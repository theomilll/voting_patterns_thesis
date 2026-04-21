"""
Nested-CV hyperparameter tuning for Stage 2 (vote choice prediction).

Structure: outer 5-fold (unbiased generalization) x inner 3-fold (Optuna TPE).
Inner objective: weighted log_loss. Outer reporting: bal_acc, log_loss, macro_f1, brier.
Survey weights (weight_post) are wired through fit() and log_loss.

Artifacts (under results/tuning/):
  - tuned_stage2_results.csv         (same schema as results/stage2_results.csv)
  - tuned_best_params.csv            (per outer fold: best params json + inner log_loss)
  - tuning_history_{approach}_{model}.csv   (per-trial Optuna log)
  - tuning_vs_baseline.csv           (side-by-side baseline vs tuned bal_acc delta)

CLI:
  python tune_stage2.py                      # full sweep (8 approaches x 3 models)
  python tune_stage2.py --smoke              # 1 approach, 1 model, 5 trials
  python tune_stage2.py --approaches A G --models XGBoost
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from train_pipeline import (
    DEMO_COLS, DEMO_STRICT, ISSUE_COLS, N_FOLDS, OUT_DIR,
    RANDOM_STATE, TARGET, WEIGHT_COL,
    _run_stage1_for_outer_fold, evaluate_fold, impute_fold,
    load_data, prepare_features,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNE_DIR = OUT_DIR / "tuning"
N_OUTER = N_FOLDS          # 5
N_INNER = 3
DEFAULT_N_TRIALS = {"LogisticRegression": 30, "RandomForest": 50, "XGBoost": 60}
ALL_MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

# Approaches & their Stage-2 feature column sets.
# "B_predicted_issues" is handled specially (Stage 1 re-run per outer fold).
APPROACH_FEATURES = {
    "A_actual_issues":      DEMO_COLS + ISSUE_COLS,
    "C_demo_with_ideology": DEMO_COLS,
    "Cs_demo_strict":       DEMO_STRICT,
    "D_party_only":         ["party_id"],
    "E_party_demo":         DEMO_COLS + ["party_id"],
    "F_issues_only":        ISSUE_COLS,
    "G_full_with_party":    DEMO_COLS + ISSUE_COLS + ["party_id"],
}
APPROACHES_STANDARD = list(APPROACH_FEATURES.keys())
APPROACHES_B = ["B_predicted_issues"]
ALL_APPROACHES = APPROACHES_STANDARD + APPROACHES_B


# ── Search spaces ──────────────────────────────────────────────────────────

def build_search_space(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [3, 5, 7, 10, 15, 20, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    raise ValueError(f"Unknown model: {model_name}")


def make_model(model_name: str, params: dict, n_classes: int):
    if model_name == "LogisticRegression":
        kwargs = dict(
            penalty="elasticnet", solver="saga",
            max_iter=5000, random_state=RANDOM_STATE, **params,
        )
        if n_classes > 2:
            kwargs["multi_class"] = "multinomial"
        return LogisticRegression(**kwargs)
    if model_name == "RandomForest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "XGBoost":
        return XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="mlogloss", use_label_encoder=False, **params)
    raise ValueError(f"Unknown model: {model_name}")


# ── Inner CV objective ─────────────────────────────────────────────────────

def _inner_score(X_tr: np.ndarray, y_tr: np.ndarray,
                 w_tr: np.ndarray | None, model_name: str, params: dict,
                 n_classes: int, seed: int, trial: optuna.Trial | None = None):
    """Mean weighted log_loss across N_INNER stratified folds.
       Reports intermediate values for pruning if trial is passed."""
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    labels = np.arange(n_classes)
    scores = []
    for fold_idx, (ti, vi) in enumerate(inner.split(X_tr, y_tr)):
        Xt, Xv = X_tr[ti].copy(), X_tr[vi].copy()
        yt, yv = y_tr[ti], y_tr[vi]
        wt = w_tr[ti] if w_tr is not None else None
        wv = w_tr[vi] if w_tr is not None else None

        impute_fold(Xt, Xv)
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xv = sc.transform(Xv)

        model = make_model(model_name, params, n_classes)
        try:
            model.fit(Xt, yt, sample_weight=wt)
            proba = model.predict_proba(Xv)
            ll = log_loss(yv, proba, labels=labels, sample_weight=wv)
        except Exception:
            return float("inf")   # tell Optuna this trial is bad
        scores.append(ll)

        if trial is not None:
            trial.report(float(np.mean(scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def _tune_outer_fold(X_tr_outer, y_tr_outer, w_tr_outer, model_name,
                     n_classes, n_trials, outer_fold, pruner=True):
    """Run Optuna on one outer-train split; return (best_params, study)."""
    seed = RANDOM_STATE + outer_fold
    sampler = TPESampler(seed=seed)
    pr = MedianPruner(n_startup_trials=10, n_warmup_steps=0) if pruner else None
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pr)

    def objective(trial):
        params = build_search_space(trial, model_name)
        return _inner_score(X_tr_outer, y_tr_outer, w_tr_outer,
                            model_name, params, n_classes, seed, trial)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _refit_and_score(X_tr_outer, y_tr_outer, w_tr, X_te_outer, y_te_outer, w_te,
                     model_name, params, n_classes):
    """Train on outer-train with best params, evaluate on outer-test."""
    Xt = X_tr_outer.copy()
    Xv = X_te_outer.copy()
    impute_fold(Xt, Xv)
    sc = StandardScaler()
    Xt_s = sc.fit_transform(Xt)
    Xv_s = sc.transform(Xv)

    model = make_model(model_name, params, n_classes)
    model.fit(Xt_s, y_tr_outer, sample_weight=w_tr)
    proba = model.predict_proba(Xv_s)
    pred = model.predict(Xv_s)
    classes = np.arange(n_classes)
    metrics = evaluate_fold(y_te_outer, proba, pred, classes, sample_weight=w_te)
    return metrics, pred, proba


# ── Orchestration: standard approaches ─────────────────────────────────────

def tune_standard_approach(df, weights, approach, model_name, n_trials,
                           n_classes):
    """Outer 5-fold x inner 3-fold Optuna for a feature-column approach."""
    feat_cols = APPROACH_FEATURES[approach]
    X, y, _, _ = prepare_features(df, feat_cols, TARGET)

    # Align weights to valid target rows (prepare_features drops NaN targets,
    # but in practice TARGET is already filtered in load_data)
    if weights is not None and len(weights) != len(y):
        weights = weights[:len(y)]   # same filter applied upstream

    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                            random_state=RANDOM_STATE)

    per_fold = []
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), n_classes))
    best_params_rows = []
    trial_history_rows = []

    for fold_idx, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        w_tr = weights[tr] if weights is not None else None
        w_te = weights[te] if weights is not None else None

        study = _tune_outer_fold(X_tr, y_tr, w_tr, model_name, n_classes,
                                 n_trials, fold_idx)
        best_params = study.best_params
        inner_best = study.best_value

        # Log every trial
        for t in study.trials:
            trial_history_rows.append({
                "approach": approach, "model": model_name,
                "outer_fold": fold_idx, "trial": t.number,
                "state": t.state.name,
                "inner_log_loss": t.value if t.value is not None else float("nan"),
                **{f"p_{k}": v for k, v in t.params.items()},
            })

        # Refit + score on outer test
        metrics, pred, proba = _refit_and_score(
            X_tr, y_tr, w_tr, X_te, y_te, w_te,
            model_name, best_params, n_classes)
        oof_pred[te] = pred
        oof_proba[te] = proba
        per_fold.append(metrics)

        best_params_rows.append({
            "approach": approach, "model": model_name,
            "outer_fold": fold_idx,
            "inner_best_log_loss": inner_best,
            "best_params": json.dumps(best_params),
            "outer_bal_acc": metrics["balanced_accuracy"],
            "outer_log_loss": metrics["log_loss"],
        })

        print(f"    fold {fold_idx} bal_acc={metrics['balanced_accuracy']:.4f} "
              f"log_loss={metrics['log_loss']:.4f}  inner_best={inner_best:.4f}")

    return per_fold, oof_pred, oof_proba, best_params_rows, trial_history_rows


# ── Orchestration: Approach B (nested Stage 1 + tuning) ────────────────────

def tune_approach_b(df, weights, model_name, n_trials, n_classes):
    """Outer loop rebuilds Stage 1 per fold; inner loop tunes Stage 2 params."""
    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                            random_state=RANDOM_STATE)
    X_demo, y_vote, _, _ = prepare_features(df, DEMO_COLS, TARGET)

    valid_mask = df[TARGET].notna()
    valid_indices = df.index[valid_mask]

    per_fold = []
    oof_pred = np.full(len(y_vote), -1, dtype=int)
    oof_proba = np.zeros((len(y_vote), n_classes))
    best_params_rows = []
    trial_history_rows = []

    for fold_idx, (tr, te) in enumerate(outer.split(X_demo, y_vote)):
        df_train = df.loc[valid_indices[tr]]
        df_val = df.loc[valid_indices[te]]
        w_tr = weights[tr] if weights is not None else None
        w_te = weights[te] if weights is not None else None

        # Build Stage 1 predictions ONCE per outer fold (not per trial).
        train_preds, val_preds = _run_stage1_for_outer_fold(
            df_train, df_val, sample_weights_train=w_tr)
        pred_cols = [c for c in train_preds.columns if c in val_preds.columns]

        X_tr = np.column_stack([X_demo[tr], train_preds[pred_cols].values])
        X_te = np.column_stack([X_demo[te], val_preds[pred_cols].values])
        y_tr = y_vote[tr]
        y_te = y_vote[te]

        study = _tune_outer_fold(X_tr, y_tr, w_tr, model_name, n_classes,
                                 n_trials, fold_idx)
        best_params = study.best_params
        inner_best = study.best_value

        for t in study.trials:
            trial_history_rows.append({
                "approach": "B_predicted_issues", "model": model_name,
                "outer_fold": fold_idx, "trial": t.number,
                "state": t.state.name,
                "inner_log_loss": t.value if t.value is not None else float("nan"),
                **{f"p_{k}": v for k, v in t.params.items()},
            })

        metrics, pred, proba = _refit_and_score(
            X_tr, y_tr, w_tr, X_te, y_te, w_te,
            model_name, best_params, n_classes)
        oof_pred[te] = pred
        oof_proba[te] = proba
        per_fold.append(metrics)

        best_params_rows.append({
            "approach": "B_predicted_issues", "model": model_name,
            "outer_fold": fold_idx,
            "inner_best_log_loss": inner_best,
            "best_params": json.dumps(best_params),
            "outer_bal_acc": metrics["balanced_accuracy"],
            "outer_log_loss": metrics["log_loss"],
        })

        print(f"    fold {fold_idx} bal_acc={metrics['balanced_accuracy']:.4f} "
              f"log_loss={metrics['log_loss']:.4f}  inner_best={inner_best:.4f}")

    return per_fold, oof_pred, oof_proba, best_params_rows, trial_history_rows


# ── Aggregation + CSV writing ──────────────────────────────────────────────

def _summarize_fold_metrics(per_fold, approach, model_name, n_samples):
    avg = {}
    for key in per_fold[0]:
        vals = [m[key] for m in per_fold]
        avg[key] = float(np.mean(vals))
        avg[f"{key}_std"] = float(np.std(vals))
    return {
        "stage": 2, "approach": approach, "model": model_name,
        "n_samples": n_samples, **avg,
    }


def _merge_and_write(new_rows: list[dict], path: Path, keys: list[str]):
    """Merge new rows with any pre-existing CSV on `keys`; new rows win on conflict."""
    new_df = pd.DataFrame(new_rows)
    if path.exists() and len(new_df) > 0:
        old_df = pd.read_csv(path)
        # drop old rows that are being overwritten by new ones
        mask = pd.Series(True, index=old_df.index)
        for _, r in new_df[keys].iterrows():
            mask &= ~((old_df[keys] == r.values).all(axis=1))
        merged = pd.concat([old_df[mask], new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False)


def _flush_results(results_rows, best_params_rows, history_rows):
    """Merge-and-write artifacts so partial re-runs don't wipe prior results.
       Called after each (approach, model)."""
    TUNE_DIR.mkdir(parents=True, exist_ok=True)

    if results_rows:
        _merge_and_write(
            results_rows, TUNE_DIR / "tuned_stage2_results.csv",
            keys=["approach", "model"])
    if best_params_rows:
        _merge_and_write(
            best_params_rows, TUNE_DIR / "tuned_best_params.csv",
            keys=["approach", "model", "outer_fold"])
    for (approach, model), rows in history_rows.items():
        # History files are per (approach, model) — overwrite is fine
        fn = TUNE_DIR / f"tuning_history_{approach}_{model}.csv"
        pd.DataFrame(rows).to_csv(fn, index=False)


def compute_vs_baseline():
    """Write tuning_vs_baseline.csv (bal_acc delta per approach x model)."""
    baseline_path = OUT_DIR / "stage2_results.csv"
    tuned_path = TUNE_DIR / "tuned_stage2_results.csv"
    if not baseline_path.exists() or not tuned_path.exists():
        print("  [skip] baseline or tuned results missing")
        return

    bl = pd.read_csv(baseline_path)[
        ["approach", "model", "balanced_accuracy", "balanced_accuracy_std",
         "log_loss", "log_loss_std"]
    ].rename(columns={
        "balanced_accuracy": "baseline_bal_acc",
        "balanced_accuracy_std": "baseline_bal_acc_std",
        "log_loss": "baseline_log_loss",
        "log_loss_std": "baseline_log_loss_std",
    })
    tu = pd.read_csv(tuned_path)[
        ["approach", "model", "balanced_accuracy", "balanced_accuracy_std",
         "log_loss", "log_loss_std"]
    ].rename(columns={
        "balanced_accuracy": "tuned_bal_acc",
        "balanced_accuracy_std": "tuned_bal_acc_std",
        "log_loss": "tuned_log_loss",
        "log_loss_std": "tuned_log_loss_std",
    })
    m = bl.merge(tu, on=["approach", "model"], how="outer")
    m["bal_acc_delta"] = m["tuned_bal_acc"] - m["baseline_bal_acc"]
    m["log_loss_delta"] = m["tuned_log_loss"] - m["baseline_log_loss"]
    m = m.sort_values(["approach", "model"])
    m.to_csv(TUNE_DIR / "tuning_vs_baseline.csv", index=False)
    print("  wrote tuning_vs_baseline.csv")
    print(m[["approach", "model", "baseline_bal_acc", "tuned_bal_acc",
             "bal_acc_delta"]].round(4).to_string(index=False))


# ── Main sweep ─────────────────────────────────────────────────────────────

def run_sweep(approaches, models, n_trials_map):
    TUNE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    weights = None
    if WEIGHT_COL in df.columns:
        w = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)
        weights = w.values

    le = LabelEncoder()
    y_enc = le.fit_transform(df[TARGET])
    n_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)}  n_classes={n_classes}")

    all_results = []
    all_best_params = []
    all_history = {}

    for approach in approaches:
        for model_name in models:
            print(f"\n--- Tuning {approach} / {model_name} "
                  f"(n_trials={n_trials_map[model_name]}) ---")
            if approach == "B_predicted_issues":
                per_fold, oof_p, oof_pr, bp, th = tune_approach_b(
                    df, weights, model_name, n_trials_map[model_name], n_classes)
                n_samples = len(y_enc)
            else:
                per_fold, oof_p, oof_pr, bp, th = tune_standard_approach(
                    df, weights, approach, model_name,
                    n_trials_map[model_name], n_classes)
                n_samples = len(prepare_features(
                    df, APPROACH_FEATURES[approach], TARGET)[1])

            summary = _summarize_fold_metrics(
                per_fold, approach, model_name, n_samples)
            all_results.append(summary)
            all_best_params.extend(bp)
            all_history[(approach, model_name)] = th

            print(f"  >>> summary: bal_acc={summary['balanced_accuracy']:.4f} "
                  f"+/-{summary['balanced_accuracy_std']:.4f}  "
                  f"log_loss={summary['log_loss']:.4f}")

            # Flush after every (approach, model) so overnight runs are resilient
            _flush_results(all_results, all_best_params, all_history)

    print("\nSweep complete.")
    compute_vs_baseline()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: A + XGBoost, 5 trials")
    ap.add_argument("--approaches", nargs="+", default=None,
                    help="Subset of approaches (e.g. A_actual_issues G_full_with_party)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of models (LogisticRegression RandomForest XGBoost)")
    ap.add_argument("--trials", type=int, default=None,
                    help="Override n_trials for all models")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        approaches = ["A_actual_issues"]
        models = ["XGBoost"]
        n_trials_map = {m: 5 for m in ALL_MODELS}
        print("[SMOKE TEST] 1 approach x 1 model x 5 trials")
    else:
        approaches = args.approaches or ALL_APPROACHES
        models = args.models or ALL_MODELS
        if args.trials is not None:
            n_trials_map = {m: args.trials for m in ALL_MODELS}
        else:
            n_trials_map = dict(DEFAULT_N_TRIALS)

    run_sweep(approaches, models, n_trials_map)


if __name__ == "__main__":
    main()
