"""
TCC Training Pipeline
"Modelagem de Padrões de Opiniões Políticas a partir de pautas ideológicas"

Two-stage modeling:
  Stage 1: Demographics → Issue opinions (one model per issue)
  Stage 2: Issue opinions + Demographics → Vote choice

Compares: Logistic Regression (Elastic Net), Random Forest, XGBoost
Evaluation: Stratified 5-fold CV, log-loss, balanced accuracy, macro-F1
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    log_loss, balanced_accuracy_score, f1_score,
    brier_score_loss, classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve
import json

# ── Config ───────────────────────────────────────────────────────────────────

DATA_PATH = Path("anes_2024_clean.csv")
OUT_DIR = Path("results")
N_FOLDS = 5
RANDOM_STATE = 42

DEMO_STRICT = [
    "age_group", "sex", "race", "education", "income_bracket",
    "marital", "employment", "social_class", "church_attendance",
    "hispanic",
]

DEMO_COLS = DEMO_STRICT + ["ideology"]  # ideology-inclusive baseline
STAGE1_DEMO_COLS = DEMO_STRICT
PREDICTED_ISSUE_STAGE2_COLS = DEMO_STRICT
NOMINAL_COLS = {"race", "marital", "employment"}

ISSUE_COLS = [
    "econ_opportunity", "econ_mobility", "econ_mobility_summary",
    "tax_millionaires", "deficit_importance", "min_wage",
    "immig_level", "immig_policy1", "immig_policy2", "immig_policy3",
    "gun_difficulty", "gun_background", "gun_assault_ban",
    "climate_impact", "climate_regulation",
    # abortion_* dropped 2026-05-31: no post-wave abortion-position variable (see prepare_anes.py)
    "racial_selfhelp", "racial_slavery", "racial_deserve", "racial_tryharder",
    "populism_insiders", "populism_rich", "fairshare_summary",
    "gender_trad_summary", "gender_equality", "gender_discrim",
    "trade_summary", "diversity_summary", "healthcare_spending",
]

TARGET = "vote_choice"
WEIGHT_COL = "weight_post"

# ── Lifestyle vs Living Conditions feature groups ───────────────────────────

MEDIA_ENTERTAINMENT = [  # 11 — non-political TV (cultural signals)
    "tv_fbi", "tv_shark_tank", "tv_ncis", "tv_daily_show",
    "tv_price_right", "tv_snl", "tv_dateline", "tv_60_minutes",
    "tv_today", "tv_gma", "tv_2020",
]
MEDIA_POLITICAL = [  # 10 — partisan TV (left + right)
    "tv_maddow", "tv_odonnell_last_word", "tv_anderson_cooper",
    "tv_chris_hayes", "tv_pbs_newshour",
    "tv_hannity", "tv_jesse_watters", "tv_the_five",
    "tv_ingraham", "tv_gutfeld",
]
MEDIA_MAINSTREAM = [  # 5 — mainstream/network news TV
    "tv_bret_baier", "tv_david_muir", "tv_lester_holt",
    "tv_meet_press", "tv_face_nation",
]
MEDIA_WEB = [  # 7 — news websites
    "web_cnn", "web_fox_news", "web_politico", "web_breitbart",
    "web_npr", "web_newsmax", "web_bbc",
]
MEDIA_SOURCES = [  # 4 — channels used for campaign news
    "media_tv", "media_newspapers", "media_internet", "media_radio",
]
ENGAGEMENT_ATTITUDES = [  # 9 — political engagement + emotions
    "political_attention", "campaign_interest", "media_trust",
    "emotion_hope", "emotion_fear", "emotion_anger",
    "emotion_pride", "emotion_worry", "emotion_happy",
]

# Union of all lifestyle sub-groups
LIFESTYLE_COLS = (MEDIA_ENTERTAINMENT + MEDIA_POLITICAL + MEDIA_MAINSTREAM
                  + MEDIA_WEB + MEDIA_SOURCES + ENGAGEMENT_ATTITUDES)

LIVING_CONDITIONS_COLS = [
    "income_bracket", "employment", "social_class",
    "econ_mobility", "econ_opportunity", "econ_mobility_summary",
]

MODELS = {
    "LogisticRegression": lambda n_classes: LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        max_iter=2000, C=1.0, random_state=RANDOM_STATE,
        multi_class="multinomial" if n_classes > 2 else "auto",
    ),
    "RandomForest": lambda n_classes: RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced",
    ),
    "XGBoost": lambda n_classes: XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss",
        use_label_encoder=False,
    ),
}


# ── Utilities ────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Drop "Other" class (only 90 obs — too few for reliable training)
    df = df[df[TARGET].isin(["Harris", "Trump"])].copy()
    print(f"Loaded {len(df)} rows (Harris/Trump only)")
    print(f"  Harris: {(df[TARGET]=='Harris').sum()}, Trump: {(df[TARGET]=='Trump').sum()}")
    return df


def _encode_feature_frame(X):
    """One-hot encode unordered nominal predictors before fold-level imputation."""
    nominal = [c for c in X.columns if c in NOMINAL_COLS]
    if nominal:
        X = pd.get_dummies(
            X, columns=nominal, dummy_na=True, prefix_sep="=", dtype=float
        )
    return X.astype(float)


def _base_feature_name(feature_name):
    return feature_name.split("=", 1)[0]


def _feature_category(feature_name):
    base = _base_feature_name(feature_name)
    return "demographic" if base in DEMO_COLS else "issue"


def prepare_features(df, feature_cols, target_col):
    """Prepare X, y — encoding only, NO imputation (done inside CV folds)."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN
    valid = y.notna()
    X, y = X[valid], y[valid]
    X = _encode_feature_frame(X)

    # Always encode target to 0-indexed integers
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)

    return X.values.astype(float), y.values, le, X.columns.tolist()


def align_sample_weights_to_target(df, sample_weights, target_col):
    """Return weights aligned to the non-missing target rows used by prepare_features."""
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype=float)
    valid_mask = df[target_col].notna().to_numpy()
    valid_count = int(valid_mask.sum())
    if len(weights) == len(df):
        return weights[valid_mask]
    if len(weights) == valid_count:
        return weights
    raise ValueError(
        f"sample_weights length {len(weights)} does not match either full "
        f"data length {len(df)} or valid {target_col} length {valid_count}"
    )


def require_matching_pred_cols(train_preds, val_preds, context):
    """Fail fast if nested Stage 1 produced inconsistent Stage 2 feature columns."""
    train_cols = list(train_preds.columns)
    val_cols = list(val_preds.columns)
    if train_cols != val_cols:
        missing_from_val = sorted(set(train_cols) - set(val_cols))
        extra_in_val = sorted(set(val_cols) - set(train_cols))
        raise RuntimeError(
            f"{context} produced mismatched predicted-issue columns: "
            f"missing_from_val={missing_from_val}, extra_in_val={extra_in_val}"
        )
    if not train_cols:
        raise RuntimeError(f"{context} produced no predicted issue columns")
    return train_cols


def impute_fold(X_train, X_val):
    """Impute NaN with train-fold medians. Mutates in place."""
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for j in range(X_train.shape[1]):
        mask_tr = np.isnan(X_train[:, j])
        if mask_tr.any():
            X_train[mask_tr, j] = medians[j]
        mask_val = np.isnan(X_val[:, j])
        if mask_val.any():
            X_val[mask_val, j] = medians[j]
    return X_train, X_val


def impute_full(X):
    """Impute NaN with column medians on full array. For SHAP/descriptive use."""
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = medians[j]
    return X


def evaluate_fold(y_true, y_pred_proba, y_pred, classes, sample_weight=None):
    """Compute metrics for one fold (optionally weighted)."""
    metrics = {}
    metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_true, y_pred, sample_weight=sample_weight)
    metrics["macro_f1"] = f1_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight)

    # Log-loss
    if y_pred_proba.ndim == 1:
        # Binary: sklearn expects (n, 2)
        proba_2d = np.column_stack([1 - y_pred_proba, y_pred_proba])
    else:
        proba_2d = y_pred_proba
    metrics["log_loss"] = log_loss(
        y_true, proba_2d, labels=classes, sample_weight=sample_weight)

    # Brier score (for binary)
    if len(classes) == 2:
        p1 = y_pred_proba if y_pred_proba.ndim == 1 else y_pred_proba[:, 1]
        metrics["brier_score"] = brier_score_loss(
            y_true, p1, sample_weight=sample_weight)

    return metrics


def cross_validate_model(model_fn, X, y, n_classes, n_folds=N_FOLDS,
                         sample_weights=None):
    """Run stratified k-fold CV. Returns per-fold metrics, OOF predictions,
       and per-fold metric lists (for significance tests)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    classes = np.arange(n_classes)

    fold_metrics = []
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx].copy(), X[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        w_train = sample_weights[train_idx] if sample_weights is not None else None
        w_val = sample_weights[val_idx] if sample_weights is not None else None

        # Impute NaN with train-fold medians (prevents leakage)
        impute_fold(X_train, X_val)

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_fn(n_classes)
        model.fit(X_train_s, y_train, sample_weight=w_train)

        y_pred = model.predict(X_val_s)
        y_proba = model.predict_proba(X_val_s)

        oof_preds[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = evaluate_fold(y_val, y_proba, y_pred, classes, sample_weight=w_val)
        fold_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(vals)
        avg_metrics[f"{key}_std"] = np.std(vals)

    return avg_metrics, oof_preds, oof_proba, fold_metrics


# ── Stage 1: Demographics → Issues ──────────────────────────────────────────

def run_stage1(df, sample_weights=None):
    print("\n" + "=" * 70)
    print("  STAGE 1: Demographics → Issue Opinions")
    print("=" * 70)

    results = []
    issue_predictions = {}  # Store OOF predicted probabilities for Stage 2
    print(f"  Predictors: strict demographics only ({len(STAGE1_DEMO_COLS)} variables)")

    for issue_col in ISSUE_COLS:
        # Simplify to 3 categories: low / mid / high
        # NOTE: qcut bins computed on full data — minor leakage of tercile
        # boundaries into CV folds. Acceptable approximation since binning
        # affects only target discretization, not features, and boundaries
        # are stable across folds.
        issue_vals = df[issue_col].dropna()
        n_unique = issue_vals.nunique()

        if n_unique <= 3:
            # Already simple enough — use as-is
            target_col = issue_col
        else:
            # Bin into 3 categories (terciles)
            target_col = f"{issue_col}_binned"
            df[target_col] = pd.qcut(
                df[issue_col], q=3, labels=False, duplicates="drop"
            )

        X, y, le, feat_names = prepare_features(df, STAGE1_DEMO_COLS, target_col)
        n_classes = len(np.unique(y[~np.isnan(y)]))

        if n_classes < 2:
            print(f"  SKIP {issue_col}: only {n_classes} class(es)")
            continue

        s1_weights = align_sample_weights_to_target(df, sample_weights, target_col)

        best_model_name = None
        best_bal_acc = -1
        best_oof_proba = None

        for model_name, model_fn in MODELS.items():
            try:
                metrics, oof_preds, oof_proba, _ = cross_validate_model(
                    model_fn, X, y, n_classes, sample_weights=s1_weights
                )
                results.append({
                    "stage": 1,
                    "target": issue_col,
                    "model": model_name,
                    "n_samples": len(y),
                    "n_classes": n_classes,
                    **metrics,
                })

                if metrics["balanced_accuracy"] > best_bal_acc:
                    best_bal_acc = metrics["balanced_accuracy"]
                    best_model_name = model_name
                    best_oof_proba = oof_proba

            except Exception as e:
                print(f"  ERROR {issue_col}/{model_name}: {e}")

        if best_oof_proba is not None:
            # Store predicted probabilities (from best model) for Stage 2
            for c in range(best_oof_proba.shape[1]):
                col_name = f"{issue_col}_pred_c{c}"
                # Align back to original df index
                valid_mask = df[target_col if target_col != issue_col else issue_col].notna()
                valid_idx = df.index[valid_mask]
                series = pd.Series(np.nan, index=df.index)
                series.loc[valid_idx] = best_oof_proba[:, c]
                issue_predictions[col_name] = series

        print(f"  {issue_col:30s} best={best_model_name} bal_acc={best_bal_acc:.3f} (n={len(y)}, k={n_classes})")

    results_df = pd.DataFrame(results)
    issue_pred_df = pd.DataFrame(issue_predictions, index=df.index)

    return results_df, issue_pred_df


# ── Stage 2: Issues + Demographics → Vote ───────────────────────────────────

def _run_approach(df, feat_cols, approach_name, n_classes, results,
                  fold_tracker, sample_weights=None):
    """Helper: evaluate all models for one approach."""
    X, y, _, feat_names = prepare_features(df, feat_cols, TARGET)
    aligned_weights = align_sample_weights_to_target(df, sample_weights, TARGET)
    for model_name, model_fn in MODELS.items():
        metrics, oof_preds, oof_proba, per_fold = cross_validate_model(
            model_fn, X, y, n_classes, sample_weights=aligned_weights
        )
        results.append({
            "stage": 2,
            "approach": approach_name,
            "model": model_name,
            "n_samples": len(y),
            **metrics,
        })
        fold_tracker[(approach_name, model_name)] = {
            "per_fold": per_fold,
            "oof_preds": oof_preds,
            "oof_proba": oof_proba,
        }
        print(f"    {model_name:25s} bal_acc={metrics['balanced_accuracy']:.4f}  "
              f"log_loss={metrics['log_loss']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    return X, y, feat_names


def _run_stage1_for_outer_fold(df_train, df_val, sample_weights_train=None):
    """For each issue:
       1. Select best Stage 1 model via CV on df_train (train-only fold).
       2. Store OOF predictions on df_train rows as Stage 2 training features.
       3. Re-train the SAME selected model on full df_train (with survey weights),
          predict on df_val rows as Stage 2 validation features.
       Both steps use the same model type, same weights, and train-only medians.
       Returns (train_preds_df, val_preds_df) aligned to df_train.index / df_val.index."""
    train_pred_cols = {}
    val_pred_cols = {}

    for issue_col in ISSUE_COLS:
        # Bin target using train-only qcut boundaries (applied to val via pd.cut)
        issue_vals_tr = df_train[issue_col].dropna()
        n_unique = issue_vals_tr.nunique()

        df_train_local = df_train
        if n_unique <= 3:
            target_col = issue_col
        else:
            target_col = f"{issue_col}_binned"
            df_train_local = df_train.copy()
            df_train_local[target_col] = pd.qcut(
                df_train[issue_col], q=3, labels=False, duplicates="drop"
            )

        X_tr, y_tr, _, feat_names = prepare_features(
            df_train_local, STAGE1_DEMO_COLS, target_col)
        n_classes_s1 = len(np.unique(y_tr))
        if n_classes_s1 < 2:
            continue

        # Align train weights to rows with valid issue target
        valid_tr_mask = df_train_local[target_col].notna()
        s1_w = None
        if sample_weights_train is not None:
            s1_w = sample_weights_train[valid_tr_mask.values]

        # 1. CV-based model selection on training fold
        best_name = None
        best_ba = -1
        best_oof = None
        errors = []
        for name, model_fn in MODELS.items():
            try:
                metrics, _, oof_proba, _ = cross_validate_model(
                    model_fn, X_tr, y_tr, n_classes_s1, sample_weights=s1_w)
                if metrics["balanced_accuracy"] > best_ba:
                    best_ba = metrics["balanced_accuracy"]
                    best_name = name
                    best_oof = oof_proba
            except Exception as e:
                errors.append(f"{name}: {e}")

        if best_name is None:
            detail = "; ".join(errors) if errors else "no model candidates ran"
            raise RuntimeError(f"All Stage 1 models failed for {issue_col}: {detail}")

        # 2. Store OOF training features (aligned to df_train index)
        valid_tr_idx = df_train_local.index[valid_tr_mask]
        for c in range(best_oof.shape[1]):
            col_name = f"{issue_col}_pred_c{c}"
            series = pd.Series(np.nan, index=df_train.index)
            series.loc[valid_tr_idx] = best_oof[:, c]
            train_pred_cols[col_name] = series

        # 3. Re-train SAME model on full train (with weights), predict on val
        # Use all val rows (including NaN issue rows) — val preds only feed Stage 2
        X_val_df = _encode_feature_frame(df_val[STAGE1_DEMO_COLS].copy())
        X_val_df = X_val_df.reindex(columns=feat_names, fill_value=0.0)
        X_val = X_val_df.values.astype(float)
        X_tr_imp = X_tr.copy()
        impute_fold(X_tr_imp, X_val)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_val_s = scaler.transform(X_val)

        try:
            model = MODELS[best_name](n_classes_s1)
            model.fit(X_tr_s, y_tr, sample_weight=s1_w)
            proba = model.predict_proba(X_val_s)
        except Exception as e:
            raise RuntimeError(
                f"Selected Stage 1 model {best_name} failed for {issue_col}"
            ) from e

        for c in range(proba.shape[1]):
            col_name = f"{issue_col}_pred_c{c}"
            val_pred_cols[col_name] = pd.Series(proba[:, c], index=df_val.index)

    return (pd.DataFrame(train_pred_cols, index=df_train.index),
            pd.DataFrame(val_pred_cols, index=df_val.index))


def run_approach_b_nested(df, n_classes, fold_tracker, results, weights):
    """Nested CV for Approach B: re-run Stage 1 inside each outer fold."""
    print("\n  --- Approach B: Predicted issues + strict demographics (NESTED CV) ---")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_demo, y_vote, le, _ = prepare_features(
        df, PREDICTED_ISSUE_STAGE2_COLS, TARGET)
    classes = np.arange(n_classes)
    valid_indices = df.index[df[TARGET].notna()]
    weights = align_sample_weights_to_target(df, weights, TARGET)

    prepared_folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_demo, y_vote)):
        df_train = df.loc[valid_indices[train_idx]]
        df_val = df.loc[valid_indices[val_idx]]

        w_train = weights[train_idx] if weights is not None else None
        w_val = weights[val_idx] if weights is not None else None

        # Stage 1 inside this outer fold: CV selection on train, re-train on
        # full train + predict val. Reuse these features for all Stage 2 models.
        train_preds, val_preds = _run_stage1_for_outer_fold(
            df_train, df_val, sample_weights_train=w_train)

        pred_cols = require_matching_pred_cols(
            train_preds, val_preds, f"Approach B fold {fold}"
        )

        X_tr = np.column_stack([
            X_demo[train_idx],
            train_preds[pred_cols].values
        ])
        X_vl = np.column_stack([
            X_demo[val_idx],
            val_preds[pred_cols].values
        ])

        impute_fold(X_tr, X_vl)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        prepared_folds.append({
            "fold": fold,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X_train": X_tr_s,
            "X_val": X_vl_s,
            "y_train": y_vote[train_idx],
            "y_val": y_vote[val_idx],
            "w_train": w_train,
            "w_val": w_val,
        })
        print(f"    prepared fold {fold} with {len(pred_cols)} predicted-issue features")

    for model_name, model_fn in MODELS.items():
        fold_metrics = []
        oof_preds = np.full(len(y_vote), -1, dtype=int)
        oof_proba = np.zeros((len(y_vote), n_classes))

        for prepared in prepared_folds:
            model = model_fn(n_classes)
            model.fit(
                prepared["X_train"],
                prepared["y_train"],
                sample_weight=prepared["w_train"],
            )

            y_pred = model.predict(prepared["X_val"])
            y_prob = model.predict_proba(prepared["X_val"])

            oof_preds[prepared["val_idx"]] = y_pred
            oof_proba[prepared["val_idx"]] = y_prob

            metrics = evaluate_fold(
                prepared["y_val"], y_prob, y_pred, classes,
                sample_weight=prepared["w_val"])
            fold_metrics.append(metrics)

        avg_metrics = {}
        for key in fold_metrics[0]:
            vals = [m[key] for m in fold_metrics]
            avg_metrics[key] = np.mean(vals)
            avg_metrics[f"{key}_std"] = np.std(vals)

        results.append({
            "stage": 2,
            "approach": "B_predicted_issues",
            "model": model_name,
            "n_samples": len(y_vote),
            **avg_metrics,
        })
        fold_tracker[("B_predicted_issues", model_name)] = {
            "per_fold": fold_metrics,
            "oof_preds": oof_preds,
            "oof_proba": oof_proba,
        }
        print(f"    {model_name:25s} bal_acc={avg_metrics['balanced_accuracy']:.4f}  "
              f"log_loss={avg_metrics['log_loss']:.4f}  macro_f1={avg_metrics['macro_f1']:.4f}")


def run_stage2(df, issue_pred_df):
    print("\n" + "=" * 70)
    print("  STAGE 2: Issues + Demographics → Vote Choice")
    print("=" * 70)

    results = []
    fold_tracker = {}  # (approach, model) → {per_fold, oof_preds, oof_proba}

    # Prepare weights
    weights = None
    if WEIGHT_COL in df.columns:
        w = df[WEIGHT_COL].copy()
        w = pd.to_numeric(w, errors="coerce").fillna(1.0)
        weights = w.values

    # Encode vote target
    le_vote = LabelEncoder()
    y_all = le_vote.fit_transform(df[TARGET])
    class_names = le_vote.classes_
    n_classes = len(class_names)
    print(f"  Classes: {list(class_names)}")

    # ── Approach A: Actual issues + strict demographics ──
    print("\n  --- Approach A: Actual issues + strict demographics ---")
    X_A, y_A, feat_names_A = _run_approach(
        df, DEMO_STRICT + ISSUE_COLS, "A_actual_issues", n_classes,
        results, fold_tracker, weights)

    # ── Approach B: Predicted issue probabilities + strict demographics ──
    run_approach_b_nested(df, n_classes, fold_tracker, results, weights)

    # ── Approach C: Demographics with ideology (original baseline) ──
    print("\n  --- Approach C: Demographics + ideology (baseline) ---")
    _run_approach(df, DEMO_COLS, "C_demo_with_ideology", n_classes,
                  results, fold_tracker, weights)

    # ── Approach C-strict: Demographics WITHOUT ideology ──
    print("\n  --- Approach C-strict: Demographics only (no ideology) ---")
    _run_approach(df, DEMO_STRICT, "Cs_demo_strict", n_classes,
                  results, fold_tracker, weights)

    # ── Approach D: Party ID only ──
    print("\n  --- Approach D: Party ID only ---")
    _run_approach(df, ["party_id"], "D_party_only", n_classes,
                  results, fold_tracker, weights)

    # ── Approach E: Party ID + demographics ──
    print("\n  --- Approach E: Party ID + demographics ---")
    _run_approach(df, DEMO_COLS + ["party_id"], "E_party_demo", n_classes,
                  results, fold_tracker, weights)

    # ── Approach F: Issues only (no demographics) ──
    print("\n  --- Approach F: Issues only ---")
    _run_approach(df, ISSUE_COLS, "F_issues_only", n_classes,
                  results, fold_tracker, weights)

    # ── Approach G: Party ID + issues + demographics (full) ──
    print("\n  --- Approach G: Party + issues + demographics (full) ---")
    _run_approach(df, DEMO_COLS + ISSUE_COLS + ["party_id"],
                  "G_full_with_party", n_classes, results, fold_tracker, weights)

    return (pd.DataFrame(results), X_A, y_A, feat_names_A, le_vote,
            fold_tracker)


# ── Significance Tests ──────────────────────────────────────────────────────

def _weighted_balanced_accuracy(y_true, y_pred, sample_weight=None):
    """Compute balanced accuracy with optional survey weights."""
    return balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)


def run_significance_tests(fold_tracker, y_true, sample_weights=None,
                           reference_model="XGBoost", n_bootstrap=10000):
    """Bootstrap paired comparison of approaches using per-sample OOF
       predictions and weighted balanced accuracy."""

    print("\n" + "=" * 70)
    print(f"  SIGNIFICANCE TESTS (bootstrap B={n_bootstrap}, weighted bal_acc)")
    print("=" * 70)

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)

    # Collect OOF predictions for the reference model
    approach_preds = {}
    for (approach, model), data in fold_tracker.items():
        if model == reference_model:
            approach_preds[approach] = data["oof_preds"]

    pairs = []
    approaches = sorted(approach_preds.keys())
    for i, a1 in enumerate(approaches):
        for a2 in approaches[i + 1:]:
            preds1 = approach_preds[a1]
            preds2 = approach_preds[a2]

            # Observed difference in weighted balanced accuracy
            ba1 = _weighted_balanced_accuracy(y_true, preds1, sample_weights)
            ba2 = _weighted_balanced_accuracy(y_true, preds2, sample_weights)
            obs_diff = ba1 - ba2

            # Bootstrap
            boot_diffs = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                w_b = sample_weights[idx] if sample_weights is not None else None
                b_ba1 = _weighted_balanced_accuracy(y_true[idx], preds1[idx], w_b)
                b_ba2 = _weighted_balanced_accuracy(y_true[idx], preds2[idx], w_b)
                boot_diffs[b] = b_ba1 - b_ba2

            # Two-sided p-value
            if obs_diff >= 0:
                p_val = np.mean(boot_diffs <= 0) * 2
            else:
                p_val = np.mean(boot_diffs >= 0) * 2
            p_val = min(p_val, 1.0)

            ci_lower = np.percentile(boot_diffs, 2.5)
            ci_upper = np.percentile(boot_diffs, 97.5)

            # McNemar's test (secondary, unweighted)
            correct1 = (preds1 == y_true).astype(int)
            correct2 = (preds2 == y_true).astype(int)
            b_c = ((correct1 == 1) & (correct2 == 0)).sum()  # 1 right, 2 wrong
            c_b = ((correct1 == 0) & (correct2 == 1)).sum()  # 1 wrong, 2 right
            if b_c + c_b > 0:
                from scipy.stats import binomtest
                mcnemar_p = binomtest(b_c, b_c + c_b, 0.5).pvalue
            else:
                mcnemar_p = 1.0

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            pairs.append({
                "approach_1": a1, "approach_2": a2,
                "mean_diff": obs_diff, "bootstrap_p": p_val,
                "ci_lower": ci_lower, "ci_upper": ci_upper,
                "mcnemar_p": mcnemar_p, "significant": sig,
            })
            print(f"  {a1:25s} vs {a2:25s}  diff={obs_diff:+.4f}  "
                  f"p={p_val:.4f} CI=[{ci_lower:+.4f},{ci_upper:+.4f}] {sig}")

    sig_df = pd.DataFrame(pairs)
    sig_df.to_csv(OUT_DIR / "significance_tests.csv", index=False)
    return sig_df


# ── Ablation Study ──────────────────────────────────────────────────────────

def run_ablation(df, sample_weights=None):
    """Leave-one-issue-out and cumulative addition analysis."""
    from sklearn.metrics import balanced_accuracy_score as ba_score

    print("\n" + "=" * 70)
    print("  ABLATION STUDY (XGBoost, Approach A)")
    print("=" * 70)

    base_cols = DEMO_COLS + ISSUE_COLS
    X_base, y_base, _, _ = prepare_features(df, base_cols, TARGET)
    n_classes = len(np.unique(y_base))
    aligned_weights = align_sample_weights_to_target(df, sample_weights, TARGET)

    # Baseline (all features)
    xgb_fn = MODELS["XGBoost"]
    base_metrics, _, _, _ = cross_validate_model(
        xgb_fn, X_base, y_base, n_classes, sample_weights=aligned_weights)
    base_ba = base_metrics["balanced_accuracy"]
    print(f"  Baseline (all issues): {base_ba:.4f}")

    # Leave-one-issue-out
    print("\n  Leave-one-issue-out:")
    loo_results = []
    for issue in ISSUE_COLS:
        reduced_cols = DEMO_COLS + [c for c in ISSUE_COLS if c != issue]
        X_r, y_r, _, _ = prepare_features(df, reduced_cols, TARGET)
        m, _, _, _ = cross_validate_model(xgb_fn, X_r, y_r, n_classes,
                                          sample_weights=aligned_weights)
        drop = base_ba - m["balanced_accuracy"]
        loo_results.append({
            "issue_removed": issue,
            "balanced_accuracy": m["balanced_accuracy"],
            "accuracy_drop": drop,
        })
        print(f"    -{issue:30s} ba={m['balanced_accuracy']:.4f}  "
              f"drop={drop:+.4f}")

    loo_df = pd.DataFrame(loo_results).sort_values("accuracy_drop",
                                                    ascending=False)
    loo_df.to_csv(OUT_DIR / "ablation_leave_one_out.csv", index=False)

    # Cumulative addition (ranked by SHAP if available, else alphabetical)
    print("\n  Cumulative feature addition:")
    shap_path = OUT_DIR / "shap_importance.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        issue_rank = [f for f in shap_df["feature"]
                      if f in ISSUE_COLS]
    else:
        issue_rank = sorted(ISSUE_COLS)

    cum_results = []
    for k in range(1, len(issue_rank) + 1):
        top_k = issue_rank[:k]
        cum_cols = DEMO_COLS + top_k
        X_c, y_c, _, _ = prepare_features(df, cum_cols, TARGET)
        m, _, _, _ = cross_validate_model(xgb_fn, X_c, y_c, n_classes,
                                          sample_weights=aligned_weights)
        cum_results.append({
            "n_issues": k,
            "last_added": top_k[-1],
            "balanced_accuracy": m["balanced_accuracy"],
        })
        if k <= 10 or k == len(issue_rank):
            print(f"    top-{k:2d} (+{top_k[-1]:30s}) ba={m['balanced_accuracy']:.4f}")

    cum_df = pd.DataFrame(cum_results)
    cum_df.to_csv(OUT_DIR / "ablation_cumulative.csv", index=False)

    return loo_df, cum_df


# ── SHAP Analysis ────────────────────────────────────────────────────────────

def run_shap(X, y, feature_names, le_vote):
    """Train final XGBoost on full data and compute SHAP values."""
    print("\n" + "=" * 70)
    print("  SHAP Feature Importance (XGBoost, Approach A)")
    print("=" * 70)

    try:
        import shap

        # Impute for descriptive SHAP (not evaluative, full-data ok)
        X = impute_full(X.copy())

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # Use sklearn RF for SHAP (more stable than XGBoost 3.x with SHAP)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        )
        model.fit(X_s, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)

        # For binary RF, shap_values is list of 2 arrays [class0, class1]
        if isinstance(shap_values, list):
            sv = np.abs(shap_values[1])
        elif shap_values.ndim == 3:
            sv = np.abs(shap_values[:, :, 1])
        else:
            sv = np.abs(shap_values)

        mean_shap = sv.mean(axis=0)
        importance = sorted(
            zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True
        )

        print("\n  Top 20 features by mean |SHAP|:")
        shap_rows = []
        for i, (feat, val) in enumerate(importance[:20]):
            category = _feature_category(feat)
            print(f"    {i+1:2d}. {feat:35s} {val:.4f}  ({category})")
            shap_rows.append({"rank": i + 1, "feature": feat, "mean_abs_shap": val, "category": category})

        # Save full SHAP importance
        shap_df = pd.DataFrame([
            {"feature": f, "mean_abs_shap": v, "category": _feature_category(f)}
            for f, v in importance
        ])
        shap_df.to_csv(OUT_DIR / "shap_importance.csv", index=False)

    except Exception as e:
        print(f"  SHAP failed: {e}")


# ── Lifestyle vs Living Conditions ──────────────────────────────────────────

def run_lifestyle_vs_conditions(df, sample_weights=None):
    """Compare lifestyle/interests vs material living conditions as
       predictors of vote choice."""
    print("\n" + "=" * 70)
    print("  LIFESTYLE vs LIVING CONDITIONS (XGBoost)")
    print("=" * 70)

    # Filter to columns that actually exist in the dataframe
    lifestyle = [c for c in LIFESTYLE_COLS if c in df.columns]
    conditions = [c for c in LIVING_CONDITIONS_COLS if c in df.columns]
    entertainment = [c for c in MEDIA_ENTERTAINMENT if c in df.columns]
    political = [c for c in MEDIA_POLITICAL if c in df.columns]
    engagement = [c for c in ENGAGEMENT_ATTITUDES if c in df.columns]

    print(f"  All media/lifestyle ({len(lifestyle)})  |  Conditions ({len(conditions)})")
    print(f"  Sub-groups: entertainment={len(entertainment)}, "
          f"political={len(political)}, engagement={len(engagement)}")

    xgb_fn = MODELS["XGBoost"]
    results = []
    aligned_weights = align_sample_weights_to_target(df, sample_weights, TARGET)

    def _eval_config(name, cols):
        cols_exist = [c for c in cols if c in df.columns]
        X, y, _, _ = prepare_features(df, cols_exist, TARGET)
        n_classes = len(np.unique(y))
        metrics, _, _, _ = cross_validate_model(
            xgb_fn, X, y, n_classes, sample_weights=aligned_weights
        )
        results.append({"config": name, "n_features": len(cols_exist), **metrics})
        print(f"  {name:35s} ({len(cols_exist):2d} feats)  "
              f"bal_acc={metrics['balanced_accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}")
        return metrics

    # Full groups
    _eval_config("all_media_lifestyle", lifestyle)
    _eval_config("conditions_only", conditions)

    # Balanced sub-group comparisons (similar feature counts)
    print("\n  Sub-group comparisons (balanced feature counts):")
    _eval_config("entertainment_only", entertainment)
    _eval_config("political_media_only", political)
    _eval_config("engagement_attitudes_only", engagement)

    # Combined
    lc_combined = list(dict.fromkeys(lifestyle + conditions))
    lcd_combined = list(dict.fromkeys(lifestyle + conditions + DEMO_STRICT))
    _eval_config("media_conditions_combined", lc_combined)
    _eval_config("media_conditions_demo", lcd_combined)

    # Hierarchical decomposition
    print("\n  Hierarchical decomposition:")
    m_c = _eval_config("_hier_conditions", conditions)
    m_cl = _eval_config("_hier_conditions+lifestyle",
                        list(dict.fromkeys(conditions + lifestyle)))
    lift_l = m_cl["balanced_accuracy"] - m_c["balanced_accuracy"]
    print(f"    Lift from adding lifestyle to conditions: {lift_l:+.4f}")

    m_l = _eval_config("_hier_lifestyle", lifestyle)
    m_lc = _eval_config("_hier_lifestyle+conditions",
                        list(dict.fromkeys(lifestyle + conditions)))
    lift_c = m_lc["balanced_accuracy"] - m_l["balanced_accuracy"]
    print(f"    Lift from adding conditions to lifestyle: {lift_c:+.4f}")

    results.append({
        "config": "incremental_lifestyle_over_conditions",
        "n_features": len(lifestyle),
        "balanced_accuracy": lift_l,
    })
    results.append({
        "config": "incremental_conditions_over_lifestyle",
        "n_features": len(conditions),
        "balanced_accuracy": lift_c,
    })

    # Remove duplicate _hier_ rows from final output
    results_df = pd.DataFrame([r for r in results if not r["config"].startswith("_hier_")])
    results_df.to_csv(OUT_DIR / "lifestyle_vs_conditions.csv", index=False)

    # SHAP on combined model
    print("\n  SHAP on combined lifestyle+conditions model:")
    try:
        import shap
        all_cols = list(dict.fromkeys(lifestyle + conditions))
        X_all, y_all, _, feat_names_all = prepare_features(df, all_cols, TARGET)
        X_all = impute_full(X_all.copy())
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_all)
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
        rf.fit(X_s, y_all)
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_s)
        if isinstance(sv, list):
            sv = np.abs(sv[1])
        elif sv.ndim == 3:
            sv = np.abs(sv[:, :, 1])
        else:
            sv = np.abs(sv)
        mean_sv = sv.mean(axis=0)
        importance = sorted(zip(feat_names_all, mean_sv),
                            key=lambda x: x[1], reverse=True)
        shap_rows = []
        ent_set = set(entertainment)
        pol_set = set(political)
        eng_set = set(engagement)
        cond_set = set(conditions)
        for i, (feat, val) in enumerate(importance[:15]):
            if feat in ent_set:
                group = "entertainment"
            elif feat in pol_set:
                group = "political_media"
            elif feat in eng_set:
                group = "engagement"
            elif feat in cond_set:
                group = "conditions"
            else:
                group = "other_media"
            print(f"    {i+1:2d}. {feat:30s} {val:.4f}  ({group})")
            shap_rows.append({
                "feature": feat, "mean_abs_shap": val, "group": group})
        shap_lc_df = pd.DataFrame(shap_rows)
        shap_lc_df.to_csv(OUT_DIR / "shap_lifestyle_conditions.csv",
                          index=False)
    except Exception as e:
        print(f"  SHAP failed: {e}")

    return results_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true",
                    help="Run Optuna nested-CV hyperparameter tuning for Stage 2 "
                         "instead of the baseline pipeline. "
                         "See tune_stage2.py for finer-grained flags.")
    args, rest = ap.parse_known_args()

    if args.tune:
        # Strip --tune before delegating so tune_stage2's own argparse works
        sys.argv = [sys.argv[0]] + rest
        from tune_stage2 import main as tune_main
        tune_main()
        return

    OUT_DIR.mkdir(exist_ok=True)

    df = load_data()

    # Prepare sample weights
    weights = None
    if WEIGHT_COL in df.columns:
        w = df[WEIGHT_COL].copy()
        w = pd.to_numeric(w, errors="coerce").fillna(1.0)
        weights = w.values

    # Stage 1
    stage1_results, issue_pred_df = run_stage1(df, sample_weights=weights)
    stage1_results.to_csv(OUT_DIR / "stage1_results.csv", index=False)

    # Stage 1 summary
    print("\n  Stage 1 Summary (average across all issues):")
    s1_summary = stage1_results.groupby("model")["balanced_accuracy"].agg(["mean", "std"])
    print(s1_summary.to_string())

    # Stage 2 (expanded: approaches A through G)
    (stage2_results, X_A, y_A, feat_names_A, le_vote,
     fold_tracker) = run_stage2(df, issue_pred_df)
    stage2_results.to_csv(OUT_DIR / "stage2_results.csv", index=False)

    # SHAP
    run_shap(X_A, y_A, feat_names_A, le_vote)

    # Significance tests (bootstrap on per-sample OOF predictions)
    run_significance_tests(
        fold_tracker, y_A,
        sample_weights=align_sample_weights_to_target(df, weights, TARGET),
    )

    # Ablation study
    run_ablation(df, sample_weights=weights)

    # Lifestyle vs Living Conditions
    run_lifestyle_vs_conditions(df, sample_weights=weights)

    # Final comparison table
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON — Stage 2 (Vote Prediction)")
    print("=" * 70)
    pivot = stage2_results.pivot_table(
        index="model",
        columns="approach",
        values=["balanced_accuracy", "log_loss", "macro_f1"],
    ).round(4)
    print(pivot.to_string())

    # Key thesis comparisons (XGBoost)
    xgb = stage2_results[stage2_results["model"] == "XGBoost"].set_index("approach")
    if len(xgb) > 0:
        ba = xgb["balanced_accuracy"]
        print("\n" + "=" * 70)
        print("  KEY COMPARISONS (XGBoost balanced accuracy)")
        print("=" * 70)
        pairs = [
            ("Ideology adds to strict demographics", "C_demo_with_ideology", "Cs_demo_strict"),
            ("Issues + strict demo vs demo+ideology", "A_actual_issues", "C_demo_with_ideology"),
            ("Issues + strict demo vs strict demo", "A_actual_issues", "Cs_demo_strict"),
            ("Issues only vs Party only", "F_issues_only", "D_party_only"),
            ("Full (party+issues+demo) vs Party only", "G_full_with_party", "D_party_only"),
            ("Predicted issues+strict demo vs actual issues+strict demo", "B_predicted_issues", "A_actual_issues"),
        ]
        for label, a, b in pairs:
            if a in ba.index and b in ba.index:
                print(f"  {label:50s}  {ba[a]:.4f} vs {ba[b]:.4f}  "
                      f"(diff: {ba[a] - ba[b]:+.4f})")

        # NOTE: All issue variables are V242xxx (post-election wave).
        # This pipeline retrodicts vote from post-election opinions,
        # not predicts from pre-election attitudes.
        print("\n  NOTE: Issue variables are post-election (V242xxx) — "
              "results reflect retrodiction, not prediction.")

    print(f"\nAll results saved to {OUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
