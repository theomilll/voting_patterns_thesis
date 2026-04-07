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

DEMO_COLS = [
    "age_group", "sex", "gender", "race", "education", "income_bracket",
    "marital", "employment", "social_class", "church_attendance",
    "ideology", "hispanic",
]

ISSUE_COLS = [
    "econ_opportunity", "econ_mobility", "econ_mobility_summary",
    "tax_millionaires", "deficit_importance", "min_wage",
    "immig_level", "immig_policy1", "immig_policy2", "immig_policy3",
    "gun_difficulty", "gun_background", "gun_assault_ban",
    "climate_impact", "climate_regulation",
    "abortion_access", "abortion_scale", "abortion_detail",
    "racial_selfhelp", "racial_slavery", "racial_deserve", "racial_tryharder",
    "populism_insiders", "populism_rich", "fairshare_summary",
    "gender_trad_summary", "gender_equality", "gender_discrim",
    "trade_summary", "diversity_summary", "healthcare_spending",
]

TARGET = "vote_choice"

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


def prepare_features(df, feature_cols, target_col):
    """Prepare X, y with basic imputation and encoding."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN
    valid = y.notna()
    X, y = X[valid], y[valid]

    # Impute numeric NaN with median
    for col in X.columns:
        if X[col].dtype in ["float64", "int64"]:
            X[col] = X[col].fillna(X[col].median())

    # Always encode target to 0-indexed integers
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)

    return X.values, y.values, le, X.columns.tolist()


def evaluate_fold(y_true, y_pred_proba, y_pred, classes):
    """Compute metrics for one fold."""
    metrics = {}
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")

    # Log-loss
    if y_pred_proba.ndim == 1:
        # Binary: sklearn expects (n, 2)
        proba_2d = np.column_stack([1 - y_pred_proba, y_pred_proba])
    else:
        proba_2d = y_pred_proba
    metrics["log_loss"] = log_loss(y_true, proba_2d, labels=classes)

    # Brier score (for binary)
    if len(classes) == 2:
        if y_pred_proba.ndim == 1:
            metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba)
        else:
            metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba[:, 1])

    return metrics


def cross_validate_model(model_fn, X, y, n_classes, n_folds=N_FOLDS):
    """Run stratified k-fold CV. Returns per-fold metrics and OOF predictions."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    classes = np.arange(n_classes)

    fold_metrics = []
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_fn(n_classes)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_val_s)
        y_proba = model.predict_proba(X_val_s)

        oof_preds[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = evaluate_fold(y_val, y_proba, y_pred, classes)
        fold_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(vals)
        avg_metrics[f"{key}_std"] = np.std(vals)

    return avg_metrics, oof_preds, oof_proba


# ── Stage 1: Demographics → Issues ──────────────────────────────────────────

def run_stage1(df):
    print("\n" + "=" * 70)
    print("  STAGE 1: Demographics → Issue Opinions")
    print("=" * 70)

    results = []
    issue_predictions = {}  # Store OOF predicted probabilities for Stage 2

    for issue_col in ISSUE_COLS:
        # Simplify to 3 categories: low / mid / high
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

        X, y, le, feat_names = prepare_features(df, DEMO_COLS, target_col)
        n_classes = len(np.unique(y[~np.isnan(y)]))

        if n_classes < 2:
            print(f"  SKIP {issue_col}: only {n_classes} class(es)")
            continue

        best_model_name = None
        best_bal_acc = -1
        best_oof_proba = None

        for model_name, model_fn in MODELS.items():
            try:
                metrics, oof_preds, oof_proba = cross_validate_model(
                    model_fn, X, y, n_classes
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

def run_stage2(df, issue_pred_df):
    print("\n" + "=" * 70)
    print("  STAGE 2: Issues + Demographics → Vote Choice")
    print("=" * 70)

    results = []

    # Encode vote target
    le_vote = LabelEncoder()
    y_all = le_vote.fit_transform(df[TARGET])
    class_names = le_vote.classes_
    n_classes = len(class_names)
    print(f"  Classes: {list(class_names)}")

    # ── Approach A: Actual issue values ──
    print("\n  --- Approach A: Actual issue values + demographics ---")
    feat_cols_A = DEMO_COLS + ISSUE_COLS
    X_A, y_A, _, feat_names_A = prepare_features(df, feat_cols_A, TARGET)

    for model_name, model_fn in MODELS.items():
        metrics, oof_preds, oof_proba = cross_validate_model(
            model_fn, X_A, y_A, n_classes
        )
        results.append({
            "stage": 2,
            "approach": "A_actual_issues",
            "model": model_name,
            "n_samples": len(y_A),
            **metrics,
        })
        print(f"    {model_name:25s} bal_acc={metrics['balanced_accuracy']:.4f}  "
              f"log_loss={metrics['log_loss']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

        # Save OOF for best model (XGBoost) for later SHAP
        if model_name == "XGBoost":
            oof_A_xgb = oof_proba
            feat_names_A_saved = feat_names_A

    # ── Approach B: Predicted issue values ──
    print("\n  --- Approach B: Predicted issue probabilities + demographics ---")
    pred_cols = [c for c in issue_pred_df.columns]
    df_B = pd.concat([df[DEMO_COLS], issue_pred_df], axis=1)
    df_B[TARGET] = df[TARGET]
    feat_cols_B = DEMO_COLS + pred_cols

    X_B, y_B, _, feat_names_B = prepare_features(df_B, feat_cols_B, TARGET)

    for model_name, model_fn in MODELS.items():
        metrics, oof_preds, oof_proba = cross_validate_model(
            model_fn, X_B, y_B, n_classes
        )
        results.append({
            "stage": 2,
            "approach": "B_predicted_issues",
            "model": model_name,
            "n_samples": len(y_B),
            **metrics,
        })
        print(f"    {model_name:25s} bal_acc={metrics['balanced_accuracy']:.4f}  "
              f"log_loss={metrics['log_loss']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    # ── Approach C: Demographics only (baseline) ──
    print("\n  --- Approach C: Demographics only (baseline) ---")
    X_C, y_C, _, feat_names_C = prepare_features(df, DEMO_COLS, TARGET)

    for model_name, model_fn in MODELS.items():
        metrics, oof_preds, oof_proba = cross_validate_model(
            model_fn, X_C, y_C, n_classes
        )
        results.append({
            "stage": 2,
            "approach": "C_demographics_only",
            "model": model_name,
            "n_samples": len(y_C),
            **metrics,
        })
        print(f"    {model_name:25s} bal_acc={metrics['balanced_accuracy']:.4f}  "
              f"log_loss={metrics['log_loss']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    return pd.DataFrame(results), X_A, y_A, feat_names_A_saved, le_vote


# ── SHAP Analysis ────────────────────────────────────────────────────────────

def run_shap(X, y, feature_names, le_vote):
    """Train final XGBoost on full data and compute SHAP values."""
    print("\n" + "=" * 70)
    print("  SHAP Feature Importance (XGBoost, Approach A)")
    print("=" * 70)

    try:
        import shap

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
            category = "demographic" if feat in DEMO_COLS else "issue"
            print(f"    {i+1:2d}. {feat:35s} {val:.4f}  ({category})")
            shap_rows.append({"rank": i + 1, "feature": feat, "mean_abs_shap": val, "category": category})

        # Save full SHAP importance
        shap_df = pd.DataFrame([
            {"feature": f, "mean_abs_shap": v, "category": "demographic" if f in DEMO_COLS else "issue"}
            for f, v in importance
        ])
        shap_df.to_csv(OUT_DIR / "shap_importance.csv", index=False)

    except Exception as e:
        print(f"  SHAP failed: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = load_data()

    # Stage 1
    stage1_results, issue_pred_df = run_stage1(df)
    stage1_results.to_csv(OUT_DIR / "stage1_results.csv", index=False)

    # Stage 1 summary
    print("\n  Stage 1 Summary (average across all issues):")
    s1_summary = stage1_results.groupby("model")["balanced_accuracy"].agg(["mean", "std"])
    print(s1_summary.to_string())

    # Stage 2
    stage2_results, X_A, y_A, feat_names_A, le_vote = run_stage2(df, issue_pred_df)
    stage2_results.to_csv(OUT_DIR / "stage2_results.csv", index=False)

    # SHAP
    run_shap(X_A, y_A, feat_names_A, le_vote)

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

    print(f"\nAll results saved to {OUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
