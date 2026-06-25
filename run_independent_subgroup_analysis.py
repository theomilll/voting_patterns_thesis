"""
Independent-subgroup issue-voting analysis for the ANES 2024 thesis.

This runner is intentionally standalone: it reads the raw ANES file so the
pre-election vote/intent and issue batteries are available, then writes new
analysis artifacts without modifying thesis prose or the main pipeline outputs.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


RAW_PATH = Path("anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv")
OUT_DIR = Path("results/independent_subgroups")
FIG_DIR = Path("figures")
RANDOM_STATE = 42
N_FOLDS = 5
N_BOOTSTRAP = 5_000


# Use codebook-verified pre-election demographics where possible. The current
# cleaned pipeline file carries some stale labels, so this analysis reads the
# raw variables directly rather than inheriting those names.
DEMOGRAPHIC_MAP = {
    "age": "V241458x",
    "sex": "V241550",
    "race_ethnicity": "V241501x",
    "education": "V241465x",
    "income": "V241567x",
    "marital": "V241461x",
    "employment_status": "V241488x",
    "church_attendance": "V241440",
    "hispanic_summary": "V241512x",
    "born_us": "V241507",
}

IDEOLOGY_MAP = {
    "ideology": "V241177",
}

# Pre-election self-placement and policy-opinion items. Excludes candidate
# placements and party-ownership items because those directly encode partisan
# evaluation rather than respondent issue position.
ISSUE_MAP = {
    "spending_services": "V241239",
    "health_insurance": "V241245",
    "abortion_7pt": "V241248",
    "guaranteed_jobs_income": "V241252",
    "govt_assistance_blacks": "V241255",
    "environment_business": "V241258",
    "social_security_spending": "V241263x",
    "public_schools_spending": "V241266x",
    "border_security_spending": "V241269x",
    "crime_spending": "V241272x",
    "welfare_spending": "V241275x",
    "highways_spending": "V241278x",
    "aid_to_poor_spending": "V241281x",
    "environment_spending": "V241284x",
    "dei_colleges": "V241290x",
    "national_economy_past": "V241294x",
    "economy_next_year": "V241297x",
    "unemployment_past": "V241300x",
    "unemployment_next": "V241301",
    "abortion_standard": "V241302",
    "abortion_importance": "V241303",
    "death_penalty": "V241308x",
    "isolationism": "V241312x",
    "military_force": "V241313",
    "voter_id": "V241319x",
    "felon_voting": "V241322x",
    "income_gap": "V241363x",
    "rising_temperatures": "V241366x",
    "paid_parental_leave": "V241369x",
    "trans_bathroom": "V241372x",
    "trans_sports": "V241375x",
    "gay_job_discrimination": "V241378x",
    "gay_adoption": "V241381x",
    "gay_marriage": "V241385x",
    "unauthorized_immigrants": "V241386",
    "birthright_citizenship": "V241389x",
    "dreamers_policy": "V241392x",
    "border_wall": "V241395x",
    "english_importance": "V241396",
    "urban_unrest": "V241397",
    "ukraine_weapons": "V241400x",
    "israel_military_aid": "V241403x",
    "palestinian_humanitarian_aid": "V241406x",
    "israel_palestine_side": "V241409x",
    "gaza_protests": "V241412x",
}

TARGET_COL = "pre_vote_intent_summary"
TARGET_CODE = "V241075x"
PARTY_ID_COL = "party_id"
PARTY_ID_CODE = "V241227x"
WEIGHT_COL = "weight_pre"
WEIGHT_CODE = "V240107a"

DEMO_COLS = list(DEMOGRAPHIC_MAP)
ISSUE_COLS = list(ISSUE_MAP)
IDEOLOGY_COLS = list(IDEOLOGY_MAP)
NOMINAL_COLS = {
    "sex",
    "race_ethnicity",
    "marital",
    "employment_status",
    "hispanic_summary",
    "born_us",
}

SUBGROUPS = {
    "pure_independents": [4],
    "independents_plus_leaners": [3, 4, 5],
}

OBSERVED_CONFIGS = {
    "pautas_only": ISSUE_COLS,
    "demographics_only": DEMO_COLS,
    "demographics+ideology": DEMO_COLS + IDEOLOGY_COLS,
    "pautas+demographics": ISSUE_COLS + DEMO_COLS,
    "pautas+demographics+ideology": ISSUE_COLS + DEMO_COLS + IDEOLOGY_COLS,
}

PREDICTED_CONFIGS = {
    "predicted_pautas_only": [],
    "predicted_pautas+demographics": DEMO_COLS,
    "predicted_pautas+demographics+ideology": DEMO_COLS + IDEOLOGY_COLS,
}


@dataclass
class CvResult:
    config: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    sample_weight: np.ndarray
    fold_id: np.ndarray
    class_names: list[str]
    metrics: dict[str, float]


def _all_raw_columns() -> list[str]:
    cols = set(DEMOGRAPHIC_MAP.values())
    cols.update(IDEOLOGY_MAP.values())
    cols.update(ISSUE_MAP.values())
    cols.update([TARGET_CODE, PARTY_ID_CODE, WEIGHT_CODE])
    return sorted(cols)


def load_analysis_data() -> pd.DataFrame:
    missing = [c for c in _all_raw_columns() if c not in pd.read_csv(RAW_PATH, nrows=0).columns]
    if missing:
        raise RuntimeError(f"Raw ANES file is missing expected columns: {missing}")

    raw = pd.read_csv(RAW_PATH, usecols=_all_raw_columns(), low_memory=False)
    rename = {
        TARGET_CODE: TARGET_COL,
        PARTY_ID_CODE: PARTY_ID_COL,
        WEIGHT_CODE: WEIGHT_COL,
    }
    rename.update({v: k for k, v in DEMOGRAPHIC_MAP.items()})
    rename.update({v: k for k, v in IDEOLOGY_MAP.items()})
    rename.update({v: k for k, v in ISSUE_MAP.items()})
    df = raw.rename(columns=rename)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target: V241075x is the ANES pre-election summary of presidential
    # vote/intent/preference. Keep Harris/Trump only for a binary test.
    target_map = {
        10: "Harris",
        20: "Harris",
        30: "Harris",
        11: "Trump",
        21: "Trump",
        31: "Trump",
    }
    target_source_map = {
        10: "early_vote",
        11: "early_vote",
        20: "intent",
        21: "intent",
        30: "preference",
        31: "preference",
    }
    target_codes = df[TARGET_COL].copy()
    df["target_source"] = target_codes.map(target_source_map)
    df[TARGET_COL] = target_codes.map(target_map)

    feature_cols = DEMO_COLS + IDEOLOGY_COLS + ISSUE_COLS + [PARTY_ID_COL]
    for col in feature_cols:
        df.loc[df[col] < 0, col] = np.nan

    # 99 means "haven't thought much" on several scales and should not be
    # treated as the largest ideological/policy value.
    for col in ISSUE_COLS + IDEOLOGY_COLS:
        df.loc[df[col] >= 90, col] = np.nan

    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)
    return df


def model_factory() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
    )


def stage1_model_factory() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


def encode_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()
    nominal = [c for c in x.columns if c in NOMINAL_COLS]
    if nominal:
        x = pd.get_dummies(x, columns=nominal, dummy_na=True, prefix_sep="=", dtype=float)
    return x.astype(float)


def impute_fold(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_missing = np.isnan(x_train)
    val_missing = np.isnan(x_val)
    if train_missing.any():
        x_train[train_missing] = np.take(medians, np.where(train_missing)[1])
    if val_missing.any():
        x_val[val_missing] = np.take(medians, np.where(val_missing)[1])
    return x_train, x_val


def weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    labels = np.array([0, 1])
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "macro_f1": f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight),
        "log_loss": log_loss(y_true, y_proba, labels=labels, sample_weight=sample_weight),
        "brier_score": brier_score_loss(y_true, y_proba[:, 1], sample_weight=sample_weight),
    }


def evaluate_config(
    df: pd.DataFrame,
    feature_cols: list[str],
    config_name: str,
    extra_features_by_fold: dict[int, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> CvResult:
    valid = df[TARGET_COL].isin(["Harris", "Trump"])
    work = df.loc[valid].copy()

    le = LabelEncoder()
    y = le.fit_transform(work[TARGET_COL])
    if list(le.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(le.classes_)}")

    weights = work[WEIGHT_COL].to_numpy(dtype=float)
    x_base = encode_feature_frame(work, feature_cols)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_base, y)):
        x_train_df = x_base.iloc[train_idx].reset_index(drop=True)
        x_val_df = x_base.iloc[val_idx].reset_index(drop=True)

        if extra_features_by_fold is not None:
            extra_train, extra_val = extra_features_by_fold[fold]
            x_train_df = pd.concat(
                [x_train_df, extra_train.reset_index(drop=True)], axis=1
            )
            x_val_df = pd.concat(
                [x_val_df, extra_val.reset_index(drop=True)], axis=1
            )

        x_train = x_train_df.to_numpy(dtype=float)
        x_val = x_val_df.to_numpy(dtype=float)
        x_train, x_val = impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model = model_factory()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])

        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)
        if proba.shape[1] != 2:
            raise RuntimeError(f"{config_name} produced non-binary probabilities")

        oof_pred[val_idx] = pred
        oof_proba[val_idx] = proba
        fold_id[val_idx] = fold

    metrics = weighted_metrics(y, oof_pred, oof_proba, weights)
    print(
        f"    {config_name:34s} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"log_loss={metrics['log_loss']:.4f}",
        flush=True,
    )
    return CvResult(
        config=config_name,
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(le.classes_),
        metrics=metrics,
    )


def _stage1_predict_proba(
    x_train_all: pd.DataFrame,
    y_issue_train: pd.Series,
    x_val_all: pd.DataFrame,
    global_classes: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = y_issue_train.notna()
    y_valid_raw = y_issue_train.loc[valid]
    if y_valid_raw.nunique() < 2:
        empty_train = pd.DataFrame(index=x_train_all.index)
        empty_val = pd.DataFrame(index=x_val_all.index)
        return empty_train, empty_val

    class_counts = y_valid_raw.value_counts()
    inner_splits = int(min(N_FOLDS, class_counts.min()))
    if inner_splits < 2:
        empty_train = pd.DataFrame(index=x_train_all.index)
        empty_val = pd.DataFrame(index=x_val_all.index)
        return empty_train, empty_val

    le = LabelEncoder()
    y_valid = le.fit_transform(y_valid_raw)
    trained_classes = list(le.classes_)
    class_to_pos = {c: i for i, c in enumerate(trained_classes)}

    encoded_train = encode_feature_frame(x_train_all, DEMO_COLS)
    encoded_val = encode_feature_frame(x_val_all, DEMO_COLS)
    encoded_val = encoded_val.reindex(columns=encoded_train.columns, fill_value=0.0)

    valid_positions = np.flatnonzero(valid.to_numpy())
    x_valid = encoded_train.iloc[valid_positions].to_numpy(dtype=float)
    train_proba = np.full((len(x_train_all), len(global_classes)), np.nan, dtype=float)

    skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
    for inner_train, inner_val in skf.split(x_valid, y_valid):
        x_inner_train = x_valid[inner_train].copy()
        x_inner_val = x_valid[inner_val].copy()
        x_inner_train, x_inner_val = impute_fold(x_inner_train, x_inner_val)
        scaler = StandardScaler()
        x_inner_train = scaler.fit_transform(x_inner_train)
        x_inner_val = scaler.transform(x_inner_val)

        model = stage1_model_factory()
        model.fit(x_inner_train, y_valid[inner_train])
        proba = model.predict_proba(x_inner_val)

        target_rows = valid_positions[inner_val]
        for global_pos, cls in enumerate(global_classes):
            if cls in class_to_pos:
                train_proba[target_rows, global_pos] = proba[:, class_to_pos[cls]]

    # Predict rows with missing issue outcomes in the outer train fold, plus all
    # validation rows, from a model trained only on outer-train respondents.
    x_full_train = encoded_train.iloc[valid_positions].to_numpy(dtype=float)
    x_train_missing = encoded_train.iloc[~valid.to_numpy()].to_numpy(dtype=float)
    x_val = encoded_val.to_numpy(dtype=float)

    x_full_train_imp = x_full_train.copy()
    arrays_to_impute = [x_full_train_imp]
    if len(x_train_missing):
        arrays_to_impute.append(x_train_missing)
    arrays_to_impute.append(x_val)

    medians = np.nanmedian(x_full_train_imp, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for arr in arrays_to_impute:
        missing = np.isnan(arr)
        if missing.any():
            arr[missing] = np.take(medians, np.where(missing)[1])

    model = stage1_model_factory()
    scaler = StandardScaler()
    x_full_train_imp = scaler.fit_transform(x_full_train_imp)
    x_val = scaler.transform(x_val)
    if len(x_train_missing):
        x_train_missing = scaler.transform(x_train_missing)
    model.fit(x_full_train_imp, y_valid)

    val_proba_model = model.predict_proba(x_val)
    val_proba = np.zeros((len(x_val_all), len(global_classes)), dtype=float)
    for global_pos, cls in enumerate(global_classes):
        if cls in class_to_pos:
            val_proba[:, global_pos] = val_proba_model[:, class_to_pos[cls]]

    if len(x_train_missing):
        missing_proba_model = model.predict_proba(x_train_missing)
        missing_rows = np.flatnonzero(~valid.to_numpy())
        for global_pos, cls in enumerate(global_classes):
            if cls in class_to_pos:
                train_proba[missing_rows, global_pos] = missing_proba_model[:, class_to_pos[cls]]

    train_proba = np.nan_to_num(train_proba, nan=0.0)
    train_cols = [f"pred_{str(c).replace('.', '_')}" for c in global_classes]
    return (
        pd.DataFrame(train_proba, columns=train_cols, index=x_train_all.index),
        pd.DataFrame(val_proba, columns=train_cols, index=x_val_all.index),
    )


def build_predicted_issue_features(df: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    valid = df[TARGET_COL].isin(["Harris", "Trump"])
    work = df.loc[valid].copy().reset_index(drop=True)
    y_vote = LabelEncoder().fit_transform(work[TARGET_COL])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    global_issue_classes = {
        issue: sorted(work[issue].dropna().unique().tolist())
        for issue in ISSUE_COLS
    }

    by_fold: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(work[DEMO_COLS], y_vote)):
        print(f"    building predicted pauta features fold {fold}", flush=True)
        train_df = work.iloc[train_idx].copy()
        val_df = work.iloc[val_idx].copy()

        train_parts = []
        val_parts = []
        for issue in ISSUE_COLS:
            classes = global_issue_classes[issue]
            if len(classes) < 2:
                continue
            train_proba, val_proba = _stage1_predict_proba(
                train_df[DEMO_COLS],
                train_df[issue],
                val_df[DEMO_COLS],
                classes,
            )
            if train_proba.empty:
                continue
            train_proba = train_proba.add_prefix(f"{issue}_")
            val_proba = val_proba.add_prefix(f"{issue}_")
            train_parts.append(train_proba.reset_index(drop=True))
            val_parts.append(val_proba.reset_index(drop=True))

        if not train_parts:
            raise RuntimeError("No predicted issue features could be built")

        by_fold[fold] = (
            pd.concat(train_parts, axis=1),
            pd.concat(val_parts, axis=1),
        )
        print(
            f"      fold {fold}: {by_fold[fold][0].shape[1]} predicted pauta columns",
            flush=True,
        )
    return by_fold


def bootstrap_differences(
    reference: CvResult,
    other: CvResult,
    subgroup: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> list[dict[str, object]]:
    if not np.array_equal(reference.y_true, other.y_true):
        raise RuntimeError("Bootstrap inputs are not paired on the same y_true")

    rng = np.random.default_rng(RANDOM_STATE)
    y = reference.y_true
    class_indices = [np.flatnonzero(y == cls) for cls in np.unique(y)]

    metric_names = ["balanced_accuracy", "macro_f1", "log_loss", "brier_score"]
    observed = {
        m: reference.metrics[m] - other.metrics[m]
        for m in metric_names
    }
    boot = {m: np.empty(n_bootstrap, dtype=float) for m in metric_names}

    for b in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(cls_idx, size=len(cls_idx), replace=True)
            for cls_idx in class_indices
        ])
        ref_metrics = weighted_metrics(
            y[idx],
            reference.y_pred[idx],
            reference.y_proba[idx],
            reference.sample_weight[idx],
        )
        oth_metrics = weighted_metrics(
            y[idx],
            other.y_pred[idx],
            other.y_proba[idx],
            other.sample_weight[idx],
        )
        for m in metric_names:
            boot[m][b] = ref_metrics[m] - oth_metrics[m]

    rows = []
    for m in metric_names:
        ci_low, ci_high = np.percentile(boot[m], [2.5, 97.5])
        obs = observed[m]
        if obs >= 0:
            p_value = min(1.0, 2 * np.mean(boot[m] <= 0))
        else:
            p_value = min(1.0, 2 * np.mean(boot[m] >= 0))
        rows.append({
            "subgroup": subgroup,
            "reference_config": reference.config,
            "comparison_config": other.config,
            "metric": m,
            "diff_reference_minus_comparison": obs,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_p_two_sided": p_value,
            "direction": "higher_is_better" if m in {"balanced_accuracy", "macro_f1"} else "lower_is_better",
            "n_bootstrap": n_bootstrap,
        })
    return rows


def sample_summary(df: pd.DataFrame, subgroup: str) -> dict[str, object]:
    counts = df[TARGET_COL].value_counts().to_dict()
    weight_by_class = df.groupby(TARGET_COL)[WEIGHT_COL].sum().to_dict()
    source_counts = df["target_source"].value_counts(dropna=False).to_dict()
    total_weight = float(df[WEIGHT_COL].sum())
    min_class = min(counts.values()) if counts else 0
    warnings_list = []
    if len(df) < 200:
        warnings_list.append("small_n_under_200")
    if min_class < 50:
        warnings_list.append("minority_class_under_50")
    return {
        "subgroup": subgroup,
        "party_id_values": json.dumps(SUBGROUPS[subgroup]),
        "n": int(len(df)),
        "weighted_n": total_weight,
        "harris_n": int(counts.get("Harris", 0)),
        "trump_n": int(counts.get("Trump", 0)),
        "harris_weight_share": weight_by_class.get("Harris", 0.0) / total_weight,
        "trump_weight_share": weight_by_class.get("Trump", 0.0) / total_weight,
        "target_source_counts": json.dumps(source_counts, sort_keys=True),
        "warnings": ";".join(warnings_list),
    }


def plot_head_to_head(bootstrap_rows: pd.DataFrame) -> None:
    ba = bootstrap_rows[bootstrap_rows["metric"] == "balanced_accuracy"].copy()
    ba = ba[ba["reference_config"] == "pautas_only"]
    comparisons = [
        "demographics_only",
        "demographics+ideology",
        "pautas+demographics",
        "pautas+demographics+ideology",
    ]
    subgroups = ["pure_independents", "independents_plus_leaners"]
    y_positions = np.arange(len(comparisons))
    offsets = {"pure_independents": -0.12, "independents_plus_leaners": 0.12}
    colors = {"pure_independents": "#4a4a4a", "independents_plus_leaners": "#8a8a8a"}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for subgroup in subgroups:
        data = ba[ba["subgroup"] == subgroup].set_index("comparison_config")
        xs = [data.loc[c, "diff_reference_minus_comparison"] for c in comparisons]
        lo = [data.loc[c, "ci_low"] for c in comparisons]
        hi = [data.loc[c, "ci_high"] for c in comparisons]
        y = y_positions + offsets[subgroup]
        xerr = np.array([
            np.array(xs) - np.array(lo),
            np.array(hi) - np.array(xs),
        ])
        ax.errorbar(
            xs,
            y,
            xerr=xerr,
            fmt="o",
            capsize=3,
            color=colors[subgroup],
            ecolor=colors[subgroup],
            label=subgroup.replace("_", " "),
            markersize=5,
            linewidth=1.2,
        )

    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(comparisons)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted balanced accuracy difference: pautas_only minus comparison")
    ax.set_title("Independent Subgroups: Pautas-Only Head-to-Head")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_independent_subgroups_head_to_head.png", dpi=220)
    fig.savefig(FIG_DIR / "fig_independent_subgroups_head_to_head.pdf")
    plt.close(fig)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_analysis_data()

    all_metrics = []
    all_bootstrap = []
    all_samples = []
    all_observed_predicted = []
    all_ideology = []

    metadata = {
        "target": TARGET_CODE,
        "target_description": "PRE summary of presidential vote/intent/preference; Harris/Trump only",
        "weight": WEIGHT_CODE,
        "party_id": PARTY_ID_CODE,
        "model": "XGBClassifier fixed across feature configurations",
        "n_folds": N_FOLDS,
        "n_bootstrap": N_BOOTSTRAP,
        "issue_feature_codes": ISSUE_MAP,
        "demographic_feature_codes": DEMOGRAPHIC_MAP,
        "ideology_feature_codes": IDEOLOGY_MAP,
        "notes": [
            "party_id is used only for subgroup definition and excluded from predictors",
            "pre-election issue features exclude candidate placements and party-ownership items",
            "V241075x combines early pre-election vote, stated intent, and stated preference",
        ],
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for subgroup, party_values in SUBGROUPS.items():
        print(f"\n== {subgroup} ==", flush=True)
        sub = df[
            df[PARTY_ID_COL].isin(party_values)
            & df[TARGET_COL].isin(["Harris", "Trump"])
        ].copy()
        sub = sub.reset_index(drop=True)
        all_samples.append(sample_summary(sub, subgroup))
        print(
            f"  n={len(sub)} class_counts={sub[TARGET_COL].value_counts().to_dict()}",
            flush=True,
        )

        results: dict[str, CvResult] = {}
        for config, cols in OBSERVED_CONFIGS.items():
            result = evaluate_config(sub, cols, config)
            results[config] = result
            all_metrics.append({
                "subgroup": subgroup,
                "config": config,
                "feature_source": "observed_pautas" if "pautas" in config else "no_pautas",
                "n": len(result.y_true),
                **result.metrics,
            })

        print("  predicted-pautas robustness", flush=True)
        predicted_by_fold = build_predicted_issue_features(sub)
        for config, cols in PREDICTED_CONFIGS.items():
            result = evaluate_config(sub, cols, config, extra_features_by_fold=predicted_by_fold)
            results[config] = result
            all_metrics.append({
                "subgroup": subgroup,
                "config": config,
                "feature_source": "predicted_pautas",
                "n": len(result.y_true),
                **result.metrics,
            })

        for comparison in [
            "demographics_only",
            "demographics+ideology",
            "pautas+demographics",
            "pautas+demographics+ideology",
        ]:
            all_bootstrap.extend(bootstrap_differences(results["pautas_only"], results[comparison], subgroup))

        for observed, predicted in [
            ("pautas_only", "predicted_pautas_only"),
            ("pautas+demographics", "predicted_pautas+demographics"),
            ("pautas+demographics+ideology", "predicted_pautas+demographics+ideology"),
        ]:
            all_observed_predicted.extend(bootstrap_differences(results[observed], results[predicted], subgroup))

        for without, with_ideology in [
            ("demographics_only", "demographics+ideology"),
            ("pautas+demographics", "pautas+demographics+ideology"),
        ]:
            diffs = bootstrap_differences(results[with_ideology], results[without], subgroup)
            for row in diffs:
                row["reference_config"] = with_ideology
                row["comparison_config"] = without
            all_ideology.extend(diffs)

        pd.DataFrame([
            {
                "subgroup": subgroup,
                "config": name,
                "fold": int(fold),
                "n": int((result.fold_id == fold).sum()),
                **weighted_metrics(
                    result.y_true[result.fold_id == fold],
                    result.y_pred[result.fold_id == fold],
                    result.y_proba[result.fold_id == fold],
                    result.sample_weight[result.fold_id == fold],
                ),
            }
            for name, result in results.items()
            for fold in range(N_FOLDS)
        ]).to_csv(OUT_DIR / f"{subgroup}_fold_metrics.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    bootstrap_df = pd.DataFrame(all_bootstrap)
    sample_df = pd.DataFrame(all_samples)
    observed_predicted_df = pd.DataFrame(all_observed_predicted)
    ideology_df = pd.DataFrame(all_ideology)

    metrics_df.to_csv(OUT_DIR / "subgroup_results.csv", index=False)
    sample_df.to_csv(OUT_DIR / "sample_summary.csv", index=False)
    bootstrap_df.to_csv(OUT_DIR / "pautas_only_bootstrap_differences.csv", index=False)
    observed_predicted_df.to_csv(OUT_DIR / "observed_vs_predicted_pautas_bootstrap.csv", index=False)
    ideology_df.to_csv(OUT_DIR / "with_vs_without_ideology_bootstrap.csv", index=False)

    for subgroup in SUBGROUPS:
        metrics_df[metrics_df["subgroup"] == subgroup].to_csv(
            OUT_DIR / f"{subgroup}_results.csv", index=False
        )
        bootstrap_df[bootstrap_df["subgroup"] == subgroup].to_csv(
            OUT_DIR / f"{subgroup}_pautas_only_bootstrap_differences.csv", index=False
        )

    plot_head_to_head(bootstrap_df)

    print("Wrote:")
    for path in [
        OUT_DIR / "metadata.json",
        OUT_DIR / "sample_summary.csv",
        OUT_DIR / "subgroup_results.csv",
        OUT_DIR / "pautas_only_bootstrap_differences.csv",
        OUT_DIR / "observed_vs_predicted_pautas_bootstrap.csv",
        OUT_DIR / "with_vs_without_ideology_bootstrap.csv",
        FIG_DIR / "fig_independent_subgroups_head_to_head.png",
        FIG_DIR / "fig_independent_subgroups_head_to_head.pdf",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    run()
