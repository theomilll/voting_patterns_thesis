"""
Thesis-ready voter-intent analysis for the ANES 2024 issue-voting claim.

Target:
  V241075x, the pre-election Harris/Trump vote, intent, or preference summary.

Predictors:
  Non-vote predictors only. The script never uses the direct vote/intent item
  as a feature. It compares issue opinions against demographics, ideology, and
  party ID so the thesis claim can be stated without hiding the party-ID test.

Outputs:
  results/voter_intent/cohort_results.csv
  results/voter_intent/fold_metrics.csv
  results/voter_intent/pairwise_bootstrap.csv
  results/voter_intent/domain_ablation.csv
  results/voter_intent/sample_summary.csv
  results/voter_intent/metadata.json
  results/voter_intent/thesis_readiness_report.md
  figures/fig_voter_intent_claim_head_to_head.{png,pdf}
  figures/fig_voter_intent_domain_ablation.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
OUT_DIR = Path("results/voter_intent")
FIG_DIR = Path("figures")
RANDOM_STATE = 42
N_FOLDS = 5
DEFAULT_BOOTSTRAP = 5_000


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

# Pre-election respondent issue positions. Candidate placements and party
# ownership items are intentionally excluded because they are not issue opinions.
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

ISSUE_DOMAINS = {
    "economic_welfare": [
        "spending_services",
        "health_insurance",
        "guaranteed_jobs_income",
        "social_security_spending",
        "public_schools_spending",
        "welfare_spending",
        "highways_spending",
        "aid_to_poor_spending",
        "income_gap",
        "national_economy_past",
        "economy_next_year",
        "unemployment_past",
        "unemployment_next",
        "paid_parental_leave",
    ],
    "race_dei": [
        "govt_assistance_blacks",
        "dei_colleges",
        "urban_unrest",
    ],
    "environment": [
        "environment_business",
        "environment_spending",
        "rising_temperatures",
    ],
    "abortion": [
        "abortion_7pt",
        "abortion_standard",
        "abortion_importance",
    ],
    "immigration_identity": [
        "border_security_spending",
        "unauthorized_immigrants",
        "birthright_citizenship",
        "dreamers_policy",
        "border_wall",
        "english_importance",
    ],
    "law_order_democracy": [
        "crime_spending",
        "death_penalty",
        "voter_id",
        "felon_voting",
    ],
    "gender_lgbt": [
        "trans_bathroom",
        "trans_sports",
        "gay_job_discrimination",
        "gay_adoption",
        "gay_marriage",
    ],
    "foreign_policy": [
        "isolationism",
        "military_force",
        "ukraine_weapons",
        "israel_military_aid",
        "palestinian_humanitarian_aid",
        "israel_palestine_side",
        "gaza_protests",
    ],
}

TARGET_COL = "pre_vote_intent_summary"
TARGET_CODE = "V241075x"
PARTY_ID_COL = "party_id"
PARTY_ID_CODE = "V241227x"
WEIGHT_COL = "weight_pre"
WEIGHT_CODE = "V240107a"

DEMO_COLS = list(DEMOGRAPHIC_MAP)
IDEOLOGY_COLS = list(IDEOLOGY_MAP)
ISSUE_COLS = list(ISSUE_MAP)
NOMINAL_COLS = {
    "sex",
    "race_ethnicity",
    "marital",
    "employment_status",
    "hispanic_summary",
    "born_us",
}

FEATURE_CONFIGS = {
    "demographics_only": DEMO_COLS,
    "ideology_only": IDEOLOGY_COLS,
    "party_id_only": [PARTY_ID_COL],
    "party_id+ideology": [PARTY_ID_COL] + IDEOLOGY_COLS,
    "issues_only": ISSUE_COLS,
    "issues+demographics": ISSUE_COLS + DEMO_COLS,
    "issues+ideology": ISSUE_COLS + IDEOLOGY_COLS,
    "issues+party_id": ISSUE_COLS + [PARTY_ID_COL],
    "issues+party_id+ideology": ISSUE_COLS + [PARTY_ID_COL] + IDEOLOGY_COLS,
    "issues+demographics+ideology": ISSUE_COLS + DEMO_COLS + IDEOLOGY_COLS,
}

COHORTS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "all_respondents": lambda df: df[TARGET_COL].isin(["Harris", "Trump"]),
    "pure_independents": lambda df: (
        df[TARGET_COL].isin(["Harris", "Trump"]) & df[PARTY_ID_COL].isin([4])
    ),
    "independents_plus_leaners": lambda df: (
        df[TARGET_COL].isin(["Harris", "Trump"]) & df[PARTY_ID_COL].isin([3, 4, 5])
    ),
}

DOMAIN_ABLATION_COHORTS = {"all_respondents", "independents_plus_leaners"}

KEY_BOOTSTRAP_PAIRS = [
    ("issues_only", "demographics_only"),
    ("issues_only", "ideology_only"),
    ("issues_only", "party_id_only"),
    ("issues+demographics+ideology", "demographics_only"),
    ("issues+party_id+ideology", "party_id_only"),
    ("issues+party_id+ideology", "party_id+ideology"),
]


@dataclass
class CvResult:
    cohort: str
    config: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    sample_weight: np.ndarray
    fold_id: np.ndarray
    class_names: list[str]
    metrics: dict[str, float]


def _required_raw_columns() -> list[str]:
    cols = set(DEMOGRAPHIC_MAP.values())
    cols.update(IDEOLOGY_MAP.values())
    cols.update(ISSUE_MAP.values())
    cols.update([TARGET_CODE, PARTY_ID_CODE, WEIGHT_CODE])
    return sorted(cols)


def load_analysis_data() -> pd.DataFrame:
    header = pd.read_csv(RAW_PATH, nrows=0)
    missing = [c for c in _required_raw_columns() if c not in header.columns]
    if missing:
        raise RuntimeError(f"Raw ANES file is missing expected columns: {missing}")

    raw = pd.read_csv(RAW_PATH, usecols=_required_raw_columns(), low_memory=False)
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

    # 90+ codes are non-substantive responses on the issue/ideology batteries.
    for col in ISSUE_COLS + IDEOLOGY_COLS:
        df.loc[df[col] >= 90, col] = np.nan

    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)
    return df


def model_factory() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
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
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", sample_weight=sample_weight
        ),
        "log_loss": log_loss(y_true, y_proba, labels=labels, sample_weight=sample_weight),
        "brier_score": brier_score_loss(
            y_true, y_proba[:, 1], sample_weight=sample_weight
        ),
    }


def evaluate_config(df: pd.DataFrame, cohort: str, config: str, feature_cols: list[str]) -> CvResult:
    work = df.copy().reset_index(drop=True)
    le = LabelEncoder()
    y = le.fit_transform(work[TARGET_COL])
    if list(le.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order for {cohort}: {list(le.classes_)}")

    weights = work[WEIGHT_COL].to_numpy(dtype=float)
    x_base = encode_feature_frame(work, feature_cols)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_base, y)):
        x_train_df = x_base.iloc[train_idx]
        x_val_df = x_base.iloc[val_idx]
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
            raise RuntimeError(f"{cohort}/{config} produced non-binary probabilities")

        oof_pred[val_idx] = pred
        oof_proba[val_idx] = proba
        fold_id[val_idx] = fold

    metrics = weighted_metrics(y, oof_pred, oof_proba, weights)
    print(
        f"  {cohort:28s} {config:30s} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"log_loss={metrics['log_loss']:.4f}",
        flush=True,
    )
    return CvResult(
        cohort=cohort,
        config=config,
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(le.classes_),
        metrics=metrics,
    )


def bootstrap_differences(
    reference: CvResult,
    comparison: CvResult,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    if not np.array_equal(reference.y_true, comparison.y_true):
        raise RuntimeError("Bootstrap inputs are not paired on the same target rows")

    rng = np.random.default_rng(RANDOM_STATE)
    y = reference.y_true
    class_indices = [np.flatnonzero(y == cls) for cls in np.unique(y)]
    metric_names = ["balanced_accuracy", "macro_f1", "log_loss", "brier_score"]
    observed = {
        metric: reference.metrics[metric] - comparison.metrics[metric]
        for metric in metric_names
    }
    boot = {metric: np.empty(n_bootstrap, dtype=float) for metric in metric_names}

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
        comp_metrics = weighted_metrics(
            y[idx],
            comparison.y_pred[idx],
            comparison.y_proba[idx],
            comparison.sample_weight[idx],
        )
        for metric in metric_names:
            boot[metric][b] = ref_metrics[metric] - comp_metrics[metric]

    rows = []
    for metric in metric_names:
        ci_low, ci_high = np.percentile(boot[metric], [2.5, 97.5])
        obs = observed[metric]
        if obs >= 0:
            p_value = min(1.0, 2 * np.mean(boot[metric] <= 0))
        else:
            p_value = min(1.0, 2 * np.mean(boot[metric] >= 0))
        rows.append({
            "cohort": reference.cohort,
            "reference_config": reference.config,
            "comparison_config": comparison.config,
            "metric": metric,
            "diff_reference_minus_comparison": obs,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_p_two_sided": p_value,
            "direction": (
                "higher_is_better"
                if metric in {"balanced_accuracy", "macro_f1"}
                else "lower_is_better"
            ),
            "n_bootstrap": n_bootstrap,
        })
    return rows


def summarize_sample(df: pd.DataFrame, cohort: str, party_values: list[int] | None) -> dict[str, object]:
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
        "cohort": cohort,
        "party_id_values": json.dumps(party_values) if party_values is not None else "all",
        "n": int(len(df)),
        "weighted_n": total_weight,
        "harris_n": int(counts.get("Harris", 0)),
        "trump_n": int(counts.get("Trump", 0)),
        "harris_weight_share": weight_by_class.get("Harris", 0.0) / total_weight,
        "trump_weight_share": weight_by_class.get("Trump", 0.0) / total_weight,
        "target_source_counts": json.dumps(source_counts, sort_keys=True),
        "warnings": ";".join(warnings_list),
    }


def plot_head_to_head(results_df: pd.DataFrame) -> None:
    selected = [
        "demographics_only",
        "ideology_only",
        "issues_only",
        "party_id_only",
        "issues+party_id+ideology",
    ]
    labels = {
        "demographics_only": "Demographics only",
        "ideology_only": "Ideology only",
        "issues_only": "Issue opinions only",
        "party_id_only": "Party ID only",
        "issues+party_id+ideology": "Issues + party ID + ideology",
    }
    colors = {
        "demographics_only": "#4878A8",
        "ideology_only": "#9D755D",
        "issues_only": "#E45756",
        "party_id_only": "#9467BD",
        "issues+party_id+ideology": "#333333",
    }
    cohorts = ["all_respondents", "independents_plus_leaners", "pure_independents"]
    cohort_labels = {
        "all_respondents": "All respondents",
        "independents_plus_leaners": "Independents + leaners",
        "pure_independents": "Pure independents",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
    y_positions = np.arange(len(selected))
    for ax, cohort in zip(axes, cohorts):
        data = results_df[results_df["cohort"] == cohort].set_index("config")
        values = [data.loc[config, "balanced_accuracy"] for config in selected]
        for y, config, value in zip(y_positions, selected, values):
            ax.plot(value, y, "o", color=colors[config], markersize=8)
            ax.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=8)
        ax.set_title(cohort_labels[cohort], fontsize=11)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([labels[c] for c in selected] if ax is axes[0] else [])
        ax.set_xlim(0.45, 0.98)
        ax.axvline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Weighted balanced accuracy, 5-fold CV")
    fig.suptitle("Voter-Intent Prediction from Non-Vote Predictors", fontsize=14)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_voter_intent_claim_head_to_head.png", dpi=220)
    fig.savefig(FIG_DIR / "fig_voter_intent_claim_head_to_head.pdf")
    plt.close(fig)


def run_domain_ablation(
    df: pd.DataFrame,
    cohort: str,
    base_result: CvResult,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    """Leave one issue domain out of the issues+ideology model.

    This is interpretive rather than model-selection evidence: it asks which
    political-opinion domains the voter-intent model misses most when removed.
    """
    rows = []
    base_issues = set(ISSUE_COLS)
    for domain, removed in ISSUE_DOMAINS.items():
        missing = [issue for issue in removed if issue not in base_issues]
        if missing:
            raise RuntimeError(f"Domain {domain} contains unknown issue columns: {missing}")

        remaining_issues = [issue for issue in ISSUE_COLS if issue not in removed]
        config = f"issues+ideology_minus_{domain}"
        result = evaluate_config(df, cohort, config, remaining_issues + IDEOLOGY_COLS)
        boot_rows = bootstrap_differences(base_result, result, n_bootstrap)
        ba_boot = next(row for row in boot_rows if row["metric"] == "balanced_accuracy")
        logloss_boot = next(row for row in boot_rows if row["metric"] == "log_loss")
        rows.append({
            "cohort": cohort,
            "base_config": base_result.config,
            "domain_removed": domain,
            "n_issue_features_removed": len(removed),
            "removed_features": json.dumps(removed),
            "base_balanced_accuracy": base_result.metrics["balanced_accuracy"],
            "ablated_balanced_accuracy": result.metrics["balanced_accuracy"],
            "balanced_accuracy_drop": (
                base_result.metrics["balanced_accuracy"]
                - result.metrics["balanced_accuracy"]
            ),
            "ba_drop_ci_low": ba_boot["ci_low"],
            "ba_drop_ci_high": ba_boot["ci_high"],
            "ba_drop_p_two_sided": ba_boot["bootstrap_p_two_sided"],
            "base_log_loss": base_result.metrics["log_loss"],
            "ablated_log_loss": result.metrics["log_loss"],
            "log_loss_increase": result.metrics["log_loss"] - base_result.metrics["log_loss"],
            "log_loss_diff_base_minus_ablated": logloss_boot["diff_reference_minus_comparison"],
            "log_loss_diff_ci_low": logloss_boot["ci_low"],
            "log_loss_diff_ci_high": logloss_boot["ci_high"],
            "log_loss_diff_p_two_sided": logloss_boot["bootstrap_p_two_sided"],
        })
    return rows


def plot_domain_ablation(domain_df: pd.DataFrame) -> None:
    if domain_df.empty:
        return

    domain_labels = {
        "economic_welfare": "Economic welfare",
        "race_dei": "Race / DEI",
        "environment": "Environment",
        "abortion": "Abortion",
        "immigration_identity": "Immigration / identity",
        "law_order_democracy": "Law, order, democracy",
        "gender_lgbt": "Gender / LGBT",
        "foreign_policy": "Foreign policy",
    }
    cohorts = ["all_respondents", "independents_plus_leaners"]
    cohort_labels = {
        "all_respondents": "All respondents",
        "independents_plus_leaners": "Independents + leaners",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
    for ax, cohort in zip(axes, cohorts):
        data = domain_df[domain_df["cohort"] == cohort].copy()
        data = data.sort_values("balanced_accuracy_drop")
        y = np.arange(len(data))
        colors = [
            "#E45756" if value > 0 else "#888888"
            for value in data["balanced_accuracy_drop"]
        ]
        ax.barh(y, data["balanced_accuracy_drop"], color=colors, height=0.65)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([domain_labels[d] for d in data["domain_removed"]])
        ax.set_title(cohort_labels[cohort], fontsize=11)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Balanced accuracy drop after removing domain")
    axes[1].set_xlabel("Balanced accuracy drop after removing domain")
    fig.suptitle("Which Political Opinion Domains Carry Voter-Intent Signal?", fontsize=14)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_voter_intent_domain_ablation.png", dpi=220)
    fig.savefig(FIG_DIR / "fig_voter_intent_domain_ablation.pdf")
    plt.close(fig)


def _lookup_metric(results_df: pd.DataFrame, cohort: str, config: str, metric: str) -> float:
    row = results_df[(results_df["cohort"] == cohort) & (results_df["config"] == config)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][metric])


def _lookup_diff(
    bootstrap_df: pd.DataFrame,
    cohort: str,
    reference: str,
    comparison: str,
    metric: str = "balanced_accuracy",
) -> pd.Series:
    row = bootstrap_df[
        (bootstrap_df["cohort"] == cohort)
        & (bootstrap_df["reference_config"] == reference)
        & (bootstrap_df["comparison_config"] == comparison)
        & (bootstrap_df["metric"] == metric)
    ]
    if row.empty:
        raise RuntimeError(f"Missing bootstrap row: {cohort} {reference} vs {comparison}")
    return row.iloc[0]


def write_readiness_report(
    results_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    domain_df: pd.DataFrame,
) -> None:
    all_issue_vs_demo = _lookup_diff(
        bootstrap_df, "all_respondents", "issues_only", "demographics_only"
    )
    all_issue_vs_ideology = _lookup_diff(
        bootstrap_df, "all_respondents", "issues_only", "ideology_only"
    )
    all_issue_vs_party = _lookup_diff(
        bootstrap_df, "all_respondents", "issues_only", "party_id_only"
    )
    all_full_vs_party = _lookup_diff(
        bootstrap_df, "all_respondents", "issues+party_id+ideology", "party_id_only"
    )
    all_full_vs_party_log_loss = _lookup_diff(
        bootstrap_df,
        "all_respondents",
        "issues+party_id+ideology",
        "party_id_only",
        metric="log_loss",
    )
    lean_issue_vs_demo = _lookup_diff(
        bootstrap_df, "independents_plus_leaners", "issues_only", "demographics_only"
    )
    lean_full_vs_party = _lookup_diff(
        bootstrap_df,
        "independents_plus_leaners",
        "issues+party_id+ideology",
        "party_id_only",
    )
    pure_n = int(sample_df.loc[sample_df["cohort"] == "pure_independents", "n"].iloc[0])
    domain_note = ""
    if not domain_df.empty:
        domain_labels = {
            "economic_welfare": "economic welfare",
            "race_dei": "race/DEI",
            "environment": "environment",
            "abortion": "abortion",
            "immigration_identity": "immigration/identity",
            "law_order_democracy": "law/order/democracy",
            "gender_lgbt": "gender/LGBT",
            "foreign_policy": "foreign policy",
        }
        top_domain_rows = (
            domain_df[domain_df["cohort"] == "all_respondents"]
            .sort_values("balanced_accuracy_drop", ascending=False)
            .head(3)
        )
        top_domains = [
            f"{domain_labels[row.domain_removed]} ({row.balanced_accuracy_drop:+.3f})"
            for row in top_domain_rows.itertuples()
        ]
        domain_note = (
            "- All-respondent domain ablation: largest balanced-accuracy drops "
            f"come from {', '.join(top_domains)}."
        )

    lines = [
        "# Thesis Readiness Report: Voter-Intent Model",
        "",
        "## Verdict",
        "",
        (
            "The model is thesis-ready if the claim is framed as: "
            "**political issue opinions are far more informative than demographics "
            "or ideology alone; party ID remains a stronger one-variable baseline "
            "among all respondents, but issue opinions improve the best non-vote "
            "model when combined with partisan identity.**"
        ),
        "",
        (
            "Do not phrase the claim as `issue opinions alone beat party ID among all "
            "respondents`. The trained evidence does not support that stronger version."
        ),
        "",
        "## Key 5-fold CV results",
        "",
        (
            f"- All respondents: issues_only balanced accuracy = "
            f"{_lookup_metric(results_df, 'all_respondents', 'issues_only', 'balanced_accuracy'):.3f}; "
            f"demographics_only = "
            f"{_lookup_metric(results_df, 'all_respondents', 'demographics_only', 'balanced_accuracy'):.3f}; "
            f"ideology_only = "
            f"{_lookup_metric(results_df, 'all_respondents', 'ideology_only', 'balanced_accuracy'):.3f}; "
            f"party_id_only = "
            f"{_lookup_metric(results_df, 'all_respondents', 'party_id_only', 'balanced_accuracy'):.3f}."
        ),
        (
            f"- All respondents: issues+party_id+ideology balanced accuracy = "
            f"{_lookup_metric(results_df, 'all_respondents', 'issues+party_id+ideology', 'balanced_accuracy'):.3f}."
        ),
        (
            f"- Independents plus leaners: issues_only balanced accuracy = "
            f"{_lookup_metric(results_df, 'independents_plus_leaners', 'issues_only', 'balanced_accuracy'):.3f}; "
            f"demographics_only = "
            f"{_lookup_metric(results_df, 'independents_plus_leaners', 'demographics_only', 'balanced_accuracy'):.3f}; "
            f"party_id_only = "
            f"{_lookup_metric(results_df, 'independents_plus_leaners', 'party_id_only', 'balanced_accuracy'):.3f}."
        ),
        (
            f"- Pure independents remain exploratory only: n = {pure_n}. "
            "Use this cohort as a robustness check, not the headline estimate."
        ),
        domain_note,
        "",
        "## Bootstrap claim checks",
        "",
        (
            f"- issues_only minus demographics_only, all respondents: "
            f"delta BA = {all_issue_vs_demo['diff_reference_minus_comparison']:+.3f}, "
            f"95% CI [{all_issue_vs_demo['ci_low']:+.3f}, {all_issue_vs_demo['ci_high']:+.3f}], "
            f"p = {all_issue_vs_demo['bootstrap_p_two_sided']:.4f}."
        ),
        (
            f"- issues_only minus ideology_only, all respondents: "
            f"delta BA = {all_issue_vs_ideology['diff_reference_minus_comparison']:+.3f}, "
            f"95% CI [{all_issue_vs_ideology['ci_low']:+.3f}, {all_issue_vs_ideology['ci_high']:+.3f}], "
            f"p = {all_issue_vs_ideology['bootstrap_p_two_sided']:.4f}."
        ),
        (
            f"- issues_only minus party_id_only, all respondents: "
            f"delta BA = {all_issue_vs_party['diff_reference_minus_comparison']:+.3f}, "
            f"95% CI [{all_issue_vs_party['ci_low']:+.3f}, {all_issue_vs_party['ci_high']:+.3f}], "
            f"p = {all_issue_vs_party['bootstrap_p_two_sided']:.4f}."
        ),
        (
            f"- issues+party_id+ideology minus party_id_only, all respondents "
            f"(balanced accuracy): "
            f"delta BA = {all_full_vs_party['diff_reference_minus_comparison']:+.3f}, "
            f"95% CI [{all_full_vs_party['ci_low']:+.3f}, {all_full_vs_party['ci_high']:+.3f}], "
            f"p = {all_full_vs_party['bootstrap_p_two_sided']:.4f}."
        ),
        (
            f"- issues+party_id+ideology minus party_id_only, all respondents "
            f"(log loss; lower is better): "
            f"delta = {all_full_vs_party_log_loss['diff_reference_minus_comparison']:+.3f}, "
            f"95% CI [{all_full_vs_party_log_loss['ci_low']:+.3f}, "
            f"{all_full_vs_party_log_loss['ci_high']:+.3f}], "
            f"p = {all_full_vs_party_log_loss['bootstrap_p_two_sided']:.4f}."
        ),
        (
            f"- issues_only minus demographics_only, independents plus leaners: "
            f"delta BA = {lean_issue_vs_demo['diff_reference_minus_comparison']:+.3f}, "
            f"95% CI [{lean_issue_vs_demo['ci_low']:+.3f}, {lean_issue_vs_demo['ci_high']:+.3f}], "
            f"p = {lean_issue_vs_demo['bootstrap_p_two_sided']:.4f}."
        ),
        (
            f"- issues+party_id+ideology minus party_id_only, independents plus leaners: "
            f"delta BA = {lean_full_vs_party['diff_reference_minus_comparison']:+.3f}, "
            f"95% CI [{lean_full_vs_party['ci_low']:+.3f}, {lean_full_vs_party['ci_high']:+.3f}], "
            f"p = {lean_full_vs_party['bootstrap_p_two_sided']:.4f}."
        ),
        "",
        "## Manuscript-safe wording",
        "",
        (
            "A defensible version is: `Across ANES 2024 respondents, pre-election "
            "issue opinions predict Harris-versus-Trump vote intent much better than "
            "demographic variables or ideological self-placement alone. Party "
            "identification remains the strongest compact baseline among all "
            "respondents, but issue opinions improve the combined non-vote model, "
            "especially in probability quality and among independents plus leaners. "
            "The best formulation is therefore not demographics -> vote, but "
            "political attitudes and partisan identity -> vote intent.`"
        ),
        "",
        "## Outputs",
        "",
        "- `results/voter_intent/cohort_results.csv`",
        "- `results/voter_intent/pairwise_bootstrap.csv`",
        "- `results/voter_intent/domain_ablation.csv`",
        "- `figures/fig_voter_intent_claim_head_to_head.png`",
        "- `figures/fig_voter_intent_domain_ablation.png`",
    ]
    (OUT_DIR / "thesis_readiness_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(n_bootstrap: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_analysis_data()

    metadata = {
        "target": TARGET_CODE,
        "target_description": "Pre-election summary of presidential vote, intent, or preference; Harris/Trump only",
        "weight": WEIGHT_CODE,
        "model": "XGBClassifier fixed across feature configurations",
        "n_folds": N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "feature_configs": FEATURE_CONFIGS,
        "issue_domains": ISSUE_DOMAINS,
        "issue_feature_codes": ISSUE_MAP,
        "demographic_feature_codes": DEMOGRAPHIC_MAP,
        "ideology_feature_codes": IDEOLOGY_MAP,
        "party_id_code": PARTY_ID_CODE,
        "notes": [
            "Direct vote/intent/preference is the target and is never used as a predictor.",
            "Issue features are pre-election respondent positions; candidate placements and party-ownership items are excluded.",
            "Party ID is reported separately because it is not a direct vote-intent question but is a strong political identity baseline.",
        ],
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_results = []
    all_fold_metrics = []
    all_samples = []
    all_bootstrap = []
    all_domain_ablation = []
    result_objects: dict[tuple[str, str], CvResult] = {}

    party_values_by_cohort = {
        "all_respondents": None,
        "pure_independents": [4],
        "independents_plus_leaners": [3, 4, 5],
    }

    for cohort, mask_fn in COHORTS.items():
        cohort_df = df.loc[mask_fn(df)].copy().reset_index(drop=True)
        print(f"\n== {cohort} n={len(cohort_df)} ==", flush=True)
        all_samples.append(
            summarize_sample(cohort_df, cohort, party_values_by_cohort[cohort])
        )
        for config, cols in FEATURE_CONFIGS.items():
            result = evaluate_config(cohort_df, cohort, config, cols)
            result_objects[(cohort, config)] = result
            all_results.append({
                "cohort": cohort,
                "config": config,
                "n": len(result.y_true),
                "n_features_raw": len(cols),
                **result.metrics,
            })
            for fold in range(N_FOLDS):
                fold_mask = result.fold_id == fold
                all_fold_metrics.append({
                    "cohort": cohort,
                    "config": config,
                    "fold": int(fold),
                    "n": int(fold_mask.sum()),
                    **weighted_metrics(
                        result.y_true[fold_mask],
                        result.y_pred[fold_mask],
                        result.y_proba[fold_mask],
                        result.sample_weight[fold_mask],
                    ),
                })

        for reference, comparison in KEY_BOOTSTRAP_PAIRS:
            all_bootstrap.extend(
                bootstrap_differences(
                    result_objects[(cohort, reference)],
                    result_objects[(cohort, comparison)],
                    n_bootstrap,
                )
            )

        if cohort in DOMAIN_ABLATION_COHORTS:
            print(f"  {cohort:28s} domain ablation on issues+ideology", flush=True)
            all_domain_ablation.extend(
                run_domain_ablation(
                    cohort_df,
                    cohort,
                    result_objects[(cohort, "issues+ideology")],
                    n_bootstrap,
                )
            )

    results_df = pd.DataFrame(all_results)
    fold_df = pd.DataFrame(all_fold_metrics)
    sample_df = pd.DataFrame(all_samples)
    bootstrap_df = pd.DataFrame(all_bootstrap)
    domain_df = pd.DataFrame(all_domain_ablation)

    results_df.to_csv(OUT_DIR / "cohort_results.csv", index=False)
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    sample_df.to_csv(OUT_DIR / "sample_summary.csv", index=False)
    bootstrap_df.to_csv(OUT_DIR / "pairwise_bootstrap.csv", index=False)
    domain_df.to_csv(OUT_DIR / "domain_ablation.csv", index=False)

    plot_head_to_head(results_df)
    plot_domain_ablation(domain_df)
    write_readiness_report(results_df, bootstrap_df, sample_df, domain_df)

    print("\nWrote:")
    for path in [
        OUT_DIR / "cohort_results.csv",
        OUT_DIR / "fold_metrics.csv",
        OUT_DIR / "pairwise_bootstrap.csv",
        OUT_DIR / "domain_ablation.csv",
        OUT_DIR / "sample_summary.csv",
        OUT_DIR / "metadata.json",
        OUT_DIR / "thesis_readiness_report.md",
        FIG_DIR / "fig_voter_intent_claim_head_to_head.png",
        FIG_DIR / "fig_voter_intent_claim_head_to_head.pdf",
        FIG_DIR / "fig_voter_intent_domain_ablation.png",
        FIG_DIR / "fig_voter_intent_domain_ablation.pdf",
    ]:
        print(f"  {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help="Bootstrap resamples for paired model comparisons.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n_bootstrap=args.bootstrap)
