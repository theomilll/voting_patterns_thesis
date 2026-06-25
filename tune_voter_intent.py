"""Model benchmark for the pre-election voter-intent thesis analysis.

This script does not replace the canonical voter-intent artifacts. It tests
alternative learners under the same weighted, fold-local 5-fold CV contract as
``run_voter_intent_analysis.py`` and writes a separate tuning evidence bundle.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

import run_voter_intent_analysis as voter


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


OUT_DIR = Path("results/voter_intent_tuning")
BASELINE_CANDIDATE = "xgb_current"
N_INNER_FOLDS = 3
CALIBRATION_METHODS = ["none", "sigmoid", "isotonic"]
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 181), 3)
MISSINGNESS_VARIANTS = ["baseline", "with_missingness_indicators"]
XGB_NAN_VARIANTS = ["median_impute_scale", "native_nan_no_scale"]
CLASS_BALANCE_VARIANTS = ["survey_weight_only", "survey_x_fold_class_balance"]
TRAINING_WEIGHT_VARIANTS = [
    "survey_weight",
    "unweighted_equal",
    "sqrt_survey_weight",
    "clipped_survey_weight_p95",
]
TWO_STAGE_CONFIGS = [
    "predicted_issues_only",
    "predicted_issues+demographics",
    "predicted_issues+party_id+ideology",
]
TWO_STAGE_STAGE1_MODELS = [
    "histgb_current",
    "histgb_smooth",
    "ridge_l2",
    "random_forest_reg",
    "extra_trees_reg",
]
TWO_STAGE_STAGE2_CANDIDATES = [
    BASELINE_CANDIDATE,
    "xgb_shallow_reg",
    "logit_l2_bal",
    "rf_leaf10",
    "extra_trees",
    "histgb",
]
DOMAIN_AGGREGATE_CONFIGS = [
    "domain_issues_only",
    "domain_issues+demographics",
    "domain_issues+party_id+ideology",
]
DOMAIN_PRUNING_CONFIGS = [
    "issues_only",
    "issues+party_id+ideology",
    "issues+demographics+ideology",
]
ISSUE_PCA_CONFIGS = [
    "pca_issues_only",
    "pca_issues+party_id+ideology",
    "pca_issues+demographics+ideology",
]
ISSUE_PCA_COMPONENTS = [5, 10, 20, 30]
ISSUE_TOPK_CONFIG_MAP = {
    "topk_issues_only": "issues_only",
    "topk_issues+party_id+ideology": "issues+party_id+ideology",
    "topk_issues+demographics+ideology": "issues+demographics+ideology",
}
ISSUE_TOPK_CONFIGS = list(ISSUE_TOPK_CONFIG_MAP)
ISSUE_TOPK_VALUES = [5, 10, 15, 25, 35]
ISSUE_ENCODING_CONFIG_MAP = {
    "issue_categorical_issues_only": "issues_only",
    "issue_categorical+party_id+ideology": "issues+party_id+ideology",
    "issue_categorical+demographics+ideology": "issues+demographics+ideology",
}
ISSUE_ENCODING_CONFIGS = list(ISSUE_ENCODING_CONFIG_MAP)
ISSUE_ENCODING_VARIANTS = ["numeric_ordinal", "issue_onehot"]
POLITICAL_ENCODING_CONFIG_MAP = {
    "political_categorical_party_id_only": "party_id_only",
    "political_categorical_party_id+ideology": "party_id+ideology",
    "issues+political_categorical": "issues+party_id+ideology",
}
POLITICAL_ENCODING_CONFIGS = list(POLITICAL_ENCODING_CONFIG_MAP)
POLITICAL_ENCODING_VARIANTS = ["numeric_ordinal", "political_onehot"]
XGB_EARLY_STOP_VARIANTS = ["fixed_300", "early_stop_1000_15pct"]
XGB_OPTUNA_DEFAULT_CONFIGS = ["issues+party_id+ideology"]
XGB_OPTUNA_DEFAULT_TRIALS = 12
XGB_OPTUNA_VARIANTS = ["fixed_300", "optuna_xgb_balanced_accuracy"]
XGB_SEED_BAG_CONFIGS = [
    "issues_only",
    "issues+party_id+ideology",
    "issues+demographics+ideology",
]
XGB_SEED_BAG_SEEDS = [42, 43, 44, 45, 46]
XGB_SEED_BAG_VARIANTS = ["xgb_current", "xgb_seed_bag_5"]
XGB_PROFILE_THRESHOLD_CONFIGS = [
    "demographics_only",
    "ideology_only",
    "party_id_only",
    "issues_only",
    "issues+party_id+ideology",
    "issues+demographics+ideology",
]
XGB_PROFILE_THRESHOLD_CANDIDATES = [
    BASELINE_CANDIDATE,
    "xgb_shallow_reg",
    "xgb_depth2_subsample",
    "xgb_stump_reg",
]
XGB_PROFILE_CALIBRATION_CONFIGS = list(XGB_PROFILE_THRESHOLD_CONFIGS)
XGB_PROFILE_CALIBRATION_CANDIDATES = [
    "xgb_shallow_reg",
    "xgb_depth2_subsample",
    "xgb_stump_reg",
]
XGB_PROFILE_CALIBRATION_METHODS = ["sigmoid", "isotonic"]
XGB_PROFILE_CALIBRATION_THRESHOLD_CONFIGS = list(XGB_PROFILE_CALIBRATION_CONFIGS)
XGB_PROFILE_CALIBRATION_THRESHOLD_CANDIDATES = list(
    XGB_PROFILE_CALIBRATION_CANDIDATES
)
XGB_PROFILE_CALIBRATION_THRESHOLD_METHODS = list(XGB_PROFILE_CALIBRATION_METHODS)
XGB_PROFILE_BLEND_CONFIGS = list(XGB_PROFILE_THRESHOLD_CONFIGS)
XGB_PROFILE_BLEND_RECIPES = {
    "xgb80+shallow20": [
        (BASELINE_CANDIDATE, 0.80),
        ("xgb_shallow_reg", 0.20),
    ],
    "xgb65+shallow35": [
        (BASELINE_CANDIDATE, 0.65),
        ("xgb_shallow_reg", 0.35),
    ],
    "xgb50+shallow50": [
        (BASELINE_CANDIDATE, 0.50),
        ("xgb_shallow_reg", 0.50),
    ],
    "xgb80+depth2_20": [
        (BASELINE_CANDIDATE, 0.80),
        ("xgb_depth2_subsample", 0.20),
    ],
    "xgb65+depth2_35": [
        (BASELINE_CANDIDATE, 0.65),
        ("xgb_depth2_subsample", 0.35),
    ],
    "xgb50+depth2_50": [
        (BASELINE_CANDIDATE, 0.50),
        ("xgb_depth2_subsample", 0.50),
    ],
    "xgb80+stump20": [
        (BASELINE_CANDIDATE, 0.80),
        ("xgb_stump_reg", 0.20),
    ],
    "xgb65+stump35": [
        (BASELINE_CANDIDATE, 0.65),
        ("xgb_stump_reg", 0.35),
    ],
    "xgb70+shallow15+depth2_15": [
        (BASELINE_CANDIDATE, 0.70),
        ("xgb_shallow_reg", 0.15),
        ("xgb_depth2_subsample", 0.15),
    ],
    "xgb70+depth2_15+stump15": [
        (BASELINE_CANDIDATE, 0.70),
        ("xgb_depth2_subsample", 0.15),
        ("xgb_stump_reg", 0.15),
    ],
}
XGB_PROFILE_BLEND_THRESHOLD_CONFIGS = [
    "issues+party_id+ideology",
    "issues+demographics+ideology",
]
XGB_PROFILE_BLEND_THRESHOLD_RECIPES = [
    "xgb65+shallow35",
    "xgb50+shallow50",
    "xgb65+stump35",
    "xgb80+shallow20",
    "xgb80+depth2_20",
    "xgb65+depth2_35",
]
EXTENDED_CANDIDATE_CONFIGS = [
    "demographics_only",
    "ideology_only",
    "party_id_only",
    "issues_only",
    "issues+party_id+ideology",
    "issues+demographics+ideology",
]
EXTENDED_CANDIDATES = [
    "rf_leaf10",
    "extra_trees",
    "logit_l2_bal",
    "histgb",
]
EXTENDED_CANDIDATE_THRESHOLD_CONFIGS = ["issues+party_id+ideology"]
EXTENDED_CANDIDATE_THRESHOLD_CANDIDATES = list(EXTENDED_CANDIDATES)
EXTENDED_CANDIDATE_CALIBRATION_CONFIGS = ["issues+party_id+ideology"]
EXTENDED_CANDIDATE_CALIBRATION_CANDIDATES = list(EXTENDED_CANDIDATES)
EXTENDED_CANDIDATE_CALIBRATION_METHODS = ["sigmoid", "isotonic"]
EXTENDED_CANDIDATE_CALIBRATION_THRESHOLD_CONFIGS = ["issues+party_id+ideology"]
EXTENDED_CANDIDATE_CALIBRATION_THRESHOLD_CANDIDATES = list(EXTENDED_CANDIDATES)
EXTENDED_CANDIDATE_CALIBRATION_THRESHOLD_METHODS = ["sigmoid", "isotonic"]
XGB_PROFILE_CANDIDATES = [
    BASELINE_CANDIDATE,
    "xgb_shallow_reg",
    "xgb_more_trees",
    "xgb_depth4",
    "xgb_stump_reg",
    "xgb_depth2_subsample",
    "xgb_depth3_heavy_reg",
    "xgb_depth4_shrinkage",
]
DEFAULT_CANDIDATES = [
    BASELINE_CANDIDATE,
    "xgb_shallow_reg",
    "xgb_more_trees",
    "xgb_depth4",
    "rf_balanced",
    "logit_elastic",
]
FIXED_CANDIDATE_BOOTSTRAP_CANDIDATES = [
    "xgb_shallow_reg",
    "xgb_more_trees",
    "xgb_depth4",
    "rf_balanced",
    "logit_elastic",
]
FEATURE_BOOTSTRAP_PAIRS = [
    ("issues+party_id", "issues_only"),
    ("issues+ideology", "issues_only"),
    ("issues+demographics", "issues_only"),
    ("issues+party_id+ideology", "issues+party_id"),
    ("issues+party_id+ideology", "issues+ideology"),
    ("issues+demographics+ideology", "issues+demographics"),
    ("issues+demographics+ideology", "issues+ideology"),
    ("issues+party_id+ideology", "issues+demographics+ideology"),
]
ENSEMBLE_RECIPES = {
    "xgb_current": [(BASELINE_CANDIDATE, 1.0)],
    "xgb_current+shallow": [(BASELINE_CANDIDATE, 0.5), ("xgb_shallow_reg", 0.5)],
    "xgb_current+logit": [(BASELINE_CANDIDATE, 0.5), ("logit_elastic", 0.5)],
    "xgb_current+rf": [(BASELINE_CANDIDATE, 0.5), ("rf_balanced", 0.5)],
    "xgb_current+shallow+logit": [
        (BASELINE_CANDIDATE, 1 / 3),
        ("xgb_shallow_reg", 1 / 3),
        ("logit_elastic", 1 / 3),
    ],
}
ENSEMBLE_BOOTSTRAP_RECIPES = [
    "xgb_current+logit",
    "xgb_current+shallow+logit",
]
ENSEMBLE_THRESHOLD_RECIPES = ENSEMBLE_BOOTSTRAP_RECIPES
CALIBRATED_EXTENDED_BLEND_CONFIGS = ["issues+party_id+ideology"]
CALIBRATED_EXTENDED_BLEND_RECIPES = {
    "xgb_current+logit_l2_sigmoid": [
        (BASELINE_CANDIDATE, "none", 0.5),
        ("logit_l2_bal", "sigmoid", 0.5),
    ],
    "xgb_current+histgb_sigmoid": [
        (BASELINE_CANDIDATE, "none", 0.5),
        ("histgb", "sigmoid", 0.5),
    ],
    "xgb_current+extra_trees_sigmoid": [
        (BASELINE_CANDIDATE, "none", 0.5),
        ("extra_trees", "sigmoid", 0.5),
    ],
    "xgb_current+logit_l2+histgb_sigmoid": [
        (BASELINE_CANDIDATE, "none", 1 / 3),
        ("logit_l2_bal", "sigmoid", 1 / 3),
        ("histgb", "sigmoid", 1 / 3),
    ],
}
CALIBRATED_EXTENDED_BLEND_THRESHOLD_CONFIGS = ["issues+party_id+ideology"]
CALIBRATED_EXTENDED_BLEND_THRESHOLD_RECIPES = list(CALIBRATED_EXTENDED_BLEND_RECIPES)
CALIBRATED_EXTENDED_BLEND_GRID_CONFIGS = ["issues+party_id+ideology"]
CALIBRATED_EXTENDED_BLEND_GRID_RECIPES = {
    "xgb80+logit_l2_sigmoid20": [
        (BASELINE_CANDIDATE, "none", 0.80),
        ("logit_l2_bal", "sigmoid", 0.20),
    ],
    "xgb65+logit_l2_sigmoid35": [
        (BASELINE_CANDIDATE, "none", 0.65),
        ("logit_l2_bal", "sigmoid", 0.35),
    ],
    "xgb35+logit_l2_sigmoid65": [
        (BASELINE_CANDIDATE, "none", 0.35),
        ("logit_l2_bal", "sigmoid", 0.65),
    ],
    "xgb20+logit_l2_sigmoid80": [
        (BASELINE_CANDIDATE, "none", 0.20),
        ("logit_l2_bal", "sigmoid", 0.80),
    ],
    "xgb80+histgb_sigmoid20": [
        (BASELINE_CANDIDATE, "none", 0.80),
        ("histgb", "sigmoid", 0.20),
    ],
    "xgb65+histgb_sigmoid35": [
        (BASELINE_CANDIDATE, "none", 0.65),
        ("histgb", "sigmoid", 0.35),
    ],
    "xgb35+histgb_sigmoid65": [
        (BASELINE_CANDIDATE, "none", 0.35),
        ("histgb", "sigmoid", 0.65),
    ],
    "xgb20+histgb_sigmoid80": [
        (BASELINE_CANDIDATE, "none", 0.20),
        ("histgb", "sigmoid", 0.80),
    ],
    "xgb80+extra_trees_sigmoid20": [
        (BASELINE_CANDIDATE, "none", 0.80),
        ("extra_trees", "sigmoid", 0.20),
    ],
    "xgb65+extra_trees_sigmoid35": [
        (BASELINE_CANDIDATE, "none", 0.65),
        ("extra_trees", "sigmoid", 0.35),
    ],
    "xgb35+extra_trees_sigmoid65": [
        (BASELINE_CANDIDATE, "none", 0.35),
        ("extra_trees", "sigmoid", 0.65),
    ],
    "xgb20+extra_trees_sigmoid80": [
        (BASELINE_CANDIDATE, "none", 0.20),
        ("extra_trees", "sigmoid", 0.80),
    ],
    "xgb70+logit_l2_sigmoid15+histgb_sigmoid15": [
        (BASELINE_CANDIDATE, "none", 0.70),
        ("logit_l2_bal", "sigmoid", 0.15),
        ("histgb", "sigmoid", 0.15),
    ],
    "xgb50+logit_l2_sigmoid25+histgb_sigmoid25": [
        (BASELINE_CANDIDATE, "none", 0.50),
        ("logit_l2_bal", "sigmoid", 0.25),
        ("histgb", "sigmoid", 0.25),
    ],
    "xgb30+logit_l2_sigmoid35+histgb_sigmoid35": [
        (BASELINE_CANDIDATE, "none", 0.30),
        ("logit_l2_bal", "sigmoid", 0.35),
        ("histgb", "sigmoid", 0.35),
    ],
}
CALIBRATED_EXTENDED_BLEND_GRID_CORE_CONFIGS = list(EXTENDED_CANDIDATE_CONFIGS)
CALIBRATED_EXTENDED_BLEND_GRID_CORE_RECIPES = [
    "xgb65+histgb_sigmoid35",
    "xgb70+logit_l2_sigmoid15+histgb_sigmoid15",
]


def candidate_factories() -> dict[str, Callable[[], object]]:
    return {
        BASELINE_CANDIDATE: lambda: XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=2.0,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "xgb_shallow_reg": lambda: XGBClassifier(
            n_estimators=500,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "xgb_more_trees": lambda: XGBClassifier(
            n_estimators=650,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=3.0,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "xgb_depth4": lambda: XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            reg_lambda=3.0,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "xgb_stump_reg": lambda: XGBClassifier(
            n_estimators=900,
            max_depth=1,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_alpha=0.1,
            reg_lambda=8.0,
            gamma=0.1,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "xgb_depth2_subsample": lambda: XGBClassifier(
            n_estimators=700,
            max_depth=2,
            learning_rate=0.025,
            subsample=0.65,
            colsample_bytree=0.65,
            min_child_weight=6,
            reg_alpha=0.2,
            reg_lambda=6.0,
            gamma=0.1,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "xgb_depth3_heavy_reg": lambda: XGBClassifier(
            n_estimators=550,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=8,
            reg_alpha=0.5,
            reg_lambda=8.0,
            gamma=0.2,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "xgb_depth4_shrinkage": lambda: XGBClassifier(
            n_estimators=650,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=5,
            reg_alpha=0.2,
            reg_lambda=6.0,
            gamma=0.3,
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "rf_balanced": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
        ),
        "rf_leaf10": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
        ),
        "logit_l2_bal": lambda: LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            random_state=voter.RANDOM_STATE,
        ),
        "logit_elastic": lambda: LogisticRegression(
            max_iter=5000,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.2,
            C=0.3,
            class_weight="balanced",
            random_state=voter.RANDOM_STATE,
        ),
        "histgb": lambda: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=voter.RANDOM_STATE,
        ),
    }


def stage1_regressor_factories() -> dict[str, Callable[[], object]]:
    return {
        "histgb_current": lambda: HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=voter.RANDOM_STATE,
        ),
        "histgb_smooth": lambda: HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.03,
            max_leaf_nodes=8,
            l2_regularization=1.0,
            random_state=voter.RANDOM_STATE,
        ),
        "ridge_l2": lambda: Ridge(alpha=10.0, random_state=voter.RANDOM_STATE),
        "random_forest_reg": lambda: RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees_reg": lambda: ExtraTreesRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=voter.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def encode_feature_frame_with_missingness(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    encoded = voter.encode_feature_frame(df, feature_cols)
    indicators = df[feature_cols].isna().astype(float)
    indicators = indicators.rename(
        columns={col: f"{col}__missing" for col in indicators.columns}
    )
    return pd.concat([encoded, indicators], axis=1).astype(float)


def encode_feature_frame_with_issue_categories(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    x = df[feature_cols].copy()
    categorical_cols = [
        col
        for col in feature_cols
        if col in voter.NOMINAL_COLS or col in voter.ISSUE_COLS
    ]
    if categorical_cols:
        x = pd.get_dummies(
            x,
            columns=categorical_cols,
            dummy_na=True,
            prefix_sep="=",
            dtype=float,
        )
    return x.astype(float)


def encode_feature_frame_with_political_categories(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    x = df[feature_cols].copy()
    political_cols = {voter.PARTY_ID_COL, *voter.IDEOLOGY_COLS}
    categorical_cols = [
        col
        for col in feature_cols
        if col in voter.NOMINAL_COLS or col in political_cols
    ]
    if categorical_cols:
        x = pd.get_dummies(
            x,
            columns=categorical_cols,
            dummy_na=True,
            prefix_sep="=",
            dtype=float,
        )
    return x.astype(float)


def evaluate_candidate_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    candidate: str,
    feature_cols: list[str],
    make_model: Callable[[], object],
) -> voter.CvResult:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_base, y)):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train, x_val = voter.impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model = make_model()
        try:
            model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        except TypeError:
            model.fit(x_train, y[train_idx])

        oof_pred[val_idx] = model.predict(x_val)
        oof_proba[val_idx] = model.predict_proba(x_val)
        fold_id[val_idx] = fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{candidate}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def evaluate_candidate(
    df: pd.DataFrame,
    feature_cols: list[str],
    make_model: Callable[[], object],
) -> dict[str, float]:
    return evaluate_candidate_result(
        df=df,
        cohort="cohort",
        config="config",
        candidate="candidate",
        feature_cols=feature_cols,
        make_model=make_model,
    ).metrics


def evaluate_ensemble_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    ensemble_name: str,
    recipe: list[tuple[str, float]],
) -> voter.CvResult:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    factories = candidate_factories()
    total_weight = sum(weight for _, weight in recipe)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_base, y)):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train, x_val = voter.impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        proba = np.zeros((len(val_idx), 2), dtype=float)
        for model_name, weight in recipe:
            model = factories[model_name]()
            model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
            proba += (weight / total_weight) * model.predict_proba(x_val)
        oof_proba[val_idx] = proba
        oof_pred[val_idx] = np.argmax(proba, axis=1)
        fold_id[val_idx] = fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{ensemble_name}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def evaluate_ensemble(
    df: pd.DataFrame,
    feature_cols: list[str],
    recipe: list[tuple[str, float]],
) -> dict[str, float]:
    return evaluate_ensemble_result(
        df=df,
        cohort="cohort",
        config="config",
        feature_cols=feature_cols,
        ensemble_name="ensemble",
        recipe=recipe,
    ).metrics


def evaluate_missingness_current(
    df: pd.DataFrame,
    feature_cols: list[str],
    add_missingness_indicators: bool,
) -> dict[str, float]:
    """Evaluate fixed XGBoost with optional row-level missingness indicators."""
    if not add_missingness_indicators:
        return evaluate_candidate(df, feature_cols, candidate_factories()[BASELINE_CANDIDATE])

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = encode_feature_frame_with_missingness(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    factory = candidate_factories()[BASELINE_CANDIDATE]

    for train_idx, val_idx in splitter.split(x_base, y):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train, x_val = voter.impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model = factory()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        oof_pred[val_idx] = model.predict(x_val)
        oof_proba[val_idx] = model.predict_proba(x_val)

    return voter.weighted_metrics(y, oof_pred, oof_proba, weights)


def evaluate_xgb_nan_current(
    df: pd.DataFrame,
    feature_cols: list[str],
    native_nan_no_scale: bool,
) -> dict[str, float]:
    """Evaluate fixed XGBoost with canonical preprocessing or native NaNs."""
    if not native_nan_no_scale:
        return evaluate_candidate(df, feature_cols, candidate_factories()[BASELINE_CANDIDATE])

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    factory = candidate_factories()[BASELINE_CANDIDATE]

    for train_idx, val_idx in splitter.split(x_base, y):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        model = factory()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        oof_pred[val_idx] = model.predict(x_val)
        oof_proba[val_idx] = model.predict_proba(x_val)

    return voter.weighted_metrics(y, oof_pred, oof_proba, weights)


def _fold_class_balanced_weights(y_train: np.ndarray, survey_weights: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    total_weight = float(survey_weights.sum())
    adjusted = survey_weights.astype(float).copy()
    for cls in classes:
        mask = y_train == cls
        class_weight = float(survey_weights[mask].sum())
        if class_weight <= 0:
            raise RuntimeError(f"Non-positive training weight for class {cls}")
        adjusted[mask] *= total_weight / (len(classes) * class_weight)
    return adjusted


def evaluate_class_balanced_current(
    df: pd.DataFrame,
    feature_cols: list[str],
    class_balance_training: bool,
) -> dict[str, float]:
    """Evaluate fixed XGBoost with optional fold-local class-balanced train weights."""
    if not class_balance_training:
        return evaluate_candidate(df, feature_cols, candidate_factories()[BASELINE_CANDIDATE])

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    factory = candidate_factories()[BASELINE_CANDIDATE]

    for train_idx, val_idx in splitter.split(x_base, y):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train, x_val = voter.impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        train_weights = _fold_class_balanced_weights(
            y[train_idx],
            weights[train_idx],
        )
        model = factory()
        model.fit(x_train, y[train_idx], sample_weight=train_weights)
        oof_pred[val_idx] = model.predict(x_val)
        oof_proba[val_idx] = model.predict_proba(x_val)

    return voter.weighted_metrics(y, oof_pred, oof_proba, weights)


def _training_weights_for_variant(
    survey_weights: np.ndarray,
    variant: str,
) -> np.ndarray:
    if variant == "survey_weight":
        adjusted = survey_weights.astype(float).copy()
    elif variant == "unweighted_equal":
        adjusted = np.ones_like(survey_weights, dtype=float)
    elif variant == "sqrt_survey_weight":
        adjusted = np.sqrt(survey_weights.astype(float))
    elif variant == "clipped_survey_weight_p95":
        cap = float(np.percentile(survey_weights, 95))
        adjusted = np.minimum(survey_weights.astype(float), cap)
    else:
        raise RuntimeError(f"Unknown training-weight variant: {variant}")

    original_sum = float(survey_weights.sum())
    adjusted_sum = float(adjusted.sum())
    if adjusted_sum <= 0:
        raise RuntimeError(f"Non-positive adjusted weight sum for {variant}")
    return adjusted * (original_sum / adjusted_sum)


def evaluate_training_weight_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    variant: str,
) -> voter.CvResult:
    """Evaluate fixed XGBoost with alternative train-fold weight patterns."""
    if variant == "survey_weight":
        return evaluate_candidate_result(
            df=df,
            cohort=cohort,
            config=config,
            candidate=f"{BASELINE_CANDIDATE}_{variant}",
            feature_cols=feature_cols,
            make_model=candidate_factories()[BASELINE_CANDIDATE],
        )

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    factory = candidate_factories()[BASELINE_CANDIDATE]

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_base, y)):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train, x_val = voter.impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        train_weights = _training_weights_for_variant(weights[train_idx], variant)
        model = factory()
        model.fit(x_train, y[train_idx], sample_weight=train_weights)
        oof_pred[val_idx] = model.predict(x_val)
        oof_proba[val_idx] = model.predict_proba(x_val)
        fold_id[val_idx] = fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{variant}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def evaluate_training_weight_current(
    df: pd.DataFrame,
    feature_cols: list[str],
    variant: str,
) -> dict[str, float]:
    return evaluate_training_weight_result(
        df=df,
        cohort="cohort",
        config="config",
        feature_cols=feature_cols,
        variant=variant,
    ).metrics


def evaluate_issue_encoding_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    canonical_config: str,
    feature_cols: list[str],
    encoding: str,
) -> voter.CvResult:
    if encoding == "numeric_ordinal":
        return evaluate_candidate_result(
            df=df,
            cohort=cohort,
            config=canonical_config,
            candidate=f"{BASELINE_CANDIDATE}_{encoding}",
            feature_cols=feature_cols,
            make_model=candidate_factories()[BASELINE_CANDIDATE],
        )
    if encoding != "issue_onehot":
        raise RuntimeError(f"Unknown issue encoding: {encoding}")

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = encode_feature_frame_with_issue_categories(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    factory = candidate_factories()[BASELINE_CANDIDATE]

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_base, y)):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train, x_val = voter.impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model = factory()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        oof_pred[val_idx] = model.predict(x_val)
        oof_proba[val_idx] = model.predict_proba(x_val)
        fold_id[val_idx] = fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{encoding}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def evaluate_political_encoding_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    canonical_config: str,
    feature_cols: list[str],
    encoding: str,
) -> voter.CvResult:
    if encoding == "numeric_ordinal":
        return evaluate_candidate_result(
            df=df,
            cohort=cohort,
            config=canonical_config,
            candidate=f"{BASELINE_CANDIDATE}_{encoding}",
            feature_cols=feature_cols,
            make_model=candidate_factories()[BASELINE_CANDIDATE],
        )
    if encoding != "political_onehot":
        raise RuntimeError(f"Unknown political encoding: {encoding}")

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = encode_feature_frame_with_political_categories(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    factory = candidate_factories()[BASELINE_CANDIDATE]

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_base, y)):
        x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train, x_val = voter.impute_fold(x_train, x_val)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model = factory()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        oof_pred[val_idx] = model.predict(x_val)
        oof_proba[val_idx] = model.predict_proba(x_val)
        fold_id[val_idx] = fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{encoding}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def _xgb_early_stop_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=2.0,
        random_state=voter.RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=25,
    )


def evaluate_xgb_early_stop_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    variant: str,
) -> tuple[voter.CvResult, list[dict[str, object]], float, float]:
    if variant == "fixed_300":
        result = evaluate_candidate_result(
            df=df,
            cohort=cohort,
            config=config,
            candidate=BASELINE_CANDIDATE,
            feature_cols=feature_cols,
            make_model=candidate_factories()[BASELINE_CANDIDATE],
        )
        return result, [], 300.0, 300.0
    if variant != "early_stop_1000_15pct":
        raise RuntimeError(f"Unknown XGBoost early-stop variant: {variant}")

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    fold_rows = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_base, y)):
        x_train_full = x_base.iloc[train_idx].to_numpy(dtype=float)
        x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
        x_train_full, x_val = voter.impute_fold(x_train_full, x_val)

        scaler = StandardScaler()
        x_train_full = scaler.fit_transform(x_train_full)
        x_val = scaler.transform(x_val)

        stopper = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=voter.RANDOM_STATE + fold,
        )
        fit_rel, stop_rel = next(stopper.split(x_train_full, y[train_idx]))
        fit_idx = train_idx[fit_rel]
        stop_idx = train_idx[stop_rel]

        model = _xgb_early_stop_model()
        model.fit(
            x_train_full[fit_rel],
            y[fit_idx],
            sample_weight=weights[fit_idx],
            eval_set=[(x_train_full[stop_rel], y[stop_idx])],
            sample_weight_eval_set=[weights[stop_idx]],
            verbose=False,
        )
        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)
        oof_pred[val_idx] = pred
        oof_proba[val_idx] = proba
        fold_id[val_idx] = fold
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None:
            best_iteration = 999
        best_iteration = int(best_iteration) + 1
        best_iterations.append(best_iteration)
        fold_metrics = voter.weighted_metrics(
            y[val_idx],
            pred,
            proba,
            weights[val_idx],
        )
        fold_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "variant": variant,
                "fold": fold,
                "n_fit": int(len(fit_rel)),
                "n_early_stop": int(len(stop_rel)),
                "n_val": int(len(val_idx)),
                "best_iteration": best_iteration,
                **fold_metrics,
            }
        )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::{variant}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return (
        result,
        fold_rows,
        float(np.mean(best_iterations)),
        float(np.median(best_iterations)),
    )


def _xgb_optuna_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }


def _xgb_from_optuna_params(params: dict[str, object]) -> XGBClassifier:
    return XGBClassifier(
        random_state=voter.RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        **params,
    )


def _score_xgb_optuna_inner(
    x_base: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    outer_train_idx: np.ndarray,
    params: dict[str, object],
    seed: int,
    trial: optuna.Trial | None = None,
) -> tuple[float, float]:
    inner = StratifiedKFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    inner_metrics = []
    outer_train_y = y[outer_train_idx]
    for inner_fold, (inner_train_rel, inner_val_rel) in enumerate(
        inner.split(x_base.iloc[outer_train_idx], outer_train_y)
    ):
        inner_train_idx = outer_train_idx[inner_train_rel]
        inner_val_idx = outer_train_idx[inner_val_rel]
        x_train, x_val = _prepare_split_arrays(x_base, inner_train_idx, inner_val_idx)
        model = _xgb_from_optuna_params(params)
        model.fit(x_train, y[inner_train_idx], sample_weight=weights[inner_train_idx])
        proba = model.predict_proba(x_val)
        pred = np.argmax(proba, axis=1)
        metrics = voter.weighted_metrics(
            y[inner_val_idx],
            pred,
            proba,
            weights[inner_val_idx],
        )
        inner_metrics.append(metrics)
        if trial is not None:
            trial.report(
                float(np.mean([m["balanced_accuracy"] for m in inner_metrics])),
                inner_fold,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

    return (
        float(np.mean([m["balanced_accuracy"] for m in inner_metrics])),
        float(np.mean([m["log_loss"] for m in inner_metrics])),
    )


def evaluate_xgb_optuna_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    n_trials: int,
) -> tuple[voter.CvResult, list[dict[str, object]], list[dict[str, object]]]:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    choice_rows = []
    trial_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        seed = voter.RANDOM_STATE + outer_fold
        sampler = TPESampler(seed=seed)
        pruner = MedianPruner(n_startup_trials=min(5, n_trials), n_warmup_steps=0)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            params = _xgb_optuna_params(trial)
            inner_ba, inner_log_loss = _score_xgb_optuna_inner(
                x_base,
                y,
                weights,
                outer_train_idx,
                params,
                seed,
                trial,
            )
            trial.set_user_attr("mean_inner_log_loss", inner_log_loss)
            return inner_ba

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_inner_ba = float(study.best_value)
        best_inner_log_loss = float(
            study.best_trial.user_attrs.get("mean_inner_log_loss", np.nan)
        )

        for trial in study.trials:
            row = {
                "cohort": cohort,
                "config": config,
                "outer_fold": outer_fold,
                "trial": trial.number,
                "state": trial.state.name,
                "inner_balanced_accuracy": trial.value
                if trial.value is not None
                else np.nan,
                "inner_log_loss": trial.user_attrs.get("mean_inner_log_loss", np.nan),
            }
            for key, value in trial.params.items():
                row[f"param_{key}"] = value
            trial_rows.append(row)

        x_train, x_val = _prepare_split_arrays(x_base, outer_train_idx, outer_val_idx)
        model = _xgb_from_optuna_params(best_params)
        model.fit(x_train, y[outer_train_idx], sample_weight=weights[outer_train_idx])
        proba = model.predict_proba(x_val)
        pred = np.argmax(proba, axis=1)
        oof_pred[outer_val_idx] = pred
        oof_proba[outer_val_idx] = proba
        fold_id[outer_val_idx] = outer_fold
        outer_metrics = voter.weighted_metrics(
            y[outer_val_idx],
            pred,
            proba,
            weights[outer_val_idx],
        )
        choice_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "outer_fold": outer_fold,
                "n_trials": n_trials,
                "selection_metric": "inner_weighted_balanced_accuracy",
                "inner_balanced_accuracy": best_inner_ba,
                "inner_log_loss": best_inner_log_loss,
                "outer_balanced_accuracy": outer_metrics["balanced_accuracy"],
                "outer_log_loss": outer_metrics["log_loss"],
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )
        print(
            f"  outer fold {outer_fold}: inner_BA={best_inner_ba:.4f} "
            f"inner_LL={best_inner_log_loss:.4f} outer_BA={outer_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::optuna_xgb_balanced_accuracy",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return result, choice_rows, trial_rows


def _xgb_current_with_seed(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
    )


def evaluate_xgb_seed_bag_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    variant: str,
    seeds: list[int],
) -> voter.CvResult:
    if variant == "xgb_current":
        return evaluate_candidate_result(
            df=df,
            cohort=cohort,
            config=config,
            candidate=BASELINE_CANDIDATE,
            feature_cols=feature_cols,
            make_model=candidate_factories()[BASELINE_CANDIDATE],
        )
    if variant != "xgb_seed_bag_5":
        raise RuntimeError(f"Unknown XGBoost seed-bag variant: {variant}")

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    splitter = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_base, y)):
        x_train, x_val = _prepare_split_arrays(x_base, train_idx, val_idx)
        proba = np.zeros((len(val_idx), 2), dtype=float)
        for seed in seeds:
            model = _xgb_current_with_seed(seed)
            model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
            proba += model.predict_proba(x_val) / len(seeds)
        oof_proba[val_idx] = proba
        oof_pred[val_idx] = np.argmax(proba, axis=1)
        fold_id[val_idx] = fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{variant}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def _fit_stage1_regressor_predict(
    x_demo: pd.DataFrame,
    issue_values: np.ndarray,
    weights: np.ndarray,
    fit_idx: np.ndarray,
    pred_idx: np.ndarray,
    stage1_model: str = "histgb_current",
) -> np.ndarray:
    valid_fit_idx = fit_idx[~np.isnan(issue_values[fit_idx])]
    if len(valid_fit_idx) < 50:
        raise RuntimeError(
            f"Stage 1 issue model has too few training rows: {len(valid_fit_idx)}"
        )
    if pd.Series(issue_values[valid_fit_idx]).nunique(dropna=True) < 2:
        raise RuntimeError("Stage 1 issue model has fewer than two target values")

    x_train = x_demo.iloc[valid_fit_idx].to_numpy(dtype=float)
    x_pred = x_demo.iloc[pred_idx].to_numpy(dtype=float)
    x_train, x_pred = voter.impute_fold(x_train, x_pred)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_pred = scaler.transform(x_pred)

    factories = stage1_regressor_factories()
    if stage1_model not in factories:
        raise RuntimeError(f"Unknown Stage 1 regressor: {stage1_model}")
    model = factories[stage1_model]()
    model.fit(
        x_train,
        issue_values[valid_fit_idx],
        sample_weight=weights[valid_fit_idx],
    )
    return model.predict(x_pred)


def _weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.average((y_true[mask] - y_pred[mask]) ** 2, weights=sample_weight[mask])))


def _crossfit_predicted_issues_for_outer_fold(
    work: pd.DataFrame,
    x_demo: pd.DataFrame,
    weights: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_val_idx: np.ndarray,
    outer_fold: int,
    stage1_model: str = "histgb_current",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    train_pred_cols = {}
    val_pred_cols = {}
    stage1_rows = []
    inner = KFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE + outer_fold,
    )

    for issue in voter.ISSUE_COLS:
        issue_values = work[issue].to_numpy(dtype=float)
        train_pred = np.full(len(outer_train_idx), np.nan, dtype=float)
        for inner_train_rel, inner_pred_rel in inner.split(outer_train_idx):
            fit_idx = outer_train_idx[inner_train_rel]
            pred_idx = outer_train_idx[inner_pred_rel]
            train_pred[inner_pred_rel] = _fit_stage1_regressor_predict(
                x_demo,
                issue_values,
                weights,
                fit_idx,
                pred_idx,
                stage1_model=stage1_model,
            )

        val_pred = _fit_stage1_regressor_predict(
            x_demo,
            issue_values,
            weights,
            outer_train_idx,
            outer_val_idx,
            stage1_model=stage1_model,
        )
        train_pred_cols[f"{issue}_predicted"] = train_pred
        val_pred_cols[f"{issue}_predicted"] = val_pred

        train_issue_values = issue_values[outer_train_idx]
        val_issue_values = issue_values[outer_val_idx]
        stage1_rows.append(
            {
                "outer_fold": outer_fold,
                "stage1_model": stage1_model,
                "issue": issue,
                "train_nonmissing": int((~np.isnan(train_issue_values)).sum()),
                "val_nonmissing": int((~np.isnan(val_issue_values)).sum()),
                "inner_oof_weighted_rmse": _weighted_rmse(
                    train_issue_values,
                    train_pred,
                    weights[outer_train_idx],
                ),
                "outer_val_weighted_rmse": _weighted_rmse(
                    val_issue_values,
                    val_pred,
                    weights[outer_val_idx],
                ),
            }
        )

    return (
        pd.DataFrame(train_pred_cols, index=outer_train_idx),
        pd.DataFrame(val_pred_cols, index=outer_val_idx),
        stage1_rows,
    )


def _two_stage_matrix(
    config: str,
    pred_df: pd.DataFrame,
    row_idx: np.ndarray,
    x_demo: pd.DataFrame,
    x_party_ideology: pd.DataFrame,
) -> np.ndarray:
    blocks = [pred_df.to_numpy(dtype=float)]
    if config == "predicted_issues_only":
        pass
    elif config == "predicted_issues+demographics":
        blocks.append(x_demo.iloc[row_idx].to_numpy(dtype=float))
    elif config == "predicted_issues+party_id+ideology":
        blocks.append(x_party_ideology.iloc[row_idx].to_numpy(dtype=float))
    else:
        raise RuntimeError(f"Unknown two-stage config: {config}")
    return np.column_stack(blocks)


def evaluate_two_stage(
    df: pd.DataFrame,
    cohort: str,
    stage1_model: str = "histgb_current",
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, voter.CvResult],
]:
    """Evaluate demographics -> predicted issues -> voter intent with outer CV."""
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_demo = voter.encode_feature_frame(work, voter.DEMO_COLS)
    x_party_ideology = voter.encode_feature_frame(
        work,
        [voter.PARTY_ID_COL] + voter.IDEOLOGY_COLS,
    )
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    trackers = {
        config: {
            "oof_pred": np.full(len(work), -1, dtype=int),
            "oof_proba": np.full((len(work), 2), np.nan, dtype=float),
            "fold_id": np.full(len(work), -1, dtype=int),
            "fold_rows": [],
        }
        for config in TWO_STAGE_CONFIGS
    }
    stage1_rows = []

    for outer_fold, (train_idx, val_idx) in enumerate(outer.split(x_demo, y)):
        print(f"  preparing predicted issues for outer fold {outer_fold}", flush=True)
        train_pred_df, val_pred_df, fold_stage1_rows = _crossfit_predicted_issues_for_outer_fold(
            work,
            x_demo,
            weights,
            train_idx,
            val_idx,
            outer_fold,
            stage1_model=stage1_model,
        )
        for row in fold_stage1_rows:
            stage1_rows.append({"cohort": cohort, **row})

        for config in TWO_STAGE_CONFIGS:
            x_train = _two_stage_matrix(config, train_pred_df, train_idx, x_demo, x_party_ideology)
            x_val = _two_stage_matrix(config, val_pred_df, val_idx, x_demo, x_party_ideology)
            x_train, x_val = voter.impute_fold(x_train, x_val)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

            model = candidate_factories()[BASELINE_CANDIDATE]()
            model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
            pred = model.predict(x_val)
            proba = model.predict_proba(x_val)
            trackers[config]["oof_pred"][val_idx] = pred
            trackers[config]["oof_proba"][val_idx] = proba
            trackers[config]["fold_id"][val_idx] = outer_fold
            fold_metrics = voter.weighted_metrics(
                y[val_idx],
                pred,
                proba,
                weights[val_idx],
            )
            trackers[config]["fold_rows"].append(
                {
                    "cohort": cohort,
                    "config": config,
                    "fold": outer_fold,
                    "n": int(len(val_idx)),
                    **fold_metrics,
                }
            )
            print(
                f"    {config:34s} BA={fold_metrics['balanced_accuracy']:.4f} "
                f"LL={fold_metrics['log_loss']:.4f}",
                flush=True,
            )

    result_rows = []
    fold_rows = []
    for config, tracker in trackers.items():
        metrics = voter.weighted_metrics(
            y,
            tracker["oof_pred"],
            tracker["oof_proba"],
            weights,
        )
        result_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "stage1_predictors": "demographics_only",
                "stage1_model": stage1_model,
                "stage2_model": BASELINE_CANDIDATE,
                "n": len(work),
                "n_predicted_issue_features": len(voter.ISSUE_COLS),
                **metrics,
            }
        )
        fold_rows.extend(tracker["fold_rows"])
    result_objects = {
        config: voter.CvResult(
            cohort=cohort,
            config=f"{config}::{stage1_model}",
            y_true=y,
            y_pred=tracker["oof_pred"],
            y_proba=tracker["oof_proba"],
            sample_weight=weights,
            fold_id=tracker["fold_id"],
            class_names=list(label_encoder.classes_),
            metrics=voter.weighted_metrics(
                y,
                tracker["oof_pred"],
                tracker["oof_proba"],
                weights,
            ),
        )
        for config, tracker in trackers.items()
    }
    return result_rows, fold_rows, stage1_rows, result_objects


def evaluate_two_stage_stage2_models(
    df: pd.DataFrame,
    cohort: str,
    stage2_candidates: list[str],
    stage1_model: str = "histgb_current",
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, str], voter.CvResult],
]:
    """Evaluate Stage 2 learner alternatives on cross-fitted predicted issues."""
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_demo = voter.encode_feature_frame(work, voter.DEMO_COLS)
    x_party_ideology = voter.encode_feature_frame(
        work,
        [voter.PARTY_ID_COL] + voter.IDEOLOGY_COLS,
    )
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    trackers = {
        (config, candidate): {
            "oof_pred": np.full(len(work), -1, dtype=int),
            "oof_proba": np.full((len(work), 2), np.nan, dtype=float),
            "fold_id": np.full(len(work), -1, dtype=int),
            "fold_rows": [],
        }
        for config in TWO_STAGE_CONFIGS
        for candidate in stage2_candidates
    }
    factories = candidate_factories()
    stage1_rows = []

    for outer_fold, (train_idx, val_idx) in enumerate(outer.split(x_demo, y)):
        print(f"  preparing predicted issues for outer fold {outer_fold}", flush=True)
        train_pred_df, val_pred_df, fold_stage1_rows = _crossfit_predicted_issues_for_outer_fold(
            work,
            x_demo,
            weights,
            train_idx,
            val_idx,
            outer_fold,
            stage1_model=stage1_model,
        )
        for row in fold_stage1_rows:
            stage1_rows.append({"cohort": cohort, **row})

        for config in TWO_STAGE_CONFIGS:
            x_train = _two_stage_matrix(
                config,
                train_pred_df,
                train_idx,
                x_demo,
                x_party_ideology,
            )
            x_val = _two_stage_matrix(
                config,
                val_pred_df,
                val_idx,
                x_demo,
                x_party_ideology,
            )
            x_train, x_val = voter.impute_fold(x_train, x_val)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

            for candidate in stage2_candidates:
                model = factories[candidate]()
                try:
                    model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
                except TypeError:
                    model.fit(x_train, y[train_idx])
                pred = model.predict(x_val)
                proba = model.predict_proba(x_val)
                tracker = trackers[(config, candidate)]
                tracker["oof_pred"][val_idx] = pred
                tracker["oof_proba"][val_idx] = proba
                tracker["fold_id"][val_idx] = outer_fold
                fold_metrics = voter.weighted_metrics(
                    y[val_idx],
                    pred,
                    proba,
                    weights[val_idx],
                )
                tracker["fold_rows"].append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "stage1_model": stage1_model,
                        "stage2_model": candidate,
                        "fold": outer_fold,
                        "n": int(len(val_idx)),
                        **fold_metrics,
                    }
                )
                print(
                    f"    {config:34s} {candidate:16s} "
                    f"BA={fold_metrics['balanced_accuracy']:.4f} "
                    f"LL={fold_metrics['log_loss']:.4f}",
                    flush=True,
                )

    result_rows = []
    fold_rows = []
    result_objects = {}
    for (config, candidate), tracker in trackers.items():
        metrics = voter.weighted_metrics(
            y,
            tracker["oof_pred"],
            tracker["oof_proba"],
            weights,
        )
        result_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "stage1_model": stage1_model,
                "stage2_model": candidate,
                "n": len(work),
                "n_predicted_issue_features": len(voter.ISSUE_COLS),
                **metrics,
            }
        )
        fold_rows.extend(tracker["fold_rows"])
        result_objects[(config, candidate)] = voter.CvResult(
            cohort=cohort,
            config=f"{config}::{stage1_model}::{candidate}",
            y_true=y,
            y_pred=tracker["oof_pred"],
            y_proba=tracker["oof_proba"],
            sample_weight=weights,
            fold_id=tracker["fold_id"],
            class_names=list(label_encoder.classes_),
            metrics=metrics,
        )
    return result_rows, fold_rows, stage1_rows, result_objects


def _domain_aggregate_split(
    work: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    issue_frame = work[voter.ISSUE_COLS].copy()
    x_train = issue_frame.iloc[train_idx].to_numpy(dtype=float)
    x_val = issue_frame.iloc[val_idx].to_numpy(dtype=float)
    x_train, x_val = voter.impute_fold(x_train, x_val)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    issue_positions = {issue: idx for idx, issue in enumerate(voter.ISSUE_COLS)}
    train_cols = []
    val_cols = []
    names = []
    for domain, issues in voter.ISSUE_DOMAINS.items():
        positions = [issue_positions[issue] for issue in issues]
        train_cols.append(np.mean(x_train[:, positions], axis=1))
        val_cols.append(np.mean(x_val[:, positions], axis=1))
        names.append(f"domain_{domain}")
    return np.column_stack(train_cols), np.column_stack(val_cols), names


def _extra_split(
    work: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_base = voter.encode_feature_frame(work, feature_cols)
    x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
    x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
    x_train, x_val = voter.impute_fold(x_train, x_val)
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_val)


def _domain_aggregate_matrix(
    config: str,
    work: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_train_domain, x_val_domain, domain_names = _domain_aggregate_split(
        work,
        train_idx,
        val_idx,
    )
    blocks_train = [x_train_domain]
    blocks_val = [x_val_domain]
    if config == "domain_issues_only":
        pass
    elif config == "domain_issues+demographics":
        x_train_extra, x_val_extra = _extra_split(work, voter.DEMO_COLS, train_idx, val_idx)
        blocks_train.append(x_train_extra)
        blocks_val.append(x_val_extra)
    elif config == "domain_issues+party_id+ideology":
        x_train_extra, x_val_extra = _extra_split(
            work,
            [voter.PARTY_ID_COL] + voter.IDEOLOGY_COLS,
            train_idx,
            val_idx,
        )
        blocks_train.append(x_train_extra)
        blocks_val.append(x_val_extra)
    else:
        raise RuntimeError(f"Unknown domain aggregate config: {config}")
    return (
        np.column_stack(blocks_train),
        np.column_stack(blocks_val),
        len(domain_names),
    )


def evaluate_domain_aggregates(
    df: pd.DataFrame,
    cohort: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    split_base = voter.encode_feature_frame(work, voter.DEMO_COLS)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    trackers = {
        config: {
            "oof_pred": np.full(len(work), -1, dtype=int),
            "oof_proba": np.full((len(work), 2), np.nan, dtype=float),
            "fold_rows": [],
            "n_domain_features": len(voter.ISSUE_DOMAINS),
        }
        for config in DOMAIN_AGGREGATE_CONFIGS
    }

    for fold, (train_idx, val_idx) in enumerate(outer.split(split_base, y)):
        for config in DOMAIN_AGGREGATE_CONFIGS:
            x_train, x_val, n_domain_features = _domain_aggregate_matrix(
                config,
                work,
                train_idx,
                val_idx,
            )
            model = candidate_factories()[BASELINE_CANDIDATE]()
            model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
            pred = model.predict(x_val)
            proba = model.predict_proba(x_val)
            trackers[config]["oof_pred"][val_idx] = pred
            trackers[config]["oof_proba"][val_idx] = proba
            trackers[config]["n_domain_features"] = n_domain_features
            fold_metrics = voter.weighted_metrics(
                y[val_idx],
                pred,
                proba,
                weights[val_idx],
            )
            trackers[config]["fold_rows"].append(
                {
                    "cohort": cohort,
                    "config": config,
                    "fold": fold,
                    "n": int(len(val_idx)),
                    **fold_metrics,
                }
            )
            print(
                f"  fold {fold} {config:32s} BA={fold_metrics['balanced_accuracy']:.4f} "
                f"LL={fold_metrics['log_loss']:.4f}",
                flush=True,
            )

    result_rows = []
    fold_rows = []
    for config, tracker in trackers.items():
        metrics = voter.weighted_metrics(
            y,
            tracker["oof_pred"],
            tracker["oof_proba"],
            weights,
        )
        extra_features = 0
        if config == "domain_issues+demographics":
            extra_features = len(voter.encode_feature_frame(work, voter.DEMO_COLS).columns)
        elif config == "domain_issues+party_id+ideology":
            extra_features = len(
                voter.encode_feature_frame(work, [voter.PARTY_ID_COL] + voter.IDEOLOGY_COLS).columns
            )
        result_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "stage2_model": BASELINE_CANDIDATE,
                "n": len(work),
                "n_domain_features": tracker["n_domain_features"],
                "n_total_features": tracker["n_domain_features"] + extra_features,
                **metrics,
            }
        )
        fold_rows.extend(tracker["fold_rows"])
    return result_rows, fold_rows


def _issue_pca_extra_cols(config: str) -> tuple[str, list[str]]:
    if config == "pca_issues_only":
        return "issues_only", []
    if config == "pca_issues+party_id+ideology":
        return "issues+party_id+ideology", [voter.PARTY_ID_COL] + voter.IDEOLOGY_COLS
    if config == "pca_issues+demographics+ideology":
        return "issues+demographics+ideology", voter.DEMO_COLS + voter.IDEOLOGY_COLS
    raise RuntimeError(f"Unknown issue-PCA config: {config}")


def _issue_pca_split(
    work: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_train = work[voter.ISSUE_COLS].iloc[train_idx].to_numpy(dtype=float)
    x_val = work[voter.ISSUE_COLS].iloc[val_idx].to_numpy(dtype=float)
    x_train, x_val = voter.impute_fold(x_train, x_val)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    pca = PCA(n_components=n_components, random_state=voter.RANDOM_STATE)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    return x_train_pca, x_val_pca, float(pca.explained_variance_ratio_.sum())


def _issue_pca_matrix(
    config: str,
    work: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    _, extra_cols = _issue_pca_extra_cols(config)
    x_train_pca, x_val_pca, variance_ratio = _issue_pca_split(
        work,
        train_idx,
        val_idx,
        n_components,
    )
    train_blocks = [x_train_pca]
    val_blocks = [x_val_pca]
    n_extra_features = 0
    if extra_cols:
        x_train_extra, x_val_extra = _extra_split(work, extra_cols, train_idx, val_idx)
        train_blocks.append(x_train_extra)
        val_blocks.append(x_val_extra)
        n_extra_features = x_train_extra.shape[1]
    return (
        np.column_stack(train_blocks),
        np.column_stack(val_blocks),
        variance_ratio,
        n_extra_features,
    )


def evaluate_issue_pca(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    n_components: int,
) -> tuple[voter.CvResult, list[dict[str, object]], float, int]:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    split_base = voter.encode_feature_frame(work, voter.ISSUE_COLS)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    fold_rows = []
    variance_ratios = []
    n_extra_features = 0

    for fold, (train_idx, val_idx) in enumerate(outer.split(split_base, y)):
        x_train, x_val, variance_ratio, n_extra_features = _issue_pca_matrix(
            config,
            work,
            train_idx,
            val_idx,
            n_components,
        )
        variance_ratios.append(variance_ratio)
        model = candidate_factories()[BASELINE_CANDIDATE]()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)
        oof_pred[val_idx] = pred
        oof_proba[val_idx] = proba
        fold_id[val_idx] = fold
        fold_metrics = voter.weighted_metrics(
            y[val_idx],
            pred,
            proba,
            weights[val_idx],
        )
        fold_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "n_components": n_components,
                "fold": fold,
                "n": int(len(val_idx)),
                "explained_variance_ratio": variance_ratio,
                **fold_metrics,
            }
        )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::pca_{n_components}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return result, fold_rows, float(np.mean(variance_ratios)), n_extra_features


def _issue_topk_extra_cols(config: str) -> tuple[str, list[str]]:
    if config not in ISSUE_TOPK_CONFIG_MAP:
        raise RuntimeError(f"Unknown issue-top-k config: {config}")
    canonical_config = ISSUE_TOPK_CONFIG_MAP[config]
    extra_cols = [
        col
        for col in voter.FEATURE_CONFIGS[canonical_config]
        if col not in voter.ISSUE_COLS
    ]
    return canonical_config, extra_cols


def _weighted_issue_gap_scores(
    x_train_issue: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(x_train_issue.shape[1], dtype=float)
    for col_idx in range(x_train_issue.shape[1]):
        values = x_train_issue[:, col_idx]
        class_means = []
        for cls in [0, 1]:
            mask = y_train == cls
            if not mask.any():
                raise RuntimeError(f"Missing class {cls} in issue top-k training fold")
            class_means.append(
                float(np.average(values[mask], weights=train_weights[mask]))
            )
        scores[col_idx] = abs(class_means[1] - class_means[0])
    return scores


def _select_topk_issue_indices(
    x_train_issue: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _weighted_issue_gap_scores(x_train_issue, y_train, train_weights)
    ordered = sorted(
        range(len(voter.ISSUE_COLS)),
        key=lambda idx: (-scores[idx], voter.ISSUE_COLS[idx]),
    )
    return np.array(ordered[:top_k], dtype=int), scores


def _issue_topk_matrix(
    config: str,
    work: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    _, extra_cols = _issue_topk_extra_cols(config)
    x_train_issue = work[voter.ISSUE_COLS].iloc[train_idx].to_numpy(dtype=float)
    x_val_issue = work[voter.ISSUE_COLS].iloc[val_idx].to_numpy(dtype=float)
    x_train_issue, x_val_issue = voter.impute_fold(x_train_issue, x_val_issue)

    scaler = StandardScaler()
    x_train_issue = scaler.fit_transform(x_train_issue)
    x_val_issue = scaler.transform(x_val_issue)

    selected_idx, scores = _select_topk_issue_indices(
        x_train_issue,
        y[train_idx],
        weights[train_idx],
        top_k,
    )
    train_blocks = [x_train_issue[:, selected_idx]]
    val_blocks = [x_val_issue[:, selected_idx]]
    n_extra_features = 0
    if extra_cols:
        x_train_extra, x_val_extra = _extra_split(work, extra_cols, train_idx, val_idx)
        train_blocks.append(x_train_extra)
        val_blocks.append(x_val_extra)
        n_extra_features = x_train_extra.shape[1]

    return (
        np.column_stack(train_blocks),
        np.column_stack(val_blocks),
        selected_idx,
        scores,
        n_extra_features,
    )


def evaluate_issue_topk(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    top_k: int,
) -> tuple[voter.CvResult, list[dict[str, object]], list[dict[str, object]], int]:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    split_base = voter.encode_feature_frame(work, voter.ISSUE_COLS)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    fold_id = np.full(len(work), -1, dtype=int)
    fold_rows = []
    selected_rows = []
    n_extra_features = 0

    for fold, (train_idx, val_idx) in enumerate(outer.split(split_base, y)):
        x_train, x_val, selected_idx, scores, n_extra_features = _issue_topk_matrix(
            config,
            work,
            train_idx,
            val_idx,
            y,
            weights,
            top_k,
        )
        model = candidate_factories()[BASELINE_CANDIDATE]()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)
        oof_pred[val_idx] = pred
        oof_proba[val_idx] = proba
        fold_id[val_idx] = fold

        fold_metrics = voter.weighted_metrics(
            y[val_idx],
            pred,
            proba,
            weights[val_idx],
        )
        fold_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "top_k": top_k,
                "fold": fold,
                "n": int(len(val_idx)),
                **fold_metrics,
            }
        )
        for rank, issue_idx in enumerate(selected_idx, start=1):
            selected_rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "top_k": top_k,
                    "fold": fold,
                    "rank": rank,
                    "issue": voter.ISSUE_COLS[issue_idx],
                    "score": scores[issue_idx],
                }
            )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::top_{top_k}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return result, fold_rows, selected_rows, n_extra_features


def _fit_predict_split(
    x_base: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    make_model: Callable[[], object],
) -> tuple[np.ndarray, np.ndarray]:
    x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
    x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
    x_train, x_val = voter.impute_fold(x_train, x_val)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    model = make_model()
    try:
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
    except TypeError:
        model.fit(x_train, y[train_idx])

    return model.predict(x_val), model.predict_proba(x_val)


def _prepare_split_arrays(
    x_base: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = x_base.iloc[train_idx].to_numpy(dtype=float)
    x_val = x_base.iloc[val_idx].to_numpy(dtype=float)
    x_train, x_val = voter.impute_fold(x_train, x_val)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    return x_train, x_val


def _fit_ensemble_proba_split(
    x_base: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    recipe: list[tuple[str, float]],
) -> np.ndarray:
    x_train, x_val = _prepare_split_arrays(x_base, train_idx, val_idx)
    factories = candidate_factories()
    total_weight = sum(weight for _, weight in recipe)
    proba = np.zeros((len(val_idx), 2), dtype=float)
    for model_name, weight in recipe:
        model = factories[model_name]()
        model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        proba += (weight / total_weight) * model.predict_proba(x_val)
    return proba


def _calibration_feature(proba: np.ndarray, method: str) -> np.ndarray:
    p1 = np.clip(proba[:, 1], 1e-6, 1 - 1e-6)
    if method == "sigmoid":
        return np.log(p1 / (1 - p1)).reshape(-1, 1)
    if method == "isotonic":
        return p1
    raise RuntimeError(f"Unknown calibration method: {method}")


def _fit_calibrator(
    raw_proba: np.ndarray,
    y_cal: np.ndarray,
    w_cal: np.ndarray,
    method: str,
) -> Callable[[np.ndarray], np.ndarray]:
    if method == "sigmoid":
        calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
        calibrator.fit(_calibration_feature(raw_proba, method), y_cal, sample_weight=w_cal)

        def predict(cal_proba: np.ndarray) -> np.ndarray:
            return calibrator.predict_proba(_calibration_feature(cal_proba, method))[:, 1]

        return predict

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(_calibration_feature(raw_proba, method), y_cal, sample_weight=w_cal)

        def predict(cal_proba: np.ndarray) -> np.ndarray:
            return calibrator.predict(_calibration_feature(cal_proba, method))

        return predict

    raise RuntimeError(f"Unknown calibration method: {method}")


def evaluate_calibrated_current(
    df: pd.DataFrame,
    feature_cols: list[str],
    method: str,
) -> dict[str, float]:
    """Evaluate fixed XGBoost with optional inner-fold probability calibration."""
    if method == "none":
        return evaluate_candidate(df, feature_cols, candidate_factories()[BASELINE_CANDIDATE])

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    factories = candidate_factories()

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        val_p1_parts = []
        outer_train_y = y[outer_train_idx]
        for inner_train_rel, cal_rel in inner.split(x_base.iloc[outer_train_idx], outer_train_y):
            inner_train_idx = outer_train_idx[inner_train_rel]
            cal_idx = outer_train_idx[cal_rel]
            x_train, x_cal = _prepare_split_arrays(x_base, inner_train_idx, cal_idx)
            _, x_outer_val = _prepare_split_arrays(x_base, inner_train_idx, outer_val_idx)

            model = factories[BASELINE_CANDIDATE]()
            model.fit(x_train, y[inner_train_idx], sample_weight=weights[inner_train_idx])
            cal_raw = model.predict_proba(x_cal)
            calibrator = _fit_calibrator(
                cal_raw,
                y[cal_idx],
                weights[cal_idx],
                method,
            )
            outer_raw = model.predict_proba(x_outer_val)
            val_p1_parts.append(calibrator(outer_raw))

        p1 = np.clip(np.mean(np.vstack(val_p1_parts), axis=0), 1e-6, 1 - 1e-6)
        proba = np.column_stack([1 - p1, p1])
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = (p1 >= 0.5).astype(int)

    return voter.weighted_metrics(y, oof_pred, oof_proba, weights)


def evaluate_calibrated_candidate_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    candidate: str,
    make_model: Callable[[], object],
    method: str,
) -> voter.CvResult:
    """Evaluate a fixed candidate with inner-fold probability calibration."""
    if method == "none":
        return evaluate_candidate_result(
            df=df,
            cohort=cohort,
            config=config,
            candidate=candidate,
            feature_cols=feature_cols,
            make_model=make_model,
        )

    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    fold_id = np.full(len(work), -1, dtype=int)

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        val_p1_parts = []
        outer_train_y = y[outer_train_idx]
        for inner_train_rel, cal_rel in inner.split(
            x_base.iloc[outer_train_idx],
            outer_train_y,
        ):
            inner_train_idx = outer_train_idx[inner_train_rel]
            cal_idx = outer_train_idx[cal_rel]
            x_train, x_cal = _prepare_split_arrays(x_base, inner_train_idx, cal_idx)
            _, x_outer_val = _prepare_split_arrays(
                x_base,
                inner_train_idx,
                outer_val_idx,
            )

            model = make_model()
            try:
                model.fit(
                    x_train,
                    y[inner_train_idx],
                    sample_weight=weights[inner_train_idx],
                )
            except TypeError:
                model.fit(x_train, y[inner_train_idx])
            cal_raw = model.predict_proba(x_cal)
            calibrator = _fit_calibrator(
                cal_raw,
                y[cal_idx],
                weights[cal_idx],
                method,
            )
            outer_raw = model.predict_proba(x_outer_val)
            val_p1_parts.append(calibrator(outer_raw))

        p1 = np.clip(np.mean(np.vstack(val_p1_parts), axis=0), 1e-6, 1 - 1e-6)
        proba = np.column_stack([1 - p1, p1])
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = (p1 >= 0.5).astype(int)
        fold_id[outer_val_idx] = outer_fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{candidate}::{method}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def _fit_calibrated_candidate_p1_split(
    x_base: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    make_model: Callable[[], object],
    method: str,
    random_state: int,
) -> np.ndarray:
    if method == "none":
        x_train, x_val = _prepare_split_arrays(x_base, train_idx, val_idx)
        model = make_model()
        try:
            model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
        except TypeError:
            model.fit(x_train, y[train_idx])
        return np.clip(model.predict_proba(x_val)[:, 1], 1e-6, 1 - 1e-6)

    inner = StratifiedKFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=random_state,
    )
    train_y = y[train_idx]
    p1_parts = []
    for model_train_rel, cal_rel in inner.split(x_base.iloc[train_idx], train_y):
        model_train_idx = train_idx[model_train_rel]
        cal_idx = train_idx[cal_rel]
        x_train, x_cal = _prepare_split_arrays(x_base, model_train_idx, cal_idx)
        _, x_val = _prepare_split_arrays(x_base, model_train_idx, val_idx)
        model = make_model()
        try:
            model.fit(
                x_train,
                y[model_train_idx],
                sample_weight=weights[model_train_idx],
            )
        except TypeError:
            model.fit(x_train, y[model_train_idx])
        cal_raw = model.predict_proba(x_cal)
        calibrator = _fit_calibrator(
            cal_raw,
            y[cal_idx],
            weights[cal_idx],
            method,
        )
        val_raw = model.predict_proba(x_val)
        p1_parts.append(calibrator(val_raw))

    return np.clip(np.mean(np.vstack(p1_parts), axis=0), 1e-6, 1 - 1e-6)


def evaluate_calibrated_threshold_candidate_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    candidate: str,
    make_model: Callable[[], object],
    method: str,
) -> tuple[voter.CvResult, list[dict[str, object]]]:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    fold_id = np.full(len(work), -1, dtype=int)
    choice_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        threshold_inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        outer_train_y = y[outer_train_idx]
        inner_oof_p1 = np.full(len(outer_train_idx), np.nan, dtype=float)
        for threshold_train_rel, threshold_rel in threshold_inner.split(
            x_base.iloc[outer_train_idx],
            outer_train_y,
        ):
            threshold_train_idx = outer_train_idx[threshold_train_rel]
            threshold_idx = outer_train_idx[threshold_rel]
            inner_oof_p1[threshold_rel] = _fit_calibrated_candidate_p1_split(
                x_base=x_base,
                y=y,
                weights=weights,
                train_idx=threshold_train_idx,
                val_idx=threshold_idx,
                make_model=make_model,
                method=method,
                random_state=voter.RANDOM_STATE + 1000 + outer_fold,
            )

        threshold, inner_ba = _choose_threshold(
            outer_train_y,
            inner_oof_p1,
            weights[outer_train_idx],
        )
        p1 = _fit_calibrated_candidate_p1_split(
            x_base=x_base,
            y=y,
            weights=weights,
            train_idx=outer_train_idx,
            val_idx=outer_val_idx,
            make_model=make_model,
            method=method,
            random_state=voter.RANDOM_STATE + 2000 + outer_fold,
        )
        proba = np.column_stack([1 - p1, p1])
        pred = (p1 >= threshold).astype(int)
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = pred
        fold_id[outer_val_idx] = outer_fold
        outer_metrics = voter.weighted_metrics(
            y[outer_val_idx],
            pred,
            proba,
            weights[outer_val_idx],
        )
        choice_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "candidate": candidate,
                "calibration": method,
                "outer_fold": outer_fold,
                "selected_threshold": threshold,
                "inner_balanced_accuracy": inner_ba,
                "outer_balanced_accuracy": outer_metrics["balanced_accuracy"],
                "outer_log_loss": outer_metrics["log_loss"],
            }
        )
        print(
            f"  {candidate:16s} {method:8s} outer fold {outer_fold}: "
            f"threshold={threshold:.3f} inner_BA={inner_ba:.4f} "
            f"outer_BA={outer_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::{candidate}::{method}::threshold",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return result, choice_rows


def evaluate_calibrated_extended_blend_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    blend_name: str,
    recipe: list[tuple[str, str, float]],
) -> voter.CvResult:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    factories = candidate_factories()
    total_weight = sum(weight for _, _, weight in recipe)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    fold_id = np.full(len(work), -1, dtype=int)

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        p1 = np.zeros(len(outer_val_idx), dtype=float)
        for candidate, calibration, component_weight in recipe:
            p1 += (component_weight / total_weight) * _fit_calibrated_candidate_p1_split(
                x_base=x_base,
                y=y,
                weights=weights,
                train_idx=outer_train_idx,
                val_idx=outer_val_idx,
                make_model=factories[candidate],
                method=calibration,
                random_state=voter.RANDOM_STATE + 3000 + outer_fold,
            )
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        proba = np.column_stack([1 - p1, p1])
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = (p1 >= 0.5).astype(int)
        fold_id[outer_val_idx] = outer_fold

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    return voter.CvResult(
        cohort=cohort,
        config=f"{config}::{blend_name}",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )


def _fit_calibrated_extended_blend_p1_split(
    x_base: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    recipe: list[tuple[str, str, float]],
    random_state: int,
) -> np.ndarray:
    factories = candidate_factories()
    total_weight = sum(weight for _, _, weight in recipe)
    p1 = np.zeros(len(val_idx), dtype=float)
    for candidate, calibration, component_weight in recipe:
        p1 += (component_weight / total_weight) * _fit_calibrated_candidate_p1_split(
            x_base=x_base,
            y=y,
            weights=weights,
            train_idx=train_idx,
            val_idx=val_idx,
            make_model=factories[candidate],
            method=calibration,
            random_state=random_state,
        )
    return np.clip(p1, 1e-6, 1 - 1e-6)


def evaluate_threshold_tuned_calibrated_extended_blend_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    blend_name: str,
    recipe: list[tuple[str, str, float]],
) -> tuple[voter.CvResult, list[dict[str, object]]]:
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    fold_id = np.full(len(work), -1, dtype=int)
    choice_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        threshold_inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        outer_train_y = y[outer_train_idx]
        inner_oof_p1 = np.full(len(outer_train_idx), np.nan, dtype=float)
        for threshold_train_rel, threshold_rel in threshold_inner.split(
            x_base.iloc[outer_train_idx],
            outer_train_y,
        ):
            threshold_train_idx = outer_train_idx[threshold_train_rel]
            threshold_idx = outer_train_idx[threshold_rel]
            inner_oof_p1[threshold_rel] = _fit_calibrated_extended_blend_p1_split(
                x_base=x_base,
                y=y,
                weights=weights,
                train_idx=threshold_train_idx,
                val_idx=threshold_idx,
                recipe=recipe,
                random_state=voter.RANDOM_STATE + 4000 + outer_fold,
            )

        threshold, inner_ba = _choose_threshold(
            outer_train_y,
            inner_oof_p1,
            weights[outer_train_idx],
        )
        p1 = _fit_calibrated_extended_blend_p1_split(
            x_base=x_base,
            y=y,
            weights=weights,
            train_idx=outer_train_idx,
            val_idx=outer_val_idx,
            recipe=recipe,
            random_state=voter.RANDOM_STATE + 5000 + outer_fold,
        )
        proba = np.column_stack([1 - p1, p1])
        pred = (p1 >= threshold).astype(int)
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = pred
        fold_id[outer_val_idx] = outer_fold
        outer_metrics = voter.weighted_metrics(
            y[outer_val_idx],
            pred,
            proba,
            weights[outer_val_idx],
        )
        choice_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "blend": blend_name,
                "outer_fold": outer_fold,
                "selected_threshold": threshold,
                "inner_balanced_accuracy": inner_ba,
                "outer_balanced_accuracy": outer_metrics["balanced_accuracy"],
                "outer_log_loss": outer_metrics["log_loss"],
            }
        )
        print(
            f"  {blend_name:36s} outer fold {outer_fold}: "
            f"threshold={threshold:.3f} inner_BA={inner_ba:.4f} "
            f"outer_BA={outer_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::{blend_name}::threshold",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return result, choice_rows


def _weighted_balanced_accuracy_for_threshold(
    y_true: np.ndarray,
    p1: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float,
) -> float:
    pred = (p1 >= threshold).astype(int)
    return voter.weighted_metrics(
        y_true,
        pred,
        np.column_stack([1 - p1, p1]),
        sample_weight,
    )["balanced_accuracy"]


def _choose_threshold(
    y_true: np.ndarray,
    p1: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[float, float]:
    scored = [
        (
            threshold,
            _weighted_balanced_accuracy_for_threshold(
                y_true,
                p1,
                sample_weight,
                threshold,
            ),
        )
        for threshold in THRESHOLD_GRID
    ]
    # Prefer the threshold closest to 0.5 when scores tie at floating precision.
    return sorted(scored, key=lambda item: (-item[1], abs(item[0] - 0.5), item[0]))[0]


def evaluate_threshold_tuned_current(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate fixed XGBoost with a threshold selected inside each outer fold."""
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    factories = candidate_factories()
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    choice_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        inner_oof_p1 = np.full(len(outer_train_idx), np.nan, dtype=float)
        outer_train_y = y[outer_train_idx]

        for inner_train_rel, threshold_rel in inner.split(
            x_base.iloc[outer_train_idx], outer_train_y
        ):
            inner_train_idx = outer_train_idx[inner_train_rel]
            threshold_idx = outer_train_idx[threshold_rel]
            x_train, x_threshold = _prepare_split_arrays(
                x_base,
                inner_train_idx,
                threshold_idx,
            )
            model = factories[BASELINE_CANDIDATE]()
            model.fit(x_train, y[inner_train_idx], sample_weight=weights[inner_train_idx])
            inner_oof_p1[threshold_rel] = model.predict_proba(x_threshold)[:, 1]

        threshold, inner_ba = _choose_threshold(
            outer_train_y,
            inner_oof_p1,
            weights[outer_train_idx],
        )

        x_train, x_val = _prepare_split_arrays(x_base, outer_train_idx, outer_val_idx)
        model = factories[BASELINE_CANDIDATE]()
        model.fit(x_train, y[outer_train_idx], sample_weight=weights[outer_train_idx])
        proba = model.predict_proba(x_val)
        pred = (proba[:, 1] >= threshold).astype(int)
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = pred
        outer_metrics = voter.weighted_metrics(
            y[outer_val_idx],
            pred,
            proba,
            weights[outer_val_idx],
        )
        choice_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "outer_fold": outer_fold,
                "selected_threshold": threshold,
                "inner_balanced_accuracy": inner_ba,
                "outer_balanced_accuracy": outer_metrics["balanced_accuracy"],
                "outer_log_loss": outer_metrics["log_loss"],
            }
        )
        print(
            f"  outer fold {outer_fold}: threshold={threshold:.3f} "
            f"inner_BA={inner_ba:.4f} outer_BA={outer_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    return voter.weighted_metrics(y, oof_pred, oof_proba, weights), choice_rows


def evaluate_threshold_tuned_candidate_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    candidate: str,
    make_model: Callable[[], object],
) -> tuple[voter.CvResult, list[dict[str, object]]]:
    """Evaluate a fixed candidate with a threshold selected inside each outer fold."""
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    fold_id = np.full(len(work), -1, dtype=int)
    choice_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        inner_oof_p1 = np.full(len(outer_train_idx), np.nan, dtype=float)
        outer_train_y = y[outer_train_idx]

        for inner_train_rel, threshold_rel in inner.split(
            x_base.iloc[outer_train_idx],
            outer_train_y,
        ):
            inner_train_idx = outer_train_idx[inner_train_rel]
            threshold_idx = outer_train_idx[threshold_rel]
            x_train, x_threshold = _prepare_split_arrays(
                x_base,
                inner_train_idx,
                threshold_idx,
            )
            model = make_model()
            model.fit(x_train, y[inner_train_idx], sample_weight=weights[inner_train_idx])
            inner_oof_p1[threshold_rel] = model.predict_proba(x_threshold)[:, 1]

        threshold, inner_ba = _choose_threshold(
            outer_train_y,
            inner_oof_p1,
            weights[outer_train_idx],
        )

        x_train, x_val = _prepare_split_arrays(x_base, outer_train_idx, outer_val_idx)
        model = make_model()
        model.fit(x_train, y[outer_train_idx], sample_weight=weights[outer_train_idx])
        proba = model.predict_proba(x_val)
        pred = (proba[:, 1] >= threshold).astype(int)
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = pred
        fold_id[outer_val_idx] = outer_fold
        outer_metrics = voter.weighted_metrics(
            y[outer_val_idx],
            pred,
            proba,
            weights[outer_val_idx],
        )
        choice_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "candidate": candidate,
                "outer_fold": outer_fold,
                "selected_threshold": threshold,
                "inner_balanced_accuracy": inner_ba,
                "outer_balanced_accuracy": outer_metrics["balanced_accuracy"],
                "outer_log_loss": outer_metrics["log_loss"],
            }
        )
        print(
            f"  {candidate:22s} outer fold {outer_fold}: "
            f"threshold={threshold:.3f} inner_BA={inner_ba:.4f} "
            f"outer_BA={outer_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::{candidate}::threshold",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return result, choice_rows


def evaluate_threshold_tuned_ensemble_result(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    ensemble_name: str,
    recipe: list[tuple[str, float]],
) -> tuple[voter.CvResult, list[dict[str, object]]]:
    """Evaluate a fixed ensemble with a threshold selected inside each outer fold."""
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    oof_pred = np.full(len(work), -1, dtype=int)
    fold_id = np.full(len(work), -1, dtype=int)
    choice_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        inner_oof_p1 = np.full(len(outer_train_idx), np.nan, dtype=float)
        outer_train_y = y[outer_train_idx]

        for inner_train_rel, threshold_rel in inner.split(
            x_base.iloc[outer_train_idx], outer_train_y
        ):
            inner_train_idx = outer_train_idx[inner_train_rel]
            threshold_idx = outer_train_idx[threshold_rel]
            inner_proba = _fit_ensemble_proba_split(
                x_base,
                y,
                weights,
                inner_train_idx,
                threshold_idx,
                recipe,
            )
            inner_oof_p1[threshold_rel] = inner_proba[:, 1]

        threshold, inner_ba = _choose_threshold(
            outer_train_y,
            inner_oof_p1,
            weights[outer_train_idx],
        )

        proba = _fit_ensemble_proba_split(
            x_base,
            y,
            weights,
            outer_train_idx,
            outer_val_idx,
            recipe,
        )
        pred = (proba[:, 1] >= threshold).astype(int)
        oof_proba[outer_val_idx] = proba
        oof_pred[outer_val_idx] = pred
        fold_id[outer_val_idx] = outer_fold
        outer_metrics = voter.weighted_metrics(
            y[outer_val_idx],
            pred,
            proba,
            weights[outer_val_idx],
        )
        choice_rows.append(
            {
                "cohort": cohort,
                "config": config,
                "ensemble": ensemble_name,
                "outer_fold": outer_fold,
                "selected_threshold": threshold,
                "inner_balanced_accuracy": inner_ba,
                "outer_balanced_accuracy": outer_metrics["balanced_accuracy"],
                "outer_log_loss": outer_metrics["log_loss"],
            }
        )
        print(
            f"  {ensemble_name:26s} outer fold {outer_fold}: "
            f"threshold={threshold:.3f} inner_BA={inner_ba:.4f} "
            f"outer_BA={outer_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    metrics = voter.weighted_metrics(y, oof_pred, oof_proba, weights)
    result = voter.CvResult(
        cohort=cohort,
        config=f"{config}::{ensemble_name}::threshold",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
        sample_weight=weights,
        fold_id=fold_id,
        class_names=list(label_encoder.classes_),
        metrics=metrics,
    )
    return result, choice_rows


def evaluate_threshold_tuned_ensemble(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    ensemble_name: str,
    recipe: list[tuple[str, float]],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    result, choice_rows = evaluate_threshold_tuned_ensemble_result(
        df=df,
        cohort=cohort,
        config=config,
        feature_cols=feature_cols,
        ensemble_name=ensemble_name,
        recipe=recipe,
    )
    return result.metrics, choice_rows


def evaluate_nested_selection(
    df: pd.DataFrame,
    cohort: str,
    config: str,
    feature_cols: list[str],
    factories: dict[str, Callable[[], object]],
    candidates: list[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Outer-CV estimate where the candidate learner is chosen by inner CV."""
    work = df.copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[voter.TARGET_COL])
    if list(label_encoder.classes_) != ["Harris", "Trump"]:
        raise RuntimeError(f"Unexpected class order: {list(label_encoder.classes_)}")

    weights = work[voter.WEIGHT_COL].to_numpy(dtype=float)
    x_base = voter.encode_feature_frame(work, feature_cols)
    outer = StratifiedKFold(
        n_splits=voter.N_FOLDS,
        shuffle=True,
        random_state=voter.RANDOM_STATE,
    )
    oof_pred = np.full(len(work), -1, dtype=int)
    oof_proba = np.full((len(work), 2), np.nan, dtype=float)
    choice_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer.split(x_base, y)):
        inner = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=voter.RANDOM_STATE + outer_fold,
        )
        candidate_rows = []
        outer_train_y = y[outer_train_idx]
        for candidate in candidates:
            inner_metrics = []
            for inner_fold, (inner_train_rel, inner_val_rel) in enumerate(
                inner.split(x_base.iloc[outer_train_idx], outer_train_y)
            ):
                inner_train_idx = outer_train_idx[inner_train_rel]
                inner_val_idx = outer_train_idx[inner_val_rel]
                pred, proba = _fit_predict_split(
                    x_base,
                    y,
                    weights,
                    inner_train_idx,
                    inner_val_idx,
                    factories[candidate],
                )
                metrics = voter.weighted_metrics(
                    y[inner_val_idx],
                    pred,
                    proba,
                    weights[inner_val_idx],
                )
                inner_metrics.append(metrics)
                choice_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "candidate": candidate,
                        "selected": False,
                        "inner_balanced_accuracy": metrics["balanced_accuracy"],
                        "inner_log_loss": metrics["log_loss"],
                        "outer_balanced_accuracy": np.nan,
                        "outer_log_loss": np.nan,
                    }
                )

            candidate_rows.append(
                {
                    "candidate": candidate,
                    "mean_inner_balanced_accuracy": float(
                        np.mean([m["balanced_accuracy"] for m in inner_metrics])
                    ),
                    "mean_inner_log_loss": float(
                        np.mean([m["log_loss"] for m in inner_metrics])
                    ),
                }
            )

        selected = sorted(
            candidate_rows,
            key=lambda row: (
                -row["mean_inner_balanced_accuracy"],
                row["mean_inner_log_loss"],
                row["candidate"],
            ),
        )[0]["candidate"]

        pred, proba = _fit_predict_split(
            x_base,
            y,
            weights,
            outer_train_idx,
            outer_val_idx,
            factories[selected],
        )
        oof_pred[outer_val_idx] = pred
        oof_proba[outer_val_idx] = proba
        outer_metrics = voter.weighted_metrics(
            y[outer_val_idx],
            pred,
            proba,
            weights[outer_val_idx],
        )

        for row in candidate_rows:
            choice_rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "outer_fold": outer_fold,
                    "inner_fold": "mean",
                    "candidate": row["candidate"],
                    "selected": row["candidate"] == selected,
                    "inner_balanced_accuracy": row["mean_inner_balanced_accuracy"],
                    "inner_log_loss": row["mean_inner_log_loss"],
                    "outer_balanced_accuracy": (
                        outer_metrics["balanced_accuracy"]
                        if row["candidate"] == selected
                        else np.nan
                    ),
                    "outer_log_loss": (
                        outer_metrics["log_loss"] if row["candidate"] == selected else np.nan
                    ),
                }
            )

        print(
            f"  outer fold {outer_fold}: selected={selected} "
            f"inner_BA={max(r['mean_inner_balanced_accuracy'] for r in candidate_rows):.4f} "
            f"outer_BA={outer_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    return voter.weighted_metrics(y, oof_pred, oof_proba, weights), choice_rows


def _cohort_frame(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    return df.loc[voter.COHORTS[cohort](df)].copy().reset_index(drop=True)


def _write_summary(rows: list[dict[str, object]], candidates: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(OUT_DIR / "model_benchmark.csv", index=False)

    best_rows = (
        benchmark.sort_values(
            ["cohort", "config", "balanced_accuracy", "log_loss"],
            ascending=[True, True, False, True],
        )
        .groupby(["cohort", "config"], as_index=False)
        .head(1)
    )
    best_rows.to_csv(OUT_DIR / "winner_by_balanced_accuracy.csv", index=False)

    baseline = benchmark[benchmark["candidate"] == BASELINE_CANDIDATE].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in benchmark.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "delta_vs_current.csv", index=False)

    metadata = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "candidates": candidates,
        "model_selection_note": (
            "model_benchmark.csv is a non-nested benchmark. If present, "
            "nested_selection_results.csv is the outer/inner-CV model-selection "
            "estimate for the specific cohorts and configs that were run. "
            "Promote a candidate only if the thesis accepts the methodological "
            "tradeoff versus a fixed learner across configs."
        ),
        "cv_contract": {
            "n_folds": voter.N_FOLDS,
            "survey_weight": voter.WEIGHT_CODE,
            "target": voter.TARGET_CODE,
            "imputation": "train-fold medians",
            "scaling": "train-fold StandardScaler",
        },
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_nested_summary(
    result_rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    candidates: list[str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(OUT_DIR / "nested_selection_results.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(OUT_DIR / "nested_selection_choices.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["nested_selection"] = {
        "candidates": candidates,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "selection_metric": "mean inner weighted balanced accuracy",
        "tie_breakers": ["mean inner weighted log_loss", "candidate name"],
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "Promote a candidate only if the thesis accepts the methodological "
        "tradeoff versus a fixed learner across configs."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_calibration_summary(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = pd.DataFrame(rows)
    calibration.to_csv(OUT_DIR / "calibration_results.csv", index=False)

    baseline = calibration[calibration["calibration"] == "none"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in calibration.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "calibration": row.calibration,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "calibration_delta_vs_uncalibrated.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["calibration"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "methods": CALIBRATION_METHODS,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "note": (
            "Calibration is fit only on inner calibration folds inside each "
            "outer training fold, then evaluated on the outer validation fold."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_threshold_summary(
    result_rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(OUT_DIR / "threshold_results.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(OUT_DIR / "threshold_choices.csv", index=False)

    threshold_results = pd.DataFrame(result_rows)
    canonical = pd.read_csv(voter.OUT_DIR / "cohort_results.csv")
    merged = threshold_results.merge(
        canonical,
        on=["cohort", "config"],
        suffixes=("_threshold", "_canonical"),
        validate="one_to_one",
    )
    deltas = []
    for row in merged.itertuples(index=False):
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "delta_balanced_accuracy": row.balanced_accuracy_threshold
                - row.balanced_accuracy_canonical,
                "delta_macro_f1": row.macro_f1_threshold - row.macro_f1_canonical,
                "delta_log_loss": row.log_loss_threshold - row.log_loss_canonical,
                "delta_brier_score": row.brier_score_threshold - row.brier_score_canonical,
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "threshold_delta_vs_current.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["threshold_tuning"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Thresholds are selected from inner out-of-fold probabilities inside "
            "each outer training fold, then applied to outer-fold probabilities."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "calibration_results.csv and threshold_results.csv test fixed-learner "
        "probability calibration and threshold tuning on the same core configs. "
        "Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_missingness_summary(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missingness = pd.DataFrame(rows)
    missingness.to_csv(OUT_DIR / "missingness_results.csv", index=False)

    baseline = missingness[missingness["missingness"] == "baseline"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in missingness.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "missingness": row.missingness,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "missingness_delta_vs_baseline.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["missingness_indicators"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "variants": MISSINGNESS_VARIANTS,
        "n_outer_folds": voter.N_FOLDS,
        "note": (
            "Missingness indicators are row-level feature-is-missing flags added "
            "before fold-local median imputation and scaling."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "calibration_results.csv, threshold_results.csv, and "
        "missingness_results.csv test fixed-learner calibration, threshold "
        "tuning, and missingness-indicator feature engineering on the same core "
        "configs. Promote a variant only if the thesis accepts the "
        "methodological tradeoff versus the simplest fixed-learner feature "
        "comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_nan_summary(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nan_results = pd.DataFrame(rows)
    nan_results.to_csv(OUT_DIR / "xgb_nan_results.csv", index=False)

    baseline = nan_results[nan_results["xgb_nan_handling"] == "median_impute_scale"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in nan_results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "xgb_nan_handling": row.xgb_nan_handling,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "xgb_nan_delta_vs_baseline.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_nan_handling"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "variants": XGB_NAN_VARIANTS,
        "n_outer_folds": voter.N_FOLDS,
        "note": (
            "median_impute_scale uses the canonical fold-local imputation and "
            "StandardScaler path; native_nan_no_scale passes numeric NaNs "
            "directly to XGBoost and skips scaling."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "calibration_results.csv, threshold_results.csv, missingness_results.csv, "
        "and xgb_nan_results.csv test fixed-learner calibration, threshold tuning, "
        "missingness indicators, and native XGBoost NaN handling on the same core "
        "configs. Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_class_balance_summary(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_balance = pd.DataFrame(rows)
    class_balance.to_csv(OUT_DIR / "class_balance_results.csv", index=False)

    baseline = class_balance[
        class_balance["class_balance"] == "survey_weight_only"
    ].set_index(["cohort", "config"])
    deltas = []
    for row in class_balance.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "class_balance": row.class_balance,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "class_balance_delta_vs_baseline.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["class_balance_training"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "variants": CLASS_BALANCE_VARIANTS,
        "n_outer_folds": voter.N_FOLDS,
        "note": (
            "survey_x_fold_class_balance multiplies survey weights in each "
            "training fold by inverse class-weight factors, then evaluates "
            "held-out predictions with the original survey weights."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "calibration_results.csv, threshold_results.csv, missingness_results.csv, "
        "xgb_nan_results.csv, and class_balance_results.csv test fixed-learner "
        "calibration, threshold tuning, missingness indicators, native XGBoost "
        "NaN handling, and class-balanced training weights on the same core "
        "configs. Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_training_weight_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training_weights = pd.DataFrame(rows)
    training_weights.to_csv(OUT_DIR / "training_weight_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "training_weight_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = training_weights[
        training_weights["training_weight"] == "survey_weight"
    ].set_index(["cohort", "config"])
    deltas = []
    for row in training_weights.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "training_weight": row.training_weight,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "training_weight_delta_vs_baseline.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["training_weight_sensitivity"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "variants": TRAINING_WEIGHT_VARIANTS,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Training weights are changed inside each training fold and rescaled "
            "to preserve the original train-fold weight sum. Evaluation always "
            "uses the original survey weights."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "training_weight_results.csv tests alternative train-fold weight patterns "
        "for the fixed XGBoost learner while keeping survey-weighted evaluation. "
        "Promote a variant only if the thesis accepts the methodological tradeoff "
        "versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_profile_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "xgb_profile_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_profile_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["candidate"] == BASELINE_CANDIDATE].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_profile_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_profile_sweep"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "candidates": candidates,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Fixed XGBoost depth, shrinkage, subsampling, and regularization "
            "profiles are compared against xgb_current on the same out-of-fold "
            "rows. This keeps the model family fixed while testing nearby "
            "training profiles."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_profile_results.csv tests fixed XGBoost-only hyperparameter "
        "profiles against xgb_current. Promote a variant only if the thesis "
        "accepts the methodological tradeoff versus the simplest fixed-learner "
        "feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_issue_encoding_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "issue_encoding_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "issue_encoding_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["encoding"] == "numeric_ordinal"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "canonical_config": row.canonical_config,
                "encoding": row.encoding,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "issue_encoding_delta_vs_numeric.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["issue_encoding"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "configs": ISSUE_ENCODING_CONFIGS,
        "canonical_config_map": ISSUE_ENCODING_CONFIG_MAP,
        "variants": ISSUE_ENCODING_VARIANTS,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "numeric_ordinal is the canonical numeric issue coding. issue_onehot "
            "treats issue responses as categorical dummy variables with a missing "
            "category while keeping the same fixed XGBoost learner and "
            "survey-weighted evaluation."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "issue_encoding_results.csv tests categorical issue-response encoding "
        "against the canonical numeric issue coding. Promote a variant only if "
        "the thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_political_encoding_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "political_encoding_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "political_encoding_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["encoding"] == "numeric_ordinal"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "canonical_config": row.canonical_config,
                "encoding": row.encoding,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "political_encoding_delta_vs_numeric.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["political_encoding"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "configs": POLITICAL_ENCODING_CONFIGS,
        "canonical_config_map": POLITICAL_ENCODING_CONFIG_MAP,
        "variants": POLITICAL_ENCODING_VARIANTS,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "numeric_ordinal is the canonical party-ID and ideology coding. "
            "political_onehot treats party ID and ideology as categorical dummy "
            "variables while keeping issue items in their canonical numeric form."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "political_encoding_results.csv tests categorical party-ID/ideology "
        "encoding against canonical numeric coding. Promote a variant only if "
        "the thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_early_stop_summary(
    rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "xgb_early_stop_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        OUT_DIR / "xgb_early_stop_fold_metrics.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_early_stop_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["early_stop_variant"] == "fixed_300"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "early_stop_variant": row.early_stop_variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_early_stop_delta_vs_fixed.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_early_stopping"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "variants": XGB_EARLY_STOP_VARIANTS,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "early_stop_eval_fraction": 0.15,
        "early_stopping_rounds": 25,
        "note": (
            "The early-stop variant splits each outer training fold into a model "
            "fit subset and a 15% early-stopping subset. The held-out outer fold "
            "is never used for stopping or preprocessing."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_early_stop_results.csv tests fold-local early stopping against "
        "the canonical fixed 300-tree XGBoost model. Promote a variant only if "
        "the thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_optuna_summary(
    rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    trial_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    n_trials: int,
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "xgb_optuna_results.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "xgb_optuna_outer_choices.csv",
        index=False,
    )
    pd.DataFrame(trial_rows).to_csv(
        OUT_DIR / "xgb_optuna_trial_history.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_optuna_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["variant"] == "fixed_300"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_optuna_delta_vs_fixed.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_optuna_tuning"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "variants": XGB_OPTUNA_VARIANTS,
        "configs": configs,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_trials": n_trials,
        "n_bootstrap": n_bootstrap,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Optuna tunes XGBoost hyperparameters inside each outer training "
            "fold. The outer held-out fold is never used for hyperparameter "
            "selection or preprocessing."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_optuna_results.csv tests same-family fold-local Optuna tuning "
        "against the canonical fixed 300-tree XGBoost model. Promote a variant "
        "only if the thesis accepts the methodological tradeoff versus the "
        "simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_seed_bag_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    seeds: list[int],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "xgb_seed_bagging_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_seed_bagging_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["variant"] == "xgb_current"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_seed_bagging_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_seed_bagging"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "variants": XGB_SEED_BAG_VARIANTS,
        "configs": configs,
        "seeds": seeds,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "xgb_seed_bag_5 averages probabilities from the canonical XGBoost "
            "profile fit with five deterministic random seeds. Evaluation uses "
            "the same outer folds and original survey weights."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_seed_bagging_results.csv tests same-profile random-seed bagging "
        "against xgb_current. Promote a variant only if the thesis accepts the "
        "methodological tradeoff versus the simplest fixed-learner feature "
        "comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_profile_threshold_summary(
    rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "xgb_profile_threshold_results.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "xgb_profile_threshold_choices.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_profile_threshold_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["variant"] == "xgb_current_threshold_0.5"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_profile_threshold_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_profile_threshold_tuning"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "candidates": candidates,
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Thresholds are selected from inner out-of-fold probabilities inside "
            "each outer training fold for fixed XGBoost profile candidates. "
            "Rows compare thresholded profiles against xgb_current at threshold 0.5."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_profile_threshold_results.csv tests inner-CV threshold tuning for "
        "fixed XGBoost profiles against xgb_current at threshold 0.5. Promote a "
        "variant only if the thesis accepts the methodological tradeoff versus "
        "the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_profile_calibration_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "xgb_profile_calibration_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_profile_calibration_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[
        (results["candidate"] == BASELINE_CANDIDATE)
        & (results["calibration"] == "none")
    ].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "calibration": row.calibration,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_profile_calibration_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_profile_calibration"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "candidates": candidates,
        "methods": methods,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Fold-local probability calibration is fit for selected fixed "
            "XGBoost profile candidates inside each outer training fold. Rows "
            "compare calibrated profiles against uncalibrated xgb_current."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_profile_calibration_results.csv tests fold-local probability "
        "calibration for selected fixed XGBoost profiles against xgb_current. "
        "Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_profile_calibration_threshold_summary(
    rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_DIR / "xgb_profile_calibration_threshold_results.csv",
        index=False,
    )
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "xgb_profile_calibration_threshold_choices.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_profile_calibration_threshold_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[
        (results["candidate"] == BASELINE_CANDIDATE)
        & (results["variant"] == "xgb_current_none_threshold_0.5")
    ].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "calibration": row.calibration,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_profile_calibration_threshold_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_profile_calibration_threshold"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "candidates": candidates,
        "methods": methods,
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Selected fixed XGBoost profiles are calibrated with inner-fold "
            "sigmoid or isotonic calibrators and then threshold-tuned using "
            "separate inner out-of-fold calibrated probabilities inside each "
            "outer training fold. Rows compare against uncalibrated "
            "xgb_current at threshold 0.5."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_profile_calibration_threshold_results.csv tests fold-local "
        "probability calibration plus threshold tuning for selected fixed "
        "XGBoost profiles against xgb_current. Promote a variant only if the "
        "thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_profile_blend_grid_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "xgb_profile_blend_grid_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_profile_blend_grid_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["blend"] == "xgb_current"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "blend": row.blend,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_profile_blend_grid_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_profile_blend_grid"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "recipes": recipes,
        "recipe_definitions": {
            name: [
                {
                    "candidate": candidate,
                    "weight": weight,
                }
                for candidate, weight in XGB_PROFILE_BLEND_RECIPES[name]
            ]
            for name in recipes
        },
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "A bounded grid of fixed probability weights averages canonical "
            "xgb_current probabilities with selected fixed XGBoost profile "
            "probabilities on the same outer folds. Rows compare each "
            "predeclared same-family blend against xgb_current."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_profile_blend_grid_results.csv tests fixed probability blends of "
        "selected XGBoost profiles against xgb_current. Promote a variant only "
        "if the thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_xgb_profile_blend_threshold_summary(
    rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_DIR / "xgb_profile_blend_threshold_results.csv",
        index=False,
    )
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "xgb_profile_blend_threshold_choices.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "xgb_profile_blend_threshold_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["variant"] == "xgb_current_threshold_0.5"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "blend": row.blend,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "xgb_profile_blend_threshold_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["xgb_profile_blend_threshold"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "recipes": recipes,
        "recipe_definitions": {
            name: [
                {
                    "candidate": candidate,
                    "weight": weight,
                }
                for candidate, weight in XGB_PROFILE_BLEND_RECIPES[name]
            ]
            for name in recipes
        },
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "The strongest same-family XGBoost profile blends from the fixed "
            "blend grid are threshold-tuned with inner out-of-fold blend "
            "probabilities inside each outer training fold."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "xgb_profile_blend_threshold_results.csv tests inner-CV threshold "
        "tuning for selected same-family XGBoost profile blends. Promote a "
        "variant only if the thesis accepts the methodological tradeoff versus "
        "the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_extended_candidate_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    candidates: list[str],
    configs: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "extended_candidate_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "extended_candidate_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["candidate"] == BASELINE_CANDIDATE].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "extended_candidate_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["extended_candidate_bootstrap"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "candidates": candidates,
        "configs": configs,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Additional fixed learners already available in candidate_factories "
            "are compared against xgb_current on the same out-of-fold rows. "
            "This artifact is separate from candidate_pairwise_bootstrap.csv so "
            "the original core-candidate evidence remains stable."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "extended_candidate_results.csv tests additional fixed learner families "
        "against xgb_current. Promote a variant only if the thesis accepts the "
        "methodological tradeoff versus the simplest fixed-learner feature "
        "comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_extended_candidate_threshold_summary(
    rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "extended_candidate_threshold_results.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "extended_candidate_threshold_choices.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "extended_candidate_threshold_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["variant"] == "xgb_current_threshold_0.5"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "extended_candidate_threshold_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["extended_candidate_threshold_tuning"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "candidates": candidates,
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Decision thresholds are selected from inner out-of-fold "
            "probabilities inside each outer training fold for additional fixed "
            "learner families. Rows compare thresholded candidates against "
            "xgb_current at threshold 0.5."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "extended_candidate_threshold_results.csv tests inner-CV threshold "
        "tuning for additional fixed learner families against xgb_current at "
        "threshold 0.5. Promote a variant only if the thesis accepts the "
        "methodological tradeoff versus the simplest fixed-learner feature "
        "comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_extended_candidate_calibration_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "extended_candidate_calibration_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "extended_candidate_calibration_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[
        (results["candidate"] == BASELINE_CANDIDATE)
        & (results["calibration"] == "none")
    ].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "calibration": row.calibration,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "extended_candidate_calibration_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["extended_candidate_calibration"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "candidates": candidates,
        "methods": methods,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Additional fixed learner families are calibrated with inner-fold "
            "sigmoid or isotonic calibrators inside each outer training fold. "
            "Rows compare calibrated candidates against uncalibrated xgb_current."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "extended_candidate_calibration_results.csv tests fold-local "
        "probability calibration for additional fixed learner families against "
        "xgb_current. Promote a variant only if the thesis accepts the "
        "methodological tradeoff versus the simplest fixed-learner feature "
        "comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_extended_candidate_calibration_threshold_summary(
    rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_DIR / "extended_candidate_calibration_threshold_results.csv",
        index=False,
    )
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "extended_candidate_calibration_threshold_choices.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "extended_candidate_calibration_threshold_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[
        (results["candidate"] == BASELINE_CANDIDATE)
        & (results["variant"] == "xgb_current_none_threshold_0.5")
    ].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "candidate": row.candidate,
                "calibration": row.calibration,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "extended_candidate_calibration_threshold_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["extended_candidate_calibration_threshold_tuning"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "candidates": candidates,
        "methods": methods,
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Additional fixed learner families are calibrated with inner-fold "
            "sigmoid or isotonic calibrators and then threshold-tuned using "
            "separate inner out-of-fold calibrated probabilities inside each "
            "outer training fold. Rows compare against uncalibrated "
            "xgb_current at threshold 0.5."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "extended_candidate_calibration_threshold_results.csv tests fold-local "
        "probability calibration plus threshold tuning for additional fixed "
        "learner families against xgb_current. Promote a variant only if the "
        "thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_calibrated_extended_blend_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "calibrated_extended_blend_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "calibrated_extended_blend_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["blend"] == "xgb_current"].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "blend": row.blend,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "calibrated_extended_blend_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["calibrated_extended_blends"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "recipes": recipes,
        "recipe_definitions": {
            name: [
                {
                    "candidate": candidate,
                    "calibration": calibration,
                    "weight": weight,
                }
                for candidate, calibration, weight in CALIBRATED_EXTENDED_BLEND_RECIPES[
                    name
                ]
            ]
            for name in recipes
        },
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Fixed probability blends average canonical xgb_current probabilities "
            "with fold-local calibrated extended learner probabilities on the "
            "same outer folds. Rows compare blends against xgb_current."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "calibrated_extended_blend_results.csv tests fixed probability blends "
        "of xgb_current with calibrated extended learners. Promote a variant "
        "only if the thesis accepts the methodological tradeoff versus the "
        "simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_calibrated_extended_blend_threshold_summary(
    rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_DIR / "calibrated_extended_blend_threshold_results.csv",
        index=False,
    )
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "calibrated_extended_blend_threshold_choices.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "calibrated_extended_blend_threshold_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["variant"] == "xgb_current_threshold_0.5"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "blend": row.blend,
                "variant": row.variant,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "calibrated_extended_blend_threshold_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["calibrated_extended_blend_threshold_tuning"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "recipes": recipes,
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Fixed probability blends of xgb_current and calibrated extended "
            "learners are threshold-tuned with inner out-of-fold blend "
            "probabilities inside each outer training fold."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "calibrated_extended_blend_threshold_results.csv tests threshold-tuned "
        "fixed probability blends of xgb_current with calibrated extended "
        "learners. Promote a variant only if the thesis accepts the "
        "methodological tradeoff versus the simplest fixed-learner feature "
        "comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_calibrated_extended_blend_grid_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "calibrated_extended_blend_grid_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "calibrated_extended_blend_grid_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["blend"] == "xgb_current"].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "blend": row.blend,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "calibrated_extended_blend_grid_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["calibrated_extended_blend_grid"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "recipes": recipes,
        "recipe_definitions": {
            name: [
                {
                    "candidate": candidate,
                    "calibration": calibration,
                    "weight": weight,
                }
                for candidate, calibration, weight in CALIBRATED_EXTENDED_BLEND_GRID_RECIPES[
                    name
                ]
            ]
            for name in recipes
        },
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "A bounded grid of fixed probability weights averages canonical "
            "xgb_current probabilities with fold-local calibrated extended "
            "learner probabilities on the same outer folds. Rows compare each "
            "predeclared weight recipe against xgb_current."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "calibrated_extended_blend_grid_results.csv tests a bounded, fixed "
        "weight grid of probability blends. Promote a variant only if the "
        "thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_calibrated_extended_blend_grid_core_summary(
    rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_DIR / "calibrated_extended_blend_grid_core_results.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "calibrated_extended_blend_grid_core_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["blend"] == "xgb_current"].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "blend": row.blend,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "calibrated_extended_blend_grid_core_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["calibrated_extended_blend_grid_core"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "configs": configs,
        "recipes": recipes,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "The BA-positive calibrated blend-grid recipes are re-run across "
            "the core feature configurations to test whether the headline-cell "
            "gain generalizes to the fixed-learner feature-comparison contract."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "calibrated_extended_blend_grid_core_results.csv tests whether the "
        "BA-positive calibrated blend-grid recipes generalize across core "
        "feature configurations. Promote a variant only if the thesis accepts "
        "the methodological tradeoff versus the simplest fixed-learner feature "
        "comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_fixed_candidate_bootstrap_summary(
    rows: list[dict[str, object]],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "candidate_pairwise_bootstrap.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["fixed_candidate_pairwise_bootstrap"] = {
        "baseline_candidate": BASELINE_CANDIDATE,
        "candidates": candidates,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "comparison": "candidate minus xgb_current on the same out-of-fold rows",
        "random_state": voter.RANDOM_STATE,
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "candidate_pairwise_bootstrap.csv compares fixed alternative learners "
        "against xgb_current on the same out-of-fold rows for the listed core "
        "configs. nested_selection_results.csv is the outer/inner-CV "
        "model-selection estimate for the listed nested-selection cohorts and "
        "configs. calibration_results.csv, threshold_results.csv, "
        "missingness_results.csv, xgb_nan_results.csv, and "
        "class_balance_results.csv test fixed-learner calibration, threshold "
        "tuning, missingness indicators, native XGBoost NaN handling, and "
        "class-balanced training weights. Promote a variant only if the thesis "
        "accepts the methodological tradeoff versus the simplest fixed-learner "
        "feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_feature_bootstrap_summary(
    rows: list[dict[str, object]],
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "feature_pairwise_bootstrap.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["feature_pairwise_bootstrap"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "pairs": [[reference, comparison] for reference, comparison in pairs],
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "comparison": "reference feature config minus comparison feature config on the same out-of-fold rows",
        "random_state": voter.RANDOM_STATE,
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "feature_pairwise_bootstrap.csv compares adjacent fixed-XGBoost feature "
        "sets on the same out-of-fold rows. candidate_pairwise_bootstrap.csv "
        "compares fixed alternative learners against xgb_current for the listed "
        "core configs. nested_selection_results.csv is the outer/inner-CV "
        "model-selection estimate for the listed nested-selection cohorts and "
        "configs. Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_domain_pruning_summary(
    result_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "domain_pruning_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "domain_pruning_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["variant"] == "baseline"].set_index(["cohort", "config"])
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "variant": row.variant,
                "domain_removed": row.domain_removed,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "domain_pruning_delta_vs_baseline.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["domain_pruning"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "configs": DOMAIN_PRUNING_CONFIGS,
        "domains": list(voter.ISSUE_DOMAINS),
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Each variant removes one thesis issue domain from the item-level "
            "issue feature set while keeping the same fixed XGBoost learner. "
            "Paired bootstrap rows compare minus-domain variants against the "
            "same config with all issue domains."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "feature_pairwise_bootstrap.csv compares adjacent fixed-XGBoost feature "
        "sets on the same out-of-fold rows. domain_pruning_results.csv tests "
        "leave-one-issue-domain-out item-level feature sets. Promote a variant "
        "only if the thesis accepts the methodological tradeoff versus the "
        "simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_issue_pca_summary(
    result_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "issue_pca_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "issue_pca_fold_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "issue_pca_pairwise_bootstrap.csv",
        index=False,
    )

    canonical = pd.read_csv(voter.OUT_DIR / "cohort_results.csv")
    merged = results.merge(
        canonical,
        left_on=["cohort", "canonical_config"],
        right_on=["cohort", "config"],
        suffixes=("_pca", "_canonical"),
        validate="many_to_one",
    )
    deltas = []
    for row in merged.itertuples(index=False):
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config_pca,
                "canonical_config": row.canonical_config,
                "n_components": row.n_components,
                "delta_balanced_accuracy": row.balanced_accuracy_pca
                - row.balanced_accuracy_canonical,
                "delta_macro_f1": row.macro_f1_pca - row.macro_f1_canonical,
                "delta_log_loss": row.log_loss_pca - row.log_loss_canonical,
                "delta_brier_score": row.brier_score_pca - row.brier_score_canonical,
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "issue_pca_delta_vs_canonical.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["issue_pca"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "configs": ISSUE_PCA_CONFIGS,
        "n_components": ISSUE_PCA_COMPONENTS,
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "Issue items are fold-locally imputed and standardized, then PCA is "
            "fit only on the outer training fold. Paired bootstrap rows compare "
            "each PCA config against the matching item-level canonical config."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "issue_pca_results.csv tests fold-local latent issue components against "
        "the matching item-level fixed-XGBoost configs. Promote a variant only "
        "if the thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_issue_topk_summary(
    result_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "issue_topk_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "issue_topk_fold_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(
        OUT_DIR / "issue_topk_selected_features.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "issue_topk_pairwise_bootstrap.csv",
        index=False,
    )

    canonical = pd.read_csv(voter.OUT_DIR / "cohort_results.csv")
    merged = results.merge(
        canonical,
        left_on=["cohort", "canonical_config"],
        right_on=["cohort", "config"],
        suffixes=("_topk", "_canonical"),
        validate="many_to_one",
    )
    deltas = []
    for row in merged.itertuples(index=False):
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config_topk,
                "canonical_config": row.canonical_config,
                "top_k": row.top_k,
                "delta_balanced_accuracy": row.balanced_accuracy_topk
                - row.balanced_accuracy_canonical,
                "delta_macro_f1": row.macro_f1_topk - row.macro_f1_canonical,
                "delta_log_loss": row.log_loss_topk - row.log_loss_canonical,
                "delta_brier_score": row.brier_score_topk - row.brier_score_canonical,
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "issue_topk_delta_vs_canonical.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["issue_topk_selection"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "configs": ISSUE_TOPK_CONFIGS,
        "canonical_config_map": ISSUE_TOPK_CONFIG_MAP,
        "top_k_values": sorted({int(row["top_k"]) for row in result_rows}),
        "n_outer_folds": voter.N_FOLDS,
        "n_bootstrap": n_bootstrap,
        "scoring": "absolute weighted standardized mean gap between Trump and Harris",
        "note": (
            "Issue items are selected inside each outer training fold after "
            "fold-local imputation and standardization. Paired bootstrap rows "
            "compare each top-k config against the matching full item-level "
            "canonical config."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "issue_topk_results.csv tests fold-local item-level issue selection "
        "against the matching full issue configs. Promote a variant only if "
        "the thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_two_stage_summary(
    result_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    stage1_rows: list[dict[str, object]],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "two_stage_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "two_stage_fold_metrics.csv", index=False)
    pd.DataFrame(stage1_rows).to_csv(OUT_DIR / "two_stage_stage1_summary.csv", index=False)

    canonical = pd.read_csv(voter.OUT_DIR / "cohort_results.csv")
    comparison_pairs = [
        ("predicted_issues_only", "demographics_only"),
        ("predicted_issues_only", "issues_only"),
        ("predicted_issues+party_id+ideology", "party_id+ideology"),
        ("predicted_issues+party_id+ideology", "issues+party_id+ideology"),
    ]
    deltas = []
    for ref_config, comparison_config in comparison_pairs:
        ref_rows = results[results["config"] == ref_config]
        comp_rows = canonical[canonical["config"] == comparison_config]
        merged = ref_rows.merge(
            comp_rows,
            on="cohort",
            suffixes=("_two_stage", "_canonical"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            deltas.append(
                {
                    "cohort": row.cohort,
                    "two_stage_config": ref_config,
                    "comparison_config": comparison_config,
                    "delta_balanced_accuracy": row.balanced_accuracy_two_stage
                    - row.balanced_accuracy_canonical,
                    "delta_macro_f1": row.macro_f1_two_stage - row.macro_f1_canonical,
                    "delta_log_loss": row.log_loss_two_stage - row.log_loss_canonical,
                    "delta_brier_score": row.brier_score_two_stage - row.brier_score_canonical,
                }
            )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "two_stage_delta_vs_canonical.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["two_stage_predicted_issues"] = {
        "stage1_predictors": "demographics_only",
        "stage1_model": "HistGradientBoostingRegressor",
        "stage2_model": BASELINE_CANDIDATE,
        "configs": TWO_STAGE_CONFIGS,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds_for_train_predictions": N_INNER_FOLDS,
        "note": (
            "Stage 2 train rows use inner out-of-fold predicted issue values; "
            "outer validation rows use issue models fit only on the outer "
            "training fold."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "calibration_results.csv, threshold_results.csv, missingness_results.csv, "
        "xgb_nan_results.csv, and two_stage_results.csv test fixed-learner "
        "calibration, threshold tuning, missingness indicators, native XGBoost "
        "NaN handling, and demographics-to-predicted-issues feature engineering. "
        "Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_two_stage_stage1_model_summary(
    result_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    stage1_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    stage1_models: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "two_stage_stage1_model_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        OUT_DIR / "two_stage_stage1_model_fold_metrics.csv",
        index=False,
    )
    pd.DataFrame(stage1_rows).to_csv(
        OUT_DIR / "two_stage_stage1_model_stage1_summary.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "two_stage_stage1_model_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["stage1_model"] == "histgb_current"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "stage1_model": row.stage1_model,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "two_stage_stage1_model_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["two_stage_stage1_model_sweep"] = {
        "stage1_predictors": "demographics_only",
        "stage1_models": stage1_models,
        "baseline_stage1_model": "histgb_current",
        "stage2_model": BASELINE_CANDIDATE,
        "configs": TWO_STAGE_CONFIGS,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds_for_train_predictions": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "This sweep tests whether the demography-to-predicted-issues branch "
            "depends on the fixed Stage 1 issue regressor. Stage 2 training rows "
            "use inner out-of-fold predicted issue values; outer validation rows "
            "use issue models fit only on the outer training fold."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "two_stage_stage1_model_results.csv tests fixed Stage 1 regressor "
        "alternatives inside the demographics-to-predicted-issues voter-intent "
        "chain. Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_two_stage_stage2_model_summary(
    result_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    stage1_rows: list[dict[str, object]],
    boot_rows: list[dict[str, object]],
    stage2_candidates: list[str],
    n_bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "two_stage_stage2_model_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        OUT_DIR / "two_stage_stage2_model_fold_metrics.csv",
        index=False,
    )
    pd.DataFrame(stage1_rows).to_csv(
        OUT_DIR / "two_stage_stage2_model_stage1_summary.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "two_stage_stage2_model_pairwise_bootstrap.csv",
        index=False,
    )

    baseline = results[results["stage2_model"] == BASELINE_CANDIDATE].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in results.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "stage1_model": row.stage1_model,
                "stage2_model": row.stage2_model,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "two_stage_stage2_model_delta_vs_current.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["two_stage_stage2_model_sweep"] = {
        "stage1_predictors": "demographics_only",
        "stage1_model": "histgb_current",
        "baseline_stage2_model": BASELINE_CANDIDATE,
        "stage2_candidates": stage2_candidates,
        "configs": TWO_STAGE_CONFIGS,
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds_for_train_predictions": N_INNER_FOLDS,
        "n_bootstrap": n_bootstrap,
        "note": (
            "This sweep keeps the current cross-fitted Stage 1 issue regressor "
            "fixed and tests whether alternative Stage 2 learners improve the "
            "demographics-to-predicted-issues voter-intent chain."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "two_stage_stage2_model_results.csv tests fixed Stage 2 learner "
        "alternatives inside the demographics-to-predicted-issues voter-intent "
        "chain. Promote a variant only if the thesis accepts the methodological "
        "tradeoff versus the simplest fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_domain_aggregate_summary(
    result_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "domain_aggregate_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "domain_aggregate_fold_metrics.csv", index=False)

    canonical = pd.read_csv(voter.OUT_DIR / "cohort_results.csv")
    comparison_pairs = [
        ("domain_issues_only", "issues_only"),
        ("domain_issues+demographics", "issues+demographics"),
        ("domain_issues+party_id+ideology", "issues+party_id+ideology"),
    ]
    deltas = []
    for domain_config, comparison_config in comparison_pairs:
        domain_rows = results[results["config"] == domain_config]
        comp_rows = canonical[canonical["config"] == comparison_config]
        merged = domain_rows.merge(
            comp_rows,
            on="cohort",
            suffixes=("_domain", "_canonical"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            deltas.append(
                {
                    "cohort": row.cohort,
                    "domain_config": domain_config,
                    "comparison_config": comparison_config,
                    "delta_balanced_accuracy": row.balanced_accuracy_domain
                    - row.balanced_accuracy_canonical,
                    "delta_macro_f1": row.macro_f1_domain - row.macro_f1_canonical,
                    "delta_log_loss": row.log_loss_domain - row.log_loss_canonical,
                    "delta_brier_score": row.brier_score_domain - row.brier_score_canonical,
                }
            )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "domain_aggregate_delta_vs_canonical.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["domain_aggregates"] = {
        "base_candidate": BASELINE_CANDIDATE,
        "configs": DOMAIN_AGGREGATE_CONFIGS,
        "n_outer_folds": voter.N_FOLDS,
        "n_domains": len(voter.ISSUE_DOMAINS),
        "note": (
            "Issue items are fold-locally imputed and standardized, then averaged "
            "within thesis issue domains before fitting the fixed voter-intent model."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "calibration_results.csv, threshold_results.csv, missingness_results.csv, "
        "xgb_nan_results.csv, two_stage_results.csv, and domain_aggregate_results.csv "
        "test fixed-learner calibration, threshold tuning, missingness indicators, "
        "native XGBoost NaN handling, demographics-to-predicted-issues feature "
        "engineering, and compact issue-domain aggregation. Promote a variant only "
        "if the thesis accepts the methodological tradeoff versus the simplest "
        "fixed-learner feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_ensemble_summary(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensembles = pd.DataFrame(rows)
    ensembles.to_csv(OUT_DIR / "ensemble_results.csv", index=False)

    baseline = ensembles[ensembles["ensemble"] == "xgb_current"].set_index(
        ["cohort", "config"]
    )
    deltas = []
    for row in ensembles.itertuples(index=False):
        key = (row.cohort, row.config)
        base = baseline.loc[key]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "ensemble": row.ensemble,
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(OUT_DIR / "ensemble_delta_vs_current.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["soft_voting_ensembles"] = {
        "recipes": ENSEMBLE_RECIPES,
        "n_outer_folds": voter.N_FOLDS,
        "note": (
            "Each recipe trains fixed candidate learners on the same outer "
            "training fold and averages predicted probabilities on the outer "
            "validation fold."
        ),
    }
    metadata["model_selection_note"] = (
        "model_benchmark.csv is a non-nested benchmark. "
        "nested_selection_results.csv is the outer/inner-CV model-selection "
        "estimate for the listed nested-selection cohorts and configs. "
        "calibration_results.csv, threshold_results.csv, missingness_results.csv, "
        "xgb_nan_results.csv, two_stage_results.csv, domain_aggregate_results.csv, "
        "and ensemble_results.csv test fixed-learner calibration, threshold tuning, "
        "missingness indicators, native XGBoost NaN handling, demographics-to-"
        "predicted-issues feature engineering, compact issue-domain aggregation, "
        "and fixed soft-voting ensembles. Promote a variant only if the thesis "
        "accepts the methodological tradeoff versus the simplest fixed-learner "
        "feature comparison."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_ensemble_bootstrap_summary(
    rows: list[dict[str, object]],
    n_bootstrap: int,
    ensembles: list[str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "ensemble_pairwise_bootstrap.csv", index=False)

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.setdefault("soft_voting_ensembles", {})
    metadata["soft_voting_ensembles"]["paired_bootstrap"] = {
        "comparison": "candidate ensemble minus xgb_current on the same out-of-fold rows",
        "ensembles": ensembles,
        "n_bootstrap": n_bootstrap,
        "random_state": voter.RANDOM_STATE,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_ensemble_threshold_summary(
    result_rows: list[dict[str, object]],
    choice_rows: list[dict[str, object]],
    ensembles: list[str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "ensemble_threshold_results.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(
        OUT_DIR / "ensemble_threshold_choices.csv",
        index=False,
    )

    ensemble_path = OUT_DIR / "ensemble_results.csv"
    if not ensemble_path.exists():
        raise RuntimeError(
            "ensemble_results.csv is required before writing ensemble threshold deltas"
        )
    untuned = pd.read_csv(ensemble_path)
    same_ensemble = untuned.merge(
        results,
        on=["cohort", "config", "ensemble"],
        suffixes=("_untuned", "_threshold"),
        validate="one_to_one",
    )
    deltas = []
    for row in same_ensemble.itertuples(index=False):
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "ensemble": row.ensemble,
                "comparison": "same_ensemble_threshold_0.5",
                "delta_balanced_accuracy": row.balanced_accuracy_threshold
                - row.balanced_accuracy_untuned,
                "delta_macro_f1": row.macro_f1_threshold - row.macro_f1_untuned,
                "delta_log_loss": row.log_loss_threshold - row.log_loss_untuned,
                "delta_brier_score": row.brier_score_threshold - row.brier_score_untuned,
            }
        )

    current = untuned[untuned["ensemble"] == "xgb_current"].set_index(["cohort", "config"])
    for row in results.itertuples(index=False):
        base = current.loc[(row.cohort, row.config)]
        deltas.append(
            {
                "cohort": row.cohort,
                "config": row.config,
                "ensemble": row.ensemble,
                "comparison": "xgb_current_threshold_0.5",
                "delta_balanced_accuracy": row.balanced_accuracy
                - base["balanced_accuracy"],
                "delta_macro_f1": row.macro_f1 - base["macro_f1"],
                "delta_log_loss": row.log_loss - base["log_loss"],
                "delta_brier_score": row.brier_score - base["brier_score"],
            }
        )
    pd.DataFrame(deltas).to_csv(
        OUT_DIR / "ensemble_threshold_deltas.csv",
        index=False,
    )

    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.setdefault("soft_voting_ensembles", {})
    metadata["soft_voting_ensembles"]["threshold_tuning"] = {
        "ensembles": ensembles,
        "threshold_grid_min": float(THRESHOLD_GRID.min()),
        "threshold_grid_max": float(THRESHOLD_GRID.max()),
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "n_outer_folds": voter.N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "selection_metric": "inner weighted balanced accuracy",
        "note": (
            "Thresholds are selected from inner out-of-fold ensemble probabilities "
            "inside each outer training fold, then applied to outer-fold ensemble probabilities."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run(cohorts: list[str], configs: list[str], candidates: list[str]) -> None:
    factories = candidate_factories()
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(f"Unknown candidates: {unknown_candidates}")

    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            feature_cols = voter.FEATURE_CONFIGS[config]
            print(f"-- {config}", flush=True)
            for candidate in candidates:
                metrics = evaluate_candidate(cohort_df, feature_cols, factories[candidate])
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "candidate": candidate,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **metrics,
                    }
                )
                print(
                    f"{candidate:16s} "
                    f"BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} "
                    f"F1={metrics['macro_f1']:.4f}",
                    flush=True,
                )

    _write_summary(rows, candidates)
    print(f"\nWrote {OUT_DIR}", flush=True)


def run_nested(cohorts: list[str], configs: list[str], candidates: list[str]) -> None:
    factories = candidate_factories()
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(f"Unknown candidates: {unknown_candidates}")

    df = voter.load_analysis_data()
    result_rows = []
    choice_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== nested {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            metrics, choices = evaluate_nested_selection(
                cohort_df,
                cohort,
                config,
                voter.FEATURE_CONFIGS[config],
                factories,
                candidates,
            )
            result_rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "selection": "nested_candidate_by_balanced_accuracy",
                    "n": len(cohort_df),
                    "n_features_raw": len(voter.FEATURE_CONFIGS[config]),
                    **metrics,
                }
            )
            choice_rows.extend(choices)
            print(
                f"nested_result BA={metrics['balanced_accuracy']:.4f} "
                f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f}",
                flush=True,
            )

    _write_nested_summary(result_rows, choice_rows, candidates)
    print(f"\nWrote nested selection artifacts in {OUT_DIR}", flush=True)


def run_calibration(cohorts: list[str], configs: list[str]) -> None:
    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== calibration {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            for method in CALIBRATION_METHODS:
                metrics = evaluate_calibrated_current(cohort_df, feature_cols, method)
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "calibration": method,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **metrics,
                    }
                )
                print(
                    f"{method:8s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

    _write_calibration_summary(rows)
    print(f"\nWrote calibration artifacts in {OUT_DIR}", flush=True)


def run_threshold(cohorts: list[str], configs: list[str]) -> None:
    df = voter.load_analysis_data()
    result_rows = []
    choice_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== threshold {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            metrics, choices = evaluate_threshold_tuned_current(
                cohort_df,
                cohort,
                config,
                voter.FEATURE_CONFIGS[config],
            )
            result_rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "thresholding": "inner_cv_balanced_accuracy",
                    "n": len(cohort_df),
                    "n_features_raw": len(voter.FEATURE_CONFIGS[config]),
                    **metrics,
                }
            )
            choice_rows.extend(choices)
            print(
                f"threshold_result BA={metrics['balanced_accuracy']:.4f} "
                f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f}",
                flush=True,
            )

    _write_threshold_summary(result_rows, choice_rows)
    print(f"\nWrote threshold artifacts in {OUT_DIR}", flush=True)


def run_missingness(cohorts: list[str], configs: list[str]) -> None:
    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== missingness {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            for variant in MISSINGNESS_VARIANTS:
                metrics = evaluate_missingness_current(
                    cohort_df,
                    feature_cols,
                    add_missingness_indicators=variant == "with_missingness_indicators",
                )
                n_features_raw = len(feature_cols) * (2 if variant == "with_missingness_indicators" else 1)
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "missingness": variant,
                        "n": len(cohort_df),
                        "n_features_raw": n_features_raw,
                        **metrics,
                    }
                )
                print(
                    f"{variant:28s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

    _write_missingness_summary(rows)
    print(f"\nWrote missingness artifacts in {OUT_DIR}", flush=True)


def run_xgb_nan(cohorts: list[str], configs: list[str]) -> None:
    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== xgb-nan {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            for variant in XGB_NAN_VARIANTS:
                metrics = evaluate_xgb_nan_current(
                    cohort_df,
                    feature_cols,
                    native_nan_no_scale=variant == "native_nan_no_scale",
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "xgb_nan_handling": variant,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **metrics,
                    }
                )
                print(
                    f"{variant:20s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

    _write_xgb_nan_summary(rows)
    print(f"\nWrote native-NaN artifacts in {OUT_DIR}", flush=True)


def run_class_balance(cohorts: list[str], configs: list[str]) -> None:
    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== class-balance {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            for variant in CLASS_BALANCE_VARIANTS:
                metrics = evaluate_class_balanced_current(
                    cohort_df,
                    feature_cols,
                    class_balance_training=variant == "survey_x_fold_class_balance",
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "class_balance": variant,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **metrics,
                    }
                )
                print(
                    f"{variant:28s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

    _write_class_balance_summary(rows)
    print(f"\nWrote class-balance artifacts in {OUT_DIR}", flush=True)


def run_training_weights(
    cohorts: list[str],
    configs: list[str],
    n_bootstrap: int,
) -> None:
    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== training weights {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            results = {}
            for variant in TRAINING_WEIGHT_VARIANTS:
                result = evaluate_training_weight_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    variant=variant,
                )
                results[variant] = result
                metrics = result.metrics
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "training_weight": variant,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **metrics,
                    }
                )
                print(
                    f"{variant:26s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

            baseline = results["survey_weight"]
            for variant, result in results.items():
                if variant == "survey_weight":
                    continue
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "training_weight": variant,
                            "comparison_training_weight": "survey_weight",
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_baseline": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_training_weight_summary(rows, boot_rows, n_bootstrap)
    print(f"\nWrote training-weight artifacts in {OUT_DIR}", flush=True)


def run_xgb_profiles(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(f"Unknown XGBoost profile candidates: {unknown_candidates}")
    if BASELINE_CANDIDATE not in candidates:
        raise RuntimeError("xgb_current must be included in XGBoost profile candidates")
    non_xgb = [candidate for candidate in candidates if not candidate.startswith("xgb_")]
    if non_xgb:
        raise RuntimeError(f"XGBoost profile sweep received non-XGB candidates: {non_xgb}")

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== XGBoost profile sweep {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline_result: voter.CvResult | None = None
            candidate_results: dict[str, voter.CvResult] = {}
            for candidate in candidates:
                result = evaluate_candidate_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    candidate=candidate,
                    feature_cols=feature_cols,
                    make_model=factories[candidate],
                )
                candidate_results[candidate] = result
                if candidate == BASELINE_CANDIDATE:
                    baseline_result = result
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "candidate": candidate,
                        "stage2_model": candidate,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **result.metrics,
                    }
                )
                print(
                    f"{candidate:22s} BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
            if baseline_result is None:
                raise RuntimeError("Missing xgb_current baseline result")
            for candidate, result in candidate_results.items():
                if candidate == BASELINE_CANDIDATE:
                    continue
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline_result,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "comparison_candidate": BASELINE_CANDIDATE,
                            "metric": metric,
                            "candidate_metric": result.metrics[metric],
                            "baseline_metric": baseline_result.metrics[metric],
                            "diff_candidate_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_xgb_profile_summary(rows, boot_rows, candidates, n_bootstrap)
    print(f"\nWrote XGBoost profile artifacts in {OUT_DIR}", flush=True)


def run_issue_encoding(
    cohorts: list[str],
    configs: list[str],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(ISSUE_ENCODING_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown issue-encoding configs: {unknown_configs}")

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== issue encoding {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            canonical_config = ISSUE_ENCODING_CONFIG_MAP[config]
            feature_cols = voter.FEATURE_CONFIGS[canonical_config]
            print(f"-- {config} ({canonical_config})", flush=True)
            results = {}
            for encoding in ISSUE_ENCODING_VARIANTS:
                result = evaluate_issue_encoding_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    canonical_config=canonical_config,
                    feature_cols=feature_cols,
                    encoding=encoding,
                )
                results[encoding] = result
                n_features_encoded = (
                    len(encode_feature_frame_with_issue_categories(cohort_df, feature_cols).columns)
                    if encoding == "issue_onehot"
                    else len(feature_cols)
                )
                metrics = result.metrics
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "canonical_config": canonical_config,
                        "encoding": encoding,
                        "stage2_model": BASELINE_CANDIDATE,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "n_features_encoded": n_features_encoded,
                        **metrics,
                    }
                )
                print(
                    f"{encoding:16s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

            baseline = results["numeric_ordinal"]
            variant = results["issue_onehot"]
            for boot_row in voter.bootstrap_differences(
                variant,
                baseline,
                n_bootstrap,
            ):
                metric = boot_row["metric"]
                boot_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "canonical_config": canonical_config,
                        "encoding": "issue_onehot",
                        "comparison_encoding": "numeric_ordinal",
                        "metric": metric,
                        "variant_metric": variant.metrics[metric],
                        "baseline_metric": baseline.metrics[metric],
                        "diff_variant_minus_numeric": boot_row[
                            "diff_reference_minus_comparison"
                        ],
                        "ci_low": boot_row["ci_low"],
                        "ci_high": boot_row["ci_high"],
                        "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                        "direction": boot_row["direction"],
                        "n_bootstrap": boot_row["n_bootstrap"],
                    }
                )

    _write_issue_encoding_summary(rows, boot_rows, n_bootstrap)
    print(f"\nWrote issue-encoding artifacts in {OUT_DIR}", flush=True)


def run_political_encoding(
    cohorts: list[str],
    configs: list[str],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(POLITICAL_ENCODING_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown political-encoding configs: {unknown_configs}")

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== political encoding {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            canonical_config = POLITICAL_ENCODING_CONFIG_MAP[config]
            feature_cols = voter.FEATURE_CONFIGS[canonical_config]
            print(f"-- {config} ({canonical_config})", flush=True)
            results = {}
            for encoding in POLITICAL_ENCODING_VARIANTS:
                result = evaluate_political_encoding_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    canonical_config=canonical_config,
                    feature_cols=feature_cols,
                    encoding=encoding,
                )
                results[encoding] = result
                n_features_encoded = (
                    len(
                        encode_feature_frame_with_political_categories(
                            cohort_df,
                            feature_cols,
                        ).columns
                    )
                    if encoding == "political_onehot"
                    else len(feature_cols)
                )
                metrics = result.metrics
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "canonical_config": canonical_config,
                        "encoding": encoding,
                        "stage2_model": BASELINE_CANDIDATE,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "n_features_encoded": n_features_encoded,
                        **metrics,
                    }
                )
                print(
                    f"{encoding:17s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

            baseline = results["numeric_ordinal"]
            variant = results["political_onehot"]
            for boot_row in voter.bootstrap_differences(
                variant,
                baseline,
                n_bootstrap,
            ):
                metric = boot_row["metric"]
                boot_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "canonical_config": canonical_config,
                        "encoding": "political_onehot",
                        "comparison_encoding": "numeric_ordinal",
                        "metric": metric,
                        "variant_metric": variant.metrics[metric],
                        "baseline_metric": baseline.metrics[metric],
                        "diff_variant_minus_numeric": boot_row[
                            "diff_reference_minus_comparison"
                        ],
                        "ci_low": boot_row["ci_low"],
                        "ci_high": boot_row["ci_high"],
                        "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                        "direction": boot_row["direction"],
                        "n_bootstrap": boot_row["n_bootstrap"],
                    }
                )

    _write_political_encoding_summary(rows, boot_rows, n_bootstrap)
    print(f"\nWrote political-encoding artifacts in {OUT_DIR}", flush=True)


def run_xgb_early_stop(
    cohorts: list[str],
    configs: list[str],
    n_bootstrap: int,
) -> None:
    df = voter.load_analysis_data()
    rows = []
    fold_rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== XGBoost early stopping {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            results = {}
            for variant in XGB_EARLY_STOP_VARIANTS:
                result, folds, mean_best, median_best = evaluate_xgb_early_stop_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    variant=variant,
                )
                results[variant] = result
                fold_rows.extend(folds)
                metrics = result.metrics
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "early_stop_variant": variant,
                        "stage2_model": BASELINE_CANDIDATE,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "mean_best_iteration": mean_best,
                        "median_best_iteration": median_best,
                        **metrics,
                    }
                )
                print(
                    f"{variant:22s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"best_iter={mean_best:.1f}",
                    flush=True,
                )

            baseline = results["fixed_300"]
            variant = results["early_stop_1000_15pct"]
            for boot_row in voter.bootstrap_differences(
                variant,
                baseline,
                n_bootstrap,
            ):
                metric = boot_row["metric"]
                boot_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "early_stop_variant": "early_stop_1000_15pct",
                        "comparison_variant": "fixed_300",
                        "metric": metric,
                        "variant_metric": variant.metrics[metric],
                        "baseline_metric": baseline.metrics[metric],
                        "diff_variant_minus_fixed": boot_row[
                            "diff_reference_minus_comparison"
                        ],
                        "ci_low": boot_row["ci_low"],
                        "ci_high": boot_row["ci_high"],
                        "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                        "direction": boot_row["direction"],
                        "n_bootstrap": boot_row["n_bootstrap"],
                    }
                )

    _write_xgb_early_stop_summary(rows, fold_rows, boot_rows, n_bootstrap)
    print(f"\nWrote XGBoost early-stop artifacts in {OUT_DIR}", flush=True)


def run_xgb_optuna(
    cohorts: list[str],
    configs: list[str],
    n_trials: int,
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown XGBoost Optuna configs: {unknown_configs}")
    if n_trials < 2:
        raise RuntimeError("XGBoost Optuna tuning requires at least 2 trials")

    df = voter.load_analysis_data()
    rows = []
    choice_rows = []
    trial_rows = []
    boot_rows = []
    factory = candidate_factories()[BASELINE_CANDIDATE]
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== XGBoost Optuna {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            feature_cols = voter.FEATURE_CONFIGS[config]
            print(f"-- {config}", flush=True)
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factory,
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "variant": "fixed_300",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "n_trials": 0,
                    **baseline.metrics,
                }
            )
            print(
                f"fixed_300                 BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )

            tuned, choices, trials = evaluate_xgb_optuna_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                feature_cols=feature_cols,
                n_trials=n_trials,
            )
            choice_rows.extend(choices)
            trial_rows.extend(trials)
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "variant": "optuna_xgb_balanced_accuracy",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "n_trials": n_trials,
                    **tuned.metrics,
                }
            )
            print(
                f"optuna_xgb_balanced_accuracy BA={tuned.metrics['balanced_accuracy']:.4f} "
                f"LL={tuned.metrics['log_loss']:.4f}",
                flush=True,
            )
            for boot_row in voter.bootstrap_differences(
                tuned,
                baseline,
                n_bootstrap,
            ):
                metric = boot_row["metric"]
                boot_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "variant": "optuna_xgb_balanced_accuracy",
                        "comparison_variant": "fixed_300",
                        "metric": metric,
                        "variant_metric": tuned.metrics[metric],
                        "baseline_metric": baseline.metrics[metric],
                        "diff_variant_minus_fixed": boot_row[
                            "diff_reference_minus_comparison"
                        ],
                        "ci_low": boot_row["ci_low"],
                        "ci_high": boot_row["ci_high"],
                        "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                        "direction": boot_row["direction"],
                        "n_bootstrap": boot_row["n_bootstrap"],
                    }
                )

    _write_xgb_optuna_summary(
        rows,
        choice_rows,
        trial_rows,
        boot_rows,
        configs,
        n_trials,
        n_bootstrap,
    )
    print(f"\nWrote XGBoost Optuna artifacts in {OUT_DIR}", flush=True)


def run_xgb_seed_bagging(
    cohorts: list[str],
    configs: list[str],
    seeds: list[int],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown XGBoost seed-bagging configs: {unknown_configs}")
    if len(seeds) < 2:
        raise RuntimeError("XGBoost seed bagging requires at least two seeds")

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== XGBoost seed bagging {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            results = {}
            for variant in XGB_SEED_BAG_VARIANTS:
                result = evaluate_xgb_seed_bag_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    variant=variant,
                    seeds=seeds,
                )
                results[variant] = result
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "variant": variant,
                        "stage2_model": BASELINE_CANDIDATE,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "n_seeds": len(seeds) if variant == "xgb_seed_bag_5" else 1,
                        **result.metrics,
                    }
                )
                print(
                    f"{variant:16s} BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
            baseline = results["xgb_current"]
            bagged = results["xgb_seed_bag_5"]
            for boot_row in voter.bootstrap_differences(
                bagged,
                baseline,
                n_bootstrap,
            ):
                metric = boot_row["metric"]
                boot_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "variant": "xgb_seed_bag_5",
                        "comparison_variant": "xgb_current",
                        "metric": metric,
                        "variant_metric": bagged.metrics[metric],
                        "baseline_metric": baseline.metrics[metric],
                        "diff_variant_minus_current": boot_row[
                            "diff_reference_minus_comparison"
                        ],
                        "ci_low": boot_row["ci_low"],
                        "ci_high": boot_row["ci_high"],
                        "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                        "direction": boot_row["direction"],
                        "n_bootstrap": boot_row["n_bootstrap"],
                    }
                )

    _write_xgb_seed_bag_summary(rows, boot_rows, configs, seeds, n_bootstrap)
    print(f"\nWrote XGBoost seed-bagging artifacts in {OUT_DIR}", flush=True)


def run_xgb_profile_threshold(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown XGBoost profile-threshold configs: {unknown_configs}")
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown XGBoost profile-threshold candidates: {unknown_candidates}"
        )
    non_xgb = [candidate for candidate in candidates if not candidate.startswith("xgb_")]
    if non_xgb:
        raise RuntimeError(
            f"XGBoost profile-threshold received non-XGB candidates: {non_xgb}"
        )

    df = voter.load_analysis_data()
    rows = []
    choice_rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== XGBoost profile thresholds {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "candidate": BASELINE_CANDIDATE,
                    "variant": "xgb_current_threshold_0.5",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current_threshold_0.5 BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )

            for candidate in candidates:
                result, choices = evaluate_threshold_tuned_candidate_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    candidate=candidate,
                    make_model=factories[candidate],
                )
                choice_rows.extend(choices)
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "candidate": candidate,
                        "variant": "inner_cv_threshold",
                        "stage2_model": candidate,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **result.metrics,
                    }
                )
                print(
                    f"{candidate:22s} threshold BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "variant": "inner_cv_threshold",
                            "comparison_variant": "xgb_current_threshold_0.5",
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_xgb_profile_threshold_summary(
        rows,
        choice_rows,
        boot_rows,
        configs,
        candidates,
        n_bootstrap,
    )
    print(f"\nWrote XGBoost profile-threshold artifacts in {OUT_DIR}", flush=True)


def run_xgb_profile_calibration(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown XGBoost profile-calibration configs: {unknown_configs}"
        )
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown XGBoost profile-calibration candidates: {unknown_candidates}"
        )
    non_xgb = [candidate for candidate in candidates if not candidate.startswith("xgb_")]
    if non_xgb:
        raise RuntimeError(
            f"XGBoost profile-calibration received non-XGB candidates: {non_xgb}"
        )
    if BASELINE_CANDIDATE in candidates:
        raise RuntimeError("Use xgb_current only as the uncalibrated baseline")
    unknown_methods = sorted(set(methods) - set(XGB_PROFILE_CALIBRATION_METHODS))
    if unknown_methods:
        raise RuntimeError(
            f"Unknown XGBoost profile-calibration methods: {unknown_methods}"
        )

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== XGBoost profile calibration {cohort} n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "candidate": BASELINE_CANDIDATE,
                    "calibration": "none",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current none BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for candidate in candidates:
                for method in methods:
                    result = evaluate_calibrated_candidate_result(
                        df=cohort_df,
                        cohort=cohort,
                        config=config,
                        feature_cols=feature_cols,
                        candidate=candidate,
                        make_model=factories[candidate],
                        method=method,
                    )
                    rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "calibration": method,
                            "stage2_model": candidate,
                            "n": len(cohort_df),
                            "n_features_raw": len(feature_cols),
                            **result.metrics,
                        }
                    )
                    print(
                        f"{candidate:22s} {method:8s} "
                        f"BA={result.metrics['balanced_accuracy']:.4f} "
                        f"LL={result.metrics['log_loss']:.4f}",
                        flush=True,
                    )
                    for boot_row in voter.bootstrap_differences(
                        result,
                        baseline,
                        n_bootstrap,
                    ):
                        metric = boot_row["metric"]
                        boot_rows.append(
                            {
                                "cohort": cohort,
                                "config": config,
                                "candidate": candidate,
                                "calibration": method,
                                "comparison_candidate": BASELINE_CANDIDATE,
                                "comparison_calibration": "none",
                                "metric": metric,
                                "candidate_metric": result.metrics[metric],
                                "baseline_metric": baseline.metrics[metric],
                                "diff_candidate_minus_current": boot_row[
                                    "diff_reference_minus_comparison"
                                ],
                                "ci_low": boot_row["ci_low"],
                                "ci_high": boot_row["ci_high"],
                                "bootstrap_p_two_sided": boot_row[
                                    "bootstrap_p_two_sided"
                                ],
                                "direction": boot_row["direction"],
                                "n_bootstrap": boot_row["n_bootstrap"],
                            }
                        )

    _write_xgb_profile_calibration_summary(
        rows,
        boot_rows,
        configs,
        candidates,
        methods,
        n_bootstrap,
    )
    print(f"\nWrote XGBoost profile-calibration artifacts in {OUT_DIR}", flush=True)


def run_xgb_profile_calibration_threshold(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown XGBoost profile calibration-threshold configs: {unknown_configs}"
        )
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            "Unknown XGBoost profile calibration-threshold candidates: "
            f"{unknown_candidates}"
        )
    non_xgb = [candidate for candidate in candidates if not candidate.startswith("xgb_")]
    if non_xgb:
        raise RuntimeError(
            "XGBoost profile calibration-threshold received non-XGB candidates: "
            f"{non_xgb}"
        )
    if BASELINE_CANDIDATE in candidates:
        raise RuntimeError("Use xgb_current only as the uncalibrated baseline")
    unknown_methods = sorted(
        set(methods) - set(XGB_PROFILE_CALIBRATION_THRESHOLD_METHODS)
    )
    if unknown_methods:
        raise RuntimeError(
            "Unknown XGBoost profile calibration-threshold methods: "
            f"{unknown_methods}"
        )

    df = voter.load_analysis_data()
    rows = []
    choice_rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== XGBoost profile calibration thresholds {cohort} "
            f"n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "candidate": BASELINE_CANDIDATE,
                    "calibration": "none",
                    "variant": "xgb_current_none_threshold_0.5",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current none threshold_0.5 "
                f"BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for candidate in candidates:
                for method in methods:
                    result, choices = evaluate_calibrated_threshold_candidate_result(
                        df=cohort_df,
                        cohort=cohort,
                        config=config,
                        feature_cols=feature_cols,
                        candidate=candidate,
                        make_model=factories[candidate],
                        method=method,
                    )
                    choice_rows.extend(choices)
                    rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "calibration": method,
                            "variant": "inner_cv_calibration_threshold",
                            "stage2_model": candidate,
                            "n": len(cohort_df),
                            "n_features_raw": len(feature_cols),
                            **result.metrics,
                        }
                    )
                    print(
                        f"{candidate:22s} {method:8s} threshold "
                        f"BA={result.metrics['balanced_accuracy']:.4f} "
                        f"LL={result.metrics['log_loss']:.4f}",
                        flush=True,
                    )
                    for boot_row in voter.bootstrap_differences(
                        result,
                        baseline,
                        n_bootstrap,
                    ):
                        metric = boot_row["metric"]
                        boot_rows.append(
                            {
                                "cohort": cohort,
                                "config": config,
                                "candidate": candidate,
                                "calibration": method,
                                "variant": "inner_cv_calibration_threshold",
                                "comparison_candidate": BASELINE_CANDIDATE,
                                "comparison_calibration": "none",
                                "comparison_variant": "xgb_current_none_threshold_0.5",
                                "metric": metric,
                                "variant_metric": result.metrics[metric],
                                "baseline_metric": baseline.metrics[metric],
                                "diff_variant_minus_current": boot_row[
                                    "diff_reference_minus_comparison"
                                ],
                                "ci_low": boot_row["ci_low"],
                                "ci_high": boot_row["ci_high"],
                                "bootstrap_p_two_sided": boot_row[
                                    "bootstrap_p_two_sided"
                                ],
                                "direction": boot_row["direction"],
                                "n_bootstrap": boot_row["n_bootstrap"],
                            }
                        )

    _write_xgb_profile_calibration_threshold_summary(
        rows,
        choice_rows,
        boot_rows,
        configs,
        candidates,
        methods,
        n_bootstrap,
    )
    print(
        f"\nWrote XGBoost profile calibration-threshold artifacts in {OUT_DIR}",
        flush=True,
    )


def run_xgb_profile_blend_grid(
    cohorts: list[str],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown XGBoost profile blend configs: {unknown_configs}")
    unknown_recipes = sorted(set(recipes) - set(XGB_PROFILE_BLEND_RECIPES))
    if unknown_recipes:
        raise RuntimeError(f"Unknown XGBoost profile blend recipes: {unknown_recipes}")
    recipe_candidates = {
        candidate
        for name in recipes
        for candidate, _ in XGB_PROFILE_BLEND_RECIPES[name]
    }
    unknown_candidates = sorted(recipe_candidates - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown XGBoost profile blend candidates: {unknown_candidates}"
        )
    non_xgb = [
        candidate
        for candidate in recipe_candidates
        if not candidate.startswith("xgb_")
    ]
    if non_xgb:
        raise RuntimeError(
            f"XGBoost profile blend received non-XGB candidates: {non_xgb}"
        )

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== XGBoost profile blend grid {cohort} n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "blend": "xgb_current",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "recipe": json.dumps([(BASELINE_CANDIDATE, 1.0)]),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for recipe_name in recipes:
                result = evaluate_ensemble_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    ensemble_name=recipe_name,
                    recipe=XGB_PROFILE_BLEND_RECIPES[recipe_name],
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "blend": recipe_name,
                        "stage2_model": "xgb_profile_blend_grid",
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(XGB_PROFILE_BLEND_RECIPES[recipe_name]),
                        **result.metrics,
                    }
                )
                print(
                    f"{recipe_name:30s} BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "blend": recipe_name,
                            "comparison_blend": "xgb_current",
                            "metric": metric,
                            "blend_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_blend_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_xgb_profile_blend_grid_summary(
        rows,
        boot_rows,
        configs,
        recipes,
        n_bootstrap,
    )
    print(f"\nWrote XGBoost profile blend-grid artifacts in {OUT_DIR}", flush=True)


def run_xgb_profile_blend_threshold(
    cohorts: list[str],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown XGBoost profile blend-threshold configs: {unknown_configs}"
        )
    unknown_recipes = sorted(set(recipes) - set(XGB_PROFILE_BLEND_RECIPES))
    if unknown_recipes:
        raise RuntimeError(
            f"Unknown XGBoost profile blend-threshold recipes: {unknown_recipes}"
        )
    recipe_candidates = {
        candidate
        for name in recipes
        for candidate, _ in XGB_PROFILE_BLEND_RECIPES[name]
    }
    unknown_candidates = sorted(recipe_candidates - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown XGBoost profile blend-threshold candidates: {unknown_candidates}"
        )
    non_xgb = [
        candidate
        for candidate in recipe_candidates
        if not candidate.startswith("xgb_")
    ]
    if non_xgb:
        raise RuntimeError(
            f"XGBoost profile blend-threshold received non-XGB candidates: {non_xgb}"
        )

    df = voter.load_analysis_data()
    rows = []
    choice_rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== XGBoost profile blend thresholds {cohort} "
            f"n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "blend": "xgb_current",
                    "variant": "xgb_current_threshold_0.5",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "recipe": json.dumps([(BASELINE_CANDIDATE, 1.0)]),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current threshold_0.5 "
                f"BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for recipe_name in recipes:
                result, choices = evaluate_threshold_tuned_ensemble_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    ensemble_name=recipe_name,
                    recipe=XGB_PROFILE_BLEND_RECIPES[recipe_name],
                )
                for choice in choices:
                    choice_row = dict(choice)
                    choice_row["blend"] = choice_row.pop("ensemble")
                    choice_row["variant"] = "inner_cv_threshold"
                    choice_rows.append(choice_row)
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "blend": recipe_name,
                        "variant": "inner_cv_threshold",
                        "stage2_model": "xgb_profile_blend_threshold",
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(XGB_PROFILE_BLEND_RECIPES[recipe_name]),
                        **result.metrics,
                    }
                )
                print(
                    f"{recipe_name:30s} threshold "
                    f"BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "blend": recipe_name,
                            "variant": "inner_cv_threshold",
                            "comparison_blend": "xgb_current",
                            "comparison_variant": "xgb_current_threshold_0.5",
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_xgb_profile_blend_threshold_summary(
        rows,
        choice_rows,
        boot_rows,
        configs,
        recipes,
        n_bootstrap,
    )
    print(
        f"\nWrote XGBoost profile blend-threshold artifacts in {OUT_DIR}",
        flush=True,
    )


def run_extended_candidates(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown extended-candidate configs: {unknown_configs}")
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(f"Unknown extended candidates: {unknown_candidates}")
    if BASELINE_CANDIDATE in candidates:
        raise RuntimeError("Do not include xgb_current in extended candidates")

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== extended candidates {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "candidate": BASELINE_CANDIDATE,
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    **baseline.metrics,
                }
            )
            print(
                f"{BASELINE_CANDIDATE:16s} BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for candidate in candidates:
                result = evaluate_candidate_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    candidate=candidate,
                    feature_cols=feature_cols,
                    make_model=factories[candidate],
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "candidate": candidate,
                        "stage2_model": candidate,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **result.metrics,
                    }
                )
                print(
                    f"{candidate:16s} BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "comparison_candidate": BASELINE_CANDIDATE,
                            "metric": metric,
                            "candidate_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_candidate_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_extended_candidate_summary(
        rows,
        boot_rows,
        candidates,
        configs,
        n_bootstrap,
    )
    print(f"\nWrote extended-candidate artifacts in {OUT_DIR}", flush=True)


def run_extended_candidate_threshold(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown extended-candidate threshold configs: {unknown_configs}"
        )
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown extended-candidate threshold candidates: {unknown_candidates}"
        )
    if BASELINE_CANDIDATE in candidates:
        raise RuntimeError("Do not include xgb_current in extended threshold candidates")

    df = voter.load_analysis_data()
    rows = []
    choice_rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== extended candidate thresholds {cohort} n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "candidate": BASELINE_CANDIDATE,
                    "variant": "xgb_current_threshold_0.5",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current_threshold_0.5 BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for candidate in candidates:
                result, choices = evaluate_threshold_tuned_candidate_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    candidate=candidate,
                    make_model=factories[candidate],
                )
                choice_rows.extend(choices)
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "candidate": candidate,
                        "variant": "inner_cv_threshold",
                        "stage2_model": candidate,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        **result.metrics,
                    }
                )
                print(
                    f"{candidate:22s} threshold BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "variant": "inner_cv_threshold",
                            "comparison_variant": "xgb_current_threshold_0.5",
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_extended_candidate_threshold_summary(
        rows,
        choice_rows,
        boot_rows,
        configs,
        candidates,
        n_bootstrap,
    )
    print(
        f"\nWrote extended-candidate threshold artifacts in {OUT_DIR}",
        flush=True,
    )


def run_extended_candidate_calibration(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown extended-candidate calibration configs: {unknown_configs}"
        )
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown extended-candidate calibration candidates: {unknown_candidates}"
        )
    unknown_methods = sorted(set(methods) - set(EXTENDED_CANDIDATE_CALIBRATION_METHODS))
    if unknown_methods:
        raise RuntimeError(
            f"Unknown extended-candidate calibration methods: {unknown_methods}"
        )
    if BASELINE_CANDIDATE in candidates:
        raise RuntimeError("Do not include xgb_current in extended calibration candidates")

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== extended candidate calibration {cohort} n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "candidate": BASELINE_CANDIDATE,
                    "calibration": "none",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current none BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for candidate in candidates:
                for method in methods:
                    result = evaluate_calibrated_candidate_result(
                        df=cohort_df,
                        cohort=cohort,
                        config=config,
                        feature_cols=feature_cols,
                        candidate=candidate,
                        make_model=factories[candidate],
                        method=method,
                    )
                    rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "calibration": method,
                            "stage2_model": candidate,
                            "n": len(cohort_df),
                            "n_features_raw": len(feature_cols),
                            **result.metrics,
                        }
                    )
                    print(
                        f"{candidate:16s} {method:8s} "
                        f"BA={result.metrics['balanced_accuracy']:.4f} "
                        f"LL={result.metrics['log_loss']:.4f}",
                        flush=True,
                    )
                    for boot_row in voter.bootstrap_differences(
                        result,
                        baseline,
                        n_bootstrap,
                    ):
                        metric = boot_row["metric"]
                        boot_rows.append(
                            {
                                "cohort": cohort,
                                "config": config,
                                "candidate": candidate,
                                "calibration": method,
                                "comparison_candidate": BASELINE_CANDIDATE,
                                "comparison_calibration": "none",
                                "metric": metric,
                                "candidate_metric": result.metrics[metric],
                                "baseline_metric": baseline.metrics[metric],
                                "diff_candidate_minus_current": boot_row[
                                    "diff_reference_minus_comparison"
                                ],
                                "ci_low": boot_row["ci_low"],
                                "ci_high": boot_row["ci_high"],
                                "bootstrap_p_two_sided": boot_row[
                                    "bootstrap_p_two_sided"
                                ],
                                "direction": boot_row["direction"],
                                "n_bootstrap": boot_row["n_bootstrap"],
                            }
                        )

    _write_extended_candidate_calibration_summary(
        rows,
        boot_rows,
        configs,
        candidates,
        methods,
        n_bootstrap,
    )
    print(
        f"\nWrote extended-candidate calibration artifacts in {OUT_DIR}",
        flush=True,
    )


def run_extended_candidate_calibration_threshold(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    methods: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown extended-candidate calibration-threshold configs: {unknown_configs}"
        )
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            "Unknown extended-candidate calibration-threshold candidates: "
            f"{unknown_candidates}"
        )
    unknown_methods = sorted(
        set(methods) - set(EXTENDED_CANDIDATE_CALIBRATION_THRESHOLD_METHODS)
    )
    if unknown_methods:
        raise RuntimeError(
            f"Unknown extended-candidate calibration-threshold methods: {unknown_methods}"
        )
    if BASELINE_CANDIDATE in candidates:
        raise RuntimeError(
            "Do not include xgb_current in extended calibration-threshold candidates"
        )

    df = voter.load_analysis_data()
    rows = []
    choice_rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== extended candidate calibration thresholds {cohort} "
            f"n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "candidate": BASELINE_CANDIDATE,
                    "calibration": "none",
                    "variant": "xgb_current_none_threshold_0.5",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current none threshold_0.5 "
                f"BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for candidate in candidates:
                for method in methods:
                    result, choices = evaluate_calibrated_threshold_candidate_result(
                        df=cohort_df,
                        cohort=cohort,
                        config=config,
                        feature_cols=feature_cols,
                        candidate=candidate,
                        make_model=factories[candidate],
                        method=method,
                    )
                    choice_rows.extend(choices)
                    rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "calibration": method,
                            "variant": "inner_cv_calibration_threshold",
                            "stage2_model": candidate,
                            "n": len(cohort_df),
                            "n_features_raw": len(feature_cols),
                            **result.metrics,
                        }
                    )
                    print(
                        f"{candidate:16s} {method:8s} threshold "
                        f"BA={result.metrics['balanced_accuracy']:.4f} "
                        f"LL={result.metrics['log_loss']:.4f}",
                        flush=True,
                    )
                    for boot_row in voter.bootstrap_differences(
                        result,
                        baseline,
                        n_bootstrap,
                    ):
                        metric = boot_row["metric"]
                        boot_rows.append(
                            {
                                "cohort": cohort,
                                "config": config,
                                "candidate": candidate,
                                "calibration": method,
                                "variant": "inner_cv_calibration_threshold",
                                "comparison_candidate": BASELINE_CANDIDATE,
                                "comparison_calibration": "none",
                                "comparison_variant": "xgb_current_none_threshold_0.5",
                                "metric": metric,
                                "variant_metric": result.metrics[metric],
                                "baseline_metric": baseline.metrics[metric],
                                "diff_variant_minus_current": boot_row[
                                    "diff_reference_minus_comparison"
                                ],
                                "ci_low": boot_row["ci_low"],
                                "ci_high": boot_row["ci_high"],
                                "bootstrap_p_two_sided": boot_row[
                                    "bootstrap_p_two_sided"
                                ],
                                "direction": boot_row["direction"],
                                "n_bootstrap": boot_row["n_bootstrap"],
                            }
                        )

    _write_extended_candidate_calibration_threshold_summary(
        rows,
        choice_rows,
        boot_rows,
        configs,
        candidates,
        methods,
        n_bootstrap,
    )
    print(
        f"\nWrote extended-candidate calibration-threshold artifacts in {OUT_DIR}",
        flush=True,
    )


def run_calibrated_extended_blends(
    cohorts: list[str],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown calibrated blend configs: {unknown_configs}")
    unknown_recipes = sorted(set(recipes) - set(CALIBRATED_EXTENDED_BLEND_RECIPES))
    if unknown_recipes:
        raise RuntimeError(f"Unknown calibrated blend recipes: {unknown_recipes}")
    factories = candidate_factories()
    recipe_candidates = {
        candidate
        for name in recipes
        for candidate, _, _ in CALIBRATED_EXTENDED_BLEND_RECIPES[name]
    }
    unknown_candidates = sorted(recipe_candidates - set(factories))
    if unknown_candidates:
        raise RuntimeError(f"Unknown calibrated blend candidates: {unknown_candidates}")

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== calibrated extended blends {cohort} n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "blend": "xgb_current",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "recipe": json.dumps([(BASELINE_CANDIDATE, "none", 1.0)]),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for recipe_name in recipes:
                result = evaluate_calibrated_extended_blend_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    blend_name=recipe_name,
                    recipe=CALIBRATED_EXTENDED_BLEND_RECIPES[recipe_name],
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "blend": recipe_name,
                        "stage2_model": "calibrated_extended_blend",
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(
                            CALIBRATED_EXTENDED_BLEND_RECIPES[recipe_name]
                        ),
                        **result.metrics,
                    }
                )
                print(
                    f"{recipe_name:36s} BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "blend": recipe_name,
                            "comparison_blend": "xgb_current",
                            "metric": metric,
                            "blend_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_blend_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_calibrated_extended_blend_summary(rows, boot_rows, configs, recipes, n_bootstrap)
    print(f"\nWrote calibrated extended blend artifacts in {OUT_DIR}", flush=True)


def run_calibrated_extended_blend_threshold(
    cohorts: list[str],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown calibrated blend threshold configs: {unknown_configs}"
        )
    unknown_recipes = sorted(set(recipes) - set(CALIBRATED_EXTENDED_BLEND_RECIPES))
    if unknown_recipes:
        raise RuntimeError(
            f"Unknown calibrated blend threshold recipes: {unknown_recipes}"
        )

    df = voter.load_analysis_data()
    rows = []
    choice_rows = []
    boot_rows = []
    factories = candidate_factories()
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== calibrated extended blend thresholds {cohort} "
            f"n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "blend": "xgb_current",
                    "variant": "xgb_current_threshold_0.5",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "recipe": json.dumps([(BASELINE_CANDIDATE, "none", 1.0)]),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current threshold_0.5 "
                f"BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for recipe_name in recipes:
                result, choices = evaluate_threshold_tuned_calibrated_extended_blend_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    blend_name=recipe_name,
                    recipe=CALIBRATED_EXTENDED_BLEND_RECIPES[recipe_name],
                )
                choice_rows.extend(choices)
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "blend": recipe_name,
                        "variant": "inner_cv_threshold",
                        "stage2_model": "calibrated_extended_blend",
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(
                            CALIBRATED_EXTENDED_BLEND_RECIPES[recipe_name]
                        ),
                        **result.metrics,
                    }
                )
                print(
                    f"{recipe_name:36s} threshold "
                    f"BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "blend": recipe_name,
                            "variant": "inner_cv_threshold",
                            "comparison_blend": "xgb_current",
                            "comparison_variant": "xgb_current_threshold_0.5",
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_calibrated_extended_blend_threshold_summary(
        rows,
        choice_rows,
        boot_rows,
        configs,
        recipes,
        n_bootstrap,
    )
    print(
        f"\nWrote calibrated extended blend threshold artifacts in {OUT_DIR}",
        flush=True,
    )


def run_calibrated_extended_blend_grid(
    cohorts: list[str],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown calibrated blend grid configs: {unknown_configs}"
        )
    unknown_recipes = sorted(set(recipes) - set(CALIBRATED_EXTENDED_BLEND_GRID_RECIPES))
    if unknown_recipes:
        raise RuntimeError(
            f"Unknown calibrated blend grid recipes: {unknown_recipes}"
        )
    factories = candidate_factories()
    recipe_candidates = {
        candidate
        for name in recipes
        for candidate, _, _ in CALIBRATED_EXTENDED_BLEND_GRID_RECIPES[name]
    }
    unknown_candidates = sorted(recipe_candidates - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown calibrated blend grid candidates: {unknown_candidates}"
        )

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== calibrated extended blend grid {cohort} n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "blend": "xgb_current",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "recipe": json.dumps([(BASELINE_CANDIDATE, "none", 1.0)]),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for recipe_name in recipes:
                result = evaluate_calibrated_extended_blend_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    blend_name=recipe_name,
                    recipe=CALIBRATED_EXTENDED_BLEND_GRID_RECIPES[recipe_name],
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "blend": recipe_name,
                        "stage2_model": "calibrated_extended_blend_grid",
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(
                            CALIBRATED_EXTENDED_BLEND_GRID_RECIPES[recipe_name]
                        ),
                        **result.metrics,
                    }
                )
                print(
                    f"{recipe_name:44s} BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "blend": recipe_name,
                            "comparison_blend": "xgb_current",
                            "metric": metric,
                            "blend_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_blend_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_calibrated_extended_blend_grid_summary(
        rows,
        boot_rows,
        configs,
        recipes,
        n_bootstrap,
    )
    print(f"\nWrote calibrated extended blend grid artifacts in {OUT_DIR}", flush=True)


def run_calibrated_extended_blend_grid_core(
    cohorts: list[str],
    configs: list[str],
    recipes: list[str],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(voter.FEATURE_CONFIGS))
    if unknown_configs:
        raise RuntimeError(
            f"Unknown calibrated blend grid-core configs: {unknown_configs}"
        )
    unknown_recipes = sorted(set(recipes) - set(CALIBRATED_EXTENDED_BLEND_GRID_RECIPES))
    if unknown_recipes:
        raise RuntimeError(
            f"Unknown calibrated blend grid-core recipes: {unknown_recipes}"
        )

    df = voter.load_analysis_data()
    rows = []
    boot_rows = []
    factories = candidate_factories()
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== calibrated extended blend grid core {cohort} "
            f"n={len(cohort_df)} ==",
            flush=True,
        )
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "blend": "xgb_current",
                    "stage2_model": BASELINE_CANDIDATE,
                    "n": len(cohort_df),
                    "n_features_raw": len(feature_cols),
                    "recipe": json.dumps([(BASELINE_CANDIDATE, "none", 1.0)]),
                    **baseline.metrics,
                }
            )
            print(
                f"xgb_current BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )
            for recipe_name in recipes:
                result = evaluate_calibrated_extended_blend_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    blend_name=recipe_name,
                    recipe=CALIBRATED_EXTENDED_BLEND_GRID_RECIPES[recipe_name],
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "blend": recipe_name,
                        "stage2_model": "calibrated_extended_blend_grid_core",
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(
                            CALIBRATED_EXTENDED_BLEND_GRID_RECIPES[recipe_name]
                        ),
                        **result.metrics,
                    }
                )
                print(
                    f"{recipe_name:44s} BA={result.metrics['balanced_accuracy']:.4f} "
                    f"LL={result.metrics['log_loss']:.4f}",
                    flush=True,
                )
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "blend": recipe_name,
                            "comparison_blend": "xgb_current",
                            "metric": metric,
                            "blend_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_blend_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_calibrated_extended_blend_grid_core_summary(
        rows,
        boot_rows,
        configs,
        recipes,
        n_bootstrap,
    )
    print(
        f"\nWrote calibrated extended blend grid-core artifacts in {OUT_DIR}",
        flush=True,
    )


def run_fixed_candidate_bootstrap(
    cohorts: list[str],
    configs: list[str],
    candidates: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_candidates = sorted(set(candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(f"Unknown candidates: {unknown_candidates}")
    if BASELINE_CANDIDATE in candidates:
        raise RuntimeError("Do not include xgb_current in fixed-candidate bootstrap candidates")

    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== fixed-candidate bootstrap {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=feature_cols,
                make_model=factories[BASELINE_CANDIDATE],
            )
            for candidate in candidates:
                result = evaluate_candidate_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    candidate=candidate,
                    feature_cols=feature_cols,
                    make_model=factories[candidate],
                )
                boot_rows = voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                )
                for boot_row in boot_rows:
                    metric = boot_row["metric"]
                    rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "candidate": candidate,
                            "comparison_candidate": BASELINE_CANDIDATE,
                            "metric": metric,
                            "candidate_metric": result.metrics[metric],
                            "comparison_metric": baseline.metrics[metric],
                            "diff_candidate_minus_comparison": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )
                print(
                    f"{candidate:18s} "
                    f"dBA={result.metrics['balanced_accuracy'] - baseline.metrics['balanced_accuracy']:+.4f} "
                    f"dLL={result.metrics['log_loss'] - baseline.metrics['log_loss']:+.4f}",
                    flush=True,
                )

    _write_fixed_candidate_bootstrap_summary(rows, candidates, n_bootstrap)
    print(f"\nWrote fixed-candidate bootstrap artifacts in {OUT_DIR}", flush=True)


def run_feature_bootstrap(
    cohorts: list[str],
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(
        {
            config
            for pair in pairs
            for config in pair
            if config not in voter.FEATURE_CONFIGS
        }
    )
    if unknown_configs:
        raise RuntimeError(f"Unknown feature configs: {unknown_configs}")

    df = voter.load_analysis_data()
    rows = []
    factory = candidate_factories()[BASELINE_CANDIDATE]
    needed_configs = sorted({config for pair in pairs for config in pair})
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== feature bootstrap {cohort} n={len(cohort_df)} ==", flush=True)
        results = {}
        for config in needed_configs:
            print(f"-- fitting {config}", flush=True)
            results[config] = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=voter.FEATURE_CONFIGS[config],
                make_model=factory,
            )
            metrics = results[config].metrics
            print(
                f"{config:32s} BA={metrics['balanced_accuracy']:.4f} "
                f"LL={metrics['log_loss']:.4f}",
                flush=True,
            )

        for reference, comparison in pairs:
            boot_rows = voter.bootstrap_differences(
                results[reference],
                results[comparison],
                n_bootstrap,
            )
            for boot_row in boot_rows:
                metric = boot_row["metric"]
                rows.append(
                    {
                        "cohort": cohort,
                        "reference_config": reference,
                        "comparison_config": comparison,
                        "metric": metric,
                        "reference_metric": results[reference].metrics[metric],
                        "comparison_metric": results[comparison].metrics[metric],
                        "diff_reference_minus_comparison": boot_row[
                            "diff_reference_minus_comparison"
                        ],
                        "ci_low": boot_row["ci_low"],
                        "ci_high": boot_row["ci_high"],
                        "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                        "direction": boot_row["direction"],
                        "n_bootstrap": boot_row["n_bootstrap"],
                    }
                )
            print(
                f"{reference:32s} vs {comparison:30s} "
                f"dBA={results[reference].metrics['balanced_accuracy'] - results[comparison].metrics['balanced_accuracy']:+.4f} "
                f"dLL={results[reference].metrics['log_loss'] - results[comparison].metrics['log_loss']:+.4f}",
                flush=True,
            )

    _write_feature_bootstrap_summary(rows, pairs, n_bootstrap)
    print(f"\nWrote feature bootstrap artifacts in {OUT_DIR}", flush=True)


def _feature_cols_without_domain(config: str, domain: str | None) -> list[str]:
    feature_cols = list(voter.FEATURE_CONFIGS[config])
    if domain is None:
        return feature_cols
    removed = set(voter.ISSUE_DOMAINS[domain])
    return [col for col in feature_cols if col not in removed]


def run_domain_pruning(
    cohorts: list[str],
    configs: list[str],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(DOMAIN_PRUNING_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown domain-pruning configs: {unknown_configs}")

    df = voter.load_analysis_data()
    result_rows = []
    boot_rows = []
    factory = candidate_factories()[BASELINE_CANDIDATE]
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== domain pruning {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            baseline = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=_feature_cols_without_domain(config, None),
                make_model=factory,
            )
            result_rows.append(
                {
                    "cohort": cohort,
                    "config": config,
                    "variant": "baseline",
                    "domain_removed": "none",
                    "n_removed_issue_features": 0,
                    "n_features_raw": len(_feature_cols_without_domain(config, None)),
                    **baseline.metrics,
                }
            )
            print(
                f"{'baseline':28s} BA={baseline.metrics['balanced_accuracy']:.4f} "
                f"LL={baseline.metrics['log_loss']:.4f}",
                flush=True,
            )

            for domain, removed_features in voter.ISSUE_DOMAINS.items():
                feature_cols = _feature_cols_without_domain(config, domain)
                result = evaluate_candidate_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=f"{config}_minus_{domain}",
                    candidate=BASELINE_CANDIDATE,
                    feature_cols=feature_cols,
                    make_model=factory,
                )
                result_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "variant": f"minus_{domain}",
                        "domain_removed": domain,
                        "n_removed_issue_features": len(removed_features),
                        "n_features_raw": len(feature_cols),
                        **result.metrics,
                    }
                )
                for boot_row in voter.bootstrap_differences(result, baseline, n_bootstrap):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "variant": f"minus_{domain}",
                            "domain_removed": domain,
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_baseline": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )
                print(
                    f"minus_{domain:22s} "
                    f"dBA={result.metrics['balanced_accuracy'] - baseline.metrics['balanced_accuracy']:+.4f} "
                    f"dLL={result.metrics['log_loss'] - baseline.metrics['log_loss']:+.4f}",
                    flush=True,
                )

    _write_domain_pruning_summary(result_rows, boot_rows, n_bootstrap)
    print(f"\nWrote domain pruning artifacts in {OUT_DIR}", flush=True)


def run_issue_pca(
    cohorts: list[str],
    configs: list[str],
    components: list[int],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(ISSUE_PCA_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown issue-PCA configs: {unknown_configs}")
    too_large = [n for n in components if n > len(voter.ISSUE_COLS)]
    if too_large:
        raise RuntimeError(f"Too many PCA components for issue columns: {too_large}")

    df = voter.load_analysis_data()
    result_rows = []
    fold_rows = []
    boot_rows = []
    factory = candidate_factories()[BASELINE_CANDIDATE]
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== issue PCA {cohort} n={len(cohort_df)} ==", flush=True)
        baselines = {}
        canonical_configs = sorted({_issue_pca_extra_cols(config)[0] for config in configs})
        for canonical_config in canonical_configs:
            print(f"-- baseline {canonical_config}", flush=True)
            baselines[canonical_config] = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=canonical_config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=voter.FEATURE_CONFIGS[canonical_config],
                make_model=factory,
            )
            metrics = baselines[canonical_config].metrics
            print(
                f"{canonical_config:34s} BA={metrics['balanced_accuracy']:.4f} "
                f"LL={metrics['log_loss']:.4f}",
                flush=True,
            )

        for config in configs:
            canonical_config, _ = _issue_pca_extra_cols(config)
            baseline = baselines[canonical_config]
            print(f"-- {config}", flush=True)
            for n_components in components:
                result, folds, variance_ratio, n_extra_features = evaluate_issue_pca(
                    cohort_df,
                    cohort,
                    config,
                    n_components,
                )
                result_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "canonical_config": canonical_config,
                        "n_components": n_components,
                        "stage2_model": BASELINE_CANDIDATE,
                        "n": len(cohort_df),
                        "n_issue_features_raw": len(voter.ISSUE_COLS),
                        "n_extra_features": n_extra_features,
                        "n_total_features": n_components + n_extra_features,
                        "mean_explained_variance_ratio": variance_ratio,
                        **result.metrics,
                    }
                )
                fold_rows.extend(folds)
                for boot_row in voter.bootstrap_differences(result, baseline, n_bootstrap):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "canonical_config": canonical_config,
                            "n_components": n_components,
                            "metric": metric,
                            "pca_metric": result.metrics[metric],
                            "canonical_metric": baseline.metrics[metric],
                            "diff_pca_minus_canonical": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )
                print(
                    f"pca_{n_components:<2d} "
                    f"var={variance_ratio:.3f} "
                    f"dBA={result.metrics['balanced_accuracy'] - baseline.metrics['balanced_accuracy']:+.4f} "
                    f"dLL={result.metrics['log_loss'] - baseline.metrics['log_loss']:+.4f}",
                    flush=True,
                )

    _write_issue_pca_summary(result_rows, fold_rows, boot_rows, n_bootstrap)
    print(f"\nWrote issue PCA artifacts in {OUT_DIR}", flush=True)


def run_issue_topk(
    cohorts: list[str],
    configs: list[str],
    top_k_values: list[int],
    n_bootstrap: int,
) -> None:
    unknown_configs = sorted(set(configs) - set(ISSUE_TOPK_CONFIGS))
    if unknown_configs:
        raise RuntimeError(f"Unknown issue-top-k configs: {unknown_configs}")
    invalid_k = [
        top_k
        for top_k in top_k_values
        if top_k < 1 or top_k > len(voter.ISSUE_COLS)
    ]
    if invalid_k:
        raise RuntimeError(f"Invalid issue top-k values: {invalid_k}")

    df = voter.load_analysis_data()
    result_rows = []
    fold_rows = []
    selected_rows = []
    boot_rows = []
    factory = candidate_factories()[BASELINE_CANDIDATE]
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== issue top-k {cohort} n={len(cohort_df)} ==", flush=True)
        baselines = {}
        canonical_configs = sorted({_issue_topk_extra_cols(config)[0] for config in configs})
        for canonical_config in canonical_configs:
            print(f"-- baseline {canonical_config}", flush=True)
            baselines[canonical_config] = evaluate_candidate_result(
                df=cohort_df,
                cohort=cohort,
                config=canonical_config,
                candidate=BASELINE_CANDIDATE,
                feature_cols=voter.FEATURE_CONFIGS[canonical_config],
                make_model=factory,
            )
            metrics = baselines[canonical_config].metrics
            print(
                f"{canonical_config:34s} BA={metrics['balanced_accuracy']:.4f} "
                f"LL={metrics['log_loss']:.4f}",
                flush=True,
            )

        for config in configs:
            canonical_config, _ = _issue_topk_extra_cols(config)
            baseline = baselines[canonical_config]
            print(f"-- {config}", flush=True)
            for top_k in top_k_values:
                result, folds, selected, n_extra_features = evaluate_issue_topk(
                    cohort_df,
                    cohort,
                    config,
                    top_k,
                )
                result_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "canonical_config": canonical_config,
                        "top_k": top_k,
                        "stage2_model": BASELINE_CANDIDATE,
                        "n": len(cohort_df),
                        "n_issue_features_raw": len(voter.ISSUE_COLS),
                        "n_selected_issue_features": top_k,
                        "n_extra_features": n_extra_features,
                        "n_total_features": top_k + n_extra_features,
                        **result.metrics,
                    }
                )
                fold_rows.extend(folds)
                selected_rows.extend(selected)
                for boot_row in voter.bootstrap_differences(result, baseline, n_bootstrap):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "canonical_config": canonical_config,
                            "top_k": top_k,
                            "metric": metric,
                            "topk_metric": result.metrics[metric],
                            "canonical_metric": baseline.metrics[metric],
                            "diff_topk_minus_canonical": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row["bootstrap_p_two_sided"],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )
                print(
                    f"top_{top_k:<2d} "
                    f"dBA={result.metrics['balanced_accuracy'] - baseline.metrics['balanced_accuracy']:+.4f} "
                    f"dLL={result.metrics['log_loss'] - baseline.metrics['log_loss']:+.4f}",
                    flush=True,
                )

    _write_issue_topk_summary(
        result_rows,
        fold_rows,
        selected_rows,
        boot_rows,
        n_bootstrap,
    )
    print(f"\nWrote issue top-k artifacts in {OUT_DIR}", flush=True)


def run_two_stage(cohorts: list[str]) -> None:
    df = voter.load_analysis_data()
    all_results = []
    all_folds = []
    all_stage1 = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== two-stage {cohort} n={len(cohort_df)} ==", flush=True)
        result_rows, fold_rows, stage1_rows, _ = evaluate_two_stage(cohort_df, cohort)
        all_results.extend(result_rows)
        all_folds.extend(fold_rows)
        all_stage1.extend(stage1_rows)

    _write_two_stage_summary(all_results, all_folds, all_stage1)
    print(f"\nWrote two-stage artifacts in {OUT_DIR}", flush=True)


def run_two_stage_stage1_models(
    cohorts: list[str],
    stage1_models: list[str],
    n_bootstrap: int,
) -> None:
    unknown_models = sorted(set(stage1_models) - set(stage1_regressor_factories()))
    if unknown_models:
        raise RuntimeError(f"Unknown two-stage Stage 1 models: {unknown_models}")
    if "histgb_current" not in stage1_models:
        raise RuntimeError("two-stage Stage 1 model sweep must include histgb_current")

    df = voter.load_analysis_data()
    all_results = []
    all_folds = []
    all_stage1 = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== two-stage Stage 1 model sweep {cohort} "
            f"n={len(cohort_df)} ==",
            flush=True,
        )
        result_objects: dict[tuple[str, str], voter.CvResult] = {}
        for stage1_model in stage1_models:
            print(f"-- Stage 1 model: {stage1_model}", flush=True)
            result_rows, fold_rows, stage1_rows, objects = evaluate_two_stage(
                cohort_df,
                cohort,
                stage1_model=stage1_model,
            )
            all_results.extend(result_rows)
            all_folds.extend(
                {**row, "stage1_model": stage1_model} for row in fold_rows
            )
            all_stage1.extend(stage1_rows)
            for config, result in objects.items():
                result_objects[(stage1_model, config)] = result

        for config in TWO_STAGE_CONFIGS:
            baseline = result_objects[("histgb_current", config)]
            for stage1_model in stage1_models:
                if stage1_model == "histgb_current":
                    continue
                result = result_objects[(stage1_model, config)]
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "stage1_model": stage1_model,
                            "comparison_stage1_model": "histgb_current",
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_two_stage_stage1_model_summary(
        all_results,
        all_folds,
        all_stage1,
        boot_rows,
        stage1_models,
        n_bootstrap,
    )
    print(f"\nWrote two-stage Stage 1 model artifacts in {OUT_DIR}", flush=True)


def run_two_stage_stage2_models(
    cohorts: list[str],
    stage2_candidates: list[str],
    n_bootstrap: int,
) -> None:
    factories = candidate_factories()
    unknown_candidates = sorted(set(stage2_candidates) - set(factories))
    if unknown_candidates:
        raise RuntimeError(
            f"Unknown two-stage Stage 2 candidates: {unknown_candidates}"
        )
    if BASELINE_CANDIDATE not in stage2_candidates:
        raise RuntimeError("two-stage Stage 2 sweep must include xgb_current")

    df = voter.load_analysis_data()
    all_results = []
    all_folds = []
    all_stage1 = []
    boot_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(
            f"\n== two-stage Stage 2 model sweep {cohort} "
            f"n={len(cohort_df)} ==",
            flush=True,
        )
        result_rows, fold_rows, stage1_rows, result_objects = (
            evaluate_two_stage_stage2_models(
                cohort_df,
                cohort,
                stage2_candidates=stage2_candidates,
            )
        )
        all_results.extend(result_rows)
        all_folds.extend(fold_rows)
        all_stage1.extend(stage1_rows)

        for config in TWO_STAGE_CONFIGS:
            baseline = result_objects[(config, BASELINE_CANDIDATE)]
            for candidate in stage2_candidates:
                if candidate == BASELINE_CANDIDATE:
                    continue
                result = result_objects[(config, candidate)]
                for boot_row in voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                ):
                    metric = boot_row["metric"]
                    boot_rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "stage1_model": "histgb_current",
                            "stage2_model": candidate,
                            "comparison_stage2_model": BASELINE_CANDIDATE,
                            "metric": metric,
                            "variant_metric": result.metrics[metric],
                            "baseline_metric": baseline.metrics[metric],
                            "diff_variant_minus_current": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )

    _write_two_stage_stage2_model_summary(
        all_results,
        all_folds,
        all_stage1,
        boot_rows,
        stage2_candidates,
        n_bootstrap,
    )
    print(f"\nWrote two-stage Stage 2 model artifacts in {OUT_DIR}", flush=True)


def run_domain_aggregates(cohorts: list[str]) -> None:
    df = voter.load_analysis_data()
    all_results = []
    all_folds = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== domain aggregates {cohort} n={len(cohort_df)} ==", flush=True)
        result_rows, fold_rows = evaluate_domain_aggregates(cohort_df, cohort)
        all_results.extend(result_rows)
        all_folds.extend(fold_rows)

    _write_domain_aggregate_summary(all_results, all_folds)
    print(f"\nWrote domain aggregate artifacts in {OUT_DIR}", flush=True)


def run_ensembles(cohorts: list[str], configs: list[str]) -> None:
    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== ensembles {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            for ensemble_name, recipe in ENSEMBLE_RECIPES.items():
                metrics = evaluate_ensemble(cohort_df, feature_cols, recipe)
                rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "ensemble": ensemble_name,
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(recipe),
                        **metrics,
                    }
                )
                print(
                    f"{ensemble_name:26s} BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

    _write_ensemble_summary(rows)
    print(f"\nWrote ensemble artifacts in {OUT_DIR}", flush=True)


def run_ensemble_bootstrap(
    cohorts: list[str],
    configs: list[str],
    ensembles: list[str],
    n_bootstrap: int,
) -> None:
    unknown = sorted(set(ensembles) - set(ENSEMBLE_RECIPES))
    if unknown:
        raise RuntimeError(f"Unknown ensemble recipes: {unknown}")

    df = voter.load_analysis_data()
    rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== ensemble bootstrap {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            baseline = evaluate_ensemble_result(
                df=cohort_df,
                cohort=cohort,
                config=config,
                feature_cols=feature_cols,
                ensemble_name="xgb_current",
                recipe=ENSEMBLE_RECIPES["xgb_current"],
            )
            for ensemble_name in ensembles:
                result = evaluate_ensemble_result(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    ensemble_name=ensemble_name,
                    recipe=ENSEMBLE_RECIPES[ensemble_name],
                )
                boot_rows = voter.bootstrap_differences(
                    result,
                    baseline,
                    n_bootstrap,
                )
                for boot_row in boot_rows:
                    metric = boot_row["metric"]
                    rows.append(
                        {
                            "cohort": cohort,
                            "config": config,
                            "reference_ensemble": ensemble_name,
                            "comparison_ensemble": "xgb_current",
                            "metric": metric,
                            "reference_metric": result.metrics[metric],
                            "comparison_metric": baseline.metrics[metric],
                            "diff_reference_minus_comparison": boot_row[
                                "diff_reference_minus_comparison"
                            ],
                            "ci_low": boot_row["ci_low"],
                            "ci_high": boot_row["ci_high"],
                            "bootstrap_p_two_sided": boot_row[
                                "bootstrap_p_two_sided"
                            ],
                            "direction": boot_row["direction"],
                            "n_bootstrap": boot_row["n_bootstrap"],
                        }
                    )
                print(
                    f"{ensemble_name:26s} "
                    f"dBA={result.metrics['balanced_accuracy'] - baseline.metrics['balanced_accuracy']:+.4f} "
                    f"dLL={result.metrics['log_loss'] - baseline.metrics['log_loss']:+.4f}",
                    flush=True,
                )

    _write_ensemble_bootstrap_summary(rows, n_bootstrap, ensembles)
    print(f"\nWrote ensemble bootstrap artifacts in {OUT_DIR}", flush=True)


def run_ensemble_threshold(
    cohorts: list[str],
    configs: list[str],
    ensembles: list[str],
) -> None:
    unknown = sorted(set(ensembles) - set(ENSEMBLE_RECIPES))
    if unknown:
        raise RuntimeError(f"Unknown ensemble recipes: {unknown}")

    df = voter.load_analysis_data()
    result_rows = []
    choice_rows = []
    for cohort in cohorts:
        cohort_df = _cohort_frame(df, cohort)
        print(f"\n== ensemble threshold {cohort} n={len(cohort_df)} ==", flush=True)
        for config in configs:
            print(f"-- {config}", flush=True)
            feature_cols = voter.FEATURE_CONFIGS[config]
            for ensemble_name in ensembles:
                metrics, choices = evaluate_threshold_tuned_ensemble(
                    df=cohort_df,
                    cohort=cohort,
                    config=config,
                    feature_cols=feature_cols,
                    ensemble_name=ensemble_name,
                    recipe=ENSEMBLE_RECIPES[ensemble_name],
                )
                result_rows.append(
                    {
                        "cohort": cohort,
                        "config": config,
                        "ensemble": ensemble_name,
                        "thresholding": "inner_cv_balanced_accuracy",
                        "n": len(cohort_df),
                        "n_features_raw": len(feature_cols),
                        "recipe": json.dumps(ENSEMBLE_RECIPES[ensemble_name]),
                        **metrics,
                    }
                )
                choice_rows.extend(choices)
                print(
                    f"{ensemble_name:26s} threshold_result "
                    f"BA={metrics['balanced_accuracy']:.4f} "
                    f"LL={metrics['log_loss']:.4f} F1={metrics['macro_f1']:.4f} "
                    f"Brier={metrics['brier_score']:.4f}",
                    flush=True,
                )

    _write_ensemble_threshold_summary(result_rows, choice_rows, ensembles)
    print(f"\nWrote ensemble threshold artifacts in {OUT_DIR}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", default=list(voter.COHORTS))
    parser.add_argument("--configs", nargs="+", default=list(voter.FEATURE_CONFIGS))
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Use inner CV to select the candidate inside each outer fold.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Benchmark fold-local sigmoid/isotonic calibration for xgb_current.",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Tune the xgb_current classification threshold inside each outer fold.",
    )
    parser.add_argument(
        "--missingness",
        action="store_true",
        help="Benchmark row-level missingness indicators for xgb_current.",
    )
    parser.add_argument(
        "--xgb-nan",
        action="store_true",
        help="Benchmark native XGBoost NaN handling without imputation/scaling.",
    )
    parser.add_argument(
        "--class-balance",
        action="store_true",
        help="Benchmark fold-local class-balanced training weights for xgb_current.",
    )
    parser.add_argument(
        "--training-weights",
        action="store_true",
        help="Benchmark alternative train-fold survey-weight patterns for xgb_current.",
    )
    parser.add_argument(
        "--xgb-profiles",
        action="store_true",
        help="Benchmark fixed XGBoost-only hyperparameter profiles against xgb_current.",
    )
    parser.add_argument(
        "--issue-encoding",
        action="store_true",
        help="Benchmark categorical issue-response encoding against numeric issue coding.",
    )
    parser.add_argument(
        "--political-encoding",
        action="store_true",
        help="Benchmark categorical party-ID/ideology encoding against numeric coding.",
    )
    parser.add_argument(
        "--xgb-early-stop",
        action="store_true",
        help="Benchmark fold-local XGBoost early stopping against fixed 300 trees.",
    )
    parser.add_argument(
        "--xgb-optuna",
        action="store_true",
        help="Benchmark fold-local XGBoost-only Optuna tuning against fixed 300 trees.",
    )
    parser.add_argument(
        "--xgb-seed-bagging",
        action="store_true",
        help="Benchmark probability averaging across canonical XGBoost random seeds.",
    )
    parser.add_argument(
        "--xgb-profile-threshold",
        action="store_true",
        help="Tune thresholds for fixed XGBoost profiles inside each outer fold.",
    )
    parser.add_argument(
        "--xgb-profile-calibration",
        action="store_true",
        help="Calibrate fixed XGBoost profiles inside each outer fold.",
    )
    parser.add_argument(
        "--xgb-profile-calibration-threshold",
        action="store_true",
        help="Calibrate and threshold-tune fixed XGBoost profiles inside each outer fold.",
    )
    parser.add_argument(
        "--xgb-profile-blend-grid",
        action="store_true",
        help="Benchmark fixed probability blends of selected XGBoost profiles.",
    )
    parser.add_argument(
        "--xgb-profile-blend-threshold",
        action="store_true",
        help="Threshold-tune selected fixed XGBoost profile probability blends.",
    )
    parser.add_argument(
        "--extended-candidates",
        action="store_true",
        help="Benchmark additional fixed learner families against xgb_current.",
    )
    parser.add_argument(
        "--extended-candidate-threshold",
        action="store_true",
        help="Tune thresholds for additional fixed learner families inside each outer fold.",
    )
    parser.add_argument(
        "--extended-candidate-calibration",
        action="store_true",
        help="Calibrate additional fixed learner probabilities inside each outer fold.",
    )
    parser.add_argument(
        "--extended-candidate-calibration-threshold",
        action="store_true",
        help="Calibrate and threshold-tune additional fixed learners inside each outer fold.",
    )
    parser.add_argument(
        "--calibrated-extended-blends",
        action="store_true",
        help="Benchmark fixed blends of xgb_current and calibrated extended learners.",
    )
    parser.add_argument(
        "--calibrated-extended-blend-threshold",
        action="store_true",
        help="Tune thresholds for fixed blends of xgb_current and calibrated extended learners.",
    )
    parser.add_argument(
        "--calibrated-extended-blend-grid",
        action="store_true",
        help="Benchmark a bounded fixed weight grid for calibrated extended blends.",
    )
    parser.add_argument(
        "--calibrated-extended-blend-grid-core",
        action="store_true",
        help="Re-test BA-positive calibrated blend-grid recipes across core configs.",
    )
    parser.add_argument(
        "--candidate-bootstrap",
        action="store_true",
        help="Paired-bootstrap fixed alternative learners against xgb_current.",
    )
    parser.add_argument(
        "--feature-bootstrap",
        action="store_true",
        help="Paired-bootstrap adjacent fixed-XGBoost feature-set comparisons.",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Benchmark demographics -> predicted issues -> voter intent.",
    )
    parser.add_argument(
        "--two-stage-stage1-models",
        action="store_true",
        help="Sweep fixed Stage 1 regressors for demographics -> predicted issues.",
    )
    parser.add_argument(
        "--two-stage-stage2-models",
        action="store_true",
        help="Sweep fixed Stage 2 learners for demographics -> predicted issues.",
    )
    parser.add_argument(
        "--domain-aggregates",
        action="store_true",
        help="Benchmark compact issue-domain aggregate features for xgb_current.",
    )
    parser.add_argument(
        "--domain-pruning",
        action="store_true",
        help="Benchmark leave-one-issue-domain-out item-level feature sets.",
    )
    parser.add_argument(
        "--issue-pca",
        action="store_true",
        help="Benchmark fold-local PCA components of issue items.",
    )
    parser.add_argument(
        "--issue-topk",
        action="store_true",
        help="Benchmark fold-local top-k issue-item selection.",
    )
    parser.add_argument(
        "--ensembles",
        action="store_true",
        help="Benchmark fixed soft-voting probability ensembles.",
    )
    parser.add_argument(
        "--ensemble-bootstrap",
        action="store_true",
        help="Paired bootstrap selected soft-voting ensembles against xgb_current.",
    )
    parser.add_argument(
        "--ensemble-threshold",
        action="store_true",
        help="Tune selected soft-voting ensemble thresholds inside each outer fold.",
    )
    parser.add_argument(
        "--ensemble-bootstrap-recipes",
        nargs="+",
        default=ENSEMBLE_BOOTSTRAP_RECIPES,
        help="Ensemble recipe names to compare against xgb_current.",
    )
    parser.add_argument(
        "--candidate-bootstrap-candidates",
        nargs="+",
        default=FIXED_CANDIDATE_BOOTSTRAP_CANDIDATES,
        help="Fixed candidate learner names to compare against xgb_current.",
    )
    parser.add_argument(
        "--xgb-profile-candidates",
        nargs="+",
        default=XGB_PROFILE_CANDIDATES,
        help="Fixed XGBoost profile names to compare in --xgb-profiles.",
    )
    parser.add_argument(
        "--ensemble-threshold-recipes",
        nargs="+",
        default=ENSEMBLE_THRESHOLD_RECIPES,
        help="Ensemble recipe names to threshold-tune.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=voter.DEFAULT_BOOTSTRAP,
        help="Bootstrap replicates for ensemble pairwise comparisons.",
    )
    parser.add_argument(
        "--issue-topk-values",
        nargs="+",
        type=int,
        default=ISSUE_TOPK_VALUES,
        help="Issue-item counts to test for --issue-topk.",
    )
    parser.add_argument(
        "--xgb-optuna-trials",
        type=int,
        default=XGB_OPTUNA_DEFAULT_TRIALS,
        help="Optuna trials per outer fold for --xgb-optuna.",
    )
    parser.add_argument(
        "--xgb-seed-bagging-seeds",
        nargs="+",
        type=int,
        default=XGB_SEED_BAG_SEEDS,
        help="Random seeds to average for --xgb-seed-bagging.",
    )
    parser.add_argument(
        "--xgb-profile-threshold-candidates",
        nargs="+",
        default=XGB_PROFILE_THRESHOLD_CANDIDATES,
        help="XGBoost profile candidates to threshold-tune.",
    )
    parser.add_argument(
        "--xgb-profile-calibration-candidates",
        nargs="+",
        default=XGB_PROFILE_CALIBRATION_CANDIDATES,
        help="XGBoost profile candidates to calibrate.",
    )
    parser.add_argument(
        "--xgb-profile-calibration-methods",
        nargs="+",
        default=XGB_PROFILE_CALIBRATION_METHODS,
        help="Calibration methods to test for XGBoost profiles.",
    )
    parser.add_argument(
        "--xgb-profile-calibration-threshold-candidates",
        nargs="+",
        default=XGB_PROFILE_CALIBRATION_THRESHOLD_CANDIDATES,
        help="XGBoost profile candidates to calibrate and threshold-tune.",
    )
    parser.add_argument(
        "--xgb-profile-calibration-threshold-methods",
        nargs="+",
        default=XGB_PROFILE_CALIBRATION_THRESHOLD_METHODS,
        help="Calibration methods to test before XGBoost profile threshold tuning.",
    )
    parser.add_argument(
        "--xgb-profile-blend-grid-recipes",
        nargs="+",
        default=list(XGB_PROFILE_BLEND_RECIPES),
        help="XGBoost profile blend-grid recipes to test.",
    )
    parser.add_argument(
        "--xgb-profile-blend-threshold-recipes",
        nargs="+",
        default=XGB_PROFILE_BLEND_THRESHOLD_RECIPES,
        help="XGBoost profile blend recipes to threshold-tune.",
    )
    parser.add_argument(
        "--two-stage-stage1-model-names",
        nargs="+",
        default=TWO_STAGE_STAGE1_MODELS,
        help="Stage 1 regressor names to sweep for --two-stage-stage1-models.",
    )
    parser.add_argument(
        "--two-stage-stage2-candidates",
        nargs="+",
        default=TWO_STAGE_STAGE2_CANDIDATES,
        help="Stage 2 candidate learner names to sweep for --two-stage-stage2-models.",
    )
    parser.add_argument(
        "--extended-candidate-names",
        nargs="+",
        default=EXTENDED_CANDIDATES,
        help="Additional fixed learner names to compare for --extended-candidates.",
    )
    parser.add_argument(
        "--extended-candidate-threshold-candidates",
        nargs="+",
        default=EXTENDED_CANDIDATE_THRESHOLD_CANDIDATES,
        help="Additional fixed learner names to threshold-tune.",
    )
    parser.add_argument(
        "--extended-candidate-calibration-candidates",
        nargs="+",
        default=EXTENDED_CANDIDATE_CALIBRATION_CANDIDATES,
        help="Additional fixed learner names to calibrate.",
    )
    parser.add_argument(
        "--extended-candidate-calibration-methods",
        nargs="+",
        default=EXTENDED_CANDIDATE_CALIBRATION_METHODS,
        help="Calibration methods to test for additional fixed learners.",
    )
    parser.add_argument(
        "--extended-candidate-calibration-threshold-candidates",
        nargs="+",
        default=EXTENDED_CANDIDATE_CALIBRATION_THRESHOLD_CANDIDATES,
        help="Additional fixed learner names to calibrate and threshold-tune.",
    )
    parser.add_argument(
        "--extended-candidate-calibration-threshold-methods",
        nargs="+",
        default=EXTENDED_CANDIDATE_CALIBRATION_THRESHOLD_METHODS,
        help="Calibration methods to test before threshold tuning.",
    )
    parser.add_argument(
        "--calibrated-extended-blend-recipes",
        nargs="+",
        default=list(CALIBRATED_EXTENDED_BLEND_RECIPES),
        help="Fixed calibrated extended blend recipes to test.",
    )
    parser.add_argument(
        "--calibrated-extended-blend-threshold-recipes",
        nargs="+",
        default=CALIBRATED_EXTENDED_BLEND_THRESHOLD_RECIPES,
        help="Fixed calibrated extended blend recipes to threshold-tune.",
    )
    parser.add_argument(
        "--calibrated-extended-blend-grid-recipes",
        nargs="+",
        default=list(CALIBRATED_EXTENDED_BLEND_GRID_RECIPES),
        help="Fixed calibrated extended blend grid recipes to test.",
    )
    parser.add_argument(
        "--calibrated-extended-blend-grid-core-recipes",
        nargs="+",
        default=CALIBRATED_EXTENDED_BLEND_GRID_CORE_RECIPES,
        help="Calibrated blend-grid recipes to re-test across core configs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one small all-respondent issues-only comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        if args.ensemble_threshold:
            run_ensemble_threshold(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                ensembles=[args.ensemble_threshold_recipes[0]],
            )
            return
        if args.ensemble_bootstrap:
            run_ensemble_bootstrap(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                ensembles=[args.ensemble_bootstrap_recipes[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.ensembles:
            run_ensembles(
                cohorts=["all_respondents"],
                configs=["issues_only"],
            )
            return
        if args.domain_aggregates:
            run_domain_aggregates(cohorts=["independents_plus_leaners"])
            return
        if args.domain_pruning:
            run_domain_pruning(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.issue_pca:
            run_issue_pca(
                cohorts=["all_respondents"],
                configs=["pca_issues_only"],
                components=[5],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.issue_topk:
            run_issue_topk(
                cohorts=["all_respondents"],
                configs=["topk_issues_only"],
                top_k_values=[5],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.two_stage:
            run_two_stage(cohorts=["independents_plus_leaners"])
            return
        if args.two_stage_stage1_models:
            smoke_stage1_models = ["histgb_current"]
            smoke_stage1_models.extend(
                name
                for name in args.two_stage_stage1_model_names
                if name != "histgb_current"
            )
            run_two_stage_stage1_models(
                cohorts=["independents_plus_leaners"],
                stage1_models=smoke_stage1_models[:2],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.two_stage_stage2_models:
            smoke_stage2_candidates = [BASELINE_CANDIDATE]
            smoke_stage2_candidates.extend(
                name
                for name in args.two_stage_stage2_candidates
                if name != BASELINE_CANDIDATE
            )
            run_two_stage_stage2_models(
                cohorts=["independents_plus_leaners"],
                stage2_candidates=smoke_stage2_candidates[:2],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_nan:
            run_xgb_nan(
                cohorts=["all_respondents"],
                configs=["issues_only"],
            )
            return
        if args.class_balance:
            run_class_balance(
                cohorts=["all_respondents"],
                configs=["issues_only"],
            )
            return
        if args.training_weights:
            run_training_weights(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_profiles:
            smoke_candidates = [
                BASELINE_CANDIDATE,
                next(
                    candidate
                    for candidate in args.xgb_profile_candidates
                    if candidate != BASELINE_CANDIDATE
                ),
            ]
            run_xgb_profiles(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                candidates=smoke_candidates,
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.issue_encoding:
            run_issue_encoding(
                cohorts=["all_respondents"],
                configs=["issue_categorical_issues_only"],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.political_encoding:
            run_political_encoding(
                cohorts=["all_respondents"],
                configs=["political_categorical_party_id_only"],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_early_stop:
            run_xgb_early_stop(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_optuna:
            run_xgb_optuna(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                n_trials=min(args.xgb_optuna_trials, 3),
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_seed_bagging:
            run_xgb_seed_bagging(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                seeds=args.xgb_seed_bagging_seeds,
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_profile_threshold:
            run_xgb_profile_threshold(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                candidates=[args.xgb_profile_threshold_candidates[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_profile_calibration:
            run_xgb_profile_calibration(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                candidates=[args.xgb_profile_calibration_candidates[0]],
                methods=[args.xgb_profile_calibration_methods[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_profile_calibration_threshold:
            run_xgb_profile_calibration_threshold(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                candidates=[args.xgb_profile_calibration_threshold_candidates[0]],
                methods=[args.xgb_profile_calibration_threshold_methods[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_profile_blend_grid:
            run_xgb_profile_blend_grid(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                recipes=[args.xgb_profile_blend_grid_recipes[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.xgb_profile_blend_threshold:
            run_xgb_profile_blend_threshold(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                recipes=[args.xgb_profile_blend_threshold_recipes[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.extended_candidates:
            run_extended_candidates(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                candidates=[args.extended_candidate_names[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.extended_candidate_threshold:
            run_extended_candidate_threshold(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                candidates=[args.extended_candidate_threshold_candidates[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.extended_candidate_calibration:
            run_extended_candidate_calibration(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                candidates=[args.extended_candidate_calibration_candidates[0]],
                methods=[args.extended_candidate_calibration_methods[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.extended_candidate_calibration_threshold:
            run_extended_candidate_calibration_threshold(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                candidates=[args.extended_candidate_calibration_threshold_candidates[0]],
                methods=[args.extended_candidate_calibration_threshold_methods[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.calibrated_extended_blends:
            run_calibrated_extended_blends(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                recipes=[args.calibrated_extended_blend_recipes[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.calibrated_extended_blend_threshold:
            run_calibrated_extended_blend_threshold(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                recipes=[args.calibrated_extended_blend_threshold_recipes[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.calibrated_extended_blend_grid:
            run_calibrated_extended_blend_grid(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                recipes=[args.calibrated_extended_blend_grid_recipes[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.calibrated_extended_blend_grid_core:
            run_calibrated_extended_blend_grid_core(
                cohorts=["all_respondents"],
                configs=["issues+party_id+ideology"],
                recipes=[args.calibrated_extended_blend_grid_core_recipes[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.candidate_bootstrap:
            run_fixed_candidate_bootstrap(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                candidates=[args.candidate_bootstrap_candidates[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.feature_bootstrap:
            run_feature_bootstrap(
                cohorts=["all_respondents"],
                pairs=[FEATURE_BOOTSTRAP_PAIRS[0]],
                n_bootstrap=min(args.n_bootstrap, 200),
            )
            return
        if args.missingness:
            run_missingness(
                cohorts=["all_respondents"],
                configs=["issues_only"],
            )
            return
        if args.threshold:
            run_threshold(
                cohorts=["all_respondents"],
                configs=["issues_only"],
            )
            return
        if args.calibration:
            run_calibration(
                cohorts=["all_respondents"],
                configs=["issues_only"],
            )
            return
        if args.nested:
            run_nested(
                cohorts=["all_respondents"],
                configs=["issues_only"],
                candidates=[BASELINE_CANDIDATE, "xgb_more_trees"],
            )
            return
        run(
            cohorts=["all_respondents"],
            configs=["issues_only"],
            candidates=[BASELINE_CANDIDATE, "xgb_more_trees"],
        )
        return
    if args.calibration:
        run_calibration(args.cohorts, args.configs)
        return
    if args.threshold:
        run_threshold(args.cohorts, args.configs)
        return
    if args.missingness:
        run_missingness(args.cohorts, args.configs)
        return
    if args.xgb_nan:
        run_xgb_nan(args.cohorts, args.configs)
        return
    if args.class_balance:
        run_class_balance(args.cohorts, args.configs)
        return
    if args.training_weights:
        run_training_weights(args.cohorts, args.configs, args.n_bootstrap)
        return
    if args.xgb_profiles:
        run_xgb_profiles(
            args.cohorts,
            args.configs,
            args.xgb_profile_candidates,
            args.n_bootstrap,
        )
        return
    if args.issue_encoding:
        run_issue_encoding(
            args.cohorts,
            ISSUE_ENCODING_CONFIGS,
            args.n_bootstrap,
        )
        return
    if args.political_encoding:
        run_political_encoding(
            args.cohorts,
            POLITICAL_ENCODING_CONFIGS,
            args.n_bootstrap,
        )
        return
    if args.xgb_early_stop:
        run_xgb_early_stop(args.cohorts, args.configs, args.n_bootstrap)
        return
    if args.xgb_optuna:
        run_xgb_optuna(
            args.cohorts,
            XGB_OPTUNA_DEFAULT_CONFIGS,
            args.xgb_optuna_trials,
            args.n_bootstrap,
        )
        return
    if args.xgb_seed_bagging:
        run_xgb_seed_bagging(
            args.cohorts,
            XGB_SEED_BAG_CONFIGS,
            args.xgb_seed_bagging_seeds,
            args.n_bootstrap,
        )
        return
    if args.xgb_profile_threshold:
        run_xgb_profile_threshold(
            args.cohorts,
            XGB_PROFILE_THRESHOLD_CONFIGS,
            args.xgb_profile_threshold_candidates,
            args.n_bootstrap,
        )
        return
    if args.xgb_profile_calibration:
        run_xgb_profile_calibration(
            args.cohorts,
            XGB_PROFILE_CALIBRATION_CONFIGS,
            args.xgb_profile_calibration_candidates,
            args.xgb_profile_calibration_methods,
            args.n_bootstrap,
        )
        return
    if args.xgb_profile_calibration_threshold:
        run_xgb_profile_calibration_threshold(
            args.cohorts,
            XGB_PROFILE_CALIBRATION_THRESHOLD_CONFIGS,
            args.xgb_profile_calibration_threshold_candidates,
            args.xgb_profile_calibration_threshold_methods,
            args.n_bootstrap,
        )
        return
    if args.xgb_profile_blend_grid:
        run_xgb_profile_blend_grid(
            args.cohorts,
            XGB_PROFILE_BLEND_CONFIGS,
            args.xgb_profile_blend_grid_recipes,
            args.n_bootstrap,
        )
        return
    if args.xgb_profile_blend_threshold:
        run_xgb_profile_blend_threshold(
            args.cohorts,
            XGB_PROFILE_BLEND_THRESHOLD_CONFIGS,
            args.xgb_profile_blend_threshold_recipes,
            args.n_bootstrap,
        )
        return
    if args.extended_candidates:
        run_extended_candidates(
            args.cohorts,
            EXTENDED_CANDIDATE_CONFIGS,
            args.extended_candidate_names,
            args.n_bootstrap,
        )
        return
    if args.extended_candidate_threshold:
        run_extended_candidate_threshold(
            args.cohorts,
            EXTENDED_CANDIDATE_THRESHOLD_CONFIGS,
            args.extended_candidate_threshold_candidates,
            args.n_bootstrap,
        )
        return
    if args.extended_candidate_calibration:
        run_extended_candidate_calibration(
            args.cohorts,
            EXTENDED_CANDIDATE_CALIBRATION_CONFIGS,
            args.extended_candidate_calibration_candidates,
            args.extended_candidate_calibration_methods,
            args.n_bootstrap,
        )
        return
    if args.extended_candidate_calibration_threshold:
        run_extended_candidate_calibration_threshold(
            args.cohorts,
            EXTENDED_CANDIDATE_CALIBRATION_THRESHOLD_CONFIGS,
            args.extended_candidate_calibration_threshold_candidates,
            args.extended_candidate_calibration_threshold_methods,
            args.n_bootstrap,
        )
        return
    if args.calibrated_extended_blends:
        run_calibrated_extended_blends(
            args.cohorts,
            CALIBRATED_EXTENDED_BLEND_CONFIGS,
            args.calibrated_extended_blend_recipes,
            args.n_bootstrap,
        )
        return
    if args.calibrated_extended_blend_threshold:
        run_calibrated_extended_blend_threshold(
            args.cohorts,
            CALIBRATED_EXTENDED_BLEND_THRESHOLD_CONFIGS,
            args.calibrated_extended_blend_threshold_recipes,
            args.n_bootstrap,
        )
        return
    if args.calibrated_extended_blend_grid:
        run_calibrated_extended_blend_grid(
            args.cohorts,
            CALIBRATED_EXTENDED_BLEND_GRID_CONFIGS,
            args.calibrated_extended_blend_grid_recipes,
            args.n_bootstrap,
        )
        return
    if args.calibrated_extended_blend_grid_core:
        run_calibrated_extended_blend_grid_core(
            args.cohorts,
            CALIBRATED_EXTENDED_BLEND_GRID_CORE_CONFIGS,
            args.calibrated_extended_blend_grid_core_recipes,
            args.n_bootstrap,
        )
        return
    if args.candidate_bootstrap:
        run_fixed_candidate_bootstrap(
            args.cohorts,
            args.configs,
            args.candidate_bootstrap_candidates,
            args.n_bootstrap,
        )
        return
    if args.feature_bootstrap:
        run_feature_bootstrap(
            args.cohorts,
            FEATURE_BOOTSTRAP_PAIRS,
            args.n_bootstrap,
        )
        return
    if args.two_stage:
        run_two_stage(args.cohorts)
        return
    if args.two_stage_stage1_models:
        run_two_stage_stage1_models(
            args.cohorts,
            args.two_stage_stage1_model_names,
            args.n_bootstrap,
        )
        return
    if args.two_stage_stage2_models:
        run_two_stage_stage2_models(
            args.cohorts,
            args.two_stage_stage2_candidates,
            args.n_bootstrap,
        )
        return
    if args.domain_aggregates:
        run_domain_aggregates(args.cohorts)
        return
    if args.domain_pruning:
        run_domain_pruning(
            args.cohorts,
            DOMAIN_PRUNING_CONFIGS,
            args.n_bootstrap,
        )
        return
    if args.issue_pca:
        run_issue_pca(
            args.cohorts,
            ISSUE_PCA_CONFIGS,
            ISSUE_PCA_COMPONENTS,
            args.n_bootstrap,
        )
        return
    if args.issue_topk:
        run_issue_topk(
            args.cohorts,
            ISSUE_TOPK_CONFIGS,
            args.issue_topk_values,
            args.n_bootstrap,
        )
        return
    if args.ensemble_threshold:
        run_ensemble_threshold(
            args.cohorts,
            args.configs,
            args.ensemble_threshold_recipes,
        )
        return
    if args.ensemble_bootstrap:
        run_ensemble_bootstrap(
            args.cohorts,
            args.configs,
            args.ensemble_bootstrap_recipes,
            args.n_bootstrap,
        )
        return
    if args.ensembles:
        run_ensembles(args.cohorts, args.configs)
        return
    if args.nested:
        run_nested(args.cohorts, args.configs, args.candidates)
        return
    run(args.cohorts, args.configs, args.candidates)


if __name__ == "__main__":
    main()
