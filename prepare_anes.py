"""
ANES 2024 Time Series — Data Preparation for TCC Research
"Modelagem de Padrões de Opiniões Políticas a partir de pautas ideológicas"

Selects, renames, and cleans variables from the raw ANES 2024 CSV.
Outputs:
  - anes_2024_clean.csv           (voters only, ~3,693 rows)
  - anes_2024_with_abstained.csv  (includes non-voters, ~4,971 rows)
  - variable_dictionary.csv       (column mapping and metadata)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_CSV = Path("anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv")
OUT_DIR = Path(".")

# ── Column mapping: (new_name, anes_code, description, category, scale) ──────

VARIABLE_MAP = [
    # === DEMOGRAPHICS ===
    ("age_group",          "V241008x",  "4-cat age group",                          "demographic", "1-4 ordinal"),
    ("sex",                "V241006",   "Binary sex (1=Male, 2=Female)",            "demographic", "1-2 nominal"),
    ("gender",             "V241004",   "5-cat gender identity",                    "demographic", "1-5 nominal"),
    ("race",               "V241501x",  "6-cat race/ethnicity summary",             "demographic", "1-6 nominal"),
    ("education",          "V242458x",  "7-cat education level (post-election)",    "demographic", "1-7 ordinal"),
    ("income_bracket",     "V241507",   "4-cat household income (low quality)",     "demographic", "1-4 ordinal"),
    ("marital",            "V241506",   "3-cat marital status",                     "demographic", "1-3 nominal"),
    ("employment",         "V241512x",  "5-cat employment status",                  "demographic", "1-7 nominal"),
    ("social_class",       "V242337",   "4-cat self-reported social class",         "demographic", "1-4 ordinal"),
    ("church_attendance",  "V241025",   "Frequency of religious attendance",        "demographic", "1-5 ordinal"),
    ("ideology",           "V241179",   "7-pt liberal-conservative self-placement", "demographic", "1-7 ordinal"),
    ("hispanic",           "V241502",   "Binary Hispanic indicator",               "demographic", "1-2 nominal"),
    ("state",              "V243001",   "State abbreviation",                       "demographic", "string"),

    # === ECONOMY & TAXES ===
    ("econ_opportunity",      "V242311",  "How much opportunity to get ahead",           "issue_economy", "1-5 ordinal"),
    ("econ_mobility",         "V242312",  "Economic mobility vs 20 years ago",           "issue_economy", "ordinal"),
    ("econ_mobility_summary", "V242314x", "Economic mobility summary",                   "issue_economy", "summary"),
    ("tax_millionaires",      "V242316a", "Favor/oppose tax on millionaires",            "issue_economy", "favor/oppose"),
    ("deficit_importance",    "V242315",  "Importance of reducing deficit",               "issue_economy", "ordinal"),
    ("min_wage",              "V242350",  "Should minimum wage be raised/kept/lowered",  "issue_economy", "ordinal"),

    # === IMMIGRATION ===
    ("immig_level",    "V242232", "Immigration level: increase/decrease/same",   "issue_immigration", "1-3 ordinal"),
    ("immig_policy1",  "V242235", "Immigration policy position 1",              "issue_immigration", "ordinal"),
    ("immig_policy2",  "V242236", "Immigration policy position 2",              "issue_immigration", "ordinal"),
    ("immig_policy3",  "V242237", "Immigration policy position 3",              "issue_immigration", "ordinal"),

    # === GUN CONTROL ===
    ("gun_difficulty",   "V242325",  "Should buying guns be harder/easier",      "issue_guns", "1-3 ordinal"),
    ("gun_background",   "V242328x", "Favor/oppose background checks summary",  "issue_guns", "summary"),
    ("gun_assault_ban",  "V242331x", "Favor/oppose assault rifle ban summary",  "issue_guns", "summary"),

    # === ENVIRONMENT / CLIMATE ===
    ("climate_impact",      "V242321",  "How much climate change affects weather",        "issue_environment", "1-5 ordinal"),
    ("climate_regulation",  "V242324x", "Favor/oppose greenhouse regulation summary",     "issue_environment", "summary"),

    # === ABORTION ===
    ("abortion_access",  "V242362",  "Abortion access position (3-cat)",       "issue_abortion", "1-3 ordinal"),
    ("abortion_scale",   "V242361x", "Abortion position summary (5-pt)",       "issue_abortion", "1-5 summary"),
    ("abortion_detail",  "V242364x", "Abortion position detail (7-pt)",        "issue_abortion", "1-7 summary"),

    # === RACIAL ATTITUDES ===
    ("racial_selfhelp",   "V242300", "Blacks should work way up w/o favors",  "issue_racial", "1-5 ordinal"),
    ("racial_slavery",    "V242301", "Past slavery makes it harder for Blacks","issue_racial", "1-5 ordinal"),
    ("racial_deserve",    "V242302", "Blacks gotten less than deserved",       "issue_racial", "1-5 ordinal"),
    ("racial_tryharder",  "V242303", "If Blacks tried harder, equal outcome", "issue_racial", "1-5 ordinal"),

    # === POPULISM / SYSTEM FAIRNESS ===
    ("populism_insiders",   "V242304",  "System only works for insiders",      "issue_populism", "ordinal"),
    ("populism_rich",       "V242305",  "Rich/powerful prevent others getting ahead", "issue_populism", "ordinal"),
    ("fairshare_summary",   "V242308x", "Gets more/less than fair share summary",    "issue_populism", "summary"),

    # === GENDER ROLES ===
    ("gender_trad_summary", "V242279x", "Better if man works/woman home summary",  "issue_gender", "summary"),
    ("gender_equality",     "V242280",  "Women demanding equality seek favors",    "issue_gender", "ordinal"),
    ("gender_discrim",      "V242281",  "Women complaining cause more problems",   "issue_gender", "ordinal"),

    # === TRADE ===
    ("trade_summary", "V242346x", "Favor/oppose free trade summary", "issue_trade", "summary"),

    # === DIVERSITY ===
    ("diversity_summary", "V242349x", "Increasing diversity better/worse summary", "issue_diversity", "summary"),

    # === HEALTHCARE ===
    ("healthcare_spending", "V242353x", "Govt help paying for healthcare summary", "issue_healthcare", "summary"),

    # === TARGET ===
    ("vote_choice", "V242067",  "Presidential vote: 1=Harris(Dem),2=Trump(Rep),4-6=Other", "target", "nominal"),
    ("party_id",    "V241227x", "7-point party identification",                   "auxiliary", "1-7 ordinal"),

    # === WEIGHTS ===
    ("weight_pre",  "V240107a", "Full-sample pre-election weight",  "weight", "continuous"),
    ("weight_post", "V240108a", "Post-election weight",             "weight", "continuous"),
]


def load_raw(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    return df


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    anes_to_new = {row[1]: row[0] for row in VARIABLE_MAP}
    missing_cols = [c for c in anes_to_new if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: columns not found in CSV: {missing_cols}")
    existing = {k: v for k, v in anes_to_new.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)
    print(f"  Selected {len(df.columns)} columns")
    return df


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ANES negative codes with NaN for numeric columns."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            df.loc[df[col] < 0, col] = np.nan

    # Convert weight_post from string to float
    if "weight_post" in df.columns and df["weight_post"].dtype == object:
        df["weight_post"] = pd.to_numeric(df["weight_post"], errors="coerce")

    # Report missingness
    miss = df.isnull().mean().sort_values(ascending=False)
    print("\n  Missingness per column:")
    for col, pct in miss.items():
        flag = " ⚠" if pct > 0.4 else ""
        print(f"    {col:30s} {pct:6.1%}{flag}")

    return df


def recode_vote(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create two versions: voters-only and with-abstained."""
    # ANES codes: 1=Democratic candidate (Harris), 2=Republican candidate (Trump)
    vote_map = {1: "Harris", 2: "Trump", 4: "Other", 5: "Other", 6: "Other"}

    # Version with abstained (NaN vote_choice → didn't vote or refused)
    df_all = df.copy()
    # Keep rows where vote_choice is valid (1,2,4,5,6) OR was -1 (didn't vote, now NaN)
    # We need the original values, but they're already NaN. Use raw approach:
    # Actually, vote_choice NaN could be: didn't vote (-1), refused (-6), DK (-7), etc.
    # Re-read raw vote to distinguish
    raw_vote = pd.read_csv(
        RAW_CSV, usecols=["V242067"], low_memory=False
    )["V242067"]
    df_all["vote_choice_raw"] = raw_vote.values

    # Voters only
    voters_mask = df_all["vote_choice_raw"].isin([1, 2, 4, 5, 6])
    df_voters = df_all[voters_mask].copy()
    df_voters["vote_choice"] = df_voters["vote_choice_raw"].map(vote_map)
    df_voters.drop(columns=["vote_choice_raw"], inplace=True)

    # With abstained
    abstained_mask = df_all["vote_choice_raw"].isin([1, 2, 4, 5, 6, -1])
    df_abstained = df_all[abstained_mask].copy()
    vote_map_abs = {1: "Harris", 2: "Trump", 4: "Other", 5: "Other", 6: "Other", -1: "Abstained"}
    df_abstained["vote_choice"] = df_abstained["vote_choice_raw"].map(vote_map_abs)
    df_abstained.drop(columns=["vote_choice_raw"], inplace=True)

    # Clean up main df
    df_all.drop(columns=["vote_choice_raw"], inplace=True)

    return df_voters, df_abstained


def export_dictionary():
    rows = []
    for new_name, anes_code, desc, category, scale in VARIABLE_MAP:
        rows.append({
            "column_name": new_name,
            "anes_code": anes_code,
            "description": desc,
            "category": category,
            "scale": scale,
        })
    dict_df = pd.DataFrame(rows)
    out = OUT_DIR / "variable_dictionary.csv"
    dict_df.to_csv(out, index=False)
    print(f"\n  Variable dictionary → {out}")


def print_summary(df: pd.DataFrame, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"\n  Vote choice distribution:")
    print(df["vote_choice"].value_counts().to_string())
    print(f"\n  Party ID distribution (valid only):")
    pid = df["party_id"].dropna()
    print(pid.value_counts().sort_index().to_string())
    print(f"\n  Demographic completeness:")
    demo_cols = [r[0] for r in VARIABLE_MAP if r[3] == "demographic"]
    for c in demo_cols:
        if c in df.columns:
            valid_pct = df[c].notna().mean()
            print(f"    {c:25s} {valid_pct:6.1%} valid")


def main():
    df = load_raw(RAW_CSV)
    df = select_and_rename(df)
    df = clean_missing(df)
    export_dictionary()

    df_voters, df_abstained = recode_vote(df)

    # Export
    out_voters = OUT_DIR / "anes_2024_clean.csv"
    out_abstained = OUT_DIR / "anes_2024_with_abstained.csv"
    df_voters.to_csv(out_voters, index=False)
    df_abstained.to_csv(out_abstained, index=False)
    print(f"\n  Voters only   → {out_voters}")
    print(f"  With abstained → {out_abstained}")

    print_summary(df_voters, "VOTERS ONLY")
    print_summary(df_abstained, "WITH ABSTAINED")


if __name__ == "__main__":
    main()
