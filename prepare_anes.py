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
# Re-audited 2026-05-31 against the ANES 2024 codebook: the demographic block and
# a few issue codes were mislabeled (e.g. V241507=birthplace, V241179=Harris
# placement, V242458x=trans sports). Corrected below to match the codebook and the
# (already-correct) pre-election script run_voter_intent_analysis.py.
# weight_post changed from V240108a (full-sample) to V240108b (respondent weight)
# since the pipeline uses post-election variables exclusively.

VARIABLE_MAP = [
    # === DEMOGRAPHICS ===
    ("age_group",          "V241458x",  "Respondent age on election day (summary)", "demographic", "ordinal"),
    ("sex",                "V241550",   "Binary sex (1=Male, 2=Female)",            "demographic", "1-2 nominal"),
    # NOTE: V241004 is actually "political attention", not gender identity.
    # The original mapping was incorrect. Gender identity is not available
    # as a separate variable in this dataset (sex covers it at V241006).
    ("race",               "V241501x",  "6-cat race/ethnicity summary",             "demographic", "1-6 nominal"),
    ("education",          "V241465x",  "5-cat education level",                    "demographic", "1-5 ordinal"),
    ("income_bracket",     "V241567x",  "6-cat household income",                   "demographic", "1-6 ordinal"),
    ("marital",            "V241461x",  "5-cat marital status",                     "demographic", "1-5 nominal"),
    ("employment",         "V241488x",  "Occupation/employment status (1-digit)",   "demographic", "nominal"),
    ("social_class",       "V242337",   "4-cat self-reported social class",         "demographic", "1-4 ordinal"),
    ("church_attendance",  "V241440",   "Frequency of religious attendance",        "demographic", "1-5 ordinal"),
    ("ideology",           "V241177",   "7-pt liberal-conservative self-placement", "demographic", "1-7 ordinal"),
    ("hispanic",           "V241512x",  "Hispanic origin summary",                 "demographic", "nominal"),
    ("state",              "V243001",   "State abbreviation",                       "demographic", "string"),

    # === ECONOMY & TAXES ===
    ("econ_opportunity",      "V242311",  "How much opportunity to get ahead",           "issue_economy", "1-5 ordinal"),
    ("econ_mobility",         "V242312",  "Economic mobility vs 20 years ago",           "issue_economy", "ordinal"),
    ("econ_mobility_summary", "V242314x", "Economic mobility summary",                   "issue_economy", "summary"),
    ("tax_millionaires",      "V242316a", "Favor/oppose tax on millionaires",            "issue_economy", "favor/oppose"),
    ("deficit_importance",    "V242315",  "Importance of reducing deficit",               "issue_economy", "ordinal"),
    ("min_wage",              "V242350",  "Should minimum wage be raised/kept/lowered",  "issue_economy", "ordinal"),

    # === IMMIGRATION ===
    ("immig_level",    "V242227", "Immigration level: increase/decrease/same",   "issue_immigration", "ordinal"),
    ("immig_policy1",  "V242235", "Immigrants good or bad for economy",          "issue_immigration", "ordinal"),
    ("immig_policy2",  "V242236", "Importance immigrants adapt to US customs",   "issue_immigration", "ordinal"),
    ("immig_policy3",  "V242234x", "Path to citizenship (summary)",              "issue_immigration", "summary"),

    # === GUN CONTROL ===
    ("gun_difficulty",   "V242325",  "Should buying guns be harder/easier",      "issue_guns", "1-3 ordinal"),
    ("gun_background",   "V242328x", "Favor/oppose background checks summary",  "issue_guns", "summary"),
    ("gun_assault_ban",  "V242331x", "Favor/oppose assault rifle ban summary",  "issue_guns", "summary"),

    # === ENVIRONMENT / CLIMATE ===
    ("climate_impact",      "V242321",  "How much climate change affects weather",        "issue_environment", "1-5 ordinal"),
    ("climate_regulation",  "V242324x", "Favor/oppose greenhouse regulation summary",     "issue_environment", "summary"),

    # === ABORTION === (DROPPED 2026-05-31: the POST-election wave has no respondent
    # abortion-POSITION variable. The codes used before (V242362, V242361x, V242364x)
    # were mislabeled trans-military / sexual-harassment items. Only V242176 (issue
    # importance) and V242186 (party handling) exist post-wave; neither is a position.)

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

    # === MEDIA & ENGAGEMENT (lifestyle proxies) ===
    ("political_attention",  "V241004",  "How often pays attention to politics/govt",  "lifestyle", "1-5 ordinal"),
    ("campaign_interest",    "V241005",  "How interested in following campaigns",      "lifestyle", "1-3 ordinal"),
    ("media_trust",          "V241335",  "How much trust in news media",               "lifestyle", "1-5 ordinal"),

    # Media sources used for campaign news (binary: 0=no, 1=yes)
    ("media_tv",             "V241600a", "Used TV for campaign news",            "lifestyle", "binary"),
    ("media_newspapers",     "V241600b", "Used newspapers for campaign news",    "lifestyle", "binary"),
    ("media_internet",       "V241600c", "Used internet for campaign news",      "lifestyle", "binary"),
    ("media_radio",          "V241600d", "Used radio for campaign news",         "lifestyle", "binary"),

    # TV shows watched regularly — ENTERTAINMENT (cultural lifestyle signals)
    ("tv_fbi",               "V241601a", "Watches FBI regularly",                "lifestyle_entertainment", "binary"),
    ("tv_shark_tank",        "V241601j", "Watches Shark Tank regularly",         "lifestyle_entertainment", "binary"),
    ("tv_ncis",              "V241601r", "Watches NCIS regularly",               "lifestyle_entertainment", "binary"),
    ("tv_daily_show",        "V241601s", "Watches Daily Show w/ Jon Stewart",    "lifestyle_entertainment", "binary"),
    ("tv_price_right",       "V241602h", "Watches The Price Is Right regularly", "lifestyle_entertainment", "binary"),
    ("tv_snl",               "V241602r", "Watches Saturday Night Live regularly","lifestyle_entertainment", "binary"),
    ("tv_dateline",          "V241602e", "Watches Dateline regularly",           "lifestyle_entertainment", "binary"),
    ("tv_60_minutes",        "V241602c", "Watches 60 Minutes regularly",         "lifestyle_entertainment", "binary"),
    ("tv_today",             "V241602j", "Watches Today show regularly",         "lifestyle_entertainment", "binary"),
    ("tv_gma",               "V241602i", "Watches Good Morning America",         "lifestyle_entertainment", "binary"),
    ("tv_2020",              "V241602d", "Watches 20/20 regularly",              "lifestyle_entertainment", "binary"),

    # TV shows watched regularly — POLITICAL NEWS (left-leaning)
    ("tv_maddow",            "V241601d", "Watches Rachel Maddow Show",           "lifestyle_political", "binary"),
    ("tv_odonnell_last_word","V241601e", "Watches Last Word Lawrence O'Donnell", "lifestyle_political", "binary"),
    ("tv_anderson_cooper",   "V241602a", "Watches Anderson Cooper 360",          "lifestyle_political", "binary"),
    ("tv_chris_hayes",       "V241602n", "Watches All In with Chris Hayes",      "lifestyle_political", "binary"),
    ("tv_pbs_newshour",      "V241602q", "Watches PBS NewsHour",                 "lifestyle_political", "binary"),

    # TV shows watched regularly — POLITICAL NEWS (right-leaning)
    ("tv_hannity",           "V241601b", "Watches Hannity regularly",            "lifestyle_political", "binary"),
    ("tv_jesse_watters",     "V241601c", "Watches Jesse Watters Primetime",      "lifestyle_political", "binary"),
    ("tv_the_five",          "V241601f", "Watches The Five regularly",           "lifestyle_political", "binary"),
    ("tv_ingraham",          "V241601g", "Watches The Ingraham Angle",           "lifestyle_political", "binary"),
    ("tv_gutfeld",           "V241601h", "Watches Gutfeld! regularly",           "lifestyle_political", "binary"),

    # TV shows watched regularly — MAINSTREAM NEWS
    ("tv_bret_baier",        "V241601k", "Watches Special Report Bret Baier",   "lifestyle_political", "binary"),
    ("tv_david_muir",        "V241601n", "Watches World News Tonight Muir",     "lifestyle_political", "binary"),
    ("tv_lester_holt",       "V241601p", "Watches Nightly News Lester Holt",    "lifestyle_political", "binary"),
    ("tv_meet_press",        "V241602g", "Watches Meet the Press",              "lifestyle_political", "binary"),
    ("tv_face_nation",       "V241602f", "Watches Face the Nation",             "lifestyle_political", "binary"),

    # Websites visited regularly
    ("web_cnn",              "V241605b", "Visits CNN regularly",                 "lifestyle_web", "binary"),
    ("web_fox_news",         "V241605f", "Visits Fox News regularly",            "lifestyle_web", "binary"),
    ("web_politico",         "V241605p", "Visits Politico regularly",            "lifestyle_web", "binary"),
    ("web_breitbart",        "V241605e", "Visits Breitbart regularly",           "lifestyle_web", "binary"),
    ("web_npr",              "V241605k", "Visits NPR News regularly",            "lifestyle_web", "binary"),
    ("web_newsmax",          "V241605c", "Visits Newsmax regularly",             "lifestyle_web", "binary"),
    ("web_bbc",              "V241605j", "Visits BBC News regularly",            "lifestyle_web", "binary"),

    # Emotions about how things are going in the country
    ("emotion_hope",         "V241118",  "How hopeful about country direction",  "lifestyle_emotion", "1-5 ordinal"),
    ("emotion_fear",         "V241119",  "How afraid about country direction",   "lifestyle_emotion", "1-5 ordinal"),
    ("emotion_anger",        "V241121",  "How angry about country direction",    "lifestyle_emotion", "1-5 ordinal"),
    ("emotion_pride",        "V241124",  "How proud about country direction",    "lifestyle_emotion", "1-5 ordinal"),
    ("emotion_worry",        "V241123",  "How worried about country direction",  "lifestyle_emotion", "1-5 ordinal"),
    ("emotion_happy",        "V241122",  "How happy about country direction",    "lifestyle_emotion", "1-5 ordinal"),

    # === TARGET ===
    ("vote_choice", "V242067",  "Presidential vote: 1=Harris(Dem),2=Trump(Rep),4-6=Other", "target", "nominal"),
    ("party_id",    "V241227x", "7-point party identification",                   "auxiliary", "1-7 ordinal"),

    # === WEIGHTS ===
    ("weight_pre",  "V240107a", "Full-sample pre-election weight",  "weight", "continuous"),
    ("weight_post", "V240108b", "Post-election respondent weight",  "weight", "continuous"),
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

    # Ideology self-placement (V241177) codes 99 = "haven't thought much about this";
    # treat as missing (matches the pre-election script). Other items use only negative
    # missing codes, already handled above.
    if "ideology" in df.columns:
        df.loc[df["ideology"] >= 90, "ideology"] = np.nan

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
