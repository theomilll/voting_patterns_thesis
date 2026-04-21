"""
Generate all thesis figures from pre-computed CSV results.
Output: figures/ directory with PNG + PDF for each figure.

Figures:
  0. Pipeline architecture schematic (inline, unnumbered)
  1. Stage 1 issue predictability (Cleveland dot plot)
  2. Stage 2 approach comparison (horizontal bars + CIs + significance)
  3. Cumulative ablation curve (+ party ID reference)
  4. SHAP feature importance (top 15, color-coded)
  5. Lifestyle vs conditions (two-panel: lift + sub-components)
  A1. SHAP beeswarm (appendix placeholder — needs raw SHAP values)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Grayscale-safe palette
C_DEMO = "#4878A8"       # steel blue → dark gray in print
C_ISSUE = "#E45756"      # muted red → medium gray
C_LIFESTYLE = "#54A24B"  # sage green
C_CONDITIONS = "#F58518" # amber
C_NEUTRAL = "#888888"

# Domain colors for Stage 1 dot plot (11 domains)
DOMAIN_COLORS = {
    "Economy":       "#E45756",
    "Immigration":   "#4878A8",
    "Guns":          "#54A24B",
    "Climate":       "#F58518",
    "Abortion":      "#B07AA1",
    "Race":          "#FF9DA7",
    "Populism":      "#9D755D",
    "Gender":        "#76B7B2",
    "Trade":         "#BAB0AC",
    "Diversity":     "#EECA3B",
    "Healthcare":    "#AF7AA1",
}

ISSUE_DOMAINS = {
    "econ_opportunity": "Economy", "econ_mobility": "Economy",
    "econ_mobility_summary": "Economy", "tax_millionaires": "Economy",
    "deficit_importance": "Economy", "min_wage": "Economy",
    "immig_level": "Immigration", "immig_policy1": "Immigration",
    "immig_policy2": "Immigration", "immig_policy3": "Immigration",
    "gun_difficulty": "Guns", "gun_background": "Guns",
    "gun_assault_ban": "Guns",
    "climate_impact": "Climate", "climate_regulation": "Climate",
    "abortion_access": "Abortion", "abortion_scale": "Abortion",
    "abortion_detail": "Abortion",
    "racial_selfhelp": "Race", "racial_slavery": "Race",
    "racial_deserve": "Race", "racial_tryharder": "Race",
    "populism_insiders": "Populism", "populism_rich": "Populism",
    "fairshare_summary": "Populism",
    "gender_trad_summary": "Gender", "gender_equality": "Gender",
    "gender_discrim": "Gender",
    "trade_summary": "Trade",
    "diversity_summary": "Diversity",
    "healthcare_spending": "Healthcare",
}

# Pretty names for issues
ISSUE_LABELS = {
    "econ_opportunity": "Economic Opportunity",
    "econ_mobility": "Economic Mobility",
    "econ_mobility_summary": "Mobility (binary)",
    "tax_millionaires": "Tax Millionaires",
    "deficit_importance": "Deficit Importance",
    "min_wage": "Minimum Wage",
    "immig_level": "Immigration Level",
    "immig_policy1": "Immigration Policy 1",
    "immig_policy2": "Immigration Policy 2",
    "immig_policy3": "Immigration Policy 3",
    "gun_difficulty": "Gun Purchase Difficulty",
    "gun_background": "Gun Background Checks",
    "gun_assault_ban": "Assault Weapon Ban",
    "climate_impact": "Climate Impact",
    "climate_regulation": "Climate Regulation",
    "abortion_access": "Abortion Access",
    "abortion_scale": "Abortion Scale",
    "abortion_detail": "Abortion (detailed)",
    "racial_selfhelp": "Racial Self-help",
    "racial_slavery": "Racial Slavery Legacy",
    "racial_deserve": "Racial Deserve",
    "racial_tryharder": "Racial Try Harder",
    "populism_insiders": "Populism: Insiders",
    "populism_rich": "Populism: Rich",
    "fairshare_summary": "Fair Share",
    "gender_trad_summary": "Gender Trad. Roles",
    "gender_equality": "Gender Equality",
    "gender_discrim": "Gender Discrimination",
    "trade_summary": "Trade Policy",
    "diversity_summary": "Diversity Value",
    "healthcare_spending": "Healthcare Spending",
}

RESULTS = Path("results")
FIGS = Path("figures")


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png")
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}")


# ── Figure 0: Pipeline Schematic ────────────────────────────────────────────

def fig0_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box_kw = dict(
        boxstyle="round,pad=0.4", edgecolor="#333", linewidth=1.5
    )

    # Boxes
    boxes = {
        "demo":      (1.0, 3.8, C_DEMO,       "Demographics\n(10 features)"),
        "stage1":    (3.5, 3.8, "#DDD",        "Stage 1\nRandom Forest\n(per issue)"),
        "pred":      (6.0, 3.8, "#FFDDDD",     "Predicted\nIssue Opinions"),
        "obs":       (6.0, 2.0, "#DDFFDD",     "Observed\nIssue Opinions"),
        "stage2":    (8.0, 2.9, "#DDD",        "Stage 2\nXGBoost / RF"),
        "vote":      (9.5, 2.9, "#FFFFCC",     "Vote\nHarris vs Trump"),
    }

    for key, (x, y, color, text) in boxes.items():
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                fontweight="bold" if "Stage" in text else "normal",
                bbox=dict(facecolor=color, **box_kw))

    # Arrows
    arrow_kw = dict(
        arrowstyle="->", color="#333", linewidth=1.5,
        connectionstyle="arc3,rad=0"
    )
    arrows = [
        ((1.8, 3.8), (2.7, 3.8)),   # demo → stage1
        ((4.3, 3.8), (5.1, 3.8)),   # stage1 → predicted
        ((6.6, 3.5), (7.3, 3.1)),   # predicted → stage2
        ((6.6, 2.3), (7.3, 2.7)),   # observed → stage2
        ((8.6, 2.9), (9.0, 2.9)),   # stage2 → vote
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=arrow_kw)

    # Demo → stage2 (dashed, curved — demographics also feed Stage 2 directly)
    ax.annotate("", xy=(7.4, 2.5), xytext=(1.0, 3.2),
                arrowprops=dict(arrowstyle="->", color=C_DEMO,
                                linewidth=1.5, linestyle="--",
                                connectionstyle="arc3,rad=-0.3"))

    # Labels on paths
    ax.text(2.5, 4.3, "Approach B\n(predicted)", fontsize=9.5,
            ha="center", color="#666", style="italic")
    ax.text(6.8, 1.6, "Approach A\n(observed)", fontsize=9.5,
            ha="center", color="#666", style="italic")

    # Nested CV note
    ax.text(3.5, 1.0,
            "Nested CV: Stage 1 retrained inside each outer fold\n"
            "to prevent information leakage from predicted issues",
            fontsize=9.5, ha="center", color="#666",
            bbox=dict(facecolor="#F8F8F8", edgecolor="#CCC",
                      boxstyle="round,pad=0.3"))

    fig.suptitle("Two-Stage Pipeline Architecture", fontsize=14,
                 fontweight="bold", y=0.98)
    save(fig, "fig0_pipeline_schematic")


# ── Figure 1: Stage 1 Issue Predictability ───────────────────────────────────

def fig1_stage1_dotplot():
    df = pd.read_csv(RESULTS / "stage1_results.csv")
    rf = df[df["model"] == "RandomForest"].copy()
    rf["domain"] = rf["target"].map(ISSUE_DOMAINS)
    rf["label"] = rf["target"].map(ISSUE_LABELS)
    rf = rf.sort_values("balanced_accuracy")

    fig, ax = plt.subplots(figsize=(8, 9))

    for i, (_, row) in enumerate(rf.iterrows()):
        color = DOMAIN_COLORS.get(row["domain"], C_NEUTRAL)
        ax.plot(row["balanced_accuracy"], i, "o", color=color,
                markersize=8, markeredgecolor="white", markeredgewidth=0.5)
        # std as horizontal error bar
        ax.plot(
            [row["balanced_accuracy"] - row["balanced_accuracy_std"],
             row["balanced_accuracy"] + row["balanced_accuracy_std"]],
            [i, i], color=color, linewidth=1.5, alpha=0.5
        )

    ax.set_yticks(range(len(rf)))
    ax.set_yticklabels(rf["label"].values, fontsize=9)
    # Reference: BA=0.5 is not chance for multiclass, but marks "halfway"
    # between random (1/K) and perfect (1.0) — useful as a visual anchor
    ax.axvline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Balanced Accuracy (Random Forest, 5-fold CV)")
    ax.set_title("Stage 1: Demographic Predictability of Issue Opinions")
    ax.set_xlim(0.30, 0.85)

    # Domain legend
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=8, label=d)
               for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2,
              framealpha=0.9)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig1_stage1_predictability")


# ── Figure 2: Stage 2 Approach Comparison ────────────────────────────────────

def fig2_stage2_comparison():
    df = pd.read_csv(RESULTS / "stage2_results.csv")
    sig = pd.read_csv(RESULTS / "significance_tests.csv")

    # Best model per approach
    best = df.loc[df.groupby("approach")["balanced_accuracy"].idxmax()].copy()
    best = best.sort_values("balanced_accuracy")

    approach_labels = {
        "Cs_demo_strict":      "Cs: Demographics only",
        "F_issues_only":       "F: Issues only",
        "C_demo_with_ideology":"C: Demo + Ideology",
        "B_predicted_issues":  "B: Predicted Issues + Demo",
        "A_actual_issues":     "A: Observed Issues + Demo",
        "D_party_only":        "D: Party ID only",
        "E_party_demo":        "E: Party + Demo",
        "G_full_with_party":   "G: Party + Issues + Demo",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))

    approaches_list = list(best["approach"])
    y_pos = list(range(len(best)))

    # Dot-and-whisker plot (not truncated bars)
    for i, (_, row) in enumerate(best.iterrows()):
        a = row["approach"]
        if "party" in a.lower():
            color = "#9467BD"
        elif "issue" in a.lower() or a == "F_issues_only":
            color = C_ISSUE
        elif "demo" in a.lower():
            color = C_DEMO
        else:
            color = C_NEUTRAL

        ba = row["balanced_accuracy"]
        std = row["balanced_accuracy_std"]
        # Whisker
        ax.plot([ba - std, ba + std], [i, i], color=color,
                linewidth=2.5, solid_capstyle="round")
        # Dot
        ax.plot(ba, i, "o", color=color, markersize=9,
                markeredgecolor="white", markeredgewidth=1, zorder=5)
        # Value label
        ax.text(ba + std + 0.003, i, f"{ba:.3f}",
                va="center", fontsize=9, color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([approach_labels.get(a, a) for a in approaches_list],
                       fontsize=10)
    ax.set_xlabel("Balanced Accuracy (best model, 5-fold CV ± std)")
    ax.set_title("Stage 2: Vote Prediction by Modeling Approach")
    ax.set_xlim(0.84, 0.99)

    # Significance brackets — read p-values from CSV
    def get_y(approach):
        return approaches_list.index(approach)

    def get_sig_label(a1, a2):
        row = sig[((sig["approach_1"] == a1) & (sig["approach_2"] == a2)) |
                  ((sig["approach_1"] == a2) & (sig["approach_2"] == a1))]
        if len(row) == 0:
            return None
        p = row["bootstrap_p"].values[0]
        stars = row["significant"].values[0]
        if pd.isna(stars) or stars == "":
            return f"p={p:.3f} n.s."
        return f"p={'<0.001' if p < 0.001 else f'{p:.3f}'} {stars}"

    key_comparisons = [
        ("A_actual_issues", "Cs_demo_strict"),
        ("A_actual_issues", "D_party_only"),
        ("G_full_with_party", "A_actual_issues"),
    ]

    x_bracket = 0.970
    for a1, a2 in key_comparisons:
        label = get_sig_label(a1, a2)
        if label is None:
            continue
        y1, y2 = get_y(a1), get_y(a2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        # Vertical line
        ax.plot([x_bracket, x_bracket], [ymin, ymax],
                color="#555", linewidth=0.8)
        # Horizontal ticks
        ax.plot([x_bracket - 0.003, x_bracket], [ymin, ymin],
                color="#555", linewidth=0.8)
        ax.plot([x_bracket - 0.003, x_bracket], [ymax, ymax],
                color="#555", linewidth=0.8)
        ax.text(x_bracket + 0.002, (ymin + ymax) / 2, label,
                fontsize=8, va="center")
        x_bracket += 0.018

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DEMO,
               markersize=8, label="Demographics-based"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ISSUE,
               markersize=8, label="Issue-based"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9467BD",
               markersize=8, label="Party-based"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig2_stage2_comparison")


# ── Figure 3: Cumulative Ablation Curve ──────────────────────────────────────

def fig3_ablation_curve():
    df = pd.read_csv(RESULTS / "ablation_cumulative.csv")
    s2 = pd.read_csv(RESULTS / "stage2_results.csv")

    # Read benchmark values from stage2 results
    best_per_approach = s2.loc[s2.groupby("approach")["balanced_accuracy"].idxmax()]
    party_ba = best_per_approach.loc[
        best_per_approach["approach"] == "D_party_only", "balanced_accuracy"
    ].values[0]
    demo_ba = best_per_approach.loc[
        best_per_approach["approach"] == "Cs_demo_strict", "balanced_accuracy"
    ].values[0]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df["n_issues"], df["balanced_accuracy"], "o-",
            color=C_ISSUE, markersize=5, linewidth=2, label="Issues + Demo")

    # Party ID reference line
    ax.axhline(party_ba, color="#9467BD", linestyle="--", linewidth=1.5,
               alpha=0.8)
    ax.text(28, party_ba + 0.001, f"Party ID only ({party_ba:.3f})",
            fontsize=9, color="#9467BD", ha="right", va="bottom")

    # 5-issue annotation
    ba_5 = df.loc[df["n_issues"] == 5, "balanced_accuracy"].values[0]
    ax.annotate(f"5 issues → {ba_5:.3f}",
                xy=(5, ba_5), xytext=(8, ba_5 - 0.012),
                fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555", linewidth=0.8))

    # 1-issue annotation
    ba_1 = df.loc[df["n_issues"] == 1, "balanced_accuracy"].values[0]
    issue_1_name = df.loc[df["n_issues"] == 1, "last_added"].values[0]
    ax.annotate(f"{issue_1_name} → {ba_1:.3f}",
                xy=(1, ba_1), xytext=(4, ba_1 - 0.010),
                fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555", linewidth=0.8))

    # Demo-only baseline
    ax.axhline(demo_ba, color=C_DEMO, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(28, demo_ba - 0.003, f"Demo only ({demo_ba:.3f})",
            fontsize=9, color=C_DEMO, ha="right", va="top")

    ax.set_xlabel("Number of Issues Added (ranked by SHAP importance)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Cumulative Issue Addition: Diminishing Returns")
    ax.set_xlim(0, 32)
    ax.set_ylim(0.87, 0.945)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, "fig3_ablation_curve")


# ── Figure 4: SHAP Feature Importance ────────────────────────────────────────

def fig4_shap_importance():
    df = pd.read_csv(RESULTS / "shap_importance.csv")
    df = df.sort_values("mean_abs_shap", ascending=False)
    top = df.head(15).copy()
    top = top.iloc[::-1]  # reverse for horizontal bar (top at top)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [C_DEMO if cat == "demographic" else C_ISSUE
              for cat in top["category"]]

    # Pretty labels
    feature_labels = {
        "age_group": "Age Group", "sex": "Sex", "ideology": "Ideology",
        "climate_impact": "Climate Impact", "gun_difficulty": "Gun Difficulty",
        "racial_tryharder": "Racial: Try Harder", "education": "Education",
        "gun_assault_ban": "Assault Weapon Ban",
        "climate_regulation": "Climate Regulation",
        "racial_slavery": "Racial: Slavery Legacy",
        "abortion_detail": "Abortion (detailed)",
        "racial_selfhelp": "Racial: Self-help",
        "immig_policy1": "Immigration Policy 1",
        "racial_deserve": "Racial: Deserve",
        "church_attendance": "Church Attendance",
    }
    labels = [feature_labels.get(f, f) for f in top["feature"]]

    ax.barh(range(len(top)), top["mean_abs_shap"].values,
            color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title("Feature Importance: Approach A (Observed Issues + Demo)")

    legend_handles = [
        mpatches.Patch(color=C_DEMO, label="Demographic"),
        mpatches.Patch(color=C_ISSUE, label="Issue"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig4_shap_importance")


# ── Figure 5: Lifestyle vs Conditions ────────────────────────────────────────

def fig5_lifestyle_conditions():
    df = pd.read_csv(RESULTS / "lifestyle_vs_conditions.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # ── Left panel: Asymmetric lift ──
    # Conditions → +Lifestyle
    cond_ba = df.loc[df["config"] == "conditions_only", "balanced_accuracy"].values[0]
    combined_ba = df.loc[df["config"] == "media_conditions_combined", "balanced_accuracy"].values[0]
    life_ba = df.loc[df["config"] == "all_media_lifestyle", "balanced_accuracy"].values[0]

    lift_life_over_cond = combined_ba - cond_ba   # +0.21
    lift_cond_over_life = combined_ba - life_ba    # +0.025

    y_positions = [1, 0]
    base_bars = [cond_ba, life_ba]
    lift_bars = [lift_life_over_cond, lift_cond_over_life]
    labels = ["Conditions\n→ +Lifestyle", "Lifestyle\n→ +Conditions"]
    lift_colors = [C_LIFESTYLE, C_CONDITIONS]

    for i, (y, base, lift, lbl, lc) in enumerate(
            zip(y_positions, base_bars, lift_bars, labels, lift_colors)):
        ax1.barh(y, base, height=0.5, color="#DDD", edgecolor="white")
        ax1.barh(y, lift, left=base, height=0.5, color=lc, edgecolor="white")
        # Lift label
        ax1.text(base + lift + 0.01, y,
                 f"Δ = +{lift:.3f}",
                 va="center", fontsize=10, fontweight="bold", color=lc)

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Balanced Accuracy")
    ax1.set_title("Asymmetric Lift")
    ax1.set_xlim(0, 1.0)
    ax1.axvline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)

    legend_handles = [
        mpatches.Patch(color="#DDD", label="Base model"),
        mpatches.Patch(color=C_LIFESTYLE, label="Lifestyle lift"),
        mpatches.Patch(color=C_CONDITIONS, label="Conditions lift"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # ── Right panel: Sub-component accuracy ──
    sub_configs = [
        ("conditions_only",          "Conditions",       C_CONDITIONS),
        ("entertainment_only",       "Entertainment\nMedia", "#BAB0AC"),
        ("political_media_only",     "Political\nMedia",    "#9467BD"),
        ("engagement_attitudes_only","Engagement\nAttitudes", C_LIFESTYLE),
        ("all_media_lifestyle",      "All\nLifestyle",      C_LIFESTYLE),
        ("media_conditions_combined","Combined",            "#333333"),
    ]

    bars_data = []
    for config, label, color in sub_configs:
        row = df[df["config"] == config]
        if len(row) > 0:
            bars_data.append((
                label, row["balanced_accuracy"].values[0],
                row["balanced_accuracy_std"].values[0] if pd.notna(row["balanced_accuracy_std"].values[0]) else 0,
                color
            ))

    labels_r = [b[0] for b in bars_data]
    vals = [b[1] for b in bars_data]
    stds = [b[2] for b in bars_data]
    colors_r = [b[3] for b in bars_data]

    x_pos = list(range(len(bars_data)))

    # Lollipop plot (stem + dot) to avoid truncated-bar problem
    for i, (v, s, c) in enumerate(zip(vals, stds, colors_r)):
        ax2.plot([i, i], [0, v], color=c, linewidth=3, solid_capstyle="round")
        ax2.plot([i, i], [v - s, v + s], color=c, linewidth=6, alpha=0.3,
                 solid_capstyle="round")
        ax2.plot(i, v, "o", color=c, markersize=10,
                 markeredgecolor="white", markeredgewidth=1, zorder=5)
        ax2.text(i, v + s + 0.015, f"{v:.3f}", ha="center", fontsize=9,
                 fontweight="bold")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_r, fontsize=9, ha="center")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title("Sub-component Decomposition")
    ax2.set_ylim(0, 0.9)
    ax2.axhline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Lifestyle vs. Living Conditions as Vote Predictors",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_lifestyle_vs_conditions")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    print("Generating figures...")
    fig0_pipeline()
    fig1_stage1_dotplot()
    fig2_stage2_comparison()
    fig3_ablation_curve()
    fig4_shap_importance()
    fig5_lifestyle_conditions()
    print("Done. All figures in figures/")
