from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
HUMAN_DIR = BASE_DIR.parent
RAW_HUMAN_CSV = HUMAN_DIR / "test_pair_attempts.csv"
COMPARISON_CSV = HUMAN_DIR / "analysis" / "tables" / "public_eval_human_vs_models.csv"

FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def frame_to_text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    human = pd.read_csv(RAW_HUMAN_CSV)
    human = human[human["task_set"] == "Public Eval"].copy()
    human["solved"] = (human["correct_submissions"] > 0).astype(int)
    human["task_pair_id"] = human["task_ID"] + "__" + human["test_index"].astype(str)

    comparison = pd.read_csv(COMPARISON_CSV)
    comparison = comparison[comparison["attempts"] >= 8].copy()
    robust_pairs = set(comparison["task_pair_id"])
    human = human[human["task_pair_id"].isin(robust_pairs)].copy()
    return human, comparison


def split_half_stats(human: pd.DataFrame, n_sims: int = 1000, seed: int = 0) -> pd.DataFrame:
    sessions = np.array(sorted(human["session_ID"].unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for _ in range(n_sims):
        perm = rng.permutation(sessions)
        half_a = set(perm[: len(perm) // 2])

        part_a = human[human["session_ID"].isin(half_a)].groupby("task_pair_id").agg(rate=("solved", "mean"), n=("solved", "size"))
        part_b = human[~human["session_ID"].isin(half_a)].groupby("task_pair_id").agg(rate=("solved", "mean"), n=("solved", "size"))
        merged = part_a.join(part_b, lsuffix="_a", rsuffix="_b", how="inner")
        merged = merged[(merged["n_a"] >= 2) & (merged["n_b"] >= 2)]

        if len(merged) < 20:
            continue

        k = max(5, int(round(len(merged) * 0.2)))
        hard_a = set(merged.nsmallest(k, "rate_a").index)
        hard_b = set(merged.nsmallest(k, "rate_b").index)
        easy_a = set(merged.nlargest(k, "rate_a").index)
        easy_b = set(merged.nlargest(k, "rate_b").index)

        rows.append(
            {
                "pearson": merged["rate_a"].corr(merged["rate_b"]),
                "spearman": merged["rate_a"].corr(merged["rate_b"], method="spearman"),
                "hard_jaccard": len(hard_a & hard_b) / len(hard_a | hard_b),
                "easy_jaccard": len(easy_a & easy_b) / len(easy_a | easy_b),
                "n_items": len(merged),
            }
        )

    return pd.DataFrame(rows)


def permutation_pvalue(y: np.ndarray, x: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = np.corrcoef(y, x)[0, 1]
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = np.corrcoef(rng.permutation(y), x)[0, 1]
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def build_benchmark_table(split_halves: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    y = comparison["solve_rate"].to_numpy()
    k = max(5, int(round(len(comparison) * 0.2)))
    hard_h = set(comparison.nsmallest(k, "solve_rate")["task_pair_id"])
    easy_h = set(comparison.nlargest(k, "solve_rate")["task_pair_id"])

    human_median = {
        "series": "Human split-half median",
        "pearson": float(split_halves["pearson"].median()),
        "spearman": float(split_halves["spearman"].median()),
        "hard_jaccard": float(split_halves["hard_jaccard"].median()),
        "easy_jaccard": float(split_halves["easy_jaccard"].median()),
        "percentile_vs_split_half": 0.5,
        "perm_p": np.nan,
    }

    rows = [human_median]
    for idx, col in [
        ("Human vs average model", "lm_mean"),
        ("Human vs best single model", "lm_best_single_model"),
        ("Human vs per-pair oracle", "lm_best_across_models"),
    ]:
        x = comparison[col].to_numpy()
        hard_m = set(comparison.nsmallest(k, col)["task_pair_id"])
        easy_m = set(comparison.nlargest(k, col)["task_pair_id"])
        pearson = float(np.corrcoef(y, x)[0, 1])
        rows.append(
            {
                "series": idx,
                "pearson": pearson,
                "spearman": float(comparison["solve_rate"].corr(comparison[col], method="spearman")),
                "hard_jaccard": len(hard_h & hard_m) / len(hard_h | hard_m),
                "easy_jaccard": len(easy_h & easy_m) / len(easy_h | easy_m),
                "percentile_vs_split_half": float((split_halves["pearson"] <= pearson).mean()),
                "perm_p": permutation_pvalue(y, x),
            }
        )

    return pd.DataFrame(rows)


def build_gap_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in [
        ("Average model", "lm_mean"),
        ("Best single model", "lm_best_single_model"),
        ("Per-pair oracle", "lm_best_across_models"),
    ]:
        diff = comparison["solve_rate"] - comparison[col]
        rows.append(
            {
                "comparison": label,
                "human_mean": comparison["solve_rate"].mean(),
                "model_mean": comparison[col].mean(),
                "mean_gap": diff.mean(),
                "median_gap": diff.median(),
                "share_human_gt": (diff > 0).mean(),
                "share_equal": (diff == 0).mean(),
                "share_model_gt": (diff < 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_divergence_table(comparison: pd.DataFrame) -> pd.DataFrame:
    cols = ["task_pair_id", "attempts", "solve_rate", "lm_mean", "lm_best_single_model", "gap_vs_lm_mean", "gap_vs_best_single_model"]
    biggest_human = comparison.nlargest(10, "gap_vs_best_single_model")[cols].copy()
    biggest_human["direction"] = "Human > best single"
    biggest_model = comparison.nsmallest(10, "gap_vs_best_single_model")[cols].copy()
    biggest_model["direction"] = "Best single > human"
    return pd.concat([biggest_human, biggest_model], ignore_index=True)


def plot_split_half_vs_models(split_halves: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(split_halves["pearson"], bins=28, color="#4C78A8", alpha=0.75, ax=ax)
    ax.axvline(benchmark.loc[benchmark["series"] == "Human split-half median", "pearson"].iloc[0], color="#1F3552", linestyle="-", linewidth=2, label="Human split-half median")
    ax.axvline(benchmark.loc[benchmark["series"] == "Human vs average model", "pearson"].iloc[0], color="#F58518", linestyle="--", linewidth=2, label="Human vs average model")
    ax.axvline(benchmark.loc[benchmark["series"] == "Human vs best single model", "pearson"].iloc[0], color="#E45756", linestyle="--", linewidth=2, label="Human vs best single")
    ax.axvline(benchmark.loc[benchmark["series"] == "Human vs per-pair oracle", "pearson"].iloc[0], color="#72B7B2", linestyle="--", linewidth=2, label="Human vs oracle")
    ax.set_title("How human-vs-model item alignment compares to human split-half reliability")
    ax.set_xlabel("Pearson correlation of Public Eval item solve rates")
    ax.set_ylabel("Split-half simulations")
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, "Split-halves use Public Eval task pairs with >=8 human attempts overall and require >=2 attempts per half.", fontsize=10)
    fig.savefig(FIGURES_DIR / "fig01_split_half_vs_models.png")
    plt.close(fig)


def write_report(split_halves: pd.DataFrame, benchmark: pd.DataFrame, gaps: pd.DataFrame, divergences: pd.DataFrame) -> None:
    split_median = split_halves["pearson"].median()
    split_lo = split_halves["pearson"].quantile(0.025)
    split_hi = split_halves["pearson"].quantile(0.975)
    avg_row = benchmark.loc[benchmark["series"] == "Human vs average model"].iloc[0]
    best_row = benchmark.loc[benchmark["series"] == "Human vs best single model"].iloc[0]
    oracle_row = benchmark.loc[benchmark["series"] == "Human vs per-pair oracle"].iloc[0]
    best_gap = gaps.loc[gaps["comparison"] == "Best single model"].iloc[0]

    lines = [
        "# Creme Thesis Checks",
        "",
        "This folder runs a few direct, low-overhead checks of the Cremieux-style thesis against the ARC human testing data we already have locally.",
        "",
        "The core question is simple: do humans and models find the same ARC items easy and hard, and how strong is that alignment compared with human-to-human alignment?",
        "",
        "## What I tested",
        "",
        "- I restricted to Public Eval task pairs with at least 8 human attempts overall.",
        "- I repeatedly split human sessions in half and correlated half-A vs half-B item solve rates. That gives a noisy but useful human-to-human baseline.",
        "- I compared the same human item solve rates to three model profiles from the existing local ARC-AGI-2 model outputs: average model, best single model, and a per-pair oracle.",
        "",
        "## Main result",
        "",
        f"- Human split-half median Pearson correlation is {split_median:.3f} with a 95% simulation interval of [{split_lo:.3f}, {split_hi:.3f}].",
        f"- Human vs average-model item difficulty correlation is {avg_row['pearson']:.3f}. That lands near the middle of the human split-half distribution ({100 * avg_row['percentile_vs_split_half']:.1f} percentile), so it is not random with respect to humans here.",
        f"- Human vs best-single-model correlation is only {best_row['pearson']:.3f}. That lands near the bottom of the human split-half distribution ({100 * best_row['percentile_vs_split_half']:.1f} percentile), which is meaningfully weaker than human-to-human alignment.",
        f"- Human vs oracle correlation is {oracle_row['pearson']:.3f}, still below the human split-half median.",
        "",
        "## Synthesis",
        "",
        "- This ARC evidence does not support the strongest version of his claim, namely that model item performance is basically random with respect to humans. The average-model profile is clearly related to human difficulty.",
        "- But it does support a milder and more defensible version: a single strong model still tracks human difficulty worse than one human subsample tracks another, so score comparability is not something we should assume.",
        "- The average-model result should be interpreted cautiously because it is an ensemble-style aggregate across many systems, not one agent.",
        "- Sparse human coverage matters a lot here. Because most items only have around 8 to 15 human attempts, the human split-half ceiling is itself noisy and fairly low.",
        "",
        "## Benchmark table",
        "",
        frame_to_text_table(benchmark.round(3)),
        "",
        "## Gap table",
        "",
        frame_to_text_table(gaps.round(3)),
        "",
        "## Biggest divergences vs the best single model",
        "",
        frame_to_text_table(divergences.round(3)),
        "",
    ]

    (BASE_DIR / "creme_thesis_synthesis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dirs()

    human, comparison = load_inputs()
    split_halves = split_half_stats(human, n_sims=1000, seed=0)
    benchmark = build_benchmark_table(split_halves, comparison)
    gaps = build_gap_table(comparison)
    divergences = build_divergence_table(comparison)

    split_halves.to_csv(TABLES_DIR / "split_half_simulations.csv", index=False)
    benchmark.to_csv(TABLES_DIR / "correlation_benchmarks.csv", index=False)
    gaps.to_csv(TABLES_DIR / "gap_summary.csv", index=False)
    divergences.to_csv(TABLES_DIR / "divergent_items.csv", index=False)

    plot_split_half_vs_models(split_halves, benchmark)
    write_report(split_halves, benchmark, gaps, divergences)

    print(f"Done. Outputs written to {BASE_DIR}")


if __name__ == "__main__":
    main()
