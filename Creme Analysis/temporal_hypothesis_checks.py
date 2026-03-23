from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
HUMAN_DIR = BASE_DIR.parent
MAIN_ANALYSIS_DIR = HUMAN_DIR / "analysis"
COMPARISON_CSV = MAIN_ANALYSIS_DIR / "tables" / "public_eval_human_vs_models.csv"

TRUTH_DIR = Path("C:/Users/cooki/Desktop/ARC-AGI/Psychometric Analysis/data/ARC-AGI-2/data/evaluation")
PREDS_DIR = Path("C:/Users/cooki/Desktop/ARC-AGI/Psychometric Analysis/data/arc_agi_v2_public_eval")

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


def normalize_grid(grid: object) -> str:
    if not isinstance(grid, list) or not grid:
        return "EMPTY"
    return ",".join(str(cell) for row in grid for cell in row)


def family_for_model(model: str) -> str:
    if model.startswith("gpt-5-1") or model.startswith("gpt-5-2") or model.startswith("gpt-5-pro") or model.startswith("gpt-4-1") or model.startswith("gpt-4-5"):
        return "OpenAI GPT"
    if model.startswith("claude-opus"):
        return "Claude Opus"
    if model.startswith("claude-sonnet"):
        return "Claude Sonnet"
    if model.startswith("claude-haiku"):
        return "Claude Haiku"
    if model.startswith("gemini-3-flash"):
        return "Gemini Flash"
    if model.startswith("gemini-3-pro") or model.startswith("gemini-3-deep"):
        return "Gemini Pro/Deep"
    return "Other"


def short_label(model: str) -> str:
    replacements = {
        "gpt-5-1-2025-11-13-thinking-low": "5.1 low",
        "gpt-5-1-2025-11-13-thinking-medium": "5.1 med",
        "gpt-5-1-2025-11-13-thinking-high": "5.1 high",
        "gpt-5-2-2025-12-11-thinking-low": "5.2 low",
        "gpt-5-2-2025-12-11-thinking-medium": "5.2 med",
        "gpt-5-2-2025-12-11-thinking-high": "5.2 high",
        "gpt-5-2-2025-12-11-thinking-xhigh": "5.2 xhigh",
        "claude-opus-4-5-20251101-thinking-8k": "Opus 8k",
        "claude-opus-4-5-20251101-thinking-16k": "Opus 16k",
        "claude-opus-4-5-20251101-thinking-32k": "Opus 32k",
        "claude-opus-4-5-20251101-thinking-64k": "Opus 64k",
        "gemini-3-flash-preview-thinking-minimal": "Flash min",
        "gemini-3-flash-preview-thinking-low": "Flash low",
        "gemini-3-flash-preview-thinking-medium": "Flash med",
        "gemini-3-flash-preview-thinking-high": "Flash high",
    }
    return replacements.get(model, model)


def load_robust_comparison() -> pd.DataFrame:
    comparison = pd.read_csv(COMPARISON_CSV)
    return comparison[comparison["attempts"] >= 8].copy()


def build_model_matrix(pair_ids: list[str]) -> pd.DataFrame:
    truth_cache = {path.name: json.loads(path.read_text()) for path in TRUTH_DIR.glob("*.json")}
    truth_outputs: dict[str, str] = {}
    pair_set = set(pair_ids)
    for path in TRUTH_DIR.glob("*.json"):
        obj = truth_cache[path.name]
        for idx, pair in enumerate(obj.get("test", [])):
            pair_id = f"{path.stem}__{idx}"
            if pair_id in pair_set:
                truth_outputs[pair_id] = normalize_grid(pair["output"])

    rows: dict[str, dict[str, int]] = {}
    for model_dir in sorted(PREDS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        row = {pair_id: 0 for pair_id in pair_ids}
        for pred_path in model_dir.glob("*.json"):
            truth_obj = truth_cache.get(pred_path.name)
            if truth_obj is None:
                continue
            pred_obj = json.loads(pred_path.read_text())
            for idx, pair in enumerate(truth_obj.get("test", [])):
                pair_id = f"{pred_path.stem}__{idx}"
                if pair_id not in row:
                    continue
                pred_entry = None
                for candidate in pred_obj:
                    if candidate.get("metadata", {}).get("pair_index") == idx:
                        pred_entry = candidate
                        break
                if pred_entry is None and idx < len(pred_obj):
                    pred_entry = pred_obj[idx]
                answer = None
                if pred_entry:
                    answer = (pred_entry.get("attempt_1") or {}).get("answer")
                    if not answer:
                        answer = (pred_entry.get("attempt_2") or {}).get("answer")
                row[pair_id] = int(normalize_grid(answer) == truth_outputs[pair_id])
        rows[model_dir.name] = row

    return pd.DataFrame.from_dict(rows, orient="index")[pair_ids]


def bootstrap_delta(y: np.ndarray, old: np.ndarray, new: np.ndarray, seed: int = 0, n_boot: int = 4000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        co = np.corrcoef(y[idx], old[idx])[0, 1]
        cn = np.corrcoef(y[idx], new[idx])[0, 1]
        deltas[i] = cn - co
    q = np.quantile(deltas, [0.025, 0.5, 0.975])
    return float(q[0]), float(q[1]), float(q[2])


def build_model_alignment_summary(comparison: pd.DataFrame, model_matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    human = comparison.set_index("task_pair_id")["solve_rate"]
    rows = []
    for model in model_matrix.index:
        profile = model_matrix.loc[model]
        if profile.nunique() < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = float(profile.corr(human))
            spearman = float(profile.corr(human, method="spearman"))
        rows.append(
            {
                "model": model,
                "family": family_for_model(model),
                "label": short_label(model),
                "pair_accuracy": float(profile.mean()),
                "human_pearson": pearson,
                "human_spearman": spearman,
            }
        )

    summary = pd.DataFrame(rows).sort_values("pair_accuracy", ascending=False)
    clean = summary.dropna(subset=["human_pearson"])
    stats = {
        "accuracy_vs_human_corr_pearson": float(clean["pair_accuracy"].corr(clean["human_pearson"])),
        "accuracy_vs_human_corr_spearman": float(clean["pair_accuracy"].corr(clean["human_pearson"], method="spearman")),
    }
    return summary, stats


def build_family_delta_summary(comparison: pd.DataFrame, model_matrix: pd.DataFrame) -> pd.DataFrame:
    human = comparison.set_index("task_pair_id")["solve_rate"]
    human_values = human.to_numpy()
    pairs = [
        ("gpt-5-1-2025-11-13-thinking-low", "gpt-5-2-2025-12-11-thinking-low", "GPT 5.1 low -> 5.2 low"),
        ("gpt-5-1-2025-11-13-thinking-medium", "gpt-5-2-2025-12-11-thinking-medium", "GPT 5.1 med -> 5.2 med"),
        ("gpt-5-1-2025-11-13-thinking-high", "gpt-5-2-2025-12-11-thinking-high", "GPT 5.1 high -> 5.2 high"),
        ("gpt-5-2-2025-12-11-thinking-high", "gpt-5-2-2025-12-11-thinking-xhigh", "GPT 5.2 high -> xhigh"),
        ("claude-opus-4-5-20251101-thinking-16k", "claude-opus-4-5-20251101-thinking-64k", "Opus 4.5 16k -> 64k"),
    ]

    rows = []
    for old, new, label in pairs:
        if old not in model_matrix.index or new not in model_matrix.index:
            continue
        old_vec = model_matrix.loc[old].to_numpy()
        new_vec = model_matrix.loc[new].to_numpy()
        old_corr = float(np.corrcoef(human_values, old_vec)[0, 1]) if np.unique(old_vec).size > 1 else np.nan
        new_corr = float(np.corrcoef(human_values, new_vec)[0, 1]) if np.unique(new_vec).size > 1 else np.nan
        newly = human[(new_vec == 1) & (old_vec == 0)]
        still_fail = human[(new_vec == 0) & (old_vec == 0)]
        lo, mid, hi = bootstrap_delta(human_values, old_vec, new_vec) if np.unique(old_vec).size > 1 else (np.nan, np.nan, np.nan)
        rows.append(
            {
                "comparison": label,
                "old_model": old,
                "new_model": new,
                "old_accuracy": float(old_vec.mean()),
                "new_accuracy": float(new_vec.mean()),
                "delta_accuracy": float(new_vec.mean() - old_vec.mean()),
                "old_human_corr": old_corr,
                "new_human_corr": new_corr,
                "delta_human_corr": new_corr - old_corr if pd.notna(old_corr) and pd.notna(new_corr) else np.nan,
                "delta_human_corr_ci_lo": lo,
                "delta_human_corr_ci_mid": mid,
                "delta_human_corr_ci_hi": hi,
                "old_new_profile_corr": float(np.corrcoef(old_vec, new_vec)[0, 1]) if np.unique(old_vec).size > 1 and np.unique(new_vec).size > 1 else np.nan,
                "newly_solved_n": int(((new_vec == 1) & (old_vec == 0)).sum()),
                "newly_solved_human_mean": float(newly.mean()) if len(newly) else np.nan,
                "still_failed_n": int(((new_vec == 0) & (old_vec == 0)).sum()),
                "still_failed_human_mean": float(still_fail.mean()) if len(still_fail) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_family_progressions(summary: pd.DataFrame) -> None:
    families = ["OpenAI GPT", "Claude Opus", "Gemini Flash"]
    data = summary[summary["family"].isin(families)].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"OpenAI GPT": "#4C78A8", "Claude Opus": "#E45756", "Gemini Flash": "#72B7B2"}

    for family in families:
        fam = data[data["family"] == family].sort_values("pair_accuracy")
        ax.plot(fam["pair_accuracy"], fam["human_pearson"], marker="o", linewidth=2, color=palette[family], label=family)
        for _, row in fam.iterrows():
            ax.text(row["pair_accuracy"], row["human_pearson"], row["label"], fontsize=8, color=palette[family])

    ax.set_title("Model progressions: accuracy vs human item-alignment")
    ax.set_xlabel("Model pair accuracy on robust Public Eval human-overlap items")
    ax.set_ylabel("Pearson correlation with human item solve rates")
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "fig02_family_progressions.png")
    plt.close(fig)


def write_creme_note(summary: pd.DataFrame, trend_stats: dict, deltas: pd.DataFrame) -> None:
    gpt_rows = deltas[deltas["comparison"].str.contains("GPT")]
    opus_row = deltas[deltas["comparison"] == "Opus 4.5 16k -> 64k"].iloc[0]

    lines = [
        "# Temporal / Family Hypothesis Checks",
        "",
        "This is a quick look for the kind of temporal or family-pattern hypothesis you asked about: are newer models just stronger versions of the same profile, or do they become more human-like in their item pattern?",
        "",
        "## Constraints",
        "",
        "- We do not have `Opus 4` in the local ARC-AGI-2 Public Eval folder, only `Opus 4.5`, so there is no direct `Opus 4` vs `Opus 4.5` time comparison to run here.",
        "- We do have useful temporal comparisons inside the GPT family (`gpt-5.1` to `gpt-5.2`) and compute-budget comparisons inside `Claude Opus 4.5` and `Gemini Flash`.",
        "",
        "## Keepers",
        "",
        f"- Across the whole non-degenerate model panel, higher pair accuracy tends to come with higher human-alignment: correlation between model accuracy and human-correlation is {trend_stats['accuracy_vs_human_corr_pearson']:.3f} Pearson and {trend_stats['accuracy_vs_human_corr_spearman']:.3f} Spearman.",
        f"- In GPT, `5.2` is not just more accurate than `5.1`; it is also more human-aligned on the item pattern. The cleanest cases are `5.1 med -> 5.2 med` ({gpt_rows.iloc[1]['old_accuracy']:.3f} -> {gpt_rows.iloc[1]['new_accuracy']:.3f} accuracy; {gpt_rows.iloc[1]['old_human_corr']:.3f} -> {gpt_rows.iloc[1]['new_human_corr']:.3f} human-correlation) and `5.1 high -> 5.2 high` ({gpt_rows.iloc[2]['old_accuracy']:.3f} -> {gpt_rows.iloc[2]['new_accuracy']:.3f}; {gpt_rows.iloc[2]['old_human_corr']:.3f} -> {gpt_rows.iloc[2]['new_human_corr']:.3f}).",
        f"- But the gains mostly look like `more of the same but better`. For `5.1 med -> 5.2 med`, newly solved items have mean human solve rate {gpt_rows.iloc[1]['newly_solved_human_mean']:.3f}, while items the new model still fails average {gpt_rows.iloc[1]['still_failed_human_mean']:.3f}. The same pattern holds for `5.1 high -> 5.2 high` ({gpt_rows.iloc[2]['newly_solved_human_mean']:.3f} vs {gpt_rows.iloc[2]['still_failed_human_mean']:.3f}).",
        f"- `Claude Opus 4.5` is interesting because it already becomes very human-aligned at moderate budgets: `16k` reaches human-correlation {opus_row['old_human_corr']:.3f} and `64k` is {opus_row['new_human_corr']:.3f}. Accuracy rises a lot ({opus_row['old_accuracy']:.3f} -> {opus_row['new_accuracy']:.3f}) but the human-correlation barely moves, which again looks like scaling up within the same ordering rather than a structural shift.",
        "- A notable exception to a pure `score = human-likeness` story is that some `Opus 4.5` settings are more human-aligned than much higher-scoring GPT settings. So the overall trend is positive, but raw accuracy and human-style item pattern are not the same thing.",
        "",
        "## Rejections / weak ideas",
        "",
        "- I do not see a strong structural-break story here. The data are more consistent with newer or higher-budget models extending along an existing human difficulty gradient.",
        "- The correlation gains within GPT are directionally positive, but bootstrap intervals on the correlation deltas are still fairly wide on this item set, so I would keep those as `suggestive`, not `definitive`.",
        "",
        "## Tables",
        "",
        frame_to_text_table(summary[["model", "family", "pair_accuracy", "human_pearson"]].dropna().sort_values("pair_accuracy", ascending=False).head(15).round(3)),
        "",
        frame_to_text_table(deltas[["comparison", "delta_accuracy", "delta_human_corr", "delta_human_corr_ci_lo", "delta_human_corr_ci_hi", "newly_solved_human_mean", "still_failed_human_mean"]].round(3)),
        "",
    ]

    (BASE_DIR / "temporal_hypotheses.md").write_text("\n".join(lines), encoding="utf-8")


def write_main_analysis_note(trend_stats: dict, deltas: pd.DataFrame) -> None:
    gpt_mid = deltas[deltas["comparison"] == "GPT 5.1 med -> 5.2 med"].iloc[0]
    gpt_high = deltas[deltas["comparison"] == "GPT 5.1 high -> 5.2 high"].iloc[0]
    opus = deltas[deltas["comparison"] == "Opus 4.5 16k -> 64k"].iloc[0]

    lines = [
        "# Temporal Hypothesis Note",
        "",
        "Quick answer: yes, there is a little signal here.",
        "",
        f"- Across the local model panel, better ARC Public Eval pair accuracy tends to come with better human item-alignment (`r = {trend_stats['accuracy_vs_human_corr_pearson']:.3f}` across models).",
        f"- `gpt-5.2` is generally more human-aligned than `gpt-5.1` at comparable thinking levels. Medium goes from {gpt_mid['old_human_corr']:.3f} to {gpt_mid['new_human_corr']:.3f}; high goes from {gpt_high['old_human_corr']:.3f} to {gpt_high['new_human_corr']:.3f}.",
        f"- The qualitative pattern still looks mostly like `more of the same but better`: newly solved GPT items are easier for humans than the items the new model still fails ({gpt_mid['newly_solved_human_mean']:.3f} vs {gpt_mid['still_failed_human_mean']:.3f} for the medium pair; {gpt_high['newly_solved_human_mean']:.3f} vs {gpt_high['still_failed_human_mean']:.3f} for the high pair).",
        f"- We do not have `Opus 4` locally, so there is no direct `Opus 4 -> 4.5` time test. What we can say is that `Opus 4.5` already looks quite human-aligned by `16k` and stays that way at `64k` ({opus['old_human_corr']:.3f} -> {opus['new_human_corr']:.3f}) while accuracy rises a lot.",
        "- One interesting exception: `Opus 4.5` can look more human-aligned than much higher-scoring GPT settings, so better raw score and more human-like item ordering are related but not identical.",
        "",
        "I kept the fuller writeup and figure in `Human data/Creme Analysis/temporal_hypotheses.md` and `fig02_family_progressions.png` because that frame is closer to the original Cremieux-style question.",
        "",
    ]

    (MAIN_ANALYSIS_DIR / "temporal_hypothesis_note.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dirs()

    comparison = load_robust_comparison()
    model_matrix = build_model_matrix(comparison["task_pair_id"].tolist())
    summary, trend_stats = build_model_alignment_summary(comparison, model_matrix)
    deltas = build_family_delta_summary(comparison, model_matrix)

    summary.to_csv(TABLES_DIR / "model_alignment_summary.csv", index=False)
    deltas.to_csv(TABLES_DIR / "family_delta_summary.csv", index=False)
    plot_family_progressions(summary)
    write_creme_note(summary, trend_stats, deltas)
    write_main_analysis_note(trend_stats, deltas)

    print(f"Done. Outputs written to {BASE_DIR}")


if __name__ == "__main__":
    main()
