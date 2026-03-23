from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

HUMAN_CSV = SCRIPT_DIR / "test_pair_attempts.csv"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
TABLES_DIR = ANALYSIS_DIR / "tables"
TASK_PANELS_DIR = ANALYSIS_DIR / "task_panels"

ARC_V2_DIR = ROOT_DIR / "Psychometric Analysis" / "data" / "ARC-AGI-2" / "data"
ARC_V2_TRAIN_DIR = ARC_V2_DIR / "training"
ARC_V2_EVAL_DIR = ARC_V2_DIR / "evaluation"
LM_V2_PREDS_DIR = ROOT_DIR / "Psychometric Analysis" / "data" / "arc_agi_v2_public_eval"


DARK_SLATE = "#2C3E50"
ACCENT_BLUE = "#2E86AB"
THINKING_TEAL = "#1ABC9C"
STANDARD_CORAL = "#E76F51"
MUTED_GOLD = "#E9C46A"
LIGHT_GRAY = "#D8DEE9"
MID_GRAY = "#8D99AE"
BG_WHITE = "#FFFFFF"
TEXT_GRAY = "#555555"
FAIL_RED = "#E31A1C"

ARC_PALETTE = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": BG_WHITE,
            "axes.facecolor": BG_WHITE,
            "savefig.facecolor": BG_WHITE,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_GRAY,
            "ytick.color": TEXT_GRAY,
            "axes.labelcolor": TEXT_GRAY,
            "text.color": TEXT_GRAY,
        }
    )


def ensure_dirs() -> None:
    for path in [ANALYSIS_DIR, FIGURES_DIR, TABLES_DIR, TASK_PANELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_grid(grid: object) -> str:
    if not isinstance(grid, list) or not grid:
        return "EMPTY"
    return ",".join(str(cell) for row in grid for cell in row)


def classify_model(model_name: str) -> str:
    model_name = model_name.lower()
    if "thinking-none" in model_name:
        return "Standard"
    if any(token in model_name for token in ["thinking", "deep", "reasoning"]):
        return "Thinking"
    if "gemini" in model_name or "gpt-5-pro" in model_name:
        return "Thinking"
    return "Standard"


def load_human_attempts() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_CSV)
    df["solved"] = (df["correct_submissions"] > 0).astype(int)
    df["task_pair_id"] = df["task_ID"] + "__" + df["test_index"].astype(str)
    df = df.sort_values(["session_ID", "start_time_seconds", "task_ID", "test_index"]).reset_index(drop=True)
    df["attempt_order"] = df.groupby("session_ID").cumcount() + 1
    session_mix = df.groupby("session_ID")["task_set"].agg(lambda s: ", ".join(sorted(set(s))))
    mix_map = {
        "Public Eval": "Public Eval Only",
        "Public Train": "Public Train Only",
        "Public Eval, Public Train": "Mixed",
    }
    df["session_mix"] = df["session_ID"].map(session_mix).map(mix_map).fillna("Mixed")
    return df


def grid_dimensions(grid: list[list[int]] | None) -> tuple[int, int, int, int]:
    if grid is None:
        return (np.nan, np.nan, np.nan, np.nan)
    n_rows = len(grid)
    n_cols = len(grid[0])
    n_cells = n_rows * n_cols
    n_colors = len({cell for row in grid for cell in row})
    return (n_rows, n_cols, n_cells, n_colors)


def load_arc_metadata() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict]]:
    task_rows: list[dict] = []
    pair_rows: list[dict] = []
    task_cache: dict[tuple[str, str], dict] = {}

    for task_set, folder in [("Public Train", ARC_V2_TRAIN_DIR), ("Public Eval", ARC_V2_EVAL_DIR)]:
        for json_path in sorted(folder.glob("*.json")):
            obj = json.loads(json_path.read_text())
            task_cache[(json_path.stem, task_set)] = obj
            train_pairs = obj.get("train", [])
            test_pairs = obj.get("test", [])

            all_inputs = [pair["input"] for pair in train_pairs + test_pairs]
            input_stats = np.array([grid_dimensions(grid) for grid in all_inputs], dtype=float)
            output_stats = np.array(
                [grid_dimensions(pair.get("output")) for pair in train_pairs + test_pairs if pair.get("output") is not None],
                dtype=float,
            )

            task_rows.append(
                {
                    "task_ID": json_path.stem,
                    "task_set": task_set,
                    "n_train_pairs": len(train_pairs),
                    "n_test_pairs": len(test_pairs),
                    "mean_input_rows": np.nanmean(input_stats[:, 0]),
                    "mean_input_cols": np.nanmean(input_stats[:, 1]),
                    "mean_input_cells": np.nanmean(input_stats[:, 2]),
                    "max_input_cells": np.nanmax(input_stats[:, 2]),
                    "mean_input_colors": np.nanmean(input_stats[:, 3]),
                    "mean_output_cells": np.nanmean(output_stats[:, 2]) if len(output_stats) else np.nan,
                    "mean_output_colors": np.nanmean(output_stats[:, 3]) if len(output_stats) else np.nan,
                }
            )

            for test_index, pair in enumerate(test_pairs):
                in_rows, in_cols, in_cells, in_colors = grid_dimensions(pair["input"])
                out_rows, out_cols, out_cells, out_colors = grid_dimensions(pair["output"])
                pair_rows.append(
                    {
                        "task_ID": json_path.stem,
                        "task_set": task_set,
                        "test_index": test_index,
                        "task_pair_id": f"{json_path.stem}__{test_index}",
                        "n_train_pairs": len(train_pairs),
                        "n_test_pairs": len(test_pairs),
                        "input_rows": in_rows,
                        "input_cols": in_cols,
                        "input_cells": in_cells,
                        "input_colors": in_colors,
                        "output_rows": out_rows,
                        "output_cols": out_cols,
                        "output_cells": out_cells,
                        "output_colors": out_colors,
                        "size_change_ratio": out_cells / max(in_cells, 1),
                    }
                )

    return pd.DataFrame(task_rows), pd.DataFrame(pair_rows), task_cache


def fit_person_item_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    design = encoder.fit_transform(df[["session_ID", "task_pair_id"]])
    y = df["solved"].to_numpy()

    model = LogisticRegression(
        C=2.0,
        solver="saga",
        max_iter=8000,
        fit_intercept=True,
        random_state=0,
    )
    model.fit(design, y)

    pred_prob = model.predict_proba(design)[:, 1]
    feature_names = pd.Index(encoder.get_feature_names_out(["session_ID", "task_pair_id"]))
    coefficients = pd.Series(model.coef_[0], index=feature_names)

    session_ability = coefficients[feature_names.str.startswith("session_ID_")]
    session_ability.index = session_ability.index.str.replace("session_ID_", "", regex=False)
    session_ability = session_ability - session_ability.mean()

    item_ease = coefficients[feature_names.str.startswith("task_pair_id_")]
    item_ease.index = item_ease.index.str.replace("task_pair_id_", "", regex=False)
    item_difficulty = -(item_ease - item_ease.mean())

    fitted = df.copy()
    fitted["pred_prob"] = pred_prob
    fitted["std_resid"] = (fitted["solved"] - fitted["pred_prob"]) / np.sqrt(
        np.clip(fitted["pred_prob"] * (1 - fitted["pred_prob"]), 1e-6, None)
    )

    diagnostics = {
        "roc_auc": roc_auc_score(y, pred_prob),
        "log_loss": log_loss(y, pred_prob),
        "brier_score": brier_score_loss(y, pred_prob),
        "intercept": float(model.intercept_[0]),
    }

    session_df = pd.DataFrame({"session_ID": session_ability.index, "ability": session_ability.values})
    item_df = pd.DataFrame({"task_pair_id": item_difficulty.index, "difficulty": item_difficulty.values})
    return session_df, item_df, fitted, diagnostics


def build_session_summary(df: pd.DataFrame, session_df: pd.DataFrame) -> pd.DataFrame:
    session_summary = (
        df.groupby("session_ID")
        .agg(
            attempts=("solved", "size"),
            total_solved=("solved", "sum"),
            solve_rate=("solved", "mean"),
            total_duration_seconds=("duration_seconds", "sum"),
            mean_duration_seconds=("duration_seconds", "mean"),
            median_duration_seconds=("duration_seconds", "median"),
            total_submissions=("submissions", "sum"),
            mean_submissions=("submissions", "mean"),
            mean_pred_prob=("pred_prob", "mean"),
            outfit=("std_resid", lambda s: float(np.mean(np.square(s)))),
            session_mix=("session_mix", "first"),
        )
        .reset_index()
    )
    session_summary["tasks_per_minute"] = session_summary["attempts"] / (
        np.maximum(session_summary["total_duration_seconds"] / 60.0, 1e-9)
    )
    session_summary = session_summary.merge(session_df, on="session_ID", how="left")
    session_summary["ability_rank"] = session_summary["ability"].rank(method="average", ascending=False)
    return session_summary.sort_values("ability", ascending=False).reset_index(drop=True)


def build_item_summary(df: pd.DataFrame, item_df: pd.DataFrame, pair_meta: pd.DataFrame) -> pd.DataFrame:
    item_summary = (
        df.groupby(["task_pair_id", "task_ID", "task_set", "test_index"])
        .agg(
            attempts=("solved", "size"),
            solve_count=("solved", "sum"),
            solve_rate=("solved", "mean"),
            mean_duration_seconds=("duration_seconds", "mean"),
            median_duration_seconds=("duration_seconds", "median"),
            mean_submissions=("submissions", "mean"),
            median_submissions=("submissions", "median"),
            mean_pred_prob=("pred_prob", "mean"),
            outfit=("std_resid", lambda s: float(np.mean(np.square(s)))),
        )
        .reset_index()
    )

    session_totals = (
        df.groupby("session_ID")["solved"]
        .agg(session_solved="sum", session_attempts="count")
        .reset_index()
    )
    rest_df = df.merge(session_totals, on="session_ID", how="left")
    rest_df["rest_score"] = (rest_df["session_solved"] - rest_df["solved"]) / (
        rest_df["session_attempts"] - 1
    ).replace(0, np.nan)

    discrim_rows: list[dict] = []
    for task_pair_id, group in rest_df.groupby("task_pair_id"):
        valid = group.dropna(subset=["rest_score"]).copy()
        point_biserial = np.nan
        ability_gap = np.nan
        if len(valid) >= 5 and valid["solved"].nunique() >= 2:
            point_biserial = float(valid["solved"].corr(valid["rest_score"]))
            if valid.loc[valid["solved"] == 1, "ability"].size and valid.loc[valid["solved"] == 0, "ability"].size:
                ability_gap = float(
                    valid.loc[valid["solved"] == 1, "ability"].mean()
                    - valid.loc[valid["solved"] == 0, "ability"].mean()
                )
        discrim_rows.append(
            {
                "task_pair_id": task_pair_id,
                "point_biserial": point_biserial,
                "ability_gap": ability_gap,
            }
        )

    item_summary = item_summary.merge(item_df, on="task_pair_id", how="left")
    item_summary = item_summary.merge(pd.DataFrame(discrim_rows), on="task_pair_id", how="left")
    item_summary = item_summary.merge(
        pair_meta,
        on=["task_pair_id", "task_ID", "task_set", "test_index"],
        how="left",
    )
    item_summary["difficulty_rank"] = item_summary["difficulty"].rank(method="average", ascending=False)
    item_summary = item_summary.sort_values("difficulty", ascending=False).reset_index(drop=True)
    return item_summary


def build_overall_summary(
    human_df: pd.DataFrame,
    session_summary: pd.DataFrame,
    item_summary: pd.DataFrame,
    task_meta: pd.DataFrame,
    model_diag: dict,
) -> pd.DataFrame:
    total_pairs_all = int(task_meta["n_test_pairs"].sum())
    attempted_task_counts = human_df[["task_ID", "task_set"]].drop_duplicates().shape[0]
    attempted_pair_counts = human_df["task_pair_id"].nunique()
    matrix_density = len(human_df) / (human_df["session_ID"].nunique() * attempted_pair_counts)
    warmup_mask = (human_df["task_ID"] == "0a1d4ef5") & (human_df["test_index"] == 0)

    summary = {
        "rows": len(human_df),
        "sessions": human_df["session_ID"].nunique(),
        "task_ids_attempted": human_df["task_ID"].nunique(),
        "task_pairs_attempted": attempted_pair_counts,
        "task_ids_attempted_of_all": attempted_task_counts,
        "task_pairs_attempted_of_all": total_pairs_all,
        "task_pair_coverage": attempted_pair_counts / total_pairs_all,
        "matrix_density": matrix_density,
        "overall_solve_rate": human_df["solved"].mean(),
        "overall_solve_rate_without_warmup": human_df.loc[~warmup_mask, "solved"].mean(),
        "public_train_solve_rate": human_df.loc[human_df["task_set"] == "Public Train", "solved"].mean(),
        "public_eval_solve_rate": human_df.loc[human_df["task_set"] == "Public Eval", "solved"].mean(),
        "median_attempts_per_session": session_summary["attempts"].median(),
        "mean_attempts_per_session": session_summary["attempts"].mean(),
        "median_attempts_per_item": item_summary["attempts"].median(),
        "warmup_attempts": int(warmup_mask.sum()),
        "warmup_solve_rate": human_df.loc[warmup_mask, "solved"].mean(),
        "logit_auc": model_diag["roc_auc"],
        "logit_log_loss": model_diag["log_loss"],
        "logit_brier_score": model_diag["brier_score"],
    }
    return pd.DataFrame([summary])


def build_sampling_bias_table(task_meta: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
    attempted_tasks = human_df[["task_ID", "task_set"]].drop_duplicates().assign(attempted=1)
    bias_table = task_meta.merge(attempted_tasks, on=["task_ID", "task_set"], how="left").fillna({"attempted": 0})
    bias_summary = (
        bias_table.groupby(["task_set", "attempted"])
        .agg(
            n_tasks=("task_ID", "count"),
            mean_input_cells=("mean_input_cells", "mean"),
            mean_input_colors=("mean_input_colors", "mean"),
            mean_train_pairs=("n_train_pairs", "mean"),
            mean_test_pairs=("n_test_pairs", "mean"),
        )
        .reset_index()
    )
    bias_summary["attempted_label"] = bias_summary["attempted"].map({0.0: "Unattempted", 1.0: "Attempted"})
    return bias_summary


def load_lm_pair_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_cache: dict[str, dict] = {}
    truth_pairs: dict[str, str] = {}
    for truth_path in sorted(ARC_V2_EVAL_DIR.glob("*.json")):
        obj = json.loads(truth_path.read_text())
        truth_cache[truth_path.name] = obj
        for idx, pair in enumerate(obj.get("test", [])):
            truth_pairs[f"{truth_path.stem}__{idx}"] = normalize_grid(pair["output"])

    model_rows: dict[str, dict[str, int]] = {}
    for model_dir in sorted(LM_V2_PREDS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        row = {pair_id: 0 for pair_id in truth_pairs}
        for pred_path in sorted(model_dir.glob("*.json")):
            truth_obj = truth_cache.get(pred_path.name)
            if truth_obj is None:
                continue
            pred_obj = json.loads(pred_path.read_text())
            for idx, pair in enumerate(truth_obj.get("test", [])):
                pair_id = f"{pred_path.stem}__{idx}"
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
                row[pair_id] = int(normalize_grid(answer) == truth_pairs[pair_id])
        model_rows[model_dir.name] = row

    lm_matrix = pd.DataFrame.from_dict(model_rows, orient="index").sort_index(axis=1)
    model_summary = pd.DataFrame(
        {
            "model": lm_matrix.index,
            "type": [classify_model(model) for model in lm_matrix.index],
            "pair_accuracy": lm_matrix.mean(axis=1).values,
        }
    ).sort_values("pair_accuracy", ascending=False)
    return lm_matrix, model_summary


def build_human_lm_comparison(item_summary: pd.DataFrame, lm_matrix: pd.DataFrame, model_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    best_single_model = model_summary.iloc[0]["model"]
    pair_summary = pd.DataFrame(
        {
            "task_pair_id": lm_matrix.columns,
            "lm_mean": lm_matrix.mean(axis=0).values,
            "lm_best_across_models": lm_matrix.max(axis=0).values,
            "lm_best_single_model": lm_matrix.loc[best_single_model].values,
        }
    )

    public_eval_items = item_summary.loc[item_summary["task_set"] == "Public Eval"].copy()
    comparison = public_eval_items.merge(pair_summary, on="task_pair_id", how="inner")
    comparison["gap_vs_lm_mean"] = comparison["solve_rate"] - comparison["lm_mean"]
    comparison["gap_vs_best_single_model"] = comparison["solve_rate"] - comparison["lm_best_single_model"]
    comparison["gap_vs_oracle"] = comparison["solve_rate"] - comparison["lm_best_across_models"]

    high_coverage = comparison.loc[comparison["attempts"] >= 8].copy()
    summary = {
        "n_overlap_pairs_all": int(len(comparison)),
        "n_overlap_pairs_ge_8": int(len(high_coverage)),
        "corr_human_vs_lm_mean_all": float(comparison["solve_rate"].corr(comparison["lm_mean"])),
        "corr_human_vs_lm_mean_ge_8": float(high_coverage["solve_rate"].corr(high_coverage["lm_mean"])),
        "corr_human_vs_best_single_ge_8": float(high_coverage["solve_rate"].corr(high_coverage["lm_best_single_model"])),
        "human_mean_ge_8": float(high_coverage["solve_rate"].mean()),
        "lm_mean_ge_8": float(high_coverage["lm_mean"].mean()),
        "best_single_model_mean_ge_8": float(high_coverage["lm_best_single_model"].mean()),
        "oracle_mean_ge_8": float(high_coverage["lm_best_across_models"].mean()),
        "best_single_model_name": str(best_single_model),
        "share_pairs_human_gt_lm_mean_ge_8": float((high_coverage["solve_rate"] > high_coverage["lm_mean"]).mean()),
        "share_pairs_human_gt_best_single_ge_8": float(
            (high_coverage["solve_rate"] > high_coverage["lm_best_single_model"]).mean()
        ),
    }
    return comparison.sort_values("gap_vs_lm_mean", ascending=False), summary


def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def annotate_points(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, label: str, n: int = 8) -> None:
    if data.empty:
        return
    for _, row in data.head(n).iterrows():
        ax.text(row[x], row[y], row[label], fontsize=8, color=TEXT_GRAY)


def frame_to_text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def plot_response_matrix(session_summary: pd.DataFrame, item_summary: pd.DataFrame, df: pd.DataFrame) -> None:
    ordered_sessions = session_summary.sort_values("ability")["session_ID"].tolist()
    ordered_items = item_summary.sort_values("difficulty")["task_pair_id"].tolist()
    matrix = (
        df.pivot_table(index="session_ID", columns="task_pair_id", values="solved", aggfunc="max")
        .reindex(index=ordered_sessions, columns=ordered_items)
        .to_numpy()
    )
    matrix = np.where(np.isnan(matrix), -1, matrix)

    cmap = ListedColormap([LIGHT_GRAY, FAIL_RED, THINKING_TEAL])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title("Human Session-by-Item Response Matrix\nSorted by latent ability and item difficulty", loc="left")
    ax.set_xlabel("Task Pairs: Easier to Harder")
    ax.set_ylabel("Sessions: Lower to Higher Ability")
    ax.set_xticks([])
    ax.set_yticks([])
    legend_handles = [
        Patch(facecolor=LIGHT_GRAY, edgecolor=LIGHT_GRAY, label="Not Attempted"),
        Patch(facecolor=FAIL_RED, edgecolor=FAIL_RED, label="Failed"),
        Patch(facecolor=THINKING_TEAL, edgecolor=THINKING_TEAL, label="Solved"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    density = df.shape[0] / (len(ordered_sessions) * len(ordered_items))
    fig.text(0.01, 0.01, f"Observed density: {density:.2%} | Warm-up item 0a1d4ef5__0 appears in 463 sessions", fontsize=10)
    fig.savefig(FIGURES_DIR / "fig01_human_response_matrix.png")
    plt.close(fig)


def plot_split_performance(df: pd.DataFrame) -> None:
    split_summary = (
        df.groupby("task_set")
        .agg(
            solve_rate=("solved", "mean"),
            mean_duration_seconds=("duration_seconds", "mean"),
            attempts=("solved", "size"),
        )
        .reset_index()
    )
    index_summary = (
        df.groupby(["task_set", "test_index"])
        .agg(
            solve_rate=("solved", "mean"),
            mean_duration_seconds=("duration_seconds", "mean"),
            attempts=("solved", "size"),
        )
        .reset_index()
    )
    index_summary["test_index_label"] = index_summary["test_index"].astype(int).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=split_summary,
        x="task_set",
        y="solve_rate",
        hue="task_set",
        palette={"Public Train": THINKING_TEAL, "Public Eval": STANDARD_CORAL},
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Solve Rate by Split")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Solve Rate")
    axes[0].set_ylim(0, 1)

    sns.barplot(
        data=index_summary,
        x="test_index_label",
        y="solve_rate",
        hue="task_set",
        palette={"Public Train": THINKING_TEAL, "Public Eval": STANDARD_CORAL},
        ax=axes[1],
    )
    axes[1].set_title("Solve Rate by Test Index")
    axes[1].set_xlabel("Test Index")
    axes[1].set_ylabel("Solve Rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="")
    fig.suptitle("Human performance is higher on Public Train and drops on later test pairs", x=0.01, ha="left")
    fig.savefig(FIGURES_DIR / "fig02_split_performance.png")
    plt.close(fig)


def plot_session_profile(session_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(session_summary["ability"], bins=30, color=ACCENT_BLUE, ax=axes[0])
    axes[0].axvline(session_summary["ability"].median(), color=STANDARD_CORAL, linestyle="--", linewidth=2)
    axes[0].set_title("Latent Ability Distribution")
    axes[0].set_xlabel("Regularized person ability")
    axes[0].set_ylabel("Sessions")

    sns.scatterplot(
        data=session_summary,
        x="ability",
        y="solve_rate",
        size="attempts",
        hue="session_mix",
        sizes=(25, 250),
        palette={"Mixed": ACCENT_BLUE, "Public Train Only": THINKING_TEAL, "Public Eval Only": STANDARD_CORAL},
        ax=axes[1],
    )
    axes[1].set_title("Ability vs Raw Solve Rate")
    axes[1].set_xlabel("Regularized person ability")
    axes[1].set_ylabel("Raw solve rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Session profile: broad human spread, but many high-performing sessions", x=0.01, ha="left")
    fig.savefig(FIGURES_DIR / "fig03_session_profile.png")
    plt.close(fig)


def plot_item_profile(item_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        data=item_summary,
        x="difficulty",
        hue="task_set",
        bins=30,
        palette={"Public Train": THINKING_TEAL, "Public Eval": STANDARD_CORAL},
        alpha=0.6,
        ax=axes[0],
    )
    axes[0].set_title("Item Difficulty Distribution")
    axes[0].set_xlabel("Regularized item difficulty")
    axes[0].set_ylabel("Task pairs")

    sns.scatterplot(
        data=item_summary,
        x="mean_duration_seconds",
        y="solve_rate",
        size="attempts",
        hue="task_set",
        sizes=(20, 250),
        palette={"Public Train": THINKING_TEAL, "Public Eval": STANDARD_CORAL},
        ax=axes[1],
    )
    axes[1].set_title("Item Speed-Accuracy Frontier")
    axes[1].set_xlabel("Mean duration (seconds)")
    axes[1].set_ylabel("Human solve rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    hardest_eval = item_summary.loc[(item_summary["task_set"] == "Public Eval") & (item_summary["attempts"] >= 8)]
    hardest_eval = hardest_eval.sort_values("difficulty", ascending=False).head(5)
    annotate_points(axes[1], hardest_eval, "mean_duration_seconds", "solve_rate", "task_pair_id", n=5)
    fig.suptitle("Time spent is the clearest simple correlate of human difficulty", x=0.01, ha="left")
    fig.savefig(FIGURES_DIR / "fig04_item_profile.png")
    plt.close(fig)


def plot_sampling_bias(item_summary: pd.DataFrame, task_bias_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    attempts = item_summary["attempts"].sort_values()
    bins = np.unique(np.geomspace(max(2, attempts.min()), attempts.max(), 20).astype(int))
    axes[0].hist(item_summary["attempts"], bins=bins, color=ACCENT_BLUE, alpha=0.85)
    axes[0].set_xscale("log")
    axes[0].axvline(9, color=MUTED_GOLD, linestyle="--", linewidth=2)
    axes[0].axvline(463, color=STANDARD_CORAL, linestyle="--", linewidth=2)
    axes[0].set_title("Exposure per Task Pair")
    axes[0].set_xlabel("Human attempts per item (log scale)")
    axes[0].set_ylabel("Count of task pairs")

    sns.barplot(
        data=task_bias_summary,
        x="attempted_label",
        y="mean_input_cells",
        hue="task_set",
        palette={"Public Train": THINKING_TEAL, "Public Eval": STANDARD_CORAL},
        ax=axes[1],
    )
    axes[1].set_title("Attempted Tasks Skew Larger than the Full Pool")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean input cells per task")
    axes[1].legend(title="")
    fig.suptitle("Coverage is sparse and non-representative, especially on Public Train", x=0.01, ha="left")
    fig.savefig(FIGURES_DIR / "fig05_sampling_bias.png")
    plt.close(fig)


def plot_discrimination(item_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    view = item_summary.loc[item_summary["attempts"] >= 5].copy()
    sns.scatterplot(
        data=view,
        x="difficulty",
        y="point_biserial",
        hue="task_set",
        size="attempts",
        sizes=(20, 250),
        palette={"Public Train": THINKING_TEAL, "Public Eval": STANDARD_CORAL},
        ax=ax,
    )
    ax.axhline(0, color=MID_GRAY, linestyle="--", linewidth=1.5)
    ax.set_title("Item Discrimination vs Difficulty")
    ax.set_xlabel("Regularized item difficulty")
    ax.set_ylabel("Point-biserial with rest score")
    low_disc = view.sort_values("point_biserial").head(8)
    annotate_points(ax, low_disc, "difficulty", "point_biserial", "task_pair_id", n=8)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.text(0.01, 0.01, "Low-discrimination estimates are noisy because most items were only seen about 9 times.", fontsize=10)
    fig.savefig(FIGURES_DIR / "fig06_discrimination.png")
    plt.close(fig)


def plot_human_vs_lm(comparison: pd.DataFrame, lm_summary: dict) -> None:
    view = comparison.loc[comparison["attempts"] >= 8].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(
        data=view,
        x="lm_mean",
        y="solve_rate",
        size="attempts",
        hue="mean_duration_seconds",
        palette="viridis",
        sizes=(25, 250),
        ax=axes[0],
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=MID_GRAY)
    axes[0].set_title("Public Eval Pair Difficulty: Humans vs Average LM")
    axes[0].set_xlabel("Average LM solve rate")
    axes[0].set_ylabel("Human solve rate")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    top_outliers = pd.concat([view.nlargest(4, "gap_vs_lm_mean"), view.nsmallest(4, "gap_vs_lm_mean")])
    annotate_points(axes[0], top_outliers, "lm_mean", "solve_rate", "task_pair_id", n=8)
    axes[0].legend(title="Mean duration", bbox_to_anchor=(1.02, 1), loc="upper left")

    benchmark_frame = pd.DataFrame(
        {
            "group": ["Humans", "Best Single Model", "Average Model", "Per-Pair Oracle"],
            "solve_rate": [
                lm_summary["human_mean_ge_8"],
                lm_summary["best_single_model_mean_ge_8"],
                lm_summary["lm_mean_ge_8"],
                lm_summary["oracle_mean_ge_8"],
            ],
            "color": [ACCENT_BLUE, THINKING_TEAL, STANDARD_CORAL, MUTED_GOLD],
        }
    )
    axes[1].bar(benchmark_frame["group"], benchmark_frame["solve_rate"], color=benchmark_frame["color"], alpha=0.9)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean solve rate on Public Eval pairs with >=8 human attempts")
    axes[1].set_title("Aggregate Comparison")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"Humans beat the mean model on {format_pct(lm_summary['share_pairs_human_gt_lm_mean_ge_8'])} of well-sampled Public Eval pairs",
        x=0.01,
        ha="left",
    )
    fig.savefig(FIGURES_DIR / "fig07_human_vs_lm_public_eval.png")
    plt.close(fig)


def draw_arc_grid(ax: plt.Axes, grid: list[list[int]], title: str) -> None:
    cmap = ListedColormap(ARC_PALETTE)
    ax.imshow(np.array(grid), cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which="minor", color="#F5F5F5", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)


def save_task_panel(
    task_pair_id: str,
    item_summary: pd.DataFrame,
    task_cache: dict[tuple[str, str], dict],
    comparison_lookup: dict[str, dict] | None,
) -> Path:
    row = item_summary.loc[item_summary["task_pair_id"] == task_pair_id].iloc[0]
    task = task_cache[(row["task_ID"], row["task_set"])]
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    test_pair = test_pairs[int(row["test_index"])]

    n_rows = len(train_pairs) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, max(2.4 * n_rows, 6)), squeeze=False)
    for idx, pair in enumerate(train_pairs):
        draw_arc_grid(axes[idx, 0], pair["input"], f"Train {idx + 1} Input")
        draw_arc_grid(axes[idx, 1], pair["output"], f"Train {idx + 1} Output")
    draw_arc_grid(axes[-1, 0], test_pair["input"], f"Test {int(row['test_index'])} Input")
    draw_arc_grid(axes[-1, 1], test_pair["output"], f"Test {int(row['test_index'])} Output")

    lm_text = ""
    if comparison_lookup and task_pair_id in comparison_lookup:
        comp = comparison_lookup[task_pair_id]
        lm_text = (
            f" | LM mean {format_pct(comp['lm_mean'])}"
            f" | Best single {format_pct(comp['lm_best_single_model'])}"
        )

    title = (
        f"{task_pair_id} | {row['task_set']} | human solve {format_pct(row['solve_rate'])}"
        f" (n={int(row['attempts'])}) | mean duration {row['mean_duration_seconds']:.0f}s"
        f"{lm_text}"
    )
    fig.suptitle("\n".join(textwrap.wrap(title, width=95)), fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = TASK_PANELS_DIR / f"{task_pair_id}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def create_montage(image_paths: list[Path], title: str, out_path: Path) -> None:
    n_images = len(image_paths)
    n_cols = 2
    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.8 * n_rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image_path in zip(axes.ravel(), image_paths):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(image_path.stem, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=16, x=0.01, ha="left")
    fig.savefig(out_path)
    plt.close(fig)


def generate_task_galleries(
    item_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    task_cache: dict[tuple[str, str], dict],
) -> tuple[list[str], list[str]]:
    comparison_lookup = comparison.set_index("task_pair_id").to_dict(orient="index")
    eligible = item_summary.loc[item_summary["attempts"] >= 8].copy()
    hardest = eligible.sort_values(["difficulty", "attempts"], ascending=[False, False]).head(4)["task_pair_id"].tolist()
    easiest = eligible.sort_values(["difficulty", "attempts"], ascending=[True, False]).head(4)["task_pair_id"].tolist()

    hard_paths = [save_task_panel(pair_id, item_summary, task_cache, comparison_lookup) for pair_id in hardest]
    easy_paths = [save_task_panel(pair_id, item_summary, task_cache, comparison_lookup) for pair_id in easiest]

    create_montage(hard_paths, "Hardest Human Task Pairs with >=8 Attempts", FIGURES_DIR / "fig08_gallery_hardest_items.png")
    create_montage(easy_paths, "Easiest Human Task Pairs with >=8 Attempts", FIGURES_DIR / "fig09_gallery_easiest_items.png")
    return hardest, easiest


def write_report(
    overall_summary: pd.DataFrame,
    item_summary: pd.DataFrame,
    task_bias_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    lm_summary: dict,
    hardest: list[str],
    easiest: list[str],
) -> None:
    overall = overall_summary.iloc[0]
    high_coverage = comparison.loc[comparison["attempts"] >= 8].copy()
    robust_human_adv = high_coverage.sort_values("gap_vs_lm_mean", ascending=False).head(5)
    robust_model_adv = high_coverage.sort_values("gap_vs_lm_mean", ascending=True).head(5)
    hardest_items = item_summary.loc[item_summary["task_pair_id"].isin(hardest), ["task_pair_id", "task_set", "solve_rate", "attempts", "mean_duration_seconds"]]
    easiest_items = item_summary.loc[item_summary["task_pair_id"].isin(easiest), ["task_pair_id", "task_set", "solve_rate", "attempts", "mean_duration_seconds"]]

    lines = [
        "# Human Testing Psychometric Analysis",
        "",
        "## Scope",
        "",
        "This report analyzes the locally downloaded `arc_agi_2_human_testing` file (`Human data/test_pair_attempts.csv`) in the context of the ARC-AGI-2 task JSONs and the ARC-AGI-2 model prediction folder in this workspace.",
        "",
        "## Data Structure",
        "",
        f"- {int(overall['rows'])} human attempts across {int(overall['sessions'])} sessions, {int(overall['task_ids_attempted'])} task IDs, and {int(overall['task_pairs_attempted'])} task-test pairs.",
        f"- Human coverage spans {format_pct(float(overall['task_pair_coverage']))} of all ARC-AGI-2 public task pairs ({int(overall['task_pairs_attempted'])} of {int(overall['task_pairs_attempted_of_all'])}).",
        f"- The observed session-by-item matrix is extremely sparse: {overall['matrix_density']:.2%} density.",
        f"- One warm-up pair (`0a1d4ef5__0`) appears in {int(overall['warmup_attempts'])} sessions with a {format_pct(float(overall['warmup_solve_rate']))} solve rate.",
        "",
        "## Exploratory Findings",
        "",
        f"- Overall human solve rate is {format_pct(float(overall['overall_solve_rate']))}; excluding the warm-up item it is still {format_pct(float(overall['overall_solve_rate_without_warmup']))}.",
        f"- Public Train is much easier than Public Eval: {format_pct(float(overall['public_train_solve_rate']))} vs {format_pct(float(overall['public_eval_solve_rate']))}.",
        f"- Sessions are short and uneven: median {overall['median_attempts_per_session']:.1f} attempts per session, mean {overall['mean_attempts_per_session']:.2f}.",
        f"- Items are also unevenly exposed: median {overall['median_attempts_per_item']:.1f} attempts per task pair.",
        "",
        "## Psychometric Results",
        "",
        f"- A regularized person-plus-item logistic model fits the sparse matrix well (AUC {overall['logit_auc']:.3f}, log loss {overall['logit_log_loss']:.3f}, Brier {overall['logit_brier_score']:.3f}).",
        f"- Item difficulty tracks human time cost more strongly than raw task size: item solve rate vs mean duration correlation is about {item_summary['solve_rate'].corr(item_summary['mean_duration_seconds']):.3f}.",
        f"- Raw visual size is a weak standalone predictor: item solve rate vs input cells correlation is about {item_summary['solve_rate'].corr(item_summary['input_cells']):.3f}.",
        f"- Discrimination estimates are usable but noisy because most items only have around {item_summary['attempts'].median():.0f} exposures.",
        "",
        "## Sampling Caveat",
        "",
        "The human sample is not representative of the full ARC-AGI-2 pool. Attempted tasks are systematically larger than unattempted tasks, especially on Public Train.",
        "",
        frame_to_text_table(task_bias_summary.round(3)),
        "",
        "## Public Eval Human vs LM Cross-Reference",
        "",
        f"- Direct overlap covers {lm_summary['n_overlap_pairs_all']} Public Eval task pairs; {lm_summary['n_overlap_pairs_ge_8']} of those have at least 8 human attempts.",
        f"- On the >=8-attempt overlap, humans average {format_pct(lm_summary['human_mean_ge_8'])}, the average model in the current local prediction folder averages {format_pct(lm_summary['lm_mean_ge_8'])}, and the best single model (`{lm_summary['best_single_model_name']}`) averages {format_pct(lm_summary['best_single_model_mean_ge_8'])}.",
        f"- The average per-pair oracle over all local models is {format_pct(lm_summary['oracle_mean_ge_8'])}, which is higher than the average human but is not achievable by any single model.",
        f"- Human and average-model difficulty are only moderately aligned (r = {lm_summary['corr_human_vs_lm_mean_ge_8']:.3f} on >=8-attempt pairs).",
        f"- Humans outperform the average model on {format_pct(lm_summary['share_pairs_human_gt_lm_mean_ge_8'])} of well-sampled Public Eval pairs and outperform the best single model on {format_pct(lm_summary['share_pairs_human_gt_best_single_ge_8'])}.",
        "",
        "Robust human-advantage items (>=8 attempts):",
        "",
        frame_to_text_table(
            robust_human_adv[["task_pair_id", "solve_rate", "attempts", "lm_mean", "lm_best_single_model", "gap_vs_lm_mean"]].round(3)
        ),
        "",
        "Robust model-advantage items (>=8 attempts):",
        "",
        frame_to_text_table(
            robust_model_adv[["task_pair_id", "solve_rate", "attempts", "lm_mean", "lm_best_single_model", "gap_vs_lm_mean"]].round(3)
        ),
        "",
        "Hardest gallery items:",
        "",
        frame_to_text_table(hardest_items.round(3)),
        "",
        "Easiest gallery items:",
        "",
        frame_to_text_table(easiest_items.round(3)),
        "",
        "## Interpretation",
        "",
        "- The human data do support a coherent difficulty structure, but this is an opportunistic sparse sample rather than a balanced exam form.",
        "- Public Eval remains challenging for humans, yet the human floor is much higher than the average model floor observed in the earlier LM psychometric work.",
        "- Difficulty is only partly shared between humans and models, suggesting some common latent structure plus substantial human-specific advantages in flexible abstraction.",
        "- Unlike the earlier LM heterogeneity report, the main limitation here is not deterministic over-consistency; it is sparse, biased coverage and low item exposure.",
        "",
    ]

    (ANALYSIS_DIR / "human_testing_psychometric_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dirs()

    print("Loading human attempts...")
    human_df = load_human_attempts()
    task_meta, pair_meta, task_cache = load_arc_metadata()
    human_df = human_df.merge(pair_meta, on=["task_ID", "task_set", "test_index", "task_pair_id"], how="left")

    print("Fitting regularized person-item model...")
    session_df, item_df, human_fitted, model_diag = fit_person_item_model(human_df)
    human_with_ability = human_fitted.merge(session_df, on="session_ID", how="left")
    session_summary = build_session_summary(human_with_ability, session_df)
    item_summary = build_item_summary(human_with_ability, item_df, pair_meta)
    overall_summary = build_overall_summary(human_with_ability, session_summary, item_summary, task_meta, model_diag)
    task_bias_summary = build_sampling_bias_table(task_meta, human_with_ability)

    print("Building LM overlap matrix...")
    lm_matrix, model_summary = load_lm_pair_matrix()
    comparison, lm_summary = build_human_lm_comparison(item_summary, lm_matrix, model_summary)

    print("Writing tables...")
    save_table(overall_summary, TABLES_DIR / "overall_summary.csv")
    save_table(session_summary, TABLES_DIR / "session_summary.csv")
    save_table(item_summary, TABLES_DIR / "item_summary.csv")
    save_table(task_bias_summary, TABLES_DIR / "sampling_bias_summary.csv")
    save_table(model_summary, TABLES_DIR / "model_pair_summary.csv")
    save_table(comparison, TABLES_DIR / "public_eval_human_vs_models.csv")

    print("Rendering figures...")
    plot_response_matrix(session_summary, item_summary, human_with_ability)
    plot_split_performance(human_with_ability)
    plot_session_profile(session_summary)
    plot_item_profile(item_summary)
    plot_sampling_bias(item_summary, task_bias_summary)
    plot_discrimination(item_summary)
    plot_human_vs_lm(comparison, lm_summary)

    print("Rendering task galleries...")
    hardest, easiest = generate_task_galleries(item_summary, comparison, task_cache)

    print("Writing narrative report...")
    write_report(overall_summary, item_summary, task_bias_summary, comparison, lm_summary, hardest, easiest)
    print(f"Done. Outputs are in: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
