#!/usr/bin/env python3
"""Generate all experiment figures from CSV results.

Reads sim_level.csv, condition_summary.csv, and task_pass_rates.csv
from --results-dir (default: PROJECT_ROOT/results) and writes six PNG
figures to results/figures/.
"""

import argparse
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# -- Style constants ----------------------------------------------------------

AGENT_PALETTE = {
    "baseline": "#888888",
    "declarative": "#4878CF",
    "imperative": "#E8792B",
}
AGENT_ORDER = ["baseline", "declarative", "imperative"]
AGENT_LABELS = {"baseline": "Baseline", "declarative": "Declarative", "imperative": "Imperative"}
RETRIEVAL_ORDER = ["golden_retrieval", "bm25"]
RETRIEVAL_LABELS = {"golden_retrieval": "Golden Retrieval", "bm25": "BM25"}

DPI = 300
FIG_WIDTH = 7  # inches, double-column
LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 12


def _apply_style():
    """Set global matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": DPI,
    })


# -- Bootstrap CI helper ------------------------------------------------------

def _pass_hat_k(n_successes: int, n_trials: int, k: int) -> float:
    """Compute Pass^k for a single task: C(n_successes, k) / C(n_trials, k)."""
    if n_trials < k or n_successes < 0:
        return 0.0
    return comb(n_successes, k) / comb(n_trials, k)


def _bootstrap_pass_k(
    sim_df: pd.DataFrame,
    agent_type: str,
    retrieval: str,
    k: int,
    n_iter: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI for Pass^k over tasks.

    Returns (point_estimate, ci_low, ci_high).
    """
    cond = sim_df[(sim_df["agent_type"] == agent_type) & (sim_df["retrieval"] == retrieval)]
    task_stats = (
        cond.groupby("task_id")
        .agg(n_trials=("success", "size"), n_successes=("success", "sum"))
        .reset_index()
    )
    task_ids = task_stats["task_id"].values
    n_tasks = len(task_ids)
    if n_tasks == 0:
        return 0.0, 0.0, 0.0

    # Point estimate
    task_stats["pass_k"] = task_stats.apply(
        lambda r: _pass_hat_k(int(r["n_successes"]), int(r["n_trials"]), k), axis=1,
    )
    point = task_stats["pass_k"].mean()

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_iter)
    pass_k_arr = task_stats["pass_k"].values
    for i in range(n_iter):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        boot_means[i] = pass_k_arr[idx].mean()

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return point, ci_low, ci_high


# -- Figure builders ----------------------------------------------------------

def _save(fig: plt.Figure, path: Path, name: str):
    """Save figure, print path, close."""
    out = path / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)


def plot_pass_k_bars(
    sim_df: pd.DataFrame,
    k: int,
    out_dir: Path,
    filename: str,
    title: str,
    *,
    ylim: float = 1.05,
    ytick_interval: float | None = None,
):
    """Grouped bar chart of Pass^k with bootstrap 95% CI."""
    rows = []
    for agent in AGENT_ORDER:
        for ret in RETRIEVAL_ORDER:
            point, ci_lo, ci_hi = _bootstrap_pass_k(sim_df, agent, ret, k)
            rows.append({
                "agent_type": AGENT_LABELS[agent],
                "retrieval": RETRIEVAL_LABELS[ret],
                "pass_k": point,
                "ci_lo": point - ci_lo,
                "ci_hi": ci_hi - point,
            })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))

    x = np.arange(len(AGENT_ORDER))
    width = 0.35
    for i, ret in enumerate(RETRIEVAL_ORDER):
        sub = plot_df[plot_df["retrieval"] == RETRIEVAL_LABELS[ret]]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            sub["pass_k"],
            width,
            yerr=[sub["ci_lo"].values, sub["ci_hi"].values],
            label=RETRIEVAL_LABELS[ret],
            color=[AGENT_PALETTE[a] for a in AGENT_ORDER],
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            alpha=0.85 if i == 0 else 0.55,
        )
        # Annotate bar values
        label_offset = ylim * 0.02
        for bar, val in zip(bars, sub["pass_k"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENT_ORDER])
    ax.set_ylabel(f"Pass^{k}")
    ax.set_ylim(0, ylim)
    tick_step = ytick_interval or (0.1 if ylim > 0.5 else 0.02)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.set_title(title)
    ax.legend(title="Retrieval")
    sns.despine(left=True)

    _save(fig, out_dir, filename)


def plot_task_heatmap(task_df: pd.DataFrame, out_dir: Path, *, solvable_only: bool = False):
    """Heatmap of per-task pass rates across 6 conditions."""
    col_order = [
        "baseline_golden", "declarative_golden", "imperative_golden",
        "baseline_bm25", "declarative_bm25", "imperative_bm25",
    ]
    col_labels = [
        "Baseline\nGolden", "Declarative\nGolden", "Imperative\nGolden",
        "Baseline\nBM25", "Declarative\nBM25", "Imperative\nBM25",
    ]

    df = task_df.copy()

    if solvable_only:
        df = df[df["mean_pass_rate"] > 0]

    # Sort by mean pass rate, highest at top
    df = df.sort_values("mean_pass_rate", ascending=True)
    matrix = df[col_order].values
    task_labels = df["task_id"].values

    n_tasks = len(task_labels)
    fig_height = max(4, n_tasks * 0.3)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    # Discrete colormap: 0=red, 0.33=orange, 0.67=light green, 1.0=green
    colors = ["#d9534f", "#f0ad4e", "#a8d97f", "#5cb85c"]
    cmap = ListedColormap(colors)
    bounds = [0, 0.165, 0.5, 0.835, 1.01]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Annotate cells
    frac_labels = {0.0: "0", 1 / 3: "\u2153", 2 / 3: "\u2154", 1.0: "1"}
    for i in range(n_tasks):
        for j in range(len(col_order)):
            val = matrix[i, j]
            closest = min(frac_labels.keys(), key=lambda fv: abs(fv - val))
            label = frac_labels[closest]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_labels, fontsize=7)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Task ID")
    n_excluded = len(task_df) - n_tasks if solvable_only else 0
    title = "Per-Task Pass Rate — Solvable Tasks" if solvable_only else "Per-Task Pass Rate Across Conditions"
    ax.set_title(title)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, ticks=[0, 0.33, 0.67, 1.0])
    cbar.ax.set_yticklabels(["0", "⅓", "⅔", "1"])
    cbar.set_label("Pass Rate")

    if solvable_only and n_excluded > 0:
        ax.text(
            0.5, -0.08, f"{n_excluded} of {len(task_df)} tasks never solved under any condition (not shown).",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#666666",
        )

    suffix = "_solvable" if solvable_only else ""
    _save(fig, out_dir, f"task_heatmap{suffix}.png")


def plot_tool_calls_boxplot(sim_df: pd.DataFrame, out_dir: Path):
    """Box plot of tool calls per successful task."""
    success_df = sim_df[sim_df["success"] == True].copy()  # noqa: E712
    success_df["retrieval_label"] = success_df["retrieval"].map(RETRIEVAL_LABELS)
    success_df["agent_label"] = success_df["agent_type"].map(AGENT_LABELS)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))

    # Check for low-count conditions
    counts = success_df.groupby(["agent_type", "retrieval"]).size()
    low_conditions = counts[counts < 10]
    subtitle = ""
    if len(low_conditions) > 0:
        names = [f"{AGENT_LABELS[a]}/{RETRIEVAL_LABELS[r]}" for a, r in low_conditions.index]
        subtitle = f"Note: <10 successes in {', '.join(names)}"

    sns.boxplot(
        data=success_df,
        x="retrieval_label",
        y="num_tool_calls",
        hue="agent_label",
        hue_order=[AGENT_LABELS[a] for a in AGENT_ORDER],
        palette=[AGENT_PALETTE[a] for a in AGENT_ORDER],
        order=[RETRIEVAL_LABELS[r] for r in RETRIEVAL_ORDER],
        ax=ax,
        fliersize=2,
        linewidth=0.8,
    )

    ax.set_xlabel("Retrieval Config")
    ax.set_ylabel("Tool Calls")
    ax.set_title("Tool Calls per Successful Task")
    if subtitle:
        ax.text(
            0.5, -0.12, subtitle, transform=ax.transAxes,
            ha="center", fontsize=8, style="italic", color="#666666",
        )
    ax.legend(title="Agent Type")
    sns.despine(left=True)

    _save(fig, out_dir, "tool_calls_boxplot.png")


def plot_turns_boxplot(sim_df: pd.DataFrame, out_dir: Path):
    """Box plot of conversation turns (all simulations)."""
    df = sim_df.copy()
    df["retrieval_label"] = df["retrieval"].map(RETRIEVAL_LABELS)
    df["agent_label"] = df["agent_type"].map(AGENT_LABELS)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))

    sns.boxplot(
        data=df,
        x="retrieval_label",
        y="num_turns",
        hue="agent_label",
        hue_order=[AGENT_LABELS[a] for a in AGENT_ORDER],
        palette=[AGENT_PALETTE[a] for a in AGENT_ORDER],
        order=[RETRIEVAL_LABELS[r] for r in RETRIEVAL_ORDER],
        ax=ax,
        fliersize=2,
        linewidth=0.8,
    )

    ax.set_xlabel("Retrieval Config")
    ax.set_ylabel("Number of Turns")
    ax.set_title("Conversation Turns by Condition")
    ax.legend(title="Agent Type")
    sns.despine(left=True)

    _save(fig, out_dir, "turns_boxplot.png")


def plot_error_categories(error_summary_path: Path, out_dir: Path):
    """Grouped bar chart of error category frequencies by condition."""
    df = pd.read_csv(error_summary_path)

    # Filter to meaningful error categories (skip E5 = all zeros, total row, and E8_exact_loop)
    keep = ["E2_action_mismatch", "E4_hallucinated_tool", "E7_premature_term",
            "E8_name_loop", "E9_discoverable_fail"]
    df = df[df["error_type"].isin(keep)].copy()

    # Combine E8 variants into one row if both exist
    error_labels = {
        "E2_action_mismatch": "E2: Action\nmismatch",
        "E4_hallucinated_tool": "E4: Hallucinated\ntool",
        "E7_premature_term": "E7: Premature\ntermination",
        "E8_name_loop": "E8: Excessive\nlooping",
        "E9_discoverable_fail": "E9: Discoverable\ntool failure",
    }

    condition_cols = [
        "baseline_golden", "baseline_bm25",
        "declarative_golden", "declarative_bm25",
        "imperative_golden", "imperative_bm25",
    ]
    condition_labels = [
        "Base+G", "Base+B", "Dec+G", "Dec+B", "Imp+G", "Imp+B",
    ]
    condition_colors = [
        AGENT_PALETTE["baseline"], AGENT_PALETTE["baseline"],
        AGENT_PALETTE["declarative"], AGENT_PALETTE["declarative"],
        AGENT_PALETTE["imperative"], AGENT_PALETTE["imperative"],
    ]
    condition_alphas = [0.85, 0.50, 0.85, 0.50, 0.85, 0.50]

    n_errors = len(df)
    n_conditions = len(condition_cols)
    x = np.arange(n_errors)
    bar_width = 0.12

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.5))

    for i, (col, label, color, alpha) in enumerate(
        zip(condition_cols, condition_labels, condition_colors, condition_alphas)
    ):
        offset = (i - (n_conditions - 1) / 2) * bar_width
        vals = df[col].values.astype(float)
        ax.bar(x + offset, vals, bar_width, label=label, color=color,
               alpha=alpha, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([error_labels[e] for e in df["error_type"]], fontsize=8)
    ax.set_ylabel("Frequency (count)")
    ax.set_title("Error Category Distribution by Condition")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    sns.despine(left=True)

    _save(fig, out_dir, "error_categories.png")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate experiment figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing CSV result files (default: PROJECT_ROOT/results)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    # Load data
    sim_df = pd.read_csv(results_dir / "sim_level.csv")
    cond_df = pd.read_csv(results_dir / "condition_summary.csv")
    task_df = pd.read_csv(results_dir / "task_pass_rates.csv")

    print(f"Loaded {len(sim_df)} simulations, {len(cond_df)} conditions, {len(task_df)} tasks")
    print(f"Saving figures to {fig_dir}/\n")

    # Figure 2: Pass^1 bar chart (full y-axis)
    plot_pass_k_bars(sim_df, k=1, out_dir=fig_dir, filename="pass1_by_condition.png",
                     title="Pass^1 by Agent Type and Retrieval Config")

    # Figure 3: Pass^3 bar chart (zoomed y-axis to show small values)
    plot_pass_k_bars(sim_df, k=3, out_dir=fig_dir, filename="pass3_by_condition.png",
                     title="Pass^3 by Agent Type and Retrieval Config",
                     ylim=0.20, ytick_interval=0.02)

    # Figure 4a: Tool calls box plot (successes only)
    plot_tool_calls_boxplot(sim_df, fig_dir)

    # Figure 4b: Turns box plot (all sims)
    plot_turns_boxplot(sim_df, fig_dir)

    # Figure 5: Task heatmap (solvable tasks only)
    plot_task_heatmap(task_df, fig_dir, solvable_only=True)

    # Figure 6: Error category distribution
    error_summary_path = results_dir / "error_summary.csv"
    if error_summary_path.exists():
        plot_error_categories(error_summary_path, fig_dir)
    else:
        print(f"  SKIP error_categories: {error_summary_path} not found")

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
