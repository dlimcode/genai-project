#!/usr/bin/env python3
"""Hypothesis testing for 3x2 factorial experiment (agent_type x retrieval).

Reads CSVs from extract_metrics.py, runs pre-registered statistical tests,
and assesses hypotheses H1-H5. Produces:
  - results/statistical_tests.csv
  - results/hypothesis_assessment.csv
  - Formatted summary to stdout
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AGENT_TYPES = ["baseline", "declarative", "imperative"]
RETRIEVAL_CONFIGS = ["golden_retrieval", "bm25"]

# Condition names match experiment.yaml keys (agent_type + short retrieval suffix)
CONDITIONS = [
    "baseline_golden", "baseline_bm25",
    "declarative_golden", "declarative_bm25",
    "imperative_golden", "imperative_bm25",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Cramer's V effect size for chi-squared test."""
    denom = n * (min(r, c) - 1)
    if denom == 0:
        return 0.0
    return math.sqrt(chi2 / denom)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def interpret_cohens_h(h: float) -> str:
    """Interpret absolute Cohen's h magnitude."""
    if math.isnan(h):
        return "N/A"
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    elif ah < 0.5:
        return "small"
    elif ah < 0.8:
        return "medium"
    else:
        return "large"


def interpret_cramers_v(v: float) -> str:
    """Interpret Cramer's V effect size magnitude."""
    if math.isnan(v):
        return "N/A"
    av = abs(v)
    if av < 0.1:
        return "negligible"
    elif av < 0.3:
        return "small"
    elif av < 0.5:
        return "medium"
    else:
        return "large"


def pass_k(successes: int, trials: int, k: int) -> float:
    """Compute Pass^k = C(successes, k) / C(trials, k).

    Returns 0.0 if trials < k or successes < 0.
    """
    if trials < k or successes < 0:
        return 0.0
    return math.comb(successes, k) / math.comb(trials, k)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns adjusted p-values (same order as input).
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keep track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        # Enforce monotonicity: adjusted p can't decrease as rank increases
        cumulative_max = max(cumulative_max, corrected)
        adjusted[orig_idx] = min(cumulative_max, 1.0)

    return adjusted


# ---------------------------------------------------------------------------
# Chi-squared tests
# ---------------------------------------------------------------------------


def chi_squared_two_way(sim_df: pd.DataFrame) -> dict:
    """6-cell chi-squared test: success/failure across all 6 conditions.

    Tests overall heterogeneity in success rates across the 3x2 design.
    Uses condition (not agent_type x retrieval separately) to build a proper
    contingency table of [successes, failures] per condition.
    """
    ct = sim_df.groupby("condition")["success"].agg(["sum", "count"])
    ct.columns = ["success", "total"]
    ct["fail"] = ct["total"] - ct["success"]
    ct = ct.reindex([c for c in CONDITIONS if c in ct.index])

    table = ct[["success", "fail"]].values
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(ct["total"].sum())
    r, c = table.shape
    v = cramers_v(chi2, n, r, c) if n > 0 else 0.0

    return {
        "test_name": "chi2_overall_6_conditions",
        "statistic": chi2,
        "p_value": p,
        "dof": dof,
        "effect_size": v,
        "effect_size_name": "Cramers_V",
        "interpretation": f"Cramer's V = {v:.3f} ({interpret_cramers_v(v)})",
    }


def chi_squared_agent_main_effect(sim_df: pd.DataFrame) -> dict:
    """1-way chi-squared: 3 agent types pooled across retrieval."""
    ct = sim_df.groupby("agent_type")["success"].agg(["sum", "count"])
    ct.columns = ["success", "total"]
    ct["fail"] = ct["total"] - ct["success"]
    ct = ct.reindex(AGENT_TYPES)

    table = ct[["success", "fail"]].values
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = ct["total"].sum()
    r, c = table.shape
    v = cramers_v(chi2, int(n), r, c) if n > 0 else 0.0

    return {
        "test_name": "chi2_agent_main_effect",
        "statistic": chi2,
        "p_value": p,
        "dof": dof,
        "effect_size": v,
        "effect_size_name": "Cramers_V",
        "interpretation": f"Cramer's V = {v:.3f} ({interpret_cramers_v(v)})",
    }


def chi_squared_retrieval_main_effect(sim_df: pd.DataFrame) -> dict:
    """1-way chi-squared: 2 retrieval types pooled across agent."""
    ct = sim_df.groupby("retrieval")["success"].agg(["sum", "count"])
    ct.columns = ["success", "total"]
    ct["fail"] = ct["total"] - ct["success"]
    ct = ct.reindex(RETRIEVAL_CONFIGS)

    table = ct[["success", "fail"]].values
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = ct["total"].sum()
    r, c = table.shape
    v = cramers_v(chi2, int(n), r, c) if n > 0 else 0.0

    return {
        "test_name": "chi2_retrieval_main_effect",
        "statistic": chi2,
        "p_value": p,
        "dof": dof,
        "effect_size": v,
        "effect_size_name": "Cramers_V",
        "interpretation": f"Cramer's V = {v:.3f} ({interpret_cramers_v(v)})",
    }


# ---------------------------------------------------------------------------
# McNemar's test (paired same-task comparisons)
# ---------------------------------------------------------------------------


def _mcnemar_test(table_2x2: np.ndarray) -> tuple[float, float]:
    """Run McNemar's test on a 2x2 table [[both_pass, a_only], [b_only, both_fail]].

    Uses exact binomial test if discordant count < 25, chi-squared otherwise.
    Returns (statistic, p_value).
    """
    b = table_2x2[0, 1]  # A pass, B fail
    c = table_2x2[1, 0]  # A fail, B pass

    if b + c == 0:
        return 0.0, 1.0

    if b + c < 25:
        # Exact binomial test: under H0, b ~ Binomial(b+c, 0.5)
        p = stats.binomtest(b, b + c, 0.5).pvalue
        return float(b), p
    else:
        chi2 = (b - c) ** 2 / (b + c)
        p = stats.chi2.sf(chi2, df=1)
        return chi2, p


def mcnemar_paired(task_pass_df: pd.DataFrame, cond_a: str, cond_b: str) -> dict:
    """McNemar's test comparing two conditions on per-task pass/fail.

    Each task is binarized as pass (pass_rate > 0) or fail (pass_rate == 0).
    """
    a_pass = task_pass_df[cond_a] > 0
    b_pass = task_pass_df[cond_b] > 0

    both_pass = (a_pass & b_pass).sum()
    a_only = (a_pass & ~b_pass).sum()
    b_only = (~a_pass & b_pass).sum()
    both_fail = (~a_pass & ~b_pass).sum()

    table = np.array([[both_pass, a_only], [b_only, both_fail]])
    stat, p = _mcnemar_test(table)

    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "statistic": stat,
        "p_raw": p,
        "both_pass": int(both_pass),
        "a_only": int(a_only),
        "b_only": int(b_only),
        "both_fail": int(both_fail),
    }


def mcnemar_all_pairs(task_pass_df: pd.DataFrame) -> pd.DataFrame:
    """Run McNemar's test for all 15 pairwise condition comparisons.

    Applies Holm-Bonferroni correction.
    """
    # Identify condition columns (exclude task_id and mean_pass_rate)
    cond_cols = [c for c in task_pass_df.columns if c not in ("task_id", "mean_pass_rate")]
    pairs = list(combinations(cond_cols, 2))

    rows = []
    for cond_a, cond_b in pairs:
        result = mcnemar_paired(task_pass_df, cond_a, cond_b)
        rows.append(result)

    df = pd.DataFrame(rows)
    df["p_adjusted"] = holm_bonferroni(df["p_raw"].tolist())
    df["significant"] = df["p_adjusted"] < 0.05

    return df


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_pass_k_ci(
    sim_df: pd.DataFrame,
    k: int,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster bootstrap CI for Pass^k, resampling at task level.

    All trials for a resampled task move together (accounts for within-task correlation).
    """
    rng = np.random.default_rng(seed)
    task_ids = sim_df["task_id"].unique()
    n_tasks = len(task_ids)

    # Pre-compute per-task, per-condition success counts and trial counts
    grouped = sim_df.groupby(["condition", "task_id"])["success"].agg(["sum", "count"]).reset_index()
    grouped.columns = ["condition", "task_id", "successes", "trials"]

    conditions = sorted(sim_df["condition"].unique())

    # Build lookup: (condition, task_id) -> (successes, trials)
    lookup: dict[tuple[str, str], tuple[int, int]] = {}
    for _, row in grouped.iterrows():
        lookup[(row["condition"], row["task_id"])] = (int(row["successes"]), int(row["trials"]))

    boot_results: dict[str, list[float]] = {c: [] for c in conditions}

    for _ in range(n_boot):
        sampled_tasks = rng.choice(task_ids, size=n_tasks, replace=True)

        for cond in conditions:
            total_pass_k = 0.0
            n_valid = 0
            for tid in sampled_tasks:
                key = (cond, tid)
                if key in lookup:
                    s, t = lookup[key]
                    total_pass_k += pass_k(s, t, k)
                    n_valid += 1
            boot_results[cond].append(total_pass_k / n_valid if n_valid > 0 else 0.0)

    rows = []
    for cond in conditions:
        samples = np.array(boot_results[cond])
        rows.append({
            "condition": cond,
            f"pass_{k}_mean": np.mean(samples),
            f"pass_{k}_ci_lower": np.percentile(samples, 100 * alpha / 2),
            f"pass_{k}_ci_upper": np.percentile(samples, 100 * (1 - alpha / 2)),
        })

    return pd.DataFrame(rows)


def bootstrap_pass_k_difference_ci(
    sim_df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    k: int,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for Pass^k(A) - Pass^k(B).

    p_value = proportion of bootstrap samples where the sign of the difference
    is opposite to the observed difference (two-sided).
    """
    rng = np.random.default_rng(seed)
    task_ids = sim_df["task_id"].unique()
    n_tasks = len(task_ids)

    # Pre-compute per-task success counts
    grouped = sim_df.groupby(["condition", "task_id"])["success"].agg(["sum", "count"]).reset_index()
    grouped.columns = ["condition", "task_id", "successes", "trials"]

    lookup: dict[tuple[str, str], tuple[int, int]] = {}
    for _, row in grouped.iterrows():
        lookup[(row["condition"], row["task_id"])] = (int(row["successes"]), int(row["trials"]))

    diffs: list[float] = []
    for _ in range(n_boot):
        sampled_tasks = rng.choice(task_ids, size=n_tasks, replace=True)

        pk_a = 0.0
        pk_b = 0.0
        n_a = 0
        n_b = 0
        for tid in sampled_tasks:
            key_a = (cond_a, tid)
            key_b = (cond_b, tid)
            if key_a in lookup:
                s, t = lookup[key_a]
                pk_a += pass_k(s, t, k)
                n_a += 1
            if key_b in lookup:
                s, t = lookup[key_b]
                pk_b += pass_k(s, t, k)
                n_b += 1

        mean_a = pk_a / n_a if n_a > 0 else 0.0
        mean_b = pk_b / n_b if n_b > 0 else 0.0
        diffs.append(mean_a - mean_b)

    diffs_arr = np.array(diffs)
    observed_diff = np.mean(diffs_arr)

    # Two-sided p: proportion of samples on the opposite side of zero
    if observed_diff > 0:
        p_value = np.mean(diffs_arr <= 0)
    elif observed_diff < 0:
        p_value = np.mean(diffs_arr >= 0)
    else:
        p_value = 1.0

    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "k": k,
        "observed_diff": observed_diff,
        "ci_lower": float(np.percentile(diffs_arr, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(diffs_arr, 100 * (1 - alpha / 2))),
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Effect size computations
# ---------------------------------------------------------------------------


def compute_pairwise_cohens_h(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's h for all 15 pairwise Pass^1 and Pass^3 comparisons."""
    conditions = summary_df["condition"].tolist()
    pairs = list(combinations(conditions, 2))

    rows = []
    for cond_a, cond_b in pairs:
        row_a = summary_df[summary_df["condition"] == cond_a].iloc[0]
        row_b = summary_df[summary_df["condition"] == cond_b].iloc[0]

        for metric in ["pass_1", "pass_3"]:
            p1 = row_a[metric]
            p2 = row_b[metric]
            h = cohens_h(p1, p2)
            rows.append({
                "cond_a": cond_a,
                "cond_b": cond_b,
                "metric": metric,
                "p_a": p1,
                "p_b": p2,
                "cohens_h": h,
                "abs_h": abs(h),
                "interpretation": interpret_cohens_h(h),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------


def evaluate_hypotheses(
    summary_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    effect_sizes_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assess each pre-registered hypothesis H1-H5."""
    results = []

    # Helper to get pass rate for a condition
    def get_pass(cond: str, metric: str = "pass_1") -> float:
        row = summary_df[summary_df["condition"] == cond]
        if row.empty:
            return 0.0
        return float(row.iloc[0][metric])

    # Helper to check if McNemar comparison is significant
    def is_significant(cond_a: str, cond_b: str) -> bool:
        mask = ((mcnemar_df["cond_a"] == cond_a) & (mcnemar_df["cond_b"] == cond_b)) | \
               ((mcnemar_df["cond_a"] == cond_b) & (mcnemar_df["cond_b"] == cond_a))
        match = mcnemar_df[mask]
        if match.empty:
            return False
        return bool(match.iloc[0]["significant"])

    # Helper to get Cohen's h between two conditions
    def get_h(cond_a: str, cond_b: str, metric: str = "pass_1") -> float:
        mask = (
            ((effect_sizes_df["cond_a"] == cond_a) & (effect_sizes_df["cond_b"] == cond_b)) |
            ((effect_sizes_df["cond_a"] == cond_b) & (effect_sizes_df["cond_b"] == cond_a))
        ) & (effect_sizes_df["metric"] == metric)
        match = effect_sizes_df[mask]
        if match.empty:
            return 0.0
        h = float(match.iloc[0]["cohens_h"])
        # Ensure sign is cond_a - cond_b
        if match.iloc[0]["cond_a"] == cond_b:
            h = -h
        return h

    # --- H1: Imperative > Declarative > Baseline under bm25 ---
    imp_bm25 = get_pass("imperative_bm25")
    dec_bm25 = get_pass("declarative_bm25")
    base_bm25 = get_pass("baseline_bm25")
    ordered = imp_bm25 > dec_bm25 > base_bm25

    imp_vs_dec_sig = is_significant("imperative_bm25", "declarative_bm25")
    dec_vs_base_sig = is_significant("declarative_bm25", "baseline_bm25")

    if ordered and imp_vs_dec_sig and dec_vs_base_sig:
        h1_status = "confirmed"
    elif ordered:
        h1_status = "directional_support"
    elif imp_bm25 > base_bm25:
        h1_status = "partially_supported"
    else:
        h1_status = "rejected"

    h1_h = get_h("imperative_bm25", "baseline_bm25")
    results.append({
        "hypothesis": "H1",
        "description": "Imperative > Declarative > Baseline under bm25",
        "status": h1_status,
        "evidence": (
            f"Pass^1 bm25: imp={imp_bm25:.3f}, dec={dec_bm25:.3f}, base={base_bm25:.3f}; "
            f"ordered={ordered}; imp>dec sig={imp_vs_dec_sig}; dec>base sig={dec_vs_base_sig}"
        ),
        "effect_size": f"h(imp-base)={h1_h:.3f} ({interpret_cohens_h(h1_h)})",
    })

    # --- H2: Performance gap shrinks under golden_retrieval ---
    gap_bm25_imp_base = imp_bm25 - base_bm25
    gap_golden_imp_base = get_pass("imperative_golden") - get_pass("baseline_golden")

    gap_bm25_dec_base = dec_bm25 - base_bm25
    gap_golden_dec_base = get_pass("declarative_golden") - get_pass("baseline_golden")

    gaps_shrink = (abs(gap_golden_imp_base) < abs(gap_bm25_imp_base)) and \
                  (abs(gap_golden_dec_base) < abs(gap_bm25_dec_base))

    if gaps_shrink:
        h2_status = "confirmed"
    elif abs(gap_golden_imp_base) < abs(gap_bm25_imp_base) or abs(gap_golden_dec_base) < abs(gap_bm25_dec_base):
        h2_status = "partially_supported"
    else:
        h2_status = "rejected"

    results.append({
        "hypothesis": "H2",
        "description": "Performance gap shrinks under golden_retrieval",
        "status": h2_status,
        "evidence": (
            f"Gap(imp-base): bm25={gap_bm25_imp_base:.3f}, golden={gap_golden_imp_base:.3f}; "
            f"Gap(dec-base): bm25={gap_bm25_dec_base:.3f}, golden={gap_golden_dec_base:.3f}; "
            f"both_shrink={gaps_shrink}"
        ),
        "effect_size": (
            f"gap_reduction_imp={abs(gap_bm25_imp_base) - abs(gap_golden_imp_base):.3f}, "
            f"gap_reduction_dec={abs(gap_bm25_dec_base) - abs(gap_golden_dec_base):.3f}"
        ),
    })

    # --- H3: bm25 hurts declarative more than imperative ---
    dec_drop = get_pass("declarative_golden") - get_pass("declarative_bm25")
    imp_drop = get_pass("imperative_golden") - get_pass("imperative_bm25")

    if dec_drop > imp_drop and dec_drop > 0:
        h3_status = "confirmed"
    elif dec_drop > imp_drop:
        h3_status = "directional_support"
    else:
        h3_status = "rejected"

    results.append({
        "hypothesis": "H3",
        "description": "bm25 hurts declarative more than imperative",
        "status": h3_status,
        "evidence": (
            f"Dec drop (golden-bm25)={dec_drop:.3f}; "
            f"Imp drop (golden-bm25)={imp_drop:.3f}; "
            f"dec_drop > imp_drop = {dec_drop > imp_drop}"
        ),
        "effect_size": f"differential_drop={dec_drop - imp_drop:.3f}",
    })

    # --- H4: Imperative uses fewer tool calls on successful tasks ---
    successful = sim_df[sim_df["success"] == 1]
    if not successful.empty:
        imp_tools = successful[successful["agent_type"] == "imperative"]["num_tool_calls"]
        dec_tools = successful[successful["agent_type"] == "declarative"]["num_tool_calls"]
        base_tools = successful[successful["agent_type"] == "baseline"]["num_tool_calls"]

        # Mann-Whitney U: imperative vs others
        h4_evidence_parts = []

        imp_median = imp_tools.median() if len(imp_tools) > 0 else float("nan")
        dec_median = dec_tools.median() if len(dec_tools) > 0 else float("nan")
        base_median = base_tools.median() if len(base_tools) > 0 else float("nan")

        h4_evidence_parts.append(
            f"Median tool calls (successful): imp={imp_median:.1f}, dec={dec_median:.1f}, base={base_median:.1f}"
        )

        imp_fewer = True
        h4_effect = ""

        if len(imp_tools) > 0 and len(dec_tools) > 0:
            u_dec, p_dec = stats.mannwhitneyu(imp_tools, dec_tools, alternative="less")
            h4_evidence_parts.append(f"MWU imp<dec: U={u_dec:.0f}, p={p_dec:.4f}")
            # Rank-biserial r as effect size
            n1, n2 = len(imp_tools), len(dec_tools)
            r_dec = 1 - (2 * u_dec) / (n1 * n2)
            h4_effect += f"r(imp_vs_dec)={r_dec:.3f}"
            if p_dec >= 0.05:
                imp_fewer = False
        else:
            imp_fewer = False

        if len(imp_tools) > 0 and len(base_tools) > 0:
            u_base, p_base = stats.mannwhitneyu(imp_tools, base_tools, alternative="less")
            h4_evidence_parts.append(f"MWU imp<base: U={u_base:.0f}, p={p_base:.4f}")
            n1, n2 = len(imp_tools), len(base_tools)
            r_base = 1 - (2 * u_base) / (n1 * n2)
            h4_effect += f", r(imp_vs_base)={r_base:.3f}"
            if p_base >= 0.05:
                imp_fewer = False
        else:
            imp_fewer = False

        h4_status = "confirmed" if imp_fewer else "rejected"
    else:
        h4_status = "inconclusive"
        h4_evidence_parts = ["No successful tasks to compare"]
        h4_effect = "N/A"

    results.append({
        "hypothesis": "H4",
        "description": "Imperative uses fewer tool calls on successful tasks",
        "status": h4_status,
        "evidence": "; ".join(h4_evidence_parts),
        "effect_size": h4_effect,
    })

    # --- H5: If declarative ~ baseline, skill files add no value ---
    h_dec_base_golden = get_h("declarative_golden", "baseline_golden")
    h_dec_base_bm25 = get_h("declarative_bm25", "baseline_bm25")

    dec_base_golden_sig = is_significant("declarative_golden", "baseline_golden")
    dec_base_bm25_sig = is_significant("declarative_bm25", "baseline_bm25")

    both_negligible = abs(h_dec_base_golden) < 0.2 and abs(h_dec_base_bm25) < 0.2
    neither_sig = not dec_base_golden_sig and not dec_base_bm25_sig

    if both_negligible and neither_sig:
        h5_status = "confirmed"
    elif both_negligible or neither_sig:
        h5_status = "partially_supported"
    else:
        h5_status = "rejected"

    results.append({
        "hypothesis": "H5",
        "description": "Declarative ~ Baseline (skill files add no value for this domain)",
        "status": h5_status,
        "evidence": (
            f"h(dec-base) golden={h_dec_base_golden:.3f} ({interpret_cohens_h(h_dec_base_golden)}), "
            f"bm25={h_dec_base_bm25:.3f} ({interpret_cohens_h(h_dec_base_bm25)}); "
            f"sig golden={dec_base_golden_sig}, sig bm25={dec_base_bm25_sig}"
        ),
        "effect_size": f"h_golden={h_dec_base_golden:.3f}, h_bm25={h_dec_base_bm25:.3f}",
    })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_all_tests(results_dir: str) -> None:
    """Run all statistical tests and produce output files."""
    sim_path = os.path.join(results_dir, "sim_level.csv")
    task_pass_path = os.path.join(results_dir, "task_pass_rates.csv")
    summary_path = os.path.join(results_dir, "condition_summary.csv")

    # Validate inputs exist
    for path, name in [(sim_path, "sim_level.csv"), (task_pass_path, "task_pass_rates.csv"), (summary_path, "condition_summary.csv")]:
        if not os.path.exists(path):
            print(f"ERROR: Required input file not found: {path}", file=sys.stderr)
            sys.exit(1)

    sim_df = pd.read_csv(sim_path)
    task_pass_df = pd.read_csv(task_pass_path)
    summary_df = pd.read_csv(summary_path)

    all_tests: list[dict] = []

    # ---- Chi-squared tests ----
    print("Running chi-squared tests...")
    all_tests.append(chi_squared_two_way(sim_df))
    all_tests.append(chi_squared_agent_main_effect(sim_df))
    all_tests.append(chi_squared_retrieval_main_effect(sim_df))

    # ---- McNemar's tests ----
    print("Running McNemar's pairwise tests...")
    mcnemar_df = mcnemar_all_pairs(task_pass_df)

    for _, row in mcnemar_df.iterrows():
        all_tests.append({
            "test_name": f"mcnemar_{row['cond_a']}_vs_{row['cond_b']}",
            "statistic": row["statistic"],
            "p_value": row["p_adjusted"],
            "effect_size": float("nan"),
            "effect_size_name": "N/A",
            "interpretation": f"{'significant' if row['significant'] else 'not significant'} (p_adj={row['p_adjusted']:.4f})",
        })

    # ---- Bootstrap CIs ----
    print("Running bootstrap CIs (Pass^1)...")
    boot_pass1 = bootstrap_pass_k_ci(sim_df, k=1)
    print("Running bootstrap CIs (Pass^3)...")
    boot_pass3 = bootstrap_pass_k_ci(sim_df, k=3)

    for _, row in boot_pass1.iterrows():
        all_tests.append({
            "test_name": f"bootstrap_pass1_ci_{row['condition']}",
            "statistic": row["pass_1_mean"],
            "p_value": float("nan"),
            "effect_size": float("nan"),
            "effect_size_name": "N/A",
            "interpretation": f"Pass^1 = {row['pass_1_mean']:.3f} [{row['pass_1_ci_lower']:.3f}, {row['pass_1_ci_upper']:.3f}]",
        })

    # Key pairwise bootstrap difference tests
    print("Running bootstrap difference CIs...")
    key_pairs = [
        ("imperative_bm25", "declarative_bm25"),
        ("declarative_bm25", "baseline_bm25"),
        ("imperative_bm25", "baseline_bm25"),
        ("imperative_golden", "baseline_golden"),
        ("declarative_golden", "baseline_golden"),
    ]
    bootstrap_diffs: list[dict] = []
    for cond_a, cond_b in key_pairs:
        diff = bootstrap_pass_k_difference_ci(sim_df, cond_a, cond_b, k=1)
        bootstrap_diffs.append(diff)
        all_tests.append({
            "test_name": f"bootstrap_diff_pass1_{cond_a}_vs_{cond_b}",
            "statistic": diff["observed_diff"],
            "p_value": diff["p_value"],
            "effect_size": float("nan"),
            "effect_size_name": "N/A",
            "interpretation": (
                f"diff = {diff['observed_diff']:.3f} "
                f"[{diff['ci_lower']:.3f}, {diff['ci_upper']:.3f}], "
                f"p = {diff['p_value']:.4f}"
            ),
        })

    # ---- Effect sizes ----
    print("Computing pairwise Cohen's h...")
    effect_sizes_df = compute_pairwise_cohens_h(summary_df)

    for _, row in effect_sizes_df.iterrows():
        all_tests.append({
            "test_name": f"cohens_h_{row['metric']}_{row['cond_a']}_vs_{row['cond_b']}",
            "statistic": row["cohens_h"],
            "p_value": float("nan"),
            "effect_size": row["abs_h"],
            "effect_size_name": "abs_cohens_h",
            "interpretation": f"|h| = {row['abs_h']:.3f} ({row['interpretation']})",
        })

    # ---- Save statistical_tests.csv ----
    tests_df = pd.DataFrame(all_tests)
    tests_out = os.path.join(results_dir, "statistical_tests.csv")
    tests_df.to_csv(tests_out, index=False)
    print(f"\nSaved: {tests_out}")

    # ---- Hypothesis evaluation ----
    print("\nEvaluating hypotheses H1-H5...")
    hyp_df = evaluate_hypotheses(summary_df, mcnemar_df, effect_sizes_df, sim_df)
    hyp_out = os.path.join(results_dir, "hypothesis_assessment.csv")
    hyp_df.to_csv(hyp_out, index=False)
    print(f"Saved: {hyp_out}")

    # ---- Print formatted summary ----
    print_summary(summary_df, tests_df, mcnemar_df, boot_pass1, boot_pass3, effect_sizes_df, hyp_df)


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------


def print_summary(
    summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    boot_pass1: pd.DataFrame,
    boot_pass3: pd.DataFrame,
    effect_sizes_df: pd.DataFrame,
    hyp_df: pd.DataFrame,
) -> None:
    """Print a clear formatted summary of all results."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("STATISTICAL ANALYSIS SUMMARY")
    print(f"{sep}")

    # -- Condition pass rates --
    print("\n--- Condition Pass Rates ---")
    for _, row in summary_df.iterrows():
        print(f"  {row['condition']:30s}  Pass^1={row['pass_1']:.3f}  Pass^3={row['pass_3']:.3f}")

    # -- Chi-squared tests --
    print("\n--- Chi-Squared Tests ---")
    chi2_tests = tests_df[tests_df["test_name"].str.startswith("chi2_")]
    for _, row in chi2_tests.iterrows():
        print(f"  {row['test_name']:40s}  chi2={row['statistic']:.2f}  p={row['p_value']:.4f}  {row['interpretation']}")

    # -- McNemar's pairwise comparisons --
    print("\n--- McNemar's Pairwise Tests (Holm-Bonferroni corrected) ---")
    sig_pairs = mcnemar_df[mcnemar_df["significant"]]
    nonsig_pairs = mcnemar_df[~mcnemar_df["significant"]]

    if not sig_pairs.empty:
        print("  Significant pairs:")
        for _, row in sig_pairs.iterrows():
            print(f"    {row['cond_a']} vs {row['cond_b']}: p_adj={row['p_adjusted']:.4f} "
                  f"(disc: {row['a_only']}|{row['b_only']})")
    else:
        print("  No significant pairs after correction.")

    print(f"  ({len(nonsig_pairs)} non-significant pairs omitted)")

    # -- Bootstrap CIs --
    print("\n--- Bootstrap 95% CIs ---")
    print("  Pass^1:")
    for _, row in boot_pass1.iterrows():
        print(f"    {row['condition']:30s}  {row['pass_1_mean']:.3f} [{row['pass_1_ci_lower']:.3f}, {row['pass_1_ci_upper']:.3f}]")
    print("  Pass^3:")
    for _, row in boot_pass3.iterrows():
        print(f"    {row['condition']:30s}  {row['pass_3_mean']:.3f} [{row['pass_3_ci_lower']:.3f}, {row['pass_3_ci_upper']:.3f}]")

    # -- Key effect sizes --
    print("\n--- Key Effect Sizes (Cohen's h, Pass^1) ---")
    key_effects = effect_sizes_df[effect_sizes_df["metric"] == "pass_1"].sort_values("abs_h", ascending=False)
    for _, row in key_effects.head(10).iterrows():
        print(f"  {row['cond_a']:25s} vs {row['cond_b']:25s}  h={row['cohens_h']:+.3f} ({row['interpretation']})")

    # -- Hypothesis assessments --
    print(f"\n{sep}")
    print("HYPOTHESIS ASSESSMENT")
    print(sep)

    status_markers = {
        "confirmed": "[CONFIRMED]",
        "rejected": "[REJECTED]",
        "partially_supported": "[PARTIAL]",
        "directional_support": "[DIRECTIONAL]",
        "inconclusive": "[INCONCLUSIVE]",
    }

    for _, row in hyp_df.iterrows():
        marker = status_markers.get(row["status"], f"[{row['status'].upper()}]")
        print(f"\n  {row['hypothesis']}: {row['description']}")
        print(f"    Status: {marker}")
        print(f"    Evidence: {row['evidence']}")
        print(f"    Effect size: {row['effect_size']}")

    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical hypothesis tests for 3x2 factorial experiment."
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Directory containing input CSVs and where output CSVs are written (default: PROJECT_ROOT/results)",
    )
    args = parser.parse_args()

    run_all_tests(args.results_dir)


if __name__ == "__main__":
    main()
