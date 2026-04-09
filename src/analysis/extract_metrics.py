#!/usr/bin/env python3
"""Extract metrics from tau3-bench experiment results into analysis-ready CSVs.

Reads: configs/experiment.yaml for condition definitions.
       results/{condition}/results.json for raw simulation data.

Produces:
  results/sim_level.csv       — one row per simulation (1,746 expected)
  results/condition_summary.csv — one row per condition (6 expected)
  results/task_pass_rates.csv  — one row per task, pass^1 per condition
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

# Agent implementation name -> friendly label used in analysis
AGENT_NAME_MAP = {
    "llm_agent": "baseline",
    "declarative_agent": "declarative",
    "imperative_agent": "imperative",
}


# ---------------------------------------------------------------------------
# Pass^k — exact match with tau2-bench/src/tau2/metrics/agent_metrics.py:126
# ---------------------------------------------------------------------------

def is_successful(reward: float) -> bool:
    """Match tau2-bench: reward is success iff it rounds to 1.0."""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """Unbiased pass^k estimator.  C(s,k) / C(n,k)."""
    if num_trials < k:
        return float("nan")
    return math.comb(success_count, k) / math.comb(num_trials, k)


# ---------------------------------------------------------------------------
# Per-simulation metric extraction
# ---------------------------------------------------------------------------

def extract_sim_row(sim: dict, condition: str, agent_type: str, retrieval: str) -> dict | None:
    """Parse one simulation dict into a flat row dict.

    Returns None for infrastructure_error simulations (skipped, matching tau2).
    """
    term = sim.get("termination_reason", "")
    if term == "infrastructure_error":
        return None

    reward = 0.0
    reward_info = sim.get("reward_info")
    if reward_info is not None:
        reward = reward_info.get("reward", 0.0)

    messages = sim.get("messages", [])

    num_messages = len(messages)
    # Turns = user or assistant messages (exclude tool-result messages)
    num_turns = sum(1 for m in messages if m.get("role") in ("user", "assistant"))

    num_tool_calls = 0
    num_kb_searches = 0
    tool_names: list[str] = []
    for m in messages:
        if m.get("role") == "assistant":
            tc = m.get("tool_calls")
            if tc:
                num_tool_calls += len(tc)
                for call in tc:
                    name = call.get("name", "")
                    tool_names.append(name)
                    if name == "KB_search":
                        num_kb_searches += 1

    # Error heuristic: tool-result messages where content contains "Error"
    num_errors = 0
    for m in messages:
        if m.get("role") == "tool":
            # Check the explicit error flag first
            if m.get("error") is True:
                num_errors += 1
            elif isinstance(m.get("content"), str) and "Error" in m["content"]:
                num_errors += 1

    return {
        "condition": condition,
        "agent_type": agent_type,
        "retrieval": retrieval,
        "task_id": sim.get("task_id", ""),
        "trial": sim.get("trial", 0),
        "reward": reward,
        "success": is_successful(reward),
        "termination_reason": term,
        "duration": sim.get("duration", 0.0),
        "agent_cost": sim.get("agent_cost"),
        "num_messages": num_messages,
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "num_kb_searches": num_kb_searches,
        "num_errors": num_errors,
        "tool_names_used": ";".join(sorted(set(tool_names))),
    }


# ---------------------------------------------------------------------------
# Load one condition's results.json
# ---------------------------------------------------------------------------

def load_condition(
    condition_name: str,
    results_dir: Path,
    agent_impl: str,
    retrieval: str,
) -> list[dict]:
    """Load results.json for one condition and return list of sim-level rows.

    Validates that the file's metadata matches the expected agent/retrieval.
    """
    path = results_dir / condition_name / "results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping condition '{condition_name}'")
        return []

    agent_type = AGENT_NAME_MAP.get(agent_impl, agent_impl)

    with open(path) as f:
        data = json.load(f)

    # Sanity-check metadata
    info = data.get("info", {})
    file_agent = info.get("agent_info", {}).get("implementation", "")
    file_retrieval = info.get("retrieval_config", "")

    if file_agent != agent_impl:
        print(f"  WARNING: {condition_name} expected agent '{agent_impl}' but file has '{file_agent}'")
    if file_retrieval != retrieval:
        print(f"  WARNING: {condition_name} expected retrieval '{retrieval}' but file has '{file_retrieval}'")

    sims = data.get("simulations", [])
    infra_skipped = 0
    rows = []
    for sim in sims:
        row = extract_sim_row(sim, condition_name, agent_type, retrieval)
        if row is None:
            infra_skipped += 1
        else:
            rows.append(row)

    if infra_skipped > 0:
        print(f"  Skipped {infra_skipped} infrastructure_error simulation(s)")

    return rows


# ---------------------------------------------------------------------------
# Condition-level summary
# ---------------------------------------------------------------------------

def build_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sim_level data into one row per condition."""
    records = []

    for condition, group in df.groupby("condition", sort=False):
        agent_type = group["agent_type"].iloc[0]
        retrieval = group["retrieval"].iloc[0]
        n_sims = len(group)
        n_tasks = group["task_id"].nunique()

        # Pass^k: average across tasks
        task_groups = group.groupby("task_id")
        pass_1_vals = []
        pass_3_vals = []
        for _tid, tg in task_groups:
            n = len(tg)
            s = int(tg["success"].sum())
            pass_1_vals.append(pass_hat_k(n, s, 1))
            if n >= 3:
                pass_3_vals.append(pass_hat_k(n, s, 3))

        pass_1 = sum(pass_1_vals) / len(pass_1_vals) if pass_1_vals else float("nan")
        pass_3 = sum(pass_3_vals) / len(pass_3_vals) if pass_3_vals else float("nan")

        successes = group[group["success"]]
        failures = group[~group["success"]]

        # Termination reason percentages
        term_counts = group["termination_reason"].value_counts()
        pct = lambda reason: term_counts.get(reason, 0) / n_sims if n_sims else 0.0

        records.append({
            "condition": condition,
            "agent_type": agent_type,
            "retrieval": retrieval,
            "pass_1": pass_1,
            "pass_3": pass_3,
            "avg_reward": group["reward"].mean(),
            "mean_tool_calls": group["num_tool_calls"].mean(),
            "mean_tool_calls_success": successes["num_tool_calls"].mean() if len(successes) else float("nan"),
            "mean_tool_calls_fail": failures["num_tool_calls"].mean() if len(failures) else float("nan"),
            "mean_kb_searches": group["num_kb_searches"].mean(),
            "mean_turns": group["num_turns"].mean(),
            "mean_duration": group["duration"].mean(),
            "pct_user_stop": pct("user_stop"),
            "pct_agent_stop": pct("agent_stop"),
            "pct_max_steps": pct("max_steps"),
            "pct_error": pct("too_many_errors"),
            "n_simulations": n_sims,
            "n_tasks": n_tasks,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Task-level pass rates pivot
# ---------------------------------------------------------------------------

def build_task_pass_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Build a task x condition pivot of pass^1 rates."""
    rows = []
    for (task_id, condition), group in df.groupby(["task_id", "condition"]):
        n = len(group)
        s = int(group["success"].sum())
        rows.append({
            "task_id": task_id,
            "condition": condition,
            "pass_1": pass_hat_k(n, s, 1),
        })

    if not rows:
        return pd.DataFrame()

    pivot = pd.DataFrame(rows).pivot(index="task_id", columns="condition", values="pass_1")
    pivot = pivot.reset_index()
    pivot.columns.name = None

    # Add mean across conditions
    condition_cols = [c for c in pivot.columns if c != "task_id"]
    pivot["mean_pass_rate"] = pivot[condition_cols].mean(axis=1)

    return pivot.sort_values("task_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract metrics from tau3-bench experiment results into CSVs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing {condition}/results.json subdirs (default: PROJECT_ROOT/results)",
    )
    args = parser.parse_args()
    results_dir: Path = args.results_dir.resolve()

    print(f"Results directory: {results_dir}")
    print(f"Config: {CONFIG_PATH}")

    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    conditions = config.get("conditions", {})
    if not conditions:
        print("ERROR: No conditions found in config", file=sys.stderr)
        sys.exit(1)

    print(f"Conditions defined: {list(conditions.keys())}\n")

    # Collect all sim-level rows across conditions
    all_rows: list[dict] = []
    for name, cond in conditions.items():
        agent_impl = cond["agent"]
        retrieval = cond["retrieval_config"]
        print(f"Loading {name} (agent={agent_impl}, retrieval={retrieval})...")
        rows = load_condition(name, results_dir, agent_impl, retrieval)
        print(f"  {len(rows)} simulations loaded")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo simulation data found. Check that results files exist.")
        sys.exit(1)

    df_sim = pd.DataFrame(all_rows)
    print(f"\nTotal simulations: {len(df_sim)}")
    print(f"Conditions with data: {df_sim['condition'].nunique()}")
    print(f"Unique tasks: {df_sim['task_id'].nunique()}")

    # Build summaries
    df_summary = build_condition_summary(df_sim)
    df_tasks = build_task_pass_rates(df_sim)

    # Write CSVs
    sim_path = results_dir / "sim_level.csv"
    summary_path = results_dir / "condition_summary.csv"
    tasks_path = results_dir / "task_pass_rates.csv"

    df_sim.to_csv(sim_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_tasks.to_csv(tasks_path, index=False)

    print(f"\nOutputs written:")
    print(f"  {sim_path}  ({len(df_sim)} rows)")
    print(f"  {summary_path}  ({len(df_summary)} rows)")
    print(f"  {tasks_path}  ({len(df_tasks)} rows)")

    # Print quick summary
    print(f"\n{'='*60}")
    print("CONDITION SUMMARY")
    print(f"{'='*60}")
    for _, row in df_summary.iterrows():
        print(
            f"  {row['condition']:25s}  pass^1={row['pass_1']:.3f}  "
            f"pass^3={row['pass_3']:.3f}  "
            f"avg_reward={row['avg_reward']:.3f}  "
            f"n={row['n_simulations']}"
        )


if __name__ == "__main__":
    main()
