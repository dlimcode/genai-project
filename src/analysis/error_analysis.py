#!/usr/bin/env python3
"""Error taxonomy analysis for failed simulations.

Reads raw results.json files, classifies failures by error taxonomy (E1-E9),
produces frequency tables, divergence analysis, and identifies case studies.

Auto-detectable errors: E2, E4, E5, E7, E8, E9.
Manual-only errors: E1, E3, E6 (flagged but not classified).

Produces:
  {results_dir}/error_analysis.csv      — one row per failed simulation
  {results_dir}/error_summary.csv       — error type x condition frequency table
  {results_dir}/divergence_analysis.csv  — tasks with divergent outcomes across conditions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

AGENT_NAME_MAP = {
    "llm_agent": "baseline",
    "declarative_agent": "declarative",
    "imperative_agent": "imperative",
}

# Standard tools always available in banking_knowledge domain
STANDARD_TOOLS = {
    "transfer_to_human_agents",
    "get_current_time",
    "get_user_information_by_id",
    "get_user_information_by_name",
    "get_user_information_by_email",
    "change_user_email",
    "get_referrals_by_user",
    "get_credit_card_transactions_by_user",
    "get_credit_card_accounts_by_user",
    "log_verification",
    "give_discoverable_user_tool",
    "unlock_discoverable_agent_tool",
    "call_discoverable_agent_tool",
    "list_discoverable_agent_tools",
    # Retrieval tools (variant-dependent but still "known")
    "KB_search",
}

# Pattern for discoverable tool names (end with _NNNN)
DISCOVERABLE_TOOL_RE = re.compile(r"_\d{4}$")


def is_successful(reward: float) -> bool:
    """Match tau2-bench: reward is success iff it rounds to 1.0."""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


# ---------------------------------------------------------------------------
# Build known tool set dynamically from simulation data
# ---------------------------------------------------------------------------

def collect_all_tool_names(results_dir: Path, conditions: dict) -> set[str]:
    """Scan all simulations to build the set of known (non-hallucinated) tool names.

    Collects from:
    - STANDARD_TOOLS (hardcoded)
    - Expected action names from evaluation criteria
    - Tool calls that got non-error responses
    Excludes tool calls that produced errors (which may be hallucinated names).
    """
    names: set[str] = set(STANDARD_TOOLS)
    for cond_name in conditions:
        path = results_dir / cond_name / "results.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        # Collect expected tool names from task action definitions
        for task in data.get("tasks", []):
            ec = task.get("evaluation_criteria") or {}
            for action in ec.get("actions") or []:
                aname = action.get("name", "")
                if aname:
                    names.add(aname)

        for sim in data.get("simulations", []):
            messages = sim.get("messages", [])
            # Build map of tool_call_id -> whether the result was an error
            tool_result_errors: dict[str, bool] = {}
            for m in messages:
                if m.get("role") == "tool":
                    tid = m.get("id", "")
                    is_err = m.get("error", False)
                    if not is_err and isinstance(m.get("content"), str):
                        is_err = "Error" in m["content"]
                    tool_result_errors[tid] = is_err

            # Collect tool names from successful (non-error) calls only
            for m in messages:
                if m.get("role") != "assistant" or not m.get("tool_calls"):
                    continue
                for tc in m["tool_calls"]:
                    name = tc.get("name", "")
                    tid = tc.get("id", "")
                    if name and not tool_result_errors.get(tid, True):
                        names.add(name)
    return names


# ---------------------------------------------------------------------------
# Error detectors
# ---------------------------------------------------------------------------

def detect_e2(sim: dict) -> bool:
    """E2: Action mismatch — at least one expected action not matched."""
    ri = sim.get("reward_info") or {}
    for check in ri.get("action_checks") or []:
        if not check.get("action_match", True):
            return True
    return False


def detect_e4(sim: dict, known_tools: set[str]) -> list[str]:
    """E4: Hallucinated tool name — tool name not in any known set."""
    hallucinated = []
    for m in sim.get("messages", []):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            name = tc.get("name", "")
            if not name:
                continue
            if name not in known_tools and not DISCOVERABLE_TOOL_RE.search(name):
                hallucinated.append(name)
    return hallucinated


def detect_e5(sim: dict, retrieval: str) -> int:
    """E5: Unnecessary KB search — KB_search calls in golden_retrieval."""
    if retrieval != "golden_retrieval":
        return 0
    count = 0
    for m in sim.get("messages", []):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            if tc.get("name") == "KB_search":
                count += 1
    return count


def detect_e7(sim: dict) -> bool:
    """E7: Premature termination — failed with normal stop reason."""
    ri = sim.get("reward_info") or {}
    reward = ri.get("reward", 0.0)
    term = sim.get("termination_reason", "")
    return not is_successful(reward) and term in ("user_stop", "agent_stop")


def detect_e8(sim: dict, window: int = 3) -> dict:
    """E8: Excessive looping — repeated tool calls.

    Returns dict with:
      name_loop: bool — 3+ consecutive same tool name
      exact_loop: bool — 3+ consecutive identical (name+args)
      looped_tools: list[str] — tool names that looped (name-only)
    """
    tool_seq_name: list[str] = []
    tool_seq_exact: list[tuple[str, str]] = []

    for m in sim.get("messages", []):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            name = tc.get("name", "")
            args = json.dumps(tc.get("arguments", {}), sort_keys=True)
            tool_seq_name.append(name)
            tool_seq_exact.append((name, args))

    name_loop = False
    looped_tools: list[str] = []
    for i in range(len(tool_seq_name) - window + 1):
        chunk = tool_seq_name[i : i + window]
        if len(set(chunk)) == 1:
            name_loop = True
            if chunk[0] not in looped_tools:
                looped_tools.append(chunk[0])

    exact_loop = False
    for i in range(len(tool_seq_exact) - window + 1):
        chunk = tool_seq_exact[i : i + window]
        if len(set(chunk)) == 1:
            exact_loop = True
            break

    return {
        "name_loop": name_loop,
        "exact_loop": exact_loop,
        "looped_tools": looped_tools,
    }


def detect_e9(sim: dict) -> bool:
    """E9: Discoverable tool failure — error on unlock or call."""
    messages = sim.get("messages", [])
    discoverable_ops = {
        "unlock_discoverable_agent_tool",
        "call_discoverable_agent_tool",
    }

    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        is_error = m.get("error", False)
        if not is_error:
            content = str(m.get("content", ""))
            if "Error" in content:
                is_error = True
        if not is_error:
            continue

        # Check if this error relates to a discoverable tool operation
        # by looking at the preceding assistant tool call
        for j in range(i - 1, -1, -1):
            prev = messages[j]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                for tc in prev["tool_calls"]:
                    if tc.get("name") in discoverable_ops:
                        return True
                break
            if prev.get("role") in ("user",):
                break
    return False


# ---------------------------------------------------------------------------
# Analyze one simulation
# ---------------------------------------------------------------------------

def analyze_simulation(
    sim: dict,
    condition: str,
    agent_type: str,
    retrieval: str,
    known_tools: set[str],
) -> dict | None:
    """Classify a single simulation by error taxonomy.

    Returns None for successful or infrastructure_error simulations.
    """
    term = sim.get("termination_reason", "")
    if term == "infrastructure_error":
        return None

    ri = sim.get("reward_info") or {}
    reward = ri.get("reward", 0.0)
    if is_successful(reward):
        return None

    # Count basic metrics
    num_tool_calls = 0
    num_errors = 0
    for m in sim.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            num_tool_calls += len(m["tool_calls"])
        if m.get("role") == "tool":
            if m.get("error") is True:
                num_errors += 1
            elif isinstance(m.get("content"), str) and "Error" in m["content"]:
                num_errors += 1

    # Run detectors
    e2 = detect_e2(sim)
    e4_tools = detect_e4(sim, known_tools)
    e5_count = detect_e5(sim, retrieval)
    e7 = detect_e7(sim)
    e8 = detect_e8(sim)
    e9 = detect_e9(sim)

    return {
        "condition": condition,
        "agent_type": agent_type,
        "retrieval": retrieval,
        "task_id": sim.get("task_id", ""),
        "trial": sim.get("trial", 0),
        "reward": reward,
        "termination_reason": term,
        "num_tool_calls": num_tool_calls,
        "num_errors": num_errors,
        "e2_action_mismatch": e2,
        "e4_hallucinated": bool(e4_tools),
        "e4_tool_names": ";".join(e4_tools) if e4_tools else "",
        "e5_unnecessary_kb": e5_count,
        "e7_premature_term": e7,
        "e8_name_loop": e8["name_loop"],
        "e8_exact_loop": e8["exact_loop"],
        "e8_looped_tools": ";".join(e8["looped_tools"]) if e8["looped_tools"] else "",
        "e9_discoverable_fail": e9,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_error_summary(error_df: pd.DataFrame) -> pd.DataFrame:
    """Frequency table: error type x condition."""
    error_cols = {
        "E2_action_mismatch": "e2_action_mismatch",
        "E4_hallucinated_tool": "e4_hallucinated",
        "E5_unnecessary_kb": lambda df: df["e5_unnecessary_kb"] > 0,
        "E7_premature_term": "e7_premature_term",
        "E8_name_loop": "e8_name_loop",
        "E8_exact_loop": "e8_exact_loop",
        "E9_discoverable_fail": "e9_discoverable_fail",
    }

    rows = []
    conditions = sorted(error_df["condition"].unique())

    for label, col_or_fn in error_cols.items():
        row = {"error_type": label}
        for cond in conditions:
            cond_df = error_df[error_df["condition"] == cond]
            if callable(col_or_fn):
                count = int(col_or_fn(cond_df).sum())
            else:
                count = int(cond_df[col_or_fn].sum())
            row[cond] = count
        row["total"] = sum(row[c] for c in conditions)
        rows.append(row)

    # Add total failures row
    total_row = {"error_type": "total_failures"}
    for cond in conditions:
        total_row[cond] = int((error_df["condition"] == cond).sum())
    total_row["total"] = len(error_df)
    rows.append(total_row)

    return pd.DataFrame(rows)


def build_divergence_analysis(
    all_sims: dict[str, list[dict]],
) -> pd.DataFrame:
    """Find tasks where outcomes diverge across conditions.

    Returns rows where one condition passes and another fails on the same
    (task_id, trial) pair.
    """
    # Build reward maps: condition -> {(task_id, trial): reward}
    reward_maps: dict[str, dict[tuple[str, int], float]] = {}
    for cond, sims in all_sims.items():
        rmap: dict[tuple[str, int], float] = {}
        for sim in sims:
            if sim.get("termination_reason") == "infrastructure_error":
                continue
            ri = sim.get("reward_info") or {}
            key = (sim.get("task_id", ""), sim.get("trial", 0))
            rmap[key] = ri.get("reward", 0.0)
        reward_maps[cond] = rmap

    conditions = sorted(reward_maps.keys())
    rows = []

    # Find all (task_id, trial) pairs
    all_keys: set[tuple[str, int]] = set()
    for rmap in reward_maps.values():
        all_keys.update(rmap.keys())

    for key in sorted(all_keys):
        task_id, trial = key
        outcomes = {}
        for cond in conditions:
            if key in reward_maps[cond]:
                outcomes[cond] = is_successful(reward_maps[cond][key])

        if len(outcomes) < 2:
            continue

        # Only report if there's at least one pass and one fail
        passes = [c for c, s in outcomes.items() if s]
        fails = [c for c, s in outcomes.items() if not s]
        if not passes or not fails:
            continue

        for cond_pass in passes:
            for cond_fail in fails:
                rows.append({
                    "task_id": task_id,
                    "trial": trial,
                    "cond_pass": cond_pass,
                    "cond_fail": cond_fail,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["task_id", "trial", "cond_pass", "cond_fail"]
    )


def identify_case_studies(
    error_df: pd.DataFrame,
    divergence_df: pd.DataFrame,
) -> None:
    """Print 3-5 case study candidates for the paper."""
    print(f"\n{'='*60}")
    print("CASE STUDY CANDIDATES")
    print(f"{'='*60}")

    # 1. Imperative-specific failures: baseline_golden passes, imperative_golden fails
    if not divergence_df.empty:
        imp_fails = divergence_df[
            (divergence_df["cond_pass"] == "baseline_golden")
            & (divergence_df["cond_fail"] == "imperative_golden")
        ]
        if not imp_fails.empty:
            tasks = sorted(imp_fails["task_id"].unique())
            print(f"\n1. IMPERATIVE PHASE-LOCK FAILURES ({len(tasks)} tasks)")
            print(f"   Tasks where baseline_golden passes but imperative_golden fails:")
            for t in tasks[:5]:
                imp_row = error_df[
                    (error_df["task_id"] == t)
                    & (error_df["condition"] == "imperative_golden")
                ]
                if not imp_row.empty:
                    r = imp_row.iloc[0]
                    flags = []
                    if r["e2_action_mismatch"]:
                        flags.append("E2")
                    if r["e4_hallucinated"]:
                        flags.append(f"E4({r['e4_tool_names']})")
                    if r["e8_name_loop"]:
                        flags.append(f"E8({r['e8_looped_tools']})")
                    if r["e9_discoverable_fail"]:
                        flags.append("E9")
                    print(
                        f"   - {t}: term={r['termination_reason']}, "
                        f"tools={r['num_tool_calls']}, errors={r['num_errors']}, "
                        f"flags={','.join(flags) if flags else 'E7-only'}"
                    )

    # 2. Declarative regressions: baseline passes, declarative fails
    if not divergence_df.empty:
        dec_fails = divergence_df[
            (divergence_df["cond_pass"] == "baseline_golden")
            & (divergence_df["cond_fail"] == "declarative_golden")
        ]
        if not dec_fails.empty:
            tasks = sorted(dec_fails["task_id"].unique())
            print(f"\n2. DECLARATIVE REGRESSIONS ({len(tasks)} tasks)")
            print(f"   Tasks where baseline_golden passes but declarative_golden fails:")
            for t in tasks[:5]:
                print(f"   - {t}")

    # 3. E4 hallucinations (all conditions)
    e4_rows = error_df[error_df["e4_hallucinated"]]
    if not e4_rows.empty:
        print(f"\n3. HALLUCINATED TOOL NAMES ({len(e4_rows)} instances)")
        for _, r in e4_rows.head(5).iterrows():
            print(
                f"   - {r['condition']}/{r['task_id']}: "
                f"hallucinated={r['e4_tool_names']}"
            )

    # 4. E8 exact looping (pathological)
    e8_exact = error_df[error_df["e8_exact_loop"]]
    if not e8_exact.empty:
        print(f"\n4. PATHOLOGICAL LOOPING (E8_exact, {len(e8_exact)} instances)")
        for _, r in e8_exact.head(5).iterrows():
            print(
                f"   - {r['condition']}/{r['task_id']}: "
                f"looped on {r['e8_looped_tools']}, {r['num_tool_calls']} total calls"
            )

    # 5. Universal retrieval sensitivity: golden passes, BM25 fails, all agents
    if not divergence_df.empty:
        retrieval_div = divergence_df[
            divergence_df["cond_pass"].str.contains("golden")
            & divergence_df["cond_fail"].str.contains("bm25")
        ]
        if not retrieval_div.empty:
            task_counts = retrieval_div.groupby("task_id").size()
            # Tasks that show this pattern for multiple agent types
            multi_agent = task_counts[task_counts >= 2]
            if not multi_agent.empty:
                print(f"\n5. RETRIEVAL-SENSITIVE TASKS ({len(multi_agent)} tasks, 2+ agents)")
                for t in list(multi_agent.index)[:5]:
                    agents = retrieval_div[retrieval_div["task_id"] == t][
                        "cond_pass"
                    ].tolist()
                    print(f"   - {t}: golden passes in {agents}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error taxonomy analysis for failed simulations."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "tau2-bench" / "data" / "simulations",
        help="Directory containing {condition}/results.json subdirs",
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

    # Phase 1: Build known tool set from all data
    print("\nCollecting known tool names across all conditions...")
    known_tools = collect_all_tool_names(results_dir, conditions)
    print(f"  {len(known_tools)} unique tool names found")

    # Phase 2: Analyze all simulations
    print(f"\nAnalyzing failures across {len(conditions)} conditions...")
    all_error_rows: list[dict] = []
    all_sims: dict[str, list[dict]] = {}
    total_sims = 0
    total_infra = 0

    for cond_name, cond_cfg in conditions.items():
        path = results_dir / cond_name / "results.json"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping '{cond_name}'")
            continue

        agent_type = AGENT_NAME_MAP.get(cond_cfg["agent"], cond_cfg["agent"])
        retrieval = cond_cfg["retrieval_config"]

        with open(path) as f:
            data = json.load(f)

        sims = data.get("simulations", [])
        all_sims[cond_name] = sims

        n_success = 0
        n_fail = 0
        n_infra = 0

        for sim in sims:
            if sim.get("termination_reason") == "infrastructure_error":
                n_infra += 1
                continue
            total_sims += 1

            row = analyze_simulation(sim, cond_name, agent_type, retrieval, known_tools)
            if row is None:
                n_success += 1
            else:
                n_fail += 1
                all_error_rows.append(row)

        total_infra += n_infra
        print(
            f"  {cond_name}: {n_success} pass, {n_fail} fail, "
            f"{n_infra} infra_error"
        )

    if not all_error_rows:
        print("\nNo failures found.")
        sys.exit(0)

    error_df = pd.DataFrame(all_error_rows)
    print(f"\nTotal: {total_sims} valid sims, {len(error_df)} failures, {total_infra} infra_errors")

    # Phase 3: Build summaries
    summary_df = build_error_summary(error_df)
    divergence_df = build_divergence_analysis(all_sims)

    # Phase 4: Write outputs
    error_path = results_dir / "error_analysis.csv"
    summary_path = results_dir / "error_summary.csv"
    divergence_path = results_dir / "divergence_analysis.csv"

    error_df.to_csv(error_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    divergence_df.to_csv(divergence_path, index=False)

    print(f"\nOutputs written:")
    print(f"  {error_path}  ({len(error_df)} rows)")
    print(f"  {summary_path}  ({len(summary_df)} rows)")
    print(f"  {divergence_path}  ({len(divergence_df)} rows)")

    # Phase 5: Print summary
    print(f"\n{'='*60}")
    print("ERROR SUMMARY (count per condition)")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    # Phase 6: Case studies
    identify_case_studies(error_df, divergence_df)

    # Phase 7: Key statistics for the paper
    print(f"\n{'='*60}")
    print("KEY STATISTICS")
    print(f"{'='*60}")

    for agent in ["baseline", "declarative", "imperative"]:
        agent_df = error_df[error_df["agent_type"] == agent]
        if agent_df.empty:
            continue
        print(f"\n  {agent.upper()} ({len(agent_df)} failures):")
        print(f"    E2 action mismatch: {agent_df['e2_action_mismatch'].sum()}")
        print(f"    E4 hallucinated:    {agent_df['e4_hallucinated'].sum()}")
        print(f"    E7 premature term:  {agent_df['e7_premature_term'].sum()}")
        print(f"    E8 name loop:       {agent_df['e8_name_loop'].sum()}")
        print(f"    E8 exact loop:      {agent_df['e8_exact_loop'].sum()}")
        print(f"    E9 disc. failure:   {agent_df['e9_discoverable_fail'].sum()}")

    # Divergence highlights
    if not divergence_df.empty:
        print(f"\n  DIVERGENCE HIGHLIGHTS:")
        for cond_pass, cond_fail, label in [
            ("baseline_golden", "imperative_golden", "Imperative-specific failures (golden)"),
            ("baseline_golden", "declarative_golden", "Declarative regressions (golden)"),
            ("baseline_bm25", "imperative_bm25", "Imperative-specific failures (bm25)"),
            ("baseline_bm25", "declarative_bm25", "Declarative regressions (bm25)"),
        ]:
            subset = divergence_df[
                (divergence_df["cond_pass"] == cond_pass)
                & (divergence_df["cond_fail"] == cond_fail)
            ]
            if not subset.empty:
                tasks = sorted(subset["task_id"].unique())
                print(f"    {label}: {len(tasks)} tasks ({', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''})")


if __name__ == "__main__":
    main()
