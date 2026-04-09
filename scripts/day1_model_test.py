#!/usr/bin/env python3
"""Day 1 model validation: test candidate agent LLMs on banking_knowledge.

NOTE: This is a historical script from the model selection phase. The final
experiment used dashscope/qwen3.5-flash (see configs/experiment.yaml), not the
OpenRouter models listed below. Retained for reproducibility documentation.

Tests each model on 3 tasks with baseline agent + bm25 retrieval.
Validates: tool_choice="auto" works, KB_search → unlock → call workflow completes.

Priority order (free models first):
  1. openrouter/z-ai/glm-4.7-flash       — FREE, #1 on tau2-bench (98.8%)
  2. openrouter/xiaomi/mimo-v2-flash:free — FREE, tau2-bench 93-95%
  3. zai/glm-4.6                          — FREE promo, BFCL #4 (72.38%)
  4. gemini/gemini-2.5-flash              — FREE tier, proven integration
  5. gpt-4.1-nano                         — budget fallback (~$0.30 for pilot)

Required env vars: OPENROUTER_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY
"""

import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run.py")

MODELS = [
    ("openrouter/z-ai/glm-4.7-flash", "FREE — tau2-bench #1 (98.8%)"),
    ("openrouter/xiaomi/mimo-v2-flash:free", "FREE — tau2-bench 93-95%"),
    ("openrouter/z-ai/glm-4.6", "FREE promo — BFCL #4 (72.38%)"),
    ("gemini/gemini-2.5-flash", "FREE tier — proven integration"),
    ("gpt-4.1-nano", "~$0.30 — budget fallback"),
]

TASK_IDS = ["task_001", "task_050", "task_070"]
USER_LLM = "gpt-4o-mini"


def run_model_test(model: str, safe_name: str) -> tuple[str, float]:
    """Run a single model test. Returns (status, elapsed_seconds)."""
    cmd = [
        sys.executable, RUN_SCRIPT,
        "run",
        "--domain", "banking_knowledge",
        "--agent-llm", model,
        "--user-llm", USER_LLM,
        "--retrieval-config", "bm25",
        "--num-trials", "1",
        "--task-ids", *TASK_IDS,
        "--verbose-logs",
        "--save-to", f"model_test_{safe_name}",
    ]

    print(f"  Command: {' '.join(cmd)}\n")
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=os.path.join(PROJECT_ROOT, "tau2-bench"),
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        return "PASS", elapsed
    else:
        return f"FAIL (exit {result.returncode})", elapsed


def main():
    results = []

    print("=" * 70)
    print("DAY 1 MODEL VALIDATION — banking_knowledge × bm25 × 3 tasks")
    print("=" * 70)
    print(f"Tasks: {', '.join(TASK_IDS)}")
    print(f"User sim: {USER_LLM}")
    print(f"Testing {len(MODELS)} models\n")

    for model, description in MODELS:
        safe_name = model.replace("/", "_").replace(":", "_")
        print(f"\n{'─' * 70}")
        print(f"  [{len(results)+1}/{len(MODELS)}] {model}")
        print(f"  {description}")
        print(f"{'─' * 70}\n")

        status, elapsed = run_model_test(model, safe_name)
        results.append((model, description, status, elapsed))

        print(f"\n  → {status} ({elapsed:.0f}s)\n")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45} {'Status':<15} {'Time':>8}")
    print("-" * 70)
    for model, desc, status, elapsed in results:
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} {model:<43} {status:<15} {elapsed:>6.0f}s")
    print("-" * 70)

    passed = [r for r in results if r[2] == "PASS"]
    if passed:
        best = passed[0]
        print(f"\nRecommendation: {best[0]}")
        print(f"  {best[1]}")
        print(f"\nUpdate configs/experiment.yaml agent_llm to this model.")
    else:
        print("\nNo models passed. Check logs in tau2-bench/results/model_test_*/")

    print("=" * 70)


if __name__ == "__main__":
    main()
