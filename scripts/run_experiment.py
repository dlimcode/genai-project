#!/usr/bin/env python3
"""Full experiment: 97 tasks x 6 conditions x 3 trials = 1,746 sims.

Runs conditions in parallel waves (default: 2 per wave, ~21h vs ~42h sequential).
Each condition logs to logs/<condition_name>.log.
Use --wave-size 1 for sequential execution.
"""

import json
import os
import signal
import subprocess
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "experiment.yaml")
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run.py")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def build_cmd(config, name, condition, task_ids=None):
    """Build the CLI command for a single condition."""
    agent_llm_args = config.get("agent_llm_args", {})
    cmd = [
        sys.executable, RUN_SCRIPT,
        "run",
        "--domain", config["domain"],
        "--agent", condition["agent"],
        "--agent-llm", config["agent_llm"],
        "--user-llm", config["user_llm"],
        "--retrieval-config", condition["retrieval_config"],
        "--num-trials", str(config["num_trials"]),
        "--max-concurrency", str(config["max_concurrency"]),
        "--seed", str(config["seed"]),
        "--verbose-logs",
        "--auto-resume",
        "--save-to", name,
    ]
    if agent_llm_args:
        cmd.extend(["--agent-llm-args", json.dumps(agent_llm_args)])
    if task_ids:
        cmd.extend(["--task-ids"] + task_ids)
    return cmd


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run full 3x2 experiment")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to experiment config YAML (default: configs/experiment.yaml)",
    )
    parser.add_argument(
        "--wave-size", type=int, default=2,
        help="Conditions per wave (default: 2, use 1 for sequential)",
    )
    parser.add_argument(
        "--conditions", type=str, default=None,
        help="Comma-separated condition names to run (default: all)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.conditions:
        selected = [c.strip() for c in args.conditions.split(",")]
        conditions = [(k, v) for k, v in config["conditions"].items() if k in selected]
        if not conditions:
            print(f"No matching conditions found. Available: {list(config['conditions'].keys())}")
            sys.exit(1)
    else:
        conditions = list(config["conditions"].items())
    task_ids = config.get("task_ids")
    wave_size = args.wave_size
    results = {}
    active_procs = []

    os.makedirs(LOG_DIR, exist_ok=True)

    # Graceful shutdown on Ctrl+C
    def handle_signal(signum, frame):
        print("\n\nInterrupted — terminating active processes...")
        for proc, lf, name in active_procs:
            proc.terminate()
        for proc, lf, name in active_procs:
            proc.wait()
            lf.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Split into waves
    waves = [conditions[i:i + wave_size] for i in range(0, len(conditions), wave_size)]
    experiment_start = time.time()

    total_conditions = len(conditions)
    done_count = 0

    print(f"Experiment: {total_conditions} conditions in {len(waves)} wave(s) "
          f"(wave_size={wave_size}, concurrency={config['max_concurrency']}/condition)")

    for wave_num, wave in enumerate(waves, 1):
        names = [name for name, _ in wave]
        print(f"\n{'='*60}")
        print(f"WAVE {wave_num}/{len(waves)}: {', '.join(names)}")
        print(f"{'='*60}")

        wave_start = time.time()
        active_procs.clear()

        for name, condition in wave:
            cmd = build_cmd(config, name, condition, task_ids=task_ids)
            log_path = os.path.join(LOG_DIR, f"{name}.log")
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=os.path.join(PROJECT_ROOT, "tau2-bench"),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active_procs.append((proc, log_file, name))
            print(f"  Started {name} (pid {proc.pid}) -> logs/{name}.log")

        # Wait for all in wave
        for proc, log_file, name in active_procs:
            proc.wait()
            log_file.close()
            done_count += 1
            elapsed = time.time() - wave_start
            if proc.returncode == 0:
                results[name] = "OK"
                print(f"  OK {name} ({elapsed / 3600:.1f}h into wave)")
            else:
                results[name] = f"FAILED (exit {proc.returncode})"
                print(f"  FAIL {name} (exit {proc.returncode}) — check logs/{name}.log")

        active_procs.clear()
        wave_elapsed = time.time() - wave_start
        total_elapsed = time.time() - experiment_start
        remaining_waves = len(waves) - wave_num
        eta = wave_elapsed * remaining_waves
        print(f"\n  Wave {wave_num} done in {wave_elapsed / 3600:.1f}h "
              f"({done_count}/{total_conditions} conditions, "
              f"total: {total_elapsed / 3600:.1f}h, "
              f"est remaining: {eta / 3600:.1f}h)")

    # Summary
    total_time = time.time() - experiment_start
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE — {total_time / 3600:.1f}h total")
    print(f"{'='*60}")
    for name, status in results.items():
        marker = "OK" if status == "OK" else "FAIL"
        print(f"  {marker} {name}: {status}")

    failed = [n for n, s in results.items() if s != "OK"]
    if failed:
        print(f"\n{len(failed)} condition(s) failed: {', '.join(failed)}")
        print("Re-run to retry (tau2-bench resumes from checkpoints).")
    else:
        print("\nAll conditions completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
