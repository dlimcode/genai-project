# Declarative vs Imperative Agent Orchestration on tau3-bench

Comparing natural-language workflow guidance (skill files) vs programmatic workflow control (state machine) for LLM agents on the [tau3-bench](https://github.com/sierra-research/tau2-bench) banking_knowledge domain.

**Course:** Gen AI with LLMs, SMU Singapore
**Deadline:** April 11, 2026

## Research Question

Does declarative orchestration (skill files appended to the system prompt) match or exceed imperative orchestration (8-phase state machine with tool restriction) for agent task completion, and does retrieval quality moderate this effect?

## Experiment Design

3×2 factorial: 3 agents × 2 retrieval configs = 6 conditions, 97 tasks × 3 trials = 1,746 simulations.

|  | golden_retrieval | bm25 |
|--|--|--|
| **Baseline** (tau3-bench default) | Condition A | Condition B |
| **Declarative** (+ skill files) | Condition C | Condition D |
| **Imperative** (state machine) | Condition E | Condition F |

## Quick Start

```bash
# 1. Setup
make setup                          # clones tau2-bench v1.0.0, installs deps

# 2. Add API keys
cp tau2-bench/.env.example tau2-bench/.env
# Edit tau2-bench/.env with OPENAI_API_KEY (user sim) and DASHSCOPE_API_KEY (agent LLM)

# 3. Verify
make check-agents                   # confirms custom agents register

# 4. Smoke test
make smoke-test-baseline            # 1 task with baseline agent

# 5. Pilot
make pilot                          # 5 tasks × 6 conditions (30 sims)

# 6. Full experiment
make experiment                     # 97 tasks × 6 conditions × 3 trials (1,746 sims)
```

## Project Structure

```
src/
  agents/
    declarative_agent.py    # Baseline + skill file injection
    imperative_agent.py     # 8-phase state machine with tool restriction
    state.py                # Shared Pydantic state model
    register.py             # Agent factory + tau2 registry integration
  skills/
    customer-interaction.md # Workflow: greet, identify, verify, confirm
    knowledge-discovery.md  # KB search + discoverable tool unlock pattern
    banking-procedures.md   # Domain knowledge per operation type
  analysis/
    extract_metrics.py      # Parse results.json -> per-simulation CSVs
    statistical_tests.py    # Chi-squared, McNemar, bootstrap CI, Cohen's h
    error_analysis.py       # Error taxonomy detection (E1-E9)
    plots.py                # Generate experiment figures
scripts/
  run.py                    # Wrapper: path setup + registration + tau2 CLI
  run_pilot.py              # Pilot orchestrator
  run_experiment.py         # Full experiment orchestrator
  day1_model_test.py        # Model selection validation (historical)
configs/
  experiment.yaml           # Main experiment config (seed=300, 6 conditions, 3 trials)
  generalizability.yaml     # GPT-4o-mini generalizability check (10 tasks, 1 trial)
tau2-bench/                 # Cloned dependency (gitignored)
```

## Models

| Role | Model | Cost |
|------|-------|------|
| Agent LLM | Qwen3.5-Flash via DashScope | ~$0.10/$0.40 per M tokens |
| User simulator | GPT-4o-mini | ~$5 total |

## Architecture

Our code plugs into tau3-bench via its registry system — **zero modifications to tau3-bench source code**. A wrapper script (`scripts/run.py`) adds our agents to the import path, registers them, then delegates to the standard `tau2` CLI.

## Reproducibility

- **Random seed:** 300 (set in `configs/experiment.yaml` and `configs/generalizability.yaml`)
- **Agent LLM:** `dashscope/qwen3.5-flash` with `max_tokens: 4096`
- **User simulator:** `gpt-4o-mini`
- **Trials:** 3 per task-condition pair (main experiment), 1 per pair (generalizability)
- **Analysis:** `make analyze` runs the full extraction, statistical testing, and plotting pipeline

To run the generalizability check with GPT-4o-mini as agent:
```bash
cd tau2-bench && uv run python ../scripts/run_experiment.py --config ../configs/generalizability.yaml
```

## Requirements

- Python ≥3.12, <3.14
- [uv](https://docs.astral.sh/uv/) package manager
- macOS or Linux
- API keys: OpenAI (user simulator), DashScope (agent LLM)
