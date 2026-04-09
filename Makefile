# GenAI Project: Declarative vs Imperative Agent Orchestration
# All commands run through tau2-bench's uv environment

PYTHON = cd tau2-bench && uv run python
RUN = $(PYTHON) ../scripts/run.py

.PHONY: setup smoke-test pilot experiment analyze clean

# --- Setup ---

setup:
	@echo "=== Cloning and setting up tau2-bench ==="
	@if [ ! -d tau2-bench ]; then \
		git clone https://github.com/sierra-research/tau2-bench.git tau2-bench; \
	fi
	cd tau2-bench && git checkout v1.0.0
	cd tau2-bench && uv sync --extra knowledge --extra voice
	@echo ""
	@echo "=== Setup complete ==="
	@echo "Create tau2-bench/.env with DASHSCOPE_API_KEY and OPENAI_API_KEY"

check-agents:
	@$(PYTHON) -c "import sys; sys.path.insert(0, '../src'); from agents.register import register_all; register_all(); from tau2.registry import registry; agents = registry.get_agents(); assert 'declarative_agent' in agents; assert 'imperative_agent' in agents; print('OK: declarative_agent and imperative_agent registered')"

# --- Smoke Tests ---

smoke-test-baseline:
	$(RUN) run --domain banking_knowledge \
		--task-ids task_001 \
		--agent-llm gpt-4o-mini --user-llm gpt-4o-mini \
		--retrieval-config bm25 \
		--verbose-logs --save-to smoke_baseline

smoke-test-declarative:
	$(RUN) run --domain banking_knowledge \
		--agent declarative_agent \
		--task-ids task_001 \
		--agent-llm gpt-4o-mini --user-llm gpt-4o-mini \
		--retrieval-config bm25 \
		--verbose-logs --save-to smoke_declarative

smoke-test-imperative:
	$(RUN) run --domain banking_knowledge \
		--agent imperative_agent \
		--task-ids task_001 \
		--agent-llm gpt-4o-mini --user-llm gpt-4o-mini \
		--retrieval-config bm25 \
		--verbose-logs --save-to smoke_imperative

smoke-test: smoke-test-baseline smoke-test-declarative smoke-test-imperative

# --- Pilot (5 tasks, 1 trial, all 6 conditions) ---

pilot:
	$(PYTHON) ../scripts/run_pilot.py

# --- Full Experiment ---

experiment:
	$(PYTHON) ../scripts/run_experiment.py

# --- Analysis ---

analyze:
	PYTHONPATH=../src $(PYTHON) -m analysis.extract_metrics
	PYTHONPATH=../src $(PYTHON) -m analysis.statistical_tests
	PYTHONPATH=../src $(PYTHON) -m analysis.plots

# --- Utilities ---

clean:
	rm -rf results/smoke_*
