#!/usr/bin/env python3
"""Wrapper script: sets up imports and registers custom agents, then runs tau2 CLI.

Usage:
    python scripts/run.py run --domain banking_knowledge --agent declarative_agent \
        --agent-llm dashscope/qwen3.5-flash --user-llm gpt-4o-mini

This avoids modifying any tau2-bench source code.
"""

import os
import sys

# Add tau2-bench/src so tau2 package is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU2_SRC = os.path.join(PROJECT_ROOT, "tau2-bench", "src")
OUR_SRC = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, TAU2_SRC)
sys.path.insert(0, OUR_SRC)

# Set working directory to tau2-bench so data paths resolve correctly
os.chdir(os.path.join(PROJECT_ROOT, "tau2-bench"))

# Load tau2-bench .env if it exists
from dotenv import load_dotenv

env_path = os.path.join(PROJECT_ROOT, "tau2-bench", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# Register our custom agents before CLI parses args
from agents.register import register_all

register_all()

# Register pricing for models not in LiteLLM's registry
import litellm

litellm.model_cost["dashscope/qwen3.5-flash"] = {
    "input_cost_per_token": 0.0000001,     # $0.10/M (Alibaba Cloud intl pricing)
    "output_cost_per_token": 0.0000004,    # $0.40/M
    "max_tokens": 65536,
    "max_input_tokens": 991000,
    "litellm_provider": "dashscope",
    "mode": "chat",
    "supports_function_calling": True,
}

# Run the tau2 CLI
from tau2.cli import main

main()
