"""Shared state model for custom agents."""

from typing import Optional

from pydantic import BaseModel, Field

from tau2.data_model.message import APICompatibleMessage, SystemMessage


class AgentState(BaseModel):
    """State shared by declarative and imperative agents.

    Extends LLMAgentState with optional phase tracking for the imperative agent.
    v3 fields (user_identified through expect_violation_count) are ignored by v1/v2 agents.
    """

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    phase: Optional[str] = None
    phase_retries: int = 0

    # v3: explicit boolean gates (Strategy 3 & 4)
    user_identified: bool = False
    verified: bool = False

    # v3: explicit task queue (Strategy 1 & 2)
    pending_tasks: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)

    # v3: per-tool retry tracking (Strategy 5)
    tool_retry_counts: dict[str, int] = Field(default_factory=dict)

    # v3: response-type violation counter (Strategy 6)
    expect_violation_count: int = 0
