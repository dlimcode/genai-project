"""Agent registration: factory functions + register_all().

Call register_all() before using tau2 CLI to make custom agents
available via --agent declarative_agent / --agent imperative_agent /
--agent declarative_agent_v2 / --agent imperative_agent_v2.
"""

from tau2.registry import registry

from agents.declarative_agent import DeclarativeAgent
from agents.imperative_agent import ImperativeAgent
from agents.imperative_agent_v2 import ImperativeAgentV2
from agents.imperative_agent_v3 import ImperativeAgentV3


def create_declarative_agent(tools, domain_policy, **kwargs):
    """Factory for DeclarativeAgent (v1 skill files: *.md)."""
    return DeclarativeAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def create_declarative_agent_v2(tools, domain_policy, **kwargs):
    """Factory for DeclarativeAgent with v2 skill files only (*-v2.md)."""
    return DeclarativeAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        skills_pattern="*-v2.md",
    )


def create_imperative_agent(tools, domain_policy, **kwargs):
    """Factory for ImperativeAgent v1."""
    return ImperativeAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def create_imperative_agent_v2(tools, domain_policy, **kwargs):
    """Factory for ImperativeAgentV2 (expanded state machine with PLANNING phase)."""
    return ImperativeAgentV2(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def create_imperative_agent_v3(tools, domain_policy, **kwargs):
    """Factory for ImperativeAgentV3 (deterministic task queue + state-driven transitions)."""
    return ImperativeAgentV3(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def register_all():
    """Register all custom agents with the tau2 registry."""
    registry.register_agent_factory(
        factory=create_declarative_agent,
        name="declarative_agent",
    )
    registry.register_agent_factory(
        factory=create_declarative_agent_v2,
        name="declarative_agent_v2",
    )
    registry.register_agent_factory(
        factory=create_imperative_agent,
        name="imperative_agent",
    )
    registry.register_agent_factory(
        factory=create_imperative_agent_v2,
        name="imperative_agent_v2",
    )
    registry.register_agent_factory(
        factory=create_imperative_agent_v3,
        name="imperative_agent_v3",
    )
