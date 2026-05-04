"""Declarative agent: LLMAgent + skill file injection.

Mirrors the built-in LLMAgent exactly, with the only difference being
skill file content appended to the system prompt in <skills> tags.
The LLM interprets skill files to guide workflow, tool selection, and
validation — no programmatic control flow.
"""

from pathlib import Path
from typing import List, Optional

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from agents.state import AgentState

SKILL_FILES_DIR = Path(__file__).parent.parent / "skills"

# Identical to tau2-bench's LLMAgent AGENT_INSTRUCTION — no extra text.
# The ONLY difference from baseline is the <skills> block appended to the prompt.
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


class DeclarativeAgent(LLMConfigMixin, HalfDuplexAgent[AgentState]):
    """Agent guided by natural-language skill files.

    Uses the full tool set at every turn. Workflow sequencing,
    tool selection, and validation are all handled by the LLM
    interpreting skill file instructions.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        skills_dir: Optional[Path] = None,
        skills_pattern: str = "*.md",
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.skills_dir = skills_dir or SKILL_FILES_DIR
        self.skills_pattern = skills_pattern
        self.skill_content = self._load_skills()

    def _load_skills(self) -> str:
        """Load skill files matching skills_pattern from directory, sorted alphabetically."""
        skills = []
        if self.skills_dir.exists():
            for path in sorted(self.skills_dir.glob(self.skills_pattern)):
                skills.append(f"## SKILL: {path.stem}\n\n{path.read_text()}")
        return "\n\n---\n\n".join(skills)

    @property
    def system_prompt(self) -> str:
        # Base prompt matches LLMAgent's SYSTEM_PROMPT exactly
        prompt = f"""
<instructions>
{AGENT_INSTRUCTION}
</instructions>
<policy>
{self.domain_policy}
</policy>
""".strip()
        # Skill files are the ONLY addition vs baseline
        if self.skill_content:
            prompt += f"\n<skills>\n{self.skill_content}\n</skills>"
        return prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AgentState:
        if message_history is None:
            message_history = []
        return AgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio messages not supported.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

        state.messages.append(assistant_message)
        return assistant_message, state
