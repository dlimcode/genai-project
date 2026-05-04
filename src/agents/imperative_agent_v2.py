"""Imperative agent v2: expanded state machine addressing τ-Banking failure modes.

Key improvements over v1
------------------------
1. **ADVISORY phase** — informational requests that require iterative KB search get a
   dedicated loop phase rather than terminating at TRIAGE. Prevents premature CONFIRMATION
   when the agent needs multiple KB search rounds for a product recommendation.

2. **PLANNING phase** (new, inserted after VERIFICATION) — the LLM enumerates ALL
   sub-tasks from the conversation, searches the KB for each required procedure and ordering
   constraint, pre-unlocks discoverable tools, and outputs a text execution plan before
   any state-changing action is taken. Directly addresses:
   - Failure to respect implicit subtask ordering (~5% of τ-Banking failures)
   - Complex interdependencies between financial offerings (~14.5% of failures)

3. **Improved EXECUTION prompt** — explicitly instructs the agent to verify user assertions
   against the database before acting on them, and to search the KB mid-execution when an
   additional tool is needed. Addresses overtrusting user assertions (~4% of failures).

4. **Better BM25 query guidance in phase prompts** — reminds the agent to use specific
   domain nouns rather than conversational queries. Addresses search inefficiency (~23%).

5. **Separated ADVISORY from TRIAGE** — TRIAGE routes correctly; ADVISORY loops until
   the agent has enough KB information to give a complete answer.

Known issues this does NOT solve
---------------------------------
- Model capability gap: Qwen-3.5-Flash (used in the original experiment) is too weak for
  τ-Banking's avg 9.52 tool calls and 18.6 required documents per task. Even frontier models
  (Claude-4.5-Opus, GPT-5.2) achieve only ~25-40% pass^1. Using a stronger model
  (claude-sonnet-4-6, gpt-4o-mini) is the single highest-impact improvement.
- BM25 retrieval gap: BM25 drops pass^1 by 8-22% vs golden retrieval even for frontier models.
  Dense retrieval (text-embedding-3-large or Qwen3-embedding-8B) may perform better.
- Context length: tasks requiring 18+ documents can saturate smaller models' context windows.
"""

from typing import List, Optional

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from agents.state import AgentState

IDENTIFICATION_TOOLS = {
    "get_user_information_by_id",
    "get_user_information_by_name",
    "get_user_information_by_email",
}

# Maximum turns allowed in PLANNING before falling through to EXECUTION.
# Prevents infinite loops when KB is unhelpful.
MAX_PLANNING_RETRIES = 6

PHASES_V2 = {
    "GREETING": {
        "goal": (
            "Greet the customer and understand their request. "
            "If their request is ambiguous or underspecified (e.g., 'which account "
            "gives the highest referral bonus?' without specifying account type), "
            "ask one targeted clarifying question before proceeding."
        ),
        "allowed_tools": [],
        "expect": "text",
    },
    "TRIAGE": {
        "goal": (
            "Based on the customer's request, take the appropriate next step.\n"
            "- If they need account operations (changes, disputes, transactions, closures, "
            "etc.), identify them first by asking for their user ID, name, or email, then "
            "look them up.\n"
            "- If they need information, recommendations, or general guidance "
            "(e.g., 'which card is best?', 'what are your rates?'), provide a direct "
            "answer or search the KB. Do NOT ask for identification.\n"
            "- IMPORTANT: If a request is underspecified (e.g., 'highest referral bonus' "
            "without specifying account type), ask which type they mean before searching."
        ),
        "allowed_tools": [
            "get_user_information_by_id",
            "get_user_information_by_name",
            "get_user_information_by_email",
            "KB_search",
            "give_discoverable_user_tool",
            "get_current_time",
        ],
        "expect": "either",
    },
    "ADVISORY": {
        "goal": (
            "You are researching an informational request. Use KB_search to find relevant "
            "product details, rates, fees, policies, or procedures.\n\n"
            "Search strategy for BM25 retrieval:\n"
            "- Use specific domain nouns: product names, operation types, policy keywords\n"
            "- Good: 'referral bonus checking savings account program'\n"
            "- Bad: 'which account has the best referral'\n"
            "- Search for each relevant product category separately if needed\n"
            "- Read ALL returned documents before drawing conclusions\n\n"
            "When you have enough information to give a complete, accurate answer, "
            "respond with text to the customer. Do not recommend products based on "
            "incomplete information or assumptions."
        ),
        "allowed_tools": ["KB_search", "get_current_time"],
        "expect": "either",
    },
    "VERIFICATION": {
        "goal": (
            "Verify the customer's identity. Require any 2 of: date of birth, email, "
            "phone number, address. Full name or user ID alone is not sufficient. "
            "Use log_verification to record the verification result."
        ),
        "allowed_tools": ["log_verification"],
        "expect": "tool_call",
    },
    "PLANNING": {
        "goal": (
            "You have verified the customer. Before executing ANY action, complete this "
            "planning phase:\n\n"
            "1. LIST ALL SUB-TASKS: Review the full conversation and enumerate every "
            "operation the customer wants done. Write them out.\n\n"
            "2. RESEARCH EACH OPERATION: For each sub-task, search the KB to find:\n"
            "   - The required procedure and exact tool name (format: tool_name_NNNN)\n"
            "   - Any eligibility requirements\n"
            "   - Any ordering constraints with other requested operations\n"
            "   BM25 query tips: use specific nouns — e.g., 'dispute transaction tool "
            "procedure', 'credit limit increase request tool', 'account closure eligibility'\n\n"
            "3. CHECK ORDERING CONSTRAINTS: Look for blocking relationships:\n"
            "   - Pending disputes BLOCK credit limit increases (submit limit request BEFORE "
            "filing a dispute, or resolve the dispute first)\n"
            "   - Account closure requires: no pending disputes, no pending replacements, "
            "minimum account age, zero balance\n"
            "   - Opening a new account may need to happen BEFORE closing an existing one\n"
            "   - Search KB for 'dispute credit limit policy' or similar if uncertain\n\n"
            "4. PRE-UNLOCK TOOLS: Use unlock_discoverable_agent_tool for each tool you "
            "found in step 2 (do not call the tools yet, just unlock them).\n\n"
            "5. SUMMARIZE THE PLAN: Output a brief text message stating the execution "
            "order and any ordering constraints you found. This ends the PLANNING phase."
        ),
        "allowed_tools": [
            "KB_search",
            "unlock_discoverable_agent_tool",
            "get_current_time",
        ],
        "expect": "either",
    },
    "EXECUTION": {
        "goal": (
            "Execute all sub-tasks according to the plan, in the determined order.\n\n"
            "For each step:\n"
            "- If you need a discoverable tool not yet unlocked: search KB first "
            "(KB_search), then unlock (unlock_discoverable_agent_tool), then call "
            "(call_discoverable_agent_tool)\n"
            "- VERIFY USER CLAIMS before acting on them: if the user says 'my dispute "
            "was approved' or 'I already paid the balance', look it up in the database "
            "first — never take irreversible action based solely on a user assertion\n"
            "- Confirm each result before moving to the next step\n"
            "- Use get_current_time() for any time-sensitive check (promotions, eligibility)\n\n"
            "When all steps are complete, send a text summary to the customer."
        ),
        "allowed_tools": [
            "unlock_discoverable_agent_tool",
            "call_discoverable_agent_tool",
            "list_discoverable_agent_tools",
            "give_discoverable_user_tool",
            "transfer_to_human_agents",
            "get_user_information_by_id",
            "get_user_information_by_name",
            "get_user_information_by_email",
            "change_user_email",
            "get_current_time",
            "get_referrals_by_user",
            "get_credit_card_transactions_by_user",
            "get_credit_card_accounts_by_user",
            "log_verification",
            "KB_search",
        ],
        "expect": "either",
    },
    "CONFIRMATION": {
        "goal": (
            "Summarize all actions taken for the customer. Confirm the outcome matches "
            "what they requested. Ask if they need help with anything else."
        ),
        "allowed_tools": [],
        "expect": "text",
    },
    "COMPLETE": {
        "goal": (
            "The task is complete. Say goodbye if the customer has no more questions. "
            "If they need to be escalated, use transfer_to_human_agents."
        ),
        "allowed_tools": ["transfer_to_human_agents"],
        "expect": "text",
    },
}

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


class ImperativeAgentV2(LLMConfigMixin, HalfDuplexAgent[AgentState]):
    """Agent v2 with expanded state machine addressing four τ-Banking failure modes.

    Phase sequence for account operations:
        GREETING → TRIAGE → VERIFICATION → PLANNING → EXECUTION → CONFIRMATION → COMPLETE

    Phase sequence for informational requests:
        GREETING → TRIAGE → ADVISORY → CONFIRMATION → COMPLETE

    The PLANNING phase is the key new addition: the LLM enumerates all sub-tasks,
    searches the KB for ordering constraints, and produces an explicit execution plan
    before any state-changing action is taken.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.all_tools = {tool.name: tool for tool in tools}

    @property
    def base_system_prompt(self) -> str:
        return (
            f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n"
            f"<policy>\n{self.domain_policy}\n</policy>"
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AgentState:
        if message_history is None:
            message_history = []
        return AgentState(
            system_messages=[
                SystemMessage(role="system", content=self.base_system_prompt)
            ],
            messages=message_history,
            phase="GREETING",
            phase_retries=0,
        )

    def _get_tools_for_phase(self, phase: str) -> list[Tool]:
        allowed = PHASES_V2[phase]["allowed_tools"]
        return [self.all_tools[name] for name in allowed if name in self.all_tools]

    def _build_phase_instruction(self, phase: str) -> str:
        phase_info = PHASES_V2[phase]
        allowed = [t for t in phase_info["allowed_tools"] if t in self.all_tools]
        return (
            f"\n<current_phase>\n"
            f"You are in the {phase} phase.\n"
            f"Your goal: {phase_info['goal']}\n"
            f"Available tools: {', '.join(allowed) if allowed else 'none (respond with text only)'}\n"
            f"</current_phase>"
        )

    def _determine_phase(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> str:
        current = state.phase

        # ── GREETING ────────────────────────────────────────────────────────────
        if current == "GREETING":
            if self._last_assistant_was_text(state):
                return "TRIAGE"
            return "GREETING"

        # ── TRIAGE ──────────────────────────────────────────────────────────────
        if current == "TRIAGE":
            last_tool = self._last_tool_name(state)
            if last_tool and last_tool in IDENTIFICATION_TOOLS:
                if self._last_was_successful_tool_result(state):
                    return "VERIFICATION"
                return "TRIAGE"  # lookup failed — retry
            if last_tool == "KB_search":
                if self._last_was_successful_tool_result(state):
                    return "ADVISORY"  # start informational research loop
                return "TRIAGE"
            if self._last_assistant_was_text(state):
                # Direct text answer (no KB search needed) → wrap up
                return "CONFIRMATION"
            return "TRIAGE"

        # ── ADVISORY ────────────────────────────────────────────────────────────
        if current == "ADVISORY":
            if self._last_assistant_was_text(state):
                return "CONFIRMATION"
            last_tool = self._last_tool_name(state)
            if last_tool in {"KB_search", "get_current_time"}:
                if self._last_was_successful_tool_result(state):
                    return "ADVISORY"  # continue researching
            return "ADVISORY"

        # ── VERIFICATION ────────────────────────────────────────────────────────
        if current == "VERIFICATION":
            last_tool = self._last_tool_name(state)
            if last_tool == "log_verification":
                if self._last_was_successful_tool_result(state):
                    return "PLANNING"
            return "VERIFICATION"

        # ── PLANNING ────────────────────────────────────────────────────────────
        if current == "PLANNING":
            # Text response = the LLM has summarised its plan → execute
            if self._last_assistant_was_text(state):
                return "EXECUTION"
            last_tool = self._last_tool_name(state)
            if last_tool in {"KB_search", "unlock_discoverable_agent_tool", "get_current_time"}:
                if self._last_was_successful_tool_result(state):
                    return "PLANNING"  # continue researching / unlocking
            # Safety valve: if the LLM has been stuck in PLANNING, proceed anyway
            if state.phase_retries >= MAX_PLANNING_RETRIES:
                return "EXECUTION"
            return "PLANNING"

        # ── EXECUTION ────────────────────────────────────────────────────────────
        if current == "EXECUTION":
            if self._last_assistant_was_text(state):
                return "CONFIRMATION"
            if self._last_was_successful_tool_result(state):
                return "EXECUTION"  # still executing
            return "EXECUTION"

        # ── CONFIRMATION ────────────────────────────────────────────────────────
        if current == "CONFIRMATION":
            if isinstance(message, UserMessage):
                return "TRIAGE"  # follow-up request
            return "COMPLETE"

        # ── COMPLETE ────────────────────────────────────────────────────────────
        if current == "COMPLETE":
            if isinstance(message, UserMessage):
                return "TRIAGE"
            return "COMPLETE"

        return current

    # ── Helpers (unchanged from v1) ─────────────────────────────────────────────

    def _last_was_successful_tool_result(self, state: AgentState) -> bool:
        if not state.messages:
            return False
        last = state.messages[-1]
        if isinstance(last, ToolMessage):
            return not getattr(last, "error", False)
        if isinstance(last, MultiToolMessage):
            return all(not getattr(tm, "error", False) for tm in last.tool_messages)
        return False

    def _last_assistant_was_text(self, state: AgentState) -> bool:
        for msg in reversed(state.messages):
            if isinstance(msg, AssistantMessage):
                return msg.has_text_content() and not msg.is_tool_call()
        return False

    def _last_tool_name(self, state: AgentState) -> Optional[str]:
        """Return the name of the tool called in the most recent AssistantMessage."""
        for msg in reversed(state.messages):
            if isinstance(msg, AssistantMessage) and msg.is_tool_call():
                tool_calls = msg.tool_calls or []
                return tool_calls[0].name if tool_calls else None
            if isinstance(msg, (ToolMessage, MultiToolMessage)):
                continue
            break
        return None

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio messages not supported.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        new_phase = self._determine_phase(message, state)
        if new_phase != state.phase:
            state.phase = new_phase
            state.phase_retries = 0
        else:
            state.phase_retries += 1

        phase_tools = self._get_tools_for_phase(state.phase)
        phase_instruction = self._build_phase_instruction(state.phase)

        phase_system = SystemMessage(
            role="system",
            content=self.base_system_prompt + phase_instruction,
        )
        messages = [phase_system] + state.messages

        expects_text = PHASES_V2[state.phase]["expect"] == "text"
        tools_arg = None if expects_text else phase_tools

        assistant_message = generate(
            model=self.llm,
            tools=tools_arg,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

        state.messages.append(assistant_message)
        return assistant_message, state
