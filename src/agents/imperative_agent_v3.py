"""Imperative agent v3: deterministic state-machine with explicit task queue.

Six deterministic improvements over v2
---------------------------------------
1. Explicit task queue — PLANNING output parsed into pending_tasks list; each
   execution step pops the next task deterministically without re-querying the LLM.
2. Topological task ordering — MUST_PRECEDE_RULES applied via Kahn's algorithm
   before execution starts; eliminates ordering errors for multi-task requests.
3. State-driven phase transitions — boolean flags (user_identified, verified) and
   queue depth drive ALL phase transitions; message-type heuristics are a fallback.
4. Verification hard gate — EXECUTION phase is blocked unless state.verified == True;
   the agent deterministically loops in VERIFICATION until log_verification succeeds.
5. Tool retry policy — per-tool retry counts in state.tool_retry_counts; exceeding
   TOOL_RETRY_POLICY[tool].max_retries triggers deterministic ESCALATE transition.
6. Strict response-type enforcement — _enforce_expect() re-prompts with
   tool_choice="required" (up to MAX_EXPECT_RETRIES) when the model returns the
   wrong response type for the current phase.
"""

import re
from dataclasses import dataclass
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

from agents.cached_generate import generate_cached as generate
from agents.state import AgentState

# ---------------------------------------------------------------------------
# Tool sets
# ---------------------------------------------------------------------------

IDENTIFICATION_TOOLS = {
    "get_user_information_by_id",
    "get_user_information_by_name",
    "get_user_information_by_email",
}

# Tools whose successful completion marks one pending task as done
EXECUTION_COMPLETION_TOOLS = {
    "call_discoverable_agent_tool",
    "change_user_email",
    "give_discoverable_user_tool",
    "transfer_to_human_agents",
}

# ---------------------------------------------------------------------------
# Strategy 5: Tool retry policy
# ---------------------------------------------------------------------------

@dataclass
class ToolRetryPolicy:
    max_retries: int
    failure_phase: str  # phase to transition to after exhausting retries

TOOL_RETRY_POLICY: dict[str, ToolRetryPolicy] = {
    "log_verification":               ToolRetryPolicy(max_retries=3, failure_phase="ESCALATE"),
    "call_discoverable_agent_tool":   ToolRetryPolicy(max_retries=2, failure_phase="ESCALATE"),
    "unlock_discoverable_agent_tool": ToolRetryPolicy(max_retries=2, failure_phase="ESCALATE"),
    "KB_search":                      ToolRetryPolicy(max_retries=4, failure_phase="ADVISORY"),
}

# ---------------------------------------------------------------------------
# Strategy 2: Ordering constraints for topological sort
# ---------------------------------------------------------------------------

# (earlier_keyword, later_keyword): if task A contains kw_a and task B contains kw_b,
# A must precede B in the execution queue.
MUST_PRECEDE_RULES: list[tuple[str, str]] = [
    ("credit limit",  "dispute"),    # request credit limit BEFORE filing a dispute
    ("open",          "clos"),       # open/create account BEFORE closing another
    ("transfer",      "clos"),       # transfer funds BEFORE account closure
    ("replacement",   "clos"),       # resolve replacement BEFORE closure
    ("upgrade",       "clos"),       # apply upgrades first, then close
    ("balance",       "clos"),       # clear balance BEFORE closure
]

MAX_PLANNING_RETRIES = 6
MAX_EXPECT_RETRIES = 2

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASES_V3 = {
    "GREETING": {
        "goal": (
            "Greet the customer and understand their request. "
            "If the request is ambiguous, ask one targeted clarifying question."
        ),
        "allowed_tools": [],
        "expect": "text",
    },
    "TRIAGE": {
        "goal": (
            "Determine the next step based on request type:\n"
            "- Account operations (changes, disputes, closures, transactions): "
            "ask for user ID, name, or email then look them up.\n"
            "- Informational requests (rates, comparisons, policies): "
            "search the KB directly; no identification needed.\n"
            "Do NOT ask for identity for purely informational requests."
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
            "Research the informational request using KB_search.\n"
            "BM25 query tips: use specific domain nouns — product names, policy keywords.\n"
            "Good: 'referral bonus checking savings account'\n"
            "Bad: 'which account has the best referral'\n"
            "Search each product category separately. Read ALL returned documents.\n"
            "When you have enough information, respond with text to the customer."
        ),
        "allowed_tools": ["KB_search", "get_current_time"],
        "expect": "either",
    },
    "VERIFICATION": {
        "goal": (
            "Verify the customer's identity. Require any 2 of: date of birth, email, "
            "phone number, address. Name or user ID alone is NOT sufficient. "
            "Call log_verification to record the result."
        ),
        "allowed_tools": ["log_verification"],
        "expect": "tool_call",
    },
    "PLANNING": {
        "goal": (
            "Plan all operations before executing any. Follow these steps:\n\n"
            "1. LIST ALL SUB-TASKS: Review the conversation and enumerate every "
            "operation the customer needs. Write each as a numbered line.\n\n"
            "2. RESEARCH EACH OPERATION: For each sub-task, use KB_search to find:\n"
            "   - The exact tool name (format: tool_name_NNNN)\n"
            "   - Eligibility requirements and ordering constraints\n"
            "   BM25 tips: 'dispute transaction procedure', 'credit limit increase tool'\n\n"
            "3. CHECK ORDERING CONSTRAINTS:\n"
            "   - Pending disputes BLOCK credit limit increases\n"
            "   - Account closure requires: no disputes, no pending replacements, "
            "minimum age, zero balance\n\n"
            "4. PRE-UNLOCK TOOLS: Call unlock_discoverable_agent_tool for each tool found.\n\n"
            "5. SUMMARIZE: Output a text message with this exact format:\n"
            "TASKS:\n"
            "1. <task description>\n"
            "2. <task description>\n"
            "END_TASKS\n"
            "Then briefly explain any ordering constraints found."
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
            "Execute sub-tasks in the planned order (see <task_queue> below).\n\n"
            "For each task:\n"
            "- If a discoverable tool is not yet unlocked: "
            "KB_search → unlock_discoverable_agent_tool → call_discoverable_agent_tool\n"
            "- VERIFY user claims against the database before acting on them\n"
            "- Confirm each result before moving to the next task\n"
            "- Use get_current_time for time-sensitive eligibility checks\n\n"
            "When all tasks are complete, send a text summary to the customer."
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
            "Summarize all actions taken. Confirm the outcome matches what the customer "
            "requested. Ask if they need anything else."
        ),
        "allowed_tools": [],
        "expect": "text",
    },
    "ESCALATE": {
        "goal": (
            "A critical tool has failed too many times. Apologize to the customer, "
            "briefly explain the issue, and transfer them to a human agent."
        ),
        "allowed_tools": ["transfer_to_human_agents"],
        "expect": "tool_call",
    },
    "COMPLETE": {
        "goal": "The task is complete. Say goodbye or handle follow-up requests.",
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

Try to be helpful and always follow the policy. Always generate valid JSON only.
""".strip()


class ImperativeAgentV3(LLMConfigMixin, HalfDuplexAgent[AgentState]):
    """Agent v3 with deterministic task queue and state-driven phase transitions.

    Phase sequences:
      Account operations:
        GREETING → TRIAGE → VERIFICATION → PLANNING → EXECUTION → CONFIRMATION → COMPLETE
      Informational:
        GREETING → TRIAGE → ADVISORY → CONFIRMATION → COMPLETE
      Tool retry failures:
        any → ESCALATE → COMPLETE
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
            system_messages=[SystemMessage(role="system", content=self.base_system_prompt)],
            messages=message_history,
            phase="GREETING",
            phase_retries=0,
            user_identified=False,
            verified=False,
            pending_tasks=[],
            completed_tasks=[],
            tool_retry_counts={},
            expect_violation_count=0,
        )

    # -------------------------------------------------------------------------
    # Strategy 5: Tool retry policy
    # -------------------------------------------------------------------------

    def _retry_limit_exceeded(self, state: AgentState) -> Optional[str]:
        """Return the failure_phase if any tool has exceeded its retry limit, else None."""
        for tool_name, count in state.tool_retry_counts.items():
            policy = TOOL_RETRY_POLICY.get(tool_name)
            if policy and count >= policy.max_retries:
                return policy.failure_phase
        return None

    # -------------------------------------------------------------------------
    # Strategies 1 & 2: Task queue with topological sort
    # -------------------------------------------------------------------------

    def _parse_tasks_from_planning(self, text: str) -> list[str]:
        """Extract task list from PLANNING summary text.

        Looks for TASKS:/END_TASKS block first, then falls back to numbered lines.
        """
        block_match = re.search(
            r"TASKS:\s*\n(.*?)END_TASKS", text, re.DOTALL | re.IGNORECASE
        )
        if block_match:
            lines = block_match.group(1).strip().split("\n")
            tasks = []
            for line in lines:
                clean = re.sub(r"^\s*\d+\.\s*", "", line).strip()
                if clean:
                    tasks.append(clean)
            return tasks

        # Fallback: any numbered list in text
        tasks = re.findall(r"^\s*\d+\.\s+(.+)$", text, re.MULTILINE)
        return [t.strip() for t in tasks if t.strip()]

    def _sort_tasks_by_dependencies(self, tasks: list[str]) -> list[str]:
        """Sort tasks via Kahn's topological sort using MUST_PRECEDE_RULES.

        For rule (kw_a, kw_b): if task A contains kw_a and task B contains kw_b,
        A must precede B.
        """
        n = len(tasks)
        predecessors: list[set[int]] = [set() for _ in range(n)]

        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                if i == j:
                    continue
                for kw_a, kw_b in MUST_PRECEDE_RULES:
                    if kw_a.lower() in ti.lower() and kw_b.lower() in tj.lower():
                        predecessors[j].add(i)

        queue = [i for i in range(n) if not predecessors[i]]
        result: list[str] = []

        while queue:
            idx = queue.pop(0)
            result.append(tasks[idx])
            for j in range(n):
                predecessors[j].discard(idx)
                if not predecessors[j] and tasks[j] not in result:
                    queue.append(j)

        # Append any tasks not reached (cycles — defensive fallback)
        already = set(result)
        for t in tasks:
            if t not in already:
                result.append(t)

        return result

    def _parse_and_queue_tasks(self, state: AgentState) -> None:
        """Parse task list from the most recent PLANNING text and store in state."""
        for msg in reversed(state.messages):
            if (
                isinstance(msg, AssistantMessage)
                and msg.has_text_content()
                and not msg.is_tool_call()
            ):
                text = self._get_text_content(msg)
                tasks = self._parse_tasks_from_planning(text)
                if tasks:
                    state.pending_tasks = self._sort_tasks_by_dependencies(tasks)
                return

    # -------------------------------------------------------------------------
    # Strategies 3 & 4: State-driven phase transitions
    # -------------------------------------------------------------------------

    def _update_task_state(self, state: AgentState) -> None:
        """Update boolean flags and task queue from the most recent ToolMessage result.

        Called BEFORE _determine_phase so that state reflects the result just received.
        """
        if not state.messages:
            return
        last_msg = state.messages[-1]
        if not isinstance(last_msg, ToolMessage):
            return

        tool_name = self._get_tool_name_for_result(last_msg, state)
        success = not getattr(last_msg, "error", False)

        if not tool_name:
            return

        if success:
            if tool_name in IDENTIFICATION_TOOLS:
                state.user_identified = True
            elif tool_name == "log_verification":
                state.verified = True
            elif tool_name in EXECUTION_COMPLETION_TOOLS and state.pending_tasks:
                done = state.pending_tasks.pop(0)
                state.completed_tasks.append(done)
        else:
            state.tool_retry_counts[tool_name] = (
                state.tool_retry_counts.get(tool_name, 0) + 1
            )

    def _get_tool_name_for_result(
        self, result_msg: ToolMessage, state: AgentState
    ) -> Optional[str]:
        """Find the tool name that produced result_msg by matching tool_call_id."""
        result_call_id = getattr(result_msg, "tool_call_id", None)
        # Scan backwards; exclude the last message (the result itself)
        for msg in reversed(state.messages[:-1]):
            if isinstance(msg, AssistantMessage) and msg.is_tool_call():
                for tc in msg.tool_calls or []:
                    tc_id = getattr(tc, "id", None)
                    if result_call_id is None or tc_id is None or tc_id == result_call_id:
                        return tc.name
        return None

    def _determine_phase(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> str:
        current = state.phase

        # Strategy 5: any phase can escalate when retry limit is exceeded
        failure_phase = self._retry_limit_exceeded(state)
        if failure_phase and current not in ("ESCALATE", "COMPLETE"):
            return failure_phase

        # ── GREETING ────────────────────────────────────────────────────────
        if current == "GREETING":
            if self._last_assistant_was_text(state):
                return "TRIAGE"
            return "GREETING"

        # ── TRIAGE ──────────────────────────────────────────────────────────
        if current == "TRIAGE":
            # Strategy 3: use explicit boolean flag, not message heuristic
            if state.user_identified:
                return "VERIFICATION"
            last_tool = self._last_tool_name(state)
            if last_tool == "KB_search" and self._last_was_successful_tool_result(state):
                return "ADVISORY"
            # Direct text answer (no lookup, no KB) → informational shortcut
            if self._last_assistant_was_text(state) and last_tool is None:
                return "CONFIRMATION"
            return "TRIAGE"

        # ── ADVISORY ────────────────────────────────────────────────────────
        if current == "ADVISORY":
            if self._last_assistant_was_text(state):
                return "CONFIRMATION"
            return "ADVISORY"

        # ── VERIFICATION ────────────────────────────────────────────────────
        if current == "VERIFICATION":
            # Strategy 4: hard gate — only exit when state.verified is True
            if state.verified:
                return "PLANNING"
            return "VERIFICATION"

        # ── PLANNING ────────────────────────────────────────────────────────
        if current == "PLANNING":
            if self._last_assistant_was_text(state):
                self._parse_and_queue_tasks(state)
                return "EXECUTION"
            if state.phase_retries >= MAX_PLANNING_RETRIES:
                return "EXECUTION"
            return "PLANNING"

        # ── EXECUTION ────────────────────────────────────────────────────────
        if current == "EXECUTION":
            # Strategy 4: never execute unless verified
            if not state.verified:
                return "VERIFICATION"
            # Strategy 1: use task queue to decide completion
            if not state.pending_tasks and self._last_assistant_was_text(state):
                return "CONFIRMATION"
            return "EXECUTION"

        # ── CONFIRMATION ────────────────────────────────────────────────────
        if current == "CONFIRMATION":
            if isinstance(message, UserMessage):
                return "TRIAGE"
            return "COMPLETE"

        # ── ESCALATE ────────────────────────────────────────────────────────
        if current == "ESCALATE":
            return "COMPLETE"

        # ── COMPLETE ────────────────────────────────────────────────────────
        if current == "COMPLETE":
            if isinstance(message, UserMessage):
                return "TRIAGE"
            return "COMPLETE"

        return current

    # -------------------------------------------------------------------------
    # Strategy 6: Strict response-type enforcement
    # -------------------------------------------------------------------------

    def _enforce_expect(
        self,
        phase: str,
        tools_arg: Optional[list],
        tool_choice: Optional[str],
        messages: list,
        state: AgentState,
    ) -> AssistantMessage:
        """Generate a response; retry up to MAX_EXPECT_RETRIES if type mismatches expect."""
        expect = PHASES_V3[phase]["expect"]

        for attempt in range(MAX_EXPECT_RETRIES + 1):
            response = generate(
                model=self.llm,
                tools=tools_arg,
                tool_choice=tool_choice,
                messages=messages,
                call_name=f"agent_{phase.lower()}",
                **self.llm_args,
            )

            got_tool = response.is_tool_call()
            got_text = response.has_text_content() and not got_tool

            if expect == "either" or (expect == "tool_call" and got_tool) or (expect == "text" and got_text):
                return response

            state.expect_violation_count += 1
            if attempt >= MAX_EXPECT_RETRIES:
                return response  # best-effort fallback

            if expect == "tool_call" and not got_tool:
                tool_choice = "required"
            elif expect == "text" and got_tool:
                correction = SystemMessage(
                    role="system",
                    content=(
                        f"You are in the {phase} phase. Respond with text only — "
                        "do NOT make a tool call. Address the customer directly."
                    ),
                )
                messages = [correction] + messages[1:]

        return response  # unreachable but satisfies type checker

    # -------------------------------------------------------------------------
    # Phase helpers
    # -------------------------------------------------------------------------

    def _get_tools_for_phase(self, phase: str) -> list[Tool]:
        allowed = PHASES_V3[phase]["allowed_tools"]
        return [self.all_tools[name] for name in allowed if name in self.all_tools]

    def _build_phase_instruction(self, phase: str, state: AgentState) -> str:
        phase_info = PHASES_V3[phase]
        allowed = [t for t in phase_info["allowed_tools"] if t in self.all_tools]

        queue_hint = ""
        if phase == "EXECUTION":
            queue_hint = (
                "\n<task_queue>\n"
                f"Pending: {', '.join(f'{i+1}. {t}' for i, t in enumerate(state.pending_tasks)) or 'none — write a summary to the customer'}\n"
                f"Completed: {', '.join(state.completed_tasks) or 'none'}\n"
                "</task_queue>"
            )

        return (
            f"\n<current_phase>\n"
            f"Phase: {phase}\n"
            f"Goal: {phase_info['goal']}\n"
            f"Available tools: {', '.join(allowed) if allowed else 'none (text response only)'}\n"
            f"</current_phase>"
            f"{queue_hint}"
        )

    # -------------------------------------------------------------------------
    # Message helpers (same as v2)
    # -------------------------------------------------------------------------

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
        for msg in reversed(state.messages):
            if isinstance(msg, AssistantMessage) and msg.is_tool_call():
                tool_calls = msg.tool_calls or []
                return tool_calls[0].name if tool_calls else None
            if isinstance(msg, (ToolMessage, MultiToolMessage)):
                continue
            break
        return None

    def _get_text_content(self, msg: AssistantMessage) -> str:
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            parts = []
            for p in msg.content:
                if hasattr(p, "text"):
                    parts.append(p.text)
                elif isinstance(p, str):
                    parts.append(p)
            return " ".join(parts)
        return ""

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Audio messages not supported.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Strategy 3: update boolean flags BEFORE phase determination
        self._update_task_state(state)

        new_phase = self._determine_phase(message, state)
        if new_phase != state.phase:
            state.phase = new_phase
            state.phase_retries = 0
        else:
            state.phase_retries += 1

        phase_tools = self._get_tools_for_phase(state.phase)
        phase_instruction = self._build_phase_instruction(state.phase, state)

        phase_system = SystemMessage(
            role="system",
            content=self.base_system_prompt + phase_instruction,
        )
        messages = [phase_system] + state.messages

        expects_text = PHASES_V3[state.phase]["expect"] == "text"
        history_has_tool_calls = any(
            (isinstance(m, AssistantMessage) and m.is_tool_call())
            or isinstance(m, (ToolMessage, MultiToolMessage))
            for m in state.messages
        )

        if expects_text and history_has_tool_calls:
            # Anthropic requires tools= in context; force tool_choice="none"
            tools_arg = list(self.all_tools.values())
            tool_choice = "none"
        elif expects_text:
            tools_arg = None
            tool_choice = None
        else:
            tools_arg = phase_tools
            tool_choice = None

        # Strategy 6: enforce expected response type
        assistant_message = self._enforce_expect(
            state.phase, tools_arg, tool_choice, messages, state
        )

        state.messages.append(assistant_message)
        return assistant_message, state
