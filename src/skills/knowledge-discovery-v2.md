# Knowledge Base and Tool Discovery (v2)

## Purpose
Guide KB search strategy to efficiently find procedures, policies, product details, and tool
names — covering both the BM25 keyword-retrieval case and the golden-retrieval case.

---

## When to Search the KB

Search the KB in ALL of these situations:
- Any operation that requires a discoverable tool (most account modifications)
- Any time you need the step-by-step procedure for an operation
- Any time you need product specs, rates, fees, or eligibility requirements
- Any time you need to check ordering constraints between operations
- Any time you need to understand a policy's edge cases or exceptions

**Do NOT skip KB search** for operations requiring discoverable tools — tool names contain
random 4-digit suffixes (e.g., `close_bank_account_7392`) and cannot be guessed.

---

## KB Search Query Construction (BM25 / Keyword-Based)

The retrieval system uses keyword matching. Poorly formed queries return irrelevant documents.

### Query principles:
- **Use specific nouns and verbs**: `credit limit increase request` not `how do I change a limit`
- **Include the product name**: `Platinum Rewards Card replacement` not `card replacement`
- **Use domain-specific terms**: `dispute`, `chargeback`, `closure`, `referral`, `replacement`,
  `freeze`, `block`, `activation`, `waiver`, `retention`, `cancellation`
- **For internal procedures**: add `internal`, `procedure`, `protocol`, or `step` to the query
- **For tool documentation**: search for the operation name + `tool` or `procedure`
- **For policy constraints**: search for `eligibility`, `requirements`, or `policy` + operation

### Example queries:
| Goal | Good query |
|---|---|
| Card replacement procedure | `order replacement credit card lost stolen procedure` |
| Credit limit tool | `credit limit increase request tool procedure` |
| Account closure eligibility | `bank account closure eligibility requirements` |
| Dispute filing | `file dispute transaction unauthorized charge internal tool` |
| Referral programs | `referral program bonus checking savings account` |
| Fee waiver | `annual fee waiver loyalty retention credit card` |

### What NOT to do:
- `"how do I help a customer"` — too vague, no keywords
- `"tool to close account"` — lacks product context; use `close bank account 7392` only after you've found the name in KB
- Guessing tool names from partial matches in memory

---

## Iterative Search Strategy

Single queries rarely capture everything needed for complex tasks. Use an iterative approach:

1. **Start with the primary operation keyword**
   - e.g., `credit limit increase request`
2. **Read ALL results fully** before drawing conclusions
3. **If results mention related documents or cross-references**, search for those too
   - e.g., if a procedure document says "see the dispute pending policy," search for that
4. **Refine with synonyms** if the first query returns irrelevant documents
   - `cancel` vs `close`, `freeze` vs `block`, `upgrade` vs `downgrade`
5. **Search separately for each operation** when the customer has multiple requests
   - Do not try to find all procedures in one query

---

## Tool Discovery Workflow

### Step 1: Search for the Procedure
```
KB_search("specific operation name + type")
```

### Step 2: Extract the Exact Tool Name
- Locate function signatures in the format: `tool_name_NNNN`
- Copy the **complete** name including the 4-digit suffix
- **Never modify, truncate, or guess the suffix** — it is not recoverable from partial information

### Step 3: Unlock the Tool
```
unlock_discoverable_agent_tool(agent_tool_name="exact_tool_name_NNNN")
```
- The unlock response describes the tool's parameters — **read the entire response**
- Only unlock tools you will **immediately use** — speculative unlocking creates database issues

### Step 4: Call the Tool
```
call_discoverable_agent_tool(
    agent_tool_name="exact_tool_name_NNNN",
    arguments='{"param1": "value1", "param2": "value2"}'
)
```
- `arguments` must be a **JSON string** (not a Python dict)
- Parameter names and types come from the unlock response — use them exactly

---

## Golden Retrieval (documents already in context)

When the relevant documents are provided directly in your system context:
- Read them carefully before acting
- Extract tool names, parameter names, and procedure steps from those documents
- Still follow the unlock → call sequence even in golden retrieval
- Do not skip required procedure steps just because documents are available

---

## Multi-Operation Research Plan

For tasks requiring multiple operations, **plan your research before executing**:

1. List every operation the customer needs
2. For each operation: search KB separately and extract the tool name and procedure
3. Search for any policy documents describing interactions between the operations
   (e.g., "does pending dispute affect credit limit?" → search `dispute credit limit policy`)
4. With a complete picture, determine the execution order
5. Begin execution only after the research phase is complete

This prevents discovering mid-execution that the operations needed to be in a different order.

---

## Error Handling

- **No useful results after 2-3 queries**: Try alternative terms; if still nothing, transfer to
  human agent and explain the limitation
- **Unlock fails**: The tool name is likely wrong — re-check the KB result for the exact name
- **Call fails with argument error**: Re-read the unlock response for correct parameter names
  and types; ensure `arguments` is a valid JSON string
- **Partial results only**: Issue a follow-up query for the missing information before proceeding

---

## Guardrails

- **NEVER guess tool names** — the numeric suffix makes guessing impossible
- **ALWAYS read the complete unlock response** before calling a tool
- **Do NOT unlock speculatively** — only unlock tools you will immediately use
- **The `arguments` parameter must be a JSON string** enclosed in single quotes or similar
  — not a Python dict (`{}`) or other data structure
- **For procedures with multiple steps**, execute each step in the documented order;
  skipping steps (e.g., skipping the balance check before account closure) causes failures
