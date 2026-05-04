# Banking Procedures Reference (v2)

## Purpose
Domain knowledge for τ-Banking operation categories: requirements, discoverable tool patterns,
known interdependencies, and ordering constraints.

---

## Operation Categories

### Account Information
- Balance inquiries, account details, transaction history, referral history
- Requires: customer identification and verification
- Tools: standard lookup tools always available (`get_credit_card_accounts_by_user`,
  `get_credit_card_transactions_by_user`, `get_referrals_by_user`)
- No KB search needed for basic lookups

### Credit Card Operations
- Card replacement, card blocking/freezing/unfreezing, closing, activation
- Requires: verified customer, card account ID
- **Always search KB** — card tier determines replacement options, fees, and waiting periods
- Replacement reasons: `lost`, `stolen`, `damaged`, `fraud` — use exact strings from KB
- After freezing all cards for "lost wallet": check if the customer also wants replacement
- Replacement flow: freeze → (confirm lost) → close old → order new → (wait for delivery) → activate
- After replacing, some card tiers impose a waiting period before a new replacement can be ordered

### Credit Limit Changes
- Increase/decrease requests, temporary limit adjustments
- Requires: verified customer
- **Always search KB** — discoverable tools required
- **Critical constraint**: The request is **automatically rejected** if the customer has any
  pending disputes on record — handle disputes last, or resolve them first
- Check for any pending applications before submitting a new request

### Disputes and Chargebacks
- Transaction disputes, unauthorized charges, cash-back reward corrections
- Requires: verified customer, transaction ID
- **Always search KB** — dispute filing uses discoverable tools with specific required fields
- After calling the dispute tool: status is `UNDER_REVIEW` — do **NOT** apply credits
  or rewards adjustments until the dispute is confirmed approved via a database lookup
- For cash-back corrections: independently recalculate the correct reward amount;
  do not use the `expected_rewards` field from the dispute record

### Payments and Transfers
- Credit card payment from checking, balance transfers, account-to-account transfers
- Requires: verified customer, source/destination account IDs, exact amount
- Some operations require discoverable tools — search KB
- Always confirm the amount and destination before executing a payment

### Account Lifecycle
- Opening, closing, upgrading, downgrading personal and business accounts
- Requires: verified customer
- **Always search KB** — lifecycle operations have strict eligibility requirements
- **Closure eligibility checklist** (verify ALL before closing):
  1. No active or pending transaction disputes
  2. No pending replacement card orders not yet received/activated
  3. Minimum account age met (varies by product — look up in KB)
  4. No outstanding balance (arrange payment first if balance exists)
  5. Check previous closure attempt history (may skip retention offers if prior attempt exists)
- **Opening eligibility**: may require an existing qualifying account; check KB for prerequisites
- Business accounts: check whether personal account status affects business account eligibility

### Profile Updates
- Email: use `change_user_email` (standard tool, no KB search needed)
- Other contact info (address, phone): search KB for the correct tool

---

## Operation Interdependencies

When the customer requests multiple operations, check these known blocking relationships:

| If the customer wants… | And also wants… | Ordering constraint |
|---|---|---|
| Dispute a charge | Credit limit increase | Submit credit limit request **BEFORE** filing the dispute (dispute creates a pending record that auto-rejects limit increases) |
| Close an account | Open a new account of a type that requires an existing account | **Open first**, then close |
| Close Account A | Open savings (tenure requirement) | If closing Account A would leave only a new account, opening savings **first** may be required to satisfy tenure |
| Transfer funds | Close source account | **Transfer first**, then close (account must reach zero balance) |
| Close a credit card | Apply fee waiver / retention offer | Follow the retention protocol order: check eligibility → check previous attempts → log reason → offer solution |
| Freeze a card | Replace a card | Freeze first (immediate security), then handle replacement |

---

## Multi-Request Handling Protocol

When the customer presents multiple requests in one message:
1. Write out all requested operations explicitly
2. Check the interdependency table above
3. Identify any operations that would block another
4. Re-sequence into a safe execution order
5. Explain the required order to the customer if it differs from their request
6. Execute in the determined order — do not follow the customer's stated order if it would fail

---

## Time-Sensitive Operations

Some operations depend on the current date:
- Active vs expired promotions (APY boosts, referral bonuses, fee waivers with end dates)
- Minimum account age checks
- Fee waiver expiration dates when applying loyalty benefits

**Always use `get_current_time()`** — never assume or guess the current date.

---

## Decision Framework

1. **Standard tool available and sufficient?** → Use it directly
2. **Requires discoverable tool?** → Search KB → unlock → call (in that order)
3. **Multiple operations?** → Check interdependency table and plan order first
4. **User claims system state (e.g., "my dispute was approved")?** → Verify with database lookup
5. **Outside your capabilities or KB?** → `transfer_to_human_agents` (ask customer first)
6. **Customer request violates policy?** → Explain the constraint; do not execute

---

## Common Patterns

- **Always confirm** before irreversible actions: account closures, payments, card cancellations
- **Never apply credits or rewards** based solely on user assertions — verify dispute/promotion status
- **Check for outstanding balances** before account closure
- **Check account age** before closure or product upgrades with tenure requirements
- **Never use `transfer_to_human_agents` prematurely** — attempt KB search first before escalating
