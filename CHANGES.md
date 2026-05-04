# Clarification Loop — Changes vs v18.7

This document describes the six fixes applied to the v18.7 clarification
loop and why each one matters.

---

## Fix #1 — LLM uses a constrained menu of real glossary terms

**File:** `core/clarification.py`

**Problem in v18.7.** The `_llm_ambiguity_check` prompt explicitly tells
the model not to generate options (rule 3), and `check_ambiguity_glossary_first`
then sets `options=[]` on the return regardless. Downstream,
`combine_with_clarification` walks `meta["options"]` to find the chosen
interpretation and build a SQL injection string. Empty options means no
injection. LLM-sourced clarifications degrade to "append free-text to
original question and re-ask the SQL LLM with no locked-in term
expression." The clarification adds almost no value.

**Fix.** `_llm_ambiguity_check_constrained` builds a menu from real
glossary terms (ranked by question-word overlap, capped at 20 terms),
gives it to the model, and constrains the model to pick 2–3 `option_id`
values **from the menu**. Returned options carry real `_term_id` and
`expression` fields. The retry prompt locks in the exact SQL formula the
user picked.

If the glossary has fewer than 2 terms, the code falls back to the plain
CLEAR/AMBIGUOUS classifier (same as before). The new constrained path
also returns CLEAR when the model invents unknown option IDs or returns
fewer than 2 valid IDs — no more hallucinated options leaking through.

---

## Fix #2 — `selected_option_id` flows from dispatch to combine

**Files:** `core/clarification.py`, `main.py`

**Problem in v18.7.** Dispatch calls `resolve_option_text(opts, text)`,
which does exact → substring → word-overlap matching. If it returns a
match, only `match["value"]` is forwarded. `combine_with_clarification`
receives that text and re-runs its *own* matching loop — only exact
match against `meta["options"]`. If the user's reply matched by
word-overlap in dispatch but doesn't equal any label/value exactly,
combine silently returns empty injection. Same user, same reply, two
matchers, two answers. On two-option menus with overlapping labels
("Late days" vs "Late/absent days"), the chosen option can flip between
the two stages non-deterministically.

**Fix.** Dispatch now captures `match["id"]` and forwards it as
`selected_option_id=...` to `combine_with_clarification`. `_pick_option`
honours the explicit ID first; if the ID isn't in the options list, it
returns None rather than falling back to text matching. The two stages
are guaranteed to agree.

---

## Fix #3 — Zero-rows clarification is gated on ambiguity signals

**File:** `main.py`

**Problem in v18.7.** Every query that returns zero rows calls
`check_ambiguity_glossary_first`. That's a full LLM ambiguity check on
every empty result set — most of which are legitimately empty
("revenue last week" at the start of a new week). Wastes tokens and
confuses users who get a clarification prompt when their filter was
correct.

**Fix.** Before calling the ambiguity check on zero rows, check whether
the question matches any glossary term with `requires_clarification=True`
OR matches two or more distinct metrics. Only those two cases trigger
the LLM path. Otherwise the user gets a clean "The query ran
successfully but returned no rows. Try broadening the filter…"
message. Saves one LLM call per empty-result query.

---

## Fix #5 — Tolerant JSON parsing for LLM ambiguity responses

**File:** `core/clarification.py`

**Problem in v18.7.** `_llm_ambiguity_check` does `json.loads(raw)` in
a try/except. If the model wraps its JSON in ` ```json … ``` ` fences,
or adds a preamble ("Here is the JSON:"), or appends trailing text, the
parse fails and falls through to a regex that looks for literal
"AMBIGUOUS:" text. If that also doesn't match, the check returns
"not ambiguous". Silent failure — every LLM clarification for that
query is lost with no log warning.

**Fix.** `_parse_ambiguity_json` strips triple-backtick fences, scans
for the first `{`, balances braces while respecting string literals,
and extracts the JSON. Handles pure JSON, fenced JSON, preamble text,
trailing text, and strings containing `{` or `}`. Logs a warning when
parsing fails (no more silent losses) and returns None cleanly.

---

## Fix #7 — User is told when their clarification expires

**Files:** `core/clarification.py`, `main.py`

**Problem in v18.7.** Pending clarifications expire in 5 minutes. If a
user sees a prompt, steps away for 6 minutes, then replies — their reply
hits `get_pending` → None → is processed as a fresh query. They wonder
why the bot "forgot" the context.

**Fix.** When `get_pending` evicts an expired row, it calls
`mark_recently_expired(account_id, zoom_user_id)`. The marker lives in
an in-process dict with a 10-minute TTL. The dispatcher, after not
finding a pending clarification, checks `was_recently_expired`. If true
AND the user's reply looks short/clarification-ish (≤6 words), it sends:

> ⏱️ Your previous clarification request timed out. Please ask your
> original question again and I'll pick it up from there.

Single-worker only; see README note about multi-worker scaling.

---

## Fix #8 — Webhook idempotency

**Files:** `core/webhook_dedup.py` (new), `main.py`

**Problem in v18.7.** Zoom, Slack, and Teams all document at-least-once
webhook delivery. A duplicate that lands within the 5-minute pending
window causes: first delivery saves pending + dispatches query; second
delivery sees the same pending (or the state has already transitioned)
and runs the message as a duplicate query. Two LLM calls, two DB
queries, two replies, possibly two interpretations.

**Fix.** New `core/webhook_dedup.py` maintains a TTL cache (120s,
bounded to 10k entries) keyed on:
- Zoom:   `zoom:{account}:{user}:{message_id or timestamp}`
- Slack:  `slack:{account}:{user}:{event_id}`
- Teams:  `teams:{account}:{user}:{activity_id}`

With no stable platform ID, falls back to a content hash. The three
webhook handlers in `main.py` call `is_duplicate_event(event)` before
`dispatch(...)` and short-circuit with a 200 on duplicates.

Single-worker only. Move to Redis if you scale workers.

---

## Fix #9 — WebSocket accepts free-text clarifications

**File:** `main.py`

**Problem in v18.7.** The WebSocket `clarification_response` handler
only accepts replies carrying an `option_id`. If a clarification has
no options (which, with Fix #1, can still happen — sparse-glossary
fallback returns no options), the user has no way to forward their
free-text reply. Their typed reply falls through to the generic
message handler and runs as a fresh query.

**Fix.** The handler now branches on whether `cmeta["options"]` is
populated. With options present, it tries `option_id` first, then
falls back to `resolve_option_text` on the provided text. With no
options, it accepts the `text` field directly as a free-text
clarification. Also passes `selected_option_id` through to combine
(Fix #2 compatibility).

---

## Items deliberately NOT in this drop

Things I considered but left for a later round:

1. **Replace `_looks_like_new_query` with a cheap LLM classifier.**
   High-value but changes dispatch semantics enough that it deserves
   its own testing cycle. The current heuristic still works for the
   common cases.

2. **Multi-turn clarifications (lift the `is_clarification=True`
   absolute block).** The current single-turn ceiling is a conscious
   safety net against infinite loops. Lifting it properly needs a
   turn counter in `pending_clarification` and clear per-turn rules.

3. **Clarification logging for glossary auto-promotion.** Writing every
   resolved clarification to a log table and using it to promote hot
   terms to `requires_clarification=True` is high-leverage but
   involves schema changes and a new review UI.

4. **`_build_glossary_hint` question-overlap ranking.** The v18.7 code
   caps at 30 terms without ranking. Fix #1 added ranking inside the
   constrained-menu path; the glossary hint used by step 3's fallback
   classifier still truncates unranked. Low priority because the
   constrained-menu path is the primary one now.


---

# LLM Audit v2 refinements (April 17, 2026)

Four follow-up fixes on top of the v18.8 LLM audit build. All four are code-only;
no schema changes. The `llm_call_log` table and `client.enable_llm_audit` flag
from v18.8 are untouched.

## Fix A — Smarter quoted-literal masking in sanitizer

**File:** `core/llm_audit.py`

**Problem.** The v1 sanitizer regex `'[^'\n]{2,120}'` was aggressive: every
quoted string in a prompt was redacted to `'[literal]'`. The audit tab became
hard to debug because prompts like `WHERE status = 'Active'` showed up as
`WHERE status = '[literal]'`, destroying the signal an admin actually wants
(which categorical value the LLM was asked to filter on).

Worse, because the regex required `{2,120}`, a single-char literal like `'Y'`
was skipped and the regex then matched **across** it — e.g. in
`'Y' AND attendance = 'Late'`, the match became `' AND attendance = '`, which
looked like one long literal and got masked.

**Fix.**
- Quoted-literal regex now matches `{1,120}` chars so short literals don't
  break the scan.
- A `_looks_like_data_value()` callback decides per-match whether to mask.
- Known-safe short categoricals (`Active`, `Late`, `Y`, `N`, `Male`, `Open`,
  etc.) are always preserved.
- Multi-word phrases, proper-noun-shaped tokens, and literals ≥10 chars are
  always masked.

## Fix B — Long-token regex no longer eats identifiers

**File:** `core/llm_audit.py`

**Problem.** `_LONG_TOKEN_RE = r"\b[A-Za-z0-9_-]{20,}\b"` was redacting
legitimate schema identifiers like `COMPOUND_PHARMACY_PRESCRIPTION_HISTORY`
(35 chars) to `[token]`. The audit tab couldn't show which table the LLM was
being pointed at.

**Fix.** The regex stays broad (so we don't miss opaque tokens) but a new
`_mask_long_token()` callback preserves:
- Pure `[A-Z0-9_]+` sequences → SCREAMING_SNAKE_CASE identifiers
- Pure `[a-z0-9_]+` sequences that contain an underscore → snake_case names

Anything mixed-case or uniform alphanumeric without underscores (API keys,
hashes, base64 blobs) still gets masked.

## Fix C — Question field sanitized before storage

**File:** `core/llm_audit.py`

**Problem.** The `question` column in `llm_call_log` stored the user's raw
input. If a user ever typed PII (`"show me alice@example.com's orders"`) the
email landed unredacted in the audit table — which is itself now a secondary
data store that auditors can see.

**Fix.** `record_llm_call` passes the question through
`sanitize_llm_text(..., limit=400)` before writing. Emails, phones, GUIDs,
long numbers get redacted the same way prompt content does.

## Fix D — Retention & purge

**Files:** `store/config_store.py`, `store/__init__.py`, `main.py`

**Problem.** `llm_call_log` grew forever. A busy client could hit tens of
thousands of rows per month.

**Fix.**
- New `purge_old_llm_calls(retention_days: int)` in `store/config_store.py`.
  Deletes rows older than N days, returns row count. Safe to call repeatedly.
- Called from `main.py` on startup with default 30 days.
- Override via env var: `LLM_AUDIT_RETENTION_DAYS=60`.

## Fix E — Admin audit-tab filters

**Files:** `admin/routes.py`, `admin/templates/client_detail.html`,
`store/config_store.py`

**Problem.** The audit tab showed 100 unfiltered rows of mixed KB builds,
clarifications, SQL generations, and analysis calls. Finding a specific row
the client was asking about was painful.

**Fix.**
- `get_recent_llm_calls()` accepts optional `component=` and `status=` args.
- Admin route reads `?audit_component=...&audit_status=...` query params.
- Template renders two dropdowns (component + status) that auto-submit on
  change, plus a "Clear filters" link.
- Component dropdown enumerates all 12 known component values.

## Tests

New file: `tests/test_llm_audit_v2_fixes.py` (15 tests).

Covers: short-categorical preservation, proper-noun masking, phrase masking,
SCREAMING_SNAKE_CASE preservation, API-key masking, hash masking, question
sanitization for emails/phones, retention (normal + zero-day no-op), filter
by component, filter by status, combined filter.

Full suite: 54 tests, all passing.


---

# UI refinements — Admin LLM Audit + Executive AI indicator (April 17, 2026)

Two UI changes following the v2 audit-tab screenshots:
1. The LLM audit tab had poor column alignment, no padding, and no explanation
   of what each `component` value actually meant.
2. The chat composer mascot was a cartoon character (eyes, pupils, smiling
   mouth) — off-brand for an executive-grade enterprise product.

## Fix F — Audit tab layout and component glossary

**File:** `admin/templates/client_detail.html`

**Layout.** Table now uses explicit `<colgroup>` column widths so the
"Component", "Time", "Request", "Model", "Status", "Chars" columns stop
collapsing toward the left edge when a long preview stretches the last
column. Headers are sticky, monospace badges for component values,
preview panel is scrollable inside its row so a 6500-char prompt
doesn't explode the row height. Proper padding, zebra hover, subtle
surface shading on the filter bar.

**Component glossary.** New collapsible "What does each component mean?"
panel above the filter row explains all 9 component values
(`sql_generation`, `sql_repair`, `clarification_menu`,
`clarification_fallback`, `analysis_narrative`, `drilldown_planner`,
`kb_business_vocab`, `kb_table_doc`, `kb_query_examples`) with one-line
descriptions. Renders only when audit logging is enabled.

**Filter bar.** Now lives in its own shaded row with clear labels; the
"Clear filters" link only appears when filters are active; component
dropdown lists all 12 known values.

## Fix G — Executive AI intelligence indicator replaces cartoon mascot

**Files:** `portal/templates/portal_chat.html` (CSS, HTML, JS)

**Old.** A ~30-line SVG cartoon with eyes, irises, pupils, blinking
eyelids, and three different mouth expressions (smile / think / oh).
The JS had ~230 lines of pupil-tracking, eye-colour, mouth-state, nod
animation, "reading behaviour" for the pupils, and contextual tooltips
with phrases like "On it! 🚀" and "Hmm…". Fun, but wrong register for
executive buyers.

**New.** A minimal abstract intelligence indicator per the design brief:
no face, no emoji, light + form + motion communicate state.

**Visual anatomy** — six layered divs, all driven by CSS:
- `.ai-halo`   — always-on soft radial glow
- `.ai-ring`   — 1px outer ring, reacts to attention
- `.ai-core`   — gradient dot at the centre
- `.ai-arc-1/2` — counter-rotating arcs, shown only in thinking / processing

**States** — driven by a single class on the root element:
- `ai-idle`     — gentle breathing halo + 7s vertical drift, very low visual intensity
- `ai-focus`    — user focused the input; ring brightens, halo strengthens
- `ai-typing`   — core pulses blue, ring rotates (attentive)
- `ai-thinking` — core shifts to violet, slower contract/expand, two arcs counter-rotate (processing)
- `ai-send`     — green, stabilized, structured 1.4s rotation (focused work)
- `ai-ready`    — single 1.6s expanding pulse, then returns to idle (subtle "done")
- `ai-error`    — muted red, no shake (executive: failure should be informative, not theatrical)

**Motion guidelines observed:**
- All transitions use ease-in-out cubic-beziers, no spring overshoot
- Horizontal position pinned at `left: 10px` — no cursor chasing
- Vertical drift limited to ~3px radius
- `@media (prefers-reduced-motion: reduce)` disables all animations (WCAG)

**API preserved.** `window._composerMascotError(msg)` and
`window._composerMascotReset()` kept the same signatures, so the
existing websocket handlers in the page continue to work unchanged.

**Tooltip copy updated** from playful ("Ask me anything!", "Press ↵ to
send", "Hmm…") to executive-tone labels ("Ready", "Listening",
"Attending", "Considering", "Processing", "Response ready",
"Connection issue"). Tooltip is shown only on hover or in error state.

**Code footprint.** Old mascot: ~30 lines of SVG cartoon + ~230 lines of
pupil/mouth/blink JS. New indicator: 6 divs + ~115 lines of clean state
machine JS. Net -140 lines.


---

# Teams adapter — feature parity with portal (April 2026)

Microsoft Teams now matches the web portal's interactive chat experience. The core dispatcher was already platform-agnostic — Teams just didn't implement the optional rich-UI methods (`send_status`, `send_chart`, `send_clarification_prompt`). The four shipped changes close that gap.

## Change A — Adaptive Card submits are now parsed

**File:** `gateway/teams_adapter.py` — `parse_event()`

When a user taps an `Action.Submit` button on an Adaptive Card, Teams delivers an activity with empty `text` and a populated `value` dict. The old parser dropped those silently because it only checked `text`. The new parser reads `value.label` (or falls back to `value.option_id`) and feeds it into the same dispatch path as a typed message — so button taps and typed replies take the same route through the core clarification resolver ("exact → substring → 2-token overlap").

## Change B — `send_status()` — typing indicator

**File:** `gateway/teams_adapter.py` — new method

On every pipeline stage (`authorization`, `retrieving_context`, `generating_sql`, `validating_sql`, `executing_query`, `repairing_query`, `chart_ready`), Teams now sends a `typing` activity. Teams auto-clears the indicator when the next message arrives. Best-effort — a failure never blocks the query. Matches `web_adapter.send_status()`'s role.

## Change C — `send_clarification_prompt()` — tappable buttons

**File:** `gateway/teams_adapter.py` — new method

Renders an Adaptive Card with one `Action.Submit` per clarification option (up to 5 — a Teams UX cap, not a logical one). Each button carries the `option_id`, the label, and the pending-clarification id in its `data` payload. The core tells the user they can also type their own clarification; the parser handles both shapes uniformly.

Empty-options case falls back to a plain-text prompt *before* fetching an OAuth token, so the fallback path makes zero network calls.

## Change D — `send_chart()` — structured chart payload

**File:** `gateway/teams_adapter.py` — new method

Accepts the same `{title, chart_type, rows, x_key, y_keys}` dict that the portal uses. Delegates rendering to the existing `core.chart.generate_chart()` matplotlib pipeline — same colors, axes, formatting as the portal. The resulting PNG is uploaded via the existing `upload_file()` (Adaptive Card image). Empty rows skip the upload silently.

## Change E — plain-text clarification fallback lists the options

**File:** `main.py` — in `handle_query()` around the clarification dispatch

When the adapter has no `send_clarification_prompt` method (Zoom, Slack today — Teams now has it), the fallback plain-text message previously showed only the clarifying question. Users then had to guess what reply shape was expected. The message now also lists each option as a bullet point so the user can reply with one of them. Options cap at 5 for readability.

## Tests

New file: `tests/test_teams_parity.py` — 12 tests covering:
- Card-submit parsing (label present / fallback to option_id / malformed)
- Text-message parsing still works (regression guard)
- Mention-prefix stripping still works
- Clarification card has one Action.Submit per option with correct data
- Clarification card caps at 5 actions
- Empty-options falls back to plain text with no network call
- `send_status` sends a typing activity
- `send_status` swallows token-fetch failures (best-effort)
- `send_chart` skips upload on empty rows
- `send_chart` triggers upload on valid payload (matplotlib-dependent)

Full suite: **66 tests, all passing.**

## Not in this change

Zoom and Slack still use the plain-text clarification fallback. The gap there is identical — adding `send_clarification_prompt` on those adapters would require: Slack block-kit buttons (straightforward — `actions` block with `interactive_message` dispatch), and Zoom's Chatbot Message Card with interactive sub-elements (more involved — requires Zoom-specific "interactive-card" JSON). Flagged as future work; not blocking for the Teams request.
