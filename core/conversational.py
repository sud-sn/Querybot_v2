"""
core/conversational.py

Deterministic front door for non-data "human" messages.

Before this module, the dispatcher had exactly one behavioral handler (the
who-are-you/capabilities regex). Everything else — "hi", "thanks", "good
morning", "this is useless", "what data do you have", "how are we doing" —
fell through every guard into the SQL pipeline, and the user got
"I couldn't find the right tables or columns to answer that" as a reply to
"thank you". This module classifies those messages with cheap regexes (no
LLM call, no latency) and builds friendly, useful replies that steer the
user toward questions the bot can actually answer.

Detection is deliberately conservative: every pattern is anchored so a real
data question containing an incidental word ("thanks to the discount, what
is revenue?") never gets swallowed. When in doubt, return None and let the
normal pipeline handle it — a wrong small-talk reply to a data question is
worse than a failed data answer to small talk.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("querybot.conversational")

# ── Detection patterns ────────────────────────────────────────────────────────
# Full-message anchored (^...$) so embedded words never trigger: "hi" matches,
# "hi, what is revenue this month" does not (comma tail exceeds the pattern).

# Trailing "tail" tolerated after any small-talk phrase: punctuation, spaces,
# and common emoji (anything outside the basic-latin word range).
_SMALLTALK_TAIL = r"[\s!.,\U0001F300-\U0001FAFF☀-➿]*"

_GREETING_RE = re.compile(
    r"^\s*(hi|hii+|hello|hey|heya|yo|greetings|good\s+(morning|afternoon|evening|day)|"
    r"howdy|hola|namaste|vanakkam)"
    r"\s*(there|team|bot|querybot|everyone|all)?" + _SMALLTALK_TAIL + r"$",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r"^\s*(thanks?|thank\s+you|thankyou|thx|ty|tysm|great,?\s*thanks?|"
    r"perfect,?\s*thanks?|awesome,?\s*thanks?|much\s+appreciated|appreciate\s+it|"
    r"(that('s| is| was)?\s+)?(great|perfect|awesome|helpful|nice)|got\s+it|cool)"
    r"(\s+(a\s+lot|so\s+much|very\s+much|again|for\s+(that|the\s+help)))?"
    + _SMALLTALK_TAIL + r"$",
    re.IGNORECASE,
)

_GOODBYE_RE = re.compile(
    r"^\s*(bye|goodbye|good\s+bye|see\s+(you|ya)( later)?|good\s+night|take\s+care|"
    r"talk\s+(to\s+you\s+)?later|ttyl|ciao|cya)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_FRUSTRATION_RE = re.compile(
    r"^\s*("
    r"(this|that|it)('s| is| was)?\s+(wrong|incorrect|useless|not\s+(right|correct|working|helpful)|bad|garbage|rubbish)"
    r"|wrong\s+answer|not\s+what\s+i\s+(asked|wanted|meant)"
    r"|(you('re| are)?\s+)?(useless|not\s+helping|no\s+help)"
    r"|(stupid|dumb|terrible|horrible)\s+(bot|answer|result)?"
    r"|this\s+(bot|thing)\s+(sucks|is\s+(broken|terrible|useless))"
    r")\s*[!.]*\s*$",
    re.IGNORECASE,
)

# "What data do you have" — meta questions about the data itself (distinct
# from _ABOUT_RE's questions about the bot). These have a real, deterministic
# answer: the schemas/tables this user is allowed to query.
_DATA_INVENTORY_RE = re.compile(
    r"\b("
    r"what\s+(data|tables?|schemas?|databases?|information)\s+(do\s+you|can\s+you|is|are)\s*(have|available|see|access|there)?"
    r"|which\s+(tables?|schemas?|data)\s+(do\s+you\s+have|are\s+(available|there)|can\s+i\s+(see|query|use|access))"
    r"|what\s+(data\s+)?(can|could)\s+i\s+(ask|query|see|access)( about)?"
    r"|show\s+me\s+(the\s+)?(available\s+)?(tables?|schemas?|data\s+sources?)"
    r"|list\s+(the\s+)?(available\s+)?(tables?|schemas?)"
    r"|what('s| is)\s+in\s+(the|my|your)\s+(database|data)"
    r"|(?:(?:explain|describe)(?:\s+me)?(?:\s+about)?|tell\s+me\s+about)\s+(?:the\s+)?(?:available\s+)?(?:data|data\s+available|data\s+sources?)"
    r")\b",
    re.IGNORECASE,
)

# Governed workspace-guide intents.  These are kept separate so a broad
# onboarding question receives a useful overview while a focused question
# (for example, "how does the semantic layer work?") receives only the
# relevant explanation.  They are deterministic metadata routes, not SQL asks.
_SEMANTIC_EXPLAINER_RE = re.compile(
    r"\b(?:"
    r"(?:(?:explain|describe)(?:\s+me)?(?:\s+about)?|what(?:'s| is)|how does)\s+(?:the\s+)?semantic\s+layer"
    r"|how\s+(?:does|do)\s+(?:the\s+)?semantic(?:\s+layer)?\s+work"
    r"|what\s+(?:are|do)\s+(?:business\s+)?(?:metrics|terms|date\s+roles)\s+(?:mean|do)"
    r")\b",
    re.IGNORECASE,
)

_TABLE_MEANINGS_RE = re.compile(
    r"\b(?:"
    r"(?:what|which)\s+tables?\s+(?:are\s+available\s+)?(?:and\s+)?what\s+do\s+they\s+mean"
    r"|(?:explain|describe)(?:\s+me)?(?:\s+about)?\s+(?:the\s+)?(?:available\s+)?tables?"
    r"|(?:business\s+)?meaning\s+of\s+(?:the\s+)?tables?"
    r"|(?:what|which)\s+(?:are\s+)?(?:the\s+)?(?:available\s+)?tables?\s+and\s+(?:their|the)\s+business\s+meanings?"
    r"|what\s+does\s+(?:each|every)\s+table\s+(?:mean|represent)"
    r")\b",
    re.IGNORECASE,
)

_BUSINESS_OVERVIEW_RE = re.compile(
    r"\b(?:"
    r"(?:explain|describe)(?:\s+me)?(?:\s+about)?\s+(?:(?:this|the|our)\s+)?(?:business|workspace)"
    r"|tell\s+me\s+about\s+(?:(?:this|the|our)\s+)?(?:business|workspace)"
    r"|what\s+(?:does|is)\s+(?:this|the|our)\s+(?:business|workspace)\s+(?:do|about|cover)"
    r"|(?:business|workspace)\s+(?:overview|summary)"
    r"|what\s+business\s+(?:areas|domains)\s+(?:are\s+)?(?:covered|available)"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_EXAMPLES_RE = re.compile(
    r"\b(?:"
    r"(?:give|show|suggest|provide)\s+(?:me\s+)?(?:some\s+)?(?:sample|example|starter)\s+questions?"
    r"|what\s+questions?\s+(?:can|should|could)\s+(?:i|we)\s+ask"
    r"|what\s+(?:types?|kinds?)\s+of\s+(?:natural[- ]language\s+|nl\s+)?questions?\s+(?:can|should|could)\s+(?:i|we)\s+ask"
    r"|what\s+(?:can|should|could)\s+(?:i|we)\s+ask"
    r"|how\s+should\s+(?:i|we)\s+(?:ask|phrase)\s+(?:a\s+)?questions?"
    r"|questions?\s+(?:i|we)\s+can\s+ask"
    r")\b",
    re.IGNORECASE,
)

_CAPABILITY_OVERVIEW_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:all\s+)?(?:can|do|could)\s+you\s+do"
    r"|how\s+(?:can|do)\s+you\s+help"
    r"|what\s+(?:are\s+)?your\s+capabilities"
    r"|(?:explain|show|list)\s+(?:your\s+)?capabilities"
    r"|what\s+is\s+querybot|how\s+does\s+querybot\s+work"
    r")\b",
    re.IGNORECASE,
)

# Opinion / judgment asks: the bot reports data; it doesn't hold views.
_OPINION_RE = re.compile(
    r"\b("
    r"(what('s| is| are)\s+your\s+(opinion|thoughts?|view|take))"
    r"|do\s+you\s+(think|feel|believe|recommend|suggest)"
    r"|should\s+(we|i)\s+\w+"
    r"|is\s+(the\s+)?business\s+(good|bad|ok|okay|healthy|doing\s+well)"
    r"|are\s+we\s+(doing\s+)?(good|well|ok|okay|badly)"
    r"|how\s+are\s+we\s+doing"
    r")\b",
    re.IGNORECASE,
)

# Vague asks with no analyzable content. Full-message anchored, tiny closed
# set — NOT a general classifier. Anything with a metric noun, a column-ish
# token, or specifics falls through to the pipeline as usual.
_VAGUE_RE = re.compile(
    r"^\s*("
    r"show\s+me\s+(the\s+)?(data|everything|numbers|stats|report)"
    r"|(give|get|send)\s+me\s+(a\s+|the\s+)?(report|data|numbers|overview|summary)"
    r"|(run|do)\s+(a\s+|an\s+)?(report|analysis)"
    r"|what('s| is)\s+(new|happening|going\s+on)"
    r"|tell\s+me\s+(something|anything|about\s+the\s+data)"
    r"|analyze\s*(this|it|the\s+data)?"
    r"|insights?"
    r"|summary"
    r"|report"
    r")\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def detect_conversational(text: str) -> str | None:
    """
    Classify a message into a behavioral kind, or None for normal routing.

    Returns one of: "greeting", "thanks", "goodbye", "frustration",
    the governed workspace-guide kinds, "opinion", or "vague".

    Order matters: greeting/thanks/goodbye/frustration are full-message
    anchored (cheap, unambiguous); data_inventory and opinion are substring
    patterns but describe unmistakably meta/judgment phrasing; vague is
    full-message anchored against a small closed set of contentless asks.
    """
    t = (text or "").strip()
    if not t or len(t) > 200:
        # Long messages are never small talk — don't even scan.
        return None
    if _GREETING_RE.match(t):
        return "greeting"
    if _THANKS_RE.match(t):
        return "thanks"
    if _GOODBYE_RE.match(t):
        return "goodbye"
    if _FRUSTRATION_RE.match(t):
        return "frustration"
    if _SEMANTIC_EXPLAINER_RE.search(t):
        return "semantic_explainer"
    if _TABLE_MEANINGS_RE.search(t):
        return "table_meanings"
    if _BUSINESS_OVERVIEW_RE.search(t):
        return "business_overview"
    if _QUESTION_EXAMPLES_RE.search(t):
        return "question_examples"
    if _CAPABILITY_OVERVIEW_RE.search(t):
        return "capability_overview"
    if _DATA_INVENTORY_RE.search(t):
        return "data_inventory"
    if _OPINION_RE.search(t):
        return "opinion"
    if _VAGUE_RE.match(t):
        return "vague"
    return None


# ── Compound-question detection ──────────────────────────────────────────────
# "revenue by region and also top 10 customers" → one SQL attempt usually
# answers half the ask or fails outright. Detect the join conservatively and
# offer to run the halves one at a time — never auto-fan-out (cost/latency
# surprise). "revenue and cost by region" is ONE intent: a bare "and" is not
# a joiner, and the right side must read like an independent ask.

_COMPOUND_JOINER_RE = re.compile(
    r"\s*(?:;|\band\s+also\b|\bas\s+well\s+as\b|\band\s+then\b|\bplus\b)\s*",
    re.IGNORECASE,
)

# The right-hand part must not be a grouping/filter continuation of the left
# ("…and also by region", "…and then for Q3" continue one intent).
_CONTINUATION_START_RE = re.compile(
    r"^(?:by|per|for|in|with|from|of|to|the|a|an)\b", re.IGNORECASE
)

# "plus" is the riskiest joiner ("revenue plus tax") — its right side must
# start like a standalone command/question before we call it compound.
_ASK_START_RE = re.compile(
    r"^(?:show|list|give|get|display|find|fetch|what|which|who|when|how"
    r"|top|bottom|count|compare|rank|break|total|average|sum)\b",
    re.IGNORECASE,
)


def detect_compound_question(text: str) -> tuple[str, str] | None:
    """
    Return (first_question, second_question) when `text` clearly bundles two
    independent asks, else None. Conservative by design: false negatives are
    cheap (the pipeline still tries), false positives hijack a valid query.
    """
    t = (text or "").strip()
    if not t or len(t) > 400:
        return None
    m = _COMPOUND_JOINER_RE.search(t)
    if not m:
        return None
    left = t[: m.start()].strip(" ,.?!")
    right = t[m.end():].strip(" ,.?!")
    if len(left.split()) < 3 or len(right.split()) < 3:
        return None
    if _CONTINUATION_START_RE.match(right):
        return None
    joiner = m.group(0).strip().lower()
    if joiner == "plus" and not _ASK_START_RE.match(right):
        return None
    # A second joiner inside the right half means 3+ asks — still offer the
    # first split; the remainder stays bundled and can be split again next turn.
    return left, right


# ── Reply builders ────────────────────────────────────────────────────────────

def _example_questions(
    account_id: str,
    limit: int = 3,
    portal_user: dict | None = None,
) -> list[str]:
    """Real questions this workspace can answer, best sources first:
    metric example_questions (admin-curated), then recent successful
    query_log questions. Best-effort — an empty list is fine."""
    # The workspace guide uses validated examples and applies the signed-in
    # user's table ACL.  Prefer it whenever identity is available; the legacy
    # account-wide lookup below is retained only for non-user system greetings.
    if portal_user:
        try:
            from core.workspace_guide import build_workspace_guide
            return build_workspace_guide(account_id, portal_user).get("examples", [])[:limit]
        except Exception as exc:
            log.debug("conversational: governed examples unavailable: %s", exc)

    examples: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = (q or "").strip().rstrip("?") + "?"
        key = q.lower()
        if len(q) > 5 and key not in seen:
            seen.add(key)
            examples.append(q)

    try:
        import store
        for metric in store.list_metrics(account_id):
            if not metric.get("is_active", 1):
                continue
            for q in str(metric.get("example_questions") or "").split("\n"):
                for part in q.split(";"):
                    if part.strip():
                        _add(part)
                if len(examples) >= limit:
                    return examples[:limit]
    except Exception as exc:
        log.debug("conversational: metric examples unavailable: %s", exc)

    try:
        from store.db import get_db
        with get_db() as conn:
            rows = conn.execute(
                "SELECT DISTINCT question FROM query_log "
                "WHERE account_id=? AND success=1 AND question IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 20",
                (account_id,),
            ).fetchall()
        for r in rows:
            q = (r["question"] or "").strip()
            # Skip long/odd historical entries — examples should look inviting.
            if 10 <= len(q) <= 90:
                _add(q)
            if len(examples) >= limit:
                break
    except Exception as exc:
        log.debug("conversational: query-log examples unavailable: %s", exc)

    return examples[:limit]


def _metric_names(account_id: str, limit: int = 5) -> list[str]:
    try:
        import store
        names = []
        for metric in store.list_metrics(account_id):
            if metric.get("is_active", 1) and metric.get("name"):
                names.append(str(metric["name"]))
            if len(names) >= limit:
                break
        return names
    except Exception:
        return []


def _format_examples_block(examples: list[str]) -> str:
    if not examples:
        return (
            "  • _What is our total revenue this month?_\n"
            "  • _Show top 10 customers by sales_\n"
            "  • _How many orders were created last week?_"
        )
    return "\n".join(f"  • _{q}_" for q in examples)


def build_reply(kind: str, account_id: str, portal_user: dict | None = None) -> str:
    """Build the reply text for a detected conversational kind."""
    try:
        from core.workspace_guide import GUIDE_KINDS, render_workspace_guide
        if kind in GUIDE_KINDS:
            return render_workspace_guide(kind, account_id, portal_user)[0]
    except Exception as exc:
        log.warning("Workspace guide rendering failed for %s: %s", kind, exc)

    if kind == "greeting":
        name = (portal_user or {}).get("name") or ""
        hello = f"Hello{', ' + name.split()[0] if name else ''}! 👋"
        return (
            f"{hello} I'm QueryBot — ask me anything about your business data.\n\n"
            "For example:\n"
            f"{_format_examples_block(_example_questions(account_id, portal_user=portal_user))}\n\n"
            "Type `help` for commands, or just ask in plain English."
        )

    if kind == "thanks":
        return "You're welcome! Ask me another question whenever you're ready."

    if kind == "goodbye":
        return "Goodbye! I'll be here whenever you need your data. 👋"

    if kind == "frustration":
        return (
            "Sorry about that — let's get it right.\n\n"
            "A couple of things that help:\n"
            "  • Name the metric and breakdown explicitly (e.g. _total revenue by customer_)\n"
            "  • Use the 👎 button on the wrong answer — your feedback goes to your "
            "administrator, who can correct the field mapping behind it\n"
            "  • If a term keeps being misunderstood, ask your admin to define it "
            "in the Semantic Layer or Metric Registry\n\n"
            "Want to try rephrasing your question?"
        )

    if kind == "opinion":
        metrics = _metric_names(account_id, limit=3)
        metric_hint = (
            f" — for example {', '.join('_' + m + '_' for m in metrics)}" if metrics else ""
        )
        return (
            "I report data — the judgment calls are yours. 🙂\n\n"
            "I can show you the numbers behind that question though. "
            f"Ask for a specific metric{metric_hint}, with a time range, like:\n"
            "  • _How did revenue this quarter compare to last quarter?_\n"
            "  • _Show gross margin by month this year_"
        )

    if kind == "vague":
        return (
            "Happy to help — I just need to know what to measure. "
            "Name a metric and (optionally) a breakdown or time range:\n\n"
            f"{_format_examples_block(_example_questions(account_id, portal_user=portal_user))}\n\n"
            "You can also type `help` for commands, or ask "
            "_what data do you have?_ to see what's available."
        )

    return ""


def build_reply_split(
    kind: str,
    account_id: str,
    portal_user: dict | None = None,
) -> tuple[str, list[str]]:
    """
    Sibling to build_reply() for adapters that can render questions as buttons.

    Returns (intro_text, questions_list) for the three kinds that embed example
    questions (greeting, data_inventory, vague). For all other kinds, returns
    (build_reply(...), []) so callers can use this uniformly without branching.

    The questions_list contains raw question strings without markdown formatting
    so the adapter can label buttons directly.
    """
    try:
        from core.workspace_guide import GUIDE_KINDS, render_workspace_guide
        if kind in GUIDE_KINDS:
            return render_workspace_guide(
                kind,
                account_id,
                portal_user,
                include_examples=False,
            )
    except Exception as exc:
        log.warning("Workspace guide split rendering failed for %s: %s", kind, exc)

    if kind == "greeting":
        name = (portal_user or {}).get("name") or ""
        hello = f"Hello{', ' + name.split()[0] if name else ''}! 👋"
        intro = (
            f"{hello} I'm QueryBot — ask me anything about your business data.\n\n"
            "Here are some questions to get you started:"
        )
        return intro, _example_questions(account_id, portal_user=portal_user)

    if kind == "vague":
        intro = (
            "Happy to help — I just need to know what to measure. "
            "Name a metric and (optionally) a breakdown or time range:"
        )
        return intro, _example_questions(account_id, portal_user=portal_user)

    # All other kinds: delegate to build_reply, no split
    return build_reply(kind, account_id, portal_user), []
