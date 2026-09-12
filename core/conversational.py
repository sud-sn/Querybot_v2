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

from core.i18n import t as _t
# Accent folding, shared with core/question_normalizer.py rather than
# reimplemented: "a bientot" and "à bientôt" have to reach the same pattern,
# because accents are the first thing a hurried typist drops.
from core.question_normalizer import _fold as _fold_accents

log = logging.getLogger("querybot.conversational")

# ── Detection patterns ────────────────────────────────────────────────────────
# Full-message anchored (^...$) so embedded words never trigger: "hi" matches,
# "hi, what is revenue this month" does not (comma tail exceeds the pattern).

# Trailing "tail" tolerated after any small-talk phrase: punctuation, spaces,
# and common emoji (anything outside the basic-latin word range).
_SMALLTALK_TAIL = r"[\s!.,\U0001F300-\U0001FAFF☀-➿]*"

# French is written with apostrophes inside half its words -- "c'est",
# "qu'est-ce", "l'entreprise" -- and with a typographic ’ as often as an ASCII
# '. detect_conversational normalises the typographic one away before matching,
# so every pattern below is written with a plain '. Stripping the apostrophe
# entirely is NOT an option: the English patterns match on "what's", "that's"
# and "you're".

# The French alternatives are matched against accent-folded text (see
# detect_conversational), so they are written unaccented: "bonsoir" covers
# "bonsoir", "re-bonjour" covers "rebonjour".
_GREETING_RE = re.compile(
    r"^\s*(hi|hii+|hello|hey|heya|yo|greetings|good\s+(morning|afternoon|evening|day)|"
    r"howdy|hola|namaste|vanakkam"
    r"|bonjour|bonsoir|salut|coucou|allo+|re-?bonjour|bien\s+le\s+bonjour)"
    r"\s*(there|team|bot|querybot|everyone|all"
    r"|a\s+tous|a\s+toutes|tout\s+le\s+monde|(?:a\s+)?l'?equipe)?"
    + _SMALLTALK_TAIL + r"$",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r"^\s*(thanks?|thank\s+you|thankyou|thx|ty|tysm|great,?\s*thanks?|"
    r"perfect,?\s*thanks?|awesome,?\s*thanks?|much\s+appreciated|appreciate\s+it|"
    r"(that('s| is| was)?\s+)?(great|perfect|awesome|helpful|nice)|got\s+it|cool"
    r"|(super|parfait|genial|top|nickel|impeccable),?\s*merci|mille\s+mercis"
    r"|merci|mercii+|(c')?est\s+(parfait|super|genial|nickel|impeccable)"
    r"|(super|parfait|genial|nickel|impeccable|tres\s+bien|ca\s+marche))"
    r"(\s+(a\s+lot|so\s+much|very\s+much|again|for\s+(that|the\s+help)"
    r"|beaucoup|bien|infiniment|encore|pour\s+(ton|votre)\s+aide))?"
    + _SMALLTALK_TAIL + r"$",
    re.IGNORECASE,
)

_GOODBYE_RE = re.compile(
    r"^\s*(bye|goodbye|good\s+bye|see\s+(you|ya)( later)?|good\s+night|take\s+care|"
    r"talk\s+(to\s+you\s+)?later|ttyl|ciao|cya"
    # "salut" is both hello and goodbye in French; the greeting branch is
    # evaluated first, so it is deliberately not repeated here.
    r"|au\s+revoir|adieu|a\s+(bientot|plus|plus\s+tard|demain|la\s+prochaine)"
    r"|bonne\s+(journee|soiree|nuit|fin\s+de\s+journee)|bonne\s+continuation)"
    + _SMALLTALK_TAIL + r"$",
    re.IGNORECASE,
)

_FRUSTRATION_RE = re.compile(
    r"^\s*("
    r"(this|that|it)('s| is| was)?\s+(wrong|incorrect|useless|not\s+(right|correct|working|helpful)|bad|garbage|rubbish)"
    r"|wrong\s+answer|not\s+what\s+i\s+(asked|wanted|meant)"
    r"|(you('re| are)?\s+)?(useless|not\s+helping|no\s+help)"
    r"|(stupid|dumb|terrible|horrible)\s+(bot|answer|result)?"
    r"|this\s+(bot|thing)\s+(sucks|is\s+(broken|terrible|useless))"
    r"|(c')?est\s+(faux|incorrect|inutile|nul|n'importe\s+quoi)"
    r"|ce\s+n'est\s+pas\s+(correct|ce\s+que\s+j'ai\s+demande|bon)"
    r"|(ca|cela)\s+ne\s+(marche|fonctionne)\s+pas"
    r"|mauvaise\s+reponse|reponse\s+(fausse|incorrecte)"
    r"|(ca|cela)\s+ne\s+sert\s+a\s+rien"
    r"|tu\s+ne\s+m'aides\s+pas|vous\s+ne\s+m'aidez\s+pas"
    r")" + _SMALLTALK_TAIL + r"$",
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
    # French. The vague reply points here in so many words ("demander
    # _quelles donnees avez-vous ?_"), so this pattern is what makes that
    # pointer land rather than fall through to SQL generation.
    r"|quelles?\s+(?:donnees|tables?|schemas?|informations?)\s+"
    r"(?:avez-vous|as-tu|sont\s+(?:disponibles|accessibles)|puis-je\s+"
    r"(?:voir|interroger|utiliser))"
    r"|(?:montre|affiche|liste)(?:-| )?(?:moi\s+)?(?:les\s+)?"
    r"(?:tables?|schemas?|sources?\s+de\s+donnees)\s+disponibles"
    r"|(?:liste|affiche)\s+(?:les\s+)?(?:tables?|schemas?)$"
    r"|qu'y\s+a-t-il\s+dans\s+(?:la\s+base|les\s+donnees)"
    r"|qu(?:e|'est-ce\s+que)\s+(?:je\s+)?(?:peux|puis)(?:-je)?\s+"
    r"(?:demander|interroger|consulter)"
    r"|(?:parle|parlez)(?:-| )?moi\s+des?\s+donnees(?:\s+disponibles)?"
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
    r"|comment\s+fonctionne\s+(?:la\s+)?couche\s+semantique"
    r"|(?:explique|expliquez|decris|decrivez)(?:-| )?(?:moi\s+)?(?:la\s+)?couche\s+semantique"
    r"|(?:qu'est-ce\s+que|c'est\s+quoi)\s+(?:la\s+)?couche\s+semantique"
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
    # The business-overview reply points here ("Demandez _quelles tables sont
    # disponibles et que signifient-elles ?_"), so this is what makes that
    # pointer land. Checked before the data-inventory pattern, as in English.
    r"|quelles?\s+tables?\s+(?:sont\s+disponibles\s+)?et\s+"
    r"(?:que\s+signifient-elles|quelles?\s+(?:sont\s+)?leurs?\s+"
    r"significations?\s+metiers?)"
    r"|(?:explique|expliquez|decris|decrivez)(?:-| )?(?:moi\s+)?"
    r"les\s+tables?(?:\s+disponibles)?"
    r"|signification\s+(?:metier\s+)?des\s+tables?"
    r"|que\s+(?:represente|signifie)\s+chaque\s+table"
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
    r"|(?:explique|expliquez|decris|decrivez)(?:-| )?(?:moi\s+)?"
    r"(?:cette\s+|l'|notre\s+|le\s+)?(?:entreprise|activite|espace\s+de\s+travail)"
    r"|(?:parle|parlez)(?:-| )?moi\s+de\s+(?:l'|cette\s+|notre\s+)?"
    r"(?:entreprise|activite|espace\s+de\s+travail)"
    r"|que\s+fait\s+(?:cette|notre|l')\s*(?:entreprise|activite)"
    r"|vue\s+d'ensemble\s+de\s+(?:l'|cette\s+|notre\s+)?(?:entreprise|activite)"
    r"|(?:apercu|resume)\s+de\s+(?:l'|cette\s+|notre\s+)?(?:entreprise|activite)"
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
    r"|quelles?\s+questions?\s+(?:puis-je|peut-on|pouvons-nous|est-ce\s+que\s+"
    r"je\s+peux)\s+poser"
    r"|(?:donne|donnez|montre|montrez|propose|proposez)(?:-| )?(?:moi\s+)?"
    r"(?:des\s+|quelques\s+)?exemples?\s+de\s+questions?"
    r"|comment\s+(?:poser|formuler)\s+(?:une\s+|mes\s+|des\s+)?questions?"
    r"|questions?\s+que\s+(?:je\s+peux|l'on\s+peut)\s+poser"
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
    r"|que\s+(?:peux|pouvez)(?:-tu|-vous)?\s+(?:faire|m'aider)"
    r"|quelles?\s+sont\s+(?:tes|vos)\s+(?:capacites|fonctionnalites)"
    r"|comment\s+(?:peux|pouvez)(?:-tu|-vous)\s+m'aider"
    r"|(?:qu'est-ce\s+que|c'est\s+quoi)\s+querybot"
    r"|comment\s+fonctionne\s+querybot"
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
    r"|(quel\s+est\s+)?(ton|votre)\s+avis"
    r"|qu'en\s+(penses-tu|pensez-vous)"
    r"|(penses-tu|pensez-vous|crois-tu|croyez-vous|recommandes-tu|recommandez-vous)"
    r"|(devrions|devons)-nous\s+\w+"
    r"|est-ce\s+que\s+(l')?(entreprise|activite)\s+va\s+bien"
    r"|comment\s+(allons-nous|se\s+porte\s+l'entreprise)"
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
    r"|montre(-| )?moi\s+(les\s+)?(donnees|chiffres|stats|tout)"
    r"|(donne|envoie)(-| )?moi\s+(un\s+|le\s+)?(rapport|resume|apercu|bilan|donnees|chiffres)"
    r"|(fais|lance)\s+(un\s+|une\s+)?(rapport|analyse)"
    r"|quoi\s+de\s+neuf"
    r"|dis(-| )?moi\s+(quelque\s+chose|ce\s+que\s+tu\s+sais)"
    r"|analyse|resume|rapport|bilan|apercu"
    r")\s*[?!.\s]*$",
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
    # Matched against accent-folded text so "a bientot" and "à bientôt" reach
    # the same pattern. Folding is a no-op for the English patterns (they are
    # ASCII and already case-insensitive), so every existing tenant classifies
    # exactly the messages it classified before. The typographic apostrophe is
    # folded to ASCII by _fold itself -- it used to be replaced again here, and
    # two copies of one rule is how they drift.
    t = _fold_accents(t)
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
    r"\s*(?:;|\band\s+also\b|\bas\s+well\s+as\b|\band\s+then\b|\bplus\b"
    # A bare "and" after a comma. This is how people actually bundle two asks
    # -- "are they priced at a premium, AND how much revenue depends on them?"
    # -- and the five explicit joiners above all missed it, so the commonest
    # two-part question in the product was never detected at all.
    #
    # The comma is doing real work: without it "revenue and tax by region" is
    # one intent, and splitting it would hijack a valid query. The right-hand
    # side is additionally required to open like a question (see
    # _COMMA_AND_JOINER below), which is what keeps "sales, and by region"
    # from splitting.
    r"|,\s+and\b)\s*",
    re.IGNORECASE,
)

# The subset of joiners that only count when the right-hand side reads as a
# fresh ask. "plus" has always been in this set; a bare comma-and joins two
# clauses far more often than it joins two questions.
_COMMA_AND_JOINER = ", and"

# A right-hand side that quantifies over the LEFT half's result set is a
# continuation of one question, however much it reads like a fresh ask.
#
#   "compare 2025 against 2024 by category which grew, which shrank,
#    and what did each contribute to the overall change?"
#
# splits cleanly on ", and" and the right half opens with "what" — but "each"
# ranges over the rows the left half produces, so running it alone asks about
# nothing. One analytical question about one result set, not two questions;
# stage 2's SECTION support is what answers it whole.
#
# Deliberately NOT ordinary anaphora. "…and how much of our revenue depends on
# THEM" refers to a noun phrase named on the left ("controlled compounds"), and
# a separate query about controlled compounds is perfectly well defined — the
# pipeline's session context already carries that referent across turns. Only
# the set-quantifiers below are unanswerable in isolation.
_RESULT_SET_REFERENCE_RE = re.compile(
    r"\b(?:each|respectively|both|the\s+rest|the\s+same|the\s+above)\b",
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
    # The permissive joiners need the right-hand side to look like a question
    # in its own right, or an ordinary sentence gets torn in half.
    if joiner in {"plus", _COMMA_AND_JOINER} and not _ASK_START_RE.match(right):
        return None
    # Applies to every joiner: a half that quantifies over the left half's rows
    # cannot stand alone, and answering it in isolation produces an answer about
    # nothing.
    if _RESULT_SET_REFERENCE_RE.search(right):
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
        q = (q or "").strip().rstrip(" ?.!;:") + "?"
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
        # The workspace has no curated examples yet, so these are the
        # product's own. They go back through the pipeline when someone types
        # one, and core/question_normalizer.py canonicalises a French question
        # to English before any detector reads it -- the same path a typed
        # question takes.
        examples = [
            _t("reply.examples.revenue"),
            _t("reply.examples.top_customers"),
            _t("reply.examples.orders"),
        ]
    return "\n".join(f"  • _{q}_" for q in examples)


def _greeting_intro(portal_user: dict | None) -> str:
    """"Hello, Ada! I'm QueryBot — ...", in the reader's language.

    Built as two whole messages rather than one sentence with an optional
    ", {name}" spliced in: French puts no comma before a name in a greeting
    ("Bonjour Ada !") and does put a space before the exclamation mark, so the
    seam English can hide is one French cannot.
    """
    name = (portal_user or {}).get("name") or ""
    hello = (_t("reply.greeting.hello_named", name=name.split()[0]) if name
             else _t("reply.greeting.hello"))
    return _t("reply.greeting.intro", hello=hello)


def build_reply(kind: str, account_id: str, portal_user: dict | None = None) -> str:
    """Build the reply text for a detected conversational kind."""
    try:
        from core.workspace_guide import GUIDE_KINDS, render_workspace_guide
        if kind in GUIDE_KINDS:
            return render_workspace_guide(kind, account_id, portal_user)[0]
    except Exception as exc:
        log.warning("Workspace guide rendering failed for %s: %s", kind, exc)

    if kind == "greeting":
        return (
            f"{_greeting_intro(portal_user)}\n\n"
            f"{_t('reply.greeting.for_example')}\n"
            f"{_format_examples_block(_example_questions(account_id, portal_user=portal_user))}\n\n"
            f"{_t('reply.greeting.help_hint')}"
        )

    if kind == "thanks":
        return _t("reply.thanks")

    if kind == "goodbye":
        return _t("reply.goodbye")

    if kind == "frustration":
        return (
            f"{_t('reply.frustration.lead')}\n\n"
            f"{_t('reply.frustration.helps')}\n"
            f"  • {_t('reply.frustration.tip_explicit')}\n"
            f"  • {_t('reply.frustration.tip_thumbs_down')}\n"
            f"  • {_t('reply.frustration.tip_semantic')}\n\n"
            f"{_t('reply.frustration.retry')}"
        )

    if kind == "opinion":
        metrics = _metric_names(account_id, limit=3)
        metric_hint = (
            _t("reply.opinion.metric_hint",
               metrics=", ".join("_" + m + "_" for m in metrics))
            if metrics else ""
        )
        return (
            f"{_t('reply.opinion.lead')}\n\n"
            f"{_t('reply.opinion.offer', metric_hint=metric_hint)}\n"
            f"  • _{_t('reply.opinion.example_compare')}_\n"
            f"  • _{_t('reply.opinion.example_margin')}_"
        )

    if kind == "vague":
        return (
            f"{_t('reply.vague.lead')}\n\n"
            f"{_format_examples_block(_example_questions(account_id, portal_user=portal_user))}\n\n"
            f"{_t('reply.vague.help_hint')}"
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
        intro = (f"{_greeting_intro(portal_user)}\n\n"
                 f"{_t('reply.greeting.starters')}")
        return intro, _example_questions(account_id, portal_user=portal_user)

    if kind == "vague":
        return _t("reply.vague.lead"), _example_questions(
            account_id, portal_user=portal_user)

    # All other kinds: delegate to build_reply, no split
    return build_reply(kind, account_id, portal_user), []
