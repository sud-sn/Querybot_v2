"""
core/masking.py
───────────────
Local, zero-network field-level data masking for KB generation.

Entry points
------------
detect_sensitive_columns(columns)  → {col_name: strategy}
    Regex scan of column names + type-based free-text detection.
    Used by the admin UI to pre-check likely-PII fields when the admin
    enables masking on a table.

scan_values_for_pii(rows, col_defs)  → {col_name: {pii_type, strategy, confidence}}
    Scan actual sample values for value-level PII patterns (email, SSN, PAN,
    Aadhaar, credit card, IP, phone).  High-confidence hits (>60%) are merged
    into masked_fields in auto mode.

mask_rows(rows, masked_fields, columns, seed_key="") → list[dict]
    Apply per-column masking strategies to a sample of real rows.
    When seed_key is non-empty, each value is masked deterministically:
    the same real value always maps to the same fake value across all
    tables — preserving FK consistency without storing a lookup table.

Performance
-----------
Faker is imported once and reused.  Falls back to stdlib-only masking
if Faker is not installed.  Deterministic path bypasses Faker entirely
and uses a seeded random.Random instance per cell.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac_mod
import logging
import random
import re
import os
import string
import threading
from functools import lru_cache
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── PII detection patterns ────────────────────────────────────────────────────
# Ordered from most-specific to least-specific.
# Each tuple: (regex applied to lower-cased column name, masking strategy)
_PII_PATTERNS: list[tuple[str, str]] = [
    # Must-redact (secrets / credentials)
    (r"password|passwd|pwd|api[_\s]?key|secret[_\s]?key|private[_\s]?key|"
     r"\btoken\b|hash\b|salt\b|\bpin\b", "redact"),
    # Government IDs
    (r"ssn|social[_\s]?sec|national[_\s]?id|nid\b|sin\b|tax[_\s]?id|"
     r"fiscal[_\s]?id|passport", "ssn"),
    # Financial card numbers
    (r"credit[_\s]?card|card[_\s]?no|card[_\s]?num|pan\b", "credit_card"),
    # Payment aliases (PCI) — tokens, bank account / routing identifiers
    (r"payment[_\s]?token|payment[_\s]?method[_\s]?id|"
     r"account[_\s]?number|acct[_\s]?no|iban|"
     r"routing[_\s]?(?:number|num|no)", "credit_card"),
    # Healthcare / provider identifiers (PHI) — treat as government-ID class
    (r"\bmrn\b|medical[_\s]?record[_\s]?(?:number|num|no|id|#)?", "ssn"),
    (r"\bnpi\b|national[_\s]?provider[_\s]?id|provider[_\s]?(?:number|num|id)", "ssn"),
    (r"\bdea\b|dea[_\s]?(?:number|num|id)", "ssn"),
    (r"(?:pharmacy|prescriber|provider|state)[_\s]?licen[sc]e|"
     r"licen[sc]e[_\s]?(?:number|num|no)", "ssn"),
    (r"policy[_\s]?(?:number|num|no|id)|member[_\s]?id|"
     r"subscriber[_\s]?id|group[_\s]?(?:number|num|id)", "ssn"),
    # Contact
    (r"email|e[_\s]mail", "email"),
    (r"phone|mobile|cell\b|tel\b|fax", "phone"),
    # Drug / medication names — MUST precede the bare 'name' pattern below,
    # which otherwise matches the _NAME suffix and substitutes a fake PERSON
    # name for a medication column, misleading KB generation into treating
    # the field as a person identifier.
    (r"(?:drug|medication|med|compound|ingredient|rx|generic)[_\s]?name", "drug_name"),
    # Name — order matters: first/last before bare 'name'
    (r"first[_\s]?name|fname|given[_\s]?name|forename", "first_name"),
    (r"last[_\s]?name|lname|surname|family[_\s]?name", "last_name"),
    (r"(?<![a-z])name(?![a-z])|full[_\s]?name|display[_\s]?name", "name"),
    # Doctor / physician — a person identifier same as bare 'name', but
    # commonly stored without a _NAME suffix in ERP schemas (DOCTOR,
    # DOCTOR_NM, PHYSICIAN) so it doesn't match the pattern above.
    (r"(?<![a-z])doctor(?![a-z])|(?<![a-z])physician(?![a-z])", "name"),
    # Address components
    (r"address|street|addr\b", "address"),
    (r"(?<![a-z])city(?![a-z])", "city"),
    (r"zip|postal|post[_\s]?code", "zip"),
    (r"(?<![a-z])state(?![a-z])", "state"),
    (r"(?<![a-z])country(?![a-z])", "country"),
    # Date of birth (shift more aggressively than normal dates)
    (r"birth[_\s]?date|date[_\s]?of[_\s]?birth|dob\b|born\b", "birthdate"),
    # Compensation / financial (shift ±15 %)
    (r"salary|wage\b|income|compensation|base[_\s]?pay|annual[_\s]?pay", "salary"),
    # IP / geo
    (r"ip[_\s]?address|ip[_\s]?addr\b", "ip_address"),
    (r"latitude|longitude|lat\b|lng\b|lon\b", "coordinate"),
    # Free-text / narrative columns — may contain anything; always redact.
    # Matched AFTER the more specific patterns above so that e.g. "address_line"
    # hits the address strategy, not the free_text strategy.
    (r"\bnotes?\b|\bcomments?\b|\bremarks?\b|description|narrative|"
     r"\bmemo\b|\bmessage\b|address_line|reason_text|feedback|"
     r"\binstructions?\b|\bsummary\b|\btranscript\b|\bdetails?\b|"
     r"\bbody\b|text_field|\bcontent\b|\bfreetext\b|free[_\s]?text|"
     r"clinical[_\s]?notes?|chief[_\s]?complaint|diagnosis[_\s]?text|"
     r"\bobservations?\b|\bassessment\b|note[_\s]?text|\breason\b",
     "free_text"),
]

# ── Type-based free-text detection ────────────────────────────────────────────
_FREE_TEXT_BASE_TYPES = {
    "text", "ntext", "longtext", "mediumtext", "tinytext",
    "clob", "nclob", "long", "blob",
}
_LONG_VARCHAR_THRESHOLD = 500

_SAFE_LONG_TEXT_NAMES = {
    "status", "type", "flag", "gender", "sex", "category", "class",
    "department", "dept", "division", "group", "team", "role", "level",
    "grade", "rank", "priority", "severity", "state", "phase", "stage",
    "mode", "method", "code", "indicator", "active", "enabled",
    "country", "region", "zone", "area", "branch", "location", "title",
    "currency", "unit", "format", "language", "locale", "timezone",
}


def _is_free_text_type(col_type: str) -> bool:
    """
    Return True if the column type indicates an unbounded or very long text
    field likely to contain unstructured, potentially sensitive content.
    """
    ct = (col_type or "").lower().strip()
    base = ct.split("(")[0].strip()
    if base in _FREE_TEXT_BASE_TYPES:
        return True
    if "varchar" in base or "character varying" in base:
        m = re.search(r"\((-?\d+)\)", ct)
        if m:
            length = int(m.group(1))
            if length < 0 or length > _LONG_VARCHAR_THRESHOLD:
                return True
    return False


# ── Value-level PII regex patterns ────────────────────────────────────────────
# Used by scan_values_for_pii() to detect PII in actual sample data.

_RE_EMAIL   = re.compile(r"^[\w.+\-]+@[\w\-]+\.[\w.]{2,}$", re.IGNORECASE)
_RE_SSN     = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_RE_PAN     = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")            # Indian PAN card
_RE_AADHAAR = re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$") # Indian Aadhaar
_RE_IP      = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

# Phone — must have 7-15 digits; allow +, spaces, hyphens, parens, dots
_RE_PHONE_FMT = re.compile(r"^\+?[\d\s\-\(\)\.]{7,20}$")


def _is_phone_like(v: str) -> bool:
    digit_count = sum(c.isdigit() for c in v)
    return 7 <= digit_count <= 15 and bool(_RE_PHONE_FMT.match(v))


# ── Embedded-PII scrubbing (defense-in-depth for free-text values) ─────────────
# Non-anchored (substring) patterns used to strip PII that is *embedded inside*
# a larger free-text value (e.g. a clinical note that mentions a patient's name,
# email, phone, SSN or date of birth). Applied as a final safety net to string
# columns that were NOT otherwise masked, so narrative content that escapes
# name/type heuristics cannot carry raw identifiers to the LLM.
_RE_EMAIL_SUB = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.]{2,}")
_RE_SSN_SUB   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Phone: require separators so we don't clobber plain numeric IDs.
_RE_PHONE_SUB = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s\-.])\d{3}[\s\-.]\d{4}(?!\d)"
)
# ISO-style date (often a DOB when embedded in clinical narrative).
_RE_DATE_SUB  = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def scrub_embedded_pii(text: str) -> str:
    """
    Redact PII tokens embedded inside a free-text string.

    Replaces emails, SSNs, phone numbers and ISO dates with bracketed
    placeholders. Order matters: email before phone (an email's domain can
    contain digits), SSN before generic date.
    """
    if not text or not isinstance(text, str):
        return text
    t = _RE_EMAIL_SUB.sub("[EMAIL]", text)
    t = _RE_SSN_SUB.sub("[SSN]", t)
    t = _RE_PHONE_SUB.sub("[PHONE]", t)
    t = _RE_DATE_SUB.sub("[DATE]", t)
    return t


# ── NER-backed person-name scrubbing (Presidio) ──────────────────────────────
# Person names have no regex-matchable shape — "Patient John Smith reported
# dizziness" sails through scrub_embedded_pii with the name intact. Presidio
# (Microsoft's local NER-based PII engine) closes exactly that gap. Scoped
# deliberately tight:
#   • PERSON entities only — LOCATION/ORG hit too many legitimate business
#     dimension values (warehouse cities, company names) to scrub safely.
#   • The question of a workspace under compliance (core2/service.py::question_scrubber: after the name rules, and
#     given the workspace's own words so its members are kept), and the free text of a regulated industry's
#     KB-discovery samples (scrub_unmasked_free_text) — never the rows of an answer.
#   • Mixed-case text only — ERP dimension values are typically
#     ALL CAPS ("MARTIN SUPPLY CO"), where NER person-detection is both
#     unreliable and business-destructive.
#   • Fully local (spaCy model in-process). The install brings it with the medium English model
#     (deploy/linux/update.sh, deploy/windows/update-querybot.ps1); without them the name rules work alone, and the
#     compliance page and proof pack say so (name_recogniser).

_presidio_analyzer: Any = None  # None = not tried yet, False = unavailable
_presidio_model = ""            # the spaCy model it reads with, once loaded
_presidio_lock = threading.Lock()
_presidio_loading = False       # a worker is loading it for a caller on the event loop
_loading_lock = threading.Lock()
# The medium model first: on questions it found the names the large one found, took fewer business words for people,
# and holds half the memory (a process of about 375 MB against 760 MB). The small one has no word vectors: a last resort.
_NER_MODELS = ("en_core_web_md", "en_core_web_lg", "en_core_web_sm")


def _ner_model(installed: Callable[[str], bool]) -> str | None:
    """The spaCy model to read names with: the one QB_NER_MODEL names (None when that one is not installed: another
    is never put in its place), else the first of _NER_MODELS installed."""
    named = os.environ.get("QB_NER_MODEL", "").strip()
    if named:
        return named if installed(named) else None
    return next((m for m in _NER_MODELS if installed(m)), None)


def ner_model_wanted() -> str:
    """The model an install should bring: the one QB_NER_MODEL names, else the medium English one."""
    return os.environ.get("QB_NER_MODEL", "").strip() or _NER_MODELS[0]


def _load_presidio() -> tuple[Any, str]:
    """Presidio's engine and the spaCy model it reads with; raises when either is missing. Seconds of work: spaCy,
    Presidio and the model are read from disk."""
    import importlib.util
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    model = _ner_model(lambda name: importlib.util.find_spec(name) is not None)
    if model is None:
        named = os.environ.get("QB_NER_MODEL", "").strip()
        raise ImportError(f"the spaCy model {named} that QB_NER_MODEL names is not installed" if named
                          else "no spaCy English model installed")
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine()), model


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _load_in_background() -> None:
    """Load the model in a worker, once: the event loop is never held for the seconds it takes."""
    global _presidio_loading
    with _loading_lock:
        if _presidio_loading:
            return
        _presidio_loading = True

    def load() -> None:
        global _presidio_loading
        try:
            _get_presidio()
        finally:
            _presidio_loading = False

    log.info("masking: the name recogniser is loading in the background; the name rules work alone until it is ready")
    threading.Thread(target=load, name="name-recogniser-load", daemon=True).start()


def _get_presidio():
    """Lazy Presidio AnalyzerEngine singleton; None when unavailable. On the event loop, before it is loaded: None
    while a worker loads it (the name rules work alone for that question). Elsewhere a caller waits for it."""
    global _presidio_analyzer, _presidio_model
    if _presidio_analyzer is not None:
        return _presidio_analyzer or None
    if _on_event_loop():
        _load_in_background()
        return None
    with _presidio_lock:                 # loaded once, though the first questions arrive together
        if _presidio_analyzer is not None:
            return _presidio_analyzer or None
        try:
            analyzer, model = _load_presidio()
            _presidio_model = model
            _presidio_analyzer = analyzer
            log.info("masking: Presidio NER person-name scrubbing enabled (spaCy model=%s)", model)
            return _presidio_analyzer
        except Exception as exc:
            log.warning("masking: the name recogniser is unavailable (%s); names typed in questions are withheld "
                        "by the name rules alone", exc)
            _presidio_analyzer = False
            return None


def name_recogniser() -> str:
    """The spaCy model a question under compliance is also read with for people's names, or "" when the name rules
    withhold them alone (Presidio or its model is not installed). Loads the model the first time."""
    if _get_presidio() is None:
        return ""
    return _presidio_model or "spaCy"


def scrub_person_names_ner(text: str, analyzer=None, known: set[str] | frozenset[str] = frozenset(),
                           phrases: set[str] | frozenset[str] = frozenset()) -> str:
    """
    Replace person-name spans in narrative text with [PERSON].

    Given the workspace's words (``known`` and ``phrases``, as scrub_typed_names takes them), a span is kept that
    lies inside a member's or a place's whole name in the text ("Johnson" of "Johnson & Johnson", a store called
    Clinton, "Texas"), or whose every word is the workspace's own or a common word of a question (a month, "Fills"),
    unless a title stands before it ("Mrs. Price"). One word of a member does not make a longer span safe: "Mary
    Franklin" is withheld though a store is called Franklin. Without them every span is withheld, as for a discovery
    sample's free text. A placeholder a scrub wrote ("[PERSON]", "[EMAIL]") is never read as a name.

    Best-effort: returns text unchanged when Presidio is unavailable, the
    value doesn't look like narrative prose (no lowercase letters), or
    analysis fails. Never raises.
    """
    if not text or not isinstance(text, str):
        return text
    return _replace_names(text, _recognised_name_spans(text, analyzer, known, phrases))


def scrub_names(text: str, known: set[str] | frozenset[str] = frozenset(),
                phrases: set[str] | frozenset[str] = frozenset(), analyzer=None) -> str:
    """
    A person's name a reader typed, withheld by the name rules (scrub_typed_names) and the name recogniser
    (scrub_person_names_ner) together: both read the question as typed, and what either takes for a name becomes
    [PERSON]. Neither reads the other's output: the recogniser taking "John" alone would leave the rules no "Smith,
    John" to see, and the rules withholding "Mary Ann" would leave the recogniser no "Mary Ann Price".
    """
    if not text or not isinstance(text, str):
        return text
    return _replace_names(text, _typed_name_spans(text, known, phrases)
                          + _recognised_name_spans(text, analyzer, known, phrases))


def _recognised_name_spans(text: str, analyzer, known: set[str] | frozenset[str],
                           phrases: set[str] | frozenset[str]) -> list[tuple[int, int]]:
    """What the name recogniser takes for a person's name, less what the workspace's words keep (see
    scrub_person_names_ner); none where it is not installed or the text has no lower case."""
    # ALL-CAPS values are ERP codes/dimension labels, not prose — NER is
    # unreliable there and a false positive corrupts a legitimate sample.
    if not any(c.islower() for c in text):
        return []
    analyzer = analyzer if analyzer is not None else _get_presidio()
    if analyzer is None:
        return []
    try:
        results = analyzer.analyze(
            text=text, entities=["PERSON"], language="en", score_threshold=0.4,
        )
        spare = bool(known or phrases)
        named = _named_spans(text, phrases) if spare else []
        spans = []
        for r in results:
            for start, end in _outside_placeholders(text, r.start, r.end):
                if spare and (any(a <= start and end <= b for a, b in named)
                              or (not _TITLE_BEFORE.search(text, 0, start)
                                  and all(_own_word(w, known) for w in _WORD.findall(text[start:end])))):
                    continue
                spans.append((start, end))
        return spans
    except Exception as exc:
        log.debug("masking: NER scrub skipped (%s)", exc)
        return []


def _replace_names(text: str, spans: list[tuple[int, int]]) -> str:
    """Each span (merged where they overlap) becomes [PERSON], a possessive kept ("[PERSON]'s claim"); from the end,
    so earlier offsets stay valid."""
    for start, end in reversed(_merged(spans)):
        tail = re.search(r"['’]s$", text[start:end])
        text = text[:start] + "[PERSON]" + (tail.group(0) if tail else "") + text[end:]
    return text


# The placeholders the scrubs write. Nothing else in square brackets is one: "[MARIA]" is a name a reader typed.
_PLACEHOLDER = re.compile(r"\[(?:PERSON|EMAIL|PHONE|SSN|DATE|MASKED|REDACTED)\]")


def _outside_placeholders(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """The parts of text[start:end] outside the placeholders a scrub put there, each from its first word to its
    last; none when it has no word."""
    parts, at = [], start
    for m in _PLACEHOLDER.finditer(text):
        if m.end() > start and m.start() < end:
            parts.append((at, max(at, m.start())))
            at = max(at, m.end())
    parts.append((at, end))
    out = []
    for a, b in parts:
        found = list(_WORD.finditer(text, a, b))
        if found:
            out.append((found[0].start(), found[-1].end()))
    return out


def _merged(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if out and start < out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


def scrub_question_pii(text: str, industry: str = "", names: bool = True) -> tuple[str, bool]:
    """
    Scrub PII the USER typed into their question before it reaches any LLM
    prompt (SQL generation, off-topic classifier, retries).

    KB samples and result rows are already protected elsewhere — this closes
    the remaining front-door channel: "why was John Smith's claim denied?"
    carries PHI in the question text itself, regardless of how well the
    schema-side masking works.

    Deliberately narrower than scrub_embedded_pii:
      • NO date scrubbing — an ISO date in an analytics question is almost
        always a period filter ("sales from 2024-01-01"), not a DOB.
        Replacing it would break the core use case.
      • NER person-name pass only for regulated industries (same gate as
        scrub_unmasked_free_text). First call pays the spaCy model load;
        subsequent calls are milliseconds on question-length text. Falls
        back to regex-only when Presidio isn't installed.

    ``names=False`` leaves the name recogniser to the caller: the new core runs it after its name rules, which
    would no longer see "Smith, John" once the recogniser had taken "John".

    Returns (scrubbed_text, changed). Idempotent — placeholders like
    [EMAIL] contain nothing the patterns re-match, so scrubbing at both the
    dispatcher and pipeline choke points is safe.
    """
    if not text or not isinstance(text, str):
        return text, False
    t = _RE_EMAIL_SUB.sub("[EMAIL]", text)
    t = _RE_SSN_SUB.sub("[SSN]", t)
    t = _RE_PHONE_SUB.sub("[PHONE]", t)
    if names and industry in ("banking", "healthcare_pharmacy"):
        t = scrub_person_names_ner(t)
    return t, t != text


# A person's name the reader typed, found without the optional NER library: a run of capitalised words that are not
# the workspace's own words. Every word below may open or sit inside a question in capitals without naming anyone.
_NOT_A_NAME = frozenset("""
a an and any are as at be by can could did do does each every for from give had has have how i if in is it its
last list me more most my next no not of on or our per please previous show so tell than that the their them these
this those to top bottom total us vs versus was we were what when where which who whom whose why will with would you
your compare break down give find get let make plot chart graph table see versus between since until during
year years quarter quarters month months week weeks day days today yesterday tomorrow ytd mtd qtd fy q1 q2 q3 q4
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
monday tuesday wednesday thursday friday saturday sunday
north south east west northeast northwest southeast southwest central
""".split())
# Words of a business question that are never a person's name, whatever the workspace calls its fields: a phrase
# made of these ("Customer Lifetime Value", "Net Sales Growth Rate") is the question's, not a person.
_COMMON = frozenset("""
average avg mean median growth rate rates ratio value values size basket lifetime trend trends count counts number
numbers amount amounts revenue revenues sales sale cost costs margin margins profit profits share shares percent
percentage change changes total totals sum gross net new old active inactive current prior first second third best
worst highest lowest high low daily weekly monthly quarterly yearly annual overall summary report reports dashboard
breakdown forecast target targets budget actual actuals variance units unit quantity qty price prices order orders
customer customers product products item items store stores region regions category categories segment segments
channel channels type types status group groups level levels performance kpi kpis metric metrics rank ranking mix
volume volumes hours minutes inventory stock fill fills prescription prescriptions patient patients claim claims
account accounts invoice invoices supplier suppliers vendor vendors brand brands department departments branch
branches location locations site sites warehouse warehouses city cities state states country countries county
""".split())
# Places a question names in capitals that are no one's name (more than one word, or a state beside another).
_PLACES = frozenset(p.strip().casefold() for p in """
alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|
indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|
nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|
pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|
west virginia|wisconsin|wyoming|district of columbia|puerto rico|los angeles|san francisco|san diego|san jose|
san antonio|las vegas|new orleans|salt lake city|st louis|saint louis|kansas city|oklahoma city|fort worth|el paso|
new york city|united states|united kingdom|new zealand|south africa|saudi arabia|hong kong|british columbia|
nova scotia|new brunswick|prince edward island|newfoundland and labrador|ontario|quebec|alberta|manitoba|saskatchewan
""".replace("\n", "").split("|"))
_TITLES = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "madam"})
# A title just before a span: whatever follows it is a person's name, a word of the workspace or not ("Mrs. Price").
_TITLE_BEFORE = re.compile(r"(?<!\w)(?:" + "|".join(sorted(_TITLES)) + r")\.?\s+$", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_](?:[^\W_]|['’\-])*")


def _bare(word: str) -> str:
    return re.sub(r"['’]s$", "", word)


def _lowered(word: str) -> str:
    return _bare(word).casefold()


def _own_word(word: str, known: set[str] | frozenset[str]) -> bool:
    """A word of the question rather than of a name: the workspace's own (``known``), a common word of business
    questions, a month, a day, a question word or a title."""
    w = _lowered(word)
    forms = {w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w}
    return any(f in _NOT_A_NAME or f in _COMMON or f in known for f in forms) or w in _TITLES


def _phrase_key(text: str) -> str:
    """A name as the scrubbers compare it: its words, in lower case, without a possessive or punctuation
    ("Dr. Reddy's" -> "dr reddy", "Johnson & Johnson" -> "johnson johnson")."""
    return " ".join(_lowered(w) for w in _WORD.findall(text))


@lru_cache(maxsize=32)
def _kept_phrase_keys(phrases: frozenset[str]) -> frozenset[str]:
    return frozenset(k for k in map(_phrase_key, phrases) if k)


def _phrase_keys(phrases: set[str] | frozenset[str]) -> frozenset[str]:
    """The members' names as compared, kept per set: a workspace's set is the same object question after question."""
    if isinstance(phrases, frozenset):
        return _kept_phrase_keys(phrases)
    return frozenset(k for k in map(_phrase_key, phrases) if k)


def _named_spans(text: str, phrases: set[str] | frozenset[str]) -> list[tuple[int, int]]:
    """Where a member's or a place's whole name stands in the text: (start, end) of each run of words that is one,
    the longest at each word."""
    keys = _phrase_keys(phrases)
    words = list(_WORD.finditer(text))
    said = [_lowered(m.group(0)) for m in words]
    out = []
    for i in range(len(words)):
        for n in range(min(6, len(words) - i), 0, -1):
            key = " ".join(said[i:i + n])
            if key in keys or key in _PLACES:
                out.append((words[i].start(), words[i + n - 1].end()))
                break
    return out


def scrub_typed_names(text: str, known: set[str] | frozenset[str] = frozenset(),
                      phrases: set[str] | frozenset[str] = frozenset()) -> str:
    """
    Replace the names of people a reader typed with [PERSON], without an NER model.

    A name is a run of two or more capitalised words ("John Smith", "Mary-Jane O'Neil's", "JOHN SMITH", "John Q.
    Smith", "J. Smith"), a title and a name ("Dr Patel"), or "Smith, John". The run's own words of the question are
    set aside first: a word the workspace knows (``known``: what its fields, metrics and tables are called, in lower
    case, singular or plural), a common word of business questions, a month, a day or a question word. What is left
    is a name unless it is a member the workspace may list (``phrases``: the member's whole name, compared by its
    words in lower case without a possessive or punctuation) or a place ("New York"); a title and a name are kept only
    when together they are a member's whole name ("Dr. Reddy's"). One word of a member does not make a name safe:
    "Mary Franklin" or "Dr Franklin" is withheld though a store is called Franklin. A word with a digit is no name's.

    Best effort, as NER is. It misses a name typed in lower case, or one surname alone ("Smith's refills"): the
    name recogniser (scrub_person_names_ner), run after these rules, catches some of those. A phrase it wrongly withholds is a word the planner
    could not have filtered on anyway (only a quoted member, matched before scrubbing, is), but it is lost from the
    question as kept in history.
    """
    if not text or not isinstance(text, str):
        return text
    return _replace_names(text, _typed_name_spans(text, known, phrases))


def _typed_name_spans(text: str, known: set[str] | frozenset[str],
                      phrases: set[str] | frozenset[str]) -> list[tuple[int, int]]:
    """Where scrub_typed_names finds a person's name: (start, end) of each."""
    words = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(text)]
    keys = _phrase_keys(phrases)
    bare, lowered = _bare, _lowered

    def initial(word: str) -> bool:
        return len(word) == 1 and word.isupper()

    def capital(word: str) -> bool:
        base = bare(word)
        if any(c.isdigit() for c in base) or len(base) < 2 or not base[0].isupper():
            return False
        if base.isupper():        # a name in capitals; a code has no vowel or fewer than three letters (NDC, OH)
            return len(base) >= 3 and any(c in "AEIOUY" for c in base.upper())
        return any(c.islower() for c in base[1:])

    def own_word(word: str) -> bool:
        return _own_word(word, known)

    def joined(i: int, j: int) -> bool:
        """Words i and j (j = i + 1) are one run: only spaces between them, or an initial's full stop."""
        gap = text[words[i][1]:words[j][0]]
        return gap.strip() == "" or (initial(words[i][2]) and gap.strip() == ".")

    def safe(run: list[int]) -> bool:
        """Every word of the run is the question's own, or part of a member's or a place's whole name."""
        said = [lowered(words[k][2]) for k in run]
        at = 0
        while at < len(said):
            longest = next((n for n in range(min(5, len(said) - at), 0, -1)
                            if " ".join(said[at:at + n]) in keys or " ".join(said[at:at + n]) in _PLACES), 0)
            if longest:
                at += longest
            elif own_word(words[run[at]][2]):
                at += 1
            else:
                return False
        return True

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(words):
        word = words[i][2]
        titled = lowered(word).rstrip(".") in _TITLES and i + 1 < len(words) and capital(words[i + 1][2]) \
            and text[words[i][1]:words[i + 1][0]].strip(" .") == ""
        if not (titled or capital(word) or initial(word)):
            i += 1
            continue
        j = i + 1 if titled else i
        while j + 1 < len(words) and (capital(words[j + 1][2]) or initial(words[j + 1][2])) and joined(j, j + 1):
            j += 1
        run = list(range(i + 1 if titled else i, j + 1))
        # The question's own words at either end of the run are the question's ("Avery Quill Sales").
        while run and own_word(words[run[0]][2]) and not titled:
            run.pop(0)
        # A title keeps the word after it, the workspace's or not: "Mrs. Price" is a person though "price" is a field.
        while len(run) > (1 if titled else 0) and own_word(words[run[-1]][2]):
            run.pop()
        names = [k for k in run if capital(words[k][2])]
        if titled and names:
            if " ".join(lowered(words[k][2]) for k in [i, *run]) not in keys:
                spans.append((words[i][0], words[run[-1]][1]))
        elif len(run) >= 2 and names and not safe(run):
            spans.append((words[run[0]][0], words[run[-1]][1]))
        elif len(run) == 1 and names and j + 1 < len(words) and capital(words[j + 1][2]) \
                and text[words[j][1]:words[j + 1][0]] == ", " and not safe([run[0]]) and not safe([j + 1]):
            spans.append((words[run[0]][0], words[j + 1][1]))      # "Smith, John"
            j += 1
        i = j + 1
    return spans


def scrub_unmasked_free_text(
    rows: list[dict],
    columns: list[dict],
    skip_fields: set[str],
    industry: str = "",
) -> list[dict]:
    """
    Final safety net: scrub embedded PII from narrative string values in
    columns that were NOT already masked.

    Only touches string values that look like free text (contain whitespace and
    are reasonably long), so short codes / identifiers are left intact. Mutates
    and returns the same row dicts (rows are short-lived discovery samples).

    For regulated industries ("banking" / "healthcare_pharmacy") an additional
    NER pass (Presidio, when installed) replaces person-name spans the regex
    scrub structurally cannot catch — see scrub_person_names_ner. Standard
    clients keep today's regex-only behavior.
    """
    if not rows:
        return rows
    skip = {f for f in (skip_fields or set())}
    ner_pass = industry in ("banking", "healthcare_pharmacy")
    for row in rows:
        for field, val in row.items():
            if field in skip:
                continue
            if isinstance(val, str) and len(val) > 12 and " " in val:
                scrubbed = scrub_embedded_pii(val)
                if ner_pass:
                    scrubbed = scrub_person_names_ner(scrubbed)
                row[field] = scrubbed
    return rows


def _luhn_check(n: str) -> bool:
    """Validate a digit string using the Luhn algorithm."""
    digits = [int(c) for c in n if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _could_be_credit_card(v: str) -> bool:
    """True if v looks like a credit card number (13-19 digits, passes Luhn)."""
    stripped = _RE_CC_STRIP.sub("", v)
    if not stripped.isdigit():
        return False
    return _luhn_check(stripped)


# ── Value-scan type-skip sets (module-level so they aren't rebuilt each call) ─
_NUMERIC_TYPE_HINTS: frozenset[str] = frozenset({
    "int", "bigint", "smallint", "tinyint", "decimal",
    "numeric", "float", "real", "money", "number", "bit",
})
_DATE_TYPE_HINTS: frozenset[str] = frozenset({
    "date", "time", "datetime", "timestamp",
})

# Pre-compiled regex for stripping separators in credit-card candidates.
# Defined at module level so it isn't recompiled on every _could_be_credit_card call.
_RE_CC_STRIP = re.compile(r"[\s\-]")

# Maps detected PII type → masking strategy
_VALUE_PII_STRATEGY: dict[str, str] = {
    "email":       "email",
    "ssn":         "ssn",
    "pan":         "ssn",        # Indian PAN — treat as government ID
    "aadhaar":     "ssn",        # Indian Aadhaar — treat as government ID
    "credit_card": "credit_card",
    "ip":          "ip_address",
    "phone":       "phone",
}

# Ordered list for stable priority when multiple types could match one value
_VALUE_PII_ORDER = ["email", "ssn", "pan", "aadhaar", "credit_card", "ip", "phone"]


def _value_pii_types(v: str) -> list[str]:
    """Return list of PII types that match value v."""
    matches: list[str] = []
    if _RE_EMAIL.match(v):
        matches.append("email")
    if _RE_SSN.match(v):
        matches.append("ssn")
    if _RE_PAN.match(v):
        matches.append("pan")
    if _RE_AADHAAR.match(v):
        matches.append("aadhaar")
    if _could_be_credit_card(v):
        matches.append("credit_card")
    if _RE_IP.match(v):
        matches.append("ip")
    elif _is_phone_like(v):          # phone check: skip if already matched as IP
        matches.append("phone")
    return matches


def _check_value_pii(values: list[str]) -> tuple[str, float] | None:
    """
    Examine a sample of string values for a single column.

    Returns (pii_type, confidence) if >60 % of values match a single PII
    pattern, else None.  When multiple types match (e.g. Aadhaar also looks
    like a phone number), the highest-priority type wins.
    """
    if not values:
        return None
    total = len(values)
    counts: dict[str, int] = {k: 0 for k in _VALUE_PII_ORDER}
    for v in values:
        for pii_type in _value_pii_types(v):
            counts[pii_type] += 1

    best_type: str | None = None
    best_ratio = 0.0
    for pii_type in _VALUE_PII_ORDER:
        ratio = counts[pii_type] / total
        if ratio >= 0.60 and ratio > best_ratio:
            best_type = pii_type
            best_ratio = ratio
    if best_type:
        return best_type, best_ratio
    return None


def scan_values_for_pii(
    rows: list[dict],
    col_defs: list[dict],
) -> dict[str, dict]:
    """
    Scan actual sample values for value-level PII patterns.

    Returns {col_name: {"pii_type": str, "strategy": str, "confidence": float}}
    for columns where >60 % of non-null values match a PII pattern.

    Only examines string/text columns — numeric and date columns are skipped
    because patterns like SSN or phone would cause false positives on
    employee IDs, revenue figures, etc.
    """
    if not rows:
        return {}

    col_type_map = {c["name"]: c.get("type", "varchar").lower() for c in col_defs}
    result: dict[str, dict] = {}

    for col_name, col_type in col_type_map.items():
        ct_base = col_type.split("(")[0].strip()
        if any(t in ct_base for t in _NUMERIC_TYPE_HINTS):
            continue
        if any(t in ct_base for t in _DATE_TYPE_HINTS):
            continue

        values = [
            str(row[col_name]).strip()
            for row in rows
            if col_name in row and row[col_name] is not None
            and str(row[col_name]).strip()
        ]
        if not values:
            continue

        hit = _check_value_pii(values)
        if hit:
            pii_type, confidence = hit
            strategy = _VALUE_PII_STRATEGY.get(pii_type, "redact")
            result[col_name] = {
                "pii_type":  pii_type,
                "strategy":  strategy,
                "confidence": round(confidence, 2),
            }
            log.info(
                "masking: value-scan detected %s in %r (%.0f%% match → strategy=%s)",
                pii_type, col_name, confidence * 100, strategy,
            )

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def detect_sensitive_columns(columns: list[dict], industry: str = "") -> dict[str, str]:
    """
    Return {column_name: masking_strategy} for columns likely to contain PII.

    Detection runs in three passes:
    1. Name-based regex scan (_PII_PATTERNS).
    2. Type-based free-text detection — flags unbounded text columns whose
       name does not indicate a known-safe business field.
    3. When `industry` is a regulated pack ("banking" or "healthcare_pharmacy"),
       also run each remaining column through the compliance classifier
       (core.compliance.classifier.classify_column) — it detects category
       patterns pass 1 doesn't cover (e.g. a bare DIAGNOSIS_CD, RX_DRUG_NAME,
       or NDC code column: PHI/PRESCRIPTION for healthcare, PCI/KYC_AML for
       banking). This is the same detector the query-time compliance system
       uses, applied here so unmasked sample rows never reach KB generation
       for a regulated client. Reuses classify_column's own industry
       filtering (banking excludes PHI/PRESCRIPTION/PAYMENT tags, healthcare
       excludes PCI/KYC_AML), so no separate pattern list to maintain here.
    """
    result: dict[str, str] = {}
    for col in columns:
        col_name = col.get("name", "")
        col_type = col.get("type", "")

        # Pass 1 — name-based
        strategy = _strategy_for_name(col_name)
        if strategy:
            result[col_name] = strategy
            continue

        # Pass 2 — type-based free-text detection
        if _is_free_text_type(col_type):
            name_lower = col_name.lower()
            if not any(hint in name_lower for hint in _SAFE_LONG_TEXT_NAMES):
                result[col_name] = "free_text"
                log.debug(
                    "masking: type-based free_text flag on %r (type=%r)",
                    col_name, col_type,
                )
                continue

        # Pass 3 — industry-specific compliance categories. Gated on the two
        # real regulated packs specifically — get_compliance_profile()'s
        # default row reports industry="standard" for every unconfigured
        # client, so a bare truthy check here would silently change
        # detection for every standard client too.
        if industry in ("banking", "healthcare_pharmacy"):
            try:
                from core.compliance.classifier import classify_column
                if classify_column(col_name, industry).get("tags"):
                    result[col_name] = strategy_for_field(col_name, col_type)
                    log.debug(
                        "masking: industry-tagged (%s) auto-masked %r",
                        industry, col_name,
                    )
            except Exception as exc:
                log.debug("masking: industry classification skipped for %r: %s", col_name, exc)

    return result


# Human-readable labels for each masking strategy (used in UI + egress log).
STRATEGY_LABELS: dict[str, str] = {
    "redact":        "→ [REDACTED]",
    "name":          "→ fake full name",
    "first_name":    "→ fake first name",
    "last_name":     "→ fake last name",
    "drug_name":     "→ fake drug name",
    "email":         "→ fake email",
    "phone":         "→ fake phone number",
    "address":       "→ fake address",
    "city":          "→ fake city",
    "zip":           "→ fake zip code",
    "state":         "→ fake state",
    "country":       "→ fake country code",
    "ssn":           "→ fake SSN",
    "credit_card":   "→ ****-****-****-XXXX",
    "ip_address":    "→ fake IP address",
    "coordinate":    "→ shifted ±0.5°",
    "birthdate":     "→ shifted ±730 days",
    "date_shift":    "→ shifted ±5 days",
    "salary":        "→ shifted ±15%",
    "numeric_shift": "→ shifted ±10%",
    "text_mask":     "→ format-preserving mask",
    "free_text":     "→ [REDACTED TEXT]",
}


def strategy_for_field(field: str, col_type: str = "varchar") -> str:
    """Resolve the masking strategy for a field name/type pair."""
    s = _strategy_for_name(field)
    if s:
        return s
    ct = (col_type or "varchar").lower()
    if any(t in ct for t in ("int", "bigint", "decimal", "numeric",
                             "float", "real", "money", "number")):
        return "numeric_shift"
    if any(t in ct for t in ("date", "time", "timestamp")):
        return "date_shift"
    if _is_free_text_type(ct):
        return "free_text"
    return "text_mask"


def get_strategy_map(masked_fields: set[str], col_defs: list[dict]) -> dict[str, str]:
    """
    Return {field_name: strategy_name} for every field in *masked_fields*.
    """
    col_type_map = {c["name"]: c.get("type", "varchar") for c in col_defs}
    result: dict[str, str] = {}
    for field in masked_fields:
        result[field] = strategy_for_field(field, col_type_map.get(field, "varchar"))
    return result


def mask_rows(
    rows: list[dict],
    masked_fields: set[str],
    columns: list[dict],
    seed_key: str = "",
    strategy_overrides: dict[str, str] | None = None,
) -> list[dict]:
    """
    Apply column-level masking to *rows*.

    Parameters
    ----------
    rows             : real sample rows from the DB
    masked_fields    : set of column names to mask
    columns          : [{name, type}, ...] column metadata
    seed_key         : when non-empty, use HMAC-based deterministic masking
                       so the same real value → same fake across all tables.
                       Typically set to the account_id.
    strategy_overrides: per-column strategy overrides (from value-scan).
                       These take highest priority over name/type detection.
    """
    if not rows or not masked_fields:
        return rows

    col_type_map = {c["name"]: c.get("type", "varchar") for c in columns}
    strategy_map: dict[str, str] = {}
    for field in masked_fields:
        # Value-based override takes highest priority
        if strategy_overrides and field in strategy_overrides:
            strategy_map[field] = strategy_overrides[field]
        else:
            strategy_map[field] = strategy_for_field(field, col_type_map.get(field, "varchar"))

    faker = _get_faker()
    return [
        _mask_row(row, strategy_map, faker,
                  seed_key=seed_key, col_type_map=col_type_map)
        for row in rows
    ]


# ── Deterministic masking (Item 3) ────────────────────────────────────────────

def _hmac_seed(value: Any, col_type: str, seed_key: str) -> int:
    """
    Derive a stable integer seed from (value, col_type, seed_key).

    Using HMAC-SHA256 ensures:
    - Same value → same seed across tables (FK consistency).
    - Different values → different seeds (no collisions within strategy).
    - seed_key isolates one account from another.

    Only called from _mask_row when seed_key is non-empty (guarded by
    ``if seed_key:``), so seed_key is always truthy here.
    """
    msg = f"{col_type}:{value}".encode("utf-8")
    digest = _hmac_mod.new(seed_key.encode("utf-8"), msg, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")


def _apply_det(value: Any, strategy: str, rng: random.Random) -> Any:
    """
    Deterministic masking using a pre-seeded RNG.

    Mirrors _apply() but replaces all random calls with rng.* so the same
    (value, strategy, seed_key) always produces the same output.
    """
    if strategy == "redact":
        return "[REDACTED]"
    if strategy == "free_text":
        return "[REDACTED TEXT]"
    if strategy == "name":
        return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    if strategy == "first_name":
        return rng.choice(_FIRST)
    if strategy == "last_name":
        return rng.choice(_LAST)
    if strategy == "drug_name":
        return rng.choice(_DRUG_NAMES)
    if strategy == "email":
        digits = "".join(rng.choices(string.digits, k=4))
        domain = rng.choice(["example.com", "mail.test", "test.org", "sample.net"])
        return f"user{digits}@{domain}"
    if strategy == "phone":
        # NNN-555-0100..0199 is the NANP-reserved fictional block — these
        # numbers can never be assigned to a real subscriber.
        return f"+1-{rng.randint(200, 999)}-555-01{rng.randint(0, 99):02d}"
    if strategy == "address":
        return f"{rng.randint(1, 9999)} {rng.choice(_STREET_NAMES)} St"
    if strategy == "city":
        return rng.choice(_CITIES)
    if strategy == "zip":
        return str(rng.randint(10000, 99999))
    if strategy == "state":
        return rng.choice(_STATES)
    if strategy == "country":
        return rng.choice(_COUNTRIES)
    if strategy == "ssn":
        # Area numbers 900-999 are never issued by the SSA — a generated
        # fake can never collide with a real person's SSN.
        return (
            f"{rng.randint(900, 999)}-"
            f"{rng.randint(10, 99)}-"
            f"{rng.randint(1000, 9999)}"
        )
    if strategy == "credit_card":
        return f"****-****-****-{rng.randint(1000, 9999)}"
    if strategy == "ip_address":
        return (
            f"{rng.randint(10, 192)}.{rng.randint(0, 255)}."
            f"{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        )
    if strategy == "coordinate":
        # No local try/except: a conversion failure must propagate to
        # _mask_row's except clause, which substitutes "[MASKED]" -- see
        # _shift_date's docstring for why returning the raw value here
        # instead is the actual defect.
        return round(float(value) + rng.uniform(-0.5, 0.5), 6)
    if strategy == "birthdate":
        return _shift_date(value, max_days=730, rng=rng)
    if strategy == "date_shift":
        return _shift_date(value, max_days=5, rng=rng)
    if strategy == "salary":
        return _shift_numeric(value, pct=0.15, rng=rng)
    if strategy == "numeric_shift":
        return _shift_numeric(value, pct=0.10, rng=rng)
    if strategy == "text_mask":
        return _mask_text(str(value), rng=rng)
    return value  # unknown strategy — pass through


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strategy_for_name(col_name: str) -> str | None:
    cn = col_name.lower()
    for pattern, strategy in _PII_PATTERNS:
        if re.search(pattern, cn):
            return strategy
    return None


def _mask_row(
    row: dict,
    strategy_map: dict[str, str],
    faker,
    seed_key: str = "",
    col_type_map: dict[str, str] | None = None,
) -> dict:
    result = dict(row)
    for field, strategy in strategy_map.items():
        if field not in result or result[field] is None:
            continue
        try:
            if seed_key:
                col_type = (col_type_map or {}).get(field, "varchar")
                seed = _hmac_seed(result[field], col_type, seed_key)
                rng = random.Random(seed)
                result[field] = _apply_det(result[field], strategy, rng)
            else:
                result[field] = _apply(result[field], strategy, faker)
        except Exception as exc:
            # Loud on purpose. This was silent, and four strategies
            # (coordinate, birthdate, date_shift, salary, numeric_shift) used
            # to catch their OWN conversion failure and return the raw value
            # instead of raising -- which made this except clause, the one
            # thing standing between a masking bug and a raw regulated value
            # reaching a KB sample or LLM prompt, unreachable for exactly
            # those failures. See core/masking.py's fixed strategies for
            # where that raise now happens.
            #
            # Logged by exception TYPE only, never str(exc) or the value
            # itself: Python's own ValueError from float() or strptime()
            # embeds the unparseable string verbatim in its message, which
            # would turn this log line into the very leak it exists to catch.
            log.warning(
                "Masking failed for field=%r strategy=%r (%s) -- "
                "substituting [MASKED] rather than the source value",
                field, strategy, type(exc).__name__,
            )
            result[field] = "[MASKED]"
    return result


def _apply(value: Any, strategy: str, faker) -> Any:  # noqa: C901
    if strategy == "redact":
        return "[REDACTED]"

    if strategy == "name":
        return faker.name() if faker else _stdlib_name()
    if strategy == "first_name":
        return faker.first_name() if faker else random.choice(_FIRST)
    if strategy == "last_name":
        return faker.last_name() if faker else random.choice(_LAST)
    if strategy == "drug_name":
        # Faker has no medication provider — use the curated fictional pool
        # in both paths so drug columns never carry person names or real
        # drug names into KB samples.
        return random.choice(_DRUG_NAMES)
    if strategy == "email":
        return faker.email() if faker else f"user{random.randint(100,9999)}@example.com"
    if strategy == "phone":
        # Never faker.phone_number() — it produces realistic numbers that
        # can collide with a real subscriber. NNN-555-0100..0199 is the
        # NANP-reserved fictional block.
        return f"+1-{random.randint(200, 999)}-555-01{random.randint(0, 99):02d}"
    if strategy == "address":
        return faker.street_address() if faker else f"{random.randint(1,9999)} Oak Street"
    if strategy == "city":
        return faker.city() if faker else random.choice(_CITIES)
    if strategy == "zip":
        return faker.zipcode() if faker else f"{random.randint(10000,99999)}"
    if strategy == "state":
        return faker.state_abbr() if faker else random.choice(_STATES)
    if strategy == "country":
        return faker.country_code() if faker else random.choice(_COUNTRIES)
    if strategy == "ssn":
        # Area 900-999 is never issued by the SSA — see _apply_det.
        return f"{random.randint(900,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
    if strategy == "credit_card":
        return f"****-****-****-{random.randint(1000,9999)}"
    if strategy == "ip_address":
        return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    if strategy == "coordinate":
        # No local try/except: see _apply_det's identical branch and
        # _shift_date's docstring below for why.
        return round(float(value) + random.uniform(-0.5, 0.5), 6)
    if strategy == "birthdate":
        return _shift_date(value, max_days=730)
    if strategy == "date_shift":
        return _shift_date(value, max_days=5)
    if strategy == "salary":
        return _shift_numeric(value, pct=0.15)
    if strategy == "numeric_shift":
        return _shift_numeric(value, pct=0.10)
    if strategy == "text_mask":
        return _mask_text(str(value))
    if strategy == "free_text":
        return "[REDACTED TEXT]"

    return value  # unknown strategy — pass through


def _shift_date(
    value: Any,
    max_days: int = 5,
    rng: random.Random | None = None,
) -> Any:
    """Shift a date value by a small random offset for de-identification.

    Raises when `value` is neither a date/datetime nor one of the five
    formats tried, rather than returning it unshifted. This function has no
    caller of its own -- it is reached only through _apply/_apply_det, inside
    _mask_row's try block, whose except clause substitutes "[MASKED]" for
    exactly this case. Returning the raw value here instead of raising made
    that safety net unreachable: an ISO timestamp with a timezone offset
    ("1985-03-02T00:00:00+05:00"), or any format outside the five tried,
    wrote the real, unmasked birthdate straight into a KB sample or LLM
    prompt with nothing to show it had happened.
    """
    from datetime import timedelta, date, datetime
    _rng = rng or random
    shift = _rng.randint(-max_days, max_days)
    if isinstance(value, datetime):
        return value + timedelta(days=shift)
    if isinstance(value, date):
        return value + timedelta(days=shift)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(str(value), fmt)
            return (d + timedelta(days=shift)).strftime(fmt)
        except ValueError:
            continue
    # Deliberately no value in the message: it reaches _mask_row's except
    # clause, and from there is logged by exception TYPE only, never
    # str(exc) -- but raising ValueError(f"...{value}...") here would still
    # be the wrong habit to leave behind for the next strategy added to this
    # function to copy.
    raise ValueError("_shift_date: value is not a recognized date format")


def _shift_numeric(
    value: Any,
    pct: float = 0.10,
    rng: random.Random | None = None,
) -> Any:
    """Shift a numeric value by a small random percentage for
    de-identification.

    Raises when `value` cannot convert to float, rather than returning it
    unshifted -- see _shift_date's docstring for why: the only safety net for
    that failure is the caller's except clause, which this function used to
    make unreachable by catching its own conversion failure and returning the
    raw value instead. A salary column containing "$85,000" (comma and
    currency symbol) is exactly the shape that fails float() and used to
    write the real, unmasked salary straight through.
    """
    _rng = rng or random
    n = float(value)
    n_new = n * (1.0 + _rng.uniform(-pct, pct))
    if isinstance(value, int):
        return max(0, int(round(n_new)))
    if isinstance(value, str) and value.isdigit():
        return max(0, int(round(n_new)))
    return round(n_new, 2)


def _mask_text(value: str, rng: random.Random | None = None) -> str:
    """Format-preserving text mask: replace alpha→alpha, digit→digit, keep rest."""
    _rng = rng or random
    result = []
    for ch in value:
        if ch.isalpha():
            pool = string.ascii_lowercase if ch.islower() else string.ascii_uppercase
            result.append(_rng.choice(pool))
        elif ch.isdigit():
            result.append(str(_rng.randint(0, 9)))
        else:
            result.append(ch)
    return "".join(result)


# ── Faker singleton ───────────────────────────────────────────────────────────

_faker_instance = None


def _get_faker():
    global _faker_instance
    if _faker_instance is not None:
        return _faker_instance
    try:
        from faker import Faker  # type: ignore
        _faker_instance = Faker()
        Faker.seed(42)
        log.debug("masking: Faker loaded")
        return _faker_instance
    except ImportError:
        log.info("masking: Faker not installed — using stdlib fallback")
        _faker_instance = False  # sentinel so we only try once
        return None


# ── Stdlib fallback data pools ────────────────────────────────────────────────
# 30 names each so deterministic sampling has good variety

_FIRST = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
    "Quinn", "Avery", "Blake", "Cameron", "Dana", "Ellis",
    "Finley", "Harper", "Hayden", "Jamie", "Jesse", "Kendall",
    "Lee", "Logan", "London", "Madison", "Mason", "Peyton",
    "Reese", "River", "Rowan", "Sage", "Skylar", "Sydney",
]
_LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Wilson", "Moore", "Anderson", "Lee",
    "Taylor", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Young", "Lewis", "Walker", "Hall", "Allen",
    "King", "Wright", "Scott", "Green", "Baker", "Adams",
]
_CITIES = [
    "Springfield", "Riverside", "Fairview", "Madison", "Georgetown",
    "Burlington", "Salem", "Greenville", "Franklin", "Arlington",
    "Clinton", "Milford", "Newport", "Chester", "Ashland",
]
_STATES   = ["CA", "TX", "NY", "FL", "IL", "WA", "OH", "GA", "NC", "PA"]
_COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "SG", "IN", "JP", "BR"]
_STREET_NAMES = [
    "Oak", "Maple", "Cedar", "Pine", "Elm", "Birch",
    "Willow", "Spruce", "Aspen", "Walnut", "Chestnut", "Magnolia",
]

# Fictional medication names — drug-shaped (common pharma suffixes) but
# invented, so KB samples look plausible without naming a real product.
# Used for drug/medication/compound name columns, which previously fell
# through to the bare 'name' pattern and got fake PERSON names.
_DRUG_NAMES = [
    "Abrovine", "Bexatrol", "Cortivane", "Delfazine", "Elutrix", "Fenolart",
    "Gastrolev", "Hydrovex", "Ivorelin", "Juvelast", "Kelotrine", "Lumizoral",
    "Mavrodine", "Nortivex", "Oprazane", "Pexolium", "Quorvast", "Rilotane",
    "Selvatrix", "Tovaprene", "Uxolane", "Ventrizol", "Wexopril", "Zeltravin",
]


def _stdlib_name() -> str:
    return f"{random.choice(_FIRST)} {random.choice(_LAST)}"
