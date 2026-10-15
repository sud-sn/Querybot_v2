"""The name recogniser (Presidio) and the name rules, together, keep a typed name from the AI and keep the workspace's
own names in the question.

Presidio ran first and the rules after it. On "prescriptions for Smith, John" Presidio found "John" alone, and the
rules, which know "Smith, John" as one name, no longer saw it: "Smith" went to the AI. Presidio also knew nothing of
the workspace, so a store called Clinton, a supplier called Johnson & Johnson or a branch called Jackson went to the AI
as [PERSON]. And the rules withheld a title and a name ("Dr. Reddy's") even when that was a member's whole name.

Now both read the question as it was typed, and what either takes for a name is withheld: Presidio adds what the rules
cannot see (a name in lower case, one first name, a surname with a possessive), the rules add what Presidio misses
("Smith, John" whole). A span of Presidio's inside a member's or a place's whole name, or made of the workspace's own
words, is kept, as the rules keep it; one word of a member does not make a name safe ("Mary Franklin" is withheld though
a store is called Franklin, "Spencer" though a supplier is Marks & Spencer), and a title makes any word a name
("Mrs. Price", though "price" is the workspace's word).

A stand-in for Presidio flags the words each case names, so these run where the library is not installed; the last
tests run the real library when it is. Invented names; nothing is sent anywhere.
"""

from __future__ import annotations

import os
import re

import pytest

from core import masking
from core.masking import scrub_person_names_ner, scrub_typed_names
from tests.test_core2_answers_a_question_end_to_end import _domain


class _Flags:
    """Presidio's AnalyzerEngine, as far as the scrubber uses it: flags each whole-word occurrence of the given words
    (case-insensitive), as a trained recogniser would flag a name."""

    def __init__(self, *words: str):
        self.words = words
        self.seen: list[str] = []

    def analyze(self, text, entities, language, score_threshold):
        assert entities == ["PERSON"] and language == "en"
        self.seen.append(text)

        class _Span:
            def __init__(self, start, end):
                self.start, self.end, self.score = start, end, 0.85

        return [_Span(m.start(), m.end()) for w in self.words
                for m in re.finditer(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text, re.I)]


@pytest.fixture
def scrubber(tmp_path, monkeypatch):
    """The scrubber of a pharmacy under compliance on the retail test workspace (stores called Clinton, Franklin ...),
    with Presidio standing in as the analyser given."""
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from core2.service import question_scrubber

    store.init_db()
    _, _, model, _ = _domain("retail")

    def make(analyser, industry="healthcare_pharmacy"):
        monkeypatch.setattr(masking, "_presidio_analyzer", analyser)
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        store.save_compliance_profile(account_id, mode="regulated", industry=industry)
        return question_scrubber(account_id, model=model)

    return make


@pytest.mark.parametrize("industry", ["healthcare_pharmacy", "banking", "standard"])
def test_a_surname_comma_first_name_is_withheld_whole_when_presidio_finds_the_first_name(scrubber, industry):
    said = scrubber(_Flags("John"), industry)("net sales for Smith, John in 2024")
    assert "Smith" not in said and "John" not in said and said == "net sales for [PERSON] in 2024", said


@pytest.mark.parametrize("asked, gone", [
    ("how many fills did john smith get last month", "john smith"),
    ("what did Maria pick up last week", "Maria"),
    ("Was Aaliyah's claim paid?", "Aaliyah"),
])
def test_presidio_still_finds_what_the_rules_cannot(scrubber, asked, gone):
    said = scrubber(_Flags(gone))(asked)
    assert gone not in said and "[PERSON]" in said, said


@pytest.mark.parametrize("asked", [
    "compare Clinton and Franklin stores",
    "net sales at Georgetown last month",
    "fills in New York and Texas",
])
def test_a_member_or_a_place_presidio_takes_for_a_person_stays(scrubber, asked):
    flagged = _Flags("Clinton", "Franklin", "Georgetown", "New York", "Texas")
    assert scrubber(flagged)(asked) == asked


def test_one_word_of_a_member_does_not_make_a_name_safe(scrubber):
    said = scrubber(_Flags("Mary Franklin"))("fills for mary franklin last week")
    assert said == "fills for [PERSON] last week", said


def test_a_name_the_rules_withheld_is_not_flagged_again(scrubber):
    said = scrubber(_Flags("PERSON", "John Smith"))("net sales for John Smith")
    assert said == "net sales for [PERSON]", said


def test_presidio_reads_the_question_as_typed(scrubber):
    """A name the rules withhold only in part ("Mary Ann" of "Mary Ann Price", whose surname is a word of the
    workspace) is withheld whole when Presidio, reading the question as typed, finds it whole."""
    flags = _Flags("Mary Ann Price", "John Q. Price")
    scrub = scrubber(flags)
    assert scrub("net sales for Mary Ann Price last week") == "net sales for [PERSON] last week"
    assert scrub("net sales for John Q. Price in May") == "net sales for [PERSON] in May"
    assert any("Mary Ann Price" in seen for seen in flags.seen), flags.seen


@pytest.mark.parametrize("asked, gone", [
    ("how many orders did Mrs. Price place last month", "Price"),
    ("patients of Dr. Gross by month", "Gross"),
    ("what did Ms. May buy", "May"),
    ("orders of Mrs West", "West"),
])
@pytest.mark.parametrize("presidio", [True, False])
def test_a_title_makes_any_word_a_name(scrubber, asked, gone, presidio):
    said = scrubber(_Flags(gone) if presidio else False)(asked)
    assert gone not in said and "[PERSON]" in said, said


def test_a_name_typed_in_brackets_is_no_placeholder(scrubber):
    said = scrubber(_Flags("MARIA"))("what did [MARIA] buy last week")
    assert "MARIA" not in said, said


def test_one_piece_of_a_members_name_is_not_a_member(tmp_path, monkeypatch):
    """Learn keeps a member's whole name, and each run of two or more of its words; never one word cut from it, so
    "Spencer" alone is a person's name though a supplier is called Marks & Spencer."""
    from core2.model.schema import TopValue
    from core2.service import _vocabulary

    _, _, model, _ = _domain("retail")
    model = model.model_copy(deep=True)
    column = next(c for c in model.columns.values() if c.name == "category_name")
    column.profile.top.extend([TopValue(value="Marks & Spencer", count=3), TopValue(value="Ben & Jerry's", count=2),
                               TopValue(value="North Region / Central Store 12", count=1)])
    _, phrases = _vocabulary(model)
    assert {"marks & spencer", "ben & jerry's", "central store", "north region"} <= phrases
    assert not {"spencer", "marks", "ben", "jerry's"} & phrases
    for asked in ("what did Spencer buy last week", "what did ben pick up"):
        said = scrub_person_names_ner(asked, analyzer=_Flags("Spencer", "ben"), known=KNOWN, phrases=phrases)
        assert "[PERSON]" in said and "Spencer" not in said and " ben " not in said, said
    kept = "net sales at Marks & Spencer and Ben & Jerry's"
    assert scrub_person_names_ner(kept, analyzer=_Flags("Spencer", "Ben", "Jerry"), known=KNOWN, phrases=phrases) == kept


def test_without_presidio_the_rules_alone_still_withhold_names(scrubber):
    assert scrubber(False)("net sales for Smith, John in 2024") == "net sales for [PERSON] in 2024"


# ── The recogniser on its own, given the workspace's words ──────────────────────────────────────────────────────────

PHRASES = frozenset({"johnson & johnson", "jackson", "dr. reddy's", "franklin"})
KNOWN = frozenset({"fills", "sales", "branches", "products"})


@pytest.mark.parametrize("asked, flagged", [
    ("sales of Johnson & Johnson products", "Johnson"),
    ("compare Jackson and Madison branches", "Jackson"),
    ("units of Dr. Reddy's generics", "Reddy"),
    ("Fills by month", "Fills"),
])
def test_a_span_of_the_workspaces_own_names_is_kept(asked, flagged):
    assert scrub_person_names_ner(asked, analyzer=_Flags(flagged), known=KNOWN, phrases=PHRASES) == asked


def test_a_span_that_is_more_than_a_member_is_withheld():
    said = scrub_person_names_ner("fills for Mary Franklin", analyzer=_Flags("Mary Franklin"), known=KNOWN,
                                  phrases=PHRASES)
    assert said == "fills for [PERSON]", said


def test_without_the_workspaces_words_every_span_is_withheld_as_before():
    said = scrub_person_names_ner("compare Jackson and Madison branches", analyzer=_Flags("Jackson"))
    assert said == "compare [PERSON] and Madison branches", said


# ── The rules: a title and a name that is a member's whole name ─────────────────────────────────────────────────────

def test_a_title_and_a_name_that_is_a_members_whole_name_stays():
    asked = "units of Dr. Reddy's generics sold"
    assert scrub_typed_names(asked, KNOWN, PHRASES) == asked


@pytest.mark.parametrize("asked, phrases", [
    ("units of Dr. Reddy's generics sold", frozenset()),
    ("patients of Dr Franklin by month", PHRASES),        # a store is called Franklin; Dr Franklin is a person
])
def test_a_title_and_a_name_is_otherwise_still_withheld(asked, phrases):
    said = scrub_typed_names(asked, KNOWN, phrases)
    assert "[PERSON]" in said and "Reddy" not in said and "Dr Franklin" not in said, said


# ── Which model ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_medium_english_model_is_preferred(monkeypatch):
    monkeypatch.delenv("QB_NER_MODEL", raising=False)
    installed = {"en_core_web_lg", "en_core_web_md", "en_core_web_sm"}
    assert masking._ner_model(installed.__contains__) == "en_core_web_md"
    assert masking._ner_model({"en_core_web_lg"}.__contains__) == "en_core_web_lg"
    assert masking._ner_model(set().__contains__) is None


def test_an_admin_names_the_model(monkeypatch):
    monkeypatch.setenv("QB_NER_MODEL", "en_core_web_lg")
    assert masking._ner_model({"en_core_web_lg", "en_core_web_md"}.__contains__) == "en_core_web_lg"
    assert masking._ner_model({"en_core_web_md"}.__contains__) is None, "a named model that is missing is not swapped"


# ── Never loaded on the event loop ──────────────────────────────────────────────────────────────────────────────────

def test_on_the_event_loop_the_model_loads_in_the_background(monkeypatch):
    """A question read on the server's event loop before the model is loaded is withheld by the rules alone while
    the model loads in a worker; the loop is never held for the seconds loading takes."""
    import asyncio
    import threading
    import time

    loaded = threading.Event()
    flags = _Flags("Maria")

    def slow_load():
        time.sleep(0.5)
        loaded.set()
        return flags, "en_core_web_md"

    monkeypatch.setattr(masking, "_presidio_analyzer", None)
    monkeypatch.setattr(masking, "_presidio_model", "")
    monkeypatch.setattr(masking, "_presidio_loading", False)
    monkeypatch.setattr(masking, "_load_presidio", slow_load)

    async def on_the_loop():
        started = time.perf_counter()
        said = masking.scrub_names("what did Maria pick up", KNOWN, PHRASES)
        return said, time.perf_counter() - started

    said, took = asyncio.run(on_the_loop())
    assert took < 0.2 and said == "what did Maria pick up", (said, took)
    assert loaded.wait(5)
    for _ in range(50):
        if masking._presidio_analyzer is flags:
            break
        time.sleep(0.02)
    assert masking.scrub_names("what did Maria pick up", KNOWN, PHRASES) == "what did [PERSON] pick up"


# ── The compliance page says which is on ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("analyser, said", [
    (False, "by rules only"),
    (_Flags(), "and by a name recogniser"),
])
def test_the_proof_pack_says_how_typed_names_are_withheld(tmp_path, monkeypatch, analyser, said):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from core.compliance.proof_pack import build_proof_pack, summary_lines

    store.init_db()
    monkeypatch.setattr(masking, "_presidio_analyzer", analyser)
    monkeypatch.setattr(masking, "_presidio_model", "en_core_web_md" if analyser else "")
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.save_compliance_profile(account_id, mode="regulated", industry="healthcare_pharmacy")
    pack = build_proof_pack(account_id)
    assert said in pack["names"]["statement"], pack["names"]
    assert pack["names"]["recogniser"] == bool(analyser)
    assert any(said in line for line in summary_lines(pack))
    assert pack["names"]["install"] == ("" if analyser else "python -m spacy download en_core_web_md")


def test_the_install_line_names_the_model_an_admin_chose(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("QB_NER_MODEL", "en_core_web_lg")
    import store
    from core.compliance.proof_pack import names_section

    store.init_db()
    monkeypatch.setattr(masking, "_presidio_analyzer", False)
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.save_compliance_profile(account_id, mode="regulated", industry="healthcare_pharmacy")
    assert names_section(account_id)["install"] == "python -m spacy download en_core_web_lg"


# ── The real library, where it is installed ─────────────────────────────────────────────────────────────────────────

_LOADED: list = []


def _real_presidio(monkeypatch):
    """The real engine, loaded once (it takes seconds); the module's own cache is put back after each test."""
    pytest.importorskip("presidio_analyzer")
    if not _LOADED:
        monkeypatch.setattr(masking, "_presidio_analyzer", None)
        monkeypatch.setattr(masking, "_presidio_model", masking._presidio_model)
        _LOADED.append(masking._get_presidio())
    if _LOADED[0] is None:
        pytest.skip("Presidio is installed without an English spaCy model")
    return _LOADED[0]


@pytest.mark.parametrize("asked, gone", [
    ("how many fills did john smith get last month", ("john", "smith")),
    ("what did Maria pick up last week", ("Maria",)),
    ("net sales for Smith, John in 2024", ("Smith", "John")),
    ("Was Aaliyah's claim paid?", ("Aaliyah",)),
])
def test_with_the_real_library_names_are_withheld(scrubber, monkeypatch, asked, gone):
    said = scrubber(_real_presidio(monkeypatch))(asked)
    assert not any(re.search(r"\b" + g + r"\b", said) for g in gone), said


@pytest.mark.parametrize("asked", [
    "compare Clinton and Franklin stores",
    "net sales at Georgetown in May",
    "net sales by store in March",
])
def test_with_the_real_library_the_workspaces_names_stay(scrubber, monkeypatch, asked):
    assert scrubber(_real_presidio(monkeypatch))(asked) == asked
