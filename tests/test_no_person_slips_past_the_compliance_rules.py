"""No person slips past the compliance rules: what a review of them found, each kept closed.

* The AI call log of a workspace under compliance kept the question as typed, a person's name and all, though the
  AI itself was given it scrubbed.
* An answer filtered on a person (the reader pointing at a patient on screen) or on a prescription's number, or a
  "why" or a forecast about one, was summarised by the AI with the person in its headline; a follow-up that
  filtered on the patient as a whole ("patient") was not read as being about people.
* A prescription's number held as a whole number was read as an amount: the catalog offered to add it up, and a
  previous plan filtering on it showed the number to the AI unmasked.
* Asked to plan again after it asked which member was meant, the AI was told the member's name, though it had only
  ever been shown its placeholder.
* Metric authoring listed the values of a column an admin had marked as people's.
* The AI call list read every kept prompt of the workspace into memory on every page.
* The name scrubber missed "Smith, John", initials, capitals and a name with a word of a member in it ("Mary
  Franklin"), and withheld ordinary phrases ("Customer Lifetime Value", "New York").
* A patient's number written PT_NO was a part number's in a parts warehouse; RXNBR and ORIG_RX_NUMBER were not
  read as prescriptions' numbers, and a column called just RX was.

Invented data throughout; the AI is replaced and keeps what it is sent.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from types import SimpleNamespace

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import about_people, build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from tests.test_core2_answers_in_the_portal import learned  # noqa: F401 - the fixture
from tests.test_core2_learned_page_shows_what_was_learned import ACCOUNT, workspace  # noqa: F401 - the fixture

TODAY = dt.date(2026, 6, 15)
YEAR = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def pharmacy():
    con = duckdb.connect()
    con.execute("CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, first_name VARCHAR, last_name VARCHAR, "
                "state VARCHAR)")
    con.execute("INSERT INTO patients VALUES (1, 'Avery', 'Quill', 'OH'), (2, 'Blair', 'Moss', 'TX'), "
                "(3, 'Casey', 'Fern', 'OH')")
    con.execute("CREATE TABLE prescriptions (rx_number INTEGER PRIMARY KEY, patient_id INTEGER, written_date DATE, "
                "drug_class VARCHAR)")
    con.execute("INSERT INTO prescriptions SELECT 778700 + i, 1 + i % 3, DATE '2025-01-01' + CAST(i % 300 AS INTEGER), "
                "CASE i % 3 WHEN 0 THEN 'statin' WHEN 1 THEN 'opioid' ELSE 'antibiotic' END FROM range(1, 61) t(i)")
    con.execute("CREATE TABLE fills (fill_id INTEGER PRIMARY KEY, rx_number INTEGER, fill_date DATE, "
                "quantity INTEGER, price_charged DECIMAL(10, 2))")
    con.execute("INSERT INTO fills SELECT i, 778700 + 1 + i % 60, DATE '2025-01-03' + CAST(i % 150 AS INTEGER), "
                "1 + i % 4, round(20 + (i * 7) % 90 + (CASE WHEN i % 3 = 0 THEN 400 ELSE 0 END), 2) "
                "FROM range(1, 901) t(i)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="rx", options=BuildOptions(workers=1))
    return con, model


class Script:
    def __init__(self, *plans):
        self.plans = [p if isinstance(p, str) else json.dumps(p) for p in plans]
        self.sent: list[str] = []

    def __call__(self, stable: str, tail: str) -> str:
        self.sent.append(stable + "\n" + tail)
        return self.plans.pop(0)


def _services(con, model, ai, *, values_allowed: bool = True) -> Services:
    warehouse = DuckDBWarehouse(con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return Services(model=model, warehouse=warehouse, complete=ai, index=build_index(model, fetch), today=TODAY,
                    values_allowed=values_allowed)


def _slug(model, name: str) -> str:
    return next(a.slug for a in model.attributes.values() if model.columns[a.column].name == name
                and not model.columns[a.column].parts) if name != "name" else next(
        a.slug for a in model.attributes.values() if model.columns[a.column].parts)


def _sales(model) -> str:
    return next(m.slug for m in model.measures.values() if "price" in m.slug and not m.hidden)


# ── the AI call log keeps the question as the AI was given it ───────────────


def test_the_ai_call_log_of_a_regulated_workspace_never_keeps_a_typed_name(learned, monkeypatch):  # noqa: F811
    store, built, _ = learned
    from core import llm
    from core2 import service

    store.save_compliance_profile(ACCOUNT, mode="regulated", industry="healthcare_pharmacy")
    plan = json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"], "time": YEAR})
    reply = SimpleNamespace(content=[SimpleNamespace(text=plan)], stop_reason="end_turn",
                            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                                  cache_creation_input_tokens=0, cache_creation=None))

    async def create(**kwargs):
        return reply

    monkeypatch.setattr(llm, "_get_anthropic_client", lambda key: SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "resolve_provider", lambda *a, **k: ("anthropic", "claude-sonnet-4-6", "key", {}))
    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    service.portal_answer(ACCOUNT, "net sales for John Smith in 2025", {"id": 1, "role": "admin"}, session_key="s1")
    with store.get_db() as conn:
        kept = [r["question"] for r in conn.execute(
            "SELECT question FROM llm_call_log WHERE account_id=? AND component='core2_planner'", (ACCOUNT,))]
    assert kept, "the planner's call is kept, the workspace being regulated"
    assert all("John Smith" not in q and "[PERSON]" in q for q in kept), kept


# ── an answer about a person never reaches the AI's summary ─────────────────


def test_an_answer_filtered_on_the_person_pointed_at_is_never_summarised(pharmacy):
    from core2.answer.summary import eligible

    con, model = pharmacy
    person, sales = _slug(model, "name"), _sales(model)
    ai = Script({"kind": "query", "intent": "breakdown", "measures": [sales], "group_by": [person], "time": YEAR},
                {"kind": "query", "intent": "value", "measures": [sales], "time": YEAR, "follow_up": "refine",
                 "filters": [{"field": person, "op": "eq", "values": ["⟨v1⟩"]}]})
    services, session = _services(con, model, ai), Session()
    answer_question("sales by patient in 2025", services, session)
    pointed = answer_question("sales for the first one", services, session)
    assert any(f["field"] == person for f in pointed["plan"]["filters"]), pointed["plan"]
    assert pointed.get("personal") and not eligible(pointed, values_allowed=True)


def test_an_answer_for_one_prescription_is_never_summarised_and_its_number_never_shown(pharmacy):
    from core2.answer.summary import eligible

    con, model = pharmacy
    rx, sales = _slug(model, "rx_number"), _sales(model)
    assert model.attributes[rx].kind == "identifier", "a prescription's number names it; it is no amount"
    ai = Script({"kind": "query", "intent": "value", "measures": [sales], "time": YEAR,
                 "filters": [{"field": rx, "op": "eq", "values": [778712]}]},
                {"kind": "query", "intent": "value", "measures": [sales], "follow_up": "refine",
                 "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}},
                 "filters": [{"field": rx, "op": "eq", "values": [778712]}]})
    services, session = _services(con, model, ai), Session()
    first = answer_question('sales for prescription "778712" in 2025', services, session)
    assert first.get("personal") and not eligible(first, values_allowed=True)
    answer_question("and in 2026?", services, session)
    assert "778712" not in ai.sent[1].split("QUESTION:")[0], "the previous plan shows the number masked"


def test_a_patient_or_a_prescription_as_a_whole_is_about_people(pharmacy):
    _, model = pharmacy
    assert about_people(model, "patient") and about_people(model, "prescription")
    assert not about_people(model, _slug(model, "state")) and not about_people(model, _slug(model, "drug_class"))


def test_asked_to_plan_again_the_ai_is_told_the_placeholder_never_the_name(pharmacy):
    from core2.plan.planner import plan_question
    from core2.plan.values import ValueMatch

    _, model = pharmacy
    drug, sales = _slug(model, "drug_class"), _sales(model)
    question = 'fills of "statin" drugs in 2025'
    start = question.index("statin")
    match = ValueMatch(text="statin", start=start, end=start + len("statin"), attribute=drug, value="statin")
    ai = Script({"kind": "clarify", "clarify": {"about": "member", "question": "Which drug class?",
                                                "options": ["⟨v1⟩"]}},
                {"kind": "query", "intent": "value", "measures": [sales], "time": YEAR,
                 "filters": [{"field": drug, "op": "eq", "values": ["⟨v1⟩"]}]})
    outcome = plan_question(model, question, ai, today=TODAY, matches=[match], values_allowed=False)
    assert len(ai.sent) == 2 and '"⟨v1⟩" is in VALUE MATCHES' in ai.sent[1]
    assert "statin" not in ai.sent[1], ai.sent[1][-800:]
    assert outcome.plan.filters[0].values == ["statin"], "the plan still filters on it"


def test_metric_authoring_lists_no_value_of_a_column_marked_as_peoples(pharmacy):
    from core2.model import authoring
    from core2.model.schema import SemanticModel

    _, model = pharmacy
    marked = SemanticModel.model_validate(model.model_dump())
    drug = next(c for c in marked.columns.values() if c.name == "drug_class")
    drug.values_allowed = True
    assert "statin" in authoring.describe_prompt(marked) or not (drug.profile and drug.profile.top)
    drug.personal = "name"          # the admin's choice on the Knowledge base page
    assert "statin" not in authoring.describe_prompt(marked)


# ── the AI call list reads no kept prompt ───────────────────────────────────


def test_the_ai_call_list_reads_the_newest_questions_and_no_kept_prompt(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    for i in range(30):
        for part in range(2):
            store.log_llm_call(account_id=account_id, question_id=f"q{i:02d}", request_id=f"r{i}-{part}",
                               question=f"question {i}", component="core2_planner", llm_provider="anthropic",
                               llm_model="m", status="success", payload_hash="h", payload_preview_sanitized="p",
                               prompt_chars=10, error_msg="", payload_sealed="SEALED-" + "x" * 1000,
                               response_sealed="SEALED-RESPONSE")
        with store.get_db() as conn:
            conn.execute("UPDATE llm_call_log SET created_at=? WHERE question_id=?",
                         (f"2026-10-{1 + i % 28:02d} {i % 24:02d}:00:00", f"q{i:02d}"))
    groups = store.get_recent_llm_calls(account_id, limit=5)
    assert [g["question_id"] for g in groups] == ["q27", "q26", "q25", "q24", "q23"], [g["question_id"] for g in groups]
    calls = [c for g in groups for c in g["calls"]]
    assert len(calls) == 10 and all(c["payload_sealed"] for c in calls), "the page still knows the prompt was kept"
    assert not any("SEALED" in str(v) for c in calls for v in c.values()), "never the kept text itself"


# ── the name scrubber ───────────────────────────────────────────────────────


KNOWN = {"net", "amount", "customer", "type", "region", "store", "sales", "wholesale"}
MEMBERS = {"franklin", "clinton", "old town store", "northline distribution"}


@pytest.mark.parametrize("asked, gone", [
    ("refills for Smith, John", "Smith, John"),
    ("fills of John Q. Smith", "John Q. Smith"),
    ("J. Smith's claims", "J. Smith"),
    ("sales for JOHN SMITH", "JOHN SMITH"),
    ("orders of Łukasz Kowalczyk", "Łukasz Kowalczyk"),
    ("sales for\nJohn  Smith", "John  Smith"),
    ("net sales for Mary Franklin", "Mary Franklin"),
    ("Bill Clinton", "Bill Clinton"),
    ("Mary Ann Jackson refills", "Mary Ann Jackson"),
])
def test_every_form_of_a_typed_name_is_withheld(asked, gone):
    from core.masking import scrub_typed_names

    said = scrub_typed_names(asked, KNOWN, MEMBERS)
    assert gone not in said and "[PERSON]" in said, said


@pytest.mark.parametrize("asked", [
    "Net Sales Growth Rate By Store",
    "Average Basket Size by store",
    "Show Customer Lifetime Value",
    "Sales in New York vs Los Angeles",
    "fills by Ohio, Texas",
    "Old Town Store sales",
    "Northline Distribution Sales",
    "Compare Franklin and Clinton",
    "sales at the NDC level in Q1 FY2026",
])
def test_the_questions_own_words_places_and_members_stay(asked):
    from core.masking import scrub_typed_names

    assert scrub_typed_names(asked, KNOWN, MEMBERS) == asked


# ── health identifiers by name ──────────────────────────────────────────────


@pytest.mark.parametrize("name", ["MEDICARE_BENEFICIARY_IDENTIFIER", "MED_REC_NO", "MRNumber", "PAT_NO"])
def test_more_spellings_of_a_patients_numbers_are_personal(name):
    from core2.bootstrap.personal import is_health_record

    assert is_health_record(name)


def test_pt_no_is_a_patients_number_only_in_a_table_about_people():
    from core2.bootstrap.personal import is_health_record

    assert is_health_record("PT_NO", people=True) and not is_health_record("PT_NO")


@pytest.mark.parametrize("name, record", [("RXNBR", True), ("RXNO", True), ("ORIG_RX_NUMBER", True),
                                          ("PHARMACY_RX_NUMBER", True), ("RX", False), ("PRESCRIPTION_ID", False),
                                          ("RX_COUNT", False)])
def test_a_prescriptions_number_by_its_spellings(name, record):
    from core2.bootstrap.personal import is_prescription_number

    assert is_prescription_number(name) is record
