"""People's data in a workspace under compliance: shown masked, or as stored to a reader cleared to see it.

Learn reads a person's name and a person's details (an email, a phone) by names and by values. They never
reach the AI. In a workspace under compliance, answers may show them: the governed warehouse masks what
comes from them for a reader who has not signed the confidentiality attestation (a name becomes a stable
alias, "Patient K-3F2", and a detail a stable token, so each person stays apart), and shows them as
stored to one who has, writing each such release to the decision log; when it cannot be written, the
values stay masked. Counts and sums are never masked, and a column the workspace's own masking covers is
not masked twice. Learn proposes what it found on the Compliance page, as classifications to review.

An answer that names people gets no summary written by the AI, in any workspace.

Invented people only.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import types

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import personal_columns
from core2.warehouse.governed import GovernedWarehouse
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)



def _store():
    """The store module the code under test resolves now: other test modules delete it from sys.modules and
    import it again, so the one this module imported may no longer be it."""
    return importlib.import_module("store")

def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, first_name VARCHAR, last_name VARCHAR, "
                "email VARCHAR, birth_date DATE, city VARCHAR)")
    con.execute("INSERT INTO patients SELECT i, ['Ana', 'Ben', 'Cleo', 'Dev', 'Eli', 'Fay'][i % 6 + 1], "
                "['Stone', 'Rivers', 'Hill', 'Moss'][i % 4 + 1] || i, 'person' || i || '@example.org', "
                "DATE '1950-03-07' + (i * 397)::INTEGER, "
                "CASE i % 3 WHEN 0 THEN 'Lakemont' WHEN 1 THEN 'Riverton' ELSE 'Pinecrest' END FROM range(1, 41) t(i)")
    con.execute("CREATE TABLE fills (fill_id INTEGER PRIMARY KEY, patient_id INTEGER, fill_date DATE, "
                "amount DECIMAL(10,2))")
    con.execute("INSERT INTO fills SELECT i, i % 40 + 1, DATE '2026-01-01' + (i % 150)::INTEGER, (i % 17) * 3.5 + 10 "
                "FROM range(1, 801) t(i)")
    return con


@pytest.fixture(scope="module")
def learned():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="people", options=BuildOptions(workers=1))
    return con, model


def _slug(model, table: str, column: str) -> str:
    return next(s for s, a in model.attributes.items()
                if model.columns[a.column].name == column and model.tables[model.columns[a.column].table].name == table)


def _name_slug(model) -> str:
    return next(s for s, a in model.attributes.items() if model.columns[a.column].parts)


def _amount(model) -> str:
    return next(m.slug for m in model.measures.values()
                if getattr(m.expr, "column", None) and model.columns[m.expr.column].name == "amount")


def test_learn_reads_names_and_details(learned):
    _, model = learned
    by_name = {c.name: c for c in model.columns.values()}
    assert by_name["first_name"].personal == by_name["last_name"].personal == "name"
    assert by_name["first_name+last_name"].personal == "name"
    assert (by_name["email"].personal, by_name["email"].sensitivity) == ("detail", "pii")
    assert by_name["city"].personal == "none"
    # Unique last names and emails are no code to show beside each patient's name.
    patients = next(e for e in model.entities.values() if model.tables[e.table].name == "patients")
    assert patients.code_column is None or model.columns[patients.code_column].personal == "none"


def test_a_detail_is_refused_unless_the_workspace_shows_it_masked(learned):
    _, model = learned
    plan = Plan(intent="breakdown", measures=[_amount(model)], group_by=[_slug(model, "patients", "email")])
    with pytest.raises(ResolveError):
        resolve(plan, model, Context(today=TODAY))
    assert resolve(plan, model, Context(today=TODAY, personal_shown=True)).groups
    marked = model.model_copy(deep=True)
    email = next(c for c in marked.columns.values() if c.name == "email")
    email.sensitivity = "confidential"          # an admin marked it sensitive: never shown, whatever the workspace
    with pytest.raises(ResolveError):
        resolve(plan, marked, Context(today=TODAY, personal_shown=True))


class _Governed:
    """The governed executor, faked at its boundary: the SQL runs on DuckDB and is analysed as for real."""

    def __init__(self, con, masking=None, rewrite=None):
        self.con, self.masking, self.rewrite = con, masking or {}, rewrite

    def __call__(self, credentials, db_type, sql, **_):
        from core.compliance.sql_guard import analyze_sql

        cursor = self.con.execute(sql)
        names = [d[0] for d in cursor.description]
        rows = [dict(zip(names, r)) for r in cursor.fetchall()]
        if self.rewrite:
            rows = [self.rewrite(r) for r in rows]
        return types.SimpleNamespace(rows=rows, analysis=analyze_sql(sql, db_type), truncated=False,
                                     decision=types.SimpleNamespace(masking=dict(self.masking)))


@pytest.fixture
def governed(learned, monkeypatch):
    con, model = learned
    released: list[dict] = []
    state = {"signed": False, "log_fails": False, "fake": _Governed(con)}

    def log(**kwargs):
        if state["log_fails"]:
            raise RuntimeError("the decision log is read-only")
        released.append(kwargs)
        return "audit-1"

    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query",
                        lambda *a, **k: state["fake"](*a, **k))
    monkeypatch.setattr("core.compliance.policy_engine.resolve_context", lambda *a, **k: types.SimpleNamespace(
        user_id="7", purpose_id="", channel="portal", policy_version=1))
    monkeypatch.setattr(_store(), "user_attestation_valid", lambda account, user: state["signed"])
    monkeypatch.setattr(_store(), "user_attestation_scope", lambda account, user: {"*"} if state["signed"] else None)
    monkeypatch.setattr(_store(), "log_policy_decision", log)
    warehouse = GovernedWarehouse("acct-people", {"id": 7}, {"db_type": "duckdb", "credentials": {}},
                                  known_tables=set(), personal=personal_columns(model))

    def ask(*group_by: str, measures: list[str] | None = None):
        plan = Plan(intent="breakdown", measures=measures or [_amount(model)], group_by=list(group_by))
        logical = resolve(plan, model, Context(today=TODAY, personal_shown=True))
        compiled = compile_query(logical, model, "duckdb")
        result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
        return result, con.execute(compiled.sql).fetchall()

    return model, ask, state, released


def test_a_reader_not_cleared_sees_stable_aliases_and_tokens(governed):
    model, ask, _, released = governed
    result, stored = ask(_name_slug(model), _slug(model, "patients", "email"))
    shown = [dict(zip(result.columns, r)) for r in result.rows]
    name_column = next(c for c in result.columns if c.endswith("name"))
    email_column = next(c for c in result.columns if "email" in c)
    names = [r[name_column] for r in shown]
    emails = [r[email_column] for r in shown]
    assert all(str(n).startswith("Patient ") for n in names), names[:3]
    assert all(str(e).startswith("TKN-") for e in emails), emails[:3]
    assert not any("@" in str(v) or "Stone" in str(v) for r in shown for v in r.values())
    assert len(set(names)) == len({r[result.columns.index(name_column)] for r in stored}), \
        "each person keeps an alias of their own"
    assert sorted(float(r[-1]) for r in result.rows) == sorted(float(r[-1]) for r in stored), "amounts untouched"
    again, _ = ask(_name_slug(model), _slug(model, "patients", "email"))
    assert again.rows == result.rows, "the same person, the same alias"
    assert released == []


def test_a_cleared_reader_sees_them_as_stored_and_the_release_is_logged(governed):
    model, ask, state, released = governed
    state["signed"] = True
    result, stored = ask(_name_slug(model))
    assert sorted(map(str, (r[0] for r in result.rows))) == sorted(map(str, (r[0] for r in stored)))
    assert released and released[0]["reason_code"] == "attested_unmasked_release"
    assert released[0]["user_id"] == "7"
    assert any("FIRST_NAME" in r for r in released[0]["resources"])


def test_a_release_that_cannot_be_logged_stays_masked(governed):
    model, ask, state, _ = governed
    state["signed"], state["log_fails"] = True, True
    result, _ = ask(_name_slug(model))
    assert all(str(r[0]).startswith("Patient ") for r in result.rows)


def test_counts_of_people_are_never_masked(governed):
    model, ask, _, _ = governed
    patients = next(m.slug for m in model.measures.values() if m.expr.agg in ("count", "count_distinct")
                    and model.tables[m.table].name == "patients")
    result, stored = ask(_slug(model, "patients", "city"), measures=[patients])
    assert sorted(result.rows) == sorted(stored)


def test_what_the_workspace_masks_already_is_not_masked_twice(governed, learned):
    model, ask, state, _ = governed
    con, _ = learned

    def redact(row):
        return {k: ("[REDACTED]" if "@" in str(v) else v) for k, v in row.items()}

    state["fake"] = _Governed(con, masking={"MAIN.PATIENTS.EMAIL": "redact"}, rewrite=redact)
    result, _ = ask(_slug(model, "patients", "email"))
    assert {r[0] for r in result.rows} == {"[REDACTED]"}


def test_a_workspace_not_under_compliance_masks_nothing_here(learned, monkeypatch):
    con, model = learned
    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", _Governed(con))
    monkeypatch.setattr("core.compliance.policy_engine.resolve_context", lambda *a, **k: types.SimpleNamespace(
        user_id="7", purpose_id="", channel="portal", policy_version=1))
    warehouse = GovernedWarehouse("acct-people", {"id": 7}, {"db_type": "duckdb", "credentials": {}},
                                  known_tables=set())
    plan = Plan(intent="breakdown", measures=[_amount(model)], group_by=[_name_slug(model)])
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb")
    assert sorted(warehouse.query(compiled.sql).rows) == sorted(con.execute(compiled.sql).fetchall())


def test_the_catalog_says_how_a_detail_is_shown(learned):
    from core2.plan.catalog import catalog_text

    _, model = learned
    assert "never shown or filtered on" in catalog_text(model)
    shown = catalog_text(model, personal_shown=True)
    assert "a personal detail: shown masked unless the reader is cleared" in shown
    assert "example.org" not in shown and "Stone" not in shown


def test_an_answer_naming_people_gets_no_summary_from_the_ai(learned):
    from core2.answer.summary import eligible
    from core2.plan.values import build_index
    from core2.service import Services, Session, answer_question

    con, model = learned
    plans = [{"kind": "query", "intent": "breakdown", "measures": [_amount(model)], "group_by": [_name_slug(model)]},
             {"kind": "query", "intent": "breakdown", "measures": [_amount(model)],
              "group_by": [_slug(model, "patients", "city")]}]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(plans.pop(0)),
                        index=build_index(model, lambda slug: []), today=TODAY)
    session = Session()
    by_person = answer_question("Amount by patient", services, session)
    by_city = answer_question("Amount by city", services, session)
    assert by_person.get("personal") is True and not eligible(by_person, values_allowed=True)
    assert not by_city.get("personal") and eligible(by_city, values_allowed=True)


def test_learn_proposes_what_it_found_on_the_compliance_page(learned, monkeypatch):
    from core2.bootstrap import service as bootstrap

    _, model = learned
    saved: list[dict] = []
    existing = {"MEMORY.MAIN.PATIENTS.LAST_NAME": {"reviewed": 1, "tags": []},          # an admin's word stands
                "MEMORY.MAIN.PATIENTS.FIRST_NAME": {"reviewed": 0, "tags": ["PII"]}}    # already known
    monkeypatch.setattr("core2.service.question_scrubber", lambda account: (lambda text: text))
    monkeypatch.setattr(_store(), "get_compliance_profile", lambda account: {"industry": "healthcare_pharmacy"})
    monkeypatch.setattr(_store(), "get_classification_map", lambda account: existing)
    monkeypatch.setattr(_store(), "save_classification", lambda account, fqn, column, **kw: saved.append(
        {"table": fqn, "column": column, **kw}) or len(saved))
    assert bootstrap.propose_classifications("acct-people", model) == 2
    by_column = {s["column"]: s for s in saved}
    assert set(by_column) == {"email", "birth_date"}, "the last name is an admin's, the first name already known"
    email = by_column["email"]
    assert email["table"].endswith("patients")
    assert "PII" in email["tags"] and email["reviewed"] is False and email["source"] == "learn"
    assert email["mask_strategy"] == "tokenize"

    monkeypatch.setattr("core2.service.question_scrubber", lambda account: None)
    saved.clear()
    assert bootstrap.propose_classifications("acct-people", model) == 0 and not saved


def test_the_ai_naming_columns_sees_shapes_and_made_up_examples_never_values(learned):
    from core2.bootstrap.labels import table_brief

    _, model = learned
    patients = next(k for k, t in model.tables.items() if t.name == "patients")
    brief = table_brief(model, patients)
    by_column = {c["column"]: c for c in brief["columns"]}
    assert by_column["email"]["holds"] == "email addresses"
    assert by_column["first_name"]["holds"] == "people's names"
    assert "range" not in by_column["birth_date"] and by_column["birth_date"]["holds"].startswith("people's dates")
    text = json.dumps(brief)
    assert "@example.org" not in text and "1950-" not in text and "Stone" not in text
    birth = next(c for c in model.columns.values() if c.name == "birth_date")
    assert birth.profile.min is None and birth.profile.max is None, "two people's birth dates are not kept"


def test_an_admin_corrects_what_learn_read_as_peoples_data(learned):
    from core2.model.knowledge import KnowledgeError, field_changes
    from core2.model.overrides import ALLOWED

    _, model = learned
    city = next(c for c in model.columns.values() if c.name == "city")
    assert field_changes(model, city.key, {"personal": "name"}) == {"personal": "name"}
    with pytest.raises(KnowledgeError):
        field_changes(model, city.key, {"personal": "secret"})
    assert "personal" in ALLOWED["column"]
