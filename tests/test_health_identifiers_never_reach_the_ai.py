"""A patient's record numbers are read as personal data, and a prescription's number never reaches the AI.

Learn read a national ID number as a personal detail only when its column said SSN or passport. A pharmacy's
patient number, medical record number (MRN) and Medicare beneficiary ID -- identifiers HIPAA's Safe Harbor lists --
were ordinary columns: their values could be listed to the AI, put in the member index, and shown unmasked. And a
prescription (Rx) number, which Safe Harbor also counts as an identifier, is the key that links a fill to its
prescription and its claims. Now:

* a medical record number, a patient's own number and a health-plan beneficiary number are personal details, as an
  SSN is: never listed, indexed or sent; masked where the workspace is under compliance;
* a prescription number is a record number: shown and linked as any key, but its values are never listed, indexed or
  sent to the AI, and where the workspace is under compliance Learn proposes it as PHI, so readers see it masked;
* an employee number or a prescriber's NPI stays as it was: neither is a patient's.

An invented pharmacy in DuckDB; a throwaway database for the compliance proposal.
"""

from __future__ import annotations

import json
import os

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse

RX = "RX-778812"
MRN = "MRN-55120"


@pytest.fixture(scope="module")
def pharmacy():
    con = duckdb.connect()
    con.execute("CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, patient_number VARCHAR, mrn VARCHAR, "
                "first_name VARCHAR, last_name VARCHAR, state VARCHAR)")
    con.execute("INSERT INTO patients SELECT i, 'P' || (10000 + i), 'MRN-' || (55000 + i), 'First' || i, "
                "'Last' || i, CASE i % 3 WHEN 0 THEN 'OH' WHEN 1 THEN 'TX' ELSE 'CA' END FROM range(1, 121) t(i)")
    con.execute("CREATE TABLE prescriptions (rx_number VARCHAR PRIMARY KEY, patient_id INTEGER, prescriber_npi VARCHAR, "
                "written_date DATE, quantity INTEGER)")
    con.execute("INSERT INTO prescriptions SELECT 'RX-' || (778700 + i), 1 + i % 120, '1' || (234567000 + i % 9), "
                "DATE '2025-01-01' + CAST(i % 300 AS INTEGER), 10 + i % 50 FROM range(1, 401) t(i)")
    con.execute("CREATE TABLE fills (fill_id INTEGER PRIMARY KEY, rx_number VARCHAR, fill_date DATE, "
                "price_charged DECIMAL(10, 2))")
    con.execute("INSERT INTO fills SELECT i, 'RX-' || (778700 + 1 + i % 400), DATE '2025-01-05' + CAST(i % 300 AS INTEGER), "
                "round(20 + (i * 7) % 200, 2) FROM range(1, 1201) t(i)")
    con.execute("CREATE TABLE claims (claim_id INTEGER PRIMARY KEY, rx_number VARCHAR, medicare_beneficiary_id VARCHAR, "
                "paid_amount DECIMAL(10, 2))")
    con.execute("INSERT INTO claims SELECT i, 'RX-' || (778700 + 1 + i % 400), '1EG4-TE5-MK' || (10 + i % 80), "
                "round(15 + (i * 3) % 150, 2) FROM range(1, 801) t(i)")
    con.execute("CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, employee_number VARCHAR, "
                "first_name VARCHAR, last_name VARCHAR)")
    con.execute("INSERT INTO employees SELECT i, 'E' || (500 + i), 'Emp' || i, 'Loyee' || i FROM range(1, 31) t(i)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="rx", options=BuildOptions(workers=1))
    return con, model


def _column(model, table: str, name: str):
    return next(c for c in model.columns.values() if model.tables[c.table].name == table and c.name == name)


def test_a_patients_record_numbers_are_personal_details(pharmacy):
    _, model = pharmacy
    for table, name in (("patients", "patient_number"), ("patients", "mrn"), ("claims", "medicare_beneficiary_id")):
        column = _column(model, table, name)
        assert (column.personal, column.sensitivity, column.values_allowed) == ("detail", "pii", False), (table, name)


def test_a_prescription_number_is_shown_and_linked_but_never_listed(pharmacy):
    _, model = pharmacy
    for table in ("prescriptions", "fills", "claims"):
        column = _column(model, table, "rx_number")
        assert column.record_number and not column.values_allowed, table
        assert (column.personal, column.sensitivity) == ("none", "none"), "still shown and grouped by"
        assert not (column.profile and (column.profile.top or column.profile.min)), "no value kept for the AI"
    assert any(j.from_columns and model.columns[j.from_columns[0]].name == "rx_number" for j in model.joins.values())


def test_neither_kind_of_number_reaches_the_ai(pharmacy):
    from core2.bootstrap.labels import table_brief
    from core2.plan.catalog import catalog_text
    from core2.plan.planner import stable_prompt

    _, model = pharmacy
    sent = stable_prompt(model) + catalog_text(model) + json.dumps(
        [table_brief(model, k) for k in model.tables], default=str)
    assert "RX-7787" not in sent and "MRN-55" not in sent and "1EG4-TE5" not in sent


def test_an_employee_number_and_a_prescribers_npi_are_not_a_patients(pharmacy):
    _, model = pharmacy
    for table, name in (("employees", "employee_number"), ("prescriptions", "prescriber_npi")):
        column = _column(model, table, name)
        assert column.personal == "none" and not column.record_number, (table, name)


def test_under_compliance_a_prescription_number_is_proposed_as_phi(pharmacy, tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from core2.bootstrap.service import propose_classifications

    store.init_db()
    _, model = pharmacy
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Pharmacy")
    store.save_compliance_profile(account_id, mode="regulated", industry="healthcare_pharmacy")
    propose_classifications(account_id, model)
    proposed = {k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1]: v
                for k, v in store.get_classification_map(account_id).items()}
    rx = proposed.get("FILLS.RX_NUMBER") or proposed.get("fills.rx_number")
    assert rx and "PHI" in (rx.get("tags") or []), sorted(proposed)
