"""Tables keyed by text identifiers (UUIDs) are linked, by name or by their values alone.

On a public health-records sample with no declared keys, Learn found none of its 40 links: every table is keyed
by a 36-character UUID in a column called Id, and a key of text longer than 24 characters was taken for a name
or a description, never a key. Text whose values are all one length, and long, is a machine's identifier: it is
a key a column may point at ("PATIENT" -> patients.Id, "PROVIDERID" -> providers.Id), and two such columns of
the same length are linked by their values alone (an appointment column holding the encounters' Id), since
identifiers of that kind never meet by chance. Names and descriptions, which vary in length, still are not keys.

Invented data only.
"""

from __future__ import annotations

import uuid

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, learn
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


def _ids(n: int, seed: int) -> list[str]:
    return [str(uuid.UUID(int=(seed << 64) + i * 7919 + 1)) for i in range(n)]


@pytest.fixture(scope="module")
def learned():
    con = duckdb.connect()
    patients, providers, visits = _ids(120, 1), _ids(30, 2), _ids(2000, 3)
    con.execute("CREATE TABLE patients (Id VARCHAR, GENDER VARCHAR, NOTE VARCHAR)")
    con.executemany("INSERT INTO patients VALUES (?, ?, ?)",
                    [(p, "FM"[i % 2], f"Patient note {'x' * (i % 40)}") for i, p in enumerate(patients)])
    con.execute("CREATE TABLE providers (Id VARCHAR, SPECIALITY VARCHAR)")
    con.executemany("INSERT INTO providers VALUES (?, ?)", [(p, ["GP", "ENT", "CARDIO"][i % 3])
                                                             for i, p in enumerate(providers)])
    con.execute("CREATE TABLE encounters (Id VARCHAR, PATIENT VARCHAR, PROVIDER VARCHAR, COST DOUBLE)")
    con.executemany("INSERT INTO encounters VALUES (?, ?, ?, ?)",
                    [(v, patients[(i * 31) % 120], providers[(i * 7) % 30], 50.0 + i % 300)
                     for i, v in enumerate(visits)])
    con.execute("CREATE TABLE claims (Id VARCHAR, PATIENTID VARCHAR, APPOINTMENTID VARCHAR, AMOUNT DOUBLE)")
    con.executemany("INSERT INTO claims VALUES (?, ?, ?, ?)",
                    [(c, patients[(i * 31) % 120], visits[i], 20.0 + i % 90)
                     for i, c in enumerate(_ids(1800, 4))])
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    found = learn(warehouse, inventory, BuildOptions(workers=1))
    name_of = {key: t.name for key, t in inventory.tables.items()}
    return {(name_of[j.from_table], j.from_column): (name_of[j.to_table], j.to_column, j.trust)
            for j in found.joins if j.trust != "rejected"}


@pytest.mark.parametrize("table, column, target", [
    ("encounters", "PATIENT", "patients"),         # named for the table it points at
    ("encounters", "PROVIDER", "providers"),
    ("claims", "PATIENTID", "patients"),
    ("claims", "APPOINTMENTID", "encounters"),      # named for nothing: its values are the encounters' Id
])
def test_a_column_holding_another_tables_identifiers_links_to_it(learned, table, column, target):
    assert learned.get((table, column)) == (target, "Id", "verified"), learned


def test_text_of_varying_length_is_no_key(learned):
    assert not any(to == "patients" and key == "NOTE" for to, key, _ in learned.values())
