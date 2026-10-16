"""The sample workbook: Solviz tried with no database, on a retail chain's data kept as an Excel file.

An admin picks "Use the sample workbook" on the Databases page (or downloads it, to put in SharePoint or OneDrive and
try a link). Its six sheets read as tables, as a team's own would: the export's title row above the order lines and
the "Grand total" row under the targets are left out. Learn finds how they link, and a business ask on it gives the
figures SQL written by hand gives on the same sheets.

Invented data only (tools/sample_workbook.py).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from unittest.mock import patch

import pytest

from core import workbook_source as W
from tests.test_a_workbook_answers_like_a_database import _request

SHEETS = {"Order lines": "order_lines", "Products": "products", "Stores": "stores", "Customers": "customers",
          "Returns": "returns", "Monthly targets": "monthly_targets"}


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUERYBOT_WORKBOOKS_DIR", str(tmp_path_factory.mktemp("workbooks")))
        loaded = W.save_sample()
        yield loaded, {"source": W.UPLOAD, "workbook": loaded.sha256, "file_name": loaded.file_name}


def test_the_sample_reads_as_six_tables_with_the_title_and_totals_left_out(sample):
    loaded, _ = sample
    tables = {t.sheet: t for t in loaded.tables}
    assert {s: t.name for s, t in tables.items()} == SHEETS and not loaded.skipped
    assert tables["Order lines"].rows > 15000 and tables["Customers"].rows == 240
    assert tables["Order lines"].left_out == ["2 rows above the header"]
    assert any("totals row" in why for why in tables["Monthly targets"].left_out)
    kinds = {c.name: c.type for c in tables["Order lines"].columns}
    assert (kinds["order_date"], kinds["net_amount"], kinds["store_id"]) == ("DATE", "DOUBLE", "VARCHAR")


def test_the_sample_is_the_one_its_script_builds():
    from tools.sample_workbook import build

    shipped, _ = W.read_tables(W.SAMPLE.read_bytes())
    rebuilt, _ = W.read_tables(build())
    assert [(t.name, rows) for t, rows in shipped] == [(t.name, rows) for t, rows in rebuilt]


@pytest.fixture(scope="module")
def learned(sample, tmp_path_factory):
    from core.schema import discover_and_write
    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_schema_json
    from core2.warehouse.querybot import QueryBotWarehouse

    _loaded, creds = sample
    folder = tmp_path_factory.mktemp("sample_schema")
    discover_and_write(creds, "excel", str(folder))
    inventory = from_schema_json(json.loads((folder / "_schema.json").read_text()), "excel")
    with QueryBotWarehouse("excel", creds) as warehouse:
        model = build_model(warehouse, inventory, client_id="sample", options=BuildOptions(workers=2))
    return creds, model


def test_learn_links_the_sheets_and_finds_the_money(learned):
    _creds, model = learned
    links = {(model.tables[j.from_table].name, model.tables[j.to_table].name) for j in model.joins.values()}
    for link in [("order_lines", "products"), ("order_lines", "stores"), ("order_lines", "customers"),
                 ("returns", "products"), ("monthly_targets", "stores")]:
        assert link in links, links
    money = {m.slug for m in model.measures.values() if m.format == "currency" and not m.hidden}
    assert "net_amount" in money and "refund_amount" in money, money


def test_a_comparison_asked_of_the_sample_is_the_sheets_own_figures(learned):
    from core.schema import run_query
    from core2.plan.values import MemberIndex
    from core2.service import Services, Session, answer_question
    from core2.warehouse.querybot import QueryBotWarehouse

    creds, model = learned
    plans = [{"kind": "query", "intent": "compare", "measures": ["net_amount"]}]
    with QueryBotWarehouse("excel", creds) as warehouse:
        services = Services(model=model, warehouse=warehouse, complete=lambda s, t: json.dumps(plans.pop(0)),
                            index=MemberIndex(), today=dt.date(2026, 10, 16))
        got = answer_question("What is my revenue comparison?", services, Session())
    # The data's last day is 30 Sep 2026: January to September, each month against the same month of 2025.
    want = run_query(creds, "excel", """
        SELECT month(order_date) AS m,
          SUM(CASE WHEN order_date BETWEEN DATE '2026-01-01' AND DATE '2026-09-30' THEN net_amount END) AS now,
          SUM(CASE WHEN order_date BETWEEN DATE '2025-01-01' AND DATE '2025-09-30' THEN net_amount END) AS before
        FROM workbook.main.order_lines
        WHERE (order_date BETWEEN DATE '2026-01-01' AND DATE '2026-09-30'
               OR order_date BETWEEN DATE '2025-01-01' AND DATE '2025-09-30')
          AND (status <> 'Cancelled' OR status IS NULL)
        GROUP BY 1 ORDER BY 1""")
    rows = got["export_rows"]
    assert len(rows) == len(want) == 9
    for row, hand in zip(rows, want):
        assert dt.date.fromisoformat(str(row["period"])[:10]).month == hand["m"]
        assert float(row["net_amount"]) == pytest.approx(hand["now"], rel=1e-9)
        assert float(row["net_amount_prior"]) == pytest.approx(hand["before"], rel=1e-9)


def test_the_databases_page_offers_the_sample_and_its_download(sample, tmp_path, monkeypatch):
    from admin import routes

    monkeypatch.setenv("QUERYBOT_WORKBOOKS_DIR", str(tmp_path / "workbooks"))
    with patch.object(routes, "_is_auth", return_value=True):
        read = json.loads(asyncio.run(routes.database_workbook_sample(
            _request("/admin/databases/workbook/sample", {}))).body)
        file = asyncio.run(routes.database_workbook_sample_file(
            _request("/admin/databases/workbook/sample.xlsx", method="GET")))
    assert read["status"] == "ok" and read["file_name"] == W.SAMPLE_NAME and len(read["workbook"]) == 64
    assert [t["sheet"] for t in read["tables"]] == list(SHEETS)
    assert file.path == str(W.SAMPLE) and "spreadsheetml" in file.media_type
    with patch.object(routes, "_is_auth", return_value=False):
        refused = asyncio.run(routes.database_workbook_sample(_request("/admin/databases/workbook/sample", {})))
    assert refused.status_code == 401
