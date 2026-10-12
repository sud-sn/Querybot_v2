"""A share asked of a part is the part over the whole period, not over itself.

"What share of 2025 net sales came in on weekends?" and "what share of 2025 receipts came from Mexican
suppliers?" were answered with the part and a share of 100%: a share was worked out over the answer's own rows,
and the filter that keeps the part left nothing else. Where the answer is one figure kept to one member filter
(or to members of the field it is grouped by), the whole is now asked for once more, without that filter, and
the share is the part over it; the sentence says the share and the whole. A breakdown keeps its shares of the
shown total, and a question with several conditions keeps them all.

The synthetic retail warehouse (invented data); figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize

Y25 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
SALES = ("order_lines o JOIN calendar c ON c.date_key = o.order_date_key JOIN stores s ON s.store_id = o.store_id "
         "JOIN regions r ON r.region_id = s.region_id")


@pytest.fixture(scope="module")
def retail():
    built = materialize(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-share",
                        options=BuildOptions(workers=1))
    return built.con, model


def _ask(con, model, plan):
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 14))
    payload = answer_question("q", services, Session())
    assert (payload.get("data") or {}).get("rows"), payload["answer"]["headline"]
    return payload


def _region(model):
    return next(a.slug for a in model.attributes.values() if a.column.endswith("regions.region_name"))


def test_the_weekends_share_is_of_the_whole_year(retail):
    con, model = retail
    payload = _ask(con, model, {"intent": "share", "measures": ["net_amount"], "time": {"window": Y25},
                                "filters": [{"field": "time:is_weekend", "op": "eq", "values": [True]}]})
    want = con.execute(f"SELECT SUM(CASE WHEN dayofweek(c.full_date) IN (0, 6) THEN o.net_amount END) / "
                       f"SUM(o.net_amount) FROM {SALES} WHERE year(c.full_date) = 2025").fetchone()[0]
    [row] = payload["data"]["rows"]
    assert row["net_amount_share"] == pytest.approx(float(want) * 100, abs=0.01)
    assert f"{float(want) * 100:.1f}% of all net amount" in payload["answer"]["headline"]


def test_one_regions_share_grouped_by_region_is_of_every_region(retail):
    con, model = retail
    region = _region(model)
    payload = _ask(con, model, {"intent": "share", "measures": ["net_amount"], "group_by": [region],
                                "time": {"window": Y25},
                                "filters": [{"field": region, "op": "eq", "values": ["North"]}]})
    want = con.execute(f"SELECT SUM(CASE WHEN r.region_name = 'North' THEN o.net_amount END) / SUM(o.net_amount) "
                       f"FROM {SALES} WHERE year(c.full_date) = 2025").fetchone()[0]
    [row] = payload["data"]["rows"]
    assert row["net_amount_share"] == pytest.approx(float(want) * 100, abs=0.01)


def test_a_breakdown_keeps_its_shares_of_the_shown_total(retail):
    con, model = retail
    payload = _ask(con, model, {"intent": "share", "measures": ["net_amount"], "group_by": [_region(model)],
                                "time": {"window": Y25}})
    shares = [r["net_amount_share"] for r in payload["data"]["rows"]]
    assert len(shares) > 1 and sum(shares) == pytest.approx(100, abs=0.05)
