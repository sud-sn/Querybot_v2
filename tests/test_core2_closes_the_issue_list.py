"""Each of the 43 issues found in today's pipeline is checked against the new core.

evals/core2/issues.yaml maps every issue to a golden question tagged with it, a
test, or the property of the design that leaves nothing to test (one, H5, is left
as it is, with its reason). These keep that map honest: every issue is there,
every test it names exists, every tag has its golden questions. The two issues
no other test covered are checked here: names with symbols and everyday words
(C3, C7), and a metric that reads another table (D4).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import yaml

from core2.model.imports import Legacy, decisions
from core2.plan.values import MemberIndex
from evals.core2 import domains
from evals.core2.compile_eval import golden, learn

ISSUES = yaml.safe_load((Path(__file__).resolve().parents[1] / "evals" / "core2" / "issues.yaml").read_text())["issues"]
IDS = [f"{letter}{n}" for letter, count in zip("ABCDEFGH", (9, 4, 7, 6, 4, 4, 4, 5)) for n in range(1, count + 1)]


def test_every_issue_is_on_the_list_with_a_check():
    assert sorted(ISSUES) == sorted(IDS) and len(IDS) == 43
    for issue, entry in ISSUES.items():
        assert entry["checks"], issue
        assert all(re.match(r"(golden|test|design|open):", c) for c in entry["checks"]), issue
        assert not [c for c in entry["checks"] if c.startswith("open:")] or issue == "H5", issue


@pytest.mark.parametrize("issue", IDS)
def test_what_an_issue_names_exists(issue):
    tags = {t for name in domains.available() if golden(name) for c in golden(name)["questions"]
            for t in c.get("tags", [])}
    for check in ISSUES[issue]["checks"]:
        kind, _, what = check.partition(":")
        if kind == "golden":
            assert what in tags, f"{issue}: no golden question is tagged {what}"
        elif kind == "test":
            module, _, name = what.partition("::")
            owner = importlib.import_module(module)
            for part in name.split("."):
                owner = getattr(owner, part, None)
                assert owner is not None, f"{issue}: {what} does not exist"


def test_names_with_symbols_and_everyday_words():
    index = MemberIndex()
    index.add("item.name", ["Hex Bolt #27", "O'Brien Fittings", "Smith & Sons", "Right Angle Supply", "Actual Tools"])
    found = [(m.text, m.value) for m in index.match("monthly revenue for hex bolt #27 and o’brien fittings")]
    assert found == [("hex bolt #27", "Hex Bolt #27"), ("o’brien fittings", "O'Brien Fittings")]
    assert [m.value for m in index.match("sales for Smith & Sons")] == ["Smith & Sons"]
    # Everyday words are not names: only a whole name in the question is one.
    assert index.match("is revenue trending right now, actually? currently right") == []
    assert [m.value for m in index.match("revenue at right angle supply")] == ["Right Angle Supply"]


def test_a_metric_reading_another_table_is_reported_with_its_reason():
    _, model = learn(domains.build("retail"), "descriptive")
    report = decisions(model, Legacy(metrics=[{
        "id": 1, "name": "Revenue per customer", "is_active": 1, "metric_status": "validated",
        "base_table": "MAIN.ORDER_LINES", "result_format": "currency",
        "sql_template": "SUM(o.NET_AMOUNT) / COUNT(DISTINCT c.CUSTOMER_ID)"}]))
    assert not [d for d in report.decisions if d.field == "define"]
    assert any("Revenue per customer" in m and "another table" in m for m in report.missed), report.missed
