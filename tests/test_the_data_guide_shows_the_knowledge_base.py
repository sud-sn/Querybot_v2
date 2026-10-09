"""The Data guide: the admin's Knowledge base as a reader may see it, and suggested on the same way.

"What you can ask" is now the Data guide. Beside each subject's metrics, breakdowns and
dates, it shows what one row of the subject is, what it means where the admin said so, and
what a field's codes mean ("C" is Cancelled) -- the names an admin gives codes on the
Knowledge base. A reader suggests names for codes as they suggest a metric's meaning: the
suggestion waits under Requests, the admin accepts it (as sent or edited) and the names join
the ones already there.

Codes are member values: a reader who may not see values (a tenant under compliance, or
value indexing off) is not shown them. A store's own code (S01) is the store, named by the
store's name: not a code to name. Invented retail data only.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

from core2.model.requests import RequestError, proposal, reader_view
from tests.test_readers_suggest_and_admins_decide import retail, site  # noqa: F401 - the fixtures

STATUS = "order_line.status_code"
STATUS_COLUMN = "memory.main.order_lines.status_code"


def _coded(view: dict) -> dict[str, list[dict]]:
    return {b["key"]: b["codes"] for s in view["subjects"] for b in s["breakdowns"] if b.get("codes")}


def test_the_guide_shows_what_a_fields_codes_mean_and_only_fields_of_codes(retail):  # noqa: F811
    _, learned = retail
    coded = _coded(reader_view(learned.model if hasattr(learned, "model") else learned, today=dt.date(2026, 6, 15)))
    assert coded[STATUS] == [{"code": c, "name": ""} for c in ("C", "D", "O", "S")]
    assert "store.code" not in coded, "a store's code is the store, named by its name"
    assert "region.name" not in coded, "names are not codes"


def test_a_reader_who_may_not_see_values_is_not_shown_codes(retail):  # noqa: F811
    _, learned = retail
    model = learned.model if hasattr(learned, "model") else learned
    assert _coded(reader_view(model, today=dt.date(2026, 6, 15), values=False)) == {}


def _guide(site, *, indexing: bool = True, compliance: bool = False) -> str:  # noqa: F811
    """The reader's Data guide page: the workspace's value indexing on or off, under compliance or not."""
    scrub = (lambda text: text) if compliance else None
    with patch("core.value_index.value_index_enabled", return_value=indexing), \
            patch("core2.service.question_scrubber", return_value=scrub):
        return site.client.get("/portal/kb").text


def test_the_tab_is_the_data_guide_and_its_codes_are_on_the_page(site):  # noqa: F811
    page = _guide(site)
    assert "<h1>Data guide</h1>" in page and "What you can ask" not in page
    assert '<h3 class="ask-h">What the codes mean</h3>' in page
    assert '<li class="ask-item ask-coded" data-kind="attribute" data-key="order_line.status_code"' in page
    assert "<dt>C</dt>" in page and "no name yet" in page


def test_without_value_indexing_or_under_compliance_the_page_shows_no_codes(site):  # noqa: F811
    for page in (_guide(site, indexing=False), _guide(site, compliance=True)):
        assert '<h3 class="ask-h">What the codes mean</h3>' not in page and "<dt>C</dt>" not in page


def test_a_reader_names_codes_and_the_admin_accepts_them(site):  # noqa: F811
    sent = site.suggest(kind="change", target_kind="attribute", target_key=STATUS,
                        codes={"C": "Cancelled", "O": "Open"}, example="cancelled orders last month")
    assert sent["ok"], sent
    request = site.store.get_core2_request(site.account, sent["request"]["id"])
    (change,) = request["changes"]
    assert (change["object_key"], change["field"], change["value"]) == (f"column:{STATUS_COLUMN}", "value_names",
                                                                        {"C": "Cancelled", "O": "Open"})
    queue = site.client.get(f"/admin/clients/{site.account}/requests").text
    assert "C = Cancelled" in queue and "What its codes mean" in queue
    assert site.accept(request["id"])["ok"]
    assert site.model().columns[STATUS_COLUMN].value_names == {"C": "Cancelled", "O": "Open"}


def test_named_codes_join_the_names_there_and_an_admin_can_edit_them_first(site):  # noqa: F811
    first = site.suggest(kind="change", target_kind="attribute", target_key=STATUS, codes={"C": "Cancelled"})
    assert site.accept(first["request"]["id"])["ok"]
    second = site.suggest(kind="change", target_kind="attribute", target_key=STATUS, codes={"S": "Shiped"})
    assert site.accept(second["request"]["id"], edits={"0": "S = Shipped\nD = Delivered"})["ok"]
    assert site.model().columns[STATUS_COLUMN].value_names == {"C": "Cancelled", "S": "Shipped", "D": "Delivered"}
    page = _guide(site)
    assert "<dt>C</dt><dd>Cancelled</dd>" in page and "<dt>S</dt><dd>Shipped</dd>" in page


def test_codes_a_field_does_not_have_or_named_alike_are_refused(retail):  # noqa: F811
    _, learned = retail
    model = learned.model if hasattr(learned, "model") else learned
    for codes, said in (({"X": "Lost"}, "X is not one of the codes"),
                        ({"C": "Closed", "O": "closed"}, "Two codes would have the same name"),
                        ({"C": "x" * 61}, "under 60 characters")):
        try:
            proposal(model, kind="attribute", key=STATUS, codes=codes)
        except RequestError as exc:
            assert said in str(exc), exc
        else:
            raise AssertionError(f"{codes} was accepted")
    try:
        proposal(model, kind="attribute", key="region.name", codes={"North": "N"})
    except RequestError as exc:
        assert "has no codes to name" in str(exc)
    else:
        raise AssertionError("a field of names took code names")


def test_an_admins_edit_naming_a_code_the_field_has_not_is_refused(site):  # noqa: F811
    sent = site.suggest(kind="change", target_kind="attribute", target_key=STATUS, codes={"O": "Open"})
    refused = site.accept(sent["request"]["id"], edits={"0": "Z = Zebra"})
    assert not refused["ok"] and "Z is not one" in refused["error"]
