"""A link an admin changes is measured again against the data, and keeps the cardinality the data shows.

A relationship's cardinality says how many rows of the table it reaches meet each row it starts from; a question
follows only links that meet one (many to one, one to one), so nothing is counted twice. Two ways the relationships
page broke that:

* a changed condition was saved without measuring the link again: a link to a table holding a member once per
  version is safe with its "current row" condition, and with that condition taken away every version would be
  counted; a condition that makes a repeating link safe left it refused, as if it still repeated;
* the page sent "many to one" for every link it saved, so a link the data had shown to be many to many was saved as
  many to one by an edit of its name or its kept rows, and followed.

Now a change to a link's conditions is measured before it is saved (and refused when it cannot be), the measured
rows are kept with the link, and a link the data showed repeating is called many to one only when the data now
shows one row each. The real store and page; the measurement is stood in for where the test is about its result.
"""

from __future__ import annotations

import pytest

from tests.test_core2_relationships_are_edited_on_their_page import (  # noqa: F401 - fixtures
    _column, _link, _model, page, retail)


def _body(link, conditions=(), cardinality=None) -> dict:
    body = {"key": link.key, "from": link.from_table, "to": link.to_table,
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)],
            "conditions": list(conditions), "keep_unmatched": link.keep_unmatched, "role": link.role or ""}
    if cardinality:
        body["cardinality"] = cardinality
    return body


def _measured(monkeypatch, most: int | None, problem: str = ""):
    from core2.model import links

    def check_link(model, spec, warehouse):
        out = {"types": [], "problem": problem, "words": ["counted more than once"] if (most or 0) > 1 else []}
        if most is not None and not problem:
            out.update({"most": most, "twice": 10 if most > 1 else 0, "match": 1.0, "with_key": 1})
        return out

    monkeypatch.setattr(links, "check_link", check_link)


def _save(client, account, body) -> dict:
    return client.post(f"/admin/clients/{account}/relationships/save-link", json=body).json()


def test_a_changed_condition_is_measured_on_the_data_and_kept(page):
    client, account, store, _warehouse = page
    model = _model(store, account)
    link, segment = _link(model, "order_lines", "customers"), _column(model, "customers", "segment")
    saved = _save(client, account, _body(link, [{"column": segment, "op": "eq", "values": ["Wholesale"]}]))
    assert saved["ok"] and saved["repeats"] is False, saved
    after = _model(store, account).joins[link.key]
    assert after.target_checked and after.to_unique and after.max_fanout == 1.0 and after.conditions


def test_a_condition_taken_away_that_lets_rows_repeat_stops_the_link(page, monkeypatch):
    client, account, store, _ = page
    model = _model(store, account)
    link, segment = _link(model, "order_lines", "customers"), _column(model, "customers", "segment")
    _measured(monkeypatch, 1)
    assert _save(client, account, _body(link, [{"column": segment, "op": "eq", "values": ["Wholesale"]}]))["ok"]
    _measured(monkeypatch, 3)          # without its condition, the table it reaches holds a member three times
    saved = _save(client, account, _body(link))
    assert saved["ok"] and saved["repeats"] is True and saved["words"]
    from core2.resolve.paths import repeats

    assert repeats(_model(store, account).joins[link.key]), "never followed: each row would be counted 3 times"


def test_a_condition_that_keeps_one_row_each_makes_a_repeating_link_safe(page, monkeypatch):
    client, account, store, _ = page
    model = _model(store, account)
    link, segment = _link(model, "order_lines", "customers"), _column(model, "customers", "segment")
    store.set_core2_override(account, None, f"join:{link.key}", "max_fanout", 3.0)
    from core2.resolve.paths import repeats

    assert repeats(_model(store, account).joins[link.key])
    _measured(monkeypatch, 1)
    assert _save(client, account, _body(link, [{"column": segment, "op": "eq", "values": ["Wholesale"]}]))["ok"]
    assert not repeats(_model(store, account).joins[link.key])


def test_a_change_that_cannot_be_measured_is_not_saved(page, monkeypatch):
    client, account, store, _ = page
    model = _model(store, account)
    link, segment = _link(model, "order_lines", "customers"), _column(model, "customers", "segment")
    _measured(monkeypatch, None, problem="The data refused the check: permission denied")
    saved = client.post(f"/admin/clients/{account}/relationships/save-link",
                        json=_body(link, [{"column": segment, "op": "eq", "values": ["Wholesale"]}]))
    assert saved.status_code == 400 and "checked against the data first" in saved.json()["error"]
    assert not _model(store, account).joins[link.key].conditions


def test_an_edit_keeps_a_link_the_data_showed_repeating_as_it_is(page, monkeypatch):
    client, account, store, _ = page
    model = _model(store, account)
    link = _link(model, "order_lines", "customers")
    store.set_core2_override(account, None, f"join:{link.key}", "cardinality", "many_to_many")
    _measured(monkeypatch, 2)
    # The page sends its "Rows" choice with every save: many to one, when the admin only changed a role.
    assert _save(client, account, {**_body(link, cardinality="many_to_one"), "keep_unmatched": False})["ok"]
    after = _model(store, account).joins[link.key]
    assert after.cardinality == "many_to_many" and after.keep_unmatched is False


def test_one_the_data_now_shows_one_row_each_is_called_so(page, monkeypatch):
    client, account, store, _ = page
    model = _model(store, account)
    link = _link(model, "order_lines", "customers")
    store.set_core2_override(account, None, f"join:{link.key}", "cardinality", "many_to_many")
    _measured(monkeypatch, 1)
    assert _save(client, account, _body(link, cardinality="many_to_one"))["ok"]
    assert _model(store, account).joins[link.key].cardinality == "many_to_one"


def test_the_page_draws_and_edits_each_link_as_the_data_shows():
    from pathlib import Path

    page_source = (Path(__file__).resolve().parents[1] / "admin/templates/client_relationships.html").read_text()
    assert "ends(j.cardinality)" in page_source and "cardinality: j.cardinality," in page_source
    assert "j.cardinality === 'one_to_one' ? '1 : 1' : 'N : 1'" not in page_source
