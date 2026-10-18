"""The Relationships list says what a link does with rows that have no match.

Each link either keeps a counted row with no match (under Unknown) or leaves it out of every total -- a LEFT or
an INNER join. The setting could only be seen by opening a link's editor; the list showed how many rows meet at
each end and nothing about the rows that meet none. Now every row of the list says it, in words.

The list is drawn by the page's own script, so the test draws it in a real browser from the page the route
serves, before and after the admin turns "keep" off.
"""

from __future__ import annotations

import os

import pytest

from tests.test_core2_relationships_are_edited_on_their_page import _link, _model, page, retail  # noqa: F401


def _row_cells(html: str) -> dict[str, dict[str, str]]:
    """Each listed link's cells by their column, as the browser draws them."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as pw:
        # Playwright's own browser, or one named in QB_CHROMIUM (as tools/ui_check takes --chromium).
        chromium = os.environ.get("QB_CHROMIUM")
        try:
            browser = pw.chromium.launch(**({"executable_path": chromium} if chromium else {}))
        except Exception as exc:  # no browser on this machine
            pytest.skip(f"no Chromium (set QB_CHROMIUM to one): {exc}")
        tab = browser.new_page()
        tab.route("**/*", lambda route: route.abort() if route.request.resource_type != "document" else route.continue_())
        tab.set_content(html, wait_until="domcontentloaded")
        tab.wait_for_selector("tbody tr td[data-label]")
        rows = tab.eval_on_selector_all("tbody tr", """rows => rows.map(r => {
            const out = {};
            r.querySelectorAll('td').forEach((td, i) => { out[td.dataset.label || (i === 0 ? 'Join' : '#' + i)] = td.innerText.trim(); });
            return out; })""")
        browser.close()
    return {r["Join"]: r for r in rows if "Join" in r}


def test_every_link_in_the_list_says_whether_rows_without_a_match_are_kept(page):  # noqa: F811
    client, account, store, _ = page
    before = _row_cells(client.get(f"/admin/clients/{account}/relationships").text)
    assert before and all(r.get("Rows without a match") == "Kept as Unknown" for r in before.values()), before

    link = _link(_model(store, account), "order_lines", "customers")
    body = {"key": link.key, "from": link.from_table, "to": link.to_table, "keep_unmatched": False,
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)]}
    assert client.post(f"/admin/clients/{account}/relationships/save-link", json=body).json()["ok"]
    after = _row_cells(client.get(f"/admin/clients/{account}/relationships").text)
    assert after["Order line → Customer"]["Rows without a match"] == "Left out"
    assert sum(r["Rows without a match"] == "Left out" for r in after.values()) == 1
