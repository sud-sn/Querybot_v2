"""
tests/test_date_choice_carries_evidence.py

"Which date should I use?" -- and nothing to answer it with.

An enterprise fact carries several dates for the same rows: invoice date,
order date, requested delivery date, payment date. When none of them is an
approved default, the product does the right thing and asks. What it asked
with was a list of NAMES:

    I found these relevant business dates, but none is an unambiguous
    approved default. Which date should I use?

        [ Invoice Date ] [ Order Date ] [ Delivery Date ] [ Payment Date ]

A reader asking about March has no way to tell which of those has March in it.
Delivery Date might be populated only from 2024; Payment Date might stop at
last quarter. Picking wrong produces a confident answer over the wrong slice of
the warehouse, and nothing on the card gives them a way to notice.

The fact that settles it is how far each date's data actually runs, and the
product already knows it: core/date_anchor.py probes exactly that -- "the
newest date present in this fact" -- and core/date_context_store.py persists it
per (tenant, fact, key). So every date this tenant has ever filtered on has a
stored answer, and the choice can be made on evidence for the cost of a local
read.

Deliberately NO warehouse query. A clarification is already a moment the reader
is waiting through; probing four facts to decorate it could take minutes, which
is the trade this whole subsystem exists to avoid.
"""

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_date_evidence.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

from core import i18n  # noqa: E402
from core.contextual_dates import (  # noqa: E402
    _as_display_date, describe_date_role_evidence,
)

FACT = "CUS_ORD_IVC_FCT"


@pytest.fixture
def tenant():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _remember(account_id, column, value):
    store.save_business_date_anchor(account_id, FACT, column, {"value": value})


class TestTheChoiceSaysHowFarEachDatesDataRuns:

    def test_a_probed_date_shows_where_its_data_ends(self, tenant):
        _remember(tenant, "IVC_DT_KEY", "2025-04-17")
        detail = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY"})
        assert "17 Apr 2025" in detail, detail

    def test_two_candidates_can_be_told_apart(self, tenant):
        """The whole point. Same fact, same question, different coverage."""
        _remember(tenant, "IVC_DT_KEY", "2025-04-17")
        _remember(tenant, "DLV_DT_KEY", "2023-12-31")
        invoice = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY"})
        delivery = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "DLV_DT_KEY"})
        assert "2025" in invoice and "2023" in delivery
        assert invoice != delivery

    def test_a_candidate_with_nothing_stored_says_nothing_rather_than_guessing(
            self, tenant):
        assert describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "NEVER_PROBED"}) == ""

    def test_the_approved_default_is_marked(self, tenant):
        detail = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY",
                     "is_default": 1})
        assert "approved default" in detail

    def test_an_approved_role_that_is_not_the_default_says_only_approved(
            self, tenant):
        detail = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY",
                     "governance_status": "approved"})
        assert detail == "approved", detail

    def test_both_facts_appear_together(self, tenant):
        _remember(tenant, "IVC_DT_KEY", "2025-04-17")
        detail = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY",
                     "is_default": 1})
        assert "approved default" in detail and "17 Apr 2025" in detail


class TestItCostsNothingAtTheWarehouse:

    def test_no_query_executor_is_ever_called(self, tenant, monkeypatch):
        """A clarification is a moment the reader is already waiting through.
        Probing four facts to decorate it could take minutes -- which is the
        trade core/date_anchor.py exists to avoid in the first place."""
        import core.schema

        calls = []
        monkeypatch.setattr(core.schema, "run_query",
                            lambda *a, **k: calls.append(a) or [])
        _remember(tenant, "IVC_DT_KEY", "2025-04-17")
        describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY"})
        assert calls == []

    def test_it_does_not_probe_for_a_missing_anchor_either(self, tenant, monkeypatch):
        import core.date_anchor as anchor

        probed = []
        monkeypatch.setattr(anchor, "resolve_business_anchor",
                            lambda *a, **k: probed.append(a) or {})
        describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "NEVER_PROBED"})
        assert probed == []


class TestItNeverBecomesTheReasonAQuestionFails:

    def test_a_broken_store_yields_less_detail_not_an_exception(
            self, tenant, monkeypatch):
        import store as store_mod

        monkeypatch.setattr(
            store_mod, "load_business_date_anchor",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
        detail = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY",
                     "is_default": 1})
        assert detail == "approved default", detail

    def test_an_unparseable_anchor_is_dropped(self, tenant):
        _remember(tenant, "JUNK_KEY", "not-a-date")
        assert describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "JUNK_KEY"}) == ""

    @pytest.mark.parametrize("value", [None, "", "2025-13-45", "yesterday", 20250417])
    def test_a_junk_value_is_not_shown(self, value):
        assert _as_display_date(value) == ""

    def test_no_account_no_lookup(self):
        assert describe_date_role_evidence(
            "", {"fact_table": FACT, "fact_column": "IVC_DT_KEY"}) == ""

    def test_no_binding_at_all_is_fine(self, tenant):
        assert describe_date_role_evidence(tenant, None) == ""


class TestItIsInTheReadersLanguage:

    def test_a_french_reader_gets_french(self, tenant):
        _remember(tenant, "IVC_DT_KEY", "2025-04-17")
        detail = describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY",
                     "is_default": 1}, lang="fr")
        assert "valeur par défaut approuvée" in detail
        assert "données jusqu'au" in detail

    def test_the_month_name_follows_the_reader(self, tenant):
        _remember(tenant, "IVC_DT_KEY", "2025-04-17")
        assert "avr." in describe_date_role_evidence(
            tenant, {"fact_table": FACT, "fact_column": "IVC_DT_KEY"}, lang="fr")

    def test_the_day_month_year_ORDER_does_not(self):
        """Swapping the order by language silently changes which number is the
        day -- the one date bug nobody spots from the screen."""
        for lang in ("en", "fr"):
            assert _as_display_date("2025-04-17", lang=lang).startswith("17 ")


class TestTheBrowserGetsItAndSubmitsTheRightThing:
    """The detail is decoration on a button whose text used to BE the answer."""

    def test_the_projection_lets_it_through(self):
        from gateway.web_adapter import _public_clarification_options

        public = _public_clarification_options([{
            "id": "date_role_1", "label": "Invoice Date", "value": "Invoice Date",
            "detail": "approved default · data through 17 Apr 2025",
            "fact_table": FACT, "fact_column": "IVC_DT_KEY",
        }])
        assert public[0]["detail"] == "approved default · data through 17 Apr 2025"

    def test_the_projection_still_withholds_the_physical_identity(self):
        from gateway.web_adapter import _public_clarification_options

        public = _public_clarification_options([{
            "id": "date_role_1", "label": "Invoice Date", "detail": "x",
            "fact_table": FACT, "fact_column": "IVC_DT_KEY",
            "resolved_question": "revenue by invoice date",
        }])
        assert "fact_table" not in public[0]
        assert "fact_column" not in public[0]
        assert "resolved_question" not in public[0]


DOM = """
function _mkEl(tag){
  return {
    tagName: tag, className: '', innerHTML: '', textContent: '',
    dataset: {}, children: [], _handlers: {},
    appendChild: function(child){ this.children.push(child); return child; },
    querySelector: function(){ return null; },
    querySelectorAll: function(){ return []; },
    addEventListener: function(name, fn){ this._handlers[name] = fn; },
    setAttribute: function(){},
    classList: { _c: [], add: function(c){ this._c.push(c); },
                 remove: function(){}, contains: function(c){ return this._c.indexOf(c) >= 0; } }
  };
}
var document = { createElement: function(tag){ return _mkEl(tag); } };
var submitted = [];
function _submitClarification(data, text, id, card){ submitted.push({text: text, id: id}); }
function escHtml(s){ return String(s); }
"""


def _render_options(options, lang="en"):
    """Run the page's real option loop over a small DOM."""
    import json

    from tests.chat_js import run as run_js

    script = """
    var wrap = _mkEl('div');
    var data = {pending_id: 'p1'};
    var card = _mkEl('div');
    var options = %s;
    %s
    JSON.stringify({
      buttons: wrap.children.map(function(b){
        return {
          text: b.textContent,
          detailed: b.classList.contains('clarification-option-detailed'),
          parts: b.children.map(function(c){
            return {cls: c.className, text: c.textContent};
          })
        };
      }),
      click: (function(){ wrap.children[0]._handlers.click(); return submitted; })()
    });
    """ % (json.dumps(options), _OPTION_LOOP)
    return run_js(script, lang=lang, preamble=DOM)


def _option_loop_source():
    """The forEach that builds the chips, lifted from the page."""
    from tests.chat_js import source

    src = source()
    start = src.index("  options.forEach((opt) => {")
    end = src.index("  const appendFreeTextClarification", start)
    return src[start:end]


_OPTION_LOOP = _option_loop_source()


class TestThePageRendersItWithoutBreakingTheAnswer:

    def test_the_detail_is_its_own_element(self):
        rendered = _render_options([
            {"id": "date_role_1", "label": "Invoice Date",
             "detail": "approved default · data through 17 Apr 2025"},
        ])
        button = rendered["buttons"][0]
        assert button["detailed"] is True
        classes = [part["cls"] for part in button["parts"]]
        assert "clarification-option-name" in classes
        assert "clarification-option-detail" in classes

    def test_the_reader_sees_both(self):
        rendered = _render_options([
            {"id": "date_role_1", "label": "Invoice Date",
             "detail": "data through 17 Apr 2025"},
        ])
        texts = [part["text"] for part in rendered["buttons"][0]["parts"]]
        assert "Invoice Date" in texts
        assert "data through 17 Apr 2025" in texts

    def test_the_submitted_answer_is_the_label_alone(self):
        """The click handler used to submit btn.textContent. With the detail
        inside the button that would have sent "Invoice Datedata through 17 Apr
        2025" back as the reader's answer, and the option matcher would have
        failed to resolve it."""
        rendered = _render_options([
            {"id": "date_role_1", "label": "Invoice Date",
             "detail": "data through 17 Apr 2025"},
        ])
        assert rendered["click"] == [{"text": "Invoice Date", "id": "date_role_1"}]

    def test_an_option_with_no_detail_renders_as_before(self):
        rendered = _render_options([
            {"id": "opt1", "label": "Invoice Date"},
        ])
        button = rendered["buttons"][0]
        assert button["detailed"] is False
        assert button["text"] == "Invoice Date"
        assert button["parts"] == []

    def test_that_option_still_submits_its_label(self):
        rendered = _render_options([{"id": "opt1", "label": "Invoice Date"}])
        assert rendered["click"] == [{"text": "Invoice Date", "id": "opt1"}]

    def test_an_option_with_neither_falls_back_to_a_word_not_a_blank(self):
        rendered = _render_options([{"id": "opt1"}])
        assert rendered["buttons"][0]["text"].strip()
