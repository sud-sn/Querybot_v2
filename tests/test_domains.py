# -*- coding: utf-8 -*-
"""Domains and corroboration — core/domains.py, store/domain_store.py.

The capability this answers is the one a competing product appears to have
and does not: it binds one app per turn and lets an agent wander into others
mid-loop, which reads as cross-app answering. But an agent that reads two
apps has no mechanism to notice they disagree, because nothing asked it to
look. A comparison step cannot miss a disagreement.

Three properties are asserted hardest:

  * adding domains must never make an un-domained workspace worse — a
    question matching nothing routes nowhere and the existing scope stands;
  * a domain narrows a user's table scope and never widens it; and
  * corroboration reports "not checked" rather than "agrees" whenever there
    was nothing to compare, because a user reads "confirmed" as evidence.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.domains import (  # noqa: E402
    CORROBORATION_TOLERANCE,
    DECISIVE_MARGIN,
    MIN_ROUTE_SCORE,
    Corroboration,
    allowed_tables_for,
    corroborate,
    describe,
    route,
    score_domain,
)
from core.i18n import MESSAGES  # noqa: E402

SALES = {
    "name": "Sales", "synonyms": "bookings, orders",
    "tables": ["SALES.ORDERS", "SALES.CUSTOMER"],
    "description": "Orders and customers",
}
FINANCE = {
    "name": "Finance", "synonyms": "ledger",
    "tables": ["FIN.GL", "FIN.AP"],
    "description": "General ledger and payables",
}


# ══════════════════════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════════════════════

class TestRouting(unittest.TestCase):

    def test_a_question_naming_a_domain_routes_to_it(self):
        routing = route("what were bookings last month", [SALES, FINANCE])
        self.assertTrue(routing.routed)
        self.assertEqual(routing.primary.name, "Sales")
        self.assertIsNone(routing.secondary)

    def test_a_synonym_routes_as_well_as_the_name(self):
        self.assertEqual(
            route("show the ledger balance", [SALES, FINANCE]).primary.name,
            "Finance")

    def test_a_question_matching_nothing_routes_nowhere(self):
        # The property that makes domains safe to add: an unrouted question
        # keeps whatever behaviour the workspace already had.
        routing = route("how many widgets shipped", [SALES, FINANCE])
        self.assertFalse(routing.routed)
        self.assertEqual(routing.reason, "no_domain_matched")

    def test_a_workspace_with_no_domains_routes_nowhere(self):
        routing = route("anything at all", [])
        self.assertFalse(routing.routed)
        self.assertEqual(routing.reason, "no_domains")

    def test_a_question_both_areas_could_answer_offers_a_second_opinion(self):
        routing = route("sales and ledger together", [SALES, FINANCE])
        self.assertTrue(routing.routed)
        self.assertTrue(routing.should_corroborate)
        self.assertEqual(routing.reason, "two_plausible_domains")
        self.assertNotEqual(routing.primary.name, routing.secondary.name)

    def test_a_decisive_winner_is_not_second_guessed(self):
        # Both domains clear the floor -- this question names each of them --
        # but one is named three ways and the other once. Running every
        # question twice would double the cost of the product to confirm
        # answers nothing disputed, so the margin has to actually decide.
        routing = route("bookings and orders and sales versus the ledger",
                        [SALES, FINANCE])
        self.assertTrue(routing.routed)
        self.assertEqual(routing.primary.name, "Sales")
        runner_up = routing.considered[1]
        self.assertGreaterEqual(runner_up.score, MIN_ROUTE_SCORE,
                                "the runner-up must clear the floor, or this "
                                "test never reaches the margin rule")
        self.assertFalse(routing.should_corroborate)

    def test_every_domain_considered_is_reported(self):
        routing = route("bookings", [SALES, FINANCE])
        self.assertEqual({m.name for m in routing.considered},
                         {"Sales", "Finance"})

    def test_routing_is_stable_regardless_of_domain_order(self):
        first = route("sales and ledger together", [SALES, FINANCE])
        second = route("sales and ledger together", [FINANCE, SALES])
        self.assertEqual(first.primary.name, second.primary.name)
        self.assertEqual(first.secondary.name, second.secondary.name)

    def test_the_thresholds_are_real_bars(self):
        self.assertGreater(MIN_ROUTE_SCORE, 0)
        self.assertGreater(DECISIVE_MARGIN, 0)
        self.assertLess(DECISIVE_MARGIN, 1)


class TestScoring(unittest.TestCase):

    def test_a_domain_name_inside_a_longer_word_does_not_match(self):
        # "Order" is a substring of "borders", and a substring match would
        # route a question about cross-border revenue into the orders area.
        # One token of overlap once bound a question to entirely the wrong
        # metric; the phrase match is anchored on word boundaries for exactly
        # that reason.
        orders = {"name": "Order", "synonyms": "", "tables": [],
                  "description": ""}
        question = "revenue across borders last quarter"
        self.assertIn("order", question)          # the substring really is there
        self.assertEqual(score_domain(question, orders).matched, ())
        self.assertLess(score_domain(question, orders).score, MIN_ROUTE_SCORE)

    def test_the_same_name_as_a_whole_word_does_match(self):
        # The negative above proves nothing unless the positive works.
        orders = {"name": "Order", "synonyms": "", "tables": [],
                  "description": ""}
        self.assertIn("Order",
                      score_domain("revenue by order last quarter", orders).matched)

    def test_a_whole_phrase_beats_incidental_overlap(self):
        supply = {"name": "Supply Chain", "synonyms": "", "tables": [],
                  "description": "supply and logistics"}
        finance = {"name": "Finance", "synonyms": "payments", "tables": [],
                   "description": ""}
        question = "how much did we owe in supplier payments"
        self.assertGreater(score_domain(question, finance).score,
                           score_domain(question, supply).score)

    def test_a_long_description_cannot_outweigh_a_name_match(self):
        # The description is the weakest signal and is capped for this
        # reason: a domain whose description happens to contain a dozen of
        # the question's words must not beat the domain the question names.
        wordy = {
            "name": "Other", "synonyms": "", "tables": [],
            "description": ("bookings orders sales revenue customers margin "
                            "region product quarter monthly totals invoices"),
        }
        question = ("bookings orders sales revenue customers margin region "
                    "product quarter monthly totals invoices")
        self.assertGreater(score_domain(question, SALES).score,
                           score_domain(question, wordy).score)

    def test_the_description_is_capped_not_merely_small(self):
        # Without a cap, enough overlapping words wins regardless of weight.
        few = {"name": "A", "synonyms": "", "tables": [],
               "description": "alpha beta gamma"}
        many = {"name": "B", "synonyms": "", "tables": [],
                "description": " ".join(f"w{i}" for i in range(40))}
        question = " ".join(f"w{i}" for i in range(40))
        self.assertLessEqual(score_domain(question, many).score,
                             score_domain("alpha beta gamma", few).score + 0.01)

    def test_a_table_name_in_the_question_contributes(self):
        with_table = score_domain("rows in SALES.ORDERS", SALES).score
        without = score_domain("rows in something else", SALES).score
        self.assertGreater(with_table, without)

    def test_what_matched_is_reported(self):
        match = score_domain("what were bookings last month", SALES)
        self.assertIn("bookings", match.matched)


class TestScopeIsNarrowedNeverWidened(unittest.TestCase):

    def test_an_unrouted_question_keeps_the_scope_it_had(self):
        existing = {"OTHER.THING"}
        routing = route("nothing matches this", [SALES])
        self.assertEqual(allowed_tables_for(routing, existing=existing), existing)

    def test_a_routed_question_is_scoped_to_the_domain(self):
        routing = route("bookings", [SALES])
        self.assertEqual(allowed_tables_for(routing, existing=None),
                         {"SALES.ORDERS", "SALES.CUSTOMER"})

    def test_a_users_own_restriction_still_wins(self):
        # A user with access to one table who asks a question routed to a
        # domain of twenty gets one. A domain must never widen access.
        routing = route("bookings", [SALES])
        self.assertEqual(
            allowed_tables_for(routing, existing={"SALES.ORDERS"}),
            {"SALES.ORDERS"})

    def test_a_user_with_no_access_to_the_domain_gets_nothing_from_it(self):
        routing = route("bookings", [SALES])
        self.assertEqual(
            allowed_tables_for(routing, existing={"FIN.GL"}), set())

    def test_case_differences_do_not_leak_access(self):
        routing = route("bookings", [SALES])
        self.assertEqual(
            allowed_tables_for(routing, existing={"sales.orders"}),
            {"sales.orders"})


# ══════════════════════════════════════════════════════════════════════════════
# Corroboration
# ══════════════════════════════════════════════════════════════════════════════

class TestCorroboration(unittest.TestCase):

    def test_two_sources_reporting_the_same_figure_agree(self):
        result = corroborate([{"V": 1000.0}], [{"V": 1000.0}],
                             primary_source="Sales", secondary_source="Finance")
        self.assertTrue(result.checked)
        self.assertTrue(result.agrees)
        self.assertEqual(result.reason, "agreement")

    def test_a_load_lag_sized_difference_still_agrees(self):
        # Two areas computing the same measure from different facts differ by
        # rounding, by late arrivals, by a day's lag. Calling that a
        # contradiction would cry wolf on every question.
        result = corroborate(
            [{"V": 1000.0}],
            [{"V": 1000.0 * (1 + CORROBORATION_TOLERANCE / 2)}])
        self.assertTrue(result.agrees)

    def test_a_material_difference_is_reported_with_both_figures(self):
        result = corroborate([{"V": 1000.0}], [{"V": 1400.0}],
                             primary_source="Sales", secondary_source="Finance")
        self.assertTrue(result.checked)
        self.assertFalse(result.agrees)
        self.assertEqual(result.primary_value, 1000.0)
        self.assertEqual(result.secondary_value, 1400.0)
        self.assertEqual(result.difference, 400.0)
        self.assertGreater(result.relative_difference, CORROBORATION_TOLERANCE)

    def test_an_empty_secondary_is_not_checked_rather_than_agreeing(self):
        # A user reads "confirmed" as evidence. Reporting agreement that was
        # never established is worse than reporting nothing.
        result = corroborate([{"V": 1000.0}], [])
        self.assertFalse(result.checked)
        self.assertFalse(result.agrees)
        self.assertEqual(result.reason, "secondary_empty")

    def test_an_empty_primary_is_not_checked(self):
        result = corroborate([], [{"V": 1000.0}])
        self.assertFalse(result.checked)
        self.assertEqual(result.reason, "primary_empty")

    def test_a_result_with_no_figure_is_not_checked(self):
        result = corroborate([{"NAME": "West"}], [{"NAME": "East"}])
        self.assertFalse(result.checked)
        self.assertEqual(result.reason, "no_comparable_figure")

    def test_zero_against_zero_agrees_without_dividing_by_it(self):
        result = corroborate([{"V": 0.0}], [{"V": 0.0}])
        self.assertTrue(result.agrees)

    def test_zero_against_a_real_figure_disagrees(self):
        result = corroborate([{"V": 0.0}], [{"V": 5000.0}])
        self.assertTrue(result.checked)
        self.assertFalse(result.agrees)

    def test_the_row_counts_are_recorded_not_the_rows(self):
        result = corroborate([{"CUSTOMER": "Ospedale", "V": 1.0}],
                             [{"CUSTOMER": "Ospedale", "V": 1.0}])
        self.assertEqual(result.detail["primary_rows"], 1)
        self.assertNotIn("Ospedale", repr(result.detail))


class TestWhatTheUserIsTold(unittest.TestCase):

    def test_an_unchecked_corroboration_says_nothing(self):
        self.assertEqual(describe(Corroboration()), "")
        self.assertEqual(describe(corroborate([{"V": 1.0}], [])), "")

    def test_agreement_names_the_source_that_confirmed_it(self):
        line = describe(corroborate([{"V": 1000.0}], [{"V": 1000.0}],
                                    secondary_source="Finance"))
        self.assertIn("Finance", line)
        self.assertIn("same figure", line)

    def test_a_disagreement_gives_both_numbers_and_the_gap(self):
        line = describe(corroborate([{"V": 1000.0}], [{"V": 1400.0}],
                                    primary_source="Sales",
                                    secondary_source="Finance"))
        self.assertIn("Finance", line)
        self.assertIn("1,400", line)
        self.assertIn("1,000", line)
        self.assertIn("%", line)

    def test_both_languages_say_it_differently(self):
        result = corroborate([{"V": 1000.0}], [{"V": 1400.0}],
                             secondary_source="Finance")
        english = describe(result, lang="en")
        french = describe(result, lang="fr")
        self.assertTrue(english and french)
        self.assertNotEqual(english, french)
        # And French writes its own numbers: a narrow no-break space groups
        # thousands and a non-breaking space precedes the percent sign.
        self.assertIn("1\u202f400", french)
        self.assertIn("28,6\u00a0%", french)
        self.assertIn("1,400", english)
        self.assertIn("28.6%", english)

    def test_every_corroboration_id_exists_in_both_languages(self):
        ids = [k for k in MESSAGES if k.startswith("corroboration.")]
        self.assertGreaterEqual(len(ids), 2)
        for msg_id in ids:
            self.assertTrue(MESSAGES[msg_id].get("en"), msg_id)
            self.assertTrue(MESSAGES[msg_id].get("fr"), msg_id)
            self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"])


# ══════════════════════════════════════════════════════════════════════════════
# The store
# ══════════════════════════════════════════════════════════════════════════════

class TestTheDomainStore(unittest.TestCase):

    def setUp(self):
        import store
        self._dir = tempfile.mkdtemp(prefix="qb-domain-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-dom-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_a_saved_domain_routes_a_question(self):
        # Write API to read API: the routing has to find what the store saved.
        import store
        store.save_domain(self.account_id, "Sales",
                          tables=["sales.orders"], synonyms="bookings")
        routing = route("what were bookings", store.list_domains(self.account_id))
        self.assertEqual(routing.primary.name, "Sales")

    def test_table_names_are_stored_uppercase(self):
        # Retrieval, the planner and the value index all key on uppercase
        # fully-qualified names. A domain storing whatever case an admin typed
        # would match none of them, and every lookup would miss silently.
        import store
        store.save_domain(self.account_id, "Sales", tables=["sales.orders"])
        self.assertEqual(store.get_domain(self.account_id, "Sales")["tables"],
                         ["SALES.ORDERS"])

    def test_saving_the_same_name_replaces_rather_than_duplicates(self):
        import store
        store.save_domain(self.account_id, "Sales", tables=["A.B"])
        store.save_domain(self.account_id, "Sales", tables=["C.D"],
                          description="now finance-facing")
        domains = store.list_domains(self.account_id)
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0]["tables"], ["C.D"])
        self.assertEqual(domains[0]["description"], "now finance-facing")

    def test_a_deleted_domain_stops_routing_but_is_not_erased(self):
        import store
        store.save_domain(self.account_id, "Sales", tables=["A.B"])
        self.assertTrue(store.delete_domain(self.account_id, "Sales"))
        self.assertEqual(store.list_domains(self.account_id), [])
        self.assertEqual(len(store.list_domains(self.account_id, active_only=False)), 1)

    def test_a_domain_needs_a_name(self):
        import store
        with self.assertRaises(ValueError):
            store.save_domain(self.account_id, "  ", tables=["A.B"])

    def test_another_workspaces_domains_are_not_visible(self):
        import store
        other = f"acct-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other, "portal")
        store.save_domain(other, "Finance", tables=["FIN.GL"])
        store.save_domain(self.account_id, "Sales", tables=["A.B"])
        self.assertEqual([d["name"] for d in store.list_domains(self.account_id)],
                         ["Sales"])


if __name__ == "__main__":
    unittest.main()


# ══════════════════════════════════════════════════════════════════════════════
# Applying a route to a user's scope
# ══════════════════════════════════════════════════════════════════════════════

class TestNarrowingTheScope(unittest.TestCase):
    """``narrow_scope`` is the decision the pipeline makes.

    Extracted from ``_handle_query_impl`` for the same reason everything else
    was: a decision nobody can call is a decision nobody can test. The three
    outcomes it distinguishes are what make domains safe to switch on.
    """

    ALL = {"SALES.ORDERS", "SALES.CUSTOMER", "FIN.GL", "FIN.AP"}

    def _narrow(self, question, *, effective=None, allowed=None, domains=None):
        from core.domains import narrow_scope
        return narrow_scope(
            question, domains if domains is not None else [SALES, FINANCE],
            effective=self.ALL if effective is None else effective,
            allowed_tables=allowed,
        )

    def test_a_routed_question_is_scoped_to_its_domain(self):
        decision = self._narrow("what were bookings last month")
        self.assertTrue(decision.applied)
        self.assertEqual(decision.domain, "Sales")
        self.assertEqual(decision.effective, {"SALES.ORDERS", "SALES.CUSTOMER"})

    def test_an_unrouted_question_keeps_every_table_it_had(self):
        # Adding domains must not make an un-domained question worse.
        decision = self._narrow("how many widgets shipped")
        self.assertFalse(decision.applied)
        self.assertEqual(decision.effective, self.ALL)
        self.assertEqual(decision.reason, "no_domain_matched")

    def test_a_workspace_with_no_domains_changes_nothing(self):
        decision = self._narrow("what were bookings", domains=[])
        self.assertFalse(decision.applied)
        self.assertEqual(decision.effective, self.ALL)

    def test_the_users_own_restriction_still_wins(self):
        decision = self._narrow(
            "what were bookings", effective={"SALES.ORDERS"},
            allowed={"SALES.ORDERS"})
        self.assertTrue(decision.applied)
        self.assertEqual(decision.effective, {"SALES.ORDERS"})
        self.assertEqual(decision.allowed_tables, {"SALES.ORDERS"})

    def test_allowed_tables_is_narrowed_alongside_effective(self):
        # Two views of one permission. Letting them disagree is how a
        # validator ends up scoped differently from the retriever.
        decision = self._narrow(
            "what were bookings",
            allowed={"SALES.ORDERS", "SALES.CUSTOMER", "FIN.GL"})
        self.assertEqual(decision.effective, {"SALES.ORDERS", "SALES.CUSTOMER"})
        self.assertEqual(decision.allowed_tables, {"SALES.ORDERS", "SALES.CUSTOMER"})
        self.assertNotIn("FIN.GL", decision.allowed_tables)

    def test_an_admin_with_no_restriction_keeps_none(self):
        # allowed_tables is None for an unrestricted admin, and None means
        # "unrestricted" everywhere downstream -- turning it into a set here
        # would silently restrict them.
        decision = self._narrow("what were bookings", allowed=None)
        self.assertTrue(decision.applied)
        self.assertIsNone(decision.allowed_tables)

    def test_a_domain_the_user_cannot_see_drops_the_route(self):
        # Narrowing to nothing would answer nothing, and would look identical
        # to the question having no answer.
        decision = self._narrow(
            "what were bookings", effective={"FIN.GL"}, allowed={"FIN.GL"})
        self.assertFalse(decision.applied)
        self.assertEqual(decision.reason, "no_visible_tables")
        self.assertEqual(decision.effective, {"FIN.GL"})
        self.assertEqual(decision.allowed_tables, {"FIN.GL"})

    def test_a_second_opinion_is_reported_when_two_areas_could_answer(self):
        decision = self._narrow("sales and ledger together")
        self.assertTrue(decision.applied)
        self.assertTrue(decision.routing.should_corroborate)
        self.assertNotEqual(decision.routing.secondary.name, decision.domain)

    def test_the_returned_scope_is_not_an_alias_of_the_callers(self):
        # The decision returns a new set even on the paths that change
        # nothing, so a caller that mutates what it got back cannot reach
        # into the pipeline's own scope. Asserting the inputs are unchanged
        # after the call is not enough -- narrow_scope does not mutate them
        # either way, so that assertion passes on an aliased return.
        effective = set(self.ALL)
        decision = self._narrow("how many widgets shipped", effective=effective)
        self.assertFalse(decision.applied)
        decision.effective.add("SOMETHING.ELSE")
        self.assertEqual(effective, self.ALL)

    def test_a_routed_scope_is_not_an_alias_either(self):
        effective = set(self.ALL)
        decision = self._narrow("what were bookings", effective=effective)
        self.assertTrue(decision.applied)
        decision.effective.add("SOMETHING.ELSE")
        self.assertEqual(effective, self.ALL)


class TestThePipelineAppliesIt(unittest.TestCase):
    """The narrowing has to be reached, and reached before anything reads scope."""

    @staticmethod
    def _source():
        import inspect

        import core.query_pipeline as qp
        return inspect.getsource(qp._handle_query_impl)

    def test_the_route_is_applied_to_both_scope_variables(self):
        source = self._source()
        self.assertIn("effective = _scope_decision.effective", source)
        self.assertIn("allowed_tables = _scope_decision.allowed_tables", source)

    def test_it_runs_before_the_query_scope_is_derived(self):
        # query_scope_tables is built from `effective`, and everything below
        # it -- retrieval, planning, validation, repair -- reads that.
        # Narrowing after it would scope the validator and leave the
        # retriever looking at the whole workspace.
        source = self._source()
        narrowed = source.index("_scope_decision = _narrow_to_domain(")
        derived = source.index("query_scope_tables = effective if")
        self.assertLess(narrowed, derived)

    def test_it_runs_after_the_selected_schema_narrows_the_scope(self):
        # A domain narrows WITHIN whatever schema the user selected, rather
        # than competing with it -- so a Sales domain spanning two schemas
        # cannot pull a question back out of the schema tab the user is on.
        source = self._source()
        schema = source.index("effective = {t for t in effective if _in_schema(t)}")
        narrowed = source.index("_scope_decision = _narrow_to_domain(")
        self.assertLess(schema, narrowed)

    def test_it_runs_after_the_users_own_acl_is_applied(self):
        # A domain narrows a permission; it must never be the thing that
        # grants one.
        source = self._source()
        acl = source.index("effective = {t for t in all_known if t in allowed_tables}")
        narrowed = source.index("_scope_decision = _narrow_to_domain(")
        self.assertLess(acl, narrowed)

    def test_only_an_applied_route_is_carried_forward(self):
        source = self._source()
        self.assertIn("if _scope_decision.applied:\n                _domain_routing = _scope_decision.routing",
                      source)

    def test_the_answering_domain_reaches_answer_confidence(self):
        self.assertIn('"domain": (', self._source())

    def test_a_routing_failure_costs_the_scope_not_the_answer(self):
        source = self._source()
        block = source[source.index("_domain_routing = None"):]
        block = block[:block.index("query_scope_tables")]
        self.assertIn("except Exception as _domain_exc", block)
        self.assertIn('log.warning("Domain routing unavailable', block)
