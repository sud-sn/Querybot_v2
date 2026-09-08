# -*- coding: utf-8 -*-
"""tests/test_plan_preview_and_stage_language.py

F2, F8 · Two producers whose output lands inside translated copy.

F2 · core/plan_preview.py builds its summary as an English sentence, and
gateway/webhooks.py interpolates it into the translated catalogue string
reply.plan.preview_suffix. What a French reader got was one sentence in two
languages, welded at the join:

    I'd answer this using **ORDERS_FCT** joined to **CUSTOMER_DMS**. Dites
    « vas-y » pour l'exécuter, ou indiquez-moi ce qu'il faut changer.

F8 · core/agent_runtime.py labels the live stage pills shown under the answer
while a question is processing. The pipeline's own stages were catalogued long
ago -- there are 36 stage.* ids -- and this second set of labels was not, so the
SAME field on the SAME run in the SAME French session reads French when the
pipeline writes it and English when agent_runtime does:

    record_stage("authorization", …)  ->  "Vérification des accès"
    start()                           ->  "Understanding your request"

── Two things the fixes have in common ─────────────────────────────────────

A SENTENCE IS NOT ASSEMBLED FROM CLAUSES. plan_preview built "I'd answer this
using {table_line}{caveat}." out of English fragments. French agreement and
word order do not survive a clause dropped into the middle of a sentence, so
each shape gets a whole sentence of its own: single table, joined tables, and
each of those with the unreviewed-join caveat. Table names are tenant data and
are interpolated.

A DEFAULT IS NOT A CONSTANT. agent_runtime's label was a dataclass default and
two keyword defaults -- all evaluated once, at import. A translated value there
would freeze whichever language loaded first and serve it to every tenant
afterwards. They become a default_factory and a resolve-inside-the-call, and a
test constructs the session under each language in turn to prove it.

Every test executes the real producer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import i18n  # noqa: E402


def under(lang, fn, *args, **kw):
    token = i18n.activate_language(lang)
    try:
        return fn(*args, **kw)
    finally:
        i18n.deactivate_language(token)


RESOLUTIONS = {
    "no_match": {"enabled": False},
    "single": {"enabled": True, "anchor": "ORDERS_FCT", "detected": [],
               "graph_scope": "confirmed"},
    "joined": {"enabled": True, "anchor": "ORDERS_FCT",
               "detected": ["CUSTOMER_DMS"], "graph_scope": "confirmed"},
    "three": {"enabled": True, "anchor": "ORDERS_FCT",
              "detected": ["CUSTOMER_DMS", "PRODUCT_DIM"],
              "graph_scope": "confirmed"},
    "unreviewed": {"enabled": True, "anchor": "ORDERS_FCT",
                   "detected": ["CUSTOMER_DMS"],
                   "graph_scope": "suggested_fallback"},
    "unreviewed_single": {"enabled": True, "anchor": "ORDERS_FCT",
                          "detected": [], "graph_scope": "suggested_fallback"},
}


def preview(case: str, lang: str) -> str:
    """The real builder, with only its two boundaries mocked."""
    import store
    import core.graph_resolver as graph_resolver
    import core.plan_preview as plan_preview

    with patch.object(store, "get_full_graph", return_value={}), \
         patch.object(graph_resolver, "resolve_for_question",
                      return_value=RESOLUTIONS[case]):
        built = under(lang, plan_preview.build_plan_preview,
                      account_id="a", question="q", db_type="azure_sql")
    return built.summary


def as_the_reader_sees_it(case: str, lang: str) -> str:
    """The summary inside the translated suffix, as webhooks assembles it."""
    return under(lang, i18n.t, "reply.plan.preview_suffix",
                 summary=preview(case, lang))


class TestThePlanPreviewIsOneLanguage(unittest.TestCase):

    def test_no_shape_answers_in_english_to_a_french_reader(self):
        for case in RESOLUTIONS:
            with self.subTest(case=case):
                self.assertNotEqual(preview(case, "en"), preview(case, "fr"))

    def test_the_welded_sentence_is_gone(self):
        # The artifact from the finding: an English clause fused to a French
        # one at the interpolation point.
        whole = as_the_reader_sees_it("joined", "fr")
        self.assertIn("vas-y", whole)
        self.assertNotIn("I'd answer this", whole)
        self.assertNotIn("joined to", whole)

    def test_the_no_match_shape_too(self):
        whole = as_the_reader_sees_it("no_match", "fr")
        self.assertNotIn("I'd generate", whole)
        self.assertNotIn("normal way", whole)

    def test_the_english_reader_still_gets_english(self):
        whole = as_the_reader_sees_it("joined", "en")
        self.assertIn("I'd answer this using", whole)
        self.assertNotIn("vas-y", whole)

    def test_the_unreviewed_caveat_is_part_of_its_sentence(self):
        # Not a fragment appended after the fact: a clause bolted onto the end
        # of a translated sentence lands in the wrong place in French.
        for case in ("unreviewed", "unreviewed_single"):
            with self.subTest(case=case):
                french = preview(case, "fr")
                self.assertIn("suggestion non revue", french)
                self.assertNotIn("unreviewed suggestion", french)

    def test_a_confirmed_join_carries_no_caveat(self):
        self.assertNotIn("suggestion", preview("joined", "fr"))

    def test_the_table_names_are_never_translated(self):
        for case in ("single", "joined", "three"):
            with self.subTest(case=case):
                french = preview(case, "fr")
                self.assertIn("**ORDERS_FCT**", french)
        self.assertIn("**PRODUCT_DIM**", preview("three", "fr"))

    def test_the_english_is_unchanged(self):
        self.assertEqual(
            preview("single", "en"),
            "I'd answer this using the **ORDERS_FCT** table.")
        self.assertEqual(
            preview("joined", "en"),
            "I'd answer this using **ORDERS_FCT** joined to **CUSTOMER_DMS**.")


class TestTheAgentStageLabels(unittest.TestCase):

    def session(self, lang):
        from core.agent_runtime import AgentRunSession

        return under(lang, AgentRunSession, account_id="a", portal_user_id=1,
                     external_thread_id="x", thread_id="t", run_id="r")

    def test_the_opening_label_is_in_the_readers_language(self):
        self.assertEqual(self.session("en").label, "Understanding your request")
        self.assertEqual(self.session("fr").label,
                         "Compréhension de votre demande")

    def test_the_label_is_resolved_per_session_not_once_at_import(self):
        # A dataclass default is evaluated once, when the module loads. A
        # translated value there would freeze whichever language happened to
        # load first and serve it to every tenant after.
        first, second = self.session("fr").label, self.session("en").label
        self.assertNotEqual(first, second)
        self.assertEqual(self.session("fr").label, first)

    def resumed(self, lang):
        """The real resume(), with only the store boundary mocked."""
        import core.agent_runtime as agent_runtime
        from core.agent_runtime import AgentRunSession

        # Patched on the module object agent_runtime itself holds. Another test
        # in the suite rebinds sys.modules["store"], so a plain `import store`
        # here is a DIFFERENT object and the patch would silently miss -- which
        # is exactly what happened: these passed alone and failed in the full
        # run, returning None because get_waiting_agent_run was never mocked.
        store = agent_runtime.store

        run = {"thread_id": "t", "id": "r"}
        with patch.object(store, "get_waiting_agent_run", return_value=run), \
             patch.object(store, "update_agent_run"), \
             patch.object(store, "update_agent_step"), \
             patch.object(store, "add_agent_message"), \
             patch.object(store, "create_agent_step"):
            return under(lang, AgentRunSession.resume, account_id="a",
                         portal_user_id=1, external_thread_id="x",
                         reply="the answer")

    def test_the_resume_label_is_translated(self):
        # Executed, not read off the catalogue: asserting on MESSAGES would
        # pass whether or not resume() uses the id, which is the exact
        # anti-pattern this sweep exists to remove.
        self.assertEqual(self.resumed("en").label, "Applying your clarification")
        self.assertEqual(self.resumed("fr").label,
                         "Prise en compte de votre précision")

    def test_the_resume_label_is_resolved_per_call(self):
        first = self.resumed("fr").label
        self.assertNotEqual(self.resumed("en").label, first)
        self.assertEqual(self.resumed("fr").label, first)

    def test_the_detail_the_store_row_records_is_translated_too(self):
        # It is written through store.update_agent_step, so the assertion is on
        # what the call was given.
        import core.agent_runtime as agent_runtime
        from core.agent_runtime import AgentRunSession

        store = agent_runtime.store
        run = {"thread_id": "t", "id": "r"}
        with patch.object(store, "get_waiting_agent_run", return_value=run), \
             patch.object(store, "update_agent_run"), \
             patch.object(store, "update_agent_step") as update_step, \
             patch.object(store, "add_agent_message"), \
             patch.object(store, "create_agent_step"):
            under("fr", AgentRunSession.resume, account_id="a",
                  portal_user_id=1, external_thread_id="x", reply="a")
        details = [str(call.kwargs.get("detail", "")) for call in
                   update_step.call_args_list]
        self.assertTrue(details, "update_agent_step was never called")
        self.assertNotIn("Clarification received", " ".join(details))

    def started(self, lang):
        """The real start(), with only the store boundary mocked."""
        import core.agent_runtime as agent_runtime
        from core.agent_runtime import AgentRunSession

        store = agent_runtime.store

        with patch.object(store, "get_allowed_tables", return_value=[]), \
             patch.object(store, "get_compliance_profile", return_value={}), \
             patch.object(store, "get_semantic_compiler_state", return_value={}), \
             patch.object(store, "ensure_agent_thread", return_value={"id": "t"}), \
             patch.object(store, "create_agent_run", return_value={"id": "r"}), \
             patch.object(store, "add_agent_message"), \
             patch.object(store, "create_agent_step") as create_step:
            session = under(lang, AgentRunSession.start, account_id="a",
                            portal_user={"id": 1}, external_thread_id="x",
                            objective="how many orders")
        return session, create_step

    def test_the_label_start_actually_writes_is_translated(self):
        # The field default and start()'s own default are two separate
        # defaults for the same label; testing the constructor alone leaves
        # this one uncovered.
        session, create_step = self.started("fr")
        self.assertEqual(session.label, "Compréhension de votre demande")
        labels = " ".join(str(call.kwargs.get("label", "")) for call in
                          create_step.call_args_list)
        self.assertNotIn("Understanding your request", labels)

    def test_the_opening_detail_is_translated_too(self):
        # The line under the pill. Asserted on what create_agent_step was
        # given, because that is where it goes.
        _, create_step = self.started("fr")
        details = [str(call.kwargs.get("detail", "")) for call in
                   create_step.call_args_list]
        self.assertTrue(details, "create_agent_step was never called")
        self.assertNotIn("Read-only governed query pipeline", " ".join(details))
        self.assertIn("lecture seule", " ".join(details))

    def test_the_english_opening_detail_is_unchanged(self):
        _, create_step = self.started("en")
        details = " ".join(str(call.kwargs.get("detail", "")) for call in
                           create_step.call_args_list)
        self.assertIn("Read-only governed query pipeline", details)

    def test_a_caller_that_names_its_own_stage_still_wins(self):
        # The analysis call site passes initial_label/initial_detail
        # explicitly. Making the defaults dynamic must not start overriding
        # them.
        import core.agent_runtime as agent_runtime
        from core.agent_runtime import AgentRunSession

        store = agent_runtime.store
        with patch.object(store, "get_allowed_tables", return_value=[]), \
             patch.object(store, "get_compliance_profile", return_value={}), \
             patch.object(store, "get_semantic_compiler_state", return_value={}), \
             patch.object(store, "ensure_agent_thread", return_value={"id": "t"}), \
             patch.object(store, "create_agent_run", return_value={"id": "r"}), \
             patch.object(store, "add_agent_message"), \
             patch.object(store, "create_agent_step"):
            session = under("fr", AgentRunSession.start, account_id="a",
                            portal_user={"id": 1}, external_thread_id="x",
                            objective="o", initial_label="Chosen by the caller")
        self.assertEqual(session.label, "Chosen by the caller")

    def test_it_agrees_with_the_pipelines_own_stage_labels(self):
        # The finding in one assertion: the same field, on the same run, in the
        # same language, from two producers.
        from core.i18n import t

        pipeline_label = under("fr", t, "stage.authorization.label")
        agent_label = self.session("fr").label
        self.assertNotEqual(pipeline_label, "Checking access")
        self.assertNotEqual(agent_label, "Understanding your request")

    def test_an_explicit_label_from_the_caller_still_wins(self):
        # The analysis call site passes its own label and detail; making the
        # defaults dynamic must not start overriding them.
        from core.i18n import t

        self.assertEqual(
            under("fr", t, "stage.analysing_results.label"),
            i18n.MESSAGES["stage.analysing_results.label"]["fr"])


if __name__ == "__main__":
    unittest.main()
