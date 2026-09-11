"""
tests/test_answer_provenance_is_business_readable.py

The most reader-facing part of a governed answer was written in warehouse
identifiers.

build_answer_grounding collects how a figure was produced, and
_format_grounding_for_prompt renders it under a heading that tells the model:

    HOW THIS RESULT WAS PRODUCED — state the relevant parts in your answer,
    and never invent one that is absent here

so whatever is in that block can reach the reader verbatim. What was in it:

    - Business date the query resolved to: metric_default on CUS_ORD_IVC_FCT
    - Tables read: DBO.CUS_ORD_IVC_FCT, DBO.DIM_DATE

A machine token, a raw fact table, and a list of physical table names. The one
distinction a reader would actually act on was the one the token hid: an
APPROVED default is a governed choice, and a discovered or inferred role is the
product's own guess. Both are legitimate; being unable to tell them apart is
what makes a number indefensible.

The tables are gone rather than translated. There is no business name for a
table in this product -- display_label names columns -- so there was nothing to
translate them into, and the answer card's trust box already shows the SQL and
the tables it read to anyone who opens it.
"""

from __future__ import annotations

import pytest

from core import i18n
from core.date_roles import provenance_phrase
from core.insight import _format_grounding_for_prompt, build_answer_grounding

FACT = "CUS_ORD_IVC_FCT"
TABLES = {"DBO.CUS_ORD_IVC_FCT", "DBO.DIM_DATE"}


def _grounding(**disclosure):
    plan = {"date_disclosures": [{"table": FACT, "column": "IVC_DT_KEY",
                                 **disclosure}]}
    return build_answer_grounding(
        semantic_plan=plan, row_count=12, truncated=False, tables=set(TABLES))


def _prompt(**disclosure):
    return _format_grounding_for_prompt(_grounding(**disclosure))


class TestNoWarehouseIdentifierReachesTheReader:

    def test_the_fact_table_is_not_in_the_business_date_line(self):
        grounding = _grounding(label="Invoice Date",
                               resolution_source="metric_default")
        assert FACT not in grounding["business_date"], grounding
        assert "CUS_ORD" not in str(grounding), grounding

    def test_no_table_name_is_anywhere_in_the_block(self):
        rendered = _prompt(label="Invoice Date",
                           resolution_source="metric_default")
        for table in TABLES:
            assert table not in rendered, rendered
        assert "Tables read" not in rendered, rendered

    def test_the_machine_token_is_not_shown_either(self):
        for source in ("metric_default", "discovered_date_role",
                       "inferred_encoded_fact_date", "user_confirmed_date_role"):
            rendered = _prompt(label="Invoice Date", resolution_source=source)
            assert source not in rendered, (source, rendered)

    def test_the_business_date_is_still_named(self):
        """Removing the identifiers must not remove the fact."""
        grounding = _grounding(label="Invoice Date",
                               resolution_source="metric_default")
        assert "Invoice Date" in grounding["business_date"]


class TestTheReaderCanTellAGovernedChoiceFromAGuess:

    @pytest.mark.parametrize("source,expected", [
        ("metric_default", "approved default"),
        ("metric_default_time_column", "approved default"),
        ("approved_metric_date_context", "approved default"),
        ("fact_default_date_role", "approved default"),
        ("single_approved_date_role", "only approved date"),
        ("single_metric_context", "only date configured"),
    ])
    def test_a_governed_choice_says_so(self, source, expected):
        assert expected in _grounding(
            label="Invoice Date", resolution_source=source)["business_date"]

    @pytest.mark.parametrize("source", [
        "discovered_date_role", "inferred_encoded_fact_date",
    ])
    def test_a_guess_says_it_is_not_an_approved_default(self, source):
        """The distinction the token hid. A reader shown a number measured on a
        date the product guessed at should be able to see that."""
        line = _grounding(label="Invoice Date",
                          resolution_source=source)["business_date"]
        assert "not an approved default" in line, line

    @pytest.mark.parametrize("source,expected", [
        ("user_confirmed_date_role", "the date you chose"),
        ("thread_date_preference", "chose earlier in this conversation"),
        ("explicit_date_role", "named in your question"),
    ])
    def test_the_readers_own_choice_is_credited_to_them(self, source, expected):
        assert expected in _grounding(
            label="Invoice Date", resolution_source=source)["business_date"]


class TestItDegradesRatherThanGuesses:

    def test_an_unknown_source_omits_the_clause_instead_of_printing_it(self):
        """A token with no phrase must not fall back to the token."""
        line = _grounding(label="Invoice Date",
                          resolution_source="some_new_source_v2")["business_date"]
        assert line == "Invoice Date", line
        assert provenance_phrase("some_new_source_v2") == ""

    def test_no_source_at_all_still_names_the_date(self):
        grounding = _grounding(label="Order Date")
        assert grounding.get("date_context") == ["Order Date"], grounding

    def test_no_label_at_all_still_says_how_it_was_chosen(self):
        line = _grounding(resolution_source="metric_default")["business_date"]
        assert "approved default" in line, line

    def test_nothing_known_emits_nothing(self):
        assert build_answer_grounding(semantic_plan={"date_disclosures": [{}]}) == {}

    def test_a_malformed_plan_is_not_a_failed_answer(self):
        """This decorates an answer; it must never be why a question fails."""
        for junk in ({"date_disclosures": "not-a-list"},
                     {"date_disclosures": [None, 7, "x"]},
                     {}):
            assert isinstance(build_answer_grounding(semantic_plan=junk), dict)


class TestTheSameDateIsNotStatedTwice:

    def test_one_date_produces_one_line(self):
        """business_date and date_context sat side by side saying "Invoice
        Date" twice, which reads like the answer is unsure of itself."""
        grounding = _grounding(label="Invoice Date",
                               resolution_source="metric_default")
        assert "date_context" not in grounding, grounding

    def test_a_second_date_is_still_carried(self):
        plan = {"date_disclosures": [
            {"label": "Invoice Date", "table": "F1",
             "resolution_source": "user_confirmed_date_role"},
            {"label": "Delivery Date", "table": "F2"},
        ]}
        grounding = build_answer_grounding(semantic_plan=plan, row_count=3)
        assert "Invoice Date" in grounding["business_date"]
        assert grounding["date_context"] == ["Delivery Date"], grounding


class TestItIsInTheReadersLanguage:

    @pytest.mark.parametrize("source,expected_fr", [
        ("metric_default", "par défaut approuvée"),
        ("user_confirmed_date_role", "que vous avez choisie"),
        ("inferred_encoded_fact_date", "déduite"),
    ])
    def test_a_french_reader_gets_french_provenance(self, source, expected_fr):
        token = i18n.activate_language("fr")
        try:
            line = _grounding(label="Date de facture",
                              resolution_source=source)["business_date"]
        finally:
            i18n.deactivate_language(token)
        assert expected_fr in line, line

    def test_an_english_reader_gets_english(self):
        line = _grounding(label="Invoice Date",
                          resolution_source="metric_default")["business_date"]
        assert "approved default date" in line


class TestEveryResolutionSourceThePipelineCanProduceHasAPhrase:
    """A source with no phrase silently drops the clause, so a new one added to
    the resolver must not be able to arrive here unnoticed."""

    @staticmethod
    def _sources_in_the_resolver():
        """Every resolution_source core/contextual_dates.py can attach.

        Read as a SYNTAX TREE, not by regex. The first version of this scan
        matched `source="..."` on one line and an allowlist of names without
        "date" in them -- and missed both of the sources it was written to
        catch: `explicit_generated_date_role`, which is written as a
        multi-line conditional inside `source=(...)`, and `business_context`,
        whose name says nothing about dates. So the guard against a source
        arriving with no reader-facing phrase passed while two had.

        Every `source=` argument is resolved through the positions that can
        actually PRODUCE its value -- a conditional's two branches, an `or`
        chain's operands, a local's assignments -- not by collecting every
        string in the subtree. Walking the whole subtree swept up
        `"_selection_status"` and `"approved"` from the branch condition and
        reported them as resolution sources. A spelling this cannot resolve
        is reported rather than skipped, so the scan cannot go quietly blind
        again.

        Scoped to the resolver, which is the one function that decides this.
        """
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        source = (root / "core" / "contextual_dates.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        resolver = next(
            (node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == "resolve_contextual_date_binding"),
            None,
        )
        assert resolver is not None, "the resolver has been renamed"

        # Every function in the module, so a source decided by a helper and
        # handed back can be followed into it.
        functions = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def returns_of(name, index=None):
            """What `name` can return -- element `index` when it returns a
            tuple. `_single_candidate_verdict` hands the resolver a
            (source, reason) pair, so without this the scan sees only the
            local it was unpacked into and reports that it is blind."""
            function = functions.get(name)
            if function is None:
                return []
            out = []
            for node in ast.walk(function):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                value = node.value
                if index is not None:
                    if isinstance(value, (ast.Tuple, ast.List)):
                        if index < len(value.elts):
                            out.append(value.elts[index])
                    continue
                out.append(value)
            return out

        # Locals the resolver assigns, so `source=explicit_source` resolves to
        # the strings that variable can hold. Plain assignments and tuple
        # unpacking both, because the source now arrives either way.
        assigned: dict[str, list[ast.expr]] = {}
        for node in ast.walk(resolver):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, []).append(node.value)
                elif isinstance(target, (ast.Tuple, ast.List)) and \
                        isinstance(node.value, ast.Call) and \
                        isinstance(node.value.func, ast.Name):
                    for position, element in enumerate(target.elts):
                        if not isinstance(element, ast.Name):
                            continue
                        assigned.setdefault(element.id, []).extend(
                            returns_of(node.value.func.id, position))

        unresolved: set[str] = set()

        def values_of(expr, seen=()):
            """The strings `expr` can evaluate to."""
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                return {expr.value}
            if isinstance(expr, ast.IfExp):
                return values_of(expr.body, seen) | values_of(expr.orelse, seen)
            if isinstance(expr, ast.BoolOp):
                out = set()
                for operand in expr.values:
                    out |= values_of(operand, seen)
                return out
            if isinstance(expr, ast.Name):
                if expr.id in seen:          # a loop; nothing more to learn
                    return set()
                bound = assigned.get(expr.id)
                if not bound:
                    unresolved.add(expr.id)
                    return set()
                out = set()
                for value in bound:
                    out |= values_of(value, seen + (expr.id,))
                return out
            unresolved.add(ast.dump(expr)[:80])
            return set()

        found = set()
        for node in ast.walk(resolver):
            # `_role_as_binding(role, source=...)`, however the value is spelled.
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "source":
                        found |= values_of(keyword.value)
            # `binding["resolution_source"] = ...`
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript)
                            and isinstance(target.slice, ast.Constant)
                            and target.slice.value == "resolution_source"):
                        found |= values_of(node.value)

        assert not unresolved, (
            "this scan cannot tell what these `source=` expressions evaluate "
            f"to, so it is not checking them: {sorted(unresolved)}")
        return {value for value in found if value}

    def test_the_scan_found_the_sources(self):
        sources = self._sources_in_the_resolver()
        assert len(sources) >= 10, sources
        # The two the regex version could not see.
        assert "explicit_generated_date_role" in sources, sources
        assert "business_context" in sources, sources

    def test_every_one_of_them_has_a_reader_facing_phrase(self):
        missing = sorted(s for s in self._sources_in_the_resolver()
                         if not provenance_phrase(s))
        assert not missing, (
            "these resolution sources reach the provenance block with no "
            f"phrase, so the clause is silently dropped: {missing}")

    def test_every_phrase_exists_in_french_too(self):
        for source in sorted(self._sources_in_the_resolver()):
            assert provenance_phrase(source, lang="fr"), source


SHIPPED_PACKAGES = ("core", "gateway", "store", "admin")


class TestASourceWrittenOutsideTheResolverHasAPhraseToo:
    """The scan above reads core/contextual_dates.py, and that is where the
    resolver decides provenance -- but it is not the only place a
    resolution_source is ASSIGNED.

    core/query_pipeline.py rewrites one: when it has inferred which fact the
    question reaches, it relabels the binding "connected_dimension_default".
    That token was written in exactly one place and read as a known provenance
    in none -- it was never added to the catalogue, so provenance_phrase
    returned "" and the clause was dropped from the answer card. Invisible to
    the resolver-scoped scan by construction.

    So this one sweeps every shipped module for a resolution_source assigned to
    a literal, wherever it lives.
    """

    @staticmethod
    def _assigned_sources():
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        found: dict[str, str] = {}
        for package in SHIPPED_PACKAGES:
            directory = root / package
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except SyntaxError:                          # pragma: no cover
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    names = set()
                    for target in node.targets:
                        if isinstance(target, ast.Subscript) and isinstance(
                                target.slice, ast.Constant):
                            names.add(target.slice.value)
                    if "resolution_source" not in names:
                        continue
                    for inner in ast.walk(node.value):
                        if isinstance(inner, ast.Constant) and isinstance(
                                inner.value, str) and inner.value:
                            found[inner.value] = (
                                f"{path.relative_to(root)}:{node.lineno}")
        return found

    def test_the_sweep_found_the_one_the_resolver_scan_cannot_see(self):
        sources = self._assigned_sources()
        assert "connected_dimension_default" in sources, sources

    def test_every_one_of_them_has_a_phrase(self):
        missing = {
            source: where for source, where in self._assigned_sources().items()
            if not provenance_phrase(source)
        }
        assert not missing, (
            "these resolution sources are written to a binding and have no "
            "reader-facing phrase, so the answer card silently drops its "
            f"provenance clause: {missing}")

    def test_every_one_of_them_speaks_french(self):
        for source in sorted(self._assigned_sources()):
            assert provenance_phrase(source, lang="fr"), source
