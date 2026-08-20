"""
tests/test_join_key_vocabulary.py

Which shared column may link two tables is ERP knowledge, so it has to come
from the tenant's vocabulary pack. Three ways that had gone wrong in
core/semantic_planner.py::_join_edges:

  1. The eligible code set was read with ``_planner_vocab(None)``, throwing
     away the vocab the function had just been handed -- while the line above
     it used that same vocab for join synonyms.
  2. Beside it sat a literal ``c.endswith("_DMS_KEY")`` -- Infor M3's
     dimension-key spelling -- under a comment claiming the set "now comes
     from the pack".
  3. ``CONO`` was listed as a join key. It is M3's company number and sits on
     essentially every table, so it made every pair of tables adjacent and the
     path search routed through edges that mean nothing.

The third is the one worth remembering: an edge exists as soon as a pair shares
ONE eligible column, so a partition code that may legitimately appear in an ON
clause must never be allowed to put it there on its own.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import core.semantic_planner as planner
from core.vocab_packs import MergedVocab, _clone_builtin, _merge_pack


PACKS_DIR = pathlib.Path(__file__).resolve().parent.parent / "packs"


def _pack_vocab(pack_id: str) -> MergedVocab:
    vocab = _clone_builtin()
    _merge_pack(vocab, json.loads((PACKS_DIR / f"{pack_id}.json").read_text(encoding="utf-8")), pack_id)
    return vocab


def _edges(table_columns: dict, vocab) -> dict:
    graph = planner._join_edges(table_columns, vocab=vocab)
    return {
        table: {edge["to"]: [tuple(c) for c in edge["conditions"]] for edge in edges}
        for table, edges in graph.items() if edges
    }


# ── The vocabulary the caller passes is the vocabulary that is used ──────────


class TestTheCallersVocabularyDecides:
    def test_an_empty_vocabulary_produces_no_edges(self):
        """The regression test for the real bug: the eligible set was read from
        the ambient vocabulary, so an explicitly passed one could not turn any
        rule off -- or on."""
        assert _edges({
            "DW.F_ENCOUNTER": {"PATIENT_SK": "int"},
            "DW.D_PATIENT": {"PATIENT_SK": "int"},
        }, MergedVocab()) == {}

    def test_a_pack_can_introduce_a_convention_core_has_never_heard_of(self):
        vocab = _clone_builtin()
        _merge_pack(vocab, {"pack_id": "house", "join_key_suffixes": ["_HKEY"]}, "house")
        assert _edges({
            "DW.F_SALE": {"CUSTOMER_HKEY": "varchar", "NET": "decimal"},
            "DW.H_CUSTOMER": {"CUSTOMER_HKEY": "varchar"},
        }, vocab) == {
            "DW.F_SALE": {"DW.H_CUSTOMER": [("CUSTOMER_HKEY", "CUSTOMER_HKEY")]},
            "DW.H_CUSTOMER": {"DW.F_SALE": [("CUSTOMER_HKEY", "CUSTOMER_HKEY")]},
        }

    def test_a_pack_can_introduce_a_code_core_has_never_heard_of(self):
        vocab = _clone_builtin()
        _merge_pack(vocab, {"pack_id": "house", "join_key_codes": ["POLICYREF"]}, "house")
        assert "DW.D_POLICY" in _edges({
            "DW.F_CLAIM": {"POLICYREF": "varchar", "PAID": "decimal"},
            "DW.D_POLICY": {"POLICYREF": "varchar"},
        }, vocab)["DW.F_CLAIM"]


# ── A partition code may qualify a join but never create one ─────────────────


class TestAPartitionCodeCannotCreateAnEdge:
    M3 = None

    @pytest.fixture(autouse=True)
    def _vocab(self):
        type(self).M3 = _pack_vocab("infor_m3")

    def test_two_unrelated_tables_sharing_only_the_company_code_stay_apart(self):
        """Every table in an M3 install carries CONO. If it could establish an
        edge, an order header and an inventory balance would be one hop apart
        with "ON left.CONO = right.CONO", and the shortest-path search would
        happily route real join plans through it."""
        assert _edges({
            "DW.OOHEAD": {"CONO": "int", "ORNO": "varchar"},
            "DW.MITBAL": {"CONO": "int", "ITNO": "varchar"},
        }, self.M3) == {}

    def test_the_company_code_still_rides_along_on_a_real_edge(self):
        """It is part of the composite key, so once an order number has linked
        the pair, leaving it out of the ON clause would be wrong."""
        edges = _edges({
            "DW.OOHEAD": {"CONO": "int", "ORNO": "varchar"},
            "DW.OOLINE": {"CONO": "int", "ORNO": "varchar", "PONR": "int"},
        }, self.M3)
        assert edges["DW.OOHEAD"]["DW.OOLINE"] == [("ORNO", "ORNO"), ("CONO", "CONO")]

    def test_the_real_key_comes_first_so_a_condition_cap_cannot_drop_it(self):
        conditions = _edges({
            "DW.OOHEAD": {"CONO": "int", "ORNO": "varchar", "PONR": "int"},
            "DW.OOLINE": {"CONO": "int", "ORNO": "varchar", "PONR": "int"},
        }, self.M3)["DW.OOHEAD"]["DW.OOLINE"]
        assert conditions[-1] == ("CONO", "CONO")
        assert ("ORNO", "ORNO") in conditions[:-1]

    def test_the_company_code_is_declared_as_a_qualifier_not_a_key(self):
        assert "CONO" in self.M3.join_qualifier_codes
        assert "CONO" not in self.M3.join_key_codes


# ── No regression for the warehouse this is deployed on ─────────────────────


class TestTheDeployedConventionStillWorks:
    def test_a_dimension_key_still_links_a_fact_to_its_dimension(self):
        edges = _edges({
            "DW.CUS_ORD_IVC_FCT": {"CUS_DMS_KEY": "int", "AMT": "decimal"},
            "DW.CUS_DMS": {"CUS_DMS_KEY": "int", "CUS_NM": "varchar"},
        }, _pack_vocab("infor_m3"))
        assert edges["DW.CUS_ORD_IVC_FCT"]["DW.CUS_DMS"] == [("CUS_DMS_KEY", "CUS_DMS_KEY")]

    def test_it_works_with_no_pack_selected_at_all(self):
        """A tenant with no pack chosen and no naming profile detected falls
        back to the builtin vocabulary, and must not lose every join it has."""
        edges = _edges({
            "DW.CUS_ORD_IVC_FCT": {"CUS_DMS_KEY": "int"},
            "DW.CUS_DMS": {"CUS_DMS_KEY": "int"},
        }, _clone_builtin())
        assert edges["DW.CUS_DMS"]["DW.CUS_ORD_IVC_FCT"] == [("CUS_DMS_KEY", "CUS_DMS_KEY")]

    def test_the_m3_spelling_is_declared_in_the_m3_pack(self):
        assert "_DMS_KEY" in _pack_vocab("infor_m3").join_key_suffixes


# ── The default set is generic, and narrow on purpose ────────────────────────


class TestTheDefaultSuffixesAreUnambiguous:
    @pytest.mark.parametrize("suffix", ["_SK", "_FK", "_DIM_KEY", "_DIMENSION_KEY"])
    def test_every_default_suffix_says_key_to_a_dimension(self, suffix):
        assert suffix in _clone_builtin().join_key_suffixes

    @pytest.mark.parametrize("suffix", ["_ID", "_KEY", "_CODE", "_NO"])
    def test_the_ambiguous_ones_are_left_to_the_pack(self, suffix):
        """A bare _ID or _KEY is a surrogate key on one warehouse and a sort
        key, hash key or natural attribute on another. A warehouse that wants
        them says so in its own pack."""
        assert suffix not in _clone_builtin().join_key_suffixes

    def test_an_id_column_does_not_link_two_tables_by_default(self):
        assert _edges({
            "DW.F_ORDER": {"BATCH_ID": "int"},
            "DW.F_SHIPMENT": {"BATCH_ID": "int"},
        }, _clone_builtin()) == {}

    def test_but_a_warehouse_that_uses_id_can_say_so(self):
        vocab = _clone_builtin()
        _merge_pack(vocab, {"pack_id": "p", "join_key_suffixes": ["_ID"]}, "p")
        assert _edges({
            "PHARMA.F_RX_FILL": {"PATIENT_ID": "int"},
            "PHARMA.D_PATIENT": {"PATIENT_ID": "int"},
        }, vocab)["PHARMA.F_RX_FILL"] == {"PHARMA.D_PATIENT": [("PATIENT_ID", "PATIENT_ID")]}


# ── The packs on disk ────────────────────────────────────────────────────────


class TestEveryPackDeclaresItsJoinVocabulary:
    PACK_IDS = sorted(p.stem for p in PACKS_DIR.glob("*.json"))

    def test_the_packs_are_the_ones_we_think_they_are(self):
        assert "infor_m3" in self.PACK_IDS
        assert len(self.PACK_IDS) >= 6

    @pytest.mark.parametrize("pack_id", PACK_IDS)
    def test_a_pack_states_something_about_joins(self, pack_id):
        """A pack that says nothing leaves its clients on the builtin default,
        which knows only the conventions generic enough to be safe everywhere.
        Every pack should say what its own ERP does."""
        pack = json.loads((PACKS_DIR / f"{pack_id}.json").read_text(encoding="utf-8"))
        declared = (
            pack.get("join_key_codes")
            or pack.get("join_qualifier_codes")
            or pack.get("join_key_suffixes")
        )
        assert declared, f"{pack_id} declares no join vocabulary"

    @pytest.mark.parametrize("pack_id", PACK_IDS)
    def test_no_code_is_both_a_key_and_a_qualifier(self, pack_id):
        pack = json.loads((PACKS_DIR / f"{pack_id}.json").read_text(encoding="utf-8"))
        keys = {str(c).upper() for c in (pack.get("join_key_codes") or [])}
        qualifiers = {str(c).upper() for c in (pack.get("join_qualifier_codes") or [])}
        assert not (keys & qualifiers)

    @pytest.mark.parametrize("pack_id", PACK_IDS)
    def test_a_declared_suffix_actually_looks_like_a_suffix(self, pack_id):
        pack = json.loads((PACKS_DIR / f"{pack_id}.json").read_text(encoding="utf-8"))
        for suffix in pack.get("join_key_suffixes") or []:
            assert str(suffix).startswith("_"), f"{pack_id}: {suffix!r}"
            assert str(suffix) == str(suffix).upper(), f"{pack_id}: {suffix!r}"
