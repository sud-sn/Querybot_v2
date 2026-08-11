"""Infor M3 terminology-pack behavior and cross-client isolation."""

from core.date_roles import detect_date_role
from core.identifier_intelligence import analyze_identifier, detect_naming_profile
from core.naming_convention import match_table_suffix
from core.schema_enrichment import enrich_columns
from core.semantic_planner import build_semantic_field_plan
from core.source_resolution import resolve_source_scope
from core.vocab_packs import _clone_builtin, _merge_pack, builtin_vocab, load_pack


def _m3_vocab():
    vocab = _clone_builtin()
    _merge_pack(vocab, load_pack("infor_m3"), "infor_m3")
    return vocab


def test_record_prefixed_columns_resolve_to_governed_m3_meanings():
    vocab = _m3_vocab()
    expected = {
        "OBORNO": "order number",
        "OBORQA": "ordered quantity (alternate unit)",
        "MBWHLO": "warehouse",
        "MBSTQT": "on-hand approved quantity",
        "MMITDS": "item description",
        "MMUNMS": "basic unit of measure",
        "IBPUNO": "purchase order number",
    }
    for physical, meaning in expected.items():
        analysis = analyze_identifier(physical, vocab=vocab)
        assert analysis.expanded_name == meaning
        assert analysis.confidence == 95
        assert any("record prefix" in value for value in analysis.evidence)


def test_unknown_or_inactive_prefixes_fail_closed():
    assert analyze_identifier("ZZORNO", vocab=_m3_vocab()).confidence == 35
    plain = analyze_identifier("OBORNO", vocab=builtin_vocab())
    assert plain.confidence < 70
    assert plain.expanded_name != "order number"


def test_m3_enrichment_assigns_identifier_measure_and_date_roles():
    vocab = _m3_vocab()
    enriched = {item.column: item for item in enrich_columns(
        ["OBORNO", "OBORQA", "MBSTQT", "MMITDS", "MMSTAT", "OAORDT", "OBDWDT"],
        vocab=vocab,
    )}
    assert enriched["OBORNO"].role == "identifier"
    assert enriched["OBORQA"].role == "measure"
    assert enriched["MBSTQT"].role == "measure"
    assert enriched["MMITDS"].role == "attribute"
    assert enriched["MMSTAT"].role == "status_filter"
    assert enriched["OAORDT"].role == "date_key"
    assert enriched["OBDWDT"].role == "date_key"
    assert all(item.confidence == 95 for item in enriched.values())


def test_m3_prefixed_business_dates_have_specific_roles():
    vocab = _m3_vocab()
    assert detect_date_role("OAORDT", vocab=vocab).key == "order_date"
    assert detect_date_role("OBDWDT", vocab=vocab).key == "requested_delivery_date"
    assert detect_date_role("FSACDT", vocab=vocab).key == "accounting_date"
    assert detect_date_role("FSDUDT", vocab=vocab).key == "due_date"
    assert detect_date_role("MMLMDT", vocab=vocab).key == "modified_date"
    assert detect_date_role("OAORDT", vocab=builtin_vocab()) is None


def test_core_m3_operational_files_are_classified_from_pack_only():
    vocab = _m3_vocab()
    for table in ("OOHEAD", "OOLINE", "MITBAL", "MITTRA", "MPHEAD", "MPLINE", "FSLEDG"):
        assert match_table_suffix(table, vocab=vocab).table_type == "fact_table"
    for table in ("OCUSMA", "MITMAS", "CIDMAS", "CSYTAB"):
        assert match_table_suffix(table, vocab=vocab).table_type == "dimension_table"
    assert match_table_suffix("OOHEAD", vocab=builtin_vocab()) is None


def test_auto_detection_uses_real_prefixed_m3_schema_evidence():
    profile = detect_naming_profile(
        ["OACONO", "OAORNO", "OAORDT", "OBPONR", "OBITNO", "OBORQA"],
        ["OOHEAD", "OOLINE"],
    )
    assert profile["auto_applied_packs"] == ["infor_m3"]
    assert profile["pack_recommendations"][0]["record_prefixed_column_matches"]


def test_m3_table_business_terms_select_the_right_source():
    vocab = _m3_vocab()
    model = {"tables": [
        {"qualified_name": "M3.OOLINE", "table": "OOLINE", "type": "fact", "entity": "order", "grain": "line"},
        {"qualified_name": "M3.MITBAL", "table": "MITBAL", "type": "fact", "entity": "inventory", "grain": "warehouse item"},
    ]}
    orders = resolve_source_scope("show customer order lines by item", model, vocab=vocab)
    stock = resolve_source_scope("show current stock balance by warehouse", model, vocab=vocab)
    assert orders["status"] == "selected" and orders["selected_fact"] == "M3.OOLINE"
    assert stock["status"] == "selected" and stock["selected_fact"] == "M3.MITBAL"


def test_semantic_planner_matches_prefixed_m3_business_fields():
    vocab = _m3_vocab()
    columns = {
        "M3.OOLINE": {
            "OBORNO": "varchar",
            "OBORQA": "decimal",
            "OBITNO": "varchar",
            "OBORDT": "int",
        }
    }
    plan = build_semantic_field_plan(
        "total ordered quantity by order number",
        columns,
        None,
        vocab=vocab,
    )
    selected = {field["column"]: field["role"] for field in plan["fields"]}
    assert selected["OBORQA"] == "measure"
    assert selected["OBORNO"] == "dimension"
