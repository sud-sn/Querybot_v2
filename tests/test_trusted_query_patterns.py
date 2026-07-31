from unittest.mock import patch

from core.vector_store import QdrantKBRetriever, upsert_kb_file


def test_raw_query_pattern_document_is_never_upserted():
    with patch("core.vector_store._qdrant") as qdrant:
        upsert_kb_file(
            "acct",
            "DB.PUBLIC.SALES",
            "queries",
            "Q: total sales\nSQL: SELECT SUM(amount) FROM sales",
            source_file="SALES_queries.md",
        )

    qdrant.assert_not_called()


def test_fact_pattern_retrieval_uses_only_governed_global_documents():
    retriever = object.__new__(QdrantKBRetriever)
    captured = {}

    def fake_search(query, n, allowed_fqns, doc_types):
        captured["doc_types"] = doc_types
        return [{"doc_type": "global", "content": "Use this JOIN relationship"}]

    retriever._hybrid_search = fake_search

    docs = retriever.retrieve_fact_patterns("sales by region")

    assert captured["doc_types"] == ["global"]
    assert docs == ["Use this JOIN relationship"]
