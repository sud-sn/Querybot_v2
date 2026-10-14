"""A person's name typed into a question never reaches the AI, with or without the optional name reader.

A workspace under compliance scrubbed its questions of emails, phone numbers and social security numbers; a person's
name ("why was John Smith's claim denied?") was found only by Presidio, an optional library the install leaves out,
and only for a pharmacy or a bank. On a default install every typed name went to the AI. Now, in every workspace
under compliance and every industry, a name-shaped phrase -- two or more capitalised words that are not the
workspace's own words (its fields, metrics and tables, months, question words), or a title and a name ("Dr Patel")
-- is replaced by [PERSON] before the question reaches the AI. Presidio, when installed, still runs first.

Invented names; nothing is sent anywhere.
"""

from __future__ import annotations

import pytest

from core.masking import scrub_typed_names
from tests.test_core2_answers_a_question_end_to_end import _domain

KNOWN = {"net", "amount", "customer", "type", "region", "store", "sales", "wholesale"}


@pytest.mark.parametrize("asked, gone", [
    ("net sales for John Smith in 2025", "John Smith"),
    ("Why was Mary-Jane O'Neil's claim denied?", "Mary-Jane O'Neil"),
    ("patients of Dr Patel by month", "Patel"),
    ("Show Avery Quill fills last year", "Avery Quill"),
    ('fills for "Blair Moss" in March', "Blair Moss"),
])
def test_a_name_shaped_phrase_is_withheld(asked, gone):
    said = scrub_typed_names(asked, KNOWN)
    assert gone not in said and "[PERSON]" in said, said


@pytest.mark.parametrize("asked", [
    "Net Amount by Customer Type in March and April 2025",
    "What were Wholesale sales in Q1 FY2026?",
    "sales by region for OH and TX",
    "Show me the top 5 stores",
])
def test_the_workspaces_own_words_months_and_codes_stay(asked):
    assert scrub_typed_names(asked, KNOWN) == asked


def test_every_compliance_workspace_scrubs_names_without_presidio(tmp_path, monkeypatch):
    import os

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from core import masking
    from core2.service import question_scrubber

    store.init_db()
    monkeypatch.setattr(masking, "_presidio_analyzer", False)      # the optional library is not installed
    _, _, model, _ = _domain("retail")
    for industry in ("healthcare_pharmacy", "banking", "standard"):
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        store.save_compliance_profile(account_id, mode="regulated", industry=industry)
        scrub = question_scrubber(account_id, model=model)
        said = scrub("net sales for John Smith by customer type in 2025")
        assert "John Smith" not in said and "net sales" in said and "customer type" in said, (industry, said)


def test_the_workspaces_own_model_is_read_when_none_is_given(tmp_path, monkeypatch):
    """The chat calls the scrubber with no model: it reads the workspace's answering model, so the workspace's own
    capitalised words are not taken for a person's name."""
    import os

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from core import masking
    from core2.bootstrap import service as bootstrap
    from core2.service import question_scrubber

    store.init_db()
    monkeypatch.setattr(masking, "_presidio_analyzer", False)
    _, _, model, _ = _domain("retail")
    monkeypatch.setattr(bootstrap, "answering_model", lambda account_id, client: model)
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.save_compliance_profile(account_id, mode="regulated", industry="healthcare_pharmacy")
    said = question_scrubber(account_id)("Net Amount by Customer Type for John Smith")
    assert said == "Net Amount by Customer Type for [PERSON]", said
