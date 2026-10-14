"""A dashboard's "Add a filter" lists the workspace's fields, under a compliance profile too.

It listed only the fields whose values may be shown to the AI. A workspace under a compliance profile (healthcare,
banking) clears few fields for the AI, so the dialog said "No other field to filter by." The picker's values are
read for the person who opens the dashboard, by the same governed query as an answer listing that field, so the
AI's clearance is not the rule here: every text field that is not hidden, not personal or sensitive, and not a
document's number or free text may be offered.

A learned retail warehouse; the filter list as the dashboard route builds it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.test_core2_answers_in_the_portal import learned  # noqa: F401 - the fixture
from tests.test_core2_learned_page_shows_what_was_learned import (
    ACCOUNT,
    workspace,  # noqa: F401 - the fixture
)


def _model(store):
    from core2.bootstrap.service import load_model

    return load_model(ACCOUNT, int(store.get_client(ACCOUNT)["db_config_id"]))


def _offered(model) -> set[str]:
    from portal import routes

    with patch("core2.bootstrap.service.answering_model", return_value=model), \
            patch.object(routes.store, "get_allowed_tables", return_value=None):
        fields = routes._filterable_fields({"id": 1, "account_id": ACCOUNT}, [])
    return {f["field"] for f in fields}


def _a_text_dimension(model) -> str:
    from core2.plan.values import listable

    return sorted(listable(model))[0]


def test_a_field_not_cleared_for_the_ai_is_offered(learned):  # noqa: F811
    model = _model(learned[0])
    slug = _a_text_dimension(model)
    for column in model.columns.values():         # a compliance profile: no field's values reach the AI
        column.values_allowed = False
    from core2.plan.values import listable

    assert not listable(model)
    assert slug in _offered(model)


def test_personal_hidden_and_free_text_fields_are_not(learned):  # noqa: F811
    model = _model(learned[0])
    personal, hidden = sorted(_offered(model))[:2]
    model.columns[model.attributes[personal].column].sensitivity = "personal"
    model.columns[model.attributes[hidden].column].hidden = True
    offered = _offered(model)
    assert personal not in offered and hidden not in offered
    for slug in offered:
        assert model.attributes[slug].kind not in ("identifier", "text")
        assert model.columns[model.attributes[slug].column].data_type == "text"
