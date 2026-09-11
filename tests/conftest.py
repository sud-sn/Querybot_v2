"""Cross-test isolation for the one piece of state that is not a fixture.

`core.i18n.activate_language` sets a ContextVar that stays set until it is
deactivated, and it is deliberately process-wide: the pipeline activates the
reader's language once and every bare `_t()` for the rest of that answer picks
it up. In a test process that same design makes a language a test leaves behind
the next module's default, and the failure lands somewhere else entirely -- a
test asserting an English phrase, three modules later, reading French.

That happened while this branch was being written: a new test looped over
SUPPORTED_LANGUAGES with a bare `activate_language(lang)`, left "fr" active, and
broke `test_the_provenance_says_it_is_not_an_approved_default` in another file.
Nothing was wrong with either test on its own, and both passed when run alone.

So the check is here, once, as an assertion rather than a silent reset: a reset
would let the leak keep happening invisibly, and the leak is the defect. Use a
scoped activation (`activate_language` paired with `deactivate_language`, or a
`with` helper) or pass `lang=` explicitly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_language_leaks_between_tests():
    yield
    from core import i18n

    active = i18n.get_active_language()
    if active not in (None, "", "en"):
        i18n.activate_language("en")
        pytest.fail(
            f"this test left {active!r} as the process-wide active language, so "
            "the next test's bare _t() calls render in it -- activate the "
            "language in a scope that ends (deactivate_language with the token "
            "activate_language returned) or pass lang= explicitly"
        )
