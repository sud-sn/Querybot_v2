"""
tests/test_admin_template_safety.py

Cross-template lints for the admin console's generated markup.

These are static checks, deliberately. The suite has no JavaScript runtime, so
the behaviour behind each rule was confirmed in a real browser first and the rule
exists to stop it regressing — not to stand in for the browser. Where a check
could pass while the code is dead, it says so.
"""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "admin" / "templates"

BACKSLASH = chr(92)

# Matches markup built by concatenation in the form
#     onclick="someFn(\''+EXPRESSION+'\')"
# capturing the function name and the interpolated expression.
_INLINE_HANDLER = re.compile(
    r'onclick="([A-Za-z_$][\w$]*)\('
    + re.escape(BACKSLASH + "''")
    + r"\s*\+\s*(.+?)\s*\+\s*"
    + re.escape("'" + BACKSLASH + "')")
)


def _handler_sites():
    for template in sorted(TEMPLATES.rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for match in _INLINE_HANDLER.finditer(source):
            yield (
                template.name,
                source[: match.start()].count("\n") + 1,
                match.group(1),
                " ".join(match.group(2).split()),
            )


def test_a_value_interpolated_into_an_inline_handler_is_escaped_for_javascript():
    """escHtml() covers & < > and " but NOT the apostrophe, and the browser
    un-escapes an attribute before parsing it as JavaScript. So a value
    containing ' closes the handler's string literal early:

        onclick="insertColumnFromMcDialog('O'BRIEN_FLG')"

    which throws "missing ) after argument list" — and because the throw happens
    inside the browser's own attribute evaluation, the click simply does nothing.
    No console error the user sees, no visible failure, the field just refuses to
    insert.

    The metric dialog's Fields panel shipped this way; the saved-metrics list two
    functions below had always escaped correctly. Verified in a browser both
    ways: before the fix the click threw and the editor stayed empty, after it
    the raw attribute reads insertColumnFromMcDialog('O\\'BRIEN_FLG') and the
    name inserts intact.
    """
    sites = list(_handler_sites())
    assert sites, (
        "no interpolated inline handlers found at all — the pattern this guards "
        "has been rewritten, so update or delete this test rather than letting it "
        "pass vacuously"
    )

    bare = [
        f"{name}:{line} {fn}() <- {expr}"
        for name, line, fn, expr in sites
        if ".replace(/'/g" not in expr
    ]
    assert not bare, (
        "these interpolate a value into an inline handler without escaping it for "
        "the JavaScript context, so any value containing an apostrophe silently "
        "breaks the click:\n  " + "\n  ".join(bare)
    )
