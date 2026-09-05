"""
tests/chat_js.py

Execute portal_chat.html's own JavaScript in duktape.

Not a test module. The page's copy lives in a catalogue now, so a source-text
assertion ("'Blocked by policy' in template") stops meaning anything: it passes
against a page that resolves the id to nothing, and fails against a page that
resolves it correctly through a different spelling of the lookup.

The catalogue handed to the harness is the REAL one from core/i18n.py, which is
what makes these tests notice an id the page uses and the catalogue does not
have -- that renders as the raw id on a customer's screen.
"""

from __future__ import annotations

import json
from pathlib import Path

from core import i18n
from tests.browser_num import preamble as _num_preamble
from tests.js_lift import function as lift

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "portal" / "templates" / "portal_chat.html"


def source() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def const_expression(src: str, name: str) -> str:
    """A `const NAME = <expr>;` whose value spans lines, by paren balance."""
    start = src.index(f"const {name} = ")
    depth = 0
    for i in range(start, len(src)):
        ch = src[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                end = src.index(";", i)
                return src[start:end + 1]
    raise AssertionError(f"unbalanced expression after {name!r}")


def run(script: str, *, lang="en", functions=(), consts=(), preamble="") -> dict:
    """Evaluate `script` with the named page functions and consts in scope."""
    import dukpy

    src = source()
    lifted = "\n".join(
        [const_expression(src, name) for name in consts]
        + [lift(src, sig) for sig in functions]
    )
    harness = f"""
{_num_preamble(lang)}
window.QB_I18N = {json.dumps(i18n.catalogue_for(lang))};
const I18N = window.QB_I18N;
function t(id, vars){{
  let out = Object.prototype.hasOwnProperty.call(I18N, id) ? I18N[id] : id;
  if (vars) for (const k in vars) out = out.split('{{' + k + '}}').join(String(vars[k]));
  return out;
}}
function plural(stem, count, vars){{
  const n = Math.abs(Number(count));
  const one = {json.dumps(lang)} === 'fr' ? n < 2 : n === 1;
  const args = Object.assign({{count: count}}, vars || {{}});
  return t(stem + (one ? '.one' : '.other'), args);
}}
{preamble}

{lifted}

{script}
"""
    return json.loads(dukpy.evaljs(harness))
