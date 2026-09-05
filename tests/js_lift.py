"""
tests/js_lift.py

Lift a whole function or object literal out of a template, by brace balance.

Not a test module. Shared by every suite that EXECUTES template JavaScript
instead of asserting on its source text -- a character window stops covering
the code it names the moment anything is inserted above it.

One difference from the older copy in tests/test_stage_trail.py, and it
matters. That version takes the first "{" after the signature, which is the
BODY only when no parameter contains a brace. `_pickerError(message, {focus =
null} = {})` has a destructured default, so the first brace is the parameter's
and the balance closes on it -- lifting one line of a function and handing the
engine a syntax error. Walk the parameter list by paren balance first, then
take the brace after it.
"""

from __future__ import annotations


def function(source: str, signature: str) -> str:
    """The whole function whose declaration starts with `signature`.

    Works for `function name(...)` and for `window.name = function (...)`
    alike: both are located by the signature, then by the parentheses and
    braces that follow it.
    """
    start = source.index(signature)
    depth = 0
    body_start = None
    for i in range(source.index("(", start), len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                body_start = source.index("{", i)
                break
    if body_start is None:
        raise AssertionError(f"unbalanced parentheses in {signature!r}")
    depth = 0
    for i in range(body_start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def const_block(source: str, name: str) -> str:
    """A `const NAME = {...}` object literal, by brace balance."""
    start = source.index(f"const {name} = {{")
    depth = 0
    for i in range(source.index("{", start), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1] + ";"
    raise AssertionError(f"unbalanced braces after {name!r}")
