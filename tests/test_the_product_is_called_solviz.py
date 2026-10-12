"""The product is called Solviz in everything a person reads.

QueryBot was renamed. Every page, the translation catalogue, the chat platforms' messages, emails, the AI's
instructions (it says its name to readers) and the names of downloaded files say Solviz. Names a running
server depends on are not the product's name and stay: the QUERYBOT_* settings, the querybot.db store, the
service, class and module names, browser storage keys, the Snowflake query tag the history reader leaves out,
and the X-QueryBot-Export-ID header. Code comments are not read by the people who use it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = "QueryBot"
PACKAGES = ("admin", "core", "core2", "gateway", "portal", "store")

# Strings that keep the old name on purpose, with why.
KEPT = {
    "core/snowflake_auth.py": {"QueryBot"},                          # QUERY_TAG: older statements carry it
    "core2/bootstrap/history.py": {"QueryBot"},                      # ...and history leaves those out
    "portal/routes.py": {"X-QueryBot-Export-ID"},                    # a header clients may read
}


def _docstrings(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def _python_files():
    for package in PACKAGES:
        yield from sorted((ROOT / package).rglob("*.py"))
    yield ROOT / "main.py"


def test_no_string_in_the_code_says_the_old_name():
    found = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstrings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip \
                    and OLD in node.value:
                kept = KEPT.get(rel, set())
                if not any(k == node.value or (k == OLD and f"'{OLD}'" in node.value) for k in kept):
                    found.append(f"{rel}:{node.lineno}: {node.value[:80]!r}")
    assert not found, "\n".join(found)


@pytest.mark.parametrize("folder", ["portal/templates", "admin/templates", "static/js"])
def test_no_page_or_script_says_the_old_name(folder):
    found = [f"{p.relative_to(ROOT)}:{n}" for p in sorted((ROOT / folder).rglob("*"))
             if p.suffix in {".html", ".js"}
             for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if OLD in line]
    assert not found, found


def test_the_whole_catalogue_says_solviz():
    from core import i18n

    found = [(key, lang) for key, entry in i18n.MESSAGES.items() for lang, text in entry.items() if OLD in text]
    assert not found, found
    assert i18n.t("ui.title.suffix", lang="en") == "Solviz Portal"
    assert i18n.t("ui.title.suffix", lang="fr") == "Portail Solviz"


def test_an_email_carries_the_name_and_comes_from_it():
    from core.brand import PRODUCT_NAME
    from core.mail import render, settings

    assert PRODUCT_NAME == "Solviz"
    html = render.page("A title", ["A paragraph."])
    assert ">Solviz</td>" in html and OLD not in html and "Query<span" not in html
    assert settings.MailSettings().from_name == "Solviz"


def test_the_api_is_titled_solviz():
    import main

    assert main.app.title == "Solviz"


def test_a_download_is_named_for_the_product():
    from core.export import build_csv_filename

    assert build_csv_filename("") == "solviz_result.csv"
    chat = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
    assert "'solviz-results.csv'" in chat and "querybot-results" not in chat
