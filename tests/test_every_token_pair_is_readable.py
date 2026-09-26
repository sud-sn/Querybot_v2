"""
Every text colour the tokens define is readable on every ground it is used on.

tests/test_palette_structure.py holds the ladder, the body inks and the control
edge. The rest of the palette -- status chips, the primary as link and as fill,
the gradient's primary action, the navigation shell, the code well, the entity
chips, the selected row -- was checked once by hand when it was chosen, which
is the check the next retune skips. These compute each pair from tokens.css.

WCAG 2.2: 4.5:1 for text, 7:1 where the token promises body-grade ink.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TOKENS = Path(__file__).resolve().parents[1] / "static" / "css" / "tokens.css"
GROUNDS = ["--paper-recessed", "--paper-sunken", "--paper", "--surface-alt", "--surface", "--surface-raised"]


@pytest.fixture(scope="module")
def tok() -> dict:
    css = re.sub(r"/\*[\s\S]*?\*/", "", TOKENS.read_text(encoding="utf-8"))
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6}|linear-gradient\([^;]+\))", css))


def _lum(hexcolour: str) -> float:
    channels = [int(hexcolour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    high, low = sorted((_lum(a), _lum(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _worst_ground(tok: dict, ink: str) -> float:
    return min(_ratio(tok[ink], tok[ground]) for ground in GROUNDS)


@pytest.mark.parametrize("family", ["success", "danger", "warning", "info"])
class TestStatus:

    def test_its_ink_reads_on_its_tint(self, tok, family):
        assert _ratio(tok[f"--{family}"], tok[f"--{family}-soft"]) >= 4.5

    def test_its_pressed_ink_reads_on_its_tint(self, tok, family):
        assert _ratio(tok[f"--{family}-strong"], tok[f"--{family}-soft"]) >= 4.5

    def test_its_ink_reads_on_every_ground(self, tok, family):
        assert _worst_ground(tok, f"--{family}") >= 4.5

    def test_white_reads_on_its_ink(self, tok, family):
        assert _ratio("#FFFFFF", tok[f"--{family}"]) >= 4.5


class TestThePrimary:

    def test_as_a_link_on_every_ground(self, tok):
        assert _worst_ground(tok, "--primary") >= 4.5

    def test_as_body_grade_ink_on_every_ground(self, tok):
        assert _worst_ground(tok, "--primary-ink") >= 7.0

    def test_its_text_on_its_fill(self, tok):
        assert _ratio(tok["--on-primary"], tok["--primary"]) >= 4.5
        assert _ratio(tok["--on-primary"], tok["--primary-hover"]) >= 4.5

    def test_its_text_on_every_stop_of_the_action_gradient(self, tok):
        stops = re.findall(r"#[0-9a-fA-F]{6}", tok["--action-gradient"])
        assert len(stops) >= 2
        for stop in stops:
            assert _ratio(tok["--on-primary"], stop) >= 4.5, stop

    def test_on_a_selected_row(self, tok):
        assert _ratio(tok["--primary-ink"], tok["--state-selected"]) >= 4.5


class TestTheShell:

    @pytest.mark.parametrize("ground", ["--shell", "--shell-surface", "--shell-selected"])
    def test_its_text_is_body_grade_on_each_of_its_grounds(self, tok, ground):
        assert _ratio(tok["--shell-text"], tok[ground]) >= 7.0

    @pytest.mark.parametrize("ground", ["--shell", "--shell-surface", "--shell-selected"])
    def test_its_muted_text_reads_on_each_of_its_grounds(self, tok, ground):
        assert _ratio(tok["--shell-muted"], tok[ground]) >= 4.5

    def test_its_accent_reads_on_it(self, tok):
        assert _ratio(tok["--shell-accent"], tok["--shell"]) >= 4.5


class TestTheCodeWell:

    def test_its_text_is_body_grade(self, tok):
        assert _ratio(tok["--code-text"], tok["--code-bg"]) >= 7.0

    @pytest.mark.parametrize("token", ["--syntax-kw", "--syntax-fn", "--syntax-col", "--syntax-str", "--syntax-num"])
    def test_each_syntax_colour_reads(self, tok, token):
        assert _ratio(tok[token], tok["--code-bg"]) >= 4.5


@pytest.mark.parametrize("kind", ["fact", "dim", "bridge"])
def test_an_entity_chip_reads(tok, kind):
    assert _ratio(tok[f"--entity-{kind}"], tok[f"--entity-{kind}-soft"]) >= 4.5


class TestTheIdentity:
    """The approved direction, held: a cool navy cast in the greys, an azure
    brand ramp under its own name, the gradient defined once, and a navy
    shell -- so a retune cannot drift back to the old palette unnoticed."""

    def test_the_brand_ramp_is_azure(self, tok):
        for step in (400, 500, 600, 700):
            r, g, b = (int(tok[f"--accent-{step}"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            assert b > g > r, f"--accent-{step} is not azure"

    def test_the_primary_is_the_ramp(self, tok):
        assert tok["--primary"] == tok["--accent-600"]

    def test_the_shell_is_navy(self, tok):
        r, g, b = (int(tok["--shell"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        assert b > g > r and _lum(tok["--shell"]) < 0.03

    def test_the_gradient_is_defined_once_for_each_use(self, tok):
        assert tok["--brand-gradient"].startswith("linear-gradient(")
        assert tok["--action-gradient"].startswith("linear-gradient(")
