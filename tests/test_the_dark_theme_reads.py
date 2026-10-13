"""
The dark theme reads: the same pairs the light palette is held to, computed
from tokens-dark.css over tokens.css.

A dark theme is the one a retune breaks without anyone looking: the page is
checked in light, and a link that read on white is a smudge on the night
ground. tokens-dark.css redefines tokens by name; whatever it leaves alone
keeps its light value, so every pair below reads the merged palette a dark
page actually gets.

WCAG 2.2: 4.5:1 for text, 7:1 where the token promises body-grade ink.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS = Path(__file__).resolve().parents[1] / "static" / "css"
GROUNDS = ["--paper-recessed", "--paper-sunken", "--paper", "--surface-alt", "--surface", "--surface-raised", "--window"]


def _clean(css: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", css)


def _declarations(body: str) -> dict[str, str]:
    return {k: v.strip() for k, v in re.findall(r"(--[\w-]+):\s*([^;]+);", body)}


def _dark_blocks() -> dict[str, dict[str, str]]:
    css = _clean((CSS / "tokens-dark.css").read_text(encoding="utf-8"))
    return {name: _declarations(body) for name, body in
            re.findall(r':root\[data-theme="(dark|auto)"\]\s*\{([^}]*)\}', css)}


@pytest.fixture(scope="module")
def light() -> dict:
    return _declarations(_clean((CSS / "tokens.css").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def tok(light) -> dict:
    merged = dict(light)
    merged.update(_dark_blocks()["dark"])
    return {k: v.upper() for k, v in merged.items() if re.fullmatch(r"#[0-9a-fA-F]{6}", v)} | \
           {k: v for k, v in merged.items() if v.startswith("linear-gradient(")}


def _lum(hexcolour: str) -> float:
    channels = [int(hexcolour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    high, low = sorted((_lum(a), _lum(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _worst_ground(tok: dict, ink: str) -> float:
    return min(_ratio(tok[ink], tok[ground]) for ground in GROUNDS)


class TestTheSheet:

    def test_it_names_only_tokens_the_light_palette_has(self, light):
        """A name only the dark sheet defines is a colour no light page has:
        a rule written against it shows nothing in the light theme."""
        for theme, declarations in _dark_blocks().items():
            unknown = sorted(set(declarations) - set(light))
            assert not unknown, (theme, unknown)

    def test_auto_is_the_dark_theme(self):
        blocks = _dark_blocks()
        assert blocks["dark"] == blocks["auto"]

    def test_every_ground_and_ink_turns_over(self, light, tok):
        """A ground or an ink left at its light value is a white card or
        black text on the night page."""
        for name in GROUNDS + ["--bg", "--text", "--text-strong", "--text-secondary", "--text-muted",
                               "--line", "--line-subtle", "--sidebar-bg", "--shell", "--code-bg"]:
            assert tok[name] != light[name].upper(), name

    def test_the_grounds_lift_a_step_per_layer(self, tok):
        """The page lies lowest and a raised card highest, so a card reads as
        on top of the page without a shadow to say so."""
        ladder = [_lum(tok[g]) for g in ["--ground", "--paper-recessed", "--paper", "--surface", "--surface-raised"]]
        assert ladder == sorted(ladder) and len(set(ladder)) == len(ladder), ladder
        assert ladder[-1] < 0.03


class TestText:

    @pytest.mark.parametrize("ink", ["--text-strong", "--text", "--text-secondary"])
    def test_body_text_is_body_grade_on_every_ground(self, tok, ink):
        assert _worst_ground(tok, ink) >= 7.0

    @pytest.mark.parametrize("ink", ["--text-muted", "--text-soft"])
    def test_secondary_text_reads_on_every_ground(self, tok, ink):
        assert _worst_ground(tok, ink) >= 4.5

    def test_the_tooltip_ink_reads_on_its_ground(self, tok):
        assert _ratio(tok["--text-inverse"], tok["--navy"]) >= 7.0


class TestThePrimary:

    def test_as_a_link_on_every_ground(self, tok):
        assert _worst_ground(tok, "--primary") >= 4.5

    def test_as_body_grade_ink_on_every_ground(self, tok):
        assert _worst_ground(tok, "--primary-ink") >= 7.0

    def test_its_text_on_its_fill(self, tok):
        assert _ratio(tok["--on-primary"], tok["--primary-fill"]) >= 4.5
        assert _ratio(tok["--on-primary"], tok["--primary-fill-hover"]) >= 4.5

    def test_its_text_on_every_stop_of_the_action_gradient(self, tok):
        stops = re.findall(r"#[0-9a-fA-F]{6}", tok["--action-gradient"])
        assert len(stops) >= 2
        for stop in stops:
            assert _ratio(tok["--on-primary"], stop) >= 4.5, stop

    def test_on_a_selected_row(self, tok):
        assert _ratio(tok["--primary-ink"], tok["--state-selected"]) >= 4.5


@pytest.mark.parametrize("family", ["success", "danger", "warning", "info"])
class TestStatus:

    def test_its_ink_reads_on_its_tint(self, tok, family):
        assert _ratio(tok[f"--{family}"], tok[f"--{family}-soft"]) >= 4.5

    def test_its_ink_reads_on_every_ground(self, tok, family):
        assert _worst_ground(tok, f"--{family}") >= 4.5

    def test_its_text_reads_on_its_fill(self, tok, family):
        assert _ratio(tok["--on-status"], tok[f"--{family}"]) >= 4.5


class TestTheShell:

    @pytest.mark.parametrize("ground", ["--sidebar-bg", "--sidebar-surface", "--sidebar-active"])
    def test_its_text_is_body_grade(self, tok, ground):
        assert _ratio(tok["--sidebar-text"], tok[ground]) >= 7.0

    @pytest.mark.parametrize("ink", ["--sidebar-muted", "--sidebar-accent", "--shell-muted", "--shell-accent"])
    def test_its_quieter_inks_read(self, tok, ink):
        for ground in ("--sidebar-bg", "--sidebar-surface", "--sidebar-active", "--shell"):
            assert _ratio(tok[ink], tok[ground]) >= 4.5, ground


class TestTheCodeWell:

    def test_its_text_is_body_grade(self, tok):
        assert _ratio(tok["--code-text"], tok["--code-bg"]) >= 7.0

    @pytest.mark.parametrize("token", ["--syntax-kw", "--syntax-fn", "--syntax-col", "--syntax-str", "--syntax-num"])
    def test_each_syntax_colour_reads(self, tok, token):
        assert _ratio(tok[token], tok["--code-bg"]) >= 4.5


@pytest.mark.parametrize("kind", ["fact", "dim", "bridge"])
def test_an_entity_chip_reads(tok, kind):
    assert _ratio(tok[f"--entity-{kind}"], tok[f"--entity-{kind}-soft"]) >= 4.5


def test_the_control_edge_is_seen(tok):
    """A text box's edge is how a reader finds it: 3:1 against the card it sits on."""
    for ground in ("--surface", "--surface-raised"):
        assert _ratio(tok["--line-control"], tok[ground]) >= 3.0, ground
