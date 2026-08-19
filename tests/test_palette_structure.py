"""
tests/test_palette_structure.py

The light palette was rebuilt because it had no working structure, not because
anyone disliked the colours. Measured on the version this replaces:

  --border            1.30:1 on --surface, against a 3:1 floor for a graphic —
                      every card edge and every input outline in the product
  ground separation   1.06-1.08 across four layers, and not even ordered:
                      --bg sat ABOVE --surface-2
  grey ramp           steps of 2.8/4.9/9.0/19.4/18.3/10.6/9.4/8.6/6.7, with
                      chroma dipping then rising, so the mid-greys were the
                      most brand-tinted objects on screen
  brand ramp          chroma jumped 50.6 to 77.8 between teal-400 and teal-500,
                      and teal-800/900 were near-neutral darks that only ever
                      existed to serve the dark theme

These tests pin the structure rather than the taste. A value can be retuned; a
ramp that stops being a ramp, a ground ladder that stops being ordered, or a
control edge that drifts back under 3:1 is the palette failing at its job.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "static" / "css" / "tokens.css"

# Grounds, ordered darkest to lightest by design.
GROUNDS = ["--paper-recessed", "--paper-sunken", "--paper",
           "--surface-alt", "--surface", "--surface-raised"]


@pytest.fixture(scope="module")
def tok() -> dict:
    css = re.sub(r"/\*[\s\S]*?\*/", "", TOKENS.read_text(encoding="utf-8"))
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", css))


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def _lin(v):
    v /= 255
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

def _lum(h):
    r, g, b = _rgb(h)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

def _ratio(a, b):
    x, y = sorted((_lum(a), _lum(b)), reverse=True)
    return (x + 0.05) / (y + 0.05)

def _hls(h):
    return colorsys.rgb_to_hls(*[c / 255 for c in _rgb(h)])


def test_the_ground_ladder_is_actually_a_ladder(tok):
    """Elevation is carried by the ground itself. If the layers are not ordered,
    nothing on the page looks like it sits on anything."""
    lums = [(name, _lum(tok[name])) for name in GROUNDS]
    for (an, a), (bn, b) in zip(lums, lums[1:]):
        assert b > a, f"{bn} is not lighter than {an}: the ladder is out of order"


def test_body_ink_clears_seven_to_one_on_every_ground(tok):
    """A floor solved against the brightest ground is not a floor. The darkest
    ground is where it has to hold."""
    for ink, need in (("--text-strong", 7.0), ("--text", 7.0),
                      ("--text-secondary", 4.5), ("--text-muted", 4.5)):
        worst = min(_ratio(tok[ink], tok[g]) for g in GROUNDS)
        assert worst >= need, (
            f"{ink} reaches only {worst:.2f} on the darkest ground it sits on "
            f"(needs {need})"
        )


def test_a_control_edge_is_a_visible_boundary(tok):
    """--border measured 1.30:1, so every input outline was below the 3:1 floor
    WCAG sets for a graphic. A field the user has to find in order to click it
    is a boundary, not decoration."""
    worst = min(_ratio(tok["--line-control"], tok[g]) for g in GROUNDS)
    assert worst >= 3.0, f"--line-control is {worst:.2f} on its worst ground"


def test_the_control_edge_token_is_actually_used(tok):
    """The failure mode the split invites: keep the token, route everything to
    the decorative rung anyway, and the WCAG failure returns while the palette
    looks fixed."""
    css = (ROOT / "static" / "css" / "base.css").read_text(encoding="utf-8")
    block = re.search(
        r"\.form-group input[^{]*\{[^}]*\}", css, re.S)
    assert block, "the form control rule is gone"
    assert "var(--line-control)" in block.group(0), (
        "form controls are back on the decorative border rung"
    )


def test_the_neutral_ramp_steps_evenly(tok):
    """The shipped ramp stepped 2.8 to 19.4. That is a set of hand-picked
    values, and it is why the greys looked arbitrary."""
    ramp = [tok[f"--gray-{n}"] for n in (50, 100, 200, 300, 400, 500, 600, 700, 800, 900)]
    steps = [abs(_hls(a)[1] - _hls(b)[1]) * 100 for a, b in zip(ramp, ramp[1:])]
    spread = max(steps) - min(steps)
    assert spread < 1.0, (
        f"ramp steps vary by {spread:.1f} lightness "
        f"({min(steps):.1f} to {max(steps):.1f}); it is not an even ramp"
    )


def test_the_neutral_ramp_keeps_its_green_cast_at_every_step(tok):
    """'Green-cast, not slate' is the identity argument. Without a test it is a
    comment in a CSS file, and the first person to add a grey will reach for
    slate."""
    for n in (50, 100, 200, 300, 400, 500, 600, 700, 800, 900):
        r, g, b = _rgb(tok[f"--gray-{n}"])
        assert g >= r and g >= b, (
            f"--gray-{n} has lost the green cast (r={r} g={g} b={b})"
        )


def test_the_brand_ramp_has_no_chroma_break(tok):
    """Chroma jumped 50.6 to 77.8 between teal-400 and teal-500, which reads as
    two ramps stitched together."""
    ramp = [tok[f"--teal-{n}"] for n in (50, 100, 200, 300, 400, 500, 600, 700, 800, 900)]
    sats = [_hls(h)[2] * 100 for h in ramp]
    jumps = [abs(a - b) for a, b in zip(sats, sats[1:])]
    assert max(jumps) < 30, (
        f"chroma jumps {max(jumps):.1f} between adjacent brand steps"
    )


def test_the_brand_ramp_is_brand_all_the_way_down(tok):
    """--teal-800 and --teal-900 used to be near-neutral darks: they existed to
    serve the dark theme, not the brand."""
    for n in (700, 800, 900):
        sat = _hls(tok[f"--teal-{n}"])[2] * 100
        assert sat > 25, (
            f"--teal-{n} has saturation {sat:.1f} and is a neutral, not a brand step"
        )


def test_hover_is_ground_aware(tok):
    """One hover value against six grounds is arithmetically impossible. The
    single value that shipped measured 1.004 against --paper — invisible — and
    was LIGHTER than --paper-recessed, so a hovered row in a table header band
    would have brightened instead of darkening. Row hover is the most-used
    interaction in a product whose main output is a table."""
    pairs = {
        "--hover-on-surface": "--surface",
        "--hover-on-alt": "--surface-alt",
        "--hover-on-paper": "--paper",
        "--hover-on-recessed": "--paper-recessed",
    }
    for hover, ground in pairs.items():
        assert hover in tok, f"{hover} is missing"
        assert _lum(tok[hover]) < _lum(tok[ground]), (
            f"{hover} is lighter than {ground}: hover would brighten the row"
        )
        assert _ratio(tok[hover], tok[ground]) >= 1.03, (
            f"{hover} is imperceptible against {ground}"
        )


def test_selection_differs_from_hover_by_more_than_brightness(tok):
    """Otherwise hovered and selected are the same gesture at two brightnesses
    and neither reads."""
    sel = _hls(tok["--state-selected"])
    hov = _hls(tok["--hover-on-surface"])
    assert sel[2] - hov[2] > 0.10, (
        "selected and hover differ only in lightness; separate them by chroma"
    )


def test_no_dark_surface_survives_in_a_light_only_product(tok):
    """--code-bg was #12211D, a near-black slab in six places, and the SQL
    syntax colours are tuned for a light ground. The navigation shell is the
    deliberate exception: it is chrome framing the page, not content."""
    CHROME = {"--shell", "--shell-surface", "--shell-line", "--shell-selected",
              "--sidebar-bg", "--sidebar-surface", "--sidebar-border",
              "--sidebar-active", "--navy", "--shadow-tint", "--scrim",
              "--text-strong", "--text", "--text-secondary", "--text-muted",
              "--text-faint", "--line-control", "--border-strong"}
    for name, value in tok.items():
        if name in CHROME or name.startswith(("--gray-", "--teal-", "--syntax-",
                                              "--entity-", "--primary", "--blue",
                                              "--success", "--danger", "--warning",
                                              "--info", "--green", "--red",
                                              "--amber", "--violet", "--code-text",
                                              "--focus-ring")):
            continue
        if name.endswith(("-strong", "-line", "-ink", "-mid", "-soft")):
            continue
        assert _lum(value) > 0.25, (
            f"{name} = {value} is a dark surface in a light-only product"
        )
