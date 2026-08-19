"""
tests/test_portal_chart_palettes.py

The chart series palettes existed twice: once in portal_chat.html and once in
portal_dashboard.html, kept in step by a comment in the dashboard that read
"matches portal_chat.html". Six palettes, two modes, eight colours apiece,
byte-identical in both files — so tuning one and not the other would have drawn
the same series in different colours on adjacent pages, with nothing to catch
it. They live in static/js/chart-palettes.js now.

The values themselves stay literal hex on purpose. ECharts paints to a canvas
and cannot resolve var(), and a categorical series ramp is a designed
qualitative set rather than a semantic colour — so these are NOT a
design-token violation, and a test that demanded tokens here would be wrong.
What matters is that there is one copy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "static" / "js" / "chart-palettes.js"
CONSUMERS = ("portal_chat.html", "portal_dashboard.html")

_HEX_ARRAY = re.compile(r"\[(?:\s*'#[0-9a-fA-F]{6}'\s*,?)+\]")


@pytest.fixture(scope="module")
def shared_source() -> str:
    assert SHARED.exists(), "the shared palette file is gone"
    return SHARED.read_text(encoding="utf-8")


def _consumer(name: str) -> str:
    return (ROOT / "portal" / "templates" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", CONSUMERS)
def test_no_page_carries_its_own_copy_of_the_palettes(name):
    source = _consumer(name)
    arrays = _HEX_ARRAY.findall(source)
    assert not arrays, (
        f"{name} declares {len(arrays)} palette array(s) again; there must be "
        f"exactly one copy, in static/js/chart-palettes.js"
    )


@pytest.mark.parametrize("name", CONSUMERS)
def test_each_page_loads_the_shared_file_before_reading_it(name):
    source = _consumer(name)
    tag = source.find('src="/static/js/chart-palettes.js')
    assert tag != -1, f"{name} does not load the shared palette file"

    use = source.find("window.QB_PALETTES")
    assert use != -1, f"{name} never reads the shared palettes"
    assert tag < use, (
        f"{name} reads window.QB_PALETTES before the script that defines it"
    )


def test_the_shared_file_defines_what_the_pages_expect(shared_source):
    assert "window.QB_PALETTES" in shared_source

    names = re.findall(r"^\s{2}(\w+):\s*\{", shared_source, re.M)
    assert set(names) >= {"default", "ocean", "sunset", "forest", "candy", "mono"}, names

    # Every palette needs both modes: a categorical ramp tuned for white does
    # not survive being dropped on the dark chart surface.
    for palette in names:
        block = re.search(rf"{palette}:\s*\{{(.*?)\}}", shared_source, re.S)
        assert block, palette
        assert "light:" in block.group(1), f"{palette} has no light mode"
        assert "dark:" in block.group(1), f"{palette} has no dark mode"


def test_the_validation_rationale_travelled_with_the_values(shared_source):
    """The specific hexes were picked against OKLCH and CVD checks, and two
    palettes ship as documented exceptions. Losing that note would invite
    someone to 'tidy' the values back into failing ones."""
    for marker in ("OKLCH", "CVD", "sunset", "forest"):
        assert marker in shared_source, (
            f"the palette rationale no longer mentions {marker}"
        )
