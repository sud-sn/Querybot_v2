"""A grouping whose members are places: US states, Canadian provinces, countries, US ZIP codes.

Read from the members themselves, with the grouping's own name as the deciding word where the members alone
could be something else ("CA" is California, Canada, or a status code): never from the warehouse or the
question. The regions a map can draw, and every way each is written ("California", "CA", "US-CA"), are in
static/geo/names.json (tools/geo/build_geo.py); the ZIP codes with a point, in static/geo/us-zip.csv.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_GEO = Path(__file__).resolve().parents[2] / "static" / "geo"
_NAMES = _GEO / "names.json"
# The map each kind is drawn on (static/geo/), a ZIP code as a point on the states.
FILES = {"us_state": "us-states.json", "ca_province": "ca-provinces.json", "country": "countries.json",
         "us_zip": "us-states.json"}
_HINTS = {"us_state": {"state", "states"}, "ca_province": {"province", "provinces", "state"},
          "country": {"country", "countries", "nation"}}
_ZIP_WORDS = {"zip", "zipcode", "zip_code", "postal", "postcode"}
_ZIP = re.compile(r"(\d{3,5})(?:-\d{4})?")


@lru_cache(maxsize=1)
def _index() -> dict[str, dict[str, str]]:
    """Each kind's spellings, folded, to the region name the map knows it by."""
    try:
        names = json.loads(_NAMES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {kind: {spelling.casefold(): name for name, spellings in regions.items() for spelling in spellings}
            for kind, regions in names.items()}


@lru_cache(maxsize=1)
def _zips() -> frozenset[str]:
    """The US ZIP codes the map has a point for."""
    try:
        lines = (_GEO / "us-zip.csv").read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return frozenset()
    return frozenset(line.split(",", 1)[0] for line in lines if line)


def zip5(value: str) -> str:
    """A ZIP code as five digits: "02101-1234" -> "02101", and one kept as a number -> "2101" -> "02101"."""
    found = _ZIP.fullmatch(str(value).strip())
    return found.group(1).zfill(5) if found else str(value).strip()


def off_map_zips(values: list[str]) -> list[str]:
    """The members that are no ZIP code the map has a point for (a PO box's own code, a typo)."""
    known = _zips()
    return [v for v in values if v and v != "Unknown" and zip5(v) not in known]


def _words(*texts: str) -> set[str]:
    return {w for t in texts for w in re.split(r"[^a-z0-9]+", (t or "").casefold()) if w}


def place_kind(label: str, column: str, values: list[str]) -> str | None:
    """Which map the members ``values`` of a grouping are places on, or None.

    A ZIP code is a grouping named for one (zip, postal code) whose members are four in five US ZIP codes --
    German and French postal codes are five digits too, and most of them are none. A state, a
    province or a country is one whose members are four in five that map's regions, named for one ("State",
    "Customer country"); or, unnamed, nearly all of them written out in full ("California"), never codes
    alone."""
    distinct = {v.strip() for v in values if v and v.strip() and v.strip() != "Unknown"}
    if len(distinct) < 3:
        return None
    words = _words(label, column)
    if words & _ZIP_WORDS:
        return "us_zip" if len(off_map_zips(sorted(distinct))) <= 0.2 * len(distinct) else None
    best: tuple[float, str] | None = None
    for kind, spellings in _index().items():
        rate = sum(v.casefold() in spellings for v in distinct) / len(distinct)
        named = bool(words & _HINTS.get(kind, set()))
        written_out = all(len(v) > 3 for v in distinct)
        if (named and rate >= 0.8) or (written_out and rate >= 0.95):
            if best is None or rate > best[0]:
                best = (rate, kind)
    return best[1] if best else None


def regions(kind: str, values: list[str]) -> dict[str, str]:
    """Each member as the map names its region ("CA" -> "California"); a member it has none for is left out."""
    spellings = _index().get(kind, {})
    return {v: spellings[v.strip().casefold()] for v in values if v and v.strip().casefold() in spellings}
