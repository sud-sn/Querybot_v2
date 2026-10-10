"""Build the map files the charts draw from (static/geo/), from public sources downloaded by hand.

    python tools/geo/build_geo.py <folder with the downloads>

The folder holds:
  admin1_50m.geojson   Natural Earth 1:50m admin-1 states and provinces (public domain)
                       github.com/nvkelso/natural-earth-vector  geojson/ne_50m_admin_1_states_provinces.geojson
  admin0_110m.geojson  Natural Earth 1:110m admin-0 countries (public domain)
                       geojson/ne_110m_admin_0_countries.geojson
  uscities.json        US ZIP codes with a latitude and longitude (MIT licence)
                       github.com/millbj92/US-Zip-Codes-JSON  USCities.json

Written: us-states.json and ca-provinces.json (each region's name and code, its outline at 0.01 degree),
countries.json (the same, by ISO code), us-zip.csv ("zip,lat,lon"), and names.json, the names and codes
of every region by map, which the answer reads to tell a grouping that is a place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "static" / "geo"
# Everyday names Natural Earth writes otherwise ("U.K.", "Czechia").
ALIASES = {"United Kingdom": ["UK", "Great Britain", "Britain"], "United States of America": ["USA", "U.S.",
           "U.S.A.", "United States", "America"], "Czechia": ["Czech Republic"], "Côte d'Ivoire": ["Ivory Coast"],
           "Dem. Rep. Congo": ["DR Congo", "DRC", "Democratic Republic of the Congo"], "Eswatini": ["Swaziland"],
           "North Macedonia": ["Macedonia"], "Myanmar": ["Burma"], "Timor-Leste": ["East Timor"]}


def _round(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 2), round(coords[1], 2)]
    out = [_round(c) for c in coords]
    if out and isinstance(out[0][0], (int, float)):        # a ring: drop points that rounded onto the last
        kept = [out[0]]
        for point in out[1:]:
            if point != kept[-1]:
                kept.append(point)
        return kept if len(kept) >= 4 else out
    return out


def _feature(name: str, code: str, geometry: dict, **extra) -> dict:
    return {"type": "Feature", "properties": {"name": name, "code": code, **extra},
            "geometry": {"type": geometry["type"], "coordinates": _round(geometry["coordinates"])}}


def main(folder: str) -> None:
    src = Path(folder)
    OUT.mkdir(parents=True, exist_ok=True)
    admin1 = json.loads((src / "admin1_50m.geojson").read_text(encoding="utf-8"))
    names: dict[str, dict[str, list[str]]] = {"us_state": {}, "ca_province": {}, "country": {}}
    for iso, key, file in (("US", "us_state", "us-states.json"), ("CA", "ca_province", "ca-provinces.json")):
        features = []
        for f in admin1["features"]:
            p = f["properties"]
            if p.get("iso_a2") != iso:
                continue
            name, code = p["name"], p["postal"]
            geometry = f["geometry"]
            if code == "HI" and geometry["type"] == "MultiPolygon":
                # The islands north-west of Kauai (to Midway) are left off, as on most maps of the states: drawn,
                # they stretch the Hawaii inset a thousand miles west.
                geometry = {"type": "MultiPolygon", "coordinates": [
                    poly for poly in geometry["coordinates"] if max(pt[0] for pt in poly[0]) > -161]}
            features.append(_feature(name, code, geometry))
            spellings = {name, code, p.get("name_en") or name, p.get("iso_3166_2") or code}
            if name == "Québec":
                spellings |= {"Quebec"}
            names[key][name] = sorted(s for s in spellings if s)
        (OUT / file).write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                           separators=(",", ":")), encoding="utf-8")
    admin0 = json.loads((src / "admin0_110m.geojson").read_text(encoding="utf-8"))
    features = []
    for f in admin0["features"]:
        p = f["properties"]
        if p["NAME"] in ("Antarctica", "Fr. S. Antarctic Lands"):
            continue                      # no business is there; drawn, it takes a fifth of the map
        code = p.get("ISO_A3_EH") if p.get("ISO_A3") in (None, "-99") else p["ISO_A3"]
        name = p["NAME"]
        features.append(_feature(name, code or name, f["geometry"]))
        spellings = {p.get(k) for k in ("NAME", "NAME_LONG", "ADMIN", "NAME_EN", "FORMAL_EN", "ISO_A2", "ISO_A3",
                                        "ISO_A2_EH", "ISO_A3_EH", "ABBREV")}
        spellings |= set(ALIASES.get(name, []))
        names["country"][name] = sorted(s for s in spellings if s and s != "-99")
    (OUT / "countries.json").write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                                   separators=(",", ":")), encoding="utf-8")
    zips = json.loads((src / "uscities.json").read_text(encoding="utf-8"))
    lines = sorted({f"{int(z['zip_code']):05d},{round(float(z['latitude']), 2)},{round(float(z['longitude']), 2)}"
                    for z in zips if z.get("latitude") not in (None, "") and z.get("longitude") not in (None, "")})
    (OUT / "us-zip.csv").write_text("zip,lat,lon\n" + "\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "names.json").write_text(json.dumps(names, ensure_ascii=False, sort_keys=True, indent=0),
                                    encoding="utf-8")
    for path in sorted(OUT.iterdir()):
        print(f"{path.name:22s} {path.stat().st_size:>9,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
