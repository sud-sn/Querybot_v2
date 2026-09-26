"""
Build static/icons/qb-icons.svg, the product's one icon sprite, from Lucide.

The product names an icon by what it means ("refresh", "chat"); ICONS says
which Lucide drawing stands for each name. To add an icon, add its name here,
run the builder against an unpacked lucide-static package of LUCIDE_VERSION,
and commit the sprite it writes:

    npm pack lucide-static@1.48.0 && tar xzf lucide-static-1.48.0.tgz
    python tools/build_icon_sprite.py package

Templates draw an icon with the ic() macro (admin/templates/icons.html,
portal/templates/icons.html) and page scripts with qbIcon()
(static/js/qb-icons.js). Both reference a symbol in the sprite, so an icon is
drawn the same way everywhere and a page carries no path data of its own.
tests/test_icons_come_from_one_sprite.py fails when the sprite and this map
disagree, and when a page names an icon the sprite does not hold.

Lucide is ISC-licensed; the licence ships beside the sprite.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LUCIDE_VERSION = "1.48.0"

ICONS: dict[str, str] = {
    # Navigation and shell
    "dashboard": "layout-dashboard",
    "chat": "message-square",
    "database": "database",
    "plug": "plug",
    "users": "users",
    "user-group": "users-round",
    "settings": "settings",
    "menu": "menu",
    "globe": "globe",
    "bell": "bell",
    "log-out": "log-out",
    "inbox": "inbox",
    "moon": "moon",
    "sun": "sun",
    # Actions
    "search": "search",
    "diagnose": "scan-search",
    "plus": "plus",
    "x": "x",
    "trash": "trash-2",
    "pencil": "pencil",
    "copy": "copy",
    "download": "download",
    "refresh": "refresh-cw",
    "rotate": "rotate-cw",
    "play": "play",
    "pin": "pin",
    "filter": "funnel",
    "list-filter": "list-filter",
    "external-link": "external-link",
    "eye": "eye",
    "eye-off": "eye-off",
    "thumbs-up": "thumbs-up",
    "thumbs-down": "thumbs-down",
    "grip": "grip-vertical",
    # Direction
    "arrow-up": "arrow-up",
    "arrow-down": "arrow-down",
    "arrow-right": "arrow-right",
    "chevron-right": "chevron-right",
    "chevron-down": "chevron-down",
    "chevron-left": "chevron-left",
    "chevron-up": "chevron-up",
    # Status
    "check": "check",
    "check-circle": "circle-check",
    "x-circle": "circle-x",
    "alert-triangle": "triangle-alert",
    "alert-circle": "circle-alert",
    "info": "info",
    "clock": "clock",
    "lock": "lock",
    "key": "key",
    "loader": "loader-circle",
    # Data and meaning
    "chart-bar": "chart-column",
    "chart-line": "chart-line",
    "layout-grid": "layout-grid",
    "link": "link",
    "book": "book-open",
    "sparkles": "sparkles",
}

SPRITE = Path(__file__).resolve().parents[1] / "static" / "icons" / "qb-icons.svg"
_SVG = "{http://www.w3.org/2000/svg}"


def _symbol(name: str, drawing: Path) -> str:
    root = ET.fromstring(drawing.read_text(encoding="utf-8").split("-->", 1)[-1])
    shapes = []
    for child in root:
        tag = child.tag.replace(_SVG, "")
        attrs = " ".join(f'{k}="{v}"' for k, v in child.attrib.items())
        shapes.append(f"<{tag} {attrs}/>")
    return f'<symbol id="{name}" viewBox="0 0 24 24">{"".join(shapes)}</symbol>'


def build(package: Path) -> str:
    icons_dir = package / "icons"
    missing = [lucide for lucide in ICONS.values() if not (icons_dir / f"{lucide}.svg").is_file()]
    if missing:
        raise SystemExit(f"not in this lucide-static package: {', '.join(missing)}")
    symbols = "\n".join(_symbol(name, icons_dir / f"{lucide}.svg") for name, lucide in sorted(ICONS.items()))
    return (f"<!-- Lucide v{LUCIDE_VERSION} (ISC, see LICENSE-Lucide.txt), built by "
            "tools/build_icon_sprite.py. Do not edit by hand. -->\n"
            f'<svg xmlns="http://www.w3.org/2000/svg">\n{symbols}\n</svg>\n')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    SPRITE.parent.mkdir(parents=True, exist_ok=True)
    SPRITE.write_text(build(Path(sys.argv[1])), encoding="utf-8")
    print(f"{SPRITE}: {len(ICONS)} icons")
