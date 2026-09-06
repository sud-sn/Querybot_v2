# -*- coding: utf-8 -*-
"""A download must not 500 because of a letter in the data.

The portal's CSV export builds its filename from the reader's question and put
it straight into a Content-Disposition header. An HTTP header is latin-1, so a
question naming a Škoda or a Łukasz — ordinary values in a European dataset —
raised UnicodeEncodeError when the response was encoded, and the reader got a
500 where they had asked for a file. French accents did not crash but arrived
as mojibake, because a raw non-ASCII byte in `filename=` is decoded
inconsistently across browsers.

The portal route had also drifted: core/export.py already owned filename
construction and the route inlined its own slug with `str.isalnum()`, which is
Unicode-aware in exactly the wrong direction — it KEEPS "Š".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.export import (  # noqa: E402
    ascii_filename,
    build_csv_filename,
    content_disposition,
)

# Values a European dataset really contains, plus the shapes that break a header.
HOSTILE = [
    "querybot_ventes_Škoda.csv",
    "querybot_commandes_Łukasz.csv",
    "querybot_売上_par_region.csv",
    "querybot_ventes_à_Zürich.csv",
    "querybot_chiffre_d_affaires_en_€.csv",
    "querybot_Ω_omega.csv",
    'querybot_a"b.csv',
    "querybot_back\\slash.csv",
    "querybot_line\r\nbreak.csv",
    "",
    "....",
    "querybot_result.csv",
]


class TestTheHeaderAlwaysEncodes(unittest.TestCase):

    def test_no_filename_can_make_the_response_fail_to_encode(self):
        """This is the 500. Starlette encodes header values as latin-1."""
        for name in HOSTILE:
            with self.subTest(name=name):
                content_disposition(name).encode("latin-1")

    def test_the_old_construction_really_did_fail(self):
        # Proves the tests above discriminate rather than passing on anything.
        question = "ventes Škoda par région"
        slug = "".join(c if c.isalnum() else "_" for c in question[:40].lower())
        with self.assertRaises(UnicodeEncodeError):
            f'attachment; filename="querybot_{slug.strip("_")}.csv"'.encode("latin-1")

    def test_a_header_never_contains_a_newline(self):
        # A newline in a header value is response splitting.
        for name in HOSTILE:
            value = content_disposition(name)
            self.assertNotIn("\n", value, name)
            self.assertNotIn("\r", value, name)

    def test_the_quoted_form_is_never_ended_early(self):
        value = content_disposition('querybot_a"b.csv')
        quoted = value.split('filename="', 1)[1].split('"', 1)[0]
        self.assertNotIn('"', quoted)
        self.assertNotIn("\\", quoted)

    def test_a_hostile_fallback_cannot_inject_a_header(self):
        """The fallback is a caller's string and reaches the header.

        It is used on exactly the path where the reader's own name folded away
        to nothing, so it is the one input nobody looks at — and it used to
        skip the filter the name goes through.
        """
        hostile = 'bad"name\r\nX-Evil: 1'
        folded = ascii_filename("", fallback=hostile)
        self.assertNotIn('"', folded)
        self.assertNotIn("\r", folded)
        self.assertNotIn("\n", folded)

        value = content_disposition("", fallback=hostile)
        value.encode("latin-1")
        self.assertEqual(value.count('"'), 2)
        self.assertNotIn("\n", value)

    def test_a_fallback_that_is_itself_unusable_still_yields_a_name(self):
        self.assertTrue(ascii_filename("", fallback='"'))
        self.assertTrue(ascii_filename("売上.csv", fallback="...").endswith(".csv"))


class TestTheReaderStillGetsTheirName(unittest.TestCase):

    def test_the_full_name_survives_in_the_utf8_parameter(self):
        # RFC 6266: current browsers prefer filename*, so the accents are kept.
        value = content_disposition("querybot_ventes_à_Zürich.csv")
        self.assertIn("filename*=UTF-8''", value)
        self.assertIn("%C3%A0", value)   # à
        self.assertIn("%C3%BC", value)   # ü

    def test_accents_fold_to_their_base_letter_in_the_ascii_form(self):
        # "region" is readable; dropping the letter entirely is not.
        self.assertEqual(ascii_filename("querybot_région.csv"), "querybot_region.csv")
        self.assertEqual(ascii_filename("querybot_Škoda.csv"), "querybot_Skoda.csv")

    def test_the_extension_survives_a_stem_that_folds_away(self):
        # Otherwise the browser is handed an extensionless file it will not open.
        self.assertTrue(ascii_filename("querybot_売上.csv").endswith(".csv"))
        self.assertTrue(ascii_filename("売上.csv").endswith(".csv"))

    def test_a_name_that_folds_to_nothing_falls_back(self):
        for name in ("", "....", "売上"):
            self.assertTrue(ascii_filename(name), name)

    def test_a_plain_ascii_name_is_left_alone(self):
        self.assertEqual(ascii_filename("querybot_revenue_by_region.csv"),
                         "querybot_revenue_by_region.csv")


class TestTheExportRouteUsesTheSharedBuilder(unittest.TestCase):

    def test_the_portal_no_longer_inlines_its_own_slug(self):
        source = (Path(__file__).resolve().parents[1]
                  / "portal" / "routes.py").read_text(encoding="utf-8")
        self.assertNotIn("c if c.isalnum() else", source)
        self.assertIn("build_csv_filename(", source)
        self.assertIn("content_disposition(", source)

    def test_every_download_header_goes_through_the_helper(self):
        import re

        for path in ("portal/routes.py", "admin/routes.py"):
            source = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
            for match in re.finditer(r'"Content-Disposition":\s*(.+)', source):
                value = match.group(1).strip()
                self.assertTrue(
                    "content_disposition(" in value or "_download_header(" in value
                    or value in ("", "\\"),
                    f"{path}: raw header value {value[:60]}",
                )

    def test_the_shared_builder_and_the_header_agree(self):
        # Write API to read API: the name the builder produces is the name the
        # header carries, folded.
        name = build_csv_filename("ventes Škoda par région")
        value = content_disposition(name)
        self.assertIn(ascii_filename(name), value)
        value.encode("latin-1")


if __name__ == "__main__":
    unittest.main()
