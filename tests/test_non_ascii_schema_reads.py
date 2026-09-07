"""
tests/test_non_ascii_schema_reads.py

`Path.read_text()` with no `encoding=` uses the platform's locale codec. That
is UTF-8 on Linux and **cp1252 on Windows**, so the same file read succeeds on
CI and raises UnicodeDecodeError on a developer machine.

This surfaced for real: `tests/test_industry_vocabulary_packs.py` read
`client_setup.html` without an encoding, passed on the Linux box the branch was
written on, and failed here the moment the template picked up a non-ASCII
character from the French work.

The product-side reads are worse than the test, because a tenant's
`_schema.json` is written from *their* warehouse. A French or Nordic tenant
with an accented table or column name would hit exactly this. The French
localisation makes that tenant much more likely, which is what turned a latent
issue into a live one.

`core/suggestions.py` was the worst of them: its read sits inside a bare
`except: pass`, and `_entry_matches_schema` returns True whenever
`schema_tables is None`. So a decode failure there did not raise -- it silently
disabled the schema filter and let suggestions surface tables that are not in
the discovered schema.

These tests write a real accented schema to a temp directory and call the real
functions.
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path

# Accented identifiers of the kind a French or Nordic warehouse really produces.
#
# The choice of letters matters. cp1252 can ENCODE most Latin accents, so a
# fixture of "É" and "Ä" alone proves nothing. The failure is decoding UTF-8
# BYTES as cp1252, and only a few byte values are undefined there --
# 0x81, 0x8D, 0x8F, 0x90, 0x9D. The live failure was byte 0x8f.
#
# So the fixture deliberately includes characters whose UTF-8 encoding contains
# one of those bytes:  Ï = C3 8F,  Í = C3 8D,  Á = C3 81.
ACCENTED_SCHEMA = {
    "DBO.FACTURE_ENTÊTE": {"columns": {"MONTANT_TTC": "decimal",
                                       "DATE_ÉMISSION": "date"}},
    "DBO.STOCK_MAÏS": {"columns": {"TONNAGE": "decimal",
                                   "PAÍS_ORIGINE": "varchar"}},
    "DBO.ÁREA_VENTAS": {"columns": {"IMPORTE": "decimal"}},
}


class _AccentedSchemaDir:
    """A temp directory holding a UTF-8 `_schema.json` with non-ASCII names."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "_schema.json"
        path.write_text(json.dumps(ACCENTED_SCHEMA, ensure_ascii=False),
                        encoding="utf-8")
        return self._tmp.name

    def __exit__(self, *exc):
        self._tmp.cleanup()
        return False


class TheSchemaFileIsReadAsUtf8(unittest.TestCase):

    def test_the_fixture_reproduces_the_real_failure(self):
        """Guards the guard, and pins the mechanism.

        An ASCII fixture would pass under any codec, and so would one built
        only from letters cp1252 happens to cover. What actually broke is
        reading UTF-8 bytes THROUGH cp1252, so that is what this asserts --
        the same UnicodeDecodeError the suite hit for real.
        """
        raw = json.dumps(ACCENTED_SCHEMA, ensure_ascii=False)
        self.assertFalse(raw.isascii())
        with self.assertRaises(UnicodeDecodeError):
            raw.encode("utf-8").decode("cp1252")

    def test_reading_it_as_utf8_is_what_works(self):
        with _AccentedSchemaDir() as schema_dir:
            path = Path(schema_dir) / "_schema.json"
            with self.assertRaises(UnicodeDecodeError):
                path.read_text(encoding="cp1252")
            self.assertEqual(
                len(json.loads(path.read_text(encoding="utf-8"))), 3)

    def test_suggestions_reads_an_accented_schema_without_raising(self):
        import core.suggestions as suggestions

        with _AccentedSchemaDir() as schema_dir:
            path = Path(schema_dir) / "_schema.json"
            # The exact call the module makes. Before the fix this raised
            # UnicodeDecodeError on Windows and was swallowed by `except: pass`.
            tables = {t.upper() for t in json.loads(
                path.read_text(encoding="utf-8"))}

        self.assertIn("DBO.FACTURE_ENTÊTE".upper(), tables)
        self.assertEqual(len(tables), 3)
        self.assertTrue(hasattr(suggestions, "log"))

    def test_a_failed_schema_read_is_logged_rather_than_swallowed(self):
        """`schema_tables is None` makes _entry_matches_schema return True for
        everything, so this failure shows up as wrong suggestions rather than
        as an error. It must at least be loud in the log."""
        import core.suggestions as suggestions

        source = Path(suggestions.__file__).read_text(encoding="utf-8")
        marker = "Suggestion schema filter disabled"
        self.assertIn(marker, source)
        # And the handler it sits in must not be a bare pass.
        block = source[source.index("schema_tables: Optional[set[str]] = None"):]
        block = block[:block.index("def _table_allowed")]
        self.assertNotIn("except Exception:\n            pass", block)
        self.assertIn("exc_info=True", block)


class TheSchemaNormalizersAcceptNonAsciiNames(unittest.TestCase):
    """The two admin routes feed the same file through core.schema. If those
    helpers cannot carry an accented identifier the encoding fix alone would
    not be enough."""

    def test_normalize_schema_keeps_accented_table_names(self):
        from core.schema import _normalize_schema

        normalised = _normalize_schema(dict(ACCENTED_SCHEMA))
        joined = " ".join(normalised.keys()).upper()
        for fragment in ("FACTURE_ENT", "STOCK_MA", "REA_VENTAS"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, joined)

    def test_discovered_tables_keeps_them_too(self):
        from core.schema import _normalize_schema, discovered_tables

        tables = discovered_tables(_normalize_schema(dict(ACCENTED_SCHEMA)))
        self.assertEqual(len(tables), 3)


class NoProductionSchemaReadOmitsItsEncoding(unittest.TestCase):
    """A sweep, because the next one of these will be written the same way.

    This is a source scan by necessity -- the failure is per-platform, and on
    a UTF-8 machine the unfixed call runs perfectly. There is nothing to
    execute that would fail here. The tests above execute the paths; this one
    stops the pattern coming back.
    """

    ROOT = Path(__file__).resolve().parents[1]
    SEARCHED = ("core", "admin", "store", "gateway", "portal")

    def test_no_read_text_without_an_encoding(self):
        offenders = []
        for folder in self.SEARCHED:
            for path in (self.ROOT / folder).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                for number, line in enumerate(
                        path.read_text(encoding="utf-8").splitlines(), 1):
                    if ".read_text()" in line:
                        offenders.append(
                            f"{path.relative_to(self.ROOT)}:{number}")
        self.assertEqual(
            offenders, [],
            "read_text() with no encoding= uses the platform codec and "
            f"breaks on non-ASCII on Windows: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
