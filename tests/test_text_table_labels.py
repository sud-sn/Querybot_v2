"""
tests/test_text_table_labels.py

The plain-text table, which for two of the three channels IS the answer.

_rows_to_table builds the table that goes to Teams and to /api/ask — readers
who get no chart and no HTML at all. It printed the warehouse's own spelling in
every header:

    WHS_NM  | BAL_VAL_AMT
    --------+------------
    Halifax | $1,200.00

The browser's <th> for the same result has said "Warehouse Name" since the L4
work, and the commit that closed that family claimed "every surface that names
a column now asks display_label". This one did not. It is a separate function
330 lines above the branch that renders it — the else branch only interpolates
`table_text`, which is computed here — so a fix aimed at the render site could
never have reached it.

The projected-rows note underneath had the second defect: an English literal
with a hand-rolled plural ("s are" / " is"), in the one channel where that
sentence is the only thing distinguishing a forecast row from a measured one.
"""
from __future__ import annotations

import unittest

from core import i18n
from core.result_renderer import _rows_to_table


ROWS = [{"WHS_NM": "Halifax", "BAL_VAL_AMT": 1200.0},
        {"WHS_NM": "Toronto", "BAL_VAL_AMT": 900.0}]


def _table(rows, lang="en"):
    token = i18n.activate_language(lang)
    try:
        return _rows_to_table(rows)
    finally:
        i18n.deactivate_language(token)


class TestTheHeadersAreBusinessNames(unittest.TestCase):

    def test_no_raw_column_code_reaches_the_reader(self):
        table = _table(ROWS)
        self.assertNotIn("WHS_NM", table)
        self.assertNotIn("BAL_VAL_AMT", table)

    def test_the_business_name_is_there_instead(self):
        """Not merely absent — replaced. Without this, a header that rendered
        as an empty string would satisfy the test above."""
        table = _table(ROWS)
        self.assertIn("Warehouse Name", table)
        self.assertIn("Balance Value Amount", table)

    def test_the_columns_still_line_up_under_the_longer_names(self):
        """A business name is longer than the code it replaces, and the widths
        were computed from the code. A table whose separator is shorter than
        its header is visibly broken in a monospaced Teams message."""
        lines = _table(ROWS).splitlines()
        header, separator = lines[0], lines[1]
        self.assertEqual(len(header), len(separator))
        for row in lines[2:4]:
            with self.subTest(row=row):
                self.assertEqual(len(row), len(header))

    def test_the_values_are_still_under_their_own_columns(self):
        lines = _table(ROWS).splitlines()
        offset = lines[0].index("Balance Value Amount")
        self.assertTrue(lines[2][offset:].startswith("$1,200.00"))

    def test_an_empty_result_is_unchanged(self):
        self.assertEqual(_rows_to_table([]), "(no results)")


class TestTheProjectedNoteSpeaksTheReadersLanguage(unittest.TestCase):
    """This sentence is the ONLY thing separating a forecast row from a
    measured one in a text channel — the marker columns are hidden."""

    def _with_forecasts(self, n):
        rows = [dict(r) for r in ROWS]
        for row in rows[-n:]:
            row["is_forecast"] = True
        return rows

    def test_it_appears_at_all(self):
        # The control for the two language tests below.
        self.assertIn("projected", _table(self._with_forecasts(1)))

    def test_and_not_when_nothing_is_projected(self):
        self.assertNotIn("projected", _table(ROWS))

    def test_a_french_reader_gets_it_in_french(self):
        note = _table(self._with_forecasts(1), "fr")
        self.assertIn("projetée", note)
        self.assertNotIn("projected", note)

    def test_the_plural_is_the_catalogues_not_a_hand_rolled_suffix(self):
        """Was `"s are" if projected != 1 else " is"` — which is English
        grammar hard-coded into a formatter, and gives "1 rows are" the moment
        anyone edits the comparison."""
        one = _table(self._with_forecasts(1), "en")
        two = _table(self._with_forecasts(2), "en")
        self.assertIn("row is projected", one)
        self.assertIn("rows are projected", two)

    def test_french_pluralises_too(self):
        one = _table(self._with_forecasts(1), "fr")
        two = _table(self._with_forecasts(2), "fr")
        self.assertIn("dernière ligne est projetée", one)
        self.assertIn("dernières lignes sont projetées", two)


if __name__ == "__main__":
    unittest.main()
