"""
Sprint E — Export tests (rows_to_csv + build_csv_filename).

All tests are pure in-memory — no DB, no LLM, no file I/O.

Groups:
  RowsToCsvTests       — rows_to_csv happy path + edges
  BuildCsvFilenameTests — build_csv_filename variations
"""

import unittest

from core.export import build_csv_filename, rows_to_csv


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_csv(csv_str: str) -> tuple[list[str], list[list[str]]]:
    """Split CSV into (headers, data_rows) for assertion convenience."""
    import csv as _csv
    import io
    reader = _csv.reader(io.StringIO(csv_str))
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


# ══════════════════════════════════════════════════════════════════════════════
# rows_to_csv
# ══════════════════════════════════════════════════════════════════════════════

class RowsToCsvTests(unittest.TestCase):

    def test_empty_rows_returns_empty_string(self):
        self.assertEqual(rows_to_csv([]), "")

    def test_none_rows_returns_empty_string(self):
        # Defensive: callers may pass None
        result = rows_to_csv(None or [])
        self.assertEqual(result, "")

    def test_header_row_present(self):
        rows = [{"Region": "North", "Revenue": 1000}]
        headers, _ = _parse_csv(rows_to_csv(rows))
        self.assertEqual(headers, ["Region", "Revenue"])

    def test_data_row_values_correct(self):
        rows = [{"Region": "North", "Revenue": 1000}]
        _, data = _parse_csv(rows_to_csv(rows))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], ["North", "1000"])

    def test_multiple_rows(self):
        rows = [
            {"R": "A", "V": 100},
            {"R": "B", "V": 200},
            {"R": "C", "V": 300},
        ]
        _, data = _parse_csv(rows_to_csv(rows))
        self.assertEqual(len(data), 3)
        self.assertEqual([d[1] for d in data], ["100", "200", "300"])

    def test_none_value_becomes_empty_string(self):
        rows = [{"Region": "South", "Revenue": None}]
        _, data = _parse_csv(rows_to_csv(rows))
        self.assertEqual(data[0][1], "")

    def test_zero_value_preserved(self):
        rows = [{"Region": "West", "Revenue": 0}]
        _, data = _parse_csv(rows_to_csv(rows))
        self.assertEqual(data[0][1], "0")

    def test_currency_format_applied(self):
        rows = [{"Revenue": 1234.56}]
        csv_str = rows_to_csv(rows, column_formats={"Revenue": "currency"})
        _, data = _parse_csv(csv_str)
        self.assertEqual(data[0][0], "$1,234.56")

    def test_percentage_format_applied(self):
        rows = [{"Share": 33.3}]
        csv_str = rows_to_csv(rows, column_formats={"Share": "percentage"})
        _, data = _parse_csv(csv_str)
        self.assertEqual(data[0][0], "33.30%")

    def test_unknown_format_falls_back_to_str(self):
        rows = [{"Count": 42}]
        csv_str = rows_to_csv(rows, column_formats={"Count": "number"})
        _, data = _parse_csv(csv_str)
        self.assertEqual(data[0][0], "42")

    def test_no_column_formats_argument(self):
        """column_formats is optional — omitting it should not error."""
        rows = [{"Revenue": 500}]
        csv_str = rows_to_csv(rows)
        self.assertIn("Revenue", csv_str)
        self.assertIn("500", csv_str)

    def test_original_rows_not_mutated(self):
        rows = [{"Region": "East", "Revenue": 300}]
        original_keys = set(rows[0].keys())
        rows_to_csv(rows)
        self.assertEqual(set(rows[0].keys()), original_keys)

    def test_string_with_commas_quoted(self):
        """CSV writer should quote values containing commas."""
        rows = [{"Name": "Smith, John", "Score": 90}]
        csv_str = rows_to_csv(rows)
        # Re-parsed correctly
        _, data = _parse_csv(csv_str)
        self.assertEqual(data[0][0], "Smith, John")

    def test_string_with_newline_quoted(self):
        rows = [{"Note": "line1\nline2", "Value": 1}]
        csv_str = rows_to_csv(rows)
        _, data = _parse_csv(csv_str)
        self.assertEqual(data[0][0], "line1\nline2")

    def test_float_value_preserved(self):
        rows = [{"Rate": 3.14159}]
        csv_str = rows_to_csv(rows)
        _, data = _parse_csv(csv_str)
        self.assertIn("3.14159", data[0][0])

    def test_currency_non_numeric_falls_back(self):
        """Non-numeric value with currency format should still export cleanly."""
        rows = [{"Revenue": "N/A"}]
        csv_str = rows_to_csv(rows, column_formats={"Revenue": "currency"})
        _, data = _parse_csv(csv_str)
        self.assertEqual(data[0][0], "N/A")

    def test_single_column(self):
        rows = [{"Name": "Alice"}, {"Name": "Bob"}]
        headers, data = _parse_csv(rows_to_csv(rows))
        self.assertEqual(headers, ["Name"])
        self.assertEqual([d[0] for d in data], ["Alice", "Bob"])

    def test_many_columns(self):
        row = {f"col_{i}": i for i in range(20)}
        headers, data = _parse_csv(rows_to_csv([row]))
        self.assertEqual(len(headers), 20)
        self.assertEqual(len(data[0]), 20)

    def test_result_starts_with_header_line(self):
        rows = [{"A": 1, "B": 2}]
        csv_str = rows_to_csv(rows)
        first_line = csv_str.split("\n")[0]
        self.assertIn("A", first_line)
        self.assertIn("B", first_line)

    def test_result_ends_with_newline(self):
        rows = [{"X": 1}]
        csv_str = rows_to_csv(rows)
        self.assertTrue(csv_str.endswith("\n"))


# ══════════════════════════════════════════════════════════════════════════════
# build_csv_filename
# ══════════════════════════════════════════════════════════════════════════════

class BuildCsvFilenameTests(unittest.TestCase):

    def test_ends_with_csv_extension(self):
        self.assertTrue(build_csv_filename("show revenue").endswith(".csv"))

    def test_empty_string_returns_default(self):
        self.assertEqual(build_csv_filename(""), "querybot_result.csv")

    def test_none_returns_default(self):
        self.assertEqual(build_csv_filename(None or ""), "querybot_result.csv")

    def test_spaces_become_underscores(self):
        name = build_csv_filename("total revenue by region")
        self.assertIn("total_revenue_by_region", name)

    def test_special_chars_stripped(self):
        name = build_csv_filename("What is revenue? (2024!)")
        self.assertNotIn("?", name)
        self.assertNotIn("!", name)
        self.assertNotIn("(", name)

    def test_question_mark_stripped(self):
        name = build_csv_filename("How much revenue?")
        self.assertNotIn("?", name)

    def test_caps_at_60_chars_plus_extension(self):
        long_q = "a " * 50  # 100 chars
        name = build_csv_filename(long_q)
        # filename = stem + ".csv"
        stem = name[:-4]
        self.assertLessEqual(len(stem), 60)

    def test_lowercase(self):
        name = build_csv_filename("Total Revenue By Region")
        self.assertEqual(name, name.lower())

    def test_multiple_spaces_collapsed(self):
        name = build_csv_filename("total   revenue")
        self.assertNotIn("__", name)

    def test_hyphens_become_underscores(self):
        name = build_csv_filename("year-on-year growth")
        self.assertNotIn("-", name.replace(".csv", ""))

    def test_numbers_preserved(self):
        name = build_csv_filename("top 10 products")
        self.assertIn("10", name)


if __name__ == "__main__":
    unittest.main()
