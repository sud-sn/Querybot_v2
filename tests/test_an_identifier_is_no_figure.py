"""
An identifier is a label, not the figure.

A result that carries a customer's number beside the customer -- CUSTOMER,
SOLD_TO_NO, UNITS_SHIPPED, as a model-written query often returns it -- was
read with the number as its measure, the first numeric column: the card said
"ZED CO leads at 40,101", the summary "(33.4% of total)" of the customer
numbers, the analysis that "Sold To No is unusually uniform", and the signals
that the customer numbers and the units "move together". The chart already
read the number as an identifier; the sentences now do too
(core.response_builder._identifier_columns): where a name or a period
labels the rows, a column named for a number, a key, a code or a rank that
holds whole numbers is set aside -- no measure, and no second label.
"""

from __future__ import annotations

import pytest

from core import i18n
from core.response_builder import (
    _regulated_analysis_fallback,
    build_answer,
    build_assistant_response,
    infer_result_scope,
    summarize_result_context,
)
from core.stat_signals import compute_signals

SHIPPED = [{"CUSTOMER": customer, "SOLD_TO_NO": number, "UNITS_SHIPPED": units}
           for customer, number, units in (("ZED CO", 40101, 900.0), ("ACME", 40002, 200.0), ("BOLT", 40000, 100.0))]
QUESTION = "units shipped by customer"


def _headline(rows: list[dict], question: str = QUESTION) -> str:
    return build_answer(rows, question, infer_result_scope(rows, question, mode="ranking"))["headline"]


class TestTheFigure:

    def test_the_headline(self):
        assert _headline(SHIPPED) == "ZED CO leads at 900."

    def test_the_summary_and_its_callout(self):
        response = build_assistant_response(question=QUESTION, rows=SHIPPED, sql="SELECT * FROM t", duration_ms=1)
        assert response["insight_summary"] == "ZED CO leads at 900 (75.0% of total) across 3 customers."
        assert [callout["message"] for callout in response["anomaly_callouts"]] == [
            "ZED CO holds 75.0% of the total"]

    def test_the_context(self):
        context = summarize_result_context(SHIPPED, QUESTION, "SELECT * FROM t")
        assert (context["value_col"], context["label_col"]) == ("UNITS_SHIPPED", "CUSTOMER")

    def test_the_analysis_and_the_signals(self):
        body = _regulated_analysis_fallback("why", SHIPPED)["body"]
        assert body == ("ZED CO alone accounts for 75% of the total Units Shipped (900). Units Shipped ranges "
                        "from 100 to 900 across the result."), body
        labels = " ".join(signal["label"] for signal in compute_signals(SHIPPED))
        assert "SOLD_TO_NO" not in labels, labels

    @pytest.mark.parametrize("column", [
        "CUSTOMER_ID", "ORDER_NUMBER", "ITEM_KEY", "STORE_NBR", "SALES_RANK", "CustomerId", "SoldToNo",
        "Numéro de commande", "Code article",
        "CUSTOMER_NUM", "ORDER_NR", "ITEM_NUMERO", "CUSTOMER_SK", "ORDER_PK", "ITEM_FK", "ITEM_CD", "ITEM_CODE",
        "ORDER_REF", "LINE_SEQ", "CD_ITEM", "ID_CUSTOMER", "ADDRESS_NO",
        "ID", "KEY", "SK", "PK", "FK", "CODE", "CD", "REF", "SEQ", "RANK"])
    def test_whatever_it_is_called(self, column):
        rows = [{"CUSTOMER": customer, column: number, "UNITS_SHIPPED": units}
                for customer, number, units in (("ZED CO", 40101, 900.0), ("ACME", 40002, 200.0))]
        assert _headline(rows) == "ZED CO leads at 900."

    def test_no_breakdown_by_what_the_result_has(self):
        from core.response_builder import compute_chip_eligibility

        context = summarize_result_context(SHIPPED, QUESTION, "SELECT * FROM t")
        plan = {"enabled": True, "available_dimensions": [{"display_column": "SOLD_TO_NO", "name": "Sold-to number"}]}
        assert "drill_dim:Sold-to number" not in {chip["id"] for chip in compute_chip_eligibility(context, None, plan)}

    def test_in_french(self):
        token = i18n.activate_language("fr")
        try:
            assert _headline(SHIPPED, "unités expédiées par client").startswith("ZED CO arrive en tête avec 900")
        finally:
            i18n.deactivate_language(token)


class TestAnIdentifierNamedWithAMeasureWord:
    """A cost centre's id beside its net amount was read as the figure: COST_CENTER_ID
    4100, 4200, 4300 headed "OPS leads at 4,300", and PRICE_LIST_ID, RATE_CODE,
    TAX_RATE_ID, MARGIN_GROUP_CD, STAFF_NO and ID_STATUS the same. A measure word
    makes a name a figure's where it ends the name (CD_BALANCE) or stands right
    before the number word that does (PRICE_REF, NET_AMOUNT_NO); an id, a key or a
    code after it names a row of what stands between."""

    ROWS = (("SALES", 4100, 98000.75), ("OPS", 4300, 12000.5), ("ADMIN", 4200, 500.25))

    @pytest.mark.parametrize("column", [
        "COST_CENTER_ID", "COST_CENTRE_NO", "COST_CENTRE_ID", "PRICE_LIST_ID", "RATE_CODE", "TAX_RATE_ID",
        "MARGIN_GROUP_CD", "VALUE_STREAM_ID", "MIN_ORDER_ID", "ITEM_COST_ID", "AMOUNT_ID", "PRICE_KEY",
        "STAFF_NO", "STAFF_NUMBER", "ID_STATUS", "CODE_SERIES", "ID_SCANS", "CODE_REVIEWS", "CD_ACCOUNTS",
        "Cost Center ID", "CostCenterId", "CostCentreNo", "Code centre de coût",
        # A word that only ends or holds "count" is no count: an account, a county, a counter.
        "ACCOUNT_NO", "ACCOUNT_NUMBER", "GL_ACCOUNT_NO", "COUNTY_NO", "COUNTER_NO", "BANK_ACCOUNT_NBR"])
    def test_it_is_still_an_identifier(self, column):
        rows = [{"COST_CENTER_NAME": name, column: number, "NET_AMT": amount} for name, number, amount in self.ROWS]
        assert _headline(rows, "net amount by cost centre") == "SALES leads at 98,000.75."

    def test_a_ledger_accounts_number_beside_its_name(self):
        rows = [{"GL_ACCOUNT_NAME": name, "GL_ACCOUNT_NO": number, "NET_AMT": amount}
                for name, number, amount in (("Rent", 4300, 12000.0), ("Sales", 4100, 98000.0), ("Fees", 4200, 500.0))]
        assert _headline(rows, "net amount by account") == "Sales leads at 98,000."

    def test_the_summary_and_the_context_agree(self):
        rows = [{"COST_CENTER_NAME": name, "COST_CENTER_ID": number, "NET_AMT": amount}
                for name, number, amount in self.ROWS]
        context = summarize_result_context(rows, "net amount by cost centre", "SELECT * FROM t")
        assert (context["value_col"], context["label_col"]) == ("NET_AMT", "COST_CENTER_NAME")
        response = build_assistant_response(
            question="net amount by cost centre", rows=rows, sql="SELECT * FROM t", duration_ms=1)
        assert response["insight_summary"].startswith("SALES leads at 98,000.75")


class TestWhatIsStillAFigure:

    @pytest.mark.parametrize("column", ["NUMBER_OF_ORDERS", "NUM_UNITS", "ORDER_COUNT"])
    def test_a_count_named_for_a_number(self, column):
        rows = [{"CUSTOMER": "ZED CO", column: 12}, {"CUSTOMER": "ACME", column: 3}]
        assert _headline(rows) == "ZED CO leads at 12."

    @pytest.mark.parametrize("column", [
        "NUM_OF_RCT", "NO_OF_EMPLOYEES", "NUM_EMPLOYEES", "NUM_PRODUCTS", "VISITS_NUMBER", "TOTAL_NUMBER",
        "ID_COUNT", "NoOfStores", "Nombre de commandes", "WEEK_NO", "NUMBER",
        # A count or a figure of one, whatever else the name holds.
        "CUSTOMER_ID_CNT", "ORDER_NO_TOT", "SUM_ORDER_NO", "AVG_ORDER_NO", "AVERAGE_ORDER_NO", "NB_ORDER_NO",
        "NBRE_COMMANDE_NO", "NOMBRE_ID",
        # A period's number.
        "WEEKS_NO", "WK_NO", "DAY_NO", "DAYS_NO", "PERIOD_NO", "PER_NO", "PRD_NO", "MONTH_NO", "MTH_NO",
        "YEAR_NO", "YR_NO", "QUARTER_NO", "QTR_NO", "HOUR_NO", "HR_NO", "FISCAL_NO", "FY_NO", "DATE_NO", "DT_KEY",
        "SEMAINE_NO", "JOUR_NO", "MOIS_NO", "ANNEE_NO", "PERIODE_NO", "TRIMESTRE_NO",
        # A quantity, and a plural beside its number.
        "QTY_NO", "STOCK_ID", "INVENTORY_ID", "ORDERS_NUM", "ORDERS_NO", "ORDERS_NBR", "ORDERS_NR",
        "CHILDREN_NO", "HEADCOUNT_NO",
        # Money: "CD balance and sales by branch" lost its balance. A measure
        # word ends the name, or stands right before the number word that does.
        "CD_BALANCE", "CD_AMOUNT", "CD_BAL", "CD_AMT", "PRICE_REF", "AMOUNT_REF", "NET_AMOUNT_NO", "COST_REF",
        "PRICE_NO", "TAX_AMOUNT_NBR", "ID_AMOUNT", "CODE_PRICE"])
    def test_a_count_beside_another_figure(self, column):
        # "Top stores by number of employees" was answered with their sales.
        rows = [{"STORE": "NORTH", column: 60, "NET_SALES": 90000.0},
                {"STORE": "SOUTH", column: 45, "NET_SALES": 310000.0}]
        assert summarize_result_context(rows, "stores by number", "SELECT * FROM t")["value_col"] == column

    def test_a_number_with_a_fraction(self):
        rows = [{"CUSTOMER": "ZED CO", "SOLD_TO_NO": 40.5, "UNITS_SHIPPED": 900.0},
                {"CUSTOMER": "ACME", "SOLD_TO_NO": 12.25, "UNITS_SHIPPED": 200.0}]
        assert _headline(rows) == "ZED CO leads at 40.50."

    def test_a_number_with_a_fraction_in_any_row(self):
        rows = [{"CUSTOMER": "ZED CO", "SOLD_TO_NO": 40101, "UNITS_SHIPPED": 900.0},
                {"CUSTOMER": "ACME", "SOLD_TO_NO": 12.25, "UNITS_SHIPPED": 200.0}]
        assert summarize_result_context(rows, QUESTION, "SELECT * FROM t")["value_col"] == "SOLD_TO_NO"

    def test_the_only_number(self):
        # A rank asked for is the answer's figure: nothing else could be.
        rows = [{"STORE": "NORTH", "SALES_RANK": 1}, {"STORE": "SOUTH", "SALES_RANK": 2}]
        assert summarize_result_context(rows, "sales rank by store", "SELECT * FROM t")["value_col"] == "SALES_RANK"

    def test_a_rank_beside_the_months_it_ranks(self):
        # The months stay the series' axis; the rank is the only figure.
        rows = [{"PRD_KEY": period, "SALES_RANK": rank} for period, rank in ((202401, 2), (202402, 1), (202403, 3))]
        context = summarize_result_context(rows, "rank of each month by sales", "SELECT * FROM t")
        assert (context["value_col"], context["mode"]) == ("SALES_RANK", "time_series")


class TestWhatAnIdentifierStaysOutOf:

    def test_a_listing_of_records(self):
        # One customer's invoices, read as a ranking: "ZED CO leads at 267".
        rows = [{"CUSTOMER": "ZED CO", "INVOICE_NO": 5001 + index, "INVOICE_AMT": amount}
                for index, amount in enumerate((40.0, 55.0, 60.0, 50.0, 62.0))]
        sql = "SELECT CUSTOMER, INVOICE_NO, INVOICE_AMT FROM INVOICES WHERE CUSTOMER = 'ZED CO'"
        assert summarize_result_context(rows, "list the invoices for zed co", sql).get("listing") is True

    def test_an_axis_where_nothing_else_labels_the_rows(self):
        # Store numbers 2001, 2014, 2033 read as years "trended down 40.0%".
        rows = [{"STORE_NO": number, "NET_SALES": sales} for number, sales in ((2001, 500.0), (2014, 900.0),
                                                                                (2033, 300.0))]
        context = summarize_result_context(rows, "net sales by store number", "SELECT * FROM t")
        assert context["mode"] != "time_series" and context.get("pct_change") is None

    def test_a_number_too_large_to_read_as_one(self):
        rows = [{"CUSTOMER": "ZED CO", "SOLD_TO_ID": 10 ** 400, "UNITS_SHIPPED": 900.0},
                {"CUSTOMER": "ACME", "SOLD_TO_ID": 10 ** 400 + 1, "UNITS_SHIPPED": 200.0}]
        assert _headline(rows) == "ZED CO leads at 900."

    def test_an_identifier_with_a_blank(self):
        rows = [{"CUSTOMER": customer, "SOLD_TO_NO": number, "UNITS_SHIPPED": units}
                for customer, number, units in (("ZED CO", 40101, 900.0), ("ACME", None, 200.0), ("BOLT", 40000, 100.0))]
        assert _headline(rows) == "ZED CO leads at 900."

    def test_with_nothing_else_to_label_the_rows(self):
        # "Sold To No ranges 40,000 to 40,555, avg 40,164.50" described the
        # customer numbers.
        rows = [{"SOLD_TO_NO": number, "UNITS_SHIPPED": units}
                for number, units in ((40101, 900.0), (40002, 200.0), (40000, 100.0), (40555, 150.0))]
        response = build_assistant_response(question="units shipped", rows=rows,
                                            sql="SELECT SOLD_TO_NO, UNITS_SHIPPED FROM T", duration_ms=1)
        assert "Units Shipped ranges 100 to 900" in response["insight_summary"]
        assert "Sold To No" not in response["insight_summary"]

    def test_the_brief_lists_it_among_the_columns(self):
        from core.insight import compute_data_brief

        brief = compute_data_brief(SHIPPED, QUESTION, context=summarize_result_context(SHIPPED, QUESTION, "SELECT 1"))
        assert brief["columns"]["SOLD_TO_NO"] == "identifier"
