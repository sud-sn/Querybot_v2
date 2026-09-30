"""
The ledger mart (tests/ledger_harness.py): purchase order lines ordered_on and
received_on dates, January 2025 to March 2026, for materials kept in
kilograms and in eaches; journal lines posted_on the 28th of each month to
accounts and cost centres; a calendar joined on its date_day.

Quantities in two units are kept apart, as in the distribution set; spend and
expenses are money and add up across anything.
"""

from __future__ import annotations

from tests import ledger_harness as ledger
from tests.question_sets import Question

NAME = "ledger"

_PO = "fct_purchase_order_lines po"
_2025 = "po.ordered_on BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'"
_JE = "fct_journal_entries je"
_JE_2025 = "je.posted_on BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'"

QUESTIONS = [
    Question("L01", "procurement", "What was our total purchase spend?",
             "Quelles ont été nos dépenses d'achat totales ?",
             f"SELECT SUM(po.line_amount) FROM {_PO}",
             today={"fr": "model: \"dépenses d'achat totales\" is not read as Purchase Spend"}),
    Question("L02", "procurement", "Purchase spend by supplier", "Dépenses d'achat par fournisseur",
             f"SELECT SUM(po.line_amount) FROM {_PO} GROUP BY po.supplier_id"),
    Question("L03", "procurement", "Purchase spend by supplier country in 2025",
             "Dépenses d'achat par pays du fournisseur en 2025",
             f"SELECT SUM(po.line_amount) FROM {_PO} JOIN dim_supplier s ON s.supplier_id = po.supplier_id "
             f"WHERE {_2025} GROUP BY s.country",
             today={"en": "model: a supplier's country is not a breakdown the compiler makes", "fr": "model: a supplier's country is not a breakdown the compiler makes"}),
    Question("L04", "procurement", "Purchase spend by material family", "Dépenses d'achat par famille de matériaux",
             f"SELECT SUM(po.line_amount) FROM {_PO} JOIN dim_material m ON m.material_id = po.material_id "
             "GROUP BY m.material_family",
             today={"en": "model: a material's family is not a breakdown the compiler makes", "fr": "model: a material's family is not a breakdown the compiler makes"}),
    Question("L05", "dates", "Purchase spend by month in 2025", "Dépenses d'achat par mois en 2025",
             f"SELECT SUM(po.line_amount) FROM {_PO} WHERE {_2025} GROUP BY date_trunc('month', po.ordered_on)",
             today={"en": "no answer: a fact dated by a date column never has its calendar join confirmed", "fr": "no answer: a fact dated by a date column never has its calendar join confirmed"}),
    Question("L06", "dates", "Purchase spend by quarter in 2025", "Dépenses d'achat par trimestre en 2025",
             f"SELECT SUM(po.line_amount) FROM {_PO} WHERE {_2025} GROUP BY date_trunc('quarter', po.ordered_on)"),
    # The receipt date, where the order date is the everyday one.
    Question("L07", "dates", "Purchase spend by receipt month in 2025",
             "Dépenses d'achat par mois de réception en 2025",
             f"SELECT SUM(po.line_amount) FROM {_PO} "
             "WHERE po.received_on BETWEEN DATE '2025-01-01' AND DATE '2025-12-31' "
             "GROUP BY date_trunc('month', po.received_on)",
             today={"en": "no answer: a fact dated by a date column never has its calendar join confirmed", "fr": "no answer: a fact dated by a date column never has its calendar join confirmed"}),
    Question("L08", "ranking", "Top 2 suppliers by purchase spend in 2025",
             "Les 2 principaux fournisseurs par dépenses d'achat en 2025",
             f"SELECT SUM(po.line_amount) AS v FROM {_PO} WHERE {_2025} GROUP BY po.supplier_id "
             "ORDER BY v DESC LIMIT 2"),
    Question("L09", "procurement", "How many purchase orders did we place in 2025?",
             "Combien de bons de commande avons-nous passés en 2025 ?",
             f"SELECT COUNT(DISTINCT po.purchase_order_number) FROM {_PO} WHERE {_2025}"),
    # A quantity in two units: one total for each.
    Question("L10", "quantities", "Quantity ordered in 2025", "Quantité commandée en 2025",
             f"SELECT SUM(po.ordered_quantity) FROM {_PO} JOIN dim_material m ON m.material_id = po.material_id "
             f"WHERE {_2025} GROUP BY m.unit_of_measure"),
    Question("L11", "quantities", "Quantity received by material", "Quantité reçue par matériau",
             f"SELECT SUM(po.received_quantity) FROM {_PO} GROUP BY po.material_id",
             today={"fr": "model: \"matériau\" is not read as the material"}),
    Question("L12", "member", "Purchase spend with Nordic Metals in 2025",
             "Dépenses d'achat avec Nordic Metals en 2025",
             f"SELECT SUM(po.line_amount) FROM {_PO} WHERE {_2025} AND po.supplier_id = 1",
             today={"en": "wrong: supplier_name is not in the value index, and the supplier named is dropped without a word", "fr": "wrong: supplier_name is not in the value index, and the supplier named is dropped without a word"}),
    Question("L13", "population", "How many suppliers do we have?", "Combien de fournisseurs avons-nous ?",
             "SELECT COUNT(*) FROM dim_supplier"),
    Question("L14", "population", "How many suppliers do we have in each country?",
             "Combien de fournisseurs avons-nous dans chaque pays ?",
             "SELECT COUNT(*) FROM dim_supplier GROUP BY country"),
    Question("L15", "finance", "Total expenses in 2025", "Total des charges en 2025",
             f"SELECT SUM(je.amount) FROM {_JE} WHERE {_JE_2025}"),
    Question("L16", "finance", "Expenses by cost center in 2025", "Charges par centre de coûts en 2025",
             f"SELECT SUM(je.amount) FROM {_JE} WHERE {_JE_2025} GROUP BY je.cost_center_id",
             today={"fr": "model: \"centre de coûts\" is not read as the cost center"}),
    Question("L17", "finance", "Expenses by department", "Charges par service",
             f"SELECT SUM(je.amount) FROM {_JE} JOIN dim_cost_center c ON c.cost_center_id = je.cost_center_id "
             "GROUP BY c.department",
             today={"fr": "model: \"service\" is not read as the department"}),
    Question("L18", "finance", "Expenses by account type in 2025", "Charges par type de compte en 2025",
             f"SELECT SUM(je.amount) FROM {_JE} JOIN dim_gl_account a ON a.gl_account_id = je.gl_account_id "
             f"WHERE {_JE_2025} GROUP BY a.account_type",
             today={"en": "model: an account's type is not a breakdown the compiler makes", "fr": "model: an account's type is not a breakdown the compiler makes"}),
    Question("L19", "dates", "Expenses by month in the first quarter of 2026",
             "Charges par mois au premier trimestre 2026",
             f"SELECT SUM(je.amount) FROM {_JE} WHERE je.posted_on BETWEEN DATE '2026-01-01' AND DATE '2026-03-31' "
             "GROUP BY date_trunc('month', je.posted_on)",
             today={"en": "no answer: a fact dated by a date column never has its calendar join confirmed", "fr": "no answer: a fact dated by a date column never has its calendar join confirmed"}),
    Question("L20", "ranking", "Which cost center had the highest expenses in 2025?",
             "Quel centre de coûts a eu les charges les plus élevées en 2025 ?",
             f"SELECT SUM(je.amount) AS v FROM {_JE} WHERE {_JE_2025} GROUP BY je.cost_center_id "
             "ORDER BY v DESC LIMIT 1", first_row=True,
             today={"fr": "model: \"centre de coûts\" is not read as the cost center"}),
    Question("L21", "dates", "Purchase spend in 2024", "Dépenses d'achat en 2024", "", kind="no data"),
]


def tenant_in(root):
    return ledger.tenant_in(root)


def ask(warehouse, question: str, lang: str, choose: str | None = None) -> dict:
    return ledger.ask(warehouse, question, lang, choose=choose)
