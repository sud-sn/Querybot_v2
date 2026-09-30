"""
The distribution mart (tests/answer_harness.py): a daily stock snapshot whose
newest day is 2026-03-31, a monthly movement fact keyed yyyymm with a
whole-year row that is never one of its months, items in two units of measure,
and the warehouse, item, item group and party dimensions.

Every reference reads the newest snapshot for stock, leaves the whole-year row
out of movements, and keeps quantities in different units apart: a total of
feet and eaches is no total.
"""

from __future__ import annotations

from tests import answer_harness as harness
from tests.question_sets import Question

NAME = "distribution"

_LATEST = "d.ITM_BAL_EFC_DT_DMS_KEY = 20260331"
_DAILY = f"ITM_BAL_DLY_FCT d JOIN ITM_DMS i USING (ITM_DMS_KEY)"
_MOVES = "ITM_BAL_PRD_FCT m JOIN ITM_DMS i USING (ITM_DMS_KEY)"
_2025 = "m.PRD_DMS_KEY BETWEEN 202501 AND 202512"

QUESTIONS = [
    Question("D01", "stock", "What is our total stock on hand?", "Quel est notre stock total en main ?",
             f"SELECT SUM(d.ON_HND_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY i.UNT_OF_MSR"),
    Question("D02", "stock", "Stock on hand by warehouse", "Stock en main par entrepôt",
             f"SELECT SUM(d.ON_HND_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY d.WHS_DMS_KEY, i.UNT_OF_MSR"),
    Question("D03", "stock", "Stock on hand by item group", "Stock en main par groupe d'articles",
             f"SELECT SUM(d.ON_HND_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY i.ITM_GRP_DMS_KEY, i.UNT_OF_MSR", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D04", "stock", "Reserved quantity by item", "Quantité réservée par article",
             f"SELECT SUM(d.RSV_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY d.ITM_DMS_KEY", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D05", "stock", "Back-ordered quantity by item", "Quantité en rupture de stock par article",
             f"SELECT SUM(d.RSV_BCK_ORD_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY d.ITM_DMS_KEY", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D06", "orders", "Ordered quantity by item", "Quantité commandée par article",
             f"SELECT SUM(d.ORD_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY d.ITM_DMS_KEY", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D07", "sales", "Units sold in 2025", "Unités vendues en 2025",
             f"SELECT SUM(m.SLD_QTY) FROM {_MOVES} WHERE {_2025} GROUP BY i.UNT_OF_MSR"),
    Question("D08", "sales", "Units sold by item in March 2025", "Unités vendues par article en mars 2025",
             f"SELECT SUM(m.SLD_QTY) FROM {_MOVES} WHERE m.PRD_DMS_KEY = 202503 GROUP BY m.ITM_DMS_KEY", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D09", "sales", "Top 2 items by units sold in 2025", "Les 2 meilleurs articles par unités vendues en 2025",
             f"SELECT SUM(m.SLD_QTY) AS v FROM {_MOVES} WHERE {_2025} GROUP BY m.ITM_DMS_KEY ORDER BY v DESC LIMIT 2", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D10", "sales", "Units sold at South Depot in 2025", "Unités vendues à South Depot en 2025",
             f"SELECT SUM(m.SLD_QTY) FROM {_MOVES} WHERE {_2025} AND m.WHS_DMS_KEY = 2 GROUP BY i.UNT_OF_MSR"),
    Question("D11", "purchases", "Units purchased by item in 2025", "Unités achetées par article en 2025",
             f"SELECT SUM(m.PCH_QTY) FROM {_MOVES} WHERE {_2025} GROUP BY m.ITM_DMS_KEY", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D12", "receipts", "Number of receipts by item group in 2025",
             "Nombre de réceptions par groupe d'articles en 2025",
             f"SELECT SUM(m.NUM_OF_RCT) FROM {_MOVES} WHERE {_2025} GROUP BY i.ITM_GRP_DMS_KEY", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D13", "receipts", "Number of receipts by warehouse in 2025", "Nombre de réceptions par entrepôt en 2025",
             f"SELECT SUM(m.NUM_OF_RCT) FROM ITM_BAL_PRD_FCT m WHERE {_2025} GROUP BY m.WHS_DMS_KEY"),
    # A population: every member but the placeholder, one with no code included.
    Question("D14", "population", "How many warehouses do we have?", "Combien d'entrepôts avons-nous ?",
             "SELECT COUNT(*) FROM WHS_DMS WHERE WHS_DMS_KEY <> 0"),
    Question("D15", "population", "How many items do we have?", "Combien d'articles avons-nous ?",
             "SELECT COUNT(*) FROM ITM_DMS WHERE ITM_DMS_KEY <> 0", today={"fr": "no answer: the French \"articles\" is not read as the item"}),
    Question("D16", "population", "How many items are in each item group?",
             "Combien d'articles y a-t-il dans chaque groupe d'articles ?",
             "SELECT COUNT(*) FROM ITM_DMS WHERE ITM_DMS_KEY <> 0 GROUP BY ITM_GRP_DMS_KEY", today={"en": "no answer: asks which dataset, where the item table is the population",
                    "fr": "no answer: the French \"articles\" is not read as the item"}),
    Question("D17", "value", "Stock value by warehouse", "Valeur du stock par entrepôt",
             f"SELECT SUM(d.ON_HND_QTY * d.ITM_CST) FROM {_DAILY} WHERE {_LATEST} GROUP BY d.WHS_DMS_KEY"),
    Question("D18", "member", "Stock on hand at North Depot", "Stock en main à North Depot",
             f"SELECT SUM(d.ON_HND_QTY) FROM {_DAILY} WHERE {_LATEST} AND d.WHS_DMS_KEY = 1 "
             "GROUP BY i.UNT_OF_MSR"),
    # The party table reached as the snapshot's buyer.
    Question("D19", "role", "Stock on hand by buyer", "Stock en main par acheteur",
             f"SELECT SUM(d.ON_HND_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY d.BYR_PTY_DMS_KEY, i.UNT_OF_MSR"),
    Question("D20", "stock", "Allocated stock by item", "Stock alloué par article",
             f"SELECT SUM(d.ALC_ON_HND_QTY) FROM {_DAILY} WHERE {_LATEST} GROUP BY d.ITM_DMS_KEY", today={"fr": "model: the French \"article\" is not read as the item"}),
    Question("D21", "sales", "Units sold by warehouse in the first half of 2025",
             "Unités vendues par entrepôt au premier semestre 2025",
             f"SELECT SUM(m.SLD_QTY) FROM {_MOVES} WHERE m.PRD_DMS_KEY BETWEEN 202501 AND 202506 "
             "GROUP BY m.WHS_DMS_KEY, i.UNT_OF_MSR"),
    Question("D22", "sales", "Units sold in 2024", "Unités vendues en 2024", "", kind="no data"),
]


def tenant_in(root):
    return harness.tenant_in(root)


def ask(warehouse, question: str, lang: str, choose: str | None = None) -> dict:
    return harness.ask(warehouse, question, lang, choose=choose)
