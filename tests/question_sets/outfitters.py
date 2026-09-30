"""
The outfitters star (tests/star_harness.py): online sales from January 2024 to
December 2025 with order, due and ship dates keyed yyyymmdd, a snowflaked
product (product, subcategory, category), customers, sales territories, a
calendar whose fiscal year starts in July and is numbered by the year it ends
in, a daily stock count for the last quarter of 2025 and a month-end balance
since January 2024.
"""

from __future__ import annotations

from tests import star_harness as star
from tests.question_sets import Question

NAME = "outfitters"

_SALES = "FactInternetSales f"
_PRODUCT = ("JOIN DimProduct p ON p.ProductKey = f.ProductKey "
            "JOIN DimProductSubcategory s ON s.ProductSubcategoryKey = p.ProductSubcategoryKey "
            "JOIN DimProductCategory c ON c.ProductCategoryKey = s.ProductCategoryKey")
_2025 = "f.OrderDateKey BETWEEN 20250101 AND 20251231"

QUESTIONS = [
    Question("O01", "sales", "What were our total sales?", "Quel est le total de nos ventes ?",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES}"),
    Question("O02", "sales", "Sales by product category", "Ventes par catégorie de produits",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} {_PRODUCT} GROUP BY c.ProductCategoryKey"),
    Question("O03", "sales", "Sales by product category in 2025", "Ventes par catégorie de produits en 2025",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} {_PRODUCT} WHERE {_2025} GROUP BY c.ProductCategoryKey"),
    Question("O04", "dates", "Sales by month in 2025", "Ventes par mois en 2025",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} WHERE {_2025} GROUP BY f.OrderDateKey // 100"),
    Question("O05", "dates", "Sales by quarter in 2025", "Ventes par trimestre en 2025",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} WHERE {_2025} "
             "GROUP BY (f.OrderDateKey // 100 % 100 - 1) // 3"),
    Question("O06", "dates", "Sales by year", "Ventes par année",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} GROUP BY f.OrderDateKey // 10000"),
    # The fiscal year the calendar keeps: fiscal 2025 is July 2024 to June 2025.
    Question("O07", "dates", "Sales in fiscal year 2025", "Ventes de l'exercice 2025",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} WHERE f.OrderDateKey BETWEEN 20240701 AND 20250630"),
    Question("O08", "ranking", "Top 3 products by sales amount", "Les 3 meilleurs produits par montant des ventes",
             f"SELECT SUM(f.SalesAmount) AS v FROM {_SALES} GROUP BY f.ProductKey ORDER BY v DESC LIMIT 3"),
    Question("O09", "ranking", "Which product category had the lowest sales in 2025?",
             "Quelle catégorie de produits a eu les ventes les plus faibles en 2025 ?",
             f"SELECT SUM(f.SalesAmount) AS v FROM {_SALES} {_PRODUCT} WHERE {_2025} "
             "GROUP BY c.ProductCategoryKey ORDER BY v LIMIT 1", first_row=True),
    Question("O10", "orders", "How many orders did we receive in 2025?",
             "Combien de commandes avons-nous reçues en 2025 ?",
             f"SELECT COUNT(DISTINCT f.SalesOrderNumber) FROM {_SALES} WHERE {_2025}"),
    Question("O11", "orders", "Number of orders by month in 2025", "Nombre de commandes par mois en 2025",
             f"SELECT COUNT(DISTINCT f.SalesOrderNumber) FROM {_SALES} WHERE {_2025} "
             "GROUP BY f.OrderDateKey // 100"),
    Question("O12", "customers", "How many customers placed an order in 2025?",
             "Combien de clients ont passé une commande en 2025 ?",
             f"SELECT COUNT(DISTINCT f.CustomerKey) FROM {_SALES} WHERE {_2025}"),
    # A population, counted on its own tables.
    Question("O13", "population", "How many products are there in each category?",
             "Combien de produits y a-t-il dans chaque catégorie ?",
             "SELECT COUNT(*) FROM DimProduct p JOIN DimProductSubcategory s "
             "ON s.ProductSubcategoryKey = p.ProductSubcategoryKey GROUP BY s.ProductCategoryKey"),
    Question("O14", "territory", "Sales by sales territory country", "Ventes par pays du territoire de vente",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} JOIN DimSalesTerritory t "
             "ON t.SalesTerritoryKey = f.SalesTerritoryKey GROUP BY t.SalesTerritoryCountry"),
    # A category the question names, broken down by its products.
    Question("O15", "member", "Sales of Climbing products in 2025", "Ventes des produits Escalade en 2025",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} {_PRODUCT} "
             f"WHERE {_2025} AND c.EnglishProductCategoryName = 'Climbing' GROUP BY f.ProductKey"),
    # The ship date, where the order date is the everyday one.
    Question("O16", "dates", "Sales by ship month in 2025", "Ventes par mois d'expédition en 2025",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} WHERE f.ShipDateKey BETWEEN 20250101 AND 20251231 "
             "GROUP BY f.ShipDateKey // 100"),
    Question("O17", "sales", "Order quantity by product category", "Quantité commandée par catégorie de produits",
             f"SELECT SUM(f.OrderQuantity) FROM {_SALES} {_PRODUCT} GROUP BY c.ProductCategoryKey"),
    # Stock is read at the newest count, never added up across days.
    Question("O18", "stock", "Units in stock by product category", "Unités en stock par catégorie de produits",
             "SELECT SUM(i.UnitsInStock) FROM FactProductInventory i JOIN DimProduct p ON p.ProductKey = i.ProductKey "
             "JOIN DimProductSubcategory s ON s.ProductSubcategoryKey = p.ProductSubcategoryKey "
             "WHERE i.DateKey = (SELECT MAX(DateKey) FROM FactProductInventory) GROUP BY s.ProductCategoryKey"),
    Question("O19", "stock", "Month-end units in stock at the end of 2024",
             "Unités en stock en fin de mois à la fin de 2024",
             "SELECT SUM(UnitsBalance) FROM FactProductInventoryMonthly WHERE MonthEndDateKey = 20241231"),
    Question("O20", "customers", "Sales by customer gender", "Ventes par sexe du client",
             f"SELECT SUM(f.SalesAmount) FROM {_SALES} JOIN DimCustomer cu ON cu.CustomerKey = f.CustomerKey "
             "GROUP BY cu.Gender"),
    # An attribute of the product's own table, averaged there.
    Question("O21", "attribute", "Average list price by product category",
             "Prix de catalogue moyen par catégorie de produits",
             "SELECT AVG(p.ListPrice) FROM DimProduct p JOIN DimProductSubcategory s "
             "ON s.ProductSubcategoryKey = p.ProductSubcategoryKey GROUP BY s.ProductCategoryKey"),
    Question("O22", "dates", "Sales in 2023", "Ventes en 2023", "", kind="no data"),
]


def tenant_in(root):
    return star.tenant_in(root)


def ask(warehouse, question: str, lang: str, choose: str | None = None) -> dict:
    return star.ask(warehouse, question, lang, choose=choose)
