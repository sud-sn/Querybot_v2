"""
core/erp_column_dict.py

ERP/M3 short-code column dictionary.

Maps raw 4-6 letter ERP column codes (Infor M3 / JDE style) to
(human_label, [synonyms]).  Used during KB generation to inject
business meanings for cryptic columns the LLM would otherwise
mark as [NEEDS CONTEXT].
"""

from __future__ import annotations

import re

ERP_COLUMN_DICT: dict[str, tuple[str, list[str]]] = {
    # ── Order / Delivery keys ──────────────────────────────────────────────
    "ORNO": ("Order Number",               ["order", "sales order", "order no", "order number"]),
    "PONR": ("Order Line Number",          ["line number", "line no", "order line"]),
    "POSX": ("Order Line Suffix",          ["suffix", "sub-line", "line suffix"]),
    "DLIX": ("Delivery Index",             ["delivery", "delivery number", "shipment", "delivery index"]),
    # ── Organisational ────────────────────────────────────────────────────
    "CONO": ("Company Number",             ["company", "company number"]),
    "DIVI": ("Division",                   ["division", "div", "business unit"]),
    "FACI": ("Facility",                   ["facility", "plant", "site"]),
    "WHLO": ("Warehouse",                  ["warehouse", "location", "whs", "warehouse location"]),
    # ── Item / Product ────────────────────────────────────────────────────
    "ITNO": ("Item Number",                ["item", "product", "sku", "part number", "item number"]),
    "ITGR": ("Item Group",                 ["item group", "product group", "category"]),
    "ITTY": ("Item Type",                  ["item type", "product type"]),
    "ITDS": ("Item Description",           ["item description", "product name", "description"]),
    # ── Customer / Supplier / People ──────────────────────────────────────
    "CUNO": ("Customer Number",            ["customer", "client", "account", "customer number"]),
    "SUNO": ("Supplier Number",            ["supplier", "vendor", "vendor number", "supplier number"]),
    "SMCD": ("Salesman Code",              ["salesman", "salesperson", "sales rep", "rep", "sales person"]),
    "BUYE": ("Buyer",                      ["buyer", "purchaser"]),
    # ── Quantity fields ───────────────────────────────────────────────────
    "TRQT": ("Transaction Quantity",       ["quantity", "transaction qty", "volume", "units", "transaction quantity"]),
    "ORQT": ("Ordered Quantity",           ["ordered qty", "order quantity", "ordered quantity"]),
    "IVQT": ("Invoiced Quantity",          ["invoiced qty", "billed quantity", "invoiced quantity"]),
    "DLQT": ("Delivered Quantity",         ["delivered qty", "shipped quantity", "delivered quantity"]),
    "ALQT": ("Allocated Quantity",         ["allocated qty", "reserved quantity", "allocated quantity"]),
    "RNQT": ("Return Quantity",            ["return qty", "returned quantity"]),
    "PLQT": ("Planned Quantity",           ["planned qty", "forecast quantity"]),
    # ── Amount / Cost fields ──────────────────────────────────────────────
    "CUAM": ("Customer Amount (CAD)",      ["amount", "revenue", "sales amount", "billed amount", "customer amount"]),
    "SAAM": ("Sales Amount",               ["sales", "net sales", "total sales", "gross sales"]),
    "SGAM": ("Gross Amount",               ["gross amount", "gross sales amount"]),
    "UCOS": ("Unit Cost",                  ["cost", "unit cost", "cogs per unit", "cost per unit"]),
    "DCOS": ("Delivery Cost",              ["delivery cost", "freight", "shipping cost", "freight cost"]),
    "MFAM": ("Manufacturing Amount",       ["manufacturing cost", "production cost"]),
    "SAPR": ("Sales Price",                ["sales price", "price", "list price"]),
    "NEPR": ("Net Price",                  ["net price", "discounted price"]),
    # ── Profit / Margin ───────────────────────────────────────────────────
    "PCLA": ("Profit Class / FIFO Layer",  ["profit class", "fifo profit", "margin tier", "pcla", "fifo layer", "fifo margin"]),
    "OFRA": ("Offered Rate / Margin %",    ["margin", "margin percent", "gross margin percent"]),
    # ── Date fields ───────────────────────────────────────────────────────
    "IVDT": ("Invoice Date",               ["invoice date", "billed date"]),
    "ORDT": ("Order Date",                 ["order date", "placed date", "order creation date"]),
    "DLDT": ("Actual Delivery Date",       ["delivery date", "shipped date", "actual delivery", "actual ship date"]),
    "DWDT": ("Requested Delivery Date",    ["requested delivery", "due date", "wanted date", "requested ship date"]),
    "CODT": ("Confirmed Delivery Date",    ["confirmed delivery", "confirmed date"]),
    "ACDT": ("Accounting Date",            ["accounting date", "posting date", "gl date", "ledger date"]),
    "DSDT": ("Dispatch Date",              ["dispatch date", "ship date"]),
    "PLDT": ("Planned Delivery Date",      ["planned delivery", "planned ship date"]),
    "RGDT": ("Registration Date",          ["registration date", "created date", "entry date"]),
    "LMDT": ("Last Modified Date",         ["last modified", "updated date", "last updated"]),
    # ── Invoice / Financial ───────────────────────────────────────────────
    "IVNO": ("Invoice Number",             ["invoice", "invoice number", "invoice no", "invoice #"]),
    "YEA4": ("Fiscal Year",                ["year", "fiscal year", "financial year"]),
    "CUCD": ("Currency Code",              ["currency", "currency code"]),
    # ── Order / Line status ───────────────────────────────────────────────
    "ORST": ("Order Status",               ["status", "order status", "order state"]),
    "ORTP": ("Order Type",                 ["order type", "transaction type"]),
    "LTYP": ("Line Type",                  ["line type"]),
    # ── Geography ─────────────────────────────────────────────────────────
    "SDST": ("Sales District",             ["district", "territory", "sales territory", "region"]),
    "CSCD": ("Country Code",               ["country", "country code"]),
    # ── Discount fields ───────────────────────────────────────────────────
    "DIA1": ("Discount Amount 1",          ["discount 1", "discount amount 1"]),
    "DIA2": ("Discount Amount 2",          ["discount 2", "discount amount 2"]),
    "DIP1": ("Discount Percent 1",         ["discount % 1", "discount percent 1"]),
    "DIP2": ("Discount Percent 2",         ["discount % 2", "discount percent 2"]),
    # ── Other common codes ────────────────────────────────────────────────
    "CHNO": ("Change Number",              ["version", "change number", "revision"]),
    "CHID": ("Changed By",                 ["changed by", "modified by", "last user"]),
    "PRMO": ("Price Model",                ["price model", "pricing model"]),
    "PYNO": ("Payer Number",               ["payer", "bill-to", "billing customer"]),
}


def get_erp_hints(column_names: list[str]) -> str:
    """
    Return a formatted hint block for any columns in `column_names` that are
    known ERP short codes.  Returns an empty string when none match so callers
    can use truthiness to skip injection for non-ERP tables.
    """
    lines: list[str] = []
    for col in column_names:
        entry = ERP_COLUMN_DICT.get(col.upper())
        if entry:
            label, synonyms = entry
            syn_str = ", ".join(f'"{s}"' for s in synonyms)
            lines.append(f"- {col} = {label} (business synonyms: {syn_str})")
    return "\n".join(lines)


def is_cryptic_table(column_names: list[str], threshold: float = 0.3) -> bool:
    """
    Return True when >= `threshold` fraction of `column_names` look like ERP
    short codes: all-uppercase, 2-6 chars, no underscores.  Useful for logging
    or conditional logic; the hint injection itself is cost-free for empty hints.
    """
    if not column_names:
        return False
    cryptic = sum(
        1 for c in column_names
        if re.match(r"^[A-Z0-9]{2,6}$", c.upper()) and "_" not in c
    )
    return (cryptic / len(column_names)) >= threshold
