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
    # ── AR / Financial Ledger (FSLEDG) ───────────────────────────────────
    "JRNO": ("Journal Number",             ["journal", "journal number", "journal no"]),
    "JSNO": ("Journal Sub-number",         ["journal sub", "journal sequence", "journal sub-number"]),
    "TRCD": ("Transaction Code",           ["transaction code", "transaction type code", "trx code"]),
    "CINO": ("Customer Invoice Number",    ["customer invoice", "invoice reference"]),
    "INYR": ("Invoice Year",               ["invoice year"]),
    "VSER": ("Voucher Series",             ["voucher series"]),
    "VONO": ("Voucher Number",             ["voucher", "voucher number", "voucher no"]),
    "IVTP": ("Invoice Type",               ["invoice type"]),
    "TDSC": ("Transaction Description",    ["description", "transaction description", "line description"]),
    "CUCL": ("Customer Class",             ["customer class", "customer category", "account class"]),
    "CRTP": ("Currency Rate Type",         ["currency rate type", "exchange rate type"]),
    "ARAT": ("Exchange Rate",              ["exchange rate", "fx rate", "currency rate"]),
    "DCAM": ("Discount Amount",            ["discount", "discount amount", "discount value"]),
    "VTAM": ("VAT / Tax Amount",           ["vat", "tax", "tax amount", "vat amount", "gst"]),
    "BKID": ("Bank ID",                    ["bank", "bank id", "bank account"]),
    "TECD": ("Payment Terms Code",         ["payment terms", "terms code", "credit terms"]),
    "PYTP": ("Payment Type",               ["payment type", "payment method"]),
    "PYCD": ("Payment Code",               ["payment code"]),
    "TEPY": ("Terms of Payment",           ["terms of payment", "payment terms", "net days"]),
    "RECO": ("Reconciliation Code",        ["reconciliation", "reconciled", "matched"]),
    "CRST": ("Credit Status",              ["credit status", "credit standing", "credit hold"]),
    "ACBL": ("Account Balance",            ["account balance", "balance", "outstanding balance", "ar balance"]),
    "RMST": ("Reminder / Dunning Status",  ["reminder status", "dunning status", "collection status"]),
    "LRDT": ("Last Reminder Date",         ["last reminder", "last dunning date"]),
    "RMBL": ("Reminder Balance",           ["reminder balance", "dunning balance"]),
    "BLBY": ("Blocked By",                 ["blocked by", "block user"]),
    "BLDT": ("Blocked Date",               ["blocked date", "hold date"]),
    "IIST": ("Interest Status",            ["interest status"]),
    "IIAM": ("Interest Amount",            ["interest", "interest amount", "interest charge"]),
    "CLST": ("Collection Status",          ["collection status", "collections"]),
    "TXID": ("Tax ID",                     ["tax id", "tax identifier"]),
    "APRV": ("Approved",                   ["approved", "approval status", "authorised"]),
    "ARCD": ("AR Code",                    ["ar code", "accounts receivable code"]),
    "IVCL": ("Invoice Class",              ["invoice class"]),
    # ── Revenue / SOP invoice columns (CUS_ORD_IVC_FCT) ─────────────────
    # IMPORTANT: SOP_CUS_IVC_LIN_AMT is the approved revenue measure.
    # The approved Revenue formula is:
    #   SUM(CASE WHEN DEL_IVC_REC_IND = 0 THEN SOP_CUS_IVC_LIN_AMT ELSE 0 END)
    # Do NOT use CUS_IVC_LIN_AMT as a substitute for revenue.
    "SOP_CUS_IVC_LIN_AMT": (
        "SOP Invoice Line Amount (Revenue)",
        ["revenue", "invoiced revenue", "sales revenue", "net revenue",
         "total revenue", "invoice revenue", "sop revenue"],
    ),
    "DEL_IVC_REC_IND": (
        "Deleted Invoice Record Indicator",
        ["deleted invoice", "invoice deleted flag", "del ivc rec ind"],
    ),
    "DEL_ORD_REC_IND": (
        "Deleted Order Record Indicator",
        ["deleted order", "order deleted flag"],
    ),
    "DEL_SOP_REC_IND": (
        "Deleted SOP Record Indicator",
        ["deleted sop", "sop deleted flag"],
    ),
    # ── Margin / pricing thresholds (CUS_ORD_IVC_FCT) ────────────────────
    "MRG_45_PCT": ("45% Margin Threshold Flag", ["45 percent margin", "margin 45", "below 45 margin"]),
    "MRG_70_PCT": ("70% Margin Threshold Flag", ["70 percent margin", "margin 70", "below 70 margin"]),
    "SGT_MRG":    ("Suggested Margin",          ["suggested margin", "target margin", "recommended margin"]),
    # ── Sales order / logistics (OOLINE / OSBSTD) ────────────────────────
    "CUOR": ("Customer Order Reference",   ["customer reference", "customer order ref", "customer po"]),
    "CUPO": ("Customer Purchase Order",    ["customer po number", "po reference", "purchase order reference"]),
    "GRWE": ("Gross Weight",               ["gross weight", "weight"]),
    "NEWE": ("Net Weight",                 ["net weight"]),
    "VOL3": ("Volume",                     ["volume", "cubic volume", "shipment volume"]),
    "PROJ": ("Project Number",             ["project", "project number", "project code"]),
    "AGNO": ("Agreement Number",           ["agreement", "agreement number", "contract number", "contract"]),
    "ALUN": ("Alternate Unit of Measure",  ["alternate unit", "alternate uom", "alt unit"]),
    "SPUN": ("Sales Price Unit",           ["sales price unit", "price uom", "price unit"]),
    "STUN": ("Stock Unit of Measure",      ["stock unit", "stocking uom"]),
    "DISY": ("Discount System",            ["discount system", "pricing system"]),
    "PRRF": ("Price Reference",            ["price reference", "price list ref"]),
    "VTCD": ("VAT Code",                   ["vat code", "tax code", "gst code"]),
    "ORTY": ("Order Type",                 ["order type"]),
    "ROUT": ("Route",                      ["route", "delivery route", "shipping route"]),
    "BRAN": ("Branch",                     ["branch", "branch code"]),
    "ORIG": ("Origin",                     ["origin", "source origin", "order origin"]),
    "SERN": ("Serial Number",              ["serial number", "serial no", "serial"]),
    # ── Remaining AR/Ledger fields (FSLEDG) ──────────────────────────────
    "RVDT": ("Reversal Date",              ["reversal date", "reversed date"]),
    "DUDT": ("Due Date",                   ["due date", "payment due date", "payment deadline"]),
    "REDE": ("Reminder Date",              ["reminder date", "dunning date"]),
    "SLOP": ("Late Payment Option",        ["late payment", "late payment option"]),
    "PYRS": ("Payment Reference",          ["payment reference", "remittance reference"]),
    "RMQT": ("Reminder Quantity",          ["reminder count", "number of reminders", "dunning level"]),
    "IICD": ("Interest Charge Code",       ["interest code", "interest charge code"]),
    "CLCD": ("Collection Code",            ["collection code"]),
    "SAGS": ("Sales Agreement",            ["sales agreement", "agreement"]),
    "GRPD": ("Group Date",                 ["group date", "batch date"]),
    "DFPT": ("Deferred Payment Type",      ["deferred payment type"]),
    "DFPD": ("Deferred Payment Date",      ["deferred payment date"]),
    "RGTM": ("Registration Time",          ["registration time", "created time"]),
    "LMTS": ("Last Modified Timestamp",    ["last modified timestamp", "last updated timestamp"]),
    "RESP": ("Responsible",                ["responsible", "responsible person", "owner"]),
    "DNRE": ("Dispute Reference",          ["dispute", "dispute reference"]),
    "DEDA": ("Deduction Date",             ["deduction date"]),
    "MIGI": ("Migration Indicator",        ["migration indicator", "migrated"]),
    # ── Alternate-unit quantity fields (QA suffix = qty in alternate unit) ─
    "ORQA": ("Ordered Quantity (Alternate Unit)",   ["ordered quantity alternate unit", "ordered alt qty"]),
    "IVQA": ("Invoiced Quantity (Alternate Unit)",  ["invoiced quantity alternate unit", "invoiced alt qty"]),
    "DLQA": ("Delivered Quantity (Alternate Unit)", ["delivered quantity alternate unit", "delivered alt qty"]),
    "ALQA": ("Allocated Quantity (Alternate Unit)", ["allocated quantity alternate unit"]),
    "PLQA": ("Planned Quantity (Alternate Unit)",   ["planned quantity alternate unit"]),
    "RNQA": ("Return Quantity (Alternate Unit)",    ["return quantity alternate unit"]),
    "IVQS": ("Invoiced Quantity (Stock Unit)",      ["invoiced quantity stock unit"]),
    "ORQS": ("Ordered Quantity (Stock Unit)",       ["ordered quantity stock unit"]),
    "ORQB": ("Ordered Quantity (Base Unit)",        ["ordered quantity base unit"]),
    # ── Useful OOLINE / OSBSTD fields ────────────────────────────────────
    "CUTP": ("Customer Type",              ["customer type"]),
    "CUST": ("Customer Status",            ["customer status"]),
    "AGNT": ("Agent",                      ["agent", "agent code"]),
    "REPI": ("Replacement Item Indicator", ["replacement", "replacement item"]),
    "MUFT": ("Manufacturing Flag",         ["manufactured", "manufacturing flag"]),
    "ITCL": ("Item Classification",        ["item classification", "item class"]),
    "SMCC": ("Salesman Commission Class",  ["commission class", "salesman commission class"]),
    "PRCH": ("Price Changed",              ["price changed", "price change indicator"]),
    "VANO": ("Voucher Number",             ["voucher", "voucher number"]),
    "JOBN": ("Job Number",                 ["job", "job number"]),
    "PROJ": ("Project",                    ["project", "project number", "project code"]),
    "INNO": ("Internal Invoice Number",    ["internal invoice", "internal invoice number"]),
    "GRWE": ("Gross Weight",               ["gross weight", "weight"]),
    "NEWE": ("Net Weight",                 ["net weight"]),
    # Common dimension display fields. These are not always raw ERP short codes,
    # but they keep KB generation from showing warehouse/customer/item keys when
    # a business-readable code or description field exists.
    "WHS_DMS_KEY": ("Warehouse Dimension Key", ["warehouse key", "warehouse id", "warehouse dimension key"]),
    "WHS_CD": ("Warehouse Code", ["warehouse code", "warehouse number"]),
    "WHS_DSC": ("Warehouse Description", ["warehouse", "warehouse name", "warehouse description", "warehouse location"]),
    "DT_DMS_KEY": ("Date Dimension Key", ["date key", "calendar date key"]),
    "ITM_DMS_KEY": ("Item Dimension Key", ["item key", "product key", "sku key"]),
    "ITM_CD": ("Item Code", ["item code", "product code", "sku"]),
    "ITM_DSC": ("Item Description", ["item", "product", "item name", "product name", "item description"]),
    "ITM_GRP_DMS_KEY": ("Item Group Dimension Key", ["item group key", "product group key"]),
    "CUS_DMS_KEY": ("Customer Dimension Key", ["customer key", "customer id", "customer dimension key"]),
    "CUS_CD": ("Customer Code", ["customer code", "customer number"]),
    "CUS_NM": ("Customer Name", ["customer", "customer name", "client name"]),
}


def get_erp_hints(column_names: list[str], vocab=None) -> str:
    """
    Return a formatted hint block for any columns in `column_names` that are
    known ERP short codes.  Returns an empty string when none match so callers
    can use truthiness to skip injection for non-ERP tables.
    """
    if vocab is None:
        from core.vocab_packs import get_active_vocab
        vocab = get_active_vocab()
    column_dict = vocab.column_dict
    lines: list[str] = []
    for col in column_names:
        entry = column_dict.get(col.upper())
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
