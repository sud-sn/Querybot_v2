/*
    Synthetic fact data for EMDW_DMART. Run after 02_seed_dimensions.sql.

    THE NEWEST FACT ROW IS DELIBERATELY OLDER THAN TODAY.

    @LatestBusinessDate below is the last day any fact carries. The calendar in
    DT_DMS runs to 2027-12-31, so anchoring a relative window on the CALENDAR
    lands on a date with no rows, while anchoring on the DATA lands here. That
    gap is the whole point of the data-relative anchor rule, and it is also what
    makes "what is today's revenue" a stale-answer trap unless the bot discloses
    the as-of date.

    Change @LatestBusinessDate to move the staleness. Set it to CAST(GETDATE()
    AS date) to test the fully-current case.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

IF NOT EXISTS (SELECT 1 FROM EMDW_DMART.DT_DMS)
    THROW 50003, 'Run 02_seed_dimensions.sql before seeding facts.', 1;

IF EXISTS (SELECT 1 FROM EMDW_DMART.CUS_ORD_IVC_FCT)
    THROW 50004, 'Facts already contain data. Rerun 01_create_model.sql to reset the test model.', 1;

DECLARE @LatestBusinessDate date = '2026-06-30';
DECLARE @DaysOfHistory      int  = 545;   /* ~18 months */

BEGIN TRANSACTION;

/* ══════════════════════════════════════════════════════════════════════════
   FACT 1 — CUSTOMER ORDER INVOICE.  ~20,000 lines.
   Every row carries all eight role-playing dates so the ambiguity is real.
   ══════════════════════════════════════════════════════════════════════════ */
;WITH N AS (
    SELECT TOP (20000) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
    FROM sys.all_objects a CROSS JOIN sys.all_objects b
), Base AS (
    SELECT
        n,
        /* Weight recent days more heavily so "last 2 days" is never empty. */
        DATEADD(day, -((n * 7) % @DaysOfHistory), @LatestBusinessDate) AS invoice_date,
        5001 + (n % 60)            AS cus_key,
        7001 + ((n * 7) % 40)      AS itm_key,
        901  + (n % 6)             AS whs_key,
        101  + (n % 8)             AS pft_key,
        CONVERT(decimal(18,3), 1 + (n % 40)) AS qty,
        CASE WHEN n % 53 = 0 THEN 1 ELSE 0 END AS cancelled
    FROM N
), Priced AS (
    SELECT *,
        CONVERT(decimal(18,2), 45.00 + ((itm_key - 7000) * 12.35) + ((n % 17) * 3.10)) AS unit_price
    FROM Base
), Amounts AS (
    SELECT *,
        CONVERT(decimal(18,2), qty * unit_price)                       AS gross_amt,
        CONVERT(decimal(18,2), qty * unit_price * (n % 7) * 0.01)      AS disc_amt
    FROM Priced
)
INSERT EMDW_DMART.CUS_ORD_IVC_FCT (
    CUS_ORD_IVC_FCT_KEY, IVC_NO, IVC_LIN_NO, ORD_NO,
    CUS_DMS_KEY, ITM_DMS_KEY, WHS_DMS_KEY, PFT_CTR_DMS_KEY,
    CUS_IVC_DT_DMS_KEY, CUS_ORD_DT_DMS_KEY, CFM_DLY_DT_DMS_KEY, RQS_DLY_DT_DMS_KEY,
    PLN_DLY_DT_DMS_KEY, CNL_ORD_DT_DMS_KEY, DUE_DT_DMS_KEY, LST_MOD_DT_DMS_KEY,
    SOP_CUS_IVC_LIN_AMT, SOP_CUS_IVC_LIN_CAD_AMT, IVC_QTY,
    IVC_GRS_AMT, IVC_DSC_AMT, IVC_TAX_AMT, IVC_CST_AMT, IVC_MGN_AMT,
    CNY_CD, IVC_STS_CD, CNL_FLG, AZ_LST_UPD_TS
)
SELECT
    100000000 + n,
    CONCAT('IVC-', RIGHT(CONCAT('00000000', n), 8)),
    1 + (n % 3),
    CONCAT('ORD-', RIGHT(CONCAT('00000000', n), 8)),
    cus_key, itm_key, whs_key, pft_key,
    /* Invoice Date — the governed default */
    CONVERT(int, CONVERT(char(8), invoice_date, 112)),
    /* Order Date — earlier than the invoice */
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(5 + (n % 10)), invoice_date), 112)),
    /* Confirmed Delivery Date */
    CONVERT(int, CONVERT(char(8), DATEADD(day, 2 + (n % 5), invoice_date), 112)),
    /* Requested Delivery Date */
    CONVERT(int, CONVERT(char(8), DATEADD(day, 1 + (n % 7), invoice_date), 112)),
    /* Planned Delivery Date */
    CONVERT(int, CONVERT(char(8), DATEADD(day, 3 + (n % 4), invoice_date), 112)),
    /* Cancelled Order Date — only on cancelled rows */
    CASE WHEN cancelled = 1
         THEN CONVERT(int, CONVERT(char(8), DATEADD(day, 1, invoice_date), 112)) END,
    /* Due Date */
    CONVERT(int, CONVERT(char(8), DATEADD(day, 30, invoice_date), 112)),
    /* Last Modified Date — an AUDIT date. Always the load date, so it is
       systematically LATER than every business date. If the bot picks this by
       mistake, a "last 2 days" window returns the whole table rather than two
       days, which is exactly how the original defect showed up. */
    CONVERT(int, CONVERT(char(8), @LatestBusinessDate, 112)),

    CONVERT(decimal(18,2), gross_amt - disc_amt),
    CONVERT(decimal(18,2), (gross_amt - disc_amt) * 1.00),   /* CAD == local here */
    qty,
    gross_amt,
    disc_amt,
    CONVERT(decimal(18,2), (gross_amt - disc_amt) * 0.13),
    CONVERT(decimal(18,2), (gross_amt - disc_amt) * 0.62),
    CONVERT(decimal(18,2), (gross_amt - disc_amt) * 0.38),
    'CAD',
    CASE WHEN cancelled = 1 THEN 'CANCELLED'
         WHEN n % 11 = 0 THEN 'CREDITED' ELSE 'POSTED' END,
    cancelled,
    CAST(@LatestBusinessDate AS datetime2(0))
FROM Amounts;

/* ══════════════════════════════════════════════════════════════════════════
   FACT 2 — PURCHASE ORDER RECEIPT.  ~8,000 lines.
   CFM_FLG carries "confirmed", so "total amount of confirmed purchase orders
   by profit centre" has a real filter and a real answer here — and must NOT
   resolve to the invoice fact.
   ══════════════════════════════════════════════════════════════════════════ */
;WITH N AS (
    SELECT TOP (8000) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
    FROM sys.all_objects a CROSS JOIN sys.all_objects b
), Base AS (
    SELECT
        n,
        DATEADD(day, -((n * 11) % @DaysOfHistory), @LatestBusinessDate) AS receipt_date,
        8001 + ((n * 3) % 10)  AS sup_key,
        7001 + ((n * 11) % 40) AS itm_key,
        901  + ((n * 5) % 6)   AS whs_key,
        101  + ((n * 3) % 8)   AS pft_key,
        CONVERT(decimal(18,3), 10 + (n % 90)) AS ord_qty,
        CASE WHEN n % 37 = 0 THEN 'REJECTED'
             WHEN n % 19 = 0 THEN 'OPEN'
             WHEN n % 9  = 0 THEN 'PARTIAL' ELSE 'RECEIVED' END AS rct_status
    FROM N
), Costed AS (
    SELECT *, CONVERT(decimal(18,4), 8.25 + ((itm_key - 7000) * 6.40)) AS unit_cost
    FROM Base
)
INSERT EMDW_DMART.PCH_ORD_RCT_FCT (
    PCH_ORD_RCT_FCT_KEY, PCH_ORD_NO, PCH_ORD_LIN_NO,
    SUP_DMS_KEY, ITM_DMS_KEY, WHS_DMS_KEY, PFT_CTR_DMS_KEY,
    PCH_ORD_DT_DMS_KEY, PCH_ORD_RCT_DT_DMS_KEY, CFM_DLY_DT_DMS_KEY, LST_MOD_DT_DMS_KEY,
    PCH_ORD_LIN_AMT, PCH_ORD_LIN_CAD_AMT, PCH_ORD_QTY, RCT_QTY, RJT_QTY,
    UNT_CST_AMT, CNY_CD, RCT_STS_CD, CFM_FLG, AZ_LST_UPD_TS
)
SELECT
    200000000 + n,
    CONCAT('PO-', RIGHT(CONCAT('00000000', n), 8)),
    1 + (n % 4),
    sup_key, itm_key, whs_key, pft_key,
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(7 + (n % 14)), receipt_date), 112)),
    CASE WHEN rct_status = 'OPEN' THEN NULL
         ELSE CONVERT(int, CONVERT(char(8), receipt_date, 112)) END,
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(2 + (n % 5)), receipt_date), 112)),
    CONVERT(int, CONVERT(char(8), @LatestBusinessDate, 112)),
    CONVERT(decimal(18,2), ord_qty * unit_cost),
    CONVERT(decimal(18,2), ord_qty * unit_cost),
    ord_qty,
    CONVERT(decimal(18,3), CASE WHEN rct_status = 'RECEIVED' THEN ord_qty
                                WHEN rct_status = 'PARTIAL'  THEN ord_qty * 0.6 ELSE 0 END),
    CONVERT(decimal(18,3), CASE WHEN rct_status = 'REJECTED' THEN ord_qty ELSE 0 END),
    unit_cost,
    'CAD',
    rct_status,
    CASE WHEN rct_status IN ('RECEIVED', 'PARTIAL') THEN 1 ELSE 0 END,
    CAST(@LatestBusinessDate AS datetime2(0))
FROM Costed;

/* ══════════════════════════════════════════════════════════════════════════
   FACT 3 — FINANCE.  ~6,000 lines. Shares PFT_CTR_DMS and CUS_DMS with the
   invoice fact, which is what lets a loose metric drag two facts into one plan.
   ══════════════════════════════════════════════════════════════════════════ */
;WITH N AS (
    SELECT TOP (6000) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
    FROM sys.all_objects a CROSS JOIN sys.all_objects b
), Base AS (
    SELECT
        n,
        DATEADD(day, -((n * 13) % @DaysOfHistory), @LatestBusinessDate) AS acg_date,
        101  + (n % 8)         AS pft_key,
        5001 + ((n * 7) % 60)  AS cus_key,
        CONVERT(decimal(18,2), 250.00 + ((n % 120) * 47.25)) AS amt
    FROM N
)
INSERT EMDW_DMART.FNN_FCT (
    FNN_FCT_KEY, VCH_NO, VCH_LIN_NO, PFT_CTR_DMS_KEY, CUS_DMS_KEY,
    ACG_DT_DMS_KEY, ENT_DT_DMS_KEY, DUE_DT_DMS_KEY, LST_MOD_DT_DMS_KEY,
    GL_ACC_CD, GL_ACC_NM, DBT_AMT, CRD_AMT, NET_AMT, NET_CAD_AMT, CNY_CD, AZ_LST_UPD_TS
)
SELECT
    300000000 + n,
    CONCAT('VCH-', RIGHT(CONCAT('00000000', n), 8)),
    1 + (n % 5),
    pft_key, cus_key,
    CONVERT(int, CONVERT(char(8), acg_date, 112)),
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(1 + (n % 3)), acg_date), 112)),
    CONVERT(int, CONVERT(char(8), DATEADD(day, 45, acg_date), 112)),
    CONVERT(int, CONVERT(char(8), @LatestBusinessDate, 112)),
    CHOOSE(((n - 1) % 6) + 1, '4000', '4100', '5000', '5100', '6000', '6100'),
    CHOOSE(((n - 1) % 6) + 1, N'Sales Revenue', N'Service Revenue', N'Cost of Goods Sold',
                              N'Freight Expense', N'Operating Expense', N'Administrative Expense'),
    CONVERT(decimal(18,2), CASE WHEN n % 2 = 0 THEN amt ELSE 0 END),
    CONVERT(decimal(18,2), CASE WHEN n % 2 = 1 THEN amt ELSE 0 END),
    CONVERT(decimal(18,2), CASE WHEN n % 2 = 0 THEN amt ELSE -amt END),
    CONVERT(decimal(18,2), CASE WHEN n % 2 = 0 THEN amt ELSE -amt END),
    'CAD',
    CAST(@LatestBusinessDate AS datetime2(0))
FROM Base;

/* ══════════════════════════════════════════════════════════════════════════
   FACT 4 — ITEM BALANCE BY PERIOD. Month-end snapshots, semi-additive.
   PRD_DMS_KEY is YYYYMM, the classification the audit flagged as producing a
   silently wrong window for "last N months" questions.
   ══════════════════════════════════════════════════════════════════════════ */
;WITH MonthN AS (
    SELECT TOP (18) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS m FROM sys.all_objects
), Grain AS (
    SELECT
        mn.m,
        w.WHS_DMS_KEY,
        i.ITM_DMS_KEY,
        w.PFT_CTR_DMS_KEY,
        EOMONTH(DATEADD(month, -(18 - mn.m), @LatestBusinessDate)) AS bal_date
    FROM MonthN mn
    CROSS JOIN EMDW_DMART.WHS_DMS w
    CROSS JOIN EMDW_DMART.ITM_DMS i
), Calc AS (
    SELECT *,
        8001 + ((WHS_DMS_KEY + ITM_DMS_KEY + m) % 10) AS sup_key,
        CONVERT(decimal(18,3), 25 + ((WHS_DMS_KEY * 7 + ITM_DMS_KEY * 11 + m * 13) % 220)) AS oh,
        CONVERT(decimal(18,3), (WHS_DMS_KEY + ITM_DMS_KEY + m) % 30) AS alc,
        CONVERT(decimal(18,4), 8.25 + ((ITM_DMS_KEY - 7000) * 6.40)) AS unit_cost
    FROM Grain
)
INSERT EMDW_DMART.ITM_BAL_PRD_FCT (
    ITM_BAL_PRD_FCT_KEY, ITM_DMS_KEY, WHS_DMS_KEY, PFT_CTR_DMS_KEY, SUP_DMS_KEY,
    PRD_DMS_KEY, BAL_DT_DMS_KEY, LST_RCT_DT_DMS_KEY,
    OH_QTY, ALC_QTY, AVL_QTY, PCH_QTY, UNT_CST_AMT, BAL_VAL_AMT
)
SELECT
    400000000 + ROW_NUMBER() OVER (ORDER BY bal_date, WHS_DMS_KEY, ITM_DMS_KEY),
    ITM_DMS_KEY, WHS_DMS_KEY, PFT_CTR_DMS_KEY, sup_key,
    (YEAR(bal_date) * 100) + MONTH(bal_date),                    /* 202606 */
    CONVERT(int, CONVERT(char(8), bal_date, 112)),
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(5 + (m % 20)), bal_date), 112)),
    oh,
    alc,
    CONVERT(decimal(18,3), CASE WHEN oh > alc THEN oh - alc ELSE 0 END),
    CONVERT(decimal(18,3), 5 + ((ITM_DMS_KEY + m) % 45)),
    unit_cost,
    CONVERT(decimal(18,2), oh * unit_cost)
FROM Calc;

COMMIT TRANSACTION;

SELECT 'CUS_ORD_IVC_FCT' AS TABLE_NAME, COUNT_BIG(*) AS ROW_COUNT FROM EMDW_DMART.CUS_ORD_IVC_FCT
UNION ALL SELECT 'PCH_ORD_RCT_FCT', COUNT_BIG(*) FROM EMDW_DMART.PCH_ORD_RCT_FCT
UNION ALL SELECT 'FNN_FCT',         COUNT_BIG(*) FROM EMDW_DMART.FNN_FCT
UNION ALL SELECT 'ITM_BAL_PRD_FCT', COUNT_BIG(*) FROM EMDW_DMART.ITM_BAL_PRD_FCT;

/* The two numbers every relative-date test depends on. They must be equal as a
   date, which is the property that makes the integer key usable as the date. */
SELECT
    (SELECT MAX(CUS_IVC_DT_DMS_KEY) FROM EMDW_DMART.CUS_ORD_IVC_FCT)  AS MAX_INVOICE_DATE_KEY,
    (SELECT MAX(d.DMS_DT) FROM EMDW_DMART.DT_DMS d
      WHERE EXISTS (SELECT 1 FROM EMDW_DMART.CUS_ORD_IVC_FCT f
                    WHERE f.CUS_IVC_DT_DMS_KEY = d.DT_DMS_KEY))        AS MAX_INVOICE_DATE,
    (SELECT MAX(DT_DMS_KEY) FROM EMDW_DMART.DT_DMS)                    AS MAX_CALENDAR_KEY,
    CAST(GETDATE() AS date)                                            AS TODAY;

/*  Referential integrity self-check. Every one of these must return 0.
    A synthetic key generated as BASE + (expr % N) yields BASE..BASE+N-1, while
    a dimension seeded as BASE + n yields BASE+1..BASE+N — an off-by-one that
    only surfaces when the modular arithmetic happens to hit it, so it must be
    asserted rather than assumed. */
SELECT 'orphan_invoice_item'      AS CHECK_NAME, COUNT_BIG(*) AS BAD_ROWS
  FROM EMDW_DMART.CUS_ORD_IVC_FCT f
  LEFT JOIN EMDW_DMART.ITM_DMS d ON d.ITM_DMS_KEY = f.ITM_DMS_KEY
 WHERE f.ITM_DMS_KEY IS NOT NULL AND d.ITM_DMS_KEY IS NULL
UNION ALL SELECT 'orphan_invoice_customer', COUNT_BIG(*)
  FROM EMDW_DMART.CUS_ORD_IVC_FCT f
  LEFT JOIN EMDW_DMART.CUS_DMS d ON d.CUS_DMS_KEY = f.CUS_DMS_KEY
 WHERE f.CUS_DMS_KEY IS NOT NULL AND d.CUS_DMS_KEY IS NULL
UNION ALL SELECT 'orphan_invoice_date', COUNT_BIG(*)
  FROM EMDW_DMART.CUS_ORD_IVC_FCT f
  LEFT JOIN EMDW_DMART.DT_DMS d ON d.DT_DMS_KEY = f.CUS_IVC_DT_DMS_KEY
 WHERE f.CUS_IVC_DT_DMS_KEY IS NOT NULL AND d.DT_DMS_KEY IS NULL
UNION ALL SELECT 'orphan_purchase_item', COUNT_BIG(*)
  FROM EMDW_DMART.PCH_ORD_RCT_FCT f
  LEFT JOIN EMDW_DMART.ITM_DMS d ON d.ITM_DMS_KEY = f.ITM_DMS_KEY
 WHERE f.ITM_DMS_KEY IS NOT NULL AND d.ITM_DMS_KEY IS NULL
UNION ALL SELECT 'orphan_purchase_supplier', COUNT_BIG(*)
  FROM EMDW_DMART.PCH_ORD_RCT_FCT f
  LEFT JOIN EMDW_DMART.SUP_DMS d ON d.SUP_DMS_KEY = f.SUP_DMS_KEY
 WHERE f.SUP_DMS_KEY IS NOT NULL AND d.SUP_DMS_KEY IS NULL
UNION ALL SELECT 'orphan_finance_customer', COUNT_BIG(*)
  FROM EMDW_DMART.FNN_FCT f
  LEFT JOIN EMDW_DMART.CUS_DMS d ON d.CUS_DMS_KEY = f.CUS_DMS_KEY
 WHERE f.CUS_DMS_KEY IS NOT NULL AND d.CUS_DMS_KEY IS NULL;

/*  Dimension coverage. Thin coverage does not fail the load, but it silently
    weakens every "by <dimension>" test — an invoice fact touching 3 of 6
    warehouses makes "revenue by warehouse" look like it works on half the data. */
SELECT 'warehouses_in_invoices' AS COVERAGE, COUNT(DISTINCT WHS_DMS_KEY) AS USED,
       (SELECT COUNT(*) FROM EMDW_DMART.WHS_DMS) AS AVAILABLE
  FROM EMDW_DMART.CUS_ORD_IVC_FCT
UNION ALL SELECT 'items_in_invoices', COUNT(DISTINCT ITM_DMS_KEY),
       (SELECT COUNT(*) FROM EMDW_DMART.ITM_DMS) FROM EMDW_DMART.CUS_ORD_IVC_FCT
UNION ALL SELECT 'customers_in_invoices', COUNT(DISTINCT CUS_DMS_KEY),
       (SELECT COUNT(*) FROM EMDW_DMART.CUS_DMS) FROM EMDW_DMART.CUS_ORD_IVC_FCT
UNION ALL SELECT 'profit_centres_in_invoices', COUNT(DISTINCT PFT_CTR_DMS_KEY),
       (SELECT COUNT(*) FROM EMDW_DMART.PFT_CTR_DMS) FROM EMDW_DMART.CUS_ORD_IVC_FCT
UNION ALL SELECT 'profit_centres_in_purchases', COUNT(DISTINCT PFT_CTR_DMS_KEY),
       (SELECT COUNT(*) FROM EMDW_DMART.PFT_CTR_DMS) FROM EMDW_DMART.PCH_ORD_RCT_FCT;
