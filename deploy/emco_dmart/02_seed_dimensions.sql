/*
    Synthetic dimension data for EMDW_DMART. Run after 01_create_model.sql.

    DT_DMS_KEY is the date as a YYYYMMDD integer, exactly as the client's mart
    stores it, so MAX(fact.<role>_DT_DMS_KEY) is directly comparable to
    MAX(DT_DMS.DMS_DT) — the property that makes an integer-range window on the
    fact possible without joining the dimension at all.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
SET DATEFIRST 1;

IF OBJECT_ID(N'EMDW_DMART.DT_DMS', N'U') IS NULL
    THROW 50001, 'Run 01_create_model.sql before seeding dimensions.', 1;

IF EXISTS (SELECT 1 FROM EMDW_DMART.DT_DMS)
    THROW 50002, 'Dimensions already contain data. Rerun 01_create_model.sql to reset the test model.', 1;

BEGIN TRANSACTION;

/* ── Date dimension: 2024-01-01 .. 2027-12-31 ──────────────────────────────
   The calendar deliberately extends well past the newest fact row, so that
   anchoring on the calendar instead of the data lands on a date with no rows.
   That is the mistake commit 975214b exists to prevent, and this data makes it
   visible rather than theoretical. */
;WITH N AS (
    SELECT TOP (DATEDIFF(day, CONVERT(date, '2024-01-01'), CONVERT(date, '2027-12-31')) + 1)
        ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
    FROM sys.all_objects a CROSS JOIN sys.all_objects b
), Dates AS (
    SELECT DATEADD(day, n - 1, CONVERT(date, '2024-01-01')) AS d FROM N
)
INSERT EMDW_DMART.DT_DMS (
    DT_DMS_KEY, DMS_DT, DAY, DAY_OF_WK_NO, DAY_NM, DAY_OF_MTH, WK_OF_YR,
    MTH_NO, MTH_NM, MTH_START_DT, YR_MTH_NO, QTR_NO, QTR_NM, CAL_YR,
    FSC_MTH_NO, FSC_QTR_NO, FSC_YR, IS_WKD_FLG
)
SELECT
    CONVERT(int, CONVERT(char(8), d, 112)),          /* 20260630 */
    d,
    d,
    DATEPART(weekday, d),
    DATENAME(weekday, d),
    DATEPART(day, d),
    DATEPART(iso_week, d),
    DATEPART(month, d),
    DATENAME(month, d),
    DATEFROMPARTS(YEAR(d), MONTH(d), 1),
    (YEAR(d) * 100) + MONTH(d),                      /* 202606 */
    DATEPART(quarter, d),
    CONCAT('Q', DATEPART(quarter, d)),
    YEAR(d),
    ((MONTH(d) + 5) % 12) + 1,
    ((((MONTH(d) + 5) % 12) + 1) - 1) / 3 + 1,
    YEAR(d) + CASE WHEN MONTH(d) >= 7 THEN 1 ELSE 0 END,
    CASE WHEN DATEPART(weekday, d) IN (6, 7) THEN 1 ELSE 0 END
FROM Dates;

INSERT EMDW_DMART.PC_DVN_DMS (PC_DVN_DMS_KEY, PC_DVN_CD, PC_DVN_NM, ACT_FLG, AZ_LST_UPD_TS)
VALUES
    (10, 'DVN-IND', N'Industrial Division',   1, '2026-06-30T02:00:00'),
    (20, 'DVN-COM', N'Commercial Division',   1, '2026-06-30T02:00:00'),
    (30, 'DVN-RES', N'Residential Division',  1, '2026-06-30T02:00:00'),
    (40, 'DVN-SVC', N'Service Division',      1, '2026-06-30T02:00:00');

INSERT EMDW_DMART.PFT_CTR_DMS (
    PFT_CTR_DMS_KEY, PFT_CTR_CD, PFT_CTR_NM, PC_DVN_DMS_KEY, RGN_NM, CNY_CD, ACT_FLG, AZ_LST_UPD_TS, AZ_LST_UPD_USR)
VALUES
    (101, 'PC-ON-01', N'Ontario Industrial',    10, N'Ontario',          'CA', 1, '2026-06-30T02:00:00', N'ETL_USER'),
    (102, 'PC-ON-02', N'Ontario Commercial',    20, N'Ontario',          'CA', 1, '2026-06-30T02:00:00', N'ETL_USER'),
    (103, 'PC-QC-01', N'Quebec Industrial',     10, N'Quebec',           'CA', 1, '2026-06-30T02:00:00', N'ETL_USER'),
    (104, 'PC-QC-02', N'Quebec Commercial',     20, N'Quebec',           'CA', 1, '2026-06-30T02:00:00', N'ETL_USER'),
    (105, 'PC-BC-01', N'British Columbia West', 30, N'British Columbia', 'CA', 1, '2026-06-30T02:00:00', N'ETL_USER'),
    (106, 'PC-AB-01', N'Alberta Central',       30, N'Alberta',          'CA', 1, '2026-06-30T02:00:00', N'ETL_USER'),
    (107, 'PC-MB-01', N'Manitoba Service',      40, N'Manitoba',         'CA', 1, '2026-06-30T02:00:00', N'ETL_USER'),
    (108, 'PC-NS-01', N'Nova Scotia Service',   40, N'Atlantic',         'CA', 1, '2026-06-30T02:00:00', N'ETL_USER');

INSERT EMDW_DMART.CUS_SEG_DMS (CUS_SEG_DMS_KEY, CUS_SEG_CD, CUS_SEG_NM)
VALUES (1, 'SEG-KEY', N'Key Account'), (2, 'SEG-MID', N'Mid Market'),
       (3, 'SEG-SML', N'Small Business'), (4, 'SEG-GOV', N'Government');

INSERT EMDW_DMART.CUS_TYP_DMS (CUS_TYP_DMS_KEY, CUS_TYP_CD, CUS_TYP_NM, ACT_FLG)
VALUES (1, 'TYP-DIS', N'Distributor', 1), (2, 'TYP-CON', N'Contractor', 1),
       (3, 'TYP-OEM', N'OEM', 1), (4, 'TYP-RET', N'Retailer', 1),
       (5, 'TYP-INT', N'Intercompany', 1);

/* 60 customers. CR_LMT_n_CHG_DT_DMS_KEY is populated so the audit-date noise is
   real: those columns are date-role shaped and do join DT_DMS. */
;WITH N AS (SELECT TOP (60) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n FROM sys.all_objects)
INSERT EMDW_DMART.CUS_DMS (
    CUS_DMS_KEY, CUS_CD, CUS_NM, CUS_TYP_DMS_KEY, CUS_SEG_DMS_KEY, PFT_CTR_DMS_KEY,
    CTY_NM, PRV_CD, PST_CD, CNY_CD,
    CR_LMT_1, CR_LMT_1_CHG_DT_DMS_KEY, CR_LMT_2, CR_LMT_2_CHG_DT_DMS_KEY,
    CR_LMT_3, CR_LMT_3_CHG_DT_DMS_KEY, FST_IVC_DT_DMS_KEY,
    ACT_FLG, AZ_LST_UPD_TS, AZ_LST_UPD_USR, AZ_EXT_ID
)
SELECT
    5000 + n,
    CONCAT('CUS-', RIGHT(CONCAT('0000', n), 4)),
    CHOOSE(((n - 1) % 12) + 1,
        N'Maple Ridge Supply', N'Northern Steel Works', N'Lakeshore Contracting',
        N'Prairie Mechanical', N'Atlantic Fasteners', N'Cascade Industrial',
        N'Confederation Builders', N'Rideau Valley Trading', N'Bow River Equipment',
        N'Acadian Distribution', N'Great Lakes Fabrication', N'Sunset Coast Hardware')
      + N' ' + CONVERT(nvarchar(4), n),
    ((n - 1) % 5) + 1,
    ((n - 1) % 4) + 1,
    101 + ((n - 1) % 8),
    CHOOSE(((n - 1) % 8) + 1, N'Toronto', N'Montreal', N'Vancouver', N'Calgary',
                              N'Winnipeg', N'Halifax', N'Ottawa', N'Edmonton'),
    CHOOSE(((n - 1) % 8) + 1, 'ON', 'QC', 'BC', 'AB', 'MB', 'NS', 'ON', 'AB'),
    CONCAT('A', RIGHT(CONCAT('0000', n), 4)),
    'CA',
    CONVERT(decimal(18,2), 25000 + (n % 10) * 5000),
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(n % 400), CONVERT(date, '2026-06-30')), 112)),
    CONVERT(decimal(18,2), 50000 + (n % 7) * 7500),
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(n % 300), CONVERT(date, '2026-06-30')), 112)),
    CONVERT(decimal(18,2), 75000 + (n % 5) * 10000),
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(n % 200), CONVERT(date, '2026-06-30')), 112)),
    CONVERT(int, CONVERT(char(8), DATEADD(day, -(400 + (n % 180)), CONVERT(date, '2026-06-30')), 112)),
    1,
    '2026-06-30T02:00:00', N'ETL_USER', CONCAT('EXT-', RIGHT(CONCAT('0000', n), 4))
FROM N;

;WITH N AS (SELECT TOP (40) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n FROM sys.all_objects)
INSERT EMDW_DMART.ITM_DMS (ITM_DMS_KEY, ITM_CD, ITM_NM, ITM_GRP_CD, ITM_GRP_NM, UOM_CD, ACT_FLG, AZ_LST_UPD_TS)
SELECT
    7000 + n,
    CONCAT('ITM-', RIGHT(CONCAT('0000', n), 4)),
    CHOOSE(((n - 1) % 10) + 1,
        N'Copper Pipe 15mm', N'Steel Elbow 90deg', N'Brass Ball Valve',
        N'PVC Conduit 20mm', N'Galvanised Bracket', N'Insulated Cable 3C',
        N'Pressure Gauge', N'Threaded Coupling', N'Hex Bolt M12', N'Sealing Gasket')
      + N' #' + CONVERT(nvarchar(4), n),
    CHOOSE(((n - 1) % 5) + 1, 'GRP-PIP', 'GRP-FIT', 'GRP-VAL', 'GRP-ELE', 'GRP-FAS'),
    CHOOSE(((n - 1) % 5) + 1, N'Piping', N'Fittings', N'Valves', N'Electrical', N'Fasteners'),
    CHOOSE(((n - 1) % 3) + 1, 'EA', 'M', 'BOX'),
    1, '2026-06-30T02:00:00'
FROM N;

INSERT EMDW_DMART.WHS_DMS (WHS_DMS_KEY, WHS_CD, WHS_NM, PFT_CTR_DMS_KEY, CTY_NM, PRV_CD, ACT_FLG)
VALUES
    (901, 'WHS-TOR', N'Toronto Distribution Centre',   101, N'Toronto',   'ON', 1),
    (902, 'WHS-MTL', N'Montreal Distribution Centre',  103, N'Montreal',  'QC', 1),
    (903, 'WHS-VAN', N'Vancouver Distribution Centre', 105, N'Vancouver', 'BC', 1),
    (904, 'WHS-CAL', N'Calgary Distribution Centre',   106, N'Calgary',   'AB', 1),
    (905, 'WHS-WPG', N'Winnipeg Branch Store',         107, N'Winnipeg',  'MB', 1),
    (906, 'WHS-HFX', N'Halifax Branch Store',          108, N'Halifax',   'NS', 1);

;WITH N AS (SELECT TOP (10) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n FROM sys.all_objects)
INSERT EMDW_DMART.SUP_DMS (SUP_DMS_KEY, SUP_CD, SUP_NM, CNY_CD, ACT_FLG)
SELECT 8000 + n, CONCAT('SUP-', RIGHT(CONCAT('000', n), 3)),
       CHOOSE(((n - 1) % 5) + 1, N'Dominion Metals', N'Laurentian Supply',
              N'Pacific Import Partners', N'Central Fabricators', N'Maritime Trading Co')
         + N' ' + CONVERT(nvarchar(3), n),
       CASE WHEN n % 4 = 0 THEN 'US' ELSE 'CA' END, 1
FROM N;

/*  Profit-centre-to-customer assignment. This is the SECOND route from an
    invoice to a customer (invoice -> profit centre -> here -> customer), beside
    the direct invoice -> customer key. Two governed routes to the same entity is
    exactly the "more than one equally governed relationship path" case. */
INSERT EMDW_DMART.PFT_CTR_CUS_DAT (PFT_CTR_DMS_KEY, CUS_DMS_KEY, ASG_DT_DMS_KEY, PRY_FLG, ACT_FLG, AZ_LST_UPD_TS)
SELECT c.PFT_CTR_DMS_KEY, c.CUS_DMS_KEY,
       CONVERT(int, CONVERT(char(8), DATEADD(day, -(c.CUS_DMS_KEY % 500), CONVERT(date, '2026-06-30')), 112)),
       1, 1, '2026-06-30T02:00:00'
FROM EMDW_DMART.CUS_DMS c
WHERE c.PFT_CTR_DMS_KEY IS NOT NULL;

COMMIT TRANSACTION;

SELECT 'DT_DMS' AS TABLE_NAME, COUNT_BIG(*) AS ROW_COUNT FROM EMDW_DMART.DT_DMS
UNION ALL SELECT 'PC_DVN_DMS',      COUNT_BIG(*) FROM EMDW_DMART.PC_DVN_DMS
UNION ALL SELECT 'PFT_CTR_DMS',     COUNT_BIG(*) FROM EMDW_DMART.PFT_CTR_DMS
UNION ALL SELECT 'CUS_SEG_DMS',     COUNT_BIG(*) FROM EMDW_DMART.CUS_SEG_DMS
UNION ALL SELECT 'CUS_TYP_DMS',     COUNT_BIG(*) FROM EMDW_DMART.CUS_TYP_DMS
UNION ALL SELECT 'CUS_DMS',         COUNT_BIG(*) FROM EMDW_DMART.CUS_DMS
UNION ALL SELECT 'ITM_DMS',         COUNT_BIG(*) FROM EMDW_DMART.ITM_DMS
UNION ALL SELECT 'WHS_DMS',         COUNT_BIG(*) FROM EMDW_DMART.WHS_DMS
UNION ALL SELECT 'SUP_DMS',         COUNT_BIG(*) FROM EMDW_DMART.SUP_DMS
UNION ALL SELECT 'PFT_CTR_CUS_DAT', COUNT_BIG(*) FROM EMDW_DMART.PFT_CTR_CUS_DAT;
