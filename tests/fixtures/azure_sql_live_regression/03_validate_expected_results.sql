/*
QueryBot live NL-to-SQL regression fixture - read-only validation queries

Run after 01_create_schema.sql and 02_seed_data.sql. This script changes no
data and contains no GO batch separators.
*/

SET NOCOUNT ON;

SELECT
    'ROW_COUNTS' AS [validation_section],
    t.[name] AS [table_name],
    SUM(p.[rows]) AS [row_count]
FROM sys.tables t
JOIN sys.schemas s ON s.[schema_id] = t.[schema_id]
JOIN sys.partitions p ON p.[object_id] = t.[object_id] AND p.[index_id] IN (0, 1)
WHERE s.[name] = N'QBOT_LIVE_TEST'
GROUP BY t.[name]
ORDER BY t.[name];

/* Case 1: star join. */
SELECT
    'CASE_01_STAR_JOIN' AS [test_case],
    w.[WAREHOUSE_CODE],
    SUM(f.[NET_REVENUE_AMOUNT]) AS [revenue]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
JOIN [QBOT_LIVE_TEST].[D_WAREHOUSE] w
    ON w.[WAREHOUSE_SK] = f.[WAREHOUSE_SK]
GROUP BY w.[WAREHOUSE_CODE]
ORDER BY w.[WAREHOUSE_CODE];

/* Case 2: snowflake fact -> warehouse -> region. */
SELECT
    'CASE_02_SNOWFLAKE_JOIN' AS [test_case],
    r.[REGION_NAME],
    SUM(f.[NET_REVENUE_AMOUNT]) AS [revenue]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
JOIN [QBOT_LIVE_TEST].[D_WAREHOUSE] w
    ON w.[WAREHOUSE_SK] = f.[WAREHOUSE_SK]
JOIN [QBOT_LIVE_TEST].[D_REGION] r
    ON r.[REGION_SK] = w.[REGION_SK]
GROUP BY r.[REGION_NAME]
ORDER BY r.[REGION_NAME];

/*
Case 3: current/previous data month. The anchor is the maximum invoice date
that exists in the fact, never the unrestricted maximum of D_DATE.
*/
;WITH [scoped_sales] AS (
    SELECT d.[FULL_DATE], f.[NET_REVENUE_AMOUNT]
    FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
    JOIN [QBOT_LIVE_TEST].[D_DATE] d
        ON d.[DATE_SK] = f.[INVOICE_DATE_SK]
),
[fact_anchor] AS (
    SELECT MAX([FULL_DATE]) AS [max_fact_date]
    FROM [scoped_sales]
),
[periods] AS (
    SELECT
        CASE
            WHEN s.[FULL_DATE] >= DATEFROMPARTS(YEAR(a.[max_fact_date]), MONTH(a.[max_fact_date]), 1)
                THEN 'current_data_month'
            ELSE 'previous_data_month'
        END AS [period_name],
        s.[NET_REVENUE_AMOUNT]
    FROM [scoped_sales] s
    CROSS JOIN [fact_anchor] a
    WHERE s.[FULL_DATE] >= DATEADD(month, -1, DATEFROMPARTS(YEAR(a.[max_fact_date]), MONTH(a.[max_fact_date]), 1))
      AND s.[FULL_DATE] < DATEADD(month, 1, DATEFROMPARTS(YEAR(a.[max_fact_date]), MONTH(a.[max_fact_date]), 1))
)
SELECT
    'CASE_03_FACT_SCOPED_RELATIVE_MONTH' AS [test_case],
    [period_name],
    SUM([NET_REVENUE_AMOUNT]) AS [revenue]
FROM [periods]
GROUP BY [period_name]
ORDER BY [period_name];

/* Case 4: last two distinct dates present in the scoped fact. */
;WITH [ranked_data_dates] AS (
    SELECT
        d.[FULL_DATE],
        DENSE_RANK() OVER (ORDER BY d.[FULL_DATE] DESC) AS [date_rank]
    FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
    JOIN [QBOT_LIVE_TEST].[D_DATE] d
        ON d.[DATE_SK] = f.[INVOICE_DATE_SK]
    GROUP BY d.[FULL_DATE]
),
[last_two_dates] AS (
    SELECT [FULL_DATE]
    FROM [ranked_data_dates]
    WHERE [date_rank] <= 2
)
SELECT
    'CASE_04_LAST_TWO_DATA_DAYS' AS [test_case],
    d.[FULL_DATE],
    SUM(f.[NET_REVENUE_AMOUNT]) AS [revenue]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
JOIN [QBOT_LIVE_TEST].[D_DATE] d
    ON d.[DATE_SK] = f.[INVOICE_DATE_SK]
JOIN [last_two_dates] selected_date
    ON selected_date.[FULL_DATE] = d.[FULL_DATE]
GROUP BY d.[FULL_DATE]
ORDER BY d.[FULL_DATE];

/* Case 5: invoice date differs from posting month for INV-1002. */
SELECT
    'CASE_05_INVOICE_DATE_NOT_POSTING_MONTH' AS [test_case],
    d.[YYYYMM_KEY] AS [invoice_yyyymm],
    SUM(f.[NET_REVENUE_AMOUNT]) AS [revenue]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
JOIN [QBOT_LIVE_TEST].[D_DATE] d
    ON d.[DATE_SK] = f.[INVOICE_DATE_SK]
WHERE d.[YYYYMM_KEY] = 202601
GROUP BY d.[YYYYMM_KEY];

/* Cases 6-8: YYYYMM, YYYYMMDD and native DATE behavior. */
SELECT
    'CASE_06_YYYYMM_MONTHLY_INVENTORY' AS [test_case],
    [PERIOD_YYYYMM],
    SUM([ENDING_INVENTORY_VALUE]) AS [inventory_value]
FROM [QBOT_LIVE_TEST].[F_INVENTORY_MONTHLY]
WHERE [PERIOD_YYYYMM] = 202602
GROUP BY [PERIOD_YYYYMM];

;WITH [latest_snapshot] AS (
    SELECT MAX([SNAPSHOT_YYYYMMDD]) AS [snapshot_yyyymmdd]
    FROM [QBOT_LIVE_TEST].[F_INVENTORY_DAILY]
)
SELECT
    'CASE_07_YYYYMMDD_LATEST_DAILY_INVENTORY' AS [test_case],
    w.[WAREHOUSE_CODE],
    SUM(i.[INVENTORY_VALUE]) AS [inventory_value]
FROM [QBOT_LIVE_TEST].[F_INVENTORY_DAILY] i
JOIN [latest_snapshot] latest
    ON latest.[snapshot_yyyymmdd] = i.[SNAPSHOT_YYYYMMDD]
JOIN [QBOT_LIVE_TEST].[D_WAREHOUSE] w
    ON w.[WAREHOUSE_SK] = i.[WAREHOUSE_SK]
GROUP BY w.[WAREHOUSE_CODE]
ORDER BY w.[WAREHOUSE_CODE];

SELECT
    'CASE_08_NATIVE_DATE' AS [test_case],
    [SNAPSHOT_DATE],
    SUM([INVENTORY_VALUE]) AS [inventory_value]
FROM [QBOT_LIVE_TEST].[F_INVENTORY_DAILY]
WHERE [SNAPSHOT_DATE] = CAST('2026-03-02' AS date)
GROUP BY [SNAPSHOT_DATE];

/* Case 9: timestamp-specific request. */
SELECT
    'CASE_09_TIMESTAMP' AS [test_case],
    [INVOICE_NUMBER],
    [UPDATED_AT_UTC]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE]
WHERE [UPDATED_AT_UTC] >= CAST('2026-03-01T00:00:00' AS datetime2(0))
ORDER BY [INVOICE_NUMBER];

/* Case 11: preserve the fact's SCD2 surrogate version. */
SELECT
    'CASE_11_SCD2' AS [test_case],
    c.[CUSTOMER_BK],
    c.[CUSTOMER_NAME],
    c.[CURRENT_FLAG],
    SUM(f.[NET_REVENUE_AMOUNT]) AS [revenue]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
JOIN [QBOT_LIVE_TEST].[D_CUSTOMER_SCD2] c
    ON c.[CUSTOMER_SK] = f.[CUSTOMER_SK]
WHERE c.[CUSTOMER_BK] = 'C001'
GROUP BY c.[CUSTOMER_BK], c.[CUSTOMER_NAME], c.[CURRENT_FLAG]
ORDER BY c.[CURRENT_FLAG];

/* Case 12: many-to-many bridge with allocation. */
SELECT
    'CASE_12_BRIDGE_ALLOCATION' AS [test_case],
    c.[CATEGORY_NAME],
    SUM(f.[NET_REVENUE_AMOUNT] * b.[ALLOCATION_PCT]) AS [allocated_revenue]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
JOIN [QBOT_LIVE_TEST].[B_PRODUCT_CATEGORY] b
    ON b.[PRODUCT_SK] = f.[PRODUCT_SK]
JOIN [QBOT_LIVE_TEST].[D_CATEGORY] c
    ON c.[CATEGORY_SK] = b.[CATEGORY_SK]
GROUP BY c.[CATEGORY_NAME]
ORDER BY c.[CATEGORY_NAME];

/*
Case 13: multi-fact comparison. Each fact is aggregated in an isolated CTE;
only the aggregate outputs are combined at warehouse grain.
*/
;WITH [march_revenue] AS (
    SELECT f.[WAREHOUSE_SK], SUM(f.[NET_REVENUE_AMOUNT]) AS [revenue]
    FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
    JOIN [QBOT_LIVE_TEST].[D_DATE] d
        ON d.[DATE_SK] = f.[INVOICE_DATE_SK]
    WHERE d.[YYYYMM_KEY] = 202603
    GROUP BY f.[WAREHOUSE_SK]
),
[latest_inventory] AS (
    SELECT i.[WAREHOUSE_SK], SUM(i.[INVENTORY_VALUE]) AS [inventory_value]
    FROM [QBOT_LIVE_TEST].[F_INVENTORY_DAILY] i
    WHERE i.[SNAPSHOT_YYYYMMDD] = (
        SELECT MAX(i2.[SNAPSHOT_YYYYMMDD])
        FROM [QBOT_LIVE_TEST].[F_INVENTORY_DAILY] i2
    )
    GROUP BY i.[WAREHOUSE_SK]
)
SELECT
    'CASE_13_ISOLATED_MULTI_FACT' AS [test_case],
    w.[WAREHOUSE_CODE],
    COALESCE(r.[revenue], 0) AS [revenue],
    COALESCE(i.[inventory_value], 0) AS [inventory_value]
FROM [QBOT_LIVE_TEST].[D_WAREHOUSE] w
LEFT JOIN [march_revenue] r ON r.[WAREHOUSE_SK] = w.[WAREHOUSE_SK]
LEFT JOIN [latest_inventory] i ON i.[WAREHOUSE_SK] = w.[WAREHOUSE_SK]
ORDER BY w.[WAREHOUSE_CODE];

/* Case 14: role-playing Return Date, not Original Invoice Date. */
SELECT
    'CASE_14_RETURN_DATE_ROLE' AS [test_case],
    d.[FULL_DATE] AS [return_date],
    SUM(r.[REFUND_AMOUNT]) AS [refund_amount]
FROM [QBOT_LIVE_TEST].[F_RETURNS] r
JOIN [QBOT_LIVE_TEST].[D_DATE] d
    ON d.[DATE_SK] = r.[RETURN_DATE_SK]
GROUP BY d.[FULL_DATE]
ORDER BY d.[FULL_DATE];

/* Case 15: governed fact-to-fact anti-join exception. */
SELECT
    'CASE_15_GOVERNED_ANTI_JOIN' AS [test_case],
    o.[ORDER_NUMBER],
    o.[ORDER_STATUS],
    o.[ORDER_AMOUNT]
FROM [QBOT_LIVE_TEST].[F_ORDERS] o
WHERE NOT EXISTS (
    SELECT 1
    FROM [QBOT_LIVE_TEST].[F_SHIPMENTS] s
    WHERE s.[ORDER_SK] = o.[ORDER_SK]
)
ORDER BY o.[ORDER_NUMBER];

/* Case 16: cryptic Infor M3-style YYYYMM period. */
SELECT
    'CASE_16_M3_CRYPTIC_YYYYMM' AS [test_case],
    [MLWHLO] AS [warehouse_code],
    SUM([MLAVAL]) AS [inventory_value]
FROM [QBOT_LIVE_TEST].[M3_MITBAL]
WHERE [MLPERY] = 202603
GROUP BY [MLWHLO]
ORDER BY [MLWHLO];

/* Case 17: intentionally unsafe relationship profile. */
SELECT
    'CASE_17_UNSAFE_RELATIONSHIP' AS [test_case],
    (SELECT COUNT(*) FROM [QBOT_LIVE_TEST].[F_BAD_CODE]) AS [source_rows],
    (SELECT COUNT(*) FROM [QBOT_LIVE_TEST].[D_DUPLICATE_CODE]) AS [target_rows],
    (SELECT COUNT(*)
     FROM (
        SELECT [DUP_CODE]
        FROM [QBOT_LIVE_TEST].[D_DUPLICATE_CODE]
        GROUP BY [DUP_CODE]
        HAVING COUNT(*) > 1
     ) duplicate_keys) AS [target_duplicate_keys],
    (SELECT COUNT(*)
     FROM [QBOT_LIVE_TEST].[F_BAD_CODE] f
     JOIN [QBOT_LIVE_TEST].[D_DUPLICATE_CODE] d
        ON d.[DUP_CODE] = f.[TARGET_CODE]) AS [joined_rows],
    CAST(
        (SELECT COUNT(*) * 1.0
         FROM [QBOT_LIVE_TEST].[F_BAD_CODE] f
         JOIN [QBOT_LIVE_TEST].[D_DUPLICATE_CODE] d
            ON d.[DUP_CODE] = f.[TARGET_CODE])
        / NULLIF((SELECT COUNT(*) FROM [QBOT_LIVE_TEST].[F_BAD_CODE]), 0)
        AS decimal(9,4)
    ) AS [fanout_ratio];

/* Case 18: presentation-only follow-up; business totals remain unchanged. */
SELECT
    'CASE_18_FORMATTING_FOLLOW_UP' AS [test_case],
    FORMAT(d.[MONTH_START_DATE], 'MMM-yy', 'en-US') AS [display_month],
    FORMAT(SUM(f.[NET_REVENUE_AMOUNT]), 'C2', 'en-US') AS [display_revenue]
FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE] f
JOIN [QBOT_LIVE_TEST].[D_DATE] d
    ON d.[DATE_SK] = f.[INVOICE_DATE_SK]
GROUP BY d.[MONTH_START_DATE]
ORDER BY d.[MONTH_START_DATE];

/* Case 19: exact ERP PRD_DMS_KEY YYYYMM pattern with zero sentinel. */
;WITH [latest_valid_period] AS (
    SELECT MAX([PRD_DMS_KEY]) AS [period_yyyymm]
    FROM [QBOT_LIVE_TEST].[ERP_ITM_BAL_PRD_FCT]
    WHERE [PRD_DMS_KEY] BETWEEN 190001 AND 299912
)
SELECT
    'CASE_19_ERP_PRD_DMS_YYYYMM' AS [test_case],
    f.[PRD_DMS_KEY],
    w.[WAREHOUSE_CODE],
    SUM(f.[INV_VAL_AMT]) AS [inventory_value]
FROM [QBOT_LIVE_TEST].[ERP_ITM_BAL_PRD_FCT] f
JOIN [latest_valid_period] p
    ON p.[period_yyyymm] = f.[PRD_DMS_KEY]
JOIN [QBOT_LIVE_TEST].[D_WAREHOUSE] w
    ON w.[WAREHOUSE_SK] = f.[WHS_DMS_KEY]
GROUP BY f.[PRD_DMS_KEY], w.[WAREHOUSE_CODE]
ORDER BY w.[WAREHOUSE_CODE];

SELECT
    [CASE_ID], [CAPABILITY], [QUESTION_TEXT], [EXPECTED_RESULT],
    [REQUIRED_TABLES], [REQUIRED_DATE_ROLE], [FORBIDDEN_BEHAVIOR]
FROM [QBOT_LIVE_TEST].[TEST_EXPECTED_CASES]
ORDER BY [CASE_ID];
