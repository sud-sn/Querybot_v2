/*
QueryBot live NL-to-SQL regression fixture - deterministic seed data

Run after 01_create_schema.sql. The script is rerunnable, contains no GO
separator, and reloads only QBOT_LIVE_TEST tables.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

BEGIN TRY
    BEGIN TRANSACTION;

    DELETE FROM [QBOT_LIVE_TEST].[TEST_EXPECTED_CASES];
    DELETE FROM [QBOT_LIVE_TEST].[F_BAD_CODE];
    DELETE FROM [QBOT_LIVE_TEST].[D_DUPLICATE_CODE];
    DELETE FROM [QBOT_LIVE_TEST].[F_SHIPMENTS];
    DELETE FROM [QBOT_LIVE_TEST].[F_ORDERS];
    DELETE FROM [QBOT_LIVE_TEST].[F_RETURNS];
    DELETE FROM [QBOT_LIVE_TEST].[M3_MITBAL];
    DELETE FROM [QBOT_LIVE_TEST].[ERP_ITM_BAL_PRD_FCT];
    DELETE FROM [QBOT_LIVE_TEST].[F_INVENTORY_MONTHLY];
    DELETE FROM [QBOT_LIVE_TEST].[F_INVENTORY_DAILY];
    DELETE FROM [QBOT_LIVE_TEST].[F_SALES_INVOICE];
    DELETE FROM [QBOT_LIVE_TEST].[B_PRODUCT_CATEGORY];
    DELETE FROM [QBOT_LIVE_TEST].[D_CATEGORY];
    DELETE FROM [QBOT_LIVE_TEST].[D_PRODUCT];
    DELETE FROM [QBOT_LIVE_TEST].[D_CUSTOMER_SCD2];
    DELETE FROM [QBOT_LIVE_TEST].[D_WAREHOUSE];
    DELETE FROM [QBOT_LIVE_TEST].[D_REGION];
    DELETE FROM [QBOT_LIVE_TEST].[D_DATE];

    /*
    The calendar deliberately extends to 2026-12-31 while the newest sales
    record is 2026-03-03. Relative periods must anchor to the scoped fact data,
    not MAX(FULL_DATE) from this complete calendar.
    */
    ;WITH [calendar_dates] AS (
        SELECT CAST('2025-12-25' AS date) AS [calendar_date]
        UNION ALL
        SELECT DATEADD(day, 1, [calendar_date])
        FROM [calendar_dates]
        WHERE [calendar_date] < CAST('2026-12-31' AS date)
    )
    INSERT INTO [QBOT_LIVE_TEST].[D_DATE] (
        [DATE_SK], [FULL_DATE], [YYYYMMDD_KEY], [YYYYMM_KEY],
        [CALENDAR_YEAR], [CALENDAR_QUARTER], [CALENDAR_MONTH_NUMBER],
        [CALENDAR_MONTH_NAME], [MONTH_START_DATE], [MONTH_END_DATE],
        [DAY_OF_MONTH], [IS_MONTH_END]
    )
    SELECT
        10000 + DATEDIFF(day, CAST('2025-01-01' AS date), [calendar_date]),
        [calendar_date],
        CONVERT(int, CONVERT(char(8), [calendar_date], 112)),
        YEAR([calendar_date]) * 100 + MONTH([calendar_date]),
        YEAR([calendar_date]),
        DATEPART(quarter, [calendar_date]),
        MONTH([calendar_date]),
        DATENAME(month, [calendar_date]),
        DATEFROMPARTS(YEAR([calendar_date]), MONTH([calendar_date]), 1),
        EOMONTH([calendar_date]),
        DAY([calendar_date]),
        CASE WHEN [calendar_date] = EOMONTH([calendar_date]) THEN 1 ELSE 0 END
    FROM [calendar_dates]
    OPTION (MAXRECURSION 0);

    INSERT INTO [QBOT_LIVE_TEST].[D_REGION]
        ([REGION_SK], [REGION_CODE], [REGION_NAME])
    VALUES
        (1, 'NORTH', 'Northern Region'),
        (2, 'SOUTH', 'Southern Region');

    INSERT INTO [QBOT_LIVE_TEST].[D_WAREHOUSE]
        ([WAREHOUSE_SK], [WAREHOUSE_CODE], [WAREHOUSE_NAME], [REGION_SK], [ACTIVE_FLAG])
    VALUES
        (10, 'W01', 'Chennai Central Warehouse', 1, 1),
        (20, 'W02', 'Bengaluru Warehouse', 1, 1),
        (30, 'W03', 'Coimbatore Warehouse', 2, 1);

    INSERT INTO [QBOT_LIVE_TEST].[D_CUSTOMER_SCD2]
        ([CUSTOMER_SK], [CUSTOMER_BK], [CUSTOMER_NAME], [CUSTOMER_SEGMENT],
         [EFFECTIVE_FROM_DATE], [EFFECTIVE_TO_DATE], [CURRENT_FLAG])
    VALUES
        (101, 'C001', 'Atlas Construction - Legacy', 'SMB',
         '2025-01-01', '2026-01-31', 0),
        (102, 'C001', 'Atlas Construction Ltd', 'Enterprise',
         '2026-02-01', '9999-12-31', 1),
        (201, 'C002', 'Beacon Infrastructure', 'Mid-Market',
         '2025-01-01', '9999-12-31', 1);

    INSERT INTO [QBOT_LIVE_TEST].[D_PRODUCT]
        ([PRODUCT_SK], [PRODUCT_CODE], [PRODUCT_NAME], [PRODUCT_FAMILY], [UNIT_OF_MEASURE])
    VALUES
        (100, 'P100', 'Compact Excavator', 'Heavy Equipment', 'EA'),
        (200, 'P200', 'Rotary Drill', 'Power Tools', 'EA'),
        (300, 'P300', 'Safety Helmet', 'Safety Equipment', 'EA');

    INSERT INTO [QBOT_LIVE_TEST].[D_CATEGORY]
        ([CATEGORY_SK], [CATEGORY_CODE], [CATEGORY_NAME])
    VALUES
        (1000, 'HARDWARE', 'Hardware'),
        (2000, 'ELECTRICAL', 'Electrical'),
        (3000, 'SAFETY', 'Safety');

    INSERT INTO [QBOT_LIVE_TEST].[B_PRODUCT_CATEGORY]
        ([PRODUCT_SK], [CATEGORY_SK], [ALLOCATION_PCT])
    VALUES
        (100, 1000, 1.000000),
        (200, 1000, 0.500000),
        (200, 2000, 0.500000),
        (300, 3000, 1.000000);

    INSERT INTO [QBOT_LIVE_TEST].[F_SALES_INVOICE] (
        [SALES_LINE_SK], [INVOICE_NUMBER], [INVOICE_DATE_SK], [ORDER_DATE_SK],
        [POSTING_YYYYMM], [CUSTOMER_SK], [PRODUCT_SK], [WAREHOUSE_SK],
        [QUANTITY], [GROSS_AMOUNT], [DISCOUNT_AMOUNT], [NET_REVENUE_AMOUNT],
        [UPDATED_AT_UTC]
    )
    SELECT
        v.[SALES_LINE_SK], v.[INVOICE_NUMBER], invoice_date.[DATE_SK], order_date.[DATE_SK],
        v.[POSTING_YYYYMM], v.[CUSTOMER_SK], v.[PRODUCT_SK], v.[WAREHOUSE_SK],
        v.[QUANTITY], v.[GROSS_AMOUNT], v.[DISCOUNT_AMOUNT], v.[NET_REVENUE_AMOUNT],
        v.[UPDATED_AT_UTC]
    FROM (VALUES
        (CAST(1 AS bigint), 'INV-1001', CAST('2026-01-30' AS date), CAST('2026-01-28' AS date), 202601, 101, 100, 10, CAST(1 AS decimal(18,3)), CAST(110 AS decimal(19,4)), CAST(10 AS decimal(19,4)), CAST(100 AS decimal(19,4)), CAST('2026-01-30T09:00:00' AS datetime2(0))),
        (CAST(2 AS bigint), 'INV-1002', CAST('2026-01-31' AS date), CAST('2026-01-29' AS date), 202602, 201, 200, 20, CAST(5 AS decimal(18,3)), CAST(55 AS decimal(19,4)), CAST(5 AS decimal(19,4)), CAST(50 AS decimal(19,4)), CAST('2026-02-01T01:00:00' AS datetime2(0))),
        (CAST(3 AS bigint), 'INV-1003', CAST('2026-02-01' AS date), CAST('2026-01-30' AS date), 202602, 102, 100, 10, CAST(2 AS decimal(18,3)), CAST(220 AS decimal(19,4)), CAST(20 AS decimal(19,4)), CAST(200 AS decimal(19,4)), CAST('2026-02-01T10:00:00' AS datetime2(0))),
        (CAST(4 AS bigint), 'INV-1004', CAST('2026-02-02' AS date), CAST('2026-02-01' AS date), 202602, 201, 200, 20, CAST(12 AS decimal(18,3)), CAST(125 AS decimal(19,4)), CAST(5 AS decimal(19,4)), CAST(120 AS decimal(19,4)), CAST('2026-02-02T11:00:00' AS datetime2(0))),
        (CAST(5 AS bigint), 'INV-1005', CAST('2026-02-28' AS date), CAST('2026-02-25' AS date), 202603, 102, 300, 30, CAST(16 AS decimal(18,3)), CAST(90 AS decimal(19,4)), CAST(10 AS decimal(19,4)), CAST(80 AS decimal(19,4)), CAST('2026-03-01T00:30:00' AS datetime2(0))),
        (CAST(6 AS bigint), 'INV-1006', CAST('2026-03-01' AS date), CAST('2026-02-27' AS date), 202603, 102, 100, 10, CAST(3 AS decimal(18,3)), CAST(330 AS decimal(19,4)), CAST(30 AS decimal(19,4)), CAST(300 AS decimal(19,4)), CAST('2026-03-01T12:00:00' AS datetime2(0))),
        (CAST(7 AS bigint), 'INV-1007', CAST('2026-03-02' AS date), CAST('2026-03-01' AS date), 202603, 201, 200, 20, CAST(15 AS decimal(18,3)), CAST(160 AS decimal(19,4)), CAST(10 AS decimal(19,4)), CAST(150 AS decimal(19,4)), CAST('2026-03-02T13:00:00' AS datetime2(0))),
        (CAST(8 AS bigint), 'INV-1008', CAST('2026-03-03' AS date), CAST('2026-03-02' AS date), 202603, 102, 300, 30, CAST(10 AS decimal(18,3)), CAST(55 AS decimal(19,4)), CAST(5 AS decimal(19,4)), CAST(50 AS decimal(19,4)), CAST('2026-03-03T14:00:00' AS datetime2(0)))
    ) v (
        [SALES_LINE_SK], [INVOICE_NUMBER], [INVOICE_DATE], [ORDER_DATE],
        [POSTING_YYYYMM], [CUSTOMER_SK], [PRODUCT_SK], [WAREHOUSE_SK],
        [QUANTITY], [GROSS_AMOUNT], [DISCOUNT_AMOUNT], [NET_REVENUE_AMOUNT],
        [UPDATED_AT_UTC]
    )
    JOIN [QBOT_LIVE_TEST].[D_DATE] invoice_date
        ON invoice_date.[FULL_DATE] = v.[INVOICE_DATE]
    JOIN [QBOT_LIVE_TEST].[D_DATE] order_date
        ON order_date.[FULL_DATE] = v.[ORDER_DATE];

    INSERT INTO [QBOT_LIVE_TEST].[F_INVENTORY_DAILY] (
        [DAILY_SNAPSHOT_SK], [SNAPSHOT_YYYYMMDD], [SNAPSHOT_DATE], [SNAPSHOT_AT_UTC],
        [PRODUCT_SK], [WAREHOUSE_SK], [ON_HAND_QUANTITY], [ALLOCATED_QUANTITY],
        [UNIT_COST], [INVENTORY_VALUE]
    )
    VALUES
        (1, 20260301, '2026-03-01', '2026-03-01T23:59:00', 100, 10, 10, 2, 100, 1000),
        (2, 20260301, '2026-03-01', '2026-03-01T23:59:00', 200, 20, 20, 4, 10, 200),
        (3, 20260301, '2026-03-01', '2026-03-01T23:59:00', 300, 30, 30, 3, 5, 150),
        (4, 20260302, '2026-03-02', '2026-03-02T23:59:00', 100, 10, 11, 2, 100, 1100),
        (5, 20260302, '2026-03-02', '2026-03-02T23:59:00', 200, 20, 18, 3, 10, 180),
        (6, 20260302, '2026-03-02', '2026-03-02T23:59:00', 300, 30, 28, 3, 5, 140),
        (7, 20260303, '2026-03-03', '2026-03-03T23:59:00', 100, 10, 12, 2, 100, 1200),
        (8, 20260303, '2026-03-03', '2026-03-03T23:59:00', 200, 20, 16, 3, 10, 160),
        (9, 20260303, '2026-03-03', '2026-03-03T23:59:00', 300, 30, 25, 2, 5, 125);

    INSERT INTO [QBOT_LIVE_TEST].[F_INVENTORY_MONTHLY] (
        [MONTHLY_SNAPSHOT_SK], [PERIOD_YYYYMM], [PRODUCT_SK], [WAREHOUSE_SK],
        [ENDING_ON_HAND_QUANTITY], [ENDING_INVENTORY_VALUE]
    )
    VALUES
        (1, 202601, 100, 10, 8, 800),
        (2, 202601, 200, 20, 15, 150),
        (3, 202601, 300, 30, 10, 50),
        (4, 202602, 100, 10, 9, 900),
        (5, 202602, 200, 20, 20, 200),
        (6, 202602, 300, 30, 20, 100),
        (7, 202603, 100, 10, 11, 1100),
        (8, 202603, 200, 20, 18, 180),
        (9, 202603, 300, 30, 24, 120);

    INSERT INTO [QBOT_LIVE_TEST].[ERP_ITM_BAL_PRD_FCT] (
        [ITM_BAL_PRD_FCT_KEY], [WHS_DMS_KEY], [ITM_DMS_KEY], [PRD_DMS_KEY],
        [ON_HND_QTY], [ALLOC_QTY], [INV_VAL_AMT]
    )
    VALUES
        (1, 10, 100, 202602, 9, 2, 900),
        (2, 20, 200, 202602, 20, 4, 200),
        (3, 30, 300, 202602, 20, 2, 100),
        (4, 10, 100, 202603, 12, 2, 1200),
        (5, 20, 200, 202603, 16, 3, 160),
        (6, 30, 300, 202603, 25, 2, 125),
        (7, 10, 300, 0, 999, 0, 9999);

    INSERT INTO [QBOT_LIVE_TEST].[M3_MITBAL]
        ([MLCONO], [MLWHLO], [MLITNO], [MLPERY], [MLSTQT], [MLALQT], [MLAVAL], [MLLMDT])
    VALUES
        (1, 'W01', 'P100', 202602, 9, 2, 900, 20260228),
        (1, 'W02', 'P200', 202602, 20, 4, 200, 20260228),
        (1, 'W03', 'P300', 202602, 20, 2, 100, 20260228),
        (1, 'W01', 'P100', 202603, 12, 2, 1200, 20260303),
        (1, 'W02', 'P200', 202603, 16, 3, 160, 20260303),
        (1, 'W03', 'P300', 202603, 25, 2, 125, 20260303);

    INSERT INTO [QBOT_LIVE_TEST].[F_RETURNS] (
        [RETURN_LINE_SK], [RETURN_NUMBER], [RETURN_DATE_SK], [ORIGINAL_INVOICE_DATE_SK],
        [CUSTOMER_SK], [PRODUCT_SK], [WAREHOUSE_SK], [RETURN_QUANTITY], [REFUND_AMOUNT]
    )
    SELECT
        v.[RETURN_LINE_SK], v.[RETURN_NUMBER], return_date.[DATE_SK], invoice_date.[DATE_SK],
        v.[CUSTOMER_SK], v.[PRODUCT_SK], v.[WAREHOUSE_SK],
        v.[RETURN_QUANTITY], v.[REFUND_AMOUNT]
    FROM (VALUES
        (CAST(1 AS bigint), 'RET-1001', CAST('2026-02-02' AS date), CAST('2026-02-01' AS date), 102, 100, 10, CAST(1 AS decimal(18,3)), CAST(20 AS decimal(19,4))),
        (CAST(2 AS bigint), 'RET-1002', CAST('2026-03-03' AS date), CAST('2026-03-01' AS date), 102, 100, 10, CAST(1 AS decimal(18,3)), CAST(30 AS decimal(19,4)))
    ) v (
        [RETURN_LINE_SK], [RETURN_NUMBER], [RETURN_DATE], [ORIGINAL_INVOICE_DATE],
        [CUSTOMER_SK], [PRODUCT_SK], [WAREHOUSE_SK], [RETURN_QUANTITY], [REFUND_AMOUNT]
    )
    JOIN [QBOT_LIVE_TEST].[D_DATE] return_date
        ON return_date.[FULL_DATE] = v.[RETURN_DATE]
    JOIN [QBOT_LIVE_TEST].[D_DATE] invoice_date
        ON invoice_date.[FULL_DATE] = v.[ORIGINAL_INVOICE_DATE];

    INSERT INTO [QBOT_LIVE_TEST].[F_ORDERS] (
        [ORDER_SK], [ORDER_NUMBER], [ORDER_DATE_SK], [CUSTOMER_SK],
        [WAREHOUSE_SK], [ORDER_STATUS], [ORDER_AMOUNT]
    )
    SELECT
        v.[ORDER_SK], v.[ORDER_NUMBER], d.[DATE_SK], v.[CUSTOMER_SK],
        v.[WAREHOUSE_SK], v.[ORDER_STATUS], v.[ORDER_AMOUNT]
    FROM (VALUES
        (CAST(1001 AS bigint), 'ORD-1001', CAST('2026-01-28' AS date), 101, 10, 'SHIPPED', CAST(100 AS decimal(19,4))),
        (CAST(1002 AS bigint), 'ORD-1002', CAST('2026-02-01' AS date), 201, 20, 'OPEN', CAST(120 AS decimal(19,4))),
        (CAST(1003 AS bigint), 'ORD-1003', CAST('2026-03-01' AS date), 102, 10, 'SHIPPED', CAST(300 AS decimal(19,4))),
        (CAST(1004 AS bigint), 'ORD-1004', CAST('2026-03-03' AS date), 102, 30, 'OPEN', CAST(50 AS decimal(19,4)))
    ) v ([ORDER_SK], [ORDER_NUMBER], [ORDER_DATE], [CUSTOMER_SK], [WAREHOUSE_SK], [ORDER_STATUS], [ORDER_AMOUNT])
    JOIN [QBOT_LIVE_TEST].[D_DATE] d ON d.[FULL_DATE] = v.[ORDER_DATE];

    INSERT INTO [QBOT_LIVE_TEST].[F_SHIPMENTS] (
        [SHIPMENT_SK], [SHIPMENT_NUMBER], [ORDER_SK], [SHIPMENT_DATE_SK],
        [WAREHOUSE_SK], [SHIPPED_AMOUNT]
    )
    SELECT
        v.[SHIPMENT_SK], v.[SHIPMENT_NUMBER], v.[ORDER_SK], d.[DATE_SK],
        v.[WAREHOUSE_SK], v.[SHIPPED_AMOUNT]
    FROM (VALUES
        (CAST(5001 AS bigint), 'SHP-1001', CAST(1001 AS bigint), CAST('2026-01-30' AS date), 10, CAST(100 AS decimal(19,4))),
        (CAST(5002 AS bigint), 'SHP-1002', CAST(1003 AS bigint), CAST('2026-03-02' AS date), 10, CAST(300 AS decimal(19,4)))
    ) v ([SHIPMENT_SK], [SHIPMENT_NUMBER], [ORDER_SK], [SHIPMENT_DATE], [WAREHOUSE_SK], [SHIPPED_AMOUNT])
    JOIN [QBOT_LIVE_TEST].[D_DATE] d ON d.[FULL_DATE] = v.[SHIPMENT_DATE];

    INSERT INTO [QBOT_LIVE_TEST].[D_DUPLICATE_CODE]
        ([DUPLICATE_CODE_SK], [DUP_CODE], [DUP_DESCRIPTION])
    VALUES
        (1, 'A', 'First A description'),
        (2, 'A', 'Second A description'),
        (3, 'B', 'Only B description');

    INSERT INTO [QBOT_LIVE_TEST].[F_BAD_CODE]
        ([BAD_FACT_SK], [TARGET_CODE], [MEASURE_AMOUNT])
    VALUES
        (1, 'A', 10),
        (2, 'B', 20);

    INSERT INTO [QBOT_LIVE_TEST].[TEST_EXPECTED_CASES] (
        [CASE_ID], [CAPABILITY], [QUESTION_TEXT], [EXPECTED_RESULT],
        [REQUIRED_TABLES], [REQUIRED_DATE_ROLE], [FORBIDDEN_BEHAVIOR]
    )
    VALUES
        (1, 'star_join',
         N'What is total revenue by warehouse?',
         N'W01=600.00; W02=320.00; W03=130.00; total=1050.00.',
         N'F_SALES_INVOICE,D_WAREHOUSE', NULL,
         N'Do not join another fact.'),
        (2, 'snowflake_join',
         N'Show revenue by region.',
         N'Northern Region=920.00; Southern Region=130.00.',
         N'F_SALES_INVOICE,D_WAREHOUSE,D_REGION', NULL,
         N'Do not invent a direct sales-to-region join.'),
        (3, 'surrogate_date_role',
         N'Compare revenue for the current month and previous month.',
         N'Current data month March 2026=500.00; previous month February 2026=400.00; increase=25%.',
         N'F_SALES_INVOICE,D_DATE', 'Invoice Date',
         N'Do not convert INVOICE_DATE_SK as YYYYMMDD and do not anchor to the future calendar maximum.'),
        (4, 'relative_last_days',
         N'What was revenue for the last 2 data days?',
         N'2026-03-02=150.00; 2026-03-03=50.00; total=200.00.',
         N'F_SALES_INVOICE,D_DATE', 'Invoice Date',
         N'Do not use server current date and do not use Order Date.'),
        (5, 'date_role_disambiguation',
         N'Show January revenue by invoice date, not posting month.',
         N'January 2026 revenue=150.00.',
         N'F_SALES_INVOICE,D_DATE', 'Invoice Date',
         N'Do not filter POSTING_YYYYMM because INV-1002 was posted in February.'),
        (6, 'yyyy_mm',
         N'Show month-end inventory value for February 2026.',
         N'PERIOD_YYYYMM=202602; total ending inventory value=1200.00.',
         N'F_INVENTORY_MONTHLY', 'Inventory Period',
         N'Do not treat 202602 as a native DATE and do not sum daily snapshots.'),
        (7, 'yyyy_mm_dd',
         N'What was inventory value by warehouse on the latest daily snapshot?',
         N'Latest data date=2026-03-03; W01=1200.00; W02=160.00; W03=125.00; total=1485.00.',
         N'F_INVENTORY_DAILY,D_WAREHOUSE', 'Inventory Snapshot Date',
         N'Do not sum all snapshots and do not choose the monthly fact.'),
        (8, 'native_date',
         N'Show daily inventory value for March 2, 2026.',
         N'SNAPSHOT_DATE=2026-03-02; total=1420.00.',
         N'F_INVENTORY_DAILY', 'Inventory Snapshot Date',
         N'Do not reinterpret SNAPSHOT_YYYYMMDD as a surrogate join key.'),
        (9, 'timestamp',
         N'Which invoices were updated after midnight on March 1, 2026?',
         N'INV-1005, INV-1006, INV-1007 and INV-1008.',
         N'F_SALES_INVOICE', 'Updated Timestamp',
         N'Do not apply the Invoice Date role to this operational timestamp request.'),
        (10, 'daily_vs_monthly_grain',
         N'Give me the February month-end inventory, then the latest daily inventory.',
         N'Monthly February=1200.00 from F_INVENTORY_MONTHLY; latest daily=1485.00 from F_INVENTORY_DAILY.',
         N'F_INVENTORY_MONTHLY,F_INVENTORY_DAILY', NULL,
         N'Do not union or raw-join facts with different snapshot grains.'),
        (11, 'scd2',
         N'Show revenue for customer C001 across its historical versions.',
         N'C001 total=730.00; legacy version=100.00; current version=630.00.',
         N'F_SALES_INVOICE,D_CUSTOMER_SCD2', NULL,
         N'Do not join CUSTOMER_BK directly and fan out historical versions.'),
        (12, 'bridge_allocation',
         N'Allocate total revenue by product category.',
         N'Hardware=760.00; Electrical=160.00; Safety=130.00; allocated total=1050.00.',
         N'F_SALES_INVOICE,D_PRODUCT,B_PRODUCT_CATEGORY,D_CATEGORY', NULL,
         N'Do not duplicate P200 revenue; apply ALLOCATION_PCT.'),
        (13, 'multi_fact_isolation',
         N'Compare March revenue with latest inventory value by warehouse.',
         N'Revenue=500.00 and latest inventory=1485.00; aggregate each fact separately before combining by warehouse.',
         N'F_SALES_INVOICE,F_INVENTORY_DAILY,D_WAREHOUSE', NULL,
         N'No SELECT scope may contain both raw facts.'),
        (14, 'role_playing_return_date',
         N'Show refunds by return date.',
         N'2026-02-02=20.00; 2026-03-03=30.00.',
         N'F_RETURNS,D_DATE', 'Return Date',
         N'Do not substitute Original Invoice Date.'),
        (15, 'governed_anti_join',
         N'Which orders have not been shipped?',
         N'ORD-1002 and ORD-1004.',
         N'F_ORDERS,F_SHIPMENTS', NULL,
         N'Use NOT EXISTS or a governed anti-join; do not aggregate raw facts together.'),
        (16, 'm3_cryptic_yyyy_mm',
         N'Using the M3 balance data, show March inventory value by warehouse.',
         N'MLPERY=202603; W01=1200.00; W02=160.00; W03=125.00.',
         N'M3_MITBAL,D_WAREHOUSE', 'M3 Inventory Period',
         N'Do not ignore MLPERY because its name is cryptic; it is YYYYMM.'),
        (17, 'unsafe_relationship',
         N'Profile whether TARGET_CODE can safely join to DUP_CODE.',
         N'Target duplicate key count=1; joined rows=3 for 2 fact rows; fanout ratio=1.5; relationship must remain blocked or warning.',
         N'F_BAD_CODE,D_DUPLICATE_CODE', NULL,
         N'Do not approve this as many-to-one.'),
        (18, 'formatting_follow_up',
         N'Now display the month as MMM-YY and revenue as currency with two decimals.',
         N'Jan-26 $150.00; Feb-26 $400.00; Mar-26 $500.00 without rerunning different business logic.',
         N'F_SALES_INVOICE,D_DATE', 'Invoice Date',
         N'Do not lose prior-result lineage or reinterpret the metric.'),
        (19, 'erp_prd_dms_yyyymm',
         N'From the ERP item balance fact, show the latest monthly inventory value by warehouse.',
         N'Latest valid PRD_DMS_KEY=202603; W01=1200.00; W02=160.00; W03=125.00; exclude sentinel period 0.',
         N'ERP_ITM_BAL_PRD_FCT,D_WAREHOUSE', 'Inventory Period',
         N'Do not treat PRD_DMS_KEY as a native date or require a D_DATE join; exclude zero sentinel values.');

    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    IF XACT_STATE() <> 0
        ROLLBACK TRANSACTION;
    THROW;
END CATCH;

SELECT [CASE_ID], [CAPABILITY], [QUESTION_TEXT], [EXPECTED_RESULT]
FROM [QBOT_LIVE_TEST].[TEST_EXPECTED_CASES]
ORDER BY [CASE_ID];
