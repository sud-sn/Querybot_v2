# QueryBot Azure SQL live regression dataset

This package creates an isolated dataset for testing QueryBot against a real
Azure SQL connection and the configured LLM. It does not modify the existing
`dbo`, `HR`, `LOGS`, `PHARMA_LAB`, `PHARMACY`, or `profitability` schemas.

## Run order

Execute each complete file against `chatbot_db` in this order:

1. `01_create_schema.sql`
2. `02_seed_data.sql`
3. `03_validate_expected_results.sql`

The scripts contain no `GO` separators. The first two scripts are rerunnable
and transaction-protected. They affect only the `QBOT_LIVE_TEST` schema.

## QueryBot onboarding

Create a dedicated test client and select these tables:

- `D_DATE`
- `D_REGION`
- `D_WAREHOUSE`
- `D_CUSTOMER_SCD2`
- `D_PRODUCT`
- `D_CATEGORY`
- `B_PRODUCT_CATEGORY`
- `F_SALES_INVOICE`
- `F_INVENTORY_DAILY`
- `F_INVENTORY_MONTHLY`
- `ERP_ITM_BAL_PRD_FCT`
- `M3_MITBAL`
- `F_RETURNS`
- `F_ORDERS`
- `F_SHIPMENTS`

Use `D_DUPLICATE_CODE` and `F_BAD_CODE` only for the unsafe-relationship test.
Do not select `TEST_EXPECTED_CASES` as a business table.

Suggested business description:

> This test workspace represents an equipment distributor that invoices
> customers, manages products across regional warehouses, records returns and
> shipments, and stores both daily and month-end inventory snapshots. Sales
> revenue is the sum of net revenue amount. Invoice Date is the default sales
> date. Inventory questions must choose daily or monthly data based on the
> requested grain. M3_MITBAL is an Infor M3-style monthly item balance table:
> MLPERY is YYYYMM, MLSTQT is stock quantity, MLALQT is allocated quantity,
> MLAVAL is inventory value, and MLLMDT is a YYYYMMDD last-modified date.

## Semantic setup to verify

QueryBot should discover the foreign-key graph automatically. Review that the
following date roles and metric defaults are present before the final run:

| Source | Physical field | Storage | Governed role | Mapping |
|---|---|---|---|---|
| `F_SALES_INVOICE` | `INVOICE_DATE_SK` | Surrogate FK | Invoice Date | `D_DATE.DATE_SK -> FULL_DATE` |
| `F_SALES_INVOICE` | `ORDER_DATE_SK` | Surrogate FK | Order Date | `D_DATE.DATE_SK -> FULL_DATE` |
| `F_RETURNS` | `RETURN_DATE_SK` | Surrogate FK | Return Date | `D_DATE.DATE_SK -> FULL_DATE` |
| `F_RETURNS` | `ORIGINAL_INVOICE_DATE_SK` | Surrogate FK | Original Invoice Date | `D_DATE.DATE_SK -> FULL_DATE` |
| `F_INVENTORY_DAILY` | `SNAPSHOT_YYYYMMDD` | Encoded `YYYYMMDD` | Inventory Snapshot Date | Direct encoded date |
| `F_INVENTORY_DAILY` | `SNAPSHOT_DATE` | Native date | Inventory Snapshot Date | Direct native date |
| `F_INVENTORY_MONTHLY` | `PERIOD_YYYYMM` | Encoded `YYYYMM` | Inventory Period | Direct encoded month |
| `ERP_ITM_BAL_PRD_FCT` | `PRD_DMS_KEY` | Encoded `YYYYMM` | Inventory Period | Direct encoded month; zero is a sentinel |
| `M3_MITBAL` | `MLPERY` | Encoded `YYYYMM` | M3 Inventory Period | Direct encoded month |
| `M3_MITBAL` | `MLLMDT` | Encoded `YYYYMMDD` | Last Modified Date | Direct encoded date |
| `F_SALES_INVOICE` | `UPDATED_AT_UTC` | Timestamp | Updated Timestamp | Direct timestamp |

Create or confirm these metrics:

- **Revenue** = `SUM(F_SALES_INVOICE.NET_REVENUE_AMOUNT)`, default time column
  `INVOICE_DATE_SK`, currency.
- **Daily Inventory Value** =
  `SUM(F_INVENTORY_DAILY.INVENTORY_VALUE)`, default time column
  `SNAPSHOT_DATE`, currency, daily snapshot grain.
- **Month-End Inventory Value** =
  `SUM(F_INVENTORY_MONTHLY.ENDING_INVENTORY_VALUE)`, default time column
  `PERIOD_YYYYMM`, currency, monthly snapshot grain.
- **M3 Inventory Value** = `SUM(M3_MITBAL.MLAVAL)`, default time column
  `MLPERY`, currency, monthly item/warehouse grain.
- **ERP Period Inventory Value** =
  `SUM(ERP_ITM_BAL_PRD_FCT.INV_VAL_AMT)`, default time column
  `PRD_DMS_KEY`, currency, monthly item/warehouse grain; ignore period zero.
- **Refund Amount** = `SUM(F_RETURNS.REFUND_AMOUNT)`, default time column
  `RETURN_DATE_SK`, currency.

For a repeatable baseline, the fixture includes an idempotent seeder for the
three sales metrics used by the compiler regression tests. Run it from the
QueryBot repository root after schema discovery / KB generation (replace the
account id if your test client uses another value):

```powershell
python tests/fixtures/azure_sql_live_regression/04_seed_querybot_metrics.py --account Test_Az
```

This creates or updates **Revenue**, **Gross Sales**, and **Discount Amount**
and binds all three to the approved Invoice Date role. It does not modify
users, access policy, or tenant-governance data.

## Live test sequence

Start a new portal thread and run the questions in
`QBOT_LIVE_TEST.TEST_EXPECTED_CASES` in `CASE_ID` order. Keep cases 3, 4 and 18
in the same thread to verify durable context and result-lineage formatting.

Important regression checks:

- Case 3 must return March 2026 versus February 2026. `D_DATE` contains future
  dates through December specifically to catch the old unrestricted-calendar
  anchor bug.
- `INVOICE_DATE_SK` is a non-date surrogate. It must be joined to `D_DATE` and
  must never be converted directly as `YYYYMMDD`.
- Cases 6, 7, 8 and 16 distinguish `YYYYMM`, `YYYYMMDD`, native date, and
  cryptic ERP date storage.
- Case 10 must choose the appropriate snapshot fact rather than mixing daily
  and monthly grains.
- Case 13 must aggregate the two facts independently before joining their
  results by warehouse.
- Case 15 is the narrow governed fact-to-fact anti-join exception. It must use
  `NOT EXISTS` or equivalent without aggregating the raw facts together.
- Case 17 must remain warning/blocked because the target code is duplicated.
- Case 18 is a presentation follow-up. It must reuse the prior result lineage
  and must not change the business calculation.

## Production gate

The implementation is ready to promote only when:

- all 19 results agree with `TEST_EXPECTED_CASES`;
- generated SQL parses and executes without repair retries;
- all surrogate date roles visibly join through `D_DATE`;
- relative periods anchor to scoped fact data;
- no raw multi-fact query is accepted;
- bridge allocation preserves the total;
- SCD2 joins use `CUSTOMER_SK`, not the repeating business key;
- the unsafe relationship is not approved;
- the server log contains no unhandled exception or semantic-plan mismatch.
