# Join Planner V2 Azure SQL acceptance runbook

## Deploy the fixture

Run `azure_sql_join_planner_v2.sql` as one Azure SQL script. It is rerunnable and intentionally contains no batch separator. It creates only the `QBOT_JOIN_TEST` schema and objects inside it.

In QueryBot, select all tables in `QBOT_JOIN_TEST`, run schema discovery, build the KB, then review the Entity Graph before asking questions.

## Expected classifications

| Table | Expected role | Grain |
|---|---|---|
| `FACT_SALES` | Fact / transaction | Invoice number + line number |
| `FACT_INVENTORY_DAILY` | Fact / periodic snapshot | Snapshot date + product + warehouse |
| `FACT_INVENTORY_MONTHLY` | Fact / periodic snapshot | Period date + product + warehouse |
| `FACT_RETURNS` | Fact / transaction | Return line |
| `DIM_DATE` | Date dimension | Calendar date |
| `DIM_CUSTOMER_SCD2` | Dimension / SCD2 | Customer surrogate version |
| `DIM_WAREHOUSE` | Dimension | Warehouse |
| `DIM_REGION` | Dimension | Region |
| `DIM_PRODUCT` | Dimension | Product |
| `BRIDGE_PRODUCT_CATEGORY` | Bridge | Product + category |

`FACT_BAD_RELATIONSHIP.TARGET_CODE -> DIM_DUPLICATE_CODE.TARGET_CODE` must not be approved as a many-to-one relationship: the target contains one duplicate key and produces 1.5x fanout.

## Required graph paths

- `FACT_SALES.WAREHOUSE_KEY -> DIM_WAREHOUSE.WAREHOUSE_KEY`
- `DIM_WAREHOUSE.REGION_KEY -> DIM_REGION.REGION_KEY`
- `FACT_SALES.INVOICE_DATE_KEY -> DIM_DATE.DATE_KEY` as **Invoice Date**
- `FACT_SALES.ORDER_DATE_KEY -> DIM_DATE.DATE_KEY` as **Order Date**
- `FACT_RETURNS.RETURN_DATE_KEY -> DIM_DATE.DATE_KEY` as **Return Date**
- `FACT_RETURNS.ORIGINAL_INVOICE_DATE_KEY -> DIM_DATE.DATE_KEY` as **Original Invoice Date**
- Facts join `DIM_CUSTOMER_SCD2` by `CUSTOMER_SK`, never by non-unique `CUSTOMER_BK`.
- Product category analysis traverses `BRIDGE_PRODUCT_CATEGORY` and uses `ALLOCATION_PCT`.

## Portal acceptance questions

1. **What is total revenue by warehouse?**
   Group totals must sum to **720.00**.

2. **Show revenue by region.**
   Northern Region = **640.00**; Southern Region = **80.00**. The SQL must use the warehouse snowflake path.

3. **Compare January and February revenue by invoice month.**
   January = **220.00**; February = **500.00**. The SQL must join `INVOICE_DATE_KEY` to `DIM_DATE.DATE_KEY`.

4. **Show revenue for customer C001 using the historical customer version.**
   Total = **320.00**. The SQL must use `CUSTOMER_SK`; joining `CUSTOMER_BK` would fan out across SCD2 versions.

5. **Allocate revenue by product category.**
   Hardware = **470.00**; Electrical = **100.00**; Safety = **150.00**. Allocated totals must remain **720.00**.

6. **What was inventory value by warehouse on the latest daily snapshot?**
   The latest scoped date is **2026-02-03** and total inventory value is **3000.00**.

7. **Show month-end inventory value for February.**
   QueryBot must select `FACT_INVENTORY_MONTHLY`, not sum daily snapshots. Total = **2300.00**.

8. **Compare February revenue and latest inventory value by warehouse.**
   The SQL must have one aggregate CTE for `FACT_SALES` and another for `FACT_INVENTORY_DAILY`. It may join only the aggregated CTEs/dimension grain. Expected totals: revenue **500.00**, inventory value **3000.00**.

9. **Show refunds by return date.**
   2026-02-02 = **20.00**; 2026-02-03 = **10.00**. The Original Invoice Date role must not replace Return Date.

10. **Profile the candidate TARGET_CODE relationship.**
    It must report duplicate target keys/fanout and remain warning or blocked.

The same cases are stored in `QBOT_JOIN_TEST.TEST_EXPECTED_CASES` for automated or manual comparison.

## Production pass criteria

- No raw fact-to-fact SQL is executed.
- Every relationship uses exact governed key pairs and required join type.
- Every selected fact has one declared grain.
- Multi-fact questions aggregate facts independently to a conformed grain.
- Bridge measures use allocation or a governed distinct business key.
- SCD2 joins use surrogate versions and respect effective/current filters when requested.
- Role-playing date questions use the requested business date.
- Invalid, ambiguous, duplicate-key, or missing paths fail closed with a useful explanation.
