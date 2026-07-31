# CHATBOT_DB.PHARMA_LAB.F_INVENTORY_SNAPSHOT

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.F_INVENTORY_SNAPSHOT`  — always use this exact three-part name in generated SQL.

**Row count:** 1,440  **Scale:** Small

**Primary Key:** `INVENTORY_SNAPSHOT_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `INVENTORY_SNAPSHOT_ID` **[PK]** | bigint(19) | No |  |
| `PHARMACY_ID` | int(10) | No |  |
| `DRUG_ID` | int(10) | No |  |
| `SUPPLIER_ID` | int(10) | No |  |
| `SNAPSHOT_DATE_ID` | int(10) | No |  |
| `EXPIRY_DATE_ID` | int(10) | No |  |
| `LAST_RECEIPT_DATE_ID` | int(10) | No |  |
| `LOT_NUMBER` | varchar(40) | No |  |
| `ON_HAND_QUANTITY` | decimal(18) | No |  |
| `ALLOCATED_QUANTITY` | decimal(18) | No |  |
| `AVAILABLE_QUANTITY` | decimal(18) | No |  |
| `REORDER_POINT_QUANTITY` | decimal(18) | No |  |
| `EXPIRED_QUANTITY` | decimal(18) | No |  |
| `UNIT_COST_AMT` | decimal(18) | No |  |
| `INVENTORY_VALUE_AMT` | decimal(18) | No |  |
| `IS_SYNTHETIC` | bit | No |  |
| `CREATED_AT_UTC` | datetime2 | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `INVENTORY_SNAPSHOT_ID` | 0.0% | 1300001 | 1301440 |
| `PHARMACY_ID` | 0.0% | 4001 | 4005 |
| `DRUG_ID` | 0.0% | 5001 | 5012 |
| `SUPPLIER_ID` | 0.0% | 7001 | 7004 |
| `SNAPSHOT_DATE_ID` | 0.0% | 116468 | 117167 |
| `EXPIRY_DATE_ID` | 0.0% | 116760 | 117493 |
| `LAST_RECEIPT_DATE_ID` | 0.0% | 116457 | 117153 |
| `ON_HAND_QUANTITY` | 0.0% | 20.000 | 179.000 |
| `ALLOCATED_QUANTITY` | 0.0% | 0.000 | 21.000 |
| `AVAILABLE_QUANTITY` | 0.0% | 0.000 | 172.000 |
| `REORDER_POINT_QUANTITY` | 0.0% | 10.000 | 44.000 |
| `EXPIRED_QUANTITY` | 0.0% | 0.000 | 6.000 |
| `UNIT_COST_AMT` | 0.0% | 12.6500 | 102.3000 |
| `INVENTORY_VALUE_AMT` | 0.0% | 253.00 | 18311.70 |
| `CREATED_AT_UTC` | 0.0% | 2025-01-31 00:00:00 | 2026-12-31 00:00:00 |

## Sample data

| INVENTORY_SNAPSHOT_ID | PHARMACY_ID | DRUG_ID | SUPPLIER_ID | SNAPSHOT_DATE_ID | EXPIRY_DATE_ID | LAST_RECEIPT_DATE_ID | LOT_NUMBER | ON_HAND_QUANTITY | ALLOCATED_QUANTITY | AVAILABLE_QUANTITY | REORDER_POINT_QUANTITY | EXPIRED_QUANTITY | UNIT_COST_AMT | INVENTORY_VALUE_AMT | IS_SYNTHETIC | CREATED_AT_UTC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1300001 | 4001 | 5001 | 7004 | 116468 | 116760 | 116457 | SYN-LOT-4001-5001-01 | 171.000 | 5.000 | 166.000 | 42.000 | 0.000 | 12.6500 | 2163.15 | True | 2025-01-31 00:00:00 |
| 1300002 | 4001 | 5002 | 7001 | 116468 | 116761 | 116457 | SYN-LOT-4001-5002-01 | 22.000 | 6.000 | 16.000 | 43.000 | 0.000 | 20.8000 | 457.60 | True | 2025-01-31 00:00:00 |
| 1300003 | 4001 | 5003 | 7002 | 116468 | 116762 | 116457 | SYN-LOT-4001-5003-01 | 33.000 | 7.000 | 26.000 | 44.000 | 0.000 | 28.9500 | 955.35 | True | 2025-01-31 00:00:00 |
| 1300004 | 4001 | 5004 | 7003 | 116468 | 116763 | 116457 | SYN-LOT-4001-5004-01 | 44.000 | 8.000 | 36.000 | 10.000 | 0.000 | 37.1000 | 1632.40 | True | 2025-01-31 00:00:00 |
| 1300005 | 4001 | 5005 | 7004 | 116468 | 116764 | 116457 | SYN-LOT-4001-5005-01 | 55.000 | 9.000 | 46.000 | 11.000 | 0.000 | 45.2500 | 2488.75 | True | 2025-01-31 00:00:00 |