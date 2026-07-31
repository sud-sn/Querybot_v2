# CHATBOT_DB.PHARMA_LAB.F_PURCHASE_RECEIPT

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.F_PURCHASE_RECEIPT`  — always use this exact three-part name in generated SQL.

**Row count:** 240  **Scale:** Small

**Primary Key:** `PURCHASE_RECEIPT_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `PURCHASE_RECEIPT_ID` **[PK]** | bigint(19) | No |  |
| `PURCHASE_ORDER_NUMBER` | varchar(30) | No |  |
| `RECEIPT_LINE_NUMBER` | smallint(5) | No |  |
| `PHARMACY_ID` | int(10) | No |  |
| `DRUG_ID` | int(10) | No |  |
| `SUPPLIER_ID` | int(10) | No |  |
| `PO_DATE_ID` | int(10) | No |  |
| `EXPECTED_DELIVERY_DATE_ID` | int(10) | No |  |
| `RECEIPT_DATE_ID` | int(10) | Yes |  |
| `EXPIRY_DATE_ID` | int(10) | No |  |
| `LOT_NUMBER` | varchar(40) | No |  |
| `ORDERED_QUANTITY` | decimal(18) | No |  |
| `RECEIVED_QUANTITY` | decimal(18) | No |  |
| `REJECTED_QUANTITY` | decimal(18) | No |  |
| `UNIT_COST_AMT` | decimal(18) | No |  |
| `EXTENDED_COST_AMT` | decimal(18) | No |  |
| `RECEIPT_STATUS` | varchar(20) | No | 'OPEN', 'PARTIAL', 'RECEIVED', 'REJECTED' |
| `TEMPERATURE_EXCEPTION_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |
| `CREATED_AT_UTC` | datetime2 | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `PURCHASE_RECEIPT_ID` | 0.0% | 1400001 | 1400240 |
| `RECEIPT_LINE_NUMBER` | 0.0% | 1 | 1 |
| `PHARMACY_ID` | 0.0% | 4001 | 4005 |
| `DRUG_ID` | 0.0% | 5001 | 5012 |
| `SUPPLIER_ID` | 0.0% | 7001 | 7004 |
| `PO_DATE_ID` | 0.0% | 116438 | 117035 |
| `EXPECTED_DELIVERY_DATE_ID` | 0.0% | 116445 | 117042 |
| `RECEIPT_DATE_ID` | 7.5% | 116443 | 117047 |
| `EXPIRY_DATE_ID` | 0.0% | 116807 | 117479 |
| `ORDERED_QUANTITY` | 0.0% | 40.000 | 200.000 |
| `RECEIVED_QUANTITY` | 0.0% | 0.000 | 200.000 |
| `REJECTED_QUANTITY` | 0.0% | 0.000 | 200.000 |
| `UNIT_COST_AMT` | 0.0% | 12.6500 | 102.3000 |
| `EXTENDED_COST_AMT` | 0.0% | 0.00 | 18830.00 |
| `CREATED_AT_UTC` | 0.0% | 2025-01-01 00:00:00 | 2026-08-21 00:00:00 |

## Sample data

| PURCHASE_RECEIPT_ID | PURCHASE_ORDER_NUMBER | RECEIPT_LINE_NUMBER | PHARMACY_ID | DRUG_ID | SUPPLIER_ID | PO_DATE_ID | EXPECTED_DELIVERY_DATE_ID | RECEIPT_DATE_ID | EXPIRY_DATE_ID | LOT_NUMBER | ORDERED_QUANTITY | RECEIVED_QUANTITY | REJECTED_QUANTITY | UNIT_COST_AMT | EXTENDED_COST_AMT | RECEIPT_STATUS | TEMPERATURE_EXCEPTION_FLAG | IS_SYNTHETIC | CREATED_AT_UTC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1400001 | SYN-PO-000001 | 1 | 4003 | 5007 | 7001 | 116441 | 116448 | 116447 | 116807 | SYN-RCT-LOT-000001 | 60.000 | 60.000 | 0.000 | 61.5500 | 3693.00 | RECEIVED | False | True | 2025-01-04 00:00:00 |
| 1400002 | SYN-PO-000002 | 1 | 4001 | 5002 | 7002 | 116444 | 116451 | 116451 | 116811 | SYN-RCT-LOT-000002 | 80.000 | 80.000 | 0.000 | 20.8000 | 1664.00 | RECEIVED | False | True | 2025-01-07 00:00:00 |
| 1400003 | SYN-PO-000003 | 1 | 4004 | 5009 | 7003 | 116447 | 116454 | 116455 | 116815 | SYN-RCT-LOT-000003 | 100.000 | 100.000 | 0.000 | 77.8500 | 7785.00 | RECEIVED | False | True | 2025-01-10 00:00:00 |
| 1400004 | SYN-PO-000004 | 1 | 4002 | 5004 | 7004 | 116450 | 116457 | 116459 | 116819 | SYN-RCT-LOT-000004 | 120.000 | 120.000 | 0.000 | 37.1000 | 4452.00 | RECEIVED | False | True | 2025-01-13 00:00:00 |
| 1400005 | SYN-PO-000005 | 1 | 4005 | 5011 | 7001 | 116453 | 116460 | 116463 | 116823 | SYN-RCT-LOT-000005 | 140.000 | 140.000 | 0.000 | 94.1500 | 13181.00 | RECEIVED | False | True | 2025-01-16 00:00:00 |