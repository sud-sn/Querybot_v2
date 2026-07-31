# CHATBOT_DB.PHARMA_LAB.F_RX_FILL

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.F_RX_FILL`  — always use this exact three-part name in generated SQL.

**Row count:** 339  **Scale:** Small

**Primary Key:** `RX_FILL_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `RX_FILL_ID` **[PK]** | bigint(19) | No |  |
| `RX_ORDER_ID` | bigint(19) | No |  |
| `PATIENT_ID` | bigint(19) | No |  |
| `PRESCRIBER_ID` | bigint(19) | No |  |
| `PHARMACY_ID` | int(10) | No |  |
| `DRUG_ID` | int(10) | No |  |
| `PAYER_ID` | int(10) | No |  |
| `ORDER_DATE_ID` | int(10) | No |  |
| `BOOKED_DATE_ID` | int(10) | No |  |
| `SCHEDULED_FILL_DATE_ID` | int(10) | No |  |
| `DISPENSE_DATE_ID` | int(10) | Yes |  |
| `PICKUP_DATE_ID` | int(10) | Yes |  |
| `REVERSAL_DATE_ID` | int(10) | Yes |  |
| `FILL_SEQUENCE_NUMBER` | tinyint(3) | No |  |
| `DISPENSED_QUANTITY` | decimal(14) | No |  |
| `DAYS_SUPPLY` | smallint(5) | No |  |
| `GROSS_REVENUE_AMT` | decimal(18) | No |  |
| `DISCOUNT_AMT` | decimal(18) | No |  |
| `NET_REVENUE_AMT` | decimal(18) | No |  |
| `ACQUISITION_COST_AMT` | decimal(18) | No |  |
| `GROSS_PROFIT_AMT` | decimal(18) | No |  |
| `PATIENT_COPAY_AMT` | decimal(18) | No |  |
| `THIRD_PARTY_PAID_AMT` | decimal(18) | No |  |
| `TAX_AMT` | decimal(18) | No |  |
| `FILL_STATUS` | varchar(20) | No | 'BOOKED', 'CANCELLED', 'DISPENSED', 'PICKED_UP', 'REVERSED' |
| `REVERSAL_FLAG` | bit | No |  |
| `DELETED_RECORD_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |
| `CREATED_AT_UTC` | datetime2 | No |  |
| `FILL_DATE` | date | Yes |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `RX_FILL_ID` | 0.0% | 1000001 | 1000339 |
| `RX_ORDER_ID` | 0.0% | 900001 | 900360 |
| `PATIENT_ID` | 0.0% | 200001 | 200020 |
| `PRESCRIBER_ID` | 0.0% | 300001 | 300008 |
| `PHARMACY_ID` | 0.0% | 4001 | 4005 |
| `DRUG_ID` | 0.0% | 5001 | 5012 |
| `PAYER_ID` | 0.0% | 6001 | 6006 |
| `ORDER_DATE_ID` | 0.0% | 116438 | 117156 |
| `BOOKED_DATE_ID` | 0.0% | 116439 | 117157 |
| `SCHEDULED_FILL_DATE_ID` | 0.0% | 116440 | 117158 |
| `DISPENSE_DATE_ID` | 12.4% | 116441 | 117159 |
| `PICKUP_DATE_ID` | 32.4% | 116442 | 117160 |
| `REVERSAL_DATE_ID` | 96.8% | 116503 | 117119 |
| `FILL_SEQUENCE_NUMBER` | 0.0% | 1 | 1 |
| `DISPENSED_QUANTITY` | 0.0% | 1.000 | 90.000 |
| `DAYS_SUPPLY` | 0.0% | 30 | 90 |
| `GROSS_REVENUE_AMT` | 0.0% | 67.00 | 286.25 |
| `DISCOUNT_AMT` | 0.0% | 0.00 | 7.50 |
| `NET_REVENUE_AMT` | 0.0% | 59.50 | 283.75 |
| `ACQUISITION_COST_AMT` | 0.0% | 27.10 | 127.20 |
| `GROSS_PROFIT_AMT` | 0.0% | 32.40 | 156.55 |
| `PATIENT_COPAY_AMT` | 0.0% | 8.00 | 267.50 |
| `THIRD_PARTY_PAID_AMT` | 0.0% | 0.00 | 273.25 |
| `TAX_AMT` | 0.0% | 1.19 | 5.68 |
| `CREATED_AT_UTC` | 0.0% | 2025-01-02 00:00:00 | 2026-12-21 00:00:00 |
| `FILL_DATE` | 12.4% | 2025-01-04 | 2026-12-23 |

## Sample data

| RX_FILL_ID | RX_ORDER_ID | PATIENT_ID | PRESCRIBER_ID | PHARMACY_ID | DRUG_ID | PAYER_ID | ORDER_DATE_ID | BOOKED_DATE_ID | SCHEDULED_FILL_DATE_ID | DISPENSE_DATE_ID | PICKUP_DATE_ID | REVERSAL_DATE_ID | FILL_SEQUENCE_NUMBER | DISPENSED_QUANTITY | DAYS_SUPPLY | GROSS_REVENUE_AMT | DISCOUNT_AMT | NET_REVENUE_AMT | ACQUISITION_COST_AMT | GROSS_PROFIT_AMT | PATIENT_COPAY_AMT | THIRD_PARTY_PAID_AMT | TAX_AMT | FILL_STATUS | REVERSAL_FLAG | DELETED_RECORD_FLAG | IS_SYNTHETIC | CREATED_AT_UTC | FILL_DATE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000001 | 900001 | 200001 | 300003 | 4002 | 5005 | 6005 | 116440 | 116441 | 116442 | 116443 | 116444 | None | 1 | 30.000 | 30 | 145.25 | 0.00 | 145.25 | 63.50 | 81.75 | 145.25 | 0.00 | 2.91 | PICKED_UP | False | False | True | 2025-01-04 00:00:00 | 2025-01-06 |
| 1000002 | 900002 | 200002 | 300006 | 4004 | 5010 | 6004 | 116442 | 116443 | 116444 | 116445 | 116446 | None | 1 | 30.000 | 30 | 245.50 | 5.00 | 240.50 | 109.00 | 131.50 | 16.00 | 224.50 | 4.81 | PICKED_UP | False | False | True | 2025-01-06 00:00:00 | 2025-01-08 |
| 1000003 | 900003 | 200003 | 300001 | 4001 | 5003 | 6003 | 116444 | 116445 | 116446 | 116447 | 116448 | None | 1 | 30.000 | 30 | 104.50 | 7.50 | 97.00 | 45.30 | 51.70 | 20.00 | 77.00 | 1.94 | PICKED_UP | False | False | True | 2025-01-08 00:00:00 | 2025-01-10 |
| 1000004 | 900004 | 200004 | 300004 | 4003 | 5008 | 6002 | 116446 | 116447 | 116448 | 116449 | 116450 | None | 1 | 60.000 | 60 | 204.75 | 0.00 | 204.75 | 90.80 | 113.95 | 24.00 | 180.75 | 4.10 | PICKED_UP | False | False | True | 2025-01-10 00:00:00 | 2025-01-12 |
| 1000005 | 900005 | 200005 | 300007 | 4005 | 5001 | 6001 | 116448 | 116449 | 116450 | 116451 | None | None | 1 | 30.000 | 30 | 80.00 | 2.50 | 77.50 | 27.10 | 50.40 | 8.00 | 69.50 | 1.55 | DISPENSED | False | False | True | 2025-01-12 00:00:00 | 2025-01-14 |