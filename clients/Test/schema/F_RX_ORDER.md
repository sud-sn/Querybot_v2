# CHATBOT_DB.PHARMA_LAB.F_RX_ORDER

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.F_RX_ORDER`  — always use this exact three-part name in generated SQL.

**Row count:** 360  **Scale:** Small

**Primary Key:** `RX_ORDER_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `RX_ORDER_ID` **[PK]** | bigint(19) | No |  |
| `RX_NUMBER` | varchar(30) | No |  |
| `PATIENT_ID` | bigint(19) | No |  |
| `PRESCRIBER_ID` | bigint(19) | No |  |
| `PHARMACY_ID` | int(10) | No |  |
| `DRUG_ID` | int(10) | No |  |
| `PAYER_ID` | int(10) | No |  |
| `WRITTEN_DATE_ID` | int(10) | No |  |
| `ORDER_DATE_ID` | int(10) | No |  |
| `THERAPY_START_DATE_ID` | int(10) | No |  |
| `THERAPY_END_DATE_ID` | int(10) | Yes |  |
| `ORDERED_QUANTITY` | decimal(14) | No |  |
| `DAYS_SUPPLY` | smallint(5) | No |  |
| `REFILLS_AUTHORIZED` | tinyint(3) | No |  |
| `ROUTE_CODE` | varchar(20) | No | 'INHALATION', 'ORAL', 'SUBCUTANEOUS' |
| `SIG_INSTRUCTIONS` | nvarchar(500) | Yes |  |
| `ORDER_STATUS` | varchar(20) | No | 'ACTIVE', 'CANCELLED', 'COMPLETED', 'ON_HOLD' |
| `PRIOR_AUTH_REQUIRED_FLAG` | bit | No |  |
| `CONTROLLED_SUBSTANCE_FLAG` | bit | No |  |
| `DELETED_RECORD_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |
| `CREATED_AT_UTC` | datetime2 | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `RX_ORDER_ID` | 0.0% | 900001 | 900360 |
| `PATIENT_ID` | 0.0% | 200001 | 200020 |
| `PRESCRIBER_ID` | 0.0% | 300001 | 300008 |
| `PHARMACY_ID` | 0.0% | 4001 | 4005 |
| `DRUG_ID` | 0.0% | 5001 | 5012 |
| `PAYER_ID` | 0.0% | 6001 | 6006 |
| `WRITTEN_DATE_ID` | 0.0% | 116436 | 117154 |
| `ORDER_DATE_ID` | 0.0% | 116438 | 117156 |
| `THERAPY_START_DATE_ID` | 0.0% | 116439 | 117157 |
| `THERAPY_END_DATE_ID` | 0.0% | 116470 | 117230 |
| `ORDERED_QUANTITY` | 0.0% | 1.000 | 90.000 |
| `DAYS_SUPPLY` | 0.0% | 30 | 90 |
| `REFILLS_AUTHORIZED` | 0.0% | 1 | 5 |
| `CREATED_AT_UTC` | 0.0% | 2025-01-01 00:00:00 | 2026-12-20 00:00:00 |

## Sample data

| RX_ORDER_ID | RX_NUMBER | PATIENT_ID | PRESCRIBER_ID | PHARMACY_ID | DRUG_ID | PAYER_ID | WRITTEN_DATE_ID | ORDER_DATE_ID | THERAPY_START_DATE_ID | THERAPY_END_DATE_ID | ORDERED_QUANTITY | DAYS_SUPPLY | REFILLS_AUTHORIZED | ROUTE_CODE | SIG_INSTRUCTIONS | ORDER_STATUS | PRIOR_AUTH_REQUIRED_FLAG | CONTROLLED_SUBSTANCE_FLAG | DELETED_RECORD_FLAG | IS_SYNTHETIC | CREATED_AT_UTC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 900001 | SYN-RX-000001 | 200001 | 300003 | 4002 | 5005 | 6005 | 116438 | 116440 | 116441 | 116470 | 30.000 | 30 | 1 | ORAL | Synthetic instruction set 1; test data o | COMPLETED | False | False | False | True | 2025-01-03 00:00:00 |
| 900002 | SYN-RX-000002 | 200002 | 300006 | 4004 | 5010 | 6004 | 116440 | 116442 | 116443 | 116472 | 30.000 | 30 | 1 | ORAL | Synthetic instruction set 2; test data o | COMPLETED | True | False | False | True | 2025-01-05 00:00:00 |
| 900003 | SYN-RX-000003 | 200003 | 300001 | 4001 | 5003 | 6003 | 116442 | 116444 | 116445 | 116474 | 30.000 | 30 | 2 | ORAL | Synthetic instruction set 3; test data o | COMPLETED | False | False | False | True | 2025-01-07 00:00:00 |
| 900004 | SYN-RX-000004 | 200004 | 300004 | 4003 | 5008 | 6002 | 116444 | 116446 | 116447 | 116506 | 60.000 | 60 | 1 | ORAL | Synthetic instruction set 4; test data o | COMPLETED | False | False | False | True | 2025-01-09 00:00:00 |
| 900005 | SYN-RX-000005 | 200005 | 300007 | 4005 | 5001 | 6001 | 116446 | 116448 | 116449 | 116478 | 30.000 | 30 | 1 | ORAL | Synthetic instruction set 5; test data o | COMPLETED | False | False | False | True | 2025-01-11 00:00:00 |