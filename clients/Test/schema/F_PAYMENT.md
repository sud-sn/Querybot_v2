# CHATBOT_DB.PHARMA_LAB.F_PAYMENT

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.F_PAYMENT`  — always use this exact three-part name in generated SQL.

**Row count:** 297  **Scale:** Small

**Primary Key:** `PAYMENT_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `PAYMENT_ID` **[PK]** | bigint(19) | No |  |
| `CLAIM_ID` | bigint(19) | Yes |  |
| `RX_FILL_ID` | bigint(19) | No |  |
| `PATIENT_ID` | bigint(19) | No |  |
| `PAYER_ID` | int(10) | No |  |
| `INVOICE_DATE_ID` | int(10) | No |  |
| `DUE_DATE_ID` | int(10) | No |  |
| `PAYMENT_DATE_ID` | int(10) | Yes |  |
| `POST_DATE_ID` | int(10) | Yes |  |
| `PAYMENT_REFERENCE` | varchar(40) | No |  |
| `PAYMENT_STATUS` | varchar(20) | No | 'OPEN', 'OVERDUE', 'PAID', 'PARTIAL', 'WRITTEN_OFF' |
| `PAYMENT_METHOD` | varchar(30) | No | 'ELECTRONIC_REMITTANCE', 'PATIENT_CARD_TOKEN' |
| `INVOICE_AMT` | decimal(18) | No |  |
| `PAID_AMT` | decimal(18) | No |  |
| `OUTSTANDING_AMT` | decimal(18) | No |  |
| `WRITE_OFF_AMT` | decimal(18) | No |  |
| `IS_SYNTHETIC` | bit | No |  |
| `CREATED_AT_UTC` | datetime2 | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `PAYMENT_ID` | 0.0% | 1200001 | 1200297 |
| `CLAIM_ID` | 0.0% | 1100001 | 1100297 |
| `RX_FILL_ID` | 0.0% | 1000001 | 1000339 |
| `PATIENT_ID` | 0.0% | 200001 | 200020 |
| `PAYER_ID` | 0.0% | 6001 | 6006 |
| `INVOICE_DATE_ID` | 0.0% | 116441 | 117159 |
| `DUE_DATE_ID` | 0.0% | 116471 | 117189 |
| `PAYMENT_DATE_ID` | 25.6% | 116451 | 117169 |
| `POST_DATE_ID` | 17.8% | 116452 | 117170 |
| `INVOICE_AMT` | 0.0% | 8.00 | 518.95 |
| `PAID_AMT` | 0.0% | 0.00 | 518.95 |
| `OUTSTANDING_AMT` | 0.0% | 0.00 | 506.34 |
| `WRITE_OFF_AMT` | 0.0% | 0.00 | 264.25 |
| `CREATED_AT_UTC` | 0.0% | 2025-01-04 00:00:00 | 2026-12-23 00:00:00 |

## Sample data

| PAYMENT_ID | CLAIM_ID | RX_FILL_ID | PATIENT_ID | PAYER_ID | INVOICE_DATE_ID | DUE_DATE_ID | PAYMENT_DATE_ID | POST_DATE_ID | PAYMENT_REFERENCE | PAYMENT_STATUS | PAYMENT_METHOD | INVOICE_AMT | PAID_AMT | OUTSTANDING_AMT | WRITE_OFF_AMT | IS_SYNTHETIC | CREATED_AT_UTC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1200001 | 1100001 | 1000001 | 200001 | 6005 | 116443 | 116473 | 116453 | 116454 | SYN-PMT-000001 | PAID | PATIENT_CARD_TOKEN | 281.79 | 281.79 | 0.00 | 0.00 | True | 2025-01-06 00:00:00 |
| 1200002 | 1100002 | 1000002 | 200002 | 6004 | 116445 | 116475 | 116455 | 116456 | SYN-PMT-000002 | PAID | ELECTRONIC_REMITTANCE | 242.07 | 242.07 | 0.00 | 0.00 | True | 2025-01-08 00:00:00 |
| 1200003 | 1100003 | 1000003 | 200003 | 6003 | 116447 | 116477 | 116457 | 116458 | SYN-PMT-000003 | PAID | ELECTRONIC_REMITTANCE | 111.18 | 111.18 | 0.00 | 0.00 | True | 2025-01-10 00:00:00 |
| 1200004 | 1100004 | 1000004 | 200004 | 6002 | 116449 | 116479 | 116459 | 116460 | SYN-PMT-000004 | PAID | ELECTRONIC_REMITTANCE | 216.47 | 216.47 | 0.00 | 0.00 | True | 2025-01-12 00:00:00 |
| 1200005 | 1100005 | 1000005 | 200005 | 6001 | 116451 | 116481 | None | None | SYN-PMT-000005 | OVERDUE | ELECTRONIC_REMITTANCE | 80.85 | 0.00 | 80.85 | 0.00 | True | 2025-01-14 00:00:00 |