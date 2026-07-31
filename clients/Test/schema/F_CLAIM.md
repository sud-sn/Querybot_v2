# CHATBOT_DB.PHARMA_LAB.F_CLAIM

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.F_CLAIM`  — always use this exact three-part name in generated SQL.

**Row count:** 297  **Scale:** Small

**Primary Key:** `CLAIM_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `CLAIM_ID` **[PK]** | bigint(19) | No |  |
| `CLAIM_CONTROL_NUMBER` | varchar(40) | No |  |
| `RX_FILL_ID` | bigint(19) | No |  |
| `PATIENT_ID` | bigint(19) | No |  |
| `PHARMACY_ID` | int(10) | No |  |
| `PAYER_ID` | int(10) | No |  |
| `SUBMIT_DATE_ID` | int(10) | No |  |
| `ADJUDICATION_DATE_ID` | int(10) | Yes |  |
| `PAID_DATE_ID` | int(10) | Yes |  |
| `DENIAL_DATE_ID` | int(10) | Yes |  |
| `MEMBER_IDENTIFIER` | varchar(40) | No |  |
| `CLAIM_STATUS` | varchar(20) | No | 'APPROVED', 'DENIED', 'PAID', 'REVERSED', 'SUBMITTED' |
| `REJECTION_CODE` | varchar(20) | Yes | 'PA_REQUIRED', 'PLAN_LIMIT' |
| `BILLED_AMT` | decimal(18) | No |  |
| `ALLOWED_AMT` | decimal(18) | No |  |
| `PAYER_PAID_AMT` | decimal(18) | No |  |
| `PATIENT_RESPONSIBILITY_AMT` | decimal(18) | No |  |
| `DISPENSING_FEE_AMT` | decimal(18) | No |  |
| `DELETED_RECORD_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |
| `CREATED_AT_UTC` | datetime2 | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `CLAIM_ID` | 0.0% | 1100001 | 1100297 |
| `RX_FILL_ID` | 0.0% | 1000001 | 1000339 |
| `PATIENT_ID` | 0.0% | 200001 | 200020 |
| `PHARMACY_ID` | 0.0% | 4001 | 4005 |
| `PAYER_ID` | 0.0% | 6001 | 6006 |
| `SUBMIT_DATE_ID` | 0.0% | 116441 | 117159 |
| `ADJUDICATION_DATE_ID` | 5.7% | 116443 | 117161 |
| `PAID_DATE_ID` | 31.3% | 116451 | 117169 |
| `DENIAL_DATE_ID` | 94.9% | 116485 | 117135 |
| `BILLED_AMT` | 0.0% | 59.50 | 283.75 |
| `ALLOWED_AMT` | 0.0% | 0.00 | 266.73 |
| `PAYER_PAID_AMT` | 0.0% | 0.00 | 273.25 |
| `PATIENT_RESPONSIBILITY_AMT` | 0.0% | 8.00 | 267.50 |
| `DISPENSING_FEE_AMT` | 0.0% | 2.50 | 4.75 |
| `CREATED_AT_UTC` | 0.0% | 2025-01-04 00:00:00 | 2026-12-23 00:00:00 |

## Sample data

| CLAIM_ID | CLAIM_CONTROL_NUMBER | RX_FILL_ID | PATIENT_ID | PHARMACY_ID | PAYER_ID | SUBMIT_DATE_ID | ADJUDICATION_DATE_ID | PAID_DATE_ID | DENIAL_DATE_ID | MEMBER_IDENTIFIER | CLAIM_STATUS | REJECTION_CODE | BILLED_AMT | ALLOWED_AMT | PAYER_PAID_AMT | PATIENT_RESPONSIBILITY_AMT | DISPENSING_FEE_AMT | DELETED_RECORD_FLAG | IS_SYNTHETIC | CREATED_AT_UTC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1100001 | SYN-CLM-000001 | 1000001 | 200001 | 4002 | 6005 | 116443 | 116445 | 116453 | None | 922-52-7760 | PAID | None | 145.25 | 136.54 | 0.00 | 145.25 | 3.25 | False | True | 2025-01-06 00:00:00 |
| 1100002 | SYN-CLM-000002 | 1000002 | 200002 | 4004 | 6004 | 116445 | 116447 | 116455 | None | 972-17-3074 | PAID | None | 240.50 | 226.07 | 224.50 | 16.00 | 4.00 | False | True | 2025-01-08 00:00:00 |
| 1100003 | SYN-CLM-000003 | 1000003 | 200003 | 4001 | 6003 | 116447 | 116449 | 116457 | None | 989-41-6092 | PAID | None | 97.00 | 91.18 | 77.00 | 20.00 | 4.75 | False | True | 2025-01-10 00:00:00 |
| 1100004 | SYN-CLM-000004 | 1000004 | 200004 | 4003 | 6002 | 116449 | 116451 | 116459 | None | 966-15-7496 | PAID | None | 204.75 | 192.47 | 180.75 | 24.00 | 2.50 | False | True | 2025-01-12 00:00:00 |
| 1100005 | SYN-CLM-000005 | 1000005 | 200005 | 4005 | 6001 | 116451 | 116453 | None | None | 920-82-9241 | APPROVED | None | 77.50 | 72.85 | 0.00 | 8.00 | 3.25 | False | True | 2025-01-14 00:00:00 |