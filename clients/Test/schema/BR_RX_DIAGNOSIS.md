# CHATBOT_DB.PHARMA_LAB.BR_RX_DIAGNOSIS

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.BR_RX_DIAGNOSIS`  — always use this exact three-part name in generated SQL.

**Row count:** 450  **Scale:** Small

**Primary Key:** `RX_ORDER_ID`, `DIAGNOSIS_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `RX_ORDER_ID` **[PK]** | bigint(19) | No |  |
| `DIAGNOSIS_ID` **[PK]** | int(10) | No |  |
| `DIAGNOSIS_SEQUENCE` | tinyint(3) | No |  |
| `PRIMARY_DIAGNOSIS_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `RX_ORDER_ID` | 0.0% | 900001 | 900360 |
| `DIAGNOSIS_ID` | 0.0% | 8001 | 8008 |
| `DIAGNOSIS_SEQUENCE` | 0.0% | 1 | 2 |

## Sample data

| RX_ORDER_ID | DIAGNOSIS_ID | DIAGNOSIS_SEQUENCE | PRIMARY_DIAGNOSIS_FLAG | IS_SYNTHETIC |
| --- | --- | --- | --- | --- |
| 900001 | 8001 | 1 | True | True |
| 900002 | 8002 | 1 | True | True |
| 900003 | 8003 | 1 | True | True |
| 900004 | 8004 | 1 | True | True |
| 900004 | 8007 | 2 | False | True |