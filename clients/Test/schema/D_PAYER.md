# CHATBOT_DB.PHARMA_LAB.D_PAYER

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_PAYER`  — always use this exact three-part name in generated SQL.

**Row count:** 6  **Scale:** Small

**Primary Key:** `PAYER_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `PAYER_ID` **[PK]** | int(10) | No |  |
| `PAYER_CODE` | varchar(20) | No | 'SYN-ASST', 'SYN-CASH', 'SYN-COMM-A', 'SYN-COMM-B', 'SYN-MCAID', 'SYN-MCARE' |
| `PAYER_NAME` | nvarchar(160) | No |  |
| `PAYER_TYPE` | varchar(30) | No | 'ASSISTANCE', 'CASH', 'COMMERCIAL', 'MEDICAID', 'MEDICARE' |
| `PLAN_NAME` | nvarchar(160) | No |  |
| `BIN_NUMBER` | char(6) | Yes |  |
| `PCN_CODE` | varchar(12) | Yes | 'SYNPCN1', 'SYNPCN2', 'SYNPCN3', 'SYNPCN4', 'SYNPCN6' |
| `ACTIVE_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `PAYER_ID` | 0.0% | 6001 | 6006 |

## Sample data

| PAYER_ID | PAYER_CODE | PAYER_NAME | PAYER_TYPE | PLAN_NAME | BIN_NUMBER | PCN_CODE | ACTIVE_FLAG | IS_SYNTHETIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6001 | SYN-COMM-A | Ellis Jackson | COMMERCIAL | Hayden Green | 990001 | SYNPCN1 | True | True |
| 6002 | SYN-COMM-B | Cameron White | COMMERCIAL | Alex Jones | 990002 | SYNPCN2 | True | True |
| 6003 | SYN-MCARE | Harper Smith | MEDICARE | Sage Thompson | 990003 | SYNPCN3 | True | True |
| 6004 | SYN-MCAID | Riley Allen | MEDICAID | Sydney Adams | 990004 | SYNPCN4 | True | True |
| 6005 | SYN-CASH | Jordan Anderson | CASH | Finley Wilson | None | None | True | True |