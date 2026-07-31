# CHATBOT_DB.PHARMA_LAB.D_SUPPLIER

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_SUPPLIER`  — always use this exact three-part name in generated SQL.

**Row count:** 4  **Scale:** Small

**Primary Key:** `SUPPLIER_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `SUPPLIER_ID` **[PK]** | int(10) | No |  |
| `SUPPLIER_CODE` | varchar(20) | No | 'SUP-DEMO-01', 'SUP-DEMO-02', 'SUP-DEMO-03', 'SUP-DEMO-04' |
| `SUPPLIER_NAME` | nvarchar(160) | No |  |
| `DEA_REGISTRATION_NUMBER` | varchar(20) | Yes |  |
| `CONTACT_NAME` | nvarchar(120) | Yes |  |
| `CONTACT_EMAIL` | varchar(160) | Yes |  |
| `ACTIVE_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `SUPPLIER_ID` | 0.0% | 7001 | 7004 |

## Sample data

| SUPPLIER_ID | SUPPLIER_CODE | SUPPLIER_NAME | DEA_REGISTRATION_NUMBER | CONTACT_NAME | CONTACT_EMAIL | ACTIVE_FLAG | IS_SYNTHETIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7001 | SUP-DEMO-01 | Hayden Harris | SYNSUPDEA01 | Sage Hall | user3554@mail.test | True | True |
| 7002 | SUP-DEMO-02 | Reese Harris | SYNSUPDEA02 | Alex Hall | user2683@mail.test | True | True |
| 7003 | SUP-DEMO-03 | Cameron Wilson | SYNSUPDEA03 | Kendall Allen | user7760@test.org | True | True |
| 7004 | SUP-DEMO-04 | Reese Lewis | SYNSUPDEA04 | Alex Miller | user5931@example.com | True | True |