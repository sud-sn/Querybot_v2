# CHATBOT_DB.PHARMA_LAB.D_PRESCRIBER

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_PRESCRIBER`  — always use this exact three-part name in generated SQL.

**Row count:** 8  **Scale:** Small

**Primary Key:** `PRESCRIBER_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `PRESCRIBER_ID` **[PK]** | bigint(19) | No |  |
| `NPI_NUMBER` | char(10) | No |  |
| `DEA_REGISTRATION_NUMBER` | varchar(20) | Yes |  |
| `STATE_LICENSE_NUMBER` | varchar(30) | No |  |
| `PRESCRIBER_FULL_NAME` | nvarchar(160) | No |  |
| `SPECIALTY_NAME` | nvarchar(100) | No |  |
| `PRACTICE_NAME` | nvarchar(160) | Yes |  |
| `PHONE_NUMBER` | varchar(40) | Yes |  |
| `EMAIL_ADDRESS` | varchar(160) | Yes |  |
| `ACTIVE_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `PRESCRIBER_ID` | 0.0% | 300001 | 300008 |

## Sample data

| PRESCRIBER_ID | NPI_NUMBER | DEA_REGISTRATION_NUMBER | STATE_LICENSE_NUMBER | PRESCRIBER_FULL_NAME | SPECIALTY_NAME | PRACTICE_NAME | PHONE_NUMBER | EMAIL_ADDRESS | ACTIVE_FLAG | IS_SYNTHETIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300001 | +1-939-555-0128 | SYNDEA0001 | 968-97-9365 | Logan Smith | Finley Thompson | Mason Smith | +1-960-555-0173 | user5396@example.com | True | True |
| 300002 | +1-555-555-0159 | SYNDEA0002 | 901-85-4016 | Avery Green | Avery Lee | Casey Garcia | +1-898-555-0198 | user2814@mail.test | True | True |
| 300003 | +1-847-555-0187 | SYNDEA0003 | 905-23-3423 | Jamie White | Dana Green | Riley Lewis | +1-304-555-0135 | user5885@test.org | True | True |
| 300004 | +1-291-555-0170 | SYNDEA0004 | 900-94-3806 | Dana Thomas | Reese Anderson | Jesse Scott | +1-720-555-0163 | user6809@example.com | True | True |
| 300005 | +1-977-555-0114 | SYNDEA0005 | 900-53-5745 | Lee Martin | Blake Moore | Jordan Smith | +1-854-555-0129 | user5728@sample.net | True | True |