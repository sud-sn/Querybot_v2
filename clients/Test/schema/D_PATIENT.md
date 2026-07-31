# CHATBOT_DB.PHARMA_LAB.D_PATIENT

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_PATIENT`  — always use this exact three-part name in generated SQL.

**Row count:** 20  **Scale:** Small

**Primary Key:** `PATIENT_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `PATIENT_ID` **[PK]** | bigint(19) | No |  |
| `PATIENT_MRN` | varchar(30) | No |  |
| `FIRST_NAME` | nvarchar(80) | No |  |
| `LAST_NAME` | nvarchar(80) | No |  |
| `DATE_OF_BIRTH_DATE_ID` | int(10) | No |  |
| `SEX_AT_BIRTH_CODE` | varchar(20) | Yes | 'F', 'M' |
| `GENDER_IDENTITY` | varchar(40) | Yes | 'Man', 'Nonbinary', 'Woman' |
| `EMAIL_ADDRESS` | varchar(160) | Yes |  |
| `PHONE_NUMBER` | varchar(40) | Yes |  |
| `ADDRESS_LINE_1` | nvarchar(160) | Yes |  |
| `CITY_NAME` | nvarchar(80) | Yes |  |
| `STATE_CODE` | char(2) | Yes |  |
| `POSTAL_CODE` | varchar(12) | Yes |  |
| `CONSENT_STATUS` | varchar(20) | No | 'GRANTED', 'RESTRICTED', 'REVOKED' |
| `DECEASED_FLAG` | bit | No |  |
| `ACTIVE_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |
| `SOURCE_SYSTEM_CODE` | varchar(30) | No | 'SYNTHETIC_EHR' |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `PATIENT_ID` | 0.0% | 200001 | 200020 |

## Sample data

| PATIENT_ID | PATIENT_MRN | FIRST_NAME | LAST_NAME | DATE_OF_BIRTH_DATE_ID | SEX_AT_BIRTH_CODE | GENDER_IDENTITY | EMAIL_ADDRESS | PHONE_NUMBER | ADDRESS_LINE_1 | CITY_NAME | STATE_CODE | POSTAL_CODE | CONSENT_STATUS | DECEASED_FLAG | ACTIVE_FLAG | IS_SYNTHETIC | SOURCE_SYSTEM_CODE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200001 | SYN-MRN-0001 | Jamie | Thomas | 101533 | F | Woman | user6074@test.org | +1-429-555-0108 | 6729 Willow St | Jordan Taylor | PA | 40364 | GRANTED | False | True | True | SYNTHETIC_EHR |
| 200002 | SYN-MRN-0002 | Sydney | Thomas | 99598 | M | Man | user0004@example.com | +1-530-555-0105 | 2508 Pine St | Jordan Taylor | PA | 28198 | GRANTED | False | True | True | SYNTHETIC_EHR |
| 200003 | SYN-MRN-0003 | Cameron | Thomas | 104207 | F | Woman | user1608@mail.test | +1-342-555-0116 | 317 Elm St | Taylor Harris | PA | 19034 | RESTRICTED | False | True | True | SYNTHETIC_EHR |
| 200004 | SYN-MRN-0004 | Mason | Thomas | 102970 | M | Man | user1467@example.com | +1-730-555-0107 | 6449 Pine St | Taylor Harris | PA | 78351 | GRANTED | False | True | True | SYNTHETIC_EHR |
| 200005 | SYN-MRN-0005 | River | Thomas | 96229 | F | Woman | user7733@example.com | +1-397-555-0175 | 4739 Spruce St | Casey Young | CA | 72168 | GRANTED | False | True | True | SYNTHETIC_EHR |