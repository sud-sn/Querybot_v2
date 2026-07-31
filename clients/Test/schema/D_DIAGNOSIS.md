# CHATBOT_DB.PHARMA_LAB.D_DIAGNOSIS

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_DIAGNOSIS`  — always use this exact three-part name in generated SQL.

**Row count:** 8  **Scale:** Small

**Primary Key:** `DIAGNOSIS_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `DIAGNOSIS_ID` **[PK]** | int(10) | No |  |
| `ICD10_CODE` | varchar(12) | No | 'E03.9', 'E11.9', 'E78.5', 'F32.9', 'G89.2', 'I10', 'I48.9', 'J45.9' |
| `DIAGNOSIS_DESCRIPTION` | nvarchar(240) | No |  |
| `DIAGNOSIS_CATEGORY` | nvarchar(120) | No | 'Behavioral Health', 'Cardiovascular', 'Endocrine', 'Metabolic', 'Pain Management', 'Respiratory' |
| `SENSITIVE_CATEGORY_FLAG` | bit | No |  |
| `ACTIVE_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `DIAGNOSIS_ID` | 0.0% | 8001 | 8008 |

## Sample data

| DIAGNOSIS_ID | ICD10_CODE | DIAGNOSIS_DESCRIPTION | DIAGNOSIS_CATEGORY | SENSITIVE_CATEGORY_FLAG | ACTIVE_FLAG | IS_SYNTHETIC |
| --- | --- | --- | --- | --- | --- | --- |
| 8001 | E11.9 | [REDACTED TEXT] | Endocrine | True | True | True |
| 8002 | I10 | [REDACTED TEXT] | Cardiovascular | True | True | True |
| 8003 | E78.5 | [REDACTED TEXT] | Metabolic | True | True | True |
| 8004 | J45.9 | [REDACTED TEXT] | Respiratory | True | True | True |
| 8005 | F32.9 | [REDACTED TEXT] | Behavioral Health | True | True | True |