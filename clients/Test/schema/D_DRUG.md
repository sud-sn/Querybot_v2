# CHATBOT_DB.PHARMA_LAB.D_DRUG

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_DRUG`  — always use this exact three-part name in generated SQL.

**Row count:** 12  **Scale:** Small

**Primary Key:** `DRUG_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `DRUG_ID` **[PK]** | int(10) | No |  |
| `NDC_11_CODE` | char(11) | No |  |
| `GENERIC_NAME` | nvarchar(160) | No |  |
| `BRAND_NAME` | nvarchar(160) | Yes |  |
| `THERAPEUTIC_CLASS` | nvarchar(120) | No | 'ACE Inhibitor', 'Antibiotic', 'Anticoagulant', 'Anticonvulsant', 'Antidiabetic', 'Bronchodilator', 'GLP-1 Agonist', 'Lipid Lowering', 'Long-acting Insulin', 'Opioid Analgesic', 'SSRI', 'Thyroid Hormone' |
| `DOSAGE_FORM` | varchar(40) | No |  |
| `STRENGTH_TEXT` | varchar(40) | No |  |
| `CONTROLLED_SCHEDULE` | varchar(10) | Yes |  |
| `PACKAGE_SIZE` | decimal(12) | No |  |
| `UOM_CODE` | varchar(10) | No | 'EA' |
| `ACTIVE_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `DRUG_ID` | 0.0% | 5001 | 5012 |
| `PACKAGE_SIZE` | 0.0% | 1.000 | 100.000 |

## Sample data

| DRUG_ID | NDC_11_CODE | GENERIC_NAME | BRAND_NAME | THERAPEUTIC_CLASS | DOSAGE_FORM | STRENGTH_TEXT | CONTROLLED_SCHEDULE | PACKAGE_SIZE | UOM_CODE | ACTIVE_FLAG | IS_SYNTHETIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5001 | +1-727-555-0165 | Cortivane | Blake Jackson | Antidiabetic | TABLET | 500 mg | None | 100.000 | EA | True | True |
| 5002 | +1-614-555-0183 | Kelotrine | Finley Allen | Lipid Lowering | TABLET | 20 mg | None | 100.000 | EA | True | True |
| 5003 | +1-506-555-0142 | Juvelast | Dana Harris | ACE Inhibitor | TABLET | 10 mg | None | 100.000 | EA | True | True |
| 5004 | +1-293-555-0194 | Oprazane | Taylor Thomas | Antibiotic | CAPSULE | 500 mg | None | 30.000 | EA | True | True |
| 5005 | +1-596-555-0140 | Hydrovex | Dana White | Thyroid Hormone | TABLET | 50 mcg | None | 100.000 | EA | True | True |