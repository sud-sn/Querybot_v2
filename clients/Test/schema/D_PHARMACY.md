# CHATBOT_DB.PHARMA_LAB.D_PHARMACY

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_PHARMACY`  — always use this exact three-part name in generated SQL.

**Row count:** 5  **Scale:** Small

**Primary Key:** `PHARMACY_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `PHARMACY_ID` **[PK]** | int(10) | No |  |
| `PHARMACY_CODE` | varchar(20) | No | 'PH-EAST-01', 'PH-FAIR-01', 'PH-LAKE-01', 'PH-NORTH-01', 'PH-WEST-01' |
| `PHARMACY_NAME` | nvarchar(160) | No |  |
| `NCPDP_ID` | varchar(12) | No |  |
| `DEA_REGISTRATION_NUMBER` | varchar(20) | Yes |  |
| `PHARMACY_TYPE` | varchar(30) | No | 'CENTRAL', 'INFUSION', 'MAIL_ORDER', 'RETAIL', 'SPECIALTY' |
| `CITY_NAME` | nvarchar(80) | No |  |
| `STATE_CODE` | char(2) | No |  |
| `REGION_NAME` | nvarchar(80) | No |  |
| `ACTIVE_FLAG` | bit | No |  |
| `IS_SYNTHETIC` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `PHARMACY_ID` | 0.0% | 4001 | 4005 |

## Sample data

| PHARMACY_ID | PHARMACY_CODE | PHARMACY_NAME | NCPDP_ID | DEA_REGISTRATION_NUMBER | PHARMACY_TYPE | CITY_NAME | STATE_CODE | REGION_NAME | ACTIVE_FLAG | IS_SYNTHETIC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4001 | PH-NORTH-01 | Skylar Hall | +1-913-555-0163 | SYNPHDEA01 | SPECIALTY | Jordan Taylor | PA | Alex Jones | True | True |
| 4002 | PH-LAKE-01 | Sage Walker | +1-221-555-0104 | SYNPHDEA02 | RETAIL | Casey Young | CA | Alex Jones | True | True |
| 4003 | PH-WEST-01 | Cameron Walker | +1-979-555-0143 | SYNPHDEA03 | INFUSION | Sydney Allen | FL | Alex Jones | True | True |
| 4004 | PH-EAST-01 | Lee Lee | +1-501-555-0103 | SYNPHDEA04 | CENTRAL | Blake Adams | GA | Lee Garcia | True | True |
| 4005 | PH-FAIR-01 | Taylor Johnson | +1-384-555-0170 | SYNPHDEA05 | MAIL_ORDER | Blake Jackson | GA | Lee Garcia | True | True |