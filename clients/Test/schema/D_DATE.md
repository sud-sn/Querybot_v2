# CHATBOT_DB.PHARMA_LAB.D_DATE

**Type:** BASE TABLE  **Schema:** PHARMA_LAB  **Database:** CHATBOT_DB

**SQL table name:** `CHATBOT_DB.PHARMA_LAB.D_DATE`  — always use this exact three-part name in generated SQL.

**Row count:** 28,489  **Scale:** Small

**Primary Key:** `DATE_ID`

## Columns

| Column | Type | Nullable | Distinct Values |
|--------|------|:--------:|-----------------|
| `DATE_ID` **[PK]** | int(10) | No |  |
| `CALENDAR_DATE` | date | No |  |
| `DATE_KEY_YYYYMMDD` | int(10) | No |  |
| `DAY_OF_WEEK_NUMBER` | tinyint(3) | No |  |
| `DAY_NAME` | varchar(10) | No |  |
| `DAY_OF_MONTH` | tinyint(3) | No |  |
| `DAY_OF_YEAR` | smallint(5) | No |  |
| `WEEK_OF_YEAR` | tinyint(3) | No |  |
| `MONTH_NUMBER` | tinyint(3) | No |  |
| `MONTH_NAME` | varchar(12) | No |  |
| `MONTH_START_DATE` | date | No |  |
| `QUARTER_NUMBER` | tinyint(3) | No |  |
| `QUARTER_NAME` | char(2) | No |  |
| `CALENDAR_YEAR` | smallint(5) | No |  |
| `FISCAL_MONTH_NUMBER` | tinyint(3) | No |  |
| `FISCAL_QUARTER_NUMBER` | tinyint(3) | No |  |
| `FISCAL_YEAR` | smallint(5) | No |  |
| `IS_WEEKEND` | bit | No |  |

## Column Statistics

| Column | Null % | Min | Max |
|--------|-------:|-----|-----|
| `DATE_ID` | 0.0% | 89044 | 117532 |
| `CALENDAR_DATE` | 0.0% | 1950-01-01 | 2027-12-31 |
| `DATE_KEY_YYYYMMDD` | 0.0% | 19500101 | 20271231 |
| `DAY_OF_WEEK_NUMBER` | 0.0% | 1 | 7 |
| `DAY_OF_MONTH` | 0.0% | 1 | 31 |
| `DAY_OF_YEAR` | 0.0% | 1 | 366 |
| `WEEK_OF_YEAR` | 0.0% | 1 | 53 |
| `MONTH_NUMBER` | 0.0% | 1 | 12 |
| `MONTH_START_DATE` | 0.0% | 1950-01-01 | 2027-12-01 |
| `QUARTER_NUMBER` | 0.0% | 1 | 4 |
| `CALENDAR_YEAR` | 0.0% | 1950 | 2027 |
| `FISCAL_MONTH_NUMBER` | 0.0% | 1 | 12 |
| `FISCAL_QUARTER_NUMBER` | 0.0% | 1 | 4 |
| `FISCAL_YEAR` | 0.0% | 1950 | 2028 |

## Sample data

| DATE_ID | CALENDAR_DATE | DATE_KEY_YYYYMMDD | DAY_OF_WEEK_NUMBER | DAY_NAME | DAY_OF_MONTH | DAY_OF_YEAR | WEEK_OF_YEAR | MONTH_NUMBER | MONTH_NAME | MONTH_START_DATE | QUARTER_NUMBER | QUARTER_NAME | CALENDAR_YEAR | FISCAL_MONTH_NUMBER | FISCAL_QUARTER_NUMBER | FISCAL_YEAR | IS_WEEKEND |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 89044 | 1950-01-01 | 19500101 | 7 | Madison Brown | 1 | 1 | 52 | 1 | Finley Johnson | 1950-01-01 | 1 | Rowan Smith | 1950 | 7 | 3 | 1950 | True |
| 89045 | 1950-01-02 | 19500102 | 1 | Sydney Miller | 2 | 2 | 1 | 1 | Finley Johnson | 1950-01-01 | 1 | Rowan Smith | 1950 | 7 | 3 | 1950 | False |
| 89046 | 1950-01-03 | 19500103 | 2 | Peyton White | 3 | 3 | 1 | 1 | Finley Johnson | 1950-01-01 | 1 | Rowan Smith | 1950 | 7 | 3 | 1950 | False |
| 89047 | 1950-01-04 | 19500104 | 3 | Lee Allen | 4 | 4 | 1 | 1 | Finley Johnson | 1950-01-01 | 1 | Rowan Smith | 1950 | 7 | 3 | 1950 | False |
| 89048 | 1950-01-05 | 19500105 | 4 | Mason King | 5 | 5 | 1 | 1 | Finley Johnson | 1950-01-01 | 1 | Rowan Smith | 1950 | 7 | 3 | 1950 | False |