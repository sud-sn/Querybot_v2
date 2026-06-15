ATTACH DATABASE ':memory:' AS FINANCE;

CREATE TABLE FINANCE.LEDGER (
    ENTRY_ID INTEGER PRIMARY KEY,
    PERIOD VARCHAR NOT NULL,
    ACCOUNT_TYPE VARCHAR NOT NULL,
    CATEGORY VARCHAR NOT NULL,
    AMOUNT DECIMAL(14, 2) NOT NULL
);

CREATE TABLE FINANCE.BUDGET (
    PERIOD VARCHAR NOT NULL,
    CATEGORY VARCHAR NOT NULL,
    BUDGET_AMOUNT DECIMAL(14, 2) NOT NULL
);

CREATE TABLE FINANCE.INVOICES (
    INVOICE_ID INTEGER PRIMARY KEY,
    CUSTOMER_NAME VARCHAR NOT NULL,
    ISSUE_DATE DATE NOT NULL,
    DUE_DATE DATE NOT NULL,
    PAID_DATE DATE,
    AMOUNT DECIMAL(14, 2) NOT NULL,
    INVOICE_STATUS VARCHAR NOT NULL
);

INSERT INTO FINANCE.LEDGER VALUES
    (1, '2026-01', 'Revenue', 'Product Sales', 100000),
    (2, '2026-01', 'Revenue', 'Services', 40000),
    (3, '2026-01', 'Expense', 'COGS', 60000),
    (4, '2026-01', 'Expense', 'Payroll', 30000),
    (5, '2026-01', 'Expense', 'Rent', 10000),
    (6, '2026-01', 'Expense', 'Marketing', 8000),
    (7, '2026-02', 'Revenue', 'Product Sales', 110000),
    (8, '2026-02', 'Revenue', 'Services', 45000),
    (9, '2026-02', 'Expense', 'COGS', 65000),
    (10, '2026-02', 'Expense', 'Payroll', 32000),
    (11, '2026-02', 'Expense', 'Rent', 10000),
    (12, '2026-02', 'Expense', 'Marketing', 12000);

INSERT INTO FINANCE.BUDGET VALUES
    ('2026-01', 'COGS', 58000),
    ('2026-01', 'Payroll', 31000),
    ('2026-01', 'Rent', 10000),
    ('2026-01', 'Marketing', 10000),
    ('2026-02', 'COGS', 62000),
    ('2026-02', 'Payroll', 32000),
    ('2026-02', 'Rent', 10000),
    ('2026-02', 'Marketing', 10000);

INSERT INTO FINANCE.INVOICES VALUES
    (1, 'Acme Ltd', '2026-01-05', '2026-02-04', '2026-01-30', 30000, 'Paid'),
    (2, 'Beta Corp', '2026-01-10', '2026-02-09', NULL, 20000, 'Overdue'),
    (3, 'Gamma PLC', '2026-02-01', '2026-03-03', '2026-02-28', 25000, 'Paid'),
    (4, 'Acme Ltd', '2026-02-15', '2026-03-17', NULL, 15000, 'Open'),
    (5, 'Delta Inc', '2026-01-20', '2026-02-19', '2026-03-05', 10000, 'Paid'),
    (6, 'Beta Corp', '2026-02-20', '2026-03-22', NULL, 18000, 'Overdue');
