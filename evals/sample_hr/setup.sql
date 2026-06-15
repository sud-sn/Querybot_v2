ATTACH DATABASE ':memory:' AS HR;

CREATE TABLE HR.DEPARTMENTS (
    DEPARTMENT_ID INTEGER PRIMARY KEY,
    DEPARTMENT_NAME VARCHAR NOT NULL
);

CREATE TABLE HR.EMPLOYEES (
    EMPLOYEE_ID INTEGER PRIMARY KEY,
    EMPLOYEE_NAME VARCHAR NOT NULL,
    DEPARTMENT_ID INTEGER,
    HIRE_DATE DATE NOT NULL,
    STATUS VARCHAR NOT NULL,
    SALARY DECIMAL(12, 2),
    CITY VARCHAR NOT NULL
);

CREATE TABLE HR.ATTENDANCE (
    EMPLOYEE_ID INTEGER NOT NULL,
    WORK_DATE DATE NOT NULL,
    ATTENDANCE_STATUS VARCHAR NOT NULL,
    HOURS_WORKED DECIMAL(5, 2)
);

INSERT INTO HR.DEPARTMENTS VALUES
    (10, 'Sales'),
    (20, 'Engineering'),
    (30, 'Finance'),
    (40, 'HR'),
    (50, 'Marketing');

INSERT INTO HR.EMPLOYEES VALUES
    (1, 'Alice', 10, '2021-01-15', 'Active', 72000, 'London'),
    (2, 'Bob', 10, '2022-03-01', 'Active', 68000, 'Leeds'),
    (3, 'Carla', 20, '2020-07-20', 'Active', 95000, 'London'),
    (4, 'Deepak', 20, '2023-02-10', 'Active', 88000, 'Manchester'),
    (5, 'Elena', 30, '2019-11-05', 'Active', 91000, 'London'),
    (6, 'Farah', 40, '2024-01-08', 'Active', 65000, 'Birmingham'),
    (7, 'George', 10, '2018-06-12', 'Inactive', 70000, 'Bristol'),
    (8, 'Hana', 20, '2025-01-12', 'Active', 82000, 'Leeds'),
    (9, 'Ivan', 30, '2022-09-30', 'Active', NULL, 'Manchester'),
    (10, 'Julia', 10, '2025-04-01', 'Active', 64000, 'London'),
    (11, 'Karim', NULL, '2026-05-18', 'Active', 60000, 'Glasgow'),
    (12, 'Lily', 20, '2021-08-23', 'Active', 102000, 'London');

INSERT INTO HR.ATTENDANCE VALUES
    (1, '2026-06-01', 'Present', 8),
    (1, '2026-06-02', 'Present', 8),
    (2, '2026-06-01', 'Absent', 0),
    (2, '2026-06-02', 'Present', 8),
    (3, '2026-06-01', 'Present', 9),
    (3, '2026-06-02', 'Present', 9),
    (4, '2026-06-01', 'Present', 8),
    (4, '2026-06-02', 'Absent', 0),
    (5, '2026-06-01', 'Present', 8),
    (5, '2026-06-02', 'Present', 8),
    (6, '2026-06-01', 'Leave', 0),
    (6, '2026-06-02', 'Present', 8),
    (8, '2026-06-01', 'Present', 7),
    (8, '2026-06-02', 'Present', 8),
    (9, '2026-06-01', 'Absent', 0),
    (9, '2026-06-02', 'Absent', 0),
    (10, '2026-06-01', 'Present', 8),
    (10, '2026-06-02', 'Present', 8),
    (11, '2026-06-01', 'Present', 8),
    (11, '2026-06-02', 'Present', 8),
    (12, '2026-06-01', 'Present', 10),
    (12, '2026-06-02', 'Present', 10);
