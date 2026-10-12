/*
Which large tables in an Azure SQL or SQL Server database would answer QueryBot faster with a columnstore index.

Read-only: it lists suggestions and changes nothing. Run it in the database QueryBot reads (SQL Server
Management Studio or Azure Data Studio), as a user who can see its tables. SQL Server 2017 or later, or Azure SQL.

QueryBot's questions and Learn add up and count many rows at once. A nonclustered columnstore index keeps each
column compressed on its own, so those queries read a fraction of the table. The table, its rows and its other
indexes stay as they are. It costs storage and some time on each insert and update, so agree it with whoever runs
the database, try it on a copy first, and create it outside working hours on a large table. Azure SQL and SQL Server
Enterprise can build it while the table stays in use: add WITH (ONLINE = ON) to the statement.

Columns a nonclustered columnstore index cannot hold (text, image, xml, the (max) types, computed, sparse and
spatial columns) are left out of the suggested statement. Change @min_rows to look at smaller tables.
*/
DECLARE @min_rows bigint = 1000000;

WITH row_counts AS (
    SELECT p.object_id, SUM(p.rows) AS row_count
    FROM sys.partitions AS p
    WHERE p.index_id IN (0, 1)
    GROUP BY p.object_id
),
sizes AS (
    SELECT p.object_id, SUM(a.total_pages) AS total_pages
    FROM sys.partitions AS p
    JOIN sys.allocation_units AS a ON a.container_id = p.partition_id
    GROUP BY p.object_id
),
indexable AS (
    SELECT c.object_id,
           COUNT(*) AS column_count,
           STRING_AGG(CAST(QUOTENAME(c.name) AS nvarchar(max)), N', ') WITHIN GROUP (ORDER BY c.column_id) AS columns
    FROM sys.columns AS c
    JOIN sys.types AS ty ON ty.user_type_id = c.user_type_id
    WHERE c.is_computed = 0
      AND c.is_sparse = 0
      AND ty.name NOT IN (N'text', N'ntext', N'image', N'xml', N'sql_variant', N'geography', N'geometry',
                          N'hierarchyid', N'timestamp')
      AND NOT (ty.name IN (N'varchar', N'nvarchar', N'varbinary') AND c.max_length = -1)
    GROUP BY c.object_id
)
SELECT
    s.name AS schema_name,
    t.name AS table_name,
    r.row_count,
    CAST(z.total_pages * 8 / 1024.0 AS decimal(18, 1)) AS size_mb,
    N'CREATE NONCLUSTERED COLUMNSTORE INDEX ' + QUOTENAME(LEFT(N'ncci_' + t.name, 128)) + N' ON '
        + QUOTENAME(s.name) + N'.' + QUOTENAME(t.name) + N' (' + i.columns + N');' AS suggested_statement
FROM sys.tables AS t
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
JOIN row_counts AS r ON r.object_id = t.object_id
JOIN sizes AS z ON z.object_id = t.object_id
JOIN indexable AS i ON i.object_id = t.object_id
WHERE t.is_ms_shipped = 0
  AND t.is_memory_optimized = 0
  AND r.row_count >= @min_rows
  AND i.column_count <= 1024
  AND NOT EXISTS (SELECT 1 FROM sys.indexes AS x WHERE x.object_id = t.object_id AND x.type IN (5, 6))
ORDER BY r.row_count DESC;
