SELECT
    schema AS schemaname,
    "table" AS tablename,
    table_id AS tableid,
    size AS size_in_mb,
    CASE
        WHEN diststyle NOT IN ('EVEN', 'ALL') THEN 1
        ELSE 0
    END AS has_dist_key,
    CASE
        WHEN sortkey1 IS NOT NULL THEN 1
        ELSE 0
    END AS has_sort_key,
    CASE
        WHEN encoded = 'Y' THEN 1
        ELSE 0
    END AS has_col_encoding,
    CAST(MAX(iq.max_blocks_per_slice - iq.min_blocks_per_slice) AS FLOAT) / 
    GREATEST(COALESCE(iq.min_blocks_per_slice, 0)::INT, 1) AS ratio_skew_across_slices,
    CAST(100 * iq.dist_slice AS FLOAT) / 
    (SELECT COUNT(DISTINCT slice) FROM stv_slices) AS pct_slices_populated
FROM
    svv_table_info ti
JOIN (
    SELECT
        tbl,
        MIN(c) AS min_blocks_per_slice,
        MAX(c) AS max_blocks_per_slice,
        COUNT(DISTINCT slice) AS dist_slice
    FROM (
        SELECT
            b.tbl,
            b.slice,
            COUNT(*) AS c
        FROM
            stv_blocklist b
        GROUP BY
            b.tbl, b.slice
    ) sub
    WHERE tbl IN (
        SELECT
            table_id
        FROM
            svv_table_info
    )
    GROUP BY
        tbl
) iq
ON iq.tbl = ti.table_id
GROUP BY
    schema,
    "table",
    table_id,
    size,
    diststyle,
    sortkey1,
    encoded,
    iq.min_blocks_per_slice,
    iq.max_blocks_per_slice,
    iq.dist_slice;

