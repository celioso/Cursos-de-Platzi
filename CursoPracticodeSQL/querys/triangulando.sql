SELECT lpad('sql', 15, '*');

SELECT LPAD('sql', id, '*')
FROM platzi.alumnos
WHERE id < 10;

SELECT LPAD('*', id, '*')
FROM platzi.alumnos
WHERE id < 10;

SELECT LPAD('*', id, '*'), carrera_id
FROM platzi.alumnos
WHERE id < 10
ORDER BY carrera_id;

SELECT * 
FROM (
	SELECT ROW_NUMBER() OVER () AS row_id, *
	FROM platzi.alumnos
) AS alumnos_with_row_id
WHERE row_id <= 5;

SELECT lpad('*', CAST(row_id AS int), '*') 
FROM (
	SELECT ROW_NUMBER() OVER (ORDER BY carrera_id) AS row_id, *
	FROM platzi.alumnos
) AS alumnos_with_row_id
WHERE row_id <= 5
ORDER BY carrera_id;

--Reto

SELECT RPAD('sql', id, '*')
FROM platzi.alumnos
WHERE id < 10;

SELECT LPAD('123', 6, '0') AS resultado;

SELECT RPAD('123', 6, '0') AS resultado;