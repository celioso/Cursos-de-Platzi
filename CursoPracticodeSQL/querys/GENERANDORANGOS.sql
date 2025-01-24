SELECT * FROM generate_series(1, 4, 2);

SELECT * FROM generate_series(5, 1, -2);

SELECT * FROM generate_series(1.1, 8, 1.3);

SELECT current_date + s.a AS dates
FROM generate_series(0, 14, 7) AS s(a);

SELECT *
FROM generate_series('2020-01-19 00:00:00'::timestamp,
					 '2020-01-26 12:00:00', '10 hours');

SELECT a.id,
	   a.nombre,
	   a.apellido,
	   a.carrera_id,
	   s.a
FROM platzi.alumnos AS a
	INNER JOIN generate_series(0, 10) AS s(a)
	ON s.a = a.carrera_id
ORDER BY a.carrera_id;

-- Reto

SELECT LPAD('*', id, '-')
FROM platzi.alumnos AS a
WHERE id IN (SELECT * FROM generate_series(0, 10, 1));

SELECT
	a.id,
	rpad(a.nombre,generate_series(0,10,2),'*'),
	a.nombre,
	a.apellido,
	a.carrera_id
FROM platzi.alumnos AS a
WHERE id <10
ORDER BY a.carrera_id;

SELECT LPAD('*', CAST(ordinality AS int), '*')
FROM generate_series(10, 2, -2) WITH ordinality;

SELECT *
FROM generate_series(100, 2, -2) WITH ordinality;

SELECT LPAD('*', CAST(ordinality AS int), '*')
FROM generate_series(100, 2, -2) WITH ordinality;