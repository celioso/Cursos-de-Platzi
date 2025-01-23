SELECT a.nombre,
		a.apellido,
		t.nombre,
		t.apellido
FROM platzi.alumnos AS a
	INNER JOIN platzi.alumnos AS t on a.tutor_id = t.id;

SELECT CONCAT (a.nombre, ' ', a.apellido) AS alumno,
		CONCAT (t.nombre, ' ', t.apellido) AS tutor
FROM platzi.alumnos AS a
	INNER JOIN platzi.alumnos AS t on a.tutor_id = t.id;


SELECT CONCAT (t.nombre, ' ', t.apellido) AS tutor,
		COUNT(*) AS alumnos_por_tutor
FROM platzi.alumnos AS a
	INNER JOIN platzi.alumnos AS t on a.tutor_id = t.id
GROUP BY tutor
ORDER BY alumnos_por_tutor DESC
LIMIT 10;

--Reto

SELECT tutor, alumnos_por_tutor, AVG(alumnos_por_tutor)
FROM(
SELECT CONCAT(t.nombre, ' ', t.apellido) AS tutor,
		COUNT(*) AS alumnos_por_tutor
FROM platzi.alumnos AS a
	INNER JOIN platzi.alumnos AS t ON a.tutor_id = t.id
GROUP BY tutor
ORDER BY alumnos_por_tutor DESC)
GROUP BY tutor, alumnos_por_tutor;

SELECT AVG(alumnos_por_tutor) AS promedio_alumnos_por_tutor
FROM(
	SELECT CONCAT (t.nombre, ' ', t.apellido) AS tutor,
			COUNT(*) AS alumnos_por_tutor
	FROM platzi.alumnos AS a
		INNER JOIN platzi.alumnos AS t on a.tutor_id = t.id
	GROUP BY tutor
) AS alumnos_tutor;
