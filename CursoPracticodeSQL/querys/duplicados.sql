insert into platzi.alumnos (id, nombre, apellido, email, colegiatura, fecha_incorporacion, carrera_id, tutor_id) 
values (1001, 'Pamelina', null, 'pmylchreestrr@salon.com', 4800, '2020-04-26 10:18:51', 12, 16);

SELECT * FROM platzi.alumnos AS ou
WHERE (
	SELECT COUNT(*)
	FROM platzi.alumnos AS inr
	WHERE ou.id = inr.id
) > 1;

SELECT (platzi.alumnos.*)::text, COUNT(*)
FROM platzi.alumnos 
GROUP BY platzi.alumnos.*
HAVING COUNT(*) > 1;

SELECT (platzi.alumnos.nombre,
		platzi.alumnos.apellido,
		platzi.alumnos.email,
		platzi.alumnos.colegiatura,
		platzi.alumnos.fecha_incorporacion,
		platzi.alumnos.carrera_id,
		platzi.alumnos.tutor_id
)::text, COUNT(*)
FROM platzi.alumnos 
GROUP BY platzi.alumnos.nombre,
		platzi.alumnos.apellido,
		platzi.alumnos.email,
		platzi.alumnos.colegiatura,
		platzi.alumnos.fecha_incorporacion,
		platzi.alumnos.carrera_id,
		platzi.alumnos.tutor_id
HAVING COUNT(*) > 1;

SELECT *
FROM (
	SELECT id,
	ROW_NUMBER() OVER(
		PARTITION BY 
			nombre,
			apellido,
			email,
			colegiatura,
			fecha_incorporacion,
			carrera_id,
			tutor_id
		ORDER BY id ASC
	) AS row,
	*
	FROM platzi.alumnos
) AS duplicados
WHERE duplicados.row > 1;

-- Reto

WITH duplicados AS (
    SELECT id,
           ROW_NUMBER() OVER(
               PARTITION BY 
                   nombre,
                   apellido,
                   email,
                   colegiatura,
                   fecha_incorporacion,
                   carrera_id,
                   tutor_id
               ORDER BY id ASC
           ) AS row
    FROM platzi.alumnos
)
DELETE FROM platzi.alumnos
WHERE id IN (
    SELECT id
    FROM duplicados
    WHERE row > 1
);

DELETE FROM platzi.alumnos
WHERE id IN(
SELECT id
FROM (
	SELECT id,
	ROW_NUMBER() OVER(
		PARTITION BY 
			nombre,
			apellido,
			email,
			colegiatura,
			fecha_incorporacion,
			carrera_id,
			tutor_id
		ORDER BY id ASC
	) AS row
	FROM platzi.alumnos
) AS duplicados
WHERE duplicados.row > 1
);