SELECT * FROM (
	SELECT ROW_NUMBER() OVER() AS row_id, *
	FROM platzi.alumnos
) AS alumnos_with_row_num
WHERE row_id IN (1, 5, 12, 15, 20, 600);

SELECT * FROM platzi.alumnos
WHERE id IN (1, 2, 8);

SELECT * FROM platzi.alumnos
WHERE id IN (
	SELECT id
	FROM platzi.alumnos
	WHERE tutor_id = 30
		AND carrera_id = 31
);

--Reto

SELECT a.*
FROM platzi.alumnos a
LEFT JOIN (
    SELECT id
    FROM platzi.alumnos
    WHERE tutor_id = 30
      AND carrera_id = 31
) excluidos
ON a.id = excluidos.id
WHERE excluidos.id IS NULL;

SELECT * FROM platzi.alumnos
WHERE id NOT IN (
	SELECT id
	FROM platzi.alumnos
	WHERE tutor_id = 30
		AND carrera_id = 31
);