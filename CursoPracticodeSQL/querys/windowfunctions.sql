SELECT *, 
		AVG(colegiatura) OVER(PARTITION BY carrera_id)
FROM platzi.alumnos;

SELECT *, 
		AVG(colegiatura) OVER()
FROM platzi.alumnos;

SELECT *, 
		SUM(colegiatura) OVER(PARTITION BY carrera_id ORDER BY colegiatura)
FROM platzi.alumnos;

SELECT *, 
		RANK() OVER(PARTITION BY carrera_id ORDER BY colegiatura DESC) AS brand_rank
FROM platzi.alumnos
ORDER BY carrera_id, brand_rank;

SELECT *
FROM (
		SELECT *, 
		RANK() OVER(PARTITION BY carrera_id ORDER BY colegiatura DESC) AS brand_rank
		FROM platzi.alumnos
) AS ranked_colegiatura_por_carrera
WHERE brand_rank < 3
ORDER BY brand_rank;