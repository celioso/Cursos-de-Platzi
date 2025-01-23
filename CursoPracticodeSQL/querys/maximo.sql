SELECT fecha_incorporacion FROM platzi.alumnos
ORDER BY fecha_incorporacion DESC
LIMIT 1;

SELECT carrera_id, fecha_incorporacion FROM platzi.alumnos
GROUP BY carrera_id, fecha_incorporacion
ORDER BY fecha_incorporacion DESC;

--Reto 

SELECT tutor_id, 
	MIN(nombre)
FROM platzi.alumnos
GROUP BY tutor_id
ORDER BY tutor_id;

