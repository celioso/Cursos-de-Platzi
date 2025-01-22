SELECT * FROM platzi.alumnos;
SELECT * FROM platzi.carreras;

SELECT * FROM platzi.alumnos AS a
JOIN platzi.carreras AS c
on a.id = c.id;