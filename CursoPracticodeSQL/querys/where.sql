SELECT * FROM platzi.alumnos
WHERE id = 1;

SELECT * FROM platzi.alumnos
WHERE carrera_id > 20;

SELECT * FROM platzi.alumnos
WHERE nombre = 'Aile'
AND (
apellido = 'Chedgey'
OR 
apellido = 'Lopez'
);

SELECT * FROM platzi.alumnos
WHERE nombre = 'Aile'
AND
apellido = 'Chedgey'
OR 
apellido = 'Lopez';

SELECT * FROM platzi.alumnos
WHERE nombre LIKE 'Is%';

SELECT * FROM platzi.alumnos
WHERE nombre LIKE 'Is_e';

SELECT * FROM platzi.alumnos
WHERE nombre NOT LIKE 'Is_ael';

SELECT * FROM platzi.alumnos
WHERE nombre IS NULL;

SELECT * FROM platzi.alumnos
WHERE nombre IS NOT NULL;

SELECT * FROM platzi.alumnos
WHERE nombre IN('Hilde', 'Doug', 'Luce');
