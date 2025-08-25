-- MUESTRA los nombre que comienzan con J
/*SELECT * FROM STUDENTS
WHERE lastname like 'J%';*/

-- Filtra la edad
/*SELECT firstname, lastname FROM STUDENTS
WHERE AGE = 20;*/

SELECT firstname, lastname FROM STUDENTS
WHERE lastname LIKE '%O%';

-- reto
SELECT firstname, lastname, age FROM Student WHERE firstname LIKE "M%" AND age=20 AND lastname LIKE "%o%";