--extrae los intructores

/*SELECT * FROM INSTRUCTORS
WHERE instructorid IN (1, 2);*/

--sea diferente 
/*SELECT * FROM INSTRUCTORS
WHERE instructorid != 1;*/

--extraer un instructor
SELECT * FROM INSTRUCTORS
WHERE firstname = 'Bob';

--Por salario por mayor
SELECT * FROM INSTRUCTORS
WHERE SALARY > 5000;

SELECT * FROM INSTRUCTORS
WHERE SALARY BETWEEN 3000 AND 9000;