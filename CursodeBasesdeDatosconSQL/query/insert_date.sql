-- Insert de estudiantes:


INSERT INTO STUDENTS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Andres', 'Badillo', '40', 'andres.badillo@example.com');
INSERT INTO STUDENTS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Jane', 'Miller', 22, 'jane.miller@example.com');
INSERT INTO STUDENTS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Mark', 'Taylor', 27, 'mark.taylor@example.com');
INSERT INTO STUDENTS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Lucy', 'Garcia', 20, 'lucy.garcia@example.com');
INSERT INTO STUDENTS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Peter', 'Gonzalez', 29, 'peter.gonzalez@example.com');

-- Insert de instructores:


INSERT INTO INSTRUCTORS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Alice', 'Smith', 40, 'alice.smith@example.com'); 
INSERT INTO INSTRUCTORS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Bob', 'Anderson', 35, 'bob.anderson@example.com');
INSERT INTO INSTRUCTORS (FIRSTNAME, LASTNAME, AGE, EMAIL)
VALUES ('Carol', 'Johnson', 45, 'carol.johnson@example.com');

--Insert de cursos:


INSERT INTO COURSES ( COURSENAME, DESCRIPTION, INSTRUCTORID,  DURATIONHOURS)
VALUES ('Introduction to SQL', 'Conceptos básicos de SQL y bases de datos relacionales', 1, 3);
INSERT INTO COURSES (COURSENAME, DESCRIPTION, INSTRUCTORID, DURATIONHOURS)
VALUES ('Advanced SQL Queries', 'Técnicas avanzadas de consultas en SQL',2 , 4);
INSERT INTO COURSES (COURSENAME, DESCRIPTION, INSTRUCTORID,  DURATIONHOURS)
VALUES ('Data Analysis with Python', 'Introducción a librerías para análisis de datos',1 , 5);
INSERT INTO COURSES (COURSENAME, DESCRIPTION, INSTRUCTORID, DURATIONHOURS)
VALUES ('Machine Learning Fundamentals', 'Conceptos básicos de aprendizaje automático', 3, 4);

--Insert de cursos x estudiante:


INSERT INTO STUDENT_COURSE (STUDENT_ID, COURSE_ID)
VALUES (1, 1);
INSERT INTO STUDENT_COURSE (STUDENT_ID, COURSE_ID)
VALUES (1, 3);
INSERT INTO STUDENT_COURSE (STUDENT_ID, COURSE_ID)
VALUES (2, 2);
INSERT INTO STUDENT_COURSE (STUDENT_ID, COURSE_ID)
VALUES (3, 1);
INSERT INTO STUDENT_COURSE (STUDENT_ID, COURSE_ID)
VALUES (4, 4);
INSERT INTO STUDENT_COURSE (STUDENT_ID, COURSE_ID)
VALUES (5, 1);
INSERT INTO STUDENT_COURSE (STUDENT_ID, COURSE_ID)
VALUES (5, 2);