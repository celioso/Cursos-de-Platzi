--- 22-08-2025 21:07:44 SQLite.2
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    age INTEGER CHECK (age >= 0 AND age <= 120),
    email TEXT UNIQUE,
    load_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Escribe aquí tu código SQL 👇
SELECT * FROM cursos;

SELECT count(*) AS cantidad FROM cursos;

SELECT nombre AS name, 
  profe AS teacher, n_calificaciones
  n_reviews FROM cursos;



