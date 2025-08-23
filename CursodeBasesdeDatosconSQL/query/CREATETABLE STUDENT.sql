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


