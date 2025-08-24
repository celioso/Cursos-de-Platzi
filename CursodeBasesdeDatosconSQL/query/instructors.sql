--- 23-08-2025 16:26:36 SQLite.6
CREATE TABLE Instructor (
    instructor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT,  
    age INTEGER,
    email TEXT UNIQUE,
    loaddate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updatedate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assignature TEXT
);


