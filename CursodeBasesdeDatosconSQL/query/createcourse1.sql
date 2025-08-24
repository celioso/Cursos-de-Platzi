--- 23-08-2025 16:28:43 SQLite.4
CREATE TABLE Courses (
    course_id INTEGER PRIMARY KEY AUTOINCREMENT,
    coursename TEXT NOT NULL,
    description TEXT,
    instructor_id INTEGER,
    durationhours INTEGER,
    load_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instructor_id) REFERENCES Instructor(instructor_id)
);


