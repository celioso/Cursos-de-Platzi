CREATE TABLE films (
    code CHAR(5),
    title VARCHAR(40),
    did INTEGER,
    date_prod DATE,
    kind VARCHAR(10),
    len INTERVAL HOUR TO MINUTE
);

INSERT INTO films (code, title, did, date_prod, kind, len)
VALUES ('12345', 'Nombre de la película', 1, '2023-10-12', 'Drama', '02:00');

SELECT * FROM films;

INSERT INTO films (code, title, did, date_prod, kind, len)
VALUES ('12346', 'Terminator', 2, '1984-10-12', 'Accion', '01:35');