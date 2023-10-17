USE metro_cdmx

--Corregir los errores enn los nombres de las stations
UPDATE `stations`
SET name = "Lázaro Cárdenas" 
WHERE id=1;

UPDATE `stations`
SET name="Ferrería"
WHERE id=2;

UPDATE `stations`
SET name="Pantitlán"
WHERE id=3;

UPDATE `stations`
SET name="Tacuba"
WHERE id=4;

UPDATE `stations`
SET name="Martín Carrera"
WHERE id=5;