SELECT MAX(precio_renta) FROM peliculas;
SELECT MIN(precio_renta) FROM peliculas;

SELECT titulo, MAX(precio_renta) FROM peliculas
GROUP BY titulo;

SELECT SUM(precio_renta) FROM peliculas;

SELECT clasificacion, COUNT(*) 
FROM peliculas
GROUP BY clasificacion;

SELECT AVG(precio_renta)
FROM peliculas;

SELECT clasificacion, AVG(precio_renta) AS precio_promedio
FROM peliculas
GROUP BY clasificacion
ORDER BY precio_promedio DESC;

SELECT clasificacion, AVG(duracion) AS duracion_promedio
FROM peliculas
GROUP BY clasificacion
ORDER BY duracion_promedio DESC;


SELECT clasificacion, AVG(duracion_renta) AS duracion_renta_promedio
FROM peliculas
GROUP BY clasificacion
ORDER BY duracion_renta_promedio DESC;