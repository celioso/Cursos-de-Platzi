SELECT n_documento, COALESCE(nombre,'No Aplica') nombre,
direccion_residencia, fecha_nacimiento
FROM public.pasajeros 
WHERE n_documento=59128269;

SELECT NULLIF (0,0);

SELECT GREATEST (0, 1, 2, 3, 4, 5, 6,2 ,4);

SELECT LEAST (0, 1, 2, 3, 4, 5, 6,2 ,4);

SELECT n_documento, nombre, direccion_residencia, fecha_nacimiento,
CASE 
WHEN fecha_nacimiento > '2000-01-01' THEN
'Niño'
ELSE
'Mayor'
END
	FROM public.pasajeros;