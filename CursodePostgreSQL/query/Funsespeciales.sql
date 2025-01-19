INSERT INTO public.estaciones(
	id_estacion, nombre, direccion)
	VALUES (1, 'casitas', 'st 5ft8')
	ON CONFLICT(id_estacion) DO UPDATE SET nombre='casitas', direccion='st 5ft8';

INSERT INTO public.estaciones(
	nombre, direccion)
	VALUES ('RET', 'RETDRI')
RETURNING* ;

SELECT nombre
	FROM public.pasajeros
	WHERE nombre ILIKE 'd%';

SELECT *
	FROM public.trenes
	WHERE modelo IS NOT NULL;