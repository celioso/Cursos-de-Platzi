INSERT INTO public.estaciones(
	id_estacion, nombre, direccion)
	VALUES (1, 'Casitas', '621 Green Drive')
	ON CONFLICT(id_estacion) DO UPDATE SET nombre='casitas', direccion='621 Green Drive';

SELECT * FROM estaciones;


INSERT INTO public.estaciones(
	id_estacion,nombre, direccion)
	VALUES (301, 'Retiro', '4 North and westh')
RETURNING* ;

SELECT nombre
	FROM public.pasajeros
	WHERE nombre ILIKE 'd%'; --% todos los que empieza _ el primero
-- like mayusculas y ilike todas
SELECT *
	FROM public.trenes
	WHERE modelo IS NOT NULL;