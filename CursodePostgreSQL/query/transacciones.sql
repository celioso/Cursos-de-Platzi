BEGIN;

INSERT INTO public.trenes(
	id_tren, modelo, capacidad)
	VALUES (52, 'Modelo Ter6754', 200);

INSERT INTO public.estaciones(
	id_estacion, nombre, direccion)
	VALUES (108, 'Estación Transac', 'direccion');

--ROLLBACK;

COMMIT;