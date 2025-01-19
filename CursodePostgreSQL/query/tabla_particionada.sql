--TAbla de particion
CREATE TABLE public.bitacoras_viaje
(
    id serial,
    id_viaje integer,
    fecha date
) PARTITION BY RANGE (fecha);

ALTER TABLE IF EXISTS public.bitacoras_viaje
    OWNER to postgres;

--insertar

INSERT INTO public.bitacoras_viaje(
	id_viaje, fecha)
	VALUES (1, '2025-04-29');

CREATE TABLE bitacoras_viaje2025_2026 PARTITION OF bitacoras_viaje
FOR VALUES FROM ('2025-01-01') to ('2026-01-01');

SELECT * FROM bitacoras_viaje;