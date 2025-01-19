--crear llave foranea trayecto-estacion
ALTER TABLE IF EXISTS public.trayectos DROP CONSTRAINT IF EXISTS estacion_fkey;

ALTER TABLE IF EXISTS public.trayectos DROP CONSTRAINT IF EXISTS tren_fkey;

ALTER TABLE IF EXISTS public.trayectos
    ADD CONSTRAINT trayecto_estacion_fkey FOREIGN KEY (id_estacion)
    REFERENCES public.estaciones (id_estacion) MATCH SIMPLE
    ON UPDATE CASCADE
    ON DELETE CASCADE
    NOT VALID;

--craer llave foranea con tren-estacion
ALTER TABLE IF EXISTS public.trayectos
    ADD CONSTRAINT trayecto_tren_fkey FOREIGN KEY (id_tren)
    REFERENCES public.trenes (id_tren) MATCH SIMPLE
    ON UPDATE CASCADE
    ON DELETE CASCADE
    NOT VALID;

--crear llaver foranea con viaje-trayecto

ALTER TABLE public.viajes
    ALTER COLUMN inicio TYPE timestamp with time zone ;

ALTER TABLE public.viajes
    ALTER COLUMN fin TYPE timestamp with time zone ;
ALTER TABLE IF EXISTS public.viajes DROP CONSTRAINT IF EXISTS id_trayecto_fkey;

ALTER TABLE IF EXISTS public.viajes DROP CONSTRAINT IF EXISTS n_documento_fkey;

ALTER TABLE IF EXISTS public.viajes
    ADD CONSTRAINT viaje_trayecto_fkey FOREIGN KEY (id_trayecto)
    REFERENCES public.trayectos (id_trayecto) MATCH SIMPLE
    ON UPDATE CASCADE
    ON DELETE CASCADE
    NOT VALID;

ALTER TABLE IF EXISTS public.viajes DROP CONSTRAINT IF EXISTS viaje_trayecto_fkey;

ALTER TABLE IF EXISTS public.viajes
    ADD CONSTRAINT viaje_pasajero_fkey FOREIGN KEY (n_documento)
    REFERENCES public.pasajeros (n_documento) MATCH SIMPLE
    ON UPDATE NO ACTION
    ON DELETE NO ACTION
    NOT VALID;