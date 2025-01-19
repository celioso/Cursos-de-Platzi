-- Tabla pasajeros
CREATE TABLE public.pasajeros
(
    n_documento serial NOT NULL,
    nombre character varying(50) NOT NULL,
    direccion_residencia character varying(50),
    fecha_nacimiento date,
    CONSTRAINT documento_pkey PRIMARY KEY (n_documento)
);

ALTER TABLE IF EXISTS public.pasajeros
    OWNER to postgres;

--tabla estaciones

CREATE TABLE public.estaciones
(
    id_estacion serial NOT NULL,
    nombre character varying(50) NOT NULL,
    direccion character varying(50),
    CONSTRAINT estacion_pkey PRIMARY KEY (id_estacion)
);

ALTER TABLE IF EXISTS public.estaciones
    OWNER to postgres;

--tabla trenes

CREATE TABLE public.trenes
(
    id_tren serial NOT NULL,
    modelo character varying(50) NOT NULL,
    capacidad integer,
    CONSTRAINT tren_pkey PRIMARY KEY (id_tren)
);

ALTER TABLE IF EXISTS public.trenes
    OWNER to postgres;


--tabla de trayecto

CREATE TABLE public.trayectos
(
    id_trayecto serial NOT NULL,
    id_estacion integer NOT NULL,
    id_tren integer NOT NULL,
    nombre character varying NOT NULL,
    CONSTRAINT trayecto_pkey PRIMARY KEY (id_trayecto),
    CONSTRAINT estacion_fkey FOREIGN KEY (id_estacion)
        REFERENCES public.estaciones (id_estacion) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID,
    CONSTRAINT tren_fkey FOREIGN KEY (id_tren)
        REFERENCES public.trenes (id_tren) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID
);

ALTER TABLE IF EXISTS public.trayectos
    OWNER to postgres;

--Tabla de viajes

CREATE TABLE public.viajes
(
    id_viaje serial NOT NULL,
    n_documento integer NOT NULL,
    id_trayecto integer NOT NULL,
    inicio date NOT NULL,
    fin date NOT NULL,
    CONSTRAINT viaje_pkey PRIMARY KEY (id_viaje),
    CONSTRAINT n_documento_fkey FOREIGN KEY (n_documento)
        REFERENCES public.pasajeros (n_documento) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID,
    CONSTRAINT id_trayecto_fkey FOREIGN KEY (id_trayecto)
        REFERENCES public.trayectos (id_trayecto) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID
);

ALTER TABLE IF EXISTS public.viajes
    OWNER to postgres;



