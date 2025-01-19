--Crear roles

CREATE ROLE usuario_consulta WITH
	LOGIN
	NOSUPERUSER
	NOCREATEDB
	NOCREATEROLE
	INHERIT
	NOREPLICATION
	NOBYPASSRLS
	CONNECTION LIMIT -1
	VALID UNTIL '2025-01-20T15:15:00-05:00' 
	PASSWORD 'xxxxxx';
COMMENT ON ROLE usuario_consulta IS 'Prueba';

-- asigner permisos

GRANT INSERT, UPDATE, SELECT ON TABLE public.estaciones TO usuario_consulta;

GRANT INSERT, UPDATE, SELECT ON TABLE public.pasajeros TO usuario_consulta;

GRANT INSERT, UPDATE, SELECT ON TABLE public.trayectos TO usuario_consulta;

GRANT INSERT, UPDATE, SELECT ON TABLE public.trenes TO usuario_consulta;

GRANT INSERT, UPDATE, SELECT ON TABLE public.viajes TO usuario_consulta;
