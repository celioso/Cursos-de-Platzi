--insertar datos
INSERT INTO public.vip(
	id, fecha)
	VALUES (50, '2012-05-23');

-- remoto
--CREATE EXTENSION dblink;

SELECT * FROM 
dblink ('dbname=remota 
		port=5432 
		host=127.0.0.1 
		user=usuario_consulta 
		password=123456',
		'SELECT id, fecha FROM vip')
		AS datos_remotos (id integer, fecha date);

--Local

SELECT * FROM pasajeros
JOIN
dblink ('dbname=remota 
		port=5432 
		host=127.0.0.1 
		user=usuario_consulta 
		password=123456',
		'SELECT id, fecha FROM vip')
		AS datos_remotos (id integer, fecha date)
ON (pasajeros.n_documento = datos_remotos.id);
-- USING (id)