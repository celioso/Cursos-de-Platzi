CREATE TRIGGER mitrigger
AFTER INSERT
ON pasajeros
FOR EACH ROW
EXECUTE PROCEDURE public.importantpl();

INSERT INTO public.pasajeros(
	n_documento, nombre, direccion_residencia, fecha_nacimiento)
	VALUES (901, 'Carlos Franco', 'str Del paso', '2001-06-15');