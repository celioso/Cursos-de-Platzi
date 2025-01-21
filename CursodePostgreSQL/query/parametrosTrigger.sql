-- FUNCTION: public.importantpl()

-- DROP FUNCTION IF EXISTS public.importantpl();

CREATE OR REPLACE FUNCTION public.importantpl(
	)
    RETURNS TRIGGER
    LANGUAGE 'plpgsql'
AS $BODY$
DECLARE
    rec RECORD;
    contador integer := 0;
BEGIN
    FOR rec IN SELECT * FROM pasajeros LOOP
        contador := contador + 1;
    END LOOP;
    INSERT INTO conteo_pasajeros (total, tiempo)
	VALUES (contador, now());
END;

$BODY$;
