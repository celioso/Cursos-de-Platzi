--Crear PL
DROP FUNCTION importantepl();
CREATE OR REPLACE FUNCTION importantePL()
RETURNS integer
AS $$
DECLARE
    rec RECORD;
    contador integer := 0;
BEGIN
    FOR rec IN SELECT * FROM pasajeros LOOP
        RAISE NOTICE 'Un pasajero se llama %', rec.nombre;
        contador := contador + 1;
    END LOOP;
    RAISE NOTICE 'El conteo es %', contador;
	RETURN contador;
END;
$$
LANGUAGE plpgsql;



SELECT importantePL();

-- Nuevo

SELECT public.importantpl()