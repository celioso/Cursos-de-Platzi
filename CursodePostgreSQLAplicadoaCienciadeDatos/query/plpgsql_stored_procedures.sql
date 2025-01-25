CREATE OR REPLACE PROCEDURE test_drcreate_procedure()
LANGUAGE SQL
AS $$
	DROP TABLE IF EXISTS aaa;
	CREATE TABLE aaa (bbb char(5) CONSTRAINT firstket PRIMARY KEY);
$$;

CALL test_drcreate_procedure();


CREATE OR REPLACE FUNCTION test_dropcreate_function()
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
	DROP TABLE IF EXISTS aaa;
	CREATE TABLE aaa (bbb char(5) CONSTRAINT firstkey PRIMARY KEY, ccc char(5));
	DROP TABLE IF EXISTS aaab;
	CREATE TABLE aaab (bbba char(5) CONSTRAINT secondkey PRIMARY KEY, ccca char(5));
END
$$;

SELECT test_dropcreate_function();