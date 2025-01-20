SELECT * FROM rango_view;

SELECT * FROM viejes Where inicio > '22:00:00';

SELECT * FROM despues_noche_mview;
REFRESH MATERIALIZED VIEW despues_noche_mview;

DELETE FROM viajes WHERE id_viajes = 2;
SELECT * FROM viajes WHERE inicio > '20:00:00';