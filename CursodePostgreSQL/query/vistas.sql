SELECT * FROM rango_view;

SELECT * FROM viajes WHERE inicio > '22:00:00';

SELECT * FROM despues_noche_mview;
REFRESH MATERIALIZED VIEW despues_noche_mview;

DELETE FROM viajes WHERE id_viaje = 27;
SELECT * FROM viajes WHERE inicio > '22:00:00';