SELECT * FROM pasajeros
join viajes ON (viajes.n_documento = pasajeros.n_documento);

SELECT * FROM pasajeros
LEFT JOIN viajes ON (viajes.n_documento = pasajeros.n_documento)
WHERE viajes.id_viaje IS NULL;



