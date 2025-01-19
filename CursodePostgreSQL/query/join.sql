SELECT * FROM pasajeros
join viajes ON (viajes.id_pasajero = pasajeros.n_documento);

SELECT * FROM pasajeros
LEFT JOIN viajes ON (viajes.id_pasajero = pasajeros.n_documento)
WHERE viajes.id_viajes IS NULL;