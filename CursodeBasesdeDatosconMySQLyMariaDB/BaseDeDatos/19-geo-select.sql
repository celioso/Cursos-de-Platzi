USE metro_cdmx;

SELECT ST_Distance_Sphere(
    POINT(-99.14912224, 19.42729875),
    POINT(-99.13303971, 19.42543703)
) AS Distance;

--Calculamos en KIlometros con datos quemados

SELECT ST_Distance_Sphere(
    POINT(-99.14912224, 19.42729875),
    POINT(-99.13303971, 19.42543703)
) /1000 AS Distance;

--Reto 
SELECT ST_Distance_Sphere(
    (SELECT `location`
(
    POINT(-99.14912224, 19.42729875),
    POINT(-99.13303971, 19.42543703)
) AS Distance;

--Calculamos en KIlometros con datos quemados

SELECT ST_Distance_Sphere(
    POINT(-99.14912224, 19.42729875),
    POINT(-99.13303971, 19.42543703)
) /1000 AS Distance;)

-- Calculamos en kilometros  con consultas anidadas

SELECT 
ST_Distance_Sphere(
    (
        SELECT `locations`.`location`
        FROM `locations`
        INNER JOIN `stations`
        ON `stations`.`id`=`locations`.`station_id`
        WHERE `stations`.`name`="Lazaro Cardenas"
    ),
    (
        SELECT `locations`.`location`
        FROM `locations`
        INNER JOIN `stations`
        ON `stations`.`id`=`locations`.`station_id`
        WHERE `stations`.`name`="Chilpancingo"
    )

)/1000 AS Distancia;