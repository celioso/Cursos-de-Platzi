--Create table
create table estudiante 
( 
id int2, 
nombre varchar(20),
apellido varchar(20),
edad int2, 
fecha_ingreso date 
);

select * from estudiante;



--Cargue con banner y delimitado por; 
copy estudiante from 's3://mibucketredshift1/primer_cargue.csv' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
region 'us-west-2'
delimiter ';'
ignoreheader 1 
IGNOREBLANKLINES
BLANKSASNULL 
;

select * from stl_load_errors; --para ver el error

--Cargue con banner y delimitado por ; con otro formato
copy estudiante from 's3://mibucketredshift1/segundo_cargue.csv' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
region 'us-west-2'
delimiter ';'
ignoreheader 1 
IGNOREBLANKLINES 
BLANKSASNULL 
dateformat 'mm-dd-yyyy';

--tercer cargue sin delimitador, solo cargue por tamaños fijos.
copy estudiante from 's3://mibucketredshift1/segundo_cargue.csv' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
region 'us-west-2'
fixedwidth '0:1,1:9,2:9,3:2,4:10'
dateformat 'mm-dd-yyyy'
;


-- Tablas 
select * from stl_load_errors order by 4 desc;
select * from STL_LOAD_COMMITS order by 10 desc ;


copy estudiante from 's3://mibucketredshift1/test.manifest' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
delimiter ';'
ignoreheader 1
manifest
region 'us-west-2';


select * from estudiante;
truncate table estudiante;

CREATE TABLE IF NOT EXISTS public.sales_compression_on
(
	salesid INTEGER NOT NULL  ENCODE raw
	,listid INTEGER NOT NULL  ENCODE raw
	,sellerid INTEGER NOT NULL  ENCODE raw
	,buyerid INTEGER NOT NULL  ENCODE raw
	,eventid INTEGER NOT NULL  ENCODE raw
	,dateid SMALLINT NOT NULL  ENCODE RAW
	,qtysold SMALLINT NOT NULL  ENCODE raw
	,pricepaid NUMERIC(8,2)   ENCODE raw
	,commission NUMERIC(8,2)   ENCODE raw
	,saletime TIMESTAMP WITHOUT TIME ZONE   ENCODE raw
);

CREATE TABLE IF NOT EXISTS public.sales_compression_off
(
	salesid INTEGER NOT NULL  ENCODE raw
	,listid INTEGER NOT NULL  ENCODE raw
	,sellerid INTEGER NOT NULL  ENCODE raw
	,buyerid INTEGER NOT NULL  ENCODE raw
	,eventid INTEGER NOT NULL  ENCODE raw
	,dateid SMALLINT NOT NULL  ENCODE RAW
	,qtysold SMALLINT NOT NULL  ENCODE raw
	,pricepaid NUMERIC(8,2)   ENCODE raw
	,commission NUMERIC(8,2)   ENCODE raw
	,saletime TIMESTAMP WITHOUT TIME ZONE   ENCODE raw
);

copy public.sales_compression_on 
from 's3://mibucketredshift1/tickitdb/sales_tab.txt'
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'
delimiter '\t' timeformat 'MM/DD/YYYY HH:MI:SS' compupdate on region 'us-west-2';

copy public.sales_compression_off
from 's3://mibucketredshift1/tickitdb/sales_tab.txt'
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'
delimiter '\t' timeformat 'MM/DD/YYYY HH:MI:SS' compupdate off region 'us-west-2';

select * from PG_TABLE_DEF
where tablename = 'sales_compression_off';

select * from PG_TABLE_DEF
where tablename = 'sales_compression_on';