create table unload_test as (
select * from cust_sales_intercalado
union all
select * from cust_sales_intercalado
union all
select * from cust_sales_intercalado
);

select count(0) from cust_sales_intercalado;

select count(0) from unload_test;

create table unload_test_2 as (
select * from unload_test
limit 2000000);

--Sube como 1.8 Gb y para con pagar use el count(0) para que no me cobren
unload ('select *from unload_test')
to 's3://mibucketredshift1/unload/unload_test_'
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'
parallel off
allowoverwrite --sobre escribir el archivo
;


unload ('select *from unload_test_2')
to 's3://mibucketredshift1/unload/unload_test_2_'
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'
allowoverwrite --sobre escribir el archivo
delimiter ';'
header
maxfilesize 500 mb --un maximo por archovo sea de 500mb
ZSTD -- usa una codificacion gzip, bzip2, zstd 
manifest
partition by (el nombre de cualquier columna) include
;

