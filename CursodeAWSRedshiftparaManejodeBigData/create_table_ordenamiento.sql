CREATE TABLE public.cust_sales_date (
	c_custkey int4 NULL,
	c_nation varchar(15) NULL,
	c_region varchar(12) NULL,
	c_mktsegment varchar(10) NULL,
	d_date date NULL,
	lo_revenue int4 NULL
);

copy cust_sales_date from 's3://mibucketredshift1/cust_sales_date_000.bz2' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
BZIP2
region 'us-west-2';

select count(0) from cust_sales_date;
select * from cust_sales_date limit 10;

create table cust_sales_simple
sortkey (c_custkey)
as (select * from cust_sales_date
);

create table auxiliar (col int);
insert into auxiliar values (1), (2), (3), (4),(5);

select * from auxiliar;

select c_custkey, c_nation, c_region, c_mktsegment, d_date, lo_revenue 
from cust_sales_date, auxiliar
limit 5;

create table cust_sales_simple
sortkey (c_custkey)
as (select c_custkey, c_nation, c_region, c_mktsegment, d_date, lo_revenue 
from cust_sales_date, auxiliar
);

select count(0) from cust_sales_simple;

create table cust_sales_compuesto
compound sortkey (c_custkey, c_region, c_mktsegment, d_date)
as (select c_custkey, c_nation, c_region, c_mktsegment, d_date, lo_revenue 
from cust_sales_date, auxiliar
);

create table cust_sales_intercalado
interleaved sortkey (c_custkey, c_region, c_mktsegment, d_date)
as (select c_custkey, c_nation, c_region, c_mktsegment, d_date, lo_revenue 
from cust_sales_date, auxiliar
);

select count(0) from cust_sales_simple;


