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

set enable_result_cache_for_session to off; -- off el cache

select max(lo_revenue), min(lo_revenue) 
from cust_sales_simple
where c_custkey < 100000; -- 5 s eoff cache 317 ms

select max(lo_revenue), min(lo_revenue) 
from cust_sales_compuesto
where c_custkey < 100000;  -- 274 ms off cache 175 ms

select max(lo_revenue), min(lo_revenue) 
from cust_sales_intercalado
where c_custkey < 100000; -- 152 ms off cache 189 ms

-------------------------------------------------------

set enable_result_cache_for_session to off; -- off el cache

select max(lo_revenue), min(lo_revenue) 
from cust_sales_simple
where c_region < 'ASIA'
and c_mktsegment = 'FURNITURE'; -- 1s off cache 986 ms

select max(lo_revenue), min(lo_revenue) 
from cust_sales_compuesto
where c_region < 'ASIA'
and c_mktsegment = 'FURNITURE'; --7s off cache 801 ms

select max(lo_revenue), min(lo_revenue) 
from cust_sales_intercalado
where c_region < 'ASIA'
and c_mktsegment = 'FURNITURE'; --5s off cache 390 ms

------------------------------------------------------

set enable_result_cache_for_session to off; -- off el cache

select max(lo_revenue), min(lo_revenue)  
from cust_sales_simple
where d_date between '01/01/1996' and '01/14/1996'
and c_mktsegment = 'FURNITURE' 
and c_region < 'ASIA'; --900 ms off cache 878 ms

select max(lo_revenue), min(lo_revenue)  
from cust_sales_compuesto
where d_date between '01/01/1996' and '01/14/1996'
and c_mktsegment = 'FURNITURE' 
and c_region < 'ASIA'; --643 ms off cache 513 ms

select max(lo_revenue), min(lo_revenue)  
from cust_sales_intercalado
where d_date between '01/01/1996' and '01/14/1996'
and c_mktsegment = 'FURNITURE'
and c_region < 'ASIA';  --393 ms off cache 211 ms
