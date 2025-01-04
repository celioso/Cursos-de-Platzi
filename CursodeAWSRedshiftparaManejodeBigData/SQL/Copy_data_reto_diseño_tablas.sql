copy public.customer from 's3://awssampledbuswest2/ssbgz/customer' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'  
gzip compupdate off region 'us-west-2';

copy dwdate from 's3://awssampledbuswest2/ssbgz/dwdate' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'  
gzip compupdate off region 'us-west-2';

copy lineorder from 's3://awssampledbuswest2/ssbgz/lineorder' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'  
gzip compupdate off region 'us-west-2';

copy part from 's3://awssampledbuswest2/ssbgz/part' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'  
gzip compupdate off region 'us-west-2';

copy supplier from 's3://awssampledbuswest2/ssbgz/supplier' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'  
gzip compupdate off region 'us-west-2';

select * from customer limit 10;

analyze compression customer;

--DROP TABLE public.customer

create table if not exists public.customer_pro
(
  c_custkey INTEGER NOT null ENCODE az64 distkey,
  c_name VARCHAR(25) NOT null ENCODE zstd,
  c_address VARCHAR(25) NOT null ENCODE zstd,
  c_city VARCHAR(10) NOT null ENCODE bytedict,
  c_nation VARCHAR(15) NOT null ENCODE bytedict,
  c_region VARCHAR(12) NOT null ENCODE bytedict,
  c_phone VARCHAR(15) NOT null ENCODE zstd,
  c_mktsegment VARCHAR(10) NOT null ENCODE bytedict
);

alter table public.customer_pro owner to platzi;

insert into customer_pro (select * from customer);

select * from customer_pro limit 10;

analyze compression customer_pro;

