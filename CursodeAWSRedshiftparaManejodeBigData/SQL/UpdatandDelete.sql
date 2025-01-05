select * from sales limit 10;

select * from sales s
where eventid in (select eventid from event e
where eventname = 'Beck');

select eventid from event e
where eventname = 'Beck';

drop table sales_auxiliar;

create table sales_auxiliar as (
select * from sales s
where eventid in (select eventid from event e
where eventname = 'Macbeth'));

select eventname from event

select * from sales_auxiliar 
where salesid = 134034
limit 10;

select * from sales
where salesid = 134034
limit 10;

update sales 
set pricepaid = sa.pricepaid
from sales_auxiliar sa
where sales.salesid = sa.salesid;

delete from sales 
using sales_auxiliar
where sales.salesid = sales_auxiliar.salesid;

insert into sales (select * from sales_auxiliar);

select eventid from event e
where eventname = 'Macbeth'
