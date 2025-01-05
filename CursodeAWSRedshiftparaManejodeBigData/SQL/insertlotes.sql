select * from estudiante;

insert  into estudiante values
(5, 'Brandon', 'Huertas', 24, '2025-01-04'),
(6, 'Cristian', 'Salas', 30, '2024-01-04'),
(7, 'Holman', 'Capos', 56, '2024-10-25'),
(8, 'Natalia', 'Montenegro', 32, '2025-06-18');

--Bulk select/insert

select e.eventname, e.starttime, 
sum(pricepaid) pricepaid, sum(commission) commission 
from sales s
inner join event e
on s.eventid = e.eventid
group by e.eventname, e.starttime;

create table total_price_by_event as (
select e.eventname, e.starttime, 
sum(pricepaid) pricepaid, sum(commission) commission 
from sales s
inner join event e
on s.eventid = e.eventid
group by e.eventname, e.starttime
);

select * from total_price_by_event;

--Deep copy

create table likesales (like sales);
insert into likesales(select * from sales);
drop table sales;
alter table likesales rename to sales

select * from sales;