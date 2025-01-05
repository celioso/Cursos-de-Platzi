explain
select eventid, eventname, event.venueid, venuename 
from event, venue;

explain
select eventid, eventname, event.venueid, venuename 
from event, venue
where event.venueid = venue.venueid;

select * from pg_table_def
where tablename in ('event', 'venue');

CREATE TABLE IF NOT EXISTS public.event_2
(
	eventid INTEGER NOT NULL  ENCODE az64
	,venueid SMALLINT NOT NULL  ENCODE az64 distkey sortkey
	,catid SMALLINT NOT NULL  ENCODE az64
	,dateid SMALLINT NOT NULL  ENCODE RAW
	,eventname VARCHAR(200)   ENCODE lzo
	,starttime TIMESTAMP WITHOUT TIME ZONE   ENCODE az64
);

insert into event_2(select * from event);

explain
select eventid, eventname, event_2.venueid, venuename 
from event_2, venue
where event_2.venueid = venue.venueid;

analyze event_2;

explain
select e.eventname, sum(pricepaid) from sales s
inner join event e
on s.eventid = e.eventid
group by e.eventname;

explain
select sum(pricepaid) from sales s

select * from pg_table_def
where tablename in ('event');

explain
select * from event;

explain
select * from event order by eventid;

explain
select * from event order by dateid;

select * from stl_alert_event_log order by query desc;

select * from stl_query
where query = 19282;

select eventid, eventname, event.venueid, venuename 
from event, venue;


