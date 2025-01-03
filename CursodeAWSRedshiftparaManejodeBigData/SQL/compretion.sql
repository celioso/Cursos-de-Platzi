select * from venue limit 10;
select count(0) from venue;
select count(*) from venue;

select venueid, venuename, venuecity, venuestate, venueseats 
from venue limit 10;

create table cartesion_venue as (
select venueid, venuename, venuecity, venuestate, venueseats 
from venue, listing);

select count(*) from cartesion_venue;

select * from cartesion_venue limit 10;

-- crear tablas
create table encoding_venue (
namerow varchar(100) encode raw,
namebytedict varchar(100) encode bytedict,
namelzo varchar(100) encode lzo,
namerunlength varchar(100) encode runlength,
nametext255 varchar(100) encode text255,
nametext32k varchar(100) encode text32k,
namezstd varchar(100) encode zstd
);

select * from encoding_venue;

insert into encoding_venue(
select venuename, venuename, venuename, venuename, venuename, venuename, venuename 
from cartesion_venue
);

select count(0) from encoding_venue limit 10;