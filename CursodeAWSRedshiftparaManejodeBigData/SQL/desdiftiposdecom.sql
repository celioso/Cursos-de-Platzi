select * from encoding_venue limit 10;

select * from stv_tbl_perm 
where name = 'encoding_venue';

select col, max(blocknum) from pg_catalog.stv_blocklist 
where tbl = 106715
and col <= 6
group by col;

select * from pg_catalog.stv_blocklist --47924
where tbl = 106715
and col = 0;

select * from pg_catalog.stv_blocklist --1177191
where tbl = 106715
and col = 6;

analyze compression cartesion_venue;