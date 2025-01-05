select * from pg_table_def
where tablename = 'sales';

select * from pg_catalog.stv_blocklist;

select * from stl_load_errors;
select * from stl_load_commits;
select * from stl_query;

select * from stl_query
where query = 10398;

select * from svl_qlog
order by starttime desc;

--para ver usuario y crearlos administrasión

select * from svl_user_info;

create user invitado password 'Password123';

select * from svv_tables
where table_schema = 'public';