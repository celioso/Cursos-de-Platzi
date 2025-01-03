copy users from 's3://mibucketredshift1/tickitdb/allusers_pipe.txt' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
delimiter '|' region 'us-west-2';

copy venue from 's3://mibucketredshift1/tickitdb/venue_pipe.txt' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
delimiter '|' region 'us-west-2';

copy category from 's3://mibucketredshift1/tickitdb/category_pipe.txt' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
delimiter '|' region 'us-west-2';

copy date from 's3://mibucketredshift1/tickitdb/date2008_pipe.txt' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
delimiter '|' region 'us-west-2';

copy event from 's3://mibucketredshift1/tickitdb/allevents_pipe.txt' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
delimiter '|' timeformat 'YYYY-MM-DD HH:MI:SS' region 'us-west-2';

copy listing from 's3://mibucketredshift1/tickitdb/listings_pipe.txt' 
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift' 
delimiter '|' region 'us-west-2';

copy sales from 's3://mibucketredshift1/tickitdb/sales_tab.txt'
credentials 'aws_iam_role=arn:aws:iam::528757819805:role/MyRoleRedshift'
delimiter '\t' timeformat 'MM/DD/YYYY HH:MI:SS' region 'us-west-2';

select *from listing limit 10;

select count(0) from listing;