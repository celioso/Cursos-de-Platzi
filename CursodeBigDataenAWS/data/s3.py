import boto3
import sys
import botocore


region = sys.argv[1]
# region = "us-east-1" para no colocar python s3.py us-east-1 al iniciar el codigo

s3 = boto3.client(
    's3',
    region_name = region
)

response = s3.list_buckets()
print (response)