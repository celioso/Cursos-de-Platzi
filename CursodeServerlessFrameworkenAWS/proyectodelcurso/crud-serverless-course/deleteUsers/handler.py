import boto3
import json
import os

table_name = os.environ['DYNAMODB_CUSTOMER_TABLE']

if os.environ.get("IS_OFFLINE"):
    dynamodb = boto3.resource(
        'dynamodb',
        region_name='localhost',
        endpoint_url='http://localhost:8000',
        aws_access_key_id='DEFAULT_ACCESS_KEY',
        aws_secret_access_key='DEFAULT_SECRET'
    )
else:
    dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table(table_name)

def deleteUsers(event, context):
    user_id = event['pathParameters']['id']

    try:
        existing = table.get_item(Key={'pk': user_id})
        if 'Item' not in existing:
            return {
                'statusCode': 404,
                'body': json.dumps({'message': f'User {user_id} not found'})
            }

        result = table.delete_item(Key={'pk': user_id})
        body = json.dumps({'message': f"user {user_id} deleted"})
        status_code = result['ResponseMetadata']['HTTPStatusCode']
    except Exception as e:
        print(f"Error deleting user {user_id}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Internal server error', 'error': str(e)})
        }

    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json'},
        'body': body
    }
