import boto3
from boto3.dynamodb.conditions import Key, Attr

if __name__ == '__main__':
    # crea datos para aws
    dynamodb = boto3.resource('dynamodb')

    table = dynamodb.Table('Personajes')


    table.put_item(
            Item = {
                'Id': 546,
                'Gender': 'F',
                'Status': 'Juana'
            }
    )


    response = table.query(
            KeyConditionExpression=Key('Id').eq(546)
    )

    print(response)

# Ver las tablas en AWS
    '''client = boto3.client('dynamodb')
    responseListTable = client.list_tables()

    print(responseListTable)'''