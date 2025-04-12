import os
import json
import uuid
import boto3

is_offline = os.environ.get("IS_OFFLINE") == "true"

dynamodb = boto3.resource(
    'dynamodb',
    endpoint_url="http://localhost:8000" if is_offline else None,
    region_name="us-east-1",
    aws_access_key_id="DEFAULT_ACCESS_KEY" if is_offline else None,
    aws_secret_access_key="DEFAULT_SECRET" if is_offline else None
)

table = dynamodb.Table(os.environ.get("USERS_TABLE"))

def handler(event, context):
    http_method = event['httpMethod']
    user_id = (event.get("pathParameters") or {}).get("id")

    if http_method == "GET":
        return get_user(user_id)
    elif http_method == "POST":
        return create_user(json.loads(event["body"]))
    elif http_method == "PUT":
        return update_user(user_id, json.loads(event["body"]))
    elif http_method == "DELETE":
        return delete_user(user_id)
    else:
        return response(400, {"error": "Método no soportado"})

def get_user(user_id):
    try:
        res = table.get_item(Key={"pk": user_id})
        if "Item" not in res:
            return response(404, {"error": "Usuario no encontrado"})
        return response(200, res["Item"])
    except Exception as e:
        return response(500, {"error": str(e)})

def create_user(data):
    try:
        new_id = str(uuid.uuid4())
        item = {"pk": new_id, **data}
        table.put_item(Item=item)
        return response(201, item)
    except Exception as e:
        return response(500, {"error": str(e)})

def update_user(user_id, data):
    try:
        res = table.update_item(
            Key={"pk": user_id},
            UpdateExpression="SET #n = :name, #a = :age",
            ExpressionAttributeNames={
                "#n": "name",
                "#a": "age"
            },
            ExpressionAttributeValues={
                ":name": data["name"],
                ":age": data["age"]
            },
            ReturnValues="ALL_NEW"
        )
        return response(200, res["Attributes"])
    except Exception as e:
        return response(500, {"error": str(e)})

def delete_user(user_id):
    try:
        table.delete_item(Key={"pk": user_id})
        return response(200, {"message": "Usuario eliminado"})
    except Exception as e:
        return response(500, {"error": str(e)})

def response(status_code, body):
    return {
        "statusCode": status_code,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json"
        }
    }
