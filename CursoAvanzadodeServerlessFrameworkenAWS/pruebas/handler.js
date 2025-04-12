const AWS = require("aws-sdk");
const { v4: uuidv4 } = require("uuid");

let dynamoDBClientParams = {};

if (process.env.IS_OFFLINE === 'true') {
  dynamoDBClientParams = {
    region: 'localhost',
    endpoint: 'http://localhost:8000',
    accessKeyId: 'DEFAULT_ACCESS_KEY',
    secretAccessKey: 'DEFAULT_SECRET'
  };
}

const dynamodb = new AWS.DynamoDB.DocumentClient(dynamoDBClientParams);
const TABLE_NAME = process.env.USERS_TABLE;

module.exports.handler = async (event) => {
  const method = event.httpMethod;
  const id = event.pathParameters?.id;

  switch (method) {
    case 'GET':
      return await getUser(id);
    case 'POST':
      return await createUser(JSON.parse(event.body));
    case 'PUT':
      return await updateUser(id, JSON.parse(event.body));
    case 'DELETE':
      return await deleteUser(id);
    default:
      return { statusCode: 400, body: "Método no soportado" };
  }
};

// Funciones CRUD
const getUser = async (id) => {
  const res = await dynamodb.get({
    TableName: TABLE_NAME,
    Key: { pk: id }
  }).promise();

  if (!res.Item) {
    return { statusCode: 404, body: JSON.stringify({ error: "Usuario no encontrado" }) };
  }

  return { statusCode: 200, body: JSON.stringify(res.Item) };
};

const createUser = async (data) => {
  const id = uuidv4();
  const item = { pk: id, ...data };

  await dynamodb.put({
    TableName: TABLE_NAME,
    Item: item
  }).promise();

  return { statusCode: 201, body: JSON.stringify(item) };
};

const updateUser = async (id, data) => {
  const params = {
    TableName: TABLE_NAME,
    Key: { pk: id },
    UpdateExpression: "set #name = :name, #age = :age",
    ExpressionAttributeNames: {
      "#name": "name",
      "#age": "age"
    },
    ExpressionAttributeValues: {
      ":name": data.name,
      ":age": data.age
    },
    ReturnValues: "ALL_NEW"
  };

  const res = await dynamodb.update(params).promise();
  return { statusCode: 200, body: JSON.stringify(res.Attributes) };
};

const deleteUser = async (id) => {
  await dynamodb.delete({
    TableName: TABLE_NAME,
    Key: { pk: id }
  }).promise();

  return { statusCode: 200, body: JSON.stringify({ mensaje: "Usuario eliminado" }) };
};
