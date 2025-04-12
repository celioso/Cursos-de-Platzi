const aws = require("aws-sdk");

let dynamoDBClienteParams = {} 

if (process.env.IS_OFFLINE) {
  dynamoDBClienteParams = {
    region: 'localhost',
    endpoint: 'http://localhost:8000',
    accessKeyId: 'DEFAULT_ACCESS_KEY',
    secretAccessKey: 'DEFAULT_SECRET'
  };
}

const dynamodb = new aws.DynamoDB.DocumentClient(dynamoDBClienteParams);

const getUsers = async (event, context) => {
  let userId = event.pathParameters.id;

  const params = {
    ExpressionAttributeValues: { ":pk": userId },
    KeyConditionExpression: "pk = :pk",
    TableName: "usersTable",
  };

  const res = await dynamodb.query(params).promise();

  return {
    statusCode: 200,
    body: JSON.stringify({ user: res })
  };
};

module.exports = {
  getUsers
};