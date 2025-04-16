const {
  DynamoDBClient,
} = require("@aws-sdk/client-dynamodb");
const {
  DynamoDBDocumentClient,
  QueryCommand,
} = require("@aws-sdk/lib-dynamodb");

let clientParams = {};

if (process.env.IS_OFFLINE) {
  clientParams = {
    region: "localhost",
    endpoint: "http://localhost:8002",
    credentials: {
      accessKeyId: "DEFAULTACCESSKEY",
      secretAccessKey: "DEFAULTSECRET",
    },
  };
}

const ddbClient = new DynamoDBClient(clientParams);
const ddbDocClient = DynamoDBDocumentClient.from(ddbClient);

const getUsers = async (event) => {
  try {
    const userId = event.pathParameters.id;

    const params = {
      ExpressionAttributeValues: {":pk": userId,},
      KeyConditionExpression: "pk = :pk", 
      TableName: process.env.DYNAMODB_CUSTOMER_TABLE,
    };

    const res = await ddbDocClient.send(new QueryCommand(params));

    return {
      statusCode: 200,
      body: JSON.stringify(res.Items),
    };
  } catch (error) {
    console.error("Error al obtener usuario:", error);

    return {
      statusCode: 500,
      body: JSON.stringify({
        message: "Internal Server Error",
      }),
    };
  }
};

module.exports = {
  getUsers,
};
