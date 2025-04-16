const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const {
  DynamoDBDocumentClient,
  PutCommand,
} = require("@aws-sdk/lib-dynamodb");
const { randomUUID } = require("crypto"); // ← Importación correcta

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

const createUsers = async (event) => {
  try {
    const id = randomUUID(); // ← ejecución correcta

    let userBody = JSON.parse(event.body);
    userBody.pk = id;

    const params = {
      TableName: process.env.DYNAMODB_CUSTOMER_TABLE,
      Item: userBody,
    };

    console.log("Saving item:", params.Item);

    await ddbDocClient.send(new PutCommand(params)); // ← usar PutCommand

    return {
      statusCode: 201,
      body: JSON.stringify({
        message: "User created successfully",
        //userId: id,
        data: userBody,
      }),
    };
  } catch (error) {
    console.error("Error al crear usuario:", error);

    return {
      statusCode: 500,
      body: JSON.stringify({
        message: "Internal Server Error",
        error: error.message,
      }),
    };
  }
};

module.exports = {
  createUsers,
};
