const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const {
  DynamoDBDocumentClient,
  UpdateCommand,
  GetCommand,
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

const updateUsers = async (event) => {
  try {
    const userId = event.pathParameters.id;
    const body = JSON.parse(event.body);

    // Verificar si el usuario existe
    const getUserParams = {
      TableName: process.env.DYNAMODB_CUSTOMER_TABLE,
      Key: { pk: userId },
    };

    const userResult = await ddbDocClient.send(new GetCommand(getUserParams));

    if (!userResult.Item) {
      return {
        statusCode: 404,
        body: JSON.stringify({ message: "User not found" }),
      };
    }

    // Construir dinámicamente la expresión de actualización esto es nuevo 
    const updateExpressions = [];
    const expressionAttributeNames = {};
    const expressionAttributeValues = {};

    for (const key in body) {
      updateExpressions.push(`#${key} = :${key}`);
      expressionAttributeNames[`#${key}`] = key;
      expressionAttributeValues[`:${key}`] = body[key];
    }

    // llega hasta aqui

    // Si existe, se actualiza
    const updateParams = {
      TableName: process.env.DYNAMODB_CUSTOMER_TABLE,
      Key: { pk: userId },
      /*UpdateExpression: "set #name = :name", del codigo de la clase
      ExpressionAttributeNames: { "#name": "name" },
      ExpressionAttributeValues: { ":name": body.name },
      ReturnValues: "ALL_NEW",*/
      UpdateExpression: `SET ${updateExpressions.join(", ")}`,
      ExpressionAttributeNames: expressionAttributeNames,
      ExpressionAttributeValues: expressionAttributeValues,
      ReturnValues: "ALL_NEW",
    };

    const res = await ddbDocClient.send(new UpdateCommand(updateParams));

    return {
      statusCode: 200,
      body: JSON.stringify(res.Attributes),
    };
  } catch (error) {
    console.error("Error al actualizar usuario:", error);

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
  updateUsers,
};