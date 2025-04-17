const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const {
  DynamoDBDocumentClient,
  PutCommand,
} = require("@aws-sdk/lib-dynamodb");
const { randomUUID } = require("crypto");
const Joi = require("joi"); // ← Importamos Joi

// Definimos el schema de validación
const userSchema = Joi.object({
  name: Joi.string()
    .pattern(/^[a-zA-ZÀ-ÿ\s]+$/)
    .min(3)
    .max(50)
    .required()
    .messages({
      "string.min": "El nombre debe tener al menos 3 caracteres.",
      "string.max": "El nombre no puede tener más de 50 caracteres.",
      "string.empty": "El nombre es requerido.",
      "string.pattern.base": "El nombre solo puede contener letras y espacios.",
    }),
  email: Joi.string()
    .pattern(/^[\w.-]+@[a-zA-Z\d.-]+\.[a-zA-Z]{2,}$/)
    .email()
    .required()
    .messages({
      "string.email": "El correo electrónico no es válido.",
      "string.empty": "El correo electrónico es requerido.",
      "string.pattern.base": "La estructura del correo no es valida.",
    }),
  age: Joi.number()
    .integer()
    .min(18)
    .max(99)
    .required()
    .messages({
      "number.min": "La edad mínima es 18.",
      "number.max": "La edad máxima es 99.",
      "number.empty": "La edad es requerida.",
    }),
});

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
    const id = randomUUID();
    let userBody = JSON.parse(event.body);

    // Validamos el cuerpo con el schema
    const { error, value } = userSchema.validate(userBody);
    if (error) {
      return {
        statusCode: 400,
        body: JSON.stringify({
          message: "Error de validación",
          detalles: error.details.map((d) => d.message),
        }),
      };
    }

    // Agregamos el id como pk si todo es válido
    userBody = value;
    userBody.pk = id;

    const params = {
      TableName: process.env.DYNAMODB_CUSTOMER_TABLE,
      Item: userBody,
    };

    console.log("Saving item:", params.Item);

    await ddbDocClient.send(new PutCommand(params));

    return {
      statusCode: 201,
      body: JSON.stringify({
        message: "User created successfully",
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