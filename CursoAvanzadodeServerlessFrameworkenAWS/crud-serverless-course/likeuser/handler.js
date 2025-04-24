/*const likeuser = (event, context) => {
    console.log("hola mundo")
}

module.exports = {
    likeuser
}*/

/*'use strict';

module.exports.likeuser = async (event) => {
  try {
    // event Records contiene los mensajes de SQS
    const records = event.Records;
    for (const record of records) {
      // Procesar cada mensaje de SQS
      console.log('Processing SQS message:', record.body);

      // Aquí va la lógica para procesar cada mensaje
      const message = JSON.parse(record.body);
      
      // Ejemplo: Imprimir el contenido del mensaje
      console.log('User ID from message:', message.userId);
      
      // Lógica de procesamiento (por ejemplo, registrar la acción del "like")
      // Aquí podrías actualizar una base de datos, enviar un evento, etc.
      
      // Si hay un error, lanza una excepción para que Lambda intente nuevamente
    }

    return {
      statusCode: 200,
      body: JSON.stringify({
        message: 'Message processed successfully!',
      }),
    };
  } catch (error) {
    console.error('Error processing message:', error);
    // Si hay un error, Lambda volverá a procesar el mensaje
    throw new Error('Error processing the SQS message');
  }
};
*/

const AWS = require("aws-sdk");
const sqs = new AWS.SQS();

const QUEUE_URL = process.env.SQS_QUEUE_URL;

module.exports.sendToQueue = async (event) => {
  const body = JSON.parse(event.body || "{}");
  const params = {
    QueueUrl: QUEUE_URL,
    MessageBody: JSON.stringify(body),
  };

  await sqs.sendMessage(params).promise();

  return {
    statusCode: 200,
    body: JSON.stringify({ message: "Mensaje enviado a la cola SQS" }),
  };
};

module.exports.processQueue = async (event) => {
  for (const record of event.Records) {
    const message = JSON.parse(record.body);
    console.log("Mensaje recibido:", message);
  }

  return { statusCode: 200 };
};
