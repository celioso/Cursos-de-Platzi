const AWS = require('aws-sdk');
const s3 = new AWS.S3({
  signatureVersion: 'v4',
});

exports.handler = async (event) => {
  // Obtener el nombre del archivo desde el query string
  const filename = event.queryStringParameters.filename;

  // Generar la URL firmada para subir el archivo al bucket
  const signedURL = s3.getSignedUrl('putObject', {
    Bucket: process.env.BUCKET, // Aquí el bucket se toma del entorno
    Key: `upload/${filename}`,  // Ruta dentro del bucket
    Expires: 300,               // URL expira en 5 minutos
  });
  
  return {
    statusCode: 200,
    body: signedURL,
  };
};
