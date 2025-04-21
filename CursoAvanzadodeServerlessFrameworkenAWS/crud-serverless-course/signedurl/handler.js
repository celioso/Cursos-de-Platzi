const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');

const s3Client = new S3Client({ region: 'us-east-1' });

module.exports.signedS3URL = async (event) => {
  try {
    const bucketName = 'bucket-serverless-course-54963217'; // Reemplaza con el nombre de tu bucket
    const filename = event.queryStringParameters?.filename;

    if (!filename) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: 'El parámetro filename es requerido.' }),
      };
    }

    const command = new PutObjectCommand({ Bucket: bucketName, Key: `uploads/${filename}` });
    const signedUrl = await getSignedUrl(s3Client, command, { expiresIn: 300 }); // La URL expira en 5 minutos (300 segundos)

    return {
      statusCode: 200,
      body: JSON.stringify({ signedUrl }),
    };
  } catch (error) {
    console.error('Error al generar la URL firmada:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Error al generar la URL firmada.' }),
    };
  }
};