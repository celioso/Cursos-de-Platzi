const AWS = require('aws-sdk');
const Sharp = require('sharp');
const S3 = new AWS.S3();

exports.thumbnailGenerator = async (event, context) => {
  const sourceBucket = event.Records[0].s3.bucket.name;
  const key = decodeURIComponent(event.Records[0].s3.object.key.replace(/\+/g, ' '));

  const imageObject = await S3.getObject({ Bucket: sourceBucket, Key: key }).promise();
  const imageBody = imageObject.Body;

  await resizeImage(imageBody, 50, 50);
  await resizeImage(imageBody, 100, 100);
  await resizeImage(imageBody, 200, 200);
};

const resizeImage = async (imageBody, width, height) => {
  const buffer = await Sharp(imageBody)
    .resize(width, height)
    .toBuffer();

  await S3.putObject({
    Bucket: 'processed-images-bucket',
    Key: `resized-${width}x${height}-${Date.now()}.jpg`,
    Body: buffer
  }).promise();
};