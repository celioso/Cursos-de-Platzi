const AWS = require('aws-sdk');

const isOffline = process.env.IS_OFFLINE;

const options = isOffline
    ? {
        region: 'localhost',
        endpoint: 'http://localhost:8000',
        accessKeyId: 'fakeMyKeyId',
        secretAccessKey: 'fakeSecretAccessKey',
        }
    : {};

const dynamoDb = new AWS.DynamoDB.DocumentClient(options);

const getUsers = async (event, context) => {
    var params = {
        ExpressionAttributeValues: { ':pk':'1'},
        KeyConditionExpression: "pk=:pk",
        TableName: "UsersTable",
    };

    return dynamodb.query(params).promise().then(res => {
        console.log(res)
        return {
            "statusCode": 200,
            "body": JSON.stringify({ 'user': res})
        };
    });
};

module.exports = {
    getUsers
};
