const aws = require('aws-sdk');

let dynamoDBClienteParams = {}

if (process.env.IS_OFFLINE){
    dynamoDBClienteParams = {
        region: 'localhost',
        endpoint: 'http://localhost:8001',
        accessKeyId: 'DEFAULT_ACCESS_KEY',
        secretAccessKey: 'DEFAULT_SECRET'}
}

const dynamodb = new aws.DynamoDB.DocumentClient(dynamoDBClienteParams);

const getUsers = async (event, context) => {

    let userId = event.pathParameters.id

    var params = {
        ExpressionAttributeValues: { ':pk': userId},
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
