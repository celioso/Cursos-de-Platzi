
const hello = async (event, context) => {
    return {
        "statusCode": 200,
        "body": JSON.stringify({ 'message': 'Creacion de otro serverless'})
    };
};

module.exports = {
    hello
};
