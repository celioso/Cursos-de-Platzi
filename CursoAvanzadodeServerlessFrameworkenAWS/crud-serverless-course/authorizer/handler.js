const authorize = async (event, context) => {
    const date = new Date();
    const minutes = date.getMinutes();
    const hour = date.getHours();

    // Genera los posibles tokens válidos (tolerancia de ±1 minuto)
    const validTokens = [
        `Bearer ${process.env.SECRET_EGG}-${hour}-${minutes}`,
        `Bearer ${process.env.SECRET_EGG}-${hour}-${minutes - 1}`,
        `Bearer ${process.env.SECRET_EGG}-${hour}-${minutes + 1}`
    ];

    // Token recibido (normalizado)
    const receivedToken = event.authorizationToken?.trim();

    // Logs para depuración
    console.log("Received token:", receivedToken);
    console.log("Valid tokens:", validTokens);

    if (validTokens.includes(receivedToken)) {
        return {
            principalId: 'anonymous',
            policyDocument: {
                Version: '2012-10-17',
                Statement: [
                    {
                        Action: 'execute-api:Invoke',
                        Effect: 'Allow',
                        Resource: event.methodArn,
                    },
                ],
            },
        };
    }

    // Si el token no es válido
    throw new Error('Unauthorized');
};

module.exports = { authorize };
