

exports.hello = async (event) => {
  return {
    statusCode: 200,
    body: JSON.stringify({
      message: "Hola a todos es la prueba de funcionamiento",
    }),
  };
};

exports.helloUser = async (event) => {
  return {
    statusCode: 200,
    body: JSON.stringify({
      message: `Hola ${event.pathParameters.name} como va tú dia`,
    }),
  };
};

exports.createUser = async (event) => {
  const body = JSON.parse(event.body)
  return {
    statusCode: 200,
    body: JSON.stringify({
      message: "Petición para crear usuarios",
      input: `Hola ${body.user}`
    }),
  };
};
