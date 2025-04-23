// authorizer.js
exports.handler = async (event) => {
    const apiKey = event.headers['x-api-key'];
    const validApiKey = 'mi-api-key-secreta';
  
    if (apiKey === validApiKey) {
      return {
        isAuthorized: true,
      };
    } else {
      return {
        isAuthorized: false,
      };
    }
  };
  