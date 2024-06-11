const { defineConfig } = require("cypress");
const mysql = require('mysql2/promise');

async function queryTestDb(query) {
    const connection = await mysql.createConnection({
        host: 'localhost', // Cambia esto según tu configuración
        user: 'root',      // Cambia esto según tu configuración
        password: 'F22Raptor@',  // Cambia esto según tu configuración
        database: 'pruebas', // Cambia esto según tu configuración
    });

    const [results] = await connection.execute(query);
    return results;
}

module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      on('task', {
        queryDb: queryTestDb,
    });

    return config;
    
    },
      // ignore los archivos
    excludeSpecPattern:[
      "**/1-getting-started/*.js",
      "**/2-advanced-examples/*.js"
    ],

    baseUrl:"http://localhost:3000/"
  },

  
});
