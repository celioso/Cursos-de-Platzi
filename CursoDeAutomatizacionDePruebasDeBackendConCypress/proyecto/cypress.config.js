const { defineConfig } = require("cypress");
const mysql = require('mysql2/promise');
const { MongoClient } = require('mongodb');

const url= 'mongodb://localhost:27017/';

async function queryTestDb(query) {
    const connection = await mysql.createConnection({
        host: 'localhost', // Cambia esto según tu configuración
        user: 'root',      // Cambia esto según tu configuración
        password: '*******',  // Cambia esto según tu configuración
        database: 'pruebas', // Cambia esto según tu configuración
    });

    const [results] = await connection.execute(query);
    return results;
};

// Empresa es el nombre de la base de datos y Nombres es la colección.

// Función para obtener listado de documentos en la colección Nombres

async function getListing() {
  const client = new MongoClient(url, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });
  console.log('Base de datos conectada')

  try {
    await client.connect();
    console.log("Connected to MongoDB");
    const db = client.db('Empresa'); // Nombre de tu base de datos
    const Nombres = db.collection('Nombres'); // Nombre de tu colección
    const result = await Nombres.find({}).limit(50).toArray();
    return result;
  } catch (e) {
    console.error("Error connecting to MongoDB", e);
    return [];
  } finally {
    await client.close();
    console.log("Disconnected from MongoDB")
  }
}

// Función para crear un documento de prueba en la colección Nombres

// Función para crear un documento de prueba en la colección Nombres
async function createName(name) {
  const client = new MongoClient(url, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });

  try {
    await client.connect();
    const db = client.db('Empresa'); // Nombre de tu base de datos
    const Nombres = db.collection('Nombres'); // Nombre de tu colección
    return await Nombres.insertOne(name);
  } catch (e) {
    console.error("Error connecting to MongoDB", e);
    return [];
  } finally {
    await client.close();
    console.log("Disconnected from MongoDB")
  }
}

// Función para limpiar la colección Nombres (eliminar todos los documentos)
async function clearNombres() {
  const client = new MongoClient(url, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });

  try {
    await client.connect();
    const db = client.db('Empresa'); // Nombre de tu base de datos
    const Nombres = db.collection('Nombres'); // Nombre de tu colección
    return await Nombres.deleteMany({});
  } catch (e) {
    console.error("Error connecting to MongoDB", e);
    return [];
  } finally {
    await client.close();
    console.log("Disconnected from MongoDB")
  }
}

module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      on('task', {
        queryDb: queryTestDb,
        getListing: getListing,
        createName: createName,
        clearNombres: clearNombres

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