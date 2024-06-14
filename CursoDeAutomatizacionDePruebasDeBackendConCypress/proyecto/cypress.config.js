const { defineConfig } = require("cypress");
const mysql = require('mysql2/promise');
const { MongoClient } = require('mongodb');

url= 'mongodb+srv://celioso1:<password>@cluster0.edssdgd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';

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

  try {
    await client.connect();
    const db = client.db('Empresa'); // Nombre de tu base de datos
    const Nombres = db.collection('Nombres'); // Nombre de tu colección
    const result = await Nombres.find({}).limit(50).toArray();
    return result;
  } catch (e) {
    console.error(e);
    return [];
  } finally {
    await client.close();
  }
}

// Función para crear un documento de prueba en la colección Nombres

// Función para crear un documento de prueba en la colección Nombres
async function createPrueba(prueba) {
  const client = new MongoClient(url, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });

  try {
    await client.connect();
    const db = client.db('Empresa'); // Nombre de tu base de datos
    const Nombres = db.collection('Nombres'); // Nombre de tu colección
    return await Nombres.insertOne(prueba);
  } catch (e) {
    console.error(e);
    return [];
  } finally {
    await client.close();
  }
}

// Función para limpiar la colección Nombres (eliminar todos los documentos)
async function clearPrueba() {
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
    console.error(e);
    return [];
  } finally {
    await client.close();
  }
}


module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      on('task', {
        queryDb: queryTestDb,
        getListing: getListing,
        createPrueba: createPrueba,
        clearPrueba: clearPrueba
    });

    return config;
    
    },
      // ignore los archivos
    excludeSpecPattern:[
      "**/1-getting-started/*.js",
      "**/2-advanced-examples/*.js"
    ],

    //baseUrl:"http://localhost:3000/"
  },

  
});