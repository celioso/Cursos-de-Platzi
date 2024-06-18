const { defineConfig } = require("cypress");
const { addMatchImageSnapshotPlugin, } = require("cypress-image-snapshot/plugin");
const webpack = require("@cypress/webpack-preprocessor");
const preprocessor = require("@badeball/cypress-cucumber-preprocessor");

const values = {};

async function setupNodeEvents(on, config){

    addMatchImageSnapshotPlugin(on, config);

    config.env.variable=process.env.NODE_ENV ?? 'NO HAY VARIABLE'; // INGRESA  ALAS VAIABLES DE ENTORNO

    on("task", {
      guardar(valor){
        const key = Object.keys(valor)[0];

        values[key] = valor[key];

        return null;
      },
      obtener(key){
        console.log('values', values);
        return values[key] ?? "No hay valor";
      },
    });

    await preprocessor.addCucumberPreprocessorPlugin(on, config);
    
    on(
      "file:preprocessor",
      webpack({
        webpackOptions: {
          resolve: {
            extensions: [".ts", ".js"],
          },
          module: {
            rules: [
              {
                test: /\.feature$/,
                use: [
                  {
                    loader: "@badeball/cypress-cucumber-preprocessor/webpack",
                    options: config,
                  },
                ],
              },
            ],
          },
        },
      })
    );
    return config;
};

module.exports = defineConfig({
  e2e: {
    baseUrl: "https://pokedexpokemon.netlify.app",
    experimentalSessionAndOrigin: true,
    specPattern: "**/*.feature",
    supportFile: false,
    setupNodeEvents,
    /*setupNodeEvents(on, config) {
      // implement node event listeners here
      addMatchImageSnapshotPlugin(on, config)
      config.env.variable=process.env.NODE_ENV ?? 'NO HAY VARIABLE'; // INGRESA  ALAS VAIABLES DE ENTORNO

      on("task", {
        guardar(valor){
          const key = Object.keys(valor)[0];

          values[key] = valor[key];

          return null;
        },
        obtener(key){
          console.log('values', values);
          return values[key] ?? "No hay valor";
        },
      });
      return config;
    },*/
    
    excludeSpecPattern:[
      "**/1-getting-started/*.js",
      "**/2-advanced-examples/*.js"
    ],
    //retries:2,
    /*retries: {
        runMode:2,
        openMode: 0
    },*/

    env:{
      credentials: {
        user: "username",
        password: "password",
      },
    },
    specPattern:"**/*.feature",
  },
});