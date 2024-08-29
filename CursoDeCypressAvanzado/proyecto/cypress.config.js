const { defineConfig } = require("cypress");
const { addMatchImageSnapshotPlugin } = require("cypress-image-snapshot/plugin");
const webpack = require("@cypress/webpack-preprocessor");
const preprocessor = require("@badeball/cypress-cucumber-preprocessor");
const allureWriter = require('@shelex/cypress-allure-plugin/writer');
const { allureCypress } = require("allure-cypress/reporter");

const values = {};

async function setupNodeEvents(on, config) {

  addMatchImageSnapshotPlugin(on, config);

  config.env.variable = process.env.NODE_ENV ?? 'NO HAY VARIABLE'; // INGRESA  ALAS VAIABLES DE ENTORNO

  on("task", {
    guardar(valor) {
      const key = Object.keys(valor)[0];

      values[key] = valor[key];

      return null;
    },
    obtener(key) {
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

  //on("file:preprocessor", webpack);
  allureWriter(on, config);
  
  allureCypress(on, {
          resultsDir: "./allure-results",
        });
  return config;
};


module.exports = defineConfig({
  projectId: 'wg5uey',
  /*reporter: "cypress-multi-reporters",
  reporterOptions: {
    configFile: "reporter-config.json",
  },*/
  e2e: {
    baseUrl: "https://pokedexpokemon.netlify.app",
    //experimentalSessionAndOrigin: true, 
    retries: 2,
    specPattern: "**/*.feature",
    supportFile: false,
    setupNodeEvents,
    excludeSpecPattern: [
      "**/1-getting-started/*.js",
      "**/2-advanced-examples/*.js"
    ],
    env: {
      credentials: {
        user: "username",
        password: "password",
      },
    },
    
    //retries: 2,
    // retries: {
    //   // Configure retry attempts for `cypress run`
    //   // Default is 0
    //   runMode: 2,
    //   // Configure retry attempts for `cypress open`
    //   // Default is 0
    //   openMode: 0,
    // },
  },
});