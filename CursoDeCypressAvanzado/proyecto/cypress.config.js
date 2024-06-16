const { defineConfig } = require("cypress");
const { addMatchImageSnapshotPlugin, } = require("cypress-image-snapshot/plugin");

module.exports = defineConfig({
  e2e: {
    baseUrl: "https://pokedexpokemon.netlify.app",
    experimentalSessionAndOrigin: true,
    setupNodeEvents(on, config) {
      // implement node event listeners here
      addMatchImageSnapshotPlugin(on, config)
      config.env.variable=process.env.NODE_ENV ?? 'NO HAY VARIABLE'; // INGRESA  ALAS VAIABLES DE ENTORNO
      return config;
    },
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

  },
});