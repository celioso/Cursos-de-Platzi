const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
    excludeSpecPattern: [
      "cypress/e2e/1-getting-started/",
      "cypress/e2e/2-advanced-examples/"
    ],
    "viewportWidth":1920,
    "viewportHeight":1080,
    "video": false,  //true activa la creacion del video
    "baseUrl": "https://demoqa.com", //para colocar la página de base
    setupNodeEvents(on, config) {
      on('task', {
        log(message){
            console.log(`Soy el console log del task ${message}`)
            return null
        }
    })
    }
  },
});
