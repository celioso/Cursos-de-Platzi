# Curso de Cypress Avanzado

## Instalación de recursos

Comandos de instalacion:

- `npm init -y`
- `npm i cypress` o `npm install cypress --save-dev`
- `npm i -D prettier`
- `npx cypress open` o `npm run test`

Tambien para el autocompletado coloquen la siguiente linea de codigo en en archivo e2e.js


/// <reference types="cypress" />

## Cookies 

**cypress.config.js**

```javascript
const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    baseUrl: "https://pokedexpokemon.netlify.app",
    experimentalSessionAndOrigin: true,
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
    excludeSpecPattern:[
      "**/1-getting-started/*.js",
      "**/2-advanced-examples/*.js"
    ],
    
  },
});
```

**cookiesWithSeccion.cy.js**

```javascript
describe("Cookies", () => {
    //sustituye al  Cypress.Cookies.preserveOnce
    // before(() => {
    //   cy.session("login", () => {
    //     cy.visit("/");
    //     cy.setCookie("nombre", "Javier");
    //   });
    // });

    //sustituye al  Cypress.Cookies.default
    beforeEach(() => {
        cy.session("login", () => {
            cy.visit("/");
            cy.setCookie("nombre", "Javier");
        });
    });

    it("Obtener una cookie", () => {
      //se necesita visitar el visit(/) porque por defecto manda a una pagina en blanco
      // pero no es mas lento porque guarda la session
        cy.visit("/");
        cy.getCookies().should("have.length", 1);
    });

    it("Obtener una cookie en especifico", () => {
        cy.visit("/");
      //esto va a fallar porque cypress limpia las cookies entre tests
        cy.getCookie("nombre").should("have.property", "value", "Javier");
    });
});
```
**cookies.cy.js**

```javascript
describe('Cookies', function(){
    beforeEach(() => {
        cy.session("Cookies",() => {
            cy.setCookie('nombre', 'Mario');
        });
    });

    it('Obtener las cookies', () => {
        cy.clearAllCookies()
        cy.visit("/")
        cy.getCookies().should('be.empty')
    });

    it('Agregar una cookie', () => {
        cy.setCookie('nombre', 'Mario')
        cy.getCookies().should('have.length', 1)
    });

    it('Obtener cookie especifica', () => {
        cy.getCookie('nombre').should('have.a.property', "value", "Mario");
    });
});
```

[Why Cypress? | Cypress Documentation](https://docs.cypress.io/guides/overview/why-cypress)

[GitHub - platzi/curso-cypress-avanzado at 2da-clase-cookies](https://github.com/platzi/curso-cypress-avanzado/tree/2da-clase-cookies)