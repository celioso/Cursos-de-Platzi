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

## Local Storage

```javascript
describe('LocalStorage', ()=>{
    beforeEach(()=>{
        //cy.visit('https://todo-cypress-iota.vercel.app/');
        //cy.get("#title").type("Titulo de prueba");
        //cy.get("#description").type("Descripción de prueba");
        //cy.contains('Create').click();
       
        
        cy.session("session todo", ()=>{
            cy.visit('https://todo-cypress-iota.vercel.app/').then(()=>{
                localStorage.setItem("react_todo_ids", JSON.stringify(["Titulo de prueba"]));
                localStorage.setItem("Titulo de prueba", JSON.stringify({
                    title: "Titulo de prueba",
                    id: "Titulo de prueba",
                    complete: false,
                    description:"Descripción de una prueba"
                    })
                );
            });    
        });
        cy.visit('https://todo-cypress-iota.vercel.app/');
    });

    it('crear una tarea',()=>{
        /*cy.visit('https://todo-cypress-iota.vercel.app/');
        cy.get("#title").type("Titulo de prueba");
        cy.get("#description").type("Descripción de prueba");
        cy.contains('Create').click();*/

        cy.contains("Titulo de prueba");

        cy.reload();

        cy.contains("Titulo de prueba").then(()=>{
            expect(localStorage.getItem("Titulo de prueba")).to.exist;
        });

        cy.contains("Remove").click().then(()=>{
            expect(localStorage.getItem("Titulo de prueba")).to.not.exist;
        });
        /*cy.clearLocalStorage("Titulo de prueba").should((ls) => {
        expect(ls.getItem("prop1")).to.be.null;
    });*/
    });

    it('valido que la tarea se crea correectamente', ()=>{
        //cy.visit('https://todo-cypress-iota.vercel.app/');
        expect(localStorage.getItem("Titulo de prueba")).to.exist;
    })
});
```

Les dejo el siguiente link por si quieren más información acerca de que es y como función el LOCAL STORAGE. [https://es.javascript.info/localstorage](https://es.javascript.info/localstorage)

[GitHub - platzi/curso-cypress-avanzado at 3era-clase-localstorage](https://github.com/platzi/curso-cypress-avanzado/tree/3era-clase-localstorage)

## Emulando dispositivos

```javascript
const dispositivos =[
    {viewport: "macbook-15", type: "desktop"},
    {viewport: "ipad-2", type: "mobile"},
    {viewport: [1280, 720], type: "desktop"},
    {viewport: [375, 667], type: "mobile"},
];

describe('Dispositivos moviles', ()=>{

    /*it('Usando el viewport',()=>{
        cy.viewport(1280, 720);
        cy.visit('/');
        cy.contains('Safari').should("exist");
    });

    it('Usando el viewport movil',()=>{
        cy.viewport(375, 667);
        cy.visit('/');
        cy.contains('Safari').should("not.be.visible");
    });

    it('Usando el viewport desktop preset',()=>{
        cy.viewport("macbook-15");
        cy.visit('/');
        cy.contains('Safari').should("exist");
    });

    it('Usando el viewport movil preset',()=>{
        cy.viewport("iphone-6+");
        cy.visit('/');
        cy.contains('Safari').should("exist");
    });*/

    dispositivos.forEach(device=>{
        it(`Prueba con el viewport ${device.viewport}`, () => {
            if (Cypress._.isArray(device.viewport)) {
              cy.viewport(device.viewport[0], device.viewport[1]);
            } else {
              cy.viewport(device.viewport);
            }
            cy.visit('/');

            if(device.type === 'desktop') {
                cy.contains("Safari").should("exist");
            }else{
                cy.contains("Safari").should("not.be.visible");
            }
        });
    });
});
```

[GitHub - platzi/curso-cypress-avanzado at 4ta-clase-emulando-dispositivos](https://platzi.com/new-home/clases/4760-cypress-avanzado/57259-emulando-dispositivos/#:~:text=GitHub%20%2D%20platzi/curso%2Dcypress%2Davanzado%20at%204ta%2Dclase%2Demulando%2Ddispositivos)

## Usando xpaths

Se instala le plugin `npm i -D cypress-xpath` y si hay algun problema se usa `npm install -D cypress-xpath`.

para que funcione, se debe ir al archivo ``e2e.js`` que se encuentra en la carpeta **support** **cypress\support**

```javascript
// Import commands.js using ES2015 syntax:
import './commands'
import 'cypress-xpath'

// Alternatively you can use CommonJS syntax:
// require('./commands')
```

[cypress-xpath - npm](https://www.npmjs.com/package/cypress-xpath)

[GitHub - platzi/curso-cypress-avanzado at 5ta-clase-plugins-xpath](https://github.com/platzi/curso-cypress-avanzado/tree/5ta-clase-plugins-xpath)