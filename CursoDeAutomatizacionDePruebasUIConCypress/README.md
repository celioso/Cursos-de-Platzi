# Curso de Automatización de Pruebas UI con Cypress

##¿Qué es Cypress?

Es una librería, pensada para englobar o ser framework de pruebas automatizadas desde pruebas e2e, unit test e integration test. Ya trae todo lo necesario para que solamente nos preocupemos para realizar las pruebas.

### Features
**Time travel**
Nos permite viajar en el tiempo, podemos ver nuestra prueba, ver como fue la ejecución de cada paso.

**Bullet debuggability**
Facilita el trabajo a la hora de debuggear el proyecto.

**Automatic waiting**
Espera automáticamente por los elementos, no tenemos que estar preocupados por esperar los elementos en pantalla y que estén listos para la interacción.

**Spies, stubs and clocks**
Útiles para las pruebas unitarias para realizar mock en funciones, espiar en ellas, ver la ejecución de las mismas (caja blanca).

**Network traffic control**
Dentro del time travel vamos a poder ver que peticiones a la red se fueron realizando (network request), que sucede a la hora de realizar la petición, el estado de la petición, etc.

**Resultados consitentes**
Cypress ataca o lucha contra los flaky test, cuando la falla es aleatoria.

Capturas de pantalla y videos
Estas son proporcionadas de forma automática.

**Trade-offs**
Restricciones en la automatización
Cypress está pensado para las pruebas, no es una herramienta de automatización en general como Puppeteer. Cypress está enfocado directamente en pruebas, no podemos hacer webscriping y otras cosas que no están relacionas con pruebas.

**Corre dentro del navegador**
Todo el entorno de Cypress se ejecuta en el navegador, lo que provoca que no sea tan fácil ejecutar código como ser las librerías de Node.

**Multi tabs**
No se puede manejar multitabs.

Múltiples navegadores al mismo tiempo
No podemos usar múltiples navegadores a la vez, por ejemplo, no podemos abrir Chrome y Opera al mismo tiempo.

**Same-origin**
Es una política de seguridad, no podemos visitar diferentes dominios a la vez en la misma prueba.

### Cypress vs Selenium
**Cypress**
- Soporte a navegadores basados en Chromium.
- Soporta JS y TS.
- Tiene una capa gratuita y una de pago.
- Reportes directos.
- Ligeramente más rápido que Selenium.
- Ataca mejor los flaky tests.
- Curva de aprendizaje corta.

**Selenium**
- Soporte a muchos navegadores.
- Soporta múltiples lenguajes más alla de JS.
- Gratis.
- Los reportes hay que integrarlos de forma manual.

### Conclusión

Todo depende el contexto del proyecto y del stack que estamos usando, basándose en eso usaremos uno u otro.

### Preparando nuestro ambiente y entendiendo la estructura de Cypress

Preparando nuestro ambiente: **Instalar o inicializar node**


`npm init -y`

crea el siguiente json

```json
{
  "name": "proyecto",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC"
}
```

creamos el archivo **.prettierrc** y escribimos el siguiente json

```json
{
    "semi":false,
    "singleQuote":true,
    "bracketSpacing": true,
    "useTabs":true,
    "tabWidth":4,
    "trailingComma": "es5"
}
```

Instalar cypress y de manera opcional prettier

`npm i -D cypress prettier`

Abrir Cypress

`npx cypress open`

otra opción es modificar el package.json en test con el siguiente código.

`"test": "cypress open"`

y lo iniciamos con:

`npm run test`

## Nuestra primera prueba

Iniciar Cypress
`npm run test`

para ignorar los archivos de prueba que trae por defecto, se debe editar el archivo `cypress.config.js` en la versión v13.9.0
como se muestra en el siguiente código.

```json
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
    "viewportHeight":1080
  },
});
```
viewportWidth es para el tamaño d ela ventan.

Creamos un archivo llamado `primerPrueba.spec.cy.js` en las carpeta **cypress** luego en la carpeta **e2e**

el archivo `primerPrueba.spec.cy.js` lleva el siguiente código.

```json
describe('Primer Suite de Pruebas', () => {
        it('primer prueba', () => {
            cy.visit('https://platzi.com');
        });
        it('segunda prueba', () => {
            cy.visit('https://platzi.com');
        });
        it('tercera prueba', () => {
            cy.visit('https://platzi.com');
        });
```

## Navegación

para este punto se utilizan los siguientes comandos:
**cy.reload()**: que es para regresar la página.
**cy.go("back") o cy.go(-1):** para regresar a la anterior página.
**cy.go("forward") o cy.go(1):** para avanzar a la siguiente página.

```json
describe('Prueba de navegacion', () => {

    it('Navegando a un sitio', () => {
        cy.visit('https://platzi.com');
    });


    it('Recargar una pagina', () => {
        cy.reload()
    });

    it('Force reload  una pagina', () => {
        cy.visit('https://google.com');
        cy.reload(true)
    });

    it('Navegar hacia atras en una pagina', () => {
        cy.visit('https://google.com');
        cy.visit('https://www.google.com/search?q=platzi&sxsrf=APq-WBsJmYoDdRVdbT5vkzyA6INN9o-OoA%3A1645072295957&source=hp&ei=p88NYtzpNpauytMPo56H6Aw&iflsig=AHkkrS4AAAAAYg3dt-lyynY6DU3aZCGsxCJKBESc0ZTy&ved=0ahUKEwic2c7u84X2AhUWl3IEHSPPAc0Q4dUDCAY&uact=5&oq=platzi&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECcyDgguEIAEELEDEMcBENEDMggIABCABBCxAzILCC4QgAQQxwEQrwEyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6BwgjEOoCECc6CwguEIAEELEDEIMBOggIABCxAxCDAToLCAAQgAQQsQMQgwE6CAguEIAEELEDOgYIIxAnEBM6BAgAEEM6BwgAELEDEEM6BwgAEMkDEEM6CgguEMcBEKMCEEM6DgguEIAEELEDEIMBENQCULcEWNgNYKYQaAFwAHgAgAGAAYgBxgWSAQMwLjaYAQCgAQGwAQo&sclient=gws-wiz');
        cy.go('back')
        // cy.go(-1)
    });

    it('Navegar hacia adelante en una pagina', () => {
        cy.visit('https://google.com');
        cy.visit('https://www.google.com/search?q=platzi&sxsrf=APq-WBsJmYoDdRVdbT5vkzyA6INN9o-OoA%3A1645072295957&source=hp&ei=p88NYtzpNpauytMPo56H6Aw&iflsig=AHkkrS4AAAAAYg3dt-lyynY6DU3aZCGsxCJKBESc0ZTy&ved=0ahUKEwic2c7u84X2AhUWl3IEHSPPAc0Q4dUDCAY&uact=5&oq=platzi&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECcyDgguEIAEELEDEMcBENEDMggIABCABBCxAzILCC4QgAQQxwEQrwEyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6BwgjEOoCECc6CwguEIAEELEDEIMBOggIABCxAxCDAToLCAAQgAQQsQMQgwE6CAguEIAEELEDOgYIIxAnEBM6BAgAEEM6BwgAELEDEEM6BwgAEMkDEEM6CgguEMcBEKMCEEM6DgguEIAEELEDEIMBENQCULcEWNgNYKYQaAFwAHgAgAGAAYgBxgWSAQMwLjaYAQCgAQGwAQo&sclient=gws-wiz');
        cy.go('back')
        cy.go('forward')
        // cy.go(1)
    });


    it('Navegar hacia adelante en una pagina con chain command', () => {
        cy.visit('https://google.com');
        cy.visit('https://www.google.com/search?q=platzi&sxsrf=APq-WBsJmYoDdRVdbT5vkzyA6INN9o-OoA%3A1645072295957&source=hp&ei=p88NYtzpNpauytMPo56H6Aw&iflsig=AHkkrS4AAAAAYg3dt-lyynY6DU3aZCGsxCJKBESc0ZTy&ved=0ahUKEwic2c7u84X2AhUWl3IEHSPPAc0Q4dUDCAY&uact=5&oq=platzi&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECcyDgguEIAEELEDEMcBENEDMggIABCABBCxAzILCC4QgAQQxwEQrwEyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6BwgjEOoCECc6CwguEIAEELEDEIMBOggIABCxAxCDAToLCAAQgAQQsQMQgwE6CAguEIAEELEDOgYIIxAnEBM6BAgAEEM6BwgAELEDEEM6BwgAEMkDEEM6CgguEMcBEKMCEEM6DgguEIAEELEDEIMBENQCULcEWNgNYKYQaAFwAHgAgAGAAYgBxgWSAQMwLjaYAQCgAQGwAQo&sclient=gws-wiz').go('back').go('forward')
    });
})
```

## Tipos de localizadores
primero colocamos el siguiente código para corregir el error de consola.

```javascript
Cypress.on('uncaught:exception', (err, runnable) => {
    // Registrar el error en la consola
    console.error('Excepción no capturada', err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
  });
```

Empezamos identificando los tag que tiene  la página.

```javascript
describe('Probando configuracion', () => {

    it('Obteniendo por un tag', () => {
        cy.visit('/automation-practice-form')
        cy.get("input")    
    });
```

El siguiente código es para identificar un tag,  atributo y un id.

```javascript
 it('Obteniendo por un de un atributo', () => {
        cy.visit('/automation-practice-form')
        cy.get('[placeholder="First Name"]')
        
    });

    it('Obteniendo por un de un atributo y un tag', () => {
        cy.visit('/automation-practice-form')
        cy.get('input[placeholder="First Name"]')
        
    });

    it('Obteniendo por un de un id', () => {
        cy.visit('/automation-practice-form')
        cy.get("#firstName")
        
    });

    it('Obteniendo por un de un class', () => {
        cy.visit('/automation-practice-form')
        cy.get(".mr-sm-2.form-control")
        
    });
```

## Encontrando elementos

Yo estoy utilizando Cypress versión 13 con Visual estudio Code, me funciona de la siguiente manera

```javascript
  it( "Usando Contains",  () =>{         

//Para encntrar un elemento q contenga cierto texto           

// cy.visit('/automation-practice-form')         

// cy.get('input[placeholder="First Name"]')       

cy.contains('Reading')       

cy.get('.header-wrapper').contains('Widgets')
```

Selector |Recommended | Notes
--------|-----------------|-------
cy.get('button').click() | Never | Worst - too generic, no context.
cy.get('.btn.btn-large').click() | Never | Bad. Coupled to styling. Highly subject to change.
cy.get('#main').click() | Sparingly | Better. But still coupled to styling or JS event listeners.
cy.get('[name="submission"]').click() |  Sparingly | Coupled to the name attribute which has HTML semantics.
cy.contains('Submit').click()  | Depends | Much better. But still coupled to text content that may change.
cy.get('[data-cy="submit"]').click() |  Always | Best. Isolated from all changes.

Es otra forma de utilizar para buscar ciertos elementos. Cómo contains y elementos parent como se muestra en el siguiente código.

```javascript
it('usando Contains', () => {
        cy.visit('/automation-practice-form')
        cy.contains("Reading")
        cy.contains(".header-wrapper","Widget")
        
    });

    it('usando parent', () => {
        cy.visit('/automation-practice-form')
        //Obteniendo el elemento el padre
        cy.get('input[placeholder="First Name"]').parent()
        //Obteniendo el elemento el padres
        cy.get('input[placeholder="First Name"]').parents()

        cy.get('input[placeholder="First Name"]').parents().find("label")
        
        cy.get("form").find("label")
    });
```

### Guardando elementos

Yields
[📚 Documentation](https://docs.cypress.io/api/commands/then#Yields "📚 Documentation") . Los "Yields" son punteros producidos mediante referencias por `.then`. Dichas referencias son modeladas idénticamente como Promesas en JavaScript, el resultado obtenido como retorno de un `then` es llamado como *Yield*. 

`cy.get('.nav').then((nav) => {})`

- Dentro de un función callback, tendremos clousers que permite manipular la referencias con el propósito de manipular valores o realizar algunas acciones. . En cuyo caso que se desea cambiar la operación a comandos de Cypress, utilizamos `.wrap`. [📚 Documentación](https://docs.cypress.io/api/commands/wrap "📚 Documentación")

```javascript
cy.wrap(1)
  .then((num) => {
    cy.wrap(num).should('equal', 1) // true
  })
  .should('equal', 1) // true
```

- Adicionalmente, con wrap podemos referencial:

- **Objetos**

```javascript
const getName = () => {
  return 'Jane Lane'
}

cy.wrap({ name: getName }).invoke('name').should('eq', 'Jane Lane')
```

- **Elementos**

```javascript
cy.get('form').within((form) => {
  cy.wrap(form).should('have.class', 'form-container')
})
```

- **Promesas como eventos**

```javascript
const myPromise = new Promise((resolve, reject) => {
  setTimeout(() => {
    resolve({
      type: 'success',
      message: 'It worked!',
    })
  }, 2500)
})

it('should wait for promises to resolve', () => {
  cy.wrap(myPromise).its('message').should('eq', 'It worked!')
})
```

## Aserciones

### TDD VS BDD

Cuando nos referimos a una **aserción**, nos estamos refiriendo a un hecho o verdad esperado y comprobado. . **TDD** o **Test Driven Development**, es un proceso de escritura de pruebas con la intención de especificar una funcionalidad. . Por su parte, **BDD** o **Behavior Driven Development**, es un proceso de escritura de pruebas con la intención de especificar una característica. . Ambas, son procesos complementarios cuya principal diferencia es el alcance. En concreto, **TDD** es una práctica de desarrollo mientras **BDD** es una metodología de aplicación, es decir, mientras una es escrita por desarrolladores, la otra es especifica por requerimientos de usuarios, respectivamente. 

**Estilos de pruebas**

Con Cypress, ya que incorpora librerías como Chai o extensiones para Sinon o jQuery, podemos describir pruebas mediante **BDD** (`expect/ should`) y/o mediante **TDD** (`assert`).

```javascript
Cypress.on('uncaught:exception', (err, runnable) => {
    // Registrar el error en la consola
    console.error('Excepción no capturada', err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
  });

describe("Aserciones", () => {

    it('Asercion', () => {
        cy.visit('/automation-practice-form')
        cy.url().should("include", "demoqa.com")
        cy.get("#firstName").should("be.visible").and("have.attr", "placeholder", "First Name")
    });

    it('Asercion 2', () => {
        cy.visit('/automation-practice-form')
        cy.url().should("include", "demoqa.com")
        cy.get("#firstName").then((element)=>{
            expect(element).to.be.visible
            expect(element).to.have.attr("placeholder", "First Name")
    })
});

    it('Asercion 3', () => {
        cy.visit('/automation-practice-form')
        cy.url().should("include", "demoqa.com")
        cy.get("#firstName").then((element)=>{
            assert.equal(element.attr("placeholder"),"First Name")
    })
})

});
```

web de **cypress asserttions**

[cypress](https://docs.cypress.io/guides/references/assertions "cypress")