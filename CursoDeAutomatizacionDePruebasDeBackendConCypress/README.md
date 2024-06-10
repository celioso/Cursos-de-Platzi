# Curso de Automatización de Pruebas de Backend con Cypress

## Backend testing

![REST vs GraphQL](images/RESTVSGraphQL.png)

## Arquitectura REST

![REST API Model](images/api-rest-model.png)

REST / RESTful es un estilo arquitectónico para diseñar y desarrollar una API a través del protocolo HTTP. . En esencia, REST posee muchos beneficios:

- **Cliente-Servidor**: Una API que posee la arquitectura REST, implementa un estilo basado en un cliente que envía solicitudes de recursos, descentralizados o distribuidos en servicios, para obtener una respuesta de información.
- **Uniformidad**: REST presenta un factor clave para su escalabilidad, la interfaz uniforme que presentamos permite identificar recursos y envío de mensajes descriptivos mediante meta datos.
- **Capas**: De manera general, REST implementa una jerarquía en capas, restringiendo y administrando restricciones de comportamiento, delegando su construcciones a patrones de servidores como MVC o MVCS.

### Verbos HTTP:

**GET**: sirve para obtener informacion.
**POST**: crea un registro.
**PUT**: sirve para modificar, sobreescribe todas las propiedades.
**PATCH**: sirve para modificar, pero solo algunas propiedades, no todas.
**DELETE**: sirve para borrar un registro/objeto.
**API**: Application Public Interface, sirve para comunicar un backend/base de datos con alguna aplicacion externa o un front-end.

### Estandares de API:

- SOAP (XML): La response de esta api esta en formato XML
- REST (JSON): Se han vuelto una norma en la industria de software, la response es en formato JSON.

GraphQL, trabaja sobre REST pero nos permite potenciar nuestra API. Usa el verbo POST solamente pero nos permite extraer información que realmente necesitamos. Evita el sobre-fetch, que es traer datos que quizas nunca los estemos utilizando.

Backend Testing: se trata del proceso de probar en conjunto las operaciones y flujos que ocurren en el backend, desde el funcionamiento de las bases de datos como tambien las respuestas de APIs y la consistencia en la información que estan entregando.

### Herramientas del Navegador

¡Rock 'n Roll! 🤟 . Adicionalmente, a los recursos como lecturas recomendadas, les comparto los enlaces a la documentación:

[🔗 DevTools Documentation](https://developer.chrome.com/docs/devtools/ "🔗 DevTools Documentation")
[🔗 Postman Documentation](https://learning.postman.com/ "🔗 Postman Documentation")

## Preparando nuestro entorno de trabajo
1. iniciamos el proyecto
`npm init -y`

2. seguir con la instalaciión de la libreria.
`npm i -D cypress prettier json-server`

3. Abrir cypress:
`npx cypress open`

4. para ignorar los archivos de test de cypress se agrega el siguente codigo en **cypress.config.js** el siguienter codigo:

```javascript
const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      // implement node event listeners here
      
    },
      // ignore los archivos
    excludeSpecPattern:[
      "**/1-getting-started/*.js",
      "**/2-advanced-examples/*.js"
    ]
  },
});
```
5. se crea el archivo **db.json** con el codigo:
```json
{
    "employees": [
    {
        "id": 1,
        "first_name": "Javier",
        "last_name": "Eschweiler",
        "email": "javier@platzi.com"
    },
    {
        "id": 2,
        "first_name": "Juan",
        "last_name": "Palmer",
        "email": "juan@platzi.com"
    },
    {
        "id": 3,
        "first_name": "Ana",
        "last_name": "Smith",
        "email": "ana@platzi.com"
    }
    ]
}
```

6. se crea el test en **package.json** en *scripts*:

```json
"scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "run:server":"json-server --watch db.json"
  },
```

7. iniciamos nuestro servidor:

`npm run run:server`


[GitHub - javierfuentesm/CypressBackendPlatzi](https://github.com/javierfuentesm/CypressBackendPlatzi "GitHub - javierfuentesm/CypressBackendPlatzi")

[GitHub - javierfuentesm/CypressBackendPlatzi at preparando-ambiente](https://github.com/javierfuentesm/CypressBackendPlatzi/tree/preparando-ambiente "GitHub - javierfuentesm/CypressBackendPlatzi at preparando-ambiente")

## Probando el header

```javascript
describe('Probando Headers de la Api',()=>{
    it("Validar el Header y el content type", ()=>{
        cy.request('employees').its('headers').its('content-type').should('include','application/json')
    });
});
```

## Probando el status code

```javascript
describe('Probando statuses', ()=>{

    it('Debe de validar el status code exitoso',()=>{

        cy.request('employees')
        .its('status')
        .should('eq',  200)
    });

    it('Debe de validar el status code fallido',()=>{
        cy.request({url:'employees/4', failOnStatusCode: false})
        .its('status')
        .should('eq', 404)
    });
});
```

## Validando el body

```javascript
describe('Probando el body', ()=>{

    it('Probar el body 2',()=>{

        cy.request('employees/1')
        .its('body')
        .its("first_name")
        .should("be.equal","Javier")

        cy.request('employees/1').then(response =>{
            
            expect(response.status).to.be.equal(200);
            expect(response.headers['content-type']).to.be.equal('application/json');
            expect(response.body.first_name).to.be.equal('Javier');
            expect(response.body.last_name).to.be.equal('Eschweiler');

        })
    });

});
```
## Validando un error

```javascript
describe("Oribando errores", ()=>{

    it("Debe validar el status code fallido y el mensaje de error", ()=>{

        cy.request({url: 'https://pokeapi.co/api/v2/aaa', failOnStatusCode: false})
        .then(response =>{
        expect(response.status).to.eq(404);
        expect(response.body).to.be.eq("Not Found");
        });
    });

    it("Debe validar el status code fallido y el mensaje de error de rick and morty", ()=>{

        cy.request({url: 'https://rickandmortyapi.com/api/location/5487', failOnStatusCode: false})
        .then(response =>{
        expect(response.status).to.eq(404);
        expect(response.body).to.have.property('error', "Location not found");
        });
    });
})
```

