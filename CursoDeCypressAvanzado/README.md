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

## ¿Qué son los Flaky Tests?

### Flaky test

Uno de los conceptos más desconocidos, pero a su vez más vividos, por los programadores es el «flaky test». Este concepto viene a referirse a la debilidad que tienen nuestros tests ya que son inestables y frágiles a cualquier cambio externo. Es decir, que el test se puede comportar diferente en cada ejecución presentando diferentes resultados (exitoso o fallido) sin haber realizado cambios en el código o en el test.
Imagina que tienes un caso de uso que matricula a un estudiante con un curso determinado.

```javascript
class EnrollmentStudentUseCase
{
    private EnrollmentRepositoryInterface $enrollmentRepository;
    private CourseApi $courseApi;

    public function __construct(
        EnrollmentRepositoryInterface $enrollmentRepository,
        CourseApi $courseApi
    ) {
        $this->enrollmentRepository = $enrollmentRepository;
        $this->courseApi = $courseApi;
    }

    public function make(
        EnrollmentId $id,
        EnrollmentStudentId $studentId,
        EnrollmentCourseId $courseId
    ): void {
        $course = $this->courseApi->findByCourseId($courseId);

        $enrollment = Enrollment::make(
            $id,
            $studentId,
            $courseId,
            $course->title()
        );

        $this->enrollmentRepository->save($enrollment);
    }
}
```
### Prueba unitaria

```javascript
final class EnrollmentStudentUseCaseTest extends MockeryTestCase
{
    /** @test */
    public function it_should_enroll_student(): void
    {
        list($id, $title, $studentId, $courseId) = EnrollmentFaker::createValueObjectOne();
        $enrollmentRepositoryInterfaceMock = $this->createEnrollmentRepositoryInterfaceMock();
        $courseApi = new CourseApi();
        $enrollmentMaker = new EnrollmentStudentUseCase($enrollmentRepositoryInterfaceMock, $courseApi);

        $enrollmentMaker->make($id, $studentId, $courseId);
    }

    private function createEnrollmentRepositoryInterfaceMock(): EnrollmentRepositoryInterface
    {
        $mock = \Mockery::mock(EnrollmentRepositoryInterface::class);

        $mock->shouldReceive('save')->once()->withArgs([Enrollment::class])->andReturnNull();

        return $mock;
    }
}
```

El test depende de un servicio externo (CourseApi), si este servicio es bastante inestable el test fallará. Con lo cual, tendremos un flaky test. Una forma de resolver esto es mockeando la dependencia que tiene el caso de uso.
A continuación dejo una lista de cosas que pueden hacer que nuestro tests se conviertan en un flaky test.
- Timeouts
- Asynchronous Waits
- Cache
- Order dependency
- Time of day

Para que la prueba se repita lo agragamos en archivo `cypress.config.js` y se agrega `retries:2,` el dos es el numero de intentos.

```javascript

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
    retries:2,
  },
});
```
para configurar los dos modos de retries
```javascript
retries: {
        runMode:2,
        openMode: 0,
    },
```

**runMode**: 2 = especifica que Cypress debe reintentar una prueba fallida hasta dos veces adicionales cuando se ejecuta en modo de ejecución.
**openMode**: 0 especifica que Cypress no debe reintentar las pruebas cuando se ejecuta en modo abierto (es decir, utilizando el Cypress Test Runner de forma interactiva).

## Interceptando Network Requests

```javascript
describe("Interceptando network requests", ()=>{

    it("Repaso de request", ()=>{
        cy.request("https://pokeapi.co/api/v2/pokemon/ditto").then(response =>{
            expect(response.body).to.have.property("name", "ditto");
            expect(response.status).to.eq(200);
            cy.log(response.body);
        });
    });

    it('Prueba de intercept simple', ()=>{
        cy.intercept("GET","https://pokeapi.co/api/v2/pokemon-species/1").as("bulbasaur");

        cy.visit("/");

        cy.contains("Bulbasaur").parent().parent().within( element=>{
            cy.wrap(element).contains("Más detalles").click();
        });
        
        /*cy.wait("@bulbasaur").then((interception) =>{
            cy.log(interception);
            expect(interception.response.body).to.have.property("name", "bulbasaur");
            expect(interception.response.statusCode).to.eq(200);
        });*/
        
        //cy.wait('@bulbasaur',{timeout: 2000});

        cy.wait('@bulbasaur').its('response.statusCode').should('eq',200);
    });
});
```

## Interceptando request

```javascript
describe("Interceptando network requests", ()=>{

    it("Repaso de request", ()=>{
        cy.request("https://pokeapi.co/api/v2/pokemon/ditto").then(response =>{
            expect(response.body).to.have.property("name", "ditto");
            expect(response.status).to.eq(200);
            cy.log(response.body);
        });
    });

    it('Prueba de intercept simple', ()=>{
        cy.intercept("GET","https://pokeapi.co/api/v2/pokemon-species/1").as("bulbasaur");

        cy.visit("/");

        cy.contains("Bulbasaur").parent().parent().within( element=>{
            cy.wrap(element).contains("Más detalles").click();
        });
        
        /*cy.wait("@bulbasaur").then((interception) =>{
            cy.log(interception);
            expect(interception.response.body).to.have.property("name", "bulbasaur");
            expect(interception.response.statusCode).to.eq(200);
        });*/
        
        //cy.wait('@bulbasaur',{timeout: 2000});

        cy.wait('@bulbasaur').its('response.statusCode').should('eq',200);
    });

    it("Probar intercept forzarlo a que falle", ()=>{
        cy.intercept("GET","https://pokeapi.co/api/v2/pokemon-species/1",{
            forceNetworkError: true,
        }).as("error");
        cy.visit("/");

        cy.contains("Bulbasaur").parent().parent().within( element=>{
            cy.wrap(element).contains("Más detalles").click();
        });

        cy.wait("@error").should("have.property", "error")
    });

    it.only('prueba intercept cambiando el body', ()=>{
        cy.intercept("GET","https://pokeapi.co/api/v2/pokemon/1",{
            stausCode:200,
            body:{
                // pegra el archivo pokemon.json aquí, para que funcione el testing

            },
        }).as("pikachu");

        cy.visit("/");

        cy.contains("Bulbasaur").parent().parent().within( element=>{
            cy.wrap(element).contains("Más detalles").click();
        });

        cy.wait('@pikachu').then(interception =>{
            expect(interception.response.body).to.have.property("name", "pikachu");
            expect(interception.response.statusCode).to.eq(200);
        });

        cy.wait(5000)
    });
});
```

## Patrones de diseño: Page Object Model

se creea una carpeta **pageObjec**t y luego el archivo **loginPage.js** i se agrega el siguiente codigo.

```javascript
export class LoginPage {
    constructor() {
        this.userInput = "#user_login";
        this.passwordInput = "#user_password";
        this.loginButton = "#login_form > div.form-actions > input";
        this.tabs = {
            account_summary_tab: "#account_summary_tab",
            account_activity_tab: "#account_activity_tab",
            transfer_founds_tab: "#transfer_funds_tab",
        };
        this.error = ".alert.alert-error";
    };

    visit() {
        cy.visit('http://zero.webappsecurity.com/login.html');
    };

    validateLoginPage(){
        cy.get(this.userInput).should("be.visible");
        cy.get(this.passwordInput).should("be.visible");
        cy.get(this.loginButton).should("be.visible");
    };

    validateErrorLogin(){
        cy.get(this.error).should("be.visible");
    };

    validateSuccessLogin(){
        cy.get(this.tabs.account_activity_tab).should("be.visible");
        cy.get(this.tabs.account_activity_tab).should("be.visible");
        cy.get(this.tabs.transfer_founds_tab).should("be.visible");
    }

    login(email, password){
        cy.get(this.userInput).type(email);
        cy.get(this.passwordInput).type(password);
        cy.get(this.loginButton).click();

    };
};

export const loginPage = new LoginPage();
```

Se crea el archivo de test:

```javascript
import { loginPage } from "./pageObjects/loginPage";

describe('login con POM', ()=>{
    beforeEach(()=>{
        loginPage.visit();
    });

    it("Login erroneo", ()=>{
        loginPage.validateLoginPage()
        loginPage.login('lalalal', 'kshhdytsd');
        loginPage.validateErrorLogin();
    });

    it("Login exitoso", ()=>{
        loginPage.validateLoginPage()
        loginPage.login('username', 'password');
        loginPage.validateSuccessLogin();
    });
});
```

[GitHub - platzi/curso-cypress-avanzado at 8-buenas-practicas](https://github.com/platzi/curso-cypress-avanzado/tree/8-buenas-practicas)

## Custom commands

utilizando commands para las pruebas:

se  agrega el sigiente codigo al archivo `commands.js`:

```javascript
// Cypress.Commands.overwrite('visit', (originalFn, url, options) => { ... })

Cypress.Commands.add("login",(email, password)=>{
    const userInput = "#user_login";
    const passwordInput = "#user_password";
    const loginButton = "#login_form > div.form-actions > input";

    cy.visit('http://zero.webappsecurity.com/login.html');

    cy.get(userInput).type(email);
    cy.get(passwordInput).type(password,{ sensitive: true });
    cy.get(loginButton).click();

});

Cypress.Commands.overwrite("type", (orginalFn, element, text, options)=>{

    if(options && options.sensitive){
        options.log = false;

        Cypress.log({  //para proteger la clave
            $el: element,
            name: "type",
            message: "*".repeat((text.length)),
        });
    };

    return orginalFn(element, text, options);
});
```

Archivo de pruebas `loginCustomCommands.cy.js`

```javascript
describe('Login con custon commands', ()=>{

    it('login error', ()=>{

        cy.login('548845','548844')
    });

    it('login correcto', ()=>{

        cy.login('username','password')
    });
});
```

[GitHub - platzi/curso-cypress-avanzado at 9-custom-commands](https://github.com/platzi/curso-cypress-avanzado/tree/9-custom-commands)

## Variables de entorno

Agregamos una mueva propiedad en **cypress.config.js**

```javascript
    env:{
      Credentials: {
        user: "username",
        password: "password",
      },
    },
```

**test con Login exitoso con cy.env**

```javascript
it("Login exitoso con cy.env.json", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(Cypress.env("Credentials").user, Cypress.env("Credentials").password);
        loginPage.validateSuccessLogin();
    });
```

**Login exitoso con cy.env.json**

se crea un archivo json en la carpeta principal con el nombre `cypress.env.json`.

```json
{
    "credentials":{
        "user":"username",
        "password":"password"
    }
}
```
iniciamos desde la terminal:

`"variable:entorno": "cypress open --env VARIABLE_DE_ENTORNO=valor"`

```javascript
{
  "name": "proyecto",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "test": "npx cypress open",
    "variable:entorno": "cypress open --env VARIABLE_DE_ENTORNO=valor"
  },
  "keywords": [],
  "author": "Mario Alexander Vargas Celis",
  "license": "ISC",
  "description": "",
  "dependencies": {
    "cypress": "^13.11.0"
  },
  "devDependencies": {
    "cypress-xpath": "^2.0.1",
    "prettier": "^3.3.2"
  }
}
```

test

```javascript
import { loginPage } from "./pageObjects/loginPage";

describe('login con POM', ()=>{
    beforeEach(()=>{
        loginPage.visit();
    });

    it("Login erroneo", ()=>{
        loginPage.validateLoginPage()
        loginPage.login('lalalal', 'kshhdytsd');
        loginPage.validateErrorLogin();
    });

    it("Login exitoso", ()=>{
        loginPage.validateLoginPage()
        loginPage.login('username', 'password');
        loginPage.validateSuccessLogin();
    });

    it("Login exitoso con cy.env", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(Cypress.env("credentials").user, Cypress.env("credentials").password);
        loginPage.validateSuccessLogin();
    });

    it("Login exitoso con cy.env.json", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(
            Cypress.env("credentials").user, 
            Cypress.env("credentials").password
    );
    loginPage.validateErrorLogin();
    });

    it.only("Login erroneo desde la termina", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(
            Cypress.env("credentials").user, 
            Cypress.env("credentials").password
    );
    });
});

```

## Cypress.env

**variables de entorno** ingresamos el siguiente codigo en **cypress.config.js**

```javascript
setupNodeEvents(on, config) {
      // implement node event listeners here
      config.env.variable=process.env.NODE_ENV ?? 'NO HAY VARIABLE'; // INGRESA  ALAS VAIABLES DE ENTORNO
      return config;
    },
```
luego en el archivo packege.json se agrega un script con el sigiente codigo:

*linux o mac*

 ```json
 "variable:entorno:sistema": "export NODE_ENV=VARIABLE_DE_DESARROLLO && cypress open"
 ````

 *windows*

 ```json
 "set NODE_ENV=VARIABLE_DE_DESARROLLO && cypress open"
 ```

[GitHub - platzi/curso-cypress-avanzado at 10-cypress-env](https://github.com/platzi/curso-cypress-avanzado/tree/10-cypress-env)

## Visual testing

instalar pluging
`npm i -D cypress-image-snapshot --legacy-peer-deps` or `npm i -D cypress-image-snapshot --force`

para activarlo vamos a cypress.config.js y agregamos el plugin:

```javascript
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
```

luego a "commnds.js" importamos el plugin con el siguiente codigo:

```javascript
import { addMatchImageSnapshotCommand } from "cypress-image-snapshot/command";

addMatchImageSnapshotCommand({
    failureThreshold:0.03,
    failureThresholdType: "percent",
    customDiffConfig: {threshold:0.1},
    capture: "viewport",
});
```

para actualizar el snapshot lo agregamos en package.json el siguiente codigo en scripts :

```json
    "test-update-snapshot": "cypress open --env updateSnapshots=true",
```
y iniciamo los test:

```javascript
describe("Visual testing", ()=>{

        it('Mi prueba de regresion', ()=>{
            cy.visit('/');

            cy.wait(4000);
            cy.scrollTo("bottom");
            cy.matchImageSnapshot();
        });

        it("Segunda prueba a un solo elemento", ()=>{
            cy.visit('/');
            cy.contains("Bulbasaur").should("be.visible").matchImageSnapshot();
    
        });
});
```

## Seguridad en Cypress

```javascript
describe("Seguridad", ()=>{

    it('Navegar entre diferentes dominios', ()=>{
        cy.visit('/');
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("#title").type("Titulo de prueba");
    });

    it.only("navego a un dominio", ()=>{
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("h1").invoke("text").as("titulo");
    });

    it.only("navego a otro dominio", ()=>{
        cy.visit("/");
        cy.log(this.titulo);
        //cy.get("h1").invoke('text').as("titulo");
    })
});
```

## Navegación entre pestañas del mismo sitio

```javascipt
describe("Seguridad", ()=>{

    it('Navegar entre diferentes dominios', ()=>{
        cy.visit('/');
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("#title").type("Titulo de prueba");
    });

    it.only("navego a un dominio", ()=>{
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("h1").invoke("text").as("titulo");
    });

    it.only("navego a otro dominio", ()=>{
        cy.visit("/");
        cy.log(this.titulo);
        //cy.get("h1").invoke('text').as("titulo");
    })
});
```

## Navegar entre diferentes dominios en diferentes tests

```javascript
it.only("navego en dos dominios en el mismo test", ()=>{
        cy.visit("/");
        cy.get("h1")
            .first()    
            .invoke("text") 
            .then((text)=>{
            Cypress.env({
                textito:text
            })
        });
        cy.origin("https://todo-cypress-iota.vercel.app", {args:{texto: "Hola"}}, function ({texto}){
            cy.visit('/');
            cy.log(texto);
            cy.log(Cypress.env());
        });
        cy.visit("/");
        cy.get("h1").first().invoke("text").should("be.equal", Cypress.env('textico'));
    });
```
[GitHub - platzi/curso-cypress-avanzado at 13/seguridad](https://github.com/platzi/curso-cypress-avanzado/tree/13/seguridad)

## Navegar entre diferentes dominios en diferentes tests y compartir información entre ambas páginas

Se agrega un nuevo plugin:

```javascript
const values = {}; // colocarlo al inicio

baseUrl: "https://pokedexpokemon.netlify.app",
    experimentalSessionAndOrigin: true,
    setupNodeEvents(on, config) {
      // implement node event listeners here
      addMatchImageSnapshotPlugin(on, config)
      config.env.variable=process.env.NODE_ENV ?? 'NO HAY VARIABLE'; // INGRESA  ALAS VAIABLES DE ENTORNO

      on("task",{
        guardar(valor){
          const key = Object.keys(valor)[0];

          values[key] = valor[key];
          return null;
        },
        obtener(key){
          console.log('values', values);
          return values[key] ?? "No hay valor";
        }
      });
      return config;
    },
```

**test**

```javascript
it.only("Compartir informacion si usar session", ()=>{
        cy.visit("/");
        cy.get("h1").first().invoke("text").then((text) =>{
            cy.task("guardar", {texto:text});
        });
    });

    it.only("Compartir informacion si usar session 2", ()=>{
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.task("obtener", "texto").then((valor) =>{
            cy.get("#title").type(valor);
        })
    });
```

[GitHub - platzi/curso-cypress-avanzado at 14/seguridad-plugin](https://github.com/platzi/curso-cypress-avanzado/tree/14/seguridad-plugin)

## Cypress fixtures

```javascript
describe("Login con fixtures",()=>{
    beforeEach(()=>{
        loginPage.visit();
    });

    it("login erroneo 2", ()=>{
        loginPage.validatePageLogin();

        cy.fixture("credentials").then( credentials =>{
            loginPage.login(credentials.email, credentials.password);
        });

        loginPage.validateErrorLogin();
    });

    it("login exitoso", ()=>{
        loginPage.validatePageLogin();

        cy.fixture("usuarios").then( credentials =>{
            loginPage.login(credentials.email, credentials.password);
        });

        loginPage.validateErrorLogin();
    });
});

const credentialsForUsers = [
    {
        nombre: "credentials",
        titulo: "Login con credentials",
    },
    {
        nombre: "usuarios",
        titulo: "Login con users",
    }
]

credentialsForUsers.forEach(credentials=>{
    describe.only(credentials.titulo, () =>{
        beforeEach(()=>{
            loginPage.visit();
        });

        it('login exitoso con fixtures', () => {
            loginPage.validatePageLogin();

            cy.fixture(credentials.nombre).then(credentials => {
                    loginPage.login(credentials.email, credentials.password);
                }
            );
            loginPage.validateErrorLogin();
        });
    });
});
```

el la carpeta de fixtures se crean dos json:

credencials.json
```json
{
    "email":"hello@cypress.io",
    "password":"123456"
}
```
usuarios.json

```json
{
    "email":"hello@cypress.io",
    "password":"123456"
}
```

[GitHub - platzi/curso-cypress-avanzado at 16/data-driven-test](https://github.com/platzi/curso-cypress-avanzado/tree/16/data-driven-test)

## Configuración de plugins y steps dinámicos

para este tema se instala mas plugin

`npm @cypress/webpack-preprocessor @badeball/cypress-cucumber-preprocessor`, `npm i @badeball/cypress-cucumber-preprocessor @cypress/webpack-preprocessor --legacy-peer-deps` y `npm install webpack --legacy-peer-deps`
Los links de las dependencias requeridas en la clase

[badeball/cypress-cucumber-preprocessor](https://www.npmjs.com/package/@badeball/cypress-cucumber-preprocessor) [cypress/webpack-preprocessor](https://www.npmjs.com/package/@cypress/webpack-preprocessor)

configuracion de `cypress.config.js`

```javascript
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
```

**login.js**

```javascript
const {
    Given,
    When,
    Then,
} = require('@badeball/cypress-cucumber-preprocessor');
const {loginPage} = require("../../pageObjects/loginPage.js");

Given('I am on the login page', () => {
    loginPage.visit();
    loginPage.validatePageLogin();
});

When(`I fill in my login and password with {string} and {string}`, (username, password) => {
    loginPage.login(username, password);
});

Then('I should validate that I\'m logged in', () => {
    loginPage.validateSuccessLogin();
});
```

**login.feature**

**gherkin**

```vbnet
Feature: Login test

    Scenario: I login with correct credentials
        Given I am on the login page
        When I fill in my login and password with "username" and "password"
        Then I should validate that I'm logged in
```

[GitHub - platzi/curso-cypress-avanzado at 17/bdd](https://github.com/platzi/curso-cypress-avanzado/tree/17/bdd)

## Shared Step Definitions
 primer paso se debe pasar la carpeta **login** a la carpeta **support**, se crea una con el nombre **step_difinitions** se crean los archivos navigation

**navigation.js**

```javascript
const {
    Given,
    When,
    Then,
} = require('@badeball/cypress-cucumber-preprocessor');
const {loginPage} = require("../../pageObjects/loginPage");

Given('I am on the home page', () => {
    loginPage.validateSuccessLogin();
});

When('I click on the Account Activity Nav', ()=> {
    cy.contains("Account Activity").click();
});

Then('I should see the Account Activity content',()=> {
    cy.contains("Show Transactions").should("be.visible");
});
```

**navigation.feature**

```vbnet
Feature: NavigationBar

#  esto va a fallar porque no esta en shared steps teneos que mover login.js en shared steps
    Background:
        Given I am on the login page
        When I fill in my email and password with "username" and "password"

    Scenario: Navigate to the Feature Navigation Bar
        Given I am on the home page
        When I click on the Account Activity Nav
        Then I should see the Account Activity content
```

[GitHub - platzi/curso-cypress-avanzado at 18/bdd-shared-steps](https://github.com/platzi/curso-cypress-avanzado/tree/18/bdd-shared-steps)

## Data Driven Test por medio de Scenarios Outline

se crea un nuevo script en package.json con el código:
`"cucumber:tag":"cypress run --env tags=@probando"`

para correrlo en modo gengles:
`npx cypress run`

para iniciar solo una prueba en especifico se usa el sigiente código `npm run cucumber:tags`
y colocar @ y darle un nombre al que se desea correr:

```vbnet
Feature: Login test

    @probando
    Scenario: I login with correct credentials
        Given I am on the login page
        When I fill in my email and password with "username" and "password"
        Then I should validate that I'm logged in
```

[GitHub - platzi/curso-cypress-avanzado at 19/bdd-scenario-outline](https://github.com/platzi/curso-cypress-avanzado/tree/19/bdd-scenario-outline)