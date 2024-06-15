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
                abilities: [
                    {
                      ability: {
                        name: "static",
                        url: "https://pokeapi.co/api/v2/ability/9/",
                      },
                      is_hidden: false,
                      slot: 1,
                    },
                    {
                      ability: {
                        name: "lightning-rod",
                        url: "https://pokeapi.co/api/v2/ability/31/",
                      },
                      is_hidden: true,
                      slot: 3,
                    },
                  ],
                  base_experience: 112,
                  forms: [
                    {
                      name: "pikachu",
                      url: "https://pokeapi.co/api/v2/pokemon-form/25/",
                    },
                  ],
                  game_indices: [
                    {
                      game_index: 84,
                      version: {
                        name: "red",
                        url: "https://pokeapi.co/api/v2/version/1/",
                      },
                    },
                    {
                      game_index: 84,
                      version: {
                        name: "blue",
                        url: "https://pokeapi.co/api/v2/version/2/",
                      },
                    },
                    {
                      game_index: 84,
                      version: {
                        name: "yellow",
                        url: "https://pokeapi.co/api/v2/version/3/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "gold",
                        url: "https://pokeapi.co/api/v2/version/4/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "silver",
                        url: "https://pokeapi.co/api/v2/version/5/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "crystal",
                        url: "https://pokeapi.co/api/v2/version/6/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "ruby",
                        url: "https://pokeapi.co/api/v2/version/7/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "sapphire",
                        url: "https://pokeapi.co/api/v2/version/8/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "emerald",
                        url: "https://pokeapi.co/api/v2/version/9/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "firered",
                        url: "https://pokeapi.co/api/v2/version/10/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "leafgreen",
                        url: "https://pokeapi.co/api/v2/version/11/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "diamond",
                        url: "https://pokeapi.co/api/v2/version/12/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "pearl",
                        url: "https://pokeapi.co/api/v2/version/13/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "platinum",
                        url: "https://pokeapi.co/api/v2/version/14/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "heartgold",
                        url: "https://pokeapi.co/api/v2/version/15/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "soulsilver",
                        url: "https://pokeapi.co/api/v2/version/16/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "black",
                        url: "https://pokeapi.co/api/v2/version/17/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "white",
                        url: "https://pokeapi.co/api/v2/version/18/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "black-2",
                        url: "https://pokeapi.co/api/v2/version/21/",
                      },
                    },
                    {
                      game_index: 25,
                      version: {
                        name: "white-2",
                        url: "https://pokeapi.co/api/v2/version/22/",
                      },
                    },
                  ],
                  height: 4,
                  held_items: [
                    {
                      item: {
                        name: "oran-berry",
                        url: "https://pokeapi.co/api/v2/item/132/",
                      },
                      version_details: [
                        {
                          rarity: 50,
                          version: {
                            name: "ruby",
                            url: "https://pokeapi.co/api/v2/version/7/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "sapphire",
                            url: "https://pokeapi.co/api/v2/version/8/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version/9/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "diamond",
                            url: "https://pokeapi.co/api/v2/version/12/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "pearl",
                            url: "https://pokeapi.co/api/v2/version/13/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version/14/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "heartgold",
                            url: "https://pokeapi.co/api/v2/version/15/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "soulsilver",
                            url: "https://pokeapi.co/api/v2/version/16/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "black",
                            url: "https://pokeapi.co/api/v2/version/17/",
                          },
                        },
                        {
                          rarity: 50,
                          version: {
                            name: "white",
                            url: "https://pokeapi.co/api/v2/version/18/",
                          },
                        },
                      ],
                    },
                    {
                      item: {
                        name: "light-ball",
                        url: "https://pokeapi.co/api/v2/item/213/",
                      },
                      version_details: [
                        {
                          rarity: 5,
                          version: {
                            name: "ruby",
                            url: "https://pokeapi.co/api/v2/version/7/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "sapphire",
                            url: "https://pokeapi.co/api/v2/version/8/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version/9/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "diamond",
                            url: "https://pokeapi.co/api/v2/version/12/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "pearl",
                            url: "https://pokeapi.co/api/v2/version/13/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version/14/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "heartgold",
                            url: "https://pokeapi.co/api/v2/version/15/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "soulsilver",
                            url: "https://pokeapi.co/api/v2/version/16/",
                          },
                        },
                        {
                          rarity: 1,
                          version: {
                            name: "black",
                            url: "https://pokeapi.co/api/v2/version/17/",
                          },
                        },
                        {
                          rarity: 1,
                          version: {
                            name: "white",
                            url: "https://pokeapi.co/api/v2/version/18/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "black-2",
                            url: "https://pokeapi.co/api/v2/version/21/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "white-2",
                            url: "https://pokeapi.co/api/v2/version/22/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "x",
                            url: "https://pokeapi.co/api/v2/version/23/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "y",
                            url: "https://pokeapi.co/api/v2/version/24/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "omega-ruby",
                            url: "https://pokeapi.co/api/v2/version/25/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version/26/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "sun",
                            url: "https://pokeapi.co/api/v2/version/27/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "moon",
                            url: "https://pokeapi.co/api/v2/version/28/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "ultra-sun",
                            url: "https://pokeapi.co/api/v2/version/29/",
                          },
                        },
                        {
                          rarity: 5,
                          version: {
                            name: "ultra-moon",
                            url: "https://pokeapi.co/api/v2/version/30/",
                          },
                        },
                      ],
                    },
                  ],
                  id: 1,
                  is_default: true,
                  location_area_encounters:
                    "https://pokeapi.co/api/v2/pokemon/25/encounters",
                  moves: [
                    {
                      move: {
                        name: "mega-punch",
                        url: "https://pokeapi.co/api/v2/move/5/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "pay-day", url: "https://pokeapi.co/api/v2/move/6/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "thunder-punch",
                        url: "https://pokeapi.co/api/v2/move/9/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "slam", url: "https://pokeapi.co/api/v2/move/21/" },
                      version_group_details: [
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 24,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 28,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "double-kick",
                        url: "https://pokeapi.co/api/v2/move/24/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 9,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "mega-kick",
                        url: "https://pokeapi.co/api/v2/move/25/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "headbutt",
                        url: "https://pokeapi.co/api/v2/move/29/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "body-slam",
                        url: "https://pokeapi.co/api/v2/move/34/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "take-down",
                        url: "https://pokeapi.co/api/v2/move/36/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "double-edge",
                        url: "https://pokeapi.co/api/v2/move/38/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "tail-whip",
                        url: "https://pokeapi.co/api/v2/move/39/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 3,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "growl", url: "https://pokeapi.co/api/v2/move/45/" },
                      version_group_details: [
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 5,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "surf", url: "https://pokeapi.co/api/v2/move/57/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "stadium-surfing-pikachu",
                            url: "https://pokeapi.co/api/v2/move-learn-method/5/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "stadium-surfing-pikachu",
                            url: "https://pokeapi.co/api/v2/move-learn-method/5/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "submission",
                        url: "https://pokeapi.co/api/v2/move/66/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "counter",
                        url: "https://pokeapi.co/api/v2/move/68/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "seismic-toss",
                        url: "https://pokeapi.co/api/v2/move/69/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "strength",
                        url: "https://pokeapi.co/api/v2/move/70/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "thunder-shock",
                        url: "https://pokeapi.co/api/v2/move/84/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "thunderbolt",
                        url: "https://pokeapi.co/api/v2/move/85/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 36,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "thunder-wave",
                        url: "https://pokeapi.co/api/v2/move/86/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 9,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 4,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "thunder",
                        url: "https://pokeapi.co/api/v2/move/87/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 43,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 41,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 58,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 58,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 58,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 30,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 44,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "dig", url: "https://pokeapi.co/api/v2/move/91/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "toxic", url: "https://pokeapi.co/api/v2/move/92/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "agility",
                        url: "https://pokeapi.co/api/v2/move/97/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 33,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 27,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 24,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "quick-attack",
                        url: "https://pokeapi.co/api/v2/move/98/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 16,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 11,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 10,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 6,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "rage", url: "https://pokeapi.co/api/v2/move/99/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "mimic", url: "https://pokeapi.co/api/v2/move/102/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "double-team",
                        url: "https://pokeapi.co/api/v2/move/104/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 15,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 23,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 23,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 23,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 12,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 8,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "defense-curl",
                        url: "https://pokeapi.co/api/v2/move/111/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "light-screen",
                        url: "https://pokeapi.co/api/v2/move/113/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 45,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 53,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 53,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 53,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 40,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "reflect",
                        url: "https://pokeapi.co/api/v2/move/115/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "bide", url: "https://pokeapi.co/api/v2/move/117/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "swift", url: "https://pokeapi.co/api/v2/move/129/" },
                      version_group_details: [
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "skull-bash",
                        url: "https://pokeapi.co/api/v2/move/130/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "flash", url: "https://pokeapi.co/api/v2/move/148/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "rest", url: "https://pokeapi.co/api/v2/move/156/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "substitute",
                        url: "https://pokeapi.co/api/v2/move/164/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "red-blue",
                            url: "https://pokeapi.co/api/v2/version-group/1/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "yellow",
                            url: "https://pokeapi.co/api/v2/version-group/2/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "thief", url: "https://pokeapi.co/api/v2/move/168/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "snore", url: "https://pokeapi.co/api/v2/move/173/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "curse", url: "https://pokeapi.co/api/v2/move/174/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "reversal",
                        url: "https://pokeapi.co/api/v2/move/179/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "protect",
                        url: "https://pokeapi.co/api/v2/move/182/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "sweet-kiss",
                        url: "https://pokeapi.co/api/v2/move/186/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "mud-slap",
                        url: "https://pokeapi.co/api/v2/move/189/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "zap-cannon",
                        url: "https://pokeapi.co/api/v2/move/192/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "detect",
                        url: "https://pokeapi.co/api/v2/move/197/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "endure",
                        url: "https://pokeapi.co/api/v2/move/203/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "charm", url: "https://pokeapi.co/api/v2/move/204/" },
                      version_group_details: [
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "rollout",
                        url: "https://pokeapi.co/api/v2/move/205/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "swagger",
                        url: "https://pokeapi.co/api/v2/move/207/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "spark", url: "https://pokeapi.co/api/v2/move/209/" },
                      version_group_details: [
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 26,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 20,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "attract",
                        url: "https://pokeapi.co/api/v2/move/213/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "sleep-talk",
                        url: "https://pokeapi.co/api/v2/move/214/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "return",
                        url: "https://pokeapi.co/api/v2/move/216/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "frustration",
                        url: "https://pokeapi.co/api/v2/move/218/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "dynamic-punch",
                        url: "https://pokeapi.co/api/v2/move/223/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "encore",
                        url: "https://pokeapi.co/api/v2/move/227/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "iron-tail",
                        url: "https://pokeapi.co/api/v2/move/231/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "hidden-power",
                        url: "https://pokeapi.co/api/v2/move/237/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "rain-dance",
                        url: "https://pokeapi.co/api/v2/move/240/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "gold-silver",
                            url: "https://pokeapi.co/api/v2/version-group/3/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "crystal",
                            url: "https://pokeapi.co/api/v2/version-group/4/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "rock-smash",
                        url: "https://pokeapi.co/api/v2/move/249/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "uproar",
                        url: "https://pokeapi.co/api/v2/move/253/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "facade",
                        url: "https://pokeapi.co/api/v2/move/263/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "focus-punch",
                        url: "https://pokeapi.co/api/v2/move/264/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "helping-hand",
                        url: "https://pokeapi.co/api/v2/move/270/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "brick-break",
                        url: "https://pokeapi.co/api/v2/move/280/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "knock-off",
                        url: "https://pokeapi.co/api/v2/move/282/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "secret-power",
                        url: "https://pokeapi.co/api/v2/move/290/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "signal-beam",
                        url: "https://pokeapi.co/api/v2/move/324/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "covet", url: "https://pokeapi.co/api/v2/move/343/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "volt-tackle",
                        url: "https://pokeapi.co/api/v2/move/344/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "calm-mind",
                        url: "https://pokeapi.co/api/v2/move/347/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "lets-go-pikachu-lets-go-eevee",
                            url: "https://pokeapi.co/api/v2/version-group/19/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "shock-wave",
                        url: "https://pokeapi.co/api/v2/move/351/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ruby-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/5/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "emerald",
                            url: "https://pokeapi.co/api/v2/version-group/6/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "firered-leafgreen",
                            url: "https://pokeapi.co/api/v2/version-group/7/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "colosseum",
                            url: "https://pokeapi.co/api/v2/version-group/12/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "xd",
                            url: "https://pokeapi.co/api/v2/version-group/13/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "natural-gift",
                        url: "https://pokeapi.co/api/v2/move/363/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "feint", url: "https://pokeapi.co/api/v2/move/364/" },
                      version_group_details: [
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 21,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 16,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "fling", url: "https://pokeapi.co/api/v2/move/374/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "magnet-rise",
                        url: "https://pokeapi.co/api/v2/move/393/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "nasty-plot",
                        url: "https://pokeapi.co/api/v2/move/417/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "discharge",
                        url: "https://pokeapi.co/api/v2/move/435/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 37,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 42,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 34,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 32,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "captivate",
                        url: "https://pokeapi.co/api/v2/move/445/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "grass-knot",
                        url: "https://pokeapi.co/api/v2/move/447/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "charge-beam",
                        url: "https://pokeapi.co/api/v2/move/451/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "diamond-pearl",
                            url: "https://pokeapi.co/api/v2/version-group/8/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "platinum",
                            url: "https://pokeapi.co/api/v2/version-group/9/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "heartgold-soulsilver",
                            url: "https://pokeapi.co/api/v2/version-group/10/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "electro-ball",
                        url: "https://pokeapi.co/api/v2/move/486/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 18,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 13,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 12,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: { name: "round", url: "https://pokeapi.co/api/v2/move/496/" },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "echoed-voice",
                        url: "https://pokeapi.co/api/v2/move/497/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "volt-switch",
                        url: "https://pokeapi.co/api/v2/move/521/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "electroweb",
                        url: "https://pokeapi.co/api/v2/move/527/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "wild-charge",
                        url: "https://pokeapi.co/api/v2/move/528/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-white",
                            url: "https://pokeapi.co/api/v2/version-group/11/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "black-2-white-2",
                            url: "https://pokeapi.co/api/v2/version-group/14/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 50,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "draining-kiss",
                        url: "https://pokeapi.co/api/v2/move/577/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "play-rough",
                        url: "https://pokeapi.co/api/v2/move/583/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "play-nice",
                        url: "https://pokeapi.co/api/v2/move/589/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 7,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 7,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 7,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 7,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "confide",
                        url: "https://pokeapi.co/api/v2/move/590/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "electric-terrain",
                        url: "https://pokeapi.co/api/v2/move/604/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "machine",
                            url: "https://pokeapi.co/api/v2/move-learn-method/4/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "nuzzle",
                        url: "https://pokeapi.co/api/v2/move/609/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 23,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "x-y",
                            url: "https://pokeapi.co/api/v2/version-group/15/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "omega-ruby-alpha-sapphire",
                            url: "https://pokeapi.co/api/v2/version-group/16/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sun-moon",
                            url: "https://pokeapi.co/api/v2/version-group/17/",
                          },
                        },
                        {
                          level_learned_at: 29,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                        {
                          level_learned_at: 1,
                          move_learn_method: {
                            name: "level-up",
                            url: "https://pokeapi.co/api/v2/move-learn-method/1/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "laser-focus",
                        url: "https://pokeapi.co/api/v2/move/673/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "ultra-sun-ultra-moon",
                            url: "https://pokeapi.co/api/v2/version-group/18/",
                          },
                        },
                      ],
                    },
                    {
                      move: {
                        name: "rising-voltage",
                        url: "https://pokeapi.co/api/v2/move/804/",
                      },
                      version_group_details: [
                        {
                          level_learned_at: 0,
                          move_learn_method: {
                            name: "tutor",
                            url: "https://pokeapi.co/api/v2/move-learn-method/3/",
                          },
                          version_group: {
                            name: "sword-shield",
                            url: "https://pokeapi.co/api/v2/version-group/20/",
                          },
                        },
                      ],
                    },
                  ],
                  name: "pikachu",
                  order: 35,
                  past_types: [],
                  species: {
                    name: "pikachu",
                    url: "https://pokeapi.co/api/v2/pokemon-species/25/",
                  },
                  sprites: {
                    back_default:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/back/25.png",
                    back_female:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/back/female/25.png",
                    back_shiny:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/back/shiny/25.png",
                    back_shiny_female:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/back/shiny/female/25.png",
                    front_default:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png",
                    front_female:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/female/25.png",
                    front_shiny:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/shiny/25.png",
                    front_shiny_female:
                      "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/shiny/female/25.png",
                    other: {
                      dream_world: {
                        front_default:
                          "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/dream-world/25.svg",
                        front_female: null,
                      },
                      home: {
                        front_default:
                          "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/home/25.png",
                        front_female:
                          "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/home/female/25.png",
                        front_shiny:
                          "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/home/shiny/25.png",
                        front_shiny_female:
                          "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/home/shiny/female/25.png",
                      },
                      "official-artwork": {
                        front_default:
                          "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
                      },
                    },
                    versions: {
                      "generation-i": {
                        "red-blue": {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/red-blue/back/25.png",
                          back_gray:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/red-blue/back/gray/25.png",
                          back_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/red-blue/transparent/back/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/red-blue/25.png",
                          front_gray:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/red-blue/gray/25.png",
                          front_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/red-blue/transparent/25.png",
                        },
                        yellow: {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/yellow/back/25.png",
                          back_gray:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/yellow/back/gray/25.png",
                          back_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/yellow/transparent/back/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/yellow/25.png",
                          front_gray:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/yellow/gray/25.png",
                          front_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-i/yellow/transparent/25.png",
                        },
                      },
                      "generation-ii": {
                        crystal: {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/back/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/back/shiny/25.png",
                          back_shiny_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/transparent/back/shiny/25.png",
                          back_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/transparent/back/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/shiny/25.png",
                          front_shiny_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/transparent/shiny/25.png",
                          front_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/crystal/transparent/25.png",
                        },
                        gold: {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/back/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/back/shiny/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/shiny/25.png",
                          front_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/transparent/25.png",
                        },
                        silver: {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/silver/back/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/silver/back/shiny/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/silver/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/silver/shiny/25.png",
                          front_transparent:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/silver/transparent/25.png",
                        },
                      },
                      "generation-iii": {
                        emerald: {
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/emerald/shiny/25.png",
                        },
                        "firered-leafgreen": {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/firered-leafgreen/back/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/firered-leafgreen/back/shiny/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/firered-leafgreen/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/firered-leafgreen/shiny/25.png",
                        },
                        "ruby-sapphire": {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/ruby-sapphire/back/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/ruby-sapphire/back/shiny/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/ruby-sapphire/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/ruby-sapphire/shiny/25.png",
                        },
                      },
                      "generation-iv": {
                        "diamond-pearl": {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/back/25.png",
                          back_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/back/female/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/back/shiny/25.png",
                          back_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/back/shiny/female/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/female/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/shiny/25.png",
                          front_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/diamond-pearl/shiny/female/25.png",
                        },
                        "heartgold-soulsilver": {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/back/25.png",
                          back_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/back/female/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/back/shiny/25.png",
                          back_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/back/shiny/female/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/female/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/shiny/25.png",
                          front_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/heartgold-soulsilver/shiny/female/25.png",
                        },
                        platinum: {
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/back/25.png",
                          back_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/back/female/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/back/shiny/25.png",
                          back_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/back/shiny/female/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/female/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/shiny/25.png",
                          front_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iv/platinum/shiny/female/25.png",
                        },
                      },
                      "generation-v": {
                        "black-white": {
                          animated: {
                            back_default:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/back/25.gif",
                            back_female:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/back/female/25.gif",
                            back_shiny:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/back/shiny/25.gif",
                            back_shiny_female:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/back/shiny/female/25.gif",
                            front_default:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/25.gif",
                            front_female:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/female/25.gif",
                            front_shiny:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/shiny/25.gif",
                            front_shiny_female:
                              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/shiny/female/25.gif",
                          },
                          back_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/back/25.png",
                          back_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/back/female/25.png",
                          back_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/back/shiny/25.png",
                          back_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/back/shiny/female/25.png",
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/female/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/shiny/25.png",
                          front_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/shiny/female/25.png",
                        },
                      },
                      "generation-vi": {
                        "omegaruby-alphasapphire": {
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/omegaruby-alphasapphire/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/omegaruby-alphasapphire/female/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/omegaruby-alphasapphire/shiny/25.png",
                          front_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/omegaruby-alphasapphire/shiny/female/25.png",
                        },
                        "x-y": {
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/x-y/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/x-y/female/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/x-y/shiny/25.png",
                          front_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vi/x-y/shiny/female/25.png",
                        },
                      },
                      "generation-vii": {
                        icons: {
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vii/icons/25.png",
                          front_female: null,
                        },
                        "ultra-sun-ultra-moon": {
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vii/ultra-sun-ultra-moon/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vii/ultra-sun-ultra-moon/female/25.png",
                          front_shiny:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vii/ultra-sun-ultra-moon/shiny/25.png",
                          front_shiny_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-vii/ultra-sun-ultra-moon/shiny/female/25.png",
                        },
                      },
                      "generation-viii": {
                        icons: {
                          front_default:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-viii/icons/25.png",
                          front_female:
                            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-viii/icons/female/25.png",
                        },
                      },
                    },
                  },
                  stats: [
                    {
                      base_stat: 35,
                      effort: 0,
                      stat: { name: "hp", url: "https://pokeapi.co/api/v2/stat/1/" },
                    },
                    {
                      base_stat: 55,
                      effort: 0,
                      stat: { name: "attack", url: "https://pokeapi.co/api/v2/stat/2/" },
                    },
                    {
                      base_stat: 40,
                      effort: 0,
                      stat: { name: "defense", url: "https://pokeapi.co/api/v2/stat/3/" },
                    },
                    {
                      base_stat: 50,
                      effort: 0,
                      stat: {
                        name: "special-attack",
                        url: "https://pokeapi.co/api/v2/stat/4/",
                      },
                    },
                    {
                      base_stat: 50,
                      effort: 0,
                      stat: {
                        name: "special-defense",
                        url: "https://pokeapi.co/api/v2/stat/5/",
                      },
                    },
                    {
                      base_stat: 90,
                      effort: 2,
                      stat: { name: "speed", url: "https://pokeapi.co/api/v2/stat/6/" },
                    },
                  ],
                  types: [
                    {
                      slot: 1,
                      type: {
                        name: "electric",
                        url: "https://pokeapi.co/api/v2/type/13/",
                      },
                    },
                  ],
                  weight: 60,

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

