# Curso de Automatización de Test con Playwright

**instalación**

1. abrimos vscode
2. presionamos extensiones y escribimos Playwright Test for VSCode
3. presionamos ctrl + shift + p
4. escribimos install Playwright y presionamos en la opción
5. nos enviara las opciones del navegador y seleccionamos el navegador a utilizar
6. en la terminal se muestra la instalación

instalación de Playwright
Playwright puede ser incorporado mediante gestores de dependencia: .

npm: `npm init playwright@latest --yes -- --quiet --browser=chromium`
o para mas navegadores
`npm init playwright@latest --yes -- --quiet --browser=chromium --browser=firefox --browser=webkit --gha`
yarn:  `yarn create playwright`
pnpm: `pnpm dlx create-playwright`
.

✨ **Concepto clave** Se recomienda anexar las extensiones o plugins para el soporte de Playwright.

- [VS Code](https://marketplace.visualstudio.com/items?itemName=ms-playwright.playwright "VS Code")
- [JetBrains](https://plugins.jetbrains.com/plugin/18100-maestro "JetBrains") (configurando un launcher script)

. Una vez ejecutado el asistente de iniciación, podemos configurar nuestro proyecto mediante las siguientes preguntas: .

- Lenguaje de programación predominante (JavaScript o TypeScript)
- Localización de nuestras pruebas (tests o e2e si nuestro proyecto cuenta con pruebas)
- Elección de automatización (GitHub Actions)
- Instalación del soporte de pruebas para navegadores

### Estructura del project
Una vez elegido nuestras preferencias de configuración, el asistente anexará un conjunto de carpetas donde residirán nuestras pruebas.  
De lo nuevo, lo más destacable será `playwright.config` . Este archivo, permitirá manipular la configuración de Playwright. .

### Ejecución de pruebas y visualización de reportes

De manera inicial, Playwright nos ofrece su binario de ejecución mediante gestores de dependencias: .

- `npm- npx playwright test`
- `yarn - yarnx playwright test`
- `pnpm - pnpx playwright test`

✨ **Concepto clave** Cada test runner (Jest, Playwright, Cypress, etc.), nos permite tener un output como resultado de una ejecución. Para Playwright, los reportes estarán localizados en la carpeta `playwright-report` .

Adicionalmente, podemos visualizar nuestros test como un reporte en web mediante gestores de dependencias: .

- `npm- npx playwright show-report`
- `yarn - yarnx playwright show-report`
- `pnpm - pnpx playwright show-report `

[Documentación](https://playwright.dev/docs/intro "Documentación")
📌 Referencia Installation | Playwright - [Link](https://playwright.dev/docs/intro "Link")

## Cualquiera puede escribir tests

se iniciacon el codigo:
`npx playwright codegen`

se realiza todos los pasos deseados y se copia el codigo generado.

se crea el archivo de test y lo iniciamos con el codigo:

`npx playwright test`

Generación de Pruebas
📌 **Referencia** [Test Generator | Playwright](https://playwright.dev/docs/codegen-intro "Test Generator | Playwright")

Con Playwright, es posible generar pruebas de manera interactiva (siguiendo, en alma, a Selenium). Con ello, podemos agilizar, y de manera sencilla, la creación de pruebas según el lenguaje de nuestra elección. .

✨ **Concepto clave** Playwright, está condicionada su configuración ya sea con TypeScript o JavaScript. Las pruebas, pueden ser generadas según el lenguaje de nuestra elección disponible.

Para disponer de esta herramienta, ejecutamos el binario de Playwright mediante `playwright codegen [url]`. 

📌 **Referencia** [Test Generator | Playwright](https://playwright.dev/docs/codegen "Test Generator | Playwright")

Por ejemplo, utilizando el sitio de pruebas de Playwright podemos evaluar nuestro ejercicio al ejecutar el binario. .

- `pnpx playwright codegen demo.playwright.dev/todomvc`
- `pnpx playwright codegen example.cypress.io`

## Ejecuta tus tests

Se crea otro test, se crean los pasos que se desea y luego lo corremos el test con el código `npx playwright test uitesting` donde está uitesting lo cambia por el nombre de su test.

Para ver lo que sucede, se utiliza headed como lo muestra el siguiente código :`npx playwright test --headed`

para que el test sea observable agregamos el archivo playwright.config.ts las siguientes líneas:

```json
use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    // baseURL: 'http://127.0.0.1:3000',
    launchOptions: {
      slowMo: 1000,
    },
```
si se desea correr otra carpeta, la colocamos en playwright.config.ts el export:
```json
export default defineConfig({
  testDir: './tests-examples',
```

También se pueden realizar los todos los test si comentamos testDir y lo iniciamos con el siguiente código `npx playwright test tests`

más información en [Running and debugging tests](https://playwright.dev/docs/running-tests "Running and debuting")

## Selectores

para observar cual selector tome utilizo la consola de desarrollador he uso el siguiente codigo document.querySelector("selector"), por ejemplo: `document.querySelector("a.nav-link")`

```javascript
import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('http://uitestingplayground.com/');
  //await page.getByRole('link', { name: 'Resources' }).click();
  await page.click('a.nav-link:has-text("Resources")'); // año 2024 en junio
  //await page.locator('a.nav-link:has-text("Resources")').click();
  await page.getByRole('link', { name: 'Home' }).click();
  await page.getByRole('link', { name: 'Click' }).click();
  await page.getByRole('button', { name: 'Button That Ignores DOM Click' }).click();
});
```

[Platzi: Cursos online profesionales de tecnología](https://platzi.com/clases/1665-preprocesadores/22294-selectores-de-css/ "Platzi: Cursos online profesionales de tecnología")

[Element selectors | Playwright](https://www.cuketest.com/playwright/docs/selectors/ "Element selectors | Playwright")

## Más sobre selectores

### Planeación de Pruebas

Al desarrollar cualquier prueba que se desee evaluar, se debe tomar a consideración el contexto con el que la característica o módulo se encuentra.

**Por ejemplo:**

- En una prueba de componente, deseamos evaluar la funcionalidad según su aportación, sea en una página o distribución de un sección.
- Por su parte, en una prueba E2E (End-2-End, Punto a punto), buscamos evaluar un todo según los requerimientos impuestos por nuestras historias de usuario.

Sin importar el caso, se debe tomar a consideración la anatomía de las pruebas para tener una guía y establecer una estructura de trabajo.

### Redactando pruebas E2E
Supongamos que tenemos una lista de actividades (TODO List). En ella, deseamos evaluar que:

Dada una nueva actividad, en una lista vacía, se deberá mostrarse una vez ingresada
Dada una nueva activad, en una lista con al menos 1 actividad, se deberá mostrarse al final de la lista.

De lo anterior, consideremos:

1. Gestionar la configuración de Playwright para pre establecer la ruta base:

`export default defineConfig({ // ... use: { // ... baseURL: "https://demo.playwright.dev", // ... }, // ... })`

2. Definir los hooks de ciclo de nuestro pool de pruebas:

```javascript
import { test, expect, type Page } from '@playwright/test'

test.beforeEach(async ({ page }) => { await page.goto('/todomvc') }) 
```

3. Establecer o requerir algún(os) mock(s) para evaluar nuestros casos:

```javascript
import { test, expect, type Page } from '@playwright/test' import todoItems from './mocks/todoItems.mock.ts'

test.beforeEach(async ({ page }) => { await page.goto('/todomvc') })

export default [ 'Clean the house', 'Feed the dog' ]
```

4. Agrupar y definir nuestros casos de pruebas

```javascript
import { test, expect } from '@playwright/test' import todoItems from './mocks/todoItems.mock.ts'

test.beforeEach(async ({ page }) => { await page.goto('/todomvc') })

test.describe('TODO app', () => { test('should be able to add a new item', async ({ page }) => { // Input activity // Validate text item }) test('should append the last items added to the bottom of the list', async ({ page }) => { // Input activity // Validate text item }) })
```

5. Completando los casos de pruebas

```javaascript 

import { test, expect, type Page } from '@playwright/test' import todoItems from './mocks/todoItems.mock.ts'

test.beforeEach(async ({ page }) => { await page.goto('/todomvc') });

test.describe('TODO app', () => { 
  test('should be able to add a new item', async ({ page }) => { 
	  const newTodo = page.getByPlaceholder('What needs to be done?');
		await newTodo.fill(todoItems[0]);
    await newTodo.press('Enter');
		await expect(page.getByTestId('todo-title')).toHaveText(todoItems[0]);
  })

	test('should append the last items added to the bottom of the list', async ({ page, }) => { 
		const newTodo = page.getByPlaceholder('What needs to be done?');
		await newTodo.fill(todoItems[0]);
    await newTodo.press('Enter'); 
		await newTodo.fill(todoItems[1]); 
    await newTodo.press('Enter'); 
		await expect(page.getByTestId('todo-item').nth(1)).toHaveText(todoItems[1]) 
  }); 
}); 
```

## Assertions

### Recomendaciones de pruebas Frontend

El desarrollo de pruebas, en una mal ejecución, terminará siendo un lastre que impedirá tanto al desarrollador como al producto, sincronizarse rápida con su público o audiencia de uso. . Para el caso de pruebas en Frontend, existen recomendaciones que permiten buscar funcionalidades (TDD) o comportamiento (BDD), que para fines de la necesidad resultará en el valor que aportará en el mercado. .

### Separa la interfaz de usuario de la funcionalidad

- En desarrollo, al planificar la arquitectura de aplicaciones Frontend, la mayoría recae en la generación de componentes e interfaces visuales.
- Un desarrollo, se deberá pensar en su desacoplamiento de elementos interaccionables con el usuario y su disposición en una página web.
- Metodologías como BEM o Atomic Design, permiten su distribución de dichas entidades. Sin embargo, queda en incógnita la arquitectura general de la aplicación.Por ejemplo, el uso de librerías como ReactJS, prefieren dejar una abstracción de Hooks o contextos como una subcarpeta más que una arquitectura de controladores y servicios.

### Consultar elementos HTML basados en atributos que es poco probable que cambien

- En desarrollo, un componente poseerá un estructura con entradas y salidas que permitirán abstraer su compleja funcionalidad para después manipular su comportamiento, modularmente. 

**Definición** Un **componente** es una entidad que interacciona con un usuario; también posee complejidad dependiente de entradas variables. Su diseño deberá ser autónoma para controlar información, errores y ambigüedad .

- En diseño, el desarrollador deberá identificar y estructurar la jerarquía de atributos esenciales de sus componentes para que, en pruebas, pueda evaluar escenarios primarios (requerimientos de usuario) y secundarios (como los consecuentes o dependientes). 

### Si es posible, desarrolla interfaces y componentes con información real

- Si bien, los wireframes y diseños finales suelen no entregarse con información y escenarios reales, el desarrollo no será la excepción.
- En desarrollo, un componente debe ser diseñado pensando en sus casos límite, los cuales definen aquellos excesos de algo. Por ejemplo, un texto muy largo, información sin formato, calidad de imágenes, etc. 
- A veces, nuestras APIs suelen no comunicarse con los desarrolladores dejando que los Frontend se las arreglen como puedan. Mockear la información es útil cuando conoces su estructura, cuando no, se interpreta, dejando la información en ambigüedad impidiendo su estabilidad al final de entrega.
- En dichos escenarios, desarrollar pruebas antes que el componente en sí permitirá reducir dichos casos de caos. Recuerda que los componentes, a veces, requieren de un paso preliminar de retroalimentación de la API, por lo que al estructurar flujos con base en datos reales, permitirán reducir “hardcore".

📌 Referencias Para más recomendaciones con código real, puedes consultarlo en [50 Best Practices](https://github.com/goldbergyoni/javascript-testing-best-practices "50 Best Practices")

```javascript
import { test, expect } from '@playwright/test';

test("test", async ({ page }) => {
    await page.goto("http://uitestingplayground.com/textinput");

    //verify input is visible
    await expect(page.locator("#newButtonName")).toBeVisible();
    //selelct input and fill the input your text
    await page.locator("#newButtonName").fill("Mario");
    //click in button
    await page.locator("#updatingButton").click();
    // verify button text update
    await expect(page.locator("#updatingButton")).toContainText("Mario");
    
});

```

### Información importante para resolver el reto

Hola, llegó el momento de enfrentar tu primer reto. Pero antes un ⚠️ ANUNCIO IMPORTANTE ⚠️

Desafortunadamente, el sitio web con el que resolveré el reto en la siguiente clase no está disponible por el momento. Pero no te preocupes, la consigna es “nunca pares de aprender”, así que te dejo esta web para que realices tus test: [https://automationexercise.com/category_products](https://automationexercise.com/category_products "https://automationexercise.com/category_products")

¿Qué diferencias vas a encontrar respecto a la forma yo resolveré el reto en la siguiente clase?

- Al hacer hover sobre un artículo a comprar no te aparecerá “quick view”, en vez de eso deberás dar click directamente en “view product”.
- Una vez ahí, la cantidad de productos a añadir se tendría que testear de forma distinta. Una pista es que observes el input para la cantidad de prendas en vez de ‘button-plus’ como yo lo hice.
- Deberás omitir el paso de seleccionar el tamaño de la prenda, ya que la web que estarás probando no tiene esa opción.
- Al verificar que el modal y el texto aparecen, en vez del texto “Success” simplemente usa el texto “Added!”

En resumen, podrás realizar 7 de los 9 test del reto, 2 de ellos con los ajustes mencionados arriba (❗):

![reto](/images/tests-reto-daad.png)

Espero que superes este reto utilizando el ejemplo de la siguiente clase y todo lo que hemos aprendido hasta ahora. Déjame en los comentarios de la siguiente clase cómo quedó tu código y si lograste hacer testing al elemento para incrementar el número de prendas 😉.

## Reto: escribe un test sin el uso de codegen

```javascript
import { test, expect } from '@playwright/test';

test("añadir producto al carrito", async ({ page }) => {
// ir a la url https://automationexercise.com/products
    await page.goto("https://automationexercise.com/products");
//hover del primer producto que encontremos
    await page.hover("body > section:nth-child(3) > div > div > div.col-sm-9.padding-right > div > div:nth-child(3)");
//click en el primer producto ver mas detalles
    await page.click("body > section:nth-child(3) > div > div > div.col-sm-9.padding-right > div > div:nth-child(3) a:has-text('View Product')");
    await expect(page).toHaveURL("https://automationexercise.com/product_details/1");
//click en el boton + (dos veces)
    await page.locator("#quantity").fill("3");
//seleccionar en el menu dropdown un nuevo tamaño
    //await page.locator('#group_1').selectOption({ index: 1 });
//click en boton añadir al carrito
    await page.click('body > section > div > div > div.col-sm-9.padding-right > div.product-details > div.col-sm-7 > div > span > button')
// verificar (expect) "Added!"
    await expect(page.locator('#cartModal > div > div > div.modal-header > h4')).toBeVisible();
    await expect(page.locator('#cartModal > div > div > div.modal-header > h4')).toContainText("Added!");
//click en boton continue shopping
    await page.click('#cartModal > div > div > div.modal-footer > button')
//el modal debe no ser visible
    await page.click('#header > div > div > div > div.col-sm-8 > div > ul > li:nth-child(1) > a');
});
```

[Page | Playwright](https://playwright.dev/docs/api/class-page#page-hover "Page | Playwright")

[Automation Exercise - All Products](https://automationexercise.com/products "Automation Exercise - All Products")

## Playwright inspector

se deseamos ver cada paso de la prueba utilizamos el inspector con el sigiente codigo: `npx playwright test <test a ejecutar> --debug` ejemplo: `npx playwright test assert --debug`.

## Debugging selectors

Para no tener que estar escribiendo siempre npx playwright test, esto se puede hacer un script para que solo tengas que escribir npm run test. . Para hacer esto en el archivo package.json, en la seccion de scripts, agregamo el siguiente script:


```javascript
"scripts": {
  "test": "npx playwright test"
}
```
Con esto ya puedes correr solamente npm run test, en lugar de correr el script original. Sin embargo, muchas veces vamos a querer correr solamente un archivo en especifico. Esto lo haces de la siguiente manera:


```javascript
npm run test -- assert
// npx playwright test assert
```
En este caso le estamos pasando parametros al comando original despues del `--`

`PWDEGUG=console npx playwright test assert`

en la consola para desarrolador podemos ver todos los comandos con `playwrigh` para buscar una etiqueta se utiliza `playwright.$("h1")` en cuentar la primera y si usa `playwright.$$("h1")` muestra todas, `playwright.inspect("input")` muestra la ubicación del elementos

[Curso de Debugging con Chrome DevTools - Platzi](https://platzi.com/cursos/devtools/ "Curso de Debugging con Chrome DevTools - Platzi")

## playwright.inspect("input")

Observability en Pruebas
ℹ️ Definición Observability es la capacidad de un sistema de ser monitoreado y saber a detalle lo que sucede en un sistema.

Para monitorear un sistema, se empieza por avaluar aquellos puntos claves que describen tanto el rendimiento como su interacción con el mundo real.
En pruebas, un podemos realizar un zondeo de múltiples parámetros en los que trabajaran nuestro producto. Por ejemplo, podemos definir el tiempo de espera `timeout`, cantidad de fallo por navegador `max-failures` , etc.
Es importante aclarar el objetivo de desarrollo para un MVP, puesto que el desarrollo puede extenderse, colisionando en las pruebas como un bloqueo de entrega.
Con Playwright, podemos configurar las pruebas con las siguientes opciones para mejorar la definición de terminado y criterios de aceptación de nuestras historias de usuarios.

✨Concepto clave Puedes ver todas las opciones de ejecución mediante npx `playwright test --hel``.

. Si fuera el caso, podemos obtener toda la salida verbosa de ejecución del depurador de Playwright mediante `DEBUG_FILE=[file]`. 
Por ejemplo, una ejecución sería:

`DEBUG=pw:api DEBUG_FILE=logs.txt npx playwright test todo`

ejecuta el comando para ejecutar tus pruebas con el depurador activado:

sh
`$env:DEBUG="pw:api"; npx playwright test`
o en CMD:
`set DEBUG=pw:api && npx playwright test`
o en Git Bash/WSL:
`export DEBUG=pw:api && npx playwright test`

## Playwright Tracing

para iniciar el tracing se utiliz ael codigo: `npx playwright test tiendaonline --trace on`

[Trace Viewer | Playwright](https://playwright.dev/docs/trace-viewer)

[trace.playwright.dev.](https://playwright.dev/docs/trace-viewer#:~:text=trace.playwright.dev.%20Make,npx%20playwright)

## Reparar un test que no funciona

reto 

[GitHub - platzi/curso-automatizacion-pruebas-playwright-reto2](https://github.com/platzi/curso-automatizacion-pruebas-playwright-reto2)