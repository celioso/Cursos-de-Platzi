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
Una vez elegido nuestras preferencias de configuración, el asistente anexará un conjunto de carpetas donde residirán nuestras pruebas. . De lo nuevo, lo más destacable será `playwright.config` . Este archivo, permitirá manipular la configuración de Playwright. .

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