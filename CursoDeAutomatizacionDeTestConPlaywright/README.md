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