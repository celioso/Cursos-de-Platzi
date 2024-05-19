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