# Curso de Fundamentos de Pruebas de Software

1. **¿Cuál es la ventaja de utilizar waitForSelector y no waitForTimeout?**

**R/:** Esperamos por un elemento en específico y no solo un tiempo muerto

2. **¿Cómo esperamos por un selector tipo Xpath?**

**R/:** await page.waitForXPath(selector)

3. **Click solo recibe selectores del tipo:**

**R/:** CSS

4. **¿Qué es puppeteer?**

**R/:** Puppeteer es una libreria de Node que proporciona una API de alto nivel para controlar Chrome en modo headless a través del protocolo DevTools.

5. **¿Qué significa el modo headless?**

**R/:** Headless es la opción que nos permite ocultar o mostrar el navegador.

6. **¿Cuál es comando para navegar a una página?**

**R/:** await page.goto('https://yahoo.com/')

7. **¿Cómo mandamos un doble click?**

**R/:** await page.click(selector, { clickCount: 2 })

8. **¿Cómo extraemos el texto de un elemento?**

**R/:** return await page.$eval(selector, (el) => el.textContent)

9. **Este hook nos permite ejecutar código antes de cada test**

**R/:** beforeEach

10. **Este hook nos permite ejecutar código antes de todos los tests**

**R/:** beforeAll

11. **Qué hace el siguiente comando page.setDefaultNavigationTimeout(10000)**

**R/:** Esta configuración cambiará el tiempo máximo de navegación predeterminado por 10000 milisegundos:

12. **¿Qué hace el comando slowMo?**

**R/:** Ralentiza las operaciones de puppeteer en la cantidad especificada de milisegundos para ayudar a la depuración.

13. **El waitForSelector solo puede recibir un selector css, esto es:**

**R/:** Verdadero, solo admite este tipo de selector

14. **¿Cómo navegamos hacia atras?**

**R/:** await page.goBack()

15. **Para que sirve el comando await page.bringToFront()**

**R/:** Sirve cuando tenemos dos páginas o más y queremos traer la página al frente (activa la pestaña) para interactuar con ella.