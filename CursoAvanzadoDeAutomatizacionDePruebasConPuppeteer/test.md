# Curso Avanzado de Automatización de Pruebas con Puppeteer - test

1. **Para emular un dispositivo forzosamente debo de usar puppeteer.devices**

**R/:** Falso

2. **¿Con qué comando puedo pasar mi configuración directamente para emular un dispositivo?**

**R/:** await page.emulate({ name: 'Emulando dispositivo', viewport: { width: 375, height: 667, deviceScaleFactor: 2, isMobile: true, hasTouch: true, isLandscape: false, }, userAgent: 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36', })

3. **Comando que me permitiría hacer una captura de pantalla completa**

**R/:** await page.screenshot({ path: './capturaDePantalla.png', fullPage: true, })

4. **Comando que me permite hacer una captura de pantalla seleccionando un área en específica**

**R/:** await page.screenshot({ path: './capturaDePantallaRecortada.png', clip: { x: 0, y: 0, width: 500, height: 500, }, })

5. **Comando que me permite obtener la accesibilidad con puppeteer**

**R/:** await page.accessibility.snapshot()

6. **Librería utilizada para ayudarnos con la accesibilidad**

**R/:** @axe-core/puppeteer

7. **¿Con qué comando podemos cambiar nuestra geolocalización?**

**R/:** await page.setGeolocation({ latitude: 90, longitude: 20 })

8. **¿Cuál es el lenguaje que se utiliza normalmente con BDD?**

**R/:** Gherkin

9. **Lenguaje que nos sirve tanto para BDD, como para historias de usuario**

**R/:** Gherkin

10. **Son las 3 palabras fundamentales reservadas de Gherkin (sin embargo, no las únicas)**

**R/:** Given, When, Then

11. **Es un patrón de diseño que se utiliza en pruebas automatizadas para evitar código duplicado y mejorar el mantenimiento de nuestras pruebas**

**R/:** Page Object Model

12. **¿De qué manera podemos hacer data driven test con Gherkin?**

**R/:** Scenarios Outline

13. **¿Qué usamos para poder hacer uso de la nueva sintaxis de Javascript ES6?**

**R/:** no es jest ni jest-puppeteer

14. **¿Se puede ejecutar pruebas con Firefox?**

**R/:** Sí se pueden ejecutar nativamente

15. **¿Cómo podemos medir el performance de la automatización (runtime metrics of Chrome Devtools)?**

**R/:** no es const metrics = await page.evaluate(() => JSON.stringify(window.performance))

16. **¿Cómo podemos medir el performance de la página?**

**R/:** const metrics = await page.evaluate(() => JSON.stringify(window.performance))

17. **¿Cómo podemos medir el "page load"?**

**R/:** await page.tracing.start({ path: 'profile.json' }) await page.goto('https://google.com') await page.tracing.stop()

18. **Comando para crear el contexto para ejecutar pruebas en modo incógnito**

**R/:** const context = await browser.createIncognitoBrowserContext()

19. **¿Cuál es el nombre del reporteador que usamos con codeceptJS?**

**R/:** allure

20. **¿Qué librería usamos para habilitar el visual testing?**

**R/:** jest-image-snapshot

21. **¿Cuál es el failureThresholdType por defecto sino se agrega en la configuración?**

**R/:** pixel

22. **¿Se puede hacer visual testing emulando dispositivos?**

**R/:** Sí. Se hace de la misma manera que se haría con puppeteer

23. **¿Podemos agregar nuestros estilos, cuando generamos un PDF?**

**R/:** Sí, usando CSS

24. **¿Qué tenemos que hacer para agregar el número de páginas en el footer de un pdf?**

**R/:** Solo usar las clases pageNumber y totalPages, Puppeteer lo hará automáticamente