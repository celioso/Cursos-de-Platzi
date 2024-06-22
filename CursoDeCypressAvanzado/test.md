### Curso de Cypress Avanzado

1. **¿Cuál es el comando con el que podemos setear una cookie?**

**R/:** cy.setCookie(name, value)cy.setCookie(name, value)

2. **¿Cuál es el comando con el que podemos obtener una cookie?**

**R/:** cy.getCookie(name)

3. **¿Cuál es el comando para obtener un valor del localstorage?**

**R/:** localStorage.getItem('prop1')

4. **¿Cuál es el comando para limpiar el localstorage?**

**R/:** cy.clearLocalStorage('prop1')

5. **¿Cypress nos permite emular dispositivos de forma nativa?**

**R/:** No, nos permite emular el webview o los headers pero no el dispositivo de forma nativa.

6. **¿Cypress nos ofrece una forma de usar xpaths?**

**R/:** No, tenemos que instalar un plugin.

7. **¿Para qué nos sirve la funcionalidad del retry?**

**R/:** Para evitar y mitigar los flaky tests.

8. **¿Cuál es el comando para interceptar requests en cypress?**

**R/:** cy.intercept(method, url)

9. **¿Qué significa POM?**

**R/:** Page Object Model

10. **¿Para qué nos sirven los Custom commands?**

**R/:** Nos sirven para abstraer funcionalidad y para sobreescribir comandos preexistentes.

11. **¿Para qué nos sirven las variables de entorno?**

**R/:** Para proteger información sensible y para personalizar nuestros datos de acuerdo con diferentes ambientes.

12. **¿Cómo se llama la librería que instalamos para hacer visual testing?**

**R/:** cypress-image-snapshot

13. **¿Para qué sirven los fixtures?**

**R/:** Son una forma de almacenar información que nos permite hacer Data Driven Tests.

14. **¿Qué librería instalamos para trabajar con BDD?**

**R/:** @badeball/cypress-cucumber-preprocessor

15. **¿Qué tipo de escenarios nos ayudan para hacer Data Driven Tests con BDD?**

**R/:** Escenarios Outline

16. **¿Cómo se llama la librería que es una alternativa al dashboard de Cypress?**

**R/:** sorry-cypress

17. **¿Cuál es la imagen de docker que usamos como base?**

**R/:** cypress/base:16

18. **¿Qué comando invocamos sobre el elemento para poder abrir la pestaña dentro del mismo sitio y no navegar a otra ventana?**

**R/:** .invoke('removeAttr', 'target')

19. **¿Con qué comando podemos cambiar el viewport?**

**R/:**  cy.viewport(1280, 720);

20. **¿Cuál es el comando que utilizamos para hacer la aserción visual?**

**R/:**  cy.matchImageSnapshot();