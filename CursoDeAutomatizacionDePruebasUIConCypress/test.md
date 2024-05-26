## Curso de Automatización de Pruebas UI con Cypress

1. **Una ventaja de Cypress es que permite trabajar con múltiples navegadores al mismo tiempo.**

**R/:** Falso

2. **¿Cypress nos permite hacer pruebas unitarias de integración y end-to-end.**

**R/:** Verdadero

3. **Cypress ya trae por defecto pruebas hechas para que puedas utilizarlas o adaptarlas para tu uso.**

**R/:** Verdadero

4. **NO es una manera de ir hacia la página anterior:**

**R/:** cy.back()

5. **NO es una manera de localizar un elemento con Cypress nativo:**

**R/:** Unittest, PyUnitReport, DDT

6. **¿Es posible obtener el mismo elemento a partir de diferentes localizadores?**

**R/:** Si

7. **¿Qué comando podemos usar para "Guardar un elemento" y poder aplicar otros comandos?**

**R/:** .then

8. **Son palabras llave para crear aserciones:**

**R/:** expect,assert,should

9. **¿Cuál de estos NO son hooks en Cypress?**

**R/:** it y describe

10. **¿Quá debemos de hacer para poder usar el debugger?**

**R/:** Abrir las devtools sino no podremos verlo en efecto.

11. **¿Cuál es la ventaja de correr una prueba en modo headless?**

**R/:** Es más rápida la ejecución y puede ahorrarnos incluso costos de operacion, si corremos nuestras pruebas en un CI/CD.

12. **¿Cypress solo permite crear videos y capturas de pantalla en el modo headless?**

**R/:** Verdadero

13. **Utilizar el flag '-browser chrome' especifica en qué navegador va a correr la prueba.**

**R/:** Verdadero

14. **No es posible elegir solo una prueba para correr de todo tu script.?**

**R/:** Falso

15. **¿Un id dinámico puede servir como identificador de un elemento?**

**R/:** Falso

16. **¿it.only permite correr solo esta prueba?**

**R/:** Verdadero

17. **¿force:true permite forzar una acción, por ejemplo el click?**

**R/:** Verdadero

18. **En un dropdown puedes hacer aserciones basadas en valor, texto o indice.**

**R/:** Verdadero

19. **¿Para qué sirve cy.on?**

**R/:** Para escuchar eventos como alertas.

20. **¿Para qué sirve cy.stub()?**

**R/:** Para reemplazar una función, registrar su uso y controlar su comportamiento.

21. **¿Para qué sirve el sufijo 'spec' en el título de un archivo de prueba?**

**R/:** Para que la prueba pueda ser reconocida como tal en Cypress.

22. **¿Por qué es importante probar el proceso de recarga una página?**

**R/:** Es importante ya que, a veces, cambios que realizamos no se veran refleajdos hasta que recarguemos la página, o para validar que los cambios en realidad se hayan guardado.

23. **NO es una manera de ir hacia la página anterior:**

**R/:** cy.back()

24. **¿Es posible hacer un debug en el log en el modo headless?**

**R/:** Verdadero.

25. **¿Cómo Cypress especifica que quiere correr las pruebas en todos navegadores excepto Chrome?**

**R/:** describe( 'Titulo de tu prueba', {browser: '!chrome'}, () =>{})

26. **¿Para qué sirve cy.stub()?**

**R/:** Para reemplazar una función, registrar su uso y controlar su comportamiento.

27. **¿Cómo limpiar el input de un formulario?**

**R/:** cy.get('#id').type('{selectAll}{backspace}') o cy.get('#id').clear()

28. **¿Qué comando debemos de usar para forzar el check si hay un elemento que esta sobre el input?**

**R/:** cy.get('#id').check({ force: true })

29. **¿Para qué sirve un alias?**

**R/:** Son de utilidad para selectores con nombres grandes y para poder hacer referencia a estos elementos y la reutilización del código.

30. **¿Qué comando utilizamos para llenar una fecha?**

**R/:** Utilizamos .type() porque el data picker es un input.

31. **Diferente de Cypress, Selenium da soporte para varios navegadores. Eso permite que pruebes tu proyecto en diferentes navegadores y es una ventaja de Selenium sobre Cypress.**

**R/:** Verdadero
