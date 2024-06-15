# Curso de Automatización de Pruebas de Backend con Cypress -text

1. **¿Qué tipos de APIs existen?**

**R/:** SOAP y REST

2. **Cómo se llaman las herramientas usada para inspeccionar las peticiones de red?**

**R/:** Chrome Dev tools

3. **¿Qué librería utiliza Cypress para manejar peticiones?**

**R/:** Ninguna, Cypress ya incluye lo necesario para manejar los requests.

4. **La siguiente línea de código debe retornar: cy.request('employees').its('headers').its('content-type').should('include', 'application/json)**

**R/:** Si el retorno es un JSON o que el contenido es un JSON.

5. **¿Cuál es el status code que se utiliza cuando una consulta se ejecutó con éxito?**

**R/:** 200

6. **¿Cuál es el status code que se utiliza cuando un registro se creó con éxito?**

**R/:** 201

7. **Para probar si "Eschweiler" es el apellido (last_name) de un empleado, la response debe ser:**

**R/:** expect(response.body.last_name).to.be.equal('Eschweiler')

8. **¿Qué propiedad podemos agregar para probar los errores?**

**R/:** failOnStatusCode: false

9. **Los errores nunca tienen un body en la respuesta solo poseen un status code.**

**R/:** Falso, también puede regresar un body la respuesta de la petición

10. **El método que permite modificar sin eliminar la información es:**

**R/:** PUT

11. **¿Qué comando nos da cypress para hacer peticiones?**

**R/:** cy.request

12. **¿Se necesita una librería extra para manejar las peticiones de GraphQL ?**

**R/:** No, cy.request es suficiente ya que Graphql trabaja sobre REST

13. **¿Cuál verbo HTTP se usa para hacer las peticiones de GraphQL?**

**R/:** POST

14. **Cypress nos da comandos necesarios para conectarnos a bases de datos**

**R/:** Falso, pero podemos crear nuestros plugins para usar las librerías nativas de las bases de datos de una mejor manera

15. **Los 'serverStatus' sirven para indicar el status del comando MySQL. Por ejemplo, serverStatus de valor 2 significa que la query fue exitosa.**

**R/:** Verdadero.

16. **¿Podemos crear una función que reciba el query como parámetro para las bases de datos no relacionales?**

**R/:** Falso por la naturaleza de las BD no relacionales , tenemos que crear funciones en específico para las acciones que queramos realizar

17. **¿Qué se puede hacer para probar de forma más completa el backend de un proyecto?**

**R/:** Unir la prueba de nuestra API vs nuestra Base de Datos

