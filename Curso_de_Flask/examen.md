# Curso de Flask
## examen

1. **¿Qué debes conocer para comenzar con Flask?**

**R//= ** Conocimientos básicos de python, pip y virtualenv.

2. **Una aplicación web utiliza el internet y un __ para comunicarse con el servidor.**

**R//= **Navegador web

3. **¿Para qué sirve Flask?**

**R//= **Todas las anteriores

4. **¿A qué nos referimos con microframework?**

**R//= **Un framework que no cuenta inicialmente con funcionalidades especificas, como ORM o autenticacion

5. **¿Con qué comando creamos una nueva instancia de Flask?**

**R//= **`app = Flask(__name__)`

6. **¿Qué variable hay que crear en la terminal para activar el debugger y reloader?**

**R//= **`FLASK_DEBUG=1`

7. **¿Qué variable hay que declarar en la terminal para prender el servidor de Flask?**

**R//= **`FLASK_APP=main.py`

8. **¿Con qué comando prendemos el servidor local?**

**R//= **`flask run`

9. **Nombre de la variable que Flask expone para acceder a la información de la petición del usuario**

**R//= **`request`

10. **¿Cuál es la sintaxis correcta para iniciar un bloque condicional?**

**R//= **`{%`

11. **¿Cuál es la sintaxis correcta para representar una variable?**

**R//= **`{{ variable }}`

12. **¿Cuál es la función correcta para crear un link interno a una ruta específica?**

**R//= **url_for()

13. **¿Cómo se llama el directorio donde Flask busca archivos estáticos por defecto?**

**R//= **`static`

14. **¿Cuál es el decorador para crear una función para manejar errores?**

**R//= **`@app.errorhandler(error)`

15. **¿Cuál es el template inicial que tenemos que extender en Bootstrap?**

**R//= **`´bootstrap/base.html'`

16. **Para desplegar una forma y encriptar la sesiones, debemos de declarar esta variable en app.config:**

**R//= **SECRET_KEY

17. **¿Qué es un flash?**

**R//= **Un mensaje que presenta informacion al usuario sobre la accion que acaba de realizar.

18. **Nombre del método que tenemos que implementar en una nueva instancia de `flask_testing.TestCase`**

**R//= **`create_app`

19. **¿Cuál es la variable que expone flask_wtf.FlaskForm para validar formas cuando son enviadas y qué tipo de variable es?**

**R//= **`validate_on_submit, boolean`

20. **¿Cómo debemos cuidar o manejar nuestro SECRET_KEY de producción?**

**R//= **No debe estar disponible en nuestro repositorio.

21. **¿Para qué nos sirve un Blueprint?**

**R//= **Para modularizar la aplicación, son un patrón de rutas, funciones y templates que nos permiten crear secciones de la aplicación.

22. **Después de crear un nuevo Blueprint, ¿cómo lo integramos en la aplicación?**

**R//= **llamamos la funcion app.register_blueprint() y pasamos nuestra nueva instancia de Blueprint como parametro

23. **Sintaxis correcta para declarar una ruta dinámica "users" que recibe "user_id" como parámetro**

**R//= **`/users/<user_id>`

24. **¿Cuál es el comando que agregamos después de instalar GCloud SDK?**

**R//= **`gcloud`

25. **¿Qué tipo de base de datos es Firestore?**

**R//= **No SQL Orientada a Documentos

26. **Flask-Login requiere la implementación de una clase UserModel con propiedades específicas.**

**R//= **Verdadero

27. **¿Para qué utilizamos `@login_manager.user_loader`?**

**R//= **En la función decorada implementamos una búsqueda a la base de datos para cargar los datos del usuario.

28. **Variable que usamos para detectar si el usuario está firmado. Disponible en cualquier template.**

**R//= **`current_user.is_authenticated`

29. **¿Cómo debemos guardar un password del usuario?**

**R//= **Nunca en el texto original. Debemos utilizar una función para cifrarlo de manera segura, solo el usuario debe saber el valor.

30. **¿Cómo se llama el archivo de configuración de AppEngine?**

**R//= **app.yaml