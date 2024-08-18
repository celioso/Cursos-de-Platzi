# Curso de Django

## ¿Qué es Django?

Django es un framework para desarrollo web escrito en Python que inicialmente fue utilizado para crear blogs, pero ha evolucionado para soportar aplicaciones complejas, como las primeras versiones de Instagram y Spotify. Su popularidad se debe a su facilidad de uso y la rapidez con la que permite desarrollar aplicaciones funcionales.

### ¿Cuáles son los requerimientos previos para aprender Django?

- Conocer Python, ya que Django está construido en este lenguaje.
  - Sintaxis básica: if, for, definición de variables.
- Comprender la programación orientada a objetos.
  - Reutilización de código mediante clases y herencia.
- Conocer HTML para diseñar la interfaz de usuario.
- Conocimientos básicos de CSS para estilizar la aplicación.

### ¿Por qué es importante usar entornos virtuales en Django?

Los entornos virtuales permiten gestionar diferentes versiones de paquetes y librerías en un mismo equipo sin conflictos. Esto es crucial cuando se trabaja en múltiples proyectos que requieren distintas versiones de Django o cualquier otro paquete de Python.

#### ¿Cómo se crea un entorno virtual en Python?
1. Abre la terminal en tu editor de código preferido, como Visual Studio Code.
2. Crea una carpeta para tu proyecto y ábrela en el editor.
3. Usa la librería venv de Python para crear un entorno virtual:
`python -m venv ~/path_to_your_folder/.venvs/my_first_env`
4. Verifica la creación del entorno con ls en la carpeta especificada.

### ¿Cómo se activa un entorno virtual?

Para activar el entorno virtual y asegurarte de que los comandos se ejecuten en este entorno específico:

`source ~/path_to_your_folder/.venvs/my_first_env/bin/activate`

Notarás que el nombre del entorno virtual aparece en la terminal, indicando que está activo.

### ¿Qué significa tener un entorno virtual activo?

Significa que cualquier comando que ejecutes utilizará las librerías instaladas en ese entorno específico, evitando conflictos con otras versiones de librerías que puedas tener en tu sistema. Esta práctica es esencial para evitar colisiones y mantener un entorno de desarrollo limpio y manejable.

## ¿Cómo instalar Django?

Para instalar Django, primero asegúrate de tener un entorno virtual configurado. Luego, usa el comando pip install django para instalarlo. Si no especificas la versión, pip instalará la última disponible compatible con tu versión de Python.

Al ejecutar este comando, verás que Django se instala junto con sus dependencias, necesarias para su funcionamiento. Esto es porque Django reutiliza librerías existentes para ciertas funcionalidades.

### ¿Qué es el comando django-admin y cómo se usa?

Una vez instalado Django, obtienes acceso al comando django-admin, que es una herramienta de línea de comandos para administrar tareas relacionadas con Django. Para ver todos los subcomandos disponibles, puedes ejecutar `django-admin help`.

### ¿Cómo crear un proyecto con django-admin?

El subcomando que más nos interesa es startproject, que se usa para crear un nuevo proyecto Django. Para hacerlo, ejecuta:

`django-admin startproject nombre_del_proyecto`

Asegúrate de no usar guiones en el nombre del proyecto, ya que Django interpretará eso como un intento de resta en Python. Usa guiones bajos en su lugar.

### ¿Qué archivos se crean con startproject?

El comando startproject crea una nueva carpeta con el nombre del proyecto. Dentro de esta carpeta, encontrarás:

- Una subcarpeta con configuraciones del proyecto.
- Un archivo manage.py, que sirve para ejecutar comandos específicos del proyecto.

### ¿Cómo usar [manage.py](http://manage.py/ "manage.py")?

El archivo manage.py se utiliza para comandos que afectan solo al proyecto actual. Para ver los comandos disponibles, ejecuta:

`python manage.py help`

¿Cómo ejecutar el servidor de desarrollo?
Para ver tu aplicación en funcionamiento, usa el comando `runserver`:

`python manage.py runserver`

Este comando inicia un servidor de desarrollo y te indica la URL y el puerto donde tu aplicación está corriendo. Puedes abrir esta URL en tu navegador para verificar que todo está configurado correctamente.

[Installing an official release with pip](https://docs.djangoproject.com/en/5.0/topics/install/#installing-official-release "Installing an official release with pip")

### Entendiendo la arquitectura de Django

La arquitectura del framework está diseñada para ser reutilizable y organizar todas tus tareas. Utiliza el modelo MVT (Model, View, Template).

### ¿Qué es el modelo en MVT (Model, View, Template)?

El modelo es la parte de los datos:

- Guarda y procesa los datos.
- Contiene la lógica del negocio, como una calculadora que suma 2 más 2.

### ¿Qué es la vista en MTV?

La vista actúa como un conector:

- Accede y dirige los datos.
- Controla el flujo de peticiones y respuestas.
- Verifica permisos y realiza comprobaciones necesarias.

### ¿Qué es el template en MTV?

El template maneja la parte gráfica:

- Usa HTML y CSS para mostrar los datos.
- Por ejemplo, muestra una lista de zapatos almacenada en el modelo.

### ¿Cómo interactúan modelo, vista y template?

El flujo de datos es el siguiente:

- El modelo pasa datos a la vista en un array.
- La vista pasa esos datos al template en un contexto.
- El template muestra los datos gráficos.

En sentido contrario:

- Un usuario busca en el template.
- La vista recibe la búsqueda y consulta al modelo.
- El modelo devuelve los resultados a la vista.
- La vista envía los datos al template para mostrarlos.

**Nota:** No debe haber conexión directa entre template y model. Siempre usa la vista para asegurar verificaciones y permisos.

Django es un framework de desarrollo web en Python que sigue el patrón de diseño **MTV (Model-Template-View)**, una variante del conocido patrón **MVC (Model-View-Controller)**. Aquí te explico cómo se organiza la arquitectura de Django:

### 1. **Modelos (Models)**
   - Representan la estructura de datos de tu aplicación. Los modelos en Django se definen como clases de Python que heredan de `django.db.models.Model`. Cada modelo se traduce en una tabla en la base de datos, y cada instancia del modelo representa una fila en esa tabla.
   - Aquí defines los campos y comportamientos de los datos que deseas almacenar, como los tipos de datos y relaciones entre modelos.

### 2. **Plantillas (Templates)**
   - Son archivos HTML que definen la presentación de los datos. Django usa un sistema de plantillas propio que permite insertar contenido dinámico, como variables y estructuras de control, en los archivos HTML.
   - Las plantillas son responsables de cómo se renderizan los datos en el navegador del usuario.

### 3. **Vistas (Views)**
   - Las vistas son funciones o clases que manejan la lógica de la aplicación. Son responsables de recibir una solicitud HTTP, interactuar con los modelos para obtener los datos necesarios y devolver una respuesta HTTP adecuada (como una página HTML, un JSON, etc.).
   - Las vistas en Django se asocian a URL específicas a través del archivo `urls.py`.

### 4. **Controlador (Controller)**
   - En el caso de Django, el controlador no es un componente explícito como en el patrón MVC tradicional. La lógica del controlador se divide entre las vistas y el sistema de despacho de URLs (`urls.py`), que asocia las URLs de la aplicación con las vistas correspondientes.

### 5. **URL Dispatcher**
   - El archivo `urls.py` es donde defines las rutas (URLs) de tu aplicación. Aquí asocias cada URL con una vista específica. Django utiliza un sistema de expresiones regulares para esta asociación.

### 6. **Migrations**
   - Django incluye un sistema de migraciones para gestionar los cambios en la estructura de la base de datos. Las migraciones son archivos generados automáticamente o manualmente que Django utiliza para sincronizar el esquema de la base de datos con los modelos definidos en el código.

### 7. **Admin Interface**
   - Django ofrece un panel de administración listo para usar que permite a los desarrolladores y administradores gestionar los datos de la aplicación sin necesidad de crear interfaces específicas.

### 8. **Middlewares**
   - Son componentes que procesan las solicitudes HTTP antes de que lleguen a las vistas o después de que las respuestas se envíen al cliente. Puedes utilizarlos para tareas como autenticación, manejo de sesiones, o gestión de errores.

### 9. **Formularios (Forms)**
   - Django tiene un sistema de formularios que facilita la creación, validación y procesamiento de formularios HTML. Los formularios pueden estar vinculados a modelos (ModelForms) para una integración más sencilla con la base de datos.

### Ejemplo de Flujo en Django:
1. El usuario envía una solicitud a una URL.
2. El **URL dispatcher** dirige la solicitud a la vista adecuada.
3. La **vista** interactúa con los **modelos** para obtener o manipular datos.
4. La vista pasa los datos a una **plantilla** para su renderización.
5. La plantilla genera una respuesta HTML que se envía de vuelta al usuario.

Esta arquitectura permite que el desarrollo sea modular y escalable, lo que facilita la creación y mantenimiento de aplicaciones web complejas.

[Glossary | Django documentation | Django](https://docs.djangoproject.com/en/5.0/glossary/#term-MTV "Glossary | Django documentation | Django")