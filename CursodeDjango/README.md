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

### Qué es el patrón MVT (Model, View y Template)

### ¿Cómo se definen los modelos en Django?

Los modelos en Django se utilizan para guardar datos. Crearemos una clase llamada Carro, que hereda de models.Model. Esta clase tendrá un campo title de tipo models.TextField, con un max_length definido para limitar la cantidad de texto que puede aceptar.

```python
from django.db import models

class Carro(models.Model):
    title = models.TextField(max_length=255)
```

### ¿Cómo se definen las vistas en Django?

Las vistas en Django se encargan de buscar datos y devolverlos al template. Una vista se define como un método que recibe un request y retorna una response. Usaremos render para pasar el request y el template a la vista.

```python
from django.shortcuts import render

def myView(request):
    car_list = [{'title': 'BMW'}, {'title': 'Mazda'}]
    context = {'car_list': car_list}
    return render(request, 'myFirstApp/carlist.html', context)
```

### ¿Cómo se crean y utilizan los templates en Django?

Los templates son archivos HTML que reciben datos de las vistas. Para que Django los reconozca, creamos una carpeta llamada templates dentro de nuestra aplicación y luego otra con el nombre de la aplicación. Dentro, creamos el archivo `carlist.html`.


```html
<html>
<head>
    <title>Car Listtitle>
head>
<body>
    <h1>Lista de Carrosh1>
    <ul>
    {% for car in car_list %}
        <li>{{ car.title }}li>
    {% endfor %}
    ul>
body>
html>
```

### ¿Cómo se registran las aplicaciones en Django?

Para que Django reconozca nuestra nueva aplicación, debemos agregarla a la lista INSTALLED_APPS en el archivo settings.py.

```html
INSTALLED_APPS = [
    ...
    'myFirstApp',
]
```

### ¿Cómo se configuran las URLs en Django?

Creamos un archivo urls.py en nuestra aplicación y definimos la ruta para nuestra vista. Luego, incluimos esta configuración en el archivo urls.py principal del proyecto.

```python
from django.urls import path
from .views import myView

urlpatterns = [
    path('carlist/', myView, name='carlist'),
]
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('myFirstApp/', include('myFirstApp.urls')),
]
```

### ¿Cómo se conectan las vistas y templates en Django?

Pasamos los datos desde la vista al template usando un contexto. En el template, usamos etiquetas Django para iterar sobre los datos y mostrarlos.


```python
{% for car in car_list %}
    
  {{ car.title }}

{% endfor %}
```

El patrón MVT (Model-View-Template) es un patrón de arquitectura utilizado en el desarrollo de aplicaciones web, particularmente en el marco de trabajo Django para Python. Es similar al patrón MVC (Model-View-Controller) pero tiene algunas diferencias clave. Aquí se describe cada componente:

1. **Model (Modelo):**
   - Es la capa que maneja la lógica de negocio y la interacción con la base de datos. Define la estructura de los datos, los comportamientos y las relaciones entre los datos.
   - En Django, los modelos se definen como clases de Python que heredan de `django.db.models.Model`. Cada clase representa una tabla en la base de datos, y los atributos de la clase representan las columnas de esa tabla.

2. **View (Vista):**
   - En MVT, la vista es responsable de la lógica de la aplicación y de procesar las solicitudes del usuario. Interactúa con el modelo para obtener los datos necesarios y selecciona la plantilla adecuada para renderizar la respuesta.
   - En Django, una vista se define como una función o clase que recibe una solicitud web y devuelve una respuesta web. Las vistas pueden manejar diferentes tipos de solicitudes (GET, POST, etc.) y realizar acciones como consultar la base de datos, procesar formularios, y mucho más.

3. **Template (Plantilla):**
   - Las plantillas son la capa de presentación que se utiliza para renderizar la interfaz de usuario. Es donde se define la estructura HTML, junto con cualquier lógica de presentación, para mostrar los datos al usuario.
   - En Django, las plantillas son archivos HTML que pueden contener un lenguaje de plantillas específico (Django Template Language) para insertar datos dinámicos, realizar iteraciones, y condicionales dentro del HTML.

### Diferencias con MVC:
- En MVC, el controlador es el que maneja la lógica de la aplicación, mientras que en MVT, esta responsabilidad recae sobre la vista. 
- En MVC, la vista solo se encarga de la presentación, mientras que en MVT, la plantilla (Template) cumple este rol.

### Ejemplo de Flujo en Django:
1. **Solicitud**: El usuario hace una solicitud a la aplicación web.
2. **Vista**: La vista correspondiente maneja la solicitud, interactúa con el modelo para obtener los datos necesarios.
3. **Modelo**: Si es necesario, se consulta el modelo para obtener o modificar datos.
4. **Plantilla**: La vista selecciona una plantilla y pasa los datos obtenidos del modelo a la plantilla.
5. **Respuesta**: La plantilla genera la respuesta en HTML que se envía de vuelta al usuario.

### Introducción a Modelos y Bases de Datos

La “M” en el patrón MVC se refiere al Modelo, que es crucial para manejar datos de la base de datos en Django. En lugar de utilizar listas con datos estáticos en las vistas, ahora trabajaremos con datos provenientes del modelo, aprovechando el ORM de Django.

### ¿Qué es el ORM en Django?

El ORM (Object-Relational Mapping) en Django nos permite definir clases de Python que se relacionan directamente con las tablas de la base de datos. De esta forma, evitamos escribir sentencias SQL, ya que todo se maneja mediante Python.

### ¿Cómo se define una clase de modelo en Django?

Para definir un modelo, creamos una clase en el archivo `models.py`. Cada clase de modelo se corresponde con una tabla en la base de datos. Por ejemplo, si definimos la clase `Car`, esta se convertirá en una tabla con el nombre `Car` en la base de datos.

### ¿Qué son las migraciones en Django?

Las migraciones son un sistema que Django usa para aplicar y revertir cambios en la base de datos. Cuando creamos o modificamos un modelo, generamos migraciones que se pueden aplicar para crear o actualizar tablas en la base de datos.

**Aplicar una migración**

- Creamos la clase `Car` con un atributo `title`.
- Ejecutamos la migración hacia adelante para crear la tabla `Car` en la base de datos.
- Si agregamos un campo `year` a la clase `Car`, otra migración aplicará este cambio a la tabla.

**Revertir una migración**

- Si es necesario, podemos revertir una migración para volver al estado anterior de la tabla.
- Por ejemplo, al revertir la migración del campo `year`, la tabla `Car` quedará como antes de agregar dicho campo.

### ¿Cómo permite Django ser independiente del motor de base de datos?

Django ORM es compatible con varios motores de base de datos. En este curso, utilizaremos SQLite para ejemplos iniciales y PostgreSQL para el proyecto final.

## Gestión de Modelos y Bases de Datos en Django con SQLite

La migración de modelos en Django es un proceso fundamental para mantener la base de datos en sincronía con las clases del proyecto. Este artículo explora el uso de comandos para migrar modelos en Django, específicamente cómo manejar la migración de un modelo llamado “carro”.

### ¿Cómo identificar migraciones pendientes en Django?

Al ejecutar el comando `python manage.py runserver`, puedes encontrar un error que indica migraciones pendientes. Este mensaje significa que las tablas correspondientes a tus clases de Django no están creadas en la base de datos, lo que impide el correcto funcionamiento del proyecto.

### ¿Cómo crear migraciones en Django?

Para crear migraciones, usa el comando `python manage.py makemigrations`. Este comando genera un archivo en la carpeta de migraciones con la creación de la tabla correspondiente al modelo “carro”.

### ¿Cómo aplicar migraciones en Django?

Una vez creadas las migraciones, se deben aplicar usando `python manage.py migrate`. Esto ejecuta todas las migraciones y crea las tablas necesarias en la base de datos.

### ¿Cómo verificar la base de datos en Django?

Puedes revisar la base de datos usando `python manage.py dbshell`. Este comando te conecta a la base de datos definida en el archivo `settings.py`. En este caso, se utilizó SQLite, que es fácil de usar pero no ideal para producción debido a su baja concurrencia.

### ¿Cómo configurar la base de datos en Django?

La configuración de la base de datos se encuentra en el archivo `settings.py` bajo el diccionario `DATABASES`. Django soporta múltiples motores de base de datos como PostgreSQL, MariaDB, MySQL, Oracle y SQLite. En este curso, se utilizará PostgreSQL.

**Lecturas recomendadas**

[SQLite Documentation](https://www.sqlite.org/docs.html "SQLite Documentation")
[django-admin and manage.py | Django documentation | Django](https://docs.djangoproject.com/en/5.0/ref/django-admin/#dbshell "django-admin and manage.py | Django documentation | Django")
[Settings | Django documentation | Django](https://docs.djangoproject.com/en/5.0/ref/settings/#databases "Settings | Django documentation | Django")

### Inserción de Datos con Django

### ¿Cómo se agrega un nuevo campo a una tabla en Django?

Para agregar un nuevo campo a una tabla existente, necesitas modificar la clase del modelo correspondiente. Por ejemplo, si deseas añadir el campo “año” a la clase Carro, lo haces así:

- Añade el campo como un `TextField` con un `MaxLength` de 4, ya que solo necesitas almacenar valores como 2022, 2023, etc.

```python
class Carro(models.Model):
    ...
    año = models.TextField(max_length=4, null=True)
```

### ¿Qué pasos se siguen después de modificar el modelo?

Después de agregar el nuevo campo al modelo, sigue estos pasos:

1. **Guardar los cambios en el archivo del modelo:** No olvides guardar el archivo después de realizar modificaciones.
2. **Crear nuevas migraciones:** Ejecuta el comando `python manage.py makemigrations`. Si no detecta cambios, verifica si guardaste el archivo.
3. **Aplicar las migraciones**: Ejecuta `python manage.py migrate`. Este comando actualiza la base de datos con la nueva estructura.

### ¿Cómo se soluciona el error de campo no nulo?

Si intentas crear un campo no nulo en una tabla que ya contiene datos, Django te pedirá resolver cómo manejar los registros existentes. Puedes:

- Proveer un valor por defecto.
- Permitir valores nulos.

En este ejemplo, se permite que el campo “año” sea nulo (`null=True`), para evitar problemas con registros anteriores.

### ¿Cómo se utiliza el ORM de Django para interactuar con los datos?

Una vez aplicado el nuevo campo, puedes usar el ORM de Django para interactuar con la base de datos. Usamos el comando `python manage.py shell` para acceder al shell interactivo de Django.

**Ejemplo de cómo crear un nuevo registro:**

1. Importar el modelo:
`from my_first_app.models import Carro`

2. Crear una instancia de Carro:
`nuevo_carro = Carro(titulo='BMW', año='2023')`

3. Guardar la instancia en la base de datos:
`nuevo_carro.save()`

### ¿Cómo mejorar la visualización de los objetos en el shell?

Define el método `__str__` en tu modelo para que la representación textual del objeto sea más clara:

```python
class Carro(models.Model):
    ...
    def __str__(self):
        return f"{self.titulo} - {self.año}"
```

### ¿Cómo agregar un nuevo atributo y practicar?

Añadir un nuevo atributo, como el color del carro, sigue los mismos pasos:

1. Modifica la clase del modelo para incluir el nuevo campo.
2. Guarda el archivo.
3. Ejecuta los comandos `makemigrations` y `migrate`.
4. Utiliza el shell para crear y guardar nuevos registros con el atributo color.

## Actualización y Eliminación de Datos en Django

Para tener en cuenta! 💡

Definir el método `__str__` en los modelos de Django es una buena práctica que proporciona una representación legible y significativa del objeto, facilitando la depuración y mejorando la usabilidad de la interfaz de administración. Si no se define, se usará la representación por defecto, que es menos informativa.

## Creación y Gestión de Relaciones entre Modelos en Django

Aprender a relacionar tablas es fundamental para manejar datos interconectados en Django

### ¿Cómo crear la clase Publisher?

Para iniciar, creamos la clase `Publisher` que hereda de `models.Model`. Incluimos atributos como `name` y `address` utilizando `models.TextField` con un `max_length` de 200, un valor que puedes ajustar según tus necesidades de datos.

```python
class Publisher(models.Model):
    name = models.TextField(max_length=200)
    address = models.TextField(max_length=200)

    def __str__(self):
        return self.name
```

### ¿Cómo definir la clase Book?

La clase `Book` también hereda de `models.Model` y contiene atributos como `title`, `publication_date` y `publisher`. Utilizamos `models.DateField` para manejar fechas y establecemos una relación con `Publisher` usando `models.ForeignKey`.

```python
class Book(models.Model):
    title = models.TextField(max_length=200)
    publication_date = models.DateField()
    publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE)

    def __str__(self):
        return self.title
```

### ¿Cómo relacionar Book con Publisher usando ForeignKey?

La relación se establece con `models.ForeignKey`, donde especificamos el modelo relacionado (`Publisher`) y el comportamiento al eliminar (`on_delete=models.CASCADE`). Esto asegura que si un editor se elimina, también se eliminarán sus libros.

### ¿Cómo aplicar migraciones?

Para aplicar estos cambios a la base de datos, creamos y aplicamos las migraciones con los comandos:

```python
python manage.py makemigrations
python manage.py migrate
```

### ¿Cómo usar la shell interactiva?

Para facilitar la interacción con la base de datos, instalamos ipython con:

`pip install ipython`

Esto mejora la experiencia en la shell permitiendo autocompletar y otras funcionalidades útiles.

### ¿Cómo crear y guardar registros en la shell?

Dentro de la shell, primero creamos un `Publisher` y luego un `Book` relacionado.

```python
from myapp.models import Publisher, Book

publisher = Publisher(name="Editorial Example", address="123 Main St")
publisher.save()

book = Book(title="Two Scoops of Django", publication_date="2024-07-17", publisher=publisher)
book.save()
```

En Django, los modelos representan la estructura de los datos en tu aplicación, y cada modelo generalmente corresponde a una tabla en la base de datos. Django proporciona varios tipos de campos para definir los modelos, cada uno de los cuales se utiliza para almacenar diferentes tipos de datos. A continuación, se presentan algunos de los modelos de campo más comunes y sus usos:

### 1. **`models.CharField`**
- **Descripción**: Se utiliza para almacenar texto de longitud limitada.
- **Argumentos clave**:
  - `max_length`: Longitud máxima del campo (obligatorio).
- **Ejemplo**:
  ```python
  class Product(models.Model):
      name = models.CharField(max_length=100)
  ```

### 2. **`models.TextField`**
- **Descripción**: Se utiliza para almacenar texto largo sin límite de longitud.
- **Ejemplo**:
  ```python
  class BlogPost(models.Model):
      content = models.TextField()
  ```

### 3. **`models.IntegerField`**
- **Descripción**: Almacena enteros.
- **Ejemplo**:
  ```python
  class Order(models.Model):
      quantity = models.IntegerField()
  ```

### 4. **`models.FloatField`**
- **Descripción**: Almacena números de punto flotante.
- **Ejemplo**:
  ```python
  class Product(models.Model):
      price = models.FloatField()
  ```

### 5. **`models.DecimalField`**
- **Descripción**: Almacena números decimales precisos, generalmente utilizados para precios y cantidades monetarias.
- **Argumentos clave**:
  - `max_digits`: Número total de dígitos en el número.
  - `decimal_places`: Número de dígitos después del punto decimal.
- **Ejemplo**:
  ```python
  class Product(models.Model):
      price = models.DecimalField(max_digits=10, decimal_places=2)
  ```

### 6. **`models.BooleanField`**
- **Descripción**: Almacena valores `True` o `False`.
- **Ejemplo**:
  ```python
  class UserProfile(models.Model):
      is_active = models.BooleanField(default=True)
  ```

### 7. **`models.DateField`**
- **Descripción**: Almacena una fecha (sin hora).
- **Argumentos clave**:
  - `auto_now_add`: Establece la fecha automáticamente cuando el objeto es creado.
  - `auto_now`: Actualiza la fecha cada vez que el objeto es guardado.
- **Ejemplo**:
  ```python
  class Event(models.Model):
      event_date = models.DateField()
  ```

### 8. **`models.DateTimeField`**
- **Descripción**: Almacena una fecha y hora.
- **Argumentos clave**:
  - `auto_now_add`: Establece la fecha y hora automáticamente cuando el objeto es creado.
  - `auto_now`: Actualiza la fecha y hora cada vez que el objeto es guardado.
- **Ejemplo**:
  ```python
  class Event(models.Model):
      event_datetime = models.DateTimeField(auto_now_add=True)
  ```

### 9. **`models.TimeField`**
- **Descripción**: Almacena una hora del día.
- **Ejemplo**:
  ```python
  class Schedule(models.Model):
      start_time = models.TimeField()
  ```

### 10. **`models.EmailField`**
- **Descripción**: Un campo de texto que valida que la entrada sea una dirección de correo electrónico.
- **Ejemplo**:
  ```python
  class Contact(models.Model):
      email = models.EmailField()
  ```

### 11. **`models.URLField`**
- **Descripción**: Un campo de texto que valida que la entrada sea una URL.
- **Ejemplo**:
  ```python
  class Website(models.Model):
      url = models.URLField()
  ```

### 12. **`models.SlugField`**
- **Descripción**: Almacena texto breve sin espacios, ideal para URLs amigables.
- **Ejemplo**:
  ```python
  class Article(models.Model):
      slug = models.SlugField(unique=True)
  ```

### 13. **`models.ForeignKey`**
- **Descripción**: Define una relación uno a muchos entre dos modelos.
- **Argumentos clave**:
  - `on_delete`: Define el comportamiento cuando el objeto relacionado es eliminado.
  - `related_name`: Nombre de la relación inversa.
- **Ejemplo**:
  ```python
  class Author(models.Model):
      name = models.CharField(max_length=100)

  class Book(models.Model):
      author = models.ForeignKey(Author, on_delete=models.CASCADE)
  ```

### 14. **`models.OneToOneField`**
- **Descripción**: Define una relación uno a uno entre dos modelos.
- **Ejemplo**:
  ```python
  class Profile(models.Model):
      user = models.OneToOneField(User, on_delete=models.CASCADE)
  ```

### 15. **`models.ManyToManyField`**
- **Descripción**: Define una relación muchos a muchos entre dos modelos.
- **Ejemplo**:
  ```python
  class Course(models.Model):
      name = models.CharField(max_length=100)
      students = models.ManyToManyField(Student)
  ```

### 16. **`models.FileField` y `models.ImageField`**
- **Descripción**: Almacena rutas a archivos y/o imágenes cargadas.
- **Argumentos clave**:
  - `upload_to`: Ruta donde se guardarán los archivos.
- **Ejemplo**:
  ```python
  class Document(models.Model):
      file = models.FileField(upload_to='documents/')
  
  class Photo(models.Model):
      image = models.ImageField(upload_to='photos/')
  ```

### 17. **`models.JSONField`**
- **Descripción**: Almacena datos en formato JSON.
- **Ejemplo**:
  ```python
  class Product(models.Model):
      metadata = models.JSONField()
  ```

### 18. **`models.UUIDField`**
- **Descripción**: Almacena un valor UUID (Identificador Único Universal).
- **Ejemplo**:
  ```python
  import uuid
  
  class MyModel(models.Model):
      id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
  ```

### 19. **`models.AutoField`**
- **Descripción**: Campo entero que se incrementa automáticamente (ID de la tabla).
- **Ejemplo**:
  ```python
  class MyModel(models.Model):
      id = models.AutoField(primary_key=True)
  ```

### 20. **`models.BigAutoField`**
- **Descripción**: Similar a `AutoField`, pero con una capacidad mayor (utilizado para tablas con gran cantidad de registros).
- **Ejemplo**:
  ```python
  class BigModel(models.Model):
      id = models.BigAutoField(primary_key=True)
  ```

### Resumen

Django proporciona una gran variedad de campos de modelo para cubrir casi cualquier necesidad de almacenamiento de datos. Estos campos permiten definir de manera clara y concisa la estructura de la base de datos, facilitando la gestión y manipulación de los datos en tu aplicación. Además, gracias a la integración de Django con diferentes bases de datos, estos modelos funcionan de manera consistente y eficiente, independientemente del motor de base de datos que estés utilizando.

shell
```shell
python manage.py shell
Type 'copyright', 'credits' or 'license' for more information
IPython 8.26.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from my_first_app.models import Book, Publisher, Author

In [2]: audry = Author(name="Audry", birth_date="2022-12-05")

In [3]: audry.save()

In [4]: pydanny = Author(name="Pydanny", birth_date="2023-12-05")

In [5]: pydanny.save()

In [6]: book
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[6], line 1
----> 1 book

NameError: name 'book' is not defined

In [7]: book = Book.objects.first()

In [8]: book
Out[8]: <Book: Two Scoops of Django>

In [9]: book.authors
Out[9]: <django.db.models.fields.related_descriptors.create_forward_many_to_many_manager.<locals>.ManyRelatedManager at 0x27443ecce90>

In [10]: book.authors.set(pydanny)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)  
Cell In[10], line 1
----> 1 book.authors.set(pydanny)

File ~\OneDrive\Escritorio\programación\platzi\CursodeDjango\venv\Lib\site-packages\django\db\models\fields\related_descriptors.py:1325, in create_forward_many_to_many_manager.<locals>.ManyRelatedManager.set(self, objs, clear, through_defaults)
   1322 def set(self, objs, *, clear=False, through_defaults=None):
   1323     # Force evaluation of `objs` in case it's a queryset whose value 
   1324     # could be affected by `manager.clear()`. Refs #19816.
hrough, instance=self.instance)
   1328     with transaction.atomic(using=db, savepoint=False):

TypeError: 'Author' object is not iterable

In [11]: authors_list = [pydanny, audry]

In [12]: print(authors_list)
[<Author: Pydanny>, <Author: Audry>]

In [13]: book.authors.set(authors_list)
```

**Lecturas recomendadas**

[Model field reference | Django documentation | Django](https://docs.djangoproject.com/en/5.0/ref/models/fields//#field-types "Model field reference | Django documentation | Django")

## Relaciones Uno a Uno (1:1) en Django

Explorar la relación uno a uno en Django puede parecer complejo, pero es fundamental para construir aplicaciones sólidas.

### ¿Cómo se crea una clase en Django?

Para empezar, imaginemos que tenemos una clase Profile que contiene información pública del autor. Este perfil incluirá:

- Un campo de URL para el sitio web del autor.
- Una biografía con un máximo de 500 caracteres.

Aquí está el código inicial para la clase `Profile`:

```python
class Profile(models.Model):
    website = models.URLField(max_length=200)
    biography = models.TextField(max_length=500)
    author = models.OneToOneField(Author, on_delete=models.CASCADE)
```

### ¿Cómo se maneja la relación uno a uno?

Para relacionar el perfil con el autor, utilizamos `OneToOneField`. Esto asegura que cada autor tenga un solo perfil y viceversa. Además, agregamos el parámetro `on_delete=models.CASCADE` para que si se elimina un autor, también se elimine su perfil.

### ¿Cómo se crean y se sincronizan las migraciones?

1. **Crear migraciones:** Ejecutamos `python manage.py makemigrations`.
2. **Sincronizar con la base de datos:** Usamos `python manage.py migrate`.

### ¿Cómo verificamos la creación de un perfil en la consola de Django?

1. **Abrir la shell de Django:** Ejecutamos `python manage.py shell`.
2. **Importar los modelos:**` from myapp.models import Author, Profile`.
3. **Buscar un autor existente:**`author = Author.objects.first()`.
4. **Crear un perfil:**

```python
profile = Profile.objects.create(
    website="http://example.com",
    biography="Lorem Ipsum",
    author=author
)
```

### ¿Cómo verificar los datos en la base de datos?

Usamos comandos SQL para verificar los datos:

`SELECT * FROM myapp_profile WHERE author_id = 1;`

### ¿Qué ocurre cuando se elimina un autor?
Si un autor se borra, su perfil también se eliminará gracias a `on_delete=models.CASCADE`.

**Lecturas recomendadas**

[Making queries | Django documentation | Django](https://docs.djangoproject.com/en/stable/topics/db/queries/ "Making queries | Django documentation | Django")
[Model field reference | Django documentation | Django](https://docs.djangoproject.com/en/stable/ref/models/fields/#django.db.models.OneToOneField "Model field reference | Django documentation | Django")


## Queries y Filtros en Django: Optimización y Estrategias Avanzadas

Los managers en Django son una herramienta poderosa que permite realizar diversas acciones dentro de las listas de objetos de un modelo, como contar, traer el primero o el último elemento, crear nuevos registros y mucho más.

Para contar los autores que están creados, utilizamos el manager por defecto llamado `objects` y el método `count`.

```python
author_count = Author.objects.count()
print(f"Hay {author_count} autores.")
```

### ¿Cómo traer el primer y último autor creado?

Para obtener el primer y último autor, podemos usar los métodos `first` y `last` del manager `objects`.

```python
primer_autor = Author.objects.first()
print(f"El primer autor es: {primer_autor.name}")

ultimo_autor = Author.objects.last()
print(f"El último autor es: {ultimo_autor.name}")
```

### ¿Cómo crear nuevos autores con el manager?

Podemos crear un nuevo autor directamente en la base de datos utilizando el método create del manager.

```python
nuevo_autor = Author.objects.create(name="Luis Martínez", birthday="1980-01-01")
print(f"Nuevo autor creado: {nuevo_autor.name}")

```
### ¿Cómo traer una lista de autores?

Para obtener una lista de todos los autores, utilizamos el método all del manager, que nos devuelve un queryset.

```python
autores = Author.objects.all()
for autor in autores:
    print(autor.name)
```

### ¿Cómo filtrar autores?

Podemos filtrar autores utilizando el método `filter`, que permite especificar condiciones basadas en los campos del modelo.

```python
autores_filtrados = Author.objects.filter(name="Pydanny")
for autor in autores_filtrados:
    print(f"Autor filtrado: {autor.name}")
```

### ¿Cómo borrar un autor filtrado?

Primero, filtramos el autor que queremos borrar y luego aplicamos el método `delete`.

```python
Author.objects.filter(name="Luis Martínez").delete()
print("Autor borrado.")
```

### ¿Cómo ordenar autores?

Podemos ordenar los autores utilizando el método `order_by`.

```python
autores_ordenados = Author.objects.order_by('name')
for autor in autores_ordenados:
    print(autor.name)
```

```shell
python manage.py shell
Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.26.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from my_first_app.models import Book, Publisher, Author, Profile, Pu 
   ...: blisher

In [2]: Author.objects.count()
Out[2]: 2

In [4]: Author.objects.first()
Out[4]: <Author: Audry>

In [5]: Author.objects.last()
Out[5]: <Author: Pydanny>

In [6]: Author??
Init signature: Author(*args, **kwargs)
Docstring:      Author(id, name, birth_date)
Source:
class Author(models.Model):
    name = models.TextField(max_length=200)
    birth_date = models.DateField()

    def __str__(self):
        return self.name
File:           c:\users\celio\onedrive\escritorio\programación\platzi\cursodedjango\django_concepts\my_first_project\my_first_app\models.py
Type:           ModelBase
Subclasses:

In [7]: Author.objects.create(name="Luis Martinez", birth_date="1990-12-05") 
   ...:
Out[7]: <Author: Luis Martinez>

In [8]: Author.objects.all()
Out[8]: <QuerySet [<Author: Audry>, <Author: Pydanny>, <Author: Luis Martinez>]>

In [10]: Author.objects.filter(name="Pydanny")
Out[10]: <QuerySet [<Author: Pydanny>]>

In [11]: Author.objects.filter(name="Luis Martinez").delete()
Out[11]: (1, {'my_first_app.Author': 1})

In [12]: Author.objects.all()
Out[12]: <QuerySet [<Author: Audry>, <Author: Pydanny>]>
artinez", birth_date="1990-12-05"
    ...: )
Out[13]: <Author: Luis Martinez>

In [14]: Author.objects.all()
Out[14]: <QuerySet [<Author: Audry>, <Author: Pydanny>, <Author: Luis Martinez>]>

In [15]: Author.objects.all().order_by('name')
Out[15]: <QuerySet [<Author: Audry>, <Author: Luis Martinez>, <Author: Pydanny>]>

In [16]:
```

## Gestión de URLs en Django: Configuración, Rutas y Mejores Prácticas

Configurar las URLs en Django es esencial para organizar tu proyecto y facilitar la navegación.

### ¿Cómo crear un archivo de URLs en Django?

Primero, debes crear un archivo urls.py en cada aplicación que desarrolles. Por ejemplo, si tienes una aplicación llamada `MyFirstApp`, debes crear un archivo `urls.py` dentro de esta aplicación.

- **Crear el archivo:** En la aplicación MyFirstApp, crea un archivo llamado urls.py.
- **Copiar y pegar configuración básica:** Puedes copiar la configuración básica de otro archivo de URLs y modificarla según sea necesario.
- **Eliminar enlaces e importaciones innecesarias:** Mantén solo lo necesario para tu aplicación.

```python
from django.urls import path
from . import views

urlpatterns = [
    path('listado/', views.myView, name='listado'),
]
```

### ¿Cómo incluir URLs de una aplicación en el proyecto?

Para incluir las URLs de una aplicación en el proyecto principal, sigue estos pasos:

1. **Modificar el archivo de URLs del proyecto:** Agrega un nuevo path que incluya las URLs de tu aplicación.

```python
from django.urls import include, path

urlpatterns = [
    path('carros/', include('myFirstApp.urls')),
]
```

2. Importar el include: Asegúrate de importar include desde django.urls.

### ¿Cómo configurar un servidor de desarrollo?

Para probar los cambios, ejecuta el servidor de desarrollo:

`python manage.py runserver`

Esto iniciará el servidor y podrás ver los cambios en tiempo real.

### ¿Cómo crear URLs dinámicas?
Para crear URLs que acepten parámetros dinámicos, sigue estos pasos:

1. **Definir una URL dinámica:** Utiliza los caracteres `< y >` para especificar el tipo de dato y el nombre del parámetro.

```python
urlpatterns = [
    path('detalle/<int:id>/', views.detalle, name='detalle'),
]
```

2. **Modificar la vista para aceptar parámetros:** Asegúrate de que tu vista acepte los parámetros correspondientes.

```python
def detalle(request, id):
    return HttpResponse(f"El ID es {id}")
```

### ¿Cómo manejar diferentes tipos de datos en URLs?

Django permite convertir diferentes tipos de datos en las URLs, como enteros y cadenas de texto:

1. **Enteros:** Utiliza `<int:nombre>` para enteros.

2. Cadenas de texto: Utiliza `<str:nombre>` para cadenas de texto.

```python
urlpatterns = [
    path('marca/<str:brand>/', views.marca, name='marca'),
]
```

### ¿Cómo probar URLs dinámicas en el navegador?
1. **Probar con enteros:** Accede a una URL que requiera un entero, como `detalle/1/`.
2. **Probar con cadenas de texto:** Accede a una URL que requiera una cadena de texto, como `marca/mazda/`.

**Lecturas recomendadas**

[URL dispatcher | Django documentation | Django](https://docs.djangoproject.com/en/5.0/topics/http/urls/#path-converters "URL dispatcher | Django documentation | Django")

### Vistas Basadas en Clases en Django

Las vistas son un componente crucial en Django, permitiendo la interacción entre las URLs y la lógica de negocio.

### ¿Cómo crear vistas en Django?

Para mantener el código organizado, es ideal ubicar las vistas en un archivo dedicado. Si tienes vistas definidas en el archivo de URLs, el primer paso es moverlas al archivo `views.py`. Asegúrate de renombrar las vistas si tienen nombres duplicados y de importar las dependencias necesarias, como HttpResponse.

### ¿Cómo manejar vistas basadas en funciones?

Las vistas basadas en funciones (FBV) son simples de implementar y adecuadas para lógica no compleja. Reciben el objeto `request` y devuelven un `HttpResponse`. Aquí un ejemplo básico:

```python
from django.http import HttpResponse

def MyTestView(request):
    return HttpResponse("Hello, this is a test view")
```

### ¿Cómo explorar el objeto request en Django?

El objeto `request` en Django contiene información relevante sobre la solicitud HTTP. Para explorar sus atributos, puedes utilizar el shell de Django:

```python
from django.http import HttpRequest

request = HttpRequest()
print(request.__dict__)
```

Esto te permitirá inspeccionar las propiedades del `request`, como el método HTTP, el usuario autenticado, entre otros.

### ¿Por qué usar vistas basadas en clases?

Las vistas basadas en clases (CBV) facilitan la reutilización de código y la modularidad. Son más adecuadas para lógica compleja y permiten utilizar métodos integrados de Django. Para convertir una vista basada en funciones a una basada en clases:

1. Define una clase que herede de una vista genérica de Django.
2. Implementa métodos como `get_context_data` para manejar el contexto.

Aquí un ejemplo de una CBV:

```python
from django.views.generic import TemplateView

class CarListView(TemplateView):
    template_name = "car_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['cars'] = Car.objects.all()
        return context
```

### ¿Cómo conectar una vista basada en clases a una URL?

Para conectar una CBV a una URL, utiliza el método `as_view()` en el archivo de URLs:

```python
from django.urls import path
from .views import CarListView

urlpatterns = [
    path('cars/', CarListView.as_view(), name='car-list')
]
```

### ¿Cómo evitar errores comunes al importar vistas?

Asegúrate de importar las vistas desde el módulo correcto. Utiliza el autocompletado del editor con precaución y verifica los importes en la documentación de Django.

### ¿Cuáles son las diferencias clave entre FBV y CBV?

- **FBV:** Simplicidad y facilidad de implementación para tareas básicas.
- **CBV:** Modularidad y reutilización, ideal para lógica compleja y uso de métodos predefinidos.

**Lecturas recomendadas**

[Class-based views | Django documentation | Django](https://docs.djangoproject.com/en/stable/topics/class-based-views/ "Class-based views | Django documentation | Django")