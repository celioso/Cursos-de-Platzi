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

## Vistas Basadas en Clases en Django

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

[Class-based views | Django documentation | Django](https://docs.djangoproject.com/en/5.1/ref/class-based-views/ "Class-based views | Django documentation | Django")

## Personalización de Interfaz con Plantillas en Django

Exploraremos los templates en Django y sus funcionalidades avanzadas que los diferencian del HTML estándar. Aprenderemos cómo los templates nos permiten mostrar contenido dinámico en el navegador, validar variables, recorrer listas y aplicar filtros para modificar valores antes de mostrarlos. También veremos cómo reutilizar contenido común mediante el archivo base HTML.

### ¿Qué son los templates en Django?

Los templates en Django son archivos HTML que incluyen funcionalidades adicionales para mostrar contenido dinámico. A diferencia del HTML puro, los Django templates permiten:

- Mostrar variables
- Realizar validaciones con `if`
- Recorrer listas con `for`

### ¿Cómo se muestran variables en un template?
Para mostrar variables, se encierran en dobles llaves `{{ }}`. Por ejemplo, para mostrar una variable llamada var del contexto, se usaría:

`{{ var }}`

### ¿Qué son y cómo se utilizan los filtros en Django?

Los filtros permiten modificar el valor de una variable antes de mostrarla. Se usan con un pipe | seguido del nombre del filtro. Por ejemplo, para mostrar solo el día y mes de una fecha:

`{{ fecha_nacimiento|date:"m/d" }}`

Los filtros pueden concatenarse. Por ejemplo, convertir el resultado en minúsculas:

`{{ fecha_nacimiento|date:"m/d"|lower }}`

### ¿Qué son los tags en Django y cómo se utilizan?

Los tags agregan funcionalidades adicionales al código HTML. Se abren con {% %} y pueden incluir:

- `if`: para validaciones
- `for`: para recorrer listas
- `url`: para mostrar URLs dinámicas

Algunos tags requieren una etiqueta de cierre. Por ejemplo, `if` y `for`:

```html
{% if condition %}
    <!-- contenido -->
{% endif %}
```

### ¿Qué es el archivo base HTML en Django?

El archivo base.html permite definir contenido común para ser reutilizado en la aplicación. Se crean bloques que pueden extenderse en otros archivos. Por ejemplo:

```html
<!-- base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
</head>
<body>
    <div id="content">
        {% block content %}
        {% endblock %}
    </div>
</body>
</html>
```

Para reutilizar este contenido:

```html
<!-- new_template.html -->
{% extends "base.html" %}
{% block content %}
    <!-- contenido específico -->
{% endblock %}
```

Las plantillas en Django son una parte esencial del framework, que permiten separar la lógica del servidor (backend) de la presentación (frontend). El sistema de plantillas de Django utiliza archivos HTML con marcadores especiales que permiten insertar datos dinámicos y lógica básica.

Aquí tienes una explicación de cómo funcionan las plantillas en Django y cómo utilizarlas:

### 1. **Creación de Plantillas**
Una plantilla es básicamente un archivo HTML que puede contener etiquetas de Django para renderizar contenido dinámico.

#### Ejemplo básico de plantilla (`car_list.html`):
```html
<!DOCTYPE html>
<html>
<head>
    <title>Listado de carros</title>
</head>
<body>
    <h1>Listado de carros</h1>
    <ul>
        {% for car in cars %}
            <li>{{ car.name }} - {{ car.brand }}</li>
        {% empty %}
            <li>No hay carros disponibles.</li>
        {% endfor %}
    </ul>
</body>
</html>
```

### 2. **Directorio de Plantillas**
Django busca las plantillas en una carpeta llamada `templates` dentro de cada aplicación o en una carpeta global especificada en tu proyecto.

#### Estructura típica de carpetas:
```
my_first_app/
    ├── templates/
        └── my_first_app/
            └── car_list.html
```

### 3. **Configuración de Plantillas en `settings.py`**
Asegúrate de que el ajuste `TEMPLATES` en tu archivo `settings.py` esté configurado correctamente. Si usas plantillas dentro de las aplicaciones, deberías tener algo como esto:

```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # Aquí puedes agregar directorios de plantillas globales si los tienes
        'APP_DIRS': True,  # Activa la búsqueda de plantillas en las aplicaciones
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

### 4. **Renderizar Plantillas en una Vista**
Para utilizar una plantilla, debes renderizarla en tu vista usando la función `render()`. Esta función toma el `request`, el nombre de la plantilla y un contexto con los datos que quieres pasar a la plantilla.

#### Ejemplo de vista:
```python
from django.shortcuts import render
from .models import Car

def car_list(request):
    cars = Car.objects.all()  # Obtén todos los carros
    return render(request, 'my_first_app/car_list.html', {'cars': cars})
```

### 5. **Sintaxis de Plantillas**

- **Variables**: Para mostrar datos dinámicos en una plantilla, utiliza las dobles llaves `{{ }}`.
  
  Ejemplo:
  ```html
  <p>Nombre del carro: {{ car.name }}</p>
  ```

- **Etiquetas**: Las etiquetas de plantilla se usan para control de flujo, como bucles, condiciones, etc. Se colocan dentro de `{% %}`.

  Ejemplo de un bucle:
  ```html
  {% for car in cars %}
      <p>{{ car.name }} - {{ car.brand }}</p>
  {% endfor %}
  ```

- **Filtros**: Los filtros permiten modificar el valor de una variable en la plantilla. Se aplican con el símbolo `|`.

  Ejemplo:
  ```html
  <p>{{ car.name|upper }}</p>  <!-- Convierte el nombre del carro a mayúsculas -->
  ```

### 6. **Herencia de Plantillas**
Django permite la herencia de plantillas, lo que significa que puedes tener una plantilla base que otras plantillas extienden.

#### Plantilla base (`base.html`):
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}Mi sitio{% endblock %}</title>
</head>
<body>
    <header>
        <h1>Mi aplicación de carros</h1>
    </header>
    <div>
        {% block content %}
        <!-- El contenido específico de cada página irá aquí -->
        {% endblock %}
    </div>
    <footer>
        <p>Derechos reservados © 2024</p>
    </footer>
</body>
</html>
```

#### Plantilla que hereda de `base.html` (`car_list.html`):
```html
{% extends "my_first_app/base.html" %}

{% block title %}Listado de carros{% endblock %}

{% block content %}
    <h2>Listado de carros</h2>
    <ul>
        {% for car in cars %}
            <li>{{ car.name }} - {{ car.brand }}</li>
        {% empty %}
            <li>No hay carros disponibles.</li>
        {% endfor %}
    </ul>
{% endblock %}
```

### 7. **Bloques de Plantilla**
Los bloques, como `block title` y `block content`, son marcadores en la plantilla base que otras plantillas pueden sobrescribir. Esto facilita la creación de estructuras de página comunes.

### 8. **Contexto en las Plantillas**
El contexto en Django es el conjunto de variables que pasas desde la vista a la plantilla. Estas variables son accesibles mediante las dobles llaves `{{ variable }}`.

### Resumen:
- **Plantillas**: Archivos HTML que contienen variables y lógica básica para generar contenido dinámico.
- **Variables y etiquetas**: Se utilizan para insertar datos dinámicos o lógica de control.
- **Herencia de plantillas**: Permite definir una estructura base y extenderla en diferentes páginas.
- **Renderización**: Usa la función `render()` en tus vistas para renderizar una plantilla con los datos del contexto.

Si necesitas más detalles o ejemplos, ¡déjame saber!

## Configuración del Proyectos en Django

Comenzamos la configuración de un proyecto Coffee Shop en Django

### ¿Cómo crear y activar el entorno virtual?

Para iniciar, nos posicionamos en la carpeta deseada en nuestro editor. Creamos el entorno virtual con:

`python -m venv <ruta_donde_guardar>/Coffee_Shop`

Activamos el entorno con:

`source Coffee_Shop/bin/activate`

Verificamos su activación y procedemos a instalar Django:

`pip install django`

### ¿Cómo iniciar un proyecto Django?

Creamos el proyecto utilizando el comando:

`django-admin startproject Coffee_Shop`

Listamos las carpetas para confirmar la creación del proyecto. Abrimos el proyecto en Visual Studio Code:

`code -r Coffee_Shop`

Ahora tenemos el archivo `manage.py` y las configuraciones listas en nuestro editor.

### ¿Qué extensiones instalar en Visual Studio Code?

Aprovechamos las alertas de Visual Studio Code para instalar extensiones esenciales como:

- **Python**
- **PyLance**
- **Python Debugger**
- **Black** (formateo de código)
- **Django** (para visualizar templates)

### ¿Cómo configurar el control de versiones con Git?

Inicializamos un repositorio Git:

`git init`

Añadimos y comiteamos los archivos iniciales creados por Django:

```bash
git add .
git commit -m "Initial setup"
```

### ¿Cómo crear y utilizar un archivo .gitignore?

Para evitar subir archivos innecesarios al repositorio, generamos un archivo `.gitignore` con [gitignore.io](https://www.toptal.com/developers/gitignore "gitignore.io") especificando “Django” como criterio. Pegamos el contenido generado en un nuevo archivo `.gitignor`e y lo comiteamos:

```bash
git add .gitignore
git commit -m "Add .gitignore"
```

### ¿Cómo manejar las dependencias del proyecto?

Creamos dos archivos para gestionar las dependencias:

1. **requirements.txt**: para dependencias de producción.
2. **requirements-dev.txt**: para dependencias de desarrollo como iPython.

Agregamos las dependencias instaladas en nuestro entorno actual:

**pip freeze > requirements.txt**

Comiteamos ambos archivos:

```bash
git add requirements.txt requirements-dev.txt
git commit -m "Add requirements files"
```

### ¿Cómo continuar con la configuración del proyecto?

Con el entorno preparado, es importante crear un archivo base HTML que sirva como plantilla. Te reto a crear `base.html` con un menú y un pie de página para usar en el curso de Django.

Lecturas recomendadas

[https://www.toptal.com/developers/gitignore/api/django](https://www.toptal.com/developers/gitignore/api/django "https://www.toptal.com/developers/gitignore/api/django")

[Python - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.python "Python - Visual Studio Marketplace")

[Pylance - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance "Pylance - Visual Studio Marketplace")

[Python Debugger - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy "Python Debugger - Visual Studio Marketplace")

[Black Formatter - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter "Black Formatter - Visual Studio Marketplace")

[Django - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=batisteo.vscode-django "Django - Visual Studio Marketplace")

## Creación del Modelo para la Aplicación 'Products' en Django

Crear una aplicación de administración de productos en una cafetería puede parecer complejo, pero siguiendo unos pasos claros, es más sencillo de lo que parece. En este artículo, exploraremos cómo crear y configurar un modelo de producto utilizando Django.

### ¿Cómo creamos la aplicación de productos?

Para empezar, debemos crear una nueva aplicación dentro de nuestro proyecto de Django. Desde la consola, ejecutamos los siguientes comandos:

- Manage
- Startup
- Products

Esto generará una nueva carpeta llamada “products”. Recuerda siempre registrar las aplicaciones creadas. Vamos a `Settings`, buscamos `Installed Apps` y añadimos `Product`.

### ¿Cómo definimos los modelos?

Después de registrar la aplicación, procedemos a crear los modelos. Iniciamos con el modelo `Product` que hereda de `Model`. El primer campo será `Name`, definido como un `TextField` con un `MaxLength` de 200 caracteres.

### ¿Qué es Verbose Name y cómo lo utilizamos?

El Verbose `Name` nos permite especificar cómo queremos que se visualice cada campo para el usuario final. Por ejemplo, para `Name` podemos definir un `verbose_name`.

### ¿Qué otros campos añadimos? 
Aparte de `Name`, añadimos otros campos importantes:

- **Description:** `TextField` con `MaxLength` de 300.
- **Price:** `DecimalField` con `max_digits` de 10 y `decimal_places` de 2.
- **Available:** `BooleanField` con `default=True`.
- **Photo**: `ImageField` con `upload_to='logos'`, permitiendo valores nulos (`null=True`) y en blanco (`blank=True`).

### ¿Cómo formateamos el código y solucionamos errores de dependencias?
Para mantener el código limpio, utilizamos la extensión `Black`. Hacemos clic derecho, seleccionamos `Format Document Width` y elegimos `Black Formatter`.

Si el editor no encuentra las dependencias, debemos asegurarnos de que Visual Studio Code esté utilizando el entorno virtual correcto. Seleccionamos el entorno correcto en la parte inferior del editor y recargamos la ventana con *Command P* o *Control P* seguido de `reload window`.

¿Cómo añadimos un método str?
Para una representación textual del modelo, añadimos un método `__str__` que retorna el nombre del producto.

## Cómo Crear Migraciones de Datos en Django

Nuestro modelo de producto ha sido actualizado con un nuevo campo: image field. Al intentar crear las migraciones, el sistema muestra un error indicando que no se puede usar image field porque Pillow no está instalado. No hay que preocuparse, la solución es instalar Pillow. Siguiendo la sugerencia del error, ejecutamos `pip install Pillow`. Ahora, volvemos a correr `make migrations` y el error desaparece, logrando así la primera migración de nuestra aplicación de productos.

### ¿Cómo se soluciona el error al crear migraciones?

El error ocurre porque Pillow, una librería necesaria para manejar campos de imagen, no está instalada. La solución es instalarla con `pip install Pillow`.

### ¿Qué hacemos después de instalar Pillow?

Después de instalar Pillow, es importante:

- Verificar que funciona corriendo nuevamente make migrations.
- Asegurarse de agregar la dependencia a `requirements.txt` para evitar problemas en producción. Utiliza `pip freeze` para ver la versión instalada y añade `Pillow` al archivo.

### ¿Por qué es importante agregar Pillow a requirements.txt?

Cuando instalamos dependencias localmente, debemos asegurarnos de que estén en `requirements.txt` para que también se instalen en el entorno de producción. Esto se hace para evitar errores y asegurar que todas las librerías necesarias estén disponibles.

### ¿Qué permite hacer Pillow con los campos de imagen?

Pillow permite realizar validaciones en imágenes, como asegurarse de que las imágenes subidas cumplan con ciertas características en cuanto a resolución.

### ¿Qué sigue después de las migraciones?

Después de realizar las migraciones, tienes la base para construir vistas, conectarlas a URLs y crear un listado de productos. Te animo a que lo intentes, lo subas a tu repositorio y compartas el enlace en el sistema de comentarios.

**Lecturas recomendadas**

[Pillow (PIL Fork) 10.4.0 documentation](https://pillow.readthedocs.io/ "Pillow (PIL Fork) 10.4.0 documentation")

## Creación de la Aplicación 'Products' con Formularios en Django

La funcionalidad de formularios en Django permite a los desarrolladores crear, validar y gestionar formularios de manera eficiente y organizada. A continuación, exploraremos cómo crear formularios en Django paso a paso.

### ¿Cómo se crean formularios en Django?

Para crear un nuevo formulario en Django, primero se debe crear una clase que herede de forms.Form. Esta clase contendrá todos los campos que queremos incluir en el formulario.

1. **Crear el archivo [forms.py](http://forms.py/ "forms.py"):**

```python
from django import forms

class ProductForm(forms.Form):
    name = forms.CharField(max_length=200, label='Nombre')
    description = forms.CharField(max_length=300, label='Descripción')
    price = forms.DecimalField(max_digits=10, decimal_places=2, label='Precio')
    available = forms.BooleanField(initial=True, label='Disponible', required=False)
    photo = forms.ImageField(label='Foto', required=False)
```

### ¿Cómo se manejan los datos del formulario en Django?

Una vez que el formulario está creado, necesitamos definir cómo manejar los datos cuando el usuario envía el formulario. Esto incluye validar los datos y guardarlos en la base de datos.

2. **Método save para guardar datos:**

```python
def save(self):
    from .models import Product
    data = self.cleaned_data
    Product.objects.create(
        name=data['name'],
        description=data['description'],
        price=data['price'],
        available=data['available'],
        photo=data['photo']
    )
```

### ¿Cómo se crea la vista para el formulario?

La vista conecta el formulario con el template y maneja el request del usuario. Usaremos una vista genérica de Django para simplificar este proceso.

3. **Crear la vista:**

from django.views.generic.edit import FormView
from django.urls import reverse_lazy
from .forms import ProductForm

```python
class ProductFormView(FormView):
    template_name = 'products/add_product.html'
    form_class = ProductForm
    success_url = reverse_lazy('product_list')

    def form_valid(self, form):
        form.save()
        return super().form_valid(form)
```

### ¿Cómo se configuran las URLs para la vista?

Es necesario configurar las URLs para que la vista esté accesible desde el navegador.

4. **Configurar [urls.py](http://urls.py/ "urls.py"):**

```python
from django.urls import path
from .views import ProductFormView

urlpatterns = [
    path('add/', ProductFormView.as_view(), name='add_product')
]
```

### ¿Cómo se crea el template para el formulario?

El template define la estructura HTML del formulario y cómo se renderiza en la página web.

5. **Crear el template add_product.html:**

```html
<h1>Agregar Producto</h1>
<form method="post" enctype="multipart/form-data">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit">Agregar</button>
</form>
```

### ¿Qué es el CSRF token y por qué es importante?

El CSRF token es una medida de seguridad que protege contra ataques de tipo Cross-Site Request Forgery. Django lo incluye automáticamente en los formularios para asegurar que las solicitudes provengan de fuentes confiables.

### ¿Cómo se maneja la redirección después de enviar el formulario?

La redirección después del envío del formulario se maneja configurando el parámetro `success_url` en la vista, utilizando `reverse_lazy` para obtener la URL de destino.

### ¿Cómo se valida y guarda el producto?

Cuando el formulario es válido, el método `form_valid` se encarga de llamar al método `save` del formulario para guardar el producto en la base de datos.

```bash
python manage.py shell
Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.26.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from products.models import Product

In [2]: Product.objects.first()
Out[2]: <Product: Latte>

In [3]: Product.objects.first().__dict__
Out[3]: 
{'_state': <django.db.models.base.ModelState at 0x14e0b965940>,
 'id': 1,
 'name': 'Latte',
 'description': 'D',
 'price': Decimal('5.00'),
 'available': True,
 'photo': ''}
```

**Lecturas recomendadas**

[Working with forms | Django documentation | Django](https://docs.djangoproject.com/en/stable/topics/forms/ "Working with forms | Django documentation | Django")

## Django Admin

Explorar la funcionalidad del Django Admin es esencial para aprovechar al máximo el potencial de Django en la gestión de aplicaciones web.

### ¿Qué es el Django Admin?

Django Admin es una herramienta integrada en Django que permite administrar modelos y objetos a través de una interfaz web intuitiva y fácil de configurar.

### ¿Cómo accedemos al Django Admin?

Primero, asegúrate de que el proyecto de Django esté corriendo. Luego, accede a la URL “/admin”. Aparecerá una página de inicio de sesión con el título “Django Administration”.

### ¿Cómo creamos un superusuario?

Para acceder al admin, necesitas un superusuario. Detén el servidor y ejecuta el comando `createsuperuse`r. Proporciona un nombre de usuario, correo electrónico y contraseña. Reinicia el servidor y usa estas credenciales para iniciar sesión en el admin.

### ¿Cómo registramos un modelo en el Django Admin?

1. Abre el archivo `admin.py` dentro de tu aplicación.
2. Crea una nueva clase que herede de `admin.ModelAdmin`.
3. Importa tu modelo con `from .models` import Product.
4. Registra el modelo usando `admin.site.register(Product, ProductAdmin)`.

### ¿Cómo personalizamos la vista de lista en el Django Admin?

Puedes añadir campos a la lista de visualización usando `list_display`:

```python
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'price')
```

Esto muestra los campos `name` y `price` en la lista de productos.

### ¿Cómo agregamos funcionalidad de búsqueda?

Añade el atributo `search_fields` en la clase del administrador:

```python
class ProductAdmin(admin.ModelAdmin):
    search_fields = ('name',)
```

Esto permite buscar productos por nombre.

### ¿Cómo editamos y guardamos productos?

Desde la lista de productos, haz clic en un producto para abrir el formulario de edición. Realiza los cambios necesarios y selecciona una de las opciones de guardado.

### ¿Cómo añadimos imágenes a los productos?

1. Asegúrate de tener un campo de imagen en tu modelo.
2. Sube una imagen a través del formulario de edición.
3. Configura las URLs para servir archivos estáticos agregando la configuración en `urls.py`:

```python
from django.conf.urls.static import static
from django.conf import settings

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### ¿Cómo administramos múltiples productos?

Selecciona varios productos usando los checkboxes y aplica acciones en masa, como eliminar.

### ¿Cómo configuramos la visualización de imágenes en la lista de productos?

Configura las URLs de los archivos estáticos y media para que Django sepa dónde encontrarlas. Asegúrate de importar y utilizar correctamente `static` y `settings` en tu archivo urls.py.

### ¿Cómo agregamos un nuevo campo al modelo?

Para agregar un nuevo campo, como la fecha de creación, modifica el modelo y actualiza la clase del administrador para mostrarlo en la lista:

```python
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'price', 'created_at')
```

**Lecturas recomendadas**

[The Django admin site | Django documentation | Django](https://docs.djangoproject.com/en/5.0/ref/contrib/admin/ "The Django admin site | Django documentation | Django")

[Crispy Tailwind](https://github.com/django-crispy-forms/crispy-tailwind "Crispy Tailwind")

## Manejo de Órdenes en CoffeShop

[GitHub coffee shop](https://github.com/platzi/django/tree/main/coffee_shop "GitHub coffee shop")

## Manejo de Pedidos en CoffeShop

[ccbv](https://ccbv.co.uk/ "ccbv")

## Mixings en vistas basadas en clases

**Lecturas recomendadas**

[GitHub - platzi/django at main](https://github.com/platzi/django/tree/main "GitHub - platzi/django at main")

django/orders/views.py at main · platzi/django · GitHub
[django/orders/views.py at main · platzi/django · GitHub](https://github.com/platzi/django/blob/main/orders/views.py "django/orders/views.py at main · platzi/django · GitHub")

## Django REST Framework

La separación de la lógica de backend y frontend es una práctica común en el desarrollo de software moderno, con el frontend generalmente escrito en JavaScript y la conexión al backend manejada a través de APIs. Django REST es una librería de Python que facilita la creación de estas APIs, permitiendo una integración eficiente entre frontend y backend.

### ¿Cómo instalar Django REST Framework?

Para instalar Django REST Framework, utilizamos el siguiente comando:

`pip install django-rest-framework`

No olvides agregarlo a tu archivo `requirements.txt` para mantener un registro de las dependencias del proyecto. Además, debes incluirlo en la configuración del proyecto dentro del archivo `settings.py` en la sección de `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    ...
    'rest_framework',
]
```

### ¿Cómo configurar un Serializer en Django REST?

Los Serializers en Django REST convierten modelos de Django en JSON. Para crear un nuevo Serializer, sigue estos pasos:

1. Crea un archivo llamado serializers.py en la aplicación correspondiente.
2. Importa ModelSerializer desde `rest_framework`:

```python
from rest_framework import serializers
from .models import Product

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'
```

### ¿Cómo crear una vista en Django REST?

Para crear una vista que devuelva datos en formato JSON:

1. Crea una vista heredando de APIView:

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Product
from .serializers import ProductSerializer

class ProductListAPI(APIView):
    def get(self, request):
        products = Product.objects.all()
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)
```

2. Define la URL para esta vista en urls.py:

```python
from django.urls import path
from .views import ProductListAPI

urlpatterns = [
    ...
    path('api/products/', ProductListAPI.as_view(), name='product-list-api'),
]
```

### ¿Cómo manejar permisos y autenticación en Django REST?

Dependiendo de tu caso de uso, puedes configurar permisos y autenticación. Para esta vista, vamos a desactivarlos:

```python
from rest_framework.permissions import AllowAny

class ProductListAPI(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        products = Product.objects.all()
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)
```

### ¿Cómo ejecutar y probar tu API?

Una vez configurado todo, puedes ejecutar tu servidor de desarrollo y acceder a la URL de la API para ver los datos en formato JSON:

`python manage.py runserver`

Luego, visita [http://localhost:8000/api/products/](http://localhost:8000/api/products/ "http://localhost:8000/api/products/") para ver la lista de productos.

**Lecturas recomendadas**

[Quickstart - Django REST framework](https://www.django-rest-framework.org/tutorial/quickstart/ "Quickstart - Django REST framework")

## Configurar PostgreSQL en AWS con Django

Preparar una aplicación para producción requiere asegurar que el entorno de desarrollo sea compatible con el entorno de producción. Aquí exploramos cómo configurar una base de datos PostgreSQL local y en AWS para asegurar una transición fluida.

### ¿Por qué cambiar de base de datos para producción?

El entorno de producción puede tener muchos usuarios simultáneos, lo que exige una base de datos capaz de manejar múltiples conexiones de manera eficiente. SQLite, aunque útil para desarrollo, no es ideal para producción. PostgreSQL, por otro lado, ofrece la capacidad necesaria para manejar estas demandas.

### ¿Cómo configurar PostgreSQL localmente?

1. Modificar configuración en Django:

- Abrir el archivo settings.py en el proyecto.
- Buscar la sección de configuración de la base de datos y reemplazar SQLite con PostgreSQL.
- Ejemplo de configuración:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'mydatabase',
        'USER': 'mydatabaseuser',
        'PASSWORD': 'mypassword',
        'HOST': 'localhost',
        'PORT': '',
    }
}
```

2. Verificar conexión:

- Ejecutar psql -h localhost para asegurarse de que PostgreSQL está instalado y configurado correctamente.
- Crear y migrar la base de datos con python manage.py migrate.

### ¿Qué errores pueden surgir al configurar PostgreSQL?

Un error común es la falta de la librería `psycopg2`. Este problema se soluciona instalando la librería necesaria:

`pip install psycopg2-binary`

Esta librería permite a Django comunicarse con PostgreSQL de manera eficiente.

### ¿Cómo configurar PostgreSQL en AWS?

1. **Crear una instancia en AWS RDS:**

- Iniciar sesión en AWS y buscar RDS.
- Crear una instancia de base de datos PostgreSQL usando la capa gratuita.
- Configurar el nombre de la base de datos, usuario y contraseña.

2. Configurar reglas de seguridad:

- Acceder a los grupos de seguridad y editar las reglas de ingreso y egreso para permitir el tráfico desde la IP local.

3. Conectar Django a AWS RDS:

- Modificar el archivo settings.py para incluir las credenciales de AWS RDS.
- Ejemplo:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'mydatabase',
        'USER': 'mydatabaseuser',
        'PASSWORD': 'mypassword',
        'HOST': 'mydatabase.amazonaws.com',
        'PORT': '5432',
    }
}
```

- Asegurarse de no incluir credenciales sensibles en el repositorio.

### ¿Cómo manejar las credenciales de manera segura?

Es crucial no almacenar las credenciales en el archivo `settings.py` para evitar comprometer la seguridad del proyecto. Utilizar variables de entorno o servicios de gestión de secretos es la mejor práctica para mantener la seguridad de la información sensible.

**Lecturas recomendadas**

[PostgreSQL: Downloads](https://www.postgresql.org/download/ "PostgreSQL: Downloads")

[Databases | Django documentation | Django](https://docs.djangoproject.com/en/stable/ref/databases/ "Databases | Django documentation | Django")

[Free Cloud Computing Services - AWS Free Tier](https://aws.amazon.com/free "Free Cloud Computing Services - AWS Free Tier")

## Variables de entorno en Django

Aprender a manejar información sensible es crucial para la seguridad de cualquier proyecto. Jango facilita este proceso mediante su librería Django Environment, la cual permite gestionar credenciales fuera del archivo de configuración principal.

### ¿Cómo instalar Django Environment?

Para comenzar, instala Django Environment desde la terminal usando el comando:

`pip install django-environ`

Luego, ve a tu archivo `settings.py` y añade la importación de la librería al principio del archivo:

`import environ`

### ¿Cómo configurar las variables de entorno?

Primero, crea una nueva instancia de la librería y define las variables en el archivo `settings.py`:

env = environ.Env()

Luego, mueve tus credenciales sensibles a un archivo `.env` en la raíz del proyecto, asegurándote de no subir este archivo al repositorio:

`DATABASE_PASSWORD=my_secure_password`

En `settings.py`, reemplaza las credenciales directas con las variables de entorno:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('DATABASE_NAME'),
        'USER': env('DATABASE_USER'),
        'PASSWORD': env('DATABASE_PASSWORD'),
        'HOST': env('DATABASE_HOST'),
        'PORT': env('DATABASE_PORT'),
    }
}
```

### ¿Cómo cargar las variables de entorno?

Para que Django reconozca el archivo `.env`, debes cargarlo en tu configuración. Agrega la siguiente línea en la parte superior de `settings.py`:

`environ.Env.read_env(os.path.join(BASE_DIR, '.env'))`

Esto permitirá que las variables definidas en `.env` sean accesibles desde el entorno de Django.

### ¿Qué hacer si la variable no se carga?

Si las variables no se cargan automáticamente, puedes exportarlas manualmente en tu entorno antes de ejecutar el servidor de Django:

`export DATABASE_PASSWORD=my_secure_password`

### ¿Cómo simplificar la configuración de la base de datos?

Jango Environment ofrece métodos útiles como db_url que simplifican aún más la configuración:

```python
DATABASES = {
    'default': env.db(),
}
```

Define todas las credenciales en una única variable en el archivo .env:

`DATABASE_URL=postgres://user:password@host:port/dbname`

Este método reduce el número de configuraciones manuales, facilitando la administración de variables.

### ¿Qué ventajas ofrece Django Environment?

Usar Django Environment para gestionar credenciales ofrece múltiples beneficios:

- **Seguridad mejorada**: Mantén credenciales fuera del código fuente.
- **Facilidad de uso**: Simplifica la configuración de la base de datos.
- **Colaboración segura**: Permite compartir código sin exponer información sensible.

**Lecturas recomendadas**

[django-environ](https://django-environ.readthedocs.io/en/latest/ "django-environ")

El error que estás viendo indica que Django no puede conectarse a la base de datos PostgreSQL alojada en Amazon RDS (o en otro servidor remoto). Este error ocurre principalmente por las siguientes razones:

### 1. **Problemas de Conexión**
   - **Conexión rechazada** o **timeout**: Esto puede deberse a problemas de conectividad o configuración incorrecta.

### 2. **Posibles causas y soluciones**

#### 1. **Base de datos no accesible**
   - **Verifica la conectividad a la base de datos**: Asegúrate de que la instancia de RDS o el servidor donde está alojada la base de datos está funcionando y aceptando conexiones desde tu máquina.
   - **Prueba de conectividad**: Desde tu terminal, intenta hacer `ping` o conectarte al puerto de la base de datos para verificar si está accesible. Por ejemplo, intenta conectarte desde `psql` (el cliente de PostgreSQL):

     ```bash
     psql -h database-1.cloka288krli.us-east-1.rds.amazonaws.com -U username -d dbname
     ```

   - Si no puedes conectarte desde la terminal, asegúrate de que la base de datos esté encendida y accesible en el puerto correcto.

#### 2. **Configuración incorrecta de la base de datos**
   - **Verifica la configuración en `settings.py`**: Asegúrate de que los detalles de la conexión a la base de datos en tu archivo `settings.py` sean correctos. La configuración debería verse algo así:

     ```python
     DATABASES = {
         'default': {
             'ENGINE': 'django.db.backends.postgresql',
             'NAME': 'nombre_de_base_de_datos',
             'USER': 'usuario',
             'PASSWORD': 'contraseña',
             'HOST': 'database-1.cloka288krli.us-east-1.rds.amazonaws.com',
             'PORT': '5432',  # El puerto estándar para PostgreSQL
         }
     }
     ```

   - **Asegúrate de que los valores sean correctos**, incluidos el nombre de la base de datos, el usuario y la contraseña.

#### 3. **Reglas de seguridad o cortafuegos**
   - **Verifica las reglas de seguridad del servidor**: Si estás utilizando Amazon RDS o algún otro servicio en la nube, asegúrate de que las reglas de seguridad permiten conexiones entrantes en el puerto 5432 (puerto estándar de PostgreSQL) desde la dirección IP de tu máquina.
   
     - Para Amazon RDS, puedes revisar las reglas de **Security Group** asociado a tu instancia y agregar las IPs de las máquinas que pueden acceder.
   
   - **Cortafuegos local**: Asegúrate de que no haya un firewall en tu red local o en tu máquina que esté bloqueando el tráfico hacia el puerto 5432.

#### 4. **Servidor PostgreSQL en el host correcto**
   - Verifica que el servidor PostgreSQL esté en el host correcto (`database-1.cloka288krli.us-east-1.rds.amazonaws.com`) y esté en ejecución. Si no tienes control directo sobre ese servidor, asegúrate de que el administrador lo esté ejecutando correctamente.

### 3. **Alternativa para tests locales: usar SQLite**
   Si sólo estás ejecutando pruebas en tu máquina local y no necesitas conectarte a la base de datos remota, puedes cambiar temporalmente a SQLite para ejecutar tus tests. Para hacerlo, ajusta el `DATABASES` en tu `settings.py` solo para entornos de prueba locales:

   ```python
   if 'test' in sys.argv:
       DATABASES = {
           'default': {
               'ENGINE': 'django.db.backends.sqlite3',
               'NAME': ':memory:',
           }
       }
   ```

   Esto cambiará la base de datos a una base de datos en memoria solo cuando ejecutes pruebas (`python manage.py test`).

### 4. **Verificar las credenciales de acceso**
   Asegúrate de que las credenciales (usuario, contraseña) que has configurado en `settings.py` tienen los permisos necesarios para conectarse y acceder a la base de datos remota.

Revisa estas posibles causas, especialmente la conectividad y la configuración en `settings.py`, y ajusta según sea necesario.

## ¿Cómo usar Unit Testing en Django?

El **unit testing** en Django es una herramienta poderosa para probar de manera automatizada las funcionalidades de tu aplicación. Django ofrece una integración directa con el módulo `unittest` de Python a través de su framework de pruebas. Aquí te explico cómo empezar a utilizar unit testing en Django:

### 1. **Configurar un entorno de prueba**

Django crea automáticamente un entorno de prueba al ejecutar pruebas, incluyendo la creación de una base de datos temporal, lo que asegura que las pruebas no afecten los datos reales.

### 2. **Estructura básica de una prueba en Django**
Las pruebas se suelen colocar en un archivo llamado `tests.py` dentro de cada aplicación de Django. El archivo básico tiene el siguiente formato:

```python
from django.test import TestCase
from .models import MiModelo

class MiModeloTestCase(TestCase):
    def setUp(self):
        # Esta función se ejecuta antes de cada test. Aquí puedes preparar datos de prueba.
        MiModelo.objects.create(campo1="dato1", campo2="dato2")

    def test_campo1_valor(self):
        # Prueba para verificar si el valor del campo1 es correcto.
        objeto = MiModelo.objects.get(campo1="dato1")
        self.assertEqual(objeto.campo1, "dato1")
```

### 3. **Métodos de prueba comunes**
- `setUp()`: Se ejecuta antes de cada prueba individual y es útil para preparar datos que serán utilizados en cada test.
- `tearDown()`: Se ejecuta después de cada prueba, y puedes usarlo para limpiar los datos si es necesario (aunque Django lo hace automáticamente al final de cada prueba).
- Métodos de `unittest` como `assertEqual`, `assertTrue`, `assertFalse`, etc., son usados para realizar las verificaciones.

### 4. **Ejecutar pruebas**
Para ejecutar las pruebas de tu proyecto Django, usa el comando:

```bash
python manage.py test
```

Esto ejecutará todas las pruebas en los archivos `tests.py` de todas las aplicaciones del proyecto.

### 5. **Pruebas para modelos**
Puedes probar la lógica de negocio y las relaciones de tus modelos de esta manera:

```python
from django.test import TestCase
from .models import MiModelo

class MiModeloTestCase(TestCase):
    def setUp(self):
        # Crea un objeto para probar
        self.objeto = MiModelo.objects.create(campo1="prueba", campo2="valor")

    def test_campo1_valor(self):
        # Verifica que el campo1 tiene el valor correcto
        self.assertEqual(self.objeto.campo1, "prueba")

    def test_string_representation(self):
        # Prueba el método __str__ del modelo
        self.assertEqual(str(self.objeto), "prueba")
```

### 6. **Pruebas para vistas**
Puedes usar el cliente de pruebas que Django ofrece para probar tus vistas:

```python
from django.test import TestCase
from django.urls import reverse

class MiVistaTestCase(TestCase):
    def test_vista_principal(self):
        # Utiliza el cliente de pruebas para simular una solicitud GET
        response = self.client.get(reverse('nombre_de_la_vista'))
        # Verifica que la respuesta es 200 OK
        self.assertEqual(response.status_code, 200)
        # Verifica si el contenido contiene algún texto específico
        self.assertContains(response, "Texto esperado")
```

### 7. **Pruebas para formularios**
Puedes probar la lógica de validación y envío de formularios de la siguiente manera:

```python
from django.test import TestCase
from .forms import MiFormulario

class MiFormularioTestCase(TestCase):
    def test_formulario_valido(self):
        form = MiFormulario(data={'campo1': 'valor válido'})
        self.assertTrue(form.is_valid())

    def test_formulario_invalido(self):
        form = MiFormulario(data={'campo1': ''})
        self.assertFalse(form.is_valid())
```

### 8. **Pruebas para URLs**
Asegúrate de que las rutas de tus vistas están configuradas correctamente:

```python
from django.test import SimpleTestCase
from django.urls import reverse, resolve
from .views import mi_vista

class URLTestCase(SimpleTestCase):
    def test_url_resuelve_a_vista(self):
        # Verifica que la URL resuelve a la vista correcta
        url = reverse('nombre_de_la_vista')
        self.assertEqual(resolve(url).func, mi_vista)
```

### 9. **Pruebas para API (si usas Django REST Framework)**
Si trabajas con una API, también puedes hacer pruebas para verificar el comportamiento de tus endpoints:

```python
from rest_framework.test import APITestCase
from rest_framework import status
from django.urls import reverse
from .models import MiModelo

class MiAPITestCase(APITestCase):
    def test_api_lista(self):
        url = reverse('mi_api_lista')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
```

### 10. **Estrategia para pruebas unitarias**
- **Cubre la lógica de negocio**: Asegúrate de probar cualquier cálculo, reglas de negocio y flujos de datos en los modelos.
- **Prueba todas las vistas críticas**: Asegúrate de que las vistas devuelvan la respuesta adecuada.
- **Prueba formularios y validaciones**: La lógica de validación en formularios debe ser robusta y a prueba de errores.
- **Prueba integraciones y APIs**: Si usas APIs, verifica que los endpoints funcionen como se espera.
  
### 11. **Ejemplo completo de pruebas**

```python
from django.test import TestCase
from django.urls import reverse
from .models import Producto

class ProductoTestCase(TestCase):
    def setUp(self):
        self.producto = Producto.objects.create(nombre="Café", precio=5.00)

    def test_producto_creado(self):
        producto = Producto.objects.get(nombre="Café")
        self.assertEqual(producto.precio, 5.00)

    def test_vista_producto_lista(self):
        response = self.client.get(reverse('productos'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Café")

    def test_formulario_producto_valido(self):
        form_data = {'nombre': 'Té', 'precio': 4.00}
        response = self.client.post(reverse('crear_producto'), data=form_data)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Producto.objects.filter(nombre='Té').exists())
```

### Resumen

- **TestCase**: Es el contenedor básico para las pruebas en Django.
- **Comandos**: Usa `python manage.py test` para ejecutar las pruebas.
- **Verificaciones**: Métodos como `assertEqual`, `assertTrue`, y `assertContains` permiten verificar el comportamiento esperado.

Este flujo de trabajo te ayudará a garantizar que tu aplicación funcione como se espera a través de pruebas automatizadas.

## Debugging en Django

Preparar un proyecto para despliegue en AWS puede ser desafiante, pero siguiendo algunos pasos esenciales, podemos asegurar que todo funcione correctamente. Aquí revisaremos cómo asegurarnos de que nuestro proyecto esté listo para ser ejecutado en un servidor de AWS, incluyendo la configuración de dependencias, ajustes en el routing y la documentación necesaria.

### ¿Cómo aseguramos que el archivo requirements.txt esté completo?

- Verificar que todas las librerías utilizadas estén listadas en el archivo `requirements.txt`.
- Asegurarnos de que las versiones de las librerías sean correctas.
- Utilizar el comando `pip install -r path/to/requirements.txt` para instalar todas las dependencias.
- Si hay errores, revisar el archivo `requirements.txt` y corregir las versiones incorrectas.
- Confirmar la instalación correcta con `pip freeze` y actualizar el archivo `requirements.txt` si es necesario.

### ¿Qué hacer si no se muestran las URLs correctas en el home del proyecto?

- Asegurarse de que no estamos retornando un 404 en la página principal.
- Mostrar la lista de productos en el home configurando las URLs adecuadamente.
- Modificar las rutas en el archivo `urls.py` para que la lista de productos sea la primera en ser validada.
- Guardar los cambios y ejecutar el proyecto para verificar que la lista de productos aparezca en la raíz del proyecto.

### ¿Por qué es importante un archivo README?

- Compartir con otros desarrolladores cómo configurar y ejecutar el proyecto.
- Incluir información sobre las diferentes aplicaciones dentro del proyecto, como `users` y `products`.
- Explicar los requerimientos del proyecto y proporcionar enlaces de clonación.
- Crear y mantener un archivo `README.md` en el root del proyecto, detallando todos estos aspectos.

### ¿Cómo formatear el código de manera consistente?

- Utilizar herramientas como Black para mantener un formato de código consistente.
- Instalar Black y ejecutarlo para unificar el uso de comillas y otros estilos de código.
- Confirmar que Black sigue las normas de PEP 8, el estándar de estilo de código en Python.
- Integrar Black en el proceso de desarrollo para mantener la consistencia en todo el proyecto.

### ¿Qué hacer antes del despliegue en AWS?

- Revisar y corregir cualquier error o bug en la aplicación.
- Crear una cuenta en AWS si aún no se tiene.
- Estar preparado para el despliegue en AWS, siguiendo las instrucciones y recomendaciones específicas para este entorno.

**Lecturas recomendadas**

[Black Formatter - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter "Black Formatter - Visual Studio Marketplace")

[PEP 8 – Style Guide for Python Code | peps.python.org](https://peps.python.org/pep-0008/ "PEP 8 – Style Guide for Python Code | peps.python.org")

[Free Cloud Computing Services - AWS Free Tier](https://aws.amazon.com/free/ "Free Cloud Computing Services - AWS Free Tier")

## Desplegar aplicaciones de Django en AWS

**NOTA**: Para este módulo utilicé wsl

Desplegar una aplicación en AWS puede ser sencillo utilizando Elastic Beanstalk, un servicio que automatiza la infraestructura necesaria.

### ¿Qué es Elastic Beanstalk y cómo funciona?

Elastic Beanstalk es un servicio de AWS que permite desplegar y gestionar aplicaciones rápidamente. Basta con enviar el código, y el servicio se encarga de crear y gestionar la infraestructura necesaria.

### ¿Cómo se configura la CLI de Elastic Beanstalk?

Con las credenciales listas, sigue estos pasos para configurar la CLI:

1. Instala Elastic Beanstalk CLI siguiendo el [enlace de instalación](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3.html "enlace de instalación").
2. Ejecuta eb init y responde las preguntas sobre la región, el ID de acceso y la clave secreta.
3. Configura el nombre de la aplicación y la versión de Python.
4. Indica si utilizarás CodeCommit (en este caso, no, ya que se usa GitHub).
5. Configura una llave SSH para conectarte a los servidores.

### ¿Cómo se despliega la aplicación?

1. Crea un environment de producción con `eb create coffee-shop-production`.
2. El servicio creará la infraestructura necesaria, incluyendo instancias y configuraciones de seguridad.
3. Verifica el estado del environment con `eb status`.

### ¿Cómo se solucionan errores comunes durante el despliegue?

- **Configuración incorrecta del módulo WSGI**: Configura el path correctamente en eb config.
- **Variable de entorno faltante**: Crea la variable con eb setenv.
- **Error en `ALLOWED_HOSTS` de Django**: Agrega el dominio correspondiente en el archivo de configuración de Django.

### ¿Cómo se gestionan archivos estáticos en Django?

Para asegurarte de que los archivos estáticos de Django se sirvan correctamente:

1. Ejecuta `python manage.py collectstatic`.
2. Configura el directorio de archivos estáticos en el archivo `settings.py`.

### ¿Qué otros proveedores de nube se pueden considerar?

AWS es una opción recomendada por su estabilidad y escalabilidad, pero también puedes explorar alternativas como DigitalOcean y Google Cloud Platform (GCP) para desplegar tus proyectos.

**Lecturas recomendadas**

[Using the Elastic Beanstalk command line interface (EB CLI) - AWS Elastic Beanstalk](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3.html "Using the Elastic Beanstalk command line interface (EB CLI) - AWS Elastic Beanstalk")

[Simplified EB CLI installation mechanism.](https://github.com/aws/aws-elastic-beanstalk-cli-setup "Simplified EB CLI installation mechanism.")