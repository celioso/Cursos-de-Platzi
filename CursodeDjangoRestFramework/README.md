# Curso de Django Rest Framework

## Crea y escala APIs con Django REST Framework

Imagina un mundo donde las aplicaciones no pueden compartir información entre ellas. Tu app de pedidos en línea no sabría tu ubicación ni si tienes saldo para pagar. ¿Qué es lo que falta? Exacto, una API. Las APIs son las autopistas de datos que permiten a las aplicaciones intercambiar información de manera efectiva, y para ello utilizan un estilo arquitectónico llamado REST. A través de métodos como GET, POST o DELETE, REST define cómo los mensajes viajan por internet. Sin embargo, crear una API desde cero puede ser complicado, y ahí es donde entra en juego Django REST Framework.

### ¿Por qué las APIs son esenciales para las aplicaciones?

- Las APIs conectan aplicaciones permitiendo que compartan información en tiempo real.
- Sin APIs, no sería posible realizar tareas básicas como verificar tu ubicación o procesar pagos.
- Permiten la comunicación eficiente entre servidores, fundamental para la funcionalidad de cualquier aplicación moderna.

### ¿Cómo facilita Django REST Framework la creación de APIs?

- Django REST Framework permite configurar y desplegar APIs sin necesidad de crear todo desde cero.
- Se encarga de la seguridad, la comunicación y la interacción con bases de datos, ofreciendo un enfoque escalable.
- Este framework se enfoca en la simplicidad y rapidez, haciendo que el desarrollo sea eficiente y sin complicaciones.

### ¿Qué hace a Django REST Framework adecuado tanto para principiantes como para expertos?

- Empresas de todos los tamaños, desde startups hasta grandes corporaciones, usan Django REST Framework debido a su versatilidad y facilidad de uso.
- No es necesario ser un experto para empezar a trabajar con él, lo que lo convierte en una opción accesible para cualquier desarrollador.
- Al utilizar Django REST Framework, puedes concentrarte en lo que realmente importa: crear experiencias digitales de calidad.

### ¿Qué beneficios ofrece Django REST Framework en la producción de APIs?

- Ahorra tiempo al evitar el desarrollo de funciones repetitivas y básicas.
- Integra funciones clave como autenticación, manejo de datos y seguridad de forma nativa.
- Facilita la escalabilidad, permitiendo que las aplicaciones crezcan sin problemas técnicos mayores.

**Lecturas recomendadas**

[Home - Django REST framework](https://www.django-rest-framework.org/)

## Introducción a las APIs, REST y JSON

Las APIs (Application Programming Interfaces) permiten que los computadores se comuniquen entre ellos de manera estructurada, usando formatos que ambos pueden entender. Son esenciales en el desarrollo moderno, automatizando procesos y facilitando la integración entre sistemas, como el caso de las plataformas de pago o la personalización de publicidad. JSON es el formato más utilizado en estas interacciones, permitiendo compartir información como texto, arreglos y objetos. Las APIs REST, basadas en JSON y HTTP, aseguran comunicaciones predecibles entre servidores y clientes.

### ¿Qué es una API y cómo funciona?

- Las APIs permiten la comunicación entre computadores de manera estructurada.
- Se utilizan principalmente para enviar solicitudes y recibir respuestas entre servidores o entre un servidor y un cliente.
- Son fundamentales para la automatización de tareas en el desarrollo web moderno.

### ¿Cómo se usan las APIs en la vida cotidiana?

- Existen APIs comunes, como la de Facebook, que utiliza tus búsquedas para mostrarte publicidad personalizada.
- Las APIs de pago, como Stripe, permiten gestionar tarjetas de crédito de manera segura.
- Estas herramientas evitan que los desarrolladores deban implementar complejas normativas de seguridad en sus propios servidores.

### ¿Qué es el formato JSON y por qué es importante?

- JSON (JavaScript Object Notation) es el formato estándar para enviar y recibir datos a través de APIs.
- Permite almacenar y estructurar información como texto, arreglos y objetos.
- Por ejemplo, un usuario puede tener varios hobbies, y estos se almacenan en un arreglo dentro de un JSON.

### ¿Cómo se estructuran las APIs REST?

- REST (Representational State Transfer) es una arquitectura que define cómo deben enviarse los mensajes a través de HTTP usando JSON.
- Garantiza que las comunicaciones sean predecibles, lo que significa que las mismas solicitudes siempre producirán los mismos resultados.

### ¿Cuáles son los métodos principales de una API REST?

- **GET**: Se utiliza para obtener información. Puede devolver una lista de recursos o un recurso específico.
- **POST**: Permite crear nuevos recursos, como agregar un nuevo usuario.
- **DELETE**: Utilizado para eliminar un recurso existente.
- **PUT y PATCH**: Modifican la información de un recurso, ya sea un solo campo o todo el contenido.

### Concepto de REST

REST, significa "Representational State Transfer". Es un estilo arquitectónico para diseñar servicios web. Se basa en una serie de principios y restricciones que permiten la comunicación entre sistemas a través de la web de manera eficiente y escalable.

### Principios Clave de REST

1. **Recursos**: En REST, todo se trata de recursos, que son entidades que pueden ser representadas en diferentes formatos (como JSON o XML). Cada recurso tiene una URL única que lo identifica.

2. **Métodos HTTP**: REST utiliza los métodos HTTP estándar para realizar operaciones sobre los recursos. Los métodos más comunes son:
 - **GET**: Para obtener información sobre un recurso.
 - **POST**: Para crear un nuevo recurso.
 - **PUT**: Para actualizar un recurso existente.
 - **DELETE**: Para eliminar un recurso.

3. **Stateless**: Cada solicitud del cliente al servidor debe contener toda la información necesaria para entender y procesar la solicitud. Esto significa que el servidor no almacena el estado del cliente entre las solicitudes, lo que mejora la escalabilidad.

4. **Representaciones**: Los recursos pueden ser representados de diferentes maneras. Por ejemplo, al solicitar un recurso, el servidor puede devolverlo en formato JSON o XML, dependiendo de lo que el cliente solicite.

5. **Navegabilidad**: REST promueve el uso de hipermedios (enlaces) para permitir a los clientes navegar entre los recursos disponibles, lo que facilita la interacción con la API.

**Conclusión**

REST se ha convertido en un estándar popular para construir APIs debido a su simplicidad y flexibilidad. Al seguir estos principios, los desarrolladores pueden crear servicios web que son fáciles de usar y mantener, permitiendo una integración fluida entre diferentes aplicaciones y plataformas.

**Herramientas de validación de JSON**

[https://jsonlint.com/](https://jsonlint.com/)

**Lecturas recomendadas**

[JSON Online Validator and Formatter - JSON Lint](https://jsonlint.com/)

[GitHub - platzi/django-rest-framework](https://github.com/platzi/django-rest-framework)

[HTTP request methods - HTTP | MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)

## Instalación de Django y Django REST Framework

Aquí tienes una guía paso a paso para instalar **Django** y **Django REST Framework**:

### 1. Preparar el entorno

Antes de comenzar, asegúrate de tener instalado **Python** y **pip**. Puedes verificarlo ejecutando los siguientes comandos en tu terminal:

```bash
python --version
pip --version
```

Si no tienes Python instalado, descárgalo desde [python.org](https://www.python.org/downloads/) e instálalo.

### 2. Crear un entorno virtual

Es recomendable crear un entorno virtual para tu proyecto. Esto te permite gestionar las dependencias de manera aislada.

```bash
# Crea un nuevo directorio para tu proyecto y navega hacia él
mkdir mi_proyecto
cd mi_proyecto

# Crea un entorno virtual
python -m venv venv

# Activa el entorno virtual
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
```

### 3. Instalar Django

Una vez que tengas el entorno virtual activo, puedes instalar Django utilizando pip:

```bash
pip install django
```

### 4. Crear un proyecto Django

Después de instalar Django, crea un nuevo proyecto:

```bash
django-admin startproject mi_proyecto .
cd mi_proyecto
```

### 5. Instalar Django REST Framework

Ahora, instala Django REST Framework:

```bash
pip install djangorestframework
```

### 6. Configurar Django REST Framework

Abre el archivo `settings.py` de tu proyecto (que se encuentra en la carpeta `mi_proyecto/mi_proyecto/`) y añade `rest_framework` a la lista de aplicaciones instaladas:

```python
# mi_proyecto/settings.py

INSTALLED_APPS = [
    ...
    'rest_framework',
]
```

### 7. Verificar la instalación

Para asegurarte de que todo está funcionando correctamente, puedes ejecutar el servidor de desarrollo de Django:

```bash
python manage.py runserver
```

Abre tu navegador y visita `http://127.0.0.1:8000/`. Deberías ver la página de inicio de Django.

### 8. Crear una aplicación (opcional)

Si deseas crear una aplicación dentro de tu proyecto:

```bash
python manage.py startapp mi_aplicacion
```

### Resumen

Ahora tienes un entorno configurado con Django y Django REST Framework. Puedes comenzar a desarrollar tu aplicación y crear APIs según sea necesario.

**Lecturas recomendadas**

[Home - Django REST framework](https://www.django-rest-framework.org/)

## Integración de Django REST Framework en proyectos Django

Django REST Framework (DRF) es una extensión poderosa de Django que permite construir APIs de manera rápida y eficiente, aprovechando las funcionalidades robustas de Django y añadiendo mejoras específicas para el desarrollo de APIs.

### ¿Cómo reutiliza Django REST las funcionalidades de Django?

Django REST reutiliza varias de las funcionalidades principales de Django, lo que permite un flujo de trabajo más sencillo:

- **ORM de Django**: DRF usa el Object-Relational Mapping (ORM) de Django para manejar modelos y realizar consultas a la base de datos sin escribir SQL.
- **Sistema de URLs**: Mejora el sistema de URLs de Django con un sistema de routers que crea automáticamente rutas de acceso a los recursos, simplificando la configuración de enrutamiento.
- **Vistas basadas en clases**: DRF extiende las vistas de Django con un nuevo concepto llamado Viewsets, que agrupa funcionalidades como listar, crear, actualizar y borrar dentro de una sola clase.

### ¿Qué añade Django REST para facilitar la creación de APIs?

Además de aprovechar la estructura de Django, Django REST agrega funcionalidades clave que hacen más fácil el desarrollo de APIs:

- **Serializadores**: Permiten transformar objetos Python a JSON y viceversa, facilitando la creación de APIs basadas en los modelos de Django sin tener que duplicar la información.
- **Viewsets**: Agrupan varias acciones en una sola clase, simplificando el código y reduciendo redundancias. Además, permiten manejar acciones según el método HTTP utilizado.
- **Mejoras en seguridad**: Gracias a la integración con Django, se pueden utilizar todas las configuraciones de seguridad como middleware y permisos.
- **Compatibilidad con Django Admin**: Permite seguir administrando la información de la aplicación a través de la interfaz administrativa de Django.

### ¿Cómo optimiza Django REST el desarrollo de APIs?

DRF optimiza varios aspectos del desarrollo de APIs al ofrecer herramientas que simplifican tareas comunes:

- **Enrutamiento automático de URLs** a través de routers.
- **Serialización eficiente** de datos basados en modelos Django, evitando la duplicación de lógica.
- **Manejo de vistas más flexible** con Viewsets que agrupan múltiples funcionalidades.
- **Continuidad con las funcionalidades de seguridad y administración de Django**, sin necesidad de configuraciones adicionales.

Para crear modelos y serializadores en Django REST Framework (DRF), sigue estos pasos:

### 1. **Instalar Django y Django REST Framework**
Asegúrate de que tienes tanto Django como Django REST Framework instalados en tu entorno.

```bash
pip install django djangorestframework
```

### 2. **Configurar un proyecto Django**
Crea un nuevo proyecto de Django si aún no lo tienes:

```bash
django-admin startproject myproject
cd myproject
```

Crea una nueva aplicación dentro del proyecto:

```bash
python manage.py startapp myapp
```

A continuación, añade `'rest_framework'` y tu aplicación `'myapp'` al archivo `settings.py` en la lista `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    ...
    'rest_framework',
    'myapp',
]
```

### 3. **Crear un Modelo en Django**
Los modelos en Django son la representación de la base de datos. Vamos a crear un modelo simple para una clase `Book` en el archivo `models.py` de tu aplicación.

```python
# myapp/models.py
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    published_date = models.DateField()
    isbn = models.CharField(max_length=13)
    pages = models.IntegerField()

    def __str__(self):
        return self.title
```

Luego, aplica las migraciones para que este modelo se cree en la base de datos:

```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. **Crear Serializadores en DRF**
Los serializadores en DRF permiten convertir los datos complejos como objetos de Django en tipos de datos nativos de Python que pueden ser fácilmente renderizados a JSON, XML o cualquier otro formato.

En el archivo `serializers.py` de tu aplicación, crea un serializador para el modelo `Book`.

```python
# myapp/serializers.py
from rest_framework import serializers
from .models import Book

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
```

Este serializador convierte las instancias del modelo `Book` en JSON y viceversa.

### 5. **Crear Vistas para manejar las solicitudes**
En el archivo `views.py`, crea vistas basadas en clases o funciones para manejar las solicitudes HTTP.

Por ejemplo, una vista genérica basada en clases para listar y crear libros:

```python
# myapp/views.py
from rest_framework import generics
from .models import Book
from .serializers import BookSerializer

class BookListCreateView(generics.ListCreateAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
```

### 6. **Configurar URLs**
Por último, configura las URLs en `urls.py` para que puedas acceder a la API.

```python
# myapp/urls.py
from django.urls import path
from .views import BookListCreateView

urlpatterns = [
    path('books/', BookListCreateView.as_view(), name='book-list-create'),
]
```

Y enlaza las URLs de tu aplicación en el archivo principal `urls.py` del proyecto:

```python
# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('myapp.urls')),
]
```

### 7. **Probar la API**
Inicia el servidor de desarrollo de Django:

```bash
python manage.py runserver
```

Ahora, puedes acceder a tu API en `http://127.0.0.1:8000/api/books/` para ver los libros o agregar uno nuevo.

### Resumen:
- **Modelos**: Representan la base de datos.
- **Serializadores**: Convierten instancias de modelos en datos JSON.
- **Vistas**: Manejan solicitudes HTTP.
- **URLs**: Rutas de acceso a las vistas.

Con estos pasos, has creado un modelo, un serializador y una API básica en Django REST Framework.

 para realizar pruebas `python manage.py shell` luego `from patients.serializers import PatientSerializer` luego `PatientSerializer`
 se crea los dato `data = {"first_name":"Luis", "last_name":"Martinez"}`
 se usa `serializer=PatientSerializer(data=data)` para ver los datos se utiliza `serializer` para ver si es valida se utiliza `serializer.is_valid()` para ver si la informacion es compatible o no el nuestra True o False, pare ver los errores ``

 ```bash
 (venv) PS C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursodeDjangoRestFramework\django-rest-framework> python manage.py shell
Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> from patients.serializers import PatientSerializer
>>> PatientSerializer
<class 'patients.serializers.PatientSerializer'>
>>> data = {"first_name":"Luis", "last_name":"Martinez"}
>>> serializer=PatientSerializer(data=data) 
>>> serializer  
PatientSerializer(data={'first_name': 'Luis', 'last_name': 'Martinez'}):
    id = IntegerField(label='ID', read_only=True)
    first_name = CharField(max_length=100)
    last_name = CharField(max_length=100)
    date_of_birth = DateField()
    contact_number = CharField(max_length=15)
    email = EmailField(max_length=254)
    address = CharField(style={'base_template': 'textarea.html'})
    medical_history = CharField(style={'base_template': 'textarea.html'})
>>> serializer.is_valid()
False
>>> serializer.errors
{'date_of_birth': [ErrorDetail(string='This field is required.', code='required')], 'contact_number': [ErrorDetail(string='This field is required.', code='required')], 'email': [ErrorDetail(string='This field is required.', code='required')], 'address': [ErrorDetail(string='This field is required.', code='required')], 'medical_history': [ErrorDetail(string='This field is required.', code='required')]}
 ```

**Lecturas recomendadas** 

[Models | Django documentation | Django](https://docs.djangoproject.com/en/5.1/topics/db/models/)

## Implementar vistas basadas en funciones en Django REST Framework

Cómo crear una vista basada en funciones que nos permita listar los pacientes en nuestra base de datos utilizando Django REST Framework y serializadores. Además, configuraremos las rutas y endpoints para consumir esta funcionalidad desde el frontend o cualquier cliente que utilice el API.

### ¿Cómo creamos una vista para listar pacientes utilizando serializadores?

Primero, abrimos nuestro archivo de vistas y realizamos las siguientes importaciones necesarias:

- Importamos el `PatientSerializer` desde los serializadores.
- Traemos el modelo `Patient` desde el archivo de Modelos.
- Importamos la clase `Response` desde Django REST Framework, que nos permitirá devolver datos en formato JSON o XML, entre otros.

Luego, creamos una función llamada `ListPatients` que será nuestra vista basada en funciones. Esta función hará una consulta a la base de datos para obtener todos los pacientes. Para esto, usamos `Patient.objects.all()` y guardamos el resultado en una variable.

### ¿Cómo usamos el serializador para manejar la lista de pacientes?

Una vez que obtenemos los pacientes, necesitamos serializar los datos. Para ello, usamos el `PatientSerializer`, pero como estamos serializando una lista de objetos, pasamos el parámetro `many=True`. Esto le indica al serializador que procese múltiples ítems.

La data serializada estará disponible en `serializer.data`, que será lo que devolvemos en el `Response`.

### ¿Cómo agregamos un decorador a nuestra vista?

Para que Django REST Framework reconozca nuestra vista, necesitamos usar el decorador `@api_view`. Lo importamos desde `rest_framework.decorators`. Este decorador se configura para que la vista solo acepte peticiones GET. De esta manera, evitamos que se utilicen otros métodos HTTP, como POST, en esta misma URL.

### ¿Cómo configuramos la URL para la vista?

Abrimos el archivo de configuración de URLs y creamos una nueva ruta. Asociamos el path `api-patients` con la vista `ListPatients`, importándola desde el archivo de vistas.

Guardamos todo y ejecutamos el servidor con el comando `python manage.py runserver`.

### ¿Qué muestra el API cuando accedemos al endpoint?

Al acceder a la URL `api-patients`, Django REST Framework nos muestra un listado de pacientes en formato JSON. Este listado incluye toda la información de los pacientes almacenados en la base de datos, con campos como nombres, apellidos y fechas. Las fechas aparecen en formato de cadena, aunque en el modelo de Python están como `DateTime`.

### ¿Qué reto sigue después de listar pacientes?

El siguiente paso es crear un nuevo endpoint que permita añadir pacientes a través del método POST. El reto será validar que los datos enviados coincidan con las reglas definidas en el modelo, usando nuevamente los serializadores.

## Gestión de Vistas Basadas en Funciones en Django REST Framework

La implementación de endpoints en Django REST Framework nos permite trabajar con recursos como pacientes, tanto para listarlos como para crearlos, mediante los métodos HTTP adecuados. El siguiente paso será extender estas funcionalidades para modificar y eliminar registros.

### ¿Cómo implementar la creación de pacientes con POST en el mismo endpoint?

Para permitir tanto la creación como la lectura de pacientes en un único endpoint, utilizamos los métodos GET y POST. GET se encarga de listar los pacientes, mientras que POST crea uno nuevo. Para lograr esto:

- Se verifica el método de la solicitud (GET o POST) usando `request.method`.
- Si es GET, se continúa listando los pacientes.
- Si es POST, se valida la información enviada en el cuerpo de la solicitud a través de un serializador.
- Si los datos son válidos, se utiliza el método `save()` para guardar el nuevo paciente en la base de datos.

### ¿Cómo se manejan los errores de validación en POST?

En caso de que los datos enviados no sean válidos, Django REST Framework captura los errores y los formatea en una respuesta JSON. Esto se hace con `raise_exception=True` en el serializador, lo que devuelve automáticamente una respuesta con los detalles de los errores sin necesidad de un condicional.

### ¿Cómo retornar una respuesta adecuada al crear un recurso?

Una vez que el paciente es creado correctamente, el servidor responde con un código de estado HTTP 201, indicando que el recurso fue creado. Esto se hace con `Response(status=status.HTTP_201_CREATED)`, asegurando que el cliente reciba la confirmación adecuada.

### ¿Cómo mostrar el detalle de un paciente con GET?

Para obtener el detalle de un paciente específico, se utiliza el método GET en un endpoint que incluye un parámetro de la URL, generalmente el ID del paciente:

- Se filtra el paciente por ID con `get_object_or_404()`.
- Si el paciente existe, se devuelve su información en formato JSON.
- Si no existe, se responde con un código de estado 404.

### ¿Cómo manejar la modificación de un paciente con PUT?

El método PUT permite modificar un paciente existente. Utiliza la misma lógica que GET para obtener el paciente, pero en lugar de devolver los datos, actualiza la información recibida:

- Se verifica si el método es PUT.
- Se validan los datos del paciente con un serializador.
- Si los datos son válidos, se guarda la actualización y se responde con un código 200 indicando éxito.

## Postman y cURL en Django REST Framework

Para probar de manera eficiente nuestras APIs, es fundamental dominar herramientas especializadas como Postman y Curl. Aunque Django ofrece una interfaz visual para pruebas, el uso de herramientas como estas nos permitirá realizar pruebas más flexibles y personalizadas en diferentes entornos, incluyendo servidores sin interfaz gráfica.

### ¿Cómo se utiliza Postman para probar una API?

Postman es una herramienta poderosa para interactuar con APIs. Permite realizar requests, gestionar colecciones y simular comportamientos de usuarios. Para probar nuestra API:

- Descarga e instala Postman desde su página principal.
- Accede a la interfaz donde puedes crear nuevos requests.
- Por ejemplo, para listar pacientes en un servidor local, usa la URL: `http://localhost:8000/api/patients`.
- Selecciona el método `GET` y presiona `Send`. Verás la lista de pacientes como respuesta.
- Postman también permite guardar cada request en una colección para su uso posterior, ideal para pruebas repetitivas.

### ¿Cómo se pueden manejar los requests en la línea de comandos con Curl?

Si no necesitas todas las funcionalidades de Postman o estás en un entorno sin ventanas, Curl es la opción adecuada. Curl te permite ejecutar requests directamente desde la consola, útil cuando estás trabajando en servidores.

- Abre una terminal y utiliza un comando Curl para hacer un request, por ejemplo, listar pacientes con:

`curl -X GET http://localhost:8000/api/patients`

- También puedes convertir fácilmente un request de Postman a Curl. En la interfaz de Postman, selecciona el ícono de código, copia el comando Curl generado y ejecútalo en la terminal.

### ¿Cómo crear un paciente nuevo usando Postman?

Para crear un nuevo recurso en nuestra API, como un paciente:

- Selecciona el método `POST` en Postman.
- Define el cuerpo de la petición en formato JSON, seleccionando `Body > Raw > JSON`. Por ejemplo:

```python
{
  "name": "Oscar Barajas",
  "age": 30,
  "email": "oscar@example.com"
}
```

- Ejecuta el request y asegúrate de que la respuesta indique que el recurso fue creado correctamente.
- También puedes generar el comando Curl correspondiente desde Postman y ejecutarlo en la consola.

### ¿Cómo combinar Postman y Curl para mejorar las pruebas?

Ambas herramientas se complementan bien. Postman facilita la creación y prueba de requests con una interfaz gráfica amigable, mientras que Curl te permite ejecutar esos mismos requests en entornos más limitados. Postman incluso puede generar el código Curl de un request, lo que es muy útil para integrar estos comandos en scripts automatizados o suites de pruebas.

**Lecturas recomendadas**

[Postman API Platform](https://www.postman.com/ "Postman API Platform")

[curl - Documentation Overview](https://curl.se/docs/ "curl - Documentation Overview")

## Refactorizar las funciones a clases en Django REST Framework

Refactorizar nuestras vistas basadas en funciones a vistas basadas en clases no solo mejora la organización del código, sino que también lo hace más escalable y reutilizable. En esta clase, hemos visto cómo Django REST Framework nos facilita aún más esta tarea al proporcionar vistas genéricas que reducen considerablemente la cantidad de código que tenemos que escribir manualmente.

### ¿Cómo refactorizar una vista basada en funciones a una basada en clases?

- Comenzamos importando APIView desde Django REST Framework.
- Creamos una nueva clase, heredando de `APIView`, donde definimos los métodos como `get`, `post`, o `delete`.
- Esto nos permite organizar mejor el código y evitar los condicionales que usamos en las vistas basadas en funciones.

### ¿Cómo conectar la vista basada en clases con una URL?

- Debemos importar la nueva vista en el archivo de URLs y reemplazar la vista basada en función por la basada en clase.
- Recordemos usar el método `as_view()` al conectarla en el archivo de URLs.

### ¿Qué beneficios ofrecen las vistas genéricas en Django REST?
Las vistas genéricas permiten simplificar aún más el código, reutilizando funcionalidad ya existente en Django REST:

- Usamos `ListAPIView` para simplificar una vista que solo lista elementos.
- Usamos `CreateAPIView` para manejar la creación de recursos.
- Podemos heredar de varias vistas genéricas a la vez para combinar funcionalidades, como listar y crear con pocas líneas de código.

### ¿Cómo funciona el QuerySet y el SerializerClass en las vistas genéricas?

- Definimos un QuerySet para obtener los datos que queremos listar o manipular.
- Asociamos una clase de serialización con `SerializerClass` para transformar los datos según las necesidades de nuestra API.
- Esto nos permite eliminar métodos como `get` o `post`, ya que se gestionan automáticamente.

### ¿Cómo evitar duplicar código?

Uno de los principales objetivos al usar clases es evitar la duplicación de código. Con vistas genéricas podemos reutilizar los mismos parámetros y métodos que ya vienen implementados, logrando que el código sea más limpio y fácil de mantener.

Breve resumen de HEAD y OPTIONS:

- **HEAD**: Se utiliza cuando solo necesitas los headers de una solicitud, sin el cuerpo (por ejemplo, para verificar la existencia o propiedades de un recurso).
- **OPTIONS**: Te dice qué métodos HTTP son soportados por un recurso, útil para seguridad y manejo de políticas entre dominios (CORS).

**Lecturas recomendadas**

[Django REST Framework 3.14 -- Classy DRF](https://www.cdrf.co/ "Django REST Framework 3.14 -- Classy DRF")

## Refactorizando vistas en Django REST Framework con vistas genéricas

Hemos visto cómo utilizar vistas genéricas en Django para crear vistas de detalle, simplificando el código y evitando la duplicación. A través de la clase `RetrieveUpdateDestroyAPIView`, podemos obtener, modificar o eliminar recursos de manera eficiente, reduciendo la cantidad de código a manejar.

### ¿Cómo evitar la duplicación de código con vistas genéricas?

- Django permite usar vistas genéricas como `RetrieveAPIView`, `UpdateAPIView` y `DestroyAPIView`.
- Sin embargo, es más eficiente usar la clase combinada **RetrieveUpdateDestroyAPIView**, que integra estas tres funcionalidades.
- Con esta clase podemos obtener, actualizar o eliminar un recurso sin necesidad de importar múltiples vistas.

### ¿Cómo funciona el refactor a las vistas genéricas?

- El código que antes obtenía el objeto y devolvía un error 404 si no se encontraba, ahora es reemplazado por una vista genérica que maneja esa lógica automáticamente.
- Al definir la vista genérica `RetrieveUpdateDestroyAPIView`, simplemente necesitamos definir las variables correspondientes, como el modelo y los permisos, y se manejan todas las operaciones CRUD (create, read, update, delete).
- Esto nos permite reducir significativamente el código y mantener la funcionalidad.

### ¿Cómo realizar validaciones con las vistas genéricas?

- Django continúa manejando validaciones, como las que se generan al enviar datos incorrectos, por ejemplo, una fecha inválida.
- Estas validaciones son útiles en formularios de frontend, ya que permiten mostrar al usuario por qué una solicitud ha fallado.

### ¿Qué sigue después de implementar vistas genéricas?

- El siguiente paso es usar view sets, que nos permitirán agrupar las vistas de una manera más eficiente y evitar la repetición de código.
- Aunque se ha logrado simplificar el código con las vistas genéricas, los view sets llevarán esta simplificación un paso más allá, agrupando operaciones similares en un solo conjunto.

Ventajas de utilizar vistas genéricas en Django Rest Framework:

1. **Reducción de código repetitivo**: Implementaciones predefinidas para operaciones comunes.
2. **Mejora de consistencia y mantenibilidad**: Aplicación de patrones uniformes en toda la API.
3. **Aceleración del desarrollo**: Permite centrarse en la lógica específica de la aplicación.
4. **Facilidad de integración y extensión**: Composición de comportamientos reutilizables.
5. **Promoción de buenas prácticas**: Adherencia a los principios de diseño de Django y REST.

**Lecturas recomendadas**

[Viewsets - Django REST framework](https://www.django-rest-framework.org/api-guide/viewsets/)

[Generic views - Django REST framework](https://www.django-rest-framework.org/api-guide/generic-views/)

## Documentación de APIs con Django REST, Swagger y OpenAPI

Cuando creamos una API, el objetivo principal es que otros sistemas o desarrolladores puedan integrarse con nuestro sistema de manera eficiente. Para lograr esto, una documentación clara y actualizada es fundamental. Herramientas como DRF Spectacular y Swagger nos facilitan esta tarea, automatizando la generación de la documentación y permitiendo que esté siempre sincronizada con nuestro código.

### ¿Cómo documentar una API automáticamente?

- Django y Django REST Framework (DRF) nos ofrecen la posibilidad de usar una librería llamada **DRF Spectacular**. Esta herramienta sigue el estándar OpenAPI para generar documentación automática.
- Este estándar permite que cualquier cambio en las vistas o en los endpoints de la API se refleje inmediatamente en la documentación, sin necesidad de modificarla manualmente.

### ¿Qué es Swagger y cómo usarlo para la documentación de tu API?

- **Swagger** es una interfaz visual que muestra la documentación generada por DRF Spectacular. Permite a los desarrolladores interactuar directamente con la API, probar los endpoints y revisar los parámetros y respuestas posibles.
- Además, ofrece la opción de descargar un archivo con el esquema OpenAPI que puede ser utilizado por otras herramientas o interfaces.

### ¿Cómo crear la aplicación de documentación en Django?

1. Crea una nueva aplicación llamada `docs` desde la terminal.
 - Registra esta aplicación en la lista de Installed Apps en el archivo settings.py.
2. Instala la librería **DRF Spectacular** ejecutando el comando `pip install drf-spectacular`.
 - Registra también esta librería en las aplicaciones instaladas.
3. Configura un esquema automático en `settings.py` asegurándote de que no haya duplicados en la configuración.

### ¿Cómo agregar las URLs de Swagger y Redoc?

- Dentro del archivo `urls.py` de la aplicación docs, agrega las URLs correspondientes a Swagger y Redoc.
- No olvides importar correctamente las rutas con `path` desde `django.urls`.
- Agrega las URLs de la aplicación `docs` en el archivo principal de URLs del proyecto para que estén accesibles.

### ¿Qué diferencia hay entre Swagger y Redoc?

- **Swagger** ofrece una interfaz donde puedes interactuar con los endpoints y probar las respuestas sin salir del navegador.
- **Redoc** es otra interfaz que permite navegar entre los endpoints de forma más organizada, con un buscador y una lista de los recursos disponibles. También muestra detalles de las respuestas y errores posibles.

### ¿Cómo mejorar la documentación de cada endpoint?

- Puedes agregar descripciones a cada uno de los endpoints en las clases de tus vistas, utilizando comentarios en Python.
- Estos comentarios aparecerán automáticamente en la documentación de Swagger o Redoc, facilitando a otros desarrolladores entender el comportamiento de cada recurso.

### ¿Qué ventajas ofrece el estándar OpenAPI?

- OpenAPI permite que cualquier herramienta que siga este estándar, como Swagger o Redoc, pueda interpretar el esquema de la API y generar documentación visual.
- Es un formato ampliamente utilizado y compatible con distintas interfaces de usuario.

### ¿Cómo actualizar la documentación al modificar el código?

- La principal ventaja de utilizar DRF Spectacular es que al modificar el código, la documentación se actualiza de forma automática. Esto garantiza que siempre esté sincronizada y evita que tengas que editar la documentación manualmente.

**Lecturas recomendadas**

[Home 2024 - OpenAPI Initiative](https://www.openapis.org/ "Home 2024 - OpenAPI Initiative")

## Listas Personalizadas y ViewSets en Django REST Framework

Los Viewsets en Django REST Framework nos ayudan a simplificar la creación de vistas al reutilizar una clase que agrupa el código necesario para manejar diferentes operaciones sobre un recurso, como listar, crear, actualizar y eliminar. Al integrarlos con los routers, evitamos la necesidad de definir cada URL manualmente, ya que el router se encarga de generar todas las rutas de manera automática.

### ¿Qué son los Viewsets y cómo funcionan?

- Un Viewset es una clase reutilizable que agrupa todas las operaciones que se suelen realizar con una vista (lista, detalle, creación, actualización, eliminación).
- Al usar Viewsets, reducimos la cantidad de clases y URLs que necesitamos escribir, ya que todas las operaciones se manejan desde un solo lugar.
- En lugar de crear múltiples clases, un solo Viewset puede manejar todas las acciones requeridas para un recurso.

### ¿Cómo se crea un Viewset?

- Importamos `ModelViewSet` desde rest_framework.viewsets.
- Definimos una clase que hereda de `ModelViewSet`, como DoctorViewset.
- Asignamos un `QuerySet` y un `Serializer` para definir cómo se gestionará la información y cómo será serializada.

```python
from rest_framework import viewsets
from .serializers import DoctorSerializer
from .models import Doctor

class DoctorViewset(viewsets.ModelViewSet):
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer
```

### ¿Cómo se registran los Viewsets con los routers?

- Los routers simplifican la creación de URLs, ya que generan las rutas automáticamente al registrar un Viewset.
- Usamos DefaultRouter de Django REST Framework para registrar el Viewset y generar las rutas correspondientes.

```python
from rest_framework.routers import DefaultRouter
from .viewsets import DoctorViewset

router = DefaultRouter()
router.register(r'doctors', DoctorViewset)
urlpatterns = router.urls
```

### ¿Cómo se prueban los Viewsets?

- Una vez registrado el Viewset, podemos verificar las URLs generadas ejecutando el servidor y accediendo a la API.
- Las operaciones de creación, actualización y eliminación de un recurso se pueden realizar directamente en las URLs generadas automáticamente.

### ¿Qué ventajas ofrecen los Viewsets y los routers?

- Evitamos la repetición de código al gestionar varias operaciones con una sola clase.
- Los routers generan automáticamente las rutas necesarias para cada recurso, lo que facilita su uso y mantenimiento.
- Las URLs generadas tienen nombres claros, lo que permite su uso programático dentro del código.

## Manejos de Acciones con ViewSet en Django REST Framework

Al crear APIs en Django REST Framework, no solo trabajamos con URLs para recursos, sino también con acciones que permiten ejecutar operaciones específicas, como el pago de una tarjeta o, en este caso, gestionar las vacaciones de un doctor.

### ¿Cómo agregar un campo para controlar el estado de vacaciones en un modelo?

- En el modelo Doctor, añadimos el campo `is_on_vacation`, que será un campo booleano con valor predeterminado `False`.
- Creamos las migraciones con `manage.py makemigrations` y luego ejecutamos `migrate` para aplicar los cambios en la base de datos.

### ¿Cómo crear una acción personalizada para activar o desactivar vacaciones?

- En los ViewSets de Django REST Framework, las acciones personalizadas se crean con el decorador `@action`. Importamos el decorador desde `rest_framework.decorators`.
- Definimos un método llamado toggle_vacation con el decorador y especificamos que solo se permitirá el método `POST`.
- El decorador también necesita el parámetro `detail=True` para que la acción se aplique a un recurso específico, como un doctor identificado por su ID en la URL.

### ¿Cómo implementar la lógica para alternar el estado de vacaciones?

- Utilizamos el método `get_object()` del ViewSet para obtener el objeto Doctor actual.
- La lógica alterna entre el valor `True` y `False` para el campo `is_on_vacation`. Si está en True, lo cambia a False y viceversa.
- Se guarda el objeto `Doctor` y se retorna una respuesta utilizando `Response` para informar el estado actualizado.

### ¿Cómo mejorar la idempotencia y claridad del endpoint?

- En lugar de alternar entre `True` y `False`, creamos dos acciones separadas: una para activar vacaciones (`set_on_vacation`) y otra para desactivarlas (`set_off_vacation`).
- Esto asegura que cada petición `POST` tenga un comportamiento predecible, lo que mejora la idempotencia del endpoint.

### ¿Cómo ajustar la URL de la acción para mejorar la legibilidad?

- Las URLs generadas a partir del nombre del método pueden tener guiones bajos, lo cual no es ideal para SEO y legibilidad. Usamos el parámetro `url_path` dentro del decorador `@action` para definir URLs con guiones, por ejemplo, `set-on-vacation`.

### ¿Cómo probar las acciones personalizadas?

- Desde la interfaz de Django REST Framework, probamos las acciones enviando peticiones `POST` a las URLs generadas.
- Verificamos que los doctores puedan ser marcados como en vacaciones o no, y que el campo `is_on_vacation` cambie correctamente en la base de datos.

### ¿Cómo replicar este proceso para otros recursos?

- Siguiendo este patrón, podemos crear acciones para otros recursos. Por ejemplo, un paciente puede necesitar obtener un reporte médico en formato JSON, lo cual sería una acción personalizada en el ViewSet de `Patient`.

## Autenticación y Gestión de Roles en Django REST Framework


En muchos sistemas, las APIs dependen de la autenticación y autorización para proteger recursos sensibles. Este artículo explica ambos conceptos a través de ejemplos y luego los implementa utilizando Django REST Framework.

¿Qué es la autenticación y cómo funciona en las APIs?
La autenticación se refiere a la comprobación de la identidad de un usuario. Imagina que llegas a un hotel, te solicitan tu documento de identificación, y de esta forma, demuestras quién eres. En el mundo digital, es similar: te identificas con un usuario y una contraseña. En Django, esta autenticación genera una cookie que luego es enviada en cada request para identificar y autorizar al usuario.

¿Cómo se implementa la autenticación en Django REST Framework?
Django REST Framework incluye múltiples sistemas de autenticación por defecto. Los más comunes son:

Session Authentication: Usa cookies y las credenciales del usuario almacenadas en la base de datos de Django.
Token Authentication: Similar a la llave de un hotel, donde el token identifica al usuario después de autenticarse.
Para implementar el sistema de autenticación en Django, se configuran las clases de autenticación dentro de settings.py, lo cual permite que solo los usuarios autenticados interactúen con los endpoints.

¿Cómo se configura la autorización en Django REST?
La autorización determina qué puede hacer un usuario autenticado. En el ejemplo del hotel, tener la llave te permite acceder solo a tu habitación, pero no a otras. En Django, se define qué usuarios tienen permiso para acceder o modificar ciertos recursos.

Para configurar esto:

Se añaden Permission Classes en los viewsets.
Se utiliza la clase IsAuthenticated para requerir que el usuario esté logueado antes de realizar cualquier acción.
¿Cómo manejar permisos más avanzados en Django REST Framework?
En algunos casos, es necesario definir permisos personalizados. Por ejemplo, solo los doctores deberían poder modificar ciertos recursos. Para implementar esto, puedes:

Crear grupos de usuarios, como el grupo “Doctors”.
Definir clases personalizadas de permisos, como IsDoctor, que verifica si el usuario pertenece al grupo adecuado.
Este sistema permite implementar roles de usuario más complejos, asegurando que solo aquellos con permisos específicos puedan realizar ciertas acciones.

¿Cómo probar la autenticación y autorización en Django?
Después de configurar todo, es importante probar que los permisos y la autenticación funcionan como se espera. Esto incluye:

Probar con usuarios que tienen acceso y verificar que pueden realizar las acciones permitidas.
Probar con usuarios sin permisos y asegurarse de que reciban los errores apropiados (como 401 o 403).
Con esta configuración, tus APIs estarán protegidas y listas para manejar autenticación y permisos de manera segura y eficiente.

## Autenticación y Gestión de Roles en Django REST Framework


En muchos sistemas, las APIs dependen de la autenticación y autorización para proteger recursos sensibles. Este artículo explica ambos conceptos a través de ejemplos y luego los implementa utilizando Django REST Framework.

### ¿Qué es la autenticación y cómo funciona en las APIs?

La autenticación se refiere a la comprobación de la identidad de un usuario. Imagina que llegas a un hotel, te solicitan tu documento de identificación, y de esta forma, demuestras quién eres. En el mundo digital, es similar: te identificas con un usuario y una contraseña. En Django, esta autenticación genera una cookie que luego es enviada en cada request para identificar y autorizar al usuario.

### ¿Cómo se implementa la autenticación en Django REST Framework?

Django REST Framework incluye múltiples sistemas de autenticación por defecto. Los más comunes son:

- **Session Authentication**: Usa cookies y las credenciales del usuario almacenadas en la base de datos de Django.
- **Token Authentication**: Similar a la llave de un hotel, donde el token identifica al usuario después de autenticarse.

Para implementar el sistema de autenticación en Django, se configuran las clases de autenticación dentro de settings.py, lo cual permite que solo los usuarios autenticados interactúen con los endpoints.

### ¿Cómo se configura la autorización en Django REST?

La autorización determina qué puede hacer un usuario autenticado. En el ejemplo del hotel, tener la llave te permite acceder solo a tu habitación, pero no a otras. En Django, se define qué usuarios tienen permiso para acceder o modificar ciertos recursos.

Para configurar esto:

1. Se añaden **Permission Classes** en los viewsets.
2. Se utiliza la clase `IsAuthenticated` para requerir que el usuario esté logueado antes de realizar cualquier acción.

### ¿Cómo manejar permisos más avanzados en Django REST Framework? 

En algunos casos, es necesario definir permisos personalizados. Por ejemplo, solo los doctores deberían poder modificar ciertos recursos. Para implementar esto, puedes:

- Crear grupos de usuarios, como el grupo “Doctors”.
- Definir clases personalizadas de permisos, como `IsDoctor`, que verifica si el usuario pertenece al grupo adecuado.

Este sistema permite implementar roles de usuario más complejos, asegurando que solo aquellos con permisos específicos puedan realizar ciertas acciones.

### ¿Cómo probar la autenticación y autorización en Django?

Después de configurar todo, es importante probar que los permisos y la autenticación funcionan como se espera. Esto incluye:

- Probar con usuarios que tienen acceso y verificar que pueden realizar las acciones permitidas.
- Probar con usuarios sin permisos y asegurarse de que reciban los errores apropiados (como 401 o 403).

Con esta configuración, tus APIs estarán protegidas y listas para manejar autenticación y permisos de manera segura y eficiente.

Crear un super usuario: `python manage.py createsuperuser`

**Lecturas recomendadas**

[How to use sessions | Django documentation | Django](https://docs.djangoproject.com/en/4.0/topics/http/sessions/ "How to use sessions | Django documentation | Django")

## Manejo de Errores y Validaciones

Validar datos correctamente es clave para asegurar que una aplicación funcione de manera segura y confiable. En este caso, exploramos cómo implementar validaciones personalizadas en serializadores de Django REST Framework para garantizar que los datos cumplan con los requisitos específicos del negocio.

### ¿Cómo implementamos una validación personalizada en Django REST Framework?

Django ya ofrece validaciones básicas, como verificar que un campo sea un email. Sin embargo, para casos específicos, como asegurarse de que un correo pertenezca a un dominio corporativo, necesitamos crear validaciones personalizadas en el serializador. Esto lo logramos usando el método `validate_<nombre_del_campo>`. Por ejemplo, para validar que un correo termine en `@example.com`, implementamos lo siguiente:

- Definimos el método `validate_email` dentro del serializador.
- Verificamos si el valor del campo contiene el dominio correcto.
- Si es válido, retornamos el valor; si no, levantamos una excepción con un mensaje de error.

### ¿Cómo manejar errores de validación en casos más complejos?

Para validaciones que dependen de múltiples campos, como validar el número de contacto y el estado de vacaciones de un doctor, usamos el método general validate. Este método permite acceder a todos los campos del serializador en forma de diccionario y aplicar lógica personalizada.
Por ejemplo:

- Validamos que el número de contacto tenga al menos 10 caracteres.
- Si el doctor está de vacaciones (`is_on_vacation es True`) y el número no es válido, lanzamos una excepción que indica que debe corregirse el número antes de continuar.

### ¿Qué debemos hacer si hay múltiples validaciones?

En casos donde existen múltiples validaciones, podemos usar un diccionario que devuelva los valores de todos los campos y agregar la lógica en consecuencia. Esto es útil cuando debemos validar múltiples condiciones que se interrelacionan.

### ¿Cómo lanzamos errores personalizados en serializadores?

Django REST Framework nos permite lanzar excepciones personalizadas que se retornan como un error en formato JSON. Usamos `raise serializers.ValidationError` para generar estos errores con mensajes específicos, ayudando a los usuarios a corregir los datos enviados antes de que se procesen.

**Ventajas de manejar errores:**

1. **Mejora la experiencia del usuario**: Proporciona mensajes claros y útiles cuando ocurren errores.
2. **Aumenta la seguridad**: Previene la exposición de información sensible a través de mensajes de error.
3. **Facilita la depuración**: Ayuda a los desarrolladores a identificar y solucionar problemas más rápidamente.

**Ventajas de manejar validaciones:**

1. **Garantiza la integridad de los datos**: Asegura que solo datos correctos y esperados sean procesados.
2. **Previene errores**: Reduce la posibilidad de errores en el sistema al validar la entrada de datos.
3. **Mejora la seguridad**: Evita la entrada de datos maliciosos que podrían comprometer el sistema.

En Django REST Framework (DRF), el manejo de errores y las validaciones son componentes cruciales para crear APIs robustas. DRF facilita la creación de validaciones personalizadas y el manejo adecuado de errores, ya sea en los serializadores, vistas o directamente en las respuestas de la API.

### Validaciones en Serializers

Los **serializers** son el lugar donde generalmente ocurre la mayoría de las validaciones de datos. DRF proporciona métodos incorporados para validar datos en un `ModelSerializer`.

#### 1. Validaciones de Campos
DRF realiza validaciones automáticas basadas en los tipos de campo definidos en el modelo, pero también puedes agregar validaciones personalizadas en cada campo.

Ejemplo:

```python
from rest_framework import serializers
from .models import Patient

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'

    # Validación personalizada en un campo específico
    def validate_age(self, value):
        if value < 0:
            raise serializers.ValidationError("La edad no puede ser negativa.")
        return value
```

#### 2. Validaciones Globales
Puedes realizar validaciones que afecten a múltiples campos a la vez usando el método `validate` en el serializador.

Ejemplo:

```python
class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'

    # Validación global (en múltiples campos)
    def validate(self, data):
        if data['age'] < 18 and not data.get('guardian'):
            raise serializers.ValidationError("Pacientes menores de edad deben tener un guardián.")
        return data
```

### Manejo de Errores en Vistas

En Django REST Framework, puedes manejar errores y excepciones en las vistas, ya sea a través de excepciones incorporadas o personalizadas.

#### 1. Excepciones Incorporadas
DRF tiene una serie de excepciones predeterminadas que puedes usar, como:

- `ValidationError`: Para errores de validación.
- `NotFound`: Cuando un recurso no existe.
- `PermissionDenied`: Cuando el usuario no tiene los permisos necesarios.
- `NotAuthenticated`: Si se requiere autenticación pero no está presente.

Ejemplo:

```python
from rest_framework.exceptions import NotFound
from rest_framework.views import APIView
from .models import Patient
from .serializers import PatientSerializer

class PatientDetailView(APIView):
    def get(self, request, pk):
        try:
            patient = Patient.objects.get(pk=pk)
        except Patient.DoesNotExist:
            raise NotFound("El paciente con este ID no existe.")

        serializer = PatientSerializer(patient)
        return Response(serializer.data)
```

#### 2. Manejo de Excepciones Globales

Si quieres manejar los errores a nivel global, puedes personalizar el comportamiento utilizando un middleware o sobreescribiendo la función de excepción de DRF.

```python
# En settings.py
REST_FRAMEWORK = {
    'EXCEPTION_HANDLER': 'my_project.my_app.utils.custom_exception_handler',
}
```

Y luego defines tu función personalizada de manejo de errores:

```python
from rest_framework.views import exception_handler

def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if response is not None:
        response.data['status_code'] = response.status_code
        response.data['error'] = str(exc)

    return response
```

### Manejo de Validaciones Personalizadas en Formularios

También puedes agregar validaciones personalizadas usando el atributo `validators` en los serializadores para campos específicos.

```python
from rest_framework import serializers
from django.core.validators import RegexValidator

class PatientSerializer(serializers.ModelSerializer):
    phone_number = serializers.CharField(
        validators=[RegexValidator(regex='^(\+)?[0-9]{10,15}$', message="Número de teléfono inválido")]
    )
    
    class Meta:
        model = Patient
        fields = '__all__'
```

### Respuestas Personalizadas en Caso de Error

Puedes crear respuestas personalizadas cuando ocurre un error en la validación de los datos en tu API. Por ejemplo, puedes sobrescribir cómo los errores son mostrados en las respuestas.

```python
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

class PatientDetailView(APIView):
    def post(self, request):
        serializer = PatientSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response({
            'status': 'error',
            'message': 'Datos inválidos',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
```

### Resumen

- **Validaciones en Serializers**: Usa `validate_<field>` para validaciones de campo, y el método `validate` para validaciones que involucran múltiples campos.
- **Manejo de Excepciones**: Puedes usar excepciones predefinidas o personalizadas en tus vistas.
- **Respuestas de Error Personalizadas**: Puedes sobrescribir el formato de los errores de validación para personalizar las respuestas JSON.
- **Excepciones Globales**: Puedes definir un manejador de excepciones global para todo el proyecto a través de la configuración de `EXCEPTION_HANDLER`.

Estos mecanismos te permitirán gestionar errores y validaciones de manera eficiente en tu proyecto Django REST Framework.

**Lecturas recomendadas**

[Serializing Django objects | Django documentation | Django](https://docs.djangoproject.com/en/stable/topics/serialization/#serializing-data "Serializing Django objects | Django documentation | Django")

[Exceptions - Django REST framework](https://www.django-rest-framework.org/api-guide/exceptions/ "Exceptions - Django REST framework")

## ¿Cómo implementar serializadores anidados en Django Django REST Framework?

Los serializadores anidados permiten incluir datos de otros modelos directamente en un serializador, lo que resulta útil al necesitar información relacionada en un solo response. En esta clase, aplicamos esta técnica para incluir una lista de citas médicas dentro del recurso de pacientes en la aplicación DoctorApp. Esto mejora la eficiencia en el manejo de relaciones entre modelos, y facilita cambios futuros en la estructura de los response de la API.

### ¿Cómo implementar un serializador anidado en Django?

- Crea un nuevo campo dentro del serializador principal.
- Importa el serializador del modelo que deseas anidar.
- Define el campo con el serializador importado y marca como `Read Only` si es necesario.
- Asegúrate de incluir el nuevo campo en la lista de `fields` del serializador para que se refleje en el response.

### ¿Cómo anidar citas dentro del serializador de pacientes?

Para incluir las citas médicas de un paciente, sigue estos pasos:

1. Abre el serializador de pacientes.
2. Agrega un nuevo campo llamado `appointments` que usará el `AppointmentsSerialize`r.
3. Importa el serializador de citas médicas desde su respectivo módulo (`Bookings.Serializers`).
4. Configura el campo con `many=True` y `read_only=True`, ya que es una lista de citas que solo puede ser visualizada.
5. Verifica que el campo se ha agregado correctamente al incluirlo en la lista de campos del serializador.

### ¿Cómo validar la implementación?

1. Ejecuta el servidor de desarrollo con manage.py runserver.
2. Accede al recurso Patients en la API y revisa si aparece el campo appointments.
3. En caso de que falte algún campo, como el ID, asegúrate de incluirlo en el serializador.

### ¿Cómo crear y visualizar citas en la consola?
Para crear una cita desde la consola de comandos:

1. Abre la consola con manage.py shell.
2. Importa los modelos relevantes (Paciente, Doctor, Appointment).
3. Define variables para el paciente y el doctor.
4. Crea una nueva cita usando el manager de appointments.
5. Recarga la página para verificar que el array de citas ya contiene información en formato JSON.

### ¿Cómo usar serializadores anidados para otros modelos?

El uso de serializadores anidados no se limita a las citas de los pacientes. Puedes replicar este mismo enfoque para otros recursos. Por ejemplo, podrías crear un serializador para listar las citas asociadas a un doctor, proporcionando una mayor flexibilidad a la API y haciendo que las relaciones entre modelos sean más visibles y accesibles.

```bash
python manage.py shell
Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> from datetime import date, time
>>> 
>>> from doctors.models import Doctor 
>>> from patients.models import Patient
>>> from bookings.models import Appointment 
>>>
>>> Appointment.objects.create(
...      patient=patient,
...      doctor=doctor,
...      appointment_date=date(2024, 10, 12),
...      appointment_time=time(9, 0),
...      notes="Ejemplo",
...      status="HECHA"
...      )
<Appointment: Appointment object (1)>
```
## ¿Cómo usar SerializerMethodField en Django REST Framework?

A veces necesitamos calcular y mostrar valores derivados en el resultado de un endpoint sin alterar el modelo de datos original. Un ejemplo común es calcular la edad de un paciente a partir de su fecha de nacimiento. Para ello, podemos utilizar el SerializerMethodField en Django REST Framework. Este campo permite realizar cálculos utilizando los datos del modelo, como veremos a continuación.

### ¿Cómo calculamos un valor con `SerializerMethodField`?

Para calcular un valor, primero debemos definir un nuevo campo en nuestro serializador usando `SerializerMethodField`. Este tipo de campo permite definir un método que realizará el cálculo y retornará el valor deseado. Aquí te mostramos cómo hacerlo:

- Importa `SerializerMethodField` desde el módulo `serializers`.
- Define un nuevo campo en el serializador, por ejemplo, “Age” para calcular la edad.
- Si no especificas un método con el argumento `method_name`, Django REST Framework generará un nombre por defecto en la forma `get_`.

### ¿Cómo calculamos la edad usando la fecha de nacimiento?

La clave del cálculo es restar la fecha de nacimiento del paciente a la fecha actual. Este proceso genera un objeto `timedelta`, que representa la diferencia en días. Para convertirlo a años, sigue estos pasos:

1. Importa `date` desde el módulo `datetime`, que es suficiente ya que trabajamos con fechas (`no datetime`).
2. Obtén la fecha actual utilizando `date.today()`.
3. Calcula la diferencia entre la fecha actual y la fecha de nacimiento.
4. Divide esta diferencia en días por 365 para obtener la edad aproximada en años.
5. Retorna el valor numérico o, si es necesario, formatea el resultado como un string.

## Ejemplo de código:

```python
from rest_framework import serializers
from datetime import date

class PatientSerializer(serializers.ModelSerializer):
    age = serializers.SerializerMethodField()

    def get_age(self, obj):
        today = date.today()
        age_timedelta = today - obj.date_of_birth
        age = age_timedelta.days // 365  # Convertimos días a años
        return age

    class Meta:
        model = Patient
        fields = ['name', 'date_of_birth', 'age']
```

### ¿Qué sucede si obtenemos resultados incorrectos?

Un problema común al calcular la edad es no acceder correctamente al atributo `days` del objeto `timedelta`. Si simplemente restamos las fechas, obtendremos un objeto `timedelta`, que necesitamos dividir por 365 para convertirlo en años.

Otro detalle importante es no incluir texto como “años” en el resultado, ya que es preferible dejar el formato de presentación (e.g., el idioma) en manos del frontend.

### ¿Cómo calculamos la experiencia de un doctor?

Siguiendo el mismo patrón que para calcular la edad, podemos calcular la experiencia de un doctor usando su fecha de inicio de trabajo. Solo es necesario reemplazar la fecha de nacimiento con la fecha de inicio laboral.

### Ejemplo de código para la experiencia:

```python
class DoctorSerializer(serializers.ModelSerializer):
    experience = serializers.SerializerMethodField()

    def get_experience(self, obj):
        today = date.today()
        experience_timedelta = today - obj.start_date
        experience = experience_timedelta.days // 365
        return experience

    class Meta:
        model = Doctor
        fields = ['name', 'start_date', 'experience']
```

### ¿Qué otras aplicaciones tiene el `SerializerMethodField`?

- Calcular otros valores derivados sin alterar el modelo de datos.
- Agregar lógica personalizada en el serializador sin tocar la base de datos.
- Permitir mostrar valores preprocesados para el frontend sin requerir cambios en el backend.

**Lecturas recomendadas**

[datetime — Basic date and time types — Python 3.12.6 documentation](https://docs.python.org/3/library/datetime.html "datetime — Basic date and time types — Python 3.12.6 documentation")

## Endpoint Anidado para Appointments Usando @action

El endpoint para agendar una cita es esencial dentro de la lógica de negocio, ya que permite la interacción entre pacientes y doctores de manera eficiente. A través de este endpoint, un paciente puede reservar una cita con un doctor, cumpliendo con las mejores prácticas de REST y utilizando un viewset anidado para aprovechar los recursos previamente creados.

### ¿Cómo se estructura la URL para agendar una cita?

La URL para agendar una cita sigue una estructura anidada basada en el ID del doctor. Utilizamos el recurso existente `/doctors/{id}` para obtener detalles de un doctor, y sobre esta misma estructura se agregan las citas con el endpoint `/appointments`. Según REST, un GET en este endpoint devolverá una lista de citas, mientras que un POST permitirá crear una nueva.

### ¿Cómo se implementa la acción para crear una cita?

Para implementar la acción, es necesario definir un método en el viewset del doctor, que maneje tanto GET como POST. El objetivo principal del POST es recibir los datos de la cita que desea agendar el usuario y crearla utilizando un `Serializer`. Aquí, el ID del doctor se obtiene de la URL, asegurando que no se pueda modificar desde el formulario.

Pasos clave:

- Se importa el AppointmentSerializer desde el módulo bookings.
- Se recibe la data del request y se agrega el ID del doctor a dicha data.
- Se valida la información a través del método isValid.
- Finalmente, se guarda la cita con Serializer.save() y se retorna un estado 201 (creado).

### ¿Cómo se filtran las citas de un doctor?

Para retornar las citas de un doctor con un GET, se filtran las citas por el ID del doctor utilizando el ORM de Django. El método `filter` se encarga de traer todas las citas asociadas al doctor, las cuales se serializan y se devuelven en formato JSON.

### ¿Cómo se valida la información recibida?

La validación se realiza utilizando el `Serializer`, el cual se asegura de que los datos cumplan con las reglas establecidas. En caso de que la información no sea válida, se lanza una excepción mostrando un error claro al usuario.

### ¿Cómo se maneja el estado de las respuestas?

Los estados HTTP se manejan a través del módulo `status` de Django REST. En el caso de crear una cita, se retorna un estado 201 para indicar que la cita fue creada correctamente. Para las demás acciones, el estado por defecto es 200, indicando que la solicitud fue exitosa.

## Pruebas Unitarias para Endpoints Anidados Usando APIClient


Las pruebas unitarias son esenciales para garantizar que nuestras APIs funcionen correctamente sin tener que gastar demasiados recursos. Django REST Framework facilita este proceso mediante la clase APIClient, que nos permite simular requests y validar los resultados de forma sencilla y eficiente. A continuación, aprenderemos cómo crear pruebas unitarias en un proyecto de Django utilizando esta herramienta.

### ¿Cómo se configuran las pruebas unitarias en Django REST Framework?

Para comenzar a crear pruebas en Django REST Framework, necesitamos trabajar con el archivo `test.py`, el cual se genera automáticamente al crear un proyecto. En este archivo, definimos nuestras pruebas heredando de la clase `TestCase`, que proporciona todas las funcionalidades necesarias para ejecutar tests en Django.

Dentro de la clase de pruebas, usamos el método `setUp` para preparar datos comunes que reutilizaremos en todas nuestras pruebas, como la creación de un paciente y un doctor. Aquí, empleamos el ORM de Django para manejar los modelos fácilmente.

#### ¿Qué es el cliente API y cómo se usa?

El cliente `APIClient` es esencial para nuestras pruebas ya que simula requests HTTP, permitiéndonos probar las respuestas de nuestra API sin hacer requests reales. Esto nos ahorra tiempo y recursos. Además, se configura automáticamente para trabajar con datos JSON, simplificando las pruebas.

Importamos el cliente usando:

`from rest_framework.test import APIClient`

Esto nos permite realizar operaciones como `GET`, `POST`, `PUT`, y más, directamente desde nuestras pruebas. Por ejemplo, para verificar que una lista de “appointments” devuelve un código 200, simplemente escribimos un test que utiliza el cliente para hacer un request `GET` a la URL de las citas.

Instalar `pip install django-extensions`

muestra todas las urls de la app `python manage.py show_urls`

### ¿Cómo validamos los resultados de las pruebas?

Django REST Framework proporciona el módulo status, que nos permite verificar los códigos de respuesta de manera sencilla. En las pruebas, utilizamos el método self.assertEqual() para comparar el código de estado devuelto por la API con el valor esperado:

```python
from rest_framework import status
self.assertEqual(response.status_code, status.HTTP_200_OK)
```

Esto nos asegura que el código de la API está funcionando correctamente según lo esperado.

### ¿Cómo se manejan las URLs en las pruebas?

Para obtener las URLs dinámicamente en nuestras pruebas, utilizamos el método reverse() de Django, que permite construir URLs basadas en sus nombres. Esto es especialmente útil cuando trabajamos con URLs que requieren parámetros, como IDs.

### ¿Cómo solucionamos errores de permisos en nuestras pruebas?

Es común que algunas vistas en Django REST Framework requieran autenticación o permisos especiales. Si nuestras pruebas fallan debido a permisos, podemos ajustar las configuraciones en el viewset, asegurándonos de que las pruebas se realicen bajo las mismas condiciones que los usuarios reales enfrentarían. Por ejemplo, si solo los doctores pueden ver ciertos datos, debemos asegurarnos de que el usuario en la prueba tenga esos permisos.

### ¿Qué hacer cuando una prueba falla inesperadamente?

Si una prueba falla, es crucial revisar el error devuelto y ajustar el código según sea necesario. A veces, la falla puede deberse a errores en los permisos o configuraciones en los viewsets. Al corregir estos errores y volver a ejecutar la prueba, podemos validar que los ajustes realizados han solucionado el problema.

Para realizar pruebas unitarias de **endpoints anidados** usando `APIClient` en Django REST Framework, puedes simular las solicitudes HTTP a tus endpoints dentro de un contexto de prueba. Aquí te explico cómo hacerlo paso a paso.

### 1. **Escenario: Endpoints Anidados**
Supongamos que tienes un endpoint de **doctores** y uno de **pacientes** anidado bajo los doctores. Un ejemplo de URLs anidadas podría ser algo como:

- `/api/doctors/` → Lista de doctores.
- `/api/doctors/{doctor_id}/patients/` → Lista de pacientes para un doctor en particular.

### 2. **Configuración de las Pruebas**
Primero, asegúrate de tener configurado tu `APIClient` y tus modelos.

#### Modelos
Si tienes los siguientes modelos:

```python
# models.py
from django.db import models

class Doctor(models.Model):
    name = models.CharField(max_length=100)
    specialization = models.CharField(max_length=100)

class Patient(models.Model):
    name = models.CharField(max_length=100)
    doctor = models.ForeignKey(Doctor, related_name='patients', on_delete=models.CASCADE)
```

#### Serializers
Los serializers podrían verse así:

```python
# serializers.py
from rest_framework import serializers
from .models import Doctor, Patient

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ['id', 'name']

class DoctorSerializer(serializers.ModelSerializer):
    patients = PatientSerializer(many=True, read_only=True)

    class Meta:
        model = Doctor
        fields = ['id', 'name', 'specialization', 'patients']
```

#### ViewSets
Las vistas para los endpoints anidados:

```python
# views.py
from rest_framework import viewsets
from .models import Doctor, Patient
from .serializers import DoctorSerializer, PatientSerializer

class DoctorViewSet(viewsets.ModelViewSet):
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer

class PatientViewSet(viewsets.ModelViewSet):
    serializer_class = PatientSerializer

    def get_queryset(self):
        return Patient.objects.filter(doctor_id=self.kwargs['doctor_pk'])
```

Y en el archivo de **urls.py**:

```python
from rest_framework.routers import DefaultRouter
from .views import DoctorViewSet, PatientViewSet

router = DefaultRouter()
router.register(r'doctors', DoctorViewSet)

# URL anidadas para pacientes de un doctor específico
from rest_framework_nested import routers
doctor_router = routers.NestedSimpleRouter(router, r'doctors', lookup='doctor')
doctor_router.register(r'patients', PatientViewSet, basename='doctor-patients')

urlpatterns = router.urls + doctor_router.urls
```

### 3. **Prueba de Endpoints Anidados**
Para realizar una prueba unitaria sobre los endpoints anidados usando `APIClient`, sigue los pasos siguientes:

#### Crear el archivo de pruebas

```python
# tests.py
from rest_framework.test import APITestCase, APIClient
from django.urls import reverse
from rest_framework import status
from .models import Doctor, Patient

class DoctorPatientAPITests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        
        # Crear un doctor y pacientes
        self.doctor = Doctor.objects.create(name="Dr. Smith", specialization="Cardiology")
        self.patient1 = Patient.objects.create(name="John Doe", doctor=self.doctor)
        self.patient2 = Patient.objects.create(name="Jane Doe", doctor=self.doctor)
        
        # URL para obtener la lista de pacientes de un doctor específico
        self.patients_url = reverse('doctor-patients-list', kwargs={'doctor_pk': self.doctor.id})

    def test_get_patients_for_doctor(self):
        """Prueba para obtener la lista de pacientes de un doctor específico"""
        
        # Simular una solicitud GET al endpoint de pacientes
        response = self.client.get(self.patients_url)
        
        # Comprobar que la respuesta tenga código 200 (éxito)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verificar que los pacientes se devuelvan correctamente en la respuesta
        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.data[0]['name'], self.patient1.name)
        self.assertEqual(response.data[1]['name'], self.patient2.name)

    def test_create_patient_for_doctor(self):
        """Prueba para crear un paciente bajo un doctor específico"""
        
        # Datos para el nuevo paciente
        data = {
            'name': 'Tom Doe'
        }
        
        # Simular una solicitud POST al endpoint de pacientes
        response = self.client.post(self.patients_url, data, format='json')
        
        # Comprobar que la respuesta tenga código 201 (creado)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Verificar que el paciente fue creado correctamente y asociado al doctor
        self.assertEqual(Patient.objects.filter(doctor=self.doctor).count(), 3)
        self.assertEqual(response.data['name'], 'Tom Doe')
```

### Explicación de la Prueba

1. **`setUp`**: Configuramos el cliente de la API (`APIClient`), creamos un `Doctor` y dos `Patient`. También generamos la URL de los pacientes anidados bajo un doctor específico.
   
2. **`test_get_patients_for_doctor`**: 
   - Enviamos una solicitud GET al endpoint anidado `/doctors/{doctor_id}/patients/`.
   - Verificamos que la respuesta tenga código de estado `200 OK`.
   - Comprobamos que se devuelvan los pacientes correctos en la respuesta.

3. **`test_create_patient_for_doctor`**: 
   - Enviamos una solicitud POST para crear un nuevo paciente bajo el doctor.
   - Verificamos que el código de estado sea `201 CREATED`.
   - Comprobamos que el paciente fue correctamente añadido a la base de datos y asociado al doctor.

### 4. **Ejecutar las pruebas**
Para ejecutar las pruebas, utiliza el siguiente comando en la terminal:

```bash
python manage.py test
```

Esto ejecutará las pruebas en tu proyecto y validará que los endpoints anidados funcionen correctamente.

## Throttling en Django REST Framework

Limitar las solicitudes a una API es fundamental para evitar abusos y proteger los recursos del servidor. El throttling es una técnica clave en este proceso, ya que permite controlar la cantidad de solicitudes que diferentes usuarios pueden hacer en un determinado periodo, previniendo ataques como DDoS y optimizando el rendimiento.

### ¿Cómo implementar throttling en Django REST?

Para controlar las solicitudes en Django REST, es importante definir reglas específicas. Estas reglas pueden basarse en el estado del usuario, como si está autenticado o es anónimo, o incluso establecer limitaciones distintas para usuarios VIP.

- Primero, debemos entender que el throttling se configura de manera similar a los permisos y autenticación.
- Definimos límites como “requests por minuto”, y estos valores pueden ser diferentes para usuarios anónimos o autenticados.

### ¿Cómo definir reglas de throttling en Django REST?

La documentación de Django REST proporciona ejemplos claros para limitar las solicitudes de acuerdo al tipo de usuario:

- Para usuarios anónimos: 100 solicitudes por día.
- Para usuarios autenticados: 1000 solicitudes por día.

Estas reglas pueden configurarse fácilmente para ser más estrictas, limitando, por ejemplo, a 5 solicitudes por minuto para usuarios anónimos.

### ¿Cómo probar la configuración?

1. **Modificar la configuración**: Añade las reglas de throttling al diccionario de configuración de Django REST. Para limitar a 5 solicitudes por minuto, establece la tasa en 'minute': 5 para usuarios anónimos.
2. **Ejecutar el servidor**: Después de realizar los cambios, corre el servidor de desarrollo y prueba enviando solicitudes repetidas.
3. **Verificación en la terminal**: Al alcanzar el límite de solicitudes, Django REST mostrará el error “too many requests” en la terminal, indicando que el sistema de throttling está funcionando correctamente.

### ¿Qué sucede cuando el límite es alcanzado?

Si un usuario anónimo intenta hacer más de 5 solicitudes en un minuto, verá un error que le informará que ha alcanzado el límite de solicitudes permitidas. Después de esperar unos segundos, el sistema volverá a permitir solicitudes. Este proceso asegura que los recursos del servidor no se saturen con solicitudes abusivas o incorrectas.

### Qué es el Throttling

El throttling es una técnica utilizada en aplicaciones web, especialmente en APIs, para **controlar la cantidad de solicitudes** que un cliente puede hacer en un período de tiempo específico. Su objetivo principal es prevenir abusos, proteger recursos y garantizar un rendimiento óptimo.

De manera visual, un throttle consiste en bloquear la invocación de funciones hasta que se **complete un tiempo determinado o una acción en específico**. De esta manera reducimos el número de peticiones a la API. Lo que conlleva reducir costos y prevenir ataques.

![thottling](thottling.png)

Un concepto similar es Debounce. Pero te lo dejo de tarea.