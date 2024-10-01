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