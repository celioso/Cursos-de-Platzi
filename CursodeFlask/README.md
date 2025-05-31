# Curso de Flask

## Creación de una Aplicación de Notas con Flask y Tailwind

Crear una aplicación de notas sencilla usando **Flask** para el backend y **Tailwind CSS** para el diseño frontend es una excelente forma de aprender desarrollo web moderno. A continuación, te explico paso a paso cómo hacerlo.

### 🗒️ Objetivo

Crear una app donde puedas:

* Ver una lista de notas
* Agregar nuevas notas
* Eliminar notas
* (Opcionalmente) Editar notas

### 🧰 Tecnologías

* **Flask** (backend en Python)
* **SQLite** (base de datos ligera)
* **Tailwind CSS** (diseño moderno y responsivo)
* **Jinja2** (templating de Flask)

### 🛠️ Paso a Paso

### 1. 📁 Estructura del Proyecto

```bash
notas_app/
├── app.py
├── templates/
│   ├── base.html
│   ├── index.html
│   └── new_note.html
├── static/
│   └── (usaremos CDN para Tailwind)
└── notes.db (se genera al correr la app)
```

### 2. 📦 Requisitos

Crea un entorno virtual y instala Flask:

```bash
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install Flask
```

### 3. ✍️ Código `app.py`

```python
from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    notes = conn.execute('SELECT * FROM notes').fetchall()
    conn.close()
    return render_template('index.html', notes=notes)

@app.route('/new', methods=('GET', 'POST'))
def new_note():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        conn = get_db_connection()
        conn.execute('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))
    return render_template('new_note.html')

@app.route('/delete/<int:id>')
def delete_note(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM notes WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Crear tabla si no existe
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL
        )
    ''')
    conn.close()
    app.run(debug=True)
```

### 4. 🖼️ Templates HTML con Tailwind

#### `templates/base.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Notas</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans">
    <div class="container mx-auto p-4">
        <h1 class="text-2xl font-bold mb-4">📝 Mis Notas</h1>
        {% block content %}{% endblock %}
    </div>
</body>
</html>
```

#### `templates/index.html`

```html
{% extends 'base.html' %}
{% block content %}
<a href="{{ url_for('new_note') }}" class="bg-blue-500 text-white px-4 py-2 rounded mb-4 inline-block">➕ Nueva Nota</a>
<div class="grid gap-4 grid-cols-1 md:grid-cols-2">
    {% for note in notes %}
    <div class="bg-white p-4 shadow rounded">
        <h2 class="font-bold text-lg">{{ note.title }}</h2>
        <p>{{ note.content }}</p>
        <a href="{{ url_for('delete_note', id=note.id) }}" class="text-red-500 mt-2 inline-block">🗑️ Eliminar</a>
    </div>
    {% endfor %}
</div>
{% endblock %}
```

#### `templates/new_note.html`

```html
{% extends 'base.html' %}
{% block content %}
<form method="post" class="bg-white p-4 shadow rounded max-w-md">
    <label class="block mb-2 font-semibold">Título</label>
    <input type="text" name="title" required class="border p-2 w-full mb-4">
    
    <label class="block mb-2 font-semibold">Contenido</label>
    <textarea name="content" required class="border p-2 w-full mb-4"></textarea>
    
    <button type="submit" class="bg-green-500 text-white px-4 py-2 rounded">Guardar</button>
</form>
{% endblock %}
```

### ✅ Resultado

Una app web minimalista donde puedes:

* Ver todas tus notas
* Crear nuevas
* Eliminar las que no necesites

### 🚀 Mejoras posibles

* Editar notas
* Autenticación de usuario
* Guardar en una base de datos remota (como PostgreSQL o RDS)
* Desplegar en servicios como **Render**, **Heroku**, o **AWS**

### Resumen

Flask es un micro framework de Python que ofrece una base sólida para construir aplicaciones web con flexibilidad y control. Su diseño minimalista permite agregar extensiones según las necesidades específicas del proyecto, convirtiéndolo en una herramienta poderosa para desarrolladores que buscan personalización y eficiencia en sus aplicaciones.

#### ¿Qué es Flask y por qué utilizarlo?

Imagina que construir una aplicación web es como construir una casa. Tienes todos los elementos y, al unirlos, vas construyendo muros, techos y suelos. Eso es Flask en esencia: una base sólida sobre la cual puedes agregar componentes según tus necesidades específicas.

Flask es un micro framework de Python que te permite tener control total sobre tu aplicación web. A diferencia de otros frameworks más robustos, Flask no impone una estructura rígida, sino que te da la libertad de diseñar tu aplicación como mejor te parezca.

Las principales ventajas de Flask incluyen:

- **Configuración minimalista** que facilita comenzar proyectos rápidamente
- **Comunidad extensa** que ha desarrollado numerosas extensiones reutilizables
- **Curva de aprendizaje accesible** especialmente si ya conoces Python
- **Herramientas de desarrollo integradas** como un servidor web y una shell para ejecutar código Python en el contexto de la aplicación

####¿Qué empresas utilizan Flask en producción?

Flask no es solo para pequeños proyectos o desarrolladores independientes. Grandes empresas confían en este framework para sus aplicaciones:

- **Netflix** utiliza Flask para crear herramientas internas que apoyan sus procesos de desarrollo
- **Spotify** implementa características importantes de su aplicación de streaming musical con Flask

Estas empresas de clase mundial han elegido Flask por su flexibilidad, rendimiento y capacidad de adaptación a necesidades específicas.

#### ¿Cómo extender Flask según tus necesidades?

Una de las características más poderosas de Flask es su capacidad de extensión. Puedes agregar funcionalidades específicas según lo requiera tu proyecto:

- Extensiones para manejar sesiones de usuario
- Componentes para gestionar consultas e inserciones en bases de datos
- Herramientas para autenticación y autorización
- Módulos para procesamiento de formularios

Esta modularidad te permite mantener tu aplicación ligera, incluyendo solo lo que realmente necesitas, sin el peso de componentes innecesarios que podrían ralentizar tu aplicación.

#### ¿Qué aprenderás en este curso sobre Flask?

Durante este curso, se desarrollará una aplicación práctica de notas utilizando Flask. Esta aplicación permitirá:

- Crear nuevas notas
- Editar notas existentes
- Eliminar notas que ya no necesites

Además, se implementará Tailwind CSS para crear una interfaz de usuario atractiva y amigable. El resultado será una aplicación completamente funcional que podrás extender y modificar según tus propias necesidades.

Flask es la herramienta ideal para crear aplicaciones web de manera rápida sin sacrificar control o flexibilidad. Si buscas un framework que te permita construir aplicaciones a medida con una base sólida, Flask es definitivamente una excelente opción para considerar en tu próximo proyecto de desarrollo web.

**Lecturas recomendadas**
[Welcome to Flask — Flask Documentation (3.1.x)](https://flask.palletsprojects.com/en/stable/)

## Creación de una Aplicación de Notas con Flask Paso a Paso

¡Perfecto! A continuación, te guiaré **paso a paso** para crear una **Aplicación de Notas con Flask**, usando **HTML + Tailwind CSS** para el frontend y **SQLite** como base de datos.

### 🛠 PASO A PASO: Aplicación de Notas con Flask

### ✅ 1. Crear la estructura del proyecto

```bash
mkdir flask-notas
cd flask-notas
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install Flask
```

### ✅ 2. Estructura de carpetas

```plaintext
flask-notas/
│
├── app.py
├── notes.db  (se crea automáticamente)
├── templates/
│   ├── base.html
│   ├── index.html
│   └── new_note.html
└── static/   (opcional si usarás Tailwind desde CDN)
```

### ✅ 3. Código base en `app.py`

```python
from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    notes = conn.execute('SELECT * FROM notes').fetchall()
    conn.close()
    return render_template('index.html', notes=notes)

@app.route('/new', methods=('GET', 'POST'))
def new_note():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        conn = get_db_connection()
        conn.execute('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))
    return render_template('new_note.html')

@app.route('/delete/<int:id>')
def delete_note(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM notes WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL
        )
    ''')
    conn.close()
    app.run(debug=True)
```

### ✅ 4. Plantillas HTML con Tailwind

#### `templates/base.html`

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Notas</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 text-gray-900">
    <div class="container mx-auto px-4 py-6">
        <h1 class="text-3xl font-bold mb-6">📝 Mis Notas</h1>
        {% block content %}{% endblock %}
    </div>
</body>
</html>
```

#### `templates/index.html`

```html
{% extends 'base.html' %}

{% block content %}
<a href="{{ url_for('new_note') }}" class="bg-blue-600 hover:bg-blue-800 text-white px-4 py-2 rounded mb-4 inline-block">➕ Nueva Nota</a>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    {% for note in notes %}
    <div class="bg-white p-4 rounded shadow">
        <h2 class="text-xl font-bold">{{ note.title }}</h2>
        <p class="mt-2">{{ note.content }}</p>
        <a href="{{ url_for('delete_note', id=note.id) }}" class="text-red-500 hover:underline mt-3 inline-block">🗑 Eliminar</a>
    </div>
    {% else %}
    <p>No hay notas aún. ¡Agrega una!</p>
    {% endfor %}
</div>
{% endblock %}
```

#### `templates/new_note.html`

```html
{% extends 'base.html' %}

{% block content %}
<form method="post" class="bg-white p-6 rounded shadow max-w-md mx-auto">
    <label class="block text-sm font-medium">Título</label>
    <input name="title" required class="border p-2 w-full mb-4" placeholder="Título de la nota">

    <label class="block text-sm font-medium">Contenido</label>
    <textarea name="content" required class="border p-2 w-full mb-4" placeholder="Escribe aquí..."></textarea>

    <button type="submit" class="bg-green-600 hover:bg-green-800 text-white px-4 py-2 rounded">Guardar</button>
</form>
{% endblock %}
```

### ✅ 5. Ejecutar la app

```bash
python app.py
```

Abre en tu navegador: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 🎉 ¡Listo!

Ya tienes una aplicación funcional de notas con:

* Flask como backend.
* Tailwind CSS como diseño responsivo.
* SQLite como base de datos embebida.

### ¿Qué más puedes hacer?

* Agregar funcionalidad de edición.
* Agregar validación de formularios.
* Autenticación de usuarios.
* Desplegarla en **Render**, **Vercel (usando Flask + Serverless)** o **AWS Elastic Beanstalk**.

### Resumen

La creación de aplicaciones web con Flask es una habilidad fundamental para desarrolladores Python que buscan construir soluciones web ligeras y eficientes. Este micro-framework ofrece la flexibilidad necesaria para desarrollar desde simples APIs hasta complejas aplicaciones web, manteniendo un enfoque minimalista que facilita el aprendizaje. A continuación, exploraremos cómo configurar un entorno de desarrollo Flask y crear nuestra primera aplicación de notas.

#### ¿Cómo configurar un entorno de desarrollo para Flask?

Antes de comenzar a programar con Flask, es fundamental establecer un entorno de desarrollo adecuado. El uso de entornos virtuales es una práctica recomendada que nos permite aislar las dependencias de cada proyecto y evitar conflictos entre versiones de paquetes.

Para crear nuestro entorno de trabajo, seguiremos estos pasos:

1.  Crear una carpeta para nuestro proyecto:

```bash
mkdir notes-app
cd notes-app
```

2. Generar un entorno virtual dentro de esta carpeta:

`python -m venv venv`

3. Activar el entorno virtual:

- En sistemas Unix/Linux/MacOS:

`source venv/bin/activate`

- En sistemas Windows (el comando específico estará disponible en los recursos adicionales)

4. Instalar Flask usando pip:

`pip install Flask`

5. Verificar la instalación:

`flask --help`

Una vez completados estos pasos, tendremos un entorno aislado con Flask instalado y listo para usar. **Esta configuración nos permite mantener las dependencias organizadas** y facilita la portabilidad del proyecto entre diferentes sistemas.

#### ¿Cómo abrir el proyecto en Visual Studio Code?

Para trabajar cómodamente con nuestro código, podemos abrir la carpeta del proyecto en Visual Studio Code directamente desde la terminal:

`code -r .`

Este comando abrirá VS Code con la carpeta actual como raíz del proyecto, permitiéndonos crear y editar archivos fácilmente.

#### ¿Cómo crear nuestra primera aplicación Flask?

Una vez configurado nuestro entorno, podemos comenzar a escribir el código para nuestra aplicación. **Flask se basa en un sistema de rutas y vistas** que nos permite definir qué contenido se mostrará en cada URL de nuestra aplicación.

Crearemos un archivo llamado `app.py` con el siguiente contenido:

```python
from Flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hola Mundo"

if __name__ == '__main__':
    app.run(debug=True)
```

Este código realiza varias acciones importantes:

1. Importa la clase Flask del paquete principal
2. Crea una instancia de la aplicación
3. Define una ruta para la URL raíz ('/')
4. Asocia una función que retorna el texto "Hola Mundo"
5. Configura la aplicación para ejecutarse en modo debug cuando se ejecuta el archivo directamente

#### ¿Cómo ejecutar nuestra aplicación Flask?

Existen dos formas principales de ejecutar una aplicación Flask:

1. **Usando Python directamente:**

`python app.py`

2. **Usando el comando Flask (recomendado):**

`flask run`

La segunda opción es preferible porque:

- Elimina la necesidad de incluir el bloque `if __name__ == '__main__'` en nuestro código
- Proporciona opciones adicionales a través de flags
- Es la forma estándar recomendada por Flask

Para activar el modo de depuración con el comando Flask, usamos:

`flask run --debug`

#### El modo debug es extremadamente útil durante el desarrollo ya que:

- Recarga automáticamente la aplicación cuando detecta cambios en el código
- Proporciona mensajes de error detallados
- Incluye una consola interactiva para depuración

Sin embargo, es importante recordar que nunca debe usarse en producción por razones de seguridad y rendimiento.

#### ¿Qué opciones adicionales ofrece el comando Flask?

Flask proporciona varias opciones para personalizar la ejecución de nuestra aplicación. Podemos explorarlas ejecutando:

`flask run --help`

Entre las opciones disponibles encontramos:

- Cambiar el host y puerto de escucha
- Habilitar o deshabilitar el modo de depuración
- Especificar archivos adicionales para vigilar cambios
- Configurar opciones de threading

#### ¿Cómo crear rutas adicionales en nuestra aplicación?

Una aplicación web típicamente necesita múltiples páginas o endpoints. En Flask, podemos crear tantas rutas como necesitemos usando el decorador `@app.route()`.

Por ejemplo, para agregar una página "Acerca de" a nuestra aplicación de notas, podríamos añadir:

```python
@app.route('/about')
def about():
    return "Esta es una aplicación para tomar y organizar notas personales. Podrás crear, editar y eliminar notas fácilmente."
```

**Cada ruta se asocia con una función específica** que determina qué contenido se mostrará cuando un usuario visite esa URL. Estas funciones pueden retornar texto simple, HTML, JSON u otros tipos de contenido según las necesidades de la aplicación.

La estructura de rutas es fundamental para organizar la navegación de nuestra aplicación y proporcionar una experiencia de usuario coherente.

Flask es un micro-framework potente y flexible que nos permite crear aplicaciones web de forma rápida y sencilla. Hemos aprendido a configurar un entorno de desarrollo, crear una aplicación básica y añadir rutas para diferentes páginas. Estos conceptos fundamentales son la base para construir aplicaciones más complejas en el futuro. ¿Qué otras funcionalidades te gustaría implementar en tu aplicación de notas? Comparte tus ideas en los comentarios.

**Lecturas recomendadas**

[Welcome to Flask — Flask Documentation (3.1.x)](https://flask.palletsprojects.com/en/stable/)

[venv — Creation of virtual environments — Python 3.13.2 documentation](https://docs.python.org/3/library/venv.html)

[GitHub - platzi/curso-flask](https://github.com/platzi/curso-flask/)

## Manejo de Decoradores y Métodos HTTP en Flask

El **manejo de decoradores y métodos HTTP en Flask** es esencial para crear rutas y controlar cómo responde tu aplicación a diferentes tipos de solicitudes. Aquí te dejo una guía paso a paso con ejemplos claros.

### 🧩 ¿Qué es un decorador en Flask?

En Flask, los decoradores como `@app.route` se usan para asociar funciones con URLs. Es decir, definen qué función debe ejecutarse cuando un cliente accede a una determinada ruta.

### ✅ Métodos HTTP comunes

| Método   | Descripción             |
| -------- | ----------------------- |
| `GET`    | Obtener datos (lectura) |
| `POST`   | Enviar datos (crear)    |
| `PUT`    | Actualizar datos        |
| `DELETE` | Eliminar datos          |

### 🧪 Ejemplo básico con varios métodos

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return "¡Hola desde Flask!"

@app.route('/saludo', methods=['GET', 'POST'])
def saludo():
    if request.method == 'POST':
        nombre = request.form.get('nombre', 'anónimo')
        return f"¡Hola, {nombre}!"
    return '''
        <form method="post">
            Nombre: <input type="text" name="nombre">
            <input type="submit">
        </form>
    '''

@app.route('/actualizar', methods=['PUT'])
def actualizar():
    datos = request.json
    return {"mensaje": "Datos actualizados", "datos": datos}, 200

@app.route('/eliminar', methods=['DELETE'])
def eliminar():
    return {"mensaje": "Elemento eliminado"}, 200

if __name__ == '__main__':
    app.run(debug=True)
```

### 🛠️ Explicación rápida

* `@app.route('/saludo', methods=['GET', 'POST'])`: responde a solicitudes `GET` y `POST`.
* `request.method`: verifica qué tipo de solicitud se recibió.
* `request.form`, `request.json`: acceden a datos del formulario o JSON respectivamente.
* Se usan condicionales para manejar el comportamiento según el método HTTP.

### Recursos

El decorador `@route` en Flask es una herramienta poderosa que permite definir cómo nuestras aplicaciones web responden a diferentes tipos de solicitudes HTTP. Dominar este decorador es fundamental para crear APIs robustas y aplicaciones web interactivas que puedan procesar diversos tipos de peticiones de los usuarios. En este artículo, exploraremos cómo extender nuestro uso del decorador `@route` para manejar diferentes métodos HTTP y retornar varios tipos de datos.

#### ¿Cómo utilizar el decorador `@route` con diferentes métodos HTTP?

El decorador `@route` en Flask no solo nos permite definir rutas para solicitudes GET, sino que también podemos configurarlo para manejar otros métodos HTTP como POST, PUT o DELETE. Esto es esencial para crear aplicaciones web completas que puedan recibir y procesar diferentes tipos de interacciones del usuario.

Para especificar qué métodos HTTP puede manejar una ruta particular, utilizamos el parámetro `methods`:

```python
@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        return "Formulario enviado correctamente", 201
    return "Página de contacto"
```

En este ejemplo, hemos creado una vista que puede responder tanto a solicitudes GET como POST. **Es importante notar que si no especificamos el parámetro `methods`, Flask asumirá por defecto que la ruta solo maneja solicitudes GET**.

#### ¿Cómo validar el tipo de método en una solicitud?

Para determinar qué tipo de método HTTP está utilizando una solicitud entrante, podemos usar el objeto `request` de Flask:

```python
from flask import request

@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        # Lógica para manejar solicitudes POST
        return "Formulario enviado correctamente", 201
    # Lógica para manejar solicitudes GET
    return "Página de contacto"
```

**El objeto `request` se importa directamente de Flask y se llena automáticamente con la información de la solicitud actual**. No necesitamos pasarlo como parámetro a nuestra función de vista.

#### ¿Cómo personalizar las respuestas HTTP en Flask?

Flask nos permite no solo retornar contenido, sino también especificar códigos de estado HTTP y otros metadatos en nuestras respuestas.

#### Retornando códigos de estado HTTP

Para retornar un código de estado específico junto con nuestra respuesta, simplemente lo incluimos como segundo elemento en una tupla:

```python
@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        # Retornamos código 201 (Created) para indicar que algo fue creado exitosamente
        return "Formulario enviado correctamente", 201
    return "Página de contacto"
```

**Los códigos de estado HTTP son importantes para seguir las buenas prácticas de desarrollo web:**

- 200: OK (éxito general)
- 201: Created (recurso creado exitosamente)
- 404: Not Found (recurso no encontrado)
- 500: Internal Server Error (error del servidor)

#### ¿Cómo retornar diferentes formatos de datos?

Flask facilita el retorno de diferentes formatos de datos, como JSON, que es especialmente útil para APIs web:

```python
from flask import jsonify

@app.route('/api/info')
def api_info():
    data = {
        "nombre": "notesApp",
        "version": "1.1.1"
    }
    return jsonify(data), 200
```

La función `jsonify()` convierte automáticamente diccionarios Python en respuestas JSON con los encabezados MIME adecuados. **Esto es fundamental cuando estamos desarrollando APIs que necesitan comunicarse con aplicaciones frontend o móviles**.

#### ¿Cómo personalizar las URLs de nuestras rutas?

Una característica interesante de Flask es que podemos definir URLs que sean diferentes del nombre de la función que maneja esa ruta:

```python
@app.route('/acerca-de')
def about():
    return "Esto es una app de notas"
```

En este ejemplo, la función se llama `about`, pero la URL que los usuarios visitarán es `/acerca-de`. **Esta flexibilidad nos permite crear URLs amigables y semánticamente significativas mientras mantenemos nombres de funciones claros en nuestro código**.

#### Probando solicitudes POST con curl

Para probar solicitudes POST sin necesidad de crear un formulario HTML, podemos utilizar herramientas como curl desde la línea de comandos:

`curl -X POST http://localhost:5000/contacto`

powershell

`Invoke-WebRequest -Uri http://127.0.0.1:5000/contacto -Method GET`

Este comando enviará una solicitud POST a nuestra ruta `/contacto` y nos mostrará la respuesta, incluyendo el código de estado HTTP.

**El uso de herramientas como curl es invaluable durante el desarrollo para probar rápidamente nuestros endpoints sin necesidad de crear interfaces de usuario completas.**

El decorador `@route` en Flask es una herramienta versátil que nos permite crear aplicaciones web robustas y APIs flexibles. Dominar su uso con diferentes métodos HTTP y tipos de respuesta es fundamental para cualquier desarrollador web que trabaje con Python. Te animo a experimentar con retornar HTML y a explorar otros métodos HTTP como PUT y PATCH para ampliar tus habilidades en el desarrollo web con Flask.

**Lecturas recomendadas**

[GitHub - platzi/curso-flask](https://github.com/platzi/curso-flask/)

## Uso de Jinja para Plantillas HTML Dinámicas en Flask

El **uso de Jinja en Flask** permite renderizar **plantillas HTML dinámicas**, es decir, páginas web que cambian en función de los datos enviados desde el backend. Flask usa **Jinja2** como su sistema de templates por defecto.

### 🚀 ¿Qué es Jinja?

**Jinja2** es un motor de plantillas para Python que permite:

* Insertar variables en HTML
* Usar estructuras de control (if, for)
* Heredar plantillas
* Reutilizar bloques comunes

### 📦 Estructura de proyecto mínima

```
mi_app/
│
├── app.py
├── templates/
│   ├── layout.html
│   └── index.html
```

### 📄 `app.py` (servidor Flask)

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    usuario = "Mario"
    tareas = ["Estudiar Flask", "Leer sobre Jinja", "Hacer una app"]
    return render_template("index.html", usuario=usuario, tareas=tareas)

if __name__ == '__main__':
    app.run(debug=True)
```

### 📄 `templates/layout.html` (plantilla base)

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>{% block titulo %}Mi App{% endblock %}</title>
</head>
<body>
    <header>
        <h1>Mi Aplicación</h1>
    </header>

    <main>
        {% block contenido %}{% endblock %}
    </main>
</body>
</html>
```

### 📄 `templates/index.html` (plantilla heredada)

```html
{% extends "layout.html" %}

{% block titulo %}Inicio{% endblock %}

{% block contenido %}
    <h2>Hola, {{ usuario }} 👋</h2>
    <ul>
        {% for tarea in tareas %}
            <li>{{ tarea }}</li>
        {% else %}
            <li>No hay tareas.</li>
        {% endfor %}
    </ul>
{% endblock %}
```

### 🧠 Conceptos Clave de Jinja

| Sintaxis         | Significado                           |
| ---------------- | ------------------------------------- |
| `{{ variable }}` | Muestra una variable de Python        |
| `{% ... %}`      | Instrucción de control (for, if, etc) |
| `{% extends %}`  | Hereda otra plantilla                 |
| `{% block %}`    | Define una sección reemplazable       |

### ✅ Ventajas

* Separación de lógica y presentación
* Reutilización de código HTML
* Renderizado dinámico basado en datos

### Resumen

La integración de Jinja en Flask revoluciona la forma en que creamos aplicaciones web dinámicas, permitiéndonos incorporar lógica de programación directamente en nuestros archivos HTML. Esta potente combinación nos brinda la flexibilidad necesaria para desarrollar interfaces de usuario interactivas y personalizadas sin sacrificar la estructura y semántica del HTML. Descubramos cómo Jinja transforma el desarrollo web con Flask y cómo podemos aprovechar sus capacidades para crear aplicaciones más robustas y dinámicas.

### ¿Qué es Jinja y por qué es importante en Flask?

Jinja es un motor de plantillas para Python que permite incorporar variables, condicionales, bucles y otras estructuras de programación directamente en archivos HTML. A diferencia del HTML estático, Jinja nos permite crear contenido dinámico que se genera en tiempo de ejecución.

### Características principales de Jinja:

- Uso de variables dentro del HTML
- Estructuras condicionales (if-else)
- Bucles (for)
- Herencia de plantillas
- Filtros para manipular datos

Flask integra Jinja de manera nativa, lo que facilita enormemente el desarrollo de aplicaciones web con contenido dinámico. Esta integración es fundamental porque separa la lógica de negocio (Python) de la presentación (HTML), manteniendo un código más limpio y mantenible.

#### ¿Cómo implementar plantillas Jinja en Flask?

Para comenzar a utilizar Jinja en nuestra aplicación Flask, necesitamos seguir algunos pasos básicos:

#### Creación de la estructura de carpetas

El primer paso es crear una carpeta llamada `templates` en la raíz de nuestro proyecto. Flask buscará automáticamente los archivos HTML en esta carpeta cuando utilicemos la función `render_template`.

```
# Estructura básica del proyecto
/proyecto
    /templates
        home.html
    app.py
```

#### Renderizando plantillas básicas

Para renderizar una plantilla, utilizamos la función `render_template` que debemos importar desde Flask:

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
```

Este código simplemente renderiza el archivo HTML sin ninguna variable dinámica. Para verificar que funciona correctamente, podemos acceder a la ruta principal de nuestra aplicación y ver el código fuente, que debería mostrar exactamente el contenido de nuestro archivo HTML.

#### Pasando variables a las plantillas

Una de las características más poderosas de Jinja es la capacidad de pasar variables desde Python a nuestras plantillas HTML:

```python
@app.route('/')
def home():
    rol = "admin"
    return render_template('home.html', rol=rol)
```

En el archivo HTML, podemos acceder a esta variable utilizando la sintaxis de dobles llaves:

`<p>Eres {{ rol }}</p>`

Cuando Flask renderice esta plantilla, reemplazará `{{ rol }}` con el valor de la variable, en este caso "admin".

#### ¿Cómo utilizar estructuras de control en Jinja?

Jinja nos permite utilizar estructuras de control como condicionales y bucles directamente en nuestro HTML, lo que hace que nuestras plantillas sean mucho más dinámicas.

#### Condicionales (if-else)

Los condicionales nos permiten mostrar u ocultar elementos HTML según ciertas condiciones:

```html
{% if rol == "admin" %}
    <button>Eliminar</button>
{% else %}
    <button>Solicitar eliminación</button>
{% endif %}
```
En este ejemplo, si el usuario tiene el rol de administrador, verá un botón de "Eliminar". De lo contrario, verá un botón de "Solicitar eliminación". **Esta funcionalidad es especialmente útil para controlar el acceso a ciertas características según el rol del usuario**.

#### Bucles (for)

Los bucles nos permiten iterar sobre colecciones de datos y generar HTML dinámicamente:

```python
@app.route('/')
def home():
    rol = "normal"
    notes = ["Nota uno", "Nota dos", "Nota tres"]
    return render_template('home.html', rol=rol, notes=notes)
```

En el HTML, podemos iterar sobre esta lista:

```html
<ul>
    {% for note in notes %}
        <li>{{ note }}</li>
    {% endfor %}
</ul>
```

Este código generará una lista HTML con un elemento `<li>` para cada nota en nuestra lista. En aplicaciones reales, estas notas podrían provenir de una base de datos, lo que hace que esta funcionalidad sea extremadamente útil.

**Mejorando la experiencia de desarrollo con Jinja**

Para facilitar el trabajo con Jinja, es recomendable instalar extensiones en nuestro editor de código. Por ejemplo, en Visual Studio Code, existe una extensión llamada "Jinja" que proporciona resaltado de sintaxis y autocompletado para el código Jinja.

**Beneficios de usar extensiones para Jinja:**

- Resaltado de sintaxis para distinguir fácilmente el código Jinja del HTML
- Autocompletado de estructuras como `{% if %}`, `{% for %}`, etc.
- Mejor legibilidad del código
- Detección de errores de sintaxis

Estas herramientas mejoran significativamente la productividad al trabajar con plantillas Jinja, especialmente en proyectos grandes con múltiples archivos HTML.

#### Trabajando con objetos y diccionarios

En lugar de pasar simples strings a nuestras plantillas, podemos pasar estructuras de datos más complejas como diccionarios u objetos:

```python
@app.route('/')
def home():
    notes = [
        {"title": "Nota uno", "description": "Descripción de la nota uno", "created_at": "2023-01-01"},
        {"title": "Nota dos", "description": "Descripción de la nota dos", "created_at": "2023-01-02"},
        {"title": "Nota tres", "description": "Descripción de la nota tres", "created_at": "2023-01-03"}
    ]
    return render_template('home.html', notes=notes)
```

En la plantilla, podemos acceder a las propiedades de cada objeto:

```html
<ul>
    {% for note in notes %}
        <li>
            <h3>{{ note.title }}</h3>
            <p>{{ note.description }}</p>
            <small>Creado el: {{ note.created_at }}</small>
        </li>
    {% endfor %}
</ul>
```

Esta capacidad nos permite crear interfaces mucho más ricas y detalladas, mostrando múltiples aspectos de nuestros datos.

Jinja es una herramienta poderosa que transforma la manera en que desarrollamos aplicaciones web con Flask. Al dominar sus funcionalidades básicas, podemos crear interfaces dinámicas y personalizadas que mejoran significativamente la experiencia del usuario. Te animo a explorar más características de Jinja en la documentación oficial y a experimentar con estructuras de datos más complejas en tus proyectos. ¿Qué otras funcionalidades de Jinja te gustaría implementar en tus aplicaciones Flask? Comparte tus ideas y experiencias en los comentarios.

**Lecturas recomendadas**

[Templates — Flask Documentation (3.1.x)](https://flask.palletsprojects.com/en/stable/tutorial/templates/)

[GitHub - platzi/curso-flask](https://github.com/platzi/curso-flask)

## Creación y Manejo de Formularios en Aplicaciones Web

La **creación y manejo de formularios** en aplicaciones web con **Flask** se realiza usando HTML en las plantillas y el módulo `request` para procesar los datos enviados desde el formulario.

Aquí tienes una **guía paso a paso**:

### 🧱 1. Crear una plantilla con el formulario (HTML + Jinja2)

Ubica este archivo en `templates/contacto.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Contacto</title>
</head>
<body>
    <h1>Formulario de Contacto</h1>
    <form action="/contacto" method="POST">
        <label for="nombre">Nombre:</label>
        <input type="text" id="nombre" name="nombre" required><br><br>

        <label for="mensaje">Mensaje:</label><br>
        <textarea id="mensaje" name="mensaje" rows="4" cols="50" required></textarea><br><br>

        <input type="submit" value="Enviar">
    </form>
</body>
</html>
```

### 🐍 2. Código Flask (`app.py`)

```python
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/contacto", methods=["GET", "POST"])
def contacto():
    if request.method == "POST":
        nombre = request.form["nombre"]
        mensaje = request.form["mensaje"]
        # Aquí podrías guardar en una base de datos, enviar un correo, etc.
        return f"Gracias, {nombre}. Tu mensaje ha sido recibido."
    return render_template("contacto.html")

if __name__ == "__main__":
    app.run(debug=True)
```

### 📦 3. Estructura del proyecto

```
notes_app/
├── app.py
└── templates/
    └── contacto.html
```

### 📌 ¿Cómo funciona?

* El navegador envía los datos al presionar "Enviar".
* Flask los recibe en `request.form`.
* Puedes hacer algo con los datos (guardar, imprimir, enviar correo, etc.).
* El servidor responde con una página de confirmación o redirección.

### Resumen

Los formularios son una parte esencial en el desarrollo web, ya que permiten la comunicación entre usuarios y servidores. Dominar el manejo de formularios en Flask te permitirá crear aplicaciones web interactivas y funcionales que respondan a las necesidades de tus usuarios. En este contenido, exploraremos cómo implementar formularios en Flask, procesar la información enviada y realizar redirecciones efectivas entre diferentes vistas.

#### ¿Cómo crear y procesar formularios en Flask?

Para comenzar a trabajar con formularios en Flask, necesitamos entender cómo se estructura un formulario HTML básico y cómo se conecta con nuestro backend. El primer paso es crear un archivo HTML que contenga nuestro formulario.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Crear Nota</title>
</head>
<body>
    <h1>Crear una nueva nota</h1>
    <form method="post">
        <label for="note">Nota:</label>
        <input type="text" name="note" id="note">
        <input type="submit" value="Crear nota">
    </form>
</body>
</html>
```

Este formulario tiene elementos clave que debemos considerar:

- El atributo `method="post"` indica que enviaremos datos al servidor
- El campo `name="note"` es **crucial** ya que es el identificador que usaremos en Python para acceder a este valor
- Evita usar nombres en español o con caracteres especiales en los atributos `name`, ya que pueden causar problemas al acceder a ellos desde Python

Una vez creado el formulario, necesitamos configurar una vista en Flask que pueda mostrar el formulario y procesar los datos enviados.

#### ¿Cómo configurar las rutas para manejar formularios?

En Flask, necesitamos configurar una ruta que pueda manejar tanto solicitudes GET (para mostrar el formulario) como POST (para procesar los datos enviados). Esto se logra especificando los métodos permitidos en la decoración de la ruta.

from flask import Flask, render_template, request, redirect, url_for

```python
app = Flask(__name__)

@app.route('/crear-nota', methods=['GET', 'POST'])
def CreateNote():
    if request.method == 'POST':
        # Procesar los datos del formulario
        note = request.form.get('note', 'No encontrada')
        print(note)  # Para verificar que estamos recibiendo los datos
        return redirect(url_for('confirmation', note=note))
    
    # Si es una solicitud GET, mostrar el formulario
    return render_template('note.html')
```

En este código:

- Importamos las funciones necesarias de Flask
- Configuramos la ruta `/crear-nota` para aceptar métodos GET y POST
- Verificamos el tipo de solicitud con `request.method`
- Accedemos a los datos del formulario mediante `request.form.get('note')`
- Redirigimos al usuario a una página de confirmación con los datos recibidos

**Es importante destacar** que `request.form` es un objeto que contiene todos los campos enviados desde el formulario. Podemos acceder a cada campo utilizando el nombre que le asignamos en el HTML.

#### ¿Cómo implementar redirecciones entre vistas?

Una práctica común después de procesar un formulario es redirigir al usuario a otra página. Esto evita problemas como el reenvío de formularios al actualizar la página y mejora la experiencia del usuario.

```python
@app.route('/confirmacion')
def confirmation():
    # Aquí deberíamos mostrar un template con la confirmación
    return "Prueba"
```

Para implementar la redirección, utilizamos dos funciones importantes:

1. `redirect()`: Redirige al usuario a otra URL
2. `url_for()`: Genera la URL para una función de vista específica

La ventaja de usar `url_for()` en lugar de escribir la URL directamente es que si cambiamos el nombre de la ruta en el futuro, no tendremos que actualizar todas las referencias a esa URL en nuestro código.

`return redirect(url_for('confirmation', note=note))`

En este ejemplo, estamos redirigiendo al usuario a la vista c`onfirmation` y pasando el valor de `note` como un parámetro en la URL. Esto permite que la vista de confirmación acceda a este valor y lo muestre al usuario.

#### ¿Cómo mostrar los datos recibidos en una plantilla HTML?
Para completar el flujo de trabajo con formularios, necesitamos mostrar los datos recibidos en una plantilla HTML. Esto se logra pasando los datos a la función `render_template()`.

```python
@app.route('/confirmacion')
def confirmation():
    note = request.args.get('note', 'No se encontró ninguna nota')
    return render_template('confirmation.html', note=note)
```

En este código, estamos:

1. Obteniendo el valor de `note` desde los parámetros de la URL con `request.args.get()`
2. Pasando ese valor a la plantilla `confirmation.html`

Luego, en nuestra plantilla HTML, podemos mostrar el valor recibido:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Confirmación</title>
</head>
<body>
    <h1>Nota creada con éxito</h1>
    <p>Tu nota: {{ note }}</p>
</body>
</html>
```

El uso de `{{ note }}` en la plantilla permite insertar el valor de la variable `note` que pasamos desde nuestra vista.

El manejo de formularios en Flask es una habilidad fundamental para cualquier desarrollador web. Con estos conocimientos básicos, puedes comenzar a crear aplicaciones interactivas que reciban y procesen datos de los usuarios. ¿Te animas a implementar tu propio sistema de formularios? Comparte tus experiencias y dudas en los comentarios.

## Integración de SQLAlchemy en Flask para Bases de Datos

La **integración de SQLAlchemy en Flask** permite trabajar con bases de datos de forma sencilla y poderosa mediante un ORM (Object-Relational Mapper). Aquí tienes una guía paso a paso para integrarlo correctamente:

### ✅ 1. Instalar dependencias

Ejecuta esto en tu entorno virtual:

```bash
pip install Flask SQLAlchemy
```

### 📁 2. Estructura del proyecto

```
notes_app/
├── app.py
├── models.py
└── templates/
    └── ...
```

### 🐍 3. Configuración en `app.py`

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configuración de la base de datos (SQLite en este caso)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notas.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializar SQLAlchemy
db = SQLAlchemy(app)
```

### 🧱 4. Crear modelos en `models.py`

```python
from app import db

class Nota(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titulo = db.Column(db.String(100), nullable=False)
    contenido = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Nota {self.titulo}>'
```

### 🔧 5. Crear la base de datos

En una terminal con el entorno virtual activado, abre el intérprete de Python y ejecuta:

```python
from app import db
db.create_all()
```

Esto generará el archivo `notas.db`.

### 🧪 6. Usar el modelo en una ruta

```python
from flask import render_template, request, redirect
from models import Nota

@app.route("/nueva-nota", methods=["GET", "POST"])
def nueva_nota():
    if request.method == "POST":
        titulo = request.form["titulo"]
        contenido = request.form["contenido"]
        nota = Nota(titulo=titulo, contenido=contenido)
        db.session.add(nota)
        db.session.commit()
        return redirect("/")
    return render_template("nueva_nota.html")
```

### 📦 7. Plantilla para crear notas: `templates/nueva_nota.html`

```html
<form method="POST">
    <label for="titulo">Título:</label>
    <input type="text" name="titulo" required><br><br>

    <label for="contenido">Contenido:</label><br>
    <textarea name="contenido" rows="5" cols="40" required></textarea><br><br>

    <input type="submit" value="Guardar">
</form>
```

### 🧠 Conclusión

Con SQLAlchemy en Flask puedes:

* Crear modelos que representan tus tablas.
* Usar métodos ORM para insertar, actualizar, eliminar y consultar datos.
* Trabajar con múltiples motores de base de datos como SQLite, PostgreSQL, MySQL, etc.

### Resumen

La integración de bases de datos en aplicaciones Flask representa un paso fundamental para desarrollar soluciones web robustas y escalables. SQL Alchemy se posiciona como una herramienta poderosa que permite a los desarrolladores Python interactuar con bases de datos relacionales sin necesidad de dominar completamente el lenguaje SQL, gracias a su implementación de ORM (Object-Relational Mapping). Este enfoque no solo simplifica el desarrollo, sino que también mejora la mantenibilidad del código al trabajar con objetos Python en lugar de consultas SQL directas.

#### ¿Cómo integrar una base de datos SQLite en una aplicación Flask?

Para integrar una base de datos en nuestra aplicación Flask, utilizaremos Flask-SQLAlchemy, una extensión que facilita el uso de SQLAlchemy con Flask. Esta librería nos permite relacionar tablas de la base de datos con modelos o clases de Python, implementando el patrón ORM (Object-Relational Mapping).

El primer paso es instalar la librería necesaria. Abrimos la terminal, activamos nuestro entorno virtual y ejecutamos:

`pip install flask-sqlalchemy`

Una vez instalada la librería, es recomendable crear un archivo `requirements.txt` para documentar las dependencias del proyecto:

```
Flask==2.x.x
Flask-SQLAlchemy==3.x.x
```

Para verificar que la instalación fue exitosa, podemos utilizar el comando `flask shell` que nos proporciona una consola interactiva con nuestra aplicación cargada:

`from flask_sqlalchemy import SQLAlchemy`

Si no aparece ningún error, significa que la librería está correctamente instalada.

#### Configuración de la base de datos en Flask

Para configurar nuestra base de datos SQLite, necesitamos modificar nuestro archivo principal de la aplicación `(app.py)`:

```python
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configuración de la ruta del archivo de base de datos
db_filepath = os.path.join(os.path.dirname(__file__), 'notes.sqlite')

# Configuración de SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_filepath}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Instancia de SQLAlchemy
db = SQLAlchemy(app)
```

En este código:

- Importamos la librería `os` para manejar rutas de archivos
- Definimos la ruta donde se creará el archivo SQLite
- Configuramos la URI de la base de datos con el formato requerido por SQLAlchemy
- Desactivamos el seguimiento de modificaciones para reducir la verbosidad de los logs
- Creamos una instancia de SQLAlchemy vinculada a nuestra aplicación

**Es importante destacar que SQLAlchemy es compatible con múltiples motores de bases de datos** como MySQL o PostgreSQL, no solo con SQLite.

#### ¿Cómo crear modelos y tablas con SQLAlchemy?

Los modelos en SQLAlchemy son clases de Python que representan tablas en la base de datos. Cada atributo de la clase corresponde a una columna en la tabla.

Para nuestro ejemplo, crearemos un modelo `Note` para almacenar notas con título y contenido:

```python
class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.String(200), nullable=False)
    
    def __repr__(self):
        return f'Note {self.id}: {self.title}'
```

En este modelo:

- `id`: Es un entero que actúa como clave primaria
- `title`: Es una cadena de texto con longitud máxima de 100 caracteres y no puede ser nula
- `content`: Es una cadena de texto con longitud máxima de 200 caracteres y tampoco puede ser nula
- El método `__repr__` define cómo se mostrará el objeto cuando se imprima

#### Creación de las tablas en la base de datos

Una vez definido el modelo, necesitamos crear las tablas correspondientes en la base de datos. Para esto, utilizamos el método `create_all()` de SQLAlchemy:

`flask shell`

Y dentro de la consola interactiva:

```python
from app import db
db.create_all()
```

Este comando creará el archivo `notes.sqlite` con la tabla `note` según la estructura definida en nuestro modelo.

#### Verificación de la estructura de la base de datos

Para verificar que la tabla se ha creado correctamente, podemos utilizar la herramienta de línea de comandos de SQLite:

`sqlite3 notes.sqlite`

Y dentro de la consola de SQLite:

`.schema`

Este comando nos mostrará la estructura de la tabla note con sus columnas `id`, `title` y `content`, confirmando que se ha creado correctamente según nuestro modelo.

#### ¿Qué ventajas ofrece el uso de ORM en aplicaciones Flask?

El uso de ORM (Object-Relational Mapping) como SQLAlchemy en aplicaciones Flask ofrece numerosas ventajas:

- **Abstracción de la base de datos**: Permite trabajar con objetos Python en lugar de consultas SQL directas
- **Portabilidad**: Facilita el cambio entre diferentes motores de bases de datos sin modificar el código
- **Seguridad**: Ayuda a prevenir ataques de inyección SQL al manejar automáticamente el escapado de caracteres
- **Productividad**: Reduce la cantidad de código necesario para interactuar con la base de datos
- **Mantenibilidad**: El código es más legible y fácil de mantener al trabajar con objetos y métodos

**El uso de modelos en SQLAlchemy también facilita la evolución del esquema** de la base de datos a medida que la aplicación crece, permitiendo agregar nuevos campos o relaciones de manera sencilla.

La integración de bases de datos en aplicaciones Flask mediante SQLAlchemy representa un paso fundamental en el desarrollo de aplicaciones web robustas. Esta aproximación nos permite centrarnos en la lógica de negocio mientras el ORM se encarga de la comunicación con la base de datos, resultando en un código más limpio, mantenible y seguro. ¿Has utilizado ORM en tus proyectos? Comparte tu experiencia en los comentarios.

**Lecturas recomendadas**

[SQLAlchemy - The Database Toolkit for Python](https://www.sqlalchemy.org/)

[Flask-SQLAlchemy — Flask-SQLAlchemy Documentation (3.1.x)](https://flask-sqlalchemy.readthedocs.io/en/stable/)

## Creación y Gestión de Notas con SQLAlchemy y Vistas en Python

Aquí tienes una **guía paso a paso para la creación y gestión de notas con SQLAlchemy y vistas en Flask**. Este ejemplo te permitirá:

✅ Crear, ver, editar y eliminar notas desde vistas web.

### 🧱 1. Estructura del proyecto

```
notes_app/
├── app.py
├── models.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── nueva_nota.html
│   └── editar_nota.html
└── notas.db  ← generado automáticamente
```

### 🛠️ 2. Instala Flask y SQLAlchemy

```bash
pip install Flask SQLAlchemy
```

### 📦 3. `app.py` – Configuración general y rutas

```python
from flask import Flask, render_template, request, redirect, url_for
from models import db, Nota

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notas.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.before_first_request
def crear_tablas():
    db.create_all()

@app.route('/')
def index():
    notas = Nota.query.all()
    return render_template('index.html', notas=notas)

@app.route('/nueva', methods=['GET', 'POST'])
def nueva_nota():
    if request.method == 'POST':
        nueva = Nota(titulo=request.form['titulo'], contenido=request.form['contenido'])
        db.session.add(nueva)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('nueva_nota.html')

@app.route('/editar/<int:id>', methods=['GET', 'POST'])
def editar_nota(id):
    nota = Nota.query.get_or_404(id)
    if request.method == 'POST':
        nota.titulo = request.form['titulo']
        nota.contenido = request.form['contenido']
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('editar_nota.html', nota=nota)

@app.route('/eliminar/<int:id>')
def eliminar_nota(id):
    nota = Nota.query.get_or_404(id)
    db.session.delete(nota)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
```

### 📄 4. `models.py` – Definición del modelo

```python
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Nota(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titulo = db.Column(db.String(100), nullable=False)
    contenido = db.Column(db.Text, nullable=False)
```

### 🖼️ 5. Plantillas HTML con Jinja2

### `templates/base.html`

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Notas</title>
</head>
<body>
    <h1>Notas</h1>
    {% block contenido %}{% endblock %}
</body>
</html>
```

### `templates/index.html`

```html
{% extends 'base.html' %}

{% block contenido %}
<a href="{{ url_for('nueva_nota') }}">Nueva Nota</a>
<ul>
    {% for nota in notas %}
    <li>
        <strong>{{ nota.titulo }}</strong><br>
        {{ nota.contenido }}<br>
        <a href="{{ url_for('editar_nota', id=nota.id) }}">Editar</a>
        <a href="{{ url_for('eliminar_nota', id=nota.id) }}">Eliminar</a>
    </li>
    {% endfor %}
</ul>
{% endblock %}
```

### `templates/nueva_nota.html`

```html
{% extends 'base.html' %}

{% block contenido %}
<h2>Nueva Nota</h2>
<form method="POST">
    <input name="titulo" placeholder="Título" required><br>
    <textarea name="contenido" placeholder="Contenido" required></textarea><br>
    <input type="submit" value="Guardar">
</form>
{% endblock %}
```

### `templates/editar_nota.html`

```html
{% extends 'base.html' %}

{% block contenido %}
<h2>Editar Nota</h2>
<form method="POST">
    <input name="titulo" value="{{ nota.titulo }}" required><br>
    <textarea name="contenido" required>{{ nota.contenido }}</textarea><br>
    <input type="submit" value="Actualizar">
</form>
{% endblock %}
```

### ✅ Resultado

* `GET /`: muestra todas las notas.
* `GET /nueva`: formulario para crear una nueva nota.
* `POST /nueva`: guarda la nueva nota.
* `GET /editar/<id>`: formulario para editar.
* `POST /editar/<id>`: guarda los cambios.
* `GET /eliminar/<id>`: elimina la nota.

### Resumen

La gestión de datos en aplicaciones web es un componente fundamental para crear experiencias interactivas y funcionales. En este artículo, exploraremos cómo implementar operaciones CRUD (Crear, Leer, Actualizar, Eliminar) en una aplicación Flask utilizando SQLAlchemy, centrándonos específicamente en la creación y visualización de notas. **Dominar estas técnicas te permitirá desarrollar aplicaciones web robustas con persistencia de datos**, una habilidad esencial para cualquier desarrollador web moderno.

#### ¿Cómo implementar la funcionalidad de notas en nuestra aplicación?

Ahora que nuestra tabla "note" tiene una estructura definida y permite la creación de nuevos registros, es momento de utilizar nuestro modelo para interactuar con la base de datos. El modelo de nota incluye varios métodos que nos permiten crear, actualizar y listar todas las notas almacenadas.

Para comenzar, necesitamos modificar nuestra vista principal (home) para mostrar las notas desde la base de datos en lugar de usar datos estáticos. Anteriormente teníamos un reto pendiente: convertir las notas de simples strings a objetos con propiedades.

```python
# Antes
@app.route('/')
def home():
    notes = ["Nota 1", "Nota 2", "Nota 3"]
    role = "admin"
    return render_template('index.html', notes=notes, role=role)

# Después
@app.route('/')
def home():
    notes = [
        {"title": "título de prueba", "content": "content de prueba"}
    ]
    return render_template('index.html', notes=notes)
```

También necesitamos actualizar nuestro archivo HTML para que sea compatible con la nueva estructura de objetos:

```python
{% for note in notes %}
    <li>
        {{ note.title }}
        <br>
        {{ note.content }}
    </li>
{% else %}
    <p>Aún no se han creado notas.</p>
{% endfor %}

<a href="{{ url_for('create_note') }}">Agregar una nueva nota</a>
```

**La implementación del bloque `else` dentro del bucle `for` es una característica poderosa de Jinja2** que nos permite mostrar un mensaje alternativo cuando la lista está vacía, mejorando así la experiencia del usuario.

#### ¿Cómo crear el formulario para añadir nuevas notas?

Para permitir a los usuarios crear nuevas notas, necesitamos un formulario adecuado. Vamos a modificar nuestro formulario existente para incluir campos tanto para el título como para el contenido:

```html
<form method="POST">
    <label for="title">Título</label>
    <input type="text" name="title" id="title">
    <br>
    <label for="content">Contenido</label>
    <input type="text" name="content" id="content">
    <br>
    <button type="submit">Crear nota</button>
</form>
```

Este formulario enviará los datos mediante el método POST a nuestra ruta de creación de notas.

#### ¿Cómo guardar las notas en la base de datos?

La parte más importante es la lógica para guardar las notas en la base de datos. Necesitamos modificar nuestra función `create_note` para capturar los datos del formulario y almacenarlos:

```python
@app.route('/create', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        
        note_db = Note(title=title, content=content)
        db.session.add(note_db)
        db.session.commit()
        
        return redirect(url_for('home'))
    
    return render_template('create.html')
```

#### Este código realiza varias operaciones clave:

1. Obtiene los datos del formulario (título y contenido)
2. Crea una nueva instancia del modelo Note con esos datos
3. Añade la nota a la sesión de la base de datos
4. Confirma la transacción con `commit()`
5. Redirige al usuario a la página principal

Es importante destacar el uso de `db.session.commit()`, que confirma los cambios en la base de datos. **Podríamos realizar múltiples operaciones (añadir varias notas) antes de hacer commit, lo que optimizaría el rendimiento en operaciones por lotes**.

#### ¿Cómo mostrar las notas desde la base de datos?

Para mostrar las notas almacenadas en la base de datos, necesitamos modificar nuestra función `home` para consultar la base de datos:

```python
@app.route('/')
def home():
    notes = Note.query.all()  # Obtiene todas las notas de la base de datos
    return render_template('index.html', notes=notes)
```

Con esta modificación, ahora estamos obteniendo todas las notas directamente desde la base de datos y pasándolas a nuestra plantilla. **SQLAlchemy se encarga de convertir los registros de la base de datos en objetos Python**, lo que facilita enormemente el trabajo con ellos en nuestras plantillas.

#### ¿Cómo mejorar la funcionalidad de nuestra aplicación?

Una mejora importante sería añadir el ID de cada nota en la lista para poder identificarlas fácilmente. Esto será especialmente útil cuando implementemos la funcionalidad de edición:

```html
{% for note in notes %}
    <li>
        ID: {{ note.id }} - {{ note.title }}
        <br>
        {{ note.content }}
    </li>
{% else %}
    <p>Aún no se han creado notas.</p>
{% endfor %}
```

**Mostrar el ID es el primer paso para implementar operaciones más avanzadas como la edición y eliminación de notas**, ya que necesitaremos identificar de manera única cada nota para realizar estas acciones.

La implementación de operaciones CRUD en aplicaciones web con Flask y SQLAlchemy es un proceso estructurado que permite crear aplicaciones robustas con persistencia de datos. Siguiendo estos pasos, has aprendido a crear y mostrar notas en una aplicación web, sentando las bases para funcionalidades más avanzadas como la edición y eliminación. **¿Qué otras funcionalidades te gustaría implementar en tu aplicación de notas?** Comparte tus ideas en los comentarios y continúa expandiendo tus conocimientos en desarrollo web.

## Edición de Contenidos en Bases de Datos con Formularios

Para editar contenidos en una base de datos usando formularios en Flask con SQLAlchemy, debes seguir estos pasos básicos:

### ✅ 1. Definir tu modelo (por ejemplo, `Note`)

Ya lo tienes:

```python
class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.String(200), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
```

### ✅ 2. Crear la ruta para editar

```python
from flask import render_template, request, redirect, url_for
from your_app import app, db
from your_app.models import Note

@app.route("/nota/editar/<int:id>", methods=["GET", "POST"])
def editar_nota(id):
    nota = Note.query.get_or_404(id)

    if request.method == "POST":
        nota.title = request.form["title"]
        nota.content = request.form["content"]
        db.session.commit()
        return redirect(url_for("listar_notas"))  # Ajusta según tu vista principal

    return render_template("editar_nota.html", nota=nota)
```

### ✅ 3. Crear el formulario HTML (`editar_nota.html`)

Ubícalo en el directorio `templates/`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Editar Nota</title>
</head>
<body>
    <h1>Editar Nota</h1>
    <form method="POST">
        <label for="title">Título:</label><br>
        <input type="text" name="title" id="title" value="{{ nota.title }}"><br><br>

        <label for="content">Contenido:</label><br>
        <textarea name="content" id="content">{{ nota.content }}</textarea><br><br>

        <button type="submit">Guardar Cambios</button>
    </form>
</body>
</html>
```

### ✅ 4. Asegúrate de tener una vista para mostrar o listar las notas

Ejemplo simple para mostrar todas las notas:

```python
@app.route("/notas")
def listar_notas():
    notas = Note.query.order_by(Note.date.desc()).all()
    return render_template("lista_notas.html", notas=notas)
```

### ✅ 5. Enlace de edición en la lista de notas

En `lista_notas.html`:

```html
{% for nota in notas %}
    <h2>{{ nota.title }}</h2>
    <p>{{ nota.content }}</p>
    <a href="{{ url_for('editar_nota', id=nota.id) }}">Editar</a>
{% endfor %}
```

### Resumen

La edición de contenidos en bases de datos es una funcionalidad esencial en cualquier aplicación web moderna. Aprender a implementar formularios de edición en Flask nos permite crear aplicaciones más completas y funcionales, donde los usuarios pueden modificar la información previamente almacenada. Este proceso, aunque parece complejo, se simplifica enormemente cuando entendemos los conceptos fundamentales de rutas dinámicas, manipulación de modelos y redirecciones.

#### ¿Cómo crear un formulario para editar contenido en Flask?

Para implementar la funcionalidad de edición en nuestra aplicación Flask, necesitamos crear un nuevo formulario que nos permita modificar una nota existente en la base de datos. Este proceso implica varios pasos importantes:

1. Crear una nueva ruta que acepte el ID de la nota como parámetro.
2. Recuperar la información existente de la base de datos.
3. Mostrar esa información en un formulario para su edición.
4. Procesar los cambios y actualizarlos en la base de datos.

Primero, debemos definir la nueva ruta en nuestro archivo principal:

```python
@app.route('/editar-nota/<int:id>', methods=['GET', 'POST'])
def edit_note(id):
    note = Note.query.get_or_404(id)
    
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        
        note.title = title
        note.content = content
        
        db.session.commit()
        return redirect(url_for('home'))
    
    return render_template('edit_note.html', note=note)
```

**Es importante destacar que utilizamos** `methods=['GET', 'POST']` **para permitir tanto la visualización del formulario como el procesamiento de los datos enviados**. El método `get_or_404()` es especialmente útil porque intenta obtener la nota con el ID especificado y, si no existe, devuelve automáticamente un error 404.

#### ¿Cómo estructurar el formulario de edición?

Para el formulario de edición, podemos crear una nueva plantilla o adaptar la existente. Lo crucial es mostrar los valores actuales de la nota en los campos del formulario:

```html
<form method="post">
    <div>
        <label for="title">Título</label>
        <input type="text" name="title" value="{{ note.title }}">
    </div>
    <div>
        <label for="content">Contenido</label>
        <textarea name="content">{{ note.content }}</textarea>
    </div>
    <button type="submit">Guardar nota</button>
</form>
```

**La clave aquí es utilizar los atributos `value` en los inputs y el contenido dentro de los elementos `textarea` para mostrar la información existente**. Esto permite al usuario ver y modificar los datos actuales.

#### ¿Por qué es importante implementar redirecciones después de editar?

Después de procesar un formulario de edición, es una buena práctica redirigir al usuario a otra página (como la página principal o la vista detallada de la nota). Esto evita problemas comunes como:

1. La reenvío accidental del formulario si el usuario recarga la página.
2. Confusión del usuario sobre si la acción se completó correctamente.
3. Problemas con el historial del navegador.

**La redirección post-edición es un patrón de diseño conocido como "Post/Redirect/Get" (PRG)**, que mejora significativamente la experiencia del usuario y previene envíos duplicados de formularios.

#### ¿Cómo mejorar la navegación para acceder a la edición?

Para facilitar el acceso a la función de edición, podemos agregar enlaces en nuestra lista de notas:

```html
{% for note in notes %}
    <div>
        <h3>{{ note.title }}</h3>
        <p>{{ note.content }}</p>
        <a href="{{ url_for('edit_note', id=note.id) }}">✏️ Editar</a>
    </div>
{% endfor %}
```

**La función `url_for()` es extremadamente útil aquí, ya que genera la URL correcta incluso si cambiamos la estructura de nuestras rutas en el futuro**. Además, al pasar el ID de la nota como parámetro, aseguramos que cada enlace apunte a la nota correcta.

#### ¿Qué consideraciones debemos tener al implementar la edición?

Al implementar funcionalidades de edición, debemos considerar varios aspectos importantes:

1. **Validación de datos**: Asegurarnos de que los datos enviados cumplan con nuestros requisitos.
2. **Control de acceso**: Verificar que el usuario tenga permisos para editar la nota.
3. **Manejo de errores**: Proporcionar mensajes claros si algo sale mal durante el proceso.
4. **Experiencia de usuario**: Hacer que el proceso sea intuitivo y proporcionar retroalimentación clara.

**La implementación de estas consideraciones puede variar según las necesidades específicas de la aplicación**, pero son fundamentales para crear una funcionalidad de edición robusta y amigable.

La capacidad de editar contenido en una aplicación web es una funcionalidad fundamental que mejora significativamente la experiencia del usuario. Con Flask, implementar esta funcionalidad es relativamente sencillo gracias a su sistema de rutas flexible y su integración con SQLAlchemy. ¿Has implementado alguna vez un sistema de edición en tus proyectos? Comparte tu experiencia en los comentarios.

## Refactorización y Organización de Proyectos en Flask

La **refactorización y organización de proyectos en Flask** es un paso clave para mantener el código limpio, escalable y fácil de mantener, especialmente cuando tu aplicación crece más allá de unos pocos archivos.

Aquí te muestro una **estructura de proyecto recomendada**, una breve **explicación de cada componente** y algunos **consejos de refactorización**:

### 🗂️ Estructura recomendada para un proyecto Flask

```
my_flask_app/
│
├── app/
│   ├── __init__.py         # Inicializa la app y registra Blueprints
│   ├── routes/
│   │   ├── __init__.py     # Registro de rutas agrupadas
│   │   └── notes.py        # Rutas relacionadas a notas
│   ├── models/
│   │   └── note.py         # Modelos SQLAlchemy
│   ├── forms/
│   │   └── note_form.py    # Formularios con Flask-WTF
│   ├── templates/
│   │   └── ...             # Archivos HTML
│   ├── static/
│   │   └── ...             # Archivos CSS, JS, imágenes
│   └── config.py           # Configuración general
│
├── migrations/             # Migraciones de base de datos (si usas Flask-Migrate)
├── tests/                  # Pruebas unitarias
├── venv/                   # Entorno virtual
├── .env                    # Variables de entorno
├── .flaskenv               # Configuración para Flask CLI
├── requirements.txt
└── run.py                  # Punto de entrada
```

### 🔧 `run.py` – Punto de entrada

```python
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
```

### 🔨 `app/__init__.py` – Crea la app y configura extensiones

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object("app.config.Config")

    db.init_app(app)

    from .routes import main
    app.register_blueprint(main)

    return app
```

### 📚 `app/routes/notes.py` – Rutas relacionadas con notas

```python
from flask import Blueprint, render_template, request, redirect, url_for
from app.models.note import Note
from app.forms.note_form import NoteForm
from app import db

main = Blueprint("main", __name__)

@main.route("/")
def index():
    notes = Note.query.all()
    return render_template("index.html", notes=notes)

@main.route("/create", methods=["GET", "POST"])
def create_note():
    form = NoteForm()
    if form.validate_on_submit():
        note = Note(title=form.title.data, content=form.content.data)
        db.session.add(note)
        db.session.commit()
        return redirect(url_for("main.index"))
    return render_template("create_note.html", form=form)
```

### 🧱 `app/models/note.py` – Modelo de datos

```python
from app import db
from datetime import datetime

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.String(200), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
```

### 📝 `app/forms/note_form.py` – Formulario

```python
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired

class NoteForm(FlaskForm):
    title = StringField("Título", validators=[DataRequired()])
    content = TextAreaField("Contenido", validators=[DataRequired()])
    submit = SubmitField("Guardar")
```

### ✅ Consejos de Refactorización

* **Divide por responsabilidad**: rutas, modelos, formularios, etc. en carpetas separadas.
* **Usa Blueprints** para modularizar rutas.
* **Crea una clase `Config`** centralizada para manejar distintos entornos (`development`, `production`, etc.).
* **Cambia variables sensibles** (como claves secretas) a `.env` usando `python-dotenv`.
* **Usa Flask-Migrate** para manejar cambios en la base de datos.

### Resumen

La refactorización de código es una práctica esencial para cualquier desarrollador que busque mantener sus proyectos escalables y fáciles de mantener. Cuando trabajamos con frameworks como Flask, organizar adecuadamente nuestro código no solo mejora la legibilidad, sino que también facilita el trabajo en equipo y la implementación de pruebas unitarias. En este artículo, exploraremos cómo transformar una aplicación Flask básica en una estructura más profesional y mantenible.

#### ¿Cómo preparar nuestro repositorio para un desarrollo profesional?

Antes de comenzar a refactorizar nuestro código, es importante asegurarnos de que nuestro repositorio esté correctamente configurado. Uno de los primeros pasos es crear un archivo `.gitignore` para evitar subir archivos innecesarios al repositorio.

#### ¿Por qué es importante el archivo .gitignore?

Cuando trabajamos con entornos virtuales y bases de datos locales, estos generan archivos que no deberían formar parte de nuestro repositorio. Para solucionar esto:

1. Crea un archivo llamado `.gitignore` en la raíz de tu proyecto.
2. Puedes utilizar plantillas predefinidas de GitHub para Python.
3. Añade extensiones específicas para tu proyecto, como `*.sqlite* `para ignorar archivos de base de datos SQLite.

**Este paso es fundamental** para mantener tu repositorio limpio y evitar conflictos innecesarios cuando trabajas en equipo.

#### ¿Cómo implementar el estándar PEP 8 en nuestro código?

El PEP 8 es el estándar de estilo para código Python que nos ayuda a mantener una estructura coherente y legible. Para implementarlo:

1. Instala herramientas como Ruff, que integra PEP 8 y otras utilidades.
2. Organiza tus imports al inicio del archivo.
3. Evita líneas demasiado largas, dividiéndolas adecuadamente.

```python
from flask import (
    Flask, render_template, request, 
    url_for, flash, redirect
)
```

**La legibilidad del código es crucial** cuando trabajas en equipos de desarrollo, ya que facilita la comprensión y modificación por parte de otros desarrolladores.

#### ¿Cómo estructurar una aplicación Flask para hacerla escalable?

Una aplicación Flask bien estructurada debe separar claramente sus componentes. Vamos a ver cómo podemos refactorizar nuestra aplicación para lograr esto.

#### ¿Cómo separar la configuración de la aplicación?

Es recomendable mantener la configuración de la aplicación en un archivo separado:

1. Crea un archivo config.py.
2. Define una clase Config que contenga todos los parámetros de configuración.

```python
import os

class Config:
    SECRET_KEY = 'tu_clave_secreta'
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database.db')}"
    # Otras configuraciones
```

3. En tu archivo principal, carga la configuración:

```python
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
```

**Esta separación permite modificar la configuración** sin tener que tocar el código principal de la aplicación, lo que es especialmente útil cuando tienes diferentes entornos (desarrollo, pruebas, producción).

#### ¿Cómo organizar los modelos de datos?

Los modelos de datos deben estar en un archivo o módulo separado:

1. Crea un archivo `models.py`.
2. Mueve tus definiciones de modelos a este archivo.

```python
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Note:
    # Definición del modelo
Importa los modelos en tu archivo principal:
from models import Note, db

# Inicializa la base de datos con la aplicación
db.init_app(app)
```

**Si tu aplicación crece y tienes muchos modelos**, considera crear un módulo `models` con archivos separados para cada dominio lógico.

#### ¿Qué hacer con las vistas y rutas?

Aunque no se cubrió completamente en la clase, una buena práctica es organizar las vistas utilizando Blueprints de Flask:

1. Agrupa las vistas relacionadas (por ejemplo, todas las operaciones de notas).
2. Elimina las vistas de prueba que ya no necesitas.
3. Organiza el código para que cada función de vista sea clara y tenga una única responsabilidad.

**Los Blueprints son una excelente manera de modularizar** tu aplicación Flask, permitiéndote dividir tu aplicación en componentes más pequeños y manejables.

La refactorización de código es un proceso continuo que mejora la calidad de tu aplicación. Siguiendo estas prácticas, no solo harás que tu código sea más mantenible, sino que también facilitarás la colaboración con otros desarrolladores y la implementación de pruebas automatizadas. ¿Has intentado refactorizar alguna de tus aplicaciones? Comparte tu experiencia en los comentarios.

**Lecturas recomendadas**

[PEP 8 – Style Guide for Python Code | peps.python.org](https://peps.python.org/pep-0008/)

[gitignore.io - Create Useful .gitignore Files For Your Project](https://gitignore.io/)

[Keyboard shortcuts VS Code](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)