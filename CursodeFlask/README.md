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

Este comando enviará una solicitud POST a nuestra ruta `/contacto` y nos mostrará la respuesta, incluyendo el código de estado HTTP.

**El uso de herramientas como curl es invaluable durante el desarrollo para probar rápidamente nuestros endpoints sin necesidad de crear interfaces de usuario completas.**

El decorador `@route` en Flask es una herramienta versátil que nos permite crear aplicaciones web robustas y APIs flexibles. Dominar su uso con diferentes métodos HTTP y tipos de respuesta es fundamental para cualquier desarrollador web que trabaje con Python. Te animo a experimentar con retornar HTML y a explorar otros métodos HTTP como PUT y PATCH para ampliar tus habilidades en el desarrollo web con Flask.

**Lecturas recomendadas**

[GitHub - platzi/curso-flask](https://github.com/platzi/curso-flask/)