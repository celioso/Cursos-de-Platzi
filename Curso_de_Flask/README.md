### Curso de Flask

### Windows
`pip install virtualenv`

### linux
`sudo apt install python3-virtualenv`

### Crear el ambiente virtual  
`virtualenv --python=pyton3 venv` o `python3 -m venv nombre-venv` o el nombre que desee darle, pero siempre se usa venv

para activar el ambiente se usa.

**windows**
`.\venv\Scriptes\activate` o `.\venv\Scriptes\activate.ps1` en PowerShell 

en linux `source venv/bin/activate`

**Para desactivar **

windows o linux `deactivate` o `.\venv\Scriptes\deactivate`

**Eliminar ambiente virtual**
**linux**
`rm -rf venv`

**Windows**
*shell*

`rmdir /s /q mi-entorno`

*powerShell*

`remove_Item -recurse -force mi-entorno`

## Instalar librerias de otro ambiente
 el archivo requirements.txt

 `pip install -r requirements.txt`

### Hello World Flask

Estos son los conceptos principales que debes entender antes de hacer un Hello World en Flask:

- **virtualenv**: es una herramienta para crear entornos aislados de Python.

- **pip:**es el instalador de paquetes para Python.

- **requirements.txt**: es el archivo en donde se colocará todas las dependencias a instalar en nuestra aplicación.

- **FLASK_APP**: es la variable para identificar el archivo donde se encuentra la aplicación.

archivo main.py
```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello world Flask"
```

luego se exporta.
`export FLASK_APP=main.py`

pare verificar
`echo $FLASK_APP`

Para activa el servidor local
`flask run`

### Debugging en Flask

Activar Debug mode:

`export FLASK_DEBUG=1`

ver la activación. 
`echo $FLASK_DEBUG`

activamos el servidor local .
`flask run`

### Ciclos de Request y Response

Request-Response: es uno de los métodos básicos que usan las computadoras para comunicarse entre sí, en el que la primera computadora envía una solicitud de algunos datos y la segunda responde a la solicitud.

Por lo general, hay una serie de intercambios de este tipo hasta que se envía el mensaje completo.

Por ejemplo: navegar por una página web es un ejemplo de comunicación de request-response.

Request-response se puede ver como una llamada telefónica, en la que se llama a alguien y responde a la llamada; es decir hacemos una petición y recibimos una respuesta.

### Templates con Jinja 2

### Archivos de la clase
**Lecturas recomendadas**

[Welcome | Jinja2 (The Python Template Engine)](https://jinja.palletsprojects.com/en/3.1.x/ "Welcome | Jinja2 (The Python Template Engine)")

### Estructuras de control

**Iteración:** es la repetición de un segmento de código dentro de un programa de computadora. Puede usarse tanto como un término genérico (como sinónimo de repetición), así como para describir una forma específica de repetición con un estado mutable.

Un ejemplo de iteración sería el siguiente:

```python
{% for key, segment in segment_details.items() %}
        <tr>
                <td>{{ key }}td>
                <td>{{ segment }}td>
        tr>
{% endfor %} 
```
En este ejemplo estamos iterando por cada *segment_details.items()* para mostrar los campos en una tabla `{{ key }}` `{{ segment }}` de esta forma dependiendo de cuantos *segment_details.items(*) haya se repetirá el código.

### Códigos de error:
**100**: no son errores sino mensajes informativos. Como usuario nunca los verás, sino que entre bambalinas indica que la petición se ha recibido y se continúa el proceso.

**200**: estos códigos también indican que todo ha ido correctamente. La petición se ha recibido, se ha procesado y se ha devuelto satisfactoriamente. Por tanto, nunca los verás en tu navegador, pues significan que todo ha ido bien.

**300**: están relacionados con redirecciones. Los servidores usan estos códigos para indicar al navegador que la página o recurso que han pedido se ha movido de sitio. Como usuario, no verás estos códigos, aunque gracias a ellos una página te podría redirigir automáticamente a otra.

**400**: corresponden a errores del cliente y con frecuencia sí los verás. Es el caso del conocido error 404, que aparece cuando la página que has intentado buscar no existe. Es, por tanto, un error del cliente (la dirección web estaba mal).

**500**: mientras que los códigos de estado 400 implican errores por parte del cliente (es decir, de parte tuya, tu navegador o tu conexión), los errores 500 son errores desde la parte del servidor. Es posible que el servidor tenga algún problema temporal y no hay mucho que puedas hacer salvo probar de nuevo más tarde.

**Framework**: es un conjunto estandarizado de conceptos, prácticas y criterios para enfocar un tipo de problemática particular que sirve como referencia, para enfrentar y resolver nuevos problemas de índole similar.

### lecturas
[Flask-Bootstrap — Flask-Bootstrap 3.3.7.1 documentation](https://pythonhosted.org/Flask-Bootstrap/ "Flask-Bootstrap — Flask-Bootstrap 3.3.7.1 documentation")

[Bootstrap · The world's most popular mobile-first and responsive front-end framework.](https://getbootstrap.com/docs/3.3/ "Bootstrap · The world's most popular mobile-first and responsive front-end framework.")

### Configuración de Flask

Para activar el *development mode* debes escribir lo siguiente en la consola:

    export FLASK_ENV=development
    echo $FLASK_ENV
	
**SESSION**: es un intercambio de información interactiva semipermanente, también conocido como diálogo, una conversación o un encuentro, entre dos o más dispositivos de comunicación, o entre un ordenador y usuario.

### Implementación de Flask-Bootstrap y Flask-WTF
[Flask-WTF — Flask-WTF 0.14](https://flask-wtf.readthedocs.io/en/1.2.x/ "Flask-WTF — Flask-WTF 0.14")

### Uso de método POST en Flask-WTF

Flask acepta peticiones **GET** por defecto y por ende no debemos declararla en nuestras rutas.

Pero cuando necesitamos hacer una petición **POST** al enviar un formulario debemos declararla de la siguiente manera, como en este ejemplo:

`@app.route('/platzi-post', methods=['GET', 'POST'])`
Debemos declararle además de la petición que queremos, **GET**, ya que le estamos pasando el parámetro methods para que acepte solo y únicamente las peticiones que estamos declarando.

De esta forma, al actualizar el navegador ya podremos hacer la petición **POST** a nuestra ruta deseada y obtener la respuesta requerida.

### Pruebas básicas con Flask-testing

La etapa de pruebas se denomina *testing* y se trata de una investigación exhaustiva, no solo técnica sino también empírica, que busca reunir información objetiva sobre la calidad de un proyecto de software, por ejemplo, una aplicación móvil o un sitio web.

El objetivo del *testing* no solo es encontrar fallas sino también aumentar la confianza en la calidad del producto, facilitar información para la toma de decisiones y detectar oportunidades de mejora.

para arrancar el test se utiliza `flask test`

### Base de datos y App Engine con Flask

- **Bases de Datos SQL**: su composición esta hecha con bases de datos llenas de tablas con filas que contienen campos estructurados. No es muy flexible pero es el más usado. Una de sus desventajas es que mientras más compleja sea la base de datos más procesamiento necesitará.

- **Base de Datos NOSQL**: su composición es no estructurada, es abierta y muy flexible a diferentes tipos de datos, no necesita tantos recursos para ejecutarse, no necesitan una tabla fija como las que se encuentran en bases de datos relacionales y es altamente escalable a un bajo costo de hardware.

[google cloud](https://console.cloud.google.com/welcome/new "google cloud")

### Configuración de proyecto en Google Cloud Platform
primer guia de instalar Instala Google Cloud CLI
[Link](https://cloud.google.com/sdk/docs/install-sdk?hl=es-419#linux "Link")

Luego de crearlo y conectarlo, se utiliza estos comando para su conexión

- `gcloud auth login`
- `gcloud auth application-default login`


### Funciona!, me sirvió para solucionar este error :

'ValueError: Project ID is required to access Firestore. Either set the projectId option, or use service account credentials. Alternatively, set the GOOGLE_CLOUD_PROJECT environment variable.'

```Python
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


project_id = 'platzi-flask-2....'
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
  'projectId': project_id,
})


db = firestore.client()


defget_users():
    return db.collection('users').get()
```

