# Curso de FastAPP

### Configuración de un servidor con FastAPI

**Creación de un entorno virtual (venv):** Se creó un entorno virtual utilizando venv. Un entorno virtual es un ambiente aislado que permite instalar y gestionar paquetes de Python específicos para un proyecto, sin afectar al sistema global.

`python -m venv nombre_del_entorno`

**Activación del entorno virtual:** Se activó el entorno virtual para asegurar que las dependencias se instalen y se ejecuten dentro de este entorno aislado.

`source nombre_del_entorno/bin/activate  # En sistemas basados en Unix`

**Actualizar pip**

`python3 -m pip install --upgrade pip`

**Instalación de Uvicorn:** Uvicorn es un servidor ASGI (Asynchronous Server Gateway Interface) que se utiliza para ejecutar aplicaciones FastAPI de manera asincrónica. Se instaló Uvicorn dentro del entorno virtual.

`pip install uvicorn`


**Desarrollo de una aplicación FastAPI simple:** Se creó un archivo Python con un código mínimo de FastAPI. El código define una instancia de la clase FastAPI y una ruta (/) que responde con un mensaje "Hello world!" cuando se realiza una solicitud GET.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def message():
    return"Hello world🔥💬"
```

**Ejecución de la aplicación con Uvicorn:** Se utilizó el servidor Uvicorn para ejecutar la aplicación FastAPI. La aplicación se configuró para escuchar en todas las interfaces (`0.0.0.0`) y en un puerto específico.

`uvicorn main:app`

`uvicorn nombre_del_archivo:app --reload --host 0.0.0.0 --port 8000`
uvicorn main:app --reload --host 127.0.0.1 --port 5000

**nombre_del_archivo** es el nombre del archivo Python que contiene la aplicación FastAPI.
- **`--reload`** habilita la recarga automática de la aplicación cuando se realizan cambios en el código.
- **`--host 0.0.0.0`** permite que la aplicación sea accesible desde cualquier dirección IP en la red.
- **`--port 8000`** especifica el puerto en el que la aplicación escuchará las solicitudes.

### Métodos HTTP en FastAPI

**Métodos HTTP**
El protocolo HTTP es aquel que define un conjunto de métodos de petición que indican la acción que se desea realizar para un recurso determinado del servidor.

Los principales métodos soportados por HTTP y por ello usados por una API REST son:
**POST:** crear un recurso nuevo.
**PUT:** modificar un recurso existente.
**GET:** consultar información de un recurso.
**DELETE:** eliminar un recurso.

Como te diste cuenta con estos métodos podemos empezar a crear un CRUD en nuestra aplicación.

**¿De qué tratará nuestra API?**
El proyecto que estaremos construyendo a lo largo del curso será una API que nos brindará información relacionada con películas, por lo que tendremos lo siguiente:

**Consulta de todas las películas**
Para lograrlo utilizaremos el método GET y solicitaremos todos los datos de nuestras películas.

**Filtrado de películas**
También solicitaremos información de películas por su id y por la categoría a la que pertenecen, para ello utilizaremos el método GET y nos ayudaremos de los parámetros de ruta y los parámetros query.

**Registro de peliculas**
Usaremos el método POST para registrar los datos de nuestras películas y también nos ayudaremos de los esquemas de la librería pydantic para el manejo de los datos.

**Modificación y eliminación**
Finalmente para completar nuestro CRUD realizaremos la modificación y eliminación de datos en nuestra aplicación, para lo cual usaremos los métodos PUT y DELETE respectivamente.

Y lo mejor es que todo esto lo estarás construyendo mientras aprendes FastAPI, te veo en la siguiente clase donde te enseñaré cómo puedes utilizar el método GET.

![API](https://i.ibb.co/HgzHhTk/Captura-de-pantalla-2024-01-22-a-la-s-6-25-41-p-m.png)

### Códigos de estado HTTP en FastAPI

[HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status "HTTP response status codes")

### Flujo de autenticación en FastAPI

#### Flujo de autenticación
Ahora empezaremos con el módulo de autenticaciones pero antes quiero explicarte un poco acerca de lo que estaremos realizando en nuestra aplicación y cómo será el proceso de autenticación y autorización.

#### Ruta para iniciar sesión
Lo que obtendremos como resultado al final de este módulo es la protección de determinadas rutas de nuestra aplicación para las cuales solo se podrá acceder mediante el inicio de sesión del usuario. Para esto crearemos una ruta que utilice el método POST donde se solicitarán los datos como email y contraseña.

#### Creación y envío de token
Luego de que el usuario ingrese sus datos de sesión correctos este obtendrá un token que le servirá para enviarlo al momento de hacer una petición a una ruta protegida.

#### Validación de token
Al momento de que nuestra API reciba la petición del usuario, comprobará que este le haya enviado el token y validará si es correcto y le pertenece. Finalmente se le dará acceso a la ruta que está solicitando.

En la siguiente clase empezaremos con la creación de una función que nos va a permitir generar tokens usando la librería pyjwt.

### Generando tokens con PyJWT

instalamos la instancia con pip para generar el token

`pip install pyjwt`

si ya esta intalado lo podemos actualizar con:

`python3 -m pip install --upgrade pip`

se crea el archivo jwt_manager.py con el siguiente codigo:

```python
from jwt import encode

def create_token(data: dict):
    token: str = encode(payload=data, key = "my_secrete_key", algorithm = "HS256")
    return token
```

en main.py importamos  from `jwt_manager import create_token`

se crea el siguiente codigo:

```python
@app.post("/login", tags=["auth"])
def login(user: User):
    return user
```

Recomendacion de compañero para guardar datos semsibles:

Es mejor guardar datos sensibles como la secretKey en las variables de entorno, son 3 pasos:

1. Instalen

`pip install python-dotenv`

2. Crean un archivo en la raíz del proyecto con nombre y extencion *.env* y agregan las variables de entorno, en nuestro caso:

`SECRET_KEY = secretKey`

Ahora se accede a las variables de entorno así, en nuestro archivo* jwt_manager.py*:

```python
import os
from dotenv import load_dotenv
from jwt import encode, decode

load_dotenv()

secretKey = os.getenv("SECRET_KEY")


def create_token(data: dict) -> str:
    token: str = encode(payload=data, key=secretKey, algorithm="HS256")
    return token


def validate_token(token: str) -> dict:
    data: dict = decode(token, key=secretKey, algorithms=["HS256"])
    return data
```

### Instalación y configuración de SQLAlchemy
- En las extensiones de VScode buscar SQLite Viewer e instalarlo.
- También instalar el modulo de sqalchemy desde la terminal

`pip install sqlalchemy`

- Crear la carpeta config

- Dentro de dicha carpeta crear el archivo __init__.py para que detecte la carpeta como un modulo.

- Crear otro archivo llamado database.py donde añadiremos las configuraciones.

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

sqlite_file_name = "database.sqlite"
base_dir = os.path.dirname(os.path.realpath(__file__))

database_url = f"sqlite:///{os.path.join(base_dir, sqlite_file_name)}"

engine = create_engine(database_url, echo=True)

Session = sessionmaker(bind=engine)

Base = declarative_base()
```

Esta clase quizás está un poco diferente al resto, por si alguien tiene dudas de lo que se realizó (para repasar)

1. **base_dir** (será nuestra url del archivo de sqlite).
2. **database_url** une un string adicional con lo que declarameos de la ruta y nombre del archivo
3. **engine** crea una instancia del resultado de lo que devuelve la funcion create_engine().
4. **Session** guarda la session creada a partir del engine.
5. **Base** se ha declarado pero su uso será explicado más adelante.

### Creación de modelos con SQLAlchemy

La "magia" detrás de la creación automática de la tabla "movies" en SQLAlchemy se debe a la clase Base de `config.database.py`, que es una instancia de `declarative_base()`. Al heredar de esta clase al definir la clase Movie, se establece una conexión con la base de datos. Al ejecutar `Base.metadata.create_all(bind=Engine)` en `main.py`, SQLAlchemy utiliza esa conexión para crear la tabla automáticamente. Este proceso simplifica la creación de tablas y se basa en la funcionalidad ORM de SQLAlchemy.

Se soluciona instalando SQLAlchemy Flask-SQLAlchemy

`sudo pip3 install SQLAlchemy Flask-SQLAlchemy`