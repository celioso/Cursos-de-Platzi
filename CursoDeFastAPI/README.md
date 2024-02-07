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

aaaaaaaaaaaaaaaaaaaaaaaaaaaaa