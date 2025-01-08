# Curso de FastAPI

## FastAPI: La herramienta definitiva para el desarrollo web

**FastAPI** es una de las frameworks más modernas y eficientes para el desarrollo de aplicaciones web en Python. Ofrece una combinación única de velocidad, simplicidad y capacidades avanzadas para la creación de APIs eficientes y escalables.

#### **Características principales de FastAPI:**

1. **Alta Velocidad**: FastAPI es extremadamente rápido gracias al uso de esquemas de tipo de datos y validaciones asincrónicas, lo que permite un rendimiento superior, especialmente en aplicaciones con mucho tráfico o carga.

2. **Tipado de Datos y Validaciones**: Utiliza `pydantic` para definir modelos de datos de manera clara y precisa. Esto garantiza una validación de datos efectiva tanto en el servidor como en el cliente.

3. **Documentación Automática**: Genera documentación interactiva automáticamente usando Swagger y Redoc, lo que facilita el uso por parte de desarrolladores y usuarios.

4. **Soporte Asincrónico**: Permite la creación de APIs asíncronas para manejar tareas simultáneas y mejorar la eficiencia de la aplicación.

5. **Seguridad**: FastAPI cuenta con soporte para autenticación, autorización y manejo de tokens de seguridad (JWT, OAuth, etc.), mejorando la seguridad del desarrollo web.

6. **Desarrollo Rápido**: Proporciona un sistema de manejo de rutas y respuestas muy eficiente que reduce el tiempo de desarrollo en comparación con otros frameworks tradicionales como Flask.

### **Pasos básicos para empezar con FastAPI:**

1. **Instalación**:
   ```bash
   pip install fastapi
   ```

2. **Creación de una API básica**:

   ```python
   from fastapi import FastAPI

   app = FastAPI()

   @app.get("/")
   def read_root():
       return {"message": "Hello, World!"}
   ```

3. **Definición de rutas y validaciones**:

   ```python
   from fastapi import FastAPI, Query

   app = FastAPI()

   @app.get("/items/{item_id}")
   def read_item(item_id: int, q: str = Query(...)):
       return {"item_id": item_id, "query": q}
   ```

4. **Generación de documentación**:

   Ejecutar el servidor y acceder a la documentación interactiva en `http://127.0.0.1:8000/docs`.

5. **Soporte Asíncrono**:

   ```python
   from fastapi import FastAPI, HTTPException
   import asyncio

   app = FastAPI()

   @app.get("/async")
   async def async_endpoint():
       await asyncio.sleep(2)  # Simulando una tarea asincrónica
       return {"message": "Asynchronous processing completed"}
   ```

### **Ventajas del uso de FastAPI:**

- **Desarrollo ágil**: Gracias a su sintaxis concisa y clara, permite un desarrollo rápido de APIs.
- **Eficiencia**: Optimización del procesamiento mediante tecnologías modernas.
- **Comunidad activa**: Contiene una amplia comunidad, documentación y soporte constante.
- **Flexibilidad**: Compatible con múltiples frameworks y bibliotecas adicionales para extender funcionalidades.

FastAPI es una opción ideal para desarrolladores web que buscan una solución rápida, robusta y segura para la creación de APIs modernas y eficientes.

**Lecturas recomendadas**

[FastAPI](https://fastapi.tiangolo.com/)

## ¿Qué es y por qué usar FastAPI?

### **Qué es FastAPI y por qué usarlo**

**FastAPI** es un framework moderno y rápido para desarrollar aplicaciones web y APIs en Python. Fue diseñado para proporcionar una combinación única de velocidad, seguridad, facilidad de uso y escalabilidad.

---

### **Características principales de FastAPI**

1. **Alta Velocidad**:
   - FastAPI utiliza esquemas de tipo y validaciones asincrónicas para asegurar un rendimiento superior.
   - Aprovecha el motor Starlette y Pydantic, lo que permite manejar grandes volúmenes de tráfico con eficiencia.

2. **Tipado y Validación de Datos**:
   - Usa `pydantic` para definir modelos de datos, lo que asegura que los datos sean validados correctamente antes de ser procesados, garantizando la calidad de los datos.
   
3. **Automatización de Documentación**:
   - Genera documentación automática utilizando Swagger y Redoc. Esto mejora la experiencia del desarrollador y facilita la comprensión de la API por parte de los usuarios.

4. **Soporte Asincrónico**:
   - Permite crear APIs asíncronas para manejar múltiples solicitudes simultáneamente, mejorando la eficiencia de aplicaciones con cargas pesadas.

5. **Seguridad**:
   - Integra soporte para autenticación y autorización, incluyendo OAuth2, JWT y otros mecanismos de seguridad.

6. **Es fácil de aprender y usar**:
   - Su sintaxis es simple y concisa, lo que facilita la creación de APIs funcionales en poco tiempo.

---

### **Razones para usar FastAPI**

1. **Rendimiento**:
   - FastAPI está diseñado para ser muy rápido gracias a su soporte nativo para métodos asíncronos y su uso eficiente de recursos.
   
2. **Seguridad**:
   - Ofrece una gestión segura de datos, validación automática, y soporte para estándares modernos de seguridad como HTTPS, autenticación y validación de token.

3. **Facilidad de Uso**:
   - Su sintaxis es muy intuitiva y limpia, ideal para desarrolladores que buscan escribir código claro y fácil de mantener.

4. **Comunidad Activa y Documentación**:
   - Gran comunidad de desarrolladores, con documentación completa, ejemplos y soporte continuo para resolver problemas o dudas.

5. **Flexibilidad y Extensibilidad**:
   - Soporte para integraciones avanzadas como Middlewares, dependencias y uso extensivo de bibliotecas externas.

6. **Compatibilidad**:
   - Es altamente compatible con bibliotecas modernas de Python como `asyncio`, `pydantic` y otras populares.

---

### **Ejemplo Básico de FastAPI**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
```

Este pequeño ejemplo muestra cómo comenzar con una simple ruta que devuelve un mensaje básico. La capacidad de manejar rutas dinámicas, validar datos y generar documentación automática hacen de FastAPI una opción ideal para proyectos de alta demanda.

**Resumen**

FastAPI es un framework de Python diseñado para crear APIs rápidas, eficientes y bien documentadas. Este framework es moderno y permite a los desarrolladores conectar sistemas, como servidores y clientes, con pocas líneas de código. Además, genera documentación de forma automática y se destaca por su compatibilidad con el tipado de Python, lo cual facilita el autocompletado en los editores de texto, acelerando el proceso de desarrollo.

**¿Por qué elegir FastAPI para construir APIs?**

- **Rapidez**: Aunque Python es conocido por su menor velocidad en comparación con otros lenguajes, FastAPI implementa asincronismo, aprovechando funcionalidades avanzadas para acelerar el rendimiento.
- **Facilidad de uso**: Su sintaxis es sencilla y directa, lo que reduce la complejidad al escribir código.
- **Documentación automática**: Una vez creada la API, FastAPI genera la documentación de endpoints de manera automática, facilitando su uso en herramientas como Swagger o Postman.
- **Compatibilidad con tipado**: Aprovecha el tipado de Python para mejorar la precisión del código y el autocompletado, lo que se traduce en un desarrollo más ágil y seguro.

### ¿Cómo funciona FastAPI internamente?

FastAPI está basado en dos frameworks principales:

1. **Starlette**: Gestiona las solicitudes HTTP, permitiendo a la API recibir y responder peticiones de manera eficiente.
2. **Pydantic**: Facilita la creación de modelos de datos. Con estos modelos, puedes estructurar la información para agregar, modificar o eliminar datos en la API de forma controlada y validada.

### ¿Cómo puedo explorar más sobre FastAPI?

Existen múltiples recursos oficiales para profundizar en el uso de FastAPI:

- **Documentación oficial**: Incluye guías detalladas sobre sus funciones principales y ejemplos prácticos.
- **Repositorio de GitHub**: Permite ver el código fuente y contribuir al desarrollo del framework. Recomendamos explorar las etiquetas como “good first issue” para empezar a colaborar y mejorar FastAPI.
- **Discord de la comunidad**: Un espacio donde desarrolladores comparten ideas, resuelven dudas y contribuyen al crecimiento de FastAPI.

### ¿Qué es OpenAPI y cómo se relaciona con FastAPI?

FastAPI genera automáticamente un archivo JSON compatible con OpenAPI. Este archivo describe todos los endpoints de la API, sus parámetros y los datos que se pueden enviar. Esta funcionalidad es útil para:

- Ver todos los endpoints y sus variables.
- Integrarse fácilmente con Swagger o Postman.
- Probar la API directamente en el navegador usando herramientas de autogeneración como el botón “Try Out” de Swagger.

### ¿Cómo comenzar con FastAPI?

Como reto, intenta instalar FastAPI en tu entorno de desarrollo. La documentación oficial de FastAPI ofrece una guía completa para instalarlo y crear una API básica. Este ejercicio te permitirá familiarizarte con la configuración inicial y prepararte para construir APIs avanzadas en el futuro.

**Lecturas recomendadas**

[Welcome to Pydantic - Pydantic](https://docs.pydantic.dev/latest/)

[Starlette](https://www.starlette.io/)

## Instalación y configuración de FastAPI

Para trabajar con un framework en Python como FastAPI, siempre es recomendable emplear entornos virtuales. Estos entornos permiten gestionar las dependencias de un proyecto sin interferir con otros. A continuación, se explican los pasos clave para crear y configurar un entorno virtual y desarrollar una primera API básica.

### ¿Cómo crear un entorno virtual para FastAPI?

1. **Crear el entorno virtual:**

 - Abre la terminal y navega a la carpeta donde se encuentra tu proyecto. Utiliza el módulo venv de Python para crear un entorno virtual:
 
`python -m venv vm`

 - Esto generará un entorno virtual en una carpeta llamada vm dentro de tu proyecto.
 
2. **Activar el entorno virtual:**

 - En sistemas Unix, ejecuta el siguiente comando:
 
`source vm/bin/activate`

 - Esto permite aislar las dependencias de tu proyecto dentro del entorno virtual.
 
### ¿Cómo instalar FastAPI y sus dependencias?

1. **Instalar FastAPI:**

 - Con el entorno virtual activo, instala FastAPI:
 
`pip install "fastapi[standard]"`

 - Si recibes errores de interpretación, agrega comillas dobles para evitar problemas con las llaves {} que incluyen dependencias adicionales para la ejecución de FastAPI en entornos locales.
 
2. **Verificar las dependencias instaladas:**

 - Tras la instalación, puedes listar las dependencias para observar los paquetes añadidos, como **Jinja** (templates), **Markdown** (manejo de texto) y **Uvicorn** (para ejecutar aplicaciones como servidor web).
 
### ¿Cómo crear un primer endpoint con FastAPI?

1. **Configurar la estructura de archivos:**

 - Crea una carpeta para el proyecto:
 
`mkdir curso_fastapi_project`

 - Dentro de esta carpeta, crea un archivo main.py para definir el primer endpoint.
 
2. **Desarrollar la API en` main.py`:**

 - Abre el archivo en tu editor y añade el siguiente código básico:
 
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"mensaje": "Hola, Mundo"}
```

 - La función root define un endpoint básico que devuelve un mensaje JSON. Utiliza el decorador `@app.get("/")` para indicar que este endpoint responde a solicitudes GET en la ruta raíz (`/`).
 
### ¿Cómo ejecutar y probar la API en desarrollo?

1. **Iniciar el servidor:**

 - Usa Uvicorn para ejecutar la aplicación:
 
`uvicorn main:app --reload`

 - El parámetro --reload activa el modo de desarrollo, permitiendo recargar la API automáticamente cada vez que guardes cambios en el código.
 
2. **Verificar en la terminal:**

Al ejecutar, Uvicorn muestra la URL de acceso a la API y la documentación generada automáticamente en `/docs`. Puedes acceder a la API en [http://localhost:8000](http://localhost:8000/ "http://localhost:8000") y la documentación en [http://localhost:8000/docs](http://localhost:8000/docs "http://localhost:8000/docs").

**Lecturas recomendadas**

[Uvicorn](https://www.uvicorn.org/)

[FastAPI](https://fastapi.tiangolo.com/)

[GitHub - fastapi/fastapi: FastAPI framework, high performance, easy to learn, fast to code, ready for production](https://github.com/fastapi/fastapi)

## Parámetros de ruta y consultas en FastAPI

Crear un endpoint en FastAPI que devuelva la hora en función del país permite ofrecer una API flexible y personalizable. A continuación, exploramos cómo construir esta funcionalidad con un endpoint dinámico y cómo incluir parámetros adicionales para definir el formato de la hora.

### ¿Cómo crear un endpoint que devuelva la hora del servidor?

1. **Función inicial para la hora del servidor**

En el archivo de sincronización (`sync`), creamos una función llamada `time` que retorne la hora actual del servidor. Para ello:

 - Importamos el módulo datetime desde la librería de Python.
 - En la función, utilizamos datetime.now() para obtener la hora actual.
 
2. **Configurar la función como endpoint**

Para que el endpoint sea accesible, decoramos la función con **@app.get("/time")**. Este decorador registra el endpoint para que esté disponible en la URL **/time**.

### ¿Cómo agregar variables en un endpoint?

Un endpoint estático es poco común en aplicaciones que requieren personalización. FastAPI permite recibir variables directamente en la URL, por lo que podemos modificar el endpoint para que acepte un código de país y devuelva la hora correspondiente en ese huso horario.

1. **Añadir el código ISO del país**

Modificamos el endpoint y añadimos un parámetro en la URL. Por ejemplo: `@app.get("/time/{iso_code}")`. Así, cuando el usuario indique el código de país, el sistema sabrá de qué huso horario obtener la hora.

2. **Tipar la variable**

Es esencial declarar el tipo de dato del parámetro `iso_code`. Al indicar `iso_code: str`, ayudamos a que FastAPI maneje correctamente el dato, garantizando que se trate de un texto. Esto también permite acceder a métodos específicos de cadenas de texto en Python, como `.upper()`.

### ¿Cómo ajustar el formato de entrada del parámetro?

Para mejorar la usabilidad:

- Convertimos `iso_code` a mayúsculas (`iso_code.upper()`). Así, la entrada será uniforme sin importar cómo el usuario ingrese el código.

- Definimos un diccionario que contiene los husos horarios por país, en el que las claves están en mayúsculas. Esto asegura que, al consultar el diccionario, se encuentre el huso horario correcto.

### ¿Cómo devolver la hora en el huso horario del país?

1. **Obtener el huso horario**
Con el código ISO del país, utilizamos `timezone.get(iso_code)` para obtener la zona horaria correspondiente.

2. **Formatear la hora según el huso horario**

Importamos el módulo `zoneinfo` y configuramos la zona horaria del resultado. De este modo, al retornar `datetime.now(tz=timezone)`, se muestra la hora correspondiente al país especificado.

### ¿Cómo agregar parámetros opcionales para formato de hora?

Finalmente, para que el usuario pueda decidir el formato de hora:

- Añadimos un parámetro opcional (`format_24`) que indique si se desea la hora en formato de 24 horas.
- En la lógica de la función, verificamos el parámetro y ajustamos el formato de salida para que muestre la hora en el formato deseado.

En **FastAPI**, los parámetros de ruta y las consultas son fundamentales para manejar la entrada de datos HTTP de manera flexible. A continuación te explico cómo funcionan y cómo utilizarlos.

### **Parámetros de ruta**  
Los parámetros de ruta son aquellos que se pasan directamente en la URL. Se pueden definir usando corchetes `{}` dentro de las funciones `app.get()`.

#### Sintaxis básica:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

**Uso**:
- `{item_id}` es un parámetro de ruta que espera un valor entero (`int`).
- El valor del `item_id` que se pase en la URL será accesible como un argumento en la función.

#### **Ejemplo**:
- URL: `/items/123`
- Respuesta: `{"item_id": 123}`

### **Parámetros de consulta**  
Los parámetros de consulta son aquellos que se incluyen después del signo de interrogación `?` en la URL, y pueden tener varios pares clave-valor separados por `&`. Se pueden definir en la función `app.get()` usando el parámetro `query`.

#### Sintaxis básica:
```python
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/search")
async def search(
    q: Optional[str] = None,       # Parámetro de consulta opcional
    limit: int = 10,                # Parámetro de consulta con un valor por defecto
    offset: int = 0
):
    return {"query": q, "limit": limit, "offset": offset}
```

**Uso**:
- `q` es un parámetro de consulta que acepta un valor de tipo `str`, que puede ser opcional (`Optional[str]`).
- `limit` es un parámetro de consulta que define el número máximo de resultados, con un valor por defecto de `10`.
- `offset` es otro parámetro de consulta que indica el número de resultados a saltar, con un valor por defecto de `0`.

#### **Ejemplo**:
- URL: `/search?q=fastapi&limit=5&offset=10`
- Respuesta: `{"query": "fastapi", "limit": 5, "offset": 10}`

### **Combinar parámetros de ruta y consulta**  
Se pueden usar ambos al mismo tiempo en una misma ruta. FastAPI los maneja de manera ordenada y los puede distinguir fácilmente.

#### Sintaxis combinada:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "query": q}
```

**Uso**:
- `item_id` es el parámetro de ruta.
- `q` es el parámetro de consulta opcional.

#### **Ejemplo**:
- URL: `/items/123?q=fastapi`
- Respuesta: `{"item_id": 123, "query": "fastapi"}`

### **Validación de parámetros**  
En FastAPI puedes especificar el tipo de validación que deseas realizar en los parámetros utilizando las anotaciones de tipo (`int`, `str`, `float`, etc.) y los posibles valores por defecto, como se ha visto en los ejemplos anteriores.

#### **Validaciones adicionales**:
- Se pueden añadir validaciones más detalladas, como especificar el rango de valores, o establecer valores por defecto.

### **Resumen**:
- **Parámetros de ruta**: Pasados directamente en la URL y accesibles por medio de `app.get("/{param}")`.
- **Parámetros de consulta**: Pasados después del `?` en la URL y accesibles por medio de `app.get("/route?param=value")`.

Esto te permite manejar de manera flexible cómo los usuarios interactúan con tus APIs en función de sus necesidades.

**Lecturas recomendadas**

[datetime — Basic date and time types — Python 3.13.0 documentation](https://docs.python.org/3/library/datetime.html)

[zoneinfo — IANA time zone support — Python 3.13.0 documentation](https://docs.python.org/3/library/zoneinfo.html)