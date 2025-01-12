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

## ¿Cómo validar datos en FastAPI con Pydantic?

En FastAPI, la validación de datos se realiza mediante modelos de **Pydantic**. Estos modelos permiten definir la estructura y los tipos de datos esperados para solicitudes entrantes, como datos JSON enviados en un `POST` o `PUT`. Pydantic valida automáticamente los datos y genera respuestas de error claras cuando los datos no cumplen con las restricciones.

### Pasos para validar datos con Pydantic

1. **Definir un modelo Pydantic**:
   Utiliza la clase `BaseModel` de Pydantic para definir los campos esperados y sus tipos.

2. **Usar el modelo como parámetro en las rutas**:
   Declara el modelo en la función de la ruta, y FastAPI automáticamente lo usará para validar la entrada.

3. **Agregar validaciones personalizadas** (opcional):
   Puedes incluir restricciones como valores máximos, mínimos, expresiones regulares, etc.

---

### Ejemplo básico

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Modelo Pydantic
class User(BaseModel):
    username: str
    email: str
    age: int

# Ruta que valida el cuerpo de la solicitud
@app.post("/create-user/")
async def create_user(user: User):
    return {"message": "User created successfully!", "user": user}
```

**Explicación**:
- `username`: debe ser una cadena (`str`).
- `email`: debe ser una cadena que represente un correo electrónico.
- `age`: debe ser un número entero (`int`).

Si envías un JSON mal formado, FastAPI responderá con un error como este:

```json
{
    "detail": [
        {
            "loc": ["body", "age"],
            "msg": "value is not a valid integer",
            "type": "type_error.integer"
        }
    ]
}
```

---

### Ejemplo avanzado con validaciones personalizadas

Puedes usar Pydantic para agregar restricciones más avanzadas:

```python
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=18, le=120, description="Age must be between 18 and 120")

@app.post("/create-user/")
async def create_user(user: User):
    return {"message": "User created successfully!", "user": user}
```

**Restricciones añadidas**:
1. `username`: Longitud mínima de 3 caracteres, máxima de 50.
2. `email`: Usa el tipo especial `EmailStr` para validar que es un correo válido.
3. `age`: El rango permitido es entre 18 y 120 (`ge=18`, `le=120`).

---

### Ejemplo con parámetros de consulta y cuerpo

FastAPI permite combinar validaciones en el cuerpo (`body`) y parámetros de consulta (`query`):

```python
from fastapi import Query, Body

@app.post("/create-user/")
async def create_user(
    username: str = Query(..., min_length=3, max_length=50),
    email: str = Body(...),
    age: int = Query(..., ge=18, le=120)
):
    return {"username": username, "email": email, "age": age}
```

Aquí:
- `Query`: Valida los parámetros de consulta.
- `Body`: Valida el cuerpo de la solicitud.

---

### Documentación automática

Cuando usas modelos de Pydantic, FastAPI genera automáticamente documentación en Swagger UI (accesible en `/docs`) y en ReDoc (accesible en `/redoc`), mostrando las restricciones de los campos esperados.

---

### Resumen
FastAPI + Pydantic te permite:
- Definir estructuras claras y tipos de datos para las entradas.
- Realizar validaciones automáticas.
- Generar documentación interactiva automáticamente.

### Resumen

Para crear un endpoint dinámico y seguro en FastAPI, es fundamental validar la información recibida, especialmente si el contenido se envía en el cuerpo de la solicitud. Los usuarios pueden ingresar datos incorrectos o no válidos, como un correo electrónico mal formateado, por lo que validar estos datos es crucial para el correcto funcionamiento de la API. FastAPI facilita esta validación a través de **Pydantic**, una biblioteca de Python que permite construir modelos de datos robustos. A continuación, exploraremos cómo crear un modelo básico de cliente para validar datos en un endpoint.

### ¿Cómo estructurar un modelo de datos en FastAPI?

Para definir un modelo de datos, FastAPI emplea Pydantic, que permite usar clases para representar un esquema y validar la información que ingresa. Los pasos iniciales incluyen:

- Importar `BaseModel` de Pydantic.
- Crear una clase llamada `Customer` que herede de `BaseModel`.
- Definir campos dentro de la clase con sus tipos, por ejemplo, `name: str` para el nombre y `age: int` para la edad.
- Utilizar `typing` para permitir múltiples tipos de datos, como en el campo description, que podría ser de tipo `str` o `None` (opcional).

FastAPI valida automáticamente los datos ingresados en cada campo según el tipo especificado. Por ejemplo, si se establece que el campo `name` debe ser un string, cualquier otro tipo de entrada generará un error de validación.

### ¿Cómo integrar el modelo en un endpoint?

Una vez definido el modelo, el siguiente paso es integrarlo en un endpoint. Esto se realiza mediante una función asincrónica, por ejemplo, `async def create_customer`, que acepta datos de tipo `Customer` en el cuerpo de la solicitud.

1. Se define el endpoint con el método `POST`, para cumplir con las recomendaciones REST al crear recursos.
2. Se registran los datos del cliente con el decorador `@app.post("/customers")`.
3. En el cuerpo de la solicitud, los datos enviados serán automáticamente validados según el esquema de `Customer`.
4. Finalmente, la función puede retornar los mismos datos recibidos para verificar su recepción o realizar acciones adicionales como guardar en una base de datos o enviar una notificación.

### ¿Qué sucede al probar el endpoint?

Para probar el endpoint, FastAPI proporciona una documentación interactiva en `/docs`. Allí, es posible ver los campos requeridos y probar el endpoint directamente:

- Al hacer clic en “Try it out”, se pueden llenar los campos y enviar la solicitud.
- La respuesta muestra el JSON recibido o los errores de validación. Por ejemplo, si name debe ser un string pero se envía un número, se mostrará un mensaje de error detallado indicando el problema y el campo afectado.

Si el servidor responde con un `200 OK`, es posible que se esté usando el código HTTP incorrecto para una creación de recurso. En estos casos, lo recomendable es devolver un `201 Created` cuando el recurso se haya almacenado correctamente.

### ¿Cómo manejar errores y otros códigos de respuesta?

FastAPI permite definir diferentes códigos de respuesta, esenciales para indicar el estado de las solicitudes:

- **200**: Solicitud exitosa.
- **201**: Recurso creado.
- **422**: Error de validación, útil cuando los datos ingresados no cumplen con el modelo definido.

Al enviar datos no válidos, como un número en el campo `name`, FastAPI devuelve automáticamente un `422`, especificando el error en el JSON de la respuesta. Este sistema facilita identificar problemas y proporciona mensajes claros para corregir errores en el frontend.

**Lecturas recomendadas**

[Welcome to Pydantic - Pydantic](https://docs.pydantic.dev/latest/)

## Modelado de Datos en APIs con FastAPI

El modelado de datos en APIs con FastAPI se basa en el uso de **Pydantic** para definir y validar esquemas de datos de manera declarativa. Esto permite crear APIs robustas y confiables que manejan entradas y salidas con estructuras bien definidas.

### **Elementos clave del modelado de datos**

1. **Modelos de Pydantic**:
   Los modelos de Pydantic son clases basadas en `BaseModel` que definen los datos esperados y sus validaciones.

   ```python
   from pydantic import BaseModel

   class User(BaseModel):
       id: int
       name: str
       email: str
       age: int
       is_active: bool = True  # Valor por defecto
   ```

   - Cada atributo se tipa explícitamente.
   - Puedes incluir valores predeterminados.
   - Se valida automáticamente cuando los datos son enviados a la API.

2. **Validación automática**:
   FastAPI valida automáticamente los datos enviados a través del cuerpo, parámetros de consulta, encabezados, cookies, etc.

   - **Cuerpo de la solicitud**:
     ```python
     from fastapi import FastAPI
     from pydantic import BaseModel

     app = FastAPI()

     class User(BaseModel):
         id: int
         name: str
         email: str

     @app.post("/users/")
     async def create_user(user: User):
         return user
     ```

     En este caso:
     - Si se envían datos incompletos o con tipos incorrectos, FastAPI devuelve automáticamente un error 422.

3. **Validación personalizada**:
   Puedes agregar validaciones adicionales usando decoradores o métodos de Pydantic.

   - **Validador personalizado**:
     ```python
     from pydantic import BaseModel, validator

     class User(BaseModel):
         name: str
         age: int

         @validator("age")
         def check_age(cls, age):
             if age < 18:
                 raise ValueError("Age must be 18 or above.")
             return age
     ```

4. **Anidamiento de modelos**:
   Los modelos pueden contener otros modelos, permitiendo estructurar datos complejos.

   ```python
   class Address(BaseModel):
       city: str
       country: str

   class User(BaseModel):
       id: int
       name: str
       address: Address
   ```

   Entrada JSON esperada:
   ```json
   {
       "id": 1,
       "name": "Mario",
       "address": {
           "city": "Bogotá",
           "country": "Colombia"
       }
   }
   ```

5. **Modelos de respuesta**:
   Puedes definir qué datos se devuelven como respuesta.

   ```python
   @app.post("/users/", response_model=User)
   async def create_user(user: User):
       return user
   ```

   Esto asegura que solo se devuelvan los campos definidos en el modelo de respuesta, incluso si el objeto tiene más datos.

6. **Tipos avanzados**:
   Pydantic soporta validación para:
   - Listas y diccionarios (`List`, `Dict`).
   - Datos opcionales (`Optional`).
   - Estructuras complejas (como `Union`).

   ```python
   from typing import List, Optional

   class Item(BaseModel):
       name: str
       tags: Optional[List[str]] = None
   ```

### Beneficios del modelado con Pydantic en FastAPI
- **Validación automática y clara**: Reduce errores manuales.
- **Código legible y mantenible**: Uso de Python estándar.
- **Manejo robusto de errores**: Mensajes de error claros para entradas inválidas.
- **Generación automática de documentación**: Los modelos son reflejados en la documentación interactiva (Swagger).

### Resumen

Para diseñar una API robusta, es esencial modelar correctamente los datos, especialmente al crear nuevos modelos que organicen y relacionen la información eficientemente. En esta guía, exploraremos cómo crear modelos en FastAPI para estructurar datos, conectar modelos y optimizar la funcionalidad de nuestra API.

### ¿Cómo crear y organizar modelos en FastAPI?

Al crear una API, tener todos los modelos en un archivo separado, como `models.py`, ayuda a evitar el “código espagueti” y mantiene el código modular y ordenado. FastAPI no exige esta organización, pero es una buena práctica. Primero, copiamos el modelo de `customer` y el BaseModel de FastAPI desde el archivo principal (`main.py`) y los pegamos en `models.py`.

### ¿Cómo construir un modelo de transacción?

Para estructurar las transacciones en nuestra API, creamos el modelo `Transaction` en `models.py`, derivado de `BaseModel`. Incluimos los siguientes campos:

- `id`: un identificador único, de tipo entero.
- `amount`: un entero en vez de `float` para representar valores financieros, evitando problemas de precisión.
- `description`: un campo de tipo `str`, obligatorio para describir la transacción.

Esta estructura permite gestionar las transacciones de manera segura y clara.

### ¿Cómo construir un modelo de factura y conectar los modelos?

El modelo `Invoice` representa una factura y también hereda de BaseModel. Además de un id, el modelo Invoice conecta los datos al incluir:

- customer: un campo de tipo `Customer`, que enlaza la factura con el cliente correspondiente.
- `transactions`: una lista de `Transaction` que contiene todas las transacciones asociadas a la factura.

Para indicar que `transactions` es una lista, usamos el tipo `List` con `Transaction`, lo que permite que FastAPI gestione adecuadamente este arreglo de datos.

### ¿Cómo calcular el total de una factura?

Para calcular el total de las transacciones en una factura, se agrega un método `total` en el modelo `Invoice`. Este método:

- Utiliza un decorador `@property`, haciendo que se acceda como un atributo.
- Suma el campo `amount` de cada `Transaction` en la lista `transactions` mediante una comprensión de listas, retornando el total.

De esta forma, el total se calcula automáticamente, y se puede utilizar para generar el subtotal al momento de consultar el endpoint de facturación.

### ¿Cómo usar los modelos en los endpoints de la API?

Los modelos se importan al archivo principal (`main.py`), y se crean nuevos endpoints para gestionar `transactions` e `invoices`. FastAPI genera automáticamente la documentación de estos endpoints, mostrando los datos necesarios para crear facturas con un cliente y una lista de transacciones, simplificando el uso de la API para el usuario final.

### ¿Cómo validar tipos de datos con Pydantic?

Utilizar Pydantic para definir los tipos de datos en nuestros modelos permite validaciones automáticas y mensajes de error claros en caso de datos incorrectos. Por ejemplo, el campo `email` en el modelo `customer` se puede validar agregando el tipo `EmailStr`, lo que permite a la API detectar y rechazar correos no válidos.

## Validación y Gestión de Modelos en FastAPI

La **validación y gestión de modelos en FastAPI** se centra en usar las capacidades de **Pydantic** para garantizar que los datos enviados a la API cumplan con los requisitos esperados. Esto incluye definir estructuras claras, realizar validaciones avanzadas y manejar respuestas correctamente.

## **1. Definición de Modelos**

Los modelos en FastAPI se basan en `BaseModel` de **Pydantic** y son el núcleo de la validación de datos.

### **Ejemplo básico:**

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    age: int
    is_active: bool = True  # Valor por defecto
```

### **Características clave:**
- Cada campo tiene un tipo explícito.
- Los valores predeterminados son opcionales.
- Se valida automáticamente al usarlo en la API.

## **2. Validación Automática**

FastAPI valida automáticamente los datos enviados a través de:
- **Cuerpo de la solicitud (`request body`)**:
  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel

  app = FastAPI()

  class User(BaseModel):
      id: int
      name: str
      email: str

  @app.post("/users/")
  async def create_user(user: User):
      return {"message": "User created", "user": user}
  ```

  Si el cliente envía un dato no válido, FastAPI responde con un error HTTP **422 Unprocessable Entity**.

- **Parámetros de consulta**:
  ```python
  @app.get("/users/")
  async def get_user(limit: int = 10, active: bool = True):
      return {"limit": limit, "active": active}
  ```

## **3. Validación Personalizada**

Puedes agregar validaciones avanzadas con los validadores de Pydantic.

### **Uso de decoradores `@validator`:**

```python
from pydantic import BaseModel, validator

class User(BaseModel):
    name: str
    age: int

    @validator("age")
    def check_age(cls, value):
        if value < 18:
            raise ValueError("Age must be 18 or above.")
        return value
```

## **4. Respuestas Controladas con Modelos**

Los modelos también pueden usarse para controlar las respuestas de la API.

### **Modelo de respuesta:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str

@app.get("/users/{user_id}", response_model=User)
async def read_user(user_id: int):
    return {"id": user_id, "name": "Mario", "email": "mario@example.com"}
```

En este ejemplo:
- El cliente solo verá los campos definidos en el modelo `User`, incluso si el objeto tiene más datos.

## **5. Estructuras de Datos Complejas**

### **Anidamiento de modelos:**
Los modelos pueden contener otros modelos para manejar datos más complejos.

```python
class Address(BaseModel):
    city: str
    country: str

class User(BaseModel):
    id: int
    name: str
    address: Address
```

Entrada JSON esperada:
```json
{
    "id": 1,
    "name": "Mario",
    "address": {
        "city": "Bogotá",
        "country": "Colombia"
    }
}
```

## **6. Validación de Tipos Compuestos**

FastAPI admite tipos avanzados como:
- **Listas y diccionarios**:
  ```python
  from typing import List

  class Item(BaseModel):
      name: str
      tags: List[str]
  ```

- **Datos opcionales (`Optional`)**:
  ```python
  from typing import Optional

  class User(BaseModel):
      id: int
      name: Optional[str] = None
  ```

## **7. Manejo de Errores Personalizados**

Puedes personalizar los errores de validación usando excepciones.

### **Ejemplo con `HTTPException`:**
```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/users/{user_id}")
async def read_user(user_id: int):
    if user_id < 0:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return {"user_id": user_id}
```

## **8. Documentación Automática**

FastAPI genera automáticamente documentación interactiva basada en los modelos y validaciones.

Accede a:
- **Swagger UI**: `/docs`
- **Redoc**: `/redoc`

## **9. Ejemplo Completo**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List

app = FastAPI()

class Address(BaseModel):
    city: str
    country: str

class User(BaseModel):
    id: int
    name: str
    email: str
    age: int
    address: Address
    hobbies: List[str] = []

    @validator("age")
    def validate_age(cls, value):
        if value < 18:
            raise ValueError("User must be at least 18 years old.")
        return value

@app.post("/users/", response_model=User)
async def create_user(user: User):
    return user
```

### **Prueba:**
Entrada:
```json
{
    "id": 1,
    "name": "Mario",
    "email": "mario@example.com",
    "age": 20,
    "address": {
        "city": "Bogotá",
        "country": "Colombia"
    },
    "hobbies": ["coding", "reading"]
}
```

Salida:
```json
{
    "id": 1,
    "name": "Mario",
    "email": "mario@example.com",
    "age": 20,
    "address": {
        "city": "Bogotá",
        "country": "Colombia"
    },
    "hobbies": ["coding", "reading"]
}
```

### Recursos

La validación de datos y la gestión de modelos en FastAPI permite crear endpoints seguros y eficientes. En este ejemplo, mostramos cómo manejar un modelo para recibir y devolver datos sin exponer identificadores innecesarios. Partimos del modelo `Customer` y desarrollamos un modelo específico para la creación, `CustomerCreate`, que omite el ID para que sea generado automáticamente en el backend.

### ¿Cómo configuramos los modelos para crear un nuevo cliente sin ID?

Para evitar enviar un ID manualmente, creamos `CustomerCreate`, que hereda de `Customer` pero excluye el ID, dejándolo en blanco hasta que se complete la validación. Esto es útil porque:

- El ID se asigna automáticamente en la base de datos o mediante código en memoria.
- Evitamos exposición de datos sensibles innecesarios en las solicitudes.

### ¿Cómo gestionamos la validación y asignación de ID en el backend?

FastAPI permite validar datos mediante modelos y gestionar IDs sin base de datos:

- Se usa una variable `current_id` inicializada en 0 que se incrementa por cada nuevo registro.
- Los datos recibidos son validados y convertidos a diccionario (`model.dict()`), creando una entrada limpia y sin errores.
- En un entorno asincrónico, no se recomienda incrementar `current_id` de forma manual, por lo que una lista simula la base de datos en memoria, donde el ID es el índice del elemento.

### ¿Cómo configuramos un endpoint para listar clientes?

El endpoint `GET` permite visualizar todos los clientes registrados:

- Definimos un modelo `List[Customer]` como response_model para mostrar un JSON con los clientes.
- FastAPI convierte automáticamente la lista de `Customer` a un JSON, haciéndola accesible desde la documentación.

### ¿Qué ocurre al crear un nuevo cliente en memoria?

Dado que estamos trabajando en memoria:

- Los datos se borran al reiniciar el servidor.
- Para cada cliente creado, asignamos un ID basado en el índice de la lista, simulando el autoincremento de una base de datos real.

### ¿Cómo crear un endpoint para obtener un cliente específico por ID?

Finalmente, para acceder a un cliente específico, añadimos un nuevo endpoint que recibe el ID en la URL:

- Este endpoint busca en la lista por ID y devuelve el cliente en formato JSON.
- Si el cliente no existe, FastAPI devuelve un error, protegiendo la integridad de los datos.

## ¿Cómo conectar FastAPI a una base de datos usando SQLModel?

Para conectar **FastAPI** a una base de datos utilizando **SQLModel**, puedes seguir estos pasos. **SQLModel** combina lo mejor de **SQLAlchemy** y **Pydantic**, permitiendo trabajar con modelos como clases Pydantic mientras administra una base de datos relacional.

### Pasos para conectar FastAPI a una base de datos con SQLModel:

1. **Instalar las dependencias necesarias**:
   ```bash
   pip install fastapi sqlmodel uvicorn sqlite
   ```

2. **Configurar los modelos de datos con SQLModel**:
   Define las tablas de la base de datos utilizando clases que heredan de `SQLModel`.

3. **Configurar la conexión a la base de datos**:
   Utiliza una base de datos como SQLite (ideal para desarrollo) o conecta a un motor como PostgreSQL o MySQL en producción.

4. **Crear los endpoints necesarios**:
   Utiliza las operaciones CRUD para interactuar con la base de datos a través de FastAPI.

### Ejemplo práctico: Gestión de clientes

#### 1. **Definir los modelos de datos**
```python
from sqlmodel import SQLModel, Field
from typing import Optional

class Customer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    age: Optional[int]
```

En este modelo:
- `table=True` indica que la clase corresponde a una tabla en la base de datos.
- El campo `id` es una clave primaria.

#### 2. **Configurar la conexión a la base de datos**
```python
from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = "sqlite:///./test.db"  # Cambia la URL según el motor de tu base de datos
engine = create_engine(DATABASE_URL, echo=True)

# Crear todas las tablas definidas en los modelos
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
```

#### 3. **Crear los endpoints en FastAPI**
```python
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Session, select

app = FastAPI()

# Crear la base de datos al iniciar la aplicación
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Dependency para manejar sesiones
def get_session():
    with Session(engine) as session:
        yield session

@app.post("/customers/", response_model=Customer)
def create_customer(customer: Customer, session: Session = Depends(get_session)):
    session.add(customer)
    session.commit()
    session.refresh(customer)
    return customer

@app.get("/customers/", response_model=list[Customer])
def list_customers(session: Session = Depends(get_session)):
    statement = select(Customer)
    results = session.exec(statement).all()
    return results

@app.get("/customers/{customer_id}", response_model=Customer)
def get_customer(customer_id: int, session: Session = Depends(get_session)):
    customer = session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: int, session: Session = Depends(get_session)):
    customer = session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    session.delete(customer)
    session.commit()
    return {"message": "Customer deleted"}
```

### 4. **Probar la aplicación**
Inicia el servidor con:
```bash
uvicorn main:app --reload
```

#### Pruebas con rutas:
1. **Crear un cliente**:
   - `POST /customers/`
   ```json
   {
       "name": "Luis",
       "email": "luis@example.com",
       "age": 30
   }
   ```

2. **Listar todos los clientes**:
   - `GET /customers/`

3. **Obtener un cliente por ID**:
   - `GET /customers/1`

4. **Eliminar un cliente**:
   - `DELETE /customers/1`

### 5. **Cambiar a otro motor de base de datos**
Para usar motores como PostgreSQL o MySQL, cambia `DATABASE_URL` a algo como:
- PostgreSQL:
  ```python
  DATABASE_URL = "postgresql://user:password@localhost/dbname"
  ```
- MySQL:
  ```python
  DATABASE_URL = "mysql+pymysql://user:password@localhost/dbname"
  ```

Asegúrate de instalar el controlador correspondiente:
```bash
pip install psycopg2  # Para PostgreSQL
pip install pymysql   # Para MySQL
```

### Beneficios de usar SQLModel con FastAPI:
1. **Simplicidad**: Combina SQLAlchemy y Pydantic en un único modelo.
2. **Flexibilidad**: Fácil de escalar a bases de datos complejas.
3. **Integración nativa con FastAPI**: Usa modelos SQLModel directamente en los endpoints.

### Resumen

Para conectar FastAPI con una base de datos real, primero configuraremos una base de datos SQLite utilizando la librería SQLModel, que facilita la integración sin necesidad de escribir SQL. SQLModel combina Pydantic y SQLAlchemy, permitiendo que nuestros modelos se almacenen directamente en bases de datos con una sintaxis simplificada.

### ¿Cómo instalar y configurar SQLModel?

1. I**nstalación**: Abre la terminal y ejecuta:

`pip install sqlmodel`

También es recomendable registrar las dependencias en un archivo `requirements.txt`, como SQLModel y FastAPI con sus respectivas versiones. Esto ayuda a instalar todas las dependencias en otros entornos fácilmente.

2. Creación del archivo de configuración:

 - Crea un archivo `db.py`.
 - Importa las clases `Session` y `create_engine` de SQLModel para gestionar la conexión.
 - Define las variables para la conexión, como la URL de la base de datos, en este caso `sqlite:///database_name.db`.
 
3. Creación del `engine`:

 - Utiliza create_engine con la URL de la base de datos para crear el motor que gestionará las sesiones.

### ¿Cómo definir la sesión para la base de datos?

Para manejar las conexiones, define una función `get_session` en `db.py`, la cual:

- Crea un contexto que inicia y cierra la sesión automáticamente.
- Facilita el uso de la sesión en varios endpoints de FastAPI.

### ¿Cómo registrar la sesión como dependencia en FastAPI?

Para que FastAPI use la sesión en sus endpoints:

- Importa `Depends` de FastAPI.
- Define una dependencia que gestione la sesión mediante `get_session`, facilitando el acceso a la base de datos desde cualquier endpoint.

### ¿Cómo adaptar los modelos para almacenar datos en la base de datos?

1. Modificación del modelo: Si usas modelos de Pydantic, ajústalos para heredar de `SQLModel` en vez de `BaseModel`. Esto conecta los modelos con la base de datos.

2. Creación de tablas:

 - En el modelo que representa una tabla, añade `table=True` para que SQLModel cree automáticamente la tabla en la base de datos.
 - Hereda los atributos comunes de un modelo base (sin` table=True`) para evitar duplicaciones y asegurar que se incluyan todos los campos necesarios.
 
3. Ejemplo de implementación:

 - Define un modelo CustomerBase para los datos comunes.
 - Crea un modelo Customer, que herede de CustomerBase y de SQLModel, con table=True para almacenar los registros en la tabla correspondiente.
 
Con esta configuración, los datos se insertarán en la base de datos sin necesidad de escribir SQL directamente. Al definir la dependencia de sesión en FastAPI, cada vez que un endpoint la requiera, FastAPI se conectará a la base de datos automáticamente.

**Lecturas recomendadas**

[SQLModel](https://sqlmodel.tiangolo.com/)v)

[sqlite3 — DB-API 2.0 interface for SQLite databases — Python 3.13.0 documentation](https://docs.python.org/3/library/sqlite3.html)

[FastAPI](https://fastapi.tiangolo.com/tutorial/sql-databases/)

## Creación y consulta de registros en SQLite con SQLModel y FastAPI

Aquí tienes un ejemplo funcional de cómo crear y consultar registros en una base de datos SQLite utilizando **SQLModel** y **FastAPI**. Este ejemplo incluye la creación de una tabla, la inserción de registros y la consulta de datos.

---

### **Requisitos Previos**
Asegúrate de instalar los paquetes necesarios:
```bash
pip install fastapi uvicorn sqlmodel
```

---

### **Código Completo**
```python
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Optional, List

# Configuración de la base de datos SQLite
sqlite_file_name = "database.sqlite"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True)

# Crear instancia de la aplicación FastAPI
app = FastAPI()

# Modelo de la tabla
class Customer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str

# Crear las tablas en la base de datos
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Dependencia para la sesión de base de datos
def get_session():
    with Session(engine) as session:
        yield session

# Crear un nuevo cliente
@app.post("/customers/", response_model=Customer)
async def create_customer(customer: Customer, session: Session = Depends(get_session)):
    session.add(customer)
    session.commit()
    session.refresh(customer)  # Refrescar para obtener el ID generado
    return customer

# Listar todos los clientes
@app.get("/customers/", response_model=List[Customer])
async def list_customers(session: Session = Depends(get_session)):
    statement = select(Customer)
    results = session.exec(statement).all()
    return results

# Obtener un cliente por ID
@app.get("/customers/{customer_id}", response_model=Customer)
async def get_customer(customer_id: int, session: Session = Depends(get_session)):
    customer = session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer
```

---

### **Explicación del Código**
1. **Modelo `Customer`:**
   - Define la estructura de la tabla con SQLModel. 
   - El campo `id` es la clave primaria, y es autoincremental.

2. **Configuración de SQLite:**
   - Usa `create_engine` para configurar la conexión a la base de datos SQLite.
   - Se crean automáticamente las tablas al iniciar la aplicación con `SQLModel.metadata.create_all(engine)`.

3. **Dependencia `get_session`:**
   - Proporciona una sesión de base de datos en cada solicitud.

4. **Endpoints:**
   - **`POST /customers/`:** Inserta un nuevo registro en la tabla `Customer`.
   - **`GET /customers/`:** Devuelve todos los registros de la tabla.
   - **`GET /customers/{customer_id}`:** Devuelve un registro específico según su `id`.

---

### **Pruebas**
1. **Ejecutar el Servidor**
   Usa el siguiente comando para iniciar el servidor FastAPI:
   ```bash
   uvicorn main:app --reload
   ```

2. **Crear un Cliente**
   Usa una herramienta como `curl`, `Postman` o `httpie`:
   ```bash
   curl -X POST "http://127.0.0.1:8000/customers/" \
   -H "Content-Type: application/json" \
   -d '{"name": "Luis", "email": "luis@example.com"}'
   ```

   Respuesta esperada:
   ```json
   {
       "id": 1,
       "name": "Luis",
       "email": "luis@example.com"
   }
   ```

3. **Listar Clientes**
   ```bash
   curl -X GET "http://127.0.0.1:8000/customers/"
   ```
   Respuesta esperada:
   ```json
   [
       {
           "id": 1,
           "name": "Luis",
           "email": "luis@example.com"
       }
   ]
   ```

4. **Consultar Cliente por ID**
   ```bash
   curl -X GET "http://127.0.0.1:8000/customers/1"
   ```
   Respuesta esperada:
   ```json
   {
       "id": 1,
       "name": "Luis",
       "email": "luis@example.com"
   }
   ```

5. **Errores:**
   - Si intentas obtener un cliente que no existe:
     ```bash
     curl -X GET "http://127.0.0.1:8000/customers/999"
     ```
     Respuesta esperada:
     ```json
     {
         "detail": "Customer not found"
     }
     ```

### Resumen

Implementar SQLModel en una aplicación requiere precisión, especialmente en la creación y manipulación de tablas y registros. Aquí te explicamos cómo gestionar correctamente las claves primarias y la creación de tablas, además de cómo integrar los endpoints de creación y consulta en FastAPI.

### ¿Por qué ocurre el error de primary key?

El error de “no se encuentra ninguna primary key” surge porque al implementar SQLModel, los campos como el ID aún no están configurados como clave primaria. Para resolver esto:

- Define el campo ID como primary_key=True.
- Configura un valor por defecto, por ejemplo `None`, para que SQLModel pueda gestionar el incremento automático de la clave primaria en la base de datos.

### ¿Cómo configurar los campos sin primary key para almacenarlos en la base de datos?

Para que cada campo sea almacenado en la base de datos, es necesario:

- Agregar la clase `Field` en cada campo sin primary key.
- Definir `default=None` en cada campo para que SQLModel los incluya en la estructura de la tabla.
- Asegúrate de que cualquier campo sin `Field` o configuración adicional no se guardará en la base de datos.

### ¿Cómo crear las tablas en SQLite?
Si aún no ves el archivo `.sqlite3` en tu directorio, es porque las tablas no se han creado. Para resolverlo:

- Crea una función `create_all_tables` que reciba la instancia de FastAPI como parámetro.
- Utiliza `SQLModel.metadata.create_all` junto con el motor de base de datos configurado para generar las tablas.
- Ejecuta esta función al iniciar la aplicación en el archivo `main.py`.

Este proceso generará automáticamente el archivo de base de datos SQLite y las tablas definidas en el modelo.

### ¿Cómo integrar el endpoint de creación de customers?

Para gestionar el endpoint de creación de customers y almacenar registros en la base de datos:

- Usa `session.add(customer)` para agregar el nuevo customer en la sesión de base de datos.
- Ejecuta `session.commit()` para guardar los cambios de manera definitiva.
- Llama a `session.refresh(customer)` para actualizar el objeto en memoria con el ID generado por la base de datos.

Esto garantiza que cada registro creado tenga su ID único asignado directamente desde la base de datos.

### ¿Cómo consultar los customers desde la base de datos?

Para listar todos los customers desde la base de datos y retornar los resultados en JSON:

- Importa `select` de SQLModel.
- Usa `session.execute(select(Customer)).all()` para obtener todos los registros de tipo customer.
- Devuelve esta lista de customers directamente en la respuesta del endpoint.

### ¿Cómo crear un endpoint para obtener un customer por ID?

Como reto, implementa un endpoint que permita obtener un customer específico según su ID. Recuerda:

- Usa la sesión para ejecutar la consulta SQL.
- Considera los casos donde el customer solicitado no exista, devolviendo un mensaje adecuado o un código de error si es necesario.

## Crear un CRUD básico en FastAPI: Eliminar

Aquí tienes un ejemplo de cómo agregar la funcionalidad para **eliminar registros** en un CRUD básico con **FastAPI** y **SQLModel**:

---

### **Código para CRUD Completo con Eliminar**

```python
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Optional, List

# Configuración de la base de datos SQLite
sqlite_file_name = "database.sqlite"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True)

# Crear instancia de la aplicación FastAPI
app = FastAPI()

# Modelo de la tabla
class Customer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str

# Crear las tablas en la base de datos
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Dependencia para la sesión de base de datos
def get_session():
    with Session(engine) as session:
        yield session

# Crear un cliente
@app.post("/customers/", response_model=Customer)
async def create_customer(customer: Customer, session: Session = Depends(get_session)):
    session.add(customer)
    session.commit()
    session.refresh(customer)
    return customer

# Listar todos los clientes
@app.get("/customers/", response_model=List[Customer])
async def list_customers(session: Session = Depends(get_session)):
    statement = select(Customer)
    results = session.exec(statement).all()
    return results

# Obtener un cliente por ID
@app.get("/customers/{customer_id}", response_model=Customer)
async def get_customer(customer_id: int, session: Session = Depends(get_session)):
    customer = session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

# Eliminar un cliente por ID
@app.delete("/customers/{customer_id}", response_model=dict)
async def delete_customer(customer_id: int, session: Session = Depends(get_session)):
    customer = session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    session.delete(customer)
    session.commit()
    return {"message": f"Customer with id {customer_id} deleted successfully"}
```

---

### **Explicación del Endpoint para Eliminar**
1. **`@app.delete("/customers/{customer_id}")`**:
   - Este endpoint recibe el `customer_id` como parámetro de la ruta.
   - Verifica si el cliente existe en la base de datos usando `session.get(Customer, customer_id)`.
   - Si no existe, lanza un error `404` con el mensaje `Customer not found`.
   - Si existe, lo elimina usando `session.delete(customer)` seguido de `session.commit()`.

2. **Respuesta**:
   - Devuelve un mensaje de confirmación indicando que el cliente fue eliminado correctamente.

---

### **Pruebas**

1. **Crear un Cliente**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/customers/" \
   -H "Content-Type: application/json" \
   -d '{"name": "Luis", "email": "luis@example.com"}'
   ```

2. **Eliminar un Cliente**:
   ```bash
   curl -X DELETE "http://127.0.0.1:8000/customers/1"
   ```
   Respuesta esperada:
   ```json
   {
       "message": "Customer with id 1 deleted successfully"
   }
   ```

3. **Intentar Eliminar un Cliente No Existente**:
   ```bash
   curl -X DELETE "http://127.0.0.1:8000/customers/999"
   ```
   Respuesta esperada:
   ```json
   {
       "detail": "Customer not found"
   }
   ```

---

### **Conclusión**

Este código demuestra cómo manejar la funcionalidad de eliminación en un CRUD básico con FastAPI y SQLModel. Puedes integrar esta funcionalidad con las operaciones de creación, lectura y actualización para construir una API completa y funcional. 🚀

### Resumen

Crear y gestionar endpoints es fundamental para un CRUD completo en FastAPI. Aquí se explora cómo implementar la funcionalidad de detalle, borrado y actualización de clientes en una API. A lo largo de este proceso, se observa cómo manejar la consulta, validación y actualización de datos en la base de datos de manera robusta.

### ¿Cómo obtener el detalle de un cliente?

Para obtener el detalle de un cliente, es esencial recibir el `customer_id` y utilizar la sesión de la base de datos. Aquí se plantea una estructura básica:

- **Declaración de Ruta**: Usamos app.get("/customer/{customer_id}") para definir la URL, en la que {customer_id} permite identificar el cliente específico.
- **Consulta en la Base de Datos**: La sesión de la base de datos ejecuta get con el modelo y el customer_id.
- **Manejo de Errores**: Si el cliente no existe, se lanza una excepción HTTPException con un código 404 Not Found y un mensaje personalizado, como Customer does not exist, asegurando claridad en la respuesta JSON.

Esta estructura asegura que el cliente pueda consultarse de manera efectiva y que, en caso de no hallarse, el usuario reciba un mensaje adecuado.

### ¿Cómo borrar un cliente en FastAPI?

Para implementar el endpoint de borrado, se reutiliza en gran medida el código de consulta:

- **Ruta de Borrado**: Cambiamos el método a `delete` para el endpoint `"/customer/{customer_id}"`, respetando la estructura previa.
- **Eliminación de Cliente**: Una vez localizado el cliente, se usa `session.delete()` con el objeto correspondiente.
- **Confirmación de Eliminació**n: Finalizamos con `session.commit()` para confirmar el cambio en la base de datos. La respuesta JSON debe incluir un mensaje como `{"detail": "Customer deleted successfully"}` para asegurar al usuario que la eliminación se realizó con éxito.

Este proceso garantiza que el cliente se elimine completamente de la base de datos solo si existe.

### ¿Cómo actualizar un cliente en FastAPI?

Para completar el CRUD, solo falta el endpoint de actualización. Este implica recibir un cuerpo (`body`) con los datos actualizados:

- **Crear la Ruta y Función**: Define un nuevo endpoint put o patch para actualizar. Se recibe el customer_id y el objeto con los nuevos datos, excluyendo el ID.
- **Validación y Actualización**: Como en los otros endpoints, verifica que el cliente exista. Luego, actualiza los campos necesarios en el objeto del cliente.
- **Confirmación de Actualización**: Usa session.commit() para guardar los cambios.

Este endpoint permite modificar los datos del cliente según lo necesite el usuario, proporcionando flexibilidad y manteniendo la integridad de los datos en la base de datos.

## Crear un CRUD básico en FastAPI: Actualizar

Aquí tienes un ejemplo de cómo agregar la funcionalidad para **actualizar registros** en un CRUD básico con **FastAPI** y **SQLModel**.

### **Código para CRUD Completo con Actualizar**

```python
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Optional, List

# Configuración de la base de datos SQLite
sqlite_file_name = "database.sqlite"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True)

# Crear instancia de la aplicación FastAPI
app = FastAPI()

# Modelo de la tabla
class Customer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str

class CustomerUpdate(SQLModel):
    name: Optional[str]
    email: Optional[str]

# Crear las tablas en la base de datos
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Dependencia para la sesión de base de datos
def get_session():
    with Session(engine) as session:
        yield session

# Crear un cliente
@app.post("/customers/", response_model=Customer)
async def create_customer(customer: Customer, session: Session = Depends(get_session)):
    session.add(customer)
    session.commit()
    session.refresh(customer)
    return customer

# Listar todos los clientes
@app.get("/customers/", response_model=List[Customer])
async def list_customers(session: Session = Depends(get_session)):
    statement = select(Customer)
    results = session.exec(statement).all()
    return results

# Obtener un cliente por ID
@app.get("/customers/{customer_id}", response_model=Customer)
async def get_customer(customer_id: int, session: Session = Depends(get_session)):
    customer = session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

# Actualizar un cliente por ID
@app.patch("/customers/{customer_id}", response_model=Customer)
async def update_customer(
    customer_id: int, 
    customer_data: CustomerUpdate, 
    session: Session = Depends(get_session)
):
    customer = session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Actualizar campos si se proporcionan
    customer_data_dict = customer_data.dict(exclude_unset=True)
    for key, value in customer_data_dict.items():
        setattr(customer, key, value)
    
    session.add(customer)
    session.commit()
    session.refresh(customer)
    return customer
```

### **Explicación del Endpoint para Actualizar**

1. **`@app.patch("/customers/{customer_id}")`**:
   - Este endpoint recibe el `customer_id` como parámetro de la ruta y un modelo `CustomerUpdate` con los datos a actualizar.
   - Si el cliente no existe, se lanza un error `404` con el mensaje `Customer not found`.
   - Los campos del modelo `CustomerUpdate` que no se envían se omiten con `exclude_unset=True`.
   - Los valores existentes se actualizan utilizando `setattr`.

2. **Modelo `CustomerUpdate`**:
   - Define los campos que son opcionales para actualizar (`name` y `email`).
   - Esto permite que solo los campos enviados sean modificados.

3. **Actualización en la Base de Datos**:
   - Se agrega el objeto actualizado de nuevo a la sesión (`session.add(customer)`).
   - Se confirman los cambios con `session.commit()`.
   - Se actualiza el objeto en la sesión con `session.refresh(customer)`.

### **Pruebas**

1. **Crear un Cliente**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/customers/" \
   -H "Content-Type: application/json" \
   -d '{"name": "Luis", "email": "luis@example.com"}'
   ```

   Respuesta:
   ```json
   {
       "id": 1,
       "name": "Luis",
       "email": "luis@example.com"
   }
   ```

2. **Actualizar un Cliente**:
   ```bash
   curl -X PATCH "http://127.0.0.1:8000/customers/1" \
   -H "Content-Type: application/json" \
   -d '{"email": "luis_updated@example.com"}'
   ```

   Respuesta:
   ```json
   {
       "id": 1,
       "name": "Luis",
       "email": "luis_updated@example.com"
   }
   ```

3. **Intentar Actualizar un Cliente No Existente**:
   ```bash
   curl -X PATCH "http://127.0.0.1:8000/customers/999" \
   -H "Content-Type: application/json" \
   -d '{"name": "Carlos"}'
   ```

   Respuesta:
   ```json
   {
       "detail": "Customer not found"
   }
   ```

---

### **Conclusión**

Este ejemplo muestra cómo implementar la funcionalidad de actualización en un CRUD básico con **FastAPI** y **SQLModel**, manejando tanto actualizaciones parciales como completas. 🚀

## Arquitectura de APIs escalables en FastAPI

**Arquitectura de APIs escalables en FastAPI** implica diseñar y estructurar tu aplicación para manejar un crecimiento en la cantidad de usuarios, datos y peticiones. Aquí tienes una guía práctica para lograrlo:

### **1. Organización del Proyecto**
Organiza tu código en módulos o capas para que sea mantenible y escalable:
- **Routers:** Divide tus rutas por funcionalidad en módulos separados (`users.py`, `products.py`, etc.) y agrégalas al proyecto principal.
- **Models:** Define tus modelos (SQLModel o Pydantic) en archivos específicos.
- **Services:** Implementa lógica de negocio en servicios separados.
- **Dependencies:** Utiliza dependencias de FastAPI para manejar configuraciones reutilizables (como bases de datos o autenticación).

Ejemplo de estructura:
```
app/
├── main.py
├── routers/
│   ├── users.py
│   ├── products.py
├── models/
│   ├── user.py
│   ├── product.py
├── services/
│   ├── user_service.py
│   ├── product_service.py
├── db/
│   ├── database.py
│   ├── migrations/
```

### **2. Base de Datos**
- **ORM Escalable:** Utiliza un ORM como SQLModel, SQLAlchemy o Tortoise ORM para manejar bases de datos relacionales.
- **Migraciones:** Integra herramientas como `Alembic` para gestionar cambios en los esquemas de la base de datos.
- **Conexión eficiente:** Configura un pool de conexiones usando `asyncpg` o conexiones síncronas bien gestionadas.

### **3. Rutas y Dependencias**
- **Modularización:** Usa routers con prefijos y dependencias específicas para cada módulo.
- **Inyección de Dependencias:** Utiliza `Depends` para manejar configuraciones como bases de datos, autenticación o validación de roles.

Ejemplo:
```python
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
async def list_users():
    return {"message": "List of users"}
```

### **4. Escalabilidad Horizontal**
Implementa mecanismos para soportar múltiples instancias de tu API:
- **Uvicorn Workers:** Ejecuta varios trabajadores con `uvicorn` o `gunicorn`.
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
  ```
- **Balanceadores de carga:** Usa Nginx o servicios como AWS Elastic Load Balancer.

### **5. Cache**
Integra un sistema de cache como Redis para reducir la carga en la base de datos:
- **Resultados de consultas:** Guarda respuestas frecuentes.
- **Tokens de autenticación:** Usa Redis para almacenar tokens de sesión.
  
Ejemplo con `aioredis`:
```python
import aioredis

redis = aioredis.from_url("redis://localhost")
```

### **6. Middleware**
- **Logging:** Agrega un middleware para registrar peticiones y respuestas.
- **Autenticación:** Implementa autenticación con OAuth2 o JWT.
- **CORS:** Configura middleware para habilitar solicitudes desde otros dominios.

Ejemplo:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### **7. Pruebas**
Escribe pruebas unitarias y de integración para garantizar la calidad:
- **Pytest:** Usa pytest para probar rutas y lógica de negocio.
- **Test de carga:** Usa herramientas como Locust o Artillery para simular alto tráfico.

---

### **8. Monitoreo y Observabilidad**
- **Logs:** Usa herramientas como `Loguru` o servicios como Datadog.
- **Métricas:** Integra Prometheus y Grafana para monitorear el rendimiento.
- **Traza de peticiones:** Implementa OpenTelemetry.

### **9. Escalabilidad en Infraestructura**
- **Contenedores:** Dockeriza tu aplicación para facilitar la implementación.
- **Orquestación:** Usa Kubernetes para gestionar múltiples contenedores.
- **Serverless:** Considera AWS Lambda o Google Cloud Functions para servicios específicos.

---

### **10. Seguridad**
- Usa HTTPS en todas las solicitudes.
- Valida y sanitiza datos de entrada.
- Configura políticas de seguridad en CORS y cabeceras.

### **Ejemplo Integrado**
```python
from fastapi import FastAPI
from app.routers import users, products
from app.db.database import create_db_and_tables

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

app.include_router(users.router)
app.include_router(products.router)
```

Con esta arquitectura, tu API en FastAPI estará lista para manejar tanto un desarrollo inicial como un crecimiento significativo.

### Resumen

La organización de endpoints en una aplicación es crucial para mantener el código limpio y fácil de escalar. Cuando todos los endpoints están en un solo archivo, se vuelve complicado de manejar y actualizar, especialmente en aplicaciones robustas. Para evitar esto, FastAPI permite estructurar la aplicación con un enfoque modular, utilizando routers y etiquetado, facilitando su mantenimiento.

### ¿Por qué dividir los endpoints en múltiples archivos?

Tener todos los endpoints en un solo archivo lleva a un código excesivamente largo y difícil de modificar. Dividir los endpoints en archivos separados permite:

- Mantener una estructura limpia y organizada.
- Facilitar la búsqueda y edición de endpoints específicos.
- Aumentar la claridad y reducir los errores al trabajar en el código.

### ¿Cómo configurar la estructura de archivos en FastAPI?

FastAPI recomienda crear una carpeta principal para la aplicación. Dentro de esta carpeta, se agregan varios archivos y subcarpetas:

- **__init__.py**: convierte la carpeta en un módulo, permitiendo importar archivos de forma más sencilla.
- **main.py**: contiene la configuración principal de la aplicación, incluyendo las dependencias y la raíz de la aplicación.
- **dependencies.py**: archivo para definir dependencias comunes a varios endpoints.
- **routers/**: subcarpeta que agrupa endpoints específicos, como customer, transactions, e invoices.

Otros archivos recomendados incluyen `db.py` para la configuración de la base de datos y `models.py` para los modelos de datos. En aplicaciones con muchos modelos, se recomienda dividir estos en múltiples archivos dentro de un módulo.

### ¿Qué es el API router y cómo agrupa endpoints?

El API router es como una mini aplicación dentro de FastAPI que permite agrupar y organizar endpoints relacionados. Este agrupamiento facilita el soporte y la implementación de cambios en los endpoints específicos sin afectar el resto. Cada router puede manejar un recurso en particular, como `customer`, `transactions` o `invoices`.

### Ejemplo de configuración de router

1. En la carpeta routers/, creamos un archivo como customer.py.
2. Copiamos los endpoints de customer a este archivo y realizamos los imports necesarios, como:
 - `models` para acceder a los modelos de datos.
 - `DB` para la configuración de la base de datos.
 - `APIRouter` de FastAPI para definir el router.
 
3. Modificamos el código para usar `router` en lugar de `app`, y así consolidamos el router de `customer`.

### ¿Cómo registrar routers y resolver errores comunes?

Al mover los endpoints a un archivo específico, debemos registrarlos en `main.py` usando `include_router`. Si al ejecutar la aplicación se encuentran errores, pueden ser causados por imports incorrectos. Es común que:

 - Los imports de `models` o `DB` requieran ajustes en las rutas de acceso.
 - Los routers deben incluir el punto y la ruta correcta, como `routers.customer`.

### ¿Cómo agrupar endpoints con tags en FastAPI?

Agregar etiquetas (tags) permite que los endpoints aparezcan agrupados en la documentación. Esto se logra fácilmente con FastAPI, lo que ayuda a:

 - Identificar rápidamente endpoints relacionados en la documentación.
 - Facilitar la navegación para desarrolladores y usuarios de la API.

### Ejemplo de etiquetado en un router

Dentro del archivo del router (`customer.py`):

- Agregamos `tags=["customer"]` en cada endpoint.
- Así, en la documentación, todos los endpoints de `customer` aparecerán agrupados bajo esta etiqueta.

**Lecturas recomendadas**

[Bigger Applications - Multiple Files - FastAPI](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

## ¿Cómo gestionar datos relacionados en SQLModel con FastAPI?

Gestionar datos relacionados en **SQLModel** con **FastAPI** implica definir relaciones entre tablas, realizar consultas eficientes y diseñar endpoints que respeten estas relaciones. Aquí te explico cómo hacerlo paso a paso:

---

### **1. Definir Relaciones en SQLModel**

SQLModel permite definir relaciones entre tablas utilizando **claves foráneas** y la opción `relationship` de SQLAlchemy. 

#### **Ejemplo: Usuarios y Órdenes**
```python
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str

    orders: List["Order"] = Relationship(back_populates="user")

class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    description: str
    user_id: int = Field(foreign_key="user.id")

    user: User = Relationship(back_populates="orders")
```

- `foreign_key`: Define la clave foránea en el modelo.
- `Relationship`: Establece una relación entre tablas.
- `back_populates`: Conecta las relaciones en ambas direcciones.

---

### **2. Crear las Tablas**

En el arranque de la aplicación, asegúrate de crear todas las tablas relacionadas:
```python
from sqlmodel import SQLModel, create_engine

sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
```

Llama a esta función durante el evento `startup` de FastAPI:
```python
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
```

---

### **3. Gestionar Sesiones de Base de Datos**

Crea una dependencia para gestionar sesiones:
```python
from sqlmodel import Session
from typing import Generator

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
```

Utilízala con `Depends` en tus rutas.

---

### **4. Crear Relaciones en la Base de Datos**

Cuando crees registros relacionados, asegúrate de gestionar las claves foráneas correctamente.

#### Crear un usuario con órdenes:
```python
from fastapi import FastAPI, Depends, HTTPException
from sqlmodel import Session

app = FastAPI()

@app.post("/users/")
def create_user(user: User, session: Session = Depends(get_session)):
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

@app.post("/orders/")
def create_order(order: Order, session: Session = Depends(get_session)):
    session.add(order)
    session.commit()
    session.refresh(order)
    return order
```

---

### **5. Consultar Datos Relacionados**

Puedes usar SQLAlchemy para cargar relaciones automáticamente con la opción `selectinload`.

#### Consultar un usuario con sus órdenes:
```python
from sqlmodel import select
from sqlalchemy.orm import selectinload

@app.get("/users/{user_id}")
def get_user_with_orders(user_id: int, session: Session = Depends(get_session)):
    statement = select(User).options(selectinload(User.orders)).where(User.id == user_id)
    result = session.exec(statement).first()
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result
```

---

### **6. Actualizar Relaciones**

Para actualizar datos relacionados, primero consulta el registro, realiza los cambios y luego guarda los datos.

```python
@app.put("/orders/{order_id}")
def update_order(order_id: int, description: str, session: Session = Depends(get_session)):
    order = session.get(Order, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    order.description = description
    session.add(order)
    session.commit()
    session.refresh(order)
    return order
```

---

### **7. Eliminar Relaciones**

Para eliminar registros relacionados, asegúrate de manejar las restricciones de claves foráneas (ON DELETE CASCADE o eliminarlas manualmente).

```python
@app.delete("/users/{user_id}")
def delete_user(user_id: int, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    session.delete(user)
    session.commit()
    return {"message": "User deleted successfully"}
```

---

### **8. Respuesta con Pydantic y Relación Anidada**

Usa modelos de Pydantic para controlar la respuesta de datos relacionados.

```python
from pydantic import BaseModel
from typing import List

class OrderResponse(BaseModel):
    id: int
    description: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    orders: List[OrderResponse]
```

Define la respuesta en las rutas:
```python
@app.get("/users/{user_id}", response_model=UserResponse)
def get_user_with_orders(user_id: int, session: Session = Depends(get_session)):
    statement = select(User).options(selectinload(User.orders)).where(User.id == user_id)
    user = session.exec(statement).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

### **Buenas Prácticas**
1. **Validaciones:** Usa modelos de entrada separados para validaciones (`UserCreate`, `OrderCreate`).
2. **Cargar relaciones solo cuando sea necesario:** Utiliza `selectinload` o `joinedload` para evitar sobrecargar las consultas.
3. **Optimización:** Limita los datos que cargas desde la base de datos para mejorar el rendimiento.

---

Esta estructura te permitirá gestionar eficientemente datos relacionados en **SQLModel** y construir APIs robustas con **FastAPI**.

### Resumen

Los modelos de datos en bases de datos relacionales permiten organizar y relacionar información sin duplicarla en múltiples tablas, optimizando así la gestión de datos. Al usar FastAPI y SQLModel, es posible configurar estas relaciones en los modelos que luego reflejarán las tablas en la base de datos, permitiendo un acceso eficiente y estructurado a los datos.

### ¿Cómo se crea una relación uno a muchos entre modelos?

En SQLModel, para crear una relación uno a muchos, como entre un `Customer` y sus `Transactions`, se establecen claves foráneas (foreign keys) que vinculan los registros de una tabla con los de otra. En este caso, la tabla `Transaction` incluirá un campo `customer_id` que hace referencia al `id` en la tabla `Customer`, garantizando que cada transacción esté asociada con un cliente.

### Pasos principales:

- **Definir el modelo `Transaction` como tabla**: Cambiando el modelo `Transaction` para que herede de `SQLModel` y estableciendo `table=True`, lo cual genera la tabla en la base de datos.
- **Configurar claves foráneas**: Agregar un campo `customer_id` en `Transaction`, que será una clave foránea vinculada al `id` de `Customer`.
- **Relaciones bidireccionales**: Usar `relationship` y `back_populates` para conectar ambos modelos, de modo que al acceder a un `Custome`r, se puedan ver todas sus `Transactions` y viceversa.

### ¿Cómo se manejan los endpoints de FastAPI para estos modelos relacionados?

Con FastAPI, la creación de endpoints para modelos relacionados implica definir operaciones de creación y consulta que respeten las relaciones establecidas.

- **Crear un endpoint para listar transacciones**: Con una query básica que retorna todas las transacciones de la base de datos.
- **Crear un endpoint para crear transacciones**: Requiere validar que el `customer_id` exista antes de añadir la transacción. Si no existe, el sistema devuelve un error `404` Not Found con un mensaje claro.
- **Ajuste de código de estado en respuestas**: En este caso, el endpoint de creación debe responder con un código `201 Created` cuando una transacción se guarda exitosamente.

### ¿Cómo probar las relaciones y endpoints configurados?

Para verificar que las configuraciones funcionan correctamente:

1. **Verificar la existencia de tablas**: Tras ejecutar el proyecto, se pueden revisar las tablas con comandos SQL (ej., `.tables` y `.schema` en SQLite).
2. **Pruebas de creación de registros**: Crear un cliente y luego una transacción asociada a este. Intentar crear una transacción para un `customer_id` inexistente debería retornar un error claro.
3. **Consulta de transacciones**: Al listar transacciones, deben mostrarse solo las asociadas al cliente indicado en `customer_id`.

### ¿Cómo optimizar la consulta de datos en FastAPI y SQLModel?

SQLModel simplifica el acceso a los datos mediante relaciones, evitando la necesidad de múltiples queries. Al usar `relationship`, se puede acceder a los datos relacionados directamente, ahorrando tiempo y simplificando el código.

**Lecturas recomendadas**

[SQLModel](https://sqlmodel.tiangolo.com/)

## ¿Cómo crear relaciones de muchos a muchos en SQLModel?

En **SQLModel**, puedes crear relaciones de muchos a muchos utilizando una **tabla intermedia** (también conocida como tabla puente). Esta tabla intermedia conecta dos entidades con claves foráneas, permitiendo que ambas tengan una relación bidireccional.

Aquí tienes una guía para crear relaciones de muchos a muchos en **SQLModel**:

### **1. Crear Modelos Base y Tabla Intermedia**

Define los modelos principales y la tabla intermedia para la relación.

#### Ejemplo: **Estudiantes y Cursos**
Un estudiante puede estar inscrito en varios cursos, y un curso puede tener varios estudiantes.

```python
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

class StudentCourseLink(SQLModel, table=True):
    student_id: Optional[int] = Field(default=None, foreign_key="student.id", primary_key=True)
    course_id: Optional[int] = Field(default=None, foreign_key="course.id", primary_key=True)

class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    courses: List["Course"] = Relationship(back_populates="students", link_model=StudentCourseLink)

class Course(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    students: List[Student] = Relationship(back_populates="courses", link_model=StudentCourseLink)
```

- **`StudentCourseLink`**: Es la tabla intermedia que conecta `Student` y `Course` mediante claves foráneas.
- **`Relationship`**:
  - `back_populates`: Conecta las relaciones en ambos lados.
  - `link_model`: Especifica la tabla intermedia que establece la relación.

### **2. Crear la Base de Datos**

Crea todas las tablas en la base de datos.

```python
from sqlmodel import create_engine

sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

create_db_and_tables()
```

### **3. Insertar Datos en la Relación**

Para insertar datos en una relación de muchos a muchos, añade registros tanto en las tablas principales como en la tabla intermedia.

#### Ejemplo de Inserción:
```python
from sqlmodel import Session

def create_sample_data():
    with Session(engine) as session:
        # Crear estudiantes
        student1 = Student(name="Alice")
        student2 = Student(name="Bob")

        # Crear cursos
        course1 = Course(title="Math")
        course2 = Course(title="Science")

        # Establecer relaciones
        student1.courses.append(course1)
        student1.courses.append(course2)
        student2.courses.append(course1)

        # Guardar en la base de datos
        session.add(student1)
        session.add(student2)
        session.commit()

create_sample_data()
```

- Las relaciones se gestionan automáticamente gracias a `Relationship`.

### **4. Consultar Datos Relacionados**

Puedes cargar datos relacionados usando la sesión de SQLModel y la relación configurada.

#### Consultar Estudiantes con sus Cursos:
```python
from sqlmodel import select

def get_students_with_courses():
    with Session(engine) as session:
        students = session.exec(select(Student)).all()
        for student in students:
            print(f"Student: {student.name}")
            for course in student.courses:
                print(f"  Enrolled in: {course.title}")

get_students_with_courses()
```

#### Consultar Cursos con sus Estudiantes:
```python
def get_courses_with_students():
    with Session(engine) as session:
        courses = session.exec(select(Course)).all()
        for course in courses:
            print(f"Course: {course.title}")
            for student in course.students:
                print(f"  Student: {student.name}")

get_courses_with_students()
```

### **5. Actualizar Relaciones**

Puedes actualizar las relaciones agregando o eliminando elementos en las listas de relaciones.

#### Agregar un Curso a un Estudiante:
```python
def add_course_to_student(student_id: int, course_id: int):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        course = session.get(Course, course_id)
        if student and course:
            student.courses.append(course)
            session.commit()

add_course_to_student(student_id=1, course_id=2)
```

#### Eliminar un Curso de un Estudiante:
```python
def remove_course_from_student(student_id: int, course_id: int):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        course = session.get(Course, course_id)
        if student and course:
            student.courses.remove(course)
            session.commit()

remove_course_from_student(student_id=1, course_id=2)
```

### **6. Buenas Prácticas**
1. **Utiliza Transacciones:** Asegúrate de usar transacciones al realizar múltiples cambios.
2. **Consultas Optimizadas:** Usa `joinedload` o `selectinload` para cargar relaciones en una sola consulta.
3. **Validaciones:** Valida los datos antes de agregarlos o eliminarlos de las relaciones.

Con este enfoque, puedes gestionar relaciones de muchos a muchos en **SQLModel** de manera eficiente en tu aplicación con **FastAPI**.

### Resumen

Las relaciones de muchos a muchos permiten conectar múltiples elementos entre sí en bases de datos, utilizando una tabla intermedia. A continuación, exploramos cómo implementar esta relación en SQLModel, especialmente para un sistema donde un cliente puede tener múltiples planes, como premium o basic, y cada plan puede asociarse con varios clientes.

### ¿Cómo se crea la tabla intermedia para una relación de muchos a muchos?

Para manejar la relación de muchos a muchos, se requiere una tabla intermedia que conecte los identificadores de ambas tablas principales:

- Primero, se crea una nueva clase `Plan`, que hereda de `SQLModel`. Esta tabla tendrá atributos como:

 - `id`: Identificador primario de tipo entero.
 - `nombre`: Nombre del plan (string, obligatorio).
 - `precio`: Precio del plan (entero).
 - `descripcion`: Descripción breve (string).
- Después, se define la tabla intermedia CustomerPlan:

 - `plan_id`: Llave foránea que se refiere al ID de `Plan`.
 - `customer_id`: Llave foránea que se refiere al ID de `Customer`.
 - Es importante definir un campo `id` como clave primaria para evitar errores al trabajar con SQLModel y Pydantic.

### ¿Cómo se establece la relación en el modelo de datos?

Para que los modelos `Customer` y `Plan` reconozcan esta relación, se usa la clase `relationship` de SQLModel en cada uno:

- En `Plan`, se define un campo llamado `customers`, que será una lista de instancias de `Customer`.

 - La relación se establece mediante `relationship`, utilizando `back_populates="plans"` y especificando el modelo de enlace `link_model=CustomerPlan`.
- En `Customer`, se configura un campo `plans`, que será una lista de instancias de `Plan`.

 - Se usa también `relationship`, pero esta vez con `back_populates="customers"` y `link_model=CustomerPlan`.

Esta configuración asegura que ambas tablas se conecten a través de la tabla intermedia `CustomerPlan`, facilitando la consulta de todos los planes asociados a un cliente y viceversa.

### ¿Cómo verificar la relación en la base de datos?

Para confirmar que las tablas y relaciones están correctamente creadas:

- Inicie la base de datos y ejecute el comando `tables` para listar todas las tablas creadas. Debería aparecer `customer_plan`.
- Al revisar el esquema, debe visualizarse `plan_id` y `customer_id` con las llaves foráneas que apuntan a `Plan` y `Customer`, respectivamente.
Esta estructura permite manejar dinámicamente la relación de muchos a muchos en SQLModel y simplifica la administración de datos asociados, como en este caso los planes de suscripción de un cliente.

**Nota**: para utilizar el sqlite3 se utiliza **wsl** ya que powershell no funciona, se instala con `sudo apt install sqlite3` luego para ingresar se utiliza `sqlite3 fastAPI/db.sqlite3` o el destino del archivo db.sqlite3 se debe colocar la ruta bien.
para ver la tablas se utiliza `.tables` y ver el contenido se utiliza `.schema <Tabla>` y salir del sqlite3 se utiliza `.quit`

## Relacionar Modelos de Datos en FastAPI: Implementación de Relaciones

Para relacionar modelos de datos en **FastAPI** con **SQLModel**, puedes usar las características de **relaciones** proporcionadas por SQLModel para conectar diferentes tablas en una base de datos relacional. Esto es útil para manejar relaciones comunes como **uno a muchos** y **muchos a muchos**.

### **1. Configurar Relación "Uno a Muchos"**

### Ejemplo: **Usuarios y Publicaciones**
- Un usuario puede tener muchas publicaciones.
- Una publicación pertenece a un usuario.

### Modelo de Datos

```python
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    posts: List["Post"] = Relationship(back_populates="owner")

class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    user_id: int = Field(foreign_key="user.id")
    owner: User = Relationship(back_populates="posts")
```

### Explicación:
- **`User` tiene una lista de publicaciones (`posts`)**: Relación uno a muchos.
- **`Post` tiene un único `owner`**: Relación inversa con una clave foránea (`user_id`).

### **2. Crear la Base de Datos**

```python
from sqlmodel import create_engine

sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

create_db_and_tables()
```

### **3. Operaciones CRUD**

#### Crear un Usuario y Publicaciones

```python
from sqlmodel import Session

def create_sample_data():
    with Session(engine) as session:
        user = User(name="Alice", email="alice@example.com")
        post1 = Post(title="My first post", content="This is my first post!", owner=user)
        post2 = Post(title="Another post", content="This is another post!", owner=user)

        session.add(user)
        session.commit()

create_sample_data()
```

#### Consultar un Usuario con sus Publicaciones

```python
from sqlmodel import select

def get_user_with_posts(user_id: int):
    with Session(engine) as session:
        user = session.get(User, user_id)
        print(f"User: {user.name}, Email: {user.email}")
        for post in user.posts:
            print(f"  Post: {post.title} - {post.content}")

get_user_with_posts(user_id=1)
```

### **4. Configurar Relación "Muchos a Muchos"**

### Ejemplo: **Estudiantes y Cursos**
Un estudiante puede inscribirse en muchos cursos, y un curso puede tener muchos estudiantes.

### Modelo de Datos

```python
class StudentCourseLink(SQLModel, table=True):
    student_id: Optional[int] = Field(default=None, foreign_key="student.id", primary_key=True)
    course_id: Optional[int] = Field(default=None, foreign_key="course.id", primary_key=True)

class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    courses: List["Course"] = Relationship(back_populates="students", link_model=StudentCourseLink)

class Course(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    students: List[Student] = Relationship(back_populates="courses", link_model=StudentCourseLink)
```

### **5. Insertar y Consultar Relaciones**

#### Insertar un Estudiante y Cursos

```python
def create_students_and_courses():
    with Session(engine) as session:
        student1 = Student(name="John Doe")
        course1 = Course(title="Math 101")
        course2 = Course(title="Physics 101")

        student1.courses.append(course1)
        student1.courses.append(course2)

        session.add(student1)
        session.commit()

create_students_and_courses()
```

#### Consultar Cursos de un Estudiante

```python
def get_student_courses(student_id: int):
    with Session(engine) as session:
        student = session.get(Student, student_id)
        print(f"Student: {student.name}")
        for course in student.courses:
            print(f"  Enrolled in: {course.title}")

get_student_courses(student_id=1)
```

### **6. Relacionar con Endpoints en FastAPI**

#### Endpoints para Usuarios y Publicaciones

```python
from fastapi import FastAPI, Depends
from sqlmodel import Session

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int, session: Session = Depends(lambda: Session(engine))):
    user = session.get(User, user_id)
    if not user:
        return {"error": "User not found"}
    return {"name": user.name, "email": user.email, "posts": [{"title": p.title, "content": p.content} for p in user.posts]}

@app.post("/users")
def create_user(user: User, session: Session = Depends(lambda: Session(engine))):
    session.add(user)
    session.commit()
    return user
```

### **Resumen**

1. **Relaciones Uno a Muchos**:
   - Define una clave foránea en el modelo hijo.
   - Usa `Relationship` con `back_populates`.

2. **Relaciones Muchos a Muchos**:
   - Define una tabla intermedia con claves foráneas de ambos modelos.
   - Usa `link_model` en `Relationship`.

3. **Consultas y CRUD**:
   - Gestiona relaciones directamente mediante las propiedades definidas.
   - Usa sesiones para consultas y cambios en la base de datos.

4. **Integración con FastAPI**:
   - Utiliza dependencias para manejar la sesión de la base de datos.
   - Expón endpoints para manejar los datos relacionados.

Estas técnicas te permiten estructurar y gestionar relaciones complejas en tus APIs con **FastAPI** y **SQLModel**.

### Resumen

Para crear un sistema de suscripción de clientes a planes en FastAPI, necesitamos desarrollar endpoints que gestionen tanto la creación de planes como la suscripción de clientes a estos planes. Además, se establecerá una relación entre los datos para permitir la consulta y gestión de estas suscripciones. A continuación, se detalla cómo implementar esta funcionalidad paso a paso.

### ¿Cómo crear el endpoint para generar planes?

1. **Configuración del archivo de rutas**: Comienza creando un archivo en la carpeta `routers`, denominado `plans`. Importa `APIRouter` de FastAPI y configura un nuevo router para gestionar los planes.
2. Definición del endpoint `create_plan`: Este será un endpoint de tipo `POST`, con la ruta `/plans`. Recibirá:
 - `plan_data`: información del plan, validada mediante un modelo (`Plan`) que se importa desde `models`.
 - `session`: la sesión de base de datos importada desde `DB`.
 
3. **Validación y almacenamiento de datos**:
 - Valida la información recibida con el modelo `Plan`, que convierte el diccionario de datos en un modelo válido.
 - Guarda el nuevo plan en la base de datos usando `session.add()` y luego confirma la transacción con `session.commit()`.
 - Para devolver el plan, refresca la sesión y retorna el objeto.

Este proceso permite registrar los planes correctamente en la base de datos. En caso de que se presente un error (por ejemplo, debido a cambios en el esquema), es posible regenerar la base de datos eliminando las tablas antiguas y ejecutando nuevamente la aplicación.

### ¿Cómo suscribir un cliente a un plan específico?

1. **Creación del endpoint de suscripción**:

 - Dirígete al archivo de rutas de `customers` y crea un nuevo método `subscribe_customer_to_plan`.
 - Define el método como `POST` y configura la URL para que reciba tanto el `customer_i`d como el `plan_id`.
 - El método recibe ambos IDs (cliente y plan) y la sesión de base de datos.
 
2. **Validación de existencia**:

 - Usa la sesión para obtener los objetos `customer` y `plan` por sus IDs.
 - Si alguno de ellos no existe, arroja una excepción `HTTPException` con un código de estado 404.
 
3. **Creación de la relación**:

 - Utiliza la relación definida en `models` (`CustomerPlan`), la cual requiere plan_id y `customer_id` para establecer la suscripción.
 - Guarda esta relación en la base de datos con `session.add()`, realiza el `commit`, y refresca la sesión antes de retornarla.
 
Este endpoint asegura que solo los clientes y planes existentes puedan ser suscritos. En el navegador, al probar con IDs válidos, se muestra la suscripción creada con sus respectivos IDs.

### ¿Cómo listar las suscripciones de un cliente?

1. **Endpoint para consultar suscripciones**:

 - Define un nuevo método de tipo `GET` en el router de `customers`.
 - La ruta debe recibir únicamente el `customer_id` como parámetro y la sesión de base de datos.
 
2. Consulta y retorno:

 - Realiza una consulta con la sesión para obtener el customer por su ID.
 - Si no existe, retorna un error. Si existe, devuelve la lista de planes suscritos.
Al probar este endpoint, se visualizan todas las suscripciones activas del cliente, incluyendo los detalles del plan.

### ¿Cómo extender la funcionalidad con un campo de estado?

Se sugiere un reto adicional: agregar un campo estado en la relación CustomerPlan, permitiendo filtrar solo los planes activos en el endpoint de suscripción. Para ello:

1. Agrega el campo estado en el modelo CustomerPlan.
2. Modifica el endpoint de listado de suscripciones para que solo devuelva las activas.
3. Realiza pruebas para confirmar que el filtro funciona correctamente.

## Consultas avanzadas con SQLModel en FastAPI

SQLModel es una herramienta poderosa para trabajar con consultas más avanzadas en FastAPI, aprovechando la combinación de SQLAlchemy y Pydantic. A continuación, te muestro cómo realizar consultas avanzadas utilizando SQLModel en un proyecto de FastAPI.


### **1. Consultas con Filtros Múltiples**
Puedes usar múltiples condiciones en una consulta para obtener resultados específicos.

```python
from sqlmodel import select, Session
from app.models import Customer

def get_customers_by_filters(session: Session, name: str = None, is_active: bool = None):
    query = select(Customer)
    if name:
        query = query.where(Customer.name.contains(name))
    if is_active is not None:
        query = query.where(Customer.is_active == is_active)
    return session.exec(query).all()
```

**Ejemplo de Uso:**

```python
@router.get("/customers/search", tags=["Customers"])
async def search_customers(name: str = None, is_active: bool = None, session: Session = Depends(get_session)):
    customers = get_customers_by_filters(session, name, is_active)
    return customers
```

### **2. Consultas Agregadas**
Las consultas agregadas permiten realizar operaciones como contar registros o calcular valores.

#### **Contar Clientes Activos**
```python
from sqlmodel import func

@router.get("/customers/count", tags=["Customers"])
async def count_active_customers(session: Session = Depends(get_session)):
    query = select(func.count()).where(Customer.is_active == True)
    result = session.exec(query).first()
    return {"active_customers": result[0]}
```

#### **Sumar y Agrupar**
Por ejemplo, sumar el total de ingresos por cliente:

```python
@router.get("/customers/revenue", tags=["Customers"])
async def calculate_revenue(session: Session = Depends(get_session)):
    query = select(Customer.name, func.sum(Order.amount)).join(Order).group_by(Customer.name)
    result = session.exec(query).all()
    return result
```

### **3. Relación entre Modelos (Join)**
Si tienes modelos relacionados, como `Customer` y `Order`, puedes realizar consultas entre ellos.

#### **Modelo de Ejemplo**
```python
class Order(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    amount: float
    customer_id: int = Field(foreign_key="customer.id")
    customer: Optional["Customer"] = Relationship(back_populates="orders")

class Customer(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str
    is_active: bool
    orders: List[Order] = Relationship(back_populates="customer")
```

#### **Consultar Pedidos por Cliente**
```python
@router.get("/customers/{customer_id}/orders", tags=["Orders"])
async def get_customer_orders(customer_id: int, session: Session = Depends(get_session)):
    query = select(Order).where(Order.customer_id == customer_id)
    orders = session.exec(query).all()
    return orders
```

### **4. Subconsultas**
Puedes utilizar subconsultas para casos más complejos.

#### **Cliente con Mayor Gasto**
```python
from sqlmodel import subquery

@router.get("/customers/top-spender", tags=["Customers"])
async def get_top_spender(session: Session = Depends(get_session)):
    subquery_orders = select(Order.customer_id, func.sum(Order.amount).label("total_spent")).group_by(Order.customer_id).subquery()
    query = select(Customer, subquery_orders.c.total_spent).join(subquery_orders, subquery_orders.c.customer_id == Customer.id).order_by(subquery_orders.c.total_spent.desc()).limit(1)
    result = session.exec(query).first()
    return result
```

### **5. Paginación**
Para manejar grandes cantidades de datos, puedes implementar paginación.

```python
@router.get("/customers", tags=["Customers"])
async def get_customers(page: int = 1, page_size: int = 10, session: Session = Depends(get_session)):
    query = select(Customer).offset((page - 1) * page_size).limit(page_size)
    customers = session.exec(query).all()
    return customers
```

### **6. Filtrar Datos Relacionados**
Filtrar pedidos que excedan un monto específico:

```python
@router.get("/orders/high-value", tags=["Orders"])
async def get_high_value_orders(min_amount: float, session: Session = Depends(get_session)):
    query = select(Order).where(Order.amount > min_amount)
    orders = session.exec(query).all()
    return orders
```

### **Conclusión**
Estas consultas avanzadas te permiten realizar operaciones complejas y específicas con SQLModel en FastAPI.

### Resumen

Aprender a realizar consultas avanzadas en SQL Model te permite filtrar datos de manera más precisa y mejorar el control sobre la información almacenada en la base de datos. En este caso, veremos cómo agregar un campo `status` para identificar si un plan de un cliente está activo o inactivo, y cómo integrarlo en las consultas de FastAPI mediante parámetros de consulta (query params).

### ¿Cómo agregar un campo de estatus al modelo?

Para gestionar los planes de clientes con distintos estados, comenzamos agregando un nuevo campo status en el modelo customer plan. Los pasos son los siguientes:

- **Crear un Enum**: definimos una clase `StatusEnum` heredada de `enum` de Python, permitiendo los valores `active` e `inactive`. Esto facilita el manejo de estatus con una lista de opciones controladas.
- **Agregar el nuevo campo**: se incluye el campo status en el modelo `customer plan`, con `active` como valor predeterminado.
- **Actualizar la base de datos**: tras modificar el modelo, es necesario regenerar la base de datos, para lo cual detenemos la aplicación y la reiniciamos.

### ¿Cómo utilizar el estatus en las consultas?

Una vez añadido el estatus, podemos integrarlo en las consultas para filtrar solo los planes activos o inactivos.

1. **Parámetros de consulta**: añadimos `plan_status` como un parámetro opcional en el router de creación de planes, utilizando `Query` de FastAPI para recibir valores a través de la URL.

2. **Filtrar por estatus**: configuramos el parámetro `plan_status` para que sea opcional, de modo que el valor predeterminado será `active` si no se especifica.

### ¿Cómo aplicar filtros en las consultas?

Para listar los planes según su estatus, configuramos la consulta de esta manera:

**Seleccionar planes específicos**: en el endpoint que lista todos los planes de un cliente, usamos session.select() con dos condiciones where:

 - **Customer ID**: la consulta verifica que el customer ID coincida con el ID del cliente recibido en la URL.
 - **Estatus**: se filtran los planes que coinciden con el plan_status especificado en la URL.
 
- **Ejecutar la consulta**: llamamos a .all() sobre la consulta para obtener todos los elementos filtrados.

### ¿Cómo probar y ajustar los filtros?

Para probar el filtro, ejecutamos la aplicación y verificamos en el endpoint de creación de suscripciones si se muestra el plan_status como parámetro de consulta. Esto permite crear planes con el estatus deseado, y probar distintos valores para asegurarse de que el filtro funcione correctamente.

**Lecturas recomendadas**

[enum — Support for enumerations](https://docs.python.org/3/library/enum.html)

## Implementación de validación de datos en FastAPI con Pydantic

La **validación de datos** en FastAPI se realiza utilizando **Pydantic**, lo que permite definir reglas de validación de manera declarativa y eficaz. A continuación, exploraremos cómo implementar esta funcionalidad en FastAPI.

### **1. Definición de Modelos de Pydantic**
En FastAPI, los modelos de Pydantic son usados para validar datos de entrada y salida. Puedes establecer tipos de datos, valores predeterminados, restricciones y más.

```python
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Nombre del usuario")
    email: EmailStr
    age: int = Field(..., gt=0, le=120, description="Edad del usuario, debe ser entre 1 y 120 años")
```

### **2. Validación Automática en Endpoints**
Cuando defines un endpoint con un modelo de Pydantic como parámetro, FastAPI valida automáticamente los datos recibidos.

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/users/")
async def create_user(user: User):
    return {"message": "Usuario creado exitosamente", "user": user}
```

#### **Validación en Acción**
Si envías datos que no cumplen con las reglas definidas (por ejemplo, un correo no válido o un nombre muy corto), FastAPI responderá con un error 422 y detalles sobre la validación.

### **3. Uso de Validaciones Personalizadas**
Puedes definir validaciones adicionales utilizando métodos en el modelo de Pydantic.

#### **Ejemplo: Validar Nombre Único**
```python
from pydantic import BaseModel, validator

class User(BaseModel):
    username: str
    email: EmailStr
    age: int

    @validator("username")
    def validate_unique_username(cls, value):
        if value.lower() in ["admin", "root"]:
            raise ValueError("El nombre de usuario no puede ser 'admin' o 'root'")
        return value
```

### **4. Validación Compleja con Dependencias**
Además de usar Pydantic, puedes implementar dependencias para manejar validaciones más dinámicas.

```python
from fastapi import Depends

def validate_age(age: int):
    if age < 18:
        raise HTTPException(status_code=400, detail="La edad debe ser mayor o igual a 18")
    return age

@app.get("/validate/")
async def check_age(age: int = Depends(validate_age)):
    return {"message": f"La edad {age} es válida"}
```

### **5. Validación en la Salida**
Puedes usar modelos de Pydantic para controlar cómo se devuelven los datos al cliente.

```python
from fastapi.responses import JSONResponse

class UserOut(BaseModel):
    username: str
    email: EmailStr

@app.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int):
    user = {"username": "john_doe", "email": "john.doe@example.com", "age": 30}  # Simulación
    return user
```

En este caso, solo se devuelven `username` y `email`, aunque el objeto `user` contiene más datos.

### **6. Validación de Entradas y Salidas Juntas**
Puedes usar modelos diferentes para entrada y salida si los datos requeridos son distintos.

```python
class UserIn(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    username: str
    email: EmailStr

@app.post("/users/", response_model=UserOut)
async def create_user(user: UserIn):
    new_user = {"username": user.username, "email": "user@example.com"}  # Simulación
    return new_user
```

### **7. Uso de Tipos Avanzados**
Puedes usar tipos avanzados como listas, diccionarios o incluso modelos anidados.

```python
from typing import List

class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    order_id: int
    items: List[Item]
    total_price: float

@app.post("/orders/")
async def create_order(order: Order):
    return order
```

### **8. Manejo de Errores de Validación**
Si necesitas personalizar el comportamiento cuando ocurre un error de validación, puedes usar los **event handlers**.

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )
```

### **Conclusión**
La validación en FastAPI con Pydantic es robusta y flexible. Permite manejar desde casos simples hasta validaciones complejas con facilidad.

### Resumen

La validación de datos es fundamental en el desarrollo de software para asegurar que los datos ingresados sean correctos y cumplan con los requisitos establecidos. Aquí exploraremos cómo implementar validaciones específicas de emails utilizando Pydantic y FastAPI, además de cómo verificar la unicidad de los datos en la base de datos.

### ¿Cómo validar emails correctamente?

Para validar un email básico, necesitamos asegurarnos de que contenga un símbolo de arroba (`@`) y termine en un dominio válido. Pydantic facilita esta validación con tipos de datos específicos, como `EmailStr`, que podemos importar directamente. Al cambiar el tipo del campo de str a `EmailStr` en el modelo, logramos:

- Validar el formato del email automáticamente.
- Recibir errores claros cuando se ingresa un email incorrecto.

Ejecutar esta validación en un endpoint nos permite identificar errores en el formato del email en tiempo real, devolviendo mensajes como “no es un email válido” cuando el formato no cumple con los requisitos.

### ¿Cómo asegurar la unicidad de un email en la base de datos?

Validar el formato del email no es suficiente en muchos casos, ya que necesitamos verificar que el email sea único en la base de datos para evitar duplicados. Para ello:

1. Definimos una nueva función `validate_email`.
2. Utilizamos el decorador `FieldValidator`, disponible en Pydantic 2, para validar campos específicos.

Con `FieldValidator`, podemos verificar la unicidad del email en la base de datos. Esto requiere acceso a una sesión de base de datos para ejecutar una consulta que busque coincidencias de emails.

¿Cómo integrar la validación con la base de datos?

Para validar contra la base de datos, necesitamos importar elementos de SQLAlchemy:

- **Sesión**: Creamos una sesión usando `SessionEngine` para gestionar la conexión con la base de datos.
- **Consulta `select`**: Realizamos una consulta a la tabla `customer` con una cláusula `where` que verifica si el email ya existe.

Esta configuración asegura que solo se cree un registro si el email es único. De lo contrario, la aplicación devuelve un error indicando la duplicación, protegiendo así la integridad de los datos.

### ¿Cómo manejar la versión de Pydantic en FastAPI? 

La transición de Pydantic de la versión 1 a la 2 trae cambios importantes en los decoradores de validación. Mientras que `validator` funcionaba en la versión 1, en la versión 2 se utiliza `FieldValidator`, que ofrece una estructura más flexible para validar múltiples campos.

- **Cambio de `self` a `cls`**: FieldValidator se aplica a métodos de clase, por lo que debemos definirlos con `cls` en lugar de `self`.

Este ajuste es esencial para que FastAPI y Pydantic funcionen de manera óptima en las versiones más recientes.

## ¿Cómo implementar paginación de datos en FastAPI?

La paginación es esencial para manejar grandes cantidades de datos en las API, mejorando el rendimiento y la experiencia del usuario. FastAPI facilita la implementación de paginación combinando parámetros de consulta (`query parameters`) y modelos de datos.

A continuación, te explico cómo implementarla paso a paso:

---

### **1. Definición del Modelo de Datos**
Usaremos un modelo para representar los datos que queremos paginar. 

```python
from typing import List
from pydantic import BaseModel

class Item(BaseModel):
    id: int
    name: str
    description: str
```

### **2. Creación de Datos Simulados**
Generemos algunos datos simulados para demostrar la paginación.

```python
items = [
    {"id": i, "name": f"Item {i}", "description": f"Description for item {i}"}
    for i in range(1, 101)
]
```

### **3. Configuración de Parámetros de Paginación**
Definimos parámetros de consulta (`query parameters`) como `skip` (elementos a omitir) y `limit` (número de elementos a devolver).

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/", response_model=List[Item])
async def get_items(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1)):
    return items[skip : skip + limit]
```

### **4. Uso del Endpoint**
- Solicitud: `GET /items/?skip=10&limit=5`
- Respuesta: Los 5 elementos que comienzan desde el índice 10.

```json
[
    {
        "id": 11,
        "name": "Item 11",
        "description": "Description for item 11"
    },
    {
        "id": 12,
        "name": "Item 12",
        "description": "Description for item 12"
    },
    ...
]
```

### **5. Respuesta Extendida con Metadatos**
Puedes incluir metadatos como el total de elementos o la página actual en la respuesta.

#### **Modelo para la Respuesta**
```python
from pydantic import BaseModel
from typing import List, Optional

class PaginatedResponse(BaseModel):
    total: int
    items: List[Item]
    skip: int
    limit: int
```

#### **Actualización del Endpoint**
```python
@app.get("/items/", response_model=PaginatedResponse)
async def get_items(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1)):
    total = len(items)
    paginated_items = items[skip : skip + limit]
    return {"total": total, "items": paginated_items, "skip": skip, "limit": limit}
```

#### **Respuesta del Endpoint**
```json
{
    "total": 100,
    "items": [
        {
            "id": 11,
            "name": "Item 11",
            "description": "Description for item 11"
        },
        {
            "id": 12,
            "name": "Item 12",
            "description": "Description for item 12"
        }
    ],
    "skip": 10,
    "limit": 2
}
```

### **6. Paginación con SQLModel o SQLAlchemy**
Si trabajas con una base de datos, puedes implementar paginación usando consultas SQL.

```python
from sqlmodel import Session, select
from fastapi import Depends
from database import get_session  # Función que proporciona la sesión de base de datos

@app.get("/items/", response_model=PaginatedResponse)
async def get_items(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1), session: Session = Depends(get_session)):
    statement = select(Item).offset(skip).limit(limit)
    results = session.exec(statement).all()
    total = session.exec(select(Item).count()).scalar()
    return {"total": total, "items": results, "skip": skip, "limit": limit}
```

### **7. Mejores Prácticas**
- **Valores predeterminados razonables:** Define un `limit` por defecto para evitar sobrecargar el servidor.
- **Máximo permitido:** Establece un valor máximo para `limit` si necesitas evitar cargas excesivas.
  
  ```python
  limit: int = Query(10, ge=1, le=100)
  ```
  
- **Filtros adicionales:** Agrega parámetros de consulta para buscar o filtrar los datos.

### **Conclusión**
La implementación de paginación en FastAPI es directa, ya sea con datos simulados o bases de datos. Aprovecha los modelos de Pydantic para estructurar las respuestas y las consultas SQL para optimizar el rendimiento en bases de datos.

### Resumen

Para manejar grandes volúmenes de datos de forma eficiente, la paginación es esencial. Esta técnica permite responder rápidamente al usuario mostrando solo una porción del conjunto total de datos y cargando más solo cuando se solicita.

### ¿Por qué la paginación mejora el rendimiento?

Cuando se trabaja con muchos datos, los endpoints pueden tardar en responder, ya que la query puede traer más información de la que el usuario necesita visualizar en ese momento. Con la paginación:

- Se muestra solo una cantidad limitada de datos por página.
- La URL incluye parámetros que permiten cambiar la cantidad y posición de datos mostrados.
- Esto optimiza los tiempos de carga y la experiencia del usuario, quien verá únicamente la información solicitada sin sobrecargar el sistema.

### ¿Cómo implementar paginación en FastAPI?

FastAPI facilita la implementación de paginación mediante parámetros en la URL y el modelo SQL de CBT. Para probarlo, puedes usar el archivo `create_multiple_transactions` disponible en el repositorio, que permite generar datos de prueba.

1. **Preparación de datos de prueba**:

 - Crea un cliente y luego, en un bucle, genera transacciones asociadas a ese cliente.
 - Aumenta el valor de cada transacción de manera secuencial para facilitar las pruebas.
 - Realiza el commit de todos los registros creados.

2. **Configuración del endpoint de transacciones**:

 - Ve al endpoint de transactions y ajusta el código para listar las transacciones.
 - Incluye los parámetros de `skip` y `limit` en la URL para definir el número de registros omitidos y el límite de registros mostrados.
 
### ¿Qué son los parámetros `skip` y `limit`?

 - **skip**: Define cuántos registros se omitirán desde el inicio. Para la primera página, este valor será cero, ya que no se omite ningún dato.
- **limit**: Determina el máximo de registros devueltos en cada página. Puedes establecerlo a diez, veinte o cualquier número deseado.

Ambos parámetros se configuran como queries en la URL y permiten ajustar la cantidad de datos mostrados y omitidos según la página solicitada.

### ¿Cómo se aplican los parámetros en la consulta?

Para que `skip` y `limit` funcionen:

- Utiliza `offset` en el select de la query, pasando el valor de `skip`.
- Configura `limit` con el valor máximo de registros a devolver.
- Guarda los cambios y prueba el endpoint; en la documentación de FastAPI, verás los nuevos campos `skip` y `limit`.

### ¿Cómo probar la paginación en FastAPI?

1. Ejecuta la aplicación y abre el endpoint en la documentación de FastAPI.
2. Prueba el parámetro limit para ver diferentes cantidades de resultados en cada ejecución.
3. Modifica el valor de skip para saltar registros y obtener datos de páginas posteriores.

La paginación permite consultar grandes volúmenes de datos de forma segmentada, optimizando el rendimiento y la experiencia del usuario al mostrar

## Implementación y Uso de Middlewares en FastAPI

### **Implementación y Uso de Middlewares en FastAPI**

En **FastAPI**, los *middlewares* son componentes que se ejecutan antes y después de que cada solicitud sea manejada por un endpoint. Son útiles para tareas como autenticación, registro de solicitudes, manejo de errores, compresión y más.

---

### **1. ¿Qué es un Middleware?**

Un middleware es una función o clase que intercepta las solicitudes HTTP entrantes y las respuestas salientes. Esto permite realizar operaciones personalizadas, como:

- Registrar logs de las solicitudes.
- Verificar tokens de autenticación.
- Modificar solicitudes o respuestas.

---

### **2. Crear un Middleware Básico**

En FastAPI, los middlewares se implementan como clases que heredan de `BaseHTTPMiddleware` o se definen como funciones personalizadas.

#### **Ejemplo básico: Registrar Logs**

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Antes de procesar la solicitud
        print(f"Request: {request.method} {request.url}")
        
        # Procesar la solicitud
        response = await call_next(request)
        
        # Después de procesar la solicitud
        print(f"Response status: {response.status_code}")
        return response

# Agregar el middleware a la aplicación
app.add_middleware(LogMiddleware)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
```

---

### **3. Uso de Middlewares Incorporados**

FastAPI permite integrar middlewares ya existentes de **Starlette** o externos, como:

- **CORS (Cross-Origin Resource Sharing):** Para permitir que dominios externos accedan a la API.
- **GZipMiddleware:** Para comprimir respuestas grandes.

#### **CORS Middleware**
Permite configurar qué dominios pueden hacer solicitudes a tu API.

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos (GET, POST, etc.)
    allow_headers=["*"],  # Encabezados permitidos
)

@app.get("/")
async def read_root():
    return {"message": "Hello, CORS!"}
```

#### **GZip Middleware**
Comprime las respuestas para mejorar el rendimiento.

```python
from starlette.middleware.gzip import GZipMiddleware

app = FastAPI()

app.add_middleware(GZipMiddleware)

@app.get("/")
async def read_root():
    return {"message": "This response is compressed!"}
```

---

### **4. Middleware Personalizado con Validaciones**

Supongamos que deseas validar un encabezado personalizado en cada solicitud.

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class HeaderValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if "x-custom-header" not in request.headers:
            raise HTTPException(status_code=400, detail="Missing x-custom-header")
        
        # Pasar al siguiente middleware o endpoint
        response = await call_next(request)
        return response

app = FastAPI()

app.add_middleware(HeaderValidationMiddleware)

@app.get("/")
async def read_root():
    return {"message": "Header is valid!"}
```

---

### **5. Orden de los Middlewares**

El orden en el que los middlewares son añadidos es importante, ya que se ejecutan en secuencia. Por ejemplo:

```python
app.add_middleware(MiddlewareA)
app.add_middleware(MiddlewareB)
```

En este caso:

1. `MiddlewareA` se ejecuta primero al procesar la solicitud.
2. `MiddlewareB` se ejecuta después al procesar la solicitud.
3. Durante la respuesta, el orden es inverso: `MiddlewareB` → `MiddlewareA`.

---

### **6. Middleware vs. Dependencias**

Aunque los middlewares son globales y afectan todas las solicitudes, las dependencias en FastAPI ofrecen una forma más granular de validar o modificar solicitudes por endpoint.

Usa middlewares para tareas generales y dependencias para validaciones específicas.

---

### **7. Ejemplo Completo: Middleware de Tiempo de Ejecución**

Calcula cuánto tiempo tarda cada solicitud en ser procesada.

```python
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class TimerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

app = FastAPI()

app.add_middleware(TimerMiddleware)

@app.get("/")
async def read_root():
    return {"message": "Check the X-Process-Time header for timing info."}
```

---

### **Conclusión**

Los middlewares en FastAPI son una herramienta poderosa para gestionar solicitudes y respuestas a nivel global. Puedes usarlos para optimizar tu API con tareas como autenticación, registro, compresión y validaciones. Asegúrate de combinarlos sabiamente con dependencias para una arquitectura eficiente.

### Resumen

Los middlewares en FastAPI son herramientas fundamentales para modificar el comportamiento de las requests de forma centralizada y eficiente. Un middleware, en términos sencillos, es una función que se ejecuta antes y después de cada request, permitiendo interceptar y extender la funcionalidad base de una API sin modificar cada endpoint individualmente.

### ¿Cómo funcionan los middlewares en FastAPI?

Un middleware captura cada request entrante, procesa alguna funcionalidad, y luego permite que el flujo continúe hacia el endpoint correspondiente. FastAPI ofrece varios middlewares predefinidos para casos comunes, pero también permite crear middlewares personalizados para necesidades específicas.

### ¿Cómo implementar un middleware personalizado en FastAPI?

Para crear un middleware personalizado, podemos definir una función asíncrona que registre el tiempo de procesamiento de cada request. Los pasos son:

- En el archivo `main.py`, donde se define la aplicación, agregar una nueva función `log_request_time`.
- Esta función recibe dos parámetros:
 - `request`: El objeto que contiene la información del request.
 - `call_next`: Una función que llama a la siguiente operación en la cadena de requests.
 
El objetivo es registrar el tiempo antes y después de ejecutar el request, y luego calcular cuánto tiempo tomó el proceso completo.

**Código de ejemplo**

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()  # Tiempo inicial del request
    response = await call_next(request)  # Llama a la funcionalidad del request
    process_time = time.time() - start_time  # Calcula el tiempo total de procesamiento
    
    # Imprime la URL y el tiempo de procesamiento en segundos
    print(f"Request {request.url} completed in {process_time:.2f} seconds")
    
    return response
```

### ¿Cómo registrar el middleware en la aplicación?

Para registrar el middleware, es necesario utilizar `app.middleware` y especificar el tipo. En este caso, usaremos el tipo `http` para que se ejecute con todos los requests HTTP.

- Registrar `log_request_time` en la aplicación con `@app.middleware("http")`.
- Importar `time` al inicio del archivo para calcular los tiempos de inicio y procesamiento.

### ¿Cómo interpretar los resultados del middleware?

Al ejecutar la aplicación y realizar un request, el middleware imprime en la consola la URL del endpoint y el tiempo de procesamiento en segundos. Este log ayuda a entender el rendimiento de los endpoints y a identificar posibles cuellos de botella.

### ¿Cuál es el reto de implementación?

Como reto, se sugiere crear un middleware adicional que imprima en la consola todos los headers enviados en cada request. Este ejercicio permite visualizar la información de los headers y comprender mejor la estructura de las requests.

## Pruebas Unitarias en FastAPI: Configuración con Pytest y SQLAlchemy

Realizar pruebas unitarias en FastAPI utilizando **Pytest** y **SQLAlchemy** permite validar el correcto funcionamiento de tus endpoints y la lógica de negocio, garantizando que tu aplicación se comporte como se espera. A continuación, te muestro cómo configurar un entorno de pruebas con estas herramientas:

---

## **Pasos para Configurar Pruebas Unitarias en FastAPI**

### **1. Instalación de Dependencias**
Asegúrate de instalar las librerías necesarias:

```bash
pip install pytest pytest-asyncio sqlalchemy
```

---

### **2. Configurar la Base de Datos para Pruebas**
Crea una base de datos en memoria para que tus pruebas no afecten los datos reales de producción.

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# Base de datos en memoria para pruebas
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Inicialización de la base de datos
def init_test_db():
    Base.metadata.create_all(bind=engine)

def drop_test_db():
    Base.metadata.drop_all(bind=engine)
```

---

### **3. Crear un Cliente de Pruebas para FastAPI**
FastAPI incluye un cliente de pruebas basado en **Starlette** para interactuar con la aplicación.

```python
from fastapi.testclient import TestClient
from myapp.main import app  # Importa tu aplicación FastAPI

# Crea un cliente de pruebas
client = TestClient(app)
```

---

### **4. Configuración de Sesiones de Base de Datos para Pruebas**
Sobrescribe la dependencia de base de datos para que utilice la base de datos en memoria.

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from myapp.database import get_db  # Tu dependencia real de la base de datos
from myapp.models import Base

# Crea una sesión de prueba
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Sobrescribe la dependencia en tu aplicación
app.dependency_overrides[get_db] = override_get_db

# Inicializa la base de datos de pruebas
init_test_db()
```

---

### **5. Escribir Pruebas Unitarias**
Crea un archivo de pruebas (por ejemplo, `test_app.py`) y escribe tus pruebas.

#### **Ejemplo: Prueba de Creación de un Registro**
```python
from myapp.schemas import CustomerCreate

def test_create_customer():
    # Datos de prueba
    data = {"name": "John Doe", "email": "johndoe@example.com"}

    # Llama al endpoint para crear un cliente
    response = client.post("/customers/", json=data)

    # Verifica el código de respuesta y los datos retornados
    assert response.status_code == 201
    assert response.json()["name"] == data["name"]
    assert response.json()["email"] == data["email"]
```

#### **Ejemplo: Prueba de Consulta de un Registro**
```python
def test_get_customer():
    # Crea un cliente
    data = {"name": "Jane Doe", "email": "janedoe@example.com"}
    client.post("/customers/", json=data)

    # Consulta el cliente creado
    response = client.get("/customers/1")

    # Verifica que los datos sean correctos
    assert response.status_code == 200
    assert response.json()["name"] == "Jane Doe"
    assert response.json()["email"] == "janedoe@example.com"
```

---

### **6. Ejecutar las Pruebas**
Ejecuta las pruebas con Pytest desde la línea de comandos:

```bash
pytest -v
```

---

## **Prácticas Recomendadas**
1. **Usa Fixtures de Pytest**: Define datos o configuraciones comunes que puedan reutilizarse entre múltiples pruebas.
   ```python
   import pytest

   @pytest.fixture
   def test_client():
       init_test_db()
       yield client
       drop_test_db()
   ```

2. **Aislar Pruebas**: Asegúrate de que cada prueba se ejecute de manera independiente, sin depender del estado de otras pruebas.

3. **Validar Errores**: Además de probar casos exitosos, incluye pruebas para errores esperados (por ejemplo, registro no encontrado).

4. **Limpiar la Base de Datos**: Si no usas una base en memoria, asegúrate de limpiar los datos después de cada prueba.

---

### **Conclusión**
Este enfoque asegura que tus endpoints en FastAPI sean robustos y se comporten correctamente. Puedes extender este modelo para probar funcionalidades más avanzadas, como autenticación, relaciones de modelos, y manejo de excepciones.

### Resumen

Configurar pruebas unitarias en FastAPI permite evaluar endpoints y lógica de negocio de manera eficiente sin comprometer el entorno de producción. A continuación, se describe cómo iniciar un entorno de pruebas en FastAPI usando Pytest y configurando una base de datos temporal en memoria, ideal para entornos de testing.

### ¿Cómo configurar el archivo de pruebas inicial en FastAPI?

Para empezar, crea un archivo conftest.py, que contendrá las configuraciones principales de prueba. En este archivo:

- **Importa Pytest** como framework base para gestionar y ejecutar las pruebas.
- `Desde FastAPI`, importa `TestClient`, que simula las acciones de un navegador para realizar peticiones `POST`, `GET` y otros métodos HTTP.

Este archivo servirá para centralizar las configuraciones de prueba, separadas del código de producción.

### ¿Por qué es importante usar una base de datos específica para pruebas?

Utilizar la base de datos de producción en pruebas puede provocar conflictos y pérdida de datos. Por eso:

- **Configura una base de datos en memoria** mediante `SQLAlchemy`, usando `create_engine` y estableciendo `check_same_thread=False` para evitar errores en entornos multihilo.
- Utiliza `StaticPool` para evitar la creación de múltiples instancias de bases de datos, asegurando un entorno limpio para cada prueba.

### ¿Cómo definir sesiones de base de datos de prueba?

Al configurar pruebas, es esencial crear sesiones de base de datos específicas:

- Crea una **fixture** llamada `session_fixture` usando `metadata.create_all(engine)`. Esto genera las tablas necesarias en la base de datos de prueba.
- Usa `yield` para devolver la sesión a las pruebas que lo requieran y ejecutar `drop_all` al finalizar, lo cual elimina las tablas y optimiza la memoria.

### ¿Cómo sobreescribir dependencias en FastAPI?

Para que las pruebas no utilicen la base de datos de producción:

1. Importa la aplicación (app) desde el archivo principal.
2. Define un método para sobrescribir la dependencia get_session. Esto permite que las pruebas utilicen la sesión de la base de datos de testing.
3. Usa dependency_overrides de FastAPI para redirigir las dependencias de la app a las que corresponden al entorno de pruebas.

### ¿Cómo se implementa el `TestClient` en una fixture?

Crear una fixture de `TestClient` permite simular peticiones HTTP:

- Declara una fixture llamada `client` que retorna un `TestClient` configurado con la app y las dependencias sobrescritas.
- Con `yield client`, puedes usar el cliente de prueba en cualquier test. Una vez finalizado, limpia las dependencias con `dependency_overrides.clear()`.

### ¿Cómo ejecutar una prueba simple con Pytest?

Para verificar la configuración de tu cliente de prueba:

1. Crea un archivo `test.py`.
2. Define una prueba que reciba el `client` como parámetro. Pytest lo inyectará automáticamente gracias a las fixtures.
3. Usa `assert` para validar el comportamiento, como el tipo de cliente.

Ejecuta las pruebas con `pytest` desde la terminal. Si la configuración es correcta, verás un indicador de éxito en verde.

### ¿Cuál es el reto adicional para mejorar las pruebas?
Como desafío, intenta crear una nueva **fixture** que genere un `customer`. Esto permitirá que futuros tests puedan reutilizar este customer como dato inicial.

## ¿Cómo implementar pruebas automáticas en endpoints de FastAPI?

Implementar pruebas automáticas para los endpoints de **FastAPI** implica automatizar la validación de los resultados esperados cuando se interactúa con la API. Esto se logra utilizando herramientas como **Pytest** y el cliente de pruebas integrado de FastAPI basado en Starlette.

### **Pasos para Implementar Pruebas Automáticas**

### **1. Instalación de Dependencias**

Primero, instala las herramientas necesarias para realizar pruebas:

```bash
pip install pytest pytest-asyncio httpx
```

### **2. Configurar el Cliente de Pruebas**

FastAPI proporciona un cliente de pruebas para interactuar con la API de manera programada. Esto permite realizar solicitudes HTTP simuladas dentro del entorno de pruebas.

```python
from fastapi.testclient import TestClient
from myapp.main import app  # Asegúrate de reemplazar `myapp.main` por la ubicación de tu aplicación

client = TestClient(app)
```

### **3. Escribir Pruebas para los Endpoints**

Las pruebas se escriben en archivos nombrados como `test_*.py`. Cada función de prueba debe comenzar con `test_`.

#### **Ejemplo: Endpoint de Creación de Usuario**
```python
def test_create_user():
    # Datos de prueba
    data = {"username": "testuser", "email": "testuser@example.com", "password": "securepassword"}

    # Realizar una solicitud POST al endpoint
    response = client.post("/users/", json=data)

    # Validar la respuesta
    assert response.status_code == 201
    assert response.json()["username"] == data["username"]
    assert response.json()["email"] == data["email"]
```

#### **Ejemplo: Endpoint de Obtención de Usuario**
```python
def test_get_user():
    # Crear un usuario previamente
    data = {"username": "johndoe", "email": "johndoe@example.com", "password": "123456"}
    client.post("/users/", json=data)

    # Consultar el usuario
    response = client.get("/users/johndoe")

    # Validar los resultados
    assert response.status_code == 200
    assert response.json()["username"] == "johndoe"
    assert response.json()["email"] == "johndoe@example.com"
```

### **4. Incluir Pruebas para Casos de Error**

Es importante probar no solo los casos exitosos, sino también errores esperados.

#### **Ejemplo: Usuario No Encontrado**
```python
def test_user_not_found():
    response = client.get("/users/nonexistent")

    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"
```

#### **Ejemplo: Validación de Datos Inválidos**
```python
def test_invalid_user_creation():
    # Datos inválidos (falta el correo electrónico)
    data = {"username": "invaliduser", "password": "123456"}
    response = client.post("/users/", json=data)

    assert response.status_code == 422  # Unprocessable Entity
    assert "email" in response.json()["detail"][0]["loc"]
```

### **5. Configurar Fixtures para Pruebas**

Las **fixtures** en Pytest ayudan a configurar estados iniciales comunes para las pruebas.

#### **Ejemplo: Base de Datos Temporal**
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from myapp.database import Base, get_db

# Crear una base de datos en memoria para pruebas
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def test_client():
    # Inicializa la base de datos
    Base.metadata.create_all(bind=engine)

    # Sobrescribe la dependencia de la base de datos
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield client

    # Limpia la base de datos después de las pruebas
    Base.metadata.drop_all(bind=engine)
```

Usa esta fixture en tus pruebas:

```python
def test_create_user_with_fixture(test_client):
    data = {"username": "janedoe", "email": "janedoe@example.com", "password": "mypassword"}
    response = test_client.post("/users/", json=data)

    assert response.status_code == 201
    assert response.json()["username"] == data["username"]
```

### **6. Ejecutar las Pruebas**

Ejecuta todas las pruebas con el siguiente comando:

```bash
pytest -v
```

### **7. Cobertura de Pruebas (Opcional)**

Para asegurarte de que todas las partes críticas de tu código están siendo probadas, puedes usar **Coverage.py**:

```bash
pip install pytest-cov
pytest --cov=your_package_name tests/
```

### **Buenas Prácticas**
1. **Aislar Pruebas**: Usa una base de datos en memoria o limpia los datos entre pruebas.
2. **Automatizar Ejecución**: Integra las pruebas con un pipeline de CI/CD como GitHub Actions, GitLab CI o Jenkins.
3. **Probar Casos Límites**: Incluye pruebas con datos extremos, valores faltantes o entradas mal formateadas.
4. **Validar Autenticación**: Si tu API utiliza autenticación, asegúrate de incluir pruebas para usuarios autenticados y no autenticados.

### **Conclusión**

Con estas herramientas y pasos, puedes implementar pruebas automáticas robustas para tus endpoints en FastAPI. Esto mejora la calidad de tu aplicación y te permite detectar errores rápidamente antes de que lleguen a producción.

**Nota**: ejecutar test `pytest app\tests\tests_customers.py` y para ver si paso se usa ` pytest app\tests\tests_customers.py -v` nuestra el nombre y si paso o no

### Resumen

Las pruebas en el desarrollo de software son cruciales para validar que la creación, edición y eliminación de recursos, en este caso “customers”, funciona correctamente. Al implementar pruebas automáticas, se identifican errores potenciales sin tener que revisar cada funcionalidad manualmente. En esta clase, desarrollamos pruebas específicas para asegurar que los endpoints de “customers” operen como se espera.

### ¿Cómo organizar y crear las pruebas?

Para organizar las pruebas, se recomienda crear una carpeta dedicada dentro de la aplicación. Esta carpeta contiene un archivo `__init__.py` para que funcione como módulo y permita agrupar todas las pruebas. Si el módulo de routers se llama `customers`, el archivo de pruebas se nombra `test_customers.py` para reflejar su propósito.

### ¿Cómo implementar una prueba de creación de customers?

1. `Configura el cliente`: Usa el cliente previamente creado para hacer un `POST` al endpoint de `customers`.
2. `Define los datos`: En el `POST`, envía un JSON con datos necesarios para crear un cliente, como nombre, email y edad.
3. **Verifica el status code**: Al ejecutar la prueba, usa `assert` para confirmar que el código de estado es 201 (creado), validando que la creación fue exitosa.
4. **Ejecuta la prueba**: Utiliza el comando `pytest` en la terminal, especificando la ruta del archivo de pruebas.

Es importante recordar el uso de `assert`, ya que sin él, la prueba se ejecuta pero no verifica nada. Si falta un `assert`, se podría pasar por alto un fallo en el código.

### ¿Cómo solucionar errores en las pruebas?

Durante la ejecución, puede aparecer un error si el código de estado no coincide con lo esperado. Por ejemplo, si se devuelve un 200 en lugar de un 201, esto indica que falta especificar el `status_code` correcto en el decorador del endpoint.

Para solucionarlo:

1. **Ajusta el endpoint**: Modifica el decorador de la ruta para que devuelva el `status_code` esperado, `201 Created`.
2. **Reejecuta la prueba**: Confirma que ahora la prueba pasa sin errores.

### ¿Cómo crear pruebas de lectura de customers?

Después de la creación, se implementa una prueba de lectura (`GET`) para verificar que los datos del cliente puedan ser recuperados:

1. **Configura la solicitud `GET`**: Usa el cliente para obtener los datos del `customer` creado, incluyendo el ID único.
2. **Usa el JSON de la respuesta**: Extrae el ID del `customer` de la respuesta JSON para construir la URL del `GET`.
3. **Verifica los datos y el status code**: Confirma que el nombre del cliente y el código de estado (200) son los esperados.
4. **Ejecuta y verifica múltiples pruebas**: Usa `pytest -v` para ejecutar las pruebas y ver los nombres, útil para identificar el estatus de cada prueba.

### ¿Qué beneficios aportan estas pruebas en la aplicación?

Implementar pruebas automáticas no solo verifica el funcionamiento de los endpoints sino que permite identificar si cambios en otros módulos, como facturación o transacciones, afectan inadvertidamente a `customers`. Así, al realizar modificaciones, las pruebas protegen la funcionalidad de la aplicación en su conjunto.

## Autenticación de API en FastAPI con HTTPBasicCredentials

La autenticación en FastAPI utilizando `HTTPBasicCredentials` es una forma sencilla de proteger los endpoints de tu API mediante autenticación básica HTTP. Este enfoque es útil para aplicaciones que no requieren un sistema de autenticación complejo o para pruebas iniciales.

### **Pasos para Implementar Autenticación con `HTTPBasicCredentials`**

### **1. Importar las Dependencias Necesarias**

FastAPI ofrece soporte nativo para la autenticación básica mediante el módulo `fastapi.security`.

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets  # Para comparar credenciales de forma segura
```

### **2. Crear una Instancia de HTTPBasic**

Esto se usa para obtener las credenciales enviadas por el cliente.

```python
security = HTTPBasic()
```

### **3. Configurar un Dependency para Validar Credenciales**

Crea una función que actúe como dependencia para validar las credenciales proporcionadas.

```python
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    valid_username = "admin"
    valid_password = "password123"

    # Comparar credenciales de manera segura
    is_correct_username = secrets.compare_digest(credentials.username, valid_username)
    is_correct_password = secrets.compare_digest(credentials.password, valid_password)

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username  # Retorna el nombre del usuario autenticado
```

### **4. Proteger Endpoints con la Dependencia**

Usa la función `authenticate` como dependencia en los endpoints que deseas proteger.

```python
app = FastAPI()

@app.get("/secure-data/")
def get_secure_data(username: str = Depends(authenticate)):
    return {"message": f"Welcome, {username}! This is your secure data."}
```

### **5. Probar la Autenticación**

#### **Cliente HTTP**
Puedes usar herramientas como **cURL**, **Postman**, o un navegador para probar la autenticación.

#### **Ejemplo con cURL**:
```bash
curl -u admin:password123 http://127.0.0.1:8000/secure-data/
```

Si las credenciales son correctas, recibirás:

```json
{
  "message": "Welcome, admin! This is your secure data."
}
```

Con credenciales incorrectas:

```json
{
  "detail": "Invalid username or password"
}
```

### **6. Mejoras y Buenas Prácticas**

1. **Uso de Variables de Entorno**: No almacenes credenciales en el código. Usa variables de entorno o un sistema de configuración segura como `python-decouple` o `dotenv`.

   ```python
   import os
   valid_username = os.getenv("BASIC_AUTH_USERNAME", "admin")
   valid_password = os.getenv("BASIC_AUTH_PASSWORD", "password123")
   ```

2. **Soporte para Múltiples Usuarios**: Si necesitas autenticar a múltiples usuarios, puedes usar un diccionario:

   ```python
   USERS = {
       "admin": "password123",
       "user1": "securepass",
   }

   def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
       if credentials.username not in USERS or not secrets.compare_digest(USERS[credentials.username], credentials.password):
           raise HTTPException(
               status_code=status.HTTP_401_UNAUTHORIZED,
               detail="Invalid username or password",
               headers={"WWW-Authenticate": "Basic"},
           )
       return credentials.username
   ```

3. **SSL/TLS Obligatorio**: Nunca uses autenticación básica sin HTTPS, ya que las credenciales se envían en texto claro.

4. **Token en Lugar de Contraseñas**: Para mayor seguridad, considera usar un sistema de tokens en lugar de autenticación básica si planeas expandir la funcionalidad.

### **7. Integrar con Middleware**

Si deseas proteger toda la aplicación o ciertos grupos de rutas, puedes usar un middleware personalizado.

#### **Middleware para Rutas Protegidas**
```python
@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/secure"):  # Proteger solo rutas específicas
        credentials = security(request)
        authenticate(credentials)
    return await call_next(request)
```

### **8. Alternativas**

Si planeas implementar un sistema de autenticación más robusto, considera:
- **JWT (JSON Web Tokens)** para APIs modernas.
- **OAuth2** con FastAPI, que proporciona soporte nativo para flujos OAuth2.

Con estos pasos, puedes implementar autenticación básica HTTP en FastAPI de manera rápida y efectiva, ideal para casos sencillos o como punto de partida para sistemas más avanzados.

### Resumen

Proteger los datos es esencial en el desarrollo de APIs. FastAPI ofrece múltiples mecanismos para asegurar nuestros endpoints; entre ellos, el uso de autenticación con usuario y contraseña, que es rápido y sencillo. En esta guía, exploramos cómo implementar autenticación básica HTTP para proteger un endpoint específico, configurando dependencias y validando credenciales de usuario.

### ¿Cómo implementar autenticación básica en FastAPI?

- FastAPI facilita la autenticación básica mediante la clase `HTTPBasicCredentials`. Esta clase permite solicitar un nombre de usuario y una contraseña al intentar acceder a un endpoint específico.
- Primero, se define una dependencia en el archivo `main`. Una dependencia en FastAPI es un parámetro que permite verificar si el valor es ingresado por el usuario o es parte de un flujo interno.
- Se declara una variable `credentials` del tipo Depends y se especifica como dependencia de seguridad (`security`) usando `HTTPBasic`.

### ¿Cómo se configura el código para proteger un endpoint?

1. **Definición de Dependencias**: Se inicia importando `HTTPBasic` desde el módulo `security` de FastAPI y configurando `Depends` para utilizarlo como dependencia.
2. **Configuración de la Función**: En la función del endpoint, se añade `credentials: HTTPBasicCredentials` como parámetro. Esto permite recibir y manejar las credenciales dentro del método.
3. **Autocompletar Variables**: FastAPI admite anotaciones que mejoran la autocompletación de variables en el IDE, facilitando la codificación de dependencias y asegurando la ejecución sin errores.

### ¿Cómo probar la autenticación en la documentación de FastAPI?

- Al ejecutar la aplicación, la documentación autogenerada de FastAPI muestra un botón de autorización. Al hacer clic, se solicita ingresar un usuario y contraseña.
- Cuando el endpoint está protegido, aparece un candado en el endpoint de la documentación, indicando que requiere autenticación.
- Una vez autorizado, los valores ingresados se almacenan en la variable `credentials`.

### ¿Cómo validar las credenciales en FastAPI?

- Dentro de la función, se valida si el `username` y `password` son correctos.
- Se utiliza un condicional `if` para verificar si las credenciales coinciden con un usuario autorizado. Si es así, se devuelve un mensaje de bienvenida. De lo contrario, se lanza una excepción `HTTPException` con un código `401 Unauthorized`.
- Este control permite garantizar que solo usuarios con credenciales válidas puedan acceder al contenido del endpoint.

### ¿Cómo simular una conexión con la base de datos para validar usuarios?

- Con el nombre de usuario capturado, es posible conectarse a una base de datos para confirmar su existencia y permisos de acceso.
- Una vez validado, se muestra la información privada correspondiente.
- Este flujo es útil para extender la lógica de autenticación a otros endpoints y realizar búsquedas de usuario en la base de datos.

**Lecturas recomendadas**

[FastAPI Template](https://github.com/fastapi/full-stack-fastapi-template)