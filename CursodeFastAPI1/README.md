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