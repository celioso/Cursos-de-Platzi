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
django-admin startproject mi_proyecto
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