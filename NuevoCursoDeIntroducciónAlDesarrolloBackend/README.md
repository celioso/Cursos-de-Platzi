### Curso de Introducción al Desarrollo Backend

**Los roles del desarrollo backend**
Tu rol principal como Backend Developer va ser escribir código que tengan que ver con:

- Reglas de negocio
- Validación
- Autorización de usuarios
- Conexiones a bases de datos
- Código que correra del lado del servidor.

El Backend developer también puede estár acercado a otro tipo de roles como:

**DB ADMIN**
Gestiona una base de datos, sus políticas y como vamos a disponer de esa DB a través del código y la seguridad que estas deberian tener.

**SERVER ADMIN**
Se encarga de gestionar la seguridad en los servidores que es donde corre el código a disposición.

### Frontend, Backend y Full Stack

La gran comparativa es algo así:

**Fronted** = Son los Avengers (todo el mundo ve su trabajo y cómo lo hacen)
Backend = Los Hombres de Negro (nadie ve su trabajo pero si no lo hacen el planeta se destruye)
**Full Stack** = El Inspector Gadget o James Bond 007 (tiene una infinidad de recursos para sacar todo adelante y evitar que el proyecto o la trama de caiga)

**Frontend, Backend y Full Stack**
- **Fronted (Cliente)**: Se enfoca en la parte del renderizado y lo que se muestra al cliente. Clientes mas populares:
 - **Navegadores**: hacen solicitudes HTML o pueden enviar datos y soportan;
   	HTML; Markdown
 	CS; Tailwind, Bootstrap, Foundation
   	JS; React, Angular , Vue
 - **APP mobile**: Pueden ser de android o iOS, q se conectan a un servicio de backend para solicitar datos y luego renderizarlo. Soportan:
	IOS; Swift, Objetive C
	Android; Kotlin, JAVA
	Cross Plataform; React Native, Flutter, .NET MAUI
 - **IOT**: Podemos conectar dispositivos que envíen datos a un SV para luego visualizarlo
- **Backend (Server):** Se enfoca en desarrollar servicios que se conectan a un frontend o sea un cliente a través de una API (application programming interface). Existen múltiples lenguajes de programación con su respectivo framework que se utilizan para el desarrollo backend;
- Python → Django
- JavaScript → Express.js
- TypeScript → NestJS
- Ruby → Ruby on Rails
- PHP → Laravel
- Java → Spring
- Go → Gin
- C# → .NET
- **FullStack developer**: Es un dev que desempeña funciones de frontend y backend, pero generalmente tiene una especialidad en la que ejerce una mayor profundidad de conocimiento.
💡 El ingeniero en “forma T” es una analogía que sugiere que el conocimiento del desarrollador debería graficarse en forma de una letra T, puesto que el conocimiento de su especialidad debe alcanzar un mayor nivel de profundidad, mientras que superficialmente entiende otras tecnologías que complementan su punto mas fuerte.

![](https://ecdisis.com/wp-content/uploads/2021/01/WhatsApp-Image-2021-07-30-at-11.03.53.jpeg)

### ¿Cómo se construye el backend?

- Los usuarios se conectan través del cliente de un dispositivo (ya sea un navegador, dispositivo móvil, etc…).
- Se realiza una solicitud en el frontend a través del cliente.
- En el listado de solicitudes, cada posible solicitud es conocida como un endpoint.
- La API (application programming interface) es la encargada de recibir la solicitud y hacerla llegar al backend, a lo que el frontend espera una respuesta.
- El backend recibe la solicitud y dispara una respuesta con el endpoint correspondiente.
- Las bases de datos porporcionan la información que requiere el backend para satisfacer la solicitud hecha por el cliente.
- Las librerías son herramientas (piezas de código) pre-fabricadas por otros desarrolladores, que pueden ser importadas al proyecto para evitar la necesidad de crear código ya existente (no hay que reinventar la rueda).
- Los framework son un conjunto de librerías que en conjunto conforman un marco de trabajo utilizado para responder a una necesidad específica existente en un proyecto.

### ¿Cómo escoger lenguajes y frameworks para backend?

Entonces: Frameworks son herramientas que nos ayuda a ir más ágil y desarrollar nuestro proyecto en el dia a dia. entre ellos tenemos para Python:

- Django
- Flask
- FastAPI

En JavaScript

- Express
- NextJS

En PHP

- Laravel
- Symphony

En Java
- Spring

En Go

- Gin

En Rubi

- Ruby Rails

En C#

- .NET

![](https://i.pinimg.com/originals/3c/9f/ff/3c9fffc1105e2c08bd0e3aff78764fe1.png)

### HTTP

HTTP Status codes:

![HTTP Status codes](https://miro.medium.com/max/920/1*w_iicbG7L3xEQTArjHUS6g.jpeg)

![Status Code](https://www.steveschoger.com/status-code-poster/img/status-code.png "Status Code")

[http cats](https://http.cat/ "http cats")
Protocolo HTTP y como interactúa con el Backend


- Protocolo: Aqui se muestra el protocolo donde hacemos la peticion, normalemente es HTPPS la S es un certificado de seguridad.

- Dominio: normalmente .com pero hay un monton de estas terminaciones (.dev .gg .org)

- PATH: Aqui nos encontramos las diferentes rutas dentro de nuestro sitio web

![Protocolo HTTP](https://static.platzi.com/media/user_upload/Untitled-fc4d2903-73ae-4f69-8257-dd0d94ec1710.jpg "Protocolo HTTP")

### ¿Qué son las APIs?

Las APIs nos permiten, a través de código, la comunicación entre sistemas. Como backend developers, nos interesan las APIs que son servicio web y corren en el protocolo HTTP. La API utiliza una lista de rutas conocidas como endpoints, que provee las respuestas a las solicitudes del cliente. La solicitud debe ser empaquetada y retornada, y existen distintos tipos de empaquetado: JSON. XML.

![Cómo funciona una API:](https://lvivity.com/wp-content/uploads/2018/07/how-api-work.jpg "Cómo funciona una API:")

![Que es una API](https://static.platzi.com/media/user_upload/7dc50204-6f44-4000-bfe3-1d8677bab50c-caf12dbc-7331-44d8-9131-c2523f325d08.jpg "Que es una API")

### Estructura REST API

- API REST es un estandar para desarrollar APIs que funcionan en el protocolo HTTP .
- A través de los endpoints se le pide información al dominio, por lo general, se nos devuelve la información empaquetada en un JSON.
- CRUD es el índice de unas plabras clave, y en el protocolo HTTP tenemos métodos para llevarlas a cabo:
 - Create (crear) → POST.
 - Read (leer) → GET.
 - Update (actualizar) → PUT / PATCH.
 - Delete (eliminar) → DELETE.
- Put envía la totalidad de los datos, mientras que patch envía solo los datos destinados a actualizarse.

![Endpoint](https://i.ibb.co/dgYfBMw/i-Screen-Shoter-20221220115650725.png "Endpoint")

### Insomnia y Postman

Exploré los métodos GET, POST y PUT, pero por algún motivo no pude utilizar DELETE con Insomnia y la API. Tampoco soporta PATCH, por lo que veo.

- Insomnia y Postman son ambos software especializados que te permiten la exploración de APIs en tiempo real.
- Las peticiones GET (lectura) no tienen información para enviar en el body.
- Insomnia te permite modificar el nombre de las peticiones para identificarlas.
- Si envías un identificador inexistente en la base de datos, recibirás un código de estado 404 NOT FOUND.
- Para enviar una petición con el método POST, debes adjuntar la información en el body. Insomnia te permite modificar todos estos parámetros.
- El identificador único lo asigna automátocamente la fakeAPI de Platzi.
- Una tarea importante del backend es asegurar y validad la integridad de la información. Si no envías toda la información necesaria de una categoría, recibirás un código de estado 400 BAD REQUEST.
- Una vez modifiques una categoría en la API conn PUT o PATCH, puedes consultarla con normalidad a través del método GET.

[The API Design Platform and API Client - Insomnia](https://insomnia.rest/)

[Postman](https://www.postman.com/ "Postman")