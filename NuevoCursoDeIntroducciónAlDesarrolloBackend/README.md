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

### La nube

- La nube es donde se alojan los servicios (código) para exponerlos, y que los clientes puedan hacer request (peticiones). Ofrecen servicios para distribuir y desplegar aplicaciones.
- La nube no está en el cielo, no es literalmente una nube, es la computadora de alguien más (tu proveedor). Tinen un CPU, una RAM y un SSD, como cualquier otra computadora.
- Los proveedores poseen granjas de servidores configurados (data centers) ubicadas en diferentes lugares del mundo.
- Mientras más cerca estés al data center que te conectes, experimentarás menor latencia, por lo que los recurso llegarán con mayor rapidez.
- Como parte de esa "nube", según la que escojas, puedes tener tu sistema replicado en diferentes lugares, y elegir en dónde estarán tus sistemas corriendo, y optimizarlos para desplegar tu aplicación.

![Azure VS AWS VS Google Computer](https://static.platzi.com/media/user_upload/comparacion%20nubes-db72aad2-9ee8-4a31-b3dc-56e4b0d14e31.jpg "Azure VS AWS VS Google Computer")

### DevOps

 DevOps no es un cargo o una persona, es una cultura que agrupa una serie de prácticas y principios para mejorar y automatizar los procesos entre los equipos de desarrollo e infraestructura (IT) para hacer el lanzamiento de software de una manera rápida, eficiente y segura. Esta es la definición en la descripción de la escuela de DevOps y Cloud Computing en Platzi.

- Existe un workflow (flujo de trabajo) para hacer que el código trabajado de forma local llegue al servidor y exponer el servicio a miles de usuarios.
- Las aplicaciones son expuestas a muchas zonas, potencialmente a todo el mundo.
- El request del cliente tiene que estar lo más cerca posible al data center para reducir la latencia, y por ende, el tiempo de respuesta.
- Git es un sistema atómico de control de versiones utilizado para crear repositorios de código. Github es un servicio de repositorios remotos.
- Centralizamos nuestro código en un repositorio remoto (Github), al que los miembros del equipo de desarrollo equipo aportarán el código. La rama principal (main) tiene todo el código que debe cumplir estándares a través de pruebas, calidad y seguridad.
- Se denomina automation al rol de los desarrolladores que se encargan de realizar las automatizaciones para hacer las verificaciones en el código.
- El servidor de repositorios nos ayuda a reunir desarrollo y operaciones; el repositorio remoto se conecta con la nube, ambos se comunican, y si cumplen con las pruebas, calidad y seguridad, se despliega la app y nos conectamos a esos servidores.
- Así el equipo de desarrollo puede lanzar rápidamente y operar el código en producción, normalmente después se vuelve un flujo:

 - Plan (planificación).
 - Code (código).
 - Build (construcción)
 - Test (pruebas).
 - Release (lanzamiento).
 - Deploy (despliegue).
 - Operate (operar).
 - Monitor (monitorear).
- Este flujo es la cultura de trabajo conocida como DevOps.

**Algunas de las Tecnologias que se pueden usar en todo el ciclo de DepOps**

![Tecnologias DEpOps](https://i0.wp.com/geniusitt.com/wp-content/uploads/2018/08/DevOpstools-1.png?fit=1024%2C543&ssl=1 "Tecnologias DEpOps")

![DevOps Concepts](https://pbs.twimg.com/media/Fv3O-osXwBENWys?format=jpg&name=4096x4096 "DevOps Concepts")

### El servidor

SaaS: Netflix. PaaS: Windows Azure. IaaS: Amazon Web Services.

- El servidor normalmente podemos implementar diferentes tipos de arquitecturas, existen tres tipos de arquitecturas principales.
- Software as a Service (software como servicio):
 - No tienes control del estado de red, ni almacenamiento, ni los datos, ni sobre la aplicación en sí, pero puedes hacer uso de su servicio.
- Platform as a Service (plataforma como servicio):
 - Tienes mayor capacidad de administración, comienzas a gestionar la aplicación y la data. Es la mas común a encontrar como backend developer. Te permite distribuir tu aplicación, el código y controlar la data de la misma (bases de datos).
- Infrastructure as a Service (infraestructura como servicio):
 - Nos permite una todavía mayor capacidad de manejo. Tenemos que gestionar la aplicación, los datos, pero tambipen el runtine, el sistema operativo, etcétera. El proveedor sigue manejando la virtualización y servidores.
- On-site:
 - Te encargas absolutamente de todo, no dependes de un proveedor, sino que utilizas directamente el computador físico.

![](https://www.artifakt.com/content/uploads/2021/07/Blog-Image-CirclesGraph-1200x627-%E2%80%93-1-1024x661.png)

### Cookies y sesiones

Las cookies son pequeños fragmentos de texto que los sitios web que visitas envían al navegador. Permiten que los sitios web recuerden información sobre tu visita, lo que puede hacer que sea más fácil volver a visitar los sitios y hacer que estos te resulten más útiles. Otras tecnologías, como los identificadores únicos que se usan para identificar un navegador, aplicación o dispositivo, los píxeles y el almacenamiento local, también se pueden usar para estos fines. Las cookies y otras tecnologías que se describen en esta página pueden usarse para los fines indicados más abajo.

**Funcionalidad**

Las cookies y otras tecnologías que se usan con fines de funcionalidad te permiten acceder a funciones esenciales de un servicio. Se consideran esenciales, por ejemplo, las preferencias, (como el idioma que has elegido), la información relacionada con la sesión (como el contenido de un carrito de la compra), y las optimizaciones de los productos que ayudan a mantener y mejorar ese servicio.

Algunas cookies y otras tecnologías se utilizan para recordar tus preferencias. Por ejemplo, la mayoría de las personas que usan los servicios de Google tienen en sus navegadores una cookie denominada "NID" o "ENID", dependiendo de sus opciones de cookies. Estas cookies se usan para recordar tus preferencias y otra información, como el idioma que prefieres, el número de resultados que quieres que se muestren en cada página de resultados de búsqueda (por ejemplo, 10 o 20) y si quieres que el filtro Búsqueda Segura de Google esté activado o desactivado. La cookie "NID" caduca 6 meses después del último uso del usuario, mientras que la cookie "ENID" dura 13 meses. Las cookies "VISITOR_INFO1_LIVE" y "YEC" tienen un propósito similar en YouTube y también se usan para detectar y resolver problemas con el servicio. Estas cookies tienen una duración de 6 y 13 meses, respectivamente.

Otras cookies y tecnologías se usan para mantener y mejorar tu experiencia durante una sesión concreta. Por ejemplo, YouTube utiliza la cookie "PREF" para almacenar información como la configuración que prefieres para tus páginas y tus preferencias de reproducción; por ejemplo, las opciones de reproducción automática que has seleccionado, la reproducción aleatoria de contenido y el tamaño del reproductor. En YouTube Music, estas preferencias incluyen el volumen, el modo de repetición y la reproducción automática. Esta cookie caduca 8 meses después del último uso del usuario. La cookie "pm_sess" también ayuda a conservar tu sesión del navegador y tiene una duración de 30 minutos.

También se pueden usar cookies y otras tecnologías para mejorar el rendimiento de los servicios de Google. Por ejemplo, la cookie "CGIC" mejora la generación de resultados de búsqueda autocompletando las consultas de búsqueda basándose en lo que introduce inicialmente un usuario. Esta cookie tiene una duración de 6 meses.

Google usa la cookie "CONSENT", que tiene una duración de 2 años, para almacenar el estado de un usuario respecto a sus elecciones de cookies. La cookie "SOCS", que dura 13 meses, también se usa con este mismo fin.

**Seguridad**

Las cookies y otras tecnologías que se usan con fines de seguridad ayudan a autenticar a los usuarios, prevenir el fraude y protegerte cuando interactúas con un servicio.

Las cookies y otras tecnologías que se usan para autenticar a los usuarios permiten asegurar que solo el propietario de una cuenta puede acceder a ella. Por ejemplo, las cookies "SID" y "HSID" contienen registros cifrados y firmados de forma digital del ID de cuenta de Google de un usuario y del momento de inicio de sesión más reciente. La combinación de estas cookies permite a Google bloquear muchos tipos de ataques, como, por ejemplo, intentos de robo del contenido de los formularios que se envían en los servicios de Google.

Algunas cookies y otras tecnologías se usan para prevenir el spam, el fraude y los abusos. Por ejemplo, las cookies "pm_sess", "YSC" y "AEC" se encargan de comprobar que las solicitudes que se hacen durante una sesión de navegación proceden del usuario y no de otros sitios. Estas cookies evitan que sitios maliciosos actúen haciéndose pasar por el usuario sin su conocimiento. La cookie "pm_sess" tiene una duración de 30 minutos, y la cookie "AEC", de 6 meses. La cookie "YSC" dura toda una sesión de navegación del usuario.

**Analíticas**

Las cookies y otras tecnologías que se usan con fines analíticos ayudan a recoger datos que permiten a los servicios entender cómo interactúas con un servicio en particular. Esta información se usa para mejorar el contenido de los servicios y sus funciones, y así ofrecerte una mejor experiencia.

Algunas cookies y otras tecnologías ayudan a los sitios y las aplicaciones a entender cómo interactúan los visitantes con sus servicios. Por ejemplo, Google Analytics utiliza un conjunto de cookies para recoger información y ofrecer estadísticas de uso de los sitios sin que Google identifique personalmente a cada visitante. La principal cookie que utiliza Google Analytics es "_ga", que permite a los servicios distinguir a un visitante de otro y tiene una duración de 2 años. La utilizan todos los sitios en los que se implementa Google Analytics, incluidos los servicios de Google. Cada cookie "_ga" es exclusiva de una propiedad específica, así que no se puede utilizar para rastrear a un usuario o navegador en sitios web no relacionados.

Los servicios de Google también usan las cookies "NID" y "ENID" en la Búsqueda de Google y "VISITOR_INFO1_LIVE" y "YEC" en YouTube con fines analíticos.

**Publicidad**

Google utiliza cookies con fines publicitarios, como publicar y renderizar anuncios, personalizar anuncios (según la configuración de anuncios que tenga el usuario en myadcenter.google.com y adssettings.google.com/partnerads), limitar el número de veces que se muestra un anuncio a un usuario, ocultar anuncios que el usuario ha indicado que no quiere volver a ver y medir la eficacia de los anuncios.

La cookie "NID" se usa para mostrar anuncios de Google en los servicios de Google a usuarios que no tengan la sesión iniciada, mientras que las cookies "ANID" e "IDE" se utilizan para mostrar anuncios de Google en sitios que no son de Google. Si has habilitado los anuncios personalizados, la cookie "ANID" se utiliza para recordar este ajuste y tiene una duración de 13 meses en el Espacio Económico Europeo (EEE), Suiza y el Reino Unido, y 24 meses en los demás lugares. Si los has desactivado, la cookie "ANID" se usa para almacenar ese ajuste hasta el 2030. La cookie "NID" caduca 6 meses después del último uso del usuario. La cookie "IDE" tiene una duración de 13 meses en el Espacio Económico Europeo (EEE), Suiza y el Reino Unido, y de 24 meses en los demás lugares.

En función de tu configuración de anuncios, otros servicios de Google, como YouTube, también pueden usar con fines publicitarios estas y otras cookies y tecnologías, como la cookie "VISITOR_INFO1_LIVE".

Algunas cookies y otras tecnologías que se usan con fines publicitarios se destinan a usuarios que inician sesión para usar servicios de Google. Por ejemplo, la cookie "DSID" se utiliza para identificar a un usuario que tenga la sesión iniciada en sitios que no son de Google y para recordar si el usuario ha aceptado la personalización de anuncios. Tiene una duración de 2 semanas.

A través de la plataforma publicitaria de Google, las empresas pueden anunciarse en servicios de Google y en sitios que no son de Google. Algunas cookies sirven de apoyo a Google para mostrar anuncios en sitios de terceros y se establecen en el dominio del sitio web que visitas. Por ejemplo, la cookie "_gads" permite a los sitios mostrar anuncios de Google. Las cookies que empiezan por "gac" proceden de Google Analytics y las utilizan los anunciantes para medir la actividad de usuario y el rendimiento de sus campañas publicitarias. Las cookies "_gads" tienen una duración de 13 meses, mientras que las cookies "gac" duran 90 días.

Algunas cookies y otras tecnologías se utilizan para medir el rendimiento de los anuncios y las campañas, así como las tasas de conversión de los anuncios de Google en los sitios que visitas. Por ejemplo, las cookies que empiezan por "gcl" se usan principalmente para ayudar a los anunciantes a determinar cuántas veces los usuarios que hacen clic en sus anuncios acaban realizando una acción en su sitio (por ejemplo, una compra). Las cookies que se usan para medir tasas de conversión no se utilizan para personalizar anuncios. Las cookies "gcl" tienen una duración de 90 días.

**¿JWT qué tan utilizado es?**

Hola tú, solo quería mencionarte que la Platzi Fake Store también cuenta con documentación para JWT y le puedas echar un vistazo de como funciona o al menos tener una idea, 

![How cookies work!](https://substackcdn.com/image/fetch/w_1200,h_600,c_limit,f_jpg,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9222a3c6-96ea-4b9e-b639-617111e8fac4_1920x1040.png "How cookies work!")

### Bases de datos

- A un servidor al que le llegan diferentes tipos de solicitud.
 - Tenemos lógica de negocio.
 - Reglas de negocio.
 - Validación.
 - Autorización.
- Dentro del servidor hay servicios de backend y servicios de datos corriendo. Puede tener múltiples servicios de cada uno, y cada uno puede consumir más o menos recursos.
 - Los servicios de backend tienden a consumir CPU y RAM.
 - Las bases de datos consumen memoria en disco (HDD o SSD).
- El backend necesia de una base de datos para conseguir información. Los drivers son el medio utilizado para conectar bases de datos con el backend, y cada base de datos en particular tiene su propio driver.
- Existen 2 tipos de bases de datos:
 **Relacionales (SQL)**.
 - MySQL.
 - PostgreSQL.
 - ORACLE.
**No relacionales (NoSQL).**
 - mongoDB.
 - CASSANDRA.
 - Couchbase.
- Las bases de datos relacionales tienen un lenguaje en común, llamado SQL.
 - Esto hace que los frameworks aprovechen dicha carácterística con Object-Relational Mapping.
- ORMS es una forma de abstraer la conexión de las bases de datos a través de la programación orientada a objetos. Se enfoca en hacer peticiones a una base de datos de manera agnóstica, permitiendo cambiar sin mucha dificultad a otra base de datos relacional.
- Por lo general el DBA (administrador de bases de datos) gestiona las bases de datos, por lo que no debería caer sobre el backend dicha resposabilidad, por ser bastante extensa en sí misma, pero nunca está de más aprender sobre su gestión.
- Existen servicios terceros que se encargan de la gestión de base de datos.

### ¿Qué es el escalamiento?

**Escalamiento vertical.**
Es cuando tenemos un servidor y eventualmente el servidor empieza a colapsar, una forma de solucionar el problema es incrementar:

- CPU

- RAM

- Disk

- **Problemas:**

 -  Costos
 -  En un black friday tu puedes escalar la aplicacion, pero algunos provedores no permiten desescalar.
 -  Disponibilidad -> Solucion, Escalamiento Horizontal
**Escalamiento horizontal**

Soluciona el problema de la disponibilidad. Se tienen varias instancias del mismo servidor. Como se tienen distintos servidores ahora se necesita de un **LOAD BALANCER**

**LOAD BALANCER **-> Tiene conocimiento de nuestras instancias/servidores (al conjunto de servidores se denomina **Clouster**). Si un nodo(instancia) se cae, el load balancer se encarga de desviarla. Distribuye las peticiones.

 -  Si tenemos la base de datos local en cada servidor va haber un problema dado que no se tienen sincronizados los datos de las distintas bases de datos de los servidores. SOLUCION Gestionar la base de datos fuera de estos servidores. Quizas como un servidor aparte que sirva como DB.
 
 ![Scaling](https://qckbot.com/wp-content/uploads/2022/03/vertical-scaling.jpg "Scaling")

### ¿Qué es la replicación?

- Soluciones a desincronización de base de datos.
 - Aislar base de datos a un servidor en particular. Se le hace escalamiento vertical solo a la base de datos para evitar un cuello de botella.
 - Se puede hacer escalamiento horizontal a la base de datos con su propio load balancer para solventar el problema de la disponibilidad.
- Cada vez que se realice escritura en una de las bases de datos, se realiza una sincronización para que el cambio ocurra en todas las bases de datos. Esto es conocido como replicación.
 - Normalmente el backend developer no se encarga de la parte de replicación y gestión de bases de datos, sino en la capa de los servidores y el escalamiento horizontal.

### ¿Qué es la caché?

- La memoria cache es un sistema de almacenamiento interno del servidor para guardarde manera temporal el retorno de información.
- Se suele utilizar con información frecuentemente solicitada para mejorar el redimiento mediante el acceso rápido de los datos.
- El sistema de memoria de cache puede estar dentro del mismo servidor guardando las peticiones que vayan llegando.
- Es importante reconocer dónde es óptimo utilizar un sistema de cache, en donde sepamos que los sitios se encuentren constantemente solicitando información y el cliente envíe de manera constante la misma información.
 - E-commerce.
 - Sitios de noticias.
 - Blogs.
- No es muy bueno su uso para aplicaciones en tiempo real, como sistemas de chat.
- Puede ser de utilidad para el bloqueo de ataques de denegación de servicio (DDoS). Esto es debido a que en una situación en la que recibas muchas peticiones al mismo endpoint en un espacio corto de tiempo tu sistema no se vería tan afectado, puesto que el sistema de caché empieza a responder en lugar del backend y la bases de datos, y podría absorber dicho ataque.

**En Conclusión, La Caché:**
Es un espacio en memoria en base de datos que almacena los datos repetitivos de una navegación cotidiana del usuario en una aplicación, sitio web, etc. . Esto va a posibilitar mejor el perfomance de carga de la plataforma y poder entregar recuersos rápidos y efecicientes a la hora de recibir las peticiones del cliente. .

- Ideal para:
- Plataformas Eccommerce.
- Blogs y sitios web de informativos.
- Sitio web de servicios de consulta estáticos.
No ideal para:
- Realtime applications como LiveChats.
La Caché es usual trabajar con ella en producción, más no recomnedable trabajar en modo desarrollo, ya que necesitamos ver los cambios en tiempo real.

### Colas de tareas

- Ciertas tareas pueden tener un tiempo de espera muy largo.
 - Reportes.
 - Backups.
 - Gráficos.
 - Zips, PDFs, CSVs.
- Para responder a los largos tiempo de espera de estos procesos, y no dejar al cliente esperando durante largos periodos de tiempo, existen las colas de tareas.
- Una cola de tareas debe tomar en cuenta la ejecución y la respuesta.
 - Eventualmente ejecuta el proceso (no es de manera instantánea).
 - Puede responder por otro medio (como correo electrónico).
- Las colas de tareas almacenan tareas pendientes para ser procesadas, las cuales son procesadas y manejadas en orden de llegada.
 - Permite el manejo simultáneo de una gran cantidad de peticiones.
 - Las tareas son manejadas de manera asíncrona, por lo que el cliente recibe una respuesta mientras la tarea está siendo procesada.
 - Permite la retención de tareas en caso de fallas en el sistema, y su debido proceso una vez vuelva a estar disponible.
 - Permite la priorización de tareas de acuerdo a su importancia y urgencia, realizando primero las tareas más críticas.
 - Es posible el desacoplamiento de los diferentes procesos en un sistema, lo que permite el escalamiento de cada proceso de manera independiente, creando un sistema más flexible.

### Server-Side Rendering

1. ¿En qué consiste el Server-Side Rendering (SSR)?

Es un enfoque de renderizado en el que se procesa y genera completamente el HTML en el servidor, antes de enviarlo al navegador del cliente.

2. ¿Cómo se compara el Server-Side Rendering con el Client-Side Rendering (CSR)?
El SSR, el HTML lo genera el servidor
El CSR, la aplicación y el HTML se generan en el navegador del cliente utilizando JavaScript y el DOM.

3. ¿Qué es la técnica de Rehydration y en qué consiste?
Es una técnica que combina características de SSR y CSR. En el Rehydration, se aprovecha el HTML y los datos renderizados desde el servidor, y luego se "hidrata" o complementa con una aplicación JavaScript que se ejecuta en el navegador.

4. ¿Qué es Prerendering y cómo funciona?
El Prerendering es una técnica de renderizado web que implica generar y renderizar una página web completa en el servidor antes de que un usuario realice una solicitud. Esto significa que las páginas web se crean de antemano, y los resultados se almacenan en forma de archivos HTML estáticos que se pueden entregar de inmediato cuando se solicitan.

5. ¿Cuáles son las ventajas de utilizar Server-Side Rendering (SSR)?

- Mejora el SEO (Motores de búsqueda)
- Carga más rápida de la página
- Mejora el rendimiento en dispositivos de baja potencia
- Mayor compatibilidad
- Mayor seguridad

6. ¿En qué situaciones es especialmente útil el Server-Side Rendering (SSR)?
Cuando se requiere:

- Una indexación SEO efectiva
- Una carga rápida de la página
- Rendimiento en dispositivos de baja potencia
- Mayor seguridad en la manipulación de datos y autenticación.