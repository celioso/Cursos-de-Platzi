# Curso de Introducción al Despliegue de Aplicaciones

## Historia de las aplicaciones

Es muy importante lo que dice el profe sobre la evolución de las apps que van desde apps de terminal de comandos(CLI) hasta las web app que conocemos como Nexflx, Platzi, Google Suite, etc... La arquitectura de las apps web modernas es muy importante a destacar como los monolitos y microservicios que no son las unicas. Pero resumidamente los monolitos es cuando todo esta en el mismo servidor (codigo, base de datos) y micro servicio es cuando tenemos múltiples servidores que se comunican entre ellos y cada servidor tiene funcionalidades diferentes. Una duda que tengo es que si uso servicios de bases de datos en la nube ya estoy rompiendo el monolito?

## Apps Monolíticas vs microservicios

1. **Apps Monolíticas**: En los 90's surgieron lenguajes como PHP y Perl, los cuales permitian desarrollar aplicaciones de una forma fácil y rápida. donde se mezclaba código HTML con código PHP para consultas a la base de datos, etc. La desventaja de esto era que se tenía mucha disponibilidad del archivo (si una linea de código fallaba se podía romper toda la aplicación). Con el paso de los años esto se fue mejorando, separando el HTML y el CSS de los archivos PHP mediante el uso de motores de templating, se dejaba la lógica en otros archivos PHP y las consultas a la Base de Datos mediante ORMs. Esto mejoró un poco la estructura de la aplicación, sin embargo teníamos el problema de tener toda la aplicación en un solo servidor. Cuando algo fallaba en el servidor, esto hacía que toda la aplicación dejara de funcionar. Con el paso de los años se empezó a buscar un concepto llamado "bajo acoplamiento, alta cohesión", que básicamente consiste en separar componentes de las aplicaciones no solo a nivel lógico sino a nivel físico, será mucho mejor.

[Apps Monolíticas](./images/mololitico.jpg)

2. **Microservicios**: Los microservicios son un enfoque para el desarrollo de aplicaciones en el que una aplicación se construye como un conjunto de servicios modulares (es decir, módulos / componentes débilmente acoplados). Cada módulo apoya un objetivo comercial específico y utiliza una interfaz simple y bien definida para comunicarse con otros conjuntos de servicios. En lugar de compartir una sola base de datos como en la aplicación Monolitica, cada microservicio tiene su propia base de datos. Tener una base de datos por servicio es esencial si desea beneficiarse de los microservicios, ya que garantiza un acoplamiento flexible. Cada uno de los servicios tiene su propia base de datos. Además, un servicio puede utilizar el tipo de base de datos que mejor se adapte a sus necesidades.

[Microservicios](./images/Microservicios.jpg)

Las imagenes y algunos textos fueron tomados de: [este artículo](https://medium.com/koderlabs/introduction-to-monolithic-architecture-and-microservices-architecture-b211a5955c63)

## Stacks LAMP, MERN, JOTL, JAM

**Stacks de desarrollo resumen**

Los stacks de tecnologias son el conjunto de herramientas para nuestra app en campos backend, frontend y bases de datos, estás pueden ser:

- **LAMP**: Compuesta por Linux, Apache, MySql y Php.

- **JOTL**: Compuesta por Java, Oracle, Tomcat y Linux.

- **MERN**: Mongo, React, Node, Express. (Cabe aclarar que los framework de javaScript pueden cambiar, entre angular Vue, React)

- **JAM**: JavaScript, Api y Markdown

Es importante tener en cuenta que estos stacks pueden variar según sea tu conveniencia.

[What is Full Stack](https://www.w3schools.com/whatis/whatis_fullstack.asp)

## En dónde se pueden desplegar aplicaciones LAMP y JOTL

### LAMP

[LAMP](./images/lamp.jpg)

El despliegue de una aplicación [LAMP](https://es.wikipedia.org/wiki/LAMP) (Linux, Apache, Mysql, PHP) puede ser de los más conocidos y populares dado la popularidad de php y mysql Cuando empezaron a aparecer las aplicaciones web .

Existen varias formas de desplegar, estas son las más comunes:

- **[Hosting compartido](https://es.wikipedia.org/wiki/Alojamiento_web)**: la fórmula más popular es comprar un servicio de hosting donde te proveen de una interfaz web llamada Cpanel donde puedes crear tu base de datos mysql, subir tus archivos php por ftp o administrador web y tener tu app en minutos.

- **Hosting gratuito**: Algunas empresas proveen hosting gratuito a cambio de que se integre publicidad en tus scripts php o de acceder a la información de tu sitio, sin embargo estas tienden a tener interfaces web menos amigables para subir archivos de la aplicación y la base de datos.

- **Usar un VPS**: utilizando plataformas como Digital Ocean, se puede crear un droplet (forma en que llaman a los VPS en esta empresa), para tener acceso SSH y poder instalar php,mysql, apache y lo que se necesite para instalar la aplicación web, puede tomar más tiempo en configurar todo, y el vps se debe administrar por la persona, a cambio, se gana acceso total al servidor para modificar php, mysql, y realizar tareas de gestión, o escalamiento de la aplicación.

### JOTL
[JOTL](./images/JOTL.jpg)

Por otra parte, en el mercado también es muy popular el stack de la empresa Oracle JOTL (Java, Oracle, Tomcat, Linux) dado el soporte y la fama que tiene Oracle de tener el sistema de base de datos más robusto, y esto sumado con Java que es un lenguaje de programación multiplataforma: funciona para hacer aplicaciones de escritorio, aplicaciones web, aplicaciones móviles para Android, etc.

Estas son las formas más comunes para desplegar una aplicación JOTL.

- Usar una plataforma como servicio: Se puede utilizar una PaaS - Platform As A Service, como es el caso de heroku, que se encarga del despliegue de la aplicación y se puede hacer un despliegue más rápido, pero se pierde el control sobre el servidor.

- Usar una Infraestructura como servicio: IaaS o Infrastructure As a Service, son empresas como AWS de Amazon, Cloud Platform de Google, Azure de microsoft o incluso IBM cloud, estas ofrecen un control mayor sobre la infraestructura, desde los servidores VPS, red, Backups, disponibilidad, escalabilidad, seguridad entre otras ventajas, sin embargo requieren de un conocimiento en manejo de infraestructuras para poder configurar todas estas opciones.

- Usar infraestructura propia: Algunas empresas prefieren disponer de una infraestructura propia, esto se conoce como on-premises, entre la razones y ventajas para este tipo de infraestructura están:
Privacidad del código fuente o aplicación, ya que este se encontrará local y no en servidores en una nube a los que terceros podrían acceder.

La segunda razón es por latencia, dado que un datacenter en la misma ciudad podrá ofrecer mejores tiempos de respuesta que uno en otro País o continente.

Finalmente por control, ya que las empresas que adoptan esto, tienen control total sobre la infraestructura física (no sólo la lógica como ocurre con las IaaS).

Como desventajas principales están: Disponibilidad física, si el lugar donde está el datacenter sufre un incendio, terremoto o cualquier situación que afecte el lugar, podría perderse el acceso físico a la información y/o a la red.

Costo, mientras la computación en la nube ofrece precios competitivos por horas, escalamiento dinámico y otros temas que parecen casi automáticos, en los entornos on-premise los costos pueden ser mayores dado que se debe costear servidores, racks de almacenamiento, el datacenter donde se almacenará la información, la energía, internet, seguridad, y demás costos asociados.

Existen muchas opciones para desplegar aplicaciones de este tipo, cada una tiene ventajas y desventajas, depende del tipo de proyecto la opción a seleccionar.

Nota: En el Stack de Java y Oracle también se puede intercambiar Apache Tomcat con GlassFish u otros, incluso en la parte de sistema operativo, cambiando Linux por Windows server.

## Despliegue en Github pages

[SpaceX Platzi](https://santiaguf1.github.io/)

[GitHub Pages | Websites for you and your projects, hosted directly from your GitHub repository. Just edit, push, and your changes are live.](https://pages.github.com/)

[Curso Profesional de Git y GitHub](https://platzi.com/clases/git-github/)

[Curso de Introducción a la Terminal y Línea de comandos](https://platzi.com/clases/terminal/)

[Santiago Bernal - Software Engineer](https://sb.js.org/)

[GitHub - santiaguf/spacex-platzi: Web application that show info of SpaceX using spaceX API](https://github.com/santiaguf/spacex-platzi)

[Repositorio Spacex](https://github.com/santiaguf/spacex-platzi)

En mi cuenta principal de github tengo el repositorio apuntando a un subdominio diferente [https://sb.js.org/](https://sb.js.org) , te reto a que luego de realizar esta clase, configures tu repo con [js.org](https://js.org/)

## Despliegue en Surge

se instala surge `npm install --global surge`
luego `surge`

Por su fuera de su interés, dejo el paso a paso del despliegue en Surge:

**Nota**: si no reconoce "npm" deberás instalar nodeJs y continuar con el paso a paso del despliegue en Surge.
[Surge.sh](./images/surge.jpg)

si hay algun problema con el dominio ya que otro lo esta usando se usa `surge login` el pedira correo y contraseña.
luego se utiliza `surge --domain your-new-subdomain.surge.sh` para el dominio deseado y listo.

[Web en suge](spacex-platzi-master-practica.surge.sh)

[Surge](http://surge.sh/)

[SpaceX Platzi](http://foregoing-loss.surge.sh/index.html)

## Despliegue en Netlify

A tomar en cuenta Netlify es mi favorito a la hora de desplegar aplicaciones a producción. Uno de los problemas con los que normalmente te puedes encontrar es el routing del servidor. Por ejemplo: Si tienes una aplicación SPA (single page application) hecha con React y manejas rutas con React-Router, tienes que tomar en cuenta que al ingresar a una de las rutas de tu aplicación directamente el servidor va a responder con un error 404.

[netlife](./images/netlife_app.jpg)

En este caso lo que tienes que hacer es configurar los redirects para tu aplicación. Netlify provee 2 formas de confirgurar las rutas, una es mediante un archivo **_redirects** y la otra que es mucho mas práctica es mediante un archivo **netlify.toml** que se coloca en la raiz de tu proyecto.

[redirects](./images/redirects.jpg)

Deber redireccionar todas las rutas a tu archivo index.html para que tu aplicación se cargue y una vez cargada se encargará del rounting en el lado cliente. te dejo un ejemplo de configuración del archivo netlify.toml a continuación:

```bash
[[redirects]]
  from = "/"
  to = "/index.html"
  status = 200

[[redirects]]
  from = "/users"
  to = "/index.html"
  status = 200

[[redirects]]
  from = "/add"
  to = "/index.html"
  status = 200

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 400
```

Para mas detalles puesdes consultar la documentación de Netlify

[app desplegada](https://66bd28bc6142150a4ac4a582--stunning-wisp-7d3051.netlify.app/)

[Netlify: All-in-one platform for automating modern web projects](https://www.netlify.com/)

[SpaceX Platzi](https://lucid-bell-432937.netlify.app/)

## Despliegue en Vercel

Vercel es una plataforma para que los desarrolladores de frontend construyan, desplieguen y compartan sus proyectos con el mundo. Está diseñada para facilitar y acelerar el paso del concepto a la producción. 

[web en  Vercel](https://spacex-platzi-master.vercel.app/)

[Develop. Preview. Ship. For the best frontend teams – Vercel](https://vercel.com/)

[SpaceX Platzi (Vercel)](https://platzinautas.now.sh/)

[SpaceX Platzi (GitHub Page)](https://santiaguf1-github-io.vercel.app/)

## Desplegando una base de datos NoSql en Mongo Atlas

[Managed MongoDB Hosting | Database-as-a-Service | MongoDB](https://www.mongodb.com/cloud/atlas)

[Compass | MongoDB](https://www.mongodb.com/products/compass)

## Desplegando una base de datos relacional en ElephantSQL

ElephantSQL es un servicio de base de datos en la nube que proporciona instancias de PostgreSQL gestionadas. Si estás trabajando con ElephantSQL y necesitas eliminar un proyecto o base de datos, aquí te dejo una guía sobre cómo hacerlo:

### Pasos para eliminar una base de datos en ElephantSQL:

1. Inicia sesión en ElephantSQL:

 - Ve a [ElephantSQL](https://www.elephantsql.com/) e inicia sesión con tus credenciales.

2. Accede a tu instancia de base de datos:

 - En el panel de control, verás una lista de las instancias de bases de datos que has creado. Haz clic en la instancia que deseas eliminar.

3. Eliminar la instancia:

 - Dentro de la página de detalles de la base de datos, busca una opción que diga "Delete" o "Eliminar instancia".
 - Haz clic en "Delete" y sigue las indicaciones para confirmar la eliminación. Es posible que te pidan confirmar el nombre de la base de datos o escribir "DELETE" como confirmación.

4. Verificación:

 - Una vez eliminada la base de datos, asegúrate de que ya no aparece en tu lista de instancias.

### Importante:

 - **Copias de seguridad**: Antes de eliminar una base de datos, asegúrate de haber hecho una copia de seguridad si la información es importante.
 - **Aplicaciones vinculadas**: Asegúrate de que no haya aplicaciones o servicios activos que dependan de esta base de datos, ya que dejarán de funcionar una vez que la base de datos sea eliminada.
Si necesitas ayuda específica sobre algún aspecto de ElephantSQL, estaré aquí para ayudarte.

[ElephantSQL - PostgreSQL as a Service](https://www.elephantsql.com/)

## Qué es Heroku

Heroku es una plataforma como servicio (PaaS) que permite a los desarrolladores crear, ejecutar y operar aplicaciones completamente en la nube. Fue una de las primeras plataformas de este tipo y es especialmente conocida por su facilidad de uso y por su integración con Git, lo que permite a los desarrolladores implementar código rápidamente y gestionar aplicaciones de manera eficiente.

### Características principales de Heroku:

1. **Facilidad de implementación:**

    - Puedes desplegar aplicaciones en Heroku con comandos simples desde la línea de comandos utilizando Git. Por ejemplo, con git push heroku master, puedes desplegar tu aplicación directamente.

2. **Soporte para múltiples lenguajes:**

 - Heroku soporta una amplia gama de lenguajes de programación, incluidos Python, Ruby, Node.js, Java, PHP, Go, y Scala, entre otros.

3. **Escalabilidad:**

 - Puedes escalar tu aplicación fácilmente añadiendo más "dynos", que son contenedores que ejecutan tus aplicaciones en Heroku. Esto permite manejar más tráfico o cargas de trabajo sin tener que preocuparte por la infraestructura subyacente.

4. Add-ons:

 - Heroku ofrece una amplia variedad de add-ons, que son servicios adicionales que puedes integrar en tu aplicación, como bases de datos (PostgreSQL, Redis), herramientas de monitoreo, servicios de correo electrónico, y más.

5. Administración y monitoreo:

 - Heroku proporciona herramientas integradas para monitorear el rendimiento de tus aplicaciones, gestionar entornos de desarrollo, staging y producción, y manejar configuraciones como variables de entorno.
6. Integración continua y entrega continua (CI/CD):

 - Puedes configurar pipelines de integración y entrega continua para automatizar pruebas y despliegues, lo que facilita el desarrollo y la entrega de software de manera ágil.

### Casos de uso comunes:

- **Aplicaciones web y API**: Heroku es popular para desplegar aplicaciones web, APIs, y aplicaciones móviles.
- **Proyectos de desarrollo rápido**: Debido a su simplicidad, es ideal para startups, prototipos, y desarrollos ágiles.
- **Aplicaciones de escala pequeña y mediana**: Aunque Heroku puede manejar aplicaciones a gran escala, su estructura de precios y arquitectura lo hacen especialmente adecuado para aplicaciones de tamaño pequeño a mediano.

### Consideraciones:

- **Costo**: Aunque Heroku ofrece un nivel gratuito, para aplicaciones en producción o con tráfico considerable, los costos pueden aumentar rápidamente a medida que escales tus recursos.
- **Limitaciones**: Heroku tiene ciertas limitaciones en cuanto a personalización y configuración avanzada, lo que puede no ser ideal para todas las aplicaciones, especialmente aquellas que requieren configuraciones específicas del servidor o acceso directo a la infraestructura subyacente.

Heroku es una plataforma poderosa para desarrolladores que buscan simplicidad y rapidez en el despliegue de sus aplicaciones sin preocuparse por la infraestructura.

[Cloud Application Platform | Heroku](https://www.heroku.com/)

## Desplegando Api en Heroku


[Cloud Application Platform | Heroku](https://www.heroku.com/)
[fly](https://fly.io/)

[GitHub - santiaguf/node-todo-app: Todo App with nodeJs, Express, Mongoose and MongoDB](https://github.com/santiaguf/node-todo-app)

### Consultando nuestra API desde Postman

**Postman** es una herramienta muy popular utilizada por desarrolladores para diseñar, probar, y documentar APIs. Es una aplicación que facilita la interacción con APIs RESTful o cualquier otra que funcione sobre HTTP.

### Funciones principales de Postman:

1. **Crear y enviar solicitudes HTTP:**

 - Puedes crear y enviar solicitudes HTTP como GET, POST, PUT, DELETE, etc., hacia una API. Puedes especificar el método HTTP, la URL, los headers, el cuerpo de la solicitud (en JSON, XML, form-data, etc.), y otros parámetros.

2. **Testing automatizado:**

 - Postman permite escribir tests para verificar el comportamiento de las respuestas de la API. Puedes validar códigos de estado, tiempos de respuesta, contenido de las respuestas, y más.

3. **Colecciones de solicitudes:**

 - Puedes agrupar varias solicitudes en colecciones, lo que facilita la organización de tests y la creación de escenarios de prueba. Las colecciones se pueden compartir con otros miembros del equipo.

4. **Variables y entornos:**

 - Puedes definir variables que se pueden usar en las solicitudes para cambiar dinámicamente valores como URLs, tokens de autenticación, u otros parámetros. Los entornos permiten cambiar rápidamente entre configuraciones para desarrollo, staging, y producción.

5. **Documentación de APIs:**

 - Postman puede generar documentación interactiva para tus APIs. Esta documentación puede ser publicada y compartida, lo que facilita a los equipos de desarrollo y a los usuarios externos conocer cómo utilizar la API.

6. **Monitoreo de APIs:**

 - Con Postman, puedes configurar monitores para verificar el rendimiento y la disponibilidad de tus APIs en intervalos específicos.

7. **Mock Servers:**

- Permite crear mock servers para simular la API antes de que esté completamente desarrollada. Esto es útil para pruebas iniciales y para equipos que trabajan en paralelo.

### Usos comunes de Postman:

- **Desarrollo de APIs:** Para probar diferentes endpoints durante el desarrollo.
- **Depuración:** Verificar y solucionar problemas con las respuestas de la API.
- **Automatización de pruebas:** Ejecutar conjuntos de tests de manera automatizada.
- **Documentación:** Crear y compartir documentación interactiva de la API.
- **Colaboración:** Compartir colecciones y entornos entre equipos.

### Cómo usar Postman para probar una API:

1. **Instalar Postman:** Puedes descargar e instalar Postman desde [postman.com](https://www.postman.com/).

2. Crear una nueva solicitud:

 - Abre Postman y haz clic en "New" para crear una nueva solicitud.
 - Selecciona el tipo de solicitud (GET, POST, PUT, DELETE).
 - Ingresa la URL de la API que deseas probar.
 - Configura los headers, parámetros de URL, o el cuerpo de la solicitud según sea necesario.

3. **Enviar la solicitud:**

 - Haz clic en "Send" para enviar la solicitud.
 - Revisa la respuesta de la API en el panel inferior, donde verás el código de estado, los headers de la respuesta, y el cuerpo de la respuesta.

4. **Agregar tests:**

 - En la pestaña "Tests", puedes escribir scripts en JavaScript para validar la respuesta.
 - Ejemplo: `pm.test("Estado 200 OK", function () { pm.response.to.have.status(200); });`

5. Guardar y organizar:

 - Guarda la solicitud en una colección para reutilizarla o compartirla con otros.
 - Organiza tus solicitudes en carpetas dentro de colecciones para mantenerlas ordenadas.

6. Ejecutar en un entorno diferente:

 - Crea y selecciona un entorno para ejecutar la misma solicitud en diferentes configuraciones (por ejemplo, desarrollo o producción).

**Integraciones y complementos:**

- **Newman:** Es una herramienta de línea de comandos para ejecutar colecciones de Postman en CI/CD pipelines.

- **Integraciones con CI/CD:** Postman se integra con herramientas de integración continua como Jenkins, Travis CI, y otros.

Postman es una herramienta versátil que facilita enormemente el trabajo con APIs, desde la etapa de desarrollo hasta la documentación y pruebas automatizadas.

[Postman | The Collaboration Platform for API Development](https://www.postman.com/)

### enerar documentación de API con Postman

Instalar postman en Ubuntu22
`sudo snap install postman`

[Postman API](https://docs.api.getpostman.com/?version=latest)
[Curso de Postman](https://platzi.com/clases/postman/)

## Capas gratuitas de grandes proveedores


Las capas gratuitas (o "free tiers") de grandes proveedores de servicios en la nube permiten a los desarrolladores y empresas probar sus servicios sin costo o con un uso limitado. Estas capas gratuitas son ideales para proyectos pequeños, pruebas, y aprendizaje. Aquí te explico las capas gratuitas ofrecidas por algunos de los principales proveedores de servicios en la nube:

### 1. Amazon Web Services (AWS)

AWS ofrece una capa gratuita que incluye servicios gratuitos durante 12 meses y otros que son siempre gratuitos.

- **Servicios gratuitos durante 12 meses:**
 - **Amazon EC2:** 750 horas por mes de instancias t2.micro o t3.micro (dependiendo de la región).
 - **Amazon RDS:** 750 horas por mes en una instancia db.t2.micro, compatible con bases de datos como MySQL, PostgreSQL, SQL Server, etc.
- **Amazon S3:** 5 GB de almacenamiento estándar en S3, 20,000 solicitudes GET y 2,000 PUT.
 - **Amazon Lambda:** 1 millón de solicitudes gratuitas por mes, y 400,000 GB-segundos de tiempo de cómputo.

### - Servicios siempre gratuitos:
 - **Amazon DynamoDB:** 25 GB de almacenamiento con 25 unidades de capacidad de lectura y 25 de escritura.
 - **Amazon SNS:** 1 millón de publicaciones o envíos de mensajes por mes.
 - **AWS Lambda:** 1 millón de solicitudes por mes y 3.2 millones de segundos de cómputo.

### 2. Google Cloud Platform (GCP)

Google Cloud ofrece una capa gratuita con crédito inicial y recursos gratuitos limitados.

- **Crédito inicial:**

 - $300 USD para gastar en cualquier servicio de Google Cloud durante los primeros 90 días.

- Servicios gratuitos:

 - **Google Compute Engine:** 1 micro instancia f1-micro por mes (usada en EE. UU.).
 - **Google Cloud Storage:** 5 GB de almacenamiento en la clase de almacenamiento estándar.
 - **Google Cloud Functions:** 2 millones de invocaciones por mes.
 - **Google BigQuery:** 1 TB de consultas gratuitas por mes.
 - **Cloud Firestore:** 1 GB de almacenamiento, 50,000 lecturas y 20,000 escrituras por día.
 - **Google Cloud Pub/Sub:** 10 GB de mensajes de salida por mes.

### 3. Microsoft Azure

Azure ofrece una capa gratuita con servicios limitados y un crédito inicial.

- Crédito inicial:

 - $200 USD para gastar en cualquier servicio de Azure durante los primeros 30 días.

- Servicios gratuitos durante 12 meses:

 - **Máquinas Virtuales (VMs):** 750 horas por mes de instancias B1S.
 - **Azure SQL Database:** 250 GB de almacenamiento estándar.
 - **Azure Blob Storage:** 5 GB de almacenamiento localmente redundante.
 - **Azure Cosmos DB:** 400 RU/s y 5 GB de almacenamiento.

- Servicios siempre gratuitos:

 - **Azure App Service:** 10 aplicaciones web, móviles o API con 1 GB de almacenamiento.
 - **Azure Functions:** 1 millón de ejecuciones por mes.
 - **Azure Cosmos DB:** 400 RU/s y 5 GB de almacenamiento (siempre gratis después de los 12 meses).

### 4. IBM Cloud
IBM Cloud ofrece una capa gratuita con servicios básicos gratuitos y créditos promocionales.

- Servicios gratuitos:
 - **Cloud Foundry:** 256 MB de tiempo de ejecución de aplicaciones.
 - **Kubernetes:** Clúster Kubernetes con un nodo y 2 GB de memoria.
 - **IBM Watson:** Algunos servicios como Watson Assistant, Text to Speech, y Visual Recognition tienen niveles gratuitos.
 - **DB2 on Cloud:** Instancia Lite con 100 MB de almacenamiento.
 - **Cloud Object Storage:** 25 GB de almacenamiento.

### 5. Oracle Cloud

Oracle Cloud tiene una capa gratuita que incluye siempre servicios gratuitos y servicios adicionales gratuitos durante 30 días.

- **Servicios siempre gratuitos:**

 - **Computación (VM.Standard.E2.1.Micro):** 2 instancias, 1 GB de RAM por instancia.
 - **Block Volumes:** 2 volúmenes, 100 GB de almacenamiento en total.
 - **Autonomous Database:** 2 bases de datos, 20 GB de almacenamiento por base de datos.
 - **Object Storage:** 10 GB de almacenamiento estándar.

- **Servicios gratuitos durante 30 días:**

 - **$300 USD** en créditos para probar cualquier servicio de Oracle Cloud.

### Consideraciones:

- **Uso limitado:** Las capas gratuitas tienen límites que, si se superan, pueden generar costos. Es importante monitorear el uso para evitar sorpresas.
- **Renovación:** Algunos servicios gratuitos caducan después de 12 meses, por lo que es importante revisar las condiciones de cada proveedor.
- **Restricciones geográficas:** Algunas ofertas pueden estar limitadas a ciertas regiones.

Estas capas gratuitas son excelentes para comenzar a explorar los servicios en la nube sin incurrir en costos, especialmente útiles para desarrollo, pruebas, y aprendizaje.

[Computación en la nube - Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/Computaci%C3%B3n_en_la_nube)

[Curso de Azure Iaas](https://platzi.com/clases/azure-iaas/)

[Fundamentos de Google Cloud Platform](https://platzi.com/clases/fundamentos-google/)

[Fundamentos de IBM Cloud](https://platzi.com/clases/ibm-cloud/)

[Curso de Fundamentos de AWS Cloud](https://platzi.com/clases/aws-cloud/)