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