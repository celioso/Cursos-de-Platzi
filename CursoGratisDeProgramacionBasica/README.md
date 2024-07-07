# Curso gratis de programación Basica 
## ¿Qué es backend?

Te encuentras muy cerca de finalizar tu primera aplicación. Es momento de desarrollar el backend de la misma, toda **la lógica del lado del servidor de tu proyecto que permitirá interactuar con otros usuarios e intercambiar mensajes**.

### Diferencias backend y front-end

Seguro ya sabes lo que es[ el front-end](https://platzi.com/blog/que-es-frontend-y-backend/ " el front-end"). **La parte visual de una aplicación web que puedes acceder a ella desde una navegador** o también denominado ““cliente””. La misma se desarrolla con tecnologías como HTML, CSS y Javascript.

Por otro lado, [el backend](https://platzi.com/blog/que-es-frontend-y-backend/ "el backend") es todo lo que ““no puedes ver””, ya que e**s la lógica de una aplicación que se ejecuta en un ““servidor”**. El backend se suele desarrollar con tecnologías como Java, PHP, C#, C++ o también con Javascript con NodeJS.

De esta breve explicación se desprenden dos conceptos claves que te acompañarán el resto de tu vida como programador. El cliente y el servidor.

### Arquitectura cliente/servidor

Si estás leyendo esto, es gracias a que tu navegador web (o aplicación mobile), el cliente, se conectó a los servidores de Platzi y los mismos le enviaron este texto.

La Internet moderna funciona a través de la arquitectura cliente/servidor. **Donde el cliente realiza peticiones de datos al servidor y este responde con los mismos**.

![Arquitectura cliente](./mokepon/assets/assetsREADME/Arquitecturacliente.png)


Un servidor puede responder de varias formas o con diferentes tipos de información dependiendo el tipo de petición del cliente.

Envío de páginas web
Cuando ingresas a [https://platzi.com/](https://platzi.com/ "https://platzi.com/") el servidor realiza un tipo de respuesta enviándole al cliente, al navegador web, archivos para la construcción de una página web.

![navegador web](./mokepon/assets/assetsREADME/navegadorweb.png "navegador web")

Cada tipo de archivo es interpretado de una forma diferente por el navegador para construir la página. Incluso puedes enviar archivos multimedia como imágenes o videos.

### Streaming de datos

Cuando te encuentras viendo un video aquí en Platzi o en YouTube, los servidores envían cada fotograma del video en el orden que les corresponde para que el navegador web pueda reproducir el video y múltiples usuarios puedan verlo en tiempo real.

![Streaming de datos](./mokepon/assets/assetsREADME/Streamingdedatos.png "Streaming de datos")

### Envió de datos

Un tercer tipo de intercambio de información entre un servidor y un cliente es el **envío de datos en crudo con una determinada estructura** de los mismos.

Un servidor puede enviar información de estudiantes, clases y cursos al cliente para que este construya la interfaz con los mismos y el usuario pueda interactuar con los datos.

![name json](./mokepon/assets/assetsREADME/namejson.png "name json")

Los datos suelen intercambiarse a través un formato de texto conocido como JSON o *Javascript Object Notation*. JSON es el estándar más utilizado hoy en día para intercambiar información entre aplicaciones y definir estructuras en los daots. El aspecto de este tipo de información es como el siguiente:

```json
{
    ""nombre"": ""Diana"",
    ""edad"": 27
}
```
Todo este intercambio de información entre un cliente y un servidor, o entre un front-end y un backend, se produce gracias a una API.

*Aplication Programming Interface* o ““Interfaz de Programación de Aplicaciones”” es otro concepto que te acompañará por mucho tiempo. Puedes verlo como una puerta de entrada para el cliente, para la obtención de datos desde un servidor.

El servidor debe permitir que un cliente haga consultas y reciba datos, a través de una API es que el intercambio de información es posible.

### Protocolo HTTP

**Internet está basado en protocolos que son formas estandarizadas de hacer las cosas**. El intercambio de datos entre un cliente y un servidor es posible gracias al protocolo HTTP.

Hypertext Transfer Protocol o ““Protocolo de Transferencia de Hipertexto”” por sus siglas en español, es el protocolo N°1 utilizado en internet para el intercambio de cualquier tipo de dato.

Seguro habrás visto que las páginas web comienzan con `http://` o `https://`. Ahora ya sabes qué significa.

### HTTP vs. HTTPS

La S de HTTPS no es más que una extensión al protocolo HTTP que lo hace más **Seguro** para el intercambio de información cifrada o codificada entre el cliente y el servidor para evitar robo de datos.

### Conclusión
Front-end, backend, cliente y servidor. El protocolo HTTP, APIs y JSON. Son solo los primeros conceptos, tecnologías o terminologías que debes conocer del mundo de la programación.

No te preocupes si aún no tienes en claro para qué sirve cada cosa, profundizarás poco a poco en cada uno de ellos y comprenderás su utilización para la construcción de un backend, y de una aplicación web completa.

## Instalación de Node.js y NPM

Así como, en el front-end, utilizas lenguajes como HTML, CSS y Javascript en el backend puedes usar otras tecnologías. Javascript es el que continuaremos utilizando de ahora en adelante.

### Tecnologías backend
Existen muchos lenguajes de programación para desarrollar en el backend. Veamos un listado de algunos de ellos.

- C/C++
- C#
- Java
- PHP
- Ruby
- Python
- Go
- Javascript (NodeJS)

Ya conoces lo que es Javascript y lo que permite construir en el front-end. También es posible utilizarlo en el backend gracias a NodeJS.

**NodeJS es un entorno de ejecución que permite interpretar y utilizar código Javascript en un servidor**, con algunas diferencias con respecto al front-end. Está construido con el motor [intérprete de código Javascript de Google Chrome denominado V8](https://platzi.com/cursos/javascript-navegador/ "intérprete de código Javascript de Google Chrome denominado V8").

En el backend no hay HTML, por lo que no podrás utilizar NodeJS para su manipulación. En su lugar, podrás usarlo para leer archivos, conectarte a una base de datos, levantar un servidor web y construir una API.

### Instalación de NodeJS

La instalación de NodeJS en tu ordenador es muy sencilla. Ingresa a su [página oficial](https://nodejs.org/en/ "página oficial") y has la descarga dependiendo tu sistema operativo.

![Descarga de NodeJS](./mokepon/assets/assetsREADME/nodejs.png "Descarga de NodeJS")

Te recomiendo que siempre instales la versión LTS (*Long Term Support*), dado que la misma tendrá soporte y mantenimiento por al menos 5 años. También utiliza versiones pares. Las versiones Current o las versiones impares suelen estar en desarrollo y pueden tener algún error o vulnerabilidad.

**NodeJS viene acompañado de otra tecnología denominada NPM** (Node Package Manager). El mismo nos ayudará a inicializar un nuevo proyecto o instalar cualquier tipo de dependencia que necesitemos para desarrollar nuestra aplicación.

![NodeJS y NPM](./mokepon/assets/assetsREADME/nodejsnpm.png "NodeJS y NPM")

Una vez realizada la instalación, puedes utilizar una serie de comandos desde una terminal para corroborar su correcto funcionamiento. Utiliza el comando node -v para verificar la versión de NodeJS y npm -v para visualizar la versión de NPM.

![version.png](./mokepon/assets/assetsREADME/version.png "version.png")

NodeJS será, tal vez, tu primer acercamiento al desarrollo backend en el lado del servidor. Mucho de lo que ya conoces sobre Javascript te servirá para NodeJS y lo complementaremos con otras características propias que exploraremos en las próximas clases.

## Terminal de comandos y Node.js

Como desarrolladores de software, una de nuestras mejores amigas será la terminal de línea de comandos para la ejecución de tareas. No debemos tenerle miedo, ya que nos facilitará mucho el trabajo en el futuro.

### ¿Qué es una terminal?

Una terminal de línea de comandos o CLI (Command-Line Interface), es **una interfaz de texto que nos permite interactuar con un proyecto, ejecutar tareas o navegar por todos los archivos y directorio de nuestro computador**.

En cualquier sistema operativo puedes ejecutar comandos en una terminal. Habrás observado que VS Code trae consigo una terminal. Existen muchas otras, todas muy similares.

A diferencia de una interfaz de usuario donde podemos observar e interactuar con archivos o directorios de forma visual y más amena, una terminal de línea de comando también lo permite a través de texto ejecutando comandos.

Parece algo más difícil, pero no te preocupes. Te acostumbrarás y te garantizo que lo agradecerás cuando seas un desarrollador de software profesional.

Con NodeJS, utilizaremos NPM que trae consigo su propio CLI para crear proyectos o instalar dependencias que nuestro proyecto necesitará.

Pero antes de eso…

### ¿Qué es una dependencia?

Llamamos dependencia o librería a una **pieza de código desarrollada por un tercero**, por otra persona. Las mismas nos permiten **solucionar problemas triviales y reutilizar código para hacer más rápido nuestro trabajo** como programadores.

[NPM](https://www.npmjs.com/ "NPM") se encargará de descargar por nosotros las dependencias que necesitamos. En la actualidad, es el gestor de dependencias más grande del mundo. Cada lenguaje de programación suele tener el suyo, como lo es *Composer* para PHP, *Maven* para Java o *PIP* para Python.

Una dependencia puede servirnos para manejar fechas, para leer archivos, para realizar solicitudes HTTP o hasta para levantar un servidor, entre muchas otras funcionalidades. Realmente te encontrarás con dependencias de todo tipo y casi para cualquier cosa que quieras hacer. Poco a poco, irás descubriendo más y más dependencias que utilizarás para construir tus proyectos.

### Comandos básicos que debes conocer

Existen [muchos comandos](https://platzi.com/cursos/terminal/ "muchos comandos") que incluso varían dependiendo el sistema operativo en el que trabajes.

Los comandos más básicos que puede probar son `ls` o `ll` para listar los archivos o directorios. También utilizarás mucho el comando `cd` para desplazarte entre directorios dentro de tu computador.

Los CLI, como NPM, incorporan a tu sistema operativo una serie de comandos específicos para trabajar con una tecnología. Suelen utilizarse estos con un prefijo como `npm <command-name>`.

### Hola Mundo con NodeJS

Siempre, y para toda tu vida, que instales una nueva tecnologías, lo primero que realizarás es el “Hola Mundo” que **permite corroborar la correcta instalación de la tecnología o herramienta**.

Para crear tu primer proyecto en NodeJS, con ayuda de NPM, basta con utilizar el comando npm init -y. El mismo creará en cuestión de segundos tu primer proyecto.

Observa que este comando ha creado un archivo llamado package.json que contiene la configuración básica de cualquier proyecto desarrollador en NodeJS.

```json
{
  "name": "prueba",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC"
}
```

No debes preocuparte tanto por su contenido, poco a poco lo irás comprendiendo. Puedes observar, entre otros datos, el nombre del proyecto, la versión, una descripción y el archivo base del proyecto, entre otros datos.

Continúa creando un archivo que llamaremos index.js que será el archivo base de nuestro proyecto. Agrégale al mismo el siguiente contenido:

`console.log("¡Hola Mundo!");`

Ya puede ejecutar tu primer proyecto en NodeJS utilizando el comando `node index.js`. Recuerda utilizar el nombre de tu archivo que acabas de crear para indicarle a Node qué archivo ejecutar.

En cuestión de segundos observarás un `¡Hola Mundo!` en la consola que hayas utilizado para lanzar el comando. Eso significa que NodeJS se encuentra correctamente instalado en tu computador y has ejecutado tu primer programa.

Ha sido bastante sencillo la creación de un proyecto con NodeJS y su ejecución. Ahora es momento de desarrollar tu primera aplicación backend apoyándote de estas tecnologías.

## Servidor web con Express.js

Crear tu primer servidor web, o servidor HTTP, con NodeJS es súper sencillo. Para esto utilizaremos una de las librerías más populares de NPM como lo es [Express.js](http://expressjs.com/ "Express.js").

### ¿Qué es ExpressJS?

Una de las dependencias más antiguas, pero completamente vigente y recomendada, para trabajar con aplicaciones web en NodeJS es Express.js. La misma te permitirá, en pocas líneas de código, **exponer tu primer servidor web y crear tu primera API**.

### Consejos sobre NPM

Siempre que instales una dependencia de terceros que se encuentren en NPM, te aconsejo que ingreses a su respectiva página para obtener más información de la librería.

![ExpressJS en NPM](./mokepon/assets/assetsREADME/ExpressJSenNPM.png "ExpressJS en NPM")

La [página de ExpressJS en NPM](https://www.npmjs.com/package/express "página de ExpressJS en NPM") indica que la mismo posee más de 24 millones de descargas en la última semana. Además de encontrar más información como el repositorio en [GitHub](https://platzi.com/cursos/git-github/ "GitHub") y ejemplos básicos de instalación y uso de la librería, licencia, entre otros datos.

Como todo el código de NPM es público, te encontrarás con miles y miles de librerías deprecadas (viejas) o de dudosa procedencia. Es aconsejable que aprendas a analizar qué librerías utilizar y cuáles no convenga utilizar, ya que no tiene mantenimiento o muy poca gente la utiliza.

### Hola Mundo con ExpressJS

Así como ya has realizado el “Hola Mundo” con NodeJS, es momento de hacerlo con ExpressJS.

### Paso 1: Instalación de ExpressJS
Levantar tu primer servidor y exponer una API con ExpressJS es súper sencillo. Comienza instalando la librería de NodeJS utilizando el comando npm i express. En cuestión de segundos, tendrás disponible la utilización de esta librería en tu proyecto.

Tal vez te llame la atención la creación de un directorio en tu proyecto llamado node_modules. Dentro de esta carpeta, encontrarás muchas otras. Cada una de ellas hace referencia a una dependencia distinta y entre ellas, seguro encontrarás a ExpressJS.

También observa el archivo package.json, en la sección dependencies, se listarán cada una de las dependencias que tu proyecto utiliza con su respectiva versión.

```json
{
  ...
  "dependencies": {
      "express": "^4.18.1"
   }
}
```

### Paso 2: Implementado el servidor con ExpressJS

En la propia [página de ExpressJS](http://expressjs.com/en/starter/hello-world.html "página de ExpressJS"), o de cualquier tecnología, encontrarás muy buena documentación de cómo empezar a usar la misma.

Será importante, a lo largo de tu carrera como programador, que aprendas a buscar esta documentación y apoyarte de ella para aprender a utilizar una librería o una tecnología.

A continuación, observemos el siguiente código de Javascript que puedes utilizar en el archivo index.js de tu aplicación.

```javascript
// Importamos Express desde la carpeta node_modules
const express = require('express');

// Creamos la aplicación de Express
const app = express();

// Escojemos un puerto por el que el servidor web escuchará
const port = 3000;

// Página para visualizar el mensaje "¡Hola Express!"
app.get('/', (req, res) => {
  res.send('¡Hola Express!');
});

// Activamos el servidor en el puerto 3000
app.listen(port, () => {
  console.log(`¡Servidor listo!`);
});
```
### ¿Qué es un puerto?

Puedes entender el concepto de “puertos” en un computador como una puerta identificada por un determinado número por la cual es posible enviar y recibir información de un sistema o subsistema hacia otro.
En este ejemplo estamos utilizando el puerto 3000. Puedes utilizar el que tú quieras desde el número 1024 hasta el 65536. De momento no es relevante el porqué de este rango de números.

### Paso 3: Ejecutando el servidor
Luego de haber instalado la librería de Express y haber preparado el código para levantar un servidor web, es momento de ejecutarlo con el comando node index.js.

En cuestión de segundos, tu servidor estará escuchando en el puerto 3000. Esto quiere decir que si ingresas a `http://localhost:3000/`, en cualquier navegador web, podrás observar el mensaje `¡Hola Express!`.

Eso nos indica que has logrado levantar tu primer servidor web. ¡Felicidades!

### Conclusión

Así de sencillo es instalar dependencias y usarlas. Aún puede parecerte poco intuitivo todo lo que hemos realizado en esta clase, pero te aseguro que en poco tiempo serán tareas triviales de todos los días, la instalación de librerías, la lectura de su documentación para aprender a utilizarlas y su correcta implementación en tu código.

Instalar express: `npm install express`
Activar el servidor: `node index.js`

## Web server with Express.js

A lo largo de tu vida como programador o programadora, te encontrarás con **múltiples conceptos que debes conocer y que verás realmente en todas partes**. Protocolos, servidores, puertos, el funcionamiento de cada tecnología. Veamos a continuación un explicativo de **conceptos básicos que tienes que comenzar a interiorizar**.

### Estructura de un dominio

Las páginas web se identifican por un dominio único e irrepetible. Conocerás el dominio de Google ([https://google.com](https://google.com/ "https://google.com")) o el dominio de Platzi ([https://platzi.com](https://platzi.com/ "https://platzi.com")).

Los dominios son también llamados URI y están compuestos por varias partes.

### ¿Qué es una URI?

URI son las siglas en español de **Identificador de Recursos Uniforme** y es ese **identificador único de una página web**. El mismo está compuesto por dos partes, una URL (Localizador de Recursos Uniforme) y una URN (Nombre de Recurso Uniforme).

![Composición de una URI](./mokepon/assets/assetsREADME/ComposiciondeunaURI.png "Composición de una URI")

### Composición de una URI

Dentro de una URI, podemos identificas varias partes que componen a la misma:

![Partes de una URI](./mokepon/assets/assetsREADME/PartesdeunaURI.png "Partes de una URI")

**- URI:**
 - Esquema: Protocolo que la URI utiliza, pudiendo ser HTTP o HTTPS.
 - Dominio: Nombre del dominio de la página.
 - Puerto: Puerto por el que el servidor se encuentra “escuchando” para responder con la información de la página.
 - Ruta: Nombre de la página concreta que queremos solicitar dentro del dominio.
 - Cadena de búsqueda: Parámetros opcionales o variables para dar más información a la ruta.

- URN:
 - Nombre: Hace referencia a una sección particular dentro de una página. También denominado “fragmento”.
Entendiendo cómo se compone un dominio, profundicemos y comprendamos el qué y el porqué de cada parte.

### Protocolo HTTP

Conoces vagamente lo que es el protocolo HTTP o HTTPS. El Protocolo de Transferencia de Hipertexto es una **forma estandarizada de hacer las cosas o de transmitir información, en este caso**. La `S` de HTTPS significa Secure o seguro y permite la transferencia de datos codificados para evitar robo de información.

Tal vez recuerdes el significado de HTML (Lenguaje de Marcado de Hipertexto), mientras que el protocolo HTTP también hace referencia a esa palabra: “Hipertexto”. HTTP es el método de transferencia de este tipo de información.

Muchas veces el protocolo que utilizamos cuando utilizamos un navegador se encuentra implícito. Observamos la URI y solo vemos [platzi.com/home](http://platzi.com/home "platzi.com/home"), el HTTPS siempre se encuentra ahí, pero para hacer más amena la lectura de un dominio, los navegadores lo ocultan.

También cabe mencionar el famoso “WWW” que muchas veces acompaña a una URI. Puedes encontrarlo con el nombre de W3 o simplemente “La Web”. El mismo significa World Wide Web o Red Informática Mundial.

### Dominio de una página

El nombre propiamente dicho de una página se lo conoce como Dominio. Cuando hablamos de “la página de Platzi”, hacemos referencia a su dominio que es [https://platzi.com](https://platzi.com/ "https://platzi.com").

Observa también el `.com` que tantas veces has visto en otras páginas. **La extensión de los dominios hacen referencia al grupo al que este pertenece**. Pudiendo tener extensiones gubernamentales como `.gob`, extensiones propias de un país como `.ar` o `.mx` o extensiones de otro tipo como **.net** que también es muy utilizado, entre muchos otros tipos de extensiones.

Los dominios son también denominados como DNS (*Domain Name System*). Es un concepto algo más avanzado que no debes preocuparte en este momento, pero en pocas palabras, **es la forma de resolver y localizar una página web en internet en todo el mundo**.

Imagina que en el mundo existen cientos de miles y miles de servidores con páginas webs. ¿Cómo encontrar el servidor de Platzi? Los DNS resuelven esto y permiten que la página de Platzi llegue a tu navegador.

### Puertos

Los puertos son ese **canal por el que se intercambia información entre un cliente y un servidor o entre subsistemas**. Es un número que va del 0 al 65535 (2^16) y se aconseja escoger uno a partir del 1024. Muchas tecnologías o protocolos tienen un puerto por defecto ya establecido como el puerto 80 para HTTP o el puerto 443 para el HTTPS.

Si utilizamos un puerto que ya se encuentra en uso, podemos tener problemas con nuestra aplicación. Por este motivo, se utilizan a partir del puerto 1024, dado que de ahí para atrás muy posiblemente ya se encuentran en uso en tu computador.

A medida que ganes experiencia en múltiples tecnologías te encontrarás con que existen muchos otros puertos ya predefinidos como el puerto 21 para el protocolo FTP, 22 para SSH, 3306 para MySQL, entre otros. Para NodeJS, se suele utilizar el puerto 3000 o el 8080.

Los puertos suelen estar ocultos y no los verás en una URI. Cuando accedes a una página a través de HTTP, implícitamente estás utilizando el puerto 80 o el puerto 443 si se trata de HTTPS. Si decides cambiar el puerto (no es recomendable), si deberás hacer referencia al mismo como [platzi.com:3000/](platzi.com:3000/ "platzi.com:3000/").

### Ruta de una página

La ruta es el nombre de una página en particular dentro de toda una página web. El nombre de cada página es asignado por el propio programador. Si te encuentras desarrollando una página de un buscador, puedes denominar a la misma como `/buscador` o `/search`, la páginas principal de una web suele ser `/home` o simplemente `/`.

Te animo a explorar la ruta de cualquier página que visites. Verás que su nombre siempre está relacionado con el contenido de la misma.

### Parámetros de consulta

Los parámetros de consulta deben ser opcionales. Una página continuará funcionando o será correctamente localizada existan o no estos.

Los mismos comienzan luego de un `?` seguido del nombre de la variable y de su valor. Si una página tiene más de un parámetro, estos se separan con el caracter `&`. Por ejemplo: `?nombre=freddy&pais=colombia`.

Estos parámetros se suelen utilizar para crear buscadores. Son variables para crear filtros de búsquedas o pasarle información dinámica que será capturada por la aplicación y manipulada.

### Sección de una página
Dentro de una misma página encontrarás varias secciones. Las mismas pueden identificarse con un nombre en particular o un ID. En el URI, puedes hacer referencia a esta sección con un # seguido del nombre de dicha sección. Por ejemplo [https://platzi.com/home#routes](https://platzi.com/home#routes "https://platzi.com/home#routes").

Observarás que, al ingresar a este tipo de URI, serás dirigido directamente a esa sección dentro de la página. Es una forma de crear un “atajo” para el usuario cuando la página tiene mucho contenido.

### Localhost o servidor local

La palabra localhost será parte de ti de ahora en adelante. La misma significa “servidor local” y hace referencia a tu propio computador. Cuando levantas un servidor con NodeJS, puedes ingresar desde un navegador con `localhost:3000/` o con el puerto que hayas elegido.

Cualquier otra aplicación o programa que se encuentra ejecutándose en tu computador, también podrás acceder a él desde `locahost:<puerto>/`.

### Peticiones HTTP

Ya sabes lo que es el protocolo HTTP, pero aún hay un concepto más que debes interiorizar.

**Las solicitudes o peticiones que realices por medio de HTTP a un servidor puede ser de varios tipos**, cada uno de ellos destinado a un propósito específico. Los diferentes tipos de solicitudes HTTP se conocen como “Verbos HTTP”. Veamos de qué tipos existen:

**GET**: El verbo GET se utiliza para la **obtención de datos**. Es el más utilizado. Siempre que ingresas a una página web, la solicitud se realiza por GET.
**POST**: Utilizarás POST para la **creación de datos** o **registros**. POST tiene la particularidad de que codifica la información para el envío de datos secretos.
**PUT**: PUT se usa para la **actualización de datos,** para indicarle al servidor de la actualización completa de un registro.
**PATCH**: PATCH es muy similar a PUT, con la diferencia de que suele implementar para **actualizar un solo** dato de un registro.
**DELETE**: Así como puedes obtener, crear y actualizar información, DELETE lo utilizarás para el **borrado de datos**.

No son todos, aún hay más tipos, pero estos son los más utilizados y que tienes que comenzar a conocer de momento.

Muchos de los verbos HTTP son intercambiable. O sea, siempre podrás obtener datos a través de PUT o POST, o borrar los mismos a través de GET. **Las buenas prácticas de desarrollo de software**, y los buenos programadores, respetan las “reglas” del protocolo y te aconsejo que tú también lo hagas.

### Conclusión

Muchos conceptos, mucha información nueva para ti. Te aconsejo que vuelvas a ver esta clase en varias oportunidades para consolidar el conocimiento, ya que todos los conceptos vistos aquí, te acompañarán el resto de tu vida como desarrollador de software.