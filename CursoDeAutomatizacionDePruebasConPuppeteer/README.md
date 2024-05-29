# Curso de Automatización de Pruebas con Puppeteer

## ¿Qué es el DOM?

![DOM](images/dom.png)

El DOM (Document Object Model en español Modelo de Objetos del Documento) es una interfaz/modelo de documento.

Se representa el documento como un árbol de nodos, en donde cada nodo representa una parte del documento (puede tratarse de un elemento, una cadena de texto o un comentario).

- Los nodos tienen jerarquía
 
- Estos nodos pueden crearse, moverse o modificarse. -La mejor forma de visualizar los elementos es con las herramientas para desarrolladores de los navegadores. -Inspeccionar te permite incluso seleccionar un elemento en específico.

## Atributos y selectores

La barra de búsqueda que menciona el profesor y que él comenta que se puede obtener con las teclas Cmd + F o Ctrl + F.

![Barar de Busqueda](images/div.png)

Más adelante el video aunque no se ve lo que escribe directamente en la barra el profesor va describiendo lo que escribe y se ve en el resultado de la búsqueda en la pantalla.

Lo primero que escribe es: div span

Las propiedades y los atributos estan Relacionados pero No son lo mismo.

En **HTML**, los **atributos** y las propiedades son elementos esenciales que afectan la estructura, el comportamiento y la apariencia de una página web. Aquí está una breve explicación de cada uno:

1. **Atributos HTML:**
 - Los atributos son valores adicionales que se agregan a las etiquetas HTML para configurar o personalizar los elementos.
 - Se definen directamente en el código HTML y se utilizan para ajustar el comportamiento de los elementos.
 - Ejemplos comunes de atributos:
`src` en la etiqueta <img> para especificar la fuente de una imagen.
`href` en la etiqueta <a> para definir la URL de un enlace.
`class` en cualquier etiqueta para aplicar estilos CSS a elementos con propiedades en común.
 `alt` en la etiqueta <img> para proporcionar un texto alternativo en caso de que la imagen no se pueda mostrar.

2. **Propiedades:**
- Las propiedades son características específicas de un elemento que se pueden acceder y modificar mediante **JavaScript** o **CSS**.
- A diferencia de los atributos, las propiedades no se definen directamente en el código HTML, sino que se manipulan a través de **scripts** o **estilos**.

- Ejemplos de propiedades:

`innerHTML`: Contenido HTML dentro de un elemento.
`style`: Estilos CSS aplicados al elemento.
`value`: Valor de un campo de entrada (por ejemplo, `<input>`).
En resumen, los atributos son parte integral de las etiquetas HTML y se utilizan para configurar elementos, mientras que las propiedades son características que se pueden manipular dinámicamente mediante JavaScript o CSS. [Ambos son fundamentales para construir páginas web efectivas y funcionales12](https://developer.mozilla.org/es/docs/Web/HTML/Attributes "Ambos son fundamentales para construir páginas web efectivas y funcionales12").

## ¿Qué es Puppeteer?

Puppeteer es una librería mantenida por el equipo de **[Chrome DevTools](https://github.com/GoogleChrome/puppeteer/blob/master/CONTRIBUTING.md "Chrome DevTools")** que están continuamente liberando versiones y corrigiendo fallos para poder utilizar las últimas novedades de Chrome. A día de hoy es posible realizar con Puppeteer lo siguiente:

- **Simular navegación web**. Es posible automatizar el acceso a un portal pudiendo hacer clics en elementos, rellenar datos, hacer envíos de formularios, etc. Además, se puede elegir la emulación de la navegación utilizando un navegador de escritorio o móvil. Si alguna vez habéis trabajado con Devtools, conoceréis la forma para emular el comportamiento de una web en estos dispositivos.
- **Generar capturas de pantallas o informes PDF**.
- **Crear crawlers de páginas SPA o generar contenido pre-renderizado SSR**.
- **Analizar rendimiento de aplicaciones web** utilizando la herramienta Timeline Trace de Devtools.
- **Automatización de tests**, pudiendo realizar pruebas con las últimas versiones de Chrome y Javascript.
- **Probar extensiones de Chrome**.

Los *scripts* creados con Puppeteer pueden ser integrados con herramientas de terceros, para monitorizar, testear o automatizar tareas.

Por contra solo es posible utilizarlo bajo Chrome y usar como lenguaje Javascript.

**Puppeteer**: es una biblioteca de Node.js que proporciona una interfaz de alto nivel para controlar los navegadores web mediante el protocolo DevTools de Chrome o Chromium. Fue desarrollada por el equipo de Chrome en Google y se utiliza comúnmente para realizar tareas automatizadas en navegadores, como web scraping, capturas de pantalla, generación de PDF, pruebas automatizadas y más.

Algunas características clave de Puppeteer incluyen:

1. **Control de Navegadores**: Puppeteer permite abrir, cerrar y controlar instancias de navegadores Chrome o Chromium.
2. **Manipulación de Páginas Web**: Puedes interactuar con páginas web, hacer clic en elementos, llenar formularios, navegar por páginas, entre otras acciones.
3. **Capturas de Pantalla y Generación de PDF**: Puppeteer facilita la captura de pantallas y la generación de archivos PDF de páginas web.
4. **Evaluación de Páginas**: Puedes ejecutar scripts en el contexto de la página que estás controlando, lo que permite realizar operaciones más avanzadas.
5. **Simulación de Dispositivos y Red**: Puppeteer permite emular diferentes dispositivos y configuraciones de red para probar cómo se comporta una página en distintos escenarios.
6. **Pruebas Automatizadas**: Se utiliza comúnmente en pruebas automatizadas para asegurar que las aplicaciones web se comporten como se espera.

Un caso de uso muy común de Puppeteer es el web scraping, donde puedes automatizar la extracción de datos de páginas web. Algunos de los comandos que se ven en los scripts de Puppeteer, como **page.goto**, **page.click**, y **page.evaluate**, son utilizados para navegar por el sitio, interactuar con elementos y ejecutar scripts en la página.

Para comenzar a usar Puppeteer, primero debes instalarlo en tu proyecto Node.js mediante npm:

`npm install puppeteer`

Después de la instalación, puedes importar Puppeteer en tu script y comenzar a utilizar sus funciones para interactuar con el navegador web de manera programática.