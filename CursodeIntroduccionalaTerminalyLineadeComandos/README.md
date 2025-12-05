# Curso de Introducción a la Terminal y Línea de Comandos

## Ventajas de dominar la terminal de comandos para profesionales

Dominar la terminal de comandos es una habilidad estratégica para cualquier profesional técnico. Lejos de ser algo anticuado, conocer esta herramienta aporta velocidad, precisión y un conocimiento profundo sobre el funcionamiento interno de sistemas operativos y aplicaciones. Con la terminal, automatizas tareas en segundos y accedes a herramientas avanzadas, ganando control total sobre tu entorno digital.

### ¿Qué ventajas ofrece el uso constante de la terminal?
El uso habitual de la terminal se refleja en una gestión más eficiente y precisa del tiempo de trabajo. Profundizar en esta habilidad permite ejecutar en segundos operaciones que normalmente tomarían horas a través de interfaces gráficas.

Entre los principales beneficios están:

- Automatización de tareas repetitivas. Acortas considerablemente el tiempo dedicado a procesos manuales.
- Control total y preciso del sistema. Conoces exactamente qué ocurre al ejecutar un comando determinado.

### ¿De qué manera mejora la terminal tu conocimiento del sistema operativo?

Al aprender a usar la terminal, no solamente mejoras tus competencias técnicas inmediatas. También te acercas de manera directa al manejo real del sistema operativo, observando claramente cómo se gestionan los archivos, directorios y procesos internos:

- Comprendes profundamente cómo se mueve la información en tu sistema.
- Adquieres claridad acerca de su estructura y rendimiento.

### ¿Qué herramientas avanzadas puedo utilizar con dominio de la terminal?

Múltiples herramientas avanzadas únicamente están disponibles mediante la línea de comandos. Con ellas, tu campo de trabajo se amplía significativamente hacia áreas especializadas e innovadoras:

- Git para control de versiones.
- Docker en gestión de contenedores.
- Htop para monitoreo eficiente de procesos.
- Nmap en analítica de redes y sistemas.

### ¿Cómo logro personalizar mi entorno digital desde la terminal?

Una de las grandes fortalezas de la terminal radica en la posibilidad de personalización extrema. Crear soluciones concretas adaptadas a tus necesidades específicas proporciona eficacia y versatilidad:

- Diseñas tus propios scripts.
- Desarrollas flujos de trabajo personalizados.
- Adaptas tu ambiente digital exactamente a lo que tu actividad requiere.

### ¿Es aplicable la terminal en cualquier plataforma?
Si bien puede existir la percepción de que la terminal es exclusiva de ciertos sistemas operativos, su uso y beneficio se extienden ampliamente:

- Windows, Linux, Mac. Cualquier sistema operativo importante cuenta con una terminal.
- Incluso dispositivos móviles poseen herramientas similares que permiten acceder a funciones avanzadas muchas veces desconocidas por la mayoría de usuarios.

### ¿Por qué aprender terminal aumenta tus oportunidades laborales?

El dominio de la terminal es altamente valorado en grandes empresas tecnológicas, donde eficiencia, rapidez y precisión técnica son fundamentales. Compañías como Google, Amazon, Facebook, Corey Cold y Red Hat destacan:

- Emplean profesionales con dominio total de la terminal.
- Consideran esta habilidad básica e imprescindible.

## Qué es una terminal y cómo funciona con comandos básicos

Una terminal es básicamente una interfaz de usuario que facilita la comunicación con el sistema operativo mediante comandos específicos. Estos comandos te permiten realizar acciones concretas, como crear, copiar o eliminar archivos, además de administrar procesos y otras tareas importantes de configuración.

### ¿Cómo funciona una terminal o Shell?

El funcionamiento de una terminal se apoya en un programa llamado shell. Este programa interpreta los comandos que introduces y los traduce en instrucciones que el sistema operativo puede entender y ejecutar. La shell convierte estos comandos escritos en acciones concretas visibles en tu equipo.

### ¿Cuáles son las tareas típicas que se hacen desde una Shell?

Desde una shell, puedes realizar diversas acciones útiles como:

- Crear, copiar y eliminar archivos.
- Administrar procesos y configurar opciones específicas del sistema.
- Navegar y listar directorios para consultar contenido alojado en tu dispositivo.

### ¿En qué sistemas operativos puedes utilizar una terminal?

Aunque la mayoría de las terminales están asociadas con sistemas Linux y Unix, es posible emplearlas también en Windows mediante herramientas específicas como PowerShell o Bash a través del sistema WSL. La elección de la terminal dependerá del sistema operativo y del entorno de trabajo.

### PowerShell vs Bash: ¿cuál utilizar en Windows?

PowerShell es una terminal específica para Windows que ofrece comandos únicos para este sistema operativo. Es utilizada principalmente en servidores Windows. Por otro lado, la terminal Bash, frecuente en Linux y sistemas Unix, es ampliamente utilizada a nivel global y será explorada mediante emulación WSL en este curso.

### ¿Dónde encontrar terminales?

Las terminales no están limitadas a un espacio concreto y pueden aparecer en diferentes entornos, por ejemplo:

- Desde el buscador en PowerShell de Windows.
- En la sección de terminal dentro de programas como Visual Studio Code.
- En páginas web o aplicaciones de teléfonos celulares conectadas adecuadamente.

Dominar el uso de una terminal es accesible para cualquier persona, no es necesario ser experto en tecnología. Todo radica en entender y ejecutar comandos básicos para facilitar diversas acciones informáticas con tu computadora.

¿Ya has utilizado antes alguna terminal? Cuéntanos tu experiencia en los comentarios.

### Comandos Powershell

`dir` = Muestra los directorios donde se  encuentra la terminal

## Instalación de terminal Bash en Windows usando WSL

[Instalación de WSL | Microsoft Learn](https://learn.microsoft.com/es-es/windows/wsl/install)

Para quienes comienzan en tecnología y usan Windows, aprender a instalar un terminal Bash puede ser clave para facilitar tareas y mejorar la eficiencia en proyectos tecnológicos. Utilizando Windows Subsystem for Linux (WSL), es posible ejecutar una versión emulada de Linux en Windows 10 u 11 fácilmente y sin complicaciones.

### ¿Qué es Windows Subsystem for Linux (WSL)?

Windows Subsystem for Linux, conocido como WSL, es una utilidad especial diseñada por Microsoft que permite ejecutar un sistema operativo Linux directamente desde Windows sin necesidad de máquinas virtuales o configuraciones complejas:

- Funciona principalmente en Windows 10 versión 2004 en adelante y Windows 11.
- Facilita la ejecución de comandos y operaciones propias de sistemas Linux mediante la terminal Bash.
- Incluye, por defecto, la distribución de Linux llamada Ubuntu, basada en Debian.

Esta comodidad ofrece la flexibilidad de experimentar y trabajar con comandos Linux directamente desde el entorno Windows, lo cual es ideal para desarrolladores y estudiantes.

### ¿Cómo instalar WSL en Windows paso a paso?

Instalar WSL en Windows involucra unos cuantos pasos sencillos, que requieren atención a ciertos detalles importantes:

### Verifica tu compatibilidad y requisitos previos

Antes de comenzar, asegúrate de contar con lo siguiente:

- Windows 10 (versión 2004 o posterior) o Windows 11 actualizado.
- Acceso a internet estable debido a la descarga de archivos que pueden ser de gran tamaño.
- Permisos de administrador para ejecutar comandos desde PowerShell.

**Instalación mediante comandos**

Para iniciar la instalación:

1. Abre la PowerShell de Windows en modo administrador:
2. Busca PowerShell desde el menú inicio.
3. Haz clic derecho y selecciona Ejecutar como administrador.
4. Una vez abierto PowerShell como administrador, ejecuta el comando siguiente:

`wsl --install`

Este comando instalará automáticamente el subsistema de Linux en tu equipo, además de descargar e instalar Ubuntu como distribución predeterminada.

**Configuración inicial de WSL y Ubuntu**

Tras la instalación:

- Busca en el menú "WSL", identifica el icono con forma de pingüino y ábrelo.
- Se iniciará una terminal especial donde se provisionará tu instancia personalizada de Ubuntu.

Aquí deberás:

- Crear un usuario nuevo (ejemplo: Code Bars).
- Establecer y memorizar una contraseña personal (es requerida en futuras operaciones).

Luego podrás verificar que Bash se instaló correctamente ejecutando el comando:

`echo $SHELL`

Si obtienes como respuesta `/bin/bash`, la terminal se encuentra lista y funcionando correctamente.

**Opciones alternativas de instalación**

Si experimentas complicaciones técnicas específicas como limitaciones de hardware o permisos que impiden la instalación con WSL, considera estas opciones:

- Instalar Ubuntu directamente en tu computadora.
- Si eres usuario de Mac, el sistema operativo ya incluye una terminal Bash por defecto, aunque con ligeras diferencias en algunos comandos por ser un sistema Unix en lugar de Linux.

Si tienes dudas o dificultades relacionadas con estos pasos, siéntete libre de compartirlo con nosotros en los comentarios y seguir avanzando en tu aprendizaje.

### Comandos

`wsl -l` = Muestra la version de linux
`echo $SHELL` = Sirve para mostrar la ruta del shell actual

## Comandos básicos de terminal para principiantes

Dominar el uso de la terminal puede parecer complicado al principio, pero conociendo comandos básicos como `whoami`, `pwd`, `ls` y `clear` puedes simplificar considerablemente muchas tareas. Detectar tu usuario actual, mostrar dónde están almacenados tus archivos o visualizar información clave del sistema son solo algunas acciones que estos comandos permiten realizar fácilmente.

### ¿Cómo identificar tu usuario actual en la terminal?

El comando `whoami` es sencillo pero esencial. Muestra con claridad el nombre de usuario con el que has iniciado sesión en tu terminal:

`whoami`

Al ejecutarlo, inmediatamente sabrás cuál es tu perfil activo en ese momento.

### ¿Dónde está tu directorio actual y cómo revisar su contenido?

Podemos conocer exactamente en qué directorio estamos utilizando el comando `pwd` (print working directory):

`pwd`

Esto es útil para ubicarnos rápidamente dentro del sistema de archivos. Para listar el contenido, se usa ls. En su forma simple solo muestra archivos visibles, pero al agregar opciones puedes revelar mucho más:

- Mostrar todo, incluyendo ocultos: `ls -a`.
- Visualizar detalles adicionales como permisos y tamaño: `ls -l`.
- Combinar ambas opciones para más información útil y fácilmente comprensible: `ls -la` o incluso en formato legible para humanos: `ls -lah`.

`ls -lah`

### ¿Cómo visualizar información específica o realizar tareas en la terminal fácilmente?

La terminal ofrece comandos útiles para diversas situaciones prácticas, tales como:

- Limpiar la pantalla completa, usando `clear` o la combinación rápida `Ctrl + L` (en Mac, `Command + L` también lo logra).
- Imprimir mensajes en consola con echo, útil en automatizaciones y scripts:

`echo "Hola Mundo"`

- Visualizar información del sistema operativo y fecha actual con:

```shell
uname -a
date
```

- Acceder al manual completo de cualquier comando usando `man`. Por ejemplo, para aprender más sobre el uso del comando `echo`:

`man echo`

Este manual proporciona descripciones detalladas, opciones adicionales, métodos de uso y autores de cada herramienta que desees explorar.

### ¿Qué significan los colores y símbolos que muestra la terminal?

Interpretar correctamente los símbolos y colores facilita la comprensión del listado de contenidos. En general, los directorios aparecen en azul y comienzan con la letra "d", mientras que los archivos suelen ser blancos. Los caracteres adicionales, letras y números en los listados indican permisos específicos que aprenderás a interpretar más adelante.

Te recomendamos practicar regularmente estos comandos para familiarizarte más rápidamente con su funcionamiento. ¿Con cuál te gustaría empezar a experimentar en profundidad?

**Resumen de comandos:**

- `whoami`: Muestra el nombre de usuario
- `pwd` "Print working directory"
- `ls`:
 - `ls -a`: Indica al comando que muestre todos los archivos.
 - `ls -l`: Lista los archivos solamente los visibles.
 - `ls -la`: Muestra toda la información de los archivos
- `ls -al`: Cambio de las opciones
- `clear "Ctrl" o "cmd" + "l"`: Borra la información de nuestra pantalla de la terminal
- `echo`: Nos va a mostrar un mensaje personalizado en la pantalla.
- `--help`: Una flag que da información de cualquier comando.
- `--help=important`: es un parámetro de ejemplo.
- `-h`: opción del comando "`ls`" que humaniza el resultado. Nos muestra información acerca del tamaño del archivo.
- `uname -a`: Muestra información de nuestro sistema
- `date`: muestra la fecha
- `man`: Es el manual de cualquier otro comando. Se usa luego de cualquier comando y nos mostrará todas las opciones.

## Navegación entre directorios en Linux con comandos de terminal

Dominar la navegación entre directorios en sistemas Linux es una habilidad fundamental para interactuar con el sistema operativo mediante comandos desde la terminal. Aprenderemos cómo desplazarnos eficientemente usando rutas absolutas y relativas, así como símbolos útiles como punto (.), doble punto (..), slash (/) y virgulilla (~).

### ¿Qué es navegar con rutas absolutas?

Una ruta absoluta comienza siempre desde la raíz del sistema operativo, especificada por un slash (/). Por ejemplo, usar el comando `cd /` nos posiciona directamente en el directorio raíz del sistema Linux, que aloja importantes carpetas del sistema tales como:

- bin
- dev
- lib64
- root
- home

Este método permite acceder directamente a cualquier directorio proporcionando su ruta completa.

### ¿Cómo funcionan las rutas relativas y símbolos especiales?

Mientras que las absolutas requieren especificar toda la ruta desde la raíz, las relativas funcionan desde el lugar actual en que nos encontramos. Uso frecuente de símbolos clave:

- Punto (`.`) para indicar el directorio actual.
- Doble punto (`..`) para subir un nivel de directorio.
- Virgulilla (`~`) para referencia rápida al directorio home del usuario actual.

Esto permite desplazarnos rápidamente usando comandos como `cd ..` para retroceder y `cd ~` para ir al home del usuario rápidamente, independientemente del directorio actual.

### ¿Qué son y cómo se emplean los comandos `pushd` y `popd`?

Ambos comandos facilitan almacenar temporalmente una ubicación y regresar posteriormente a ella:

- `pushd`: guarda la ubicación actual en una pila.
- `popd`: recupera la última ubicación almacenada y nos desplaza automáticamente a ella.

Este mecanismo es muy útil para navegar cómodamente cuando trabajamos en múltiples directorios simultáneamente.

¡Comienza a practicar estos comandos en tu sistema Linux para familiarizarte con la organización y navegar fácilmente por tu terminal! ¿Tienes alguna consulta sobre este proceso? ¡Comenta debajo!

## Comandos Linux para crear, mover, copiar y eliminar archivos

Manejar archivos y directorios desde la terminal de Linux es esencial para quienes trabajan con grandes volúmenes de datos, especialmente en áreas como data science. La interfaz de línea de comandos ofrece rapidez y eficiencia incomparable con las interfaces gráficas tradicionales, optimizando considerablemente las tareas diarias.

### ¿Cómo crear archivos y directorios con comandos básicos?

Los archivos se crean fácilmente con el comando `touch`. Por ejemplo, generamos un archivo de texto usando:

`touch archivo.txt`

Para crear directorios, se utiliza `mkdir`. Puedes hacerlo de manera sencilla:

`mkdir documentos`

Al añadir la flag `-p`, puedes crear varias subcarpetas simultáneamente:

`mkdir -p documentos/escuela/matematicas`

### ¿De qué forma mover o renombrar archivos y directorios fácilmente?

Mover elementos se hace con el comando `mv`. Por ejemplo, para mover la carpeta matemáticas fuera de escuela:

`mv escuela/matematicas .`

Para renombrar directorios o archivos también empleas `mv`. Por ejemplo, cambiar la carpeta escuela a colegio:

`mv escuela colegio`

### ¿Qué necesitas saber para copiar y eliminar archivos/directorios con seguridad?

Para copiar archivos utilizas el comando `cp`:

`cp saludo.txt adios.txt`

Cuando copias directorios completos, debes aplicar el modo recursivo con `-r`:

`cp -r documentos documentos_respaldo`

Para eliminar archivos, puedes usar `rm`, pero con mucho cuidado, ya que no existe papelera en la terminal:

`rm archivo.txt`

Eliminar directorios requiere la opción recursiva `-r` y la opción `-f` para forzar la eliminación, siendo extremadamente cuidadoso:

`rm -rf documentos_respaldo`

Te recomiendo siempre verificar los comandos ejecutados usando opciones menos agresivas o el modo interactivo para evitar errores críticos.

¿Te animas a probar estos comandos con diferentes opciones y descubrir nuevas posibilidades? ¡Comparte tu experiencia o duda en los comentarios!

## Comandos para explorar y manipular archivos de texto en terminal

En cualquier sistema operativo, los archivos de texto plano son esenciales al procesar datos o descargar información específica. Para interactuar con ellos desde una terminal, existen comandos avanzados que permiten una manipulación efectiva y cómoda. A continuación, encontrarás claves prácticas para explorar y manipular archivos de texto mediante comandos como `cat`, `less`, `head`, `tail`, `nl`, `wc` y `awk`.

### ¿Cómo explorar rápidamente el contenido de archivos en terminal?

Uno de los comandos más utilizados para visualizar contenido rápidamente es `cat`, que muestra todo el texto dentro del archivo instantáneamente. Sin embargo, para archivos largos como los CSV, `cat` puede resultar poco práctico, ya que muestra mucha información de golpe.

Para una visualización más controlada y con la posibilidad de navegar dentro del archivo, existe el comando `less`, que presenta una interfaz interactiva donde:

- Puedes moverte con facilidad por el texto.
- Para salir de la visualización, solo debes presionar la tecla "q".

### ¿Qué comandos son útiles para mostrar partes específicas de un archivo?

Cuando buscas enfocarte solo en un segmento particular del archivo, son especialmente útiles los comandos:

- `head`: muestra las primeras líneas de tu archivo.
- `tail`: presenta las últimas líneas del archivo.

Ambos tienen opciones para especificar cuántas líneas deseas visualizar. Por ejemplo, con la opción `-n`:

- `head -n 20 nombre_archivo`: muestra las primeras 20 líneas.
- `tail -n 20 nombre_archivo`: presenta las últimas 20 líneas.

### ¿Cómo obtener información detallada del archivo?

Para información adicional sobre archivos de texto, puedes emplear diversos comandos:

### ¿Contar líneas y palabras fácilmente?

- ``nl``: numera las líneas directamente desde la terminal, facilitando la identificación rápida del texto dentro de un archivo.
- ``wc`` (Word Count): brinda múltiples posibilidades para obtener estadísticas textuales claves, incluyendo:
- Cantidad total de palabras (``wc archivo -w``).
- Número total de líneas (``wc archivo -l``).

### ¿Manipular archivos CSV con eficiencia?

El comando `awk` es una herramienta poderosa diseñada especialmente para manipular y explorar archivos CSV:

- Permite seleccionar columnas específicas con sintaxis sencilla. Por ejemplo, para imprimir la primera columna:

`awk '{print $1}' archivo.csv`

- Es posible imprimir múltiples columnas indicando separadores (`-F`) y columnas específicas con `$`:

`awk -F"," '{print $1, $3}' archivo.csv`

Los archivos CSV suelen usar comas para separar sus valores (`Comma Separated Values`), facilitando tanto su exploración como análisis.

### ¿Cómo aprovechar al máximo estos comandos?

Además de aplicar los comandos mencionados, invita la exploración adicional consultando los manuales en línea con `man`. Investigar las opciones y banderas específicas de cada comando ayudará a obtener mejores resultados al trabajar con información textual y al realizar análisis de sistemas operativos o archivos de registros (`logs`).

¡Te invito a experimentar con estos comandos en diferentes contextos y a compartir tus experiencias y consultas en los comentarios!

## Uso de wildcards para búsquedas masivas en la terminal

Dominar los wildcards o caracteres comodines en la terminal es una habilidad esencial para trabajar eficientemente con archivos. Los wildcards son caracteres especiales que te permiten hacer coincidir patrones específicos en nombres de archivos, simplificando enormemente tareas repetitivas como listar, copiar o mover múltiples archivos rápidamente.

### ¿Qué es un wildcard y para qué sirve?

Un wildcard es un carácter especial utilizado como comodín para hacer coincidir múltiples archivos en base a un patrón determinado. Principalmente, es útil con comandos como:

- `ls` (listar archivos)
- `cp` (copiar archivos)
- `mv` (mover archivos)
- `rm` (eliminar archivos)
-Y otros comandos comunes de Unix/Linux, como `head`, `tail` o `grep`.

### ¿Cuáles son los principales tipos de wildcards?

- Asterisco (*): representa cualquier combinación de caracteres.
- Signo de interrogación (?): coincide específicamente con un único carácter.
- Corchetes [ ]: agrupan caracteres específicos.
- Llaves { }: agrupan patrones o palabras.

### ¿Cómo usar el wildcard asterisco (*) en tus comandos?

El wildcard más común es el asterisco (*), que coincide con cualquier combinación de caracteres. ¿Cómo aplicarlo?

- Para listar archivos con una extensión específica:

`ls *.txt`

- Para listar todos los archivos que comiencen con una palabra específica:

`ls file*`

### ¿Cómo funciona el wildcard del signo de interrogación (?)?

Este comodín hace que coincida únicamente un solo carácter en la posición exacta del patrón. Ejemplo:

`ls file?.txt`

El comando anterior listará "file1.txt", "file2.txt" pero no "filelargo.txt".

### ¿Cómo usar corchetes para buscar por caracteres específicos?

Con corchetes [] agrupamos caracteres precisos para afinar aún más la búsqueda:

- Listar archivos terminados en letra específica antes del punto:

`ls *[o].*`

Esto listará archivos como "archivo.ccv" o "filelargo.txt" que tienen una "o" antes del punto.

### ¿Cómo usar agrupación con llaves { } para patrones específicos?

Las llaves te permiten indicar diferentes patrones o palabras específicas de manera sencilla y potente:

- Para listar archivos con extensiones específicas múltiples:

`ls *.{md,log}`

Esto mostrará todos los archivos terminados en ".md" y ".log", como "data.log" y "fileb.md".

### Ejemplos prácticos para mover archivos en lotes

Usar wildcards también es útil para organizar archivos rápidamente:

```bash
mkdir backup
mv *.txt backup/
```

Con estos comandos, mueves todos los archivos `.txt` hacia una carpeta llamada backup.

### Precauciones al usar comandos con wildcards

Aunque los wildcards son extremadamente útiles, requieren precaución para evitar acciones involuntarias como borrar archivos esenciales. Recuerda siempre verificar bien el comando antes de ejecutarlo, especialmente usando comandos como `rm`.

Además, toma en cuenta que los wildcards pueden variar ligeramente según la Shell que utilices (bash, ZSH, entre otras). Si algo no funciona exactamente como esperas, verifica la documentación específica de tu Shell.

¿Ya utilizas wildcards regularmente en tu terminal? ¡Cuéntanos sobre tu experiencia o tus trucos favoritos en los comentarios!

## Comandos GREP y FIND para búsquedas avanzadas en Linux

Aprender a buscar archivos y contenidos específicos en Linux es esencial para optimizar tus tareas cotidianas en este sistema operativo. Dos comandos clave que debes dominar son GREP y FIND, ambos utilizan expresiones regulares, permitiendo búsquedas avanzadas por tipo de archivo, contenido específico, tamaño o ubicación.

### ¿Qué es el comando GREP y cómo usarlo?

El comando GREP es una potente herramienta que permite buscar cadenas o patrones específicos dentro del contenido de archivos. GREP facilita la obtención y filtrado de información específica en archivos, fundamentalmente útil en análisis de datos o administración de sistemas.

### ¿Cómo buscar patrones específicos con GREP?

Para comenzar la búsqueda, GREP espera dos cosas fundamentales: el patrón que buscas y la ubicación del archivo a revisar. A continuación, ejemplo práctico:

`grep -i "spider" marvel_wiki.csv`

De esta forma, GREP buscará la palabra "Spider" sin importar si contiene mayúsculas o minúsculas.

### ¿Cómo contar coincidencias con GREP?

Si deseas contar cuántas veces aparece un término, utiliza la opción -c:

`grep -i -c "spider" marvel_wiki.csv`

Con esto GREP entregará directamente el número de líneas donde aparece la palabra "spider".

### ¿Cómo excluir patrones de una búsqueda con GREP?

Para mostrar líneas que no coincidan con tu búsqueda específica:

`grep -i -v "spider" marvel_wiki.csv`

Este método es útil cuando incluyes filtros para analizar grandes cantidades de datos.

### ¿Qué es el comando FIND y cómo usarlo para búsquedas avanzadas?

Mientras GREP busca dentro del contenido, el comando FIND te permite localizar archivos o directorios por nombre o atributos como tamaño o tipo, partiendo desde una ubicación específica.

### ¿Cómo buscar carpetas con FIND?

Para buscar todos los directorios desde tu ubicación actual:

`find . -type d -name "*"`

La opción `type` permite especificar qué buscar: `d` para directorios y `f` para archivos.

### ¿Cómo buscar archivos según tamaño con FIND?

Buscar archivos mayores a cierto tamaño (por ejemplo, 1 MB):

`find . -type f -size +1M`

Este comando devuelve la ruta directa y simplifica enormemente la gestión de grandes volúmenes de archivos según sus características.

### ¿Qué son las expresiones regulares y para qué sirven?

Las expresiones regulares no son más que patrones específicos que buscan correspondencias en texto. Actúan como moldes que encuentran coincidencias exactas o aproximadas, facilitando tareas avanzadas de filtros y búsquedas.

Si aún no estás familiarizado con expresiones regulares, explora algún curso especializado para entender mejor esta potente herramienta.

¿Has utilizado estos comandos en tu trabajo diario? Cuéntanos cómo te han facilitado tus tareas en Linux.

## Tipos de comandos en Linux y cómo identificarlos

Cuando trabajamos en Linux, usualmente ejecutamos comandos sin reflexionar acerca de su naturaleza y origen. Un comando en Linux puede referirse a varias cosas: desde un script en Shell, pasando por archivos binarios compilados en lenguajes como C++, hasta utilidades propias del sistema o alias personalizados.

### ¿Cómo identificar qué es exactamente cada comando?

Son varias las herramientas que Linux ofrece para identificar el tipo y las características de los comandos que usamos diariamente. Aquí exploraremos algunas que debes conocer:

### ¿Qué es un alias en Linux y cómo reconocerlo?

Un alias funciona como un apodo en el sistema operativo Linux, permite ejecutar comandos con ciertas opciones de manera más fácil o vistosa. Por ejemplo, el comando `ls` que generalmente usamos para listar directorios, en realidad es un alias de `ls --color=auto` para mostrar resultados con colores. Para verificar si un comando es un alias, puedes ejecutar:

`type ls`

Esta instrucción genera una salida que indica si el comando es un alias y cuál es su composición específica.

### ¿Dónde se ubican los comandos originales?

Si deseas encontrar la ruta precisa del comando que estás usando, la instrucción `which` será fundamental:

`which ls`

Este comando te mostrará la ubicación exacta del archivo ejecutable original, distinto al alias que podrías estar usando.

Además, para obtener más ubicaciones relacionadas con un comando específico, puedes utilizar `whereis`. Este comando te dará un panorama más amplio:

`whereis ls`

Generalmente verás rutas como `/usr/bin/ls`, indicando dónde residen la mayoría de los comandos importantes y esenciales del sistema operativo.

### ¿Cómo obtener información rápida sobre la función de un comando?

Para obtener rápidamente información sobre qué función cumple cualquier comando, utiliza `whatis`. Este comando te devuelve breve y claramente la tarea principal del comando consultado:

`whatis grep`

La respuesta, en este caso, indica que `grep` es capaz de encontrar líneas según patrones específicos.

### ¿Qué tipos de comandos existen en Linux?

En términos prácticos, los comandos ejecutables dentro de Linux pueden pertenecer a alguna de estas categorías:

- Binarios compilados: usualmente creados en lenguajes como C++.
- Scripts en diversos lenguajes: incluyendo shell scripts o scripts de Python o JavaScript.
- Alias: comandos personalizados que simplifican o embellecen la experiencia visual en la terminal.
- Utilidades propias del sistema operativo: herramientas básicas esenciales para la operación general del sistema.

Te recomendamos explorar dentro del directorio `/usr/bin`, donde se ubica gran parte de los comandos estándares del sistema. Es importante que recuerdes que algunos comandos adicionales pueden provenir de paquetes gestionados mediante `npm` (JavaScript), paquetes de Python u otros lenguajes que generen binarios globales.

Comparte en los comentarios cuáles comandos interesantes descubriste al explorar `/usr/bin` usando las herramientas mencionadas.

## Redirecciones de terminal en Linux con operadores básicos

Las redirecciones del sistema son una herramienta esencial para manejar eficientemente la información que generan los comandos en la terminal de Linux. Al utilizar operadores básicos, como mayor que (>), menor que (<), doble mayor que (>>) y pipe operator (|), podemos redirigir salidas, entradas y errores hacia diferentes destinos, ampliando así las posibilidades de nuestro trabajo en terminal.

### ¿Qué es una redirección de terminal y para qué sirve?

Una redirección permite transferir la salida estándar (standard output) de cualquier comando hacia un archivo de texto o como entrada estándar (standard input) de otro comando. Esto es fundamental porque:

- Puedes almacenar resultados que normalmente aparecen en pantalla directamente en archivos.
- Facilita el envío de datos entre comandos en una cadena o flujo.

### ¿Cómo almacenar resultados de comandos en archivos?

La forma más sencilla consiste en utilizar el operador de redirección mayor que (>), como en el siguiente ejemplo:

```bash
echo "hola mundo" > archivo_hola.txt
cat archivo_hola.txt # Resultado: hola mundo
```

Si deseas agregar contenido adicional en lugar de sobreescribir, utilizas un doble mayor que (>>):

```bash
echo "hola personas" >> archivo_hola.txt
cat archivo_hola.txt # Resultado: hola mundo, hola personas
```

### ¿Cómo funciona la redirección de entradas y salidas entre comandos?

Es posible usar la salida de un comando como entrada directa (standard input) de otro utilizando el denominado pipe operator (|). Por ejemplo, los comandos `LOLCAT` o `Cowsay` usados habitualmente en ejercicios educativos, pueden recibir entradas de esta manera:

```bash
echo "saludo colorido" | lolcat
cat archivo_hola.txt | lolcat
cowsay "hola mundo" | lolcat
```

### ¿Cómo capturar errores utilizando la terminal en Linux?

Los errores que se generan al ejecutar comandos pueden ser almacenados específicamente. En Linux, el flujo de errores estándar (standard error) es referenciado con el número 2, y puede capturarse separadamente así:

```bash
ls archivo_inexistente 2> errores.log
cat errores.log # Contiene el mensaje de error correspondiente
```

Si quieres concatenar varios errores dentro de un mismo archivo, la redirección se realiza con:

```bash
# Utilizando doble mayor que,
ls archivo_inexistente 2>> errores.log
```

### ¿Cómo registrar salidas y errores simultáneamente?

Otra práctica frecuente es guardar tanto la información correcta generada como los errores en un único archivo.

`sudo apt install vim &> instalación.log`

Aquí, `&>` indica que se almacenarán ambas informaciones (salida y error estándar) en "instalación.log".

¿Ya utilizas estos comandos habitualmente en tu trabajo o estudio con Linux? Cuéntanos tu experiencia y cómo aprovechas estas herramientas en tu día a día.