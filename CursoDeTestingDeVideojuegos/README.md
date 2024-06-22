# Curso de Testing de Videojuegos

## Qué aprenderás sobre el testing de videojuegos

¿Qué significa ser tester?
¿Por qué es importante el control de calidad?
¿Qué se espera de ti como tester?

Ricardo Izquiero ha trabajado en la industria de los videojuegos desde el 2006, en empresas como Electronic Arts, PyroStudios y King. Su trabajo se ha centrado en el área de control de calidad. En este curso Ricardo te comparte los conocimientos que ha adquirido en estos 12 años de experiencia para enseñarte a ser un tester de videojuegos. Pero, ¿qué es un tester? Es la persona que se asegura de que el videojuego tenga la mejor calidad al momento del lanzamiento. De todos los departamentos que existen en la industria de videojuegos el más estricto es el de Control de Calidad.
Somos la última línea de defensa antes de que el producto salga a la luz y esa responsabilidad recae sobre testing.

## ¿Cómo funciona el testing de Black Box en un estudio de videojuegos?

Hay dos áreas dentro del testing:
**Whitebox**: se refiere al testeo del código fuente del juego.
**Blackbox**: se refiere al testeo desde el punto de vista del usuario final. Este curso se enfica exclusivamente al testing de blackbox.
En esta clase veremos cómo funciona un estudio de videojuegos. Se parte de una idea que propone el departamento de diseño, luego el departamento de programación hace un prototipo sobre el cual el departamento de grafismo implementa los gráficos. El resultado de este ciclo es la primera versión del juego. A continuación, el equipo de control de calidad (testing) hace una serie de pruebas para detectar los errores de esta primera versión. Programación hará las correcciones necesarias y las enviará nuevamente a pruebas con el departamento de testing. Este ciclo se repite las veces necesarias para garantizar que el videojuego funcione como debería. Cuando se tiene una versión que funciona a la perfección, el departamento de diseño vuelve a entrar para implementar más detalles en el juego, los cuales serán programados y graficados por los departamentos de programación y grafismo, respectivamente. Este ciclo se repite para cada versión del juego (también llamada build).

## Diferencia entre los 2 QA’s

QA es una sigla que puede traducirse como “Quality Assurance” (control de calidad) y “Quality Assistance” (asistencia de calidad). En esta clase veremos la diferencia entre ambos términos.
**Quality Assurance**: somos nosotros los responsables directos de la calidad del producto. Firmamos un documento asegurando que el juego está listo para ser lanzado. Y, en caso de que no lo esté, tenemos la potestad de detener el lanzamiento.
**Quality Assistance**: somos asesores de calidad y damos nuestro veredicto sobre calidad, pero no somos los responsables. No podemos detener un lanzamiento, sólo el estudio tiene la potestad.

Nosotros somos Testers **¿Somos Quality Assurances o Quality Assistances?** Dentro de los videojuegos, del control de calidad, hay **dos grandes diferencias entre QAs**.

**Quality Assurance**: responsables de la calidad del producto, sobre quien recaería la responsabilidad si sale con mala calidad el producto. Cuando estemos a punto de lanzar el juego, se nos preguntará si el juego está libre de errores, si de verdad se puede lanzar Y nos harán firmar un documento explicando que, efectivamente, no quedan errores. Tenemos la potestad - si aún vemos errores - de parar la producción, demorar ese lanzamiento y revisar el juego. Mucho más responsables que en Quality Assistance (+ responsabilidad) .

**Quality Assistance**: más o menos somos asesores de calidad, vamos a indicarles lo que más o menos debería ser la calidad del producto, pero no somos los responsables directos de esta calidad. Normalmente, la potestad al final la tiene el productor, nosotros no podemos parar el lanzamiento; va a ser el productor o el jefe de estudio quien tenga esa última palabra (no podemos parar el lanzamiento, **- responsabilidad** ).

## Testing de Regresión y Testing Exploratorio

Veamos los dos tipos de testing: exploratorio y de regresión.

- De regresión: se ejecutan pruebas sobre una batería de pruebas, se envía el feedback y se reciben los cambios. Una **batería de pruebas** es un conjunto de pruebas o casos diseñado para testear la calidad del juego. Pueden ser generales o estar agrupadas por sectores.

- Exploratorio: no contamos con una batería de pruebas sino que exploramos el videojuego en busca de posibles errores. Se asume que somos expertos en el juego que estamos probando y por lo tanto podemos explorarlo con libertad.

## Tipos de Testing: Funcional, Lingüístico y de Localización

Hablemos ahora de tres tipos de testing: Testing Funcional, Testing de Localización y Testing Lingüístico

1. **Funcional**: se chequean todas las funcionalidades del juego. Es el tipo de testing más mecánico y el que abarca la mayor cantidad de detalles.
2. **De localización y lingüístico**: ambos se relacionan con el texto del juego.
- El de **localización** se refiere a las convenciones del país donde se comercializará el juego. No es necesario ser nativo del país objetivo para hacer este tipo de testing, la empresa nos entrega una lista para verificar estas características que cambian de acuerdo a la región.
- El **lingüístico** se enfoca en el texto en sí, es decir, que la redacción de los textos que aparecen en el juego sea correcta. Es necesario contar con un nivel avanzado o nativo del idioma que estamos chequeando, ya que este testing se encarga de revisar la redacción, ortografía y la distribución en los campos de texto.

## Tipos de Testing: Online, Compliance, Usabilidad y Play Test

Continuando con los tipos de testing, en esta clase veremos los siguientes tipos:

- **Online**: chequear todas la funcionalidades online del videojuego. Es decir, todas las funciones que requieran una conexión a internet.
- **Compliance**: Es un testing muy específico que se hace con los fabricantes de hardware. Consiste en garantizar que el juego cumple con los requisitos del fabricante de la consola.
- **Usabilidad**: chequear que los flujos del videojuego, como los menús y las opciones, tengan sentido en cuanto a la usabilidad. Es decir, que la interacción del usuario con el videojuego fluya de una manera natural.
- **Play test**: se hace al final del desarrollo. Se enfoca en garantizar que el videojuego sea adecuado y entretenido para el público objetivo para el que se diseñó. Para esto se busca un punto de vista ajeno al desarrollo del juego. A partir de estas pruebas se hacen cambios de última hora con base en el feedback del público objetivo que prueba el juego.

## Ejemplos de casos de pruebas de baterías de testing

### Testing Funcional:

- Comprueba que el juego se ha instalado correctamente y que arranca sin problemas.
- Comprueba que se puede terminar el juego sin ningún error notable (critical o blocker).
- Comprueba que se puede acceder a cualquier nivel y salir de él sin perder el progreso guardado por el jugador.
- Comprueba que no existen problemas de colisión (paredes falsas) en todos los niveles del juego.
- Desinstala el juego y vuélvelo a instalar. Comprueba que los datos de la partida guardada del jugador se pueden recuperar desde la nube sin problemas.

### Testing Online:

- Comprueba que el usuario puede conectarse y desconectarse de Facebook sin problema.
- Comprueba que el usuario puede enviar y recibir vidas y que éstas llegan y se pueden usar.
- Comprueba que el jugador puede entrar en el modo deathmatch y que puede emparejarse sin problema.
- Comprueba que el usuario puede enviar y recibir mensajes en tiempo real del resto de usuarios.
- Comprueba el nivel de lag que hay en una partida usando la referencia de ping 80, si el ping es superior, la prueba se dará por fallida.

### Testing Legal:

- Comprueba que los logos de las empresas desarrolladoras del juego se muestran correctamente.
- Comprueba que los logotipos de los equipos de fútbol y su equipamiento se muestran correctamente.
- Dirígete al apartado del EULA (End User License Agreement ) y comprueba que éste se muestra correctamente en el idioma que tiene el usuario activado por defecto.
- Comprueba que las banderas de los países que participan en el juego se muestran correctamente y sin errores.
- Comprueba que los nombres de los jugadores de los equipos aparecen correctamente y sin errores, además que sus fotografías corresponden a dichos jugadores.

### Testing Compliance:

- Realiza un guardado de la partida sobre un disco duro externo, y mientras esta se realiza, extráelo. Comprueba que el texto que se muestra es el correcto (“El dispositivo de almacenamiento fue extraído durante su uso. Error al guardar.) para Xbox360.
- Llena el dispositivo de almacenamiento interno y realiza un intento de salvado de partida, comprueba que el texto que se muestra es el correcto (“No hay suficiente espacio libre en el dispositivo de almacenamiento %1.”) donde “%1” es el nombre del dispositivo de almacenamiento. Para Xbox360.
- Al inicio del juego, comprueba que el texto de auto-guardado se muestra correctamente (“Este juego guarda datos automáticamente en determinados puntos. No extraigas el Memory Stick™ ni reinicies o apagues el sistema mientras el indicador de acceso al Memory Stick™ esté parpadeando.”) Para PSP.
- Realiza un guardado en cualquier punto del juego y comprueba que el texto asociado se muestra correctamente (“Comprobando el Memory Stick™. No extraigas el Memory Stick™ ni reinicies o apagues el sistema.”) Para PSP.
- Sobrecarga el ancho de banda de la conexión a internet e intenta conectarte con el servicio online de Nintendo, comprueba que el texto asociado se muestra correctamente (“La Conexión Wi-Fi de Nintendo tiene un volumen de tráfico demasiado elevado o el servicio se ha interrumpido. Vuelve a intentar conectarte más tarde. Para obtener ayuda, visita el sitio web: [www.nintendowifi.com](http://www.nintendowifi.com/ "www.nintendowifi.com")”) Para Nintendo Wii.

### Testing de Localización

- Comprueba que la fecha se muestra correctamente en el idioma inglés/americano (MM/DD/YYYY).
- Comprueba que los números cardinales se muestran correctamente en castellano/español (1º, 2º, 3º, 4º, 5º…).
- Comprueba que los números cardinales se muestran correctamente en Inglés (1st, 2nd, 3rd, 4th, 5th…).
- Selecciona el idioma americano/Inglés y comprueba que la separación de unidades de millar y coma decimal está escrita correctamente (12,500.50).
- Selecciona el idioma Castellano/Español y comprueba que la separación de unidades de millar y coma decimal está escrita correctamente (12.500,50).

### Testing Lingüístico

- Comprueba que el texto del tutorial está escrito correctamente, sin errores ortográficos y su contenido es coherente.
- Comprueba que el texto de los menús de ayuda está escrito correctamente, sin errores ortográficos y su contenido es coherente.
- Comprueba que el texto que puedes encontrar en la sección de descripción de los personajes está escrito correctamente, sin errores ortográficos y su contenido es coherente.
- Comprueba que el texto del menú online (incluyendo la sección in-game) está escrito correctamente, sin errores ortográficos y su contenido es coherente.
- Comprueba que el texto que aparece al finalizar una temporada completa está escrito correctamente, sin errores ortográficos y su contenido es coherente.

## Qué es un bug y la importancia del reporte de errores

Un bug es un error. El nombre viene de los computadores primitivos que funcionaban con tarjetas perforadas. En una de las pruebas con este equipo, Grace Hopper encontró una polilla, un insecto (*bug*) entre el computador que estaba produciendo un fallo en el sistema.

Reportar adecuadamente los errores es vital porque le permite al programador identificar rápidamente las inconsistencias en su código y facilitar una pronta corrección. Por esta razón, en este curso aprenderemos a redactar reportes de bugs de forma clara y directa mendiante un formato de reporte de errores (*bug writing format*) que está pensado para ahorrarle tiempo al programador.

## Bug Writing Format

Cuando vamos a reportar un error no podemos reportarlo de cualquier manera, esto sería caótico. En esta clase aprenderemos la manera correcta de hacer estos reportes.

Cada empresa tiene su propio formato para reportar bugs. Toma esta clase como una guía pero ten siempre presente que cada estudio o equipo de desarrollo tendrá su propio formato y debemos ser capaces de adaptarnos a ellos. El lenguaje en el que suelen hacerte estos reportes es inglés.

Un bug se compone de dos partes: el encabezado y el cuerpo del bug. El encabezado contiene información como el título del juego y su versión, la plataforma y su versión, el tipo de bug, el área del juego donde se encuentra y una breve descripción del mismo, de alrededor de 5 palabras.

En el cuerpo ampliaremos esta descripción que hemos dado brevemente en el encabezado; también contiene unos steps to reproduce, es decir el paso a paso para encontrarnos con este error; el nivel de prioridad del bug; un *repro rate*, que se refiere a la frecuencia con la que se observa el error; el actual result, es decir, lo que estamos observando que sucede como consecuencia del bug; *expected result* es lo que debería hacer el juego si no existiera el bug: y unos archivos adjuntos para soportar este reporte.

## Prioridades de los bugs y prioridad según su ruta

En esta clase aprenderemos a clasificar los bugs según su importancia y según su ruta.

Las prioridades de los bugs se dividen en:
**Minor**: errores muy pequeños. Por ejemplo pequeños desajustes en la gráfica o en el texto. Se deben reportar todos los errores por minúsculos que sean.
**Major**: errores estándar que no impiden el progreso del jugador en el juego.
**Critical**: errores extremadamente vistosos que impiden el progreso en el juego pero no impiden terminar el juego.
**Blocker**: errores graves que detienen el progreso del jugador en el juego.

La clasificación de los bugs según su ruta se refiere a los pasos que se deben seguir para llegar al punto donde encontramos el bug. En este sentido, la prioridad se divide en:
**Low**: se requieren muchos pasos muy específicos para reproducir el bug. Es poco probable que un jugador llegue a encontrarlo. Un bug de tipo blocker que se encuentre en una ruta low puede considerarse critical.
**High**: son muy pocos los pasos y las condiciones para llegar a este bug y por lo tanto es muy probable que un jugador se encuentre con él. Un bug de tipo major que se encuentre en una ruta high puede considerarse critical.

## Tipos de bugs: texto, gráfico, funcional, gameplay

Hablemos ahora de los tipos de bugs: funcional y de gameplay son similares en que se refieren a problemas en las funciones del juego.

Se diferencian en que los bugs de **gameplay** afectan funcionalidades que encontramos en el interior del juego, mientras que los **funcionales** se refieren a funcionalidades como menús, integración con redes sociales y otros aspectos que no afectan el interior del juego.

**Gráfico** y **de texto** se refieren, respectivamente, a problemas con las gráficas y los textos del juego.

## Tipos de bugs: Crash, Freezee, Framerate, Audio, Legal

En el proceso de testing podemos encontrarnos con los siguientes tipos de bugs:

- **Crash**: el juego se cierra y nos regresa a la pantalla de inicio.
- **Freezee**: el juego se congela y no responde. Generalmente corresponde a un bucle en el código.
- **Framerate**: el juego va a un framerate inferior al que debería funcionar. Visualmente se observa en forma de saltos en la imagen y movimientos poco fluidos.
- **Audio**: el sonido del juego no corresponde con lo que está ocurriendo en pantalla. Puede suceder con la música, las voces o los efectos de sonido.
- **Legal**: se refiere a problemas con el uso de marcas registradas y propiedad intelectual dentro del juego.