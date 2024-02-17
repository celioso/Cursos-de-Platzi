# Fundamentos de Ingeniería de Software

### ¿Qué es un byte? ¿Qué es un bit?

El **byte** es una unidad de información en tecnología y telecomunicaciones compuesta por 8 bits.

Cada bit corresponde a un lugar en una tabla binaria y aunque no hay un símbolo universal para representar el byte, se usa “B” en países de habla inglesa y “o” en países de habla francesa. Además, se le conoce indistintamente como “octeto”.

**¿Para qué sirve un byte?**

Un byte se utiliza para medir la capacidad de almacenamiento de un dispositivo o la cantidad de información que se transmite por segundo en una red, lo que significa que puede representar 256 combinaciones diferentes de ceros y unos.

Si tienes un archivo de 1 megabyte, esto significa que estás almacenando aproximadamente 1 millón de bytes de información.

Además, la capacidad de almacenamiento de un dispositivo se mide en términos de bytes, por lo que es importante conocer la cantidad que necesitas para almacenar tus archivos y programas.

**Historia del byte**

Este concepto se adoptó en la década de 1950 como una unidad de almacenamiento de información en los primeros ordenadores.

Originalmente, un byte estaba compuesto por cualquier número de bits necesarios para representar un único carácter alfanumérico, como una letra o un número. Sin embargo, diez años después, en los años 60, se estandarizó en 8 bits por byte.

**¿Qué es un bit?**

El bit es la unidad mínima de información que se emplea en informática, este puede tener dos estados: uno o cero y comúnmente están asociados a que un dispositivo se encuentre apagado o encendido.

Proviene del funcionamiento del transistor, lo que quiere decir que tiene un impulso eléctrico o no lo tiene, es decir, la contraposición entre dos valores:

- 0 y 1
- Apagado y encendido
- Falso y verdadero
- Abierto y cerrado

Comprender el funcionamiento de los bits nos permitirá realizar conteos u operaciones matemáticas en un sistema que entienden las computadoras: [el sistema binario](https://platzi.com/clases/3221-pensamiento-logico/50671-que-es-el-sistema-binario/ "el sistema binario").

Este tipo de numeración nos permite codificar valores como números, letras o imágenes por series de unos y ceros. Estas después serán decodificadas para ser interpretadas en una forma más sencilla y menos abstracta.

También los unos y ceros se podrían agrupar en diferentes longitudes, pero el estándar de agrupación es de una longitud de ocho valores.

**¿Cuántos bits tiene un byte?**

Un byte tiene 8 bits y puede tomar valores entre cero y 255. Cada uno de estos, representa algún tipo específico de valor y para conocerlo, usamos la tabla ASCII.

Cada archivo de texto, cada imagen, cada canción que está en nuestra computadora tiene un peso en bytes, que depende de la cantidad de información que contiene.

**¿Cuál es el bit más significativo?**

El bit dentro de un byte con el valor más alto se le conoce como el bit más significativo o ***Most Significant Bit (MSB)*** por sus siglas en inglés. Este suele ser por convención el bit del extremo izquierdo, además, los bits dentro de un byte poseen diferentes valores que van incrementando de acuerdo a su posición.

**Tabla de equivalencia de bytes, Kilobytes, Megabytes y Gigabytes**

Medida | Equivalencia
------------- | -------------
1 byte | 8 bits
1 Kilobyte | 1024 Bytes
1 Megabyte | Son 1024 Kilobytes
1 Gigabyte | 1024 Megabytes
1 Terabyte | 1024 Gigabytes

**Valor total de un byte**
Para conocer el valor total de un byte solo se necesita hacer la suma de los bits que tiene activos (que están en uno) para determinar su valor. Si sumas todos los valores dentro de un byte te darás cuenta de que el valor máximo que puede tener es 255.

![bits-bytes](https://static.platzi.com/media/files/bits-bytes_72187e0e-0518-4aef-b9f3-997530c7e374.png "bits-bytes")

### Cómo funcionan los correos electrónicos

Conocer qué es la **Ingeniería de Software**, es un requisito básico para trabajar en el mundo de la tecnología. Debemos tener una idea muy clara de cómo funcionan procesos tan simples y cotidianos como lo es el enviar un correo electrónico, qué es un servidor y los protocolos utilizados,. Es cierto que no es un proceso simple, pero no es algo imposible de entender paso a paso.
Veamos este proceso más a detalle.
![](https://static.platzi.com/media/user_upload/insofware-1250c431-7bea-425c-a44b-fd9d40cd30f7.jpg)

a) Desde tu computador, en tu servidor de correo, estás redactando un email.

b) Cuando acabas de escribir, le das al botón “Enter”. Este botón manda un impulso eléctrico a tu tarjeta madre, y es procesado por el CPU (Central Processing Unit).

c) Mediante el Sistema Operativo (SO), la señal del CPU se identifica y reconoce. Así nuestro SO sabe lo que significa ese impulso creado por esa tecla especifica del teclado.

d) Como nos encontramos en un navegador web, nuestro sistema operativo le indica que ocurrió un evento, es decir, nuestro Enter.

e) Este evento lo que hace, es tomar todo lo que escribimos en nuestro correo y lo encapsula para enviarlo a un servidor, mediante protocolos ya establecidos.

f) Los servidores son computadoras, y ahí se reciben estos paquetes de datos. Mediante sus bases de datos, asignan este paquete de datos al remitente correspondiente

g) La persona que tiene su correo asociado a este servidor, recibe este paquete encapsulado, ya listo para leer en forma de correo electrónico.

Como vez, mandar un correo electrónico es más complejo de lo que parece, pero no por eso deja de ser trivial para lo que podemos hacer actualmente.

**Glosario básico necesario para entender como funciona el envío de un correo electrónico**

**ASCII**= American Standard Code for Information Interchange
**API**= Application Program Interface
**AJAX**= Asynchronous JavaScript and XML
**JSON**= JavaScript Object Notation
**REST**= Protocolo preestablecido de envio de datos= Representational State Transfer
**HTTPS**= protocolo de transferencia de envio de datos= Hypertext Transfer Protocol
**FTP**= File Transfer Protocol
**URL**= Unifor Resource Locator
**DNS**= Domain Name System
**IP**= Internet Protocol
**SMTP**= Simple Mail Transfer Protocol
**SoC**= [System on a Chip](https://platzi.com/clases/1098-ingenieria/6552-que-es-un-system-on-a-chip/ "System on a Chip")
**POP**= protocolo de oficina de correo

### Cómo funcionan los circuitos electrónicos

Los circuitos electrónicos son las bases de la tecnología moderna. La electricidad se crean en plantas de energía y se almacena en baterías o se transmite por cable hasta los enchufes de tu casa.

**¿Qué es la electricidad?**

La electricidad es un flujo constante de electrones y tiene ciertos parámetros que ayudan a describirla.

- El **Voltaje** representa el Voltio que es la fuerza que mueve la electricidad por un cable.

- El **Ohmio** representa la Resistencia que se opone a la electricidad.

- El **Amperio** representa la Intensidad de Corriente o cuanta corriente pasa por un circuito.

**¿Para qué sirve la electricidad?**

La electricidad se vuelve sonido cuando haces vibrar una membrana por medio de una onda eléctrica que representa un audio. Puede ser movimiento cuando hace trabajar un motor. Pero la forma más poderosa de usar electricidad es cuando la usamos para obtener información dentro de una computadora que está trabajando.

![](https://static.platzi.com/media/public/uploads/image_9c2c52eb-e5a1-41b5-990e-b2a43639993d.png)

### Procesadores y arquitecturas de CPU

Sin importar la marca o tamaño, los computadores tienen componentes principales similares. Es necesario conocer estos componentes y sus características para poder tomar una buena decisión sobre cuál comprar, basándonos en el uso que le vamos a dar.

**CPU - Central Processing Unit**: Procesador central (con marcas como Intel, AMD). Para conocer la capacidad de tu CPU te guías por los GHz (velocidad a la que procesan una instrucción) y Cores (# de CPU’s en un mismo chip, cuantas instrucciones pueden hacer al mismo tiempo).

**BIOS**: Chip especial que está instalado en la tarjeta madre, es un sistema operativo de arranque. Cuando arranca intenta detectar todas las cosas que están conectadas a un computador.

**Disco Duro**: Es dónde se guarda toda la información, es dónde está el sistema operativo.

Aprende más sobre: [las ventajas de system on a chip](https://platzi.com/clases/1098-ingenieria/6552-que-es-un-system-on-a-chip/ "las ventajas de system on a chip").

**RAM - Random Access Memory:** Es un tipo de intermediario entre el Sistema Operativo que está en el disco duro y la CPU. Es una memoria de alta velocidad (memoria volátil), solo funciona cuando hay electricidad.

**Memristor**: Pieza electrónica que logra guardar la onda eléctrica que pasa por ella incluso cuando se desconecta. Es posible que sea el reemplazo del disco duro y la memoria RAM en el futuro.

**Periféricos**: Pantalla, teclado, mouse, puertos, etc.

**Drivers**: Convierte la interacción de los accesorios periféricos en Bits y Bytes para que el computador pueda entender las instrucciones que les damos a través de ellos.

**GPU**: Canal de comunicación entre la pantalla y la CPU. Es quién se encarga de mostrar todo en la pantalla, desde que arranca hasta reproducir videos y videojuegos.

![](https://static.platzi.com/media/files/arq_9b73b2cf-5dd0-4ed3-b11b-20784ce43a0b.png)

### ¿Qué es un system on a chip?

Un **System on a Chip** es un circuito integrado que incluye dentro de sí todo un sistema electrónico o informático. Es, como su nombre lo indica, un sistema completo que funciona integrado en un solo chip. El término System on a Chip es conocido como SoC y significa en español ‘sistema en un chip’.

Los componentes que un SoC busca incorporar dentro de sí incluyen, por lo general, una unidad central de procesamiento (CPU), puertos de entrada y salida, memoria interna, así como bloques de entrada y salida analógica, entre otras cosas.

Dependiendo del tipo de sistema que se haya reducido al tamaño de un chip, puede ejecutar diversas funciones, como el procesamiento de señales, la comunicación inalámbrica o la inteligencia artificial, entre otras.

**Ventajas de un System on a Chip**

Estas son algunas ventajas que más destacan:

1. **Mayor densidad de funcionalidad:** los SoC permiten que todos los componentes electrónicos se coloquen en un solo chip, lo que reduce el tamaño y el peso del dispositivo.
2. **Menor costo:** la integración de los componentes en un solo chip reduce los costos de fabricación.
3. **Mejor rendimiento:** los SoC permiten un mejor rendimiento gracias al aumento de la velocidad de procesamiento y del ancho de banda.
4. **Mayor flexibilidad**: los SoC permiten la integración y el uso de diferentes tecnologías en un solo chip.

**Componentes de un System on a Chip (SoC)
Un System On a Chip se compone de:**

- **BIOS:** arranca nuestro sistema.
- **RAM**: se guardan rápidamente todos los datos a los que queremos acceder.
- **CPU**: se encarga de procesar todo por dentro.
- **Chip de radio**: controla todas las señales wifi, bluetooth o señales de celular (3G, 4G…)
- **GPU**: se encarga de hacer toda la representación gráfica de todo en pantalla para nuestro sistema.
- **Periféricos**: un sistema de periféricos va actuar como el intermedio del sistema operativo, los drivers y el hardware de tal manera que un smartphone pueda ser expandido.
- **Pantalla:** es otro gran periférico, la pantalla representa todo lo que hacemos dentro de un sistema embebido (no todos los sistemas embebidos tienen una pantalla).
- **Batería**: la batería de nuestro celular no está dentro del System on a chip, la batería tiene su propio controlador eléctrico que es una pequeña CPU que se encarga de manipular como nosotros internamente estamos obtenido la electricidad.

**Ejemplo de System on a Chip**

En esta imagen, podemos ver a una Raspberry Pi y todos sus componentes:

![](https://static.platzi.com/media/user_upload/raspberry_pi_b--c84316d7-9bb1-4e0d-93a7-19a09158d581.jpg)

- **System on a Chip (SoC):** un circuito integrado que incorpora muchos componentes de la computadora en un solo chip: la CPU, la memoria y la memoria RAM.
- **Conector de pantalla DSI**: se usa para conectar un panel LCD. En el otro lado de la placa hay una ranura para tarjetas microSD que contiene el sistema operativo Disco Duro.
- **Pines GPIO (entrada / salida de propósito general)**: Pasadores utilizados para conectar dispositivos electrónicos. Bus de Datos 1 Byte cada uno.
- **Puerto HDMI:** Se usa para conectarse a un monitor o TV. HDMI puede transportar sonido e imagen.
- **Puerto Ethernet:** Un puerto Ethernet estándar de 10/100 Mbit/s que se utiliza para conectar su dispositivo con el resto de la red.
- **Puertos USB**: Puertos USB 2.0 estándar que se emplean para conectar periféricos, como el teclado y el mouse.
- **Puerto de audio:** Un conector de 3.5 mm usado para conectar los altavoces.
- **Conector de alimentación micro-USB**: se emplea para alimentar la energía de la Raspberry Pi.
- **Chip de interfaz USB y Ethernet:** Se emplean para transferir información.
- **Conector de cámara**: Permite la captura de fotografías y videos.

### Diferencia entre memoria RAM y disco duro

La importancia de la **memoria RAM y el disco duro** radica en que son elementos donde se guarda información y datos de un dispositivo. La memoria RAM se diferencia del disco duro porque no guarda los datos de manera persistente, mientras que los discos duros sí lo hacen.

Este tipo de memoria funciona a alta velocidad porque puede acceder a cualquier lugar donde se guardan los datos de manera instantánea. En cambio, los discos duros son lentos porque deben llegar al punto exacto donde están alojados los archivos para poder abrirlos.

**¿Qué es la memoria RAM?**

La memoria RAM es una parte importante de dispositivos como computadoras y celulares porque es donde se almacenan los datos de las aplicaciones que estás usando y necesitarás más de ella si estás ejecutando muchos programas.

![](https://static.platzi.com/media/user_upload/memoriaram%20%281%29-5c206cf5-762f-4920-95c6-7f7b6171f211.jpg)

Este elemento almacena información de manera temporal, es decir, esos datos se borrarán cuando apagues el dispositivo, por lo cual si la información excede la capacidad de la memoria RAM, la CPU empezará a guardarla en el almacenamiento de tu computador ralentizado el desempeño del dispositivo.

Repasa y aprende más sobre: [¿Qué es un system on a chip?](https://platzi.com/clases/1098-ingenieria/6552-que-es-un-system-on-a-chip/ "¿Qué es un system on a chip?")

**¿Cómo funciona la memoria RAM?**

La memoria RAM guarda temporalmente los programas y archivos en uso en la computadora. La CPU accede a ella mediante un índice compartido y un bus de datos que transfiere datos entre la CPU, el disco duro y la memoria principal.

**¿Qué tipos de memoria RAM existen?**

Si tienes una computadora encontrarás que tiene alguno de estos dos tipos de memorias RAM: la Memoria RAM estática o la Memoria RAM Dinámica.

1. **Memoria RAM estática**
La estática es un tipo de memoria conocida como SRAM que preserva los datos siempre que tenga suficiente energía en su sistema. No necesita actualizarse constantemente para retener la información, por lo que es más rápida. Sin embargo, su costo es elevado y no suele ser la primera opción para un dispositivo.

2. **Memoria RAM dinámica**
En cambio, la memoria RAM dinámica, también conocida como DRAM, suele ser la memoria principal de los sistemas informáticos porque es más asequible. Está compuesta por un capacitor y un transistor, por lo que su sistema está diseñado para actualizarse constantemente con el fin de retener información.

**¿Qué es un disco duro?**
Un **disco duro**  o *hard drive* es una pieza de hardware que almacena datos en un disco de manera permanente. El usuario puede acceder a estos datos para leer y escribir archivos.

**¿Cómo funciona el disco duro?**

Antes, los discos duros tenían un brazo mecánico que leía y escribía datos en un disco de metal que giraba, como un disco de vinilo. El brazo se movía para acceder a diferentes partes del disco.

Hoy en día, existen los discos de estado sólido (SSD) que no tienen brazo ni disco que gira. Funcionan como la memoria RAM y evitan la pérdida de información al apagar la computadora. Son más rápidos para leer y escribir y duran más porque no tienen piezas que se muevan y se puedan romper.

![](https://static.platzi.com/media/user_upload/inside-SSD-1024x683-4891497e-fa39-401c-a66e-4b0fb401981f.jpg)

**¿Cuál es la diferencia entre el disco duro y la memoria RAM?**

Los discos duros son lentos porque requieren posicionarse en el lugar exacto donde se encuentra el archivo. En cambio, la RAM es más rápida porque puede acceder instantáneamente a los datos almacenados.

La diferencia radica en que los discos duros no son volátiles y retienen la información aunque no tengan energía. La memoria RAM, por otro lado, pierde los datos al apagar el computador. Los discos duros almacenan archivos de manera secuencial, dividiéndolos en pedazos y guardando su posición y ubicación en el disco para accederlos ordenadamente.

**¿Para qué sirven los sistemas de archivos de un disco?**

Para poder almacenar los archivos de forma adecuada, un disco duro necesita un sistema de archivos que son convenciones internas de los sistemas operativos para poder acceder a los archivos almacenados.

- En Linux existe ext3 o ext4.
- En Windows existía FAT16 o FAT32 (File Allocation Table), que fue reemplazado por NTFS (New Technology File System).
- En Mac OSX el sistema de archivos se llamaba HFS (Hierarchical File System) pero ahora se llama AFS (Apple File System) en macOS Sierra.
Cuando abrimos un archivo, la CPU (Unidad Central de Procesamiento) se lo pide al disco duro y luego lo lleva a la memoria de acceso aleatorio, o random access memory, para leerlo.

![https://static.platzi.com/media/files/archivos_624fac99-b611-4a4a-a578-914206c29626.png](https://static.platzi.com/media/files/archivos_624fac99-b611-4a4a-a578-914206c29626.png "https://static.platzi.com/media/files/archivos_624fac99-b611-4a4a-a578-914206c29626.png")

