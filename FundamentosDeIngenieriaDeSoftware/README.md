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

### GPUs, tarjetas de video y sonido

Sabemos cómo los archivos se cargan en memoria, pero ¿cómo veo en pantalla que el archivo se ha abierto?

Esto se logra gracias a la Graphic Processing Unit o GPU.

![](https://static.platzi.com/media/user_upload/gpus-2c491d6e-e15d-4302-a54c-666fea3873aa.jpg)
La CPU puede ejecutar cualquier proceso, incluido el dibujado en pantalla de ciertos datos. Pero no es ella quien se encarga, sino la GPU: **tarjetas especialmente fabricadas para realizar estas tareas**.

La comunicación entre la CPU y la GPU se ejecuta actualmente a través de un socket llamado PCI-Express.

Estas placas de vídeo tienen sus propias unidades o núcleos de procesamiento y su propia memoria RAM.

Lo que sucede es que la GPU divide la pantalla en una matriz y cada núcleo se encarga de dibujar una parte de esa matriz, para lograr una mejor performance.

### Periféricos y sistemas de entrada de información

Nuestra computadora, por si sola, es capaz de muchas cosas. Pero, si le agregamos una serie de periféricos, es capaz de muchas más cosas.

**¿Cuáles son los periféricos de una computadora?**

Las impresoras, micrófonos, unidades de disco externo y en general todos aquellos dispositivos que se pueden quitar y poner de una computadora para aumentar sus funciones, son considerados como dispositivos periféricos.

Estos sistemas no deben tener acceso total a nuestra computadora, necesitan diferentes niveles de permisos. Estos permisos se organizan en niveles de anillos.

- **Primer anillo - Kernel:** El Kernel lo podemos entender como la capa más profunda de nuestro S.O. por lo tanto tiene acceso completo a archivos, drivers, programas, etc…Igual que cualquier otro proceso, se carga en la RAM como la cualidad de que es lo primero en cargar.En esta capa también viven programas capaces de encriptar y desencriptar información, de tal forma que ninguna otra capa del S.O. tenga acceso a ellos.
- **Segundo anillo - Drivers:** Como se ha dicho antes, los drivers son código que se encargan de interpretar las señales del hardware y establecer una comunicación con el software del PC. Estos primeros drivers pertenecen a piezas de hardware bastante importantes como la pantalla, el teclado, el mouse, etc.
- **Tercer anillo - Más Drivers:** Otra capa de drivers carga en un tercer “puesto” en la RAM. Dado que están más alejados del Kernel, tienen menos permisos y privilegios que los drivers del segundo anillo. Dado que mediante los drivers de este anillo, se comunican en su mayoría las Apps, es necesario que primero los drivers del tercer anillo pidan permisos a los del segundo anillo para luego así comunicarse con el hardware.
- **Cuarto anillo - Apps:** Finalmente, en la última capa del modelo de anillos del S.O. nos encontramos con las apps, que se cargan en la RAM para ejecutar procesos. Sin embargo, a diferencia de los otros anillos, no tienen ningún tipo de acceso directo al hardware del PC. Es relevante tener en cuenta que así debería ser, pues de lo contrario cualquier Software escrito por terceros tendría la capacidad de acceder casi por completo al PC y a sus piezas de Hardware.

![Anillos de un software](https://static.platzi.com/media/files/operating-rings_9e1109d1-3440-45c3-864d-6da3eb4bb7cc.png)

### La evolución de la arquitectura de la computación

La primera computadora mecánica la creó Charles Babbage en 1822, el primer motor de cálculo automático que además podía realizar algunas copias en papel -por lo cual, también era una especie de impresora-. Pero Babbage no consiguió la financiación necesaria para construir a gran escala esta computadora rudimentaria y su invento quedó en el olvido.

Sin embargo, podemos situar el origen de las computadoras en un sentido estricto en el año 1936, cuando Konrad Zuse inventó la Z1, la primera computadora programable. Aquí comienza la llamada primera generación, que abarca hasta el año 1946, teniendo propósitos básicamente militares.

**Cómo eran las primeras computadoras actuales**

En 1946 se construye la primera computadora con propósitos generales, llamada ENIAC (Integrador Numérico Electrónico e Informático). Pesaba 30 toneladas, por lo que básicamente no era parecida a lo que hoy conocemos como computadora, podía realizar una única tarea y consumía grandes cantidades de energía. Otra característica particular es que esta computadora no tenía sistema operativo.

![](https://static.platzi.com/media/user_upload/intelhistory_infograph_SPA-b1166af7-e363-460e-8df7-5dff325a2a5c.jpg)

**Cómo son las computadoras actuales**

Hoy en día tenemos computadoras en nuestros propios bolsillos y las cargamos a todos lados, tenemos laptops cuyos monitores se pueden desacoplar y funcionan como tablets, tenemos microchips que sirven como una computadora común y corriente.

Ese salto evolutivo en la computación ocurre gracias a la estandarización de la arquitectura de las computadoras: decidimos que un Byte son 8 bits, que la CPU es la encargada de procesar, que la GPU representa datos visualmente, que 1024 Bytes son un KiloByte, y que 1024 KB son 1 MB, que exista un puerto usual como el USB que nos permite conectar otros dispositivos externos.

Estandarizamos la transferencia de datos y los protocolos de comunicación. Hay un formato definido para cada tipo de imágenes, hay una forma de escribir HTML para que el navegador lo interprete y pueda mostrarnos elementos visuales en la pantalla. Definimos una manera para comprimir un archivo.

### Introducción a las redes y protocolos de Interne

Internet es una gran herramienta, pero no todo mundo conoce realmente como es qué nuestras computadoras están conectadas unas a otras y como es qué podemos comunicarnos e intercambiar información, a pesar de las distancias y los idiomas.

**Cómo nos conectamos a Internet.**

Tradicionalmente, pensamos que todas nuestras computadoras conectan a servidor en la nube y de ahí tenemos accesos a todos los servicios que nos ofrece internet. El procedimiento es más complejo, requiere de múltiples protocolos de transferencia y de hardware especializado que se encarga de transmitir los paquetes de datos involucrados en nuestra navegación. Veamos algunos de estos.

![](https://static.platzi.com/media/user_upload/freddy%20image-56f8e6b8-679b-410a-a675-e5e702e0e3f8-62b2f539-1f96-48bf-824a-2fbeaf9cd27a.jpg)

**Conceptos básicos**

**Ethernet:** cable de Red.
**Switch:** aparato que conecta varios dispositivos a una Red mediante el Cable Ethernet a una serie de puertos de conexión.
**Router:** aparato que interconecta varios dispositivos inalámbricamente, o por medio del Ethernet directamente al “computador” o a un “Switch”. Funciona como una red, se encarga de enrutar a c/paquete de datos dentro de una red informática.
**DHCP:** protocolo que asigna dinámicamente una IP y otros parámetros de configuración de Red a c/dispositivo en una, para que puedan comunicarse con otras redes IP.**IP **(Internet Protocol), dirección compuesta por una serie de n° que identifica a un computador.
**MAC Address:** Identificador único, que está grabado en el hardware del dispositivo.
**Modem:** aparato que convierte las señales digitales en analógicas y viceversa.
**ISP** (Internet Server Provider): Proveedor de Servicio de Internet.

**Conectividad: Tipos de Conexiones de Red Interna**

Hay muchas más formas de hacer una Red Interna de manera Local.

Por el Switch, que comparte y conecta múltiples dispositivos entre sí, generando una Red Local. ConectividadAl mandar datos de un dispositivo a otro, un algoritmo se encarga de saber de dónde es y enrutar el mensaje, internamente va preguntando el IP de c/dispositivo conectado, para saber a/por donde tiene que ir. Lo encuentra y envía la información, una vez recibido finaliza eliminado la conexión. Los algoritmos Internamente en las Redes, generan que datos, como la “verificación del receptor” (que viene en cabeceras especiales) este compartido, porque la señal rebota; siendo la forma para encontrar cualquier camino, es ineficiente pero funciona.

**Red Wi-Fi. **Con Users/Password, necesita del Router Wi-Fi, aparato con dos antenas y una serie de puertos conexiones (que funcionan como “Switch”, si no necesita uno; se puede conectar a uno, o a otros lados, para que sean parte de la Red), que da acceso a Internet. Emite una señal que permite conectar a varios Dispositivos. Su función consiste en enviar o enrutar paquetes de datos de una a otra Red, interconectando sub-redes (conjunto de dispositivos) que se pueden comunicar [no interviene un enrutador (Puente de Red o ‘Switch’)], estos pueden o no tener Wi-Fi.ConectividadAl ingresar Users/Password, internamente se crea un cable virtual entre el Router y el Dispositivo del que se está, y con otros dispositivos.

**Modem del ISP. **Hay varias formas para que se conecte el Modem al ISP, por medio de:

- ADSL: Cable de teléfono.
- Teléfono: Línea telefónica
- 4G/LTE: Antena de Radio.
- Fibra Óptica: Forma más óptima de conectarse.

### Puertos y protocolos de red

Los **routers** son las puertas de enlace a diferentes redes. El router asigna IPs dentro de la red local y esa IP es única en esa red, hacia afuera todos los equipos se conectan con la IP que te da el proveedor de internet que tienes contratado.

Para asignar IPs un software se encarga de revisar la MAC address de cada dispositivo y asignarle una IP que esté disponible.

En los esquemas de red se crea un **red virtual** dentro de los sistemas operativos con un concepto interno que se le conoce como los **puertos**.

![](https://static.platzi.com/media/user_upload/internet-9650ef5d-636f-4407-af41-9a395a7f3bc5.jpg)

**¿Qué son los puertos y los protocolos de Red?**

Un puerto es una puerta específica para un programa específico.

Cada solicitud que tú haces desde tu PC a través de una red trabaja con una ip y un puerto amigo, los puertos sirven para identificar los miles de servicios que maneja un SO, ejemplo: cuando tu entras a twitter desde tu navegador tú estás haciendo una petición a (102.102.20.02, ejemplo de ip de twitter), y el puerto 80, pero si quisieras subir un archivo por protocolo ftp sería 102.102.20.02 por puerto 21 que se ve reflejado como 102.102.20.02:21 y así sucesivamente cambia el puerto dependiendo del servicio. Los [protocolos de red](https://platzi.com/clases/2225-redes/35583-protocolos-de-red/ "protocolos de red") son como un lenguaje de comunicación entre máquinas y los puertos son autopistas donde los mensajes del protocolo pueden transitar.

### Qué es una dirección IP y el protocolo de Internet

**Una dirección IP** es el número único con el cual una computadora o un dispositivo se identifica cuando está conectada a una red con el **protocolo IP**. El protocolo IP (Internet Protocol) es una serie de reglas que se deben de seguir para que un dispositivo se pueda conectar a internet.

Las direcciones IP son números con los cuales cada computador del mundo se identifica tanto en Internet como las redes LAN. Las IPs están conformadas por 4 bytes lo que a su vez son 32 bits.

**Cómo funciona una dirección IP**

Recuerda que el router es quien asigna una IP a nuestra computadora, a través de DHCP, este debe tener una IP maestra que también lo identifique en la red. Gracias a esta IP es que se permite que todos los dispositivos puedan intercambiar información y persistir en la red.

Cada dirección IP está compuesta por 4 números separados por puntos y son una forma de comprender números más grandes y complejos. Las direcciones IP tienen una estructura que las convierten en privadas o públicas y que además hacen parte de la máscara de red y el getaway.

Las direcciones IP permiten que cada computador o dispositivo pueda conectarse al exterior, es decir a Internet, esto a través de tecnologías como NAT o Network Address Translation.

![](https://static.platzi.com/media/user_upload/ips_f6a4455a-6660-47e1-933e-a81ba037ef03-5987e6fd-a3e2-4f98-b319-1b69b099e3ba.jpg)

**Máscaras de red y dirección IP**

El protocolo DHCP es el protocolo encargado de asignarle la o las IPs a una determinada computadora. De igual manera, el protocolo puede usarse para una LAN o para internet. Este protocolo usa como base un concepto llamado máscaras de red.

Una máscara de red es una forma en la que se le indica a el protocolo DHCP cómo es que debe de asignar esas IPs. Las máscaras de red en pocas palabras, se basan en definir qué números o secciones de una IP pueden ser modificables y cuáles no. Esto lo hace por medio de 255s y 0s. Los 255, se refieren a qué esos números o secciones de la dirección IP no pueden cambiar. Y los 0 significan que sí pueden cambiar.

La decisión de cuáles son los números o secciones se pueden cambiar va a depender casi por completo de la persona u organización que crea la red./Entonces la máscara de red es 255.255.0.1, le indica al protocolo DHCP que en una IP las 2 primeras secciones no pueden cambiar, pero las otras 2 sí pueden.

Entonces usando esa máscara de red, se pueden hacer IPs como: 192.168.0.1. Es decir, 192.168 no pueden cambiar mientras que 0.1 si puede (192.168 es un número que casi siempre se usa al principio de una IP, ya que se ha vuelto un estándar y no es recomendable cambiarlo, al menos no en redes LAN.

**IP de internet y gateways**

Los gateways son en sí la dirección IP que tiene un router. Estas existen debido a que, al igual que las computadores comunes, los router también necesitan algo que los identifique en todo Internet. Esto se hace así una vez que un modem hace que nos conectemos a internet.

El sitio de internet cuya información queremos obtener, le manda la información solicitada al modem, luego el modem le pasa la información al router. Este mediante el protocolo NAT (Network Address Translator) traduce esa información recibida para que los dispositivos conectados al modem puedan también recibir la información y usarla. Esto es porque de otro modo, sin el protocolo NAT, las computadoras ordinarias por sí mismas jamás entenderían la información recibida de internet.

**Cuáles son los tipos de IP**

La manera más fácil de ver si una IP es clase A, B o C es utilizando la máscara de red estándar y estas son:

- **Clase A**. 255.0.0.0
- **Clase B** 255.255.0.0
- **Clase C** 255.255.255.0

También existe el método de máscara de subred variable, la cual es muy útil para segmentar una red IP con máscara de red estándar

Toda sección representada por un 0 en la máscara de red va a ser destinada a host o clientes y las que están representadas por 255 son las destinadas a la red.

### Cables submarinos, antenas y satélites en Internet

A menudo, imaginamos el acceso a Internet como una interacción con **satélites en órbita**. Sin embargo, esta creencia es inexacta, ya que los satélites son principalmente para áreas remotas. Internet en realidad se basa en cables que se extienden por todo el mundo, en lugar de satélites.

**¿Cómo nos conectamos a internet?**

Ya sea en casa con un módem o en movimiento con nuestro celular, nos conectamos a la infraestructura que nuestros proveedores mantienen. Esto nos conecta con diferentes puntos alrededor del mundo mediante cables submarinos, que varían en su composición entre fibra óptica y cobre.

Estos cables pueden comenzar en una ciudad como Nueva York y terminar en Japón y aunque no parezca, la red de Internet un poco frágil pues los cables pueden romperse por diferentes causas, como las anclas de los barcos.

![](https://static.platzi.com/media/user_upload/world-submarine-cable-map-652c0888-f4bf-4a53-8a1e-323febc6bfbb.jpg)

**¿Qué es la fibra óptica?**

La fibra óptica, un filamento dieléctrico, es el corazón de las transmisiones de alta velocidad.Estos cables son muy frágiles, y cuando alguno de ellos sufre alguna alteración se presentan problemas en el internet mundial. Pueden romperse debido a diversas causas, como anclas de barcos, lo que puede interrumpir nuestra conexión en cualquier momento.

![](https://static.platzi.com/media/user_upload/A86-285c8d27-0789-4ce5-88a8-259bc5ad20b8.jpg)

### Qué es un dominio, DNS o Domain Name System

Cuando queremos alojar a nuestro sitio en internet, necesitamos darle una “casa” y “una localización”. Así podemos decirles a los demas donde encontrarnos. Para esto necesitamos un dominio y este debe ser asignado a un DNS.

**¿Qué es un dominio?**

Los dominios son los nombres únicos de los sitios web. Se utilizan porque son más fáciles de recordar que la **IP**. Por ejemplo es más fácil recordar [http://misitio.com](http://misitio.com/ "http://misitio.com") a 190.02.4.123.

**¿Qué es un DNS?**

DNS significa Domine Name System. La manera de relacionar una IP con el dominio que le queremos asignar es mediante los DNS. Es decir, sirve como una base de datos. Nosotros podemos cambiar hacia donde apunta nuestro dominio, por ejemplo cuando queremos cambiar de proveedor de hosting.

### Cómo los ISP hacen Quality of Service o QoS

Cuando queremos conectarnos a puntos muy lejanos, como un servidor ubicado en el otro lado del mundo para nuestro proveedor de internet, puede resultar costoso. Los servidores manejan esta situación priorizando las conexiones mediante el QoS (Quality of Service), lo que significa que regulan la velocidad según el servicio al que deseas acceder.

Cuando el proveedor de internet establece una conexión más cercana, crea una red MAN (Metropolitan Area Network) con un costo casi nulo.Existe una forma de evadir las limitaciones de QoS y simular una conexión más cercana mediante el uso de un CDN (Content Delivery Network).

**¿Qué es un CDN?**

Un CDN es un servicio que almacena una copia del contenido estático, como imágenes o videos, que originalmente se encuentran en otro servidor. Es como tener una réplica más cercana de dicho material. Esto se traduce en una mejor velocidad de navegación y un costo menor.

Repasa: [¿Qué es un ISP?](https://platzi.com/clases/2053-intro-historia-internet/32965-isp/ "¿Qué es un ISP?")

### Cómo funciona la velocidad en internet

Nada puede molestarnos más al navegar en internet, que tener lag. Todos odiamos cuando las páginas tardan en cargar, o cuando no podemos ver un video de manera fluida. Pero, no siempre comprendemos por qué en unos lugares tenemos más velocidad que en otros y a qué se debe que a veces naveguemos más lento que otras veces

**¿Qué velocidad de internet tengo?**

La velocidad que nos ofrecen los proveedores de Internet, no es constante y muchos menos, es igual. Para entender nuestra velocidad de internet, es necesario conocer que esta depende de dos cosas. El ping y el ancho de banda.

**¿Qué es el ancho de banda?**

El ancho de banda es la capacidad máxima de información que se puede mandar. La mayoría de los ISPs (Internet Service Providers) nos venden ancho de banda en Mb. Es decir se mide en la cantidad de bits (no bytes) que transmite por segundo. Esta velocidad es variable y depende también del ping.

**¿Qué es el ping?**

La velocidad del internet se mide obteniendo el tiempo que le toma a la información viajar a través de un punto a otro en milisegundos. Por ejemplo entre tu computadora y el servidor. El ping tiene un límite y eso no va a cambiar nunca. Nuestra señal siempre va a viajar a una velocidad menor o igual a la velocidad de la luz, pero nunca podrá viajar más rápido. Puede ser un ping muy grande, cuando tenemos mucha distancia que recorrer.

Entonces, el ancho de banda es como el tamaño de un tubo de agua y el ping es la velocidad a la que puede viajar el agua.

![](https://static.platzi.com/media/files/bandwidth_df3967e6-f876-43da-b7af-f32d912ab1d1.png)

### Qué es el Modelo Cliente/Servidor

Tener un sitio web, es cada vez más complejo, tanto así que podemos tener a diferentes personas de nuestro equipo especializadas totalmente a cada una de estas partes. Por ejemplo, podemos tener un equipo dedicado al Frontend y otro al Backend. Sus funciones son complementarias y vinculantes, más no prescindibles.

**¿Cuál es la diferencia entre Frontend y Backend?**

El frontend es la parte que se “ve” de un sitio. Incluye el texto, los botones, imágenes, animaciones, etc. El backend son las acciones que se realizan cuando le damos clic al botón, son las conexiones a las bases de datos, y todo el código que hace que el sitio funcione.

**¿Qué es el modelo cliente-servidor?**

Se le llama modelo cliente-servidor a la relación que existe entre el frontend y el backend. El proceso de un modelo Cliente/Servidor es así:

- Cliente (Navegador que lee HTML, CSS y JS).
- Se envía una solicitud al Backend (Python, Go, Node, Java, etc.) a través de una URI.
- El Backend recibe la solicitud y toma decisiones en base a ella.
- El Backend consulta la Base de Datos (MySQL, Oracle, MongoDB, etc.) en caso de ser necesario.
- El Backend devuelve una respuesta que el navegador pueda leer, muchas veces datos en formato JSON.
- El Cliente recibe los datos JSON y los parsea para mostrarlos en HTML.

A un grupo de tecnologías se les conoce como Stack

![](https://static.platzi.com/media/public/uploads/image_fa321527-4b74-463d-8deb-36b5d6c4563f.png)

### Cómo funciona un sitio web

Llevamos ya varios años navegando en internet en páginas informativas, foros de discusión, redes sociales, etc. Pero ¿sabemos realmente como es que estos sitios funcionan? ¿Cómo es que se guardan y envían nuestros mensajes?

**¿Cómo es que navegamos en internet?**

Cuando escribimos una dirección web y damos clic, se ejecutan una serie de pasos, que no vemos, pero que son responsables de que lleguemos o no al sitio elegido:

- El navegador le hace una petición al sistema operativo para ver si tiene una versión en caché.
- GET le pide al servidor los datos y se los envía a la IP del servidor.
- El servidor responde con un número, como 200 (OK), 404 (No encontrado), 500 (error del servidor).
- Se buscan los archivos que ya tenemos en caché.
- Se empieza a desplegar el sitio web empezando por el texto.
- Por último se solicitan las imágenes, videos y otros assets del sitio.
- Las cookies son datos guardados en variables y van ambos lados, tanto en el servidor como en el navegador. Las cookies pesan, entonces es importante limitarlas para no afectar la velocidad de las peticiones.

![](https://static.platzi.com/media/files/http_f00292a5-0e1e-4582-a6c8-1da21dafcac1.png)

### Internet es más grande de lo que crees

Internet es más grande y más complejo de lo que llegamos a creer, pues existen muchos protocolos y formas de conectarnos. Hemos logrado evolucionar y revolucionar nuestras velocidades de conexión. Algunos dependen del tipo de emisión o recepción de los datos y otros de la velocidad, otros del propósito de esta información y lo cifrada que debe estar.

**TCP/IP vs UDP**

Son protocolos de transmisión de datos que se encuentran en la capa de transporte en el modelo OSI, la [diferencia más importante entre ellos](https://platzi.com/clases/2225-redes/35596-comparacion-entre-ambos-modelos/ "diferencia más importante entre ellos") es que el protocolo TCP/IP funciona con un protocolo de confirmación y el envío en orden de los datos, para asegurar la recepción de los datos.

Es decir que la transmisión es bidireccional, lo que lo hace relativamente más lento que el protocolo UDP. El protocolo UDP es un sistema rápido de transmisión de datos, pero no asegura la entrega de datos, ya que es unidireccional y no necesita conexión como el TCP.

**Tipos de Wifi**

- **802.11:** Este fue el primer estándar de transmisión inalámbrica creado en 1997, admitía un ancho de banda de 2 Mbps.
- **802.11b: **En 1999 el IEEE aumentaron el ancho de banda a 11 Mbps, pero usaba la frecuencia de señal de radio de 2.4 GHz(No regulada) y podían llegar a tener interferencias con otros dispositivos, se usaban más en hogares.
- **802.11a:**Se desarrolló a la vez que el 802.11b, tiene un ancho de banda de 54 Mbps y una frecuencia regulada de 5 GHz aprox. lo que la hacía más potente, pero cubría menos área, a más frecuencia menos área cubierta.
- **802.11g:** o Wireless G es un estandar creado en el año 2002 con un ancho de banda de 54 Mbps y una frecuencia de 2.4 GHz, compatible con estándar 802.1b.
- **802.11n:** También conocido como Wireless N y desarrolada en el año 2009, se usaron múltiples señales y antenas para lograr un ancho de banda de 300 Mbps y además es compatible con 802.11 b/g.
- **802.11AC:** Desarrollada en 2013, opera en frecuencia de 5 GHz con un alcance un 10% menor que sus antecesores pero con una velocidad de 1.3 Gbps, compatible con 802.11b/g/n.

Los tipos de cifrado de wifi son WEP, WPA, WPA2,WPA2-PSK(TKIP(Temporal Key Integrity Protocol), WPA2-PSK(AES(Advanced Encryption Standard)), WPA3, entre otros.

**Otros conceptos necesarios sobre Internet**

**TOR**: Es una red supersegura desarrollada por los militares y liberada al público, funciona con VPN y es sobre todo usada por periodistas y ciber-activistas.
**Firewalls**: protocolos de gestión de conexiones para reforzar la seguridad.
**Sockets:** Método para hacer conexiones persistentes, utilizada en chats, videojuegos por ejemplo.
**Tethering:** Tecnología para compartir internet desde un teléfono móvil actuando como router o modem.
**P2P:** Peer to Peer es una forma de conexión entre dos ordenadores conectados a un mismo IXP pudiendo compartir información sin necesidad de pasar por el IXP.
**Redes Mesh:** son redes diseñadas con dispositivos especiales que agilizan y hacen más seguras las conexiones, funcionan como repetidores inteligentes que se conectan a los dispositivos dependiendo cuál dé mejor rendimiento. El mayor de los pocos inconvenientes es el precio.

Repasa las redes [WAN, MAN y LAN ](https://platzi.com/clases/1277-redes-2017/11149-que-es-la-red-internet-lan-wan-y-topologias-de-red/ "WAN, MAN y LAN ")y [redes empresariales](https://platzi.com/clases/2225-redes/35599-ejemplo-de-una-red-empresarial/ "redes empresariales").

**Multi-WAN Round Robin:** La técnica multi-wan consiste en tener varios ISP conectados a una red para evitar, si hubiese un problema con alguno de los ISP, quedarse sin conexión durante ese periodo; el sistema multi-WAN Round-Robin consiste en emplear todos los recursos brindados por todos los ISP conectados y distribuirla de manera equitativa al todos los dispositivos conectados.

**IP fija vs. IP Dinámica:** La IP Fija se utiliza para aplicaciones como skype, videojuegos, VPN, entre otras, ya que son más estables y dan más velocidad de carga y descarga, pero son más inseguras, más caras, con disponibilidad limitada y se necesita tener conocimiento de informática, puesto que se tiene que configurar a mano.
La IP Dinámica es la más usada actualmente, porque su configuración suele ser establecida por parte del ISP, es relativamente más segura, no hay cargos económicos extra y son más eficientes, pero la conexión es más inestable.

**VPN:** Es una aplicación que repite tu localización a diferentes partes del mundo mediante una red virtual privada.

**TTL:** Time To Live, es el tiempo máximo que espera un paquete de datos para conectarse, hasta cancelarse.

**“Paquetes”**: Son paquetes de datos empleados por los ISP para facilitar la transmisión de los mismos.

**SYN/ACK:** son bits de control en el protocolo TCP para especificar el envío y recepción del mismo.

### Diferencias entre Windows, Linux, Mac, iOS y Android

Un Sistema Operativo es un programa o conjunto de programas que actúa como interfase entre el usuario o programador y la máquina física (el hardware) (a veces también citado mediante su forma abreviada OS en inglés) se encarga de crear el vínculo entre los recursos materiales, el usuario y las aplicaciones (procesador de texto, videojuegos, etcétera). Cuando un programa desea acceder a un recurso material, no necesita enviar información específica a los dispositivos periféricos; simplemente envía la información al sistema operativo, el cual la transmite a los periféricos correspondientes a través de su driver (controlador).

**¿Cuáles son los sistemas operativos más comunes?**

**Windows** es el sistema operativo de propósito general más usado a nivel mundial, es un sistema operativo cerrado y se encuentra en la gran mayoría de computadoras para consumidores, además utiliza un núcleo propietario perteneciente a Microsoft.

**Linux** es el sistema operativo más empleado en servidores, es libre y su creador Linus Torvalds aún sigue desarrollando su núcleo destacado por su alto rendimiento y alta seguridad, tienen una licencia del tipo GNU-GPL que no solo permite redistribuir sino también garantiza que las personas que redistribuyen el código deban aportar a la licencia entre otras cosas.

**FreeBSD** es el sistema operativo en el que está basado Mac OS .

![](https://static.platzi.com/media/files/os-types_18fb69f4-c8ae-4b54-9688-7201001c67be.png)

### Permisos, niveles de procesos y privilegios de ejecución

Los archivos digitales que se manejan tiene grados de importancia diferentes. Es decir, un documento de texto con un trabajo escolar, es mucho menos importante para el buen funcionamiento de nuestra computadora, que los archivos de sistema. Entonces, la seguridad de nuestra computadora, o hasta de nuestro server, depende que niveles de premiso tienen ciertos archivos y cuáles son los usuarios con los privilegios para modificarlos.

**Permisos, niveles de procesos y privilegios de ejecución**

En la administración de archivos la capacidad de utilizar permisos te permite definir entre las siguientes características, los permisos existen en todos los sistemas operativos de diversas formas y se crean con las siguientes opciones:

Read ®: permisos de lectura.
Write (w): permisos de escritura.
Execute (x): permisos de ejecución.

Una manera fácil de entender el sistema de permisos es el siguiente, tengamos en cuenta que usualmente vemos comandos parecidos al chmod 777, estos 3 numeros significan los 3 grupos de permisos de los cuales se hablan en el video, admin, team y public

**Cómo dar permisos de lectura y escritura**

La representación de estos números se toma en un sistema octal, teniendo en cuenta lo siguiente,

7 representa permisos de escritura, lectura y ejecución
6 representa lectura y escritura
5 representa lectura y ejecución
4 representa lectura
3 representa escritura y ejecución
2 representa escritura
1 representa ejecución
0 representa ningún permiso

Teniendo esto en cuenta, ahora cada vez que veas un comando chmod, recuerda que cada número representa el grupo de permiso y el número representa los permisos asignados que tiene. Un comando chmod 777 representan entonces, que los administradores, el team y los usuarios públicos puedes, escribir, leer y ejecutar archivos o lo que sea.!

![](https://static.platzi.com/media/user_upload/permissions_33dfc086-5fa0-42d1-8833-9455205f1e98-7a4e562e-2dd0-4a78-9e37-f28e009b59cf.jpg)

[ Domina la Administración de Usuarios y Permisos en Servidores Linux](https://platzi.com/blog/administracion-usuarios-servidores-linux/ " Domina la Administración de Usuarios y Permisos en Servidores Linux")

### Fundamentos de sistemas operativos móviles

A diferencia de los sistemas para escritorio, los sistemas operativos móviles tienen extrema seguridad en la forma en la que se instalan apps y en la que se accede a partes específicas de hardware. Los dispositivos móviles son diferentes completamente a una computadora normal, y es más fácil que un usuario inexperto caiga en situaciones vulnerables.

**Cómo funcionan los Sistemas Operativos móviles**

En Android existe la **Google Play Store**. También **Amazon Fire Store**. Para lanzar una app, previamente se debe enviar a los que permitan distribuirla. Se debe declarar que permisos se usarán (escritura en disco, GPS, Cámara). Este es uno de los motivos por los que debemos tener cuidado sobre que tipo de aplicaciones cargamos a nuestro Android.

En **iOS** solo existe la **App Store**.
Una serie de hackers rompieron la seguridad del sistema operativo para saltarse los anillos de privilegios y teníamos a Cydia, para poder instalar apps con Cydia había que hacer Jailbreak al iPhone.

**Permisos de acceso**

Se pide permiso por cosas como: GPS, cámara, acelerómetro, micrófono, contactos, galería, sistema de archivos.

**Android:** Permite acceder a la SD card, y al sistema de archivos linux (*nix).

**iOS:** Usa contenedores internos para las Apps llamado “SandBox”. Aísla a las apps para que no se pueda acceder a los archivos desde una app a otra.
Su sistema interno se basa en un API llamado “File Sharing API”. Cuando le damos a compartir archivo a una app, se crea un puente temporal entre ellas.
La forma en la que Apple espera que alguien guarde información desde a una app es iOS Cloud.

El único sistema de archivos que comparte tanto iOS como Android es la galería de fotos.

En ambos sistemas operativos podemos modificar los permisos, el tema es que muchas apps dependen de algunos de ellos.

![](https://static.platzi.com/media/user_upload/mobile-dev_317cde74-3955-4c7d-92ac-f4c1783ae8c6-afbfe651-ee27-46a5-85e4-63bd02e1798b.jpg)

**Cómo están desarrollados los SO de los moviles**

iOS y Android = Nativamente C++
Pocos usan C++ para desarrollar sus apps.

**Android** = Nativamente JAVA con Api Dalvik.

**iOS **= Historicamente era Objective-C. Pero ahora es SWIFT.
Objective-C era un lenguaje viejo basado en Next, de más bajo nivel pero superrápido.
Swift es más similar a Ruby o al nuevo JavaScript.

En ambos se puede con otros lenguajes como JavaScript c#, c++ gracias a API’s. Se puede desarrollar Apps multiplataforma en entornos para juegos como Unity y Unreal.

Como entornos de desarrollo específicos para apps se puede utilizar Android Studio y Xcode para iOS.

### Sistemas operativos embebidos e Internet of Things

Los sistemas embebidos son dispositivos que se encuentran en una gran variedad de lugares, estos son los sistemas de procesamiento que se utilizan en dispositivos diferentes a nuestros computadores, por ejemplo el microcontrolador que tiene programadas las secuencias de tu lavadora, el sistema embebido que tiene tu vehículo y que se encarga de coordinar tareas de seguridad básicas, el microcontrolador que tiene programadas las funciones de tu horno de microondas, el sistema de control de una estufa de inducción, la computadora embebida en un cajero automático, el sistema de navegación, estabilización y seguridad de un avión y muchos dispositivos más.

**¿Cómo se crean los sistemas embebidos?**

Hay plataformas para poder prototipar estos sistemas embebidos, las más populares son Arduino o Raspberry Pi, etc. Hay sistemas embebidos que no crearas que son un computador como la SIM Card (En la tarjeta SIM hay CPU, memoria RAM, un disco, un S.O., etc.)

**Arduino**

Los Arduino son muy populares porque sirven para prototipar muy rápido lo que quieras.
Las CPU ARM son un tipo de CPU especial. Las CPU normales como Intel o AMD son sistemas que se llaman X86. La diferencia está en la forma en que los transistores están por dentro organizados y en algunos elementos fundamentales. Por ejemplo Intel siempre ha optimizado los procesadores Intel y la arquitectura x86 para que estos sean ultra veloces, sin importar nada más. En cambio, ARM la velocidad es una prioridad, pero mucho más prioritario que esta es el uso de la electricidad, ARM usa la misma energía para procesar la misma cantidad de datos que un Intel, obviamente por ahora un Intel siempre va a hacer más veloz aunque ARM está alcanzando la velocidad de este.

**Raspberry Pi**

Los Raspberry pi tiene puertos básicamente de entrada eléctrica, por lo que se pueden mandar 2 tipos de señales, análogas o digitales.

Cuentan con una CPU ARM, pero tienen algo particular y es que esta CPU es quad-core, esto significa que son 4 CPU realmente. Los Raspberry Pi no se programan directamente como un Arduino, estos son un **PC COMPLETO** y eso es una diferencia radical en comparación con un Arduino. Los Raspberry pi tienen puertos (USB, HDMI, eléctrico, etc). Una Raspberry Pi tambien tiene una GPU que tiene un chip llamado Broadcom videocore que hace rénder de cualquier cosa gráfica que necesites.

Históricamente, los Raspberry corrían Linux, una versión optimizada para esta llamada Raspian; sin embargo, desde hace algunos años hay una versión especial de Windows llamada Windows ARM

**Microsoft**

Microsoft tiene una historia de S.O embebidos, por eso has visto en aeropuertos, cajeros, centros comerciales, fotos de la pantalla azul en lugares inesperados como el lugar donde se ven los vuelos, como pantallas de publicidad, etc. Esto es porque Windows creo una versión para sistemas embebidos llamada Windows CE y también tenemos otro llamado Windows Mobile o Pocket edición que fue evolucionando hasta crear el Windows Phone, etc. Al día de hoy solo existe un Windows desde la perspectiva de Microsoft (Windows 10) pero hay una versión especial llamada ARM edition que corre en sistemas embebidos como el Raspberry Pi.

**Chips en tarjetas**

La SIM Card es un SoC que tiene un S.O. Nacieron a partir de las Smart Cards. Probablemente, tienes una tarjeta de crédito que tiene un chip igual al que tiene una SIM Card, o las tarjetas para entrar al trabajo, internamente seguro, tienen uno de esos chips. Todos estos tienen una CPU, una memoria RAM, memoria de únicamente lectura donde está el S.O.

![](https://static.platzi.com/media/user_upload/iot_2cc925fd-f6f1-4069-a8a9-02d1d550ff72-c294769c-17d2-4aee-a40f-cd792d646484.jpg)

### Metadatos, cabeceras y extensiones de archivos

La mayoría de extensiones son de tres caracteres, porque en los inicios de la computación, solo era posible asignar 3 bytes para la extensión (tipo) de archivo, y 8 para el nombre. Sin embargo, todo avanzo y ya no estamos limitados a usar únicamente tres caracteres para denotar el tipo de archivo. Uno de los conocimientos básico que debemos tener, es el saber identificar las diferentes extensiones que pueden tener los archivos que más utilizamos y reconocer que tipos de programas pueden abrir dichos archivos.

**¿Cómo funcionan las extensiones de los archivos?**

Los **Sistemas Operativos** tienen en una base de datos guardada la correspondencia de cada extensión de archivo. Es decir, en la base de datos dirá algo como: jpg = imagen, mp3 = música, html = página web y así. En todo caso, para llegar a esta base de datos y encontrar a que corresponde, es necesario primero identificar que tipo de archivo es. Para ello el S.O. lee los primeros bits de cada archivo hasta que encuentra cierto patrón, esos bits de identificación se llaman la cabecera. Una vez identificado el patrón, el SO ya conoce que tipo de archivo es, con que programa se debe abrir y cómo se debe mostrar gracias a la base de datos.

Todo esto en la web funciona a través de un estándar denominado **MIME TYPES** (Extensión para emails multipropósito). Eventualmente, funcionaba solo con emails, pero hoy en día está en la mayoría de protocolos de comunicación. La forma en que funciona es sencilla, en la cabecera del protocolo de comunicación (independiente de si es http, ftp) se envían metadatos con la información del archivo. De tal forma que si es una página web, envía text/html, si es un video mp4 envía video/mp4 y así se logra la identificación de los archivos

![](https://static.platzi.com/media/files/filetypes_f84a50c0-2501-476d-b0e3-7d7194e2fb41.png)

### Cómo funciona el formato JPG

Los diferentes tipos de archivo de imagen, tienen diferentes pesos y diferente calidad, dependiendo de su formato. Asumamos que tenemos una foto de 600*800, si esto estuviera en un formato sin compresión como el formato .bmp pesaría 840KB (solo representando un color por pixel).

1 Bit representa como número máximo es el 256: Por lo tanto, en un .bmp tiene **256 colores** y para determinar su tamaño se debe multiplicar el ancho por el alto de la imagen.

Para representar colores de **16 bits**: Se necesitan 2 bytes.
Para representar colores (ultra-reales) de **32 bits**: Se necesitan 4 bytes.

**Nota**: Sí, cada bite tiene 8 bits, entonces; 16 bits es igual a 2 bytes y 32 bits es igual a 4 bytes.

Para tener una calidad de 32 bit la imagen debe pesar casi 1.9 MB, así que podemos comprimir las imágenes y usar formatos como jpeg.

JPEG es un algoritmo que lo que hace es identificar coordenadas para agrupar áreas de color. De esta manera minimiza la utilización de bytes y logra que la imagen tenga un peso mucho menor

![](https://static.platzi.com/media/files/jpgcompression_70f14183-f6ae-48c4-8fd4-b7ebc0395272.png)

### Videos, contenedores, codecs y protocolos

Los videos en internet serian muy pesados si nada más fueran secuencia de imágenes, sin comprimir. Para optimizar esto tenemos a los diferentes contenedores, codecs protocolos y keyframes. De esta manera hemos podido optimizar esta tecnología a los que ahora tenemos, sin necesitar grandes velocidades de descargas o archivos muy grandes.

**Contenedores**

Son los tipos de archivos donde se guarda el video, porque no es simplemente una secuencia de imágenes colocadas de manera consecutiva, los videos son la animación del movimiento, el sonido, los subtítulos, en dvd diferentes tracks de video, audio y subtítulos, más cosas internas, etc. Por eso se han creado múltiples contenedores como:
.avi,.MP4, .flv (flash video), .mpg, WebM (lo empuja mucho Google), etc.

**Codecs**

El codec es un algoritmo, es una pieza de código especial que comprime un video y lo descomprime. Históricamente, el codec que se usaba mucho antes era DivX. El más popular de nuestra época y el que más se encuentran es H.264.

**Protocolos**

Son la forma de transmitir videos. Uno muy obvio es HTTP, pero tenía el problema de que las descargas se interrumpían de manera muy fácil.

RTMP: Es una manera especial de transmitir video que te permite varias cosas, primero enviar y recibir, de tal manera que tu puedes ser el emisor del video.

**Keyframes**

Cada cierta cantidad de frames, existe un frame que vuelve a definir toda el área.

![](https://static.platzi.com/media/files/videocodecs_b338f03f-b6a5-4126-8367-5bab73a605ae.png)

### Cómo funciona .zip: Árboles binarios

El poder reducir el tamaño de los archivos que estamos compartiendo, es, sin lugar a dudas, una gran ayuda. Entender como funciona la comprensión de estos archivos podrían ser necesaria para poder usarlos, pero, sin duda, es fascinante.

**¿Cómo funciona la compresión de archivos?**

Los **árboles binarios** nos permiten comprimir sin perder información. En este caso, vamos a comprimir “amo leer panama papers”.

1. Debemos ver cuantas veces se repite cada letra

![](https://static.platzi.com/media/user_upload/Captura%20de%20Pantalla%202020-08-11%20a%20la%28s%29%208.57.22%20a.%20m.-7fb4d293-2b82-4574-8b83-6cf013f6dc73-6374a869-9d33-4cb9-b444-2ad4fea9f9b4.jpg)

2. La letra con más frecuencia va a estar en el primer punto de la rama. Cuando se encuentra es 1, y cuando no se encuentra es cero

![](https://static.platzi.com/media/user_upload/Captura%20de%20Pantalla%202020-08-11%20a%20la%28s%29%209.03.41%20a.%20m.-e3caf1ea-f981-402f-b70b-6513d57f3c8f-04e76ced-aa47-4343-b746-04e9bc31f230.jpg)

3. Con esto debemos volver a construir nuestro mensaje siguiendo el árbol, esto quedaría

![](https://static.platzi.com/media/user_upload/Captura%20de%20Pantalla%202020-08-11%20a%20la%28s%29%209.09.02%20a.%20m.-99f4b50f-618d-4d2f-bbf8-db653acd267b-28c905f8-55e4-4c98-96ba-cebbdc0499f6.jpg)

`1 00001 0000001 01 00000001 001 001 000001 01 0001 1 0000000001 1 00001 1 01 0001 1 0001 001 000001 000000001`

Aunque en este ejemplo no se reduce drásticamente el tamaño. Imagina párrafos más grandes u otro tipo de archivos.

### Qué es una red neuronal

![](https://static.platzi.com/media/public/uploads/flag_of_europe_f9fe7b5e-410b-4170-9eac-6b9d1b622ad7.png)

### ¿Qué es SQL y NoSQL?

![](https://static.platzi.com/media/user_upload/Diferencias-SLQ-NoSQL-6b2452bb-e44a-4dcf-b7c5-7797c907e244.jpg)

### Qué es un algoritmo

![](https://static.platzi.com/media/user_upload/Ciclos%20pokemon%20%283%29-10eb9b36-9d70-481b-bb93-fab16c351a51.jpg)

### El poder de un Raspberry Pi

Raspberry Pi Contiene un SoC, que controla todos los procesos de la tarjeta, un puerto hdmi, lector de tarjeta SD, GPIO - entradas y salidas de propósito general Creado con la intención de proveer un acercamiento con el software y hardware de manera fácil y barato, con un enfoque a la educación contiene todos los componentes para ser una computadora completa es del tamaño de una cartera servidor multimedia, servidor iot

### Principios de la ingeniería de software sostenible

¡Hola! Esta es una clase especial que forma parte del entrenamiento de la Escuela de Cloud Computing con Azure. Si es la primera vez que haces unos de estos laboratorios por favor realiza los siguientes pasos.

**Crea una cuenta en Microsoft Learn y regístrate a los laboratorios.**

Para ello solo tienes que seguir estos pasos:

1. Ingresa a la [página de Microsoft Learn](https://docs.microsoft.com/es-es/learn/ "página de Microsoft Learn").

![](https://i.imgur.com/i3zJvyM.png)

En parte superior derecha encontrarás un botón para iniciar sesión.

2. Inicia sesión en el portal. Lo podrás hacer con cualquier cuenta de Microsoft existente que tengas. En caso de que no tengas una da clic en el enlace para crear una.

![](https://i.imgur.com/OoxdXtx.png)

3. Una vez inicies sesión, te pedirá llenar unos datos extra para completar tu perfil dentro de la plataforma.

![](https://i.imgur.com/KsCvh6p.png)

4. Por último, regístrate al [desafío de la Escuela de Cloud Computing con Azure](https://docs.microsoft.com/es-mx/learn/challenges?id=2cceec57-42f9-4350-8b82-1fc4fe0034fa "desafío de la Escuela de Cloud Computing con Azure").

![](https://i.imgur.com/Aigzdiq.png)

5. ¡Listo! Ya tienes tu cuenta en Microsoft Learn vinculada a la Escuela de Azure de Platzi y puedes comenzar a realizar los laboratorios de práctica asociados a este curso.

**Ingeniería de Software Sostenible**

Si estás tomando la escuela de Cloud Computing con Azure te recomendamos hacer el siguiente módulo en Microsoft Learn donde podrás aprender los principios de la ingeniería de software sostenible.

La ingeniería de software sostenible es una disciplina emergente en la intersección de la climatología, el software, el hardware, los mercados de la electricidad y el diseño de centros de datos. Los principios de la ingeniería de software sostenible son un conjunto básico de competencias necesarias para definir, compilar y ejecutar aplicaciones de software sostenibles.

En este módulo, aprenderás a:

- Identificar los ocho principios de la ingeniería de software sostenible
- Entender las dos filosofías de la ingeniería de software sostenible
Para iniciar este módulo solo necesitas acceder al **[siguiente enlace](https://docs.microsoft.com/es-es/learn/modules/sustainable-software-engineering-overview/?ns-enrollment-type=Collection&ns-enrollment-id=xgg5bxjg1owzm7 "siguiente enlace")**.