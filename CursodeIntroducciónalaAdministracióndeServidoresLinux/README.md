# Curso de Introducción a la Administración de Servidores Linux

### Habilidades y Roles de un Administrador
Habilidades clave:
- Control de accesos.
- Monitoreo del sistema.
- Administración de recursos.
- Troubleshooting.
- Instalación y mantenimiento de software.
- Creación de respaldos.
- Documentación.
**
Roles que puedes desempeñar:**

- **DevOps Engineer**:
 - Se enfocan en los procesos y metodologías para la correcta liberación en el proceso de desarrollo de software.
- **Site Reliability Engineer**:
 - Se enfocan en que los sistemas de software operen de manera correcta y con el mayor grado de confiabilidad posible.
- **Security Operations Engineer:**
 - Encargados de mantener la seguridad de los sistemas a nivel de red y aplicaciones.
- Algunos otros roles:
 - **Network Engineer**.
 - **Database Administrator.**
 - **Network Operation Center Engineer.**
 - **MLOps Engineer.**
 - Cloud Engineer.

### ¿Qué son los servidores?

**simple definition:** Un servidor es como una gran biblioteca que guarda información en internet. Puedes imaginar que cada página web es como un libro y el servidor es el lugar donde se guardan todos los libros. Cuando quieres ver una página web, tu computadora le pide al servidor que le muestre el libro (página web) que está buscando. Y el servidor le envía el libro (página web) a tu computadora para que puedas leerlo. De esta manera, todos podemos acceder a información en internet gracias a los servidores.

 ### ¿Qué es un sistema Linux/UNIX?

Linux y UNIX son sistemas operativos de tipo Unix-like (similares a Unix) que comparten una arquitectura y una filosofía común basada en la idea de que "todo es un archivo" (en inglés, "everything is a file"), lo que significa que los dispositivos, servicios y recursos del sistema se presentan y manipulan a través del sistema de archivos.

### Arquitectura de un sistema UNIX/Linux

![](https://static.platzi.com/media/user_upload/Arquitectura%20linux-c07d90e8-4d0c-4932-90c4-47b23f6a50b7.jpg)

**Hardware: **son las partes físicas del computador que nos permiten darle instrucciones.
**Kernel:** Es el nucleo del sistema operativo que permite la comunicación entre el hardware y software.
**Shell:** Es la capa intermedia entre el sistema operativo y las aplicaciones. Nos podemos comunicar con ella atraves de la terminal.
**Utilities:** Son las aplicaciones presentes en nuestra computadora con interfaz visual.

Contestando la pregunta del profe Devars, creo que es el Kernel la parte más importante ya que debe gestionar todos los procesos de las demas capas de la arquitectura UNIX/Linux.

### Breve historia del software libre y el Open Source

Característica | Software Libre | Open Source
----|----|-----
Definición | El software otorga al usuario final la libertad de usar, estudiar, copiar, modificar y distribuir el software y sus fuentes | El software cuyo código fuente está disponible públicamente y puede ser examinado, modificado y distribuido por cualquier persona
Filosofía | Se basa en el valor de la libertad del usuario y la creencia de que el conocimiento y la información deben ser libres | Se centra en la colaboración y la revisión del código fuente para mejorar y hacer avanzar el software
Licencias | Utiliza licencias GPL, LGPL, BSD, MIT, Apache, entre otras | Utiliza licencias como la Apache, BSD, MIT, entre otras
Desarrollo | El desarrollo se realiza a través de una comunidad colaborativa, generalmente no hay una empresa que controle el software | El desarrollo puede ser realizado tanto por una comunidad como por una empresa privada
Beneficios | Proporciona libertad y flexibilidad a los usuarios finales | Fomenta la innovación y el avance del software al permitir que un gran número de personas colaboren y contribuyan al proyecto
Ejemplos de software | Linux, GNU, Firefox, Apache, LibreOffice, etc. | MySQL, Perl, Python, etc.

![](https://static.platzi.com/media/user_upload/03483589-35105ae1-1941-4ab0-97d7-1871010f7c85.jpg)

### Sistemas operativos y distribuciones

Una lista de algunas de las distros mas usadas de Linux organizadas por dificultad de uso, esta lista no esta escrita en piedra y puede variar un poco para cada uno según su experiencia de uso pero sirve bastante para tener un punto de partida a los que recién empiezan en Linux. 

![](https://static.platzi.com/media/user_upload/Linux%20Distros-171b7d64-67ae-4129-863e-cb8406ff4882.jpg)

Hay quienes prefieren empezar en Hard Mode desde el principio y otros prefieren ir avanzando de dificultad poco a poco. Eso ya depende de cada uno.

Una distribución de Linux es una variante de Linux creada por una organización o comunidad específica que toma el kernel de Linux y lo combina con otras aplicaciones y herramientas de software para crear un sistema operativo completo. Las distribuciones de Linux pueden variar en características, propósitos y enfoques.

Hay muchas distribuciones de Linux disponibles, y algunas de las más populares son:

**Rolling Release:**

*Arch Linux, Gentoo, Solus, Manjaro,  openSUSE Tumbleweed*

**Fixed Release:**

*Debian, Ubuntu, CentOS, Fedora, Red Hat Enterprise Linux (RHEL)*

Las distribuciones de Rolling Release reciben actualizaciones continuas y no tienen versiones específicas. Las actualizaciones se entregan a medida que se lanzan y se prueban. Las distribuciones de Fixed Release, por otro lado, tienen una versión específica que se lanza en un momento determinado y reciben actualizaciones de seguridad y mantenimiento regulares, pero no se actualizan con nuevas características de forma regular.

Cada distribución de Linux tiene su propia comunidad de usuarios y su propia filosofía y enfoque. La elección de una distribución de Linux depende de las necesidades y preferencias individuales del usuario, como el propósito de la distribución, el nivel de experiencia técnica del usuario y las características específicas que se buscan.

### ¿Dónde viven nuestros servidores?

Todo el hardware y software del servidor es alojado y mantenido por la organización. Cloud

**Publica:** son todos los proveedores de servicios que otorgan recursos de hardware y software para montar los servidores, tales como Google cloud, Azure, AWS, entre otros.

**Privada:** todos los recursos y software pueden vivir en otro lado pero ningun recurso o servicio se comparte con otra empresa para tener un mayor grado de seguridad. Hybrid (Híbrida) Es una combinación de servicios on premise y cloud.

Los servidores pueden estar clasificados según el lugar donde están alojados en diferentes categorías, tales como: _ 
**Servidores en las instalaciones (On-premises):** Son aquellos servidores que se ubican físicamente en la misma ubicación que la empresa o la organización que los posee. Estos servidores son administrados directamente por el personal interno de la empresa, lo que les da un mayor control y seguridad, pero también requieren una inversión significativa en hardware y mantenimiento.
**Servidores dedicados:** Son servidores físicos que se alquilan por completo a una empresa o una organización y que son administrados por un proveedor de servicios de hosting. Estos servidores se pueden ubicar en los centros de datos del proveedor de servicios o en los locales del cliente, y pueden ser administrados de forma remota por el personal de la empresa. Los servidores dedicados ofrecen un alto nivel de personalización y control, pero pueden ser costosos. 
**Servidores virtuales privados (VPS):** Son servidores virtuales que se crean mediante la partición de un servidor físico en varias máquinas virtuales, cada una de las cuales funciona como un servidor independiente. Estos servidores se pueden alojar en centros de datos de terceros o en la nube, y son administrados por el proveedor de servicios de hosting. Los servidores VPS ofrecen una solución más económica que los servidores dedicados, pero aún ofrecen un alto nivel de control y personalización.
** Servidores en la nube**: Son servidores que se alojan en la infraestructura de la nube de un proveedor de servicios de hosting, como Amazon Web Services o Microsoft Azure. Estos servidores son escalables y se pueden adaptar fácilmente a las necesidades cambiantes de la empresa. Los servidores en la nube ofrecen un alto nivel de flexibilidad y reducen la necesidad de inversión en hardware, pero pueden tener limitaciones en cuanto a personalización y control.  En resumen, los servidores se pueden clasificar según el lugar donde están alojados en instalaciones propias, servidores dedicados, servidores virtuales privados y servidores en la nube, cada una con sus propias ventajas y desventajas. La elección dependerá de las necesidades específicas de la empresa u organización en cuestión.

### Formas de montar un servidor

Hay varias formas de montar un servidor Linux, dependiendo de las necesidades y recursos de cada organización. Algunas de las formas comunes de montar un servidor Linux son las siguientes:

1. Servidores físicos: Consiste en instalar Linux en un servidor físico en las instalaciones de la organización. Este enfoque puede ser más adecuado para organizaciones que tienen un alto nivel de control sobre el hardware y la seguridad del servidor.

2. Servidores virtuales: Consiste en instalar Linux en una máquina virtual que se ejecuta en un servidor físico. Este enfoque puede ser más adecuado para organizaciones que necesitan flexibilidad y escalabilidad, pero que no tienen los recursos para adquirir y administrar un servidor físico.

3. Servidores en la nube: Consiste en instalar Linux en un servidor virtual alojado en la nube de un proveedor de servicios en la nube. Este enfoque puede ser más adecuado para organizaciones que desean acceso remoto, escalabilidad y flexibilidad sin tener que administrar su propio hardware.

4. Contenedores: Consiste en utilizar tecnología de contenedores para alojar aplicaciones en Linux. Los contenedores pueden ser más eficientes que las máquinas virtuales porque comparten recursos de hardware, lo que significa que pueden alojar más aplicaciones en un solo servidor.

5. Kubernetes: Consiste en utilizar una plataforma de orquestación de contenedores como Kubernetes para gestionar y escalar contenedores en un clúster de servidores Linux.

Cada uno de estos enfoques tiene sus propias ventajas y desventajas, y la elección depende de las necesidades específicas de la organización, como la escalabilidad, la flexibilidad y el control sobre el hardware y la seguridad del servidor.

### Instalación de VirtualBox

Si tú, como yo, eres usuario **Ubuntu** te dejaré una pequeña guía de cómo instalarlo.
**(Si utilizas otra distribución, puedes expandir este aporte con tu distribución y cómo instalar virtualbox).**

1. Primer paso, debes dirigirte a la página oficial de [VirtualBox](https://www.virtualbox.org/ "VirtualBox").

![](https://static.platzi.com/media/user_upload/Captura%20desde%202023-02-23%2016-01-12-78cfe8da-fdd0-4fb8-8fdd-06f25ae11ab4.jpg)

2. Debes darle click al botón gigante que te aparece para instalar VirtualBox y te redirige a otra página.

![](https://static.platzi.com/media/user_upload/Captura%20desde%202023-02-23%2016-03-33-86cc5f69-7548-468f-a874-3519879db169.jpg)

**Nota: es muy importante tener en cuenta que si tu país no cumple con los requisitos políticos para adquirir servicios de Oracle, te recomiendo utilizar una VPN para poder descargar este programa. (🇻🇪🇨🇺🇳🇮)**

3. Una vez estándo en la lista, debes elegir la opción que dice: **Linux distributions**.

![](https://static.platzi.com/media/user_upload/Captura%20desde%202023-02-23%2016-05-13-f6bac4e5-1d3e-49a8-aa03-27034f265747.jpg)

4. Ahí podemos observar una lista de distribuciones y debemos elegir la distribución que tengamos; en mi caso es Ubuntu y su versión es la 22.04.

![](https://static.platzi.com/media/user_upload/Captura%20desde%202023-02-23%2016-11-59-0767764d-a368-4567-a22d-561681e97272.jpg)

5. Debemos esperar a que el archivo se descargue.

6. Una vez descargado el archivo, en este caso es un archivo "deb". Esto quiere decirnos que el gestor de paquetes que utilizamos es el mismo que el de Debian (deb) entonces podemos hacer la instalación gráfica con click e instalar o por la terminal. Tú decides cuál crees que te sea más fácil.

![](https://static.platzi.com/media/user_upload/Captura%20desde%202023-02-23%2016-15-00-37e36c99-9af0-43c3-a5b5-3d124bdaafbc.jpg)

7. Para instalarla de forma gráfica, solamente debes pulsar **click derecho > abrir con otra aplicación > Instalar software** Y te abrirá una forma gráfica de instalar.

![](https://static.platzi.com/media/user_upload/Captura%20desde%202023-02-23%2016-17-45-ad076fff-7353-4763-a1e5-41c2aa9dabb0.jpg)

7.1. O también la puedes instalar por la terminal con este comando  `sudo dpkg -i  virtualbox-7.0` Por supuesto el nombre del paquete debe de ser idéntico a como está descargado.

### Instalación de Ubuntu Server

![](https://static.platzi.com/media/user_upload/Ubuntu-e9e759ce-9033-42d3-906e-5e7951385634.jpg)

Quiero hacer dos aportes:
- LTS Long Term Support
- Filosofá Ubuntu Con todo ese ambiente GNU, en mis primeros pasos con linux un amigo me hablo de que Ubuntu proveneia del Zúlu:

### Instalación de RHEL 8

[VirtualBOx](https://www.virtualbox.org/wiki/Download_Old_Builds_6_1 "VirtualBOx")

[https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8)

### Configuración básica para un servidor remoto

Comandos: **ssh**: lo usamos para confirmar que tengamos instalado openssh en el sistema, en caso de no estar instalado podemos instalarlo con el comando:

`sudo apt install openssh`

Para el caso de ubuntu server, o para el caso de RHEL con el comando:

`sudo dnf install openssh`

**systemctl status sshd**: Para verificar que el proceso de ssh este activo y corriendo en el sistema (si no les funcione agréguenle la palabra **sudo** al principio del comando para abrirlo con permisos de superusuario)

**ip address:** nos da datos sobre nuestros dispositivos de red, incluyendo la IP interna del servidor.

Ahora, para conectarse al servidor desde un dispositivo en la misma red, se puede usar el comando:

`ssh username@localip`

Desde la PowerShell de Windows o la consola del sistema operativo que estés usando.

**EXTRA**

En caso de querer acceder remotamente a un servidor, el comando es el mismo, solo que ahora en vez de usar la ip local se usaría la IP pública, la cual se puede ver desde un navegador en internet buscando myip estando conectado a la misma red del servidor o desde el servidor usando algún comando como lo puede ser el siguiente:

`curl ifconfig.me`

Es importante tener en cuenta que para poder tener este acceso, se debe tener abierto a la red el puerto de internet número 22, adicionalmente es una buena práctica utilizar un firewall para que solo ciertas IPs puedan conectarse al servidor y así evitar accesos no autorizados.

### ¿Qué son los sistemas de archivos?

En Linux, un sistema de archivos es la estructura que se utiliza para organizar y almacenar datos en dispositivos de almacenamiento, como discos duros, unidades flash USB, CD-ROM y otros dispositivos de almacenamiento.

El sistema de archivos determina cómo se organizan los datos en un dispositivo de almacenamiento y cómo se accede a ellos. En un sistema de archivos típico, los datos se organizan en archivos y directorios (también llamados carpetas), que se pueden acceder y manipular mediante comandos en la línea de comandos o mediante una interfaz gráfica de usuario.

Linux admite varios sistemas de archivos, incluidos los siguientes:

- **Ext4:** Es el sistema de archivos predeterminado en la mayoría de las distribuciones de Linux modernas. Es conocido por su rendimiento y confiabilidad.

- **Btrfs:** Es un sistema de archivos de alta tecnología diseñado para manejar grandes conjuntos de datos y proporcionar una alta redundancia y escalabilidad.

- **XFS:** Es un sistema de archivos de alto rendimiento que es adecuado para sistemas de archivos muy grandes.

- **NTFS:** Es un sistema de archivos utilizado en sistemas operativos Windows, pero que también puede ser utilizado en Linux.

- **FAT32:** Es un sistema de archivos compatible con muchos sistemas operativos y es utilizado comúnmente en unidades flash USB y otros dispositivos de almacenamiento portátiles.

Cada sistema de archivos tiene sus propias características y ventajas, y la elección del sistema de archivos adecuado depende de las necesidades y requisitos específicos de cada caso de uso.

![ComparativaSistemasDeArchivosSegúnVariablesPrincipales](https://static.platzi.com/media/user_upload/sistemadearchivos-a9552368-750e-48a8-ad0c-ea85c380b05b.jpg "ComparativaSistemasDeArchivosSegúnVariablesPrincipales")

### Particiones de un Servidor Linux

**Comando mas usados**

Comando | Descripción
----|----
`lsblk` | Lista los dispositivos de bloques y las particiones en el sistema
`fdisk` | Herramienta para administrar particiones de disco
`parted` | Herramienta para crear y administrar particiones de disco
`mkfs` | Formatea una partición con un sistema de archivos
`mount` | Monta un sistema de archivos en una partición o un directorio
`umount` | Desmonta un sistema de archivos
`df` | Muestra el espacio libre y utilizado en las particiones montadas
`du` | Muestra el tamaño de un archivo o directorio
`resize2fs` | Ajusta el tamaño de un sistema de archivos ext2, ext3 o ext4
`lvcreate` | Crea un volumen lógico en un grupo de volúmenes LVM
`lvextend` | Amplía el tamaño de un volumen lógico
`lvresize` | Ajusta el tamaño de un volumen lógico
`lvremove` | Elimina un volumen lógico
`vgcreate` | Crea un grupo de volúmenes LVM
`vgextend` | Amplía un grupo de volúmenes LVM
`vgreduce` | Reduce un grupo de volúmenes LVM
`pvcreate` | Crea un volumen físico LVM en una partición o dispositivo
`pvextend` | Amplía un volumen físico LVM
`pvresize` | Ajusta el tamaño de un volumen físico LVM
`pvremove` | Elimina un volumen físico LVM

**df:** (Disk Free) en Linux se utiliza para mostrar información sobre el espacio en disco utilizado y disponible en el sistema de archivos. Cuando se ejecuta el comando "df" sin argumentos, se muestra una lista de todas las particiones montadas en el sistema junto con su uso de espacio y su capacidad total. Algunos de los argumentos más comunes que se utilizan con el comando "df" son:

- **-h**: muestra la información de uso de espacio en formato legible por humanos, lo que significa que muestra la capacidad y el espacio utilizado en unidades como GB, MB, KB, etc., en lugar de bytes.
- **-T**: muestra el tipo de sistema de archivos en lugar del tipo de dispositivo.
- **-i:** muestra información sobre el uso de inodos en lugar de bloques.
- **-t:** muestra solo las particiones que coinciden con el tipo de sistema de archivos especificado.

El comando `lsblk` en Linux se utiliza para listar información acerca de los dispositivos de almacenamiento del sistema, incluyendo discos duros, unidades flash USB, tarjetas SD, particiones, entre otros.

Cuando se ejecuta el comando `lsblk` sin argumentos, muestra una lista jerárquica de los dispositivos de almacenamiento conectados al sistema, incluyendo el tamaño, el nombre del dispositivo y el tipo de partición. También muestra información acerca de cómo los dispositivos están conectados al sistema, como los controladores SCSI, SATA o USB.

Linux lista los discos como sda: sda, sdb, sdc, etc. Estos discos se pueden particionar a nivel logico y cada particion va a estar enumerada: sda1, sda2, sdb1, sdb2

Algunos de los argumentos más comunes que se utilizan con el comando `lsblk` son:

- **-a:** muestra todos los dispositivos, incluso aquellos que no están en uso o no tienen sistemas de archivos asociados.
- **-f:** muestra información adicional sobre los sistemas de archivos asociados con cada dispositivo.
- **-n:** suprime la cabecera y muestra solo la lista de dispositivos en una columna.
- **-o:** permite al usuario especificar qué columnas deben mostrarse en la salida.

El comando `fdisk `en Linux se utiliza para crear, editar y administrar particiones en el disco duro de un sistema. Con fdisk, se pueden ver las particiones existentes, crear nuevas particiones, modificar sus tamaños, tipos y formatos de sistema de archivos. Además, fdisk permite realizar otras tareas, como imprimir la tabla de particiones, verificar la integridad de las particiones, o escribir una tabla de particiones en un archivo. . La partición swap en Linux es un área de almacenamiento temporal en el disco duro que se utiliza cuando se agota la memoria RAM del sistema. Permite al sistema operativo manejar eficientemente los recursos de memoria y actúa como una extensión de la memoria RAM. Es importante asignar un tamaño apropiado a la partición swap para evitar un uso excesivo que pueda perjudicar el rendimiento del sistema.

### Manejo de un archivo swap

La partición swap en Linux es una partición del disco duro que se utiliza como un área de almacenamiento temporal para datos que no se utilizan actualmente en la memoria RAM del sistema. Es una forma de memoria virtual que permite al sistema operativo manejar eficientemente los recursos de memoria.

Cuando se agota la memoria RAM disponible en un sistema, el kernel de Linux mueve los datos menos utilizados a la partición swap, liberando espacio en la memoria RAM para datos más críticos. De esta manera, la partición swap actúa como una extensión de la memoria RAM, permitiendo que el sistema siga funcionando incluso cuando se agota la memoria RAM física.

Es importante tener en cuenta que, si bien la partición swap puede ser útil en situaciones de escasez de memoria, su uso también puede ser perjudicial para el rendimiento del sistema si se utiliza en exceso. Por lo tanto, se recomienda asignar un tamaño apropiado a la partición swap según las necesidades del sistema.

**CREAR TU MEMORIA SWAP**

Mover al ROOT

`cd /`

Confirmar si tengo un archivo swap actualmente

`ls -lh`

ver la cantidad de memoria disponible

`free -h`

ver si tengo suficiente espacio en disco para aumentar la particion swap actual o crear una nueva

`df -h`

creo un archivo especial para mi memoria swap

`sudo fallocate -l 2G /swapfile`

configurar permisos para que solo el root acceda a este archivo acceda en lectura y escritura

`sudo chmod 600 /swapfile`

Confirmo que tengo mi archivo swap creado

`ls -lh`

Convertir el archivo swap creado en un archivo swap valido

`sudo mkswap /swapfile`

Configurar el fstab

`sudo vim /etc/fstab`

Escribir nueva linea en el archivo fstab

`/swapfile swap swap defaults 0 0`

1. Para salir de vim solo teclea ESCAPE luego :**wq** y dar ENTER,el archivo se guarda y sales del editor
2. comprobar los cambios que hice al archivo fstab **cat /etc/fstab**
3. activamos nuestro archivo especial como memoria swap **sudo swapon /swapfile**
4. si queremos desactivar nuestro archivo para ya no usarlo como memoria swap solo usamos **sudo swapoff /swapfile y eliminar la linea del archivo fstab que configuramos anteriormente**.