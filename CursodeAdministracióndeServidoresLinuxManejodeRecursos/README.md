# Curso de Administración de Servidores Linux: Manejo de Recursos

### ¿Cómo es el arranque del sistema?

Cuando se enciende un ordenador, la BIOS se ejecuta y verifica que todos los componentes estén funcionando correctamente. Cuando termina el chequeo, busca la partición de arranque (bootloader). Éste se encarga de cargar el sistema operativo en la RAM del ordenador. En el caso de Linux, GRUB es el bootloader más común.

Una vez que el bootloader ha cargado el sistema operativo, lo primero que se ejecuta es el kernel, que es el encargado de administrar los recursos del sistema. Cuando el kernel ha cargado, se ejecuta Init, que es el primer proceso que realiza el sistema operativo. Este se encarga de iniciar todos los procesos necesarios para el correcto funcionamiento del sistema operativo.

Luego, cuando el sistema operativo ha cargado sus procesos, se carga el entorno gráfico, que es la interfaz que los nosotros como usuarios vemos y con la que interactuamos.

![](https://static.platzi.com/media/user_upload/linuxbootprocess-175c4b8d-cbcf-4b6c-b206-4f71c4dc6426.jpg)

[https://docs.oracle.com/cd/E50691_01/html/E50101/gnchj.html](https://docs.oracle.com/cd/E50691_01/html/E50101/gnchj.html)

[https://www.intel.com/content/www/us/en/support/articles/000033003/server-products.html](https://www.intel.com/content/www/us/en/support/articles/000033003/server-products.html)

**Arranque en el sistema:**

1. El SO es almacenado en memoria: En este proceso se busca encontrar un espacio dentro de la memoria para poder ejecutarse sin problemas.
2. Revisión del hardware: A través de este proceso se busca monitorizar todos los dispositivos conectados a través del Firmware.
3. Selección del dispositivo de arranque: En ella se muestra un grub donde seleccionaremos que sistema operativo a utilizar.
4. Detección de la partición EFI: Luego de seleccionar la partición se verificará el espacio de almacenamiento del sistema elegido. 5.Carga del boot loader: Este proceso arrancara el sistema elegido anteriormente.
5. Determina el kernel a usar: A través de este paso el sistema elige el kernel mediante el cual el sistema funcionara este proceso lo hace el sistema tomando en cuenta lo que ya se tiene instalado.
6. Carga del kernel: Paso donde se ira iniciando el sistema para mostrar al usuario.
7. Se carga el proceso (PID1): Este proceso es que se comenzara a iniciar el proceso de INIT en Linux.
8. Ejecuta scripts de inicio: Para este proceso se cargarán todos los scripts que se irán ejecutando en segundo plano conocidos como daemons o demonios.
9. Por último: El sistema se habrá iniciado correctamente para poder ser usado por el usuario administrador o usuario normal.

### Modo recovery

- En el contexto de Linux, el "modo de recuperación" (recovery mode en inglés) es una opción de arranque especial que proporciona un entorno de trabajo mínimo con privilegios de administrador (root) para solucionar problemas o realizar tareas de mantenimiento en un sistema Linux. También se conoce como "modo de emergencia" o "modo de rescate".

- Cuando se inicia en el modo de recuperación, el sistema operativo Linux se inicia con un conjunto mínimo de servicios y controladores. Esto permite al usuario realizar diversas acciones, como:

- Recuperación de contraseñas: En el modo de recuperación, se puede modificar o restablecer contraseñas de usuarios, lo que resulta útil si se ha olvidado una contraseña y no se puede iniciar sesión en el sistema.

- Reparación del sistema de archivos: Si el sistema de archivos tiene errores o problemas de integridad, se puede utilizar el modo de recuperación para realizar una comprobación y reparación del sistema de archivos.

- Modificación de configuraciones: El modo de recuperación permite editar archivos de configuración importantes, como el archivo /etc/passwd, /etc/group, /etc/sudoers y otros, lo que permite realizar cambios necesarios para solucionar problemas.

- Actualización de software: En algunos casos, se puede utilizar el modo de recuperación para actualizar o instalar software adicional o corregir paquetes rotos que están causando problemas en el sistema.

- Resolución de problemas de red: En el modo de recuperación, se pueden realizar pruebas de conectividad de red y solucionar problemas relacionados con la configuración de red, como la configuración de interfaces de red o problemas con el servidor DHCP.

- Es importante tener en cuenta que el modo de recuperación proporciona un acceso de nivel de administrador al sistema, por lo que se debe tener cuidado al realizar cambios para evitar daños o alteraciones no deseadas.

### ¿Qué son los grupos y usuarios en Linux?

Una tarea de administración del sistema básica es configurar una cuenta de usuario para cada usuario en un sitio. Una cuenta de usuario típica incluye la información que necesita un usuario para iniciar sesión y utilizar un sistema, sin tener la contraseña root del sistema. Los componentes de información de cuenta de usuario se describen en Componentes de cuentas de usuario.

Al configurar una cuenta de usuario, puede agregar el usuario a grupos de usuarios predefinidos. Un uso habitual de grupos es configurar permisos de grupo en un archivo y directorio, lo que permite el acceso sólo a los usuarios que forman parte de ese grupo.

Por ejemplo, puede tener un directorio que contenga archivos confidenciales a los que sólo unos pocos usuarios deberían tener acceso. Puede configurar un grupo denominado topsecret que incluya los usuarios que trabajan en el proyecto topsecret. Y, puede configurar los archivos topsecret con permiso de lectura para el grupo topsecret. De esta manera, sólo los usuarios del grupo topsecret podrán leer los archivos.

Un tipo especial de cuenta de usuario, denominado un rol, sirve para brindar a usuarios seleccionados privilegios especiales. Para obtener más información, consulte Control de acceso basado en roles (descripción general) de Guía de administración del sistema: servicios de seguridad.

¿Qué son las cuentas de usuario y los grupos? - Guía de administración del sistema: administración básica. (2011, 1 enero). [https://docs.oracle.com/cd/E24842_01/html/E23289/userconcept-36940.html](https://docs.oracle.com/cd/E24842_01/html/E23289/userconcept-36940.html "https://docs.oracle.com/cd/E24842_01/html/E23289/userconcept-36940.html")

**Usuario:** Nos permite separar las responsabilidades y permisos de acciones en el sistema. Dependiendo los permisos que tengan son las acciones que podrán ejecutar.

**Características: UID:** Identificador único del sistema GIDs: uno o más IDs que los relacionan a un grupo Directorio Home: está en la ruta /home/<username> Archivos **/etc/passwd**: contiene info de nuestros usuarios en formato `name:password:UID:GID:GECOS:directory:shell GECOS`: info extra **SHELL**: la shell de inicio **/etc/shadow**: contraseña de forma sifrada, si contiene un asterico * significa que jamas tuvo una contraseña asignada, si contiene un simbolo de exclamacion ! ha sido bloqueado

**Grupos:** Agrupan usuario y conjunto de permisos, estos son muy usados por servicios como demonios, docker, postgres, etc

Aquí comparto un artículo muy interesante sobre el archivo shadow y cómo leer la información almacenada aquí. Link:
[https://linuxize.com/post/etc-shadow-file/](https://linuxize.com/post/etc-shadow-file/)
![](https://static.platzi.com/media/user_upload/Captura%20de%20Pantalla%202023-03-28%20a%20la%28s%29%2012.09.09%20p.%C2%A0m.-02a5423c-91ba-47e4-9076-4b9337b15ed5.jpg)

### Manejo de usuarios

**ls: **listar directorio actual

**ls /home**: listar un directorio especifico

**if config**: configuracion de red

**ssh username@ip**: conexion por ssh

**cat /etc/passwd**: ver archivo donde estan los usuarios

**less /etc/shadow**: ver usuarios con su contraseña de forma cifrada

**(podemos usar tanto less o cat)**

**su root**: cambiar a usuario admin

**pwd**: nombre del directorio actual

**adduser username**: agregar nuevo usuario

**whoami**: ver usuario actual

**id**: ver info del usuario (uid,gid,groups)

**chfn username**: cambiar informacion del usuario

**usermod --lock username**: bloquear usuario (accesos futuros)

**htop**: ver todos los procesos actuales (podemos ver los procesos referentes a un usuario) con F9 matamos el proceso por ejemplo los que aparecen con ssh

**deluser username**: eliminar un usuario (aunque la carpeta en home permanece)

**rm -rf /home/username/**: eliminar un directorio como la carpeta del usuario

**usermod --unlock username**: para desbloquear al usuario

### Manejo de grupos

Para manejar grupos de usuarios en Linux, se pueden seguir los siguientes pasos:

1. **Verificar los grupos existentes**: se puede usar el comando `cat /etc/group` para mostrar una lista de los grupos existentes en el sistema.
2. **Crear un nuevo grupo**: se puede utilizar el comando `sudo groupadd <nombre_del_grupo>` para crear un nuevo grupo en el sistema.
3. **Agregar usuarios a un grupo**: se puede utilizar el comando `sudo usermod -a -G <nombre_del_grupo> <nombre_de_usuario>` para agregar un usuario existente al grupo.
4. **Verificar los usuarios de un grupo**: se puede utilizar el comando` id <nombre_del_grupo>` para mostrar una lista de usuarios que pertenecen a un grupo.
5. **Eliminar un grupo**: se puede utilizar el comando sudo groupdel <nombre_del_grupo> para eliminar un grupo existente en el sistema.
6. **Cambiar los permisos de un archivo o directorio para un grupo**: se puede utilizar el comando `sudo chgrp <nombre_del_grupo> <ruta_al_archivo_o_directorio>` para cambiar el grupo propietario de un archivo o directorio. Luego se puede utilizar el comando `sudo chmod g+<permisos> <ruta_al_archivo_o_directorio>` para otorgar permisos específicos para el grupo propietario.
7. **Cambiar el nombre de un grupo**: se puede utilizar el comando `sudo groupmod -n <nuevo_nombre> <nombre_actual>` para cambiar el nombre de un grupo existente.
8. **Verificar los permisos de un grupo**: se puede utilizar el comando `sudo getfacl /ruta/al/archivo-o-directorio` para verificar los permisos de un archivo o directorio. En la salida del comando, se puede observar información sobre los permisos de usuario, grupo y otros.
9. `Agregar usuarios a un grupo secundario`: se puede utilizar el comando sudo usermod -a -G <nombre_del_grupo_secundario> <nombre_de_usuario> para agregar un usuario existente a un grupo secundario.
10. Eliminar usuarios de un grupo: se puede utilizar el comando `sudo gpasswd -d <nombre_de_usuario> <nombre_del_grupo>` para eliminar un usuario existente de un grupo.
11. Verificar los permisos de un grupo específico: se puede utilizar el comando sudo visudo para abrir el archivo de configuración sudoers, y luego agregar la siguiente línea para permitir que los usuarios del grupo tengan permisos de superusuario: `%<nombre_del_grupo> ALL=(ALL) ALL`.
12. `Verificar los grupos de un usuario`: se puede utilizar el comando `groups <nombre_de_usuario>` para mostrar una lista de grupos a los que pertenece un usuario específico.

**Comandos**

- **groups**: ver todos mis grupos

- **getent group sudo**: ver usuarios del grupo sudo

- **groupadd groupname**: crear nuevo grupo

- **groupdel groupname**: eliminar grupo

- **groupmod -n newGroupName oldGroupName**: cambiar el nombre de un grupo

- **usermod -aG groupname username**: agregar usuario existente a grupo existente

- **useradd username -m -g groupname**: crear usuario y asignarlo a un grupo, -m crea el directorio personal y -g asigna el usuario a un grupo ya creado

- **gpaswd -d username groupname**: quitar usuario de un grupo

- **mkdir shared**: crear carpeta

- **ls -la**: ver permisos de las carpetas

- **chgrp groupname  /shared/**: asignar carpeta a grupo

- **chmod 770 /shared/**: cambiar permisos, todos los permisos al owner, todos al grupo y denegar acceso a todos los que no pertenezcan al grupo

- **chmod +s /shared/**: asignar permisos especiales, cualquier archivo que creemos dentro del directorio se va a crear con el group owner

- **less /etc/group | grep username**: buscar usuario especifico dentro de los grupos

- **less /etc/group**: listar grupos

- **ls**: listar directorio actual

- **ls /home**: listar un directorio especifico

- **if config**: configuracion de red

- **ssh username@ip**: conexion por ssh

- **cat /etc/passwd**: ver archivo donde estan los usuarios

- **less /etc/shadow**: ver usuarios con su contraseña de forma cifrada (podemos usar tanto less o cat)

- **su username**: cambiarse de usuario

- **pwd**: nombre del directorio actual

- **adduser username**: agregar nuevo usuario

- **whoami**: ver usuario actual

- **id**: ver info del usuario (uid,gid,groups)

- **chfn username**: cambiar informacion del usuario

- **usermod --lock username**: bloquear usuario (accesos futuros)

- **htop**: ver todos los procesos actuales (podemos ver los procesos referentes a un usuario) con F9 matamos el proceso por ejemplo los que aparecen con ssh

- **deluser username**: eliminar un usuario (aunque la carpeta en home permanece)

- **rm -rf /home/username/**: eliminar un directorio como la carpeta del usuario

### El control de accesos en Linux

Para Linux todo es un “objeto” **Control de accesos**

- Depende del usuario y las acciones que quiera realizar (permitidas o denegadas).
- Quien crea el “objeto” es dueño de él.
- La cuenta root puede acceder a cualquier objeto que quiera de las demás cuentas.
- Solo la cuenta root puede hacer ciertas operaciones sensibles.

Buenas prácticas

- No acceder directamente desde la cuenta root, mejor usar su.
- Otorgar permisos de administrador solo a los usuarios necesarios y revocar accesos después de cierto tiempo.

**El control de acceso en Linux** se refiere al conjunto de medidas y configuraciones utilizadas para gestionar y regular el acceso a recursos y funciones del sistema operativo Linux. El objetivo principal del control de acceso es proteger la integridad, la confidencialidad y la disponibilidad de los datos y recursos del sistema.

En Linux, el control de acceso se implementa a través de varios mecanismos y componentes. A continuación, se presentan algunos de los principales:

- Permisos de archivos: Linux utiliza un sistema de permisos de archivos basado en usuarios, grupos y otros. Los permisos de lectura, escritura y ejecución se asignan a propietarios, grupos y otros usuarios, lo que determina qué acciones pueden realizar en un archivo o directorio.

- Control de usuarios: Linux administra los usuarios a través de cuentas de usuario individuales. Los administradores pueden crear y eliminar cuentas de usuario, asignar contraseñas y definir los permisos y privilegios asociados a cada cuenta.

- Control de grupos: Los grupos en Linux permiten organizar y administrar usuarios con características y permisos similares. Los archivos y directorios pueden estar asignados a un grupo específico, lo que permite que los miembros de ese grupo compartan acceso a esos recursos.

- Directivas de seguridad: Linux admite la implementación de directivas de seguridad a través de herramientas como SELinux (Security-Enhanced Linux) o AppArmor. Estas herramientas permiten especificar reglas detalladas sobre qué acciones están permitidas o restringidas para procesos y aplicaciones específicas.

- Firewall: Linux incluye herramientas de firewall, como iptables o nftables, que permiten controlar las conexiones de red entrantes y salientes, lo que ayuda a proteger el sistema de posibles amenazas externas.

- Autenticación y autorización: Linux utiliza sistemas de autenticación para verificar la identidad de los usuarios, como el uso de contraseñas o autenticación de clave pública. La autorización se basa en la asignación de permisos y privilegios específicos a usuarios y grupos.

Estas son solo algunas de las medidas de control de acceso en Linux. El control de acceso es un aspecto fundamental de la seguridad en el sistema operativo y su implementación adecuada contribuye a proteger los recursos del sistema contra accesos no autorizados.

### Creación de un usuario administrador

Introducción

La separación de privilegios es uno de los paradigmas de seguridad fundamentales implementados en Linux y en los sistemas operativos tipo Unix. Los usuarios regulares operan con privilegios limitados para reducir el alcance de su influencia en su propio entorno, y no en el sistema operativo en general.

Un usuario especial, llamado root, tiene privilegios de superusuario. Esta es una cuenta administrativa sin las restricciones que tienen los usuarios normales. Los usuarios pueden ejecutar comandos con privilegios de superusuario o root de varias maneras.

En este artículo, explicaremos cómo obtener privilegios root de forma correcta y segura, con un enfoque especial en la edición del archivo /etc/sudoers.

Completaremos estos pasos en un servidor de Ubuntu 20.04, pero la mayoría de las distribuciones de Linux modernas, como Debian y CentOS, deberían funcionar de manera similar.

Esta guía asume que usted ya ha completado la configuración inicial del servidor que se mencionó aquí. Inicie sesión en su servidor como usuario no root regular y continúe como se indica abajo.

Si estas en fedora el grupo sudo es wheel, para que asignarle permisnos de admin a un usuario el comando correcto es:
`usermod -aG wheel username`

### Particionando y montando una unida

Montar particiones en Linux. Mount Cada sistema de ficheros que se desea incorporar se tiene que montar en un directorio. Se suelen crear subdirectorios en /mnt/ o en /media/. Por otra parte, cada dispositivo que se monta en el sistema debe poseer información sobre su montaje en un fichero ubicado en /dev/.

Es habitual encontrarnos con los subdirectorios floppy y cdrom, que estarán vacíos hasta que no introduzcamos los correspondientes dispositivos. Debido a la frecuencia con la que se solían introducir y extraer este tipo unidades el sistema tiene los mecanismos necesarios para permitir que el contenido se muestre automáticamente en los directorios anteriormente mencionados.

Cuando vinculamos un directorio vacío con el contenido de un nuevo sistema de ficheros, bien sea un dispositivo externo, otra partición de disco, un directorio compartido, etc. se dice que estamos “montando” dicha partición.

Listar particiones después de crear el disco .

`lsblk`

`sudo fdisk -l`

Comando para crear particiones : `sudo fdisk /dev/sdb`

F : listar las particiones en fdisk

1. Creando nueva partición fdisk: `n`

2. Seleccionar el tipo de partición: `p` (primaria)

3. Seleccionar entre 1-4 tamaño inicial por defecto: 1

4. Indicar el tamaño que quieres la partición : +4G tamaño en Gigabytes.

5. Por defecto te crea una unidad llamada "sdb1" con el tamaño asignado.

6. Realiza el proceso con la segunda unidad presionando ENTER para que use el tamaño de disco restante.

7. Para guardar los cambios presiona : `w`

8. Formatear partición con ext4 :` sudo mkfs.ext4 /dev/sdb1`

9. Montar la partición: crea una carpeta con mkdir "scripts" : `sudo mkdir scripts`.

10. Comando para montar: `sudo mount /dev/sdb1 /scripts`

11. Desmontar unidad : sudo umount /dev/sdb1/scripts

12. Editar el file system para que el disco se monte en el inicio del sistema:

`sudo vim /etc/fstab`

linea que debes agregar: `/dev/sdb1 /scripts ext4 defaults 0 0`

12+1: Guardar cambios y reiniciar.

 14. Si tienes problemas o escribiste mal puedes ingresar en modo recovery.

 ### ¿Qué es RAID y LVM?
¿Qué es RAID y LVM?
El primer concepto, RAID (redundant Array of Independent Disk), nos permite hacer un arreglo de discos redundantes, similar a aun back up, pero con la diferencia de que no se almacena en un sitio aislado del servidor de producción.

Existen distintos tipos de RAID: 1, 2, 3, etc.

En ellos, podremos usar dos discos duros para usarlos como si fuera uno solo, alojando siempre la misma información, esto con el objetivo de generar redundancia.

**LVM**
Es un gestor lógico de discos, que nos evita crear o modificar particiones, creando un volumen lógico que agrupa los volúmenes físicos para hacer particiones más dinámicas: lo cual nos permite trabajar más eficiente y cómodamente.

**LVM sobre RAID**

Nos permite hacer uso de los beneficios de ambos arreglos: tener la redundancia que nos ofrece RAID, y el dinamismo en la administración de particiones que nos ofrece LVM.

![](https://static.platzi.com/media/user_upload/RAID%2BLVM-df96322e-96ed-4fc2-bdc4-4eaac1f5929a.jpg)

### Creacion de sistema RAID 1

Proyecto del curso
Creación de un sistema RAID 1

- Añadimos 2 discos físicos en nuestro VirtualBox, las cuales deben ser del mismo tamaño.
- Iniciamos nuestra máquina virtual.
- Creamos las particiones virtuales con los comandos:

```shell
fdisk /dev/sdb
n → crea partición
p → crea partición primaria
Elegimos el primer sector para guardar nuestra partición.
Elegimos qué tamaño deseamos para la partición: +3G
p → Nos muestra tabla de particiones.
t → cambia el tamaño de las particiones.
w → Guardar los cambios (cuidado, una vez hecho, no hay vuelta atrás).
```

- Repetimos el proceso para el segundo disco.
- Validamos la creación con:
`lsblk`

- Creamos nuestro arreglo RAID con una herramienta de instalación de medios, que deberemos instalar:
`apt install mdadm`

- luego, ejecutamos el programa para crear el volumen lógico:
`mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sd{c,d}`

- Validamos la creación con:
`lsblk`

- Para mayor información debemos ingresar el siguiente comando:
`mdadm --misc --detail /dev/md0`

### Creación de LVM sobre RAID 1

Creación de LVM sobre RAID1
- Instalamos el paquete:
`apt install lvm2`

- Creamos primero el PV (Physical Volume) que alojará el LV (Logical Volume):
`pvcreate /dev/md0`

- Validamos que se haya creado el PV con el siguiente comando:
`pvdisplay`

- Creamos ahora nuestro grupo de volúmenes (VG):
`vgcreate volumegroup /dev/md0`

- Validamos que se haya creado el VG con el siguiente comando:
`vgdisplay`

- Finalmente, crearemos las particiones más pequeñas:

```shell
lvcreate --name newname1 --size 1Gb volumegroupname1
lvcreate --name newname2 --size 1Gb volumegroupname2
lvdisplay
```

### Agregando el sistema a fstab

Una vez creados los RAID y los Volúmenes lógicos, ya podemos pasar a montarlos en nuestro sistema para poderles dar uso.

**Pasos:**

1. Formateamos a EXT4.
2. Agregar a fstab para poderlas usar.
3. Extras: puedes crear permisos, grupos, etc.
- Para formatear el sistema, escribimos:
`mkfs.ext4 /dev/volumegroup/public`

- Confirmamos con:
`lsblk`

- Ahora procederemos a montarlos:

```shell
cd / →Vamos al directorio principal
ls -la → Listamos
mkdir public && mkdir private → Creamos las carpetas.
sudo mount /dev/volumegroup/public /public →Montamos la unidad sobre esa carpeta
touch /public/new_file.txt
ls /public
```

- Finalmente, los añadiremos al archivo fstab, escribiendo:
`vim /etc/fstab`

- En el archivo de configuración añadimos:
```shell
/dev/volumegroup/private /private ext4  defaults        0       0
/dev/volumegroup/public/public ext4  defaults        0       0
```

### Preparando el sistema

Proyecto 2: Recuperar GRUB
Cuando nuestro GRUB se desconfigura (que es un evento bastante común), debemos estar en la capacidad de restaurarlo, para ello debemos seguir las siguientes indicaciones:

La carpeta donde se aloja el archivo de configuración es el siguiente: /boot/grub/grub.cfg

- Primero, romperemos nuestro grub, de tal manera que podamos repararlo:
`sudo mv /boot/grub/grub.cfg.bck → Cambia el nombre del archivo original de grub.cfg`

- Reiniciamos nuestra máquina virtual con:
`sudo reboot`

Los archivos en /etc/grub.d son hooks que poseen scripts que son ejecutados al ejecutar grub-mkconfig. Por lo general, la mayoria de distros tiene hooks para crear las entradas:

- kernel predeterminado con el initrd predeterminado,
- kernel predeterminado con initrd fallback (sin el hook autodetect)
- un kernel alternativo (probablemente en una version lts)
- un script que detecta la presencia de otros ejecutables efi (os-prober)

Pueden escribir sus propios scripts para, por ejemplo, detectar imagenes uki en su particion /boot y anadirlos como entradas, sin embargo, no es muy facil que digamos.

### Escalando el sistema con chroot e instalando Grub

Reparando el archivo grub con chroot
Para poder reparar el archivo debemos hacer un escalamiento de privilegios a través de chroot, esto aprovechando una vulnerabilidad del sistema de linux. Para esto es importante tener en cuenta que los discos no han sido cifrados.

Iniciamos como si fuéramos a instalar una nueva máquina virtual, pero en la misma máquina de Ubuntu.

- Vamos a configuración > Storage >Controller IDE > seleccionamos nuestra distribución de Ubuntu.
- Iniciamos la máquina virtual.
- Seleccionamos try or install Ubuntu.
- En la pantalla de login, presionamos: fn + f2 para abrir una nueva consola.
- Lo primero que haremos será crear una contraseña al usuario root.
`sudo passwd`

- Luego validamos las particiones o discos existentes con:
`fdisk -l | less`

- Vamos a la carpeta home del usuario:
`cd /`

- Montamos la unidad que contiene el file system que validamos con el comando fdisk, en este caso es el sda2:
`mount /dev/sda2 /mnt`

- Para acceder a los archivos dañados bastará con acceder a la carpeta que los contiene, para ello deberemos saber la ruta:
`ls /mnt/boot/grub/`

- Ahora haremos el montaje de todo el sistema uniendo los archivos de nuestro sistema live con los del SO estropeado:

```shell
mount -o bind /dev/     /mnt/dev
mount -o bind /dev/pts  /mnt/dev/pts
mount -o bind /proc     /mnt/proc
mount -o bind /run      /mnt/run
mount -o bind /sys      /mnt/sys
```

- Luego, para volver a ser el usuario root del sistema corrupto (en lugar del root del live SO), debemos emplear el comando chroot (change root), así:

```shell
chroot /mnt /bin/bash → Lo cual nos abrirá una shell con permisos del usuario root del SO que fue reparado. 
```

- Después, para reparar el grub, debemos tipear:
`grub-mkconfig -o /boot/grub/grub.cfg`

Finalmente, si queremos instalar este grub debemos escribir:
`grub-install --boot-directory=/boot/ --recheck /dev/sda`

- Hacemos un ls /boot/grub para validar que todo esté correcto; cerramos sesiones con exit y apagamos la máquina.

- Ya solo queda eliminar la imagen live, iniciar la máquina e iniciar sesión.