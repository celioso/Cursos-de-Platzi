# Curso de Programación en Bash Shell

## Todo lo que aprenderás para programar en Bash Shell

**Lecturas recomendadas**

[Matoosfe - YouTube](https://www.youtube.com/user/matoosfe)

[Not Acceptable!](http://matoosfe.com/)

## Componentes de Linux, Tipos de Shell y Comandos de información

Linux tiene 3 partes principales:

- **Kernel**: Es el núcleo del Sistema Operativo y se gestionan los recursos de hardware como la memoria, el procesamiento y los dispositivos periféricos conectados al computador.
- **Shell**: Es el interprete, un programa con una interfaz de usuario permitiendo ejecutar las aplicaciones en un lenguaje de alto nivel y procesarlas en un lenguaje de bajo nivel para manipular y controlar aplicaciones y programas como nuestro proyecto.
- **Aplicaciones**: Son las aplicaciones con las que interactuamos día a día.

Tipos de Shells:

- SH
- KSH
- CSH
- BASH

![Tipos de Shells](images/tipos_de_shells.png)

Algunos comandos para conocer información sobre el resto de comandos:

man [comando]
info [comando]

## Bash scripting

- Bash Scripting

La idea básica de generar programas en bash es poder ejecutar múltiples comandos de forma secuencial en muchas ocasiones para automatizar una tarea en especifico. Estos comandos son colocados en un archivo de textos de manera secuencial para poder ejecutarlos a posterioridad.

Un archivo `.vimrc` podremos configurar de mejor manera nuestro editor VIM.

Más editores `vi` y `nano`.

![editor_vim.png](images/editor_vim.png)

Presionamos `I` para poder escribir en nuestro editor.
Presionamos `ESC` para salir del modo edición, luego escribimos `:wq` para salir y guardar nuestro archivo.

## Crear nuestro primer Script

**1_utilityPostgres.sh**

```bash
# !/bin/bash
# Programa para realizar algunas operaciones utilitarios de Postgres

echo "Hola bienvenido al curso de Programación bash"
```

**1_comments.sh**

```bash
#! /bin/bash
# PROGRAMA: U-POSG
echo "Programa Utilidades Postgres"
    <<"COMENTARIO 1"
    Programa para administrar las utilidades de la Base
    de Datos Postgres
   "COMENTARIO 1"
    
exit 0
```

El comando `ls -l 1_utilityPostgres.sh` en un sistema Unix o Linux sirve para mostrar detalles sobre el archivo llamado `1_utilityPostgres.sh`. A continuación te explico lo que hace:

1. **`ls`**: Lista archivos y directorios en el directorio actual.
2. **`-l`**: Muestra la lista de archivos en formato largo (detallado), proporcionando información como permisos, número de enlaces, propietario, grupo, tamaño y fecha de la última modificación.
3. **`1_utilityPostgres.sh`**: Es el nombre del archivo específico que deseas listar (en este caso, parece ser un script de shell relacionado con PostgreSQL).

Ejemplo de salida:

```bash
-rwxr-xr-x 1 usuario grupo 2048 sep  9 14:32 1_utilityPostgres.sh
```

Esto indica:

- **`-rwxr-xr-x`**: Los permisos del archivo (lectura, escritura, ejecución para el propietario, y lectura, ejecución para grupo y otros).
- **`1`**: Número de enlaces (hard links).
- **`usuario`**: Propietario del archivo.
- **`grupo`**: Grupo al que pertenece el archivo.
- **`2048`**: Tamaño del archivo en bytes.
- **`sep 9 14:32`**: Fecha y hora de la última modificación.
- **`1_utilityPostgres.sh`**: El nombre del archivo.

Este comando te permite verificar las propiedades y permisos del archivo para asegurarte, por ejemplo, de que puedes ejecutarlo como un script de shell.

## Ejecutar nuestro script con un nombre único

**pwd**: El comando pwd (abreviatura de Print Working Directory) en sistemas Unix o Linux se utiliza para mostrar la ruta completa del directorio en el que te encuentras actualmente en la terminal.

El comando `chmod` en sistemas Unix/Linux se utiliza para cambiar los permisos de archivos y directorios. Los permisos definen quién puede leer, escribir o ejecutar un archivo o directorio.

### Sintaxis básica:
```bash
chmod [opciones] permisos archivo
```

### Tipos de permisos:

- **r** (read) – Permiso de lectura.
- **w** (write) – Permiso de escritura.
- **x** (execute) – Permiso de ejecución.

### Categorías de usuarios:

- **u** (user) – El propietario del archivo.
- **g** (group) – Los usuarios del grupo.
- **o** (others) – Todos los demás usuarios.
- **a** (all) – Todos los usuarios (u, g, o).

### Modificar permisos con letras:
Puedes agregar (`+`), quitar (`-`) o asignar (`=`) permisos para un archivo o directorio.

#### Ejemplos:

1. **Añadir permiso de ejecución al propietario:**
   ```bash
   chmod u+x archivo.sh
   ```

2. **Eliminar permiso de escritura para el grupo:**
   ```bash
   chmod g-w archivo.txt
   ```

3. **Dar a todos los usuarios permiso de lectura:**
   ```bash
   chmod a+r archivo.txt
   ```

### Modificar permisos con números (notación octal):

Cada tipo de permiso tiene un valor numérico asociado:

- **r = 4**
- **w = 2**
- **x = 1**

Estos valores se suman para definir los permisos de cada categoría. La estructura es:
```bash
chmod ### archivo
```
Donde cada número representa los permisos de **u**, **g**, **o** en ese orden.

#### Ejemplos:

1. **Dar permisos completos al propietario (7), lectura y ejecución al grupo (5) y solo lectura a otros (4):**
   ```bash
   chmod 754 archivo.sh
   ```

   Esto sería equivalente a:
   - Propietario: lectura, escritura, ejecución (7 = 4+2+1)
   - Grupo: lectura, ejecución (5 = 4+1)
   - Otros: solo lectura (4)

2. **Dar permisos de lectura y escritura a todos:**
   ```bash
   chmod 666 archivo.txt
   ```

3. **Dar permisos completos a todos (lectura, escritura, ejecución):**
   ```bash
   chmod 777 archivo.sh
   ```

El comando `chmod` es muy flexible y te permite controlar exactamente cómo se puede acceder a cada archivo o directorio en tu sistema.

### Crear un contenedor con docker de ubuntu

En Docker, no se crea exactamente una "máquina virtual", sino que se crean **contenedores**. Los contenedores son entornos ligeros y aislados que pueden ejecutar aplicaciones y sistemas operativos, como Ubuntu, de manera eficiente. A continuación, te explico cómo crear y ejecutar un contenedor de **Ubuntu** usando Docker:

### Pasos para crear un contenedor de Ubuntu en Docker:

#### 1. **Instalar Docker**

Si no tienes Docker instalado en tu máquina, primero debes instalarlo. Aquí te dejo los comandos para instalar Docker en un sistema Ubuntu:

```bash
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

Para verificar que Docker esté funcionando correctamente:

```bash
sudo docker --version
```

#### 2. **Buscar la imagen de Ubuntu en Docker Hub**

Las imágenes de Docker son plantillas que puedes usar para crear contenedores. Para obtener la imagen oficial de Ubuntu, usa el siguiente comando:

```bash
sudo docker search ubuntu
```

Esto mostrará una lista de imágenes relacionadas con Ubuntu. Normalmente, utilizarás la imagen oficial, que se llama simplemente `ubuntu`.

#### 3. **Descargar la imagen de Ubuntu**

Descarga la imagen oficial de Ubuntu con el siguiente comando:

```bash
sudo docker pull ubuntu
```

Esto descargará la última versión de Ubuntu disponible.

#### 4. **Crear y ejecutar un contenedor de Ubuntu**

Ahora que tienes la imagen de Ubuntu, puedes crear y ejecutar un contenedor basado en esa imagen. Para hacer esto, ejecuta:

```bash
sudo docker run -it ubuntu
```

Aquí:
- `-it` te permite interactuar con el contenedor en modo interactivo (como si fuera una terminal).
- `ubuntu` es el nombre de la imagen que descargaste.

Este comando creará un contenedor basado en la imagen de Ubuntu y te proporcionará una sesión de terminal dentro del contenedor.

#### 5. **Instalar software dentro del contenedor**

Una vez dentro del contenedor, puedes ejecutar comandos como si estuvieras en una instalación de Ubuntu normal. Por ejemplo, puedes actualizar los paquetes e instalar software como lo harías normalmente:

```bash
apt update
apt install vim
```

#### 6. **Salir del contenedor**

Cuando termines de trabajar dentro del contenedor, puedes salir escribiendo `exit` o presionando `Ctrl+D`. Esto detendrá el contenedor.

```bash
exit
```

#### 7. **Listar los contenedores activos e inactivos**

Para ver los contenedores que están corriendo:

```bash
sudo docker ps
```

Para ver **todos** los contenedores, incluidos los detenidos:

```bash
sudo docker ps -a
```

#### 8. **Reiniciar un contenedor detenido**

Si ya has creado un contenedor pero lo detuviste, puedes reiniciarlo con su ID o nombre:

```bash
sudo docker start <ID-del-contenedor>
sudo docker attach <ID-del-contenedor>
```

El comando `start` inicia el contenedor, y `attach` te permite interactuar con él de nuevo.

#### 9. **Guardar cambios en una nueva imagen (opcional)**

Si has hecho cambios en el contenedor y quieres crear una nueva imagen con esos cambios, puedes "commit" esos cambios a una nueva imagen:

```bash
sudo docker commit <ID-del-contenedor> mi_ubuntu_personalizado
```

Esto crea una nueva imagen llamada `mi_ubuntu_personalizado` que puedes usar más tarde.

#### 10. **Eliminar un contenedor o imagen**

Para eliminar un contenedor cuando ya no lo necesites:

```bash
sudo docker rm <ID-del-contenedor>
```

Para eliminar una imagen:

```bash
sudo docker rmi <nombre-de-la-imagen>
```

### Resumen:
Con Docker puedes crear contenedores de Ubuntu fácilmente y ejecutarlos como si fueran máquinas virtuales ligeras. La principal diferencia con una máquina virtual tradicional es que los contenedores son más ligeros y comparten el kernel del sistema operativo anfitrión, lo que los hace más eficientes.

Si necesitas más información o ayuda con un paso en particular, ¡no dudes en preguntar!

para ejecutar el programa se utiliza lo siguiente:  `bash 1_utilityPostgres.sh` o `./1_utilityPostgres.sh`

`type` es para ver el tipo d earchivo

## Declaración de Variables y Alcance en Bash Shell

En **Bash** (el **Bourne Again Shell**), la declaración y el alcance de variables tienen un comportamiento particular. Aquí te explico cómo funcionan:

### Declaración de Variables

1. **Declaración Básica**

   Para declarar una variable en Bash, simplemente asigna un valor a un nombre de variable sin espacios alrededor del signo igual (`=`). Por ejemplo:

   ```bash
   variable="valor"
   ```

   - No se usan espacios antes y después del signo igual (`=`).
   - Las variables en Bash son sensibles a mayúsculas y minúsculas (`Variable` y `variable` son distintas).

2. **Variables de Entorno**

   Para que una variable esté disponible en otros procesos o subprocesos (es decir, como variable de entorno), debes exportarla usando el comando `export`:

   ```bash
   export variable="valor"
   ```

### Alcance de Variables

1. **Alcance Local**

   Las variables definidas en un script o en una sesión de shell son locales a esa sesión. Por ejemplo, si defines una variable dentro de una función, esa variable solo es accesible dentro de esa función:

   ```bash
   function mi_funcion {
     local variable_local="valor"
     echo "$variable_local"  # Esto funciona
   }

   echo "$variable_local"  # Esto no funcionará, ya que variable_local no está definida fuera de la función
   ```

   - **`local`**: La palabra clave `local` se usa dentro de funciones para declarar variables locales.

2. **Alcance Global**

   Las variables que no están marcadas como locales y se declaran fuera de funciones son globales y están disponibles en todo el script y en subprocesos si se exportan:

   ```bash
   variable_global="valor"

   function mi_funcion {
     echo "$variable_global"  # Esto funciona porque variable_global es global
   }

   mi_funcion
   echo "$variable_global"  # También funciona aquí
   ```

3. **Alcance en Subprocesos**

   Si exportas una variable, estará disponible en cualquier subproceso que inicie el shell. Por ejemplo:

   ```bash
   export variable="valor"

   (echo "$variable")  # Imprime "valor" en el subproceso
   ```

   - Los subprocesos heredan las variables de su proceso padre, pero los cambios realizados en el subproceso no afectan al proceso padre.

4. **Variables en Scripts**

   Cuando ejecutas un script, las variables definidas dentro del script solo afectan al script y a sus subprocesos. No afectan a tu sesión de shell actual a menos que uses `source` para ejecutar el script en el entorno actual:

   ```bash
   ./mi_script.sh  # Ejecuta el script en un subshell, variables no afectan al shell actual
   source mi_script.sh  # Ejecuta el script en el entorno actual, variables afectan al shell actual
   ```

### Ejemplos Prácticos

1. **Declarar y Usar Variables**

   ```bash
   #!/bin/bash

   mensaje="Hola, Mundo!"
   echo "$mensaje"
   ```

2. **Variable de Entorno**

   ```bash
   #!/bin/bash

   export PATH="/usr/local/bin:$PATH"
   ```

3. **Variable Local en Función**

   ```bash
   #!/bin/bash

   function saludo {
     local mensaje="Hola, desde la función!"
     echo "$mensaje"
   }

   saludo
   echo "$mensaje"  # No mostrará nada porque mensaje es local a la función
   ```

Estos conceptos te ayudarán a entender cómo manejar variables y su alcance en **Bash**. Si tienes alguna pregunta adicional o necesitas más detalles sobre algún aspecto, ¡déjamelo saber!

**2_variables_2.sh**

```bash
# !/bin/bash
# Programa para revisar la declaración de variables
# Autor: Marco Toscano Freire - @martosfre

echo "Opción nombre pasada del script anterior: $nombre"
```

**2_variables.sh**

```bash
# !/bin/bash
# Programa para revisar la declaración de variables
# Autor: Marco Toscano Freire - @martosfre

opcion=0
nombre=Marco

echo "Opción: $opcion y Nombre: $nombre"

# Exportar la variable nombre para que este disponible a los demás procesos
export nombre

# Llamar al siguiente script para recuperar la variable
./2_variables_2.sh
```

Ingresar a la variables de entorno se utiliza `sudo vim /etc/profile`

y se crea al final la bariable

```bash
# Variables de Entorno S.O
COURSE_NAME=pProgramación Bash
export COURSE_NAME
```

para copiar un archivo `cp <archivo_a_copiar> <nombre_que_la_asigna_a_la_copia>`

para eliminar una linea en vin se presciona `ESC` y luego 2 veses `D`