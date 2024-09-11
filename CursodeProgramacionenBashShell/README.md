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

## Tipos de Operadores

En Bash, hay varios tipos de operadores que se utilizan para realizar diferentes tipos de operaciones. Aquí te presento una lista de los operadores más comunes en Bash:

### 1. **Operadores Aritméticos**
Estos operadores se utilizan para realizar operaciones matemáticas.

- `+` : Suma
- `-` : Resta
- `*` : Multiplicación
- `/` : División
- `%` : Módulo (residuo)
- `**` : Exponenciación (Bash 4.0+)

**Ejemplo:**
```bash
a=5
b=3
echo $((a + b))  # Salida: 8
```

### 2. **Operadores de Comparación (Numérica)**
Se usan para comparar valores numéricos.

- `-eq` : Igual a
- `-ne` : No igual a
- `-lt` : Menor que
- `-le` : Menor o igual a
- `-gt` : Mayor que
- `-ge` : Mayor o igual a

**Ejemplo:**
```bash
if [ $a -gt $b ]; then
  echo "$a es mayor que $b"
fi
```

### 3. **Operadores de Comparación (Cadenas)**
Se utilizan para comparar cadenas de texto.

- `=` : Igual a
- `!=` : No igual a
- `<` : Menor que (lexicográficamente, en `[[ ]]`)
- `>` : Mayor que (lexicográficamente, en `[[ ]]`)

**Ejemplo:**
```bash
if [ "$string1" = "$string2" ]; then
  echo "Las cadenas son iguales"
fi
```

### 4. **Operadores Lógicos**
Estos operadores se utilizan para combinar expresiones booleanas.

- `&&` : AND lógico (si la primera condición es verdadera, evalúa la segunda)
- `||` : OR lógico (si la primera condición es falsa, evalúa la segunda)
- `!` : NOT lógico (invierte el valor de la condición)

**Ejemplo:**
```bash
if [ $a -gt 0 ] && [ $b -lt 10 ]; then
  echo "Condiciones verdaderas"
fi
```

### 5. **Operadores de Redirección**
Se usan para redirigir la entrada y salida de comandos.

- `>` : Redirige la salida estándar a un archivo (sobrescribe)
- `>>` : Redirige la salida estándar a un archivo (añade al final)
- `<` : Redirige la entrada estándar desde un archivo
- `2>` : Redirige la salida de error a un archivo
- `&>` : Redirige tanto la salida estándar como la de error a un archivo

**Ejemplo:**
```bash
echo "Hola" > archivo.txt
```

### 6. **Operadores de Expansión de Variables**
Estos operadores se utilizan para manipular el contenido de las variables.

- `${variable:-default}` : Usa el valor de `variable` si está definida; si no, usa `default`.
- `${variable:=default}` : Usa el valor de `variable` si está definida; si no, asigna `default`.
- `${variable:+value}` : Usa `value` si `variable` está definida y no está vacía.

**Ejemplo:**
```bash
echo ${var:-"Valor por defecto"}
```

### 7. **Operadores de Evaluación Condicional**
Se utilizan en pruebas condicionales.

- `-e` : Verifica si un archivo existe
- `-d` : Verifica si un archivo es un directorio
- `-f` : Verifica si un archivo es un archivo regular
- `-r` : Verifica si un archivo es legible
- `-w` : Verifica si un archivo es escribible
- `-x` : Verifica si un archivo es ejecutable

**Ejemplo:**
```bash
if [ -f archivo.txt ]; then
  echo "El archivo existe"
fi
```
**3_tipoOperadores.sh**

```bash
# ! /bin/bash
# Programa para revisar los tipos de operadores
# Autor: Marco Toscano - @martosfre

numA=10
numB=4

echo "Operadores Aritméticos"
echo "Números: A=$numA y B=$numB"
echo "Sumar A + B =" $((numA + numB))
echo "Restar A - B =" $((numA - numB))
echo "Multiplicar A * B =" $((numA * numB))
echo "Dividir A / B =" $((numA / numB))
echo "Residuo A % B =" $((numA % numB))


echo -e "\nOperadores Relaciones"
echo "Números: A=$numA y B=$numB"
echo "A > B =" $((numA > numB))
echo "A < B =" $((numA < numB))
echo "A >= B =" $((numA >= numB))
echo "A <= B =" $((numA <= numB))
echo "A == B =" $((numA == numB))
echo "A != B =" $((numA != numB))


echo -e "\nOperadores Asignación"
echo "Números: A=$numA y B=$numB"
echo "Sumar A += B" $((numA += numB))
echo "Restar A -= B" $((numA -= numB))
echo "Multiplicación A *= B" $((numA *= numB))
echo "Dividir A /= B" $((numA /= numB))
echo "Residuo A %= B" $((numA %= numB))
```

para deshacer se usa `ESC` + `U`
para copiar una linea s eutiliza `ESC`, dos veces `y`, y luego `p` para que lo copie en la parte inferior

## Script con Argumentos

Hay algunos identificadores para cuando ejecutamos un script con argumentos

Identificador | Descripción
---|---
**$0** | Se refiere al nombre del Script
**$1 al ${10}** | Se refiere al número de argumento. Si es más de uno lo colocamos dentro de llaves.
**$#** | Es útil para conocer el número de argumentos enviados.
**$** | Con este podemos conocer todos los argumentos enviados.

```bash
# ! /bin/bash
# Programa para ejemplificar el paso de argumentos
# Autor: Marco Toscano Freire - @martosfre

nombreCurso=$1
horarioCurso=$2

echo "El nombre del curso es: $nombreCurso dictado en el horario de $horarioCurso"
echo "El número de parámetros enviados es: $#"
echo "Los parámetros enviados son: $*"
```

para ingresar los argumentos son: `Archivo.sh <agumento_1> <argumento_2>` Ejemplo: `./4_argumentos.sh "Programación Bash" "18:00 a 20:00"`

En Bash, un script puede recibir argumentos que se pasan al ejecutar el script desde la línea de comandos. Estos argumentos se pueden utilizar dentro del script para realizar diferentes acciones. Aquí te muestro cómo trabajar con argumentos en un script de Bash.

### Crear un Script con Argumentos

1. **Crear el Script**

   Abre tu editor de texto y crea un archivo llamado `example_script.sh`.

   ```bash
   nano example_script.sh
   ```

2. **Escribir el Script**

   Aquí hay un ejemplo simple de un script que utiliza argumentos:

   ```bash
   #!/bin/bash

   # Verifica si se proporcionaron al menos dos argumentos
   if [ "$#" -lt 2 ]; then
     echo "Usage: $0 <name> <age>"
     exit 1
   fi

   # Asigna los argumentos a variables
   NAME=$1
   AGE=$2

   # Muestra un mensaje usando los argumentos
   echo "Hello, $NAME! You are $AGE years old."
   ```

   - `$0` es el nombre del script.
   - `$1`, `$2`, etc., son los argumentos proporcionados al script.

3. **Hacer el Script Ejecutable**

   Cambia los permisos del archivo para hacerlo ejecutable:

   ```bash
   chmod +x example_script.sh
   ```

4. **Ejecutar el Script con Argumentos**

   Ejecuta el script pasando dos argumentos:

   ```bash
   ./example_script.sh Alice 30
   ```

   Salida esperada:

   ```
   Hello, Alice! You are 30 years old.
   ```

### Manejo de Argumentos

- **Número de Argumentos**: Usa `$#` para obtener el número de argumentos pasados al script.

  ```bash
  echo "Number of arguments: $#"
  ```

- **Argumentos Opcionales**: Puedes proporcionar argumentos opcionales y manejar su ausencia en el script.

  ```bash
  # Verifica si un argumento opcional fue proporcionado
  if [ -z "$3" ]; then
    echo "No optional argument provided."
  else
    echo "Optional argument: $3"
  fi
  ```

- **Argumentos con Nombres**: Puedes usar `getopts` para manejar argumentos con nombres.

  ```bash
  #!/bin/bash

  while getopts "n:a:" opt; do
    case ${opt} in
      n )
        NAME=$OPTARG
        ;;
      a )
        AGE=$OPTARG
        ;;
      \? )
        echo "Invalid option: -$OPTARG" >&2
        ;;
      : )
        echo "Invalid option: -$OPTARG requires an argument" >&2
        ;;
    esac
  done

  echo "Name: $NAME"
  echo "Age: $AGE"
  ```

  Ejecuta el script usando opciones con nombres:

  ```bash
  ./example_script.sh -n Alice -a 30
  ```

### Resumen

Los scripts de Bash pueden recibir y manejar argumentos de diversas maneras. Puedes usarlos para pasar datos al script y adaptar su comportamiento según estos datos. Los ejemplos anteriores te proporcionan una base para comenzar a trabajar con argumentos en tus propios scripts.

## Sustitución de Comandos en variables

Para la sustitución de comandos es importante tener en cuenta que el resultado servirá para realizar otras tareas de otras sentencias de nuestro programa.

Las dos maneras de hacerlo:

- Usando el backtick caracter. (`)
- Usando el signo de dólar con el formato $(comando)

```bash
# ! /bin/bash
# Programa para revisar como ejecutar comados dentro de un programa y almacenar en una variable para su posterior utilización 
# Autor: Marco Toscano Freire - @martosfre

ubicacionActual=`pwd`
infoKernel=$(uname -a)

echo "La ubicación actual es la siguiente: $ubicacionActual"
echo "Información del Kernel: $infoKernel"
```
ESC  y luego `clr` + `B`y activa visual block y con ship marcamos todo lo que se desea borrar

## Debug en Script

Para realizar debugging en un script tenemos dos opciones en el comando de bash:

- **-v**: Utilizado para ver el resultado detallado de nuestro script, evaluado línea por línea. `bash -v <nombre>.sh`
- **-x**: Utilizado para desplegar la información de los comandos que son usados, capturando el comando y su salida. `bash -x <nombre>.sh`

El proceso de depuración (debugging) en scripts Bash es crucial para identificar y solucionar errores. Aquí hay varias técnicas y herramientas para depurar tus scripts:

### **1. Usar el Modo de Depuración**

Puedes ejecutar un script en modo de depuración para ver cómo se ejecutan los comandos y cómo cambian las variables.

- **Modo de Depuración Completo**: Usa `bash -x script.sh` para ejecutar el script en modo de depuración, lo que imprimirá cada comando antes de ejecutarlo.

  ```bash
  bash -x script.sh
  ```

- **Modo de Depuración Dentro del Script**: Puedes activar el modo de depuración dentro del script añadiendo `set -x` al principio del script. Para desactivar el modo de depuración, usa `set +x`.

  ```bash
  #!/bin/bash
  set -x  # Activar depuración

  echo "This is a test"
  ls -l

  set +x  # Desactivar depuración
  ```

### **2. Usar `echo` para Imprimir Variables y Mensajes**

Una técnica común es imprimir el valor de las variables y mensajes en diferentes partes del script para entender el flujo y los valores en tiempo de ejecución.

```bash
#!/bin/bash

var="Hello"
echo "The value of var is: $var"

# Más lógica
```

### **3. Verificar Errores de Comandos**

Puedes usar `$?` para verificar el estado de salida del último comando ejecutado. Un estado de salida `0` indica éxito, y cualquier otro valor indica un error.

```bash
#!/bin/bash

cp /source/file /destination/
if [ $? -ne 0 ]; then
  echo "Error: The copy command failed."
fi
```

### **4. Usar `trap` para Capturar Errores**

El comando `trap` puede ser usado para ejecutar un comando cuando el script recibe una señal o en caso de errores.

```bash
#!/bin/bash

trap 'echo "Error occurred at line $LINENO"; exit 1;' ERR

# Código que puede fallar
cp /source/file /destination/
```

### **5. Comprobar el Script Paso a Paso**

Puedes ejecutar el script paso a paso utilizando un depurador de scripts, como `bashdb`, que proporciona un entorno interactivo para depurar scripts Bash.

```bash
bashdb script.sh
```

### **6. Verificar la Sintaxis del Script**

Usa `bash -n script.sh` para verificar la sintaxis del script sin ejecutarlo. Esto te ayuda a detectar errores de sintaxis.

```bash
bash -n script.sh
```

### **7. Revisar el Log de Ejecución**

Si el script está ejecutando comandos que escriben en un archivo de log, revisa esos archivos para encontrar información sobre lo que está ocurriendo.

```bash
#!/bin/bash

echo "Starting script..." >> /var/log/myscript.log
```

### **8. Validar Entrada y Salida**

Asegúrate de que las entradas y salidas de los comandos sean las esperadas. Esto incluye verificar si los archivos o directorios existen, si tienes los permisos adecuados, etc.

```bash
#!/bin/bash

if [ ! -f /path/to/file ]; then
  echo "File not found!"
  exit 1
fi
```

Estas técnicas te ayudarán a identificar y resolver problemas en tus scripts Bash. La depuración efectiva es clave para crear scripts robustos y confiables.

## Capturar información usuario

Para poder capturar información tenemos dos formas dentro de un programa Bash.

- Utilizando en conjunto con el comando **echo**
- Utilizando directamente el comando **read**

**6_readEcho.sh**

```bash
# ! /bin/bash
# Programa para ejemplificar como capturar la información del usuario utilizando el comando echo, read y $REPLY
# # Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""

echo "Programa Utilidades Postgres"
echo -n "Ingresar una opción:"
read
option=$REPLY
echo -n "Ingresar el nombre del archivo del backup:"
read
backupName=$REPLY
echo "Opción:$option , backupName:$backupName"
```

**7_read.sh**

```bash
# ! /bin/bash
# Programa para ejemplificar como capturar la información del usuario utilizando el comando read
# # Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""

echo "Programa Utilidades Postgres"
read -p "Ingresar una opción:" option
read -p "Ingresar el nombre del archivo del backup:" backupName
echo "Opción:$option , backupName:$backupName"
```

Para capturar información del usuario en un script Bash, puedes utilizar varios métodos que permiten interactuar con el usuario y almacenar sus respuestas en variables. Aquí te muestro algunos ejemplos y técnicas comunes:

### **1. Usar `read` para Capturar Entrada**

El comando `read` permite capturar la entrada del usuario y almacenarla en una variable.

**Ejemplo Básico:**
```bash
#!/bin/bash

echo "Enter your name:"
read name
echo "Hello, $name!"
```

### **2. Leer Entrada con Mensaje**

Puedes proporcionar un mensaje que se muestra antes de que el usuario ingrese su información.

**Ejemplo:**
```bash
#!/bin/bash

read -p "Enter your age: " age
echo "You are $age years old."
```

### **3. Capturar Entrada Silenciosa**

Para capturar información de manera silenciosa (por ejemplo, contraseñas), usa la opción `-s` con `read`.

**Ejemplo:**
```bash
#!/bin/bash

read -sp "Enter your password: " password
echo
echo "Password received."
```

### **4. Leer Múltiples Valores en una Línea**

Puedes leer múltiples valores en una sola línea separándolos con espacios.

**Ejemplo:**
```bash
#!/bin/bash

read -p "Enter your first name and last name: " first_name last_name
echo "First Name: $first_name"
echo "Last Name: $last_name"
```

### **5. Proporcionar un Valor Predeterminado**

Puedes establecer un valor predeterminado en caso de que el usuario no ingrese nada.

**Ejemplo:**
```bash
#!/bin/bash

read -p "Enter your city [default: New York]: " city
city=${city:-New York}
echo "City: $city"
```

### **6. Validar la Entrada del Usuario**

Puedes validar la entrada del usuario para asegurarte de que cumpla con ciertos criterios antes de continuar.

**Ejemplo:**
```bash
#!/bin/bash

while true; do
    read -p "Enter a number between 1 and 10: " number
    if [[ "$number" =~ ^[0-9]+$ ]] && [ "$number" -ge 1 ] && [ "$number" -le 10 ]; then
        echo "Valid number: $number"
        break
    else
        echo "Invalid input. Please enter a number between 1 and 10."
    fi
done
```

### **7. Leer desde un Archivo**

Puedes leer información desde un archivo y procesarla línea por línea.

**Ejemplo:**
```bash
#!/bin/bash

while IFS= read -r line; do
    echo "Line from file: $line"
done < file.txt
```

### **8. Capturar Entrada de Usuario en un Formulario**

Puedes capturar múltiples entradas de usuario en un formulario simple, almacenarlas en variables, y luego procesarlas.

**Ejemplo:**
```bash
#!/bin/bash

echo "Please fill out the following form:"
read -p "Name: " name
read -p "Email: " email
read -p "Phone number: " phone

echo "Thank you for your submission!"
echo "Name: $name"
echo "Email: $email"
echo "Phone number: $phone"
```

Estas técnicas te permitirán interactuar con el usuario y capturar información de manera efectiva en tus scripts Bash. Puedes combinar estas técnicas según tus necesidades específicas para crear interfaces de usuario simples y funcionales en la línea de comandos.

## Expresiones Regulares

Cuando se solicita ingresar información través de un programa por parte del usuario que está utilizando el programa, independientemente el lenguaje que esté realizado; es importante considerar la validación de la información no solo en su tamaño sino también en los tipos de datos, formatos soportados lo cual nos permite asegurar la calidad de la información que recibimos, almacenamos y procesamos.

Dentro de este contexto en la programación bash para cumplir con este objetivo se utiliza expresiones regulares, las cuales son básicamente cadenas de caracteres que definen un patrón de búsqueda que se valida frente a una información específica para asegurar que cumple la validación definida.

Se necesita conocer ciertos criterios utilizados en las expresiones regulares que son los siguientes:

- ^.- Caracter que representa el inicio de la expresión regular.
- $.- Caracter que representa el final de la expresión regular.
- *.- Caracter que representa cero o más ocurrencias de la expresión
- +.- Caracter que representa una o más ocurrencias de la expresión.
- {n}.-Representa n veces de una expresión.
- [ ] .- Representa un conjunto de caracteres, por ejemplo: [a-z] representa las letras del abecedario de la a a la z.

Tomando en cuenta estos criterios se realizará un programa que valida la siguiente información:
Número de Identificación de un tamaño de 10 números. Ejemplo: 1717836520
País de Origen denotado por dos letras en un rango específico. Ejemplo: EC, CO, US
Fecha de Nacimiento en el formato yyyyMMDD. Ejemplo: 20181222

Primero se definirá las expresiones regulares y se solicitará la información del usuario:

![Captura de pantalla 2019-01-16 a la(s) 15.58.48.png](images/primera.png)

Luego se validará cada expresión regular comenzando con la identificación, para lo cual para cada validación se utilizará la sentencia condicional if y para comparar la expresión se debe utilizar el siguiente formato especial `if [[ $variable =~ $expresionRegular ]]` como se muestra a continuación.

![Captura de pantalla 2019-01-16 a la(s) 15.59.26.png](images/validacion_identidad.png)

Se realizará la ejecución de la aplicación con los dos escenarios el correcto y el incorrecto como se muestra a continuación:

![Captura de pantalla 2019-01-16 a la(s) 15.59.55.png](images/validariformacion.png)

Finalmente el código fuente lo pueden encontrar en el repositorio de GitHub en el branch 7.ValidarInformacion