# Curso de Git y GitHub

## ¿Qué son Git y GitHub?

### **Git**  
Git es un **sistema de control de versiones distribuido** que permite a los desarrolladores rastrear cambios en el código fuente, colaborar en proyectos y administrar diferentes versiones del mismo proyecto sin perder información. Fue creado por **Linus Torvalds** en 2005 para el desarrollo del kernel de Linux.  

Algunas características clave de Git:  
- Permite trabajar en **ramas (branches)** para desarrollar nuevas características sin afectar la versión principal.  
- Facilita la colaboración entre múltiples desarrolladores mediante **fusiones (merges)** y **resolución de conflictos**.  
- Almacena de manera eficiente los cambios en los archivos utilizando un modelo basado en instantáneas.  
- Funciona de manera **descentralizada**, lo que significa que cada copia de un repositorio es independiente.  

### **GitHub**  
GitHub es una **plataforma de alojamiento de repositorios Git** basada en la nube que permite almacenar, compartir y colaborar en proyectos de software. Aunque GitHub usa Git como tecnología base, agrega funcionalidades adicionales como:  
- **Interfaz web** para administrar repositorios sin usar la línea de comandos.  
- **GitHub Actions** para automatizar pruebas e implementaciones.  
- **Issues y Pull Requests**, herramientas para gestionar cambios y reportar problemas.  
- **GitHub Pages**, que permite alojar sitios web estáticos desde un repositorio.  

Existen otras plataformas similares a GitHub que también usan Git, como **GitLab, Bitbucket y SourceForge**.  

Si quieres empezar con Git, puedes instalarlo desde [git-scm.com](https://git-scm.com/) y aprender a usar comandos básicos como:  
```bash
git init       # Inicializa un repositorio Git en una carpeta
git clone URL  # Clona un repositorio remoto
git add .      # Agrega cambios al área de preparación
git commit -m "Mensaje"  # Guarda los cambios con un mensaje
git push origin main  # Sube los cambios al repositorio remoto
```  

**Resumen**
Aprender a gestionar versiones en proyectos de software es fundamental para evitar el caos de múltiples archivos llamados “versión final” y mejorar la colaboración en equipo. Git, un sistema de control de versiones, permite a los desarrolladores trabajar de manera ordenada, manteniendo solo los cambios realizados en los archivos y simplificando el trabajo en equipo al coordinar y sincronizar las modificaciones.

**¿Qué es Git y por qué debería importarte?**

Git es la herramienta de control de versiones más utilizada por programadores. Su función es clara: gestiona versiones de archivos de forma eficaz, algo vital en proyectos colaborativos. Sin Git, los desarrolladores enfrentaban problemas de organización y errores en la sincronización manual de archivos, un proceso que era tan lento como propenso a fallos.

**¿Quién creó Git y por qué es tan relevante?**

El creador de Git es Linus Torvalds, el mismo desarrollador detrás del núcleo de Linux, quien creó esta herramienta para resolver sus propias necesidades de control de versiones. Además, Git es open source, lo que permite a cualquier persona contribuir a su mejora constante, garantizando que siga siendo una herramienta poderosa y en evolución.

**¿Cómo te beneficia aprender Git desde el inicio?**

Desde que comienzas a programar, aprender Git te brinda una ventaja profesional. Esta herramienta te permitirá organizar tu código, colaborar con otros desarrolladores y, eventualmente, contribuir a proyectos externos. En el desarrollo de software, los productos suelen ser el resultado del esfuerzo de un equipo, y Git se vuelve esencial para gestionar el trabajo de manera efectiva.

**¿Cómo funciona Git en tu entorno de trabajo?**

Git se usa en la terminal o en editores como Visual Studio Code, utilizando comandos clave como `commit`, `pull`, `push`, `merge`, entre otros. Para colaborar en proyectos con otros desarrolladores, se usa una plataforma en la nube como GitHub. Allí puedes almacenar tu código, gestionar versiones y acceder a funcionalidades adicionales que aumentan la productividad y facilitan la colaboración.

**¿Qué aprenderás en este curso de Git?**

En este curso, aprenderás a:

- Configurar Git en tu computadora y crear repositorios locales.
- Realizar cambios, crear ramas, fusionarlas y gestionar el flujo de trabajo en equipo.
- Configurar un repositorio remoto en GitHub para colaborar en proyectos.
- Revisar y fusionar aportes de otros programadores, resolver conflictos y entender un flujo de trabajo profesional.

Al final, dominarás tanto los comandos básicos como las herramientas avanzadas, desde ramas y fusiones hasta Code Spaces y automatizaciones, para que realmente te destaques en el uso de Git.

**Lecturas recomendadas**

[Git](https://git-scm.com/)

[GitHub · Build and ship software on a single, collaborative platform · GitHub](https://github.com/)

## Configuración Inicial de Git: init y config

Para comenzar a usar Git en tu máquina, es importante hacer una configuración inicial después de instalarlo. Esto implica inicializar un repositorio (`git init`) y establecer configuraciones globales (`git config`).

## **1. Inicializar un Repositorio: `git init`**
El comando `git init` se usa para convertir una carpeta en un repositorio de Git.  

### **Pasos:**
1. Abre una terminal y navega hasta la carpeta donde quieres iniciar el repositorio.
2. Ejecuta:  
   ```bash
   git init
   ```
3. Esto creará una carpeta oculta llamada `.git` que almacenará toda la información del repositorio.

🔹 **Ejemplo:**
```bash
mkdir mi_proyecto
cd mi_proyecto
git init
```
Después de esto, `mi_proyecto` se convierte en un repositorio Git.

## **2. Configuración Global de Git: `git config`**
Antes de comenzar a hacer commits, Git necesita conocer tu identidad para etiquetar correctamente los cambios.

### **Configurar nombre y correo electrónico**  
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tuemail@example.com"
```
Estos datos se guardan en el archivo `~/.gitconfig`.

### **Ver la configuración actual**  
Para verificar tu configuración, usa:
```bash
git config --list
```

### **Configurar el editor de texto predeterminado**  
Si quieres usar un editor específico (ej. `nano` o `vim`), ejecuta:
```bash
git config --global core.editor "nano"
```

### **Configurar el comportamiento de las ramas principales**  
A partir de Git 2.28, puedes definir el nombre predeterminado de la rama inicial:
```bash
git config --global init.defaultBranch main
```
Esto hará que, al ejecutar `git init`, la rama principal se llame `main` en lugar de `master`.

## **3. Configuración Local vs. Global**
- **Global (`--global`)**: Aplica la configuración a todos los repositorios de tu usuario.
- **Local (sin `--global`)**: Aplica la configuración solo al repositorio actual.

Ejemplo de configuración local:
```bash
git config user.name "Otro Nombre"
```

## **Resumen**
| Comando | Descripción |
|---------|------------|
| `git init` | Inicializa un repositorio Git en la carpeta actual |
| `git config --global user.name "Tu Nombre"` | Configura el nombre del usuario a nivel global |
| `git config --global user.email "tuemail@example.com"` | Configura el email del usuario a nivel global |
| `git config --list` | Muestra la configuración actual de Git |
| `git config --global core.editor "nano"` | Establece el editor de texto predeterminado |
| `git config --global init.defaultBranch main` | Define el nombre de la rama principal como `main` |

**Resumen**

Trabajar con Git en la terminal permite a los desarrolladores gestionar sus proyectos de manera eficiente. A continuación, revisamos cómo instalar, configurar y utilizar Git en Linux, Mac y WSL de Windows, junto con algunas recomendaciones prácticas para dominar los comandos iniciales de esta herramienta.

### ¿Cómo confirmar que Git está instalado en tu sistema?

Para verificar la instalación de Git:

1. Abre la terminal y escribe el comando `git --version`.
2. Si el comando devuelve un número de versión, Git está listo para usarse.
3. Si no aparece la versión, revisa los recursos adjuntos donde se explican las instalaciones para cada sistema operativo.

### ¿Cómo crear y preparar el primer proyecto con Git?

El primer paso para crear un proyecto en Git es:

1. Limpia la terminal para evitar confusión visual.
2. Crea una carpeta para el proyecto con `mkdir nombre_del_proyecto`.
3. Navega a la carpeta con `cd nombre_del_proyecto`.

### ¿Cómo inicializar un repositorio en Git?

Al estar dentro de la carpeta de tu proyecto, inicia el repositorio con:

- `git init`: Esto crea la rama inicial “master” por defecto.

Si prefieres la rama principal como “main”:

1. Cambia la configuración global escribiendo `git config --global init.defaultBranch main`.
2. Actualiza la rama en el proyecto actual con `git branch -m main`.

### ¿Cómo personalizar tu configuración de usuario en Git?

Configura el nombre de usuario y correo electrónico de Git, que identificará todas tus contribuciones:

1. Usa `git config --global user.name "Tu Nombre o Apodo"`.
2. Configura el correo electrónico con `git config --global user.email "tu.email@example.com"`.

**Tip**: Si necesitas corregir algún error en el comando, puedes usar la tecla de flecha hacia arriba para recuperar y editar el último comando escrito.

¿Cómo confirmar la configuración de Git?
Para revisar tu configuración, ejecuta:

- `git config --list`: Aquí verás los datos de usuario y el nombre de la rama principal.

Esta configuración se aplicará a todos los repositorios que crees en adelante.

### ¿Qué hacer si olvidas un comando?

Git incluye un recurso rápido y útil para recordar la sintaxis de comandos:

1. Escribe `git help` en la terminal.
2. Navega la lista de comandos disponibles y consulta la documentación oficial de cada uno cuando sea necesario.

**Lecturas recomendadas**

[Git](https://git-scm.com/)

[Git - git-init Documentation](https://git-scm.com/docs/git-init)

[Git Cheat Sheet - GitHub Education](https://education.github.com/git-cheat-sheet-education.pdf)

[Git - git-config Documentation](https://git-scm.com/docs/git-config)

[Git - Configurando Git por primera vez](https://git-scm.com/book/es/v2/Inicio---Sobre-el-Control-de-Versiones-Configurando-Git-por-primera-vez)

[Configurar Windows para WSL - Platzi](https://platzi.com/home/clases/6900-configuracion-windows/60922-configurar-windows-11-para-soportar-la-instalacion/)

[GitHub · Build and ship software on a single, collaborative platform · GitHub](https://github.com/)

## Comandos Básicos de Git: add, commit y log

## **Comandos Básicos de Git: `add`, `commit` y `log`**  

Una vez que tienes Git configurado e inicializado, puedes comenzar a gestionar cambios en tu código. Tres de los comandos más importantes son `git add`, `git commit` y `git log`.

---

## **1. Agregar Archivos al Área de Preparación: `git add`**  
Antes de confirmar los cambios en el historial de Git, debes agregarlos al **staging area** (área de preparación).

### **Sintaxis:**
```bash
git add <archivo>       # Agrega un archivo específico
git add .               # Agrega todos los archivos modificados
git add *.txt           # Agrega solo archivos con cierta extensión
```

🔹 **Ejemplo:**  
Si creas un archivo `index.html` y lo modificas, puedes agregarlo al área de preparación con:
```bash
git add index.html
```
Esto indica a Git que este archivo será parte del próximo commit.

---

## **2. Confirmar Cambios: `git commit`**  
El comando `git commit` guarda una **instantánea permanente** de los archivos en el historial de versiones.

### **Sintaxis:**
```bash
git commit -m "Mensaje descriptivo"
```
🔹 **Ejemplo:**  
```bash
git commit -m "Agregada la estructura inicial del proyecto"
```
Después de este comando, los cambios se guardan en el historial de Git, pero aún no se han enviado a un repositorio remoto.

**Opción avanzada:** Si quieres escribir un mensaje más detallado en varias líneas, usa:
```bash
git commit
```
Esto abrirá el editor de texto configurado (como `nano` o `vim`), donde puedes escribir una descripción más extensa del commit.

---

## **3. Ver el Historial de Commits: `git log`**  
Para ver el historial de confirmaciones en el repositorio, usa `git log`.

### **Sintaxis básica:**
```bash
git log
```
🔹 **Ejemplo de salida:**
```
commit 1a2b3c4d5e6f7g8h9i (HEAD -> main)
Author: Mario Alexander Vargas Celis <mario@example.com>
Date:   Wed Jan 30 12:00:00 2025 -0500

    Agregada la estructura inicial del proyecto
```

### **Opciones útiles:**
- **Mostrar commits en una línea resumida:**
  ```bash
  git log --oneline
  ```
  🔹 **Ejemplo de salida:**  
  ```
  1a2b3c4 Agregada la estructura inicial del proyecto
  ```

- **Ver cambios en cada commit:**  
  ```bash
  git log -p
  ```
- **Filtrar por autor:**  
  ```bash
  git log --author="Mario"
  ```
- **Ver commits de los últimos 7 días:**  
  ```bash
  git log --since="7 days ago"
  ```

---

## **Flujo de Trabajo Básico con Git**
1️⃣ **Crear o modificar archivos:**  
   ```bash
   echo "Hola Mundo" > archivo.txt
   ```

2️⃣ **Agregar cambios al área de preparación:**  
   ```bash
   git add archivo.txt
   ```

3️⃣ **Confirmar los cambios con un mensaje:**  
   ```bash
   git commit -m "Primer archivo agregado"
   ```

4️⃣ **Ver el historial de commits:**  
   ```bash
   git log --oneline
   ```

Con estos comandos, ya puedes comenzar a gestionar versiones en Git.

## Resumen

Aprender a utilizar Git desde los primeros pasos puede parecer desafiante, pero es esencial para registrar cambios y manejar versiones de cualquier proyecto. Siguiendo un flujo de trabajo sencillo y utilizando los comandos adecuados, puedes dominar el control de versiones y llevar un seguimiento preciso de tus archivos.

### ¿Cómo inicia el control de versiones con Git?

El primer paso es iniciar un repositorio con el comando `git init`, que crea una carpeta oculta llamada `.git` en el directorio de trabajo. Esta carpeta actúa como una bitácora, almacenando cada cambio y movimiento de los archivos que se manejan en el proyecto.

### ¿Cómo se crean y agregan archivos a Git?

Para crear un archivo desde la terminal, utiliza un editor como `nano`. Una vez creado, puedes verificar su existencia y estado con `git status`, que te mostrará el archivo como no registrado. Para incluirlo en el área de staging, donde estará listo para el commit, usa `git add nombre_del_archivo.txt`. Esta área de staging es un “limbo” donde decides qué archivos entrarán en el control de versiones.

- **Ejemplo de comandos:**
 - nano testing.txt para crear el archivo.
 - git add testing.txt para agregarlo al área de staging.
 
### ¿Qué es el área de staging y cómo funciona?

El área de staging permite revisar los cambios antes de que se registren oficialmente en el repositorio. Los archivos en staging aún no forman parte del historial de versiones; están en espera de que se realice un commit o de ser devueltos a su estado original con `git rm --cached nombre_del_archivo.txt`.

### ¿Cómo realizar el commit de los archivos en Git?

Una vez en staging, se ejecuta git commit -m "mensaje descriptivo" para registrar los cambios en el repositorio. El mensaje en el commit es crucial porque indica la acción realizada, como “nuevo archivo de testing”. Este mensaje permite identificar los cambios de forma clara y ordenada en el historial del proyecto.

- **Ejemplo de commit:**
 - `git commit -m "nuevo archivo de testing"`
 
### ¿Cómo gestionar múltiples archivos en Git?

Para trabajar con varios archivos a la vez, utiliza `git add .` que agrega todos los archivos sin registrar en el área de staging. Puedes decidir entre realizar commits individuales o múltiples en función de la cantidad de archivos y los cambios realizados en cada uno.

### ¿Cómo visualizar el historial de cambios en Git?

El comando `git log` muestra el historial de commits, proporcionando una vista completa de cada cambio realizado en el proyecto. Esta bitácora permite ver el estado de cada archivo y la información de cada commit.

### ¿Qué sucede al modificar un archivo en Git?

Cuando un archivo se edita, Git lo detecta como “modificado”. El flujo de trabajo para registrar este cambio es el mismo que para un archivo nuevo: `git add` para llevarlo a staging y `git commit` para guardar la modificación. Esto asegura que Git mantenga un registro detallado de cada cambio, actualización o eliminación en el proyecto.

### ¿Cómo maneja Git diferentes tipos de archivos?

Git trata cualquier archivo de igual manera, sin importar su extensión o tipo, ya sea de texto, código o imagen. Con `git add` y `git commit`, cualquier cambio en estos archivos se registra, facilitando el control de versiones sin importar el tipo de contenido.

![comandos Basicos de git](images/comandosBasicosdegit.png)

**Terminos basicos**

- cd → cambiar directorio y/o regresar al directorio raiz
- cd .. → retroceder 1 carpeta dentro del directorio
- mkdir → crear directorio
- rmdir → remover directorio
- ls → contenido de un directorio
- .. → volver 1 carpeta atrás
- mkdir repo → crear repo
- rmdir repo → eliminar repo
- git init → iniciar repositorio
- git add → añadir archivos
- git status → estado del repo
- git rm —cached → eliminar archivo añadido al repositorio
- git commit → subir todo al repositorio

**Lecturas recomendadas**

[Git - git-add Documentation](https://git-scm.com/docs/git-add)

[Git - git-commit Documentation](https://git-scm.com/docs/git-commit)

[Git - git-log Documentation](https://git-scm.com/docs/git-log)

[Póngase en marcha - Documentación de GitHub](https://docs.github.com/es/get-started/start-your-journey)

## Ramas y Fusión de Cambios: branch, merge, switch y checkout

### **Ramas y Fusión de Cambios en Git**  
Las **ramas** en Git permiten trabajar en diferentes versiones de un proyecto sin afectar la rama principal. Esto es útil para desarrollar nuevas funcionalidades o corregir errores sin modificar el código estable.  

Los comandos más importantes para manejar ramas son:  
- `git branch` → Crear y listar ramas.  
- `git switch` y `git checkout` → Cambiar entre ramas.  
- `git merge` → Fusionar cambios entre ramas.  

### **1. Listar y Crear Ramas: `git branch`**  
### **Ver ramas existentes:**  
```bash
git branch
```
🔹 **Ejemplo de salida:**
```
* main
  nueva_funcionalidad
```
El asterisco (*) indica la rama en la que estás trabajando.

### **Crear una nueva rama:**  
```bash
git branch nombre_rama
```
🔹 **Ejemplo:**  
```bash
git branch nueva_funcionalidad
```
Esto crea la rama `nueva_funcionalidad`, pero **no cambia a ella**.

### **2. Cambiar de Rama: `git switch` y `git checkout`**  
Para cambiar de rama, puedes usar:  

### **Usando `git switch` (Recomendado desde Git 2.23)**  
```bash
git switch nombre_rama
```
🔹 **Ejemplo:**  
```bash
git switch nueva_funcionalidad
```

### **Usando `git checkout` (Método antiguo, aún válido)**  
```bash
git checkout nombre_rama
```

### **Crear y cambiar a una nueva rama en un solo paso:**  
```bash
git switch -c nueva_rama
```
O con `checkout` (versión antigua):  
```bash
git checkout -b nueva_rama
```

### **3. Fusionar Cambios entre Ramas: `git merge`**  
Cuando terminas de trabajar en una rama, puedes fusionar sus cambios en la rama principal.

### **Pasos para fusionar ramas:**  
1️⃣ Cambiar a la rama donde se quiere fusionar (por ejemplo, `main`):  
   ```bash
   git switch main
   ```

2️⃣ Ejecutar el merge:  
   ```bash
   git merge nueva_funcionalidad
   ```

🔹 **Ejemplo:**  
Si trabajaste en `nueva_funcionalidad` y quieres fusionarla en `main`:  
```bash
git switch main
git merge nueva_funcionalidad
```

### **Posibles resultados al hacer `merge`:**
✅ **Fusión rápida (`Fast-forward`)**  
Si no hubo otros cambios en `main`, Git moverá directamente la referencia:  
```
  main --> nueva_funcionalidad
```
✅ **Fusión con commit de merge**  
Si hay cambios en ambas ramas, Git creará un **nuevo commit de merge**.  

⚠️ **Si hay conflictos, Git pedirá resolverlos manualmente.**  
Para ver los archivos en conflicto:  
```bash
git status
```
Después de resolverlos, hacer:
```bash
git add archivo_con_conflicto
git commit -m "Resuelto conflicto en archivo.txt"
```

### **Resumen de Comandos**
| Comando | Descripción |
|---------|------------|
| `git branch` | Muestra todas las ramas |
| `git branch nueva_rama` | Crea una nueva rama |
| `git switch nueva_rama` | Cambia a otra rama (nuevo método) |
| `git checkout nueva_rama` | Cambia a otra rama (método antiguo) |
| `git switch -c nueva_rama` | Crea y cambia a una nueva rama |
| `git checkout -b nueva_rama` | Crea y cambia a una nueva rama (método antiguo) |
| `git merge nombre_rama` | Fusiona una rama en la actual |
| `git status` | Muestra el estado del repositorio |

---

### 🚀 **Flujo de Trabajo con Ramas**
1️⃣ Crear una nueva rama:  
   ```bash
   git branch feature_x
   ```
2️⃣ Cambiar a la nueva rama:  
   ```bash
   git switch feature_x
   ```
3️⃣ Hacer cambios y confirmarlos:  
   ```bash
   git add .
   git commit -m "Agregada nueva funcionalidad"
   ```
4️⃣ Volver a `main` y fusionar los cambios:  
   ```bash
   git switch main
   git merge feature_x
   ```

Con este flujo, puedes trabajar en diferentes funcionalidades sin afectar la rama principal. 🎯  
¿Quieres aprender a **eliminar ramas** o **resolver conflictos** en Git?

### Resumen
El uso de ramas en Git permite trabajar en un entorno aislado sin interferir con otros, facilitando la organización y el control del proyecto. Aprender a crear, gestionar y fusionar ramas optimiza la colaboración y ayuda a mantener la limpieza en el historial de cambios.

### ¿Por qué son útiles las ramas en Git?

Las ramas son una herramienta que permite trabajar en tareas específicas sin alterar la rama principal. Entre sus ventajas se encuentran:

- Aislamiento de cambios individuales.
- Posibilidad de desechar una rama sin afectar la principal.
- Organización de actividades múltiples en diferentes ramas.

### ¿Cómo verificar la rama actual?

Para saber en qué rama estás trabajando, ejecuta:

`git branch`

El asterisco (`*`) indica la rama activa. Inicialmente, suele ser `main`, pero al crear más ramas, la lista crecerá, permitiéndote ver todas las disponibles y cuál es la actual.

### ¿Cómo crear una nueva rama en Git?

La creación de ramas permite desarrollar sin riesgo en paralelo. Para crear y moverte a una nueva rama, usa:

`git checkout -b` 

Por ejemplo, `git checkout -b Amin` crea y mueve a la rama `Amin`. Puedes verificar que estás en esta rama ejecutando git branch.

### ¿Cómo agregar y confirmar cambios en una rama?

Dentro de una nueva rama, los archivos se editan y confirman sin que impacten otras ramas. Sigue estos pasos para agregar y confirmar:

1. Crea o edita un archivo.
2. Añádelo con:

`git add .`

3. Confirma el cambio:

`git commit -m "mensaje de confirmación"`

Los cambios ahora son parte de la rama en la que trabajas y no afectan la principal.

### ¿Cómo fusionar cambios de una rama secundaria a la principal?

Para unificar el trabajo en la rama principal:

1. Cambia a la rama principal:

`git switch main`

**Nota**: Puedes usar también git checkout main.

2. Fusiona la rama secundaria:

`git merge`

Git indicará que el proceso fue exitoso y actualizará el contenido en la rama `main` con los cambios de la rama secundaria.

### ¿Por qué es importante eliminar ramas que ya no se usan?

Una vez fusionada una rama, es buena práctica eliminarla para evitar desorden. Hazlo con:

`git branch -d`

Eliminar ramas que ya cumplieron su propósito previene conflictos y mantiene el entorno de trabajo limpio y organizado.

- git reset: Este comando devuelve a un commit anterior, eliminando los cambios en el historial como si nunca hubieran ocurrido.
- Permite deshacer cambios y mover el puntero HEAD a un commit específico. Hay tres modos principales:
- git reset --soft: Mueve HEAD al commit especificado, pero mantiene los cambios en el área de preparación.
- git reset --mixed: (Por defecto) Mueve HEAD y deshace los cambios en el área de preparación, pero mantiene los cambios en el directorio de trabajo.
- git reset --hard: Mueve HEAD y descarta todos los cambios, tanto en el área de preparación como en el directorio de trabajo.
- git revert: Crea un nuevo commit que deshace los cambios de un commit específico. Es útil para deshacer cambios de forma segura en repositorios compartidos.

Estos comandos son útiles para corregir errores o volver a estados anteriores del proyecto de manera controlada, limpieza de historial y manejo de conflictos.

nano error.txt clear ls git add . git commit -m "nuevo archivo especial creado" git log clear

**git revert**

git revert"hash commit"

**Crea un nuevo commit que deshace los cambios del último commit**

"Revert "nuevo archivo especial creado" por "autor revert""

git log clear ls

nano reset.txt git add . git commit -m "nuevo archivo para reiniciar" git log clear ls

**git reset**

git reset --hard "hash"

**Lecturas recomendadas**

[Git - git-branch Documentation](https://git-scm.com/docs/git-branch)

[Git - git-merge Documentation](https://git-scm.com/docs/git-merge)

[Git - git-switch Documentation](https://git-scm.com/docs/git-switch)

[Git - git-checkout Documentation](https://git-scm.com/docs/git-checkout)

## Volviendo en el Tiempo en Git: reset y revert

En Git, puedes **deshacer cambios** y regresar a estados anteriores usando los comandos `reset` y `revert`. Sin embargo, tienen diferencias clave:

| Comando | Descripción | Afecta historial? | Se recomienda en remoto? |
|---------|------------|-------------------|--------------------------|
| `git reset` | Mueve la referencia del commit actual a otro punto, eliminando o manteniendo cambios en `working directory`. | ❌ Sí, reescribe historial. | 🚫 No recomendado. |
| `git revert` | Crea un nuevo commit que revierte los cambios de un commit anterior. | ✅ No reescribe historial. | ✅ Seguro para repositorios remotos. |

### **1. Deshacer Commits con `git reset`**  

El comando `git reset` mueve la referencia de la rama a un commit anterior. Puede afectar los cambios en **tres niveles** según la opción que elijas:  

### **Modos de `git reset`:**
1️⃣ **`--soft`**: Mantiene los cambios en el área de preparación (staging).  
2️⃣ **`--mixed` (por defecto)**: Mantiene los cambios en el directorio de trabajo pero los saca del área de preparación.  
3️⃣ **`--hard`**: **Elimina completamente** los cambios, sin posibilidad de recuperarlos.

### **Ejemplos:**
- **Volver al commit anterior pero mantener los cambios en staging (`--soft`)**  
  ```bash
  git reset --soft HEAD~1
  ```
  🔹 Esto mueve la rama un commit atrás, pero los cambios siguen en el área de preparación.  

- **Volver al commit anterior y sacar los cambios de staging (`--mixed`, por defecto)**  
  ```bash
  git reset HEAD~1
  ```
  🔹 La rama retrocede, y los cambios quedan en el directorio de trabajo (sin agregar).  

- **Eliminar completamente el último commit y los cambios (`--hard`)**  
  ```bash
  git reset --hard HEAD~1
  ```
  ⚠️ **¡Cuidado! Esto borra los cambios sin opción de recuperación.**  

### **Volver a un commit específico:**
Si quieres regresar a un commit en particular, usa su **hash**:
```bash
git reset --hard <ID_DEL_COMMIT>
```
Para ver los commits anteriores y obtener el hash:
```bash
git log --oneline
```

---

### **2. Deshacer Cambios con `git revert` (Recomendado para repositorios remotos)**  
El comando `git revert` crea un **nuevo commit** que deshace los cambios de un commit específico, sin eliminar el historial.  

🔹 **Ejemplo:**  
```bash
git revert HEAD
```
Esto deshace el último commit y crea un nuevo commit con la reversión.

### **Revertir un commit específico:**
```bash
git revert <ID_DEL_COMMIT>
```
Esto aplicará los cambios inversos de ese commit en la rama actual.

Si quieres revertir varios commits:  
```bash
git revert HEAD~2..HEAD
```
Este comando revierte los últimos **dos commits**.

---

### **3. Comparación entre `reset` y `revert`**
| Acción | `git reset` | `git revert` |
|--------|------------|-------------|
| Deshace commits | ✅ Sí | ✅ Sí |
| Mantiene historial | ❌ No (lo reescribe) | ✅ Sí (agrega un nuevo commit) |
| Seguro para repositorios remotos | 🚫 No | ✅ Sí |
| Permite eliminar cambios en archivos | ✅ Sí (con `--hard`) | ❌ No |

---

### **Casos de Uso**
1️⃣ **Si ya subiste un commit a un repositorio remoto y quieres deshacerlo:**  
   → Usa `git revert` para evitar problemas con otros colaboradores.  
   ```bash
   git revert HEAD
   git push origin main
   ```

2️⃣ **Si hiciste un commit por error y aún no lo subiste a GitHub:**  
   → Usa `git reset` para deshacerlo.  
   ```bash
   git reset --soft HEAD~1
   ```

3️⃣ **Si quieres descartar completamente los últimos cambios:**  
   → Usa `git reset --hard`.  
   ```bash
   git reset --hard HEAD~1
   ```

---

### **Resumen de Comandos**
| Comando | Acción |
|---------|--------|
| `git reset --soft HEAD~1` | Mueve el commit atrás, pero mantiene los cambios en staging. |
| `git reset --mixed HEAD~1` | Mueve el commit atrás y deja los cambios en el directorio de trabajo. |
| `git reset --hard HEAD~1` | Borra el último commit y los cambios (¡Irreversible!). |
| `git revert HEAD` | Crea un nuevo commit que revierte el último commit. |
| `git revert <ID_DEL_COMMIT>` | Revierte un commit específico sin modificar el historial. |

### Resumen

Para quienes se inician en el manejo de versiones con Git, comandos como `git reset` y `git revert` se vuelven herramientas indispensables, ya que permiten deshacer errores y ajustar el historial de cambios sin complicaciones. Aunque al avanzar en la experiencia puedan dejarse de lado, dominar su uso resulta clave para un control de versiones eficiente.

### ¿Cuál es la diferencia entre Git Reset y Git Revert?

- **Git Reset:** mueve el puntero de los commits a uno anterior, permitiendo “volver en el tiempo” y explorar el historial de cambios. Es útil para deshacer actualizaciones recientes o revisar lo que se hizo en cada commit.
- **Git Revert**: crea un nuevo commit que revierte los cambios de un commit específico, permitiendo conservar el historial original sin eliminaciones. Es ideal para regresar a un estado anterior sin afectar los commits de otros usuarios.

### ¿Cómo se utiliza Git Reset?

1. Ejecuta git log para identificar el historial de commits. El commit actual se marca con `HEAD` apuntando a `main`.
2. Si quieres eliminar cambios recientes:
 - Crea un archivo temporal (ejemplo: `error.txt`) y realiza un commit.
 - Verifica el historial con git log y localiza el hash del commit que deseas restablecer.
 
3. Para revertir a un estado anterior:
- Usa git reset con parámetros:
 - --soft: solo elimina el archivo del área de staging.
 - --mixed: remueve los archivos de staging, manteniendo el historial de commits.
 - --hard: elimina los archivos y el historial hasta el commit seleccionado.
- Este último parámetro debe ser una última opción debido a su impacto irreversible en el historial.

### ¿Cómo funciona Git Revert?

Identificación del commit: usa git log para encontrar el commit a revertir.
Ejecuta git revert seguido del hash del commit: crea un nuevo commit inverso, preservando el historial.
Editar el mensaje de commit: permite dejar claro el motivo de la reversión, ideal en equipos colaborativos para mantener claridad.

### ¿Cuándo es recomendable utilizar Git Reset o Git Revert?

Ambos comandos resultan útiles en diversas situaciones:

- **Corrección de errores**: si has subido un archivo incorrecto, git revert es rápido y seguro para deshacer el cambio sin afectar el historial.
- **Limpieza del historial**: en proyectos sólidos, puede que quieras simplificar el historial de commits; git reset ayuda a limpiar entradas innecesarias.
- **Manejo de conflictos**: en casos extremos de conflicto de archivos, git reset es útil, aunque puede ser mejor optar por resolver conflictos manualmente.

### ¿Cómo aseguras una correcta comunicación en el uso de estos comandos?

- Utiliza estos comandos en sincronización con el equipo.
- Evita el uso de git reset --hard sin coordinación para prevenir la pérdida de trabajo ajeno.
- Documenta cada reversión con un mensaje claro para asegurar el seguimiento de cambios.

- git reset: Este comando devuelve a un commit anterior, eliminando los cambios en el historial como si nunca hubieran ocurrido.
- Permite deshacer cambios y mover el puntero HEAD a un commit específico. Hay tres modos principales:
- git reset --soft: Mueve HEAD al commit especificado, pero mantiene los cambios en el área de preparación.
- git reset --mixed: (Por defecto) Mueve HEAD y deshace los cambios en el área de preparación, pero mantiene los cambios en el directorio de trabajo.
- git reset --hard: Mueve HEAD y descarta todos los cambios, tanto en el área de preparación como en el directorio de trabajo.
- git revert: Crea un nuevo commit que deshace los cambios de un commit específico. Es útil para deshacer cambios de forma segura en repositorios compartidos.

Estos comandos son útiles para corregir errores o volver a estados anteriores del proyecto de manera controlada, limpieza de historial y manejo de conflictos.

nano error.txt clear ls git add . git commit -m "nuevo archivo especial creado" git log clear

**git revert**

git revert"hash commit"

Crea un nuevo commit que deshace los cambios del último commit
"Revert "nuevo archivo especial creado" por "autor revert""

git log clear ls

nano reset.txt git add . git commit -m "nuevo archivo para reiniciar" git log clear ls

**git reset**

git reset --hard "hash"

**Lecturas recomendadas**

[Git - git-reset Documentation](https://git-scm.com/docs/git-reset)

[Git - git-revert Documentation](https://git-scm.com/docs/git-revert)

## Gestión de versiones: tag y checkout

En Git, los **tags** (etiquetas) se usan para marcar versiones específicas del código, por ejemplo, cuando se lanza una nueva versión de un software (`v1.0`, `v2.0.1`). Además, puedes utilizar `checkout` (o `switch` en versiones recientes de Git) para navegar entre diferentes versiones del código.

### **1. Crear y Listar Etiquetas (`git tag`)**  

Las etiquetas son snapshots (instantáneas) de un commit específico y se dividen en dos tipos:  
- **Anotadas** (`-a`): Guardan información adicional como autor, fecha y mensaje.  
- **Ligeras** (Lightweight): Son solo un alias del commit, sin información extra.

### **Listar todas las etiquetas disponibles:**  
```bash
git tag
```
🔹 **Ejemplo de salida:**  
```
v1.0
v1.1
v2.0-beta
```

### **Crear una Etiqueta Ligera**
```bash
git tag v1.0
```
Esto etiqueta el commit actual con `v1.0`, pero sin información adicional.

### **Crear una Etiqueta Anotada**
```bash
git tag -a v1.0 -m "Versión estable 1.0"
```
🔹 Esto crea una etiqueta con un mensaje y metadatos.

### **Etiquetar un Commit Anterior**  
Si necesitas etiquetar un commit específico, usa su hash:
```bash
git tag -a v1.1 123abc -m "Versión 1.1 con correcciones"
```
(El `123abc` es el ID del commit, obtenido con `git log --oneline`).

### **2. Compartir Etiquetas en un Repositorio Remoto**  

Las etiquetas **no** se suben automáticamente a GitHub. Para enviarlas, usa:
```bash
git push origin v1.0
```
Si quieres subir **todas las etiquetas** de una vez:
```bash
git push --tags
```

### **3. Eliminar Etiquetas**
- **Eliminar una etiqueta localmente:**
  ```bash
  git tag -d v1.0
  ```
- **Eliminar una etiqueta en el repositorio remoto:**
  ```bash
  git push --delete origin v1.0
  ```

### **4. Cambiar a una Versión Etiquetada (`git checkout`)**  
Si quieres ver el código de una versión específica, puedes "viajar en el tiempo" con:

```bash
git checkout v1.0
```
🔹 Esto coloca el código en un estado de solo lectura (`HEAD detached`). Para volver a la rama principal:  
```bash
git switch main
```

### **5. Crear una Rama desde una Etiqueta**
Si necesitas hacer cambios en una versión etiquetada:
```bash
git checkout -b fix-v1.0 v1.0
```
Esto crea una rama `fix-v1.0` basada en la versión `v1.0`.

### **Resumen de Comandos**
| Comando | Acción |
|---------|--------|
| `git tag` | Lista todas las etiquetas. |
| `git tag v1.0` | Crea una etiqueta ligera. |
| `git tag -a v1.0 -m "Mensaje"` | Crea una etiqueta anotada. |
| `git tag -a v1.1 <commit_id> -m "Mensaje"` | Etiqueta un commit específico. |
| `git push --tags` | Envía todas las etiquetas al repositorio remoto. |
| `git tag -d v1.0` | Elimina una etiqueta localmente. |
| `git push --delete origin v1.0` | Elimina una etiqueta en GitHub. |
| `git checkout v1.0` | Cambia a una versión específica. |
| `git checkout -b rama_nueva v1.0` | Crea una nueva rama desde una etiqueta. |

¿Quieres aprender más sobre versionado semántico (`v1.0.0`, `v2.1.3`)?

### **Versionado Semántico en Git (`vX.Y.Z`)**  

El **Versionado Semántico (SemVer)** es una convención usada en software para nombrar versiones de manera clara y predecible. Se usa el formato:  

```
MAJOR.MINOR.PATCH
```
Ejemplo: **`v2.1.3`**  
- **MAJOR (`2`)** → Cambios incompatibles o grandes reestructuraciones.  
- **MINOR (`1`)** → Nuevas funcionalidades sin romper compatibilidad.  
- **PATCH (`3`)** → Correcciones de errores sin agregar nuevas funciones.  

### **1. Ejemplo de Uso en Git**  
### **Crear una Etiqueta con Versionado Semántico**
```bash
git tag -a v1.0.0 -m "Primera versión estable"
```

### **Lanzar una Nueva Versión con Cambios Menores**
```bash
git tag -a v1.1.0 -m "Agregada nueva funcionalidad X"
```

### **Lanzar un Parche para una Corrección de Bug**
```bash
git tag -a v1.1.1 -m "Corrección de bug en la funcionalidad X"
```

### **2. Comparar Versiones**
Puedes comparar dos versiones para ver qué cambió entre ellas:
```bash
git diff v1.0.0 v1.1.0
```
También puedes ver qué commits hay entre dos versiones:
```bash
git log v1.0.0..v1.1.0 --oneline
```

### **3. Automatizar Versionado con Git y Tags**
Si quieres lanzar una nueva versión de forma automática, puedes usar:
```bash
git tag -a v$(date +%Y.%m.%d) -m "Versión automática con fecha"
```
Esto generará etiquetas como `v2025.01.30` (formato `AÑO.MES.DÍA`).

### **4. Eliminar o Reemplazar una Versión**
Si necesitas cambiar una versión mal etiquetada:
```bash
git tag -d v1.0.0  # Borra la etiqueta local
git push --delete origin v1.0.0  # Borra en GitHub
```
Y luego la vuelves a crear correctamente:
```bash
git tag -a v1.0.0 -m "Versión corregida"
git push origin v1.0.0
```

### **Conclusión**
El versionado semántico ayuda a organizar versiones en proyectos y facilita la colaboración en equipos. **Git y los tags hacen que la gestión de versiones sea fácil y estructurada.**  

### Resumen

Git facilita el control de versiones y organización de proyectos, y los comandos `git tag` y `git checkout` son piezas clave para una gestión eficiente y ordenada de los cambios en el código. Ambos comandos ayudan a crear puntos de referencia y explorar cambios sin afectar el desarrollo principal, ofreciendo opciones robustas para pruebas y organización.

### ¿Cómo se utiliza `git tag` para organizar versiones?

El comando `git tag` permite marcar un commit con una etiqueta descriptiva, ideal para señalar versiones estables o hitos importantes en el proyecto. Esto resulta útil en proyectos donde el equipo necesita identificar fácilmente puntos clave de avance. Al etiquetar, se añade una nota visible en el historial, lo cual facilita encontrar versiones específicas en un flujo de trabajo con muchos commits.

Para crear un tag:

- Escribe `git tag -a v1.0 -m "primera versión estable"` y presiona Enter.
- Al consultar `git log`, se verá el tag junto al commit en el historial.

Además, `git show` muestra detalles de la etiqueta, quién la creó, el mensaje de la versión y los cambios asociados a ese commit. Esto es especialmente útil cuando el historial es extenso, ya que permite regresar a puntos específicos sin necesidad de revisar cada commit en el log completo.

Para eliminar un tag:

- Usa `git tag -d v1.0` para remover el tag sin afectar el historial ni los archivos. Esto es conveniente si el nombre del tag necesita ser corregido o ajustado.

### ¿Qué permite `git checkout` al explorar el historial?

El comando `git checkout` tiene usos más amplios que solo cambiar entre ramas. También permite revisar commits previos para explorar o probar cambios sin alterar la rama principal. Al usar `git checkout` , puedes regresar a un punto específico en el historial y evaluar cómo afectaban los cambios al proyecto en ese momento.

Por ejemplo:

1. Cambia a un commit específico con `git checkout` 
2. Realiza pruebas o modificaciones. Esto te permite simular cambios o ver el estado del proyecto en esa versión. 
3. Para regresar a la rama principal, escribe `git checkout main`. Esto restaura el proyecto al estado actual y evita que los cambios temporales afecten el historial o la estructura del proyecto. 

Al navegar entre commits y regresar a `main`, es importante notar que no se crean ramas adicionales, ni se modifican commits previos, lo cual asegura la integridad del historial y la rama principal.

### ¿Cómo integran `git tag` y `git checkout` una experiencia de desarrollo ordenada?

Ambos comandos permiten explorar y organizar sin interferir en el flujo principal del trabajo. `git tag` marca versiones y puntos importantes, actuando como separadores en el historial, mientras que `git checkout` permite regresar a esos puntos y probar sin comprometer la rama actual. Esto proporciona una estructura en la que el equipo puede trabajar con libertad para realizar pruebas, versionar cambios y retornar al estado actual en cualquier momento sin temor a alterar el trabajo original.

### COMO AGREGARLE ETIQUETAS A LOS COMMITS

- `git tag` > Es como una etiqueta que le puedes aplicar a un commit que ayuda a identificar en un futuro, al aplicar este comando me lista todos los tag creados en la actualidad
- `-a` > El flag `-a` indica que el tag será anotado
- `-m` > El flag `-m` permite añadir un mensaje descriptivo al tag.
- `-d` > El flag `-d` Permite borrar el tag escrito. Esto no altera los archivos solo etiqueta

Un **flag** es una opción o parámetro que modifica el comportamiento de un comando

### COMO PODER MOSTRAR LOS DETALLES DE UN TAG

- `git show` > ayuda a mostrar el detalle de un commit o una etiqueta tag y a su vez combinaciones de ramas y archivos.

### COMO ALTERTAR Y CHEQUIAR CAMBIOS ANTES DE INTEGRARLOS EN EL MAIN

- `git checkout` > cambiar de una rama o un commit especifico

**Lecturas recomendadas**

[Git - git-tag Documentation](https://git-scm.com/docs/git-tag)

[Git - git-checkout Documentation](https://git-scm.com/docs/git-checkout)

## Cómo Resolver Conflictos de Ramas en Git

Cuando trabajas con Git, los conflictos de fusión ocurren cuando dos ramas modifican la misma parte de un archivo y Git no sabe cuál versión mantener.  

🚀 **¿Cómo se resuelven estos conflictos?** Vamos paso a paso.

## **1. Detectar un Conflicto**  
Si intentas hacer un `merge` y Git detecta un conflicto, verás un mensaje como este:

```bash
git merge feature-branch
```
🔴 **Salida:**  
```
Auto-merging archivo.txt
CONFLICT (content): Merge conflict in archivo.txt
Automatic merge failed; fix conflicts and then commit the result.
```
📌 Esto significa que el archivo `archivo.txt` tiene cambios en ambas ramas que entran en conflicto.

## **2. Ver los Archivos con Conflicto**  
Ejecuta:
```bash
git status
```
🔹 Verás algo como:
```
both modified: archivo.txt
```
Esto indica que `archivo.txt` tiene conflictos.

### **3. Editar el Archivo para Resolver el Conflicto**  
Abre el archivo en conflicto (`archivo.txt`). Git lo marcará así:

```
<<<<<<< HEAD
Esta es la versión en la rama actual.
=======
Esta es la versión en la rama "feature-branch".
>>>>>>> feature-branch
```

### **Opciones para Resolverlo:**
1️⃣ **Mantener la versión de `HEAD` (rama actual)**  
2️⃣ **Mantener la versión de `feature-branch`**  
3️⃣ **Fusionar ambas versiones manualmente**  

Ejemplo de **solución manual**:
```
Esta es la versión final combinada de ambas ramas.
```

Guarda el archivo después de editarlo.

### **4. Marcar el Conflicto como Resuelto**
Después de editar el archivo, agrégalo a staging:
```bash
git add archivo.txt
```

Luego, finaliza el merge con:
```bash
git commit -m "Resolviendo conflicto en archivo.txt"
```

### **5. Continuar con la Fusión**
Si estás haciendo un `rebase`, después de resolver el conflicto usa:
```bash
git rebase --continue
```
Si decides abortar el merge y deshacer los cambios:
```bash
git merge --abort
```

## **Evitar Conflictos en el Futuro**
✔ **Sincroniza tu rama antes de hacer cambios**  
```bash
git pull origin main
```
✔ **Haz commits pequeños y frecuentes**  
✔ **Comunica los cambios con tu equipo**  


### **Conclusión**  
Resolver conflictos en Git requiere revisar el código con cuidado y elegir qué cambios mantener.  

## Resumen

Cuando trabajamos en equipo, el manejo de conflictos de ramas en Git es esencial para evitar problemas y asegurar una integración fluida de cambios en los archivos compartidos. Aquí te mostramos cómo se genera un conflicto de ramas y la forma efectiva de resolverlo paso a paso.

### ¿Qué es un conflicto de ramas en Git?

En un entorno colaborativo, es común que varias personas realicen modificaciones en archivos compartidos. Esto puede llevar a conflictos de ramas cuando intentamos fusionar cambios y estos alteran las modificaciones previas realizadas por otro colaborador. En estos casos, se debe elegir qué cambios se mantendrán en la rama principal.

### ¿Cómo crear un conflicto de ramas para aprender a resolverlo?

Para experimentar y entender cómo resolver un conflicto, podemos crear uno intencionalmente. Aquí están los pasos básicos:

- Verifica tu rama actual con `git branch`. Si solo tienes la rama `main`, estás listo para iniciar.
- Crea un archivo, por ejemplo, `conflict.txt`, añade contenido inicial (e.g., “línea original”) y realiza un commit:

```shell
git add conflict.txt
git commit -m "Archivo de conflicto creado"
```

- Crea una nueva rama con `git checkout -b developer` y modifica el archivo con nuevos cambios, como “cambios desde la rama dev”, realiza un commit.
- Vuelve a la rama `main` con `git checkout main` y modifica el mismo archivo en esta rama, por ejemplo, añadiendo “segundo cambio desde main”, y realiza otro commit.

Al regresar a `main` y realizar la fusión de `developer`, verás el conflicto.

### ¿Cómo resolver un conflicto de ramas en Git?

Cuando Git detecta un conflicto, te indicará las diferencias entre las ramas con etiquetas que facilitan la identificación de cambios:

- Abre el archivo en conflicto. Verás secciones como `<<<<< HEAD` y `>>>>>`, que marcan los cambios en `main` y en la rama que intentas fusionar (`developer`).
- Edita el archivo eliminando las líneas de marcación y decide cuáles cambios deseas conservar, combinar o incluso reescribir.
- Guarda el archivo sin las señalizaciones de conflicto y realiza un commit para registrar la resolución:

```shell
git add conflict.txt
git commit -m "Conflicto resuelto"
```

### ¿Qué hacer después de resolver un conflicto?

Una vez resuelto el conflicto y unificada la versión final en `main`, considera eliminar la rama `developer` para evitar conflictos futuros. Esto ayuda a mantener el historial de cambios limpio y reduce la posibilidad de cometer errores en el futuro.

Aquí voy con un step-by-step porque así lo entendí (espero les sirva y agradezco algún tip u observación):

1. Creo archivo propenso a conflictos y lo edito.

-----

2.Se crea una nueva rama donde:

2.1 Se edita archivo de conflicto.

2.2 git status, git add . , git commit -m "mensaje".

2.3 Cambiamos la rama a main.

-----

3.En rama main se modificó una vez más el archivo de conflicto y se le aplicó:

3.1 git status, git add . , git commit -m "mensaje".

-----

4.Fusionamos la rama de cambios local con la rama main -> git merge [2nd branch].

-----

5.Obtenemos mensaje de conflicto/error en las ramas por parte de Git al fusionar.

5.1Abrimos archivo de conflicto y se decide que se conserva/descarta.

5.2 git status, git add . , git commit -m "mensaje" -> commit de cambios "aprobados" donde ya no hay conflictos <3

-----

6.De nuevo hacemos el merge/fusionamos la rama de cambios local/alterna con la rama main.

-----

7.Por buena practica eliminamos la rama donde se hicieron los cambios inicialmente. -> git branch -D [branch]

**Lecturas recomendadas**

[Git - git-branch Documentation](https://git-scm.com/docs/git-branch)

[Git - git-merge Documentation](https://git-scm.com/docs/git-merge)

## Usando Git desde Visual Studio Code

Visual Studio Code (VS Code) tiene integración nativa con Git, lo que facilita el control de versiones sin necesidad de usar la terminal. Aquí te muestro cómo trabajar con Git en VS Code paso a paso. 🚀  

### **1. Configurar Git en VS Code**  

### **Verificar si Git está instalado**  
Abre VS Code y presiona `Ctrl + ñ` para abrir la terminal integrada. Luego, escribe:  
```bash
git --version
```
🔹 Si Git no está instalado, descárgalo desde [git-scm.com](https://git-scm.com/).  

### **Configurar tu usuario y correo**  
Si es la primera vez que usas Git, configura tu nombre y correo:  
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tuemail@example.com"
```

### **2. Inicializar un Repositorio en VS Code**  

Si estás en un proyecto nuevo y quieres iniciar Git:  
1️⃣ Abre VS Code y carga tu proyecto.  
2️⃣ Abre la terminal (`Ctrl + ñ`) y ejecuta:  
   ```bash
   git init
   ```
3️⃣ Verás una nueva sección en el panel de control con el icono de **Git**.

### **3. Clonar un Repositorio desde GitHub**  
Si el proyecto ya está en GitHub:  
1️⃣ Copia la URL del repositorio desde GitHub.  
2️⃣ En VS Code, abre la terminal y escribe:  
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```
3️⃣ Abre la carpeta del proyecto en VS Code (`File > Open Folder`).

### **4. Realizar Commits desde VS Code**  

### **Agregar archivos al control de versiones**  
1️⃣ En el menú de Git (`Ctrl + Shift + G`), verás una lista de archivos modificados.  
2️⃣ Haz clic en `+` junto a cada archivo para agregarlo a staging (equivalente a `git add`).  
3️⃣ Escribe un mensaje de commit en la caja de texto y presiona `Ctrl + Enter` o haz clic en ✅ (equivalente a `git commit -m "mensaje"`).

### **5. Subir Cambios a GitHub**  

### **Conectar el repositorio a GitHub**  
Si el repositorio no está vinculado a GitHub, usa:  
```bash
git remote add origin <URL_DEL_REPOSITORIO>
git branch -M main  # Asegura que la rama principal sea "main"
git push -u origin main
```

### **Enviar cambios a GitHub**  
Cada vez que hagas cambios:  
1️⃣ Realiza `commit` como en el paso anterior.  
2️⃣ Haz clic en el botón `Sync Changes` en la barra de Git.  
3️⃣ O usa la terminal:  
   ```bash
   git push origin main
   ```

### **6. Descargar Cambios desde GitHub**  

Para actualizar tu código con la última versión del repositorio remoto:  
1️⃣ Ve a la sección de Git en VS Code.  
2️⃣ Haz clic en el botón `Pull` para traer cambios del repositorio.  
3️⃣ O usa la terminal:  
   ```bash
   git pull origin main
   ```

### **7. Crear y Cambiar de Ramas en VS Code**  

### **Crear una nueva rama**
1️⃣ Abre la barra de comandos (`Ctrl + Shift + P`).  
2️⃣ Escribe **"Git: Create Branch"** y selecciona la opción.  
3️⃣ Ingresa el nombre de la nueva rama y presiona Enter.  

📌 También puedes usar la terminal:  
```bash
git branch nueva-rama
git switch nueva-rama  # Para cambiar de rama
```

### **Cambiar entre ramas**  
1️⃣ Haz clic en el nombre de la rama en la esquina inferior izquierda.  
2️⃣ Selecciona la rama a la que quieres cambiar.  

### **8. Resolver Conflictos de Fusión en VS Code**  
Si hay un conflicto, VS Code te mostrará opciones en el archivo afectado:  
✅ **Accept Current Change** → Mantener la versión de tu rama actual.  
✅ **Accept Incoming Change** → Mantener la versión de la otra rama.  
✅ **Accept Both Changes** → Combinar ambas versiones manualmente.  

📌 Luego, guarda el archivo y haz `commit` para finalizar.

### **Conclusión**  
VS Code hace que trabajar con Git sea mucho más intuitivo. Puedes hacer **commits, push, pull, trabajar con ramas y resolver conflictos** sin salir del editor.  

### Resumen

Visual Studio Code ofrece una interfaz visual y eficiente para gestionar versiones con Git, simplificando muchas tareas complejas y ahorrando tiempo a los desarrolladores. Integrar VS Code en nuestro flujo de trabajo diario puede facilitar considerablemente el manejo de ramas, commits y conflictos sin depender tanto de comandos en la terminal.

### ¿Cómo abrir VS Code desde la terminal?

- Inicia VS Code en la ubicación del proyecto con `code .`.
- Esto abre una instancia de VS Code en el directorio actual, incluyendo todos los archivos versionados con Git.

### ¿Cómo visualizar y gestionar ramas en VS Code?

- Dentro de VS Code, identifica tu rama activa en la sección de control de versiones.
- Selecciona la rama para ver las opciones de cambio, como alternar entre ramas o crear nuevas.
- Los cambios en las ramas se presentan en una gráfica visual, diferenciando fusiones y ramas en colores, una ventaja significativa sobre `git log`.

### ¿Cómo hacer un commit de cambios en VS Code?

- Al editar un archivo, el ícono de control de versiones muestra un indicador de cambio.
- En lugar de usar `git commit -m "mensaje"`, puedes simplemente añadir un mensaje y presionar commit en la interfaz de VS Code.

### ¿Cómo crear y alternar entre ramas en VS Code?

1. Haz clic en “Create New Branch” y nómbrala, por ejemplo, “VS Code Dev”.
2. VS Code marca esta nueva rama como activa, heredando los cambios de la rama principal.
3. Al editar archivos en esta rama, puedes realizar commits directamente en la interfaz.

### ¿Cómo resolver conflictos de fusión en VS Code?

- Selecciona la rama con la que deseas fusionar (por ejemplo, VS Code Dev con Main) usando el menú de Branch > Merge.
- Cuando ocurre un conflicto, VS Code despliega opciones de resolución con colores para cada cambio, simplificando la selección entre el cambio actual, el entrante o ambos.
- Puedes optar por “Merge Editor” para una vista más visual y confirmar la fusión con un “Complete Merge” al finalizar.

### ¿Cómo iniciar un nuevo repositorio en VS Code?

1. Crea un nuevo directorio y abre VS Code en esa ubicación.
2. Al no haber archivos, selecciona “Inicializar repositorio” para configurar un nuevo repositorio.
3. Esto ejecuta `git init`, crea la rama principal (main) y permite añadir nuevas ramas y hacer commits sin usar comandos.

**Lecturas recomendadas**

[Visual Studio Code - Code Editing. Redefined](https://code.visualstudio.com/)

## Por qué GitHub es clave para los Desarrolladores

GitHub es una plataforma basada en la nube que permite a los desarrolladores **colaborar, almacenar, compartir y gestionar código de manera eficiente**. Su integración con Git lo convierte en una herramienta esencial para cualquier programador.  

### **🔹 Razones clave por las que GitHub es importante:**

1️⃣ **Control de versiones y colaboración**  
   - Permite a múltiples desarrolladores trabajar en el mismo proyecto sin sobrescribir cambios.  
   - Soporta ramas (`branches`) y fusiones (`merges`) para un desarrollo estructurado.  

2️⃣ **Trabajo en equipo con repositorios remotos**  
   - Facilita la colaboración global en proyectos de código abierto y privados.  
   - Equipos pueden trabajar en diferentes partes del código y fusionarlas fácilmente.  

3️⃣ **Historial de cambios y seguridad**  
   - Cada cambio queda registrado, permitiendo volver a versiones anteriores si es necesario.  
   - Ofrece control de acceso y permisos para proteger el código.  

4️⃣ **Integración con CI/CD (Integración y Despliegue Continuo)**  
   - Se puede automatizar la compilación, pruebas y despliegue con herramientas como **GitHub Actions, Travis CI y Jenkins**.  

5️⃣ **Portafolio y oportunidades laborales**  
   - Sirve como una **carta de presentación para desarrolladores**:  
     - Empresas pueden revisar proyectos y contribuciones en repositorios públicos.  
     - Participar en código abierto aumenta la visibilidad profesional.  

6️⃣ **Facilita la gestión de proyectos**  
   - Con **GitHub Issues y Projects**, se pueden organizar tareas y reportar errores.  
   - Permite una gestión ágil de equipos con tableros estilo **Kanban**.  

7️⃣ **Acceso a un ecosistema de herramientas y comunidad**  
   - Tiene una gran comunidad de desarrolladores y empresas contribuyendo activamente.  
   - Permite usar herramientas como **GitHub Copilot** para mejorar la productividad.  

### **🔹 ¿Cómo empezar con GitHub?**  
1️⃣ **Crea una cuenta en [GitHub](https://github.com/)**  
2️⃣ **Crea un repositorio** con:  
   ```bash
   git init
   git add .
   git commit -m "Primer commit"
   git branch -M main
   git remote add origin <URL-del-repositorio>
   git push -u origin main
   ```
3️⃣ **Colabora con otros mediante `fork`, `pull requests` y `issues`.**  

### **Conclusión:**  
GitHub no solo **facilita la colaboración y gestión de código**, sino que también es un espacio para aprender, mejorar y conectarse con otros desarrolladores. 

### Resumen

La colaboración en proyectos de software depende de sistemas de control de versiones, y Git es una herramienta central para lograrlo. Usar GitHub, una plataforma en la nube basada en Git, permite que los desarrolladores compartan sus proyectos, trabajen en equipo y accedan a herramientas avanzadas para asegurar y escalar sus desarrollos. Con un enfoque en inteligencia artificial (IA), colaboración, productividad, seguridad y escalabilidad, GitHub ha pasado de ser una red social de programadores a una herramienta integral que optimiza el desarrollo de software moderno.

### ¿Qué opciones existen para hospedar proyectos en Git?

- **GitHub**: la plataforma más destacada, adquirida por Microsoft en 2018, ofrece amplias herramientas de colaboración y desarrollo.
- **Bitbucket (Atlassian)**, GitLab, Azure DevOps (Microsoft), CodeCommit (Amazon), y Cloud Source (Google): todas permiten el control de versiones en la nube.
- **Servidores propios de Git:** para quienes prefieren un ambiente privado y controlado.

### ¿Cómo ha evolucionado GitHub desde su lanzamiento?

Inicialmente, GitHub era un simple repositorio de código en la nube; sin embargo, ha evolucionado hasta ofrecer una plataforma avanzada que incluye una interfaz web, herramientas de línea de comandos y flujos de trabajo colaborativos. En lugar de limitarse a compartir proyectos, permite a los usuarios colaborar en tiempo real, automatizar tareas y utilizar inteligencia artificial para mejorar la seguridad y productividad del código.

### ¿Qué funcionalidades destacan en GitHub actualmente?

GitHub ahora integra IA y facilita procesos clave en el desarrollo de software mediante:

- **Colaboración eficiente**: herramientas para trabajo en equipo, seguimiento de cambios y mejoras en el flujo de trabajo.
- **Automatización y productividad**: automatiza tareas repetitivas, permitiendo a los desarrolladores enfocarse en resolver problemas complejos.
- **Seguridad integrad**a: herramientas avanzadas de seguridad que aseguran el código desde el inicio, minimizando riesgos.
- **Escalabilidad**: una infraestructura robusta que permite gestionar millones de repositorios y usuarios globalmente.

### ¿Qué oportunidades brinda GitHub para los desarrolladores?

Con GitHub, cualquier desarrollador puede contribuir a proyectos relevantes, como mejoras en lenguajes de programación o incluso en el kernel de Linux. Esta capacidad de colaboración global eleva el nivel de la ingeniería de software, fomentando el trabajo en equipo entre profesionales de todo el mundo.

### ¿Cómo puede ayudarte GitHub en el desarrollo profesional?

Además de ser una herramienta de colaboración y desarrollo, GitHub ofrece la GitHub Foundation Certification, una certificación ideal para validar habilidades en GitHub y dar un primer paso hacia un perfil profesional sólido en desarrollo colaborativo.

**Lecturas recomendadas**

[GitHub: Let’s build from here · GitHub](https://github.com/)

## Creación y configuración de cuenta de GitHub

GitHub es una plataforma clave para gestionar proyectos y colaborar con otros desarrolladores. Aquí tienes una guía paso a paso para crear y configurar tu cuenta correctamente.  

### **1️⃣ Crear una cuenta en GitHub**
### 📌 Pasos:
1. **Ir a la página oficial:** [GitHub](https://github.com/)  
2. **Haz clic en "Sign up" (Registrarse).**  
3. **Ingresa tus datos:**
   - Nombre de usuario (único y profesional).
   - Correo electrónico.
   - Contraseña segura.  
4. **Verificación de cuenta:**  
   - GitHub puede pedirte resolver un captcha para confirmar que no eres un bot.  
5. **Escoge un plan:**  
   - **Gratis** (suficiente para la mayoría de los desarrolladores).  
   - Planes pagos con funciones avanzadas (opcional).  
6. **Confirma tu correo electrónico:**  
   - Revisa tu bandeja de entrada y haz clic en el enlace de verificación.  

### **2️⃣ Configurar Git en tu PC**
Antes de usar GitHub, necesitas configurar Git en tu computadora.  

### 📌 **Instalar Git**
Si no lo tienes instalado, descárgalo desde:  
🔗 [https://git-scm.com/downloads](https://git-scm.com/downloads)  

Después de instalarlo, verifica que está funcionando:  
```bash
git --version
```
Debe mostrar una versión como `git version 2.x.x`.

## **3️⃣ Configurar Git con GitHub**
Para enlazar Git con GitHub, sigue estos pasos:  

### 📌 **Configurar tu nombre y correo en Git**  
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tuemail@example.com"
```
👉 Usa el **mismo correo** con el que te registraste en GitHub.  

Verifica la configuración con:  
```bash
git config --list
```

### **4️⃣ Generar y añadir una clave SSH a GitHub (Opcional pero recomendado)**  
Esto evita que ingreses tu usuario y contraseña cada vez que uses GitHub.  

### 📌 **Generar una clave SSH**  
Ejecuta este comando en tu terminal (Git Bash en Windows):  
```bash
ssh-keygen -t rsa -b 4096 -C "tuemail@example.com"
```
Presiona **Enter** para aceptar la ubicación predeterminada y no pongas contraseña (opcional).  

### 📌 **Agregar la clave SSH a GitHub**  
1. Copia la clave SSH con:  
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
2. Ve a **GitHub > Settings > SSH and GPG keys**.  
3. Haz clic en **New SSH Key**, ponle un nombre y pega la clave.  
4. Guarda y verifica con:  
   ```bash
   ssh -T git@github.com
   ```
   Si ves el mensaje `Hi <usuario>! You've successfully authenticated`, todo está listo. 🎉  

### **5️⃣ Crear y subir tu primer repositorio**  
1. En GitHub, ve a **"Repositories" > "New"**.  
2. Asigna un nombre, descripción y selecciona si será público o privado.  
3. Copia la URL del repositorio y en la terminal escribe:  
   ```bash
   git init
   git add .
   git commit -m "Primer commit"
   git branch -M main
   git remote add origin <URL_DEL_REPOSITORIO>
   git push -u origin main
   ```

### 🎯 **¡Listo! Ya tienes tu cuenta y Git configurado con GitHub.**  
Ahora puedes comenzar a trabajar en proyectos y colaborar con otros desarrolladores.

**Lecturas recomendadas**

[GitHub · Build and ship software on a single, collaborative platform · GitHub](https://github.com/)

## ¿Cómo integrar Git y GitHub en un flujo de trabajo profesional?

En un entorno profesional, Git y GitHub se utilizan para gestionar código de manera eficiente, asegurando **colaboración, control de versiones y despliegue continuo**. A continuación, te explico cómo estructurar un flujo de trabajo profesional utilizando Git y GitHub.  

### **1️⃣ Configuración Inicial del Proyecto**  
Antes de comenzar a trabajar en un proyecto, sigue estos pasos:  

### 📌 **Crear un repositorio en GitHub**  
1. Ve a [GitHub](https://github.com/).  
2. Haz clic en **"New Repository"**.  
3. Asigna un nombre y elige si será **público o privado**.  
4. Inicializa el repositorio con un **README.md** y un archivo **.gitignore** según el lenguaje del proyecto.  
5. Copia la URL del repositorio.  

### 📌 **Clonar el repositorio en tu máquina local**  
```bash
git clone <URL_DEL_REPOSITORIO>
cd nombre-del-repositorio
```

### **2️⃣ Trabajar con Ramas (Branching Model)**  
Para evitar conflictos en el código, usa ramas correctamente. Un flujo profesional sigue la estrategia **Git Flow o GitHub Flow**.  

### 📌 **Estructura de ramas recomendada:**  
✅ `main` → Contiene la versión estable en producción.  
✅ `develop` → Rama de integración donde se prueban nuevas funcionalidades.  
✅ `feature/nueva-funcionalidad` → Para cada nueva característica o mejora.  
✅ `hotfix/fix-bug` → Para corrección de errores críticos en producción.  

### 📌 **Crear una nueva rama de trabajo**  
```bash
git checkout -b feature/nueva-funcionalidad
```
🚀 **Trabaja en la rama sin afectar `main` ni `develop`.**  

### **3️⃣ Realizar Cambios y Guardarlos en Git**  
Cada cambio en el código debe ser registrado con **commits claros y descriptivos**.  

### 📌 **Buenas prácticas al hacer commits:**  
- **Mensajes cortos y descriptivos** (`feat: agrega autenticación con JWT`).  
- **Commits atómicos** (un solo cambio por commit).  

### 📌 **Añadir y confirmar cambios**  
```bash
git add .
git commit -m "feat: agrega autenticación con JWT"
```

### **4️⃣ Subir Cambios a GitHub y Crear un Pull Request**  
Para integrar la nueva funcionalidad, **sube los cambios y crea un Pull Request (PR)**.  

### 📌 **Subir la rama al repositorio remoto**  
```bash
git push origin feature/nueva-funcionalidad
```
### 📌 **Crear un Pull Request en GitHub**  
1. Ve a **GitHub > Pull Requests**.  
2. Selecciona la rama `feature/nueva-funcionalidad` y compárala con `develop`.  
3. Agrega una descripción clara y solicita revisión del equipo.  
4. Una vez aprobado, haz **merge** de la rama a `develop`.  

### **5️⃣ Fusionar y Eliminar la Rama**  
Si los cambios son aprobados y fusionados correctamente, elimina la rama para mantener el repositorio limpio.  

### 📌 **Fusionar en `develop` y eliminar la rama**  
```bash
git checkout develop
git merge feature/nueva-funcionalidad
git branch -d feature/nueva-funcionalidad
git push origin develop
```

### **6️⃣ Implementación y Despliegue (CI/CD)**  
Un flujo profesional suele incluir **Integración y Despliegue Continuo (CI/CD)** con herramientas como:  

✅ **GitHub Actions**  
✅ **Jenkins, Travis CI, CircleCI**  
✅ **Docker y Kubernetes**  

Esto permite **automatizar pruebas y despliegues**, asegurando que solo código estable llegue a producción.  

### 🎯 **Conclusión:**  
Este flujo de trabajo con Git y GitHub permite:  
✅ **Colaboración organizada** entre desarrolladores.  
✅ **Manejo eficiente de versiones** con ramas estructuradas.  
✅ **Automatización con CI/CD** para mayor eficiencia.  

### Resumen
Para entender cómo Git y GitHub funcionan en conjunto en un flujo de trabajo profesional, debemos recordar que Git es una herramienta de control de versiones basada en comandos, mientras que GitHub facilita su implementación al ofrecer una plataforma que permite manejar proyectos de Git de forma colaborativa y accesible en la nube.

### ¿Cuál es la relación entre Git y GitHub?

Aunque Git y GitHub son complementarios, no fueron creados por los mismos desarrolladores ni comparten una dependencia directa. Git es el sistema de control de versiones en sí mismo, mientras que GitHub es un servicio que permite alojar repositorios Git en la nube, facilitando el trabajo colaborativo. La única conexión entre ambos es que GitHub permite gestionar proyectos que usan Git para el control de versiones.

### ¿Cómo se inicia el flujo de trabajo en GitHub?

Para trabajar en un proyecto en GitHub, en lugar de comenzar con `git init` en tu máquina local, creas un repositorio en GitHub. Este repositorio vacío se descarga al equipo y, desde ahí, se pueden hacer cambios locales. La estructura básica del flujo de trabajo incluye los siguientes pasos:

- **Crear un commit**: Guardar los cambios realizados localmente.
- **Subir cambios a GitHub**: Una vez los cambios estén listos, se suben a una rama separada en el repositorio remoto.

### ¿Por qué es importante trabajar en ramas?

Trabajar en una rama separada permite mantener el código principal estable mientras trabajas en nuevas características. Al subir la rama a GitHub, el proceso de **Code Review** comienza. Otros compañeros revisarán y aprobarán los cambios antes de integrarlos en la rama principal.

### ¿Qué reglas se pueden seguir para crear tareas?

Para facilitar la revisión de código y evitar conflictos, es ideal mantener las tareas pequeñas y con un solo objetivo. Esto hace que:

- El proceso de revisión sea sencillo.
- Los cambios sean menos propensos a conflictos al integrarse al proyecto principal.

Algunos equipos imponen reglas como limitar el número de archivos modificados o la cantidad de líneas de código en una tarea, aunque una recomendación básica es “una tarea, un objetivo”.

**Lecturas recomendadas**

[Póngase en marcha - Documentación de GitHub](https://docs.github.com/es/get-started/start-your-journey)

## Gestión de Repositorios en GitHub

Un **repositorio en GitHub** es donde se almacena y gestiona el código de un proyecto, permitiendo colaborar con otros desarrolladores. A continuación, te explico cómo gestionar eficazmente un repositorio en GitHub.  

### **1️⃣ Crear y Configurar un Repositorio en GitHub**  

### 📌 **Crear un nuevo repositorio**  
1. Inicia sesión en [GitHub](https://github.com/).  
2. Ve a **"Repositories" > "New"**.  
3. Ingresa:  
   - **Nombre del repositorio** (único y descriptivo).  
   - **Descripción** (opcional pero recomendada).  
   - **Visibilidad**:  
     - **Público**: Cualquier persona puede verlo.  
     - **Privado**: Solo tú y los colaboradores autorizados.  
   - Opcionalmente, inicializa con:  
     - **README.md** (Descripción del proyecto).  
     - **.gitignore** (Para ignorar archivos innecesarios).  
     - **Licencia** (Ejemplo: MIT, Apache, GPL).  
4. Haz clic en **"Create repository"**.  

### **2️⃣ Clonar un Repositorio**  
Para trabajar en el código localmente, clónalo con:  
```bash
git clone <URL_DEL_REPOSITORIO>
cd nombre-del-repositorio
```

Verifica la conexión con:  
```bash
git remote -v
```

### **3️⃣ Administración de Ramas**  
Un repositorio bien gestionado usa ramas (`branches`) para organizar el desarrollo.  

### 📌 **Ver ramas disponibles**  
```bash
git branch -a
```

### 📌 **Crear una nueva rama y cambiar a ella**  
```bash
git checkout -b feature/nueva-funcionalidad
```

### 📌 **Subir una rama al repositorio remoto**  
```bash
git push origin feature/nueva-funcionalidad
```

### **4️⃣ Gestión de Colaboradores y Permisos**  
Para trabajar en equipo, es importante gestionar accesos.  

### 📌 **Agregar colaboradores**  
1. Ve a **Settings > Collaborators**.  
2. Agrega el usuario de GitHub del colaborador.  
3. Define el rol:  
   - **Admin**: Control total.  
   - **Maintainer**: Puede gestionar ramas y PRs.  
   - **Write**: Puede subir código y hacer PRs.  
   - **Read**: Solo puede ver el repositorio.  

### 📌 **Trabajar con Forks y Pull Requests**  
Si no eres colaborador directo, puedes:  
1. **Hacer un Fork** (crear una copia del repositorio en tu cuenta).  
2. **Realizar cambios en tu Fork**.  
3. **Crear un Pull Request (PR)** para sugerir cambios al repositorio original.  

### **5️⃣ Gestionar Versiones y Releases**  
Para marcar hitos en el desarrollo, se pueden crear **tags y releases**.  

### 📌 **Crear un tag para una versión**  
```bash
git tag -a v1.0 -m "Versión estable 1.0"
git push origin v1.0
```

### 📌 **Publicar una Release en GitHub**  
1. Ve a **"Releases" > "Create a new release"**.  
2. Selecciona un tag (o crea uno nuevo).  
3. Escribe una descripción y sube archivos adicionales si es necesario.  
4. Publica la versión.  

### **6️⃣ Mantenimiento y Seguridad del Repositorio**  
Un buen mantenimiento del repositorio mejora la calidad del código.  

### 📌 **Actualizar el código del repositorio local**  
```bash
git pull origin main
```

### 📌 **Eliminar ramas obsoletas**  
```bash
git branch -d feature/antigua
git push origin --delete feature/antigua
```

### 📌 **Configurar GitHub Actions para Automatización**  
- Usa **GitHub Actions** para pruebas, despliegues y automatización.  
- Configura un flujo de trabajo en `.github/workflows/ci.yml`.  

### 🎯 **Conclusión:**  
🔹 **GitHub permite gestionar proyectos de manera eficiente** mediante ramas, colaboraciones, versiones y automatización.  
🔹 **Una buena estructura de repositorio facilita la colaboración y el mantenimiento del código.** 

### Resumen

Crear y gestionar un repositorio en GitHub es una habilidad esencial para colaborar y mantener proyectos de software de forma ordenada. Aquí aprenderás cómo crear, configurar, invitar colaboradores y clonar un repositorio de manera efectiva.

### ¿Cómo crear un repositorio en GitHub?

Para empezar, accede a la pantalla principal de tu perfil en GitHub y selecciona el símbolo de “+”. Aquí, selecciona la opción “Nuevo repositorio”, lo que abrirá un formulario para configurarlo:

- **Propietario**: Elige tu usuario actual como propietario del repositorio.
- **Nombre del repositorio**: Puedes asignarle un nombre como “mi-primer-repo”. Este nombre puede adaptarse a tu usuario, permitiendo reutilizar nombres similares.
- **Descripción**: Añade una breve descripción del proyecto para facilitar su identificación.
- **Visibilidad**: Decide si el repositorio será público o privado según las necesidades del proyecto.
- **Inicialización**: Puedes agregar un archivo README para documentar el repositorio desde el inicio. Aunque opcional, es una buena práctica.

Finalmente, selecciona el botón verde de “Crear repositorio” para completar este proceso. Al hacerlo, tendrás acceso directo a tu repositorio con el archivo README visible.

### ¿Cómo agregar un colaborador a un repositorio en GitHub?

Para trabajar en equipo, es fundamental añadir colaboradores. Esto se hace desde la sección de “Settings” del repositorio:

- Dirígete a “Colaboradores” en la configuración.
- Asegúrate de que el colaborador tenga una cuenta de GitHub.
- Selecciona la opción de agregar usuarios y elige a quien quieras invitar.

Una vez enviada la invitación, deberás esperar que el colaborador la acepte para que pueda acceder al repositorio y trabajar en él.

### ¿Cómo clonar un repositorio en tu máquina local?

Clonar el repositorio te permite trabajar desde tu entorno local y sincronizar cambios con GitHub. Para ello:

1. Ve a la sección de “Code” dentro de tu repositorio.
2. Selecciona la opción HTTPS y copia la URL del repositorio.
3. En tu terminal, escribe git clone seguido de la URL copiada y presiona “enter”.

Este comando descargará el repositorio en tu máquina. Podrás ver todos los archivos en una carpeta con el nombre del repositorio y comenzar a trabajar de manera local.

### ¿Cómo integrar Git y GitHub para un flujo de trabajo colaborativo?

Una vez que el repositorio está clonado en tu entorno local, puedes editar archivos, guardar cambios y subirlos de nuevo a GitHub mediante Git. Al hacer esto, permites que todos los colaboradores se mantengan sincronizados y al día con el desarrollo del proyecto.

**Lecturas recomendadas**

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github "GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub")

## Productos de GitHub: precios, planes y apps

Ahora que ya vimos como poder crear un repositorio en Github y usar sus repositorios, es momento de hablar acerca de los diferentes productos que veremos durante todo el curso y sus consideraciones, principalmente los costos de cada uno de los servicios que vamos a utilizar.

Recuerda que esta sección es de gran importancia porque como programadores podemos ver todos estos servicios como una variedad de opciones en donde podemos jugar como niños chiquitos en la arena; sin embargo, como parte de alguna organización debemos tener presente que los costos derivados de ello pueden jugar en nuestra contra si no sabemos como hacer para obtener un beneficio de todo esto, ten siempre presente la regla más importante de cualquier servicio que contrates.

### Si un servicio o herramienta que estás utilizando no está ayudando a tu organización, entonces la está perjudicando

Bueno, hora de dejar la clase de negocio y comenzar a ver el costo de los diferentes productos.

### Repositorios

Los repositorios de Github ya sean públicos o privados son gratuitos y sin un límite en específico en la cantidad de cuántos puedes tener, es decir, sin importar si se trata de una cuenta de pago o gratuita podrás crear tantos repositorios como gustes, así que por este tema no es necesario preocuparte, esta no es una diferencia entre todos los planes, tanto gratuitos como de pago.

### Codespaces

¡Huy! Aquí la cosa se pone buena. Codespaces es una herramienta que vamos a utilizar muchísimo en este curso y que es muy importante tener presente que es de costo. ¿Quieres un adelanto? Te recordaré todo el tiempo jugar con esta herramienta y luego apagarla, pero bueno, es momento de ver los costos.

Núcleos | Costo por hora | Tiempo de uso gratuito
---|---|
2 núcleos | $0.18 USD por hora | 60 horas gratuitas
4 núcleos | $0.36 USD por hora | 30 horas gratuitas
8 núcleos | $0.72 USD por hora | 15 horas gratuitas
16 núcleos | $1.44 USD por hora | No aplica
32 núcleos | $2.88 USD por hora | No aplica

En cuánto a almacenamiento también hay un costo asociado a ello.

Categoría | Costo | Datos gratuitos
---|---|---
Almacenamiento | $0.07 USD por mes | 15 GB gratuitos mensuales

Lo único que te puedo decir en esta categoría es que esas 30 horas de uso con 4 núcleos van a ser mucho más que suficientes para este curso y jugar un rato más, además, recuerda que cada mes se renuevan estos datos, así que si algo sucede simplemente tocará esperar.

### Github web editor

¡Buenas noticias aquí! Al igual que los repositorios, esta característica está presente en todos los planes de todos los niveles, sin costo en ningún escenario y sin límite de uso, esencialmente se trata de una característica que podemos aprovechar y aprender a utilizar mucho si preocuparnos por el costo.

### Github Actions

Github Actions es un tema de lo más complicado, el costo de las Actions depende mucho del sistema operativo, la capacidad del agente, obviamente el hardware y muchas cosas más; sin embargo, para los principiantes (y me incluyo en esta categoría porque ni de broma recuerdo todas las configuraciones) la mejor manera de evaluar y de guiarte es por medio del consumo por minutos, en la siguiente tabla podrás ver una buena referencia de los planes.

Plan | Consumo de minutos
---|--
Gratuito | 2,000 minutos de ejecución
Team | 3,000 minutos de ejecución
Enterprise | 50,000 minutos de ejecución

La verdad es que hay mucho que considerar en el tema de costos y beneficios de todas las herramientas y lo mejor es que dediques un tiempo a esto para saber como aprovechar al máximo los beneficios aquí solo mencionamos los productos que usaremos en el curso, sin embargo, hay muchas más consideraciones, lo ideal es que comiences por la página de referencia por excelencia para aprender de todo lo necesario acerca de esto, la puedes visitar [aquí](https://github.com/pricing "aquí").

## Cómo configurar SSH para GitHub: Guía paso a paso 

Configurar una clave **SSH** en GitHub permite autenticarse de forma segura sin necesidad de ingresar usuario y contraseña en cada operación con Git.  

### **1️⃣ Verificar si ya tienes una clave SSH**  
Antes de generar una nueva clave, revisa si ya tienes una en tu sistema.  

🔹 Abre una terminal y ejecuta:  
```bash
ls -al ~/.ssh
```
Si ves archivos como `id_rsa` y `id_rsa.pub`, significa que ya tienes una clave.  

Si ya tienes una, **puedes usarla** en GitHub o crear una nueva.  

### **2️⃣ Generar una Nueva Clave SSH**  
Si no tienes una clave SSH o quieres generar una nueva, sigue estos pasos.  

🔹 Ejecuta el siguiente comando en la terminal:  
```bash
ssh-keygen -t rsa -b 4096 -C "tuemail@example.com"
```
📌 **Explicación:**  
- `-t rsa`: Algoritmo RSA (recomendado).  
- `-b 4096`: Tamaño de la clave para mayor seguridad.  
- `-C "tuemail@example.com"`: Asocia la clave a tu correo de GitHub.  

🔹 **Presiona ENTER** para aceptar la ubicación predeterminada:  
```
Enter file in which to save the key (/home/tuusuario/.ssh/id_rsa):
```
🔹 **Opcionalmente, asigna una contraseña** para mayor seguridad.  

### **3️⃣ Agregar la Clave SSH al Agente SSH**  
Para que Git use la clave correctamente, agrégala al **agente SSH**.  

🔹 Inicia el agente SSH:  
```bash
eval "$(ssh-agent -s)"
```
🔹 Agrega la clave generada al agente:  
```bash
ssh-add ~/.ssh/id_rsa
```

### **4️⃣ Copiar la Clave SSH Pública**  
Ahora, copia la clave SSH para agregarla en GitHub.  

🔹 Ejecuta:  
```bash
cat ~/.ssh/id_rsa.pub
```
🔹 Copia la clave completa que se muestra en pantalla.  

### **5️⃣ Agregar la Clave SSH en GitHub**  
1. Ve a **GitHub** y accede a [Configuración SSH](https://github.com/settings/keys).  
2. Haz clic en **"New SSH Key"**.  
3. Ingresa un **nombre** para identificar la clave (Ejemplo: "Mi PC personal").  
4. **Pega la clave SSH copiada** en el campo correspondiente.  
5. Haz clic en **"Add SSH Key"**.  

### **6️⃣ Verificar la Conexión con GitHub**  
🔹 Para comprobar que todo está funcionando, ejecuta:  
```bash
ssh -T git@github.com
```
Si todo está correcto, deberías ver un mensaje como:  
```
Hi <tu-usuario>! You've successfully authenticated, but GitHub does not provide shell access.
```

### **7️⃣ Configurar Git para Usar SSH**  
Si quieres que Git use SSH en lugar de HTTPS, cambia la URL del repositorio.  

🔹 Para un repositorio existente, ejecuta:  
```bash
git remote set-url origin git@github.com:usuario/repositorio.git
```

🔹 Para clonar un nuevo repositorio con SSH:  
```bash
git clone git@github.com:usuario/repositorio.git
```

### 🎯 **Conclusión**  
✅ Ahora puedes trabajar con GitHub de manera segura sin ingresar tu usuario y contraseña en cada `push` o `pull`.  
✅ La autenticación SSH es **más rápida y segura** que HTTPS.  

### Resumen

Usar SSH para interactuar con GitHub es una excelente forma de aumentar la seguridad y mejorar la comodidad en el manejo de repositorios. A continuación, te explicamos el paso a paso para generar y configurar tus llaves SSH en distintos sistemas operativos y cómo integrarlas en tu perfil de GitHub para mejorar la experiencia de clonación y autenticación.

### ¿Por qué es mejor usar SSH en lugar de HTTPS para GitHub?

- **Seguridad adicional:** SSH permite autenticar la computadora específica que accede a los repositorios sin necesidad de ingresar una contraseña cada vez.
- **Comodidad**: Evita la necesidad de escribir tu contraseña en cada operación con GitHub, agilizando el flujo de trabajo.

### ¿Cómo generar una llave SSH en Windows y Linux?

1. **Instalar WSL** si estás en Windows (opcional si usas Linux nativo).
2. **Verificar que no tienes llaves previas**: Ve al menú de “Code” en GitHub y verifica que la opción de SSH no tenga llaves configuradas.
3. **Generar la llave SSH**: En la terminal, usa el comando:
`ssh-keygen -t ed25519 -C "tu_correo@example.com"`

 - -t ed25519 establece el nivel de encriptación.
 - -C añade un comentario con tu correo, útil para identificar la llave en GitHub.
 
4. **Guardar y proteger la llave**:

 - Usa el nombre por defecto y añade una contraseña segura.
 - La llave pública se guarda en el directorio `.ssh`, generalmente nombrada `id_ed25519.pub`.
 
5. Configurar el agente SSH: Activa el agente de SSH y añade la llave privada:

    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519

### ¿Cómo añadir la llave SSH a GitHub?

1. **Abrir el archivo de la llave pública** (id_ed25519.pub) y copia el contenido.
2. En GitHub, ve a **Settings** > **SSH and GPG keys** > **New SSH key** y pega la llave.
3. Nombra la llave de acuerdo a la computadora en la que estás configurándola.

### ¿Qué pasos adicionales seguir en Mac?

1. **Crear el archivo de configuración SSH**: Abre o crea el archivo `config` dentro del directorio `.ssh`.

2. **Agregar parámetros específicos de Mac:** Añade la configuración para integrar SSH con el sistema Keychain:

```shell
Host github.com
   AddKeysToAgent yes
   UseKeychain yes
   IdentityFile ~/.ssh/id_ed25519
```

3. Añadir la llave al agente SSH con Keychain: Usa el comando:

`ssh-add --apple-use-keychain ~/.ssh/id_ed25519`

### ¿Cómo verificar la conexión con GitHub?

1. Comprobar autenticación: En la terminal, ejecuta el comando:

`ssh -T git@github.com`

 - Escribe “yes” para confirmar la conexión.
 - Aparecerá un mensaje de GitHub que confirma la autenticidad.
 
### ¿Cómo clonar un repositorio usando SSH?

1. En GitHub, selecciona el repositorio deseado, elige la opción de clonación por SSH y copia la URL.
2. En la terminal, ejecuta:
`git clone git@github.com:usuario/repositorio.git`

3. Esto clona el repositorio sin solicitar contraseña, aprovechando la autenticación SSH.

**Lecturas recomendadas**

[Conectar a GitHub con SSH - Documentación de GitHub](https://docs.github.com/es/authentication/connecting-to-github-with-ssh)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Clone, fork y estrellas a repositorios 

Cuando trabajas con GitHub, hay tres acciones clave que te permiten interactuar con repositorios: **clonar (clone), bifurcar (fork) y marcar con estrella (star)**. Cada una tiene un propósito específico.  

### **1️⃣ Clonar un Repositorio (`git clone`)**  
**Clonar un repositorio** significa descargar una copia completa del código fuente en tu computadora. Esto te permite trabajar en él localmente.  

### 📌 **Clonar un repositorio público**  
```bash
git clone <URL_DEL_REPOSITORIO>
```
🔹 La URL se obtiene en GitHub desde el botón **"Code" > SSH o HTTPS**.  

Ejemplo:  
```bash
git clone https://github.com/octocat/Hello-World.git
```

### 📌 **Clonar un repositorio privado**  
Si el repositorio es privado, necesitas permisos y autenticación con SSH o un token de acceso.  

### 📌 **Trabajar en un repositorio clonado**  
Una vez clonado, entra al directorio del proyecto:  
```bash
cd Hello-World
```
Puedes trabajar en el código, hacer commits y subir cambios si tienes permisos.  

### **2️⃣ Fork: Bifurcar un Repositorio**  
**Hacer un fork** significa crear una copia del repositorio en tu cuenta de GitHub. Esto es útil cuando quieres contribuir a un proyecto en el que no tienes acceso directo.  

### 📌 **Pasos para hacer un fork**  
1. Ve al repositorio en GitHub.  
2. Haz clic en **"Fork"** (en la parte superior derecha).  
3. Ahora el repositorio aparece en tu cuenta con la URL:  
   ```
   https://github.com/tu-usuario/Hello-World
   ```
4. Clona el repositorio bifurcado en tu PC:  
   ```bash
   git clone https://github.com/tu-usuario/Hello-World.git
   ```

### 📌 **Cómo sincronizar cambios del repositorio original**  
Si el repositorio original cambia y quieres actualizar tu fork:  
```bash
git remote add upstream https://github.com/original-usuario/Hello-World.git
git fetch upstream
git merge upstream/main
```
Esto traerá los últimos cambios al fork.  

### 📌 **Enviar cambios al proyecto original**  
Si mejoras el código en tu fork y quieres que el dueño del repositorio original lo incluya, envía un **Pull Request (PR)** desde GitHub.  

### **3️⃣ Estrellas en un Repositorio (`⭐ Star`)**  
Las **estrellas (stars)** en GitHub sirven para:  
✅ **Marcar repositorios interesantes**.  
✅ **Apoyar proyectos que te gustan**.  
✅ **Guardar repositorios para acceder fácilmente después**.  

Para dar una estrella:  
1. Ve al repositorio en GitHub.  
2. Haz clic en el botón **"Star"** en la parte superior derecha.  

Puedes ver tus repositorios marcados con estrella en:  
```
https://github.com/stars
```

### 🎯 **Conclusión**  
🔹 **`git clone`** → Descarga una copia del repositorio en tu computadora.  
🔹 **Fork** → Crea una copia en tu cuenta para contribuir sin permisos directos.  
🔹 **Star ⭐** → Marca repositorios interesantes para guardarlos y apoyarlos.  

### Resumen

Entender el uso de forks y estrellas en GitHub optimiza la gestión de proyectos y recursos al trabajar en esta plataforma. Aquí exploraremos cómo funcionan estos elementos y cómo pueden ayudarte a organizar tus repositorios en función de tus necesidades.

### ¿Qué es un fork y cómo se utiliza?

Un fork en GitHub es una copia de un repositorio alojado en la cuenta de otra persona y que puedes transferir a tu propia cuenta. Este proceso crea una réplica del repositorio en su estado actual, sin reflejar futuros cambios del original a menos que se sincronice manualmente. Esto permite:

- Trabajar de manera independiente en un proyecto sin afectar el repositorio original.
- Personalizar el contenido según tus necesidades sin modificar el repositorio fuente.
- Crear una base para hacer contribuciones posteriores al repositorio original.

Para crear un fork, debes abrir el repositorio, seleccionar el botón de Fork y seguir los pasos para copiarlo en tu cuenta. Así, GitHub duplicará el repositorio, manteniendo el nombre y descripción del original. Puedes optar por copiar solo la rama principal (main) o todo el proyecto. Luego, desde tu cuenta, podrás modificar el contenido sin interferir con el repositorio original.

### ¿Qué beneficios aporta usar estrellas en GitHub?

Las *estrellas* en GitHub funcionan como un sistema de marcado para resaltar los repositorios que deseas tener a mano como referencia o favoritos. Son útiles para:

- Crear un índice de repositorios de referencia o inspiración.
- Acceder rápidamente a recursos clave desde tu perfil.
- Seguir el desarrollo de proyectos importantes para tus intereses.

Al seleccionar la estrella en un repositorio, ésta se ilumina para indicar que has marcado este recurso. Puedes acceder a todos tus repositorios marcados desde la sección de “tus estrellas” en tu perfil. Aquí se listan los proyectos que has destacado, ayudándote a centralizar tus fuentes de consulta.

### ¿Cómo clonar un repositorio forkeado?

Después de realizar un fork, puedes clonar este repositorio a tu entorno local para trabajar de forma directa en tu equipo. Para hacerlo:

1. Ve a tu repositorio forkeado.
2. Selecciona el botón **Code** y copia la URL del proyecto en formato SSH.
3. Abre la terminal y usa el comando `git clone`.

De esta manera, tendrás una versión local del repositorio en la que podrás modificar y gestionar el código. Esta técnica de fork y clonación es útil para personalizar proyectos o experimentar sin afectar el original, ofreciendo flexibilidad para hacer cambios sin alterar el repositorio fuente.

### ¿Por qué usar forks en lugar de clonar directamente el repositorio original?

Hacer un fork en lugar de una clonación directa del repositorio original permite que trabajes de manera independiente. Puedes hacer ajustes sin el riesgo de cambiar el repositorio base, especialmente útil cuando el original es de terceros o si planeas realizar cambios extensivos. Además, el fork es una base ideal para hacer contribuciones futuras, ya que se puede sincronizar y enviar cambios al proyecto original a través de un proceso estructurado.

**Lecturas recomendadas**

[Conectar a GitHub con SSH - Documentación de GitHub](https://docs.github.com/es/authentication/connecting-to-github-with-ssh)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Trabajo con repositorios remotos: push, pull y fetch

Cuando trabajas con **Git y GitHub**, necesitas sincronizar tu código local con un repositorio remoto. Los comandos clave para hacerlo son:  

- **`git push`** → Envía cambios al repositorio remoto.  
- **`git pull`** → Descarga y fusiona cambios desde el repositorio remoto.  
- **`git fetch`** → Obtiene cambios remotos sin fusionarlos automáticamente.  

### **1️⃣ Enviar Cambios con `git push`**  
Una vez que has hecho cambios en tu código y creado un commit, necesitas enviarlos al repositorio remoto.  

### 📌 **Pasos para hacer un `push`**  
1. Asegúrate de estar en la rama correcta:  
   ```bash
   git branch
   ```
2. Verifica los cambios que se enviarán:  
   ```bash
   git status
   ```
3. Agrega los archivos al área de staging (si no lo hiciste antes):  
   ```bash
   git add .
   ```
4. Crea un commit con un mensaje descriptivo:  
   ```bash
   git commit -m "Descripción de los cambios"
   ```
5. **Envía los cambios al repositorio remoto:**  
   ```bash
   git push origin main
   ```
   📌 **`origin`** → Nombre del repositorio remoto.  
   📌 **`main`** → Nombre de la rama donde se subirán los cambios.  

✅ **Si es la primera vez que subes una rama nueva**, usa:  
```bash
git push -u origin nueva-rama
```

### **2️⃣ Obtener Cambios con `git pull`**  
Si otras personas han hecho cambios en el repositorio remoto, debes actualizar tu código local antes de seguir trabajando.  

### 📌 **Ejecutar `git pull` para actualizar tu código local**  
```bash
git pull origin main
```
Esto descarga y fusiona automáticamente los cambios de la rama `main` en tu código local.  

### 📌 **Cuando usar `git pull`?**  
✅ Antes de comenzar a trabajar en nuevas características.  
✅ Para mantener tu código actualizado con el equipo.  

Si hay **conflictos de fusión**, Git te pedirá que los resuelvas manualmente antes de continuar.  

### **3️⃣ Obtener Cambios sin Fusionar con `git fetch`**  
A veces, solo quieres ver si hay actualizaciones sin aplicarlas todavía. Para eso, usa `git fetch`.  

### 📌 **Descargar cambios sin fusionarlos automáticamente**  
```bash
git fetch origin
```
Esto descarga las actualizaciones desde el repositorio remoto, pero **no modifica** tu código local.  

### 📌 **Ver los cambios antes de fusionarlos**  
```bash
git diff main origin/main
```
Si decides traer los cambios después de revisar, usa:  
```bash
git merge origin/main
```

### **📌 Diferencias Clave entre `git pull` y `git fetch`**  

| Comando    | ¿Qué hace? |
|------------|-----------|
| `git fetch` | Descarga cambios remotos sin aplicarlos. |
| `git pull`  | Descarga y aplica cambios automáticamente. |

Si no quieres que Git **modifique automáticamente** tu código, usa `fetch` en lugar de `pull`.  

### 🎯 **Conclusión**  
🔹 **`git push`** → Sube tus cambios al repositorio remoto.  
🔹 **`git pull`** → Descarga y fusiona cambios del repositorio remoto.  
🔹 **`git fetch`** → Obtiene actualizaciones sin fusionarlas automáticamente.  

**Lecturas recomendadas**

[Git - git-fetch Documentation](https://git-scm.com/docs/git-fetch)

[Git - git-push Documentation](https://git-scm.com/docs/git-push)

[Git - git-pull Documentation](https://git-scm.com/docs/git-pull)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Gestión de Issues y Discussions en GitHub

Cuando trabajas en un proyecto en **GitHub**, es importante mantener un buen flujo de comunicación y seguimiento de tareas. Para esto, GitHub ofrece dos herramientas clave:  

- **Issues** → Para reportar errores, sugerir mejoras o gestionar tareas.  
- **Discussions** → Para debatir ideas, recibir retroalimentación y compartir conocimientos.  


### **1️⃣ Issues: Reportar Errores y Gestionar Tareas**  
Los **Issues** (incidencias) son la mejor forma de **documentar problemas, sugerencias y tareas pendientes** en un repositorio.  

### 📌 **Cómo Crear un Issue**  
1. Ve al repositorio en GitHub.  
2. Haz clic en la pestaña **"Issues"**.  
3. Presiona **"New Issue"**.  
4. Agrega un **título claro** y una descripción detallada.  
5. Puedes asignar etiquetas (*labels*) para clasificar el Issue.  
6. Asigna responsables si es necesario.  
7. Presiona **"Submit new issue"**.  

### 📌 **Ejemplo de Issue Bien Escrito**  
**Título:** Bug: Error al autenticar usuario con Google  

**Descripción:**  
- **Pasos para reproducir:**  
  1. Ir a la pantalla de inicio de sesión.  
  2. Hacer clic en "Iniciar sesión con Google".  
  3. Aparece un error "500 Internal Server Error".  
- **Comportamiento esperado:** El usuario debería autenticarse correctamente.  
- **Dispositivo y entorno:** Windows 10, navegador Chrome 114.  
- **Posible solución:** Revisar la configuración de la API de Google.  

✅ **Beneficio:** Un issue bien descrito facilita su solución.  

### **2️⃣ Discussions: Espacio para Debatir y Compartir Ideas**  
Las **Discussions** permiten a los desarrolladores, colaboradores y la comunidad hablar sobre el proyecto sin necesidad de abrir un Issue.  

### 📌 **Cuándo usar Discussions en lugar de Issues**  
✅ Para debatir **nuevas funcionalidades** antes de implementarlas.  
✅ Para **preguntas generales** o ayuda técnica.  
✅ Para recibir **sugerencias** de la comunidad.  

### 📌 **Cómo Crear una Discussion**  
1. Ve a la pestaña **"Discussions"** del repositorio.  
2. Haz clic en **"New Discussion"**.  
3. Elige una categoría (idea, pregunta, anuncio, etc.).  
4. Agrega un título y redacta tu mensaje.  
5. Presiona **"Start discussion"**.  

💬 **Ejemplo de Discussion:**  
> 🎯 **Propuesta: Agregar soporte para idioma español**  
> ¡Hola equipo! ¿Qué opinan sobre traducir la aplicación al español?  
> Creo que podríamos empezar agregando un archivo `locales/es.json`.  


### **3️⃣ Mejoras con Etiquetas (`Labels`) y Asignaciones (`Assignees`)**  
- **Etiquetas (`labels`)** → Para clasificar los Issues (Ej: `bug`, `feature request`, `documentation`).  
- **Asignaciones (`assignees`)** → Para designar a la persona encargada de resolver el Issue.  
- **Milestones** → Para agrupar Issues en objetivos o versiones específicas.  

### 📌 **Ejemplo de Etiquetas en un Issue**  
| Etiqueta | Descripción |
|----------|------------|
| `bug` | Un error en la aplicación. |
| `enhancement` | Una mejora o nueva funcionalidad. |
| `documentation` | Problemas o mejoras en la documentación. |
| `good first issue` | Ideal para nuevos colaboradores. |

### 🎯 **Conclusión**  
🔹 **Issues** → Para reportar errores y gestionar tareas en el proyecto.  
🔹 **Discussions** → Para debatir ideas y recibir retroalimentación.  
🔹 **Etiquetas y asignaciones** → Facilitan la organización del trabajo en equipo.  

### Resumen

Usar los Issues de GitHub permite gestionar y documentar problemas en un repositorio, desde errores simples en la redacción hasta defectos complejos en la funcionalidad. Con una interfaz intuitiva, permite que cualquier usuario señale y describa aspectos mejorables, y con plantillas de Issues, mejora la claridad y colaboración en el proceso.

### ¿Qué es un Issue en GitHub?

Un Issue es una forma de señalar problemas o sugerencias dentro de un repositorio. Puede ser usado para:

- Notificar errores en la documentación, como faltas de ortografía.
- Reportar problemas en el funcionamiento esperado del código.
- Informar mejoras o cambios necesarios.

Los Issues permiten una comunicación bidireccional entre los colaboradores y el creador del repositorio, lo que facilita la resolución de problemas.

### ¿Cómo crear un nuevo Issue?

1. En el repositorio de GitHub, selecciona la pestaña Issues.
2. Haz clic en **New Issue** y describe el problema en dos campos principales:
 - Título: Una breve descripción.
 - Descripción: Detalles del problema, pasos para reproducirlo, etc.
 
Es posible agregar elementos adicionales, como asignar el Issue a otra persona o etiquetarlo.

### ¿Cómo crear una plantilla de Issues?

Para facilitar el proceso a otros colaboradores, es útil crear plantillas de Issues. Para hacerlo:

1. Desde el repositorio, abre Visual Studio Code con el comando `code .`.
2. Crea una carpeta llamada `.github` y dentro otra carpeta llamada `ISSUE_TEMPLATE`.
3. Dentro de `ISSUE_TEMPLATE`, crea un archivo Markdown (por ejemplo, `bug_report.md`).
4. Copia la estructura de la plantilla, que usualmente incluye secciones como descripción, pasos para reproducir el error y detalles adicionales.

Con esta plantilla, los colaboradores tendrán un formato estándar para reportar problemas, lo que ayuda a una mejor gestión y resolución.

### ¿Cómo sincronizar los cambios en GitHub?

1. Una vez creada la plantilla, verifica que los archivos cambiados estén marcados en verde en Visual Studio Code.
2. Realiza un commit (por ejemplo, “Bug Report agregado”).
3. Sincroniza con el repositorio de GitHub mediante el botón de flecha hacia arriba (push).
4. En GitHub, verifica que la plantilla esté disponible en la sección de Issues.

### ¿Qué ventajas tiene una plantilla de Issues?

Las plantillas simplifican el proceso de documentación de problemas y mejoran la comunicación al estandarizar la información que se solicita a los colaboradores. Esto ayuda a identificar los problemas de forma precisa y rápida.

### ¿Cómo personalizar las plantillas de Issues para casos específicos?

Además de la plantilla básica para bugs, puedes crear plantillas personalizadas como:

- **Document Report**: Para señalar errores en la documentación.
- **Mejores prácticas**: Para sugerir mejoras en la estructura del código.

Estas plantillas permiten a los colaboradores elegir el tipo de Issue que mejor se adapta al problema y agilizan la gestión del repositorio.

**Lecturas recomendadas**

[Acerca de las propuestas - Documentación de GitHub](https://docs.github.com/es/issues/tracking-your-work-with-issues/about-issues)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Colaboración sin errores: Pull Requests en GitHub

Cuando trabajas en equipo en **GitHub**, necesitas una forma organizada de proponer cambios en el código sin afectar directamente la versión principal. Aquí es donde entran en juego los **Pull Requests (PRs)**.  

Un **Pull Request (PR)** permite:  
✅ Sugerir cambios en un repositorio antes de fusionarlos (**merge**).  
✅ Revisar código en equipo para evitar errores.  
✅ Mantener un historial claro de contribuciones.  

### **1️⃣ ¿Cómo Crear un Pull Request en GitHub?**  
Sigue estos pasos para proponer cambios a un repositorio:  

### 📌 **1. Crear una Rama Nueva**  
Antes de hacer cambios, crea una rama separada:  
```bash
git checkout -b feature-nueva-funcionalidad
```
✅ Usar nombres descriptivos para las ramas ayuda a la organización.  

### 📌 **2. Hacer Cambios y Confirmarlos (`commit`)**  
Modifica los archivos y guarda los cambios:  
```bash
git add .
git commit -m "Añadida nueva funcionalidad X"
```

### 📌 **3. Subir los Cambios a GitHub (`push`)**  
```bash
git push origin feature-nueva-funcionalidad
```

### 📌 **4. Crear el Pull Request en GitHub**  
1. Ve al repositorio en GitHub.  
2. Haz clic en **"Pull Requests" > "New Pull Request"**.  
3. Selecciona la rama **feature-nueva-funcionalidad** y compárala con **main**.  
4. Agrega un **título claro** y una **descripción detallada**.  
5. Presiona **"Create Pull Request"**.  

### **2️⃣ ¿Cómo Revisar y Aprobar un Pull Request?**  
Después de crear un PR, otros desarrolladores pueden **revisarlo** antes de aprobarlo.  

### 📌 **Revisión de Código en un PR**  
1. Ve al PR en GitHub.  
2. Revisa los cambios en la pestaña **"Files changed"**.  
3. Agrega comentarios en líneas específicas si es necesario.  
4. Aprueba el PR con **"Approve"** o solicita cambios con **"Request changes"**.  

### 📌 **Fusionar (`merge`) un Pull Request**  
Si el código está listo, puedes **fusionarlo** con la rama principal:  
1. En el PR, haz clic en **"Merge pull request"**.  
2. Confirma con **"Confirm merge"**.  
3. (Opcional) Borra la rama después de fusionar.  

### **3️⃣ Diferencias entre `merge` y `rebase` en PRs**  
- **Merge** → Mantiene el historial de commits.  
- **Rebase** → Integra los cambios en una línea más limpia, sin historial de fusión.  

Si prefieres un historial más ordenado, puedes hacer:  
```bash
git rebase main
```

### **4️⃣ Resolver Conflictos en un Pull Request**  
Si GitHub detecta que hay cambios en la rama `main` que chocan con los de tu PR, verás un mensaje de **"This branch has conflicts that must be resolved"**.  

### 📌 **Cómo Resolver Conflictos en Git**  
1. Descarga los cambios más recientes:  
   ```bash
   git fetch origin
   git checkout feature-nueva-funcionalidad
   git merge main
   ```
2. Edita los archivos en conflicto (verás marcas como `<<<<<<<` en el código).  
3. Guarda los cambios y haz un nuevo commit:  
   ```bash
   git add .
   git commit -m "Resueltos conflictos de fusión"
   ```
4. Sube los cambios:  
   ```bash
   git push origin feature-nueva-funcionalidad
   ```

Ahora el PR estará listo para ser fusionado.  

### 🎯 **Conclusión**  
🔹 **Pull Requests** permiten colaborar sin afectar directamente la rama principal.  
🔹 **Revisión de código** mejora la calidad y seguridad del proyecto.  
🔹 **Resolver conflictos** es clave para mantener el código limpio.

### Resumen

Colaborar en GitHub requiere evitar modificar directamente la rama principal, lo que podría causar conflictos con el trabajo de otros compañeros. En su lugar, trabajar en ramas individuales y fusionarlas mediante *Pull Requests (PR)* es clave para un flujo de trabajo colaborativo y seguro.

### ¿Por qué evitar cambios directos en la rama principal?

Realizar cambios directamente en la rama principal (main) puede sobrescribir el trabajo no sincronizado de otros colaboradores. Este error común se evita al:

- Crear una rama separada para cada contribuyente.
- Fusionar cambios mediante una revisión en el Pull Request, antes de unirlos a la rama principal.

### ¿Cómo funciona un Pull Request?

1. **Crear una Rama Nueva**: Al iniciar cambios, crea una rama local específica. Por ejemplo, `developer01`.
2. **Subir la Rama a GitHub**: Usa `git push -u origin` para subir tu rama.
3. **Notificar al Equipo**: Al crear un Pull Request, notificas al equipo sobre tus cambios, lo que permite una revisión colaborativa (*Code Review*).

### ¿Qué papel juega la revisión de código?

El Code Review en los Pull Requests permite:

- Evaluar y comentar los cambios antes de fusionarlos.
- Aumentar la calidad y la visibilidad de los cambios propuestos.

Aunque puede ser intimidante al principio, esta práctica asegura transparencia y mejora continua en el equipo.

### ¿Cómo se fusiona un Pull Request?

- **Comparación y Revisión**: Una vez que el equipo revisa los cambios y los aprueba, GitHub facilita la fusión con la rama principal.
- **Resolver Conflictos**: GitHub verifica automáticamente conflictos potenciales, mostrando una marca verde si está listo para fusionarse sin problemas.
- **Eliminar la Rama**: Tras la fusión, se elimina la rama para mantener el repositorio ordenado y listo para nuevas tareas.

### ¿Cómo puedo practicar Pull Requests de forma efectiva?

Para mejorar, colabora con un amigo o colega, practicando la creación y revisión de Pull Requests. Esta interacción entre ramas te ayudará a familiarizarte y a fluir con confianza en el proceso de colaboración en GitHub.

**Lecturas recomendadas**

[Acerca de las solicitudes de incorporación de cambios - Documentación de GitHub](https://docs.github.com/es/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)

[Documentación de solicitudes de incorporación de cambios - Documentación de GitHub](https://docs.github.com/es/pull-requests)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Introducción a GitHub Projects

**GitHub Projects** es una herramienta integrada en **GitHub** que permite gestionar proyectos de forma visual y organizada, combinando **issues, pull requests y notas** en tableros personalizables.  

### 🎯 **¿Por qué usar GitHub Projects?**  
✅ Facilita la planificación y seguimiento del trabajo.  
✅ Se integra con Issues y Pull Requests automáticamente.  
✅ Permite organizar tareas en tableros estilo **Kanban** o listas.  
✅ Soporta automatización con GitHub Actions.  
✅ Ideal para **equipos** y **proyectos individuales**.

### **1️⃣ Cómo Crear un GitHub Project**  
Sigue estos pasos para iniciar un **Project** en GitHub:  

### 📌 **1. Acceder a GitHub Projects**  
1. Ve al repositorio donde quieres gestionar el proyecto.  
2. Haz clic en la pestaña **"Projects"**.  
3. Presiona **"New project"**.  

### 📌 **2. Elegir el Tipo de Proyecto**  
Puedes elegir entre:  
- **Table** (tablas personalizadas con filtros avanzados).  
- **Board** (tablero estilo **Kanban**, similar a Trello).  

Elige el que mejor se adapte a tu flujo de trabajo.  

### 📌 **3. Configurar el Proyecto**  
1. Asigna un **nombre y descripción**.  
2. Agrega **columnas o estados** (Ej: "To Do", "In Progress", "Done").  
3. Agrega **Issues o Pull Requests** arrastrándolos al tablero.  

### **2️⃣ Uso Básico de GitHub Projects**  
Una vez creado, puedes gestionar el trabajo de manera eficiente:  

### 📌 **1. Agregar Tareas (Items)**  
- Puedes **añadir Issues o PRs** existentes al proyecto.  
- También puedes **crear notas personalizadas** para otras tareas.  

### 📌 **2. Organizar el Trabajo con Columnas**  
- **To Do** → Tareas pendientes.  
- **In Progress** → Tareas en desarrollo.  
- **Done** → Tareas completadas.  

Puedes **personalizar columnas** según tu flujo de trabajo.  

### 📌 **3. Automatizar con GitHub Actions**  
Puedes configurar **automatizaciones** como:  
✅ Mover un Issue a **"In Progress"** cuando se asigne.  
✅ Marcar como **"Done"** cuando se cierre un Pull Request.

### **3️⃣ Ejemplo de Flujo de Trabajo con GitHub Projects**  
🔹 **1. Crear un Issue** → Se registra una tarea o bug.  
🔹 **2. Mover a "In Progress"** → Se asigna a un desarrollador.  
🔹 **3. Crear un Pull Request** → Se suben los cambios.  
🔹 **4. Revisar el Código** → Se aprueba el PR.  
🔹 **5. Fusionar y Cerrar el Issue** → Se mueve a "Done".  

🚀 **Resultado:** Un equipo bien organizado y un código de mejor calidad.  

### 🎯 **Conclusión**  
✅ **GitHub Projects** ayuda a organizar y visualizar tareas en un proyecto.  
✅ Se integra con **Issues, Pull Requests y GitHub Actions**.  
✅ Es una herramienta poderosa para **equipos y desarrolladores individuales**.  

### Resumen

Las herramientas colaborativas en GitHub han evolucionado, haciendo posible una organización eficaz y centralizada del trabajo en equipo sin la necesidad de mensajes o correos constantes. GitHub Projects destaca como una herramienta clave que permite gestionar tareas, fechas y recursos dentro de un solo espacio.

### ¿Cómo ayuda GitHub Projects en la organización de proyectos?

GitHub Projects facilita la organización de actividades dentro de equipos, permitiendo asignar tareas y priorizarlas según su urgencia. La herramienta cuenta con plantillas para diferentes tipos de proyectos, como la de **Team Planning** (planificación en equipo), que organiza tareas en un tablero Kanban con tres columnas: “Por hacer”, “En progreso” y “Hechas”.

### ¿Cómo crear y gestionar tareas en GitHub Projects?

Para crear una tarea en un proyecto de GitHub:

- Accede a “Projects” en tu perfil y selecciona **New Project**.
- Elige la plantilla que mejor se adapte al tipo de proyecto.
- Crea una nueva actividad, por ejemplo, “Actualizar proyecto HTML”. Esto permitirá que todos en el equipo sepan qué tareas están pendientes.

Además, puedes definir prioridades, tamaños y tiempos estimados para cada tarea, basándote en metodologías ágiles, lo cual ayuda a coordinar el esfuerzo del equipo y optimizar el rendimiento en cada iteración.

### ¿Cómo asignar tareas y gestionar las responsabilidades del equipo?

Dentro de GitHub Projects, es posible asignar tareas a miembros del equipo, mostrando claramente las responsabilidades de cada persona. Cada tarea puede vincularse a un repositorio específico seleccionando la opción **Convertir a un issue**, lo que permite una relación directa entre la actividad y el código. Esta vinculación muestra un número identificador, útil para nombrar ramas en el repositorio y facilitar la organización.

### ¿Cómo gestionar múltiples proyectos y tareas personales?

Para aquellos que participan en varios proyectos o trabajan como freelancers, GitHub Projects es una solución que permite centralizar actividades y manejar múltiples repositorios. Al seleccionar la pestaña **My Items**, se filtran todas las tareas asignadas a ti, ayudándote a enfocarte sin distracciones.

### ¿Cuál es la importancia de estimar tiempos en GitHub Projects?

Aprender a estimar tiempos de manera precisa es esencial para el trabajo en equipo. GitHub Projects permite practicar la estimación del tiempo y la carga de trabajo en cada tarea, lo que ayuda a los desarrolladores a mejorar en la gestión de su propio tiempo y a contribuir de manera eficiente al equipo, evitando interrupciones.

**Lecturas recomendadas**
[Acerca de Projects - Documentación de GitHub](https://docs.github.com/es/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Automatización y Gestión de Proyectos en GitHub

**GitHub Projects** permite no solo gestionar tareas de manera visual, sino también automatizar procesos para mejorar la eficiencia del equipo. A través de **GitHub Actions** y reglas personalizadas, puedes automatizar la gestión de issues, pull requests y cambios en el tablero del proyecto.  

### **1️⃣ Beneficios de Automatizar GitHub Projects**  
✅ Reduce tareas manuales repetitivas.  
✅ Mantiene el tablero actualizado automáticamente.  
✅ Facilita el seguimiento del progreso del proyecto.  
✅ Mejora la colaboración dentro del equipo.  

### **2️⃣ Métodos de Automatización en GitHub Projects**  

### 📌 **1. Reglas de Automatización (Built-in Automation)**  
GitHub Projects ofrece automatizaciones predefinidas que puedes activar sin necesidad de código.  

Ejemplo de automatización en un **tablero Kanban**:  
- Cuando se **abre un Issue**, se mueve automáticamente a la columna "To Do".  
- Cuando un **Pull Request se vincula a un Issue**, el Issue se mueve a "In Progress".  
- Cuando un **Pull Request se fusiona**, el Issue se cierra y se mueve a "Done".  

💡 **Cómo configurarlas:**  
1. Ve a **Projects** en tu repositorio.  
2. Abre tu proyecto y haz clic en **"Automate"**.  
3. Selecciona las reglas que deseas activar.  

### 📌 **2. Automatización con GitHub Actions**  
Si necesitas personalización avanzada, puedes usar **GitHub Actions** para ejecutar flujos de trabajo automáticos.  

Ejemplo: **Mover un Issue cuando se le asigne un responsable**  
```yaml
name: Move Issue to In Progress

on:
  issues:
    types: [assigned]

jobs:
  move_issue:
    runs-on: ubuntu-latest
    steps:
      - name: Move Issue to In Progress
        uses: actions/add-to-project@v1
        with:
          project-url: https://github.com/orgs/MiOrganizacion/projects/1
          github-token: ${{ secrets.GITHUB_TOKEN }}
          column-name: "In Progress"
```
✅ **Explicación:**  
- **`on: issues`** → Se ejecuta cuando se asigna un Issue.  
- **`actions/add-to-project@v1`** → Mueve el Issue al tablero de proyectos.  
- **`column-name: "In Progress"`** → Especifica a qué columna moverlo.  

💡 **Cómo implementarlo:**  
1. Crea un archivo en `.github/workflows/move-issue.yml`.  
2. Copia y pega el código anterior.  
3. Guarda y haz un `commit`.  

### 📌 **3. Integración con Pull Requests**  
Puedes automatizar cambios en el proyecto cuando un **Pull Request** es aprobado o fusionado.  

Ejemplo: **Cerrar un Issue automáticamente cuando se fusiona un Pull Request**  
```yaml
name: Close Issue on PR Merge

on:
  pull_request:
    types: [closed]

jobs:
  close_issue:
    runs-on: ubuntu-latest
    steps:
      - name: Close linked Issue
        uses: peter-evans/close-issue@v2
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          comment: "Este issue ha sido resuelto en el PR #${{ github.event.pull_request.number }}"
```
✅ **Explicación:**  
- **`on: pull_request: types: [closed]`** → Se ejecuta cuando se fusiona un PR.  
- **`peter-evans/close-issue@v2`** → Cierra automáticamente los Issues vinculados.  
- **Agrega un comentario** informando que el PR resolvió el Issue.  

### **3️⃣ Casos de Uso Comunes**  
🔹 **Asignar automáticamente Issues a un usuario específico.**  
🔹 **Etiquetar Issues basados en palabras clave.**  
🔹 **Publicar notificaciones en Slack cuando se cierre un Issue.**  
🔹 **Reabrir Issues si el bug no se resolvió correctamente.**  

### 🎯 **Conclusión**  
✅ **Automatizar GitHub Projects** ahorra tiempo y mejora la organización.  
✅ Las **reglas predefinidas** son ideales para configuraciones rápidas.  
✅ **GitHub Actions** permite personalizaciones avanzadas con código.

### Resumen

Automatizar tareas en proyectos de software es esencial para ahorrar tiempo y mejorar la productividad. Con GitHub Projects, puedes configurar flujos de trabajo automáticos que simplifican la gestión y seguimiento de actividades, permitiendo un enfoque directo en el código.

### ¿Cómo vincular y personalizar un proyecto en GitHub?

1. Accede a tu repositorio en GitHub y selecciona la categoría de “Projects.”
2. Si no tienes un proyecto vinculado, selecciona la opción “Enlazar a un proyecto.”
3. Edita el proyecto sin título agregando un nombre relevante, como “Mi proyecto individual,” y, opcionalmente, una descripción y un README.
4. Guarda los cambios y regresa al repositorio para enlazar este proyecto.

### ¿Cómo gestionar y actualizar actividades dentro del proyecto?

- Dentro de tu proyecto vinculado, crea actividades como “Actualizar archivo HTML,” “Actualizar archivo CSS,” o “Actualizar archivo JavaScript.”
- Marca el estado de cada tarea: en progreso, pendiente o completada.
- Usa la sección “Insights” para ver un gráfico del estado de las actividades y medir la eficacia del equipo.

### ¿Cómo automatizar los cambios de estado en actividades?

- Entra en los flujos de trabajo (Workflows) seleccionando los tres puntos en la esquina superior de tu proyecto.
- Configura las reglas, por ejemplo, para que un issue o pull request cerrado cambie automáticamente el estado de la actividad a “Hecho.”
- Personaliza otros workflows como el cambio de estado cuando un pull request es aprobado, asegurando que la automatización se adapte a tus necesidades.

### ¿Cómo crear y enlazar un issue desde una actividad?

1. Selecciona una actividad como “Actualizar archivo CSS,” presiona los tres puntos y conviértela en un issue en el repositorio.
2. Crea una nueva rama desde la sección de ramas, nómbrala de forma clara, y agrega los cambios necesarios en el archivo, por ejemplo, un nuevo archivo style.css.
3. Guarda los cambios, crea un pull request y describe los cambios. Usa la palabra clave closes seguido del número de issue para que GitHub lo cierre automáticamente cuando se apruebe el pull request.

### ¿Qué ventajas ofrece el flujo automatizado en GitHub?

Con esta automatización:

- El estado de las tareas se actualiza solo, sin necesidad de hacerlo manualmente.
- Los workflows pueden expandirse para notificar por Slack, Teams o correo electrónico cada vez que un pull request se cierra, facilitando la comunicación y el seguimiento en equipo.
- GitHub Projects, junto con estas integraciones, permite un flujo de trabajo robusto y ágil.

**Lecturas recomendadas**

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

[Planning and tracking with Projects - GitHub Docs](https://docs.github.com/en/issues/planning-and-tracking-with-projects)

[GitHub Issues · Project planning for developers · GitHub](https://github.com/features/issues)

## Documentación en GitHub con Markdown

### Herramientas útiles para documentación con Markdown

En las clases anteriores has visto la relevancia de trabajar con Markdown y lo mucho que este lenguaje te puede ayudar para crear una gran documentación. En esta clase lo que veremos son algunos de los muchísimos recursos que puedes utilizar para poder escribir de una gran manera utilizando Markdown de la mejor manera posible. ¡Comencemos!

### [Documentación de Markdown](https://www.markdownguide.org/ "Documentación de Markdown")

Simplemente, la mejor referencia para conocer todo lo que se puede hacer con Markdown dentro de los documentos, aquí puedes comenzar a leer mucho.

![Markdown page](images/Markdownpage.png)

Mi primera sugerencia es irte a la opción de Cheat Sheet, en esta sección podrás encontrar todo lo que puedes hacer, desde la sintaxis básica hasta la extendida. Lo mejor que puedes hacer comenzar a practicar aquí con esto, la verdad es que si sabes usar estas características ya estás dominando el 90% de todo el trabajo.

También considera que Markdown es compatible con algunas funciones de html como , lo que te permitiría jugar un poco más con el diseño de tu documento.

Si tienes un poco más de tiempo libre estaría fenomenal visitar la sección de Get Started en donde el sitio explica como funciona Markdown lo que es una lectura muy buena para aprender un poco más. ¡Dale un vistazo!

### Extensión de Markdown para VS Code

Ya que conoces lo que Markdown puede hacer y su sintaxis lo mejor que puedes hacer es instalar la extensión de Markdown dentro de VS Code, esto te puede llevar a un nivel mucho más avanzado de documentación porque te puede ayudar con la estructura del proyecto mostrándote las reglas que es recomendable no dejar en el documento.

Puedes encontrar el enlace de la extensión [aquí](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint "aquí").

Dentro de VS Code la imagen es como la siguiente:

![Markdownlint](images/Markdownlint.png)

Una vez que lo hayas instalado entonces es momento de ponerla en prueba y para ello debes simplemente cometer un par de errores al momento de escribir tu documento. Yo lo hice con este que ahora estás leyendo. Podrás ver las líneas amarillas en cada línea por corregir.

![Markdownlint](images/Markdownlint1.png)

¿Quieres lo mejor? Solo basta que te coloques encima de las líneas para que puedas conocer el error que puedes corregir.

![Markdownlint1](images/Markdownlint2.png)

Solo es cosa de que veas la regla y la modifiques, te debo confesar que esta extensión me ha hecho aprender a redactar de manera más eficiente mis documentos. ¡Me encantaría recordar quién me la enseñó para poder agradecerle por el gran tip!

### Previsualización de Markdown

Dentro de VS Code puedes previsualizar todos los documentos markdown antes de colocarlos en un control de versiones, solo es necesario que te ubiques en la esquina superior derecha para encontrar un ícono con una lupa y que te permite previsualizar el documento.

![VS Code](images/VS_Code.png)

Al hacerlo podrás ver una división entre el documento que estás editando y su presentación final dándote no solo una vista previa, sino que también podrá mostrar cualquier error como una ruta de imágenes mal direccionada o cosas por el estilo.

![VS Code](images/VS_Code1.png)

Usar esta vista es un recurso que puedes utilizar para muchas opciones, como evitar un commit que repare los errores de uno anterior. Lo importante es que si usas el monitor de una laptop podrá ser un poco complicado y es aquí donde podrás ansiar tener un monitor ultra wide para trabajar con total felicidad (¡yo quiero uno de esos!).

### Diagramas Mermaid

Dejando de lado la funcionalidad básica de lo que puedes hacer con los markdown y VS Code podemos dar un paso adelante y utilizar una herramienta que te hará hacer documentos de otro nivel con los diagramas mermaid.

Estos diagramas te permiten diseñar gráficas de muchos niveles y personalizarlas con la complejidad que deseas.

Por ejemplo, gracias a un código similar al siguiente podrás representar el flujo de interacción entre diferentes ramas, muy acorde a nuestro curso ¿no?

```shell
gitGraph
    commit
    commit
    branch develop
    checkout develop
    commit
    commit
    checkout main
    merge develop
    commit
    commit
```

Al insertar el código en tu documento podrás ver el resultado luciendo como esta imagen.

![VS Code](images/VS_Code2.png)

Hacer diagramas así es muy útil para representar flujos de trabajo de una manera visual y mucho más cómodos de entender, además, una ventaja adicional, es que no se requiere ninguna instalación o configuración adicional, simplemente agregas el diagrama y todo aparece de maravilla.

Para poder jugar más con el código mermaid en tus documentos, lo mejor es [visitar el visualizador de diagramas de mermaid aquí](https://mermaid.live/edit "visitar el visualizador de diagramas de mermaid aquí").

Ojalá te animes a usar todas estas herramientas para hacer lo que todo desarrollador de software debe hacer ¡Una gran documentación!

## Personaliza tu Perfil de GitHub con Markdown


En **GitHub**, puedes crear un perfil atractivo y profesional utilizando **Markdown**. Un perfil bien diseñado ayuda a destacar tu trabajo, proyectos y habilidades.  

### **1️⃣ ¿Cómo crear un perfil personalizado en GitHub?**  

### 📌 **Paso 1: Crear un Repositorio Especial**  
1️⃣ Ve a [GitHub](https://github.com/) y haz clic en **"New Repository"**.  
2️⃣ Nombra el repositorio **exactamente igual** a tu usuario de GitHub (Ejemplo: `MarioVargasCelis`).  
3️⃣ Marca la opción **"Public repository"**.  
4️⃣ Activa **"Add a README file"** y haz clic en **"Create repository"**.  

GitHub reconocerá este repositorio como tu perfil personal y mostrará el contenido del `README.md` en tu perfil.  

### **2️⃣ Personaliza tu README con Markdown**  

Tu `README.md` es la base de tu perfil. Aquí tienes algunos elementos clave para mejorar su presentación:  

### **📌 Encabezado con una Bienvenida**
```md
# ¡Hola, soy Mario Vargas Celis! 👋
🚀 Ingeniero Mecatrónico | 🤖 IA & Data Science | 💻 Desarrollo Web  
```

### **📌 Agregar una Imagen o GIF**
```md
![Mi Banner](https://mi-imagen.com/banner.png)
```

💡 **Tip:** Puedes subir imágenes a un Issue en GitHub y usar su URL.

### **3️⃣ Agrega Secciones Claves**  

### **📌 Sobre Mí**
```md
### 📌 Sobre mí  
Soy un apasionado de la tecnología, la inteligencia artificial y el desarrollo de software. Me encanta trabajar con **Python, Machine Learning y GitHub Actions**.
```

### **📌 Habilidades Técnicas**
```md
### 🚀 Habilidades  
- 🔹 **Lenguajes:** Python, JavaScript, C++  
- 🔹 **Frameworks:** Django, React, PyTorch  
- 🔹 **Herramientas:** Git, Docker, VS Code  
```

### **📌 Proyectos Destacados**
```md
### 📂 Proyectos Destacados  
🌟 [Mi Portafolio Web](https://miportafolio.com) - Hecho con React y TailwindCSS  
🌟 [Análisis de Datos con Pandas](https://github.com/usuario/proyecto)  
🌟 [Modelo de IA para clasificación](https://github.com/usuario/ia-clasificacion)  
```

### **📌 Estadísticas de GitHub**
Puedes usar widgets dinámicos como **GitHub Stats**:
```md
![Estadísticas de GitHub](https://github-readme-stats.vercel.app/api?username=tu-usuario&show_icons=true&theme=dark)
```

### **4️⃣ Agrega Contacto y Redes Sociales**
```md
## 📫 Contáctame  
💼 [LinkedIn](https://www.linkedin.com/in/mi-perfil/)  
📧 **Correo:** mario@ejemplo.com  
🌐 [Mi Portafolio](https://miportafolio.com)  
```

💡 **Tip:** Usa **badges** con Shields.io:
```md
![LinkedIn](https://img.shields.io/badge/LinkedIn-Perfil-blue?logo=linkedin)
```

### **🎯 Conclusión**  
✅ **Personalizar tu perfil de GitHub** con Markdown lo hace más profesional.  
✅ Puedes agregar **imágenes, estadísticas y enlaces** para destacar.  
✅ Un perfil bien diseñado te ayuda a **conseguir oportunidades laborales**.  

### Resumen

Aprender a crear una portada atractiva y funcional en GitHub usando Markdown no solo mejora la presentación profesional de un perfil, sino que también permite personalizarlo con elementos visuales, enlaces y badges que muestran actividad e información de contacto. Aquí encontrarás un paso a paso para utilizar Markdown y hacer que tu perfil destaque.

### ¿Cómo iniciar el repositorio especial en GitHub?

- **Crear un repositorio**: Ve a la sección de repositorios y crea uno nuevo usando el mismo nombre que tu nombre de usuario en GitHub; esto permite que el repositorio funcione como portada.
- **Descripción y visibilidad**: Añade una descripción breve (ej. “Portada de perfil”), hazlo público y agrega un archivo README.md. Esto es esencial para que el contenido sea visible en tu perfil.
- **Clonación del repositorio**: Clona el repositorio usando git clone en la terminal. La opción HTTPS es conveniente para nuevos perfiles, aunque SSH es más seguro.

### ¿Cómo personalizar el README con Markdown?

Markdown facilita la creación de secciones y elementos visuales. Puedes agregar:

- **Títulos y subtítulos**: Usa `#`, `##` o `###` según la jerarquía. Por ejemplo, `## Contacto`.
- **Enlaces y badges**: Incluye enlaces usando `[Texto](URL)`. Para badges, visita [shields.io](https://shields.io/ "shields.io") donde encontrarás diferentes opciones (ej. actividad de commits, sitios web).
- **Iconos y emojis**: Puedes agregar emojis como `:computer:` o `:pencil:` para destacar roles o actividades.

### ¿Cómo previsualizar y ajustar el archivo en Visual Studio Code?

- **Vista previa de Markdown**: Selecciona Open Preview en la esquina superior derecha para ver cómo se verán los cambios en GitHub.
- **Extensiones recomendadas**: Markdown Lint ayuda a mejorar el estilo al sugerir tips sobre el formato, como evitar espacios en blanco innecesarios.
- **Limpieza del código**: Markdown Lint también ayuda a mantener el archivo ordenado eliminando líneas en blanco que no son necesarias.

### ¿Cómo añadir y ajustar badges en el perfil de GitHub?

1. Visita shields.io y busca categorías de badges como:
 - **Website**: Permite agregar un enlace a tu sitio web.
 - **Actividad de Commits**: Muestra la frecuencia de tus commits en GitHub. Puedes seleccionar el intervalo: `weekly`, `monthly`, `yearly`, o `total`.
2. **Insertar badges en Markdown**: Selecciona “Markdown” en [shields.io](http://shields.io/ "shields.io") para obtener el código y pégalo en el `README.md`.
3. **Prueba de visualización**: Asegúrate de que los badges se muestren correctamente en Visual Studio Code.

### ¿Qué estrategias aplicar para mejorar la portada en GitHub?

- **Referencias a otros repositorios**: Examina perfiles de otros usuarios o proyectos con buena documentación. Visualiza su archivo README.md en modo “RAW” para ver el código en Markdown.
- **Explora y adapta**: La práctica es clave; revisa diferentes perfiles para encontrar ideas y técnicas que puedas adaptar.

**Lecturas recomendadas**

[Sintaxis de escritura y formato básicos - Documentación de GitHub](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

[Static Badge | Shields.io](https://shields.io/badges)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Uso de Wikis

En las clases hemos visto como utilizar el archivo [README.md](http://readme.md/ "README.md") para mostrar la documentación del proyecto, con el tiempo esta práctica ha ganado cada vez más adopción por su sencillez, pero eso no es la única manera de crear documentación.

Para ello existe dentro de Github la opción de crear una Wiki en donde puedes generar un nivel más estructurado de documentación. Puedes ver la sección de Wiki en tus proyectos en la sección superior del portal de Github.

![Wiki](images/Wiki.png)

Si seleccionas esta opción entonces podrás ver un botón que te invita a crear tu primera página, hazlo presiónalo.

![Primera página](images/Primera_pagina.png)

El formulario te da la opción de crear una nueva página a la que llama Home lo que es una gran opción, puedes usar esta página para mostrar la documentación inicial, pero también la puedes usar como un índice para poder llevar a tu lector a diferentes secciones y eso es lo que vamos a hacer. Escribe lo siguiente en tu formulario.

```rst
# ¡Bienvenido a la wiki!

Aquí podrás encontrar todas las secciones para poder implementar tu proyecto de manera rápida y simple.

## Índice de navegación

[Explicación del proyecto](/proyecto.md)
```

Si presionas el botón de guardar los cambios, el resultado de esta primera edición es igual al de esta imagen.

![Home](images/Home.png)

Repite la misma operación, ahora con una nueva página llamada Proyecto, en su descripción puedes agregar cualquier contenido, en la imagen que sigue puedes ver un ejemplo de como debe lucir tu nueva página.

![Proyecto](images/Proyecto.png)

Si guardas los cambios, entonces el resultado de la estructura de tu wiki lucirá de la siguiente manera.

![Páginas](images/Paginas.png)

Comienza a crear algunas páginas, no te diré cuantas ni con qué nombres para que te diviertas, lo que sí haré es mostrarte como luce mi wiki al final.

![Páginas 2](images/Paginas1.png)

Ahora vuelve a tu página Home en donde agregaste un poco de texto y además un enlace, en la esquina superior derecha hay un botón con el que puedes editarla. En el código markdown de aquí abajo podrás ver una manera fácil en la que puedes navegar entre secciones.

```rst
# ¡Bienvenido a la wiki!

Aquí podrás encontrar todas las secciones para poder implementar tu proyecto de manera rápida y simple.

## Índice de navegación

[Explicación del proyecto](./Proyecto)

[Arquitectura](./Arquitectura)

[Documentación](./Documentación)
```

Con este índice es fácil que tus usuarios puedan navegar entre las secciones de una manera cómoda, lo que estaría fenomenal es que ahora les facilites a tus usuarios volver a la sección principal en cada una de tus secciones para que la navegación se vuelva cíclica y así les sea muy fácil moverse entre todas las secciones. ¿Cómo harías eso?

Volvamos a la pantalla principal de la wiki y observa que debajo del menú de páginas está una sección que te permite crear una barra lateral personalizada, selecciona esta opción.

![Sidebar](images/Sidebar.png)

Al hacerlo notarás un formulario idéntico a los anteriores, solo que con un título diferente en donde podrás personalizar todos los detalles de la barra lateral, intenta copiar y pegar aquí el mismo markdown que acabamos de usar en la página Home. ¡Oh! Cierto, no cambies el título, la palabra _Sidebar es lo que permite que Github sepa que estamos hablando de una barra lateral y no de otra sección más.

![Custom Sidebar](images/Custom_Sidebar.png)

Guarda tus cambios y disfruta de tu nueva barra de navegación.

![Custom Sidebar 1](images/Custom_Sidebar1.png)

Una característica superinteresante es que puedes clonar esta wiki dentro de tu entorno local sin mayor problema, observa que hacer esto significa que solo vas a clonar todos estos documentos y no vas a hacer lo mismo con el repositorio lo que se me hace superinteresante porque puede ser que el portal de Github sea fantástico, pero no tanto como para pasar ahí horas leyendo documentos por lo que de esta manera puedes hacerlo desde tu lector de documentos Markdown favorito.

¡Invierte tiempo en tus wikis! Visita las de otros proyectos y toma muchas ideas de ahí, practica mucho con tu documentación aprendiendo a usar el lenguaje Markdown y cuando tengas una wiki fantástica no olvides compartirla con todos nosotros.

## Cómo usar GitHub Gist

**GitHub Gist** es una herramienta de GitHub que permite crear y compartir fragmentos de código, notas y documentos en formato **Markdown**. Es ideal para almacenar pequeñas piezas de código reutilizable, documentar ideas y compartir con otros desarrolladores.  

### **1️⃣ ¿Qué es GitHub Gist y para qué sirve?**  

✅ Guardar y compartir fragmentos de código.  
✅ Crear notas en **Markdown** con documentación técnica.  
✅ Publicar snippets de código accesibles desde cualquier lugar.  
✅ Colaborar con otros a través de comentarios en los Gists.  

### **2️⃣ Crear un Gist en GitHub**  

### 📌 **Paso 1: Acceder a GitHub Gist**  
🔹 Ve a [GitHub Gist](https://gist.github.com/).  
🔹 Inicia sesión con tu cuenta de GitHub.  

### 📌 **Paso 2: Crear un Nuevo Gist**  
1️⃣ Haz clic en **"New Gist"**.  
2️⃣ Escribe un **nombre de archivo** con su extensión (ejemplo: `script.py`).  
3️⃣ Agrega el **código o texto** dentro del editor.  
4️⃣ Opcionalmente, agrega una **descripción** para explicar el contenido.  

### 📌 **Paso 3: Elegir la Privacidad**  
🔹 **Público:** Visible para todos.  
🔹 **Secreto:** Solo accesible con el enlace directo.  

5️⃣ Haz clic en **"Create secret gist"** o **"Create public gist"**.  

### **3️⃣ Administrar y Compartir un Gist**  

### **📌 Compartir un Gist**  
✅ Cada Gist tiene una **URL única** para compartir fácilmente.  
✅ Puedes incrustarlo en blogs o foros con:  
```md
<script src="https://gist.github.com/tu-usuario/gist-ID.js"></script>
```

### **📌 Editar y Eliminar un Gist**  
🔹 Para editarlo, abre el Gist y haz clic en **"Edit"**.  
🔹 Para eliminarlo, haz clic en **"Delete"** (⚠️ No se puede recuperar después).  

### **4️⃣ Usar GitHub Gist con Git**  

Puedes clonar un Gist en tu computadora como un repositorio Git:  
```sh
git clone https://gist.github.com/tu-usuario/gist-ID.git
```
Luego puedes editar archivos, hacer commits y subir cambios:  
```sh
git add .
git commit -m "Actualización del Gist"
git push origin main
```

### **🎯 Conclusión**  
✅ **GitHub Gist** es una excelente herramienta para compartir código y notas.  
✅ Soporta **Markdown**, código de múltiples lenguajes y control de versiones.  
✅ Se puede **clonar, editar y actualizar** con Git.  

### Resumen

GitHub Gist permite compartir y discutir fragmentos de código de forma sencilla, sin necesidad de crear un repositorio completo. Esta herramienta es ideal para obtener retroalimentación rápida y colaborativa sin comprometer los cambios en un proyecto principal.

### ¿Qué es GitHub Gist y cómo se utiliza?
GitHub Gist es una funcionalidad de GitHub diseñada para almacenar y compartir pequeños fragmentos de código. A diferencia de un repositorio tradicional, un Gist no se vincula a un proyecto completo, sino que permite discutir una pieza de código de manera aislada, ideal para colaboración rápida.

- **Crear un Gist**: Ingresa a gist.github.com, pega el fragmento de código y añade una descripción breve.
- **Compartir el enlace**: Copia la URL generada y compártela con tus colaboradores para abrir la discusión.
- **Feedback en tiempo real**: Los colaboradores pueden comentar directamente en el Gist, permitiendo iteraciones y mejoras rápidas.

### ¿Cómo se usa GitHub Gist para colaboración?
La simplicidad de los Gists facilita el trabajo en equipo al ofrecer un espacio directo de intercambio de ideas y mejoras sin alterar el proyecto base.

- **Conversación activa**: Puedes recibir y responder comentarios sobre el fragmento de código.
- **Actualización en tiempo real**: Si el colaborador sugiere cambios, puedes editar el Gist y mejorar el código sin necesidad de crear nuevas ramas.
- **Ventajas en pair programming**: Un Gist puede ser usado como base en sesiones de pair programming, manteniendo el enfoque en mejoras puntuales y rápidas.

### ¿Cómo se gestionan los Gists en GitHub?

GitHub permite gestionar y organizar fácilmente los Gists en tu perfil, lo que facilita tener una colección de snippets reutilizables.

- **Acceso rápido**: Los Gists se encuentran en tu perfil y pueden organizarse en una colección para referencias futuras.
- **Eliminar Gists innecesarios:** Si un Gist ya no es útil, puede eliminarse sin afectar otros proyectos.
- **Edición y actualización**: Los Gists pueden editarse directamente para mantener el código actualizado según las necesidades del proyecto.

### ¿Qué beneficios adicionales ofrece GitHub Gist?

Además de la colaboración, los Gists son útiles para mantener una biblioteca personal de snippets de código, mejorando la eficiencia en nuevos proyectos.

- **Biblioteca personal**: Guarda configuraciones iniciales o fragmentos reutilizables para evitar escribir código repetitivo.
- **Probar ideas antes de integrarlas**: Permite experimentar con variantes de código antes de incorporarlas oficialmente.
- **Ahorro de tiempo**: Facilita el acceso y reutilización de código en proyectos similares, optimizando el flujo de trabajo.

**Lecturas recomendadas**

[Discover gists · GitHub](https://gist.github.com/)

[Crear gists - Documentación de GitHub](https://docs.github.com/es/get-started/writing-on-github/editing-and-sharing-content-with-gists/creating-gists)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Cómo usar GitHub Pages

**GitHub Pages** es una herramienta gratuita de GitHub que permite **hospedar sitios web estáticos** directamente desde un repositorio de GitHub. Es ideal para mostrar proyectos, crear blogs o publicar portafolios personales.  

### **1️⃣ Configurar GitHub Pages en un Repositorio**  

### **📌 Paso 1: Crear o Usar un Repositorio**  
1️⃣ Ve a [GitHub](https://github.com/) y crea un nuevo repositorio o usa uno existente.  
2️⃣ Sube los archivos de tu sitio web (**HTML, CSS, JavaScript, etc.**).  

### **📌 Paso 2: Activar GitHub Pages**  
1️⃣ Ve a la pestaña **Settings** del repositorio.  
2️⃣ Desplázate hasta **Pages** en la barra lateral izquierda.  
3️⃣ En la sección **"Source"**, selecciona la rama donde está tu código (`main` o `gh-pages`).  
4️⃣ Haz clic en **Save**.  

✅ **¡Listo!** Tu sitio estará disponible en `https://tu-usuario.github.io/nombre-del-repositorio/` después de unos minutos.  

### **2️⃣ Usar GitHub Pages con Jekyll (Opcional para Blogs)**  
Si quieres un blog sin programar demasiado, GitHub Pages soporta **Jekyll**, un generador de sitios estáticos basado en Markdown.  

1️⃣ Crea un repositorio con un archivo `_config.yml`.  
2️⃣ Usa un tema de Jekyll agregando esto a `config.yml`:  
   ```yaml
   theme: minima
   ```  
3️⃣ Sube archivos `.md` y GitHub Pages generará tu blog automáticamente.  

### **3️⃣ Personalizar tu Dominio con GitHub Pages**  
Si tienes un dominio propio, puedes vincularlo a tu sitio:  

1️⃣ Ve a **Settings > Pages** y agrega tu dominio en **Custom Domain**.  
2️⃣ Configura los registros `CNAME` en tu proveedor de dominio.  

### **🎯 Conclusión**  
✅ **GitHub Pages** es una forma fácil y gratuita de hospedar sitios web.  
✅ Puedes usarlo con **HTML puro, frameworks o Jekyll**.  
✅ Se integra perfectamente con repositorios y control de versiones.  

**Lecturas recomendadas**

[GitHub Pages | Websites for you and your projects, hosted directly from your GitHub repository. Just edit, push, and your changes are live.](https://pages.github.com/)

[Guía de inicio rápido para GitHub Pages - Documentación de GitHub](https://docs.github.com/es/pages/quickstart)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Cómo usar GitHub Codespaces: Configuración y Desarrollo en la Nube

**GitHub Codespaces** es un entorno de desarrollo en la nube basado en **Visual Studio Code**, que te permite escribir, compilar y ejecutar código directamente desde el navegador sin necesidad de configuraciones locales.  

### **1️⃣ Habilitar GitHub Codespaces**  

### 📌 **Requisitos**  
✅ Tener una cuenta en [GitHub](https://github.com/).  
✅ Ser parte de un plan que incluya Codespaces (usuarios gratuitos tienen horas limitadas).  
✅ Usar un repositorio donde quieras trabajar.  

### 📌 **Cómo crear un Codespace**  
1️⃣ Ve al repositorio en GitHub.  
2️⃣ Haz clic en el botón **"Code"** (verde).  
3️⃣ Selecciona la pestaña **"Codespaces"**.  
4️⃣ Haz clic en **"Create codespace on main"**.  
5️⃣ Espera unos segundos y tendrás un entorno **VS Code en el navegador** listo para programar.

### **2️⃣ Personalizar el Entorno de Desarrollo**  

GitHub Codespaces permite personalizar el entorno con archivos de configuración:  

### 📌 **Instalar Extensiones**  
Puedes agregar extensiones de **VS Code** desde la barra lateral.  

### 📌 **Configurar con un Devcontainer**  
Puedes definir un entorno específico agregando un archivo `.devcontainer/devcontainer.json`:  
```json
{
  "name": "Mi Codespace",
  "image": "mcr.microsoft.com/devcontainers/python:3.9",
  "extensions": ["ms-python.python"]
}
```
Esto configura un entorno de desarrollo con Python 3.9 y extensiones preinstaladas.  

### **3️⃣ Trabajar con Git y GitHub**  

### **📌 Clonar un Repositorio**  
Si creaste un Codespace vacío, puedes clonar un repositorio con:  
```sh
git clone https://github.com/usuario/repositorio.git
```

### **📌 Subir Cambios**  
Después de hacer modificaciones, usa:  
```sh
git add .
git commit -m "Actualización del código"
git push origin main
```

## **4️⃣ Ejecutar y Depurar Código**  

✅ Puedes ejecutar código desde la terminal integrada (`Ctrl + ` `)  
✅ Puedes usar **depuración interactiva** con `F5` en VS Code  
✅ Soporta múltiples lenguajes como **Python, JavaScript, C++ y más**  

### **🎯 Conclusión**  
✅ **GitHub Codespaces** te permite **desarrollar sin instalar nada localmente**.  
✅ Es una opción ideal para **colaborar en proyectos sin problemas de configuración**.  
✅ Puedes personalizar el entorno con **devcontainers** y usar **VS Code en la nube**.

### Resumen

GitHub Codespaces es una herramienta poderosa que permite crear y gestionar entornos de desarrollo en la nube, aumentando la flexibilidad y productividad para desarrolladores en cualquier lugar. Con una interfaz similar a Visual Studio Code, Codespaces permite desarrollar proyectos desde cualquier dispositivo, sin importar si está instalado el entorno completo en la máquina local.

### ¿Qué es GitHub Codespaces y cómo funciona?

GitHub Codespaces ofrece entornos de desarrollo alojados en máquinas virtuales en la nube. Esto permite a los desarrolladores trabajar desde cualquier dispositivo, como una tableta o teléfono, en proyectos alojados en repositorios de GitHub. Con acceso a herramientas de compilación y despliegue, se puede trabajar con múltiples lenguajes de programación sin necesidad de instalarlos localmente.

### ¿Cómo se crea un Codespace?

Para iniciar un Codespace:

- Selecciona “New Codespace” en el menú.
- Escoge el repositorio en el que vas a trabajar.
- Elige la rama y región que prefieras.
- Configura el tipo de máquina virtual, seleccionando entre diferentes núcleos y memoria RAM según la necesidad del proyecto.

Una vez creado, se abre una interfaz de desarrollo completa, que incluye explorador de archivos, terminal integrada y control de versiones.

### ¿Cuáles son las características clave de Codespaces?

Algunas funcionalidades destacadas incluyen:

- **Explorador de archivos y extensiones**: permite administrar archivos y añadir extensiones, igual que en VS Code.
- **Terminal integrada**: facilita ejecutar comandos sin salir del entorno.
- **Preconfiguración de entornos**: lenguajes como Node, Python y .NET ya están instalados, permitiendo trabajar sin configurarlos en la máquina local.

### ¿Cómo se gestiona un proyecto en Codespaces?

Puedes crear y gestionar proyectos en múltiples lenguajes. Por ejemplo, para un proyecto en .NET:

- Crea una carpeta para el proyecto en la rama del repositorio.
- Usa comandos como `.NET new console` para iniciar el proyecto, sin preocuparte por tener .NET instalado localmente.
- La terminal permite ejecutar el proyecto con `.NET run` y hacer ajustes sobre la marcha.

### ¿Cómo hacer commit y sincronizar cambios?

Al realizar cambios en el proyecto, estos se pueden ver en la sección de control de versiones:

1. Añade un mensaje de commit que describa los cambios.
2. Sincroniza los cambios para reflejarlos en el repositorio de GitHub.

Esta integración asegura que cada modificación quede guardada y reflejada en GitHub, como si se hubiese hecho en un entorno local.

### ¿Por qué es importante eliminar Codespaces al terminar?

Cada Codespace utiliza recursos de GitHub y, en cuentas gratuitas, existe un límite de 120 horas de uso al mes. Al completar una tarea:

- Elimina el Codespace para evitar cargos adicionales.
- Desde “My Codespaces”, selecciona el Codespace y elige “delete” para confirmar la eliminación.

Este proceso garantiza que solo uses el tiempo necesario y no excedas el límite de la cuenta gratuita.

**Lecturas recomendadas**

[Codespaces documentation - GitHub Docs](https://docs.github.com/en/codespaces)

[Codespaces · GitHub](https://github.com/codespaces)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Cómo configurar entornos de desarrollo avanzados en GitHub Codespaces

**GitHub Codespaces** permite personalizar el entorno de desarrollo utilizando **containers de desarrollo (Dev Containers)**. Esto es útil para proyectos complejos que requieren herramientas específicas, dependencias o configuraciones avanzadas.  

### **1️⃣ Habilitar GitHub Codespaces en un Proyecto**  
Antes de personalizar el entorno, asegúrate de que GitHub Codespaces está habilitado en tu repositorio:  
1. Ve a tu repositorio en GitHub.  
2. Haz clic en el botón **"Code"** y selecciona la pestaña **"Codespaces"**.  
3. Crea un nuevo Codespace en la rama principal (`main`).  

### **2️⃣ Personalizar el Entorno con un Dev Container**  
Para configurar un entorno avanzado, debes agregar un **devcontainer** a tu repositorio.  

### 📌 **Paso 1: Crear la Carpeta de Configuración**  
En la raíz del repositorio, crea una carpeta llamada **`.devcontainer/`** y dentro de ella, un archivo llamado **`devcontainer.json`**.  

### 📌 **Paso 2: Definir la Configuración del Contenedor**  
Aquí tienes un ejemplo de `devcontainer.json` para un entorno de desarrollo con **Python 3.9 y Node.js**:  
```json
{
  "name": "Mi Entorno de Desarrollo",
  "image": "mcr.microsoft.com/devcontainers/python:3.9",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {}
  },
  "extensions": [
    "ms-python.python",
    "ms-vscode.vscode-typescript-tslint-plugin"
  ],
  "postCreateCommand": "pip install -r requirements.txt"
}
```
🔹 **Explicación:**  
✅ Usa una imagen de contenedor con **Python 3.9**.  
✅ Agrega soporte para **Node.js**.  
✅ Instala extensiones de VS Code.  
✅ Ejecuta `pip install -r requirements.txt` después de crear el Codespace.  

### **3️⃣ Configurar Variables de Entorno**  
Si tu proyecto necesita credenciales o claves API, puedes agregarlas en `devcontainer.json`:  
```json
"remoteEnv": {
  "API_KEY": "tu_clave_aqui"
}
```
También puedes definirlas en **GitHub Secrets** para mayor seguridad.  

### **4️⃣ Instalar Dependencias Automáticamente**  
Puedes agregar un script de configuración en `.devcontainer/postCreateCommand.sh` para instalar dependencias automáticamente:  
```sh
#!/bin/bash
echo "Instalando dependencias..."
pip install -r requirements.txt
npm install
```
Luego, agrégalo en `devcontainer.json`:  
```json
"postCreateCommand": "./.devcontainer/postCreateCommand.sh"
```

### **5️⃣ Trabajar con Bases de Datos y Servicios**  
Si tu proyecto necesita **MySQL, PostgreSQL o Redis**, puedes definirlos en `.devcontainer/docker-compose.yml`:  
```yaml
version: '3'
services:
  db:
    image: postgres:13
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydatabase
```
Y en `devcontainer.json`:  
```json
"dockerComposeFile": ".devcontainer/docker-compose.yml",
"service": "app"
```

### **🎯 Conclusión**  
✅ **GitHub Codespaces** permite configurar entornos avanzados sin necesidad de instalar nada localmente.  
✅ Puedes personalizar el entorno con **Dev Containers**, instalar dependencias automáticamente y conectar servicios externos.  
✅ Esto es ideal para proyectos complejos, colaborativos y con necesidades específicas de configuración.  

### Resumen

Los Codespaces de GitHub permiten una experiencia de desarrollo altamente personalizable sin necesidad de configurar entornos locales, gracias a su integración de plantillas y configuraciones específicas. Aquí exploramos cómo aprovechar al máximo estas plantillas, como una basada en Django con Python, y cómo ajustar la configuración para mejorar la experiencia en proyectos colaborativos o personales.

### ¿Cómo encontrar y utilizar plantillas de Codespaces?

1. Accede a tus Codespaces en [github.com/codespaces](github.com/codespaces "github.com/codespaces") para ver todos los entornos que has creado.
2. En lugar de crear un Codespace desde cero, selecciona una plantilla. Por ejemplo, puedes usar una con Django, lo cual permite omitir configuraciones iniciales, ya que la plantilla incluye todo lo necesario.
3. Las plantillas permiten crear un entorno en blanco o preconfigurado para un lenguaje o framework específico, sin necesidad de cuestionarios de configuración.

### ¿Cómo configurar y editar un Codespace en tiempo real?

- Al crear un proyecto con una plantilla, como Django, puedes modificar el código en tiempo real. Los cambios, como actualizar el texto en un archivo HTML, se reflejan sin necesidad de compilar, como sería el caso en entornos .NET.
- Para probar la visibilidad de tu proyecto, puedes copiar la URL pública que se genera para compartir avances en tiempo real.

### ¿Qué configuraciones están disponibles en el archivo devcontainer.json?

Dentro de cada plantilla, encontrarás una carpeta `.devcontainer` que contiene el archivo `devcontainer.json`. Este archivo:

- Define el entorno que tu Codespace utilizará, configurando lenguajes y herramientas específicos, como Python en el caso de un proyecto Django.
- Permite agregar extensiones de Visual Studio Code necesarias para el proyecto. Por ejemplo, al agregar la extensión “Live Share”, puedes activarla en el archivo `devcontainer.json` para que esté disponible en futuras sesiones.

### ¿Cómo optimizar el desarrollo remoto en Visual Studio Code Desktop?

1. Usa la opción “Abrir en Visual Studio Code Desktop” para una experiencia completa de edición fuera del navegador.
2. Instala la extensión de GitHub Codespaces en Visual Studio si es la primera vez que usas este entorno.
3. Asegúrate de que tu Codespace solo se ejecute en un entorno a la vez (navegador o desktop), para evitar conflictos de puerto.
4. Puedes controlar el estado del servidor desde la terminal en cualquiera de los entornos, y detener la aplicación con `Control-C` para reiniciar el servidor cuando sea necesario.

### ¿Por qué utilizar plantillas preconfiguradas en Codespaces?

- Las plantillas permiten explorar nuevas tecnologías sin necesidad de configurarlas en tu equipo local. Esto es útil para probar frameworks o lenguajes sin modificar entornos personales.
- Ejemplos de plantillas populares incluyen Jupyter para análisis de datos, Node para desarrollo de aplicaciones web, y Django para aplicaciones con Python.

Usando estas plantillas, puedes acelerar la curva de aprendizaje y concentrarte en el desarrollo sin preocuparte por la configuración inicial.

**Lecturas recomendadas**

[Ciclo de vida de un codespace - Documentación de GitHub](https://docs.github.com/es/codespaces/getting-started/understanding-the-codespace-lifecycle)

[Codespaces · GitHub](https://github.com/codespaces)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Pair Programming con Codespaces y Visual Studio Code

**Pair Programming** (Programación en Pareja) con **GitHub Codespaces y Visual Studio Code** permite colaborar en tiempo real en un mismo entorno de desarrollo, sin necesidad de configurar nada localmente.  

### **🚀 Beneficios de Pair Programming con Codespaces**
✅ **Entorno en la nube:** No es necesario instalar dependencias en cada equipo.  
✅ **Colaboración en tiempo real:** Puedes escribir y editar código con otras personas simultáneamente.  
✅ **Seguridad y control:** Puedes administrar permisos de acceso fácilmente.  

### **1️⃣ Habilitar Codespaces en el Proyecto**  
Antes de colaborar, asegúrate de que **Codespaces está habilitado** en el repositorio.  
1️⃣ Ve a tu repositorio en GitHub.  
2️⃣ Haz clic en el botón **"Code"** y selecciona **"Codespaces"**.  
3️⃣ Crea un nuevo Codespace en la rama principal (`main`).  

### **2️⃣ Compartir el Codespace con Live Share**  
**Visual Studio Code** tiene una extensión llamada **Live Share**, que permite la colaboración en tiempo real.  

### 📌 **Paso 1: Instalar Live Share**  
Dentro de tu Codespace en VS Code, abre la pestaña de **Extensiones (`Ctrl + Shift + X`)** y busca:  
🔍 **"Live Share"** ➝ Instálala.  

### 📌 **Paso 2: Iniciar una Sesión de Live Share**  
1️⃣ Haz clic en el icono de **Live Share** en la barra lateral de VS Code.  
2️⃣ Presiona el botón **"Start collaboration session"**.  
3️⃣ Copia el enlace generado y compártelo con tu compañero.  

### 📌 **Paso 3: Unirse a la Sesión**  
Tu compañero solo necesita:  
✅ Abrir **VS Code** (puede ser en local o en otro Codespace).  
✅ Instalar **Live Share**.  
✅ Abrir el enlace de la sesión en su navegador.  

📌 **Ambos pueden editar el código en tiempo real** y compartir la terminal.  

### **3️⃣ Colaborar con Git y Codespaces**  
Mientras trabajan juntos, pueden sincronizar los cambios con GitHub:  

### 📌 **Subir Cambios**  
```sh
git add .
git commit -m "Mejoras en el código"
git push origin main
```

### 📌 **Obtener Últimos Cambios**  
```sh
git pull origin main
```

⚠️ **Si hay conflictos**, GitHub te avisará para que los resuelvas antes de continuar.  

### **🎯 Conclusión**  
✅ **GitHub Codespaces + Live Share** permiten Pair Programming de manera sencilla y sin instalaciones locales.  
✅ Puedes compartir código, depurar errores y escribir en tiempo real con otros desarrolladores.  
✅ Todo se sincroniza automáticamente con **GitHub**.  

### Resumen

Trabajar en equipo y en tiempo real desde la nube se ha vuelto esencial, y GitHub Codespaces junto con la extensión de Live Share de Visual Studio Code permite precisamente eso: una colaboración dinámica y segura. Aquí te explicamos cómo puedes usar estas herramientas para hacer pair programming y trabajar en equipo sin comprometer tu espacio de desarrollo.

### ¿Cómo iniciar una sesión de colaboración en Codespaces?

Para compartir tu Codespace con otra persona, asegúrate de que la extensión de Live Share esté instalada en tu entorno de Visual Studio Code. Luego:

- Detén la ejecución de cualquier aplicación en el Codespace.
- Accede a la sección de Live Share en VS Code y selecciona “Iniciar sesión compartida”.
- Copia el enlace de invitación y compártelo con tu colaborador. Al ingresar, podrá editar y sugerir cambios en tiempo real.

Tu invitado puede unirse desde VS Code o, de preferencia, a través del navegador usando la opción “Continuar en web” si busca un acceso más rápido y liviano.

### ¿Qué visualiza el colaborador en la sesión?

El colaborador que accede mediante el enlace ve tu código en tiempo real y puede realizar modificaciones que se reflejan en tu Codespace. Puedes observar sus movimientos, las ediciones sugeridas y trabajar en conjunto sin perder control sobre el entorno, ya que todos los cambios quedan bajo tu responsabilidad en el commit.

### ¿Qué ventajas tiene trabajar en la nube con Codespaces y Live Share?

- **Colaboración segura**: Permites acceso solo al entorno de Codespaces en la nube, manteniendo tu espacio local aislado.
- **Facilidad para múltiples colaboradores**: Puedes compartir el enlace con más de un participante, y al terminar la sesión todos los cambios pueden unificarse en un solo commit.
- **Entorno unificado**: Todos los participantes trabajan con el mismo set de extensiones y configuración, lo que facilita la integración y el seguimiento del proyecto.

### ¿Cómo finalizar la sesión de colaboración?

Para detener la colaboración, ve a la sección de Live Share en tu Codespace y selecciona “Detener sesión de colaboración”. Esto retirará el acceso de los invitados y volverás a un modo de trabajo individual. Además, para cerrar por completo, regresa a GitHub Codespaces, revisa tu plantilla y elige la opción de eliminarla para liberar recursos.

### ¿Qué configuraciones adicionales puedes incluir en Codespaces?

Puedes personalizar tu Codespace añadiendo extensiones de lenguaje y herramientas de trabajo (como Markdown o Python) que optimicen tu flujo de trabajo. Estas configuraciones aseguran que, al reabrir tu Codespace, esté listo con las herramientas necesarias, siempre que las hayas especificado en el archivo `devcontainer.json`.

**Lecturas recomendadas**

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

[Live Share - Visual Studio](https://code.visualstudio.com/blogs/2017/11/15/live-share)

## Cómo usar GitHub.dev Editor 

**GitHub.dev** es un editor basado en la web que te permite modificar y visualizar archivos en repositorios de GitHub sin necesidad de clonar el proyecto ni instalar software adicional. Es una versión ligera de **Visual Studio Code** en el navegador, ideal para ediciones rápidas y colaboración en proyectos.

### **🚀 1️⃣ Acceder a GitHub.dev**
### 📌 **Método 1: Atajo de Teclado**
1. Abre un repositorio en GitHub.  
2. Presiona `.` (punto) en tu teclado.  
3. Se abrirá **GitHub.dev** en una nueva pestaña.  

### 📌 **Método 2: URL Manual**
1. Abre un repositorio en GitHub.  
2. Cambia `github.com` por `github.dev` en la barra de direcciones.  
   - Ejemplo:  
     ```
     https://github.com/usuario/repositorio
     ```
     ➝ Cambiar a:  
     ```
     https://github.dev/usuario/repositorio
     ```

### **⚙️ 2️⃣ Funcionalidades de GitHub.dev**
### **📂 Navegación y Edición**
- Puedes explorar archivos y carpetas del repositorio.  
- Hacer cambios en los archivos como en **Visual Studio Code**.  
- Compatible con resaltado de sintaxis y autocompletado.  

### **📌 3️⃣ Commit y Gestión de Archivos**
1. **Modificar archivos**: Edita cualquier archivo en el repositorio.  
2. **Hacer un commit**:  
   - Abre el **panel de control de Git** en la barra lateral izquierda.  
   - Escribe un mensaje de commit.  
   - Haz clic en **"Commit changes"**.  
3. **Trabajar en ramas**: Puedes cambiar de rama o crear una nueva desde el editor.  

### **🔄 4️⃣ Diferencias entre GitHub.dev y Codespaces**
| Característica       | **GitHub.dev** | **GitHub Codespaces** |
|----------------------|---------------|----------------------|
| **Editor**          | Basado en navegador (VS Code) | Entorno en la nube completo |
| **Ejecución de código** | ❌ No permite ejecutar código | ✅ Sí, permite correr código |
| **Acceso a terminal** | ❌ No disponible | ✅ Disponible |
| **Uso de extensiones** | Limitado | Completo |
| **Configuración avanzada** | ❌ No soporta `devcontainer.json` | ✅ Soporta `devcontainer.json` |

**¿Cuándo usar GitHub.dev?**  
✅ Para ediciones rápidas de código.  
✅ Cuando no necesitas ejecutar programas.  
✅ Si no quieres configurar un entorno local.  

### **🎯 Conclusión**
**GitHub.dev** es una excelente opción para editar código desde el navegador sin instalar herramientas adicionales. Sin embargo, si necesitas ejecutar código o configurar un entorno completo, **GitHub Codespaces** es la mejor alternativa.  

### Resumen

GitHub ha incorporado una herramienta que, aunque pasa desapercibida, puede hacer que editar archivos sea rápido y sencillo sin necesidad de un entorno completo como Codespaces. Este pequeño editor de código integrado en el navegador se activa al presionar Enter en la página de un repositorio, llevando al usuario a una experiencia muy similar a Visual Studio Code para ediciones rápidas y precisas.

### ¿Cómo funciona el GitHub Dev Editor?

El GitHub Dev Editor se activa al presionar Enter dentro de cualquier repositorio en GitHub. Esta acción abre un editor web donde puedes:

- Visualizar y editar archivos de texto de inmediato.
- Realizar cambios rápidos en archivos como README o en cualquier otro documento de texto dentro del repositorio.
- Guardar automáticamente los cambios al cerrar la pestaña o realizar un “commit” y “push” de forma directa.

Este editor no cuenta con terminal ni opciones avanzadas de desarrollo; está diseñado únicamente para realizar ediciones en texto.

### ¿Qué ventajas ofrece frente a otros entornos de desarrollo?

GitHub Dev Editor es ideal para tareas de edición de texto simples. Aquí algunos beneficios:

- **Sin costo adicional**: A diferencia de Codespaces, el uso del Dev Editor no genera cargos.
- **Accesibilidad**: Disponible en cualquier navegador, sin requerir configuraciones complejas.
- **Rapidez**: Acceso rápido para editar y confirmar cambios de inmediato, útil para pequeñas modificaciones o correcciones ortográficas.

### ¿Cuándo deberías utilizar Codespaces en lugar de GitHub Dev Editor?

El uso de Codespaces es recomendable cuando el proyecto requiere un entorno completo de desarrollo. Algunas de las principales razones para elegir Codespaces son:

- **Compilación y ejecución**: Codespaces permite compilar y probar código, algo que no es posible en el Dev Editor.
- **Terminal y depuración**: Incluye terminal y herramientas avanzadas para ejecutar comandos, ideal para proyectos complejos.
- **Colaboración avanzada**: Permite trabajar con equipos en un entorno en la nube, manteniendo sincronización en tiempo real.

Sin embargo, hay un costo asociado a su uso cuando se superan las horas gratuitas mensuales, por lo que es importante considerar esta opción solo cuando el proyecto lo requiera.

### ¿Qué debes considerar al escoger entre GitHub Dev Editor y Codespaces?

- **Frecuencia y tipo de cambios**: Si solo necesitas editar unas pocas líneas, GitHub Dev Editor es suficiente.
- **Costo**: Codespaces tiene un costo en la nube, mientras que el Dev Editor es gratuito.
- **Necesidades de desarrollo**: Para compilaciones y pruebas de código, Codespaces es indispensable; para correcciones menores, el Dev Editor es más práctico.

**Lecturas recomendadas**

[El editor basado en web de github.dev - Documentación de GitHub](https://docs.github.com/es/codespaces/the-githubdev-web-based-editor)

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

## Cómo Usar Tokens en GitHub para Acceso Seguro a Repositorios Privados

Los **tokens de acceso personal (PAT - Personal Access Tokens)** en GitHub permiten autenticarse de manera segura sin necesidad de usar contraseñas. Son esenciales para interactuar con repositorios privados desde la terminal, Git, API o herramientas externas.  

### **1️⃣ Generar un Token de Acceso Personal en GitHub**  
1. Ve a [GitHub → Settings](https://github.com/settings/tokens).  
2. Haz clic en **"Generate new token (classic)"**.  
3. Asigna un **nombre descriptivo** al token.  
4. Define la **fecha de expiración** (opcional, pero recomendable).  
5. Selecciona los permisos necesarios:  
   - `repo`: Acceder a repositorios privados.  
   - `workflow`: Administrar GitHub Actions.  
   - `admin:repo_hook`: Gestionar webhooks del repositorio.  
6. Haz clic en **"Generate token"** y **guárdalo en un lugar seguro**.  

⚠️ **Importante**: No podrás ver el token después de salir de la página.  

### **2️⃣ Usar el Token en Git**  
### 📌 **Autenticación con HTTPS**  
Si tu repositorio es privado y quieres hacer `git push` o `git pull`, usa este formato:  
```sh
git clone https://<TOKEN>@github.com/usuario/repositorio.git
```
Ejemplo:  
```sh
git clone https://ghp_xxxxxxx@github.com/miusuario/mirepo.git
```
Luego, puedes usar `git pull`, `git push`, etc., sin necesidad de ingresar credenciales.  

### **3️⃣ Configurar el Token en Git (Credenciales Guardadas)**
Si quieres evitar ingresar el token cada vez que interactúas con GitHub, usa el **credential helper**:  
```sh
git config --global credential.helper store
```
Luego, la primera vez que hagas `git push`, Git te pedirá el usuario y el token como contraseña.  

### **4️⃣ Usar el Token con la API de GitHub**
Si necesitas interactuar con la API de GitHub, puedes hacer una solicitud con `curl`:  
```sh
curl -H "Authorization: token <TOKEN>" https://api.github.com/user/repos
```

### **5️⃣ Revocar o Regenerar un Token**
Si pierdes un token o ya no lo necesitas:  
🔹 Ve a [GitHub → Settings → Tokens](https://github.com/settings/tokens).  
🔹 Selecciona el token y haz clic en **"Revoke"** o **"Regenerate"**.  

### **🎯 Conclusión**
✅ **Los tokens de GitHub son más seguros que las contraseñas**.  
✅ **Permiten interactuar con GitHub desde Git, API o herramientas externas**.  
✅ **Es importante administrar y revocar tokens cuando ya no se necesiten**.  

### Resumen
Para quienes buscan facilitar el trabajo colaborativo sin comprometer la seguridad de su repositorio privado, GitHub ofrece una solución eficaz a través de los tokens de acceso personal (PAT, por sus siglas en inglés). Estas llaves temporales otorgan acceso controlado a usuarios o aplicaciones, asegurando que solo puedan realizar acciones específicas y por el tiempo necesario. En este artículo, exploramos cómo crear y gestionar tokens en GitHub para maximizar la seguridad y funcionalidad en distintos entornos.

### ¿Cómo invitar a otros sin hacerlos colaboradores permanentes?

- **Escenarios comunes**: Cuando deseas que alguien contribuya de forma temporal, pero sin añadirlo como colaborador permanente.
- **Solución con tokens**: En lugar de agregarlo como colaborador, puedes crear un token con permisos limitados para que acceda al repositorio solo en el tiempo y con las acciones necesarias.

### ¿Qué tipos de tokens existen en GitHub?

GitHub ofrece dos tipos de tokens:

- **Tokens Clásicos**: Permiten seleccionar permisos básicos y pueden tener duración indefinida (aunque no es recomendable).
- **Tokens Detallados**: Versiones más nuevas que permiten control granular y una duración máxima de 90 días. Ideal para asegurar un acceso mucho más restringido.

### ¿Cómo se crea un token clásico?

1. **Acceso a Configuración**: Ve a “Settings” en tu perfil y selecciona “Developer Settings”.
2. **Generar Token**: Elige “Generate New Token” y configura:
 - Nombre: Ayuda a identificar el propósito del token.
 - Expiración: Ajusta a la duración esperada del proyecto.
 - Permisos: Elige los accesos que consideres necesarios (repos, paquetes, notificaciones, etc.).
 
3. Guardar Token: Copia el token en un lugar seguro, ya que no será visible nuevamente después de cerrar la página.

### ¿Cómo funcionan los tokens detallados?

A diferencia de los tokens clásicos, los detallados permiten:

- **Duración Máxima de 90 Días**: Ofrece seguridad adicional al limitar el tiempo de acceso.
- **Control de Repositorio Específico**: Puedes configurar que el token tenga acceso solo a repositorios específicos, incluso privados o públicos.
- **Permisos Granulares**: Permiten ajustar más a detalle el alcance y las acciones que el token puede realizar.

### ¿Cómo utilizar el token en un entorno externo?

Al clonar un repositorio en un equipo sin credenciales de GitHub, el sistema pedirá el nombre de usuario y, en lugar de la contraseña de la cuenta, el token de acceso personal. Al ingresar el token en la terminal, podrás acceder al repositorio de forma segura, replicando el proceso de clonación estándar sin exponer tus credenciales personales.

### ¿Para qué otras tareas se pueden utilizar los tokens?

Los tokens no solo sirven para acceder desde equipos remotos; su funcionalidad se extiende a:

- **Automatización con GitHub Actions**: Automatiza flujos de trabajo en tu repositorio.
- **Scripts Personalizados**: Ideal para automatizar tareas repetitivas como commits periódicos o sincronización de proyectos.
- **Integración con Terceros**: Configura accesos específicos para aplicaciones que interactúan con tu repositorio.

### ¿Qué medidas de seguridad deben tomarse con los tokens?

- **Configurar expiración**: Limita la duración del token según las necesidades.
- **Reducir permisos**: Otorga solo los permisos mínimos necesarios.
- **Revisión y eliminación**: Revisa periódicamente los tokens en uso y elimina aquellos que ya no sean necesarios para evitar riesgos de acceso no autorizado.

**Lecturas recomendadas**

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

[Sign in to GitHub · GitHub](https://github.com/settings/tokens)

## Gestión de Dependencias y Seguridad con Dependabot en GitHub

**Dependabot** es una herramienta integrada en GitHub que ayuda a mantener actualizadas las dependencias de tu proyecto y a detectar vulnerabilidades de seguridad en paquetes desactualizados.  

### **📌 1️⃣ ¿Qué es Dependabot y cómo funciona?**  
**Dependabot** automatiza la actualización de dependencias en proyectos gestionados con **npm, pip, Maven, Gradle, Composer, Cargo**, entre otros.  

🛠️ **Funciones principales:**  
✔ **Actualización automática de dependencias.**  
✔ **Alertas de seguridad sobre paquetes vulnerables.**  
✔ **Creación de Pull Requests para solucionar problemas detectados.**  

### **🔄 2️⃣ Activar Dependabot en un Repositorio**  
1️⃣ Ve a tu repositorio en GitHub.  
2️⃣ Accede a **"Settings" → "Security & analysis"**.  
3️⃣ Habilita **Dependabot alerts** y **Dependabot security updates**.  

### **📥 3️⃣ Configurar Actualizaciones Automáticas de Dependencias**  
Para activar Dependabot en la actualización de paquetes, crea un archivo de configuración en `.github/dependabot.yml`:  

```yaml
version: 2
updates:
  - package-ecosystem: "npm" # Reemplázalo según tu gestor (pip, maven, etc.)
    directory: "/" # Ruta del archivo de dependencias
    schedule:
      interval: "daily" # Opciones: daily, weekly, monthly
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
    ignore:
      - dependency-name: "lodash" # Puedes excluir paquetes específicos
```

📌 **Personalización:**  
- **`package-ecosystem`**: Define el gestor de dependencias (npm, pip, composer, etc.).  
- **`directory`**: Ubicación del archivo de dependencias.  
- **`schedule`**: Frecuencia de actualización.  
- **`ignore`**: Evita actualizaciones de paquetes específicos.  

### **⚠ 4️⃣ Seguridad: Alertas y Parches Automáticos**  
Cuando GitHub detecta una vulnerabilidad en una dependencia:  
🔴 Se muestra una alerta en la pestaña **"Security" → "Dependabot alerts"**.  
🟢 Dependabot puede generar automáticamente un **Pull Request** con una versión segura del paquete afectado.  

### **🎯 Conclusión**  
✅ **Mejora la seguridad** al detectar y corregir vulnerabilidades automáticamente.  
✅ **Mantiene las dependencias actualizadas** con mínimo esfuerzo.  
✅ **Facilita la gestión de proyectos** al automatizar actualizaciones en múltiples entornos. 

### Resumen

La gestión de dependencias es esencial para mantener la seguridad y estabilidad de las aplicaciones. Dependabot es una herramienta eficaz que, al integrarse en el flujo de trabajo, identifica y soluciona problemas en las versiones de paquetes de terceros, minimizando vulnerabilidades. Aquí exploramos cómo configurar y aprovechar al máximo Dependabot.

### ¿Por qué es importante mantener actualizados los paquetes de terceros?

Los paquetes de terceros son un recurso común en el desarrollo para simplificar tareas como la lectura de archivos JSON o la creación de APIs. Sin embargo, estas dependencias pueden convertirse en un riesgo si no se actualizan, ya que las versiones desactualizadas pueden contener vulnerabilidades que comprometan la seguridad de la aplicación.

### ¿Cómo activar Dependabot en un repositorio?

Para activar Dependabot:

1. Accede a Settings o Security dentro del repositorio.
2. Ve a Code Security and Analysis y selecciona la categoría de Dependabot.
3. Activa las alertas de seguridad y actualizaciones de versión.
4. Dependabot generará un archivo dependabot.yml, donde puedes ajustar la frecuencia de las revisiones, como cambiar de semanal a diaria para detectar actualizaciones con mayor regularidad.

### ¿Cómo utilizar Dependabot para gestionar versiones específicas?

En el caso de proyectos .NET, se pueden elegir versiones específicas de paquetes:

1. Navega a la pestaña del paquete deseado (por ejemplo, **Newtonsoft.Json**).
2. Escoge una versión con vulnerabilidades conocidas (como 12.0.3 en este ejemplo) para ver cómo Dependabot detecta y notifica el problema.
3. Dependabot genera un pull request automáticamente para actualizar la versión del paquete y solucionar la vulnerabilidad detectada.

### ¿Qué sucede cuando Dependabot detecta una vulnerabilidad?

Cuando Dependabot encuentra una vulnerabilidad:

- Notifica con prioridad la versión insegura del paquete.
- Crea un pull request para actualizar el paquete a una versión segura.
- Permite revisar y aceptar la actualización directamente desde la sección de **Security** o en **Pull Requests**.

Dependabot analiza la compatibilidad de versiones para asegurar que la actualización sea estable y, en algunos casos, puede incluso eliminar la rama creada una vez fusionada la actualización.

### ¿Por qué integrar Dependabot en el flujo de trabajo?

Dependabot simplifica la gestión de actualizaciones:

- Detecta y repara vulnerabilidades sin intervención manual.
- Mantiene el proyecto actualizado con las versiones estables más recientes de cada dependencia.
- Agiliza la revisión y aplicación de actualizaciones, evitando que el equipo trabaje con versiones obsoletas.

**Lecturas recomendadas**

[GitHub - platzi/git-github: Repositorio del Curso de Git y GitHub](https://github.com/platzi/git-github)

[Dependabot · GitHub](https://github.com/dependabot)

## Mantenimiento de repositorios seguros

Mantener un repositorio seguro en GitHub es fundamental para proteger el código y la información sensible de un proyecto. A continuación, te presento **buenas prácticas y herramientas** que te ayudarán a garantizar la seguridad en tus repositorios.  

### **1️⃣ Configuración de Seguridad en GitHub**  

### 📌 **Habilita la autenticación en dos pasos (2FA)**  
- Ve a **Settings → Password and authentication**.  
- Activa **2FA** para proteger tu cuenta contra accesos no autorizados.  

### 📌 **Usa claves SSH o Tokens de Acceso Personal (PAT)**  
- **Claves SSH**: Autenticación segura para Git en lugar de contraseñas.  
- **PAT (Personal Access Tokens)**: Se usan para acceso seguro a repositorios y API de GitHub.  

### **2️⃣ Control de Accesos y Permisos**  

### 👥 **Configura los roles de los colaboradores**  
- **Admin**: Control total sobre el repositorio.  
- **Maintainer**: Gestión de Issues y Pull Requests.  
- **Developer**: Puede hacer commits y forks.  
- **Read-only**: Solo lectura del código.  

### 🔒 **Restringe los cambios en ramas protegidas**  
- En **Settings → Branches**, activa **"Branch protection rules"**.  
- Habilita **"Require pull request reviews"** para evitar cambios sin revisión.

### **3️⃣ Protección contra Vulnerabilidades**  

### ⚠ **Habilita Dependabot** para actualizar dependencias  
- Activa **Dependabot alerts** en **Settings → Security & analysis**.  
- Configura `.github/dependabot.yml` para actualizaciones automáticas.  

### 🔎 **Habilita el análisis de código y escaneo de secretos**  
- Usa **GitHub Advanced Security** para detectar credenciales filtradas.  
- Configura **CodeQL** para análisis de seguridad del código fuente.  

### **4️⃣ Buenas Prácticas para la Seguridad del Código**  

### 🚫 **Evita subir archivos sensibles**  
- Agrega un archivo `.gitignore` para excluir archivos como:  
  ```plaintext
  .env
  config/secrets.yml
  node_modules/
  ```

### 🔑 **Usa variables de entorno en lugar de credenciales en código**  
- Almacena claves API en archivos `.env` y usa `dotenv` en tu código.  

### 📜 **Firma tus commits para mayor autenticidad**  
- Usa GPG para firmar commits y evitar alteraciones:  
  ```sh
  git config --global user.signingkey <GPG_KEY>
  git commit -S -m "Commit firmado"
  ```

### **5️⃣ Monitoreo y Auditoría**  

### 📊 **Revisa el historial de seguridad en GitHub**  
- Ve a **Settings → Audit Log** para ver registros de cambios y accesos.  
- Activa **notificaciones de seguridad** en la pestaña **"Security"** del repositorio.  

### 🛠 **Automatiza revisiones de seguridad con GitHub Actions**  
- Usa workflows de GitHub Actions para correr análisis de seguridad antes de cada `merge`.  

### **🎯 Conclusión**  
✅ **Usa autenticación segura** con 2FA y claves SSH.  
✅ **Controla permisos y protege ramas críticas**.  
✅ **Mantén dependencias seguras** con Dependabot.  
✅ **Evita exponer información sensible** en los commits.  
✅ **Monitorea accesos y cambios en el repositorio**.  

**Lecturas recomendadas**

[GitHub - github/gitignore: A collection of useful .gitignore templates](https://github.com/github/gitignore)

[Ignoring files - GitHub Docs](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files)

[Curso de GitHub Copilot - Platzi](https://platzi.com/cursos/github-copilot/)

## Gestión de datos sensibles y políticas de seguridad

La **seguridad de los datos sensibles** en GitHub es clave para evitar filtraciones de credenciales, claves API y otra información confidencial. Implementar **buenas prácticas y políticas de seguridad** ayuda a proteger tanto el código como la infraestructura del proyecto.  

### **1️⃣ ¿Qué son los Datos Sensibles y por qué protegerlos?**  

🔴 **Datos sensibles** son aquellos que pueden comprometer la seguridad si son expuestos, como:  
- Claves de API y tokens de autenticación.  
- Credenciales de bases de datos.  
- Claves SSH y certificados.  
- Información personal o financiera.  
- Configuraciones privadas de servidores.  

🛑 **Si se exponen estos datos en un repositorio público, podrían ser utilizados por atacantes para acceder a sistemas críticos.**  

### **2️⃣ Evitar la Exposición de Datos Sensibles**  

### 📌 **Usa un `.gitignore` adecuado**  
Agrega archivos sensibles a `.gitignore` para que no sean rastreados por Git. Ejemplo:  

```plaintext
# Archivo de configuración y claves API
.env
config/secrets.yml
*.pem
*.key
```

### 📌 **Usa Variables de Entorno en Lugar de Datos en el Código**  
En lugar de incrustar claves en el código, usa variables de entorno.  

Ejemplo en **Python** con `dotenv`:  
```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
```

Ejemplo en **JavaScript** con `process.env`:  
```javascript
require('dotenv').config();
const apiKey = process.env.API_KEY;
```

### **3️⃣ Políticas de Seguridad en GitHub**  

### 🔑 **Configurar Accesos y Permisos**  
- **Usa la autenticación en dos pasos (2FA)** para mejorar la seguridad.  
- **Asigna roles adecuados** en equipos para evitar accesos innecesarios.  
- **Restringe cambios en ramas protegidas** en **Settings → Branches**.  

### 🛠 **Escaneo de Seguridad con GitHub Advanced Security**  
- **Habilita el escaneo de secretos** para detectar credenciales filtradas en commits.  
- **Activa Dependabot** para recibir alertas sobre vulnerabilidades en dependencias.  

```yaml
# Archivo .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "daily"
```

### **4️⃣ Auditoría y Monitoreo**  

### 📊 **Habilita el Registro de Actividad (Audit Log)**  
- Ve a **Settings → Audit Log** para ver accesos, cambios y acciones sospechosas.  

### ⚠ **Configura Alertas de Seguridad**  
- Recibe notificaciones cuando GitHub detecte vulnerabilidades o accesos no autorizados.  
- Usa **GitHub Actions** para automatizar auditorías de seguridad.  

Ejemplo de **workflow en GitHub Actions** para escaneo de seguridad:  
```yaml
name: Security Check
on: push
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run TruffleHog
        uses: edplato/trufflehog-actions-scan@main
```

### **5️⃣ ¿Qué Hacer si ya Cometiste un Error y Subiste Datos Sensibles?**  

### 🚨 **1. Revocar credenciales comprometidas de inmediato**  
Si subiste una clave API o token, revócalo y genera uno nuevo desde tu proveedor de servicios.  

### 🧹 **2. Eliminar el archivo comprometido de Git**  
Si subiste datos sensibles, elimínalos completamente del historial:  

```sh
git rm --cached archivo_sensible.txt
git commit -m "Elimina archivo sensible"
git push origin main
```

Si ya fue publicado en un commit, usa `filter-branch` o `BFG Repo-Cleaner`:  
```sh
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch archivo_sensible.txt' --prune-empty --tag-name-filter cat -- --all
git push origin --force --all
```

### **🎯 Conclusión**  
✅ **Usa `.gitignore` y variables de entorno** para evitar exponer datos sensibles.  
✅ **Configura permisos y autenticación segura** en GitHub.  
✅ **Activa Dependabot y escaneo de seguridad** para detectar vulnerabilidades.  
✅ **Monitorea accesos y acciones sospechosas** con Audit Log.  
✅ **Si cometes un error, revoca credenciales y limpia el historial de Git.**

### Resumen

Al configurar un proyecto en GitHub, es crucial proteger la información sensible, como llaves de API o datos de conexión, evitando que sean subidas por error al repositorio. Aunque el uso de archivos como .gitignore es clave para excluir información sensible, existen herramientas adicionales en GitHub para detectar y mitigar estos riesgos. Aquí te explico cómo activar y aprovechar al máximo CodeQL y Secret Scanning para mejorar la seguridad de tu código en repositorios públicos.

### ¿Cómo activar CodeQL y Secret Scanning en GitHub?

Para comenzar, es necesario hacer el repositorio público, ya que algunas de estas funciones no están disponibles en repositorios privados. Sigue estos pasos para activar la seguridad de tu código:

1. Ve a **Settings** en el repositorio y desplázate a **Code Security and Analysis**.
2. Activa **CodeQL Analysis** en la sección de Code Scanning. GitHub detectará el lenguaje de tu proyecto y comenzará a analizar el código en segundo plano.
3. Luego, asegúrate de que **Secret Scanning** esté habilitado. Esta función, que viene activada por defecto, detecta automáticamente secretos o llaves sensibles en el código.

### ¿Cómo probar si Secret Scanning detecta información sensible?

Para verificar el funcionamiento de Secret Scanning, puedes añadir intencionadamente una clave sensible en el código. Por ejemplo:

- Agrega una cadena como `string stripeApiKey = "clave_sensible"`; en tu archivo de código y sube el cambio.
- GitHub realizará el análisis en segundo plano y, si detecta una llave API, emitirá una alerta en la sección de Security.

En cambio, una cadena menos evidente, como `string connectionString = "cadena_sensible";`, podría pasar desapercibida. Este ejemplo muestra que, aunque Secret Scanning es útil, no es infalible y puede omitir cadenas sensibles si no siguen patrones comunes.

### ¿Qué hacer si GitHub detecta una alerta de seguridad?

Si GitHub te notifica sobre una alerta de secreto en el código, sigue estos pasos para mitigar el riesgo:

- **Renueva las credenciales** en la cuenta del servicio de terceros para evitar accesos no autorizados.
- **Elimina la clave del historial** de commits, ya que eliminarla solo del código actual no la quita de versiones anteriores.
- **Modifica tu código** para utilizar archivos de configuración, como `.env` o `appsettings.json`, y exclúyelos con `.gitignore` para evitar futuras filtraciones.

### ¿Por qué CodeQL y Secret Scanning no son suficientes?

Aunque GitHub ofrece herramientas poderosas de análisis, no garantizan la detección de todas las cadenas sensibles. Por eso, la mejor práctica sigue siendo prevenir las filtraciones configurando correctamente `.gitignore` y almacenando secretos en archivos de entorno.

Estas prácticas te ayudarán a evitar el riesgo de exponer información crítica, reduciendo los problemas de seguridad y permitiéndote mantener un código limpio y seguro.

**Lecturas recomendadas**

[Managing your personal access tokens - GitHub Docs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

## Administración de organizaciones en GitHub

GitHub permite gestionar equipos de desarrollo de manera eficiente a través de **Organizaciones**, proporcionando herramientas avanzadas para la colaboración, seguridad y control de acceso a los repositorios.  

### **1️⃣ ¿Qué es una Organización en GitHub?**  

Una **Organización en GitHub** es un **espacio de trabajo compartido** donde varios desarrolladores pueden colaborar en proyectos con permisos y roles bien definidos.  

🔹 **Ventajas de usar una Organización:**  
✅ Centraliza el desarrollo en equipos grandes.  
✅ Permite una mejor gestión de acceso y permisos.  
✅ Facilita la administración de múltiples repositorios.  
✅ Incluye herramientas de seguridad y auditoría avanzadas.  

### **2️⃣ Creación de una Organización en GitHub**  

### 📌 Pasos para crear una organización:  
1. **Ir a** [GitHub Organizations](https://github.com/account/organizations).  
2. Hacer clic en **"New organization"**.  
3. Elegir un plan (Gratis o GitHub Enterprise).  
4. Asignar un nombre a la organización y agregar miembros.  
5. Configurar permisos y roles iniciales.  

### **3️⃣ Gestión de Miembros y Equipos**  

### **👥 Roles en una Organización**  
GitHub ofrece distintos niveles de acceso para gestionar el equipo:  

| **Rol**      | **Permisos** |
|-------------|-------------|
| **Owner** | Control total sobre la organización. Puede administrar repositorios, facturación y seguridad. |
| **Member** | Acceso a los repositorios de la organización con permisos asignados por los Owners. |
| **Billing Manager** | Puede gestionar la facturación, pero no modificar repositorios. |

🔹 **Configurar permisos de usuario** en **Settings → People**.  
🔹 **Asignar roles** para limitar el acceso según las responsabilidades de cada miembro.  

### **👥 Creación de Equipos**  
- Ir a **Organization → Teams → New Team**.  
- Crear equipos por función, como `Frontend`, `Backend`, `DevOps`.  
- Asignar **repositorios y permisos específicos** a cada equipo.  

📌 **Ejemplo de niveles de acceso en un equipo:**  
- `Read` → Solo lectura del código.  
- `Write` → Puede hacer commits y push.  
- `Admin` → Control total del repositorio.  

### **4️⃣ Gestión de Repositorios en una Organización**  

### 🔹 **Creación y Configuración de Repositorios**  
- Desde la organización, ir a **Repositories → New**.  
- Elegir el nivel de visibilidad:  
  - **Público** (accesible para todos).  
  - **Privado** (solo para miembros de la organización).  
  - **Interno** (disponible solo dentro de la empresa en GitHub Enterprise).  

### 🔹 **Administrar Acceso a los Repositorios**  
1. Ir a **Repository → Settings → Manage Access**.  
2. Asignar permisos según equipos o usuarios individuales.  

📌 **Consejo:** Habilita **branch protection rules** para evitar cambios sin revisión.  

```sh
git branch -m main
git push --set-upstream origin main
```

### **5️⃣ Seguridad y Control en una Organización**  

### 🔑 **Habilitar Autenticación en Dos Pasos (2FA)**  
- **Settings → Security → Require Two-Factor Authentication**.  
- Obliga a los miembros a usar 2FA para mayor seguridad.  

### 🔎 **Auditoría con GitHub Audit Log**  
- **Settings → Audit Log** permite revisar actividades como:  
  - Commits y cambios de código.  
  - Modificaciones en los permisos.  
  - Accesos de usuarios y tokens API.  

### 🚀 **Activar GitHub Advanced Security**  
- **Escaneo de código y secretos** para detectar vulnerabilidades.  
- **Habilitar Dependabot** para gestionar dependencias seguras.  

### **6️⃣ Automatización con GitHub Actions**  

Las organizaciones pueden **automatizar flujos de trabajo** con **GitHub Actions**:  
- **CI/CD** para pruebas y despliegues automáticos.  
- **Escaneo de seguridad** en cada pull request.  
- **Notificaciones automáticas** para revisiones de código.  

Ejemplo de workflow para ejecutar pruebas en cada push:  

```yaml
name: CI/CD Workflow
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Ejecutar pruebas
        run: npm test
```

### **🎯 Conclusión**  

✅ **Usar organizaciones en GitHub facilita la colaboración y control de acceso.**  
✅ **Configurar equipos y roles permite gestionar el trabajo eficientemente.**  
✅ **Implementar seguridad y auditoría protege la organización de accesos no autorizados.**  
✅ **Automatizar procesos con GitHub Actions mejora la eficiencia del equipo.**  

**Lecturas recomendadas**

[Comunidad Platzi · GitHub](https://github.com/platzi)

## Cómo personalizar tu perfil de GitHub con GitHub Actions

GitHub permite personalizar el perfil con un **README dinámico**, que se actualiza automáticamente usando **GitHub Actions**. Esto es útil para mostrar estadísticas, proyectos recientes o cualquier otro contenido en tiempo real.  

### **1️⃣ ¿Qué es GitHub Actions?**  
**GitHub Actions** es una herramienta de automatización en GitHub que permite ejecutar scripts en respuesta a eventos como commits, push o cron jobs (tareas programadas).  

🔹 Se usa para:  
✅ Actualizar automáticamente tu perfil.  
✅ Mostrar estadísticas en tiempo real.  
✅ Integrar APIs para obtener datos dinámicos.  

### **2️⃣ Crear y Configurar el Perfil Personal**  

📌 **Paso 1: Crear un Repositorio Especial**  
1. Ve a [GitHub](https://github.com/) y haz clic en **"New Repository"**.  
2. Nómbralo exactamente como tu usuario de GitHub. Ejemplo: `MarioVargasCelis`.  
3. **Marcarlo como público**.  
4. Selecciona **"Add a README file"** y crea el repositorio.  

📌 **Paso 2: Editar el README.md**  
Abre el archivo `README.md` y personalízalo con Markdown:  

```md
# ¡Hola! Soy Mario Alexander Vargas Celis 🚀

💡 **Ingeniero Mecatrónico** apasionado por la **IA, el análisis de datos y la automatización**.  
📈 Actualmente aprendiendo **PyTorch, Pandas y Machine Learning**.  
🌎 Visitando Argentina, Uruguay y Brasil en mi próximo viaje.  
🔗 [LinkedIn](https://linkedin.com/in/mariovargascelis) | [GitHub](https://github.com/MarioVargasCelis)
```

📌 **Paso 3: Confirmar y Subir Cambios**  

```sh
git add README.md
git commit -m "Personalizando mi perfil"
git push origin main
```

### **3️⃣ Personalización Dinámica con GitHub Actions**  

📌 **Paso 4: Crear un Workflow Automático**  
1. Dentro del repositorio, crea la carpeta **`.github/workflows/`**.  
2. Dentro de esa carpeta, crea un archivo llamado `update-profile.yml`.  

📌 **Paso 5: Escribir el Workflow**  
Este workflow actualiza el perfil con la fecha y hora actual cada 24 horas.  

```yaml
name: Update README

on:
  schedule:
    - cron: "0 0 * * *"  # Ejecutar todos los días a medianoche
  push:
    branches:
      - main

jobs:
  update-readme:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Update README
        run: |
          echo "### Última actualización: $(date)" > README.md
          git config --global user.name "GitHub Actions"
          git config --global user.email "actions@github.com"
          git add README.md
          git commit -m "Actualización automática del README"
          git push
```

📌 **Paso 6: Confirmar y Subir el Workflow**  

```sh
git add .github/workflows/update-profile.yml
git commit -m "Agregando GitHub Actions para actualizar README"
git push origin main
```

### **4️⃣ Agregar Contenido Dinámico al Perfil**  

### **📊 Estadísticas del Perfil con GitHub Actions**  

Para mostrar estadísticas de GitHub en el perfil, usa:  

```md
![GitHub Stats](https://github-readme-stats.vercel.app/api?username=MarioVargasCelis&show_icons=true&theme=tokyonight)
```

### **💻 Lenguajes Más Usados**  

```md
![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=MarioVargasCelis&layout=compact&theme=tokyonight)
```

## **🎯 Conclusión**  

✅ **GitHub Actions permite actualizar automáticamente el README del perfil.**  
✅ **Se pueden agregar datos dinámicos como la fecha de actualización o estadísticas.**  
✅ **Con un workflow programado, puedes mantener tu perfil actualizado sin esfuerzo.**  

### Resumen

Optimizar tu perfil de GitHub puede marcar una gran diferencia al mostrar tus habilidades y actividad de desarrollo, y GitHub Actions es la herramienta ideal para automatizar tareas que mantengan tu perfil siempre actualizado. A continuación, te explicamos cómo crear una GitHub Action que muestre tu actividad reciente directamente en tu portada de perfil.

### ¿Qué es GitHub Actions y cómo puede ayudarte?

GitHub Actions permite automatizar diversos flujos de trabajo, desde revisiones de seguridad con Dependabot hasta integraciones continuas y despliegues (CI/CD). Con una GitHub Action, puedes extraer automáticamente tu actividad más reciente y mostrarla en tu perfil. Esto ayuda a que otros vean tu progreso y compromiso en tus repositorios.

### ¿Cómo configurar tu repositorio de portada en GitHub?

1. **Selecciona tu repositorio de perfil**: Este suele llevar el nombre de usuario como nombre de repositorio.
2. **Edita el README**: Abre el archivo **README.md** en modo edición y asegúrate de contar con una sección específica para la actividad reciente.
3. **Agrega un título personalizable**: Por ejemplo, “Actividad reciente” o “Últimos commits”, para que la sección tenga un toque personal.

### ¿Cómo crear una GitHub Action para actualizar tu actividad?

- **Accede a la sección de Actions** en el repositorio de tu perfil.
- **Elige una plantilla básica** para crear un flujo de trabajo simple.
- **Configura el archivo YAML**:
 - **Ubicación**: Guarda el archivo como `.github/workflows/update-readme.yml`.
 - **Cron Job**: Programa la frecuencia de actualización. Un formato común es `*/12 * * * *` para actualizar cada 12 horas.
 - **Paso de checkout**: Este paso accede a tus archivos.
 - **Recent Activity Workflow**: Incluye el paso para extraer tu actividad reciente desde un flujo preconstruido disponible en el Marketplace de GitHub Actions.
 
### ¿Cómo asegurarte de que tu GitHub Action funcione correctamente?

Guarda y ejecuta la acción: Haz un commit directo en la rama principal.
Verifica el estado: Si ocurre algún error, GitHub Actions lo destacará en rojo, señalando el paso problemático.
Ajustes finales: Corrige cualquier error en el archivo YAML o en los comentarios del `README`, y vuelve a ejecutar el flujo de trabajo.

### ¿Qué opciones adicionales puedes explorar en GitHub Actions?

- **Marketplace**: Encuentra acciones preconfiguradas para desplegar en Docker, compilar aplicaciones en Java, entre otras.
- **Variables de ambiente:** Usa `GITHUB_TOKEN` para autenticar la Action y mantenerla segura.
- **Contenido personalizado**: Puedes agregar secciones adicionales en tu perfil, como listas de videos de YouTube o portafolios enlazados.

### ¿Cómo hacer que tu perfil destaque con otras personalizaciones?

GitHub Actions ofrece un sinfín de posibilidades para personalizar tu perfil y mostrar contenido relevante. Explora perfiles inspiradores en los recursos de esta clase, descarga archivos YAML, y pruébalos en tu repositorio para continuar optimizando tu presencia profesional.

**Lecturas recomendadas**

[Curso de GitHub Actions - Platzi](https://platzi.com/cursos/github-actions/80449878)

## GitHub CLI para administración

La **GitHub CLI** (Interfaz de Línea de Comandos de GitHub) es una herramienta que permite interactuar con GitHub directamente desde la terminal, facilitando diversas tareas de administración de repositorios, gestión de issues, pull requests, y otras funcionalidades de GitHub sin necesidad de utilizar la interfaz web.

### Instalación de GitHub CLI

1. **Para macOS** (con Homebrew):
   ```bash
   brew install gh
   ```

2. **Para Windows** (con Chocolatey):
   ```bash
   choco install gh
   ```

3. **Para Linux** (usando APT en Debian/Ubuntu):
   ```bash
   sudo apt install gh
   ```

### Autenticación con GitHub

Para autenticarte en GitHub desde la CLI, puedes usar el siguiente comando:

```bash
gh auth login
```

Esto abrirá una serie de pasos para elegir cómo deseas autenticarte (por ejemplo, mediante navegador o token de acceso personal).

### Comandos Básicos de GitHub CLI

1. **Clonar un repositorio:**
   ```bash
   gh repo clone <usuario>/<repositorio>
   ```

2. **Crear un nuevo repositorio:**
   ```bash
   gh repo create <nombre-del-repo> --private
   ```

3. **Ver repositorios de un usuario:**
   ```bash
   gh repo list <usuario>
   ```

4. **Gestionar Issues:**
   - **Listar issues:** 
     ```bash
     gh issue list
     ```
   - **Crear un nuevo issue:** 
     ```bash
     gh issue create --title "Nuevo Issue" --body "Descripción del problema"
     ```

5. **Gestionar Pull Requests:**
   - **Ver pull requests abiertos:**
     ```bash
     gh pr list
     ```
   - **Crear un pull request:**
     ```bash
     gh pr create --title "Nueva característica" --body "Descripción de la PR" --base main --head feature-branch
     ```

6. **Ver información del repositorio:**
   ```bash
   gh repo view <usuario>/<repositorio> --web
   ```

7. **Ver actividades recientes:**
   ```bash
   gh activity
   ```

8. **Ver eventos de un repositorio:**
   ```bash
   gh repo events
   ```

### Integración con Git

GitHub CLI se puede usar junto con los comandos de Git. Puedes gestionar tus repositorios, ramas y cambios de manera integrada con Git. Además, puedes abrir Pull Requests y Issues desde la misma terminal sin necesidad de salir de la consola.

### Más Comandos

Puedes ver todos los comandos disponibles ejecutando:
```bash
gh help
```

La **GitHub CLI** es una herramienta poderosa para usuarios que prefieren trabajar directamente desde la terminal, optimizando su flujo de trabajo con GitHub.

**Lecturas recomendadas**

[GitHub CLI | Take GitHub to the command line](https://cli.github.com/)

[gh - Manual GitHub CLI](https://cli.github.com/manual/gh)

### Importancia de los Pull Requests y Ramas en GitHub

Los **Pull Requests** y las **Ramas** son fundamentales para un flujo de trabajo eficiente y colaborativo en proyectos que utilizan GitHub. Estos elementos ayudan a gestionar cambios en el código, permitiendo trabajar de manera simultánea sin comprometer la integridad del código principal.

### Importancia de las **Ramas** en GitHub:

1. **Aislamiento de Características y Correcciones:**
   - Las ramas permiten trabajar en nuevas características, correcciones de errores o mejoras sin afectar el código principal (usualmente en la rama `main` o `master`).
   - Esto facilita el desarrollo paralelo por parte de varios desarrolladores en un mismo proyecto.

2. **Trabajo en Equipo:**
   - Cada miembro del equipo puede crear una rama para realizar sus cambios sin interferir con el trabajo de otros.
   - Se pueden organizar ramas específicas para cada tarea o característica (por ejemplo, `feature/login`, `bugfix/fix-button`).

3. **Manejo de Conflictos:**
   - Trabajar en ramas separadas facilita la resolución de conflictos cuando varios desarrolladores modifican archivos similares.
   - Los conflictos se pueden resolver dentro de la rama antes de realizar una fusión con la rama principal.

### Importancia de los **Pull Requests (PR)** en GitHub:

1. **Revisión de Código:**
   - Un **Pull Request** es la solicitud para fusionar una rama con otra (por lo general, una rama de características o corrección de errores en la rama principal).
   - Permite a otros miembros del equipo revisar el código antes de integrarlo, mejorando la calidad y reduciendo errores.
   - Se pueden realizar comentarios en líneas específicas del código y sugerir cambios o mejoras.

2. **Colaboración:**
   - Los **PR** facilitan la colaboración entre diferentes miembros del equipo y permiten mantener un control sobre los cambios que se van integrando.
   - Además, se pueden añadir **etiquetas** para clasificar PRs (por ejemplo, `bug`, `feature`, `enhancement`).

3. **Automatización de Pruebas:**
   - Muchas veces, al abrir un PR, se ejecutan **acciones de CI/CD (integración continua/despliegue continuo)** que automatizan pruebas, asegurándose de que los cambios no rompan el código existente.
   - Esto ayuda a identificar problemas temprano en el proceso de desarrollo.

4. **Historial Claro:**
   - Los PRs proporcionan un registro claro y estructurado de los cambios realizados en el proyecto. El historial de PRs se puede consultar fácilmente para ver quién hizo qué y cuándo.

5. **Documentación del Proceso:**
   - Cada PR puede incluir una descripción detallada de lo que se está cambiando y por qué. Esto documenta el propósito de los cambios, facilitando la comprensión del código para los miembros del equipo y otros colaboradores.

6. **Fusión Controlada:**
   - Antes de fusionar el código en la rama principal, los PRs permiten realizar una revisión y asegurarse de que el código esté listo para ser integrado.
   - Esto garantiza que los errores o problemas no se introduzcan directamente en el código de producción.

### Resumen:

- **Ramas**: Aislan los cambios, lo que permite un trabajo simultáneo y ordenado sin interferir con el código principal.
- **Pull Requests**: Son solicitudes para fusionar esas ramas, permitiendo revisiones de código, colaboración, pruebas automatizadas y un historial claro de cambios.

En conjunto, las ramas y los pull requests permiten un flujo de trabajo eficiente, organizado y controlado, lo cual es crucial para proyectos colaborativos, especialmente cuando múltiples desarrolladores trabajan simultáneamente en diferentes partes del código.

**Lecturas recomendadas**

[Creating a pull request - GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)