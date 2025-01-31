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
