# Curso de Configuración Profesional de Entorno de Trabajo para Ciencia de Datos

## ¿Qué son y por qué utilizar plantillas de proyectos?

### ¿Qué son las plantillas de proyectos?

Las plantillas de proyectos son un medio que posibilita portar o construir un diseño predefinido. Estas te permiten definir carpetas, notebooks, scripts, archivos de configuración, etc.

### ¿Por qué usar plantillas de proyectos?

Algunas razones para usar plantillas en proyectos se debe a que:

- Agiliza tu trabajo y reduce la fatiga por decisión.
- Es más fácil de personalizar un proyecto hecho con plantillas que hecho desde cero.
- La reproducibilidad de un proyecto es más viable.
- Es más fácil encontrar algo entre los componentes del proyecto.

**Las plantillas de proyectos** son estructuras predefinidas y organizadas que proporcionan un punto de partida estandarizado para la creación de proyectos. Su propósito es facilitar el inicio rápido de un proyecto al proporcionar una estructura básica que ya incluye archivos, carpetas y configuraciones esenciales. Estas plantillas pueden ser personalizadas para diferentes tipos de proyectos, como desarrollo de software, análisis de datos, creación de sitios web, entre otros.

### ¿Por qué utilizar plantillas de proyectos?

1. **Eficiencia**: Ahorra tiempo al evitar la configuración inicial repetitiva de cada nuevo proyecto. La estructura básica y las configuraciones esenciales ya están listas para su uso.

2. **Estandarización**: Garantiza consistencia y uniformidad en los proyectos, lo cual es particularmente útil en equipos, asegurando que todos los miembros sigan el mismo formato.

3. **Mejor organización**: Proporciona una estructura clara para los archivos y carpetas del proyecto, lo que facilita la navegación, la colaboración y el mantenimiento a largo plazo.

4. **Buenas prácticas**: Las plantillas suelen incluir configuraciones recomendadas, como las mejores prácticas para la gestión de dependencias, control de versiones, y arquitectura de código, ayudando a evitar errores comunes.

5. **Facilita la colaboración**: Al tener una estructura predefinida, facilita que otros miembros del equipo puedan integrarse rápidamente en el proyecto, comprendiendo la disposición de archivos y configuraciones.

6. **Automatización**: Las plantillas pueden incluir scripts o herramientas para automatizar ciertas tareas repetitivas, como la configuración de entornos, la ejecución de pruebas, o la compilación de código.

### Ejemplo de uso de plantillas

- **Desarrollo web**: Una plantilla para un proyecto de desarrollo web puede incluir carpetas como `src` para el código fuente, `static` para archivos estáticos, y configuraciones de herramientas como Webpack o Babel.

- **Análisis de datos**: Una plantilla de análisis de datos puede tener carpetas como `data` para conjuntos de datos, `notebooks` para cuadernos Jupyter, y configuraciones para entornos de Python como `requirements.txt`.

Utilizar plantillas reduce la complejidad inicial del desarrollo y promueve la eficiencia, organización y estandarización en los proyectos.

## Instalar Cookiecutter

###  ¿Qué es Cookiecutter?

Es un manejador de plantillas multiplataforma (Windows, Mac OS, Linux) que te permite hacer plantillas en lenguaje de programación o formato de marcado. Puede ser usado como herramienta de línea de comandos o como librería de Python.

Cookiecutter funciona con Jinja, un motor de plantillas extensible con el cual puedes crear plantillas como si estuvieras escribiendo código en Python.

### ¿Cómo funciona?

Hay 3 pasos para entender la manera en que funciona:

- Detectará una sintaxis especial en los documentos y carpetas de tu proyecto.
- Buscará variables y valores a reemplazar.
- Finalmente, entregará un proyecto con la estructura definida en la plantilla.

### Sintaxis de Jinja

Existen 3 tipos diferentes de bloques:

- **Bloques de expresión:** se usan para incluir variables en la plantilla:

`{{ cookiecutter.saluda }}`

- **Bloques de declaración:** se usan para el uso de condicionales, ciclos, etc.:

```python
{% if coockiecutter.eres_asombroso %}
. . .
{% endif %}
```

- **Bloques de comentario:** se usan para dejar comentarios o recomendaciones a los propios desarrolladores:

`{# Esto es un comentario #}`

### Instalación de Cookiecutter

**Importante:** Todos los pasos son vía consola y con Anaconda instalado.

**Instalar conda:**

para wsl en windows 

`wget -O anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh`

se inicia la instalacion con:

`bash anaconda.sh`

darle a todo yes:

Al instalarse debe aparecer (base) en el anviente, siempre hay que cambiar la terminal a otra de wsl para que aplique los cambios. sino lo detecta se utiliza los siguientes pasos:

Si tu sistema no detecta el comando `conda`, es posible que Anaconda no esté instalada correctamente, o que no se haya agregado a tu variable de entorno PATH. Aquí te dejo los pasos para solucionarlo:

### 1. Verificar la instalación de Anaconda

Primero, verifica si **Anaconda** está correctamente instalada ejecutando el siguiente comando:

```bash
which conda
```

Si no devuelve ninguna ruta, significa que **Anaconda** no está instalada correctamente o que no se agregó al **PATH** del sistema.

### 2. Agregar Anaconda al PATH manualmente

Si Anaconda está instalada pero no se encuentra en el PATH, puedes agregarla manualmente. Sigue estos pasos dependiendo de tu sistema operativo:

#### Para **Linux** o **macOS**:

1. Abre la terminal.
2. Agrega la ruta de Anaconda al archivo de configuración de tu shell (`~/.bashrc`, `~/.zshrc`, etc.). La ruta típica de Anaconda es `~/anaconda3` o `~/miniconda3`.

   Edita tu archivo de configuración con el siguiente comando:

   ```bash
   nano ~/.bashrc
   ```

3. Añade esta línea al final del archivo:

   ```bash
   export PATH="$HOME/anaconda3/bin:$PATH"
   ```

   Si instalaste Miniconda, cambia `anaconda3` por `miniconda3`.

4. Guarda el archivo y actualiza la sesión de la terminal con:

   ```bash
   source ~/.bashrc
   ```

   Para **zsh**:

   ```bash
   source ~/.zshrc
   ```

5. Verifica que `conda` se puede detectar ahora:

   ```bash
   conda --version
   ```

#### Para **Windows**:

1. Abre **Anaconda Prompt** (busca "Anaconda Prompt" en el menú Inicio).
2. Si `conda` funciona desde **Anaconda Prompt** pero no desde **cmd** o **PowerShell**, probablemente sea un problema de PATH.
3. Para agregar **Anaconda** al PATH en **Windows**:
   - Ve al **Panel de control** → **Sistema y seguridad** → **Sistema** → **Configuración avanzada del sistema**.
   - Haz clic en **Variables de entorno**.
   - En **Variables del sistema**, selecciona **Path** y haz clic en **Editar**.
   - Agrega la ruta de instalación de **Anaconda** (ejemplo: `C:\Users\TuUsuario\Anaconda3`) y el subdirectorio `Scripts` (ejemplo: `C:\Users\TuUsuario\Anaconda3\Scripts`).

4. Después de hacer esto, abre una nueva ventana de **cmd** o **PowerShell** y ejecuta:

   ```bash
   conda --version
   ```

### 3. Reinstalar Anaconda

Si agregar la ruta al PATH no soluciona el problema, es posible que la instalación esté corrupta. Puedes intentar reinstalar Anaconda:

1. Descarga la última versión de Anaconda desde [aquí](https://www.anaconda.com/products/individual).
2. Sigue las instrucciones de instalación para tu sistema operativo.

Asegúrate de seleccionar la opción "Agregar Anaconda al PATH" durante la instalación en **Windows**.

### 4. Usar Miniconda (alternativa ligera)

Si sigues teniendo problemas con Anaconda, puedes optar por **Miniconda**, que es una versión más ligera y rápida de instalar:

- [Descargar Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Después de instalar Miniconda, el comando `conda` debería funcionar correctamente.

### 5. Reiniciar la terminal/sistema

Después de hacer cambios en el PATH o instalar Anaconda, asegúrate de cerrar todas las terminales y reiniciarlas o reiniciar tu computadora para que los cambios surtan efecto.

Intenta seguir estos pasos y asegúrate de que `conda` esté correctamente configurado en tu entorno.

Para instalar **Anaconda** desde la terminal, sigue estos pasos según el sistema operativo que utilices.

### 1. Descargar el instalador de Anaconda

#### Linux/Mac:
Puedes descargar la versión más reciente de Anaconda para **Linux** o **MacOS** desde la línea de comandos con **wget** o **curl**.

**Para Linux**:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
```

**Para macOS**:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-latest-MacOSX-x86_64.sh
```

O puedes usar `curl`:
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
```

### 2. Ejecutar el instalador

Después de descargar el archivo, ejecuta el instalador con el siguiente comando:

```bash
bash Anaconda3-latest-Linux-x86_64.sh
```

Este comando es para **Linux**, pero si estás usando macOS, el nombre del archivo puede variar ligeramente, simplemente reemplázalo por el nombre correcto.

### 3. Aceptar el Acuerdo de Licencia

Sigue las instrucciones en la terminal. Se te pedirá que aceptes el acuerdo de licencia. Puedes hacerlo escribiendo:

```bash
yes
```

Luego elige el directorio donde deseas instalar Anaconda (por defecto será `~/anaconda3`).

### 4. Configurar Anaconda

El instalador te preguntará si deseas inicializar **Anaconda** en tu archivo `.bashrc` o `.zshrc`. Si deseas que se agregue automáticamente a tu entorno de terminal, selecciona **yes**.

### 5. Activar el entorno base de Anaconda

Para activar Anaconda, ejecuta este comando:

```bash
source ~/.bashrc
```

En **macOS** con **zsh**, podrías tener que usar:

```bash
source ~/.zshrc
```

### 6. Verificar la instalación

Finalmente, verifica que **Anaconda** se haya instalado correctamente ejecutando:

```bash
conda --version
```

Si ves la versión de **conda**, significa que la instalación fue exitosa.

1. Crea una carpeta un entrar en ella:

```python
mkdir <nombre_carpeta>
cd <nombre_carpeta>
```

2. Agrega el canal Conda-Forge a tu configuración global:

conda config --add channels conda-forge

3. Crea un ambiente virtual que contenga a Coockiecutter:

`conda create --name <nombre_ambiente> coockiecutter=1.7.3`

4. Activa el ambiente virtual:

`conda activate <nombre_ambiente>`

5. Definir en dónde estará tu ambiente:

`conda env export --from-history --file environment.yml`

Para desactivar el ambiente virtual:

`conda deactivate`cookiecutter

Crear un nuevo proyecto

`cookiecutter https://github.com/platzi/curso-entorno-avanzado-ds --checkout cookiecutter-personal-platzi`

**Lecturas recomendadas**

[Cookiecutter — cookiecutter 1.7.2 documentation](https://cookiecutter.readthedocs.io/en/1.7.2/README.html "Cookiecutter — cookiecutter 1.7.2 documentation")

[Home - Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/ "Home - Cookiecutter Data Science")

[GitHub - platzi/curso-entorno-avanzado-ds](https://github.com/platzi/curso-entorno-avanzado-ds "GitHub - platzi/curso-entorno-avanzado-ds")

[Instalar conda](https://platzi.com/home/clases/2434-jupyter-notebook/40394-instalar-conda-a-traves-de-la-terminal/ "Instalar conda")

### Crear plantillas de proyecto personalizadas

Estructura inicial de la plantilla
Dentro de la carpeta principal crea la carpeta que contendrá todo lo que necesitarás en tu proyecto con el nombre:

`{{ coockiecutter.project.slug }}`

En la carpeta recién creada agrega los siguientes archivos:

```
README.md
environment.yml
coockiecutter.json
```

También crea las carpetas que necesitará tu proyecto:

```bash
/data
/notebooks
```

Afuera de la carpeta, pero dentro de la carpeta principal, crea el siguiente archivo:

`environment.yml`

Hay dos archivos environment.yml, el de configuración de entorno (dentro de la carpeta que creaste) y el que configura las dependencias y paquetes (en la carpeta principal).

### Información de [README.md](http://readme.md/ "README.md")

Adentro del archivo [README.md](http://readme.md/ "README.md") agrega las siguientes líneas que lo harán un archivo dinámico:

```python
# {{ coockiecutter.project_title }}
By: {{ coockiecutter.project_author_name }}
{{ coockiecutter.project_description }}
```

Estas líneas, hechas en Jinja, permitirán a tu archivo acceder a las variables que contienen la información del título, autor y descripción del proyecto.

### Información de environment.yml (entorno)

```python
# conda env create --file environment.yml
name: cookiecutter-personal-platzi
channels:
  - anaconda
  - conda-forge
  - defaults
dependencies:
  - cookiecutter
```

### Información de environment.yml (configuración)

```python
# conda env create --file environment.yml
name: {{ cookiecutter.project_slug }}
channels:
  - anaconda
  - conda-forge
  - defaults
dependencies:
  {% if cookiecutter.project_packages == "All" -%}
  - fs
  - jupyter
  - jupyterlab
  - pathlib
  {% endif -%}
  - pip
  {% if cookiecutter.project_packages == "All" -%}
  - pyprojroot
  {% endif -%}
  - python={{ cookiecutter.python_version }}
  - pip:
    {% if cookiecutter.project_packages == "All" -%}
    - pyhere
    {% endif -%}
```

### Agregando información a coockiecutter.json

Dentro de este archivo configurarás todos los valores de las variables que utilizas en los demás archivos:

```json
{
    "project_title": "Cookiecutter Personal",
    "project_slug": "{{ coockiecutter.project_title.lower().replace(" ", "_").replace("-", "_") }}",
    "project_description": "Tu primer proyecto con Cookiecutter.",
    "project_author_name": "Tu nombre",
    "project_packages": ["All, Minimal"],
    "python_version": "3.7"
}
```

### Ejecuta el proyecto

- Inicializas el Coockiecutter con conda.
- Configuras la instalación, como en la clase anterior.

Para crear plantillas de proyectos personalizadas, puedes utilizar herramientas como **Cookiecutter**, que facilita la creación de proyectos a partir de una plantilla predefinida. A continuación te explico los pasos para crear una plantilla personalizada utilizando **Cookiecutter**, o si prefieres, cómo hacerlo manualmente.

### Crear una plantilla personalizada con **Cookiecutter**

#### Paso 1: Instalar **Cookiecutter**

Si aún no lo has hecho, instala Cookiecutter con el siguiente comando:

```bash
pip install cookiecutter
```

#### Paso 2: Crear la estructura de la plantilla

Crea un directorio que represente tu plantilla de proyecto. Dentro de este directorio, puedes usar variables para que **Cookiecutter** las reemplace con los valores que el usuario proporcionará.

Por ejemplo, crea un directorio llamado `project-template` con la siguiente estructura:

```
project-template/
│
├── cookiecutter.json
├── {{cookiecutter.project_name}}/
│   ├── README.md
│   ├── setup.py
│   ├── {{cookiecutter.module_name}}/
│   │   ├── __init__.py
│   │   └── main.py
└── LICENSE
```

#### Paso 3: Crear el archivo `cookiecutter.json`

Este archivo es donde defines las variables que deseas que **Cookiecutter** solicite al usuario cuando genere un nuevo proyecto. Por ejemplo:

```json
{
    "project_name": "my_project",
    "module_name": "my_module",
    "author_name": "Your Name",
    "license": ["MIT", "BSD", "GPL"]
}
```

#### Paso 4: Crear los archivos de tu plantilla

Dentro del directorio `{{cookiecutter.project_name}}`, puedes colocar los archivos que formarán parte de tu proyecto. Usa las variables definidas en el archivo `cookiecutter.json` para personalizar el contenido. Por ejemplo, el archivo `README.md` podría verse así:

```markdown
# {{cookiecutter.project_name}}

Autor: {{cookiecutter.author_name}}

Este es el proyecto {{cookiecutter.project_name}}. Fue creado utilizando una plantilla personalizada.
```

El archivo `setup.py` para un proyecto de Python podría ser algo así:

```python
from setuptools import setup, find_packages

setup(
    name="{{cookiecutter.project_name}}",
    version="0.1",
    packages=find_packages(),
    author="{{cookiecutter.author_name}}",
    license="{{cookiecutter.license}}",
)
```

#### Paso 5: Crear un módulo o código base

Dentro del directorio `{{cookiecutter.module_name}}`, crea el archivo `main.py`, que puede contener el código base para tu proyecto:

```python
def main():
    print("Bienvenido al proyecto {{cookiecutter.project_name}}")

if __name__ == "__main__":
    main()
```

#### Paso 6: Generar un nuevo proyecto usando la plantilla

Una vez que tengas tu plantilla creada, puedes usar **Cookiecutter** para generar un proyecto basado en ella. Usa el siguiente comando, indicando la ruta hacia tu plantilla:

```bash
cookiecutter path/to/project-template
```

**Cookiecutter** te pedirá que ingreses los valores para `project_name`, `module_name`, `author_name`, y cualquier otra variable que hayas definido. Generará un nuevo proyecto en base a esos valores.

### Crear una plantilla manualmente

Si no deseas usar Cookiecutter, puedes crear una plantilla de manera manual, simplemente configurando la estructura y archivos de tu proyecto en un directorio, y copiando esa estructura cada vez que inicies un nuevo proyecto. Por ejemplo, podrías hacer un script bash que copie la estructura de directorios y archivos necesarios:

```bash
#!/bin/bash

mkdir $1
cd $1
mkdir src tests
touch README.md setup.py .gitignore
echo "# $1" >> README.md
echo "Proyecto creado con estructura básica." >> README.md
```

Luego, podrías ejecutar el script proporcionando un nombre de proyecto:

```bash
./create_project.sh nombre_del_proyecto
```

Esto copiará la estructura base de tu plantilla manual para que puedas empezar rápidamente un nuevo proyecto.

## Implementar hooks

### Introducción a Hooks

Los Hooks son sentencias que se van a ejecutar antes o después de generar la plantilla de datos. Por ejemplo, puedes usarlos para verificar el nombre de una carpeta, actualizar git, etc.

### Implementación de Hooks

- Se crea la carpeta “hooks”, adentro de la carpeta principal de tu proyecto.
- Dentro de la carpeta se agregan los archivos “pre_gen_project.py” (lo que se ejecuta antes de generar la plantilla) y “pos_gen_project.py” (lo que se ejecuta después de generar la plantilla).

Por ejemplo, en “pre_gen_project.py” se puede inicializar git o validar nombres y archivos para evitar errores.

En el archivo “pos_gen_project.py” se puede hacer el primer commit en git o mostrar la finalización de la instalación de dependencias.

Los **hooks** en **Cookiecutter** son scripts que te permiten ejecutar código adicional antes o después de que se genere el proyecto a partir de una plantilla. Puedes usarlos para automatizar tareas adicionales, como la instalación de dependencias, la creación de archivos dinámicos o cualquier configuración extra que quieras realizar en tu plantilla personalizada.

### Tipos de hooks en Cookiecutter
1. **Pre-hook** (`pre_gen_project`): Se ejecuta antes de que Cookiecutter genere el proyecto.
2. **Post-hook** (`post_gen_project`): Se ejecuta después de que Cookiecutter genera el proyecto.

Estos hooks son scripts de Python o shell que se ejecutan durante el proceso de generación de la plantilla.

### Cómo implementar hooks en una plantilla Cookiecutter

#### Paso 1: Crear el directorio `hooks`

Dentro de tu estructura de plantilla, crea un directorio llamado `hooks`. Este directorio contendrá los scripts que quieres ejecutar antes o después de la generación del proyecto.

La estructura de la plantilla se verá así:

```
project-template/
├── cookiecutter.json
├── {{cookiecutter.project_name}}/
│   ├── README.md
│   ├── setup.py
│   └── {{cookiecutter.module_name}}/
│       ├── __init__.py
│       └── main.py
└── hooks/
    ├── pre_gen_project.py
    └── post_gen_project.py
```

#### Paso 2: Escribir el hook `pre_gen_project.py`

Este hook se ejecuta **antes** de que Cookiecutter genere el proyecto. Puedes usarlo para validar entradas o preparar el entorno.

Por ejemplo, si deseas asegurarte de que el nombre del proyecto no contiene espacios, puedes escribir un script `pre_gen_project.py`:

```python
import sys
from cookiecutter.utils import prompt_for_config

project_name = '{{ cookiecutter.project_name }}'

if " " in project_name:
    print("ERROR: El nombre del proyecto no debe contener espacios.")
    sys.exit(1)
```

#### Paso 3: Escribir el hook `post_gen_project.py`

Este hook se ejecuta **después** de que se ha generado el proyecto. Puedes usarlo para automatizar tareas adicionales, como inicializar un repositorio de Git, instalar dependencias o imprimir mensajes personalizados.

Por ejemplo, si quieres inicializar un repositorio de Git después de generar el proyecto, puedes crear un script `post_gen_project.py`:

```python
import os
import subprocess

# Inicializar un repositorio de Git
subprocess.run(['git', 'init'])

# Crear un commit inicial
subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', 'Initial commit'])
```

#### Paso 4: Probar los hooks

Una vez que has creado los hooks, puedes ejecutar **Cookiecutter** para probarlos:

```bash
cookiecutter path/to/project-template
```

Si todo está bien, el pre-hook se ejecutará antes de que se genere el proyecto y el post-hook después de que se genere el proyecto.

### Ejemplos de tareas comunes con hooks

- **Validar entradas**: Con un pre-hook, puedes validar que los valores ingresados en `cookiecutter.json` cumplan con ciertos requisitos.
- **Instalar dependencias**: Un post-hook puede ejecutar `pip install` o `npm install` automáticamente después de generar un proyecto.
- **Inicializar repositorios**: Usar hooks para crear un repositorio Git automáticamente y hacer un commit inicial.
- **Generar archivos adicionales**: Después de la generación del proyecto, un post-hook puede generar archivos adicionales o modificar archivos existentes.

### Resumen

Los **hooks** en Cookiecutter son una poderosa forma de personalizar aún más tus plantillas de proyectos, permitiéndote automatizar tareas complejas y mantener la flexibilidad. Con ellos puedes ejecutar validaciones previas o tareas de post-procesamiento de manera eficiente, optimizando el flujo de trabajo del desarrollo de proyectos.

## Distribuir plantilla de proyecto

Para distribuir una plantilla de proyecto de manera eficiente, puedes utilizar varias estrategias que permiten compartir y reutilizar la plantilla con otros desarrolladores o equipos. A continuación te explico algunos enfoques comunes para distribuir tu plantilla:

### 1. **Distribuir la plantilla usando Git**

Puedes almacenar la plantilla del proyecto en un repositorio Git y compartirlo con otras personas. Esto permite que cualquiera clone o descargue el repositorio para crear nuevos proyectos.

#### Paso 1: Crear un repositorio en Git

- Crea un repositorio Git, ya sea localmente o en una plataforma como **GitHub**, **GitLab**, **Bitbucket**, etc.
- Inicializa el repositorio en tu plantilla:
  ```bash
  git init
  git add .
  git commit -m "Plantilla de proyecto"
  ```

#### Paso 2: Subir el repositorio a GitHub (u otro servicio)

- Si usas GitHub, crea un nuevo repositorio en la plataforma, luego sigue las instrucciones para conectarlo a tu repositorio local:

  ```bash
  git remote add origin https://github.com/tu_usuario/tu_repositorio.git
  git push -u origin master
  ```

#### Paso 3: Clonar el repositorio

Quienquiera que quiera utilizar la plantilla puede clonar el repositorio y empezar a trabajar:

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
```

### 2. **Distribuir con Cookiecutter**

Si estás usando **Cookiecutter** para tu plantilla, también puedes compartirla a través de un repositorio Git, y los usuarios podrán generar nuevos proyectos basados en la plantilla de forma interactiva.

#### Paso 1: Crear el repositorio Git con tu plantilla Cookiecutter

Sigue los mismos pasos mencionados anteriormente para crear un repositorio con tu plantilla de Cookiecutter.

#### Paso 2: Distribuir la plantilla con Cookiecutter

Una vez que la plantilla esté disponible en GitHub u otra plataforma de control de versiones, otros desarrolladores pueden generar un proyecto basado en tu plantilla usando **Cookiecutter**.

Por ejemplo, si la plantilla está en `https://github.com/tu_usuario/mi_plantilla_cookiecutter`, puedes ejecutar el siguiente comando:

```bash
cookiecutter https://github.com/tu_usuario/mi_plantilla_cookiecutter
```

### 3. **Distribuir como paquete en PyPI (para plantillas Python)**

Si has creado una plantilla específica para proyectos en Python, puedes empaquetarla como un paquete de Python y distribuirla a través de **PyPI**. Esto permite que cualquiera instale tu plantilla con `pip`.

#### Paso 1: Crear el archivo `setup.py`

Define el archivo `setup.py` en tu plantilla para describir el paquete:

```python
from setuptools import setup, find_packages

setup(
    name='nombre_de_tu_plantilla',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'iniciar_proyecto=nombre_de_tu_plantilla.main:start_project',
        ],
    },
)
```

#### Paso 2: Subir el paquete a PyPI

Regístrate en [PyPI](https://pypi.org/) y usa las herramientas como **twine** para subir tu paquete:

```bash
pip install twine
python setup.py sdist
twine upload dist/*
```

Después de subir el paquete, otros usuarios pueden instalar y ejecutar tu plantilla con `pip`:

```bash
pip install nombre_de_tu_plantilla
iniciar_proyecto
```

### 4. **Compartir la plantilla como un archivo comprimido (ZIP/TAR)**

Si prefieres no usar control de versiones, puedes distribuir tu plantilla como un archivo comprimido.

#### Paso 1: Empaquetar la plantilla

Crea un archivo ZIP o TAR de la plantilla:

```bash
zip -r mi_plantilla.zip /ruta/a/mi_plantilla
```

#### Paso 2: Compartir el archivo

Puedes compartir este archivo a través de correo electrónico, almacenamiento en la nube (Google Drive, Dropbox, etc.), o cualquier otro medio. Los usuarios solo tendrán que descomprimir el archivo y comenzar a usar la plantilla.

### 5. **Distribuir usando gestores de plantillas (como Yeoman o otros)**

Si estás desarrollando plantillas para otras tecnologías como JavaScript, puedes usar herramientas específicas como **Yeoman**, que es un generador de scaffolding para proyectos web. Crearías un generador con Yeoman, lo empaquetarías, y otros desarrolladores podrían usarlo para generar nuevos proyectos con el comando:

```bash
yo nombre-de-tu-generador
```

### 6. **Publicar en un repositorio de plantillas (marketplaces o comunidades)**

- **Cookiecutter Templates**: Publica tu plantilla en el directorio oficial de Cookiecutter para que esté disponible para otros usuarios. Puedes encontrarlo en [cookiecutter's repository](https://github.com/cookiecutter/cookiecutter).
- **GitHub Marketplace**: Puedes publicar tus plantillas en GitHub Marketplace si tienes scripts o configuraciones que pueden ser útiles para otros.
- **Plantillas en PyPI**: Si tienes plantillas para Python, puedes publicarlas en PyPI para que estén fácilmente disponibles a través de `pip`.

### Resumen

- **Git**: Almacena y distribuye la plantilla en un repositorio remoto.
- **Cookiecutter**: Genera nuevos proyectos basados en plantillas interactivamente.
- **PyPI**: Empaqueta tu plantilla como un paquete Python.
- **ZIP**: Empaqueta la plantilla en un archivo comprimido y distribúyela.
- **Yeoman**: Usado para plantillas JavaScript/web.

## Manejo de rutas del sistema: OS

### Objetivo

Crear la ruta “./data/raw/” independiente del sistema operativo. En este caso usaremos os, un módulo de Python que sirve para manejar rutas.

**IMPORTANTE**: cerciórate de que estás trabajando en el entorno correcto.

### Implementación

Dentro del notebook de jupyter:

```bash
import os

CURRENT_DIR = os.getcwd()  # Ruta actual de trabajo
DATA_DIR = os.path.join(CURRENT_DIR, os.pardir, "data", "raw")  # Ruta objetivo (os.pardir: ruta padre)

os.path.exists(DATA_DIR)  # Revisa si el path existe
os.path.isdir(DATA_DIR)  # Revisa si es un directorio

os.listdir(DATA_DIR)  # Itera por los archivos dentro del directorio

os.mkdir(os.path.join(DATA_DIR, "os"))  # Crea la carpeta *"os"*
```

**Lecturas recomendadas**

[os.path — Common pathname manipulations — Python 3.9.7 documentation](https://docs.python.org/3/library/os.path.html "os.path — Common pathname manipulations — Python 3.9.7 documentation")

## Manejo de rutas del sistema: Pathlib

### Objetivo

Crear la ruta “./data/raw/” independiente del sistema operativo. Ahora usaremos pathlib, otro módulo de Python.

### Implementación

Dentro del notebook de jupyter:

```python
import pathlib

pathlib.Path()  # Genera un objeto Unix Path o 

CURRENT_DIR = pathlib.Path().resolve()  # Path local completo
DATA_DIR = CURRENT_DIR.parent.joinpath("data", "raw")  # Directorio objetivo

DATA_DIR.exists()  # Revisa si el directorio existe
DATA_DIR.is_dir()  # Revisa si es un directorio
```

Utiliza el método “parent” para obtener el directorio padre y de ahí concatenar el path de las carpetas “data” y “raw”.

Puedes crear una carpeta dentro de un directorio, usando el método “mkdir”:

`DATA_DIR.joinpath("<nombre_carpeta>").mkdir()`

Para buscar la ruta de un archivo dentro del proyecto, usando regex:

`list(DATA_DIR.glob("<nombre_archivo>"))`

**Lecturas recomendadas**

[pathlib — Object-oriented filesystem paths — Python 3.9.7 documentation](https://docs.python.org/3/library/pathlib.html "pathlib — Object-oriented filesystem paths — Python 3.9.7 documentation")

## Manejo de rutas del sistema: PyFilesystem

### Objetivo

Crear la ruta “./data/raw/” independiente del sistema operativo. Ahora usaremos PyFilesystem2.

### Implementación

Dentro del notebook de jupyter:

```python
import fs

fs.open_fs(".")  # Abre una conexión con el path actual (OSFS)

CURRENT_DIR = fs.open_fs(".")

CURRENT_DIR.exists(".")  # Revisa si el directorio existe
DATA_DIR.listdir(".")  # Muestra el contenido dentro de la ruta.
```

- PyFilesystem2 genera un objeto OSFS (Operating System Filesystem).

- El inconveniente con este módulo es que el objeto OSFS solo detecta los objetos que existen en la ruta actual, por lo que si intentas acceder a un archivo ubicado en el directorio padre “…” te saltará un *IndexError*.

- Si necesitas que el objeto OSFS también detecte el directorio padre, además de las carpetas “data” y “raw”, vuelve a generar el objeto de la siguiente forma:

`fs.open_fs("../data/raw/")  # Ruta objetivo`

**Lecturas recomendadas**

[Introduction — PyFilesystem 2.4.13 documentation](https://docs.pyfilesystem.org/en/latest/introduction.html "Introduction — PyFilesystem 2.4.13 documentation")