# Curso de Python: PIP y Entornos Virtuales

# Game Proyect

para correr el juego debe seguir las siguientes instrucciones en la terminal:

```sh
cd game
python3 main.py
```

# App Project

```sh
git clone
cd app
python3 -m venv env
souerce venv/local/bin/activate
pip3 install -r requirements.txt
python3 main.py
```


### Instalar en Windows

#### Comandos Utilizados

- python

- python3

- exit() para salir de la interfaz de python

#### Instalación

- apt update

- sudo apt update

- sudo apt -y upgrade

#### Verificar Instalación de python

- python3 -V

#### Instalación de gestor de paquetes de dependencias

- sudo apt install -y python3-pip

#### Verificar Instalación del gestor

- pip3 -V

#### Dependencias en entorno profesional

- sudo apt install -y build-essential libssl-dev libffi-dev python3-dev

### Instalar en MAC

### Comandos utilizados

- python o python3

- exit()

**Normalmente viene instalado en Mac, en caso de que no lo tenga continuar con estos comandos Herramientas de codigo**

- sudo xcode-select --install

- sudo xcode-select --reset

### Instalación de python

- brew install python3

### Verificar la Instalación

- python3

### Crear el .gitignore

[gitignore.io](https://www.toptal.com/developers/gitignore "gitignore.io")

### ¿Qué es un ambiente virtual?

Instalar a nivel global puede causar distintos problemas al momento de manejar diferentes proyectos, por ejemplo para algunos proyectos necesitaras otro tipo de version, libreria o modulos y para solucionar esto se creo un ambiente virtual en python el cual encapsula cada proyecto y no lo deja de forma compartida.

![¿Qué es un ambiente virtual?1](https://static.platzi.com/media/user_upload/Captura%20de%20Pantalla%202022-10-21%20a%20la%28s%29%2012.52.15%20a.m.-ce482717-55b4-464b-964f-db56ce4adce1.jpg "¿Qué es un ambiente virtual?1")
![¿Qué es un ambiente virtual?2](https://static.platzi.com/media/user_upload/Captura%20de%20Pantalla%202022-10-21%20a%20la%28s%29%2012.57.43%20a.m.-d3eac289-2ffb-465c-a818-7a91f9b3a661.jpg "¿Qué es un ambiente virtual?2")

 ### Usando entornos virtuales en Python

#### Verificar donde esta python y pip
- which python3
- which pip3

#### Si estas en linus o wsl debes instalar

- sudo apt install -y python3-venv

#### Poner cada proyecto en su propio ambiente, entrar en cada carpeta.

- python3 -m venv env o virtualenv --python=python3 venv

#### Activar el ambiente

- source env/bin/activate

### Salir del ambiente virtual

- deactivate

#### Podemos instalar las librerias necesarias en el ambiente virtual como por ejemplo

- pip3 install matplotlib==3.5.0

#### Verificar las instalaciones

- pip3 freeze

### requirements.txt

Requirements.txt = Archivo que gestiona todas las dependencias y en que versiones se necesitan.

Generar el archivo con el siguiente comando

- pip3 freeze > requirements.txt
Revisar lo que hay dentro del archivo

* cat requirements.txt
Instalar las dependencias necesarias para contribuir más rápido en proyectos

- pip3 install -r requirements.txt
Preparar archivo para contribución

```sh
git clone
cd app
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 main.py
```

![pasos](https://static.platzi.com/media/user_upload/Purple%20and%20White%20Simple%20Memo-13837b27-dd0d-40da-8767-4ac3f84afd91.jpg "pasos")

### Python para Backend: web server con FastAPI

- Navegar a proyecto Web

`cd ../web-server`

- Activar ambiente del proyecto

`source env/bin/activate `

- Agregar nuevas librerías FastAPI

`pip3 install fastapi`

- Agregar ASGI (Asynchronous Server Gateway Interface) Uvicorn

`pip3 install "uvicorn[standard]"`

- Verificar librerías instaladas

`pip3 freeze`

- Actualizar Requirements

` pip3 freeze > requirements.txt`

**Python para Backend: web server con FastAPI**

uvicorn servidor web para correr mis aplicaciones como servidor

```sh
uvicorn main:app --reload # Para correr mi aplicación en un servidor # reaload para volver a cargar el comando cada vez que modifique el archivo principal
```

#### Lecturas recomendadas

[fastAPI](https://fastapi.tiangolo.com/#installation "fastAPI")

[Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#html-response "Custom Response - HTML, Stream, File, others")

[curso-python-pip/web_server at master platzi/curso-pip*GitHub](https://github.com/platzi/curso-python-pip/tree/master/web-server "curso-python-pip/web_server at master platzi/curso-pip*GitHub")