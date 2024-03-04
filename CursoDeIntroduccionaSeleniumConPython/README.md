# Curso de Introducción a Selenium con Python

### Historia de Selenium

¿Qué es Selenium? Es una SUIT de herramientas para la automatización de navegadores Web. El objetivo de Selenium NO fue para el Testing ni para el Web Scraping (aunque se puede usar para eso), por lo tanto, no es el más optimo para estas actividades. Protocolo: WebDriver, herramienta que se conecta a un API. Selenium WebDriver es la herramienta que utilizaremos en el curso. -Selenium NO es un Software, ES una SUIT de Softwares. *DDT: Data Drive Testing: Ingresar datos para que realice varias pruebas (sin intervención humana).

**¿Qué es Selenium?** Es una **SUIT de herramientas** para la automatización de navegadores Web. El objetivo de Selenium NO fue para el **Testing** ni para el **Web Scraping** (aunque se puede usar para eso), por lo tanto, no es el más optimo para estas actividades. **Protocolo: WebDriver**, herramienta que se conecta a un API. Selenium WebDriver es la herramienta que utilizaremos en el curso. -Selenium NO es un Software, ES una SUIT de Softwares. **+DDT: Data Drive Testing: **Ingresar datos para que realice varias pruebas (sin intervención humana).

**Selenium IDE**

**Pros**

- Excelente para iniciar
- No requiere saber programar
- Exporta scripts para Selenium RC y Selenium WebDriver
- Genera reportes

**Contras**

- Disponible para Google Chrome y FireFox
- No soporta DDT. No permite colocar datos para múltiples pruebas.

**Selenium RC**

**Pros**

- Soporte para
 - Varias plataformas, navegadores y lenguajes.
 - Operaciones lógicas y condicionales
 - DDT
- Posee una API madura

**Contras**

- Complejo de instalar
- Necesita de un servidor corriendo.
- Comandos redundantes en una API
- Navegación no tan realista

**Selenium Web Driven**

**Pros**

- Soporte para múltiples lenguajes
- Facil de instalar.
- Comunicación directa con el navegador.
- Interacción más realista.

**Contra**

- No soporta nuevos navegadores tan rápido.
- No genera reportes o resultados de pruebas.
- Requiere de saber programar.

### Otras herramientas de testing y automatización

**Puppeteer:**

- *PROS*: Soporte por parte de Google, te brinda datos del Performance Analysis de Chrome y un mayor control de este navegador. No requiere archivos externos como lo hace Selenium con WebDriver.

- *CONTRAS*: Solo funciona para Google Chrome con JavaScript, tiene una comunidad pequeña así que el apoyo será poco.

**Cypress.io:**

- *PROS*: Tiene una comunidad emergente y va creciendo a pasos acelerados, tiene muy buena documentación para implementar Cypress en los proyectos. Es muy ágil en pruebas E2E, está orientado a desarrolladores y tiene un excelente manejo del asincronismo, logrando que las esperas sean dinámicas y también se puedan manejar fácilmente.

- *CONTRAS*: Solo funciona en Google Chrome con JavaScript, se pueden realizar pruebas en paralelo únicamente en la versión de pago.

### Configurar entorno de trabajo

pasos:

1. Verificar las version de Python Instalada en el equipo

`python --version` o linux `python3 --version`

2. Instalar Selenium con el commando: `pip install selenium`

3. Instalar PyUnitReport con el comando: `pip install pyunitreport`

Nota: para eso yo creo un ambiente virtual:
en PowerShell: `python -m venv venv`
Linux: `python3 --python=python3 venv`

Para cargar el requirements.txt
`pip install -r requirements.txt`

Para Activar el ambiente:
`source venv/Scripts/activate`

### Compatibilidad con Python 3.9 y aprendiendo a utilizar múltiples versiones

Yo hice lo siguiente para Windows:

Desde la consola instalé virtualenv:

`py -m pip install --user virtualenv`

Luego creo el entorno virtual con la versión de Python que quiera:

`virtualenv env -p python3.8` # yo instale 3.11.1

para activar el entorno es:

`source env\Scripts\activate`

Y por último se verifica la versión de Python instalada:

`python --version`

### Compatibilidad de Selenium con Python 3.9

¡Es aquí cuando das un gran paso en tu camino para convertirte en una developer profesional! Al crear un ambiente virtual estás aislando tu proyecto del resto de tu computadora y haciendo que funcione con módulos independientes. Es decir, para llevar este curso puedes tener una versión de Python y Selenium y para hacer otro proyecto puedes tener versiones distintas. Esto hace que los proyectos no se rompan.

Usualmente, sin hacer uso de ambientes virtuales, los proyectos en tu computadora se verían así:

![](https://static.platzi.com/media/user_upload/Untitled-bf9d42f1-5c44-4521-8b1f-8052334b96c0.jpg)

Pero, al organizarlo profesionalmente, tus proyectos aislados en ambientes virtuales se verían de esta forma:

![](https://static.platzi.com/media/user_upload/Untitled%20%281%29-8f77947f-7ca7-4c49-9ab4-c462d734678f.jpg)

**¿Cómo crear y activar un ambiente virtual?**
Primero veamos cómo hacerlo en sistemas basados en Unix como Linux y MacOS.
Te ubicas en la carpeta root del proyecto y corres los siguientes comandos:

```python
#Para crear el ambiente virtual
python3 -m venv nameOfVirtualEnv

#Luego lo tienes que activar
source nameOfVirtualEnv/bin/activate

#Lo puedes desactivar así
deactivate
```

Si trabajas en Windows puedes poner lo siguiente:

```python
#Crear
py -m venv nameOfVirtualEnv

#Activar
.\nameOfVirtualEnv\Scripts\activate

#Desactivar 
deactivate
```

Te sugiero que, si trabajas en Windows, uses una terminal basada en Unix como Cmder o un WSL. Además, el[ Curso de Introducción a la Terminal y Línea de Comandos](https://platzi.com/cursos/terminal/ " Curso de Introducción a la Terminal y Línea de Comandos") te viene perfecto para dominar la terminal.

**Se muestran errores en la terminal, ¿qué hago?**

Tranquila, tranquilo. Los errores son tus amigos. Si te sale un error, lee, interpreta o googlea. Al final actúa racionalmente. Pero lo más probable es que haya errores por no tener paquetes descargados o no tener Python actualizado. Soluciónalo así:

```python
sudo apt update
sudo apt -y upgrade

#Instalando el módulo para ambientes virtuales

sudo apt-get install python3.8-venv #o python3.9-venv según la versión
```

Instalando dependencias
Ahora que ya sabes qué es un ambiente virtual, cómo crearlo y cómo activarlo, llegó la hora de instalar dependencias usando pip. En el caso particular del curso, instalaremos Selenium en su versión más actualizada que es 4.1.1 y Python 3.9. Puedes ver el historial de versiones de Selenium y su compatibilidad con las versiones de Python dando click aquí.

Recuerda tener activado el ambiente virtual. Luego pones esto en la terminal.

```python
pip install selenium==4.1.3
sudo apt-get install python3.9-venv
```

¡Listo, ya puedes seguir con el curso! Existen otros instaladores como pyenv. Pero con pip puedes hacer cosas muy avanzadas.

Te animo a revisar el [Curso de Python Intermedio: Comprehensions, Lambdas y Manejo de Errores](https://platzi.com/cursos/python-intermedio/ "Curso de Python Intermedio: Comprehensions, Lambdas y Manejo de Errores") en donde se explican los ambientes virtuales y más cosas interesantes sobre Python.deactivate
