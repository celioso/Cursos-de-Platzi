# Curso de PostgreSQL

## ¿Qué es Postgresql?

**PostgreSQL** es un **sistema de gestión de bases de datos relacional (RDBMS)** de código abierto, altamente robusto, escalable y confiable. Es conocido por su cumplimiento con los estándares SQL y por ofrecer extensibilidad avanzada. Fue desarrollado originalmente en la Universidad de California, Berkeley, y desde entonces ha evolucionado hasta convertirse en una de las opciones más populares para manejar bases de datos complejas.

### **Características principales de PostgreSQL:**

1. **Código abierto:** Es gratuito y su código fuente está disponible para modificaciones personalizadas.

2. **Soporte para estándares SQL:** Cumple con las normas SQL, lo que garantiza portabilidad y compatibilidad.

3. **Extensibilidad:** Permite crear funciones definidas por el usuario, tipos de datos personalizados y módulos para ampliar sus capacidades.

4. **Soporte para datos estructurados y no estructurados:**
   - Tablas relacionales tradicionales (estructuradas).
   - Datos JSON/JSONB (no estructurados).
   - Funciones avanzadas como índices GIN/JSON.

5. **Integridad transaccional:** Utiliza el modelo **ACID** (Atomicidad, Consistencia, Aislamiento, Durabilidad), asegurando confiabilidad en transacciones.

6. **Rendimiento avanzado:**
   - Índices potentes: B-tree, GIN, GiST, etc.
   - Optimización de consultas.
   - Paralelización para consultas complejas.

7. **Alta disponibilidad y recuperación:** Incluye características como:
   - Replicación en streaming.
   - Recuperación de datos en caso de fallos.
   - Failover automático con herramientas externas.

8. **Soporte para lenguajes de programación:** PostgreSQL permite trabajar con múltiples lenguajes de programación como **PL/pgSQL**, Python, Java, entre otros.

9. **Escalabilidad:** Puede manejar desde pequeñas aplicaciones hasta bases de datos empresariales masivas con petabytes de datos.

10. **Seguridad avanzada:**
    - Autenticación y control de acceso basados en roles.
    - Conexiones cifradas mediante TLS/SSL.
    - Encriptación de datos sensibles.

### **Casos de uso comunes:**

- **Aplicaciones web y móviles:** Usado por grandes plataformas como Instagram, Reddit y Spotify.
- **Sistemas analíticos:** Ideal para consultas complejas y modelado de datos.
- **Almacenamiento de datos geoespaciales:** Con la extensión PostGIS.
- **Integración con Big Data:** Compatible con herramientas como Hadoop y Spark.

### **Ventajas de usar PostgreSQL:**
- Comunidad activa y soporte continuo.
- Herramientas modernas para administración y monitoreo.
- Flexibilidad para adaptar el sistema a necesidades específicas.

Si trabajas en proyectos de ETL, ciencia de datos o desarrollo de aplicaciones, PostgreSQL es una opción poderosa y confiable para gestionar tus bases de datos.

**Lecturas recomendadas** 

[Qué es el teorema CAP y cómo elegir la base de datos para tu proyecto](https://platzi.com/blog/que-es-el-teorema-cap-y-como-elegir-la-base-de-datos-para-tu-proyecto)

[PostgreSQL: The world's most advanced open source database](https://www.postgresql.org/)

## Instalación y configuración de la Base de Datos

Vamos a instalar PostgreSQL en nuestra computadora. A continuación veremos el paso a paso y algunos consejos útiles para instalar y configurar correctamente PostgreSQL en nuestro equipo. En éste caso, usaremos Windows, pero los pasos son bastante similares entre los diferentes sistemas operativos.

Primer paso: ir a [https://www.postgresql.org/.](https://www.postgresql.org/ "https://www.postgresql.org/.")

Actualmente, la página web oficial de postgres luce así:

![1](images/1.png)

Ten en cuenta que puedes ver esta página en diferentes idiomas, depende de la configuración predeterminada de idioma de tu navegador.

Hacer clic en el botón ‘Download’ (Descarga) que se encuentra en la parte inferior derecha. Veremos lo siguiente:

![1](images/2.png)

Veremos lo siguiente:
Seleccionamos la opción que corresponda con tu sistema operativo, para éste caso hacemos clic en “Windows”:

Veremos en la parte inferior:

![3](images/3.png)

Haz clic en el enlace “Download the installer”. Esto nos va a llevar a la Web de Enterprise DB o EDB. EDB es una empresa que ofrece servicios sobre el motor de base de datos PostgreSQL y ofrece un instalador para Postgres de manera gratuita.

![4](images/4.png)

Es altamente recomendable seleccionar la penúltima o antepenúltima versión. Si bien la última versión estable está disponible, en éste caso la 12.0, no es recomendable instalarla en nuestro equipo, ya que al momento de instalarla o usar un servicio en la Nube para Postgres, lo más seguro es que no esté disponible y sólo esté hasta la versión 11.5, que no es la última versión. Esto porque todos los proveedores de Infraestructura no disponen de la versión de Postgres más actual siempre (tardan un poco en apropiar los nuevos lanzamientos).

Si tienes un equipo con Linux, la instalación la puedes hacer directamente desde los repositorios de Linux, EDB ya no ofrece soporte para instaladores en Linux debido a que se ha vuelto innecesario, el repositorio de Linux con PostgreSQL ofrece una manera mucho más sencilla y estándar para instalar PostgreSQL en linux.

Segundo paso: descargamos la versión “Windows x86-64” (porque nuestro sistema operativo es de 64 bits). En caso de que tu equipo sea de 32 bits debes seleccionar la opción “Windows x86-32”.

Vamos a descargar la versión 11.5. Hacemos clic en Download y guardamos el archivo que tendrá un nombre similar a:
“postgresql-11.5-2-windows-x64.exe”

Ahora vamos a la carpeta donde descargamos el archivo .exe, debe ser de aproximadamente 190 MB, lo ejecutamos.

Veremos lo siguiente:

![5](images/5.png)

Hacemos clic en siguiente. Si deseas cambiar la carpeta de destino, ahora es el momento:

![6](images/6.png)

Seleccionamos los servicios que queremos instalar. En este caso dejamos seleccionados todos menos “Stack Builder”, pues ofrece la instalación de servicios adicionales que no necesitamos hasta ahora. Luego hacemos clic en siguiente:

![7](images/7.png)

Ahora indicamos la carpeta donde iran guardados los datos de la base de datos, es diferente a la ruta de instalación del Motor de PostgreSQL, pero normalmente será una carpeta de nuestra carpeta de instalación. Puedes cambiar la ruta si quieres tener los datos en otra carpeta. Hacemos clic en siguiente.

![8](images/8.png)

Ingresamos la contraseña del usuario administrador. De manera predeterminada, Postgres crea un usuario super administrador llamado postgres que tiene todos los permisos y acceso a toda la base de datos, tanto para consultarla como para modificarla. En éste paso indicamos la clave de ese usuario super administrador.

Debes ingresar una clave muy segura y guardarla porque la vas a necesitar después. Luego hacemos clic en siguiente.

![9](images/9.png)

Ahora si queremos cambiar el puerto por donde el servicio de Postgresql estará escuchando peticiones, podemos hacerlo en la siguiente pantalla, si queremos dejar el predeterminado simplemente hacemos clic en siguiente.

![10](images/10.png)

La configuración regional puede ser la predeterminada, no es necesario cambiarla, incluso si vamos a usarla en español, ya que las tildes y las eñes estarán soportadas si dejas la configuración regional predeterminada. Es útil cambiarla cuando quieras dejar de soportar otras funciones de idiomas y lenguajes diferentes a uno específico. Luego hacemos clic en siguiente:

![11](images/11.png)

En pantalla aparecerá el resumen de lo que se va a instalar:

![12](images/12.png)

Al hacer clic en siguiente se muestra una pantalla que indica que PostgreSQL está listo para instalar, al hacer clic de nuevo en siguiente iniciará la instalación, espera un par de minutos hasta que la aplicación termine.

Una vez terminada la instalación, aparecerá en pantalla un mensaje mostrando que PostgreSQL ha sido instalado correctamente.

![13](images/13.png)

Podemos cerrar ésta pantalla y proceder a comprobar que todo quedó instalado correctamente.

Vamos a buscar el programa PgAdmin, el cual usaremos como editor favorito para ejecutar en él todas las operaciones sobre nuestra base de datos.

También vamos a buscar la consola… Tanto la consola como PgAdmin son útiles para gestionar nuestra base de datos, una nos permite ingresar comando por comandos y la otra nos ofrece una interfaz visual fácil de entender para realizar todas las operaciones.

En el menú de Windows (o donde aparecen instalados todos los programas) buscamos “PgAdmin…”

![14](images/14.png)

Ahora buscamos “SQL Shell…”

![15](images/15.png)

Efectivamente, ahora aparecen las herramientas que vamos a utilizar en éste curso.
Ahora vamos a crear una base de datos de prueba usando la consola y comprobaremos si existe usando PgAdmin, la crearemos para validar que la conexión con el servicio de base de datos interno funciona correctamente.

Para ello abrimos la consola, buscamos SQL Shell y lo ejecutamos. Veremos algo así:

![16](images/16.png)

Lo que vemos en pantalla es la consola esperando que ingresemos cada parámetro para la conexión.

Primero está el nombre del parámetro. En éste caso es “Server” seguido de unos corchetes que contienen el valor predeterminado. Si presionamos “Enter” sin digitar nada la consola asumirá que te refieres al valor predeterminado, si en éste caso presionamos “Enter” el valor asumido será “Localhost”. Localhost se refiere a nuestra propia máquina, si instalaste la base de datos en el mismo pc que estás usando para la consola, el valor correcto es Localhost o 127.0.0.1 (representan lo mismo).

Podemos dejar todos los valores predeterminados (presionando “Enter”) hasta que la consola pregunte por la clave del usuario maestro:

![17](images/17.png)

Debemos ingresar la clave que usamos cuando estábamos instalando Postgres, de lo contrario no podremos acceder. Presionamos Enter y veremos a continuación una pantalla que nos indica que estamos logueados en la base de datos y estamos listos para hacer modificaciones.

De manera predeterminada, la base de datos instalada es Postgres, la cual no debemos tocar, ya que ejecuta funciones propias del motor. Es usada por el Motor de PostgreSQL para interactuar con todas las bases de datos que vayamos a crear en el futuro.

La siguiente imagen indica que estamos conectados a la base de datos Postgres. Vamos a crear una base de datos nueva y luego saltar el cursor a ésta base de datos recién creada.

![18](images/18.png)

Para ello escribimos el comando “CREATE DATABASE transporte_publico;” y presionamos “Enter”. Veremos:

![19](images/19.png)

El mensaje “CREATE DATABASE” justo después de la línea que acabamos de escribir indica que la base de datos fue creada correctamente.

Para saltar a la base de datos recién creada ejecutamos el comando “\c transporte_publico”, el cursor mostrará lo siguiente:

![20](images/20.png)

Ahora vamos a validar desde PgAdmin que la base de datos fué creada correctamente. Abrimos PgAdmin y nos encontramos con una lista de items a la izquierda, lo que significa que de manera predeterminada PgAdmin ha creado un acceso a nuestra base de datos local, el cual llamó “PostgreSQL 11”:

![21](images/21.png)

Al hacer hacer doble clic sobre éste elemento (“PostgreSQL 11”) nos pedirá ingresar la clave que hemos determinado para el super usuario postgres, al igual que la consola, hasta no ingresarla correctamente no nos podremos conectar:

![22](images/22.png)

Ingresamos la clave. Te recomiendo seleccionar la opción “Save Password” o “Guardar Contraseña”. Si la máquina sobre la que estás trabajando es de confianza, que seas sólo tú o tu equipo quien tenga acceso a ella, de lo contrario, no guardes la contraseña para mantenerla segura.
Veremos la lista de bases de datos disponibles, la predeterminada “postgres” y la que acabamos de crear usando la consola, lo que comprueba que la base de datos y la consola funcionan correctamente.

![23](images/23.png)

Ahora procedemos a eliminar la base de datos recién creada para comprobar que PgAdmin está correctamente configurada y si pueda realizar cambios sobre la base de datos.

Para ello hacemos clic derecho sobre el elemento “transporte_publico” y seleccionamos la opción “Delete/Drop”. Al mensaje de confirmar hacemos clic en OK.

Con ello, si el elemento “transporte_publico” desaparece del menú de la izquierda comprobamos que PgAdmin funcionan correctamente.

## Interacción con Postgres desde la Consola

Interactuar con **PostgreSQL desde la consola** implica utilizar el cliente de línea de comandos llamado **psql**, que es una herramienta interactiva para ejecutar comandos SQL y administrar bases de datos PostgreSQL. A continuación, se describen los pasos clave para hacerlo:

---

### **1. Acceso a la Consola de PostgreSQL**
Para iniciar sesión en la consola de PostgreSQL:

```bash
psql -U <usuario> -d <nombre_base_datos>
```

- **`-U`**: Especifica el nombre de usuario de PostgreSQL.
- **`-d`**: Especifica la base de datos a la que deseas conectarte.
- Te pedirá la contraseña del usuario.

Si no especificas la base de datos, puedes conectarte directamente al servidor:

```bash
psql -U <usuario>
```

---

### **2. Comandos Básicos de psql**

#### a) **Conectarse a una Base de Datos**
```sql
\c <nombre_base_datos>
```

#### b) **Listar Bases de Datos**
```sql
\l
```

#### c) **Listar Tablas**
Para ver las tablas de la base de datos actual:
```sql
\dt
```

#### d) **Ver Detalles de una Tabla**
Para inspeccionar la estructura de una tabla:
```sql
\d <nombre_tabla>
```

#### e) **Salir de la Consola**
Para salir de psql:
```sql
\q
```

#### f) **cambiar de base de datos**
Para cambiar base de datos:
```sql
\c
```

### **3. Consultas SQL Básicas**

#### a) **Crear una Tabla**
```sql
CREATE TABLE empleados (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(50),
    salario NUMERIC(10, 2),
    fecha_ingreso DATE
);
```

#### b) **Insertar Datos**
```sql
INSERT INTO empleados (nombre, salario, fecha_ingreso)
VALUES ('Juan Pérez', 50000.00, '2023-01-15');
```

#### c) **Consultar Datos**
```sql
SELECT * FROM empleados;
```

#### d) **Actualizar Datos**
```sql
UPDATE empleados
SET salario = 55000.00
WHERE id = 1;
```

#### e) **Eliminar Datos**
```sql
DELETE FROM empleados
WHERE id = 1;
```

---

### **4. Comandos de Administración**

#### a) **Crear una Base de Datos**
```sql
CREATE DATABASE <nombre_base_datos>;
```

#### b) **Eliminar una Base de Datos**
```sql
DROP DATABASE <nombre_base_datos>;
```

#### c) **Crear un Usuario**
```sql
CREATE USER <nombre_usuario> WITH PASSWORD '<contraseña>';
```

#### d) **Asignar Privilegios**
```sql
GRANT ALL PRIVILEGES ON DATABASE <nombre_base_datos> TO <nombre_usuario>;
```

---

### **5. Importar y Exportar Datos**

#### a) **Exportar una Base de Datos a un Archivo**
Desde la consola del sistema operativo:
```bash
pg_dump -U <usuario> -d <nombre_base_datos> -F c -f <archivo_exportado>.dump
```

#### b) **Importar una Base de Datos desde un Archivo**
```bash
pg_restore -U <usuario> -d <nombre_base_datos> <archivo_exportado>.dump
```

#### c) **Cargar Datos desde un Archivo CSV**
Dentro de psql:
```sql
COPY empleados FROM '/ruta/archivo.csv' DELIMITER ',' CSV HEADER;
```

---

### **6. Ayuda y Documentación**
Para obtener ayuda en cualquier momento dentro de **psql**:
- **Ayuda de comandos SQL**:  
  ```sql
  \h <comando>
  ```
- **Ayuda de comandos de psql**:  
  ```sql
  \?
  ```

Con estos pasos, puedes interactuar de manera eficiente con PostgreSQL desde la consola, ya sea para gestionar bases de datos o realizar consultas y transformaciones en los datopostgres

RESUMEN

1. ENTRAR A LA CONSOLA DE POSTGRES `psql -U postgres -W`
2. VER LOS COMANDOS \ DE POSTGRES `\?`
3. LISTAR TODAS LAS BASES DE DATOS `\l`
4. VER LAS TABLAS DE UNA BASE DE DATOS `\dt`
5. CAMBIAR A OTRA BD `\c nombre_BD`
6. DESCRIBIR UNA TABLA `\d nombre_tabla`
7. VER TODOS LOS COMANDOS SQL `\h`
8. VER COMO SE EJECTUA UN COMANDO SQL `\h nombre_de_la_funcion`
9. CANCELAR TODO LO QUE HAY EN PANTALLA `Ctrl + C`
10. VER LA VERSION DE POSTGRES INSTALADA, IMPORTANTE PONER EL ';' `SELECT version();`
11. VOLVER A EJECUTAR LA FUNCION REALIADA ANTERIORMENTE `\g`
12. INICIALIZAR EL CONTADOR DE TIEMPO PARA QUE LA CONSOLA TE DIGA EN CADA EJECUCION ¿CUANTO DEMORO EN EJECUTAR ESA FUNCION? `\timing`
LIMPIAR PANTALLA DE LA CONSOLA PSQL `Ctrl + L`

## PgAdmin: Interacción con Postgres desde la Interfaz Gráfica

### Interacción con PostgreSQL desde PgAdmin:

**PgAdmin** es una herramienta potente para gestionar bases de datos PostgreSQL mediante una interfaz gráfica (GUI). Aquí te explico algunos pasos básicos para interactuar con Postgres desde PgAdmin:

#### 1. **Conexión a la base de datos:**
   - Abre PgAdmin.
   - Haz clic en el botón **"Add New Server"** (Agregar nuevo servidor).
   - Llena los campos con la información de la conexión:
     - **Name:** Nombre que quieras darle al servidor.
     - **Host:** Dirección del servidor (por ejemplo, `localhost` o la IP del servidor).
     - **Port:** Puerto donde PostgreSQL escucha (por defecto suele ser `5432`).
     - **Maintenance Database:** Base de datos predeterminada para la conexión (si tienes una predefinida).
     - **Username:** Nombre de usuario de PostgreSQL.
     - **Password:** Contraseña del usuario.
   - Haz clic en **"Save"** para guardar la configuración.

#### 2. **Explorar la base de datos:**
   - Una vez conectados, podrás ver las bases de datos del servidor en la parte izquierda de la pantalla bajo **"Databases"**.
   - Haz doble clic sobre la base de datos que quieras explorar.

#### 3. **Ejecutar consultas SQL:**
   - Haz clic derecho sobre la base de datos o un esquema y selecciona **"Query Tool"** para abrir la ventana de ejecución de SQL.
   - Escribe tus sentencias SQL (consultas, procedimientos almacenados, etc.) en el editor.
   - Haz clic en **"Execute"** para ejecutar la consulta.

#### 4. **Explorar objetos dentro de la base de datos:**
   - PgAdmin muestra todos los objetos dentro de la base de datos, como tablas, vistas, funciones, etc.
   - Puedes expandir las carpetas para ver las tablas y demás elementos.

#### 5. **Ver registros y estructura de tablas:**
   - Haz doble clic en cualquier tabla para ver su estructura o los registros actuales.
   - Puedes modificar datos directamente desde la vista de registros si tienes los permisos adecuados.

#### 6. **Realizar operaciones básicas:**
   - Desde PgAdmin puedes realizar operaciones como:
     - **Crear tablas**: Clic derecho en "Tables" -> **"Create Table"**.
     - **Insertar datos**: Haz clic derecho sobre la tabla y selecciona **"Query Tool"** para ejecutar INSERT.
     - **Actualizar datos**: Realiza UPDATE directamente desde la ventana de consultas.
     - **Eliminar datos**: Ejecuta DELETE desde la ventana de consultas.

#### 7. **Seguridad y gestión de usuarios:**
   - Desde PgAdmin puedes gestionar los usuarios y permisos de la base de datos. Simplemente, explora la sección de **"Roles"** y **"Privileges"** en la base de datos.

### Ventajas de usar PgAdmin:
   - **Interfaz gráfica intuitiva**: Ideal para gestionar bases de datos sin necesidad de escribir comandos.
   - **Gestión completa**: Permite desde consultas simples hasta la administración de seguridad y configuración avanzada.
   - **Soporte para múltiples bases de datos**: No solo PostgreSQL, también soporta otros sistemas de bases de datos como MySQL y SQLite.

**Lecturas recomendadas**

[pgAdmin - PostgreSQL Tools](https://www.pgadmin.org/)

## Archivos de Configuración

Un archivo de configuración es un componente esencial en muchos sistemas de software. Estos archivos permiten almacenar parámetros y configuraciones que determinan cómo debe comportarse una aplicación o sistema. Son particularmente útiles porque permiten separar la lógica del código de los valores que podrían cambiar según el entorno o las necesidades.

### ¿Para qué sirven los archivos de configuración?
1. **Separación de la lógica y configuración:** Permiten mantener el código independiente de los valores configurables, como rutas de archivos, credenciales o parámetros específicos.
2. **Facilitan la administración:** Los valores se pueden modificar sin cambiar el código fuente.
3. **Portabilidad:** Ayudan a implementar la misma aplicación en diferentes entornos (desarrollo, prueba, producción).
4. **Seguridad:** Pueden almacenar credenciales y datos sensibles, aunque se recomienda gestionarlos de forma segura.

### Ejemplos de uso
1. **Bases de datos:** Configuración de host, usuario, contraseña, nombre de la base de datos, etc.
2. **API:** Llaves de acceso, rutas base, y parámetros de conexión.
3. **Aplicaciones web:** Puerto del servidor, claves de sesión, y configuraciones específicas del entorno.

### Formatos comunes de archivos de configuración
1. **JSON:** 
   - Estructurado y legible por humanos.
   - Ejemplo:
     ```json
     {
         "database": {
             "host": "localhost",
             "port": 5432,
             "user": "admin",
             "password": "password123"
         }
     }
     ```

2. **YAML:**
   - Similar a JSON, pero más legible debido a su formato simple.
   - Ejemplo:
     ```yaml
     database:
       host: localhost
       port: 5432
       user: admin
       password: password123
     ```

3. **INI:**
   - Formato sencillo con secciones y pares clave-valor.
   - Ejemplo:
     ```ini
     [database]
     host = localhost
     port = 5432
     user = admin
     password = password123
     ```

4. **TOML:**
   - Similar a INI, pero más moderno.
   - Ejemplo:
     ```toml
     [database]
     host = "localhost"
     port = 5432
     user = "admin"
     password = "password123"
     ```

5. **XML:**
   - Estructurado, pero más verboso.
   - Ejemplo:
     ```xml
     <config>
         <database>
             <host>localhost</host>
             <port>5432</port>
             <user>admin</user>
             <password>password123</password>
         </database>
     </config>
     ```

6. **ENV (Variables de Entorno):**
   - Simple y común para aplicaciones en contenedores.
   - Ejemplo:
     ```
     DATABASE_HOST=localhost
     DATABASE_PORT=5432
     DATABASE_USER=admin
     DATABASE_PASSWORD=password123
     ```

### Uso en Python
En Python, puedes usar diferentes bibliotecas para trabajar con estos formatos:
1. **JSON:**
   ```python
   import json
   with open("config.json") as file:
       config = json.load(file)
   print(config["database"]["host"])
   ```

2. **YAML:**
   ```python
   import yaml
   with open("config.yaml") as file:
       config = yaml.safe_load(file)
   print(config["database"]["host"])
   ```

3. **ConfigParser (para INI):**
   ```python
   import configparser
   config = configparser.ConfigParser()
   config.read("config.ini")
   print(config["database"]["host"])
   ```

4. **dotenv (para ENV):**
   ```python
   from dotenv import load_dotenv
   import os
   load_dotenv(".env")
   print(os.getenv("DATABASE_HOST"))
   ```

### Buenas prácticas
1. **No incluir credenciales sensibles en los archivos:** Usa servicios de gestión de secretos.
2. **Versionar con cuidado:** No incluir archivos con datos sensibles en sistemas de control de versiones (usar `.gitignore` si es necesario).
3. **Validación:** Implementa validaciones para asegurar que los valores del archivo son correctos antes de usarlos.
4. **Cifrado:** Considera cifrar los valores sensibles almacenados.

Los archivos de configuración son fundamentales para mantener la flexibilidad y la organización en cualquier proyecto de software.

Los archivos de configuración son tres principales:

- postgreql.conf
- pg.hba.conf
- pg_ident.conf

La ruta de los mismos depende del sistema Operarivo, para saber que que ruta estan, basta con hacer una Query

- SHOW config_file;

**NOTA**: siempre es bueno hacer una copia original de los archivos antes de modificarlos por si movemos algo que no.
**Lecturas recomendadas**

[Curso de Administración de Servidores Linux | Platzi](https://platzi.com/clases/linux/)

[Domina la Administración de Usuarios y Permisos en Servidores Linux](https://platzi.com/blog/administracion-usuarios-servidores-linux/)

[Visual Studio Code - Code Editing. Redefined](https://code.visualstudio.com/)

## Comandos más utilizados en PostgreSQL

**La Consola**

La consola en PostgreSQL es una herramienta muy potente para crear, administrar y depurar nuestra base de datos. podemos acceder a ella después de instalar PostgreSQL y haber seleccionado la opción de instalar la consola junto a la base de datos.

PostgreSQL está más estrechamente acoplado al entorno UNIX que algunos otros sistemas de bases de datos, utiliza las cuentas de usuario nativas para determinar quién se conecta a ella (de forma predeterminada). El programa que se ejecuta en la consola y que permite ejecutar consultas y comandos se llama psql, psql es la terminal interactiva para trabajar con PostgreSQL, es la interfaz de línea de comando o consola principal, así como PgAdmin es la interfaz gráfica de usuario principal de PostgreSQL.

Después de emitir un comando PostgreSQL, recibirás comentarios del servidor indicándote el resultado de un comando o mostrándote los resultados de una solicitud de información. Por ejemplo, si deseas saber qué versión de PostgreSQL estás usando actualmente, puedes hacer lo siguiente:

![24](images/24.png)

### Comandos de ayuda

En consola los dos principales comandos con los que podemos revisar el todos los comandos y consultas son:

- \? Con el cual podemos ver la lista de todos los comandos disponibles en consola, comandos que empiezan con backslash ( \ )

![25](images/25.png)


- \h Con este comando veremos la información de todas las consultas SQL disponibles en consola. Sirve también para buscar ayuda sobre una consulta específica, para buscar información sobre una consulta específica basta con escribir \h seguido del inicio de la consulta de la que se requiera ayuda, así: \h **ALTER**
De esta forma podemos ver toda la ayuda con respecto a la consulta **ALTER**

![26](images/26.png)

### Comandos de navegación y consulta de información

- \c Saltar entre bases de datos

- \l Listar base de datos disponibles

- \dt Listar las tablas de la base de datos

- \d <nombre_tabla> Describir una tabla

- \dn Listar los esquemas de la base de datos actual

- \df Listar las funciones disponibles de la base de datos actual

- \dv Listar las vistas de la base de datos actual

- \du Listar los usuarios y sus roles de la base de datos actual

### Comandos de inspección y ejecución

- \g Volver a ejecutar el comando ejecutando justo antes

- \s Ver el historial de comandos ejecutados

- \s <nombre_archivo> Si se quiere guardar la lista de comandos ejecutados en un archivo de texto plano

- \i <nombre_archivo> Ejecutar los comandos desde un archivo

- \e Permite abrir un editor de texto plano, escribir comandos y ejecutar en lote. \e abre el editor de texto, escribir allí todos los comandos, luego guardar los cambios y cerrar, al cerrar se ejecutarán todos los comandos guardados.

- \ef Equivalente al comando anterior pero permite editar también funciones en PostgreSQL

### Comandos para debug y optimización

- \timing Activar / Desactivar el contador de tiempo por consulta

### Comandos para cerrar la consola

- \q Cerrar la consola

### Ejecutando consultas en la base de datos usando la consola

De manera predeterminada PostgreSQL no crea bases de datos para usar, debemos crear nuestra base de datos para empezar a trabajar, verás que existe ya una base de datos llamada postgres pero no debe ser usada ya que hace parte del CORE de PostgreSQL y sirve para gestionar las demás bases de datos.

Para crear una base de datos debes ejecutar la consulta de creación de base de datos, es importante entender que existe una costumbre no oficial al momento de escribir consultas; consiste en poner en mayúsculas todas las palabras propias del lenguaje SQL cómo **CREATE**, **SELECT**, **ALTE**, etc y el resto de palabras como los nombres de las tablas, columnas, nombres de usuarios, etc en minúscula. No está claro el porqué de esta especie de “estándar” al escribir consultas SQL pero todo apunta a que en el momento que SQL nace, no existían editores de consultas que resaltaran las palabras propias del lenguaje para diferenciar fácilmente de las palabras que no son parte del lenguaje, por eso el uso de mayúsculas y minúsculas.

Las palabras reservadas de consultas SQL usualmente se escriben en mayúscula, ésto para distinguir entre nombres de objetos y lenguaje SQL propio, no es obligatorio, pero podría serte útil en la creación de Scripts SQL largos.

Vamos ahora por un ligero ejemplo desde la creación de una base de datos, la creación de una tabla, la inserción, borrado, consulta y alteración de datos de la tabla.

Primero crea la base de datos, “**CREATE DATABASE transporte;**” sería el primer paso.

![27](images/27.png)

Ahora saltar de la base de datos postgres que ha sido seleccionada de manera predeterminada a la base de datos transporte recién creada utilizando el comando **\c transporte**.

![28](images/28.png)

Ahora vamos a crear la tabla tren, el SQL correspondiente sería:

`CREATE TABLE tren ( id serial NOT NULL, modelo character varying, capacidad integer, CONSTRAINT tren_pkey PRIMARY KEY (id) );`

La columna id será un número autoincremental (cada vez que se inserta un registro se aumenta en uno), modelo se refiere a una referencia al tren, capacidad sería la cantidad de pasajeros que puede transportar y al final agregamos la llave primaria que será id.

![29](images/29.png)

Ahora que la tabla ha sido creada, podemos ver su definición utilizando el comando **\d tren**

![30](images/30.png)

PostgreSQL ha creado el campo id automáticamente cómo integer con una asociación predeterminada a una secuencia llamada ‘**tren_id_seq**’. De manera que cada vez que se inserte un valor, id tomará el siguiente valor de la secuencia, vamos a ver la definición de la secuencia. Para ello,** \d tren_id_seq** es suficiente:

![31](images/31.png)

Vemos que la secuencia inicia en uno, así que nuestra primera inserción de datos dejará a la columna id con valor uno.

`INSERT INTO tren( modelo, capacidad ) VALUES (‘Volvo 1’, 100);`

![32](images/32.png)

Consultamos ahora los datos en la tabla:

`SELECT * FROM tren;`

![33](images/33.png)

Vamos a modificar el valor, establecer el tren con id uno que sea modelo Honda 0726. Para ello ejecutamos la consulta tipo` UPDATE tren SET modelo = 'Honda 0726' Where id = 1;`

![34](images/34.png)

Verificamos la modificación `SELECT * FROM tren;`

![35](images/35.png)

Ahora borramos la fila: `DELETE FROM tren WHERE id = 1;`

![36](images/36.png)

Verificamos el borrado `SELECT * FROM tren;`

![37](images/37.png)

El borrado ha funcionado tenemos 0 rows, es decir, no hay filas. Ahora activemos la herramienta que nos permite medir el tiempo que tarda una consulta \timing

![38](images/38.png)

Probemos cómo funciona al medición realizando la encriptación de un texto cualquiera usando el algoritmo md5:

![39](images/39.png)

La consulta tardó 10.011 milisegundos

Ahora que sabes como manejar algunos de los comandos más utilizados en PostgreSQL es momento de comenzar a practicar!!!

## Tipos de datos

En programación, los **tipos de datos** representan las distintas clases de valores que una variable puede almacenar. Identificar y trabajar correctamente con los tipos de datos es fundamental para desarrollar aplicaciones robustas y funcionales.


### **Clasificación General de Tipos de Datos**
1. **Tipos de Datos Primitivos:** Representan valores básicos y fundamentales.
2. **Tipos de Datos Compuestos:** Formados por combinaciones de datos primitivos.
3. **Tipos de Datos Abstractos:** Diseñados para resolver problemas específicos, basados en estructuras complejas.

### **1. Tipos de Datos Primitivos**
Son los más básicos en cualquier lenguaje de programación.

- **Numéricos:**
  - Enteros (**int**): Números sin decimales (e.g., 10, -5, 0).
  - Flotantes (**float**): Números con decimales (e.g., 3.14, -0.01).
  - Complejos (**complex**): Números con parte real e imaginaria (e.g., 3 + 4j).
  
- **Texto:**
  - **String (str):** Secuencia de caracteres (e.g., "Hola", "123").
  
- **Booleanos:**
  - **bool:** Representa valores de verdad (True, False).

- **Carácter:**
  - Algunos lenguajes como C tienen el tipo **char** para un único carácter.

### **2. Tipos de Datos Compuestos**
Agrupan múltiples valores, ya sean homogéneos o heterogéneos.

- **Listas (List o Array):**
  - Colección ordenada de elementos (e.g., [1, 2, 3], ["a", "b", "c"]).
  - En Python: `mi_lista = [10, "texto", True]`.

- **Tuplas (Tuple):**
  - Colección ordenada e inmutable (e.g., (1, 2, 3)).
  - En Python: `mi_tupla = (1, "texto", False)`.

- **Conjuntos (Set):**
  - Colección no ordenada de elementos únicos (e.g., {1, 2, 3}).
  - En Python: `mi_conjunto = {1, 2, 2, 3}` → Resultado: `{1, 2, 3}`.

- **Diccionarios (Dictionary):**
  - Colección de pares clave-valor (e.g., {"clave": "valor"}).
  - En Python: `mi_dict = {"nombre": "Ana", "edad": 30}`.

### **3. Tipos de Datos Abstractos**
Diseñados para solucionar problemas específicos, suelen estar implementados como estructuras más avanzadas.

- **Pilas (Stack):** Sigue la regla LIFO (Last In, First Out). Ejemplo: una pila de platos.
- **Colas (Queue):** Sigue la regla FIFO (First In, First Out). Ejemplo: una fila de espera.
- **Listas Enlazadas (Linked List):** Elementos conectados mediante referencias a otros nodos.
- **Árboles (Tree):** Estructura jerárquica con un nodo raíz y nodos hijos.
- **Grafos (Graph):** Representan nodos conectados por aristas.

### **4. Tipos de Datos en Lenguajes Específicos**
#### **Python:**
- Enteros, flotantes, cadenas, booleanos.
- Estructuras como listas, conjuntos, diccionarios y tuplas.
  
#### **Java:**
- Primitivos: `int`, `float`, `char`, `boolean`.
- Objetos: `String`, `ArrayList`, `HashMap`.

#### **C/C++:**
- Primitivos: `int`, `float`, `char`, `double`.
- Estructuras como arrays, estructuras (`struct`) y punteros.

#### **SQL:**
- **Datos numéricos:** `INT`, `FLOAT`, `DECIMAL`.
- **Datos de texto:** `CHAR`, `VARCHAR`, `TEXT`.
- **Datos de fecha:** `DATE`, `TIME`, `DATETIME`.

### **Conversión entre Tipos de Datos**
Es común convertir datos de un tipo a otro para manipularlos correctamente.

- **Conversión implícita (automática):** Realizada por el lenguaje (e.g., `int` a `float`).
- **Conversión explícita:** Requiere el uso de funciones o métodos específicos:
  - Python: 
    ```python
    entero = int("10")  # Convierte de string a entero
    flotante = float(5)  # Convierte de entero a flotante
    cadena = str(123)  # Convierte de entero a string
    ```

### **Importancia de los Tipos de Datos**
1. **Precisión en cálculos:** Evitar errores al realizar operaciones.
2. **Eficiencia:** Elegir el tipo de dato adecuado optimiza el rendimiento.
3. **Seguridad:** Restringir el uso indebido de datos.
4. **Flexibilidad:** Facilita la manipulación y procesamiento de datos complejos.

**Lecturas recomendadas**

[PostgreSQL data types, tipos de datos más utilizados - TodoPostgreSQL](https://todopostgresql.com/postgresql-data-types-los-tipos-de-datos-mas-utilizados/)

[PostgreSQL: Documentation: 11: Chapter 8. Data Types](https://www.postgresql.org/docs/11/datatype.html)

## Diseñando nuestra base de datos: estructura de las tablas

Diseñar una base de datos implica estructurar tablas que representen entidades y sus relaciones en el dominio del problema. A continuación, se detalla un enfoque práctico y organizado para diseñar la estructura de las tablas de una base de datos.

### **1. Identificación de Entidades**
Las entidades representan objetos o conceptos importantes en el sistema que deben ser modelados como tablas. Ejemplos:
- Usuarios
- Productos
- Pedidos
- Categorías

### **2. Definición de Tablas**
Cada entidad se traduce en una tabla. Para cada tabla:
- **Nombre:** Describa claramente la entidad (e.g., `Usuarios`, `Productos`).
- **Columnas:** Representan los atributos de la entidad.
  - Ejemplo: La tabla `Usuarios` puede tener `id_usuario`, `nombre`, `email`, `fecha_registro`.

### **3. Establecer Tipos de Datos**
Defina el tipo de datos más adecuado para cada columna:
- **Numéricos:**
  - `INT`: Para enteros (e.g., identificadores, cantidades).
  - `FLOAT` o `DECIMAL`: Para valores con decimales (e.g., precios, porcentajes).
- **Texto:**
  - `VARCHAR`: Para cadenas de texto (e.g., nombres, correos electrónicos).
  - `TEXT`: Para descripciones largas.
- **Fecha y hora:**
  - `DATE`: Para fechas (e.g., `2025-01-18`).
  - `DATETIME`: Fecha y hora combinadas.
- **Booleanos:**
  - `BOOLEAN` o `TINYINT(1)`: Para valores de verdadero/falso.

### **4. Relaciones entre Tablas**
Defina cómo se relacionan las tablas. Los tipos principales de relaciones son:

1. **Uno a Uno (1:1):**
   - Un registro en una tabla corresponde exactamente a un registro en otra tabla.
   - Ejemplo: Tabla `Usuarios` y tabla `DetallesUsuario`.

2. **Uno a Muchos (1:N):**
   - Un registro en una tabla está relacionado con varios registros en otra tabla.
   - Ejemplo: Un usuario puede hacer varios pedidos (`Usuarios` y `Pedidos`).

3. **Muchos a Muchos (M:N):**
   - Varios registros en una tabla están relacionados con varios registros en otra tabla.
   - Ejemplo: `Productos` y `Categorías` se relacionan a través de una tabla intermedia `ProductoCategoria`.

### **5. Claves Primarias y Foráneas**
- **Clave primaria (Primary Key):** Identifica de forma única cada registro en una tabla.
  - Ejemplo: `id_usuario` en `Usuarios`.
- **Clave foránea (Foreign Key):** Relaciona tablas y asegura la integridad referencial.
  - Ejemplo: `id_usuario` en la tabla `Pedidos` es una clave foránea que apunta a `Usuarios`.

### **6. Ejemplo de Estructura de Tablas**
#### **Usuarios**
| Nombre columna    | Tipo de dato | Restricciones          |
|-------------------|-------------|------------------------|
| id_usuario        | INT         | PRIMARY KEY, AUTO_INCREMENT |
| nombre            | VARCHAR(100) | NOT NULL              |
| email             | VARCHAR(150) | UNIQUE, NOT NULL      |
| fecha_registro    | DATETIME    | DEFAULT CURRENT_TIMESTAMP |

#### **Productos**
| Nombre columna    | Tipo de dato | Restricciones          |
|-------------------|-------------|------------------------|
| id_producto       | INT         | PRIMARY KEY, AUTO_INCREMENT |
| nombre_producto   | VARCHAR(100) | NOT NULL              |
| precio            | DECIMAL(10,2)| NOT NULL              |
| stock             | INT         | DEFAULT 0             |

#### **Pedidos**
| Nombre columna    | Tipo de dato | Restricciones          |
|-------------------|-------------|------------------------|
| id_pedido         | INT         | PRIMARY KEY, AUTO_INCREMENT |
| id_usuario        | INT         | FOREIGN KEY (`id_usuario`) REFERENCES `Usuarios`(`id_usuario`) |
| fecha_pedido      | DATETIME    | DEFAULT CURRENT_TIMESTAMP |
| total             | DECIMAL(10,2)| NOT NULL              |

#### **PedidoDetalles**
| Nombre columna    | Tipo de dato | Restricciones          |
|-------------------|-------------|------------------------|
| id_detalle        | INT         | PRIMARY KEY, AUTO_INCREMENT |
| id_pedido         | INT         | FOREIGN KEY (`id_pedido`) REFERENCES `Pedidos`(`id_pedido`) |
| id_producto       | INT         | FOREIGN KEY (`id_producto`) REFERENCES `Productos`(`id_producto`) |
| cantidad          | INT         | NOT NULL              |
| subtotal          | DECIMAL(10,2)| NOT NULL              |


### **7. Normalización**
Aplica principios de normalización para evitar redundancia:
- **Primera Forma Normal (1NF):** Elimina atributos multivaluados.
- **Segunda Forma Normal (2NF):** Asegura que cada atributo dependa completamente de la clave primaria.
- **Tercera Forma Normal (3NF):** Elimina dependencias transitivas.

### **8. Herramientas de Diseño**
Utiliza herramientas como:
- **MySQL Workbench**
- **pgAdmin**
- **DB Designer**
- **Lucidchart**

[dbdiagram](https://dbdiagram.io/d)

```dbml
// Use DBML to define your database structure
// Docs: https://dbml.dbdiagram.io/docs

Table estaciones {
  id_estacion integer [primary key]
  nombre varchar(52)
  direccion varchar(52)
}
Ref: estaciones.id_estacion < trayectos.id_estacion

Table trenes {
  id_tren integer [primary key]
  modelo varchar(52)
  capacidad integer(52)
}
Ref: trenes.id_tren < trayectos.id_tren

Table pasajeros {
  n_documento integer [primary key]
  nombre varchar(52)
  direccion_residencia integer(52)
  fecha_nacimiento date
}

Ref: viajes.id_viaje > pasajeros.n_documento // ono a muchos


Table trayectos {
  id_trayecto integer [primary key]
  id_estacion integer
  id_tren integer
  nombre integer
}
Ref: trayectos.id_trayecto <> viajes.id_trayecto //muchos a muchos
 
 // - uno a uno
Table viajes {
  id_viaje integer [primary key]
  n_documento integer
  id_trayecto integer
  inicio time
  fin time
}
```

**Lecturas recomendadas**

[Fundamentos de Bases de Datos](https://platzi.com/clases/bd/)

### Jerarquía de Bases de Datos

Toda jerarquía de base de datos se basa en los siguientes elementos:

- **Servidor de base de datos**: Computador que tiene un motor de base de datos instalado y en ejecución.

- **Motor de base de datos**: Software que provee un conjunto de servicios encargados de administrar una base de datos.

- **Base de datos**: Grupo de datos que pertenecen a un mismo contexto.

- **Esquemas de base de datos en PostgreSQL**: Grupo de objetos de base de datos que guarda relación entre sí (tablas, funciones, relaciones, secuencias).

- **Tablas de base de datos**: Estructura que organiza los datos en filas y columnas formando una matriz.

**PostgreSQL es un motor de base de datos.**

La estructura de la base de datos diseñada para el reto corresponde a los siguientes
elementos:

![40](images/40.png)

La base de datos se llama transporte, usaremos su esquema predeterminado public.

El esquema public contiene las siguientes tablas:

- Estación

- Pasajero

- Tren

Y las tablas de relaciones entre cada uno de los elementos anteriores son:

- Trayecto

- Viaje

El esquema relacional entre las tablas corresponde al siguiente diagrama:

![41](images/41.png)

***Estación***
Contiene la información de las estaciones de nuestro sistema, incluye datos de nombre con tipo de dato texto y dirección con tipo de dato texto, junto con un número de identificación único por estación.

***Tren***
Almacena la información de los trenes de nuestro sistema, cada tren tiene un modelo con tipo de dato texto y una capacidad con tipo de dato numérico que representa la cantidad de personas que puede llevar ese tren, también tiene un ID único por tren.

***Trayecto***
Relaciona los trenes con las estaciones, simula ser las rutas que cada uno de los trenes pueden desarrollar entre las estaciones

***Pasajero***
Es la tabla que contiene la información de las personas que viajan en nuestro sistema de transporte masivo, sus columnas son nombre tipo de dato texto con el nombre completo de la persona, direccion_residencia con tipo de dato texto que indica dónde vive la persona, fecha_nacimiento tipo de dato texto y un ID único tipo de dato numérico para identificar a cada persona.

***Viaje***
Relaciona Trayecto con Pasajero ilustrando la dinámica entre los viajes que realizan las personas, los cuales parten de una estación y se hacen usando un tren.

## Creación de Tablas

Crear tablas en **PostgreSQL** implica definir su estructura, tipos de datos, restricciones y relaciones. Esto se realiza utilizando el comando **`CREATE TABLE`** en SQL. A continuación, te explico cómo se hace y te doy ejemplos prácticos.

### **Estructura Básica de `CREATE TABLE`**
```sql
CREATE TABLE nombre_tabla (
    nombre_columna tipo_dato [restricciones],
    ...
);
```

### **Tipos de Datos Comunes en PostgreSQL**
- **Numéricos**: `INTEGER`, `BIGINT`, `NUMERIC`, `REAL`, `DOUBLE PRECISION`.
- **Texto**: `CHAR(n)`, `VARCHAR(n)`, `TEXT`.
- **Fecha y Hora**: `DATE`, `TIME`, `TIMESTAMP`, `INTERVAL`.
- **Booleanos**: `BOOLEAN`.
- **Serial (auto-incremental)**: `SERIAL`, `BIGSERIAL`.

### **Restricciones Comunes**
- **`PRIMARY KEY`**: Identifica la clave principal.
- **`NOT NULL`**: Evita valores nulos en una columna.
- **`UNIQUE`**: Garantiza que los valores sean únicos.
- **`CHECK`**: Verifica una condición lógica.
- **`FOREIGN KEY`**: Define una clave foránea.
- **`DEFAULT`**: Especifica un valor predeterminado.

### **Ejemplo 1: Crear una Tabla Simple**
```sql
CREATE TABLE usuarios (
    id_usuario SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Descripción**:
1. `id_usuario` es una columna auto-incremental.
2. `nombre` no puede ser nulo.
3. `email` debe ser único.
4. `fecha_registro` tiene un valor predeterminado.

### **Ejemplo 2: Tablas con Relaciones**
```sql
CREATE TABLE pedidos (
    id_pedido SERIAL PRIMARY KEY,
    id_usuario INT NOT NULL,
    fecha_pedido DATE NOT NULL,
    total DECIMAL(10, 2),
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario)
);
```

**Descripción**:
1. La columna `id_usuario` es una clave foránea que hace referencia a `usuarios`.
2. `fecha_pedido` no puede ser nula.
3. `total` permite hasta 10 dígitos, con 2 decimales.

### **Ejemplo 3: Relación Uno a Uno**
```sql
CREATE TABLE perfiles (
    id_perfil SERIAL PRIMARY KEY,
    id_usuario INT UNIQUE NOT NULL,
    bio TEXT,
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario)
);
```

**Nota**: La restricción `UNIQUE` asegura una relación uno a uno entre `usuarios` y `perfiles`.

### **Comprobando la Creación de la Tabla**
Después de crear una tabla, puedes verificar su estructura con:
```sql
\d nombre_tabla
```

### **Buenas Prácticas**
1. **Nombrar columnas y tablas con consistencia** (ej., en minúsculas y usando snake_case).
2. **Definir claves primarias y foráneas** para garantizar integridad referencial.
3. **Usar índices** si esperas buscar frecuentemente por una columna.
4. **Probar las restricciones** con datos reales antes de popular la base.

## Particiones

Las **particiones en bases de datos en PostgreSQL** permiten dividir una tabla grande en tablas más pequeñas (particiones) para mejorar el rendimiento y la gestión de datos. Cada partición contiene un subconjunto de los datos, pero se comporta como si fueran parte de la tabla principal.

### Ventajas de usar particiones:
1. **Rendimiento mejorado**: Consultas más rápidas, ya que pueden buscar en una partición específica en lugar de toda la tabla.
2. **Gestión eficiente**: Más fácil eliminar o archivar datos antiguos al trabajar con particiones específicas.
3. **Concurrencia mejorada**: Menor bloqueo entre transacciones que acceden a diferentes particiones.

### Crear una tabla particionada en PostgreSQL

#### 1. Crear una tabla principal como particionada
Debes especificar el tipo de partición (`RANGE` o `LIST`) al crear la tabla principal.

```sql
CREATE TABLE ventas (
    id_venta SERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    total NUMERIC(10, 2) NOT NULL
) PARTITION BY RANGE (fecha);
```

- **`RANGE`**: Divide datos por un rango de valores.
- **`LIST`**: Divide datos por una lista de valores.
- **`HASH`**: Divide datos usando una función hash (para distribuciones uniformes).

#### 2. Crear las particiones específicas
Las particiones deben ser creadas explícitamente, asignando el subconjunto de datos que manejarán.

##### Ejemplo de particiones por rango:
```sql
CREATE TABLE ventas_2023 PARTITION OF ventas
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE ventas_2024 PARTITION OF ventas
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

##### Ejemplo de particiones por lista:
```sql
CREATE TABLE ventas_america PARTITION OF ventas
    FOR VALUES IN ('AMERICA');

CREATE TABLE ventas_europa PARTITION OF ventas
    FOR VALUES IN ('EUROPA');
```

#### 3. Insertar datos en la tabla principal
PostgreSQL automáticamente asigna los datos a la partición correcta según las reglas definidas.

```sql
INSERT INTO ventas (fecha, total) VALUES ('2023-05-10', 100.00);
INSERT INTO ventas (fecha, total) VALUES ('2024-03-15', 200.00);
```

#### 4. Consultar datos en tablas particionadas
Las consultas a la tabla principal incluyen automáticamente todas las particiones.

```sql
SELECT * FROM ventas WHERE fecha BETWEEN '2023-01-01' AND '2023-12-31';
```

### Ver las particiones
Usa el comando `\d+` en `psql` para ver las particiones asociadas a una tabla.

```sql
\d+ ventas
```

### Eliminar particiones
Puedes eliminar una partición como si fuera una tabla normal:

```sql
DROP TABLE ventas_2023;
```

### Buenas prácticas:
1. Define particiones que representen un subconjunto claro y uniforme de los datos.
2. Usa índices en las particiones individuales si las consultas son frecuentes.
3. Automatiza la creación y eliminación de particiones con scripts periódicos.

## Creación de Roles

La creación de roles en PostgreSQL es una parte clave de la administración de usuarios y permisos. Un **rol** en PostgreSQL puede representar un usuario, un grupo, o ambos. Los roles son fundamentales para la seguridad y control de acceso en la base de datos.

### 1. **Crear un rol**
Para crear un rol en PostgreSQL, usa el comando `CREATE ROLE`. Puedes especificar varias opciones dependiendo de las necesidades.

#### Ejemplo básico:
```sql
CREATE ROLE usuario_prueba;
```

Este comando crea un rol llamado `usuario_prueba`. Por defecto:
- El rol no puede iniciar sesión.
- No tiene permisos adicionales.

### 2. **Crear un rol con permisos para iniciar sesión**
Si necesitas que el rol actúe como un usuario que pueda conectarse a la base de datos, agrega la opción `LOGIN`.

```sql
CREATE ROLE usuario_login WITH LOGIN PASSWORD 'mi_contraseña';
```

Opciones adicionales:
- **`PASSWORD`**: Establece una contraseña para el rol.
- **`CREATEDB`**: Permite que el rol cree bases de datos.
- **`SUPERUSER`**: Asigna permisos de superusuario.
- **`INHERIT`**: Permite que herede permisos de roles a los que pertenece.
- **`NOCREATEDB`** y **`NOSUPERUSER`**: Opciones inversas para restringir permisos.

### 3. **Asignar un rol a otro rol (Grupos de usuarios)**
Un rol puede actuar como un grupo para simplificar la administración de permisos.

#### Crear un rol de grupo:
```sql
CREATE ROLE grupo_lectura;
```

#### Agregar un usuario al grupo:
```sql
GRANT grupo_lectura TO usuario_login;
```

Ahora, el usuario `usuario_login` hereda todos los permisos asignados al rol `grupo_lectura`.

### 4. **Modificar un rol existente**
Para cambiar las propiedades de un rol, usa el comando `ALTER ROLE`.

#### Ejemplo:
```sql
ALTER ROLE usuario_prueba WITH LOGIN PASSWORD 'nueva_contraseña';
ALTER ROLE usuario_prueba SET search_path = 'mi_esquema';
```

### 5. **Eliminar un rol**
Elimina roles con el comando `DROP ROLE`. Asegúrate de que el rol no tenga dependencias (como bases de datos o pertenencia a grupos).

```sql
DROP ROLE usuario_prueba;
```

### 6. **Ver roles existentes**
Usa el siguiente comando en `psql` para listar roles:

```sql
\du
```

Esto muestra todos los roles, sus atributos y permisos.

### 7. **Ejemplo completo**
Supongamos que deseas crear un rol administrador y un rol para usuarios con permisos de solo lectura.

```sql
-- Crear roles
CREATE ROLE admin WITH SUPERUSER CREATEDB CREATEROLE LOGIN PASSWORD 'admin123';
CREATE ROLE solo_lectura;

-- Crear usuario y asignarlo al grupo de solo lectura
CREATE ROLE usuario_lectura WITH LOGIN PASSWORD 'user123';
GRANT solo_lectura TO usuario_lectura;

-- Asignar permisos específicos al grupo
GRANT SELECT ON ALL TABLES IN SCHEMA public TO solo_lectura;

-- Verificar roles
\du
```

Este ejemplo configura un administrador con todos los permisos y un usuario que solo puede leer tablas del esquema `public`.

### Buenas prácticas:
1. **Usa roles de grupo** para gestionar permisos de múltiples usuarios.
2. **Restringe el uso de superusuarios** para evitar errores o brechas de seguridad.
3. **Documenta los roles y permisos asignados** para facilitar el mantenimiento.

**NOTAS**: 
`\h` CREATE ROL opciones de la creacion de roles.
`\dg` o `\du` consulta los roles.
`CREATE ROLE usuario_consulta;` CREA un usuario.
`ALTER ROLE usuario_consulta WITH LOGIN` altera el usuario y le asigna una clava.
`ALTER ROLE usuario_consulta WITH PASSWORD '123456'` asigna clave al usuario.
`ALTER ROLE usuario_consulta WITH SUPERUSER;` Cambia el rol al usuario.
`DROP ROLE usuario_consulta;` Eliminar usuario.

En PostgreSQL, los roles se utilizan para administrar permisos y controlar el acceso a las bases de datos. Existen varios tipos de roles según sus características y funcionalidades, que permiten definir distintos niveles de acceso y responsabilidades.

### **1. Roles de Usuario**
- Un rol que tiene la capacidad de iniciar sesión en PostgreSQL.
- Se utiliza para representar a individuos que acceden a la base de datos.
  
#### Características:
- Tiene habilitada la opción `LOGIN`.
- Puede tener permisos específicos sobre objetos (tablas, vistas, esquemas, etc.).
  
#### Ejemplo:
```sql
CREATE ROLE usuario_login WITH LOGIN PASSWORD 'mi_contraseña';
```

### **2. Roles de Grupo**
- No tienen acceso directo al sistema.
- Actúan como contenedores de permisos para simplificar la administración.
  
#### Características:
- No tienen la opción `LOGIN`.
- Se pueden asignar a otros roles (usuarios) que heredan los permisos del grupo.
  
#### Ejemplo:
```sql
CREATE ROLE grupo_admin;
GRANT grupo_admin TO usuario_login;
```

### **3. Rol de Superusuario**
- Tiene todos los privilegios posibles en PostgreSQL.
- Puede realizar cualquier operación, incluyendo la administración de bases de datos y roles.
  
#### Características:
- Habilitado con la opción `SUPERUSER`.
- Debe ser usado con cuidado debido a su nivel de acceso.

#### Ejemplo:
```sql
CREATE ROLE superadmin WITH SUPERUSER LOGIN PASSWORD 'admin123';
```

### **4. Roles con Permisos Administrativos**
- **`CREATEDB`**: Permite al rol crear bases de datos.
- **`CREATEROLE`**: Permite crear, modificar o eliminar otros roles.
- **`REPLICATION`**: Permite realizar tareas relacionadas con la replicación de datos.

#### Ejemplo:
```sql
CREATE ROLE db_creator WITH CREATEDB LOGIN PASSWORD 'creator123';
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'replica123';
```

### **5. Rol sin Privilegios (por defecto)**
- Es un rol básico sin permisos adicionales.
- No tiene acceso a objetos o funciones hasta que se le otorguen explícitamente.

#### Ejemplo:
```sql
CREATE ROLE usuario_básico WITH LOGIN PASSWORD 'user123';
```

### **6. Roles Personalizados**
Se crean según las necesidades del sistema. Por ejemplo:

- **Rol de Lectura**: Solo puede consultar datos.
- **Rol de Escritura**: Puede consultar y modificar datos.
- **Rol de Administrador de Esquema**: Puede gestionar objetos en un esquema.

#### Ejemplo:
```sql
-- Rol de solo lectura
CREATE ROLE solo_lectura;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO solo_lectura;

-- Rol de solo escritura
CREATE ROLE solo_escritura;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO solo_escritura;

-- Rol de administrador de esquema
CREATE ROLE admin_esquema;
GRANT ALL PRIVILEGES ON SCHEMA public TO admin_esquema;
```

### **Comparación de Roles**

| **Rol**              | **Puede Iniciar Sesión** | **Privilegios**                          |
|-----------------------|--------------------------|------------------------------------------|
| Usuario              | Sí                      | Depende de los permisos asignados.       |
| Grupo                | No                      | Permisos agrupados para otros roles.     |
| Superusuario         | Sí                      | Todos los privilegios posibles.          |
| Administrador de DB  | Opcional                | Puede crear y administrar bases de datos.|
| Replicador           | Opcional                | Permisos para replicación.               |

### **Visualizar Roles Existentes**
Para listar los roles en PostgreSQL, utiliza:

```sql
\du
```

Esto muestra los roles creados, sus atributos y los permisos asociados.

### Buenas Prácticas para el Uso de Roles
1. **Usar roles de grupo** para administrar permisos comunes.
2. **Evitar el uso excesivo de superusuarios**.
3. **Asignar permisos mínimos necesarios** para cada usuario o rol.
4. **Documentar los roles y permisos** para facilitar el mantenimiento.
5. **Revisar regularmente los permisos y roles asignados** para garantizar la seguridad.

## Llaves foráneas

En PostgreSQL, una **clave foránea** (foreign key) se utiliza para garantizar la integridad referencial entre dos tablas. Define una relación entre una columna de una tabla (tabla hija) y una columna de otra tabla (tabla padre). Esto asegura que los valores en la columna de la tabla hija deben coincidir con los valores de la columna de la tabla padre o ser nulos.

### **Sintaxis de Llave Foránea**

```sql
CREATE TABLE nombre_tabla_hija (
    columna_hija tipo_de_dato,
    ...
    CONSTRAINT nombre_llave_foranea FOREIGN KEY (columna_hija)
        REFERENCES nombre_tabla_padre (columna_padre)
        [ON UPDATE acción]
        [ON DELETE acción]
);
```

### **Acciones para `ON UPDATE` y `ON DELETE`**

Puedes definir lo que sucede cuando los valores de la clave referenciada en la tabla padre cambian o se eliminan:
- **CASCADE**: Propaga el cambio o eliminación a la tabla hija.
- **SET NULL**: Establece la columna de la tabla hija como `NULL`.
- **SET DEFAULT**: Establece el valor predeterminado en la columna de la tabla hija.
- **RESTRICT**: Impide la acción si hay registros relacionados.
- **NO ACTION**: Similar a `RESTRICT`, pero permite que otras reglas intervengan antes de lanzar un error.

### **Ejemplo Práctico**

#### **Crear Tablas con Llaves Foráneas**

```sql
-- Tabla padre
CREATE TABLE departamentos (
    id_departamento SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL
);

-- Tabla hija
CREATE TABLE empleados (
    id_empleado SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    id_departamento INTEGER NOT NULL,
    CONSTRAINT fk_departamento FOREIGN KEY (id_departamento)
        REFERENCES departamentos (id_departamento)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);
```

#### **Insertar Datos**

```sql
-- Insertar en la tabla padre
INSERT INTO departamentos (nombre)
VALUES ('Recursos Humanos'), ('IT'), ('Finanzas');

-- Insertar en la tabla hija
INSERT INTO empleados (nombre, id_departamento)
VALUES ('Juan Pérez', 1), ('Ana Gómez', 2), ('Carlos Ruiz', 3);
```

### **Modificar Llaves Foráneas Existentes**

#### **Agregar una Llave Foránea**

Si la tabla ya existe y no tiene una clave foránea, puedes agregarla:

```sql
ALTER TABLE empleados
ADD CONSTRAINT fk_departamento FOREIGN KEY (id_departamento)
REFERENCES departamentos (id_departamento)
ON DELETE CASCADE;
```

#### **Eliminar una Llave Foránea**

Si necesitas eliminar una llave foránea:

```sql
ALTER TABLE empleados
DROP CONSTRAINT fk_departamento;
```

### **Ver Llaves Foráneas en una Tabla**

Para verificar las llaves foráneas en una tabla, puedes usar:

```sql
SELECT
    conname AS nombre_llave,
    conrelid::regclass AS tabla_hija,
    a.attname AS columna_hija,
    confrelid::regclass AS tabla_padre,
    af.attname AS columna_padre
FROM
    pg_constraint c
    JOIN pg_attribute a ON a.attnum = ANY (c.conkey) AND a.attrelid = c.conrelid
    JOIN pg_attribute af ON af.attnum = ANY (c.confkey) AND af.attrelid = c.confrelid
WHERE
    contype = 'f'; -- 'f' significa foreign key
```

### **Buenas Prácticas**
1. **Nombrar claramente las claves foráneas** para facilitar el mantenimiento.
   - Ejemplo: `fk_<tabla_hija>_<tabla_padre>`
2. **Definir correctamente las acciones `ON DELETE` y `ON UPDATE`** según las necesidades de la aplicación.
3. **Asegurarse de que las columnas relacionadas tengan el mismo tipo de dato y longitud**.
4. **Evitar valores huérfanos en tablas hijas** mediante el uso de `ON DELETE CASCADE` o validaciones.

## Inserción y consulta de datos

A continuación se explica cómo realizar inserciones y consultas de datos en PostgreSQL, con ejemplos prácticos.

## **Inserción de Datos**

### **Sintaxis Básica**
Para insertar datos en una tabla, utiliza el comando `INSERT INTO`.

```sql
INSERT INTO nombre_tabla (columna1, columna2, columna3)
VALUES (valor1, valor2, valor3);
```

### **Ejemplo Práctico**

#### Crear la Tabla
```sql
CREATE TABLE empleados (
    id_empleado SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    cargo VARCHAR(50),
    salario NUMERIC(10, 2)
);
```

#### Insertar Datos
```sql
INSERT INTO empleados (nombre, cargo, salario)
VALUES 
('Juan Pérez', 'Analista', 2500.50),
('Ana Gómez', 'Gerente', 5000.00),
('Carlos Ruiz', 'Asistente', 1800.75);
```

## **Consulta de Datos**

### **Sintaxis Básica**
Para consultar datos, utiliza el comando `SELECT`.

```sql
SELECT columna1, columna2
FROM nombre_tabla
WHERE condición;
```

### **Tipos de Consultas**

#### 1. **Consulta de Todos los Registros**
```sql
SELECT * FROM empleados;
```

**Resultado**:
| id_empleado | nombre       | cargo      | salario  |
|-------------|--------------|------------|----------|
| 1           | Juan Pérez   | Analista   | 2500.50  |
| 2           | Ana Gómez    | Gerente    | 5000.00  |
| 3           | Carlos Ruiz  | Asistente  | 1800.75  |

---

#### 2. **Consulta con Filtros**
Consulta empleados con salario mayor a 2000.

```sql
SELECT nombre, cargo, salario
FROM empleados
WHERE salario > 2000;
```

**Resultado**:
| nombre      | cargo    | salario  |
|-------------|----------|----------|
| Juan Pérez  | Analista | 2500.50  |
| Ana Gómez   | Gerente  | 5000.00  |

---

#### 3. **Consulta Ordenada**
Ordenar los empleados por salario de manera descendente.

```sql
SELECT nombre, cargo, salario
FROM empleados
ORDER BY salario DESC;
```

#### 4. **Consulta con Límites**
Obtener los 2 empleados con mayor salario.

```sql
SELECT nombre, cargo, salario
FROM empleados
ORDER BY salario DESC
LIMIT 2;
```

#### 5. **Consulta con Agregación**
Calcular el salario promedio de los empleados.

```sql
SELECT AVG(salario) AS salario_promedio
FROM empleados;
```

**Resultado**:
| salario_promedio |
|------------------|
| 3100.42          |

#### 6. **Consulta con Funciones de Grupo**
Número de empleados por cargo.

```sql
SELECT cargo, COUNT(*) AS cantidad_empleados
FROM empleados
GROUP BY cargo;
```

**Resultado**:
| cargo     | cantidad_empleados |
|-----------|--------------------|
| Analista  | 1                  |
| Gerente   | 1                  |
| Asistente | 1                  |

## **Combinar Inserción y Consulta**
### Insertar y consultar de inmediato
```sql
WITH nuevo_empleado AS (
    INSERT INTO empleados (nombre, cargo, salario)
    VALUES ('Laura Sánchez', 'Supervisora', 3000.00)
    RETURNING id_empleado, nombre, cargo, salario
)
SELECT * FROM nuevo_empleado;
```

## **Buenas Prácticas**
1. **Verificar las restricciones antes de insertar datos**: Define claves primarias, claves foráneas, y restricciones únicas.
2. **Utilizar transacciones para operaciones críticas**: Asegura que las inserciones o actualizaciones sean atómicas.
3. **Evitar consultas sin filtros en tablas grandes**: Usa `WHERE` y `LIMIT` para optimizar el rendimiento.
4. **Utilizar índices para mejorar las consultas**: Especialmente en columnas utilizadas en filtros o uniones.

## Inserción masiva de datos

La inserción masiva de datos en PostgreSQL es útil cuando necesitas cargar grandes volúmenes de información de manera eficiente. Hay dos enfoques principales para lograrlo:

### **1. Usar el comando `COPY`**
Este método permite importar datos desde un archivo directamente a una tabla.

#### **Ejemplo:**
Supongamos que tienes un archivo CSV llamado `estaciones.csv` con el siguiente contenido:

```csv
id_estacion,nombre,direccion
2,Estación Norte,St 100 # 112
3,Estación Sur,Av. 45 # 98-23
4,Estación Este,Cra 12 # 24-15
5,Estación Oeste,Calle 8 # 16-10
```

Puedes usar el siguiente comando para cargar estos datos:

```sql
COPY public.estaciones(id_estacion, nombre, direccion)
FROM '/ruta/completa/estaciones.csv'
DELIMITER ','
CSV HEADER;
```

#### **Notas:**
1. Reemplaza `'/ruta/completa/estaciones.csv'` con la ruta real al archivo CSV en tu sistema.
2. Asegúrate de que el archivo sea accesible y que el usuario de PostgreSQL tenga permisos de lectura.
3. Usa el parámetro `HEADER` si el archivo contiene nombres de columnas en la primera fila.

### **2. Usar múltiples filas con `INSERT INTO`**
Si no quieres usar un archivo externo, puedes insertar múltiples filas en un solo comando `INSERT`.

#### **Ejemplo:**
```sql
INSERT INTO public.estaciones(id_estacion, nombre, direccion)
VALUES
    (2, 'Estación Norte', 'St 100 # 112'),
    (3, 'Estación Sur', 'Av. 45 # 98-23'),
    (4, 'Estación Este', 'Cra 12 # 24-15'),
    (5, 'Estación Oeste', 'Calle 8 # 16-10');
```

### **3. Usar herramientas externas**
- **pgAdmin**: Puedes usar la opción "Import/Export Data" en la interfaz gráfica para cargar datos desde un archivo.
- **Herramientas ETL**: Programas como Apache Nifi, Talend, o Python con librerías como `psycopg2` o `SQLAlchemy` permiten automatizar la carga masiva.

#### **Ejemplo con Python (`psycopg2`):**
```python
import psycopg2

# Conexión a la base de datos
conn = psycopg2.connect(
    host="localhost",
    database="mi_base_datos",
    user="mi_usuario",
    password="mi_contraseña"
)
cur = conn.cursor()

# Archivo CSV
with open('estaciones.csv', 'r') as f:
    cur.copy_expert("COPY public.estaciones FROM STDIN WITH CSV HEADER", f)

# Confirmar y cerrar
conn.commit()
cur.close()
conn.close()
```

### **Consideraciones:**
- Si trabajas con archivos grandes, prefiere `COPY` por su velocidad.
- Verifica la estructura de tu archivo y la tabla destino para evitar errores de formato.

-- Fecha

`SELECT current_date;`

-- hora

`SELECT CURRENT_TIMESTAMP;`

**Lecturas recomendadas**

[Mockaroo - Random Data Generator and API Mocking Tool | JSON / CSV / SQL / Excel](https://mockaroo.com/)

## Cruzar tablas: SQL JOIN

En SQL, un **JOIN** se utiliza para combinar filas de dos o más tablas en función de una condición relacionada entre ellas. Aquí tienes una guía de los tipos de **JOIN** más comunes y cómo utilizarlos:

### **1. INNER JOIN**
Devuelve solo las filas que tienen coincidencias en ambas tablas.

```sql
SELECT tabla1.columna1, tabla2.columna2
FROM tabla1
INNER JOIN tabla2
ON tabla1.columna_comun = tabla2.columna_comun;
```

- **Ejemplo:**
  Si tienes una tabla `clientes` y otra `pedidos`, y quieres ver los pedidos realizados por cada cliente:

  ```sql
  SELECT clientes.nombre, pedidos.fecha
  FROM clientes
  INNER JOIN pedidos
  ON clientes.id_cliente = pedidos.id_cliente;
  ```

### **2. LEFT JOIN (LEFT OUTER JOIN)**
Devuelve todas las filas de la primera tabla (izquierda) y las filas coincidentes de la segunda tabla. Si no hay coincidencia, las columnas de la segunda tabla tendrán valores `NULL`.

```sql
SELECT tabla1.columna1, tabla2.columna2
FROM tabla1
LEFT JOIN tabla2
ON tabla1.columna_comun = tabla2.columna_comun;
```

- **Ejemplo:**
  Si quieres listar todos los clientes, incluso aquellos que no tienen pedidos:

  ```sql
  SELECT clientes.nombre, pedidos.fecha
  FROM clientes
  LEFT JOIN pedidos
  ON clientes.id_cliente = pedidos.id_cliente;
  ```

### **3. RIGHT JOIN (RIGHT OUTER JOIN)**
Es similar al `LEFT JOIN`, pero devuelve todas las filas de la segunda tabla (derecha) y las filas coincidentes de la primera tabla. Si no hay coincidencia, las columnas de la primera tabla tendrán valores `NULL`.

```sql
SELECT tabla1.columna1, tabla2.columna2
FROM tabla1
RIGHT JOIN tabla2
ON tabla1.columna_comun = tabla2.columna_comun;
```

- **Ejemplo:**
  Si quieres listar todos los pedidos, incluso aquellos que no están asociados a un cliente:

  ```sql
  SELECT clientes.nombre, pedidos.fecha
  FROM clientes
  RIGHT JOIN pedidos
  ON clientes.id_cliente = pedidos.id_cliente;
  ```

### **4. FULL JOIN (FULL OUTER JOIN)**
Devuelve todas las filas cuando hay una coincidencia en cualquiera de las tablas. Si no hay coincidencia, las columnas no coincidentes tendrán valores `NULL`.

```sql
SELECT tabla1.columna1, tabla2.columna2
FROM tabla1
FULL JOIN tabla2
ON tabla1.columna_comun = tabla2.columna_comun;
```

- **Ejemplo:**
  Si quieres listar todos los clientes y todos los pedidos, incluso si no tienen coincidencias entre ellos:

  ```sql
  SELECT clientes.nombre, pedidos.fecha
  FROM clientes
  FULL JOIN pedidos
  ON clientes.id_cliente = pedidos.id_cliente;
  ```

### **5. CROSS JOIN**
Devuelve el producto cartesiano de las dos tablas (todas las combinaciones posibles). Se utiliza cuando no hay una relación entre las tablas o cuando quieres todas las combinaciones posibles.

```sql
SELECT tabla1.columna1, tabla2.columna2
FROM tabla1
CROSS JOIN tabla2;
```

### **Claves a tener en cuenta al usar JOIN:**
1. **Relación de las tablas:** Asegúrate de que las tablas tengan una clave común (como `id_cliente`) para realizar el **JOIN** correctamente.
2. **Optimización:** Los **JOIN** pueden ser costosos en términos de rendimiento, especialmente si las tablas son grandes. Utiliza índices en las columnas involucradas.
3. **Alias:** Usa alias para simplificar las consultas y hacerlas más legibles.

   ```sql
   SELECT c.nombre, p.fecha
   FROM clientes AS c
   INNER JOIN pedidos AS p
   ON c.id_cliente = p.id_cliente;
   ```

**Lecturas recomendadas**

[Fundamentos de Bases de Datos](https://platzi.com/clases/bd/)

[http://www.postgresqltutorial.com/wp-content/uploads/2018/12/PostgreSQL-Joins.png](http://www.postgresqltutorial.com/wp-content/uploads/2018/12/PostgreSQL-Joins.png)

[PostgreSQL Joins](http://www.postgresqltutorial.com/postgresql-joins/)

## Funciones Especiales Principales

Las funciones especiales o principales en bases de datos SQL son herramientas que permiten realizar cálculos, transformaciones y manipulaciones de datos. Estas funciones pueden dividirse en varias categorías:

### **1. Funciones de Agregación**
Se usan para realizar cálculos en un conjunto de valores y devolver un único resultado.

- **SUM():** Suma todos los valores de una columna.
  ```sql
  SELECT SUM(salario) AS total_salarios FROM empleados;
  ```

- **AVG():** Calcula el promedio.
  ```sql
  SELECT AVG(edad) AS edad_promedio FROM empleados;
  ```

- **COUNT():** Cuenta el número de filas o valores no nulos.
  ```sql
  SELECT COUNT(*) AS total_empleados FROM empleados;
  ```

- **MAX():** Devuelve el valor máximo.
  ```sql
  SELECT MAX(salario) AS salario_mayor FROM empleados;
  ```

- **MIN():** Devuelve el valor mínimo.
  ```sql
  SELECT MIN(edad) AS edad_menor FROM empleados;
  ```

### **2. Funciones de Fecha y Hora**
Ayudan a trabajar con valores de tipo `DATE`, `TIME` y `TIMESTAMP`.

- **NOW():** Devuelve la fecha y hora actual.
  ```sql
  SELECT NOW() AS fecha_actual;
  ```

- **CURRENT_DATE:** Devuelve la fecha actual.
  ```sql
  SELECT CURRENT_DATE AS fecha_hoy;
  ```

- **DATE_PART():** Extrae una parte específica de una fecha (PostgreSQL).
  ```sql
  SELECT DATE_PART('year', fecha_nacimiento) AS anio_nacimiento FROM empleados;
  ```

- **DATEDIFF():** Calcula la diferencia entre dos fechas (MySQL).
  ```sql
  SELECT DATEDIFF('2025-01-18', '2025-01-01') AS dias_diferencia;
  ```

### **3. Funciones de Texto**
Manipulan cadenas de texto.

- **UPPER():** Convierte el texto a mayúsculas.
  ```sql
  SELECT UPPER(nombre) AS nombre_mayusculas FROM empleados;
  ```

- **LOWER():** Convierte el texto a minúsculas.
  ```sql
  SELECT LOWER(nombre) AS nombre_minusculas FROM empleados;
  ```

- **CONCAT():** Combina varias cadenas en una sola.
  ```sql
  SELECT CONCAT(nombre, ' ', apellido) AS nombre_completo FROM empleados;
  ```

- **LENGTH():** Devuelve la longitud de una cadena.
  ```sql
  SELECT LENGTH(nombre) AS longitud_nombre FROM empleados;
  ```

- **SUBSTRING():** Extrae una parte de una cadena.
  ```sql
  SELECT SUBSTRING(nombre FROM 1 FOR 3) AS primeras_letras FROM empleados;
  ```

### **4. Funciones Matemáticas**
Realizan cálculos numéricos.

- **ROUND():** Redondea un número al número especificado de decimales.
  ```sql
  SELECT ROUND(salario, 2) AS salario_redondeado FROM empleados;
  ```

- **FLOOR():** Devuelve el mayor número entero menor o igual a un valor.
  ```sql
  SELECT FLOOR(edad / 10) AS decadas FROM empleados;
  ```

- **CEIL():** Devuelve el menor número entero mayor o igual a un valor.
  ```sql
  SELECT CEIL(edad / 10) AS decadas_superiores FROM empleados;
  ```

- **ABS():** Devuelve el valor absoluto.
  ```sql
  SELECT ABS(-15) AS valor_absoluto;
  ```

- **POWER():** Eleva un número a la potencia especificada.
  ```sql
  SELECT POWER(edad, 2) AS edad_cuadrada FROM empleados;
  ```

### **5. Funciones de Ventana (Window Functions)**
Se utilizan para realizar cálculos en un conjunto de filas relacionadas.

- **ROW_NUMBER():** Asigna un número secuencial a cada fila.
  ```sql
  SELECT ROW_NUMBER() OVER (ORDER BY salario DESC) AS ranking, nombre FROM empleados;
  ```

- **RANK():** Asigna un rango a cada fila, permitiendo empates.
  ```sql
  SELECT RANK() OVER (ORDER BY salario DESC) AS rango, nombre FROM empleados;
  ```

- **NTILE():** Divide las filas en un número de grupos aproximadamente iguales.
  ```sql
  SELECT NTILE(4) OVER (ORDER BY salario) AS cuartil, nombre FROM empleados;
  ```

Estas funciones te permiten manejar datos de manera eficiente y son esenciales para el análisis, transformación y reporte de información.

Aquí tienes una explicación detallada de cada uno de estos conceptos de SQL en PostgreSQL:

---

### **1. `ON CONFLICT DO`**
`ON CONFLICT DO` se utiliza para manejar conflictos que surgen al intentar insertar datos en una tabla con restricciones únicas.

- **Sintaxis básica**:
  ```sql
  INSERT INTO table_name (column1, column2, ...)
  VALUES (value1, value2, ...)
  ON CONFLICT (conflict_column)
  DO UPDATE SET column1 = value, column2 = value
  WHERE condition;
  ```

- **Opciones**:
  - **`DO NOTHING`**: Ignora el conflicto y no realiza ninguna acción.
    ```sql
    INSERT INTO usuarios (email, nombre)
    VALUES ('user@example.com', 'Juan Pérez')
    ON CONFLICT (email)
    DO NOTHING;
    ```
  - **`DO UPDATE`**: Realiza una actualización de la fila existente.
    ```sql
    INSERT INTO usuarios (email, nombre)
    VALUES ('user@example.com', 'Juan Pérez')
    ON CONFLICT (email)
    DO UPDATE SET nombre = EXCLUDED.nombre;
    ```
    Donde `EXCLUDED` se refiere a los valores de la fila que intentabas insertar.

---

### **2. `RETURNING`**
La cláusula `RETURNING` se utiliza para devolver valores después de una operación de `INSERT`, `UPDATE` o `DELETE`. Esto es útil para obtener los valores generados automáticamente (como los de una columna serial) o verificar cambios.

- **Sintaxis básica**:
  ```sql
  INSERT INTO table_name (column1, column2)
  VALUES (value1, value2)
  RETURNING column1, column2;
  ```

- **Ejemplo**:
  Obtener el ID generado automáticamente:
  ```sql
  INSERT INTO usuarios (email, nombre)
  VALUES ('user@example.com', 'Juan Pérez')
  RETURNING id;
  ```

  Devolver valores después de actualizar:
  ```sql
  UPDATE usuarios
  SET nombre = 'Juan Pérez'
  WHERE email = 'user@example.com'
  RETURNING *;
  ```

---

### **3. `LIKE` e `ILIKE`**
Se utilizan para realizar búsquedas en cadenas de texto con comodines (`%` y `_`).

- **`LIKE`**: Realiza búsquedas **sensibles a mayúsculas**.
  ```sql
  SELECT * FROM usuarios
  WHERE nombre LIKE 'Juan%'; -- Coincide con "Juan" y "Juan Pérez"
  ```

- **`ILIKE`**: Realiza búsquedas **insensibles a mayúsculas**.
  ```sql
  SELECT * FROM usuarios
  WHERE nombre ILIKE 'juan%'; -- Coincide con "juan" o "Juan Pérez"
  ```

- **Comodines**:
  - `%`: Representa cero o más caracteres.
  - `_`: Representa un solo carácter.
  
  Ejemplo:
  ```sql
  SELECT * FROM usuarios
  WHERE nombre LIKE '_uan%'; -- Coincide con "Juan" y "Auan"
  ```

---

### **4. `IS` e `IS NOT`**
Se utilizan para verificar si un valor es `NULL` o si cumple con una condición especial.

- **`IS NULL`**: Verifica si un valor es `NULL`.
  ```sql
  SELECT * FROM usuarios
  WHERE nombre IS NULL;
  ```

- **`IS NOT NULL`**: Verifica si un valor **no** es `NULL`.
  ```sql
  SELECT * FROM usuarios
  WHERE nombre IS NOT NULL;
  ```

- **`IS DISTINCT FROM`**: Compara valores, incluyendo `NULL`, de forma que `NULL IS DISTINCT FROM NULL` devuelve `FALSE`.
  ```sql
  SELECT * FROM usuarios
  WHERE email IS DISTINCT FROM 'user@example.com';
  ```

- **`IS TRUE`, `IS FALSE`**: Evalúan valores booleanos.
  ```sql
  SELECT * FROM usuarios
  WHERE activo IS TRUE; -- Solo selecciona filas donde "activo" sea verdadero
  ```

---

### **Combinación de estos conceptos**

Ejemplo práctico:
```sql
INSERT INTO usuarios (email, nombre, activo)
VALUES ('user@example.com', 'Juan Pérez', TRUE)
ON CONFLICT (email)
DO UPDATE SET nombre = EXCLUDED.nombre
RETURNING id, nombre;

SELECT * FROM usuarios
WHERE nombre ILIKE 'juan%'
AND activo IS TRUE;
```

Esto inserta o actualiza un registro, devuelve el ID y el nombre, y luego consulta todos los usuarios cuyo nombre comienza con "juan" (insensible a mayúsculas) y que están activos.

## Funciones Especiales Avanzadas

Aquí tienes una descripción de cada una de las funciones especiales avanzadas que mencionas, con ejemplos prácticos en PostgreSQL:

### **1. `COALESCE`**
La función `COALESCE` devuelve el primer valor no nulo de una lista de argumentos. Es útil para manejar valores nulos en consultas.

- **Sintaxis**:
  ```sql
  COALESCE(value1, value2, ..., valueN)
  ```

- **Ejemplo**:
  ```sql
  SELECT nombre, COALESCE(telefono, 'Sin teléfono') AS telefono
  FROM usuarios;
  ```
  Si `telefono` es `NULL`, se mostrará "Sin teléfono".

### **2. `NULLIF`**
La función `NULLIF` devuelve `NULL` si dos valores son iguales; de lo contrario, devuelve el primer valor.

- **Sintaxis**:
  ```sql
  NULLIF(value1, value2)
  ```

- **Ejemplo**:
  ```sql
  SELECT nombre, NULLIF(ventas, 0) AS ventas_validas
  FROM empleados;
  ```
  Si `ventas` es `0`, se devolverá `NULL`; de lo contrario, se devolverá el valor de `ventas`.

### **3. `GREATEST`**
La función `GREATEST` devuelve el valor máximo de una lista de argumentos.

- **Sintaxis**:
  ```sql
  GREATEST(value1, value2, ..., valueN)
  ```

- **Ejemplo**:
  ```sql
  SELECT nombre, GREATEST(sueldo_base, bono) AS mayor_ingreso
  FROM empleados;
  ```
  Devuelve el mayor valor entre `sueldo_base` y `bono`.

---

### **4. `LEAST`**
La función `LEAST` devuelve el valor mínimo de una lista de argumentos.

- **Sintaxis**:
  ```sql
  LEAST(value1, value2, ..., valueN)
  ```

- **Ejemplo**:
  ```sql
  SELECT nombre, LEAST(dias_vacaciones, dias_restantes) AS dias_a_usar
  FROM empleados;
  ```
  Devuelve el menor valor entre `dias_vacaciones` y `dias_restantes`.

### **5. Bloques Anónimos (`DO`)**
Un bloque anónimo en PostgreSQL permite ejecutar código PL/pgSQL sin necesidad de crear una función almacenada. Es útil para tareas únicas o scripts.

- **Sintaxis**:
  ```sql
  DO $$
  BEGIN
    -- Código PL/pgSQL aquí
  END;
  $$
  ```

- **Ejemplo**:
  Insertar datos automáticamente:
  ```sql
  DO $$
  BEGIN
    FOR i IN 1..10 LOOP
      INSERT INTO usuarios (nombre, email)
      VALUES ('Usuario ' || i, 'user' || i || '@example.com');
    END LOOP;
  END;
  $$;
  ```

- **Uso Avanzado con Condicionales**:
  ```sql
  DO $$
  BEGIN
    IF EXISTS (SELECT 1 FROM empleados WHERE nombre = 'Juan Pérez') THEN
      RAISE NOTICE 'El empleado ya existe';
    ELSE
      INSERT INTO empleados (nombre, sueldo) VALUES ('Juan Pérez', 5000);
    END IF;
  END;
  $$;
  ```

### **Combinación de funciones avanzadas**
Ejemplo práctico:
```sql
SELECT 
  COALESCE(telefono, 'Sin teléfono') AS telefono,
  NULLIF(sueldo, 0) AS sueldo_validado,
  GREATEST(sueldo, bono) AS mayor_ingreso,
  LEAST(dias_vacaciones, dias_restantes) AS dias_minimos
FROM empleados;
```

Esto aplica múltiples funciones avanzadas para manejar valores nulos, comparar valores y calcular mínimos/máximos en una sola consulta.

## Vistas

Las **vistas volátiles** y las **vistas materializadas** son conceptos clave en bases de datos relacionales que se usan para manejar datos consultados de manera repetitiva o compleja. A continuación, te explico las diferencias, ventajas, desventajas y cómo implementarlas en PostgreSQL.

---

### **1. Vista Volátil**
Una vista volátil, también conocida como vista normal, es una consulta almacenada en la base de datos que no guarda físicamente los datos. Cada vez que accedes a la vista, la consulta subyacente se ejecuta dinámicamente.

- **Características**:
  - Los datos no se almacenan; se obtienen directamente de las tablas subyacentes al momento de la consulta.
  - Siempre muestra datos actualizados porque ejecuta la consulta en tiempo real.
  - Más lenta para consultas complejas debido a la ejecución dinámica.

- **Sintaxis**:
  ```sql
  CREATE VIEW vista_volatil AS
  SELECT columna1, columna2
  FROM tabla
  WHERE condiciones;
  ```

- **Ejemplo**:
  ```sql
  CREATE VIEW ventas_anuales AS
  SELECT cliente_id, SUM(monto) AS total_ventas
  FROM ventas
  WHERE fecha BETWEEN '2023-01-01' AND '2023-12-31'
  GROUP BY cliente_id;
  ```

- **Ventajas**:
  - Siempre devuelve datos actualizados.
  - No ocupa espacio adicional en disco.

- **Desventajas**:
  - Puede ser más lenta para consultas complejas o con grandes volúmenes de datos.

---

### **2. Vista Materializada**
Una vista materializada almacena físicamente los resultados de una consulta en la base de datos. Se actualiza manualmente mediante comandos específicos.

- **Características**:
  - Los datos se almacenan físicamente, lo que mejora el rendimiento para consultas repetitivas.
  - No se actualiza automáticamente; se debe usar el comando `REFRESH MATERIALIZED VIEW` para actualizar los datos.
  - Ocupa espacio en disco.

- **Sintaxis**:
  ```sql
  CREATE MATERIALIZED VIEW vista_materializada AS
  SELECT columna1, columna2
  FROM tabla
  WHERE condiciones;
  ```

- **Ejemplo**:
  ```sql
  CREATE MATERIALIZED VIEW ventas_anuales_materializada AS
  SELECT cliente_id, SUM(monto) AS total_ventas
  FROM ventas
  WHERE fecha BETWEEN '2023-01-01' AND '2023-12-31'
  GROUP BY cliente_id;
  ```

- **Actualizar una vista materializada**:
  ```sql
  REFRESH MATERIALIZED VIEW ventas_anuales_materializada;
  ```

- **Ventajas**:
  - Más rápida para consultas repetitivas, ya que no se ejecuta la consulta subyacente.
  - Útil para escenarios donde los datos no cambian frecuentemente.

- **Desventajas**:
  - Los datos pueden quedar obsoletos si no se actualizan manualmente.
  - Ocupa espacio adicional en disco.

---

### **Diferencias Principales**

| Característica            | Vista Volátil                  | Vista Materializada            |
|---------------------------|--------------------------------|--------------------------------|
| **Almacenamiento**        | No guarda datos físicamente.  | Guarda datos físicamente.     |
| **Actualización**         | Siempre muestra datos actuales. | Debe ser actualizada manualmente. |
| **Rendimiento**           | Más lenta para consultas complejas. | Más rápida para consultas repetitivas. |
| **Espacio en Disco**      | No ocupa espacio adicional.   | Ocupa espacio adicional.      |

---

### **¿Cuándo Usar Cada Una?**
- **Vista Volátil**:
  - Cuando los datos cambian frecuentemente y necesitas que las consultas reflejen siempre el estado más reciente.
  - Ejemplo: Tableros en tiempo real, consultas dinámicas.
  
- **Vista Materializada**:
  - Cuando los datos no cambian con frecuencia y se requiere alto rendimiento para consultas repetitivas.
  - Ejemplo: Reportes periódicos o cálculos agregados costosos.

---

### **Uso Combinado**
En algunos casos, puedes usar una combinación de ambas:
1. Usa vistas volátiles para datos que cambian rápidamente.
2. Usa vistas materializadas para cálculos complejos que necesitan ser ejecutados periódicamente, pero no en tiempo real.

## PL/SQL

En PL/SQL, la consulta que compartiste puede ser adaptada para ejecutarse dentro de un bloque PL/SQL si necesitas procesar los resultados o realizar alguna acción en base a los datos obtenidos. Aquí te dejo una forma de hacerlo:

### Consulta Adaptada en PL/SQL

```plsql
DECLARE
    CURSOR cur_pax_without_travel IS
        SELECT p.n_documento, p.nombre, p.apellido
        FROM pasajeros p
        LEFT JOIN viajes v
        ON v.n_documento = p.n_documento
        WHERE v.id_viaje IS NULL;
    -- Variables para almacenar los datos del cursor
    v_n_documento pasajeros.n_documento%TYPE;
    v_nombre pasajeros.nombre%TYPE;
    v_apellido pasajeros.apellido%TYPE;
BEGIN
    -- Abrimos el cursor
    OPEN cur_pax_without_travel;
    LOOP
        -- Obtenemos cada registro del cursor
        FETCH cur_pax_without_travel INTO v_n_documento, v_nombre, v_apellido;
        -- Salimos del bucle cuando no hay más registros
        EXIT WHEN cur_pax_without_travel%NOTFOUND;
        
        -- Aquí puedes realizar acciones con los datos obtenidos
        DBMS_OUTPUT.PUT_LINE('Documento: ' || v_n_documento || 
                             ', Nombre: ' || v_nombre || 
                             ', Apellido: ' || v_apellido);
    END LOOP;
    -- Cerramos el cursor
    CLOSE cur_pax_without_travel;
END;
/
```

### Explicación

1. **Declaración del Cursor:**
   El cursor `cur_pax_without_travel` contiene la consulta SQL para seleccionar los pasajeros que no tienen un viaje asociado.

2. **Variables:**
   Se declaran variables (`v_n_documento`, `v_nombre`, `v_apellido`) para almacenar temporalmente los datos extraídos del cursor.

3. **Bucle `LOOP`:**
   - Se abre el cursor y se recorren los registros uno por uno con `FETCH`.
   - La condición `EXIT WHEN cur_pax_without_travel%NOTFOUND` asegura que el bucle se detenga cuando se procesen todos los registros.

4. **Acciones:**
   Dentro del bucle puedes realizar cualquier acción necesaria, como imprimir datos con `DBMS_OUTPUT.PUT_LINE` o realizar actualizaciones/inserciones en otras tablas.

5. **Cerrar el Cursor:**
   Después de procesar todos los registros, se cierra el cursor con `CLOSE`.

### Consideraciones

- **Habilitar `DBMS_OUTPUT`:** Si estás ejecutando este bloque desde SQL Developer u otra herramienta, asegúrate de habilitar la salida de `DBMS_OUTPUT` para ver los resultados.
- **Manejo de Excepciones:** Es buena práctica agregar un bloque `EXCEPTION` para manejar posibles errores:
  ```plsql
  EXCEPTION
    WHEN OTHERS THEN
      DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
  ```
- **Validar el Esquema:** Asegúrate de que las tablas `pasajeros` y `viajes` existen y los nombres de las columnas coinciden.

**Lecturas recomendadas**

[PostgreSQL: Documentation: 9.2: PL/pgSQL - SQL Procedural Language](https://www.postgresql.org/docs/9.2/plpgsql.html)

## Triggers

Un **trigger** en PostgreSQL es una función especial que se ejecuta automáticamente en respuesta a un evento en una tabla o vista. Es una herramienta poderosa para mantener la integridad de los datos, realizar auditorías, o automatizar procesos.

### Tipos de Triggers
1. **Basados en el momento:**
   - `BEFORE`: Se ejecuta antes de que ocurra el evento.
   - `AFTER`: Se ejecuta después de que el evento haya ocurrido.
   - `INSTEAD OF`: Reemplaza la operación en vistas.

2. **Basados en el evento:**
   - `INSERT`: Se ejecuta al insertar un registro.
   - `UPDATE`: Se ejecuta al actualizar un registro.
   - `DELETE`: Se ejecuta al eliminar un registro.
   - `TRUNCATE`: Se ejecuta al truncar la tabla.

3. **Nivel de ejecución:**
   - **`FOR EACH ROW`**: Se ejecuta una vez por cada fila afectada.
   - **`FOR EACH STATEMENT`**: Se ejecuta una vez por cada declaración SQL.

### Estructura General
```sql
CREATE OR REPLACE FUNCTION nombre_funcion_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- Lógica del trigger
    RETURN NEW; -- Para operaciones que modifican filas
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER nombre_trigger
{ BEFORE | AFTER | INSTEAD OF } { INSERT | UPDATE | DELETE | TRUNCATE }
ON nombre_tabla
FOR EACH { ROW | STATEMENT }
EXECUTE FUNCTION nombre_funcion_trigger();
```

### Ejemplo 1: Auditoría de Cambios en una Tabla

#### 1. Crear la tabla principal y la tabla de auditoría
```sql
CREATE TABLE empleados (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100),
    salario NUMERIC
);

CREATE TABLE auditoria_empleados (
    id SERIAL PRIMARY KEY,
    accion VARCHAR(50),
    id_empleado INT,
    nombre_empleado VARCHAR(100),
    salario NUMERIC,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 2. Crear la función del trigger
```sql
CREATE OR REPLACE FUNCTION auditoria_cambios()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        INSERT INTO auditoria_empleados (accion, id_empleado, nombre_empleado, salario)
        VALUES ('INSERT', NEW.id, NEW.nombre, NEW.salario);
    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO auditoria_empleados (accion, id_empleado, nombre_empleado, salario)
        VALUES ('UPDATE', NEW.id, NEW.nombre, NEW.salario);
    ELSIF (TG_OP = 'DELETE') THEN
        INSERT INTO auditoria_empleados (accion, id_empleado, nombre_empleado, salario)
        VALUES ('DELETE', OLD.id, OLD.nombre, OLD.salario);
    END IF;
    RETURN NEW; -- Necesario para `INSERT` o `UPDATE`
END;
$$ LANGUAGE plpgsql;
```

#### 3. Crear el trigger
```sql
CREATE TRIGGER trigger_auditoria_empleados
AFTER INSERT OR UPDATE OR DELETE ON empleados
FOR EACH ROW
EXECUTE FUNCTION auditoria_cambios();
```

### Ejemplo 2: Validación Antes de un INSERT

#### 1. Crear la función del trigger
```sql
CREATE OR REPLACE FUNCTION validar_salario()
RETURNS TRIGGER AS $$
BEGIN
    IF (NEW.salario < 0) THEN
        RAISE EXCEPTION 'El salario no puede ser negativo';
    END IF;
    RETURN NEW; -- Permite completar la operación
END;
$$ LANGUAGE plpgsql;
```

#### 2. Crear el trigger
```sql
CREATE TRIGGER trigger_validacion_salario
BEFORE INSERT OR UPDATE ON empleados
FOR EACH ROW
EXECUTE FUNCTION validar_salario();
```

### Ejemplo 3: Trigger para Vistas con `INSTEAD OF`

#### 1. Crear una vista y la función del trigger
```sql
CREATE VIEW empleados_vista AS
SELECT id, nombre FROM empleados;

CREATE OR REPLACE FUNCTION gestionar_vista_empleados()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO empleados (nombre, salario) VALUES (NEW.nombre, 0);
    RETURN NULL; -- La operación original es reemplazada
END;
$$ LANGUAGE plpgsql;
```

#### 2. Crear el trigger en la vista
```sql
CREATE TRIGGER trigger_vista_empleados
INSTEAD OF INSERT ON empleados_vista
FOR EACH ROW
EXECUTE FUNCTION gestionar_vista_empleados();
```

### Consultar Triggers Existentes
Para listar los triggers en una tabla:
```sql
SELECT event_object_table AS tabla, trigger_name AS nombre_trigger, event_manipulation AS evento, action_timing AS momento
FROM information_schema.triggers
WHERE event_object_table = 'nombre_tabla';
```

## Simulando una conexión a Bases de Datos remotas

En PostgreSQL, la extensión `dblink` permite conectarse y ejecutar consultas en bases de datos remotas directamente desde una base de datos local. Esto es útil para simular conexiones a bases de datos remotas en un entorno controlado. A continuación, se describe cómo configurarlo y usarlo:

### **1. Activar la extensión `dblink`**
Primero, asegúrate de que la extensión `dblink` esté instalada y activada en tu base de datos.

Ejecuta este comando en tu base de datos:
```sql
CREATE EXTENSION IF NOT EXISTS dblink;
```

### **2. Configurar una conexión remota con `dblink`**
Usa la función `dblink` para conectarte a una base de datos remota. Puedes especificar los detalles de conexión en una cadena de conexión.

#### Ejemplo básico:
Supongamos que tienes dos bases de datos:
- **Local:** `db_local`
- **Remota:** `db_remota`

##### Crear una conexión y ejecutar una consulta:
```sql
SELECT * 
FROM dblink(
    'host=127.0.0.1 port=5432 dbname=db_remota user=postgres password=admin',
    'SELECT id, nombre FROM usuarios'
) AS remote_data(id INT, nombre TEXT);
```

### **3. Crear un objeto de conexión persistente**
Puedes crear un objeto de conexión persistente con `dblink_connect`, lo que te permite reutilizar la conexión en múltiples consultas.

#### Paso 1: Establecer la conexión
```sql
SELECT dblink_connect(
    'conexion_remota',
    'host=127.0.0.1 port=5432 dbname=db_remota user=postgres password=admin'
);
```

#### Paso 2: Consultar la base de datos remota
```sql
SELECT * 
FROM dblink(
    'conexion_remota',
    'SELECT id, nombre FROM usuarios'
) AS remote_data(id INT, nombre TEXT);
```

#### Paso 3: Cerrar la conexión
```sql
SELECT dblink_disconnect('conexion_remota');
```

### **4. Insertar datos en la base de datos remota**
También puedes usar `dblink` para insertar datos en una base de datos remota.

#### Ejemplo:
```sql
SELECT dblink_exec(
    'conexion_remota',
    'INSERT INTO usuarios (id, nombre) VALUES (3, ''Carlos'')'
);
```

### **5. Simular una conexión con bases de datos locales**
Si deseas simular una conexión a una base de datos remota pero no tienes acceso a una, puedes usar otra base de datos local como "remota". Por ejemplo:

1. Crea otra base de datos en tu servidor PostgreSQL:
   ```bash
   createdb db_remota
   ```

2. Llénala con datos de prueba:
   ```sql
   \c db_remota
   CREATE TABLE usuarios (id SERIAL PRIMARY KEY, nombre TEXT);
   INSERT INTO usuarios (nombre) VALUES ('Alice'), ('Bob');
   ```

3. Usa `dblink` desde la base de datos local para conectarte a esta nueva base de datos.

### **6. Ejemplo completo**
1. Configura las bases de datos:
   ```sql
   CREATE DATABASE db_local;
   CREATE DATABASE db_remota;
   ```

2. En `db_remota`:
   ```sql
   CREATE TABLE usuarios (id SERIAL PRIMARY KEY, nombre TEXT);
   INSERT INTO usuarios (nombre) VALUES ('Alice'), ('Bob');
   ```

3. En `db_local`:
   ```sql
   CREATE EXTENSION dblink;

   SELECT * 
   FROM dblink(
       'host=127.0.0.1 port=5432 dbname=db_remota user=postgres password=admin',
       'SELECT id, nombre FROM usuarios'
   ) AS remote_data(id INT, nombre TEXT);
   ```

### **Consideraciones**
- Asegúrate de que las configuraciones de red y autenticación (`pg_hba.conf`) permitan conexiones remotas.
- Usa conexiones seguras (TLS/SSL) si estás trabajando con bases de datos en entornos de producción.
- La extensión `postgres_fdw` es una alternativa más moderna y robusta a `dblink`.

**Lecturas recomendadas**
[Mockaroo - Random Data Generator and API Mocking Tool | JSON / CSV / SQL / Excel](https://mockaroo.com/)

## Transacciones

