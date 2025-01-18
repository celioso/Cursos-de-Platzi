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