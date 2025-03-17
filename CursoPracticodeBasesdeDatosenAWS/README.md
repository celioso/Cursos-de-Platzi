# Curso Práctico de Bases de Datos en AWS

## Qué aprenderás sobre bases de datos en AWS

En **AWS**, aprenderás sobre diferentes tipos de **bases de datos** según el caso de uso y las necesidades de escalabilidad, rendimiento y administración. Aquí te dejo un resumen de los principales conceptos y servicios:

### **1️⃣ Tipos de bases de datos en AWS**  
AWS ofrece bases de datos **relacionales y no relacionales**, optimizadas para distintos casos de uso:

✅ **Bases de datos relacionales (SQL):**  
   - **Amazon RDS (Relational Database Service)** → Servicio administrado para bases como **MySQL, PostgreSQL, MariaDB, SQL Server y Oracle**.  
   - **Amazon Aurora** → Base de datos relacional escalable y de alto rendimiento compatible con **MySQL y PostgreSQL**.  

✅ **Bases de datos NoSQL:**  
   - **Amazon DynamoDB** → Base de datos NoSQL de clave-valor altamente escalable.  
   - **Amazon DocumentDB** → Base de datos para documentos compatible con **MongoDB**.  
   - **Amazon ElastiCache** → Bases de datos en memoria para caching con **Redis y Memcached**.  

✅ **Bases de datos especializadas:**  
   - **Amazon Redshift** → Almacén de datos (Data Warehouse) para análisis de grandes volúmenes de información.  
   - **Amazon Neptune** → Base de datos de grafos para relaciones complejas (como redes sociales).  
   - **Amazon Timestream** → Base de datos de series temporales para IoT y análisis de datos en tiempo real.  
   - **Amazon QLDB (Quantum Ledger Database)** → Base de datos inmutable para registros contables y auditorías.

### **2️⃣ Qué aprenderás sobre bases de datos en AWS**  

📌 **1. Creación y administración de bases de datos**  
   - Configurar y desplegar bases de datos con **Amazon RDS, Aurora y DynamoDB**.  
   - Elegir entre bases relacionales y NoSQL según el caso de uso.  

📌 **2. Seguridad y encriptación**  
   - Control de accesos con **IAM, roles y políticas**.  
   - Encriptación de datos en tránsito y en reposo con **AWS KMS y SSL/TLS**.  

📌 **3. Replicación y alta disponibilidad**  
   - Configurar **Multi-AZ y Read Replicas** en RDS y Aurora.  
   - Uso de **DynamoDB Streams** y **Cross-Region Replication**.  

📌 **4. Backup y recuperación**  
   - Configurar backups automáticos y snapshots en **RDS, Aurora y DynamoDB**.  
   - Restauración de bases de datos en caso de fallos.  

📌 **5. Monitoreo y optimización**  
   - Uso de **CloudWatch** para métricas y alertas.  
   - Análisis de consultas con **Performance Insights en RDS**.  
   - Escalabilidad automática en DynamoDB.  

📌 **6. Integración con otros servicios de AWS**  
   - Uso de **Lambda** para eventos en bases de datos.  
   - Integración con **AWS Glue** para ETL y análisis de datos.  
   - Uso de **Athena** para consultas en **S3** sin necesidad de servidores.

### **3️⃣ Casos de uso comunes**  
📌 Aplicaciones web escalables con RDS o Aurora.  
📌 Bases NoSQL para aplicaciones en tiempo real con DynamoDB.  
📌 Almacenes de datos para análisis de grandes volúmenes con Redshift.  
📌 Bases de datos en memoria con ElastiCache para mejorar el rendimiento.  
📌 Bases de datos de grafos para modelar relaciones complejas con Neptune.

### **📌 Conclusión:**  
AWS ofrece una gran variedad de bases de datos según el caso de uso. Aprenderás a elegir la mejor opción, configurarla, asegurarla, hacer respaldos, monitorearla y optimizar su rendimiento. 🚀

**Resumen**

Bienvenido al **Curso de Bases de Datos en AWS** de Platzi. Vamos a aprender sobre los servicios de bases de datos relacionales en AWS, principalmente el servicio de **RDS**, y sobre el servicio de bases de datos no relacionales en AWS: **DynamoDB**.

Esta vez nuestro profesor será Carlos Andrés Zambrano, que tiene más de 4 años de experiencia trabajando con AWS.

## Características de Relational Database Service (RDS)

Amazon **RDS** es un servicio administrado de base de datos relacional que facilita la configuración, operación y escalado de bases de datos en la nube de AWS. Soporta varios motores de bases de datos populares y ofrece alta disponibilidad, seguridad y escalabilidad.

### **🔹 1. Compatibilidad con motores de bases de datos**  
AWS RDS soporta múltiples motores de bases de datos:  
✅ **Amazon Aurora** (compatible con MySQL y PostgreSQL)  
✅ **MySQL**  
✅ **PostgreSQL**  
✅ **MariaDB**  
✅ **SQL Server**  
✅ **Oracle Database**

### **🔹 2. Administración simplificada**  
💡 **RDS se encarga de tareas administrativas**, como:  
🔹 Instalación y configuración del motor de base de datos.  
🔹 Aplicación de parches de seguridad.  
🔹 Administración de backups y snapshots automáticos.  
🔹 Recuperación ante fallos con Multi-AZ.

### **🔹 3. Escalabilidad y rendimiento**  
📌 **Escalado vertical**: Puedes aumentar el tamaño de la instancia con mayor RAM y CPU.  
📌 **Escalado horizontal**: Usar **Read Replicas** para mejorar el rendimiento de lectura.  
📌 **Almacenamiento escalable**: Permite **Auto Scaling de almacenamiento** sin interrupciones.  
📌 **Optimización con caché**: Compatible con Amazon **ElastiCache** para mejorar la velocidad de consultas.

### **🔹 4. Alta disponibilidad y replicación**  
🔹 **Multi-AZ (Alta disponibilidad)**: Mantiene una réplica sincronizada en otra zona de disponibilidad (AZ).  
🔹 **Read Replicas**: Permite crear copias solo de lectura para distribuir la carga de consultas.  
🔹 **Respaldo automático y snapshots**: Se pueden restaurar bases de datos fácilmente.

### **🔹 5. Seguridad y cumplimiento**  
🔹 **Cifrado de datos** con **AWS KMS** (en reposo y en tránsito con SSL/TLS).  
🔹 **Control de acceso** con **IAM** y políticas de seguridad.  
🔹 **Integración con AWS CloudTrail** para auditoría de accesos y eventos.  

### **🔹 6. Monitoreo y mantenimiento**  
🔹 **Amazon CloudWatch**: Permite rastrear métricas de rendimiento y configurar alertas.  
🔹 **Performance Insights**: Identifica consultas lentas y cuellos de botella en la base de datos.  
🔹 **RDS Proxy**: Mejora la administración de conexiones para bases de datos de alto tráfico.

### **🔹 7. Costos y modelo de pago**  
💰 **Pago por uso**: Se paga solo por lo que se consume, incluyendo:  
✅ Tipo y tamaño de la instancia.  
✅ Almacenamiento y transferencia de datos.  
✅ Uso de **Read Replicas** y Multi-AZ.  

También existe el **modo serverless con Amazon Aurora**, que escala automáticamente según la demanda.

### **📌 Conclusión**  
AWS RDS es una solución potente y administrada para bases de datos relacionales, ideal para empresas que buscan alto rendimiento, seguridad y escalabilidad sin preocuparse por la administración manual. 🚀

**Resumen**

RDS (Relational Database Service) es un servicio de AWS enfocado a bases de datos relacionales con compatibilidad a 6 motores de bases de datos: Amazon Aurora, MySQL, MariaDB, PostgreSQL, Oracle y Microsoft SQL Server, cada uno con sus características, integraciones y limitaciones.

Entre sus características principales podemos destacar los **backups automáticos** con un tiempo de retención de hasta 35 días, es decir, si encontramos algún problema con nuestras bases de datos podemos restablecerlas a la hora, minuto y segundo que necesitemos dentro del periodo de retención. Recuerda que por defecto este periodo es de 7 días. También tenemos la opción de hacer **backups manuales**, podemos tomar **snapshots** manuales en cualquier momento si nuestra aplicación lo necesita. Además, AWS por defecto tomará un snapshot final de nuestras bases de datos antes de eliminarlas, así podremos restablecerla si tenemos algún inconveniente.

Todas las bases de datos relacionales utilizan un **sistema de almacenamiento**, si la carga de lectura y escritura son constantes, el sistema General Purpose funciona muy bien, sin embargo, podemos utilizar el sistema Provisioned Storage cuando requerimos de altas cantidades de consumo y operaciones de disco.

RDS es un sistema completamente administrado, esto quiere decir que AWS reduce nuestra carga operativa automatizando muchas tareas de nuestra base de datos, por ejemplo, las actualizaciones. A nivel de seguridad contamos con muchas opciones, una de ellas es la posibilidad de encriptar nuestra base de datos para que solo nosotros y las personas o roles que especifiquemos tengan acceso.

También tenemos integración con otros servicios de AWS, por ejemplo, IAM para administrar a los usuarios, roles, grupos y políticas de conexión a la base de datos por medio de tokens con máximo 20 conexiones por segundo (recomendado para escenarios de prueba), o la integración de Enhanced monitoring para hacer monitoreo en tiempo real nuestras bases de datos (recuerda que además de subir el precio, no está disponible para instancias small).

**Lecturas recomendadas**

[https://docs.aws.amazon.com/rds/index.html](https://docs.aws.amazon.com/rds/index.html)

## Desplegando nuestra primer base de datos

Crear una base de datos en **Amazon RDS** es un proceso sencillo que se puede hacer a través de la **Consola de AWS**, la **CLI** o **Terraform**. A continuación, veremos el proceso paso a paso utilizando la **Consola de AWS**.

### **🛠 Paso 1: Iniciar Sesión en AWS**
1. Ingresa a la consola de AWS: [AWS Console](https://aws.amazon.com/console/)
2. En el **buscador**, escribe **RDS** y selecciona **Amazon RDS**.

### **🛠 Paso 2: Crear una Nueva Base de Datos**
1. En la página de Amazon RDS, haz clic en **"Crear base de datos"**.
2. Selecciona el **método de creación**:
   - **Estándar** (personalizado)
   - **Facilitado** (configuración automática para pruebas)

### **🛠 Paso 3: Elegir el Motor de Base de Datos**
AWS RDS soporta varios motores, elige el que necesites:
✅ **Amazon Aurora** (MySQL/PostgreSQL compatible)  
✅ **MySQL**  
✅ **PostgreSQL**  
✅ **MariaDB**  
✅ **Oracle**  
✅ **SQL Server**  

*Para este ejemplo, seleccionaremos **MySQL**.*

### **🛠 Paso 4: Configurar la Instancia**
1. **Versión del motor**: Elige la versión más reciente recomendada.  
2. **Identificador de la base de datos**: Escribe un nombre único, por ejemplo: `mi-base-datos`.  
3. **Autenticación**:
   - Usuario administrador: `admin`
   - Contraseña: Elige una fuerte o permite que AWS la genere. 

### **🛠 Paso 5: Configuración de la Instancia**
1. **Clase de instancia** (elige según los recursos que necesitas):  
   - `db.t3.micro` (gratis en el **Free Tier**)  
   - Instancias más grandes para producción (`db.m5.large`, `db.r5.large`, etc.).  
2. **Almacenamiento**:  
   - **General Purpose SSD (gp3/gp2)** – Recomendado para cargas estándar.  
   - **Provisioned IOPS (io1/io2)** – Para alta velocidad de E/S.  
   - **Magnetic (st1/sc1)** – Para almacenamiento barato y accesible.  
3. **Tamaño de almacenamiento**: 20 GB (mínimo, puede aumentar automáticamente).

### **🛠 Paso 6: Configurar Alta Disponibilidad y Conectividad**
1. **Alta Disponibilidad (Multi-AZ)**:  
   - **Activado** para producción.  
   - **Desactivado** para pruebas y entornos de desarrollo.  
2. **VPC y Subredes**:  
   - Selecciona una **VPC** o usa la predeterminada.  
   - Habilita **acceso público** si deseas conectarte desde fuera de AWS.  
3. **Grupo de seguridad**:  
   - Crea un nuevo grupo de seguridad o usa uno existente.  
   - Permite tráfico en el **puerto 3306** (para MySQL) desde IPs seguras.

### **🛠 Paso 7: Configurar Backups y Monitoreo**
1. **Backups automáticos**:  
   - Define el **período de retención** (1-35 días).  
   - Habilita **copias automáticas** en otra región si es necesario.  
2. **Monitoreo**:  
   - Activa **Amazon CloudWatch** para rastrear métricas.  
   - Habilita **Performance Insights** si deseas un análisis detallado.

### **🛠 Paso 8: Crear la Base de Datos**
1. **Revisa todas las configuraciones**.  
2. **Haz clic en "Crear base de datos"**.  
3. AWS desplegará la instancia (toma unos minutos).

### **🛠 Paso 9: Conectar a la Base de Datos**
Cuando la base de datos esté activa:  
1. Ve a la consola de RDS → **Instancias**.  
2. Copia el **endpoint** de conexión (algo como `mi-base.cdgk3m1w.us-east-1.rds.amazonaws.com`).  
3. Usa **MySQL Workbench**, **DBeaver** o la **línea de comandos** para conectarte:  

```bash
mysql -h mi-base.cdgk3m1w.us-east-1.rds.amazonaws.com -u admin -p
```

### **🎯 ¡Listo! Ya tienes tu primera base de datos en AWS RDS.**  
Puedes probar creando una tabla y agregando datos en MySQL:

```sql
CREATE DATABASE mi_app;
USE mi_app;

CREATE TABLE usuarios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100),
    email VARCHAR(100) UNIQUE
);

INSERT INTO usuarios (nombre, email) VALUES ('Mario Vargas', 'mario@example.com');
SELECT * FROM usuarios;
```

### **📌 Conclusión**
AWS RDS es una excelente opción para manejar bases de datos en la nube sin preocuparse por la infraestructura. Con este proceso, ya puedes empezar a usar tu base de datos de manera segura, escalable y eficiente. 🚀

### Resumen

### ¿Cómo desplegar una base de datos en Amazon RDS?

Desplegar una base de datos en Amazon RDS te permitirá aprovechar la escalabilidad y flexibilidad que ofrece la nube de AWS. En este tutorial, realizaremos un recorrido por el proceso para desplegar tu primera base de datos de cero, entendiendo las distintas opciones y configuraciones necesarias para adaptar una base de datos a tus necesidades.

### ¿Cuál es el primer paso?

Para desplegar una base de datos en RDS, lo primero que necesitamos es acceder a la consola de AWS. Dentro de ésta, seleccionaremos el servicio de RDS. Una vez dentro, haremos clic en "Create database" para iniciar el proceso. Podrás observar un panel que muestra los distintos motores de bases de datos disponibles:

- MySQL
- MariaDB
- PostgreSQL
- Oracle
- SQL Server
- Amazon Aurora

### ¿Cómo elegir el motor de base de datos adecuado?

Cada motor de base de datos tiene características distintas que se ajustan a diferentes necesidades. Por ejemplo:

- **MySQL**: Soporta bases de datos hasta 32 TB y ofrece servicios como backups automáticos y réplicas de lectura.
- **MariaDB y PostgreSQL**: Tienen características similares, pero el proceso de selección cambia según el licenciamiento que necesites.
- **Oracle y SQL Server**: Ofrecen distintas versiones (Enterprise, Standard) cada una con diferentes funcionalidades y precios.

### ¿Qué sucede al seleccionar MySQL?

Al elegir MySQL, AWS recomienda utilizar Amazon Aurora para bases de datos de producción debido a su alto rendimiento y disponibilidad. Sin embargo, en esta demostración, procederemos con MySQL para la práctica.

- **Versión del motor**: Podrás elegir entre diferentes versiones del motor de base de datos.
- **Instancia**: AWS sugerirá una instancia apta para la capa gratuita (por defecto db.t2.micro), útil para pruebas y desarrollo.

### Configuración de la base de datos

Al configurar la base de datos, surgen varias opciones críticas:

- **Tipo de despliegue**: Puedes elegir despliegues simples o multi-AZ para alta disponibilidad.
- **Almacenamiento**: Selecciona entre almacenamiento general o provisioned IOPS para un rendimiento específico.
- **Identificadores**: Define un nombre para la instancia y las credenciales para acceder a la base de datos (username y password).

### VPC y configuración de red

Aquí decidirás si tu base de datos será accesible públicamente o no, realizando una configuración en una Virtual Private Cloud (VPC):

Acceso público: No se recomienda por seguridad, pero puedes configurarlo para pruebas.
Grupo de seguridad: Se puede crear uno nuevo o usar uno existente para definir reglas de acceso.

### Otras características avanzadas

- **Encriptación**: Está disponible con una instancia adecuada. Puedes habilitarla y seleccionar una clave KMS.
- **Monitoreo y mantenimiento**: Activa el monitoreo en tiempo real y define ventanas de mantenimiento según se requiera.

Después de revisar y ajustar todas las configuraciones, procederemos a crear la base de datos haciendo clic en "Create database". Esto marca el despliegue exitoso de tu base de datos en RDS con características de alta disponibilidad y ajustes personalizados según tus necesidades. ¡Buena suerte!

## Conexión gráfica a nuestra base de datos

Después de desplegar nuestra base de datos en **Amazon RDS**, podemos conectarnos usando una herramienta gráfica. Algunas opciones populares son:  

✅ **MySQL Workbench** (para MySQL y MariaDB)  
✅ **DBeaver** (compatible con MySQL, PostgreSQL, Oracle, SQL Server, etc.)  
✅ **PgAdmin** (para PostgreSQL)  
✅ **SQL Server Management Studio (SSMS)** (para SQL Server)  

A continuación, te mostraré cómo conectar gráficamente una base de datos **MySQL en RDS** usando **MySQL Workbench**.

### **🛠 Paso 1: Obtener las Credenciales de Conexión**
1. **Abre la Consola de AWS** y navega a **Amazon RDS**.  
2. **Selecciona tu base de datos** en la lista de instancias.  
3. **Copia el "Endpoint"**, que es la URL para conectarte.  
   - Ejemplo: `mi-base.cdgk3m1w.us-east-1.rds.amazonaws.com`  
4. Asegúrate de tener:  
   - **Usuario administrador:** `admin` (o el que configuraste)  
   - **Contraseña:** (la que definiste al crear la base)  
   - **Puerto:** `3306` (para MySQL)

### **🛠 Paso 2: Configurar el Acceso en el Grupo de Seguridad**
Si la base de datos está **en una VPC privada**, debes permitir conexiones:  
1. **Ir a "EC2" → "Grupos de Seguridad"**.  
2. **Selecciona el grupo de seguridad asignado a RDS**.  
3. **Edita las reglas de entrada** y **agrega una nueva regla**:
   - **Tipo:** MySQL/Aurora  
   - **Protocolo:** TCP  
   - **Puerto:** `3306`  
   - **Fuente:** `Tu dirección IP` o `0.0.0.0/0` *(para acceso público, no recomendado en producción)*.

### **🛠 Paso 3: Descargar e Instalar MySQL Workbench**
1. **Descargar** desde: [MySQL Workbench](https://dev.mysql.com/downloads/workbench/).  
2. Instalar y abrir la aplicación.

### **🛠 Paso 4: Crear la Conexión en MySQL Workbench**
1. **Abrir MySQL Workbench**.  
2. En la pestaña **"Database"** → **"Manage Connections"** → **"New"**.  
3. Configurar la conexión:  
   - **Connection Name:** `AWS-RDS-MySQL`  
   - **Connection Method:** `Standard (TCP/IP)`  
   - **Hostname:** `mi-base.cdgk3m1w.us-east-1.rds.amazonaws.com`  
   - **Port:** `3306`  
   - **Username:** `admin`  
   - **Password:** Click en **"Store in Vault"** para guardarla.  
4. **Haz clic en "Test Connection"**.  

Si la conexión es exitosa, MySQL Workbench confirmará el acceso. 

### **🛠 Paso 5: Explorar y Administrar la Base de Datos**
1. En la pestaña de conexiones, selecciona **AWS-RDS-MySQL**.  
2. Se abrirá el panel con tu base de datos.  
3. Puedes ejecutar consultas, administrar tablas y visualizar datos gráficamente.  

Ejemplo:  
```sql
SHOW DATABASES;
USE mi_app;
SELECT * FROM usuarios;
```

### **🎯 Conclusión**
Ahora puedes gestionar tu base de datos en **Amazon RDS** de forma gráfica usando **MySQL Workbench**. Esta conexión facilita la administración, creación de tablas y consultas de datos sin necesidad de usar la línea de comandos. 🚀

### Resumen

En esta clase vamos a conectarnos a la base de datos que creamos en la clase anterior usando la herramienta *MySQL Workbench*, que nos permite ejecutar y visualizar nuestros comandos muy fácilmente.

Cuando utilizamos el servicio RDS con el motor de MySQL podemos crear multiples bases de datos con un solo *endpoint* (una sola conexión), ya que entre las características de este motor encontramos la cantidad de bases de datos ilimitada. Obviamente, debemos tener en cuenta que nuestras instancias deberían soportar la cantidad de bases de datos que vamos a utilizar, y las herramientas de monitoreo nos pueden ayudar a medir esta relación de tamaño y rendimiento.

Recuerda que si necesitamos un permiso de usuarios diferente para cada base de datos vamos a necesitar configuraciones diferentes en las keys (*llaves de acceso*) de nuestra instancia.

Nota: si la base de datos no conecta toca ir al grupo de seguridad y ver si esta creada:

| Type | Protocol | Port Range | Source | Description |
|---|---|---|---|---|
| MYSQL/Aurora | TCP | 3306 | MY IP | Conect DB |

**Lecturas recomendadas**

[MySQL :: MySQL Workbench](https://www.mysql.com/products/workbench/)

## Creación de una tabla 

Después de conectarnos a nuestra base de datos en **Amazon RDS**, podemos crear tablas para almacenar información estructurada.  

### **🛠 1. Seleccionar la Base de Datos**  
Antes de crear la tabla, seleccionamos la base de datos donde se almacenará:  
```sql
USE mi_base_de_datos;  -- Solo para MySQL/MariaDB
```
En **PostgreSQL**, el comando sería:  
```sql
\c mi_base_de_datos;
```

### **🛠 2. Crear la Tabla**  
A continuación, creamos una tabla de ejemplo llamada **usuarios** con los siguientes campos:  

- `id` (clave primaria, autoincremental)  
- `nombre` (texto, máximo 100 caracteres)  
- `email` (texto, único)  
- `fecha_registro` (fecha y hora de registro)  

#### **📌 Código para MySQL y PostgreSQL**
```sql
CREATE TABLE usuarios (
    id SERIAL PRIMARY KEY,  -- Autoincremental
    nombre VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **📌 Código para SQL Server**
```sql
CREATE TABLE usuarios (
    id INT IDENTITY(1,1) PRIMARY KEY,  -- Autoincremental
    nombre NVARCHAR(100) NOT NULL,
    email NVARCHAR(150) UNIQUE NOT NULL,
    fecha_registro DATETIME DEFAULT GETDATE()
);
```

### **🛠 3. Verificar la Creación de la Tabla**
Para asegurarnos de que la tabla fue creada correctamente, podemos ejecutar:  

```sql
SHOW TABLES;  -- MySQL
SELECT table_name FROM information_schema.tables WHERE table_schema = 'mi_base_de_datos';  -- PostgreSQL
EXEC sp_tables;  -- SQL Server
```

Para ver la estructura de la tabla:  
```sql
DESCRIBE usuarios;  -- MySQL
SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'usuarios';  -- PostgreSQL
EXEC sp_columns 'usuarios';  -- SQL Server
```

### **🛠 4. Insertar Datos de Prueba**
Después de crear la tabla, podemos insertar algunos registros de prueba:  
```sql
INSERT INTO usuarios (nombre, email) VALUES 
('Mario Vargas', 'mario@example.com'),
('Ana López', 'ana@example.com');
```

### **🛠 5. Consultar los Datos**
Para ver los registros insertados, usamos:  
```sql
SELECT * FROM usuarios;
```

### **🎯 Conclusión**
Ahora tenemos una tabla creada y lista para almacenar datos en **AWS RDS**. 🚀

### Resumen

### ¿Cómo crear una tabla en MySQL Workbench?
MySQL Workbench es una herramienta poderosa que simplifica el manejo y la gestión de bases de datos de manera gráfica. Crear tablas e ingresar datos nunca había sido más accesible. En esta guía, te mostraré cómo usar esta herramienta para crear una tabla llamada "trabajadores" y llenarla con información relevante.

### ¿Cómo añadir columnas a una tabla?

Para empezar, crea una nueva tabla haciendo clic derecho en "Tables" y seleccionando "Create Table". Pide un nombre, así que ponle "trabajadores". Luego, crea las columnas necesarias:

1. **ID de trabajadores**: de tipo entero.
2. **Nombre de trabajadores**: para almacenar nombres.
3. **Fecha de ingreso**: de tipo DATETIME.
4. **Fecha de nacimiento**: también de tipo DATE.
5. **Cargo**: un campo adicional para el puesto del trabajador.

Por defecto, la herramienta toma la columna ID de trabajadores como clave primaria, pero esto se puede modificar según tus necesidades.

### ¿Cómo se aplican cambios en la base de datos?

Después de definir las columnas, haz clic en "Apply". MySQL Workbench genera una consulta SQL automática que se puede ejecutar para implementar los cambios en la base de datos. Revisa y aplica estos cambios para crear realmente la tabla en el sistema.

### ¿Cómo se ingresa datos en la tabla?

Con la tabla creada, puedes visualizarla y empezar a introducir datos. Supongamos que deseas añadir un nuevo trabajador:

- **ID**: 1
- **Nombre**: Carlos
- **Fecha de ingreso**: 20/08/2018
- **Fecha de nacimiento**: 05/03/1988
- **Cargo**: Arquitecto

Completa estos datos y selecciona "Apply". MySQL ejecutará una consulta `INSERT INTO`, lo que asegura que los datos se guarden en la base de datos de manera segura.

### ¿Es posible usar comandos SQL directamente?

Además de la interfaz gráfica, MySQL Workbench permite ejecutar comandos SQL directamente en la consola. Esto es útil si prefieres escribir tus propias consultas o automatizar procesos de ingreso de datos.

### ¿Cuáles son las ventajas de usar la interfaz gráfica de MySQL Workbench?

La interfaz gráfica no solo simplifica el proceso de creación y gestión de tablas, sino que también facilita la configuración, visualización y manejo de múltiples bases de datos. Ideal para administradores de bases de datos que buscan eficiencia sin comprometer la funcionalidad.

### ¿Qué es importante recordar al usar MySQL Workbench?

- **Facilidad de acceso**: Ofrece una manera intuitiva de interactuar con bases de datos.
- **Variedad de funcionalidades**: Desde crear tablas hasta gestionar bases de datos completas.
- **Soporte de MySQL Server**: Totalmente integrado y optimizado para trabajar en este entorno.

Explora la herramienta y descubre cómo puede transformar tu experiencia en la gestión de bases de datos. Si eres principiante, considera este tu primer paso hacia un manejo más avanzado y eficiente de datos. ¡Sigue aprendiendo y expandiendo tus habilidades!

## Conexión por consola a nuestra base de datos

Para conectarnos a una base de datos en **Amazon RDS** desde la consola, seguimos los siguientes pasos según el motor de base de datos elegido. 

### **🛠 1. Obtener la Información de la Base de Datos**  
Desde la **Consola de AWS**:  
1. Ir a **RDS** → **Bases de datos**.  
2. Seleccionar la base de datos creada.  
3. Buscar el **Endpoint** (nombre del host) y el **puerto**.  
4. Asegurarse de que la base de datos permita conexiones remotas (ajustando **Grupos de Seguridad** si es necesario).

### **🛠 2. Conexión por Consola según el Motor de Base de Datos**  

### **🔹 MySQL o MariaDB**  
#### 📌 **Comando desde la terminal (Linux/Mac/WSL) o cmd (Windows)**  
```bash
mysql -h tu-endpoint-rds.amazonaws.com -P 3306 -u tu_usuario -p
```
Luego, ingresa tu contraseña cuando se solicite.

#### 📌 **Ejemplo con base de datos específica**
```bash
mysql -h tu-endpoint-rds.amazonaws.com -P 3306 -u tu_usuario -p tu_base_de_datos
```

### **🔹 PostgreSQL**  
#### 📌 **Instalar el Cliente de PostgreSQL si no lo tienes instalado**  
```bash
sudo apt install postgresql-client  # Ubuntu/Debian
brew install libpq  # macOS
```

#### 📌 **Comando de conexión**  
```bash
psql -h tu-endpoint-rds.amazonaws.com -p 5432 -U tu_usuario -d tu_base_de_datos
```
Luego, ingresa la contraseña cuando se solicite.

### **🔹 SQL Server**  
#### 📌 **Usando `sqlcmd` en Windows**  
```cmd
sqlcmd -S tu-endpoint-rds.amazonaws.com -U tu_usuario -P "tu_contraseña" -d tu_base_de_datos
```
Si `sqlcmd` no está disponible, puedes instalar **SQL Server Command Line Tools**.

### **🛠 3. Probar la Conexión**  
Si la conexión es exitosa, deberías poder ejecutar comandos SQL como:  

```sql
SELECT NOW();  -- PostgreSQL / MySQL
SELECT GETDATE();  -- SQL Server
```

### **🎯 Solución de Problemas**
✔ **Error de conexión**: Revisar reglas del **Grupo de Seguridad** en AWS para permitir conexiones desde tu IP.  
✔ **Acceso denegado**: Verificar usuario y contraseña.  
✔ **Puerto bloqueado**: Asegurar que el firewall local permita conexiones al puerto (3306, 5432, 1433 según el motor).  

Con esto, ¡ya estás conectado a tu base de datos en AWS RDS desde la consola! 🚀

**Resumen**

En esta clase vamos a conectarnos a nuestra base de datos por medio del bash de Linux. Para esto, debemos crear la instancia de un servidor de AWS con un grupo de seguridad que posteriormente vamos a configurar para que la base de datos y el servidor sean accesibles entre ellos.

El desafió de esta clase es identificar al menos 2 características de RDS que actualmente no tenemos en otros sistemas bases de datos.

**NOTA:** Parece que **Amazon Linux 2023** no tiene `mysql` en sus repositorios por defecto. Para instalar el cliente de MySQL en **Amazon Linux 2023**, prueba con los siguientes pasos:  

### 🔹 **1. Instalar `mysql` desde el repositorio correcto**  
Ejecuta este comando en tu instancia EC2:  
```bash
sudo yum install mysql-community-client
```
Si esto no funciona, intenta habilitar el repositorio correcto:  
```bash
sudo amazon-linux-extras enable mariadb10.5
sudo yum install mariadb
```
**Nota**: `mariadb` es compatible con MySQL y puede usarse como cliente para conectarse a bases de datos MySQL.

### 🔹 **2. Verificar la instalación**
Una vez instalado, verifica que el cliente de MySQL está disponible con:  
```bash
mysql --version
```

### 🔹 **3. Conectarte a tu Base de Datos en RDS**
Si la instalación es correcta, usa este comando para conectarte:  
```bash
mysql -h tu-endpoint-rds.amazonaws.com -P 3306 -u tu_usuario -p
```
Te pedirá la contraseña y luego estarás conectado.

Si sigues teniendo problemas, dime qué error exacto aparece para darte una solución más precisa. 🚀

**NOTA:**

SSH cambiar dato de grupo de seguridad

![Gruposdeseguridad.png](images/Gruposdeseguridad.png)

se nombran con cada uno el primero sgec2 que coresponde a la instacian de la maquina virtual en linux y se configura asi:


![security Groups ec2](images/sgec2.png)

y luego el de la base de datos:

![security Groups RBS](images/sgrdb.png)

```bash
Windows PowerShell
Copyright (C) Microsoft Corporation. Todos los derechos reservados.

Instale la versión más reciente de PowerShell para obtener nuevas características y mejoras. https://aka.ms/PSWindows

PS C:\Users\celio> cd .\OneDrive\Escritorio\programación\platzi\CursoPracticodeBasesdeDatosenAWS\
PS C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoPracticodeBasesdeDatosenAWS> ssh -i "prueba.pem" ec2-user@ec2-54-236-237-225.compute-1.amazonaws.com
Warning: Identity file prueba.pem not accessible: No such file or directory.
ssh: connect to host ec2-54-236-237-225.compute-1.amazonaws.com port 22: Connection timed out
PS C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoPracticodeBasesdeDatosenAWS> ssh -i "keydatabase.pem" ec2-user@ec2-3-82-22-6.compute-1.amazonaws.com
   ,     #_
   ~\_  ####_        Amazon Linux 2023
  ~~  \_#####\
  ~~     \###|
  ~~       \#/ ___   https://aws.amazon.com/linux/amazon-linux-2023
   ~~       V~' '->
    ~~~         /
      ~~._.   _/
         _/ _/
       _/m/'
Last login: Sun Mar 16 21:32:19 2025 from 181.32.29.193
[ec2-user@ip-172-31-23-105 ~]$ mysql --version

[ec2-user@ip-172-31-23-105 ~]$ sudo rpm --import https://repo.mysql.com/RPM-GPG-KEY-mysql-2023
[ec2-user@ip-172-31-23-105 ~]$ sudo yum clean all
38 files removed
[ec2-user@ip-172-31-23-105 ~]$ sudo yum install mysql
Amazon Linux 2023 repository              52 MB/s |  33 MB     00:00
Amazon Linux 2023 Kernel Livepatch repos 125 kB/s |  14 kB     00:00
MySQL 8.0 Community Server                55 MB/s | 2.3 MB     00:00
MySQL Connectors Community               5.9 MB/s |  74 kB     00:00
MySQL Tools Community                     22 MB/s | 946 kB     00:00
Dependencies resolved.
=========================================================================
 Package                     Arch   Version      Repository         Size
=========================================================================
Installing:
 mysql-community-client      x86_64 8.0.41-1.el9 mysql80-community 3.4 M
Installing dependencies:
 mysql-community-client-plugins
                             x86_64 8.0.41-1.el9 mysql80-community 1.4 M
 mysql-community-common      x86_64 8.0.41-1.el9 mysql80-community 556 k
 mysql-community-libs        x86_64 8.0.41-1.el9 mysql80-community 1.5 M

Transaction Summary
=========================================================================
Install  4 Packages

Total download size: 6.8 M
Installed size: 96 M
Is this ok [y/N]: y
Downloading Packages:
(1/4): mysql-comm  0% [                ] ---  B/s |   0  B     --:-- ETA
(1/4): mysql-community-client-8.0.41-1.e  53 MB/s | 3.4 MB     00:00
(2/4): mysql-community-client-plugins-8.  20 MB/s | 1.4 MB     00:00
(3/4): mysql-community-common-8.0.41-1.e 7.4 MB/s | 556 kB     00:00
(4/4): mysql-community-libs-8.0.41-1.el9  58 MB/s | 1.5 MB     00:00
-------------------------------------------------------------------------
Total                                     67 MB/s | 6.8 MB     00:00
Running transaction check
Transaction check succeeded.
Running transaction test
Transaction test succeeded.
Running transaction
  Preparing        :                                                 1/1
  Installing       : mysql-community-client-plugins-8.0.41-1.el9.x   1/4
  Installing       : mysql-community-common-8.0.41-1.el9.x86_64      2/4
  Installing       : mysql-community-libs-8.0.41-1.el9.x86_64        3/4
  Running scriptlet: mysql-community-libs-8.0.41-1.el9.x86_64        3/4
  Installing       : mysql-community-client-8.0.41-1.el9.x86_64      4/4
  Running scriptlet: mysql-community-client-8.0.41-1.el9.x86_64      4/4
  Verifying        : mysql-community-client-8.0.41-1.el9.x86_64      1/4
  Verifying        : mysql-community-client-plugins-8.0.41-1.el9.x   2/4
  Verifying        : mysql-community-common-8.0.41-1.el9.x86_64      3/4
  Verifying        : mysql-community-libs-8.0.41-1.el9.x86_64        4/4

Installed:
  mysql-community-client-8.0.41-1.el9.x86_64
  mysql-community-client-plugins-8.0.41-1.el9.x86_64
  mysql-community-common-8.0.41-1.el9.x86_64
  mysql-community-libs-8.0.41-1.el9.x86_64

Complete!
[ec2-user@ip-172-31-23-105 ~]$ mysql --version
mysql  Ver 8.0.41 for Linux on x86_64 (MySQL Community Server - GPL)
[ec2-user@ip-172-31-23-105 ~]$ mysql -h tu-endpoint-rds.amazonaws.com -P 3306 -u admin -p
Enter password:
ERROR 2005 (HY000): Unknown MySQL server host 'tu-endpoint-rds.amazonaws.com' (-2)
[ec2-user@ip-172-31-23-105 ~]$ mysql -h platzipruebaidentificador.cyt0kcygsc7w.us-east-1.rds.amazonaws.com -P 3306 -u admin -p
Enter password:
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 47
Server version: 8.0.40 Source distribution

Copyright (c) 2000, 2025, Oracle and/or its affiliates.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| PlatziDB           |
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
5 rows in set (0.00 sec)

mysql> USE PlatziDB
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
mysql> USE PlatziDB;
Database changed
mysql> use PlatziDB;
Database changed
mysql> show tables;
+--------------------+
| Tables_in_PlatziDB |
+--------------------+
| trabajadores       |
+--------------------+
1 row in set (0.00 sec)

mysql> SELECT * FROM trabajadores;
+----------------+--------+---------------------+---------------------+------------+
| idtrabajadores | Nombre | Fecha_Ingreso       | Fecha_de_Nacimiento | Cargo      |
+----------------+--------+---------------------+---------------------+------------+
|              1 | Carlos | 2018-08-08 00:00:00 | 1988-03-05 00:00:00 | arquitecto |
+----------------+--------+---------------------+---------------------+------------+
1 row in set (0.01 sec)

mysql> SELECT * FROM trabajadores;
+----------------+--------+---------------------+---------------------+------------+
| idtrabajadores | Nombre | Fecha_Ingreso       | Fecha_de_Nacimiento | Cargo      |
+----------------+--------+---------------------+---------------------+------------+
|              1 | Carlos | 2018-08-08 00:00:00 | 1988-03-05 00:00:00 | arquitecto |
|              2 | Mario  | 2019-02-25 00:00:00 | 1984-03-25 00:00:00 | Tecnico    |
+----------------+--------+---------------------+---------------------+------------+
2 rows in set (0.00 sec)

mysql> INSERT INTO `PlatziDB`.`trabajadores` (`idtrabajadores`, `Nombre`, `Fecha_Ingreso`, `Fecha_de_Nacimiento`, `Cargo`) VALUES ('3', 'Camila',
'2015-12-25', '1998-08-03', 'Secretaria');
Query OK, 1 row affected (0.00 sec)

mysql> SELECT * FROM trabajadores;
+----------------+--------+---------------------+---------------------+------------+
| idtrabajadores | Nombre | Fecha_Ingreso       | Fecha_de_Nacimiento | Cargo      |
+----------------+--------+---------------------+---------------------+------------+
|              1 | Carlos | 2018-08-08 00:00:00 | 1988-03-05 00:00:00 | arquitecto |
|              2 | Mario  | 2019-02-25 00:00:00 | 1984-03-25 00:00:00 | Tecnico    |
|              3 | Camila | 2015-12-25 00:00:00 | 1998-08-03 00:00:00 | Secretaria |
+----------------+--------+---------------------+---------------------+------------+
3 rows in set (0.01 sec)

mysql>
```

## Base de Datos corporativa en RDS

¡Hola! Como primer proyecto para este curso vas a poner en práctica tus conocimientos para desplegar, conectar, consultar y recuperar una base de datos en RDS.

Eres el arquitecto de soluciones de una empresa y el CEO te ha pedido que despliegues una base de datos que contenga información de los trabajadores que ingresaron durante la primer semana del mes, la información es la siguiente:

Tabla # 1 - Trabajadores.

![Tabla de trabajadores](images/Tabladetrabajadores.png)

Captura de pantalla 2018-11-21 a la(s) 13.44.14.png
- Despliega la Base de datos RDS (MySQL) y conéctate a través de MySQL Workbench.
- Crea una tabla de trabajadores con los campos ID, Nombre, Fecha de Ingreso, Fecha de Nacimiento y Cargo.
Ingresa los datos mostrados en la tabla # 1 - Trabajadores.
- Ahora conéctalos a la base de datos a través de una instancia EC2 usando la CLI y observa la tabla que creaste gráficamente.
- Luego de haber creado la tabla, ingresó un empleado:

Juan Camilo Rodriguez.
Fecha de Ingreso → 10/10/2018
Fecha de Nacimiento → 25/08/1991
Cargo → Software Engineer Senior
Ingresar el registro del nuevo empleado.

Ahora quieres probar las funcionalidades de Backup de la base de datos y para eso, vas a restaurar la base de datos al momento anterior al cual agregaste el último ingreso (Juan Camilo Rodriguez).

- Restaura la base de datos al momento anterior al ingreso del último usuario.
- Consulta la tabla trabajadores y verifica su estado.
- Verifica la tabla y evidencia que contenga solo los 5 registros iniciales.
Por último, envía un diagrama de arquitectura al CIO de la Base de Datos en Alta Disponibilidad y con optimización de performance.

No olvides compartir tus resultados, desafíos y aciertos en el panel de discusiones.

## Estrategias de backup

Las estrategias de **backup** en bases de datos y sistemas críticos dependen del nivel de disponibilidad y recuperación que necesites. Aquí te dejo algunas de las más usadas en AWS y en general: 

### 🔹 **1. Backup Completo (Full Backup)**
✅ **Descripción:** Se realiza una copia completa de todos los datos.  
✅ **Ventajas:** Fácil de restaurar, proporciona una imagen exacta del sistema.  
✅ **Desventajas:** Consume mucho tiempo y espacio en almacenamiento.  
✅ **Ejemplo en AWS:**  
- **Amazon RDS Snapshots** para bases de datos.
- **Amazon S3 Glacier** para almacenamiento a largo plazo.

📌 **SQL Backup Manual:**  
```sql
BACKUP DATABASE mi_base TO DISK = '/backups/mi_base_full.bak';
```

### 🔹 **2. Backup Incremental**
✅ **Descripción:** Solo copia los datos que han cambiado desde el último backup.  
✅ **Ventajas:** Menos almacenamiento y más rápido que un backup completo.  
✅ **Desventajas:** Restaurar los datos requiere combinar el backup completo y todos los incrementales.  
✅ **Ejemplo en AWS:**  
- **Amazon S3 Versioning** permite almacenar múltiples versiones de un mismo archivo.

📌 **SQL Backup Incremental:**  
```sql
BACKUP DATABASE mi_base TO DISK = '/backups/mi_base_inc.bak' WITH DIFFERENTIAL;
```

### 🔹 **3. Backup Diferencial**
✅ **Descripción:** Guarda todos los cambios desde el último backup completo.  
✅ **Ventajas:** Más rápido que el backup completo y más sencillo que el incremental.  
✅ **Desventajas:** Requiere un backup completo previo para restaurar.  
✅ **Ejemplo en AWS:**  
- **RDS Automated Backups** con retención personalizada.

📌 **SQL Backup Diferencial:**  
```sql
BACKUP DATABASE mi_base TO DISK = '/backups/mi_base_diff.bak' WITH DIFFERENTIAL;
```

### 🔹 **4. Backup en Tiempo Real (Continuous Backup)**
✅ **Descripción:** Permite la recuperación punto en el tiempo (PITR).  
✅ **Ventajas:** Recuperación precisa de datos sin pérdidas.  
✅ **Desventajas:** Alto consumo de almacenamiento y requiere más configuración.  
✅ **Ejemplo en AWS:**  
- **AWS Backup** con recuperación punto en el tiempo.
- **Aurora Backtrack** permite revertir cambios en Amazon Aurora.

📌 **SQL Backup con Logs de Transacción (PITR):**  
```sql
BACKUP LOG mi_base TO DISK = '/backups/mi_base_log.trn';
```

### 🔹 **5. Backup en la Nube**
✅ **Descripción:** Se almacena en servicios como **AWS S3, Google Drive, Azure Blob Storage, etc.**  
✅ **Ventajas:** Seguridad, escalabilidad y redundancia.  
✅ **Desventajas:** Puede generar costos adicionales por almacenamiento y transferencia de datos.  
✅ **Ejemplo en AWS:**  
- **AWS Backup** permite centralizar backups de RDS, EBS, S3 y DynamoDB.  

📌 **Comando AWS CLI para copiar un backup a S3:**  
```sh
aws s3 cp /backups/mi_base_full.bak s3://mi-bucket-backups/
```

### 🔹 **Recomendaciones Generales**
✅ **Automatiza los backups** con herramientas como **AWS Backup** o **cron jobs**.  
✅ **Usa múltiples estrategias**: combinación de backup completo + incremental o diferencial.  
✅ **Prueba la restauración periódicamente** para asegurar que los backups funcionan.  
✅ **Usa almacenamiento seguro y cifrado** para proteger los datos.

## Demo estrategias de backup

Aquí tienes una demostración de estrategias de backup en **SQL Server** y su integración con **AWS S3** para almacenamiento en la nube.  

### 🔹 **Paso 1: Crear un Backup Completo (Full Backup)**
📌 **Este backup contiene toda la base de datos.**  
```sql
BACKUP DATABASE mi_base 
TO DISK = 'C:\backups\mi_base_full.bak' 
WITH FORMAT, INIT;
```
📝 **Opción AWS:** Puedes almacenar este backup en **Amazon S3**.  

📌 **Copiar el backup a S3 con AWS CLI:**  
```sh
aws s3 cp C:\backups\mi_base_full.bak s3://mi-bucket-backups/
```

### 🔹 **Paso 2: Crear un Backup Diferencial**
📌 **Guarda solo los cambios desde el último backup completo.**  
```sql
BACKUP DATABASE mi_base 
TO DISK = 'C:\backups\mi_base_diff.bak' 
WITH DIFFERENTIAL;
```
✅ **Menos espacio y tiempo que un full backup.**

### 🔹 **Paso 3: Backup Incremental con Logs de Transacciones**
📌 **Registra cada cambio en la base de datos para restauración punto en el tiempo.**  
```sql
BACKUP LOG mi_base 
TO DISK = 'C:\backups\mi_base_log.trn';
```
✅ **Ideal para bases de datos críticas que requieren restauración precisa.** 

### 🔹 **Paso 4: Restauración desde un Backup**
### 📌 **Restaurar desde un Backup Completo**
```sql
RESTORE DATABASE mi_base 
FROM DISK = 'C:\backups\mi_base_full.bak' 
WITH NORECOVERY;
```

### 📌 **Restaurar un Backup Diferencial**
```sql
RESTORE DATABASE mi_base 
FROM DISK = 'C:\backups\mi_base_diff.bak' 
WITH NORECOVERY;
```

### 📌 **Restaurar desde un Backup de Logs**
```sql
RESTORE LOG mi_base 
FROM DISK = 'C:\backups\mi_base_log.trn' 
WITH RECOVERY;
```
✅ **Restaura todos los cambios desde el último backup diferencial o completo.**

### 🔹 **Paso 5: Automatización con un Job de SQL Server**
📌 **Automatiza backups con SQL Server Agent**  
```sql
USE msdb;
EXEC sp_add_job @job_name = 'Backup Diario';
EXEC sp_add_jobstep @job_name = 'Backup Diario',
    @step_name = 'Backup Completo',
    @command = 'BACKUP DATABASE mi_base TO DISK = ''C:\backups\mi_base_full.bak''',
    @database_name = 'mi_base';
```

### 🔹 **Paso 6: Backup en la Nube (AWS S3 o Glacier)**
📌 **Guardar backups en S3 para retención a largo plazo.**  
```sh
aws s3 cp C:\backups\mi_base_full.bak s3://mi-bucket-backups/
```
📌 **Para archivos históricos, mover a Glacier (almacenamiento frío)**  
```sh
aws s3 mv s3://mi-bucket-backups/mi_base_full.bak s3://mi-bucket-glacier/ --storage-class GLACIER
```

### 🔹 **Conclusión**
✅ **Full Backup:** Mejor para restauraciones completas.  
✅ **Diferencial:** Menos almacenamiento, rápida recuperación.  
✅ **Incremental (Logs):** Permite recuperación punto en el tiempo.  
✅ **Automatización:** Uso de **Jobs en SQL Server** o **AWS Backup**.  
✅ **Almacenamiento en AWS:** **S3, Glacier o RDS Snapshots**.  

### Resumen

### ¿Cómo restaurar una base de datos en Amazon RDS?

Restaurar una base de datos en Amazon RDS es una tarea crítica que requiere atención a los detalles y comprensión de las opciones disponibles. En esta sección, vamos a desglosar cómo proceder con la restauración y las diferentes opciones que ofrece RDS para realizar restauraciones automáticas de bases de datos.

### ¿Dónde encontrar las opciones de restauración?

Para comenzar, ingresa a la consola de Amazon RDS. Una vez dentro, ubica tu base de datos ya creada y haz clic en "Modify". Aquí encontrarás varias opciones que te permiten ajustar configuraciones hechas durante la creación inicial:

- Modificar el período de retención, esencial para auditorías o cambios en requerimientos.
- Cambiar características de la infraestructura, como tipos de instancias o configuraciones de almacenamiento.

### ¿Cómo restaurar a un punto en el tiempo?

Un método clave es "Restore to point in time", que te permite regresar tu base de datos a un momento específico. Esto se puede lograr seleccionando desde la fecha y hora exacta en que deseas restaurar:

1. **Último punto posible**: Se muestran el año, mes, día y hora.
2. **Opción personalizada**: Para restauraciones más específicas, elige "Custom" y selecciona días, horas, minutos y segundos de manera precisa.

### ¿Qué configuraciones se pueden ajustar durante una restauración?

Las restauraciones no son meramente una vuelta atrás en el tiempo; también ofrecen la oportunidad de ajustar e incluso mejorar configuraciones:

- **DB Engine**: Asegúrate de que el motor de base de datos sea el correcto (e.g., MySQL).
- **Tipo de instancia**: Cambia el tipo de instancia si es necesario.
- **Infraestructura adicional**: Decide si deseas que la instancia sea multi-AZ y modifica el almacenamiento.

Estas opciones brindan flexibilidad y control durante el proceso de restauración, permitiendo una personalización detallada según las necesidades del negocio.

### ¿Qué prácticas son recomendables al manejar el período de retención?

El periodo de retención es crucial, especialmente para aplicaciones críticas. Aunque por defecto es de siete días, se recomienda extenderlo hasta 35 días o más para garantizar la posibilidad de restauración en entornos productivos con gran cantidad de información. Además, puedes complementar esta configuración con snapshots manuales para asegurar mayor integridad de datos.

### ¿Qué otras funcionalidades pueden activarse durante el proceso?

Durante el proceso de restauración, RDS permite activar funcionalidades extras que pueden ser útiles:

- **Autenticación IAM**: Mejora la seguridad de acceso.
- **Logs y mantenimiento**: Configura ventanas de mantenimiento y registros para un control más detallado.
- **Subredes y accesibilidad**: Decide sobre qué VPC restaurar y si la base será accesible públicamente.

Estas funcionalidades extra no solo te permiten restaurar una base de datos sino optimizar su configuración para mejorar su rendimiento y seguridad.

### ¿Cómo RDS maneja los backups de forma integral?

RDS provee un enfoque integral para la gestión de backups, ofreciendo tanto copias automáticas como la posibilidad de crear y gestionar snapshots manualmente. Esto asegura que, independientemente de la complejidad del entorno, siempre exista una estrategia que mantenga la data segura y accesible.

Para concluir, el dominio de estas herramientas de restauración en RDS no solo asegura la recuperación efectiva de datos, sino que también incrementa el rendimiento y seguridad general de tus bases de datos. Cada ajuste realizado puede hacer una gran diferencia en el manejo cotidiano de la información crítica.

## Estrategias de performance en RDS

Amazon RDS ofrece varias estrategias para mejorar el rendimiento de la base de datos. Aquí están las más importantes: 

### 🔹 **1. Elección del Motor de Base de Datos**
Amazon RDS soporta motores como **MySQL, PostgreSQL, SQL Server, MariaDB, Oracle y Aurora**.  
✅ **Aurora:** Ofrece mejor rendimiento y escalabilidad con hasta 15 réplicas de lectura.  
✅ **MySQL/PostgreSQL:** Usar las versiones más recientes optimizadas para AWS.

### 🔹 **2. Tamaño y Tipo de Instancia**
📌 **Elige la instancia adecuada según la carga de trabajo:**  
- **T3/M5** → Para bases de datos pequeñas a medianas.  
- **R5/X** → Para cargas intensivas de memoria.  
- **I3/D** → Para bases de datos con alto uso de disco.  
- **Burstable (T3)** → Ideal para cargas ligeras.  

📌 **Ejemplo: Cambiar tipo de instancia en AWS CLI**  
```sh
aws rds modify-db-instance --db-instance-identifier mi-base --db-instance-class db.m5.large
```

### 🔹 **3. Uso de Almacenamiento SSD (IOPS)**
📌 **Tipos de almacenamiento:**  
- **General Purpose SSD (gp3/gp2):** Para bases de datos pequeñas o medianas.  
- **Provisioned IOPS (io1/io2):** Para bases de datos de alto rendimiento.  

📌 **Ejemplo: Cambiar almacenamiento en AWS CLI**  
```sh
aws rds modify-db-instance --db-instance-identifier mi-base --allocated-storage 100 --storage-type io1 --iops 5000
```

### 🔹 **4. Réplicas de Lectura**
📌 **Distribuye la carga de lectura en múltiples réplicas.**  
✅ **Aurora:** Soporta hasta **15 réplicas de lectura**.  
✅ **MySQL/PostgreSQL:** Soporta réplicas con latencia mínima.  

📌 **Ejemplo: Crear una réplica de lectura en AWS CLI**  
```sh
aws rds create-db-instance-read-replica --db-instance-identifier mi-replica --source-db-instance-identifier mi-base
```

### 🔹 **5. Caché de Consultas**
📌 **Habilita caché en motores compatibles:**  
✅ **PostgreSQL:** Usa **pg_stat_statements** y **pgtune** para optimización.  
✅ **MySQL:** Activa **query_cache_size** (solo en versiones más antiguas).  
✅ **Aurora:** Usa **Aurora Query Cache** para mejorar performance.  

📌 **Ejemplo: Configurar caché en PostgreSQL**  
```sql
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET work_mem = '64MB';
```

### 🔹 **6. Auto Scaling de Capacidad**
📌 **RDS soporta escalado automático para bases de datos Aurora.**  
✅ **Horizontal:** Agregar más réplicas de lectura.  
✅ **Vertical:** Aumentar CPU/memoria automáticamente.  

📌 **Ejemplo: Habilitar Auto Scaling en Aurora**
```sh
aws rds modify-db-cluster --db-cluster-identifier mi-cluster --scaling-configuration MinCapacity=2,MaxCapacity=8
```

### 🔹 **7. Optimización de Índices**
📌 **Usar EXPLAIN ANALYZE en PostgreSQL/MySQL para identificar consultas lentas.**  
📌 **Añadir índices en columnas de búsqueda frecuente.**  

📌 **Ejemplo: Crear un índice en MySQL**  
```sql
CREATE INDEX idx_usuarios_email ON usuarios(email);
```

### 🔹 **8. Monitoreo y Ajustes con CloudWatch**
📌 **Monitorear métricas clave con AWS CloudWatch:**  
✅ **CPUUtilization** → Uso de CPU.  
✅ **DatabaseConnections** → Número de conexiones activas.  
✅ **ReadIOPS/WriteIOPS** → Operaciones de lectura/escritura.  

📌 **Ejemplo: Obtener métricas en AWS CLI**
```sh
aws cloudwatch get-metric-statistics --metric-name CPUUtilization --namespace AWS/RDS --statistics Average --period 300
```

### 🔹 **9. Uso de AWS ElastiCache**
📌 **Reduce carga en RDS usando caché en memoria (Redis/Memcached).**  
✅ **Redis** → Soporta persistencia y clustering.  
✅ **Memcached** → Alta velocidad, ideal para caché simple.  

📌 **Ejemplo: Conectar aplicación a Redis**  
```python
import redis
cache = redis.Redis(host='mi-redis.xxxx.cache.amazonaws.com', port=6379)
cache.set('usuario:1', 'Mario')
print(cache.get('usuario:1'))
```

### 🔹 **10. Uso de Particionamiento y Sharding**
📌 **Para bases de datos grandes, divide datos en múltiples nodos:**  
✅ **Sharding:** Divide los datos en múltiples bases de datos.  
✅ **Partitioning:** Divide los datos en segmentos dentro de una tabla.  

📌 **Ejemplo: Crear partición en PostgreSQL**  
```sql
CREATE TABLE ventas_2025 PARTITION OF ventas
    FOR VALUES FROM ('2025-01-01') TO ('2025-12-31');
```

### 🔹 **Conclusión**
✅ **Elige la instancia correcta según carga de trabajo.**  
✅ **Usa almacenamiento IOPS para bases de datos críticas.**  
✅ **Implementa réplicas de lectura para escalar lectura.**  
✅ **Aprovecha caché con ElastiCache o Query Cache.**  
✅ **Usa CloudWatch para monitoreo y auto scaling.**  

**Resumen**

En esta clase vamos a aprender cómo identificar el rendimiento de nuestra base de datos, estrategias para mejorar su rendimiento actual y todas las opciones de performance de AWS.

A nivel de monitoreo, AWS nos provee un servicio llamado **CloudWatch** que nos permite visualizar los niveles de lectura, escritura, CPU, disco y memoria de la instancia dónde corre nuestra base de datos, también podemos analizar las métricas de conexiones para determinar la carga y la concurrencia de nuestras instancias.

La primer estrategia para mejorar el performance son las **replicas de lectura**, copias asíncronas de nuestra base de datos principal con un nuevo endpoint que vamos a utilizar solo en tareas de lectura, así obtenemos mucha más disponibilidad para tareas de escritura en nuestra base de datos principal. Recuerda que este servicio no esta disponible para los motores de Oracle y SQL Server.

También podemos mejorar el storage de nuestra base de datos utilizando **provisioned iops** para soportar altas operaciones de entrada y salida sobre la base de datos, principalmente para transacciones OLTP (*OnLine Transaction Processing*).

Existen otras alternativas como las bases de datos en memoria (*ElastiCache, por ejemplo*). Estas opciones resultan muy útiles para guardar la información más consultada en cache, así aliviamos un poco la carga de nuestra base de datos principal. Si estamos muy saturados y agotamos todas las opciones para mejorar el performance, la recomendación es dividir nuestra base de datos en otras más pequeñas.

## Despliegues Multi AZ

Amazon RDS permite configuraciones de alta disponibilidad mediante **Multi-AZ**, asegurando redundancia y failover automático.

### 🎯 **¿Qué es Multi-AZ en RDS?**  
Multi-AZ crea una **réplica en espera (standby)** en otra **Zona de Disponibilidad (AZ)** dentro de la misma región.  

📌 **Características principales:**  
✅ **Alta disponibilidad:** Failover automático en caso de fallas.  
✅ **Datos replicados sincrónicamente.**  
✅ **Ideal para cargas críticas de producción.**  
✅ **Compatible con MySQL, PostgreSQL, MariaDB, SQL Server y Oracle.**  
✅ **Aurora usa un modelo distinto con múltiples réplicas en diferentes AZs.**

### ⚡ **Ventajas del Despliegue Multi-AZ**  
✔ **Failover automático** en caso de caída del nodo primario.  
✔ **Cero pérdida de datos** gracias a la replicación sincrónica.  
✔ **Mantenimiento sin interrupciones**, ya que las actualizaciones ocurren en la réplica antes de aplicarse en el nodo principal.  
✔ **Mejor recuperación ante desastres** al estar los datos distribuidos en múltiples AZs.

### 🔹 **Cómo Configurar Multi-AZ en RDS**  

### **1️⃣ Crear una Instancia RDS con Multi-AZ desde la Consola AWS**  
1️⃣ Ir a **Amazon RDS** → **Crear Base de Datos**.  
2️⃣ Elegir **Motor de base de datos** (MySQL, PostgreSQL, etc.).  
3️⃣ En **Alta Disponibilidad y Durabilidad**, activar **Despliegue Multi-AZ**.  
4️⃣ Configurar almacenamiento y opciones de seguridad.  
5️⃣ Crear la base de datos.  

### **2️⃣ Configurar Multi-AZ con AWS CLI**  
📌 **Crear una base de datos Multi-AZ:**  
```sh
aws rds create-db-instance \
    --db-instance-identifier mi-db \
    --allocated-storage 20 \
    --db-instance-class db.m5.large \
    --engine mysql \
    --master-username admin \
    --master-user-password MiClaveSegura \
    --multi-az
```

📌 **Modificar una instancia existente a Multi-AZ:**  
```sh
aws rds modify-db-instance \
    --db-instance-identifier mi-db \
    --multi-az \
    --apply-immediately
```

### 🔥 **Cómo Funciona el Failover en Multi-AZ**  
En caso de falla en la instancia principal:  
1️⃣ AWS detecta automáticamente el problema.  
2️⃣ Se redirige el tráfico a la réplica en espera.  
3️⃣ La nueva instancia primaria toma el control.  
4️⃣ AWS crea una nueva réplica en espera automáticamente.  

🕒 **Tiempo de failover:** Generalmente **60-120 segundos**.

### 📌 **Diferencias Entre Multi-AZ y Réplicas de Lectura**  
| Característica       | Multi-AZ              | Réplicas de Lectura |
|----------------------|----------------------|----------------------|
| **Propósito**       | Alta disponibilidad   | Escalabilidad de lectura |
| **Tipo de Replicación** | Sincrónica          | Asincrónica |
| **Failover Automático** | ✅ Sí | ❌ No |
| **Uso de Endpoint**  | Un solo endpoint | Diferentes endpoints |
| **Costo**           | Mayor (por réplica en espera) | Menor (solo lectura) |

📌 **Multi-AZ ≠ Escalabilidad** → No mejora el rendimiento de lectura, solo la disponibilidad.  
📌 **Para escalabilidad**, usar **Réplicas de Lectura**.

### 🔚 **Conclusión**  
✔ **Multi-AZ es ideal para cargas de producción críticas**.  
✔ **Failover automático sin intervención manual**.  
✔ **Protege contra fallos de hardware o zonas de disponibilidad.**  
✔ **Compatible con varias bases de datos en AWS.**  

**Resumen**

El servicio de Multi AZ nos permite aumentar la disponibilidad de nuestro servicio realizando despliegues de nuestra base de datos en diferentes zonas. Cuando nuestra base de datos principal tenga problemas de disponibilidad, AWS automáticamente conectará nuestra aplicación con la base de datos replica en la segunda zona de disponibilidad. Recuerda que el precio de este servicio es equivalente a tener 2 bases de datos.

El desafío de esta clase es identificar un caso de uso en tu empresa, universidad o algún proyecto personal dónde podemos utilizar RDS, recuerda explicar cuál es la funcionalidad qué más llama tu atención y por qué.

## Estrategias de migración a RDS

Migrar bases de datos a Amazon RDS puede mejorar la escalabilidad, disponibilidad y mantenimiento de tu infraestructura. Sin embargo, es fundamental elegir la mejor estrategia según el caso de uso.

### 🎯 **Principales Estrategias de Migración a RDS**  

### 1️⃣ **Migración con AWS Database Migration Service (AWS DMS)**  
✅ **Ideal para migraciones con mínimo downtime.**  
✅ Soporta migración de bases de datos **homogéneas y heterogéneas**.  
✅ Permite replicación en tiempo real.  

📌 **Ejemplo de migración homogénea (misma tecnología):**  
- MySQL → RDS MySQL  
- PostgreSQL → RDS PostgreSQL  

📌 **Ejemplo de migración heterogénea (diferente tecnología):**  
- Oracle → RDS PostgreSQL  
- SQL Server → RDS MySQL  

👉 **Pasos con AWS DMS:**  
1️⃣ Crear una instancia de AWS DMS.  
2️⃣ Configurar los endpoints de origen y destino.  
3️⃣ Crear y ejecutar la tarea de migración.  
4️⃣ Validar la integridad de los datos migrados.

### 2️⃣ **Migración Manual con Exportación e Importación de Datos**  
✅ **Recomendada para bases de datos pequeñas o sin requisitos de alta disponibilidad.**  

📌 **Ejemplo para MySQL:**  
1️⃣ **Exportar los datos desde la base de datos de origen:**  
```sh
mysqldump -u usuario -p --all-databases > backup.sql
```
2️⃣ **Subir el archivo a RDS:**  
```sh
mysql -h mi-db.rds.amazonaws.com -u admin -p < backup.sql
```

📌 **Ejemplo para PostgreSQL:**  
1️⃣ **Exportar datos:**  
```sh
pg_dump -U usuario -h origen -d mi_db > backup.sql
```
2️⃣ **Importar a RDS:**  
```sh
psql -h mi-db.rds.amazonaws.com -U admin -d mi_db < backup.sql
```

### 3️⃣ **Replicación Binlog para Migraciones en Vivo (MySQL y MariaDB)**  
✅ **Útil para migraciones sin interrupciones prolongadas.**  
✅ **Replica cambios en tiempo real.**  

📌 **Pasos:**  
1️⃣ Habilitar el **binlog** en la base de datos de origen.  
2️⃣ Configurar **replicación en RDS** con un usuario replicador.  
3️⃣ Mantener sincronización hasta hacer el switch final.

### 4️⃣ **Migración con Amazon Aurora (Backtrack & Cloning)**  
✅ **Recomendada si se migra de MySQL o PostgreSQL.**  
✅ Aurora permite **clonar bases de datos** y **restaurar en segundos**.  

📌 **Pasos:**  
1️⃣ Crear un snapshot de la base de datos de origen.  
2️⃣ Restaurar el snapshot en una instancia de Amazon Aurora.  
3️⃣ Conectar la aplicación a la nueva base de datos.

### 📌 **Comparación de Estrategias**  

| Estrategia | Downtime | Complejidad | Escenario Ideal |
|------------|---------|-------------|----------------|
| **AWS DMS** | Bajo | Medio | Migraciones en vivo |
| **Exportar/Importar** | Alto | Bajo | Bases pequeñas |
| **Binlog Replication** | Bajo | Alto | MySQL/MariaDB con replicación |
| **Aurora Cloning** | Bajo | Medio | Migraciones a Aurora |

### 🔥 **Recomendaciones Finales**  
✔ **Evaluar compatibilidad** antes de migrar.  
✔ **Realizar pruebas previas en un entorno de staging.**  
✔ **Monitorear la migración para evitar pérdidas de datos.**  
✔ **Optimizar índices y configuraciones post-migración.**

**Resumen**

DMS (*Database Migration Service*) es un servicio de AWS que nos permite migrar nuestras bases de datos con otros motores al servicio de RDS u otros servicios de bases de datos en AWS.

Este servicio tiene las siguientes características:

- Podemos realizar migraciones de bases de datos on premise o en la nube a los servicios de bases de datos en AWS sin afectar el downtime de la base de datos que vamos a migrar.
- La carga de trabajo durante las migraciones es adaptable.
- Solo pagamos por los recursos que utilizamos en la migración.
- AWS administra la infraestructura necesaria para el trabajo de la migración, Hardware, Software, parches, etc.
- Conmutación por error automática, si AWS detecta un error en el proceso automáticamente creará una nueva instancia para remplazar la anterior, así el proceso de replicación no se ve afectado por estos problemas.
- Los datos en reposo están cifrados con KMS (Key Management Service) y la migración utiliza el protocolo de seguridad SSL.

## Migraciones homogéneas y heterogéneas

Cuando migramos bases de datos a Amazon RDS u otros sistemas, podemos categorizar la migración en **homogénea** o **heterogénea**, dependiendo de si el motor de la base de datos cambia o no.

### ✅ **1. Migración Homogénea**  
📌 **Se mantiene el mismo motor de base de datos.**  
📌 **Ideal cuando solo se cambia de infraestructura**, por ejemplo, de un servidor on-premise a la nube.  
📌 **Ejemplo:**  
- MySQL → RDS MySQL  
- PostgreSQL → RDS PostgreSQL  
- SQL Server → RDS SQL Server  

📌 **Métodos de migración homogénea:**  
- **AWS Database Migration Service (DMS)**  
- **Backup y Restore** (Ejemplo: `mysqldump` o `pg_dump`)  
- **Replicación Binlog** (para MySQL y MariaDB)  

👉 **Ventaja:** Fácil y rápida, ya que no hay necesidad de cambiar estructura ni código SQL.

### 🔄 **2. Migración Heterogénea**  
📌 **Se cambia el motor de base de datos.**  
📌 **Requiere conversión de esquema y datos.**  
📌 **Ejemplo:**  
- Oracle → RDS PostgreSQL  
- SQL Server → RDS MySQL  
- MySQL → RDS Aurora PostgreSQL  

📌 **Métodos de migración heterogénea:**  
1️⃣ **AWS Schema Conversion Tool (AWS SCT):** Convierte automáticamente estructuras incompatibles.  
2️⃣ **AWS DMS con transformación de datos:** Permite ajustar tipos de datos.  
3️⃣ **Exportación manual y transformación:** Se extraen datos, se adaptan y se importan en la nueva base de datos.  

👉 **Desafíos:**  
- **Conversión de tipos de datos**  
- **Adaptación de consultas SQL**  
- **Cambio en funciones y procedimientos almacenados** 

### 📊 **Comparación Rápida**  

| **Migración** | **Cambio de Motor** | **Dificultad** | **Ejemplo** |
|--------------|----------------|--------------|-----------|
| **Homogénea** | ❌ No | 🟢 Fácil | MySQL → RDS MySQL |
| **Heterogénea** | ✅ Sí | 🔴 Compleja | Oracle → RDS PostgreSQL |

### 🚀 **Conclusión**  
✔ Si el motor de BD **se mantiene**, una migración homogénea es más rápida y sencilla.  
✔ Si el motor **cambia**, se debe hacer una conversión de esquema y datos con herramientas como **AWS SCT + DMS**.  
✔ **Siempre se recomienda probar la migración en un entorno de prueba antes de ejecutarla en producción.** 

**Resumen**

Las migraciones homogéneas son migraciones donde la base de datos de origen y la de destino puede tener diferentes versiones del mismo motor, o son bases de datos compatibles entre sí (*MySQL y Aurora, por ejemplo*).

También podemos realizar migraciones heterogéneas, donde la base de datos de origen no es compatible con la de destino. Estas migraciones NO siempre son posibles, y antes de realizar la migración vamos a necesitar convertir el esquema de la base de datos con la herramienta AWS Schema Conversion Tool.