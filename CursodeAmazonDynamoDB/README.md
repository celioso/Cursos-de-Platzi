# Curso de Amazon DynamoDB

## Bienvenida a DynamoDB

¡Claro! Aquí tienes una introducción a **DynamoDB** para que te familiarices con este servicio:

---

## Bienvenida a **Amazon DynamoDB**

### ¿Qué es DynamoDB?

Amazon **DynamoDB** es una base de datos NoSQL completamente gestionada, rápida y flexible, diseñada para aplicaciones que requieren baja latencia a cualquier escala.

### Características principales

* **Totalmente administrada:** No tienes que preocuparte por hardware, configuración o mantenimiento.
* **Alta escalabilidad:** Soporta millones de solicitudes por segundo y ajusta automáticamente la capacidad según la demanda.
* **Baja latencia:** Respuestas en milisegundos, ideal para aplicaciones web, móviles, juegos, IoT, etc.
* **Modelo NoSQL:** Utiliza tablas con pares clave-valor o documentos (JSON).
* **Alta disponibilidad:** Replicación automática entre múltiples zonas de disponibilidad (AZs).
* **Seguridad integrada:** Control de acceso mediante IAM, cifrado en reposo y en tránsito.

### Casos de uso comunes

* Aplicaciones móviles y web con alta demanda.
* Juegos en línea con millones de usuarios.
* Sistemas de comercio electrónico.
* IoT y análisis en tiempo real.
* Aplicaciones con datos semi estructurados o no estructurados.

### ¿Cómo funciona?

* **Tablas:** Son los contenedores de datos.
* **Ítems:** Registros dentro de las tablas, similares a filas.
* **Atributos:** Campos dentro de cada ítem.
* **Clave primaria:** Identifica unívocamente cada ítem (puede ser simple o compuesta).
* **Índices secundarios:** Permiten consultas eficientes usando otros atributos.

## ¿Qué es una base de datos?

Una **base de datos** es un sistema organizado para **almacenar, gestionar y recuperar datos** de forma eficiente.

Es como un archivo digital donde guardas información estructurada para que puedas acceder a ella, modificarla o analizarla rápidamente cuando la necesites.

### Características principales:

* **Organización:** Los datos se guardan en tablas, registros o documentos para facilitar su uso.
* **Acceso rápido:** Permite consultar datos específicos sin revisar todo el contenido.
* **Actualización:** Se pueden agregar, modificar o eliminar datos fácilmente.
* **Seguridad:** Controla quién puede ver o cambiar la información.

### Ejemplo cotidiano:

Imagina una lista organizada de contactos en tu teléfono:

* Cada contacto es un registro.
* Cada dato como nombre, teléfono o email es un campo.

Una base de datos hace lo mismo pero a mucha mayor escala y con herramientas para manejar millones de registros.

## Amazon DynamoDB

**Amazon DynamoDB** es un servicio de base de datos **NoSQL** totalmente administrado por AWS, diseñado para ofrecer un rendimiento rápido y predecible con una escalabilidad sencilla.

### Características principales de DynamoDB

* **NoSQL:** No utiliza tablas relacionales tradicionales, sino que maneja datos en formato de clave-valor o documentos.
* **Escalabilidad automática:** Ajusta la capacidad de lectura y escritura según la demanda sin necesidad de intervención manual.
* **Baja latencia:** Ofrece acceso a los datos con latencias de milisegundos, ideal para aplicaciones en tiempo real.
* **Totalmente gestionado:** AWS se encarga del hardware, configuración, parcheo y replicación.
* **Alta disponibilidad y durabilidad:** Replicación automática de datos entre varias zonas de disponibilidad (AZ) para garantizar resiliencia.
* **Modelo flexible:** Permite almacenar datos estructurados, semi-estructurados y no estructurados.
* **Seguridad:** Integración con AWS IAM para control de acceso, cifrado en reposo y en tránsito.

### Conceptos básicos de DynamoDB

* **Tabla:** Contenedor para almacenar los datos.
* **Ítem:** Cada registro dentro de una tabla (similar a una fila).
* **Atributo:** Campos dentro de cada ítem (como columnas en una tabla relacional).
* **Clave primaria:** Identifica un ítem de forma única. Puede ser simple (clave de partición) o compuesta (clave de partición + clave de ordenación).
* **Índices secundarios:** Permiten consultas adicionales basadas en atributos distintos de la clave primaria.

### ¿Para qué se usa DynamoDB?

* Aplicaciones web y móviles con alta demanda y necesidad de baja latencia.
* Juegos en línea.
* IoT y aplicaciones de análisis en tiempo real.
* Comercio electrónico y gestión de sesiones.
* Cualquier aplicación que necesite escalabilidad sin complicaciones de gestión de base de datos.

**Lecturas recomendadas**

[https://twitter.com/tpschmidt_/status/1493958661391233026/photo/1](https://twitter.com/tpschmidt_/status/1493958661391233026/photo/1)

[Fast NoSQL Key-Value Database – Amazon DynamoDB – Amazon Web Services](https://aws.amazon.com/dynamodb/)

## Crear una tabla en DynamoDB: primary keys

Para **crear una tabla en Amazon DynamoDB**, es esencial definir correctamente las **claves primarias (primary keys)**, ya que estas determinan cómo se almacenan y acceden los datos. Aquí te explico paso a paso cómo hacerlo, con ejemplos:

### 🔑 Tipos de claves primarias en DynamoDB

### 1. **Clave primaria simple (Partition Key)**

* Se compone **solo de una clave de partición**.
* Todos los elementos deben tener un valor **único** para esa clave.

📌 Ejemplo:

```json
"UserId": "123"
```

* Ejemplo de tabla con clave primaria simple:

  ```bash
  aws dynamodb create-table \
    --table-name Usuarios \
    --attribute-definitions AttributeName=UserId,AttributeType=S \
    --key-schema AttributeName=UserId,KeyType=HASH \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
  ```

### 2. **Clave primaria compuesta (Partition Key + Sort Key)**

* Usa dos atributos:

  * **Partition Key (HASH)**: para distribuir datos entre particiones.
  * **Sort Key (RANGE)**: permite almacenar múltiples ítems con la misma Partition Key, ordenados por Sort Key.

📌 Ejemplo:

```json
"UserId": "123",
"Fecha": "2025-05-22"
```

* Ejemplo de tabla con clave primaria compuesta:

  ```bash
  aws dynamodb create-table \
    --table-name Pedidos \
    --attribute-definitions AttributeName=UserId,AttributeType=S AttributeName=Fecha,AttributeType=S \
    --key-schema AttributeName=UserId,KeyType=HASH AttributeName=Fecha,KeyType=RANGE \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
  ```

### 🛠️ Atributos de definición

| Tipo de atributo | Código |
| ---------------- | ------ |
| String           | `S`    |
| Number           | `N`    |
| Binary           | `B`    |

### 🧠 Recomendaciones

* Usa una clave de partición **uniformemente distribuida** (como un UUID) para evitar **hot partitions**.
* Si necesitas buscar por múltiples atributos, considera crear **índices secundarios (LSI / GSI)**.

## Hot partitions

### 🔥 ¿Qué son las *Hot Partitions* en DynamoDB?

En **Amazon DynamoDB**, una *hot partition* ocurre cuando **una partición física específica recibe una carga desproporcionada de tráfico**, causando:

* **Cuellos de botella en el rendimiento**
* **Límites de lectura o escritura excedidos**
* **Latencia alta o errores de throughput**

### 🧱 ¿Cómo funciona el almacenamiento en particiones?

DynamoDB divide internamente las tablas en **particiones físicas**, y cada partición:

* Tiene capacidad limitada (\~3,000 lecturas o 1,000 escrituras por segundo).
* Se determina por el **valor de la Partition Key**.

📌 Si muchos ítems comparten el mismo valor de clave de partición, todos caen en la **misma partición física** → ahí nace una *hot partition*.

### 🧨 Ejemplo de *hot partition*

Supón una tabla con esta clave primaria simple:

```json
"UserId": "12345"
```

Y todos los usuarios que acceden a la app son `"12345"`...
❌ **Todos los accesos van a la misma partición** → esto crea una *hot partition*.

### ✅ Cómo evitar *Hot Partitions*

1. **Diseño de clave de partición más distribuido**

   * Usa un UUID o un valor hash distribuido uniformemente.
   * Si tienes una fecha o categoría muy popular, inclúyela con un sufijo aleatorio.

   📌 Ejemplo:

   ```json
   "PartitionKey": "login-2025-05-22-3"
   ```

2. **Usar claves compuestas (partition key + sort key)**

   * Permite organizar datos dentro de una clave de partición y facilita búsquedas ordenadas.

3. **Agregar entropía (sal)**

   * Añadir un valor aleatorio o hash como parte del partition key:

     ```json
     "UserId": "12345#4"
     ```

4. **Utilizar Global Secondary Indexes (GSI)**

   * Para consultar por otros atributos sin sobrecargar la clave primaria.

5. **On-demand capacity mode**

   * Deja que DynamoDB escale automáticamente, aunque no soluciona completamente *hot partitions* si el diseño es deficiente.

### 📌 Regla de oro

> "Diseña tu clave de partición como si millones de usuarios accedieran al mismo tiempo"

## Usuarios y permisos IAM en DynamoDB

En **Amazon DynamoDB**, el acceso a las tablas y operaciones está controlado mediante **IAM (Identity and Access Management)**. Esto permite definir **quién puede hacer qué sobre qué recursos**.

### 🔐 ¿Qué es IAM?

IAM permite:

* Crear **usuarios**, **grupos** y **roles**
* Asignarles **políticas** que otorgan permisos sobre servicios como DynamoDB

### 🔑 Permisos comunes en DynamoDB

| Acción IAM             | Descripción                               |
| ---------------------- | ----------------------------------------- |
| `dynamodb:GetItem`     | Leer un ítem por clave primaria           |
| `dynamodb:PutItem`     | Insertar o sobrescribir un ítem           |
| `dynamodb:UpdateItem`  | Actualizar uno o más atributos de un ítem |
| `dynamodb:DeleteItem`  | Eliminar un ítem                          |
| `dynamodb:Query`       | Consultar ítems usando una clave          |
| `dynamodb:Scan`        | Leer todos los ítems de la tabla          |
| `dynamodb:CreateTable` | Crear una tabla                           |
| `dynamodb:DeleteTable` | Eliminar una tabla                        |

### 🧪 Ejemplo: Política IAM para acceso de solo lectura

Esta política da permiso a un usuario o rol para hacer solo operaciones de lectura:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/NombreDeTuTabla"
    }
  ]
}
```

### 🧑‍💻 Crear usuario IAM y asignar permisos a DynamoDB

### 1. Ve a la consola de IAM: [https://console.aws.amazon.com/iam/](https://console.aws.amazon.com/iam/)

### 2. Crea un usuario

* Tipo de acceso: ✅ Programático o acceso a consola
* Asigna permisos:

  * Elige **"Adjuntar políticas existentes directamente"**
  * Puedes usar `AmazonDynamoDBFullAccess` (para pruebas)

### 3. Crear o adjuntar una política personalizada

* Para acceso más controlado, crea una política como el ejemplo anterior y asígnala al usuario/rol.

### 🔁 Buenas prácticas

✅ Usa **roles** si estás trabajando con:

* **Lambda**
* **EC2**
* **ECS/Fargate**

✅ Usa **usuarios IAM** con acceso programático (por clave) solo si realmente lo necesitas.

✅ Limita el **principio de menor privilegio**: da solo los permisos necesarios.

✅ Usa **condiciones** (como claves específicas, tags o IPs) para restringir aún más los accesos.

## Instalación de SDK y AWS CLI

Aquí tienes una guía paso a paso para **instalar el AWS CLI y el SDK de AWS** según tu entorno:

---

## 🧰 1. Instalación de AWS CLI (Command Line Interface)

La **AWS CLI** permite interactuar con AWS desde la terminal.

### 🔹 Windows

1. Descargar el instalador:
   [https://awscli.amazonaws.com/AWSCLIV2.msi](https://awscli.amazonaws.com/AWSCLIV2.msi)

2. Ejecuta el instalador y sigue las instrucciones.

3. Verifica instalación:

   ```bash
   aws --version
   ```

---

### 🔹 macOS

```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
aws --version
```

---

### 🔹 Linux

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

---

## 🔐 2. Configuración inicial del AWS CLI

Después de instalarlo, ejecuta:

```bash
aws configure
```

Te pedirá:

* AWS Access Key ID
* AWS Secret Access Key
* Región (por ejemplo: `us-east-1`)
* Formato de salida (ej: `json`)

---

## 🧑‍💻 3. Instalación del SDK (según lenguaje)

### ✅ Python (Boto3)

```bash
pip install boto3
```

Verifica:

```python
import boto3
print(boto3.__version__)
```

### ✅ JavaScript / Node.js (AWS SDK v3)

```bash
npm install @aws-sdk/client-dynamodb
```

O para todo el SDK:

```bash
npm install aws-sdk
```

### ✅ Java

Agrega al `pom.xml` de Maven:

```xml
<dependency>
  <groupId>software.amazon.awssdk</groupId>
  <artifactId>dynamodb</artifactId>
  <version>2.25.16</version>
</dependency>
```

### ✅ Go

```bash
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/aws/aws-sdk-go-v2/service/dynamodb
```

### Lecturas recomendadas

[Command Line Interface - AWS CLI - AWS](https://aws.amazon.com/cli/ "Command Line Interface - AWS CLI - AWS")

[GitHub - aws/aws-cli: Universal Command Line Interface for Amazon Web Services](https://github.com/aws/aws-cli "GitHub - aws/aws-cli: Universal Command Line Interface for Amazon Web Services")

[https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html")

[AWS SDK para Python](https://aws.amazon.com/es/sdk-for-python/ "AWS SDK para Python")

## Crear tablas y cargar datos en DynamoDB

Crear tablas y cargar datos en **Amazon DynamoDB** con **Python** y **Boto3** es un proceso sencillo. A continuación te muestro los pasos completos:

### ✅ 1. Requisitos Previos

Instala las dependencias necesarias (si no lo has hecho):

```bash
pip install boto3 awscli
```

Configura tus credenciales de AWS:

```bash
aws configure
import boto3

# Crear cliente
dynamodb = boto3.client('dynamodb', region_name='us-east-1')  # Cambia la región si es necesario

# Crear tabla
response = dynamodb.create_table(
    TableName='Usuarios',
    KeySchema=[
        {
            'AttributeName': 'usuario_id',
            'KeyType': 'HASH'  # Clave primaria
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'usuario_id',
            'AttributeType': 'S'  # Tipo S = String
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Tabla creada. Estado:", response['TableDescription']['TableStatus'])
```

### ✅ 3. Cargar Datos en DynamoDB

```python
import boto3

dynamodb = boto3.resource('dynamodb')
tabla = dynamodb.Table('Usuarios')

# Insertar ítems
usuarios = [
    {"usuario_id": "1", "nombre": "Ana", "edad": 28},
    {"usuario_id": "2", "nombre": "Luis", "edad": 34},
    {"usuario_id": "3", "nombre": "Carlos", "edad": 22}
]

for usuario in usuarios:
    tabla.put_item(Item=usuario)

print("Datos insertados correctamente.")
```

### ✅ 4. Verificar contenido (opcional)

```python
response = tabla.scan()
for item in response['Items']:
    print(item)
```

### 🧠 Notas importantes

* Puedes usar `boto3.client()` o `boto3.resource()`. Este último es más orientado a objetos.
* Si estás usando **DynamoDB local**, avísame para darte el código con `endpoint_url`.
* Puedes usar **atributos compuestos** (HASH + RANGE) si necesitas claves primarias compuestas.

**Comandos utilizados**

Ver documentación = `aws help` para filtrar las palabras que deseo ver como dynamodb `aws help | grep dynamodb` y pedir ayuda de solo ver dynamodb `aws dynamodb help`

ver las tablas creadas = `aws dynamodb list-tables`

crear una tabla Dinamodb = `aws dynamodb create-table --table-name DynamoDBPrueba --key-schema AttributeName=EstudianteId,KeyType=HASH --attribute-definitions AttributeName=EstudianteId,AttributeType=N --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5`

crear tabla Personajes = `aws dynamodb create-table --table-name Personajes --key-schema AttributeName=Id,KeyType=HASH --attribute-definitions AttributeName=Id,AttributeType=N --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5`

crear una tabla con wizard toca instalar aws-cli2 esta [GitHub - aws/aws-cli at v2](https://github.com/aws/aws-cli/tree/v2#installation)

`aws dynamodb wizard new-table`

cargar datos:

`aws dynamodb batch-write-item --request-items file://Personajes.json`

**Archivos de la clase**

[createtable.json](https://static.platzi.com/media/public/uploads/createtable_58340ff8-7bc8-415e-92f6-9db83db2ac5e.json)

[personajes.json](https://static.platzi.com/media/public/uploads/personajes_c2743b85-de05-42c8-88de-d0c2f0ed4a9b.json)

**Lecturas recomendadas**

[GitHub - aws/aws-cli at v2](https://github.com/aws/aws-cli/tree/v2#installation)

[createtable.json - Google Drive](https://drive.google.com/file/d/1Njoi_OiRQmjlQVIUlxiO9bz9h0BC0u92/view?usp=share_link)

[Personajes.json - Google Drive](https://drive.google.com/file/d/1pE2NCho-LHd3FkZmr5aOZa9sUvwVJhVV/view?usp=share_link)

## Consultar y eliminar tablas en DynamoDB

Aquí tienes cómo **consultar** y **eliminar** tablas en DynamoDB usando AWS CLI:

### 🔍 Consultar Tablas en DynamoDB

### 1. ✅ Listar todas las tablas

```bash
aws dynamodb list-tables
```

### 2. ✅ Obtener información detallada de una tabla específica

```bash
aws dynamodb describe-table --table-name NOMBRE_TABLA
```

Esto te muestra:

* Claves primarias
* Atributos
* Estado de la tabla
* Configuración de capacidad, etc.

### ❌ Eliminar una Tabla en DynamoDB

### Comando:

```bash
aws dynamodb delete-table --table-name NOMBRE_TABLA
```

🔒 **Precaución**: Este comando elimina la tabla **y todos sus datos** permanentemente. No se puede deshacer.

### 🧠 Ejemplo práctico

Supongamos que tienes una tabla llamada `DynamoDBPrueba`:

```bash
aws dynamodb describe-table --table-name DynamoDBPrueba
aws dynamodb delete-table --table-name DynamoDBPrueba
```

### 📌 Agrega región si es necesario:

Si tu tabla no está en la región por defecto, agrega el parámetro:

```bash
--region us-east-1  # o la región que estés usando
```

**Eliminar una tabla desde aws-cli** 
ver las tablas `aws dynamodb list-tables`
 eliminar tablas se usa `aws dynamodb delete-table --table-name DynamoDBPrueba`

 ## Índices locales

 ### 📘 Índices Locales Secundarios en DynamoDB (LSI – *Local Secondary Index*)

### 🔹 ¿Qué es un índice local secundario (LSI)?

Un **LSI** permite crear **índices alternativos** sobre los datos en una tabla **sin cambiar la clave de partición**. Se utiliza para hacer consultas más eficientes basadas en otros atributos además de la clave principal.

### 🔑 Diferencias clave:

* **LSI** usa **la misma clave de partición** que la tabla base.
* Te permite crear **una clave de ordenación (sort key)** alternativa.
* Solo puedes definir LSIs **cuando creas la tabla**.
* Hasta **5 LSIs por tabla**.
* **Comparte el almacenamiento** con la tabla principal.

### 📦 Ejemplo: Crear una tabla con un LSI

```bash
aws dynamodb create-table \
  --table-name Estudiantes \
  --attribute-definitions \
    AttributeName=EstudianteId,AttributeType=S \
    AttributeName=Curso,AttributeType=S \
    AttributeName=Nota,AttributeType=N \
  --key-schema \
    AttributeName=EstudianteId,KeyType=HASH \
    AttributeName=Curso,KeyType=RANGE \
  --local-secondary-indexes '[
    {
      "IndexName": "NotaIndex",
      "KeySchema": [
        { "AttributeName": "EstudianteId", "KeyType": "HASH" },
        { "AttributeName": "Nota", "KeyType": "RANGE" }
      ],
      "Projection": {
        "ProjectionType": "ALL"
      }
    }
  ]' \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

### 🔍 Consultar usando el índice

Una vez creado, puedes hacer una consulta usando el índice así:

```bash
aws dynamodb query \
  --table-name Estudiantes \
  --index-name NotaIndex \
  --key-condition-expression "EstudianteId = :id" \
  --expression-attribute-values '{":id":{"S":"123"}}'
```

### 🧠 Proyección en LSI

Con `ProjectionType` decides qué atributos incluir:

* `KEYS_ONLY`: Solo claves primarias e índice.
* `INCLUDE`: Añades atributos específicos.
* `ALL`: Todos los atributos de la tabla.

**TablaMusic.sh**

```bash
aws dynamodb create-table --table-name 'Music' --attribute-definitions AttributeName=Artist,AttributeType=S AttributeName=SongTitle,AttributeType=S AttributeName=AlbumTitle,AttributeType=S --key-schema AttributeName=Artist,KeyType=HASH AttributeName=SongTitle,KeyType=RANGE --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 --local-secondary-indexes file://lsi_config.json
```

**lsi_config.json**

```json
[
  {
    "IndexName": "AlbumTitleIndex",
    "KeySchema": [
      { "AttributeName": "Artist", "KeyType": "HASH" },
      { "AttributeName": "AlbumTitle", "KeyType": "RANGE" }
    ],
    "Projection": {
      "ProjectionType": "INCLUDE",
      "NonKeyAttributes": ["Genre", "Year"]
    }
  }
]
```

Cargar el archivo songs.json a dynamodb

`aws dynamodb batch-write-item --request-items file://songs.json`

## Consultas con índices locales

Cuando usas **Índices Secundarios Locales (LSI)** en Amazon DynamoDB, puedes hacer **consultas más flexibles** que no están limitadas únicamente a las claves primarias. Aquí te explico cómo hacer consultas usando un **LSI**, y te doy un ejemplo concreto basado en la tabla `Music` que definiste.

### 🧠 ¿Qué es un índice secundario local (LSI)?

Un **LSI** permite consultar datos usando la misma clave de partición (`HASH`) que la tabla, pero **con una clave de ordenamiento secundaria distinta**.

### 🎯 Escenario: Tabla `Music` con LSI `AlbumTitleIndex`

```json
Tabla: Music
Clave primaria: Artist (HASH) + SongTitle (RANGE)
LSI: AlbumTitleIndex
  - HASH: Artist
  - RANGE: AlbumTitle
  - Proyección: INCLUDE (Genre, Year)
```

### 🔍 Consulta usando `AlbumTitleIndex`

### 📘 Objetivo:

Obtener todas las canciones del artista `"The Beatles"`, ordenadas por título del álbum (`AlbumTitle`), y mostrar el género y el año.

### ✅ Comando AWS CLI:

```bash
aws dynamodb query \
  --table-name Music \
  --index-name AlbumTitleIndex \
  --key-condition-expression "Artist = :artist" \
  --expression-attribute-values '{":artist":{"S":"The Beatles"}}' \
  --projection-expression "AlbumTitle, Genre, Year"
```

### 📝 Notas importantes:

* `--index-name` indica que estás consultando usando un índice (obligatorio).
* `--key-condition-expression` debe usar solo los atributos definidos como claves en ese índice.
* Puedes incluir atributos proyectados como `Genre` y `Year`.

## Índices globales

### 🌍 Índices Secundarios Globales (GSI) en DynamoDB

Un **Índice Secundario Global (GSI)** en Amazon DynamoDB permite consultar una tabla **usando una clave de partición y/o ordenamiento distinta a la clave principal** de la tabla. Es una herramienta muy poderosa para realizar consultas flexibles sin duplicar datos.

### 🧠 Diferencia entre LSI y GSI:

| Característica          | LSI                             | GSI                                                 |
| ----------------------- | ------------------------------- | --------------------------------------------------- |
| Clave de partición      | Igual a la tabla principal      | Puede ser diferente a la tabla principal            |
| Número máximo por tabla | Hasta 5                         | Hasta 20                                            |
| Creación                | Solo al crear la tabla          | Se puede agregar en cualquier momento               |
| Capacidad de lectura    | Comparte capacidad con la tabla | Tiene su propia capacidad (on-demand o provisioned) |

### 🏗️ Ejemplo de GSI

### 🎯 Tabla: `Orders`

```json
Clave primaria: CustomerId (HASH) + OrderId (RANGE)
```

### ➕ GSI: `StatusIndex`

```json
Clave GSI: Status (HASH) + OrderDate (RANGE)
Proyección: ALL
```

### 🔧 Crear un GSI con AWS CLI

```bash
aws dynamodb update-table \
  --table-name Orders \
  --attribute-definitions \
    AttributeName=Status,AttributeType=S \
    AttributeName=OrderDate,AttributeType=S \
  --global-secondary-index-updates \
    '[{
      "Create": {
        "IndexName": "StatusIndex",
        "KeySchema": [
          { "AttributeName": "Status", "KeyType": "HASH" },
          { "AttributeName": "OrderDate", "KeyType": "RANGE" }
        ],
        "Projection": {
          "ProjectionType": "ALL"
        },
        "ProvisionedThroughput": {
          "ReadCapacityUnits": 5,
          "WriteCapacityUnits": 5
        }
      }
    }]'
```

### 🔍 Consultar usando un GSI

```bash
aws dynamodb query \
  --table-name Orders \
  --index-name StatusIndex \
  --key-condition-expression "Status = :status" \
  --expression-attribute-values '{":status": {"S": "SHIPPED"}}'
```

### ✅ ¿Cuándo usar un GSI?

Usa un GSI cuando necesitas:

* Consultar por atributos distintos a la clave primaria.
* Realizar búsquedas eficientes por estado, categoría, ubicación, etc.
* Separar el rendimiento de consultas secundarias.

## Unidades de lectura y escritura

En Amazon DynamoDB, **las unidades de lectura y escritura** determinan la capacidad de procesamiento de una tabla (cuando se usa el modo de **capacidad aprovisionada**). Aquí tienes una explicación clara:

### 📘 **Unidades de Lectura (Read Capacity Units - RCU)**

* **1 RCU** permite:

  * 1 lectura **fuertemente coherente** por segundo de un ítem de hasta **4 KB**.
  * 2 lecturas **eventualmente coherentes** por segundo de un ítem de hasta **4 KB**.

> Si tu ítem es mayor a 4 KB, se consume 1 RCU por cada múltiplo de 4 KB.

### ✍️ **Unidades de Escritura (Write Capacity Units - WCU)**

* **1 WCU** permite:

  * 1 escritura por segundo de un ítem de hasta **1 KB**.

> Si el ítem es mayor a 1 KB, se consume 1 WCU por cada múltiplo de 1 KB.

### 🛠️ **Ejemplo práctico**

Supón que tu tabla tiene:

* Items de **2 KB**
* Y requiere:

  * **100 lecturas eventualmente coherentes por segundo**
  * **50 escrituras por segundo**

**Cálculo**:

* Lecturas:

  * 2 KB → 1 RCU por lectura eventualmente coherente permite 2 lecturas/sec → necesitas **50 RCU**.
* Escrituras:

  * 2 KB → 2 WCU por escritura → necesitas **2 × 50 = 100 WCU**.

### 🧩 Modos de capacidad en DynamoDB

1. **Capacidad aprovisionada**: Defines RCU y WCU manualmente.
2. **Capacidad bajo demanda**: DynamoDB escala automáticamente según el tráfico (no defines RCU/WCU).

## SDK y DynamoDB

El **SDK (Software Development Kit)** de AWS te permite interactuar con DynamoDB desde distintos lenguajes de programación como **Python, JavaScript, Java, Go, etc.**

Aquí te explico cómo usarlo con el lenguaje más común: **Python**, usando el paquete `boto3`.

### 🧰 1. Instalación del SDK para Python (`boto3`)

```bash
pip install boto3
```

### 🔐 2. Configurar credenciales de AWS

Puedes hacerlo con el siguiente comando:

```bash
aws configure
```

Esto guarda tus credenciales de acceso (Access Key, Secret Key, región, y formato) en `~/.aws/credentials`.

### 📁 3. Crear una tabla DynamoDB con SDK (Python)

```python
import boto3

dynamodb = boto3.resource('dynamodb')

table = dynamodb.create_table(
    TableName='Estudiantes',
    KeySchema=[
        {
            'AttributeName': 'EstudianteId',
            'KeyType': 'HASH'  # Clave primaria
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'EstudianteId',
            'AttributeType': 'N'
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Esperando a que la tabla esté disponible...")
table.wait_until_exists()
print("Tabla creada con éxito")
```

### 📥 4. Insertar un ítem en la tabla

```python
table = dynamodb.Table('Estudiantes')

table.put_item(
   Item={
        'EstudianteId': 1,
        'Nombre': 'Mario Vargas',
        'Curso': 'Ingeniería'
    }
)
```

### 📤 5. Consultar un ítem

```python
response = table.get_item(
    Key={
        'EstudianteId': 1
    }
)
print(response['Item'])
```

### 🧹 6. Eliminar una tabla

```python
table = dynamodb.Table('Estudiantes')
table.delete()
```

**Archivos de la clase**

[main.py](https://static.platzi.com/media/public/uploads/main_e80cf8e5-b208-45b5-bda2-587228d38513.py "main.py")

**Lecturas recomendadas**

[https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html")

[main.py - Google Drive](https://drive.google.com/file/d/1StJr-02bbERETJrLz5y4al45S2HQsvNS/view?usp=share_link "main.py - Google Drive")

## Creación de AWS Lambda

¡Perfecto! Aquí te explico cómo crear una función **AWS Lambda** paso a paso, usando la consola, CLI y también un ejemplo en código (Python):

### 🖥️ OPCIÓN 1: Crear una Lambda desde la **Consola AWS**

1. Ve a la consola de AWS: [https://console.aws.amazon.com/lambda](https://console.aws.amazon.com/lambda)
2. Haz clic en **“Crear función”**
3. Elige:

   * **Nombre**: `MiPrimeraLambda`
   * **Tiempo de ejecución (runtime)**: Python 3.10 (u otro que prefieras)
   * **Permisos**: crea un nuevo rol con permisos básicos de Lambda
4. Clic en **Crear función**
5. En el editor de código, puedes poner algo como esto:

```python
def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': 'Hola desde Lambda!'
    }
```

6. Dale clic a “Implementar” y luego en “Probar” para ver el resultado.

### 💻 OPCIÓN 2: Crear Lambda con la CLI de AWS

Primero asegúrate de tener:

* AWS CLI instalado (`aws --version`)
* Configurado (`aws configure`)
* ZIP del código Python

### Paso 1: Crea el archivo de código

`lambda_function.py`:

```python
def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': '¡Hola desde Lambda CLI!'
    }
```

### Paso 2: Comprimirlo

```bash
zip function.zip lambda_function.py
```

### Paso 3: Crear la función Lambda

```bash
aws lambda create-function \
  --function-name MiPrimeraLambdaCLI \
  --runtime python3.10 \
  --role arn:aws:iam::123456789012:role/rol_lambda_ejemplo \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip
```

> ⚠️ Reemplaza el ARN del rol con el tuyo real que tenga permiso `AWSLambdaBasicExecutionRole`.

### ⚙️ ¿Qué necesitas para ejecutar Lambda?

* **Código comprimido**
* **Un rol IAM** con al menos el permiso: `AWSLambdaBasicExecutionRole`
* **Nombre de función**, `runtime`, y `handler`

crear un role:

`aws iam create-role --role-name dynamodbStreamsRole --assume-role-policy-document file://file/stream/role.json`

crea policy:

`aws iam put-role-policy --role-name dynamodbStreamsRole --policy-name lambdapermissions --policy-document file://file/stream/policy.json`

comprimir el archivo main.py:

`zip function.zip file/stream/main.py`

Crea la lambda:

`aws lambda create-function --function-name lambda-export --role arn:aws:iam::376129853411:role/dynamodbStreamsRole --runtime python3.9 --handler main.dynamodb_events --publish --zip-file fileb://file/stream/function.zip`

**Archivos de la clase**

[role.json](https://static.platzi.com/media/public/uploads/role_04a74542-c867-4561-a900-169fbd85750d.json "role.json")
[policy.json](https://static.platzi.com/media/public/uploads/policy_d61ca807-9f9a-4583-bb86-8d5ee9aaef54.json "policy.json")
[main.py](https://static.platzi.com/media/public/uploads/main_71b8ca35-2d19-4569-91b6-4578e2087cb5.py "main.py")

**Lecturas recomendadas**

[role.json - Google Drive](https://drive.google.com/file/d/1GwpeEmrLQdOshibFWnZKwzzpx1U6jiMR/view?usp=share_link "role.json - Google Drive")

[policy.json - Google Drive](https://drive.google.com/file/d/1faMSgI3VSTlWFCUVXzzN0hZytRvJO1Q9/view?usp=share_link "policy.json - Google Drive")

[main.py - Google Drive](https://drive.google.com/file/d/1pQ-CMWv7ksKnkupN8jsPxzgSNVBY8U2P/view?usp=share_link "main.py - Google Drive")

## DynamoDB Streams

### 🔄 **DynamoDB Streams: Explicación Clara y Directa**

**DynamoDB Streams** es una característica de Amazon DynamoDB que permite capturar **cambios (inserciones, actualizaciones y eliminaciones)** en una tabla de DynamoDB **en tiempo real**.

### 📌 ¿Para qué sirve?

Con DynamoDB Streams puedes:

* **Disparar funciones Lambda automáticamente** cuando los datos cambian.
* Sincronizar datos con otros servicios o bases de datos.
* Auditar cambios.
* Implementar replicación entre regiones.

### 🧠 ¿Qué contiene un stream?

Cada registro en el stream puede contener:

* El **elemento anterior** (old image)
* El **elemento nuevo** (new image)
* Ambas imágenes
* Solo las **claves** modificadas

Esto depende de la configuración que elijas al habilitar el stream.

### ⚙️ ¿Cómo habilitar DynamoDB Streams?

Puedes hacerlo al crear o actualizar una tabla:

```bash
aws dynamodb update-table \
  --table-name NombreDeTuTabla \
  --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES
```

**Tipos de `StreamViewType`:**

* `KEYS_ONLY`: solo claves primarias
* `NEW_IMAGE`: solo la nueva versión
* `OLD_IMAGE`: solo la versión anterior
* `NEW_AND_OLD_IMAGES`: ambas versiones

### 🚀 Conectar Streams a Lambda

1. Crea la función Lambda.
2. Asóciala al stream de la tabla:

```bash
aws lambda create-event-source-mapping \
  --function-name lambda-export \
  --event-source-arn arn:aws:dynamodb:REGION:ACCOUNT_ID:table/TuTabla/stream/FECHA \
  --starting-position LATEST
```

### 📥 Ejemplo de evento recibido por Lambda

```json
{
  "Records": [
    {
      "eventID": "1",
      "eventName": "INSERT",
      "dynamodb": {
        "Keys": {
          "ID": { "S": "123" }
        },
        "NewImage": {
          "ID": { "S": "123" },
          "Nombre": { "S": "Mario" }
        }
      },
      "eventSource": "aws:dynamodb"
    }
  ]
}
```

## Integrar API Gateway con DynamoDB

Integrar **API Gateway con DynamoDB** permite exponer tu base de datos como una API RESTful sin necesidad de servidores. Esto se puede hacer de dos formas principales:

### 🛠️ Opción 1: **Usando Lambda como intermediario**

**Arquitectura recomendada (flexible y segura):**

```
API Gateway → AWS Lambda → DynamoDB
```

### 🔹 Paso 1: Crear una tabla DynamoDB (si no existe)

```bash
aws dynamodb create-table \
  --table-name Productos \
  --attribute-definitions AttributeName=productoId,AttributeType=S \
  --key-schema AttributeName=productoId,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### 🔹 Paso 2: Crear una función Lambda (ejemplo en Python)

```python
import boto3
import json

dynamodb = boto3.resource('dynamodb')
tabla = dynamodb.Table('Productos')

def lambda_handler(event, context):
    if event['httpMethod'] == 'GET':
        producto_id = event['queryStringParameters']['productoId']
        response = tabla.get_item(Key={'productoId': producto_id})
        return {
            'statusCode': 200,
            'body': json.dumps(response.get('Item', {}))
        }

    if event['httpMethod'] == 'POST':
        item = json.loads(event['body'])
        tabla.put_item(Item=item)
        return {'statusCode': 200, 'body': 'Producto agregado'}
```

### 🔹 Paso 3: Crear la función Lambda en AWS

```bash
aws lambda create-function \
  --function-name productos-api \
  --runtime python3.12 \
  --role arn:aws:iam::<tu-account-id>:role/<rol-con-permisos-dynamodb-lambda> \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://funcion.zip
```

### 🔹 Paso 4: Crear API REST en API Gateway

1. Ve a **API Gateway** > **Crear API REST**.
2. Crea un recurso (ej: `/productos`).
3. Añade métodos (GET, POST).
4. En la integración selecciona **Lambda Function** y elige `productos-api`.
5. Despliega la API (crear un stage, como `dev`).

### ✅ Opción 2: **Integración directa (sin Lambda)**

API Gateway puede conectarse directamente a DynamoDB usando **integraciones VTL (Velocity Template Language)**, pero esto es más complejo y menos flexible. Solo se recomienda si quieres evitar Lambda por costos.

### ✅ Permisos necesarios

Asegúrate de que el **rol Lambda tenga esta política** mínima:

```json
{
  "Effect": "Allow",
  "Action": [
    "dynamodb:GetItem",
    "dynamodb:PutItem"
  ],
  "Resource": "arn:aws:dynamodb:*:*:table/Productos"
}
```

**Archivos de la clase**

[platzitemplate.json](https://static.platzi.com/media/public/uploads/platzitemplate_221657b4-63ab-4a47-8b3d-a510251b7224.json "platzitemplate.json")

**Lecturas recomendadas**

[Amazon API Gateway | API Management | Amazon Web Services](https://aws.amazon.com/api-gateway/ "Amazon API Gateway | API Management | Amazon Web Services")

[platzitemplate.json - Google Drive](https://drive.google.com/file/d/1rvExRPGz96DlrKFfIreD74lMQnaH7Wkx/view?usp=share_link "platzitemplate.json - Google Drive")

## Casos de uso de DynamoDB

Aquí tienes algunos de los **principales casos de uso de Amazon DynamoDB**, una base de datos NoSQL totalmente gestionada, altamente escalable y con baja latencia:

### 🔹 1. **Aplicaciones web y móviles a gran escala**

* **Casos:** perfiles de usuario, configuración personalizada, sesiones.
* **Ventaja:** baja latencia de milisegundos, perfecta para apps en tiempo real.
* **Ejemplo:** una app de redes sociales que almacena publicaciones, likes y comentarios.

### 🔹 2. **Gaming (juegos en línea)**

* **Casos:** seguimiento de puntuaciones, progreso de jugador, partidas multijugador.
* **Ventaja:** escalabilidad automática con millones de usuarios concurrentes.
* **Ejemplo:** tabla `Jugadores` con estadísticas, inventario y progreso.

### 🔹 3. **Carritos de compra y e-commerce**

* **Casos:** carritos activos, historial de pedidos, catálogos de productos.
* **Ventaja:** velocidad de lectura/escritura rápida sin preocuparse por la infraestructura.
* **Ejemplo:** tabla `Pedidos` con el historial de cada usuario.

### 🔹 4. **Internet de las cosas (IoT)**

* **Casos:** almacenamiento de datos de sensores, eventos en tiempo real.
* **Ventaja:** manejo eficiente de grandes volúmenes de datos que ingresan constantemente.
* **Ejemplo:** tabla `LecturasSensor` con timestamp como clave de ordenamiento.

### 🔹 5. **Gestión de catálogos**

* **Casos:** bibliotecas digitales, menús, inventarios.
* **Ventaja:** flexibilidad del esquema, ideal para catálogos con distintos atributos por ítem.
* **Ejemplo:** tabla `Productos` con SKU, categoría, precio y atributos personalizados.

### 🔹 6. **Mensajería y chat**

* **Casos:** historial de conversaciones, usuarios conectados.
* **Ventaja:** latencia baja y consistencia eventual, ideal para mensajes rápidos.
* **Ejemplo:** tabla `Mensajes` con claves compuestas por ID de conversación + timestamp.

### 🔹 7. **Sistemas de recomendación y personalización**

* **Casos:** seguimiento de actividad del usuario, historial de navegación.
* **Ventaja:** DynamoDB Streams permite analizar eventos en tiempo real con Lambda.
* **Ejemplo:** tabla `ActividadUsuario` con eventos que luego alimentan un motor de IA.

## Exportación de datos a S3

Exportar datos de DynamoDB a Amazon S3 es una tarea común para respaldos, análisis o integraciones. Aquí te muestro las **opciones disponibles** y cómo hacerlo paso a paso:

### ✅ OPCIÓN 1: **Exportar directamente desde DynamoDB a S3 (sin código)**

### 🔸 Requisitos:

* La tabla debe tener **DynamoDB Streams habilitado**.
* Debes tener un **bucket S3** disponible.
* Rol de IAM con permisos adecuados.

### 🔸 Pasos (consola AWS):

1. Ve a la **Consola de DynamoDB**.
2. Selecciona la tabla que quieres exportar.
3. Haz clic en **"Exportar a S3"**.
4. Elige el bucket S3 y configura el prefijo (carpeta).
5. Haz clic en **"Exportar"**.
6. Se generará un archivo `.parquet` por cada fragmento exportado.

### ✅ OPCIÓN 2: **Usar AWS Data Pipeline** *(más personalizable, pero está quedando en desuso)*

### Recomendado solo si necesitas exportar a formatos como CSV/JSON automáticamente.

### ✅ OPCIÓN 3: **Exportación personalizada con AWS Lambda + Streams**

Ideal si quieres:

* Exportar solo ciertos datos.
* Exportar en tiempo real a medida que se insertan.

### Flujo:

1. **Habilita Streams** en tu tabla DynamoDB.
2. Crea una función **Lambda** que se dispare con los eventos del Stream.
3. Desde Lambda, escribe los datos en un archivo y súbelos a S3 usando `boto3`.

#### Ejemplo básico de función Lambda en Python:

```python
import boto3
import json

s3 = boto3.client('s3')
bucket_name = 'mi-bucket-export'

def lambda_handler(event, context):
    for record in event['Records']:
        if record['eventName'] == 'INSERT':
            new_item = record['dynamodb']['NewImage']
            key = f"dynamodb-backup/{new_item['ID']['S']}.json"
            s3.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=json.dumps(new_item)
            )
```

### ✅ OPCIÓN 4: **Usar AWS Glue para transformar y exportar datos**

Si deseas analizar los datos en Athena, QuickSight o Redshift, AWS Glue puede leer DynamoDB, transformarlos y almacenarlos en S3.

### 🔐 Permisos IAM mínimos

Asegúrate de que el rol de IAM tenga permisos como:

```json
{
  "Effect": "Allow",
  "Action": [
    "dynamodb:DescribeTable",
    "dynamodb:Scan",
    "s3:PutObject"
  ],
  "Resource": "*"
}
```

## Monitoreo de DynamoDB con CloudWatch

El **monitoreo de DynamoDB con CloudWatch** te permite observar el rendimiento, consumo de recursos y detectar posibles problemas en tus tablas. Aquí tienes un resumen claro con lo esencial para que lo uses de forma efectiva:

### 📊 ¿Qué métricas de DynamoDB se monitorean en CloudWatch?

AWS DynamoDB envía métricas automáticamente a CloudWatch. Algunas de las **más importantes** incluyen:

| Métrica                         | Descripción                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| `ConsumedReadCapacityUnits`     | Unidades de lectura consumidas por segundo                      |
| `ConsumedWriteCapacityUnits`    | Unidades de escritura consumidas por segundo                    |
| `ProvisionedReadCapacityUnits`  | Capacidad de lectura aprovisionada                              |
| `ProvisionedWriteCapacityUnits` | Capacidad de escritura aprovisionada                            |
| `ThrottledRequests`             | Número de solicitudes rechazadas por exceder capacidad          |
| `ReadThrottleEvents`            | Lecturas rechazadas                                             |
| `WriteThrottleEvents`           | Escrituras rechazadas                                           |
| `SuccessfulRequestLatency`      | Latencia promedio de solicitudes exitosas                       |
| `ReturnedItemCount`             | Número de ítems devueltos por operaciones de consulta o escaneo |

### 🛠️ ¿Cómo ver estas métricas?

### Opción 1: **Consola de AWS**

1. Ve a **CloudWatch > Métricas > DynamoDB**.
2. Selecciona una tabla específica o un grupo de métricas.
3. Puedes crear **gráficas, paneles y alarmas personalizadas**.

### Opción 2: **Línea de comandos (AWS CLI)**

```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/DynamoDB \
  --metric-name ConsumedReadCapacityUnits \
  --dimensions Name=TableName,Value=MiTablaDynamo \
  --statistics Average \
  --period 60 \
  --start-time 2025-05-21T00:00:00Z \
  --end-time 2025-05-22T00:00:00Z
```

### 🚨 Alarmas con CloudWatch

Puedes configurar alarmas para ser notificado (por email, SMS, Lambda, etc.) si una métrica supera cierto umbral.

### Ejemplo: Crear una alarma por escritura limitada

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "AlarmaEscrituraLimitada" \
  --metric-name WriteThrottleEvents \
  --namespace AWS/DynamoDB \
  --statistic Sum \
  --period 60 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --dimensions Name=TableName,Value=MiTablaDynamo \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:mi-topic-sns
```

### 🧠 Buenas prácticas

* Usa **Auto Scaling** para evitar rechazos (`ThrottledRequests`).
* Monitorea la latencia (`SuccessfulRequestLatency`) si tienes tiempos de respuesta altos.
* Crea **tableros de CloudWatch Dashboards** para ver todo en un solo lugar.
* Si usas Streams o TTL, también puedes ver métricas relacionadas.

## Mantenimiento

El **mantenimiento de Amazon DynamoDB** se refiere a las prácticas necesarias para asegurar que tus tablas y datos estén disponibles, optimizados y seguros a lo largo del tiempo. Aunque DynamoDB es un servicio **completamente administrado por AWS**, tú como usuario eres responsable de ciertas tareas clave de mantenimiento.

### 🔧 ¿Qué incluye el mantenimiento en DynamoDB?

### 1. **Monitoreo continuo**

* Usa **Amazon CloudWatch** para observar:

  * Capacidad consumida (lectura/escritura).
  * Eventos de throttling (cuando se supera la capacidad).
  * Latencia de operaciones.
* Crea **alarmas** para recibir notificaciones proactivas.

### 2. **Optimización del rendimiento**

* Revisa si estás usando bien los **índices secundarios** (GSI/LSI).
* Verifica el diseño de claves (evita las **hot partitions**).
* Evalúa si necesitas ajustar la capacidad provisionada o usar modo on-demand.

### 3. **Backups y recuperación**

* Activa **on-demand backups** o **point-in-time recovery (PITR)** para cada tabla.
* Permite restaurar datos a cualquier segundo dentro de los últimos 35 días.

```bash
aws dynamodb update-continuous-backups \
  --table-name MiTabla \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true
```

### 4. **Limpieza de datos**

* Usa **TTL (Time To Live)** para eliminar automáticamente ítems antiguos/innecesarios.
* Esto ahorra almacenamiento y reduce costos.

```bash
aws dynamodb update-time-to-live \
  --table-name MiTabla \
  --time-to-live-specification "Enabled=true, AttributeName=expiraEn"
```

### 5. **Gestión de seguridad y permisos**

* Revisa y actualiza regularmente políticas de acceso en IAM.
* Usa cifrado en reposo (AES-256 o AWS KMS).
* Habilita el **registro de auditoría con AWS CloudTrail**.

### 📅 Recomendaciones periódicas

| Frecuencia          | Actividad                                     |
| ------------------- | --------------------------------------------- |
| **Diariamente**     | Revisar métricas y errores                    |
| **Semanalmente**    | Verificar alarmas y eventos de CloudWatch     |
| **Mensualmente**    | Evaluar necesidad de optimizaciones o índices |
| **Trimestralmente** | Auditar IAM y políticas de seguridad          |

## Escalabilidad

La **escalabilidad** en **Amazon DynamoDB** es una de sus características más potentes y se refiere a la capacidad del servicio para **aumentar o disminuir automáticamente su rendimiento y capacidad de almacenamiento** conforme cambian las necesidades de tu aplicación. Aquí te explico cómo funciona y qué debes tener en cuenta:

### 🚀 ¿Qué significa que DynamoDB sea escalable?

DynamoDB está diseñado para manejar:

* **Millones de solicitudes por segundo**.
* **Almacenamiento desde unos pocos KB hasta cientos de TB**.
* **Escalado automático y sin interrupciones**.

### ⚙️ Tipos de escalabilidad en DynamoDB

### 1. **Escalabilidad horizontal**

* DynamoDB divide automáticamente las tablas grandes en **particiones** físicas.
* Cada partición puede manejar una cantidad específica de lecturas, escrituras y almacenamiento.
* Si superas los límites de una partición, DynamoDB crea nuevas para repartir la carga.

### 2. **Escalado automático (Auto Scaling)**

DynamoDB puede **ajustar automáticamente** la capacidad provisionada (si no usas el modo On-Demand):

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace dynamodb \
  --resource-id table/MiTabla \
  --scalable-dimension dynamodb:table:WriteCapacityUnits \
  --min-capacity 5 \
  --max-capacity 100
```

Puedes establecer políticas para que se escale según métricas como `ConsumedWriteCapacityUnits`.

### 3. **Modo On-Demand**

* No necesitas definir capacidad de lectura o escritura.
* Paga **solo por las solicitudes que haces**.
* Escala automáticamente sin configuración adicional.

Ideal para cargas **variables o impredecibles**.

```bash
aws dynamodb update-table \
  --table-name MiTabla \
  --billing-mode PAY_PER_REQUEST
```

### 🔥 Evitar particiones calientes (Hot Partitions)

* Una **partición caliente** ocurre cuando muchas operaciones se concentran en una sola clave de partición.
* Esto **afecta el rendimiento** y limita la escalabilidad.

### ✅ Buenas prácticas:

* Usa claves de partición con **alta cardinalidad y distribución uniforme**.
* Evita usar atributos como fechas fijas (`2025-05-22`) como partición primaria.

### 📊 ¿Cómo saber si necesitas escalar?

Monitorea métricas como:

* `ConsumedReadCapacityUnits`
* `ThrottledRequests`
* `ProvisionedThroughputExceeded`

Con estos datos puedes ajustar tus políticas de escalado.