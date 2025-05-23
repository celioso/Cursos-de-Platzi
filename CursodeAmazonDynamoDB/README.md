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