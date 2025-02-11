# Curso de Base de Datos NoSQL

## NoSQL: El Otro Tipo de Bases de Datos

Las bases de datos **NoSQL** son una alternativa a las bases de datos relacionales tradicionales, diseñadas para manejar grandes volúmenes de datos, escalabilidad y flexibilidad en el modelado de datos.  

### 🚀 **¿Qué es NoSQL?**  
**NoSQL** (Not Only SQL) es un tipo de base de datos que **no sigue el modelo relacional** basado en tablas y esquemas rígidos. En su lugar, ofrece estructuras más flexibles como:  
- Documentos  
- Claves-valor  
- Columnas anchas  
- Grafos  

### 🔥 **Características Clave**  
✅ **Escalabilidad horizontal**: Se distribuyen fácilmente en múltiples servidores.  
✅ **Alto rendimiento**: Optimizadas para lecturas y escrituras rápidas.  
✅ **Modelo de datos flexible**: No requieren esquemas predefinidos.  
✅ **Alta disponibilidad**: Diseñadas para tolerancia a fallos y replicación automática.  

### 🔍 **Tipos de Bases de Datos NoSQL**  
1️⃣ **Bases de Datos de Documentos**  
   - Almacenan datos en formato JSON, BSON o XML.  
   - 📌 Ejemplo: MongoDB, CouchDB  

2️⃣ **Bases de Datos Clave-Valor**  
   - Datos almacenados como un par clave-valor.  
   - 📌 Ejemplo: Redis, DynamoDB  

3️⃣ **Bases de Datos de Columnas Anchas**  
   - Almacenan datos en columnas en lugar de filas, ideales para Big Data.  
   - 📌 Ejemplo: Apache Cassandra, HBase  

4️⃣ **Bases de Datos de Grafos**  
   - Diseñadas para representar relaciones complejas entre datos.  
   - 📌 Ejemplo: Neo4j, ArangoDB  

### 📌 **¿Cuándo Usar NoSQL?**  
✅ Cuando necesitas **escalabilidad** y **rendimiento** en grandes volúmenes de datos.  
✅ Cuando los datos son **semiestructurados o no estructurados**.  
✅ Para aplicaciones **en tiempo real** (chats, redes sociales, IoT).  
✅ Para almacenar y consultar **relaciones complejas** en bases de datos de grafos.  

### ⚖️ **¿SQL o NoSQL?**  
| Característica | SQL | NoSQL |
|--------------|-----|-------|
| **Estructura** | Tablas y esquemas rígidos | Modelos flexibles (documentos, clave-valor, etc.) |
| **Escalabilidad** | Vertical (mejor hardware) | Horizontal (más servidores) |
| **Consultas** | SQL (JOINs, ACID) | API flexible, sin necesidad de JOINs |
| **Casos de uso** | Finanzas, ERP, CRM | Big Data, redes sociales, IoT |

### 🎯 **Conclusión**  
Las bases de datos **NoSQL** son ideales para aplicaciones modernas que requieren escalabilidad y flexibilidad. Sin embargo, no reemplazan a **SQL**, sino que **complementan** su uso en distintos escenarios.  

### Resumen

El cambio de gigantes tecnológicos como Meta, Twitter y Adobe de bases de datos SQL a no SQL marcó una tendencia significativa en la tecnología. Empresas como Uber, Netflix y Google también han adoptado no SQL debido a la necesidad de adaptarse rápidamente a los cambios en el uso de datos.

La flexibilidad de no SQL es crucial para startups, permitiendo agregar valor en cada sprint. No SQL no implica la exclusión total del lenguaje SQL; existen bases de datos no SQL que también soportan SQL. Este término se popularizó en los 2000 por la necesidad de manejar datos de manera diferente a medida que el costo de almacenamiento disminuyó y las aplicaciones requerían almacenar y consultar más datos.

### ¿Qué es no SQL?

- SQL, que significa Structured Query Language, es un lenguaje estándar utilizado en muchas tecnologías de bases de datos.
- No SQL no significa necesariamente no usar SQL; algunas bases de datos no SQL soportan SQL.
- Originalmente, no SQL se asociaba con bases de datos no relacionales, pero hoy en día tiene un significado más amplio.

### ¿Por qué surgió la necesidad de no SQL?

- Las bases de datos no relacionales existen desde finales de los 60.
- En los 2000, la necesidad de manejar grandes volúmenes de datos y la reducción en el costo de almacenamiento impulsaron la adopción de no SQL.
- No SQL permite almacenar datos estructurados, semiestructurados y polimórficos, ofreciendo flexibilidad y nuevas posibilidades.

### ¿Cuáles son los tipos de bases de datos no SQL?

- Bases de datos de grafos.
- Basadas en documentos.
- De llave-valor.
- Columnares.
- Vectoriales.

### ¿Cómo escalan las bases de datos no SQL?

- Las bases de datos SQL escalan verticalmente (más RAM, CPU, capacidad de almacenamiento).
- No SQL permite escalar horizontalmente, facilitando la partición de datos mediante estrategias como sharding.
- Esta técnica distribuye datos en distintos clusters, mejorando la eficiencia de lecturas y escrituras.

### ¿Qué características diferencian a no SQL?

- Enfoque en consistencia, disponibilidad y tolerancia a particiones (Teorema CAP).
- Las bases de datos no SQL pueden cumplir solo dos de estas tres características, lo que representa una ventaja dependiendo del uso.
- Importancia de conocer las ventajas y desventajas para elegir la mejor base de datos para cada aplicación.

### ¿Cuándo utilizar bases de datos no SQL?

- Cuando se requiere flexibilidad para agregar valor rápidamente en proyectos con cambios constantes.
- Al manejar grandes volúmenes de datos que necesitan escalar horizontalmente.
- En aplicaciones donde la consistencia y disponibilidad son más críticas que la tolerancia a particiones, o viceversa.

**Lecturas recomendadas**

[What Is NoSQL? NoSQL Databases Explained | MongoDB](https://www.mongodb.com/nosql-explained)

[Teorema CAP - Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/Teorema_CAP)

[NoSQL Resources - Couchbase](https://www.couchbase.com/resources/)

## ¿Qué debo elegir? NoSQL vs SQL

Elegir entre **NoSQL** y **SQL** depende de las necesidades de tu proyecto. Aquí tienes una comparación clara para ayudarte a decidir:

### ✅ **Cuándo elegir SQL** (Bases de datos relacionales)
💾 **Ejemplos**: MySQL, PostgreSQL, Oracle, SQL Server  
📌 **Úsalo si necesitas**:  
✔ **Estructura y relaciones**: Los datos tienen un esquema definido y dependen unos de otros (relaciones).  
✔ **Transacciones ACID**: Necesitas garantizar Atomicidad, Consistencia, Aislamiento y Durabilidad (por ejemplo, en bancos y sistemas financieros).  
✔ **Consultas complejas**: Usas `JOINs`, filtros avanzados y relaciones múltiples entre tablas.  
✔ **Datos bien estructurados**: Cuando los datos tienen un formato predecible (ejemplo: registros de clientes en una tienda).  
✔ **Integridad de datos**: La precisión y consistencia de los datos es una prioridad.  

🔹 **Ejemplo de uso:**  
- Bancos, contabilidad, ERP, CRM, comercio electrónico con datos estructurados.  

### ✅ **Cuándo elegir NoSQL** (Bases de datos no relacionales)  
💾 **Ejemplos**: MongoDB, Redis, Cassandra, Firebase  
📌 **Úsalo si necesitas**:  
✔ **Escalabilidad horizontal**: Manejo eficiente de grandes volúmenes de datos distribuidos.  
✔ **Alta velocidad**: Rápido acceso a la información, especialmente para lecturas y escrituras masivas.  
✔ **Datos flexibles y dinámicos**: No tienes un esquema fijo y los datos pueden cambiar con frecuencia.  
✔ **Big Data y análisis en tiempo real**: Manejas grandes cantidades de datos en redes sociales, IoT, logs, etc.  
✔ **Almacenamiento de datos semiestructurados o no estructurados**: JSON, BSON, XML, documentos, etc.  

🔹 **Ejemplo de uso:**  
- Redes sociales, aplicaciones de mensajería, streaming, análisis de datos en tiempo real, IoT.  

### ⚖ **Comparación SQL vs NoSQL**  

| **Característica**      | **SQL**                            | **NoSQL**                          |
|----------------------|--------------------------------|----------------------------------|
| **Estructura**      | Tablas con filas y columnas   | Documentos, Clave-Valor, Grafos |
| **Esquema**        | Fijo y predefinido            | Flexible y dinámico            |
| **Escalabilidad**  | Vertical (mejor hardware)    | Horizontal (más servidores)    |
| **Consultas**      | Complejas con `JOINs`         | Simples y rápidas, sin `JOINs`  |
| **Transacciones**  | ACID (consistencia fuerte)    | BASE (eventual consistency)    |
| **Casos de uso**   | Finanzas, ERP, CRM, e-commerce | Redes sociales, Big Data, IoT  |

### 🚀 **Conclusión: ¿Cuál elegir?**  
✅ **Usa SQL si necesitas** estructura, relaciones complejas y transacciones seguras.  
✅ **Usa NoSQL si necesitas** escalabilidad, flexibilidad y manejar grandes volúmenes de datos sin relaciones estrictas.  

Si tienes un **proyecto híbrido**, puedes combinar ambas tecnologías, por ejemplo:  
- **SQL** para gestionar usuarios y transacciones.  
- **NoSQL** para almacenar logs, notificaciones o datos en tiempo real.

### Resumen

Elegir la tecnología adecuada para un proyecto puede ser complejo y tener un impacto significativo en términos económicos, de tiempo y en la experiencia del equipo. La decisión entre utilizar SQL o NoSQL depende del caso de uso, la infraestructura y la naturaleza de los datos. A continuación, se presentan algunos escenarios y ventajas de cada tecnología para ayudar en esta elección.

### ¿Cuándo es ventajoso utilizar bases de datos NoSQL?

- **Datos semiestructurados**: Cuando los datos no están bien definidos desde el inicio o tienden a ser semiestructurados.
- **Datos sin relaciones fuertes**: Ideal cuando no hay relaciones fuertes entre los datos.
- **Distribución geográfica**: Necesidad de distribuir datos localmente o geográficamente para cumplir con leyes de protección de datos.
- **Esquemas cambiantes**: Útil cuando los datos son definidos por aplicaciones o terceros, como en middleware de APIs o almacenamiento de logs.
- **Disponibilidad rápid**a: Priorizar la disponibilidad rápida de los datos sobre la consistencia fuerte, no enfatizando el modelo ACID.

Ejemplo: Un e-commerce podría almacenar información de productos en una base de datos orientada a documentos como MongoDB, utilizar un motor de búsqueda como Elasticsearch para búsquedas rápidas y bases de datos vectoriales para recomendaciones.

### ¿Cuándo es ventajoso utilizar bases de datos SQL?

- **Esquemas bien definidos**: Cuando los datos y el esquema están bien definidos y no cambiarán con el tiempo.
- **Relaciones claras**: Cuando existen relaciones importantes y claras entre las entidades desde el inicio del proyecto.
- **Consistencia de datos**: La ventaja de ACID, que asegura la consistencia, disponibilidad y otros factores cruciales para datos que requieren alta integridad.

Ejemplo: Un sistema bancario que necesita mantener la consistencia de los datos y maneja relaciones claras y definidas puede beneficiarse del uso de SQL.

### ¿Qué es la persistencia políglota y por qué es común?

La persistencia políglota es la práctica de utilizar múltiples tecnologías de bases de datos en un solo proyecto para aprovechar las ventajas de cada una. Es común en grandes compañías que han escalado, ya que permite combinar bases de datos SQL y NoSQL para obtener mejores resultados.

**Lecturas recomendadas**

[ACID - Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/ACID)

[Power BI documentation - Power BI | Microsoft Learn](https://docs.microsoft.com/en-us/power-bi/)

[MongoDB Documentation](https://docs.mongodb.com/)

## Tus primeros pasos con MongoDB

MongoDB es una base de datos NoSQL orientada a documentos, diseñada para almacenar datos en formato **JSON/BSON** de manera flexible y escalable. Aquí tienes una guía rápida para empezar con MongoDB.  

### 🔹 **1. Instalación de MongoDB**
### 🖥️ **En Windows, Mac o Linux**
1. **Descargar MongoDB Community Server** desde [mongodb.com](https://www.mongodb.com/try/download/community).
2. **Instalar** siguiendo las instrucciones para tu sistema operativo.
3. **Verificar instalación**:  
   - En la terminal o CMD, ejecuta:
     ```sh
     mongod --version
     ```

### 🔹 **2. Iniciar el Servidor de MongoDB**
Antes de usar MongoDB, debes iniciar el **servidor**:  
```sh
mongod
```
Esto iniciará MongoDB en el puerto **27017** (por defecto).  

Si deseas conectarte a la base de datos desde otro terminal:  
```sh
mongo
```

### 🔹 **3. Primeros Pasos con MongoDB**
### 🗂️ **Crear y Usar una Base de Datos**
```sh
use mi_base_de_datos
```
Si la base de datos no existe, se crea automáticamente.

### 📁 **Crear una Colección y Agregar Datos**
MongoDB almacena los datos en **colecciones** (similares a tablas en SQL).  
Ejemplo de insertar un documento en la colección `usuarios`:  
```sh
db.usuarios.insertOne({
    nombre: "Mario Vargas",
    edad: 30,
    ciudad: "Bogotá"
})
```
Para insertar múltiples documentos:  
```sh
db.usuarios.insertMany([
    { nombre: "Ana Pérez", edad: 25, ciudad: "Medellín" },
    { nombre: "Carlos López", edad: 35, ciudad: "Cali" }
])
```

### 🔍 **Consultar Datos**
- Ver todos los documentos:
  ```sh
  db.usuarios.find()
  ```
- Buscar por nombre:
  ```sh
  db.usuarios.find({ nombre: "Mario Vargas" })
  ```
- Mostrar de manera legible:
  ```sh
  db.usuarios.find().pretty()
  ```

### ✏️ **Actualizar Datos**
```sh
db.usuarios.updateOne(
    { nombre: "Mario Vargas" },
    { $set: { edad: 31 } }
)
```
Para actualizar múltiples documentos:
```sh
db.usuarios.updateMany(
    { ciudad: "Bogotá" },
    { $set: { ciudad: "Bogotá, Colombia" } }
)
```

### 🗑️ **Eliminar Datos**
- **Eliminar un usuario**:
  ```sh
  db.usuarios.deleteOne({ nombre: "Carlos López" })
  ```
- **Eliminar todos los usuarios de una ciudad**:
  ```sh
  db.usuarios.deleteMany({ ciudad: "Cali" })
  ```
- **Eliminar toda la colección**:
  ```sh
  db.usuarios.drop()
  ```

### 🔹 **4. Conectar MongoDB con Python**
Para usar MongoDB con Python, instala la librería `pymongo`:
```sh
pip install pymongo
```
Ejemplo de conexión y consulta con Python:
```python
from pymongo import MongoClient

# Conectar con MongoDB
cliente = MongoClient("mongodb://localhost:27017/")
db = cliente["mi_base_de_datos"]
coleccion = db["usuarios"]

# Insertar un documento
coleccion.insert_one({"nombre": "Luis", "edad": 28})

# Mostrar documentos
for usuario in coleccion.find():
    print(usuario)
```

## 🚀 **Conclusión**
MongoDB es una excelente opción para almacenar datos flexibles, escalar horizontalmente y manejar grandes volúmenes de información.

### Resumen

La adopción de bases de datos NoSQL ha crecido exponencialmente y MongoDB se ha posicionado como una de las más populares. Su versatilidad y capacidad para manejar datos semiestructurados son algunas de sus ventajas destacadas, lo que facilita su integración con tecnologías como SQL y herramientas de análisis de datos. Vamos a profundizar en las razones para elegir MongoDB y cómo comenzar a utilizarlo.

### ¿Por qué elegir MongoDB?

MongoDB representa aproximadamente el 44% del mercado de bases de datos NoSQL, lo que habla de su aceptación y robustez. Su API es compatible con otras tecnologías como Cosmos de Azure, permitiendo replicar comandos y ejemplos fácilmente.

### ¿Cuáles son las características clave de MongoDB?

- **Compatibilidad con SQL**: Permite realizar análisis y consultas utilizando SQL, conectándose con herramientas como Power BI para la visualización de datos.
- **Persistencia políglota**: Utiliza distintas tecnologías de almacenamiento y consulta, incluyendo Lucene de Apache para búsquedas avanzadas, similar a Elasticsearch.
- **Datos semiestructurados**: Maneja datos en formato JSON y BSON (binary JSON), optimizando la codificación y decodificación, y soportando una amplia variedad de tipos de datos.

### ¿Cómo empezar con MongoDB?

Para crear una base de datos en MongoDB, sigue estos pasos:

- **Accede a la página de MongoDB**: Utiliza tu cuenta de Google o GitHub para iniciar sesión.
- **Crea un clúster**: Selecciona la opción M0 para un clúster gratuito. Nombrar adecuadamente tu clúster sin espacios es una buena práctica.
- **Selecciona el proveedor y región**: La opción por defecto suele ser suficiente para empezar.
- **Configura el clúster**: Genera una réplica set con tres nodos (uno primario y dos secundarios) para mayor disponibilidad y resistencia a fallos.

### ¿Cómo gestionar la replicación y disponibilidad?

- **Operaciones de escritura**: Se realizan en el nodo primario, mientras que los nodos secundarios replican esta información para asegurar disponibilidad.
- **Gestión de fallos**: Si el nodo primario falla, uno de los secundarios se convierte en el nuevo primario sin downtime, asegurando continuidad del servicio.
- **Distribución de cargas de trabajo**: Los nodos secundarios pueden manejar lecturas, distribuyendo así la carga y mejorando el rendimiento.

### ¿Qué recursos adicionales necesitas?

Para una gestión más avanzada, descarga MongoDB Compass, la herramienta oficial que permite interactuar con tu base de datos desde tu sistema operativo. Esto es útil para realizar operaciones más complejas que pueden estar limitadas en el navegador.

**Lecturas recomendadas**

[MongoDB Atlas Database | Multi-Cloud Database Service | MongoDB](https://www.mongodb.com/cloud/atlas)

[MongoDB Compass | MongoDB](https://www.mongodb.com/products/compass)

[MongoDB Documentation](https://docs.mongodb.com/)

## Creación de Documentos en MongoDB

En MongoDB, los datos se almacenan en **documentos BSON (Binary JSON)** dentro de **colecciones**, similares a las tablas en bases de datos SQL. Veamos cómo crear documentos en MongoDB paso a paso.  

### 🔹 **1. Insertar Documentos con `insertOne()`**
Para insertar un solo documento en una colección, usamos `insertOne()`.  
📌 **Ejemplo**: Insertar un usuario en la colección `usuarios`.  
```sh
db.usuarios.insertOne({
    nombre: "Mario Vargas",
    edad: 30,
    ciudad: "Bogotá"
})
```
🛠️ **Si la colección `usuarios` no existe, MongoDB la creará automáticamente.**  

### 🔹 **2. Insertar Múltiples Documentos con `insertMany()`**
Cuando necesitas agregar varios documentos de una sola vez, usa `insertMany()`.  

📌 **Ejemplo**: Insertar varios usuarios en la colección `usuarios`.  
```sh
db.usuarios.insertMany([
    { nombre: "Ana Pérez", edad: 25, ciudad: "Medellín" },
    { nombre: "Carlos López", edad: 35, ciudad: "Cali" },
    { nombre: "Laura Gómez", edad: 28, ciudad: "Barranquilla" }
])
```
Cada documento insertado tendrá un campo `_id` único generado automáticamente.  

### 🔹 **3. Insertar Documentos con un `_id` Personalizado**  
Si deseas definir manualmente un `_id`, puedes hacerlo al insertar el documento.  

📌 **Ejemplo**:  
```sh
db.usuarios.insertOne({
    _id: 1001,
    nombre: "Sofía Mendoza",
    edad: 27,
    ciudad: "Cartagena"
})
```
⚠️ **Si intentas insertar otro documento con el mismo `_id`, MongoDB generará un error de duplicado.**  

### 🔹 **4. Insertar Documentos en MongoDB con Python (`pymongo`)**
Si trabajas con Python, usa la librería `pymongo` para insertar datos en MongoDB.  

📌 **Ejemplo en Python**:  
```python
from pymongo import MongoClient

# Conectar a MongoDB
cliente = MongoClient("mongodb://localhost:27017/")
db = cliente["mi_base_de_datos"]
coleccion = db["usuarios"]

# Insertar un solo documento
coleccion.insert_one({"nombre": "Luis", "edad": 28})

# Insertar múltiples documentos
coleccion.insert_many([
    {"nombre": "Carla", "edad": 23, "ciudad": "Medellín"},
    {"nombre": "Jorge", "edad": 31, "ciudad": "Bogotá"}
])

# Mostrar documentos insertados
for usuario in coleccion.find():
    print(usuario)
```

## 🎯 **Resumen**
✅ `insertOne()` → Inserta un solo documento.  
✅ `insertMany()` → Inserta múltiples documentos.  
✅ Puedes definir un `_id` personalizado.  
✅ MongoDB crea la colección automáticamente si no existe.  
✅ Puedes insertar documentos desde Python con `pymongo`.

### Resumen

MongoDB Compass es una herramienta esencial para gestionar bases de datos en la nube y en tu computadora local.

### ¿Cómo crear un usuario en MongoDB Atlas?

- Accede a Atlas y navega a Database Access.
- Genera un nuevo usuario con el nombre deseado, como “Demo Test”.
- Asigna una contraseña segura.
- Define los roles del usuario, como “Atlas Admin” para bases de datos de prueba.
- Guarda la contraseña para la URI de conexión.

### ¿Cómo configurar el acceso a la red?

- En la sección de red, asegúrate de agregar las IPs de tus servidores y computadoras.
- Añade tu IP actual para asegurar que sólo dispositivos autorizados puedan acceder a la base de datos.
- Configura las IPs adicionales según tus necesidades, como servidores locales o instancias en la nube.

### ¿Cómo conectar MongoDB Compass a tu base de datos?

- Abre Compass y selecciona la opción de conexión.
- Usa la URI proporcionada por Atlas, que incluye el prefijo MongoDB+SRV, el nombre de usuario y la contraseña.
- Guarda y conecta para acceder a tu base de datos.

### ¿Cómo crear y gestionar colecciones en MongoDB Compass?

- Navega a la opción de Database y haz clic en Connect.
- Crea una nueva base de datos, por ejemplo, “mi red social”, y una colección llamada “users”.
- Añade datos manualmente o importa desde un archivo JSON o CSV.

### ¿Cómo insertar datos en MongoDB Compass?

- Inserta documentos directamente en la interfaz, utilizando el formato JSON.
- Agrega campos como nombre, apellido, roles y fechas.
- La flexibilidad de NoSQL permite insertar documentos con diferentes campos sin restricciones estrictas.

### ¿Cómo usar el intérprete de JavaScript en MongoDB?

- MongoDB ofrece un intérprete de JavaScript para ejecutar código directamente en la base de datos.
- Inserta documentos de manera programática usando comandos como `insertOne`.
- Aprovecha la capacidad de crear funciones dinámicas y realizar operaciones complejas con JavaScript.

### ¿Qué ventajas ofrece MongoDB para startups?

- Flexibilidad en el esquema de datos, permitiendo cambios rápidos según las necesidades del negocio.
- Facilidad para almacenar datos de diversos dispositivos con diferentes variables.
- Eficiencia en la creación y gestión del backend gracias a la naturaleza no estructurada de NoSQL.

### Recursos técnicos

- [MongoDB Atlas Documentation](https://docs.mongodb.com/manual/)
- [Compass User Guide](https://docs.mongodb.com/compass/master/)
- [JavaScript for MongoDB](https://developer.mongodb.com/learn/?content=javascript)

### Nombres SEO para la clase

- Conexión y Gestión de Bases de Datos con MongoDB Compass
- Cómo Configurar Usuarios y Accesos en MongoDB Atlas
- Inserción de Datos y Uso del Intérprete de JavaScript en MongoDB

**Lecturas recomendadas**

[What is MongoDB Compass? - MongoDB Compass](https://www.mongodb.com/docs/compass/master/)

[JavaScript & MongoDB | Support & Compatibility | MongoDB | MongoDB](https://www.mongodb.com/resources/languages/javascript)

## Uso de la Consola de MongoDB: Creación de Datos con insertOne e insertMany

En la consola de **MongoDB (`mongosh`)**, puedes insertar documentos en una colección usando los métodos `insertOne` e `insertMany`.

### **📌 Insertar un solo documento con `insertOne`**
El método `insertOne` permite agregar un **único documento** a una colección.

### **Ejemplo:**
```sh
db.usuarios.insertOne({
    "nombre": "Juan",
    "apellido": "Pérez",
    "rol": ["admin", "developer"],
    "fechaNac": ISODate("1990-02-14T00:00:00.000Z"),
    "edad": 34
})
```
✅ **Salida esperada**:
```json
{
  "acknowledged": true,
  "insertedId": ObjectId("64a9b1234c5d6e7f8a9b0123")
}
```
🔹 **MongoDB genera automáticamente un `_id`** si no se especifica.

### **📌 Insertar múltiples documentos con `insertMany`**
El método `insertMany` permite agregar **varios documentos** al mismo tiempo en una colección.

### **Ejemplo:**
```sh
db.usuarios.insertMany([
    {
        "nombre": "Laura",
        "apellido": "Gómez",
        "rol": ["editor", "marketing"],
        "fechaNac": ISODate("1995-05-20T00:00:00.000Z"),
        "edad": 28
    },
    {
        "nombre": "Pedro",
        "apellido": "López",
        "rol": ["designer", "content creator"],
        "fechaNac": ISODate("1988-09-12T00:00:00.000Z"),
        "edad": 35
    }
])
```
✅ **Salida esperada**:
```json
{
  "acknowledged": true,
  "insertedIds": {
    "0": ObjectId("64a9b1234c5d6e7f8a9b0124"),
    "1": ObjectId("64a9b1234c5d6e7f8a9b0125")
  }
}
```

### **🔹 Consideraciones Importantes**
- **`insertOne`** solo permite insertar **un documento a la vez**.
- **`insertMany`** permite insertar **varios documentos a la vez**.
- MongoDB genera un **`_id` automáticamente** si no lo especificas.
- **`ISODate()`** solo se usa dentro de `mongosh`; si insertas desde otro cliente, usa **strings de fecha en formato ISO 8601**.

### **🛠️ Verificar la Inserción**
Después de insertar los datos, puedes ver los documentos almacenados con:
```sh
db.usuarios.find().pretty()
```

🚀 **¡Ahora ya sabes cómo insertar datos en MongoDB desde la consola!** 🎯

### Resumen

Explora las poderosas capacidades de inserción en MongoDB y cómo maximizar su eficiencia. Aprenderás a usar comandos para insertar datos de manera efectiva y entenderás las implicaciones en el rendimiento del clúster.

### ¿Cómo insertar documentos en MongoDB desde la terminal de Compass? 

En Compass, accede a la terminal `mongush` para insertar comandos de MongoDB. Puedes usar el intérprete de JavaScript, lo que te permite crear variables y funciones para generar datos aleatorios. Primero, selecciona la base de datos con use, luego usa `db.nombre_de_tu_colección.insert` para insertar documentos.

### ¿Qué ventajas ofrece la flexibilidad de la estructura de documentos en MongoDB?

MongoDB permite insertar documentos con diferentes estructuras, una gran ventaja para startups o proyectos con cambios frecuentes en el esquema de datos. Por ejemplo, puedes tener un documento con un campo `name` y otro con `nombre`, permitiendo adaptaciones rápidas sin complicaciones.

### ¿Cómo insertar múltiples documentos simultáneamente?

Usa `insertMany` para insertar varios documentos a la vez. Esto se logra creando un array de objetos JSON:

```json
db.users.insertMany([
  { name: "Raúl", role: "admin" },
  { name: "Samanta", age: 25 }
]);
```

### ¿Qué consideraciones de rendimiento debes tener al insertar documentos?

Cada escritura afecta el rendimiento del disco (IOPS). Insertar documentos uno por uno puede generar cuellos de botella. Es mejor usar `insertMany` para insertar varios documentos de una vez, optimizando las operaciones de entrada y salida.

### ¿Cómo crear una función en JavaScript para insertar documentos aleatorios?

Puedes crear una función en JavaScript para automatizar la inserción de documentos con datos aleatorios:

```java
function insertRandomUsers() {
  const roles = ["admin", "user", "editor"];
  const names = ["Ana", "Raúl", "Samanta", "Carlos"];
  const users = [];

  for (let i = 0; i < 10; i++) {
    const user = {
      name: names[Math.floor(Math.random() * names.length)],
      role: roles[Math.floor(Math.random() * roles.length)],
      age: Math.floor(Math.random() * 50) + 20
    };
    users.push(user);
  }

  db.users.insertMany(users);
}
insertRandomUsers();
```

**Lecturas recomendadas**

[db.collection.insert() - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/reference/method/db.collection.insert/)

## Eliminar Documentos en MongoDB

MongoDB permite eliminar documentos de una colección utilizando los métodos `deleteOne` y `deleteMany`. Aquí te explico cómo usarlos correctamente.

### **📌 1. Eliminar un solo documento con `deleteOne`**
Este método elimina **el primer documento** que coincida con el filtro.

### **Ejemplo:**
```sh
db.usuarios.deleteOne({ "nombre": "Laura" })
```
✅ **Salida esperada**:
```json
{ "acknowledged": true, "deletedCount": 1 }
```
🔹 Si no se encuentra ningún documento con el criterio dado, `deletedCount` será `0`.

### **📌 2. Eliminar múltiples documentos con `deleteMany`**
Este método elimina **todos los documentos** que coincidan con el filtro.

### **Ejemplo:**
```sh
db.usuarios.deleteMany({ "rol": "marketing" })
```
✅ **Salida esperada**:
```json
{ "acknowledged": true, "deletedCount": 2 }
```
🔹 Todos los documentos que tengan `"rol": "marketing"` serán eliminados.

### **📌 3. Eliminar todos los documentos de una colección**
⚠️ **¡Cuidado!** Esto eliminará **todos los documentos** en la colección.

```sh
db.usuarios.deleteMany({})
```
✅ **Salida esperada**:
```json
{ "acknowledged": true, "deletedCount": 10 }
```
🔹 Borra **todos los documentos** de la colección `usuarios`, pero **la colección sigue existiendo**.

### **📌 4. Eliminar una colección completa**
Si quieres eliminar **una colección entera**, usa:

```sh
db.usuarios.drop()
```
✅ **Salida esperada**:
```json
true
```
🔹 Esto **elimina por completo la colección `usuarios`**, incluyendo su estructura.

### **🛠️ Verificar la Eliminación**
Después de eliminar documentos, puedes verificar con:
```sh
db.usuarios.find().pretty()
```
Si la colección está vacía, **no mostrará resultados**.

🚀 **¡Ahora ya sabes cómo eliminar documentos en MongoDB!**  

### Resumen

Eliminar documentos en MongoDB puede parecer una tarea intimidante, pero con los pasos adecuados, se puede realizar de manera eficiente tanto desde la interfaz gráfica como desde la terminal.

### ¿Cómo eliminar documentos desde la interfaz gráfica?

Para eliminar documentos desde la interfaz gráfica de MongoDB:

- Coloca el cursor dentro del documento que deseas eliminar.
- Haz clic en el último botón, que es “remove document”.
- Confirma la acción haciendo clic en “delete”.

¡Y listo! El documento se ha eliminado de la base de datos.

### ¿Cómo eliminar documentos desde la terminal?

Eliminar documentos desde la terminal es un poco más complejo, pero sigue siendo bastante manejable con los siguientes pasos.

1. **Seleccionar la base de datos**:

 - Asegúrate de que estás trabajando en la base de datos correcta utilizando el comando `use`.
 
2. **Eliminar un solo documento**:

 - Usa el comando `deleteOne`.
 - Necesitas especificar un filtro que coincida con el documento que deseas eliminar. Por ejemplo:
`db.users.deleteOne({_id: ObjectId("tu_object_id")});`

 - Este comando eliminará el documento que coincida con el ObjectId especificado.
 
3. Eliminar múltiples documentos:

 - Usa el comando `deleteMany`.
 - Especifica un filtro para los documentos que deseas eliminar. Por ejemplo, para eliminar todos los documentos donde la edad es menor de 30 años:
`db.users.deleteMany({edad: {$lt: 30}});`
 - Este comando eliminará todos los documentos que coincidan con el filtro.
 
### ¿Qué precauciones debo tomar al eliminar documentos?

- **Verificar los filtros**: Siempre revisa cuidadosamente los filtros que estás utilizando para evitar eliminar documentos incorrectos.
- **Pruebas previas**: Si es posible, prueba tus comandos en una copia de la base de datos antes de ejecutarlos en la base de datos principal.
- **Backups**: Mantén copias de seguridad actualizadas para poder restaurar la base de datos en caso de errores.

### ¿Cómo puedo practicar la eliminación de documentos?

Te reto a que encuentres un patrón común en los documentos que te quedan, como documentos que comienzan con una letra específica o tienen un campo específico. Utiliza filtros compuestos para practicar la eliminación de una gran cantidad de documentos de manera segura.

**Eliminar por id**
`db.users.deleteOne({_id: ObjectId('67a92a55eda20b289d582fe7')})`

**Eliminar barios $gte = mayor que y $lte = menor que**
`db.users.deleteMany({edad:{$lte: 30}})`


**Lecturas recomendadas**

[Delete Documents - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/tutorial/remove-documents/)

[Query and Projection Operators - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/reference/operator/query/)

## Cómo Leer Documentos en MongoDB con find()

En MongoDB, el método `find()` se usa para buscar documentos en una colección.

### **1. Mostrar Todos los Documentos**
Para obtener **todos los documentos** de la colección `users`:  
```javascript
db.users.find()
```
🔹 **Esto devolverá todos los documentos**, pero en un formato poco legible.  

Para formatearlo de manera más clara:
```javascript
db.users.find().pretty()
```
🔹 **Esto mostrará los documentos con indentación y formato JSON**.

### **2. Buscar Documentos con un Filtro**  
Si quieres encontrar usuarios con el nombre `"Juan"`:
```javascript
db.users.find({ nombre: "Juan" })
```
🔹 **Esto mostrará todos los documentos donde `"nombre"` sea `"Juan"`**.

### **3. Buscar un Solo Documento con `findOne()`**
Si solo necesitas **un documento**, usa:
```javascript
db.users.findOne({ nombre: "Juan" })
```
🔹 **Esto traerá el primer documento** que cumpla la condición.

### **4. Buscar con Múltiples Condiciones**
Si quieres buscar un usuario llamado `"Juan"` con el rol `"admin"`:
```javascript
db.users.find({ nombre: "Juan", rol: "admin" }).pretty()
```
🔹 **Esto devolverá solo los documentos que cumplan ambas condiciones**.

### **5. Buscar con Operadores**
#### **Mayor que (`$gt`) y Menor que (`$lt`)**
Buscar personas mayores de 30 años:
```javascript
db.users.find({ edad: { $gt: 30 } })
```
Buscar personas menores de 25 años:
```javascript
db.users.find({ edad: { $lt: 25 } })
```

#### **Buscar con `$or` (Cualquiera de las condiciones)**
```javascript
db.users.find({ $or: [{ nombre: "Juan" }, { rol: "admin" }] })
```
🔹 **Esto traerá los documentos donde el nombre sea "Juan" o el rol sea "admin"**.

### **6. Buscar Solo Algunos Campos (`Projection`)**
Si solo quieres ver los nombres y roles sin mostrar los `_id`, usa:
```javascript
db.users.find({}, { nombre: 1, rol: 1, _id: 0 })
```
🔹 **Esto devolverá solo los campos `nombre` y `rol`**.

### Resumen

Consultar datos en MongoDB puede ser más fácil o intuitivo dependiendo de tu experiencia previa, y para facilitar este proceso utilizaremos MongoDB Atlas y Compass. En Atlas, puedes cargar un conjunto de datos de muestra que te permitirá experimentar sin problemas. Luego, puedes realizar consultas tanto en la terminal como en la interfaz de Compass.

### ¿Cómo cargar datos de muestra en MongoDB Atlas?

Para poblar nuestra base de datos con datos uniformes y variados:

- Ve al panel de Atlas de MongoDB.
- En tu clúster, haz clic en los tres puntos y selecciona “Load Sample Dataset”.
- Espera a que se carguen las bases de datos y colecciones de muestra.

### ¿Cómo consultar datos en la terminal de MongoDB?

Para realizar consultas desde la terminal:

- Usa la base de datos: `use sample_mflix`.
- Consulta una colección con `db.Movies.findOne()` para obtener un ejemplo.
- Usa `db.Movies.find({type: "movie"})` para filtrar por tipo de documento.

### ¿Cómo realizar consultas avanzadas en MongoDB?

Puedes aplicar filtros adicionales:

- Filtra películas lanzadas después del año 2000:

`db.Movies.find({type: "movie", release_date: {$gte: new Date("2000-01-01")}})`

- Usa `it` para iterar sobre los resultados.

### ¿Cómo utilizar Compass para consultas en MongoDB?

En Compass:

- Selecciona la colección y ve los documentos.
- Realiza una consulta similar a la terminal:
`{type: "movie"}`
- Añade filtros como `release_date: {$gte: new Date("2000-01-01")}`.

### ¿Cómo mostrar y ordenar campos específicos en Compass?

Para mostrar campos específicos:

- Usa “Project” para seleccionar campos, por ejemplo, solo el título de la película:
`{title: 1}`
- Ordena por fecha de lanzamiento en orden descendente:
`{release_date: -1}`

### ¿Cómo generar consultas con inteligencia artificial en Compass?

Compass incorpora IA para generar consultas:

- Escribe un prompt como “Give me type movies released after 2000 with awards”.
- La IA generará la consulta y podrás ejecutarla para obtener resultados precisos.

## Consultas Avanzadas en MongoDB: Dominando el Framework de Agregación

MongoDB proporciona el **Framework de Agregación** (`aggregate()`) para realizar consultas avanzadas, procesar grandes volúmenes de datos y obtener información específica mediante múltiples etapas (`stages`).  

### **1. Estructura Básica del Framework de Agregación**  
El método `aggregate()` permite aplicar varias etapas secuenciales de procesamiento a los documentos de una colección.  

```javascript
db.users.aggregate([
    { /* Stage 1 */ },
    { /* Stage 2 */ },
    { /* Stage 3 */ }
])
```

Cada **etapa** transforma los datos y los pasa a la siguiente.  


### **2. Principales Etapas (`Stages`) en Agregación**  

### **🔹 `$match` → Filtrar documentos (Similar a `find()`)**  
```javascript
db.users.aggregate([
    { $match: { rol: "admin" } }
])
```
🔹 **Filtra solo los documentos donde `rol` sea `"admin"`**.  

### **🔹 `$group` → Agrupar datos y calcular valores**  
```javascript
db.users.aggregate([
    {
        $group: {
            _id: "$rol",
            totalUsuarios: { $sum: 1 }
        }
    }
])
```
🔹 **Agrupa por el campo `rol` y cuenta cuántos usuarios hay por cada rol**.  

### **🔹 `$sort` → Ordenar documentos**  
```javascript
db.users.aggregate([
    { $sort: { edad: -1 } } // Ordena de mayor a menor edad
])
```
🔹 **Ordena los usuarios por `edad` en orden descendente** (`-1` = descendente, `1` = ascendente).  

### **🔹 `$project` → Seleccionar y modificar campos**  
```javascript
db.users.aggregate([
    {
        $project: {
            _id: 0, // Ocultar el campo _id
            nombreCompleto: { $concat: ["$nombre", " ", "$apellido"] },
            edad: 1
        }
    }
])
```
🔹 **Crea un nuevo campo `nombreCompleto` concatenando `nombre` y `apellido`** y solo muestra `edad`.  

### **🔹 `$lookup` → Unir colecciones (Equivalente a `JOIN` en SQL)**  
```javascript
db.users.aggregate([
    {
        $lookup: {
            from: "pedidos",  // Colección a unir
            localField: "_id", // Campo en `users`
            foreignField: "userId", // Campo en `pedidos`
            as: "historialPedidos"
        }
    }
])
```
🔹 **Une la colección `pedidos` con `users` donde `_id` de `users` coincida con `userId` en `pedidos`**.

### **🔹 `$unwind` → Expandir arrays en múltiples documentos**  
Si cada usuario tiene varios roles en un array, `$unwind` permite tratarlos como documentos individuales.
```javascript
db.users.aggregate([
    { $unwind: "$rol" }
])
```
🔹 **Convierte un solo documento con múltiples roles en varios documentos, cada uno con un rol**.

### **🔹 `$limit` y `$skip` → Paginación de resultados**  
```javascript
db.users.aggregate([
    { $sort: { edad: -1 } }, // Ordenar por edad descendente
    { $skip: 10 }, // Saltar los primeros 10 registros
    { $limit: 5 } // Tomar solo 5 registros
])
```
🔹 **Útil para paginación de resultados**.

### **Ejemplo Completo: Análisis de Usuarios**  
Supongamos que queremos:  
1️⃣ Filtrar usuarios con más de 30 años  
2️⃣ Agruparlos por rol y contar cuántos hay  
3️⃣ Ordenarlos de mayor a menor cantidad  
4️⃣ Mostrar solo los primeros 3 resultados  

```javascript
db.users.aggregate([
    { $match: { edad: { $gt: 30 } } }, 
    { $group: { _id: "$rol", totalUsuarios: { $sum: 1 } } }, 
    { $sort: { totalUsuarios: -1 } }, 
    { $limit: 3 }
])
```
🔹 **Consulta avanzada que permite obtener análisis de datos más detallados**.  

### **Conclusión**  
El Framework de Agregación de MongoDB es extremadamente poderoso para **consultas avanzadas, transformación de datos y análisis en tiempo real**.  

### Resumen

MongoDB ofrece un framework de agregación poderoso y flexible, conocido como aggregation framework, que permite realizar consultas complejas mediante la combinación de diferentes etapas en pipelines.

### ¿Qué es el aggregation framework?

El aggregation framework de MongoDB permite realizar operaciones avanzadas en los datos utilizando una serie de etapas en una tubería, conocida como pipeline. Cada etapa procesa los datos y los pasa a la siguiente, similar a cómo el agua fluye a través de una tubería con válvulas que se abren y cierran.

### ¿Cómo se configura una etapa de agregación?

1. **Iniciar Aggregation en MongoDB Compass**:

 - Abre MongoDB Compass y selecciona “Aggregations”.
 - Verás un resumen de documentos sin ninguna consulta aplicada.
 
2. **Agregar una etapa Matc**h:

 - Selecciona “Match” como la primera etapa para filtrar documentos, similar a la operación find.
 - La eficiencia mejora al filtrar grandes cantidades de documentos primero.
 
3. **Ejecutar una consulta Match**:

 - Copia una query existente y pégala en la etapa de match.
 - Los resultados se mostrarán automáticamente a la derecha.
 
### ¿Cómo se utilizan otras etapas en el pipeline?

1. **Agregar una etapa Project**:

 - Selecciona “Project” para mostrar solo ciertos campos como `title` y `release`.
 - La salida de esta etapa se convierte en la entrada de la siguiente.

2. **Insertar una etapa Group**:

 - Selecciona “Group” para agrupar documentos, similar a `Group By` en SQL.
 - Por ejemplo, agrupa por año y calcula el promedio de la duración de las películas (`runtime`).
 
3. **Ordenar los resultados**:

 - Añade una etapa “Sort” para ordenar los resultados por el campo calculado, como el promedio de duración (average).
 
### ¿Cómo se manejan las etapas en el pipeline?

- Las etapas se pueden reordenar y modificar según sea necesario.
- Cada cambio en una etapa afecta la entrada de las siguientes etapas.
- Es posible eliminar y añadir nuevas etapas para ajustar el flujo de datos.

### ¿Cómo exportar y editar el pipeline?

- MongoDB Compass permite exportar el pipeline a diferentes lenguajes de programación.
- Puedes editar el pipeline como texto para modificarlo de manera más sencilla.

### ¿Qué retos existen al usar el aggregation framework?

- Explorar las diferentes etapas disponibles en el aggregation framework.
- Realizar operaciones como Joins para combinar colecciones, por ejemplo, unir comentarios a películas utilizando el identificador de película en la colección `Comments`.

**Lecturas recomendadas**

[Aggregation Operations - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/aggregation/)

[Create an Aggregation Pipeline - MongoDB Compass](https://www.mongodb.com/docs/compass/current/create-agg-pipeline/)

## Cómo Eliminar Datos en MongoDB

En MongoDB, puedes eliminar documentos de una colección utilizando varios métodos, como `deleteOne()`, `deleteMany()` y `remove()`. Aquí te explico cada uno de ellos:

### 1. **Eliminar un solo documento con `deleteOne()`**
Este método elimina el primer documento que coincide con el filtro proporcionado.

**Sintaxis**:
```javascript
db.collection.deleteOne({ <filtro> })
```

**Ejemplo**:
```javascript
db.users.deleteOne({ "nombre": "Carlos" })
```
Este comando eliminará el primer documento donde el campo `nombre` sea igual a "Carlos".

### 2. **Eliminar múltiples documentos con `deleteMany()`**
Este método elimina todos los documentos que coincidan con el filtro.

**Sintaxis**:
```javascript
db.collection.deleteMany({ <filtro> })
```

**Ejemplo**:
```javascript
db.users.deleteMany({ "edad": { $lt: 30 } })
```
Este comando eliminará todos los documentos donde el campo `edad` sea menor a 30.

### 3. **Eliminar todos los documentos con `deleteMany()` sin filtro**
Si deseas eliminar todos los documentos de una colección, puedes usar `deleteMany()` sin un filtro.

**Sintaxis**:
```javascript
db.collection.deleteMany({})
```

**Ejemplo**:
```javascript
db.users.deleteMany({})
```
Este comando eliminará **todos** los documentos de la colección `users`.

### 4. **Usar `remove()`** *(obsoleto en versiones más recientes)*
Este método también permite eliminar documentos, pero está en desuso. En lugar de `remove()`, se recomienda utilizar `deleteOne()` o `deleteMany()`.

**Sintaxis**:
```javascript
db.collection.remove({ <filtro> })
```

**Ejemplo**:
```javascript
db.users.remove({ "nombre": "Andrea" })
```
Este comando eliminaría todos los documentos en los que el campo `nombre` sea igual a "Andrea".

### Consideraciones:
- **Precaución**: Eliminar datos es una acción irreversible. Siempre asegúrate de que el filtro esté correctamente especificado para evitar eliminar datos accidentalmente.
- **Uso de `find()` antes de eliminar**: Si no estás seguro de qué documentos se eliminarán, es una buena práctica hacer primero una consulta `find()` para verificar los documentos que coinciden con tu filtro.

**Ejemplo**:
```javascript
db.users.find({ "edad": { $lt: 30 } })
```

Esto te permitirá ver qué documentos serán eliminados antes de ejecutar `deleteMany()` o `deleteOne()`.

### Operaciones avanzadas de reemplazo en MongoDB

En MongoDB, las operaciones avanzadas de **reemplazo** permiten actualizar completamente un documento con un nuevo conjunto de datos, manteniendo el mismo `_id`. Se utiliza principalmente el método `replaceOne()`.

### 🔹 **Método `replaceOne()`**
Este método reemplaza **un solo documento** que cumpla con un criterio de búsqueda.

📌 **Sintaxis:**
```javascript
db.coleccion.replaceOne(
  { filtro },        // Documento de búsqueda
  { nuevoDocumento } // Documento de reemplazo
)
```
📌 **Ejemplo:**
```javascript
db.usuarios.replaceOne(
  { nombre: "Carlos" },
  {
    nombre: "Carlos",
    apellido: "Ramírez",
    edad: 30,
    rol: ["admin", "developer"]
  }
)
```
⚠️ **Nota:** Este método **reemplaza completamente** el documento, eliminando cualquier campo que no esté en el nuevo documento.

### 🔹 **Diferencia con `updateOne()`**
Mientras `replaceOne()` **sustituye** el documento entero, `updateOne()` **modifica solo campos específicos**.

📌 **Ejemplo con `updateOne()`:**
```javascript
db.usuarios.updateOne(
  { nombre: "Carlos" },
  { $set: { edad: 31 } }  // Solo modifica el campo "edad"
)
```
🔹 **Casos de uso de `replaceOne()`**  
✅ Cuando deseas sobrescribir por completo un documento.  
✅ Cuando quieres mantener el `_id` pero actualizar todos los datos.  

🔹 **Casos de uso de `updateOne()`**  
✅ Cuando solo necesitas modificar algunos campos del documento.

### Resumen

La operación de reemplazo en MongoDB permite sustituir el contenido completo de un documento. Esto es útil cuando se necesita modificar todo el documento en lugar de solo algunos campos. Aunque existen varias funciones para actualizar documentos, cada una tiene su propósito específico y utiliza los recursos del clúster de manera distinta.

### ¿Cómo se realiza la operación de reemplazo en MongoDB?

- Utiliza replaceOne para reemplazar un documento completo.
- El primer argumento es el filtro, usualmente el ObjectId.
- El segundo argumento es el nuevo documento que reemplazará al existente.

**Ejemplo:**

```shell
db.customers.replaceOne(
  { _id: ObjectId("identificador_del_documento") },
  { username: "elitry" }
);
```

Este ejemplo elimina todos los campos excepto `username` y lo reemplaza con el valor especificado.

### ¿Qué hacer si no se encuentra un documento?

- Utiliza la opción upsert: true.
- Si el filtro no encuentra un documento, se crea uno nuevo con los valores proporcionados.

**Ejemplo:**

```shell
db.customers.replaceOne(
  { username: "xyz" },
  { username: "xyz", name: "Andrés", email: "test@gmail.com" },
  { upsert: true }
);
```

Este comando crea un nuevo documento si no encuentra uno existente con el `username` especificado.

### ¿Cómo usar Find and Modify para reemplazar documentos?

`findAndModify` no solo actualiza el documento, sino que también devuelve el documento antes o después de la modificación.

**Ejemplo:**

```shell
db.customers.findAndModify({
  query: { username: "lintco1" },
  update: { username: "lintc", name: "Catherine Davis" },
  new: true
});
```

Este comando reemplaza el documento y puede configurarse para mostrar el documento modificado.

### ¿Cuáles son las diferencias clave entre Update y Replace?

- `updateOne` se usa para modificar ciertos campos del documento.
- `updateMany` se usa para modificar múltiples documentos que cumplen con un filtro.
- `replaceOne` reemplaza el documento completo.
- `findAndModify` realiza dos operaciones: actualización y retorno del documento modificado.

### ¿Qué consideraciones tener al escalar?

- Las operaciones atómicas como updateOne son preferibles para minimizar el uso de CPU y disco.
- `findAndModif`y puede ser menos eficiente debido a las operaciones adicionales que realiza.

**Querys d ela clase**

`db.customers.replaceOne({_id: ObjectId('5ca4bbcea2dd94ee58162a78')},{username:"eliray"})` = remplasa el username
`db.customers.replaceOne({username: "xyz"},{username:"xyz", name:"Andres", email: "test@hotmail.com"},{upsert: true})` = Remplaza varios elementos con el username
`db.customers.findAndModify({query:{username: "taylorbullock"}, update:{username:"Lindc", name: "Shirley Rodriguez"}})` = cambia el nombre del username

**Lecturas recomendadas**

[db.collection.replaceOne() - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/reference/method/db.collection.replaceOne/)

[db.collection.findAndModify() - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/reference/method/db.collection.findAndModify/)

## Cómo Actualizar Documentos en MongoDB

En MongoDB, puedes actualizar documentos utilizando los métodos `updateOne()`, `updateMany()` y `replaceOne()`. Aquí tienes ejemplos de cómo usarlos:  

### 🔹 **Actualizar un solo documento con `updateOne()`**  
Este método actualiza el **primer documento** que coincida con la consulta.  

```js
db.usuarios.updateOne(
  { nombre: "Carlos" },  // Filtro para encontrar el documento
  { $set: { edad: 30 } } // Modificación en el campo "edad"
)
```

### 🔹 **Actualizar múltiples documentos con `updateMany()`**  
Si deseas modificar varios documentos que cumplan la condición:  

```js
db.usuarios.updateMany(
  { rol: "developer" },  // Filtro
  { $set: { status: "activo" } }  // Cambia el campo "status"
)
```

### 🔹 **Reemplazar un documento con `replaceOne()`**  
Este método **sustituye completamente** el documento encontrado por uno nuevo:  

```js
db.usuarios.replaceOne(
  { nombre: "Andrea" },  // Filtro para encontrar el documento
  { nombre: "Andrea", apellido: "González", edad: 37, rol: ["admin"] }  // Nuevo documento
)
```

### 🔹 **Actualizar un campo incrementando su valor**  
Si quieres aumentar en **5** años la edad de todos los usuarios:  

```js
db.usuarios.updateMany({}, { $inc: { edad: 5 } })
```

### 🔹 **Añadir elementos a un array dentro del documento**  
Si deseas agregar un nuevo rol a un usuario:  

```js
db.usuarios.updateOne(
  { nombre: "Sofía" },
  { $push: { rol: "project manager" } }
)
```

### 🔹 **Eliminar un campo dentro de un documento**  
Para eliminar el campo `status` de todos los usuarios:  

```js
db.usuarios.updateMany({}, { $unset: { status: "" } })
```

🔹 **📌 Nota:**  
- **`$set`** → Modifica o agrega campos nuevos.  
- **`$inc`** → Incrementa valores numéricos.  
- **`$push`** → Agrega elementos a un array.  
- **`$unset`** → Elimina un campo.

### Resumen

Modificar o actualizar documentos en MongoDB es esencial para manejar datos dinámicos. Aunque existen varios métodos para realizar estas acciones, los más comunes son `update` y `replace`. Cada uno se utiliza en diferentes escenarios y es crucial entender sus diferencias y aplicaciones.

### ¿Cómo se utiliza `update` en MongoDB?

#### ¿Qué es update?

El método `update` permite modificar ciertos valores de un documento que cumplen con un filtro específico. Existen variantes como `updateOne` y `updateMany`.

### ¿Cómo usar `updateOne`?

Para actualizar un solo documento en MongoDB, se utiliza `updateOne`. Este método requiere un filtro para identificar el documento y el operador `$set` para especificar los cambios. Por ejemplo, para cambiar el nombre de un cliente:

```shell
db.customers.updateOne(
  { _id: ObjectId("5f3e5a3a29f1e8b7c2c69d62") },
  { $set: { name: "Elizabeth" } }
);
```
Este comando busca el documento con el `_id `especificado y actualiza el campo `name`.

### ¿Cómo usar `updateMany`?

Para actualizar múltiples documentos que cumplen con un criterio, se usa `updateMany`. Este método también requiere un filtro y los cambios a realizar:

```shell
db.customers.updateMany(
  { birthYear: { $gte: 1990 } },
  { $set: { membership: "Platinum" } }
);
```

En este ejemplo, todos los documentos donde `birthYear` es mayor o igual a 1990 se actualizarán para incluir el campo `membership` con el valor `Platinum`.

### ¿Qué es `replace` en MongoDB?

#### ¿Cómo funciona replaceOne?

El método `replaceOne` reemplaza un documento completo excepto su identificador. Esto es útil cuando se necesita reestructurar un documento:

```shell
db.customers.replaceOne(
  { _id: ObjectId("5f3e5a3a29f1e8b7c2c69d62") },
  { name: "John Doe", age: 30, city: "New York" }
);
```

Este comando reemplaza el documento identificado por `_id` con uno nuevo que tiene los campos `nam`e, `age` y `city`.

### ¿Qué otras alternativas existen a `update` y `replace`?

Además de `update` y `replace`, MongoDB ofrece otros métodos para la manipulación de datos, como `bulkWrite`, que permite realizar múltiples operaciones en una sola llamada, y `findAndModify`, que devuelve y modifica documentos en un solo paso.

### Ejercicio Práctico

Para seguir practicando, intenta el siguiente reto: en la base de datos de Airbnb, actualiza todos los apartamentos con menos de tres habitaciones restando 10 al precio. Esto te ayudará a aplicar los conceptos aprendidos.

```shell
db.airbnb.updateMany(
  { bedrooms: { $lt: 3 } },
  { $inc: { price: -10 } }
);
```

**Querys**

`db.customers.updateOne({"_id": ObjectId('5ca4bbcea2dd94ee58162a6a')},{$set:{name: "Camilo Torres"}})` 

`db.customers.updateOne({"_id": ObjectId('5ca4bbcea2dd94ee58162a6a')},{$set:{name: "Camilo Torres", plus: true}})` = Agrega plus 

`db.customers.updateOne({"_id": ObjectId('5ca4bbcea2dd94ee58162a6a')},{$unset:{plus: true}})` = Elimina el plus

`db.customers.updateMany({ birthdate: { $gte: new Date("1990-01-01T00:00:00Z")}}, { $set: { platinum: true }})` = se crea para los que nacieron despues del año se crea un platino

``db.customers.updateMany({ birthdate: { $gte: new Date("1990-01-01T00:00:00Z")}}, { $unset: { platinum: true }})` = elimina los platinum
**Lecturas recomendadas**

[db.collection.update() - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/reference/method/db.collection.update/ "db.collection.update() - MongoDB Manual v7.0")

[db.collection.replaceOne() - MongoDB Manual v7.0](https://www.mongodb.com/docs/manual/reference/method/db.collection.replaceOne/ "db.collection.replaceOne() - MongoDB Manual v7.0")

## Bases de Datos de Grafos: Conceptos y Aplicaciones Prácticas

Las **bases de datos de grafos** son una alternativa a las bases de datos relacionales y NoSQL, diseñadas para modelar y gestionar datos altamente conectados. Se basan en la teoría de grafos, donde los datos se representan mediante **nodos**, **aristas (relaciones)** y **propiedades**.

### **📌 Conceptos Clave en Bases de Datos de Grafos**  

### **1️⃣ Nodos (Nodes)**  
Los **nodos** representan entidades, como personas, productos o ubicaciones. Son equivalentes a las filas en una base de datos relacional.  
✅ **Ejemplo**: Un nodo puede representar un usuario en una red social.  

### **2️⃣ Aristas (Edges o Relaciones)**  
Las **aristas** conectan nodos y representan relaciones entre ellos. A diferencia de las bases de datos relacionales, donde las relaciones se gestionan con JOINs, en las bases de datos de grafos estas relaciones son nativas y eficientes.  
✅ **Ejemplo**: "Juan SIGUE a María" en una red social.  

### **3️⃣ Propiedades (Properties)**  
Tanto los nodos como las aristas pueden tener **propiedades**.  
✅ **Ejemplo**: Un nodo de usuario puede tener una propiedad `edad: 30`, y una relación de "amistad" puede tener `desde: 2020`.  

### **4️⃣ Etiquetas (Labels)**  
Los nodos pueden pertenecer a una o más categorías mediante **etiquetas**.  
✅ **Ejemplo**: Un nodo puede tener la etiqueta `Usuario` o `Cliente`.

### **📌 Aplicaciones Prácticas de las Bases de Datos de Grafos**  

🚀 **1. Redes Sociales**  
   - Modelado de relaciones entre usuarios (amistades, seguidores, interacciones).  
   - Algoritmos de recomendación (¿a quién deberías seguir?).  
   - Análisis de comunidades.  

📦 **2. Gestión de Recomendaciones**  
   - Netflix, Amazon y Spotify usan grafos para sugerir contenido.  
   - Basado en conexiones entre usuarios y productos.  

🔍 **3. Detección de Fraude**  
   - Identificación de patrones sospechosos en transacciones bancarias.  
   - Análisis de conexiones entre cuentas fraudulentas.  

🚛 **4. Logística y Optimización de Rutas**  
   - Empresas como Uber y Google Maps usan grafos para encontrar rutas más rápidas.  
   - Modelado de redes de transporte y tráfico.  

🏥 **5. Ciencias de la Salud**  
   - Descubrimiento de relaciones entre genes, enfermedades y tratamientos.  
   - Uso en investigación biomédica y análisis de redes neuronales.

### **📌 Ejemplo en Neo4j (Consulta Cypher)**  
Neo4j es una de las bases de datos de grafos más populares.  

```cypher
CREATE (juan:Persona {nombre: "Juan"}) 
CREATE (maria:Persona {nombre: "María"}) 
CREATE (juan)-[:SIGUE]->(maria);
```

Esta consulta **crea dos nodos** (`Juan` y `María`) y establece una relación `SIGUE` entre ellos.

### **📌 Bases de Datos de Grafos Populares**  

🔹 **Neo4j** → La más popular, usa Cypher como lenguaje de consulta.  
🔹 **ArangoDB** → Multi-modelo (documentos + grafos).  
🔹 **OrientDB** → Soporta relaciones jerárquicas y grafos.  
🔹 **Amazon Neptune** → Base de datos de grafos en la nube.  
🔹 **JanusGraph** → Open-source, escalable para big data.

### **📌 ¿Cuándo Usar una Base de Datos de Grafos?**  
✅ **Si los datos tienen muchas relaciones** (redes sociales, recomendaciones, detección de fraude).  
✅ **Si necesitas consultas rápidas en grafos grandes** (optimización de rutas, análisis de redes).  
✅ **Si los JOINs en SQL son lentos y complicados** (gestión de relaciones complejas).  

⛔ **No es ideal si los datos son estructurados y sin muchas relaciones** (en este caso, SQL puede ser mejor).

### **🎯 Conclusión**  
Las bases de datos de grafos son una solución poderosa cuando se necesita modelar datos altamente interconectados. Son ideales para redes sociales, sistemas de recomendación y detección de fraude, entre otros.  

### Resumen

Las redes sociales manejan datos complejos mediante estructuras avanzadas como grafos. LinkedIn y Facebook utilizan sistemas especializados (Expresso y Tau) para gestionar publicaciones, comentarios y reacciones interconectadas, algo que sería costoso con bases de datos SQL tradicionales. Exploraremos cómo funcionan estas estructuras y sus ventajas.

### ¿Cómo se representan los grafos en redes sociales?


Los grafos se representan con nodos y aristas:

- **Nodos**: Representan entidades (p. ej., personas) con propiedades (nombre, correo, edad).
- **Aristas**: Conexiones entre nodos, que representan las relaciones de manera natural y eficiente.

### ¿Cuáles son las ventajas de los grafos frente a las bases de datos SQL?

- **Eficiencia**: Los nodos tienen conexiones inherentes, evitando las costosas llaves foráneas y consultas complejas de SQL.
- **Escalabilidad**: La estructura de grafos permite escalar fácilmente con el crecimiento de datos y conexiones.

### ¿Cómo se implementan y consultan las estructuras de grafos?

- **Implementación**: Sistemas como Expresso y Tau están diseñados para manejar grandes volúmenes de datos interconectados.
- **Consultas**: Las consultas en grafos son más rápidas y naturales, facilitando la obtención de datos relacionados sin la complejidad de SQL.

**Lecturas recomendadas**

[Caso de aplicación de bases de datos NoSQL: Espresso (LinkedIn)](https://es.linkedin.com/pulse/caso-de-aplicaci%C3%B3n-bases-datos-nosql-espresso-linkedin-danny-prol)

[Facebook](https://research.facebook.com/publications/tao-facebooks-distributed-data-store-for-the-social-graph/)

[TAO: The power of the graph - Engineering at Meta](https://engineering.fb.com/2013/06/25/core-infra/tao-the-power-of-the-graph/)

## Bases de Datos de Grafos: Ejercicios y Casos de Uso

Aquí tienes algunos **ejercicios prácticos y casos de uso** para bases de datos de grafos, utilizando **Neo4j** como referencia.  

### **📌 Ejercicios Prácticos con Bases de Datos de Grafos (Neo4j)**  

### **1️⃣ Crear una Red Social con Neo4j**
📌 **Objetivo:** Modelar usuarios y sus relaciones en una red social.  
🔹 **Entidad:** `Persona` (nombre, edad).  
🔹 **Relaciones:** `SIGUE`.  

✅ **Ejercicio:**  
1. **Crea dos usuarios y una relación entre ellos.**  
```cypher
CREATE (juan:Persona {nombre: "Juan", edad: 30})
CREATE (maria:Persona {nombre: "María", edad: 28})
CREATE (juan)-[:SIGUE]->(maria);
```

2. **Consulta quién sigue a quién.**  
```cypher
MATCH (p:Persona)-[:SIGUE]->(q:Persona) 
RETURN p.nombre, q.nombre;
```

3. **Obtener todos los seguidores de "María".**  
```cypher
MATCH (p:Persona)-[:SIGUE]->(m:Persona {nombre: "María"}) 
RETURN p.nombre;
```

### **2️⃣ Sistema de Recomendación (Netflix, Spotify, Amazon)**
📌 **Objetivo:** Modelar relaciones entre usuarios y películas.  
🔹 **Entidad:** `Usuario`, `Película`.  
🔹 **Relaciones:** `VIÓ`, `GUSTA`.  

✅ **Ejercicio:**  
1. **Crear datos de usuarios y películas.**  
```cypher
CREATE (ana:Usuario {nombre: "Ana"})
CREATE (pedro:Usuario {nombre: "Pedro"})
CREATE (pelicula1:Película {titulo: "Matrix"})
CREATE (pelicula2:Película {titulo: "Inception"})
CREATE (ana)-[:VIÓ]->(pelicula1)
CREATE (pedro)-[:VIÓ]->(pelicula1)
CREATE (pedro)-[:GUSTA]->(pelicula2);
```

2. **Recomendar películas vistas por otros usuarios con gustos similares.**  
```cypher
MATCH (u:Usuario)-[:VIÓ]->(p:Película)<-[:VIÓ]-(similar:Usuario)-[:VIÓ]->(recomendada:Película)
WHERE u.nombre = "Ana" AND NOT (u)-[:VIÓ]->(recomendada)
RETURN DISTINCT recomendada.titulo;
```

### **3️⃣ Detección de Fraude en Transacciones Bancarias**
📌 **Objetivo:** Modelar transacciones entre cuentas para detectar fraudes.  
🔹 **Entidad:** `Cuenta` (ID, saldo).  
🔹 **Relaciones:** `TRANSFIERE`.  

✅ **Ejercicio:**  
1. **Crear cuentas y transacciones sospechosas.**  
```cypher
CREATE (c1:Cuenta {id: "001", saldo: 5000})
CREATE (c2:Cuenta {id: "002", saldo: 1000})
CREATE (c3:Cuenta {id: "003", saldo: 50})
CREATE (c1)-[:TRANSFIERE {monto: 4500}]->(c2)
CREATE (c2)-[:TRANSFIERE {monto: 900}]->(c3);
```

2. **Detectar flujos de dinero sospechosos (cadenas de transacciones).**  
```cypher
MATCH path = (c1:Cuenta)-[:TRANSFIERE*2..]->(c3:Cuenta)
RETURN path;
```

### **4️⃣ Optimización de Rutas (Google Maps, Uber)**
📌 **Objetivo:** Modelar una red de ciudades y calcular la mejor ruta.  
🔹 **Entidad:** `Ciudad`.  
🔹 **Relaciones:** `CONECTADO_A` con distancia.  

✅ **Ejercicio:**  
1. **Crear ciudades y conexiones.**  
```cypher
CREATE (bogota:Ciudad {nombre: "Bogotá"})
CREATE (medellin:Ciudad {nombre: "Medellín"})
CREATE (cali:Ciudad {nombre: "Cali"})
CREATE (bogota)-[:CONECTADO_A {distancia: 400}]->(medellin)
CREATE (medellin)-[:CONECTADO_A {distancia: 300}]->(cali)
CREATE (bogota)-[:CONECTADO_A {distancia: 500}]->(cali);
```

2. **Encontrar la ruta más corta entre dos ciudades.**  
```cypher
MATCH (start:Ciudad {nombre: "Bogotá"}), (end:Ciudad {nombre: "Cali"}),
path = shortestPath((start)-[:CONECTADO_A*]->(end))
RETURN path;
```

### **📌 Casos de Uso Reales**
🔹 **Redes Sociales:** Facebook, Twitter, LinkedIn usan bases de datos de grafos para conexiones entre usuarios.  
🔹 **Motores de Recomendación:** Netflix, Amazon, Spotify sugieren contenido con análisis de grafos.  
🔹 **Seguridad y Fraude:** Bancos detectan fraudes con patrones en transacciones.  
🔹 **Logística y Rutas:** Google Maps y Uber optimizan tiempos y rutas.  

### Resumen

Implementar una base de datos basada en grafos puede parecer desafiante, pero con Neo4j y su servicio Aura, es más sencillo de lo que parece. Solo necesitas un correo electrónico para empezar y crear instancias gratuitas donde puedes almacenar hasta 200.000 nodos y 400.000 conexiones.

### ¿Cómo crear una instancia en Neo4j Aura?

Para comenzar, debes ingresar a la página de Neo4j y hacer clic en “Aura Login”. Una vez allí, crea una nueva instancia gratuita. Descarga las credenciales que se generan, ya que las necesitarás más adelante. Este proceso puede tardar unos minutos, pero una vez completado, tendrás tu instancia lista para usar.

### ¿Cómo acceder y utilizar tu instancia de Neo4j?

Con tu instancia creada, haz clic en “Open” e ingresa las credenciales descargadas. Este será tu entorno de trabajo. Para empezar a crear nodos, usa la palabra reservada `create` seguida de paréntesis que contienen el identificador y el tipo de entidad del nodo.

### ¿Cómo crear tus primeros nodos?

Para crear un nodo, usa la siguiente sintaxis:

`create (n:Person {name: "Alice", age: 24})`

Este comando crea un nodo de tipo `Person` con las propiedades `name` y `age`. Puedes crear varios nodos en una sola línea:

`create (n:Person {name: "Alice", age: 24}), (m:Person {name: "Samantha", age: 30}), (o:Person {name: "Bob", age: 29})`

### ¿Qué es el lenguaje Cypher?

Cypher es el lenguaje de consulta para Neo4j. Se parece a SQL pero está optimizado para trabajar con datos en forma de grafos. Si deseas profundizar en Cypher, revisa la documentación oficial para aprender a realizar operaciones más complejas.

### ¿Cómo agregar conexiones entre nodos?

Para agregar conexiones, usa `match` para encontrar los nodos y `create` para establecer las relaciones:

```shell
match (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
create (a)-[:FRIEND]->(b)
```

Este comando crea una relación de amistad entre Alice y Bob.

### ¿Cómo realizar consultas en tu base de datos de grafos?

Utiliza `match` para buscar nodos y sus relaciones, y `return` para especificar qué datos deseas obtener:

```shell
match (a:Person {name: "Alice"})-[:FRIEND]->(b)
return a.name, collect(b.name) as friends
```

Esta consulta devuelve el nombre de Alice y una lista de los nombres de sus amigos.

### ¿Por qué elegir una base de datos basada en grafos?

Las bases de datos de grafos permiten relaciones nativas entre nodos, lo que las hace más eficientes para ciertas consultas comparado con las bases de datos SQL tradicionales. Las operaciones complejas como los `joins` en SQL pueden ser costosas y lentas a medida que aumentan los datos, mientras que en Neo4j, las conexiones y consultas son rápidas y directas.

**Querys**
`CREATE (Alice:PERSON {name: 'Alice', age:24}), (Samanta:PERSON {name: 'Samanta', age:45}), (Bob:PERSON {name: 'Bob', age:19})` = Crea personas
`MATCH (a:PERSON), (b:PERSON) WHERE a.name = 'Alice' AND b.name ='Bob' CREATE (a)-[:FRIEND] -> (b)` = Crea una relación
`MATCH (Alice:PERSON {name: 'Alice'}) - [:FRIEND] -> (friends)  RETURN Alice.name, collect(friends.name) AS FriendName`= Para ver con cual esta relacionados
**Lecturas recomendadas**

[Introduction - Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)

[Fully Managed Graph Database Service | Neo4j AuraDB](https://neo4j.com/cloud/aura/)

## Introducción a las Bases de Datos basadas en Documentos

Las **bases de datos basadas en documentos** son un tipo de bases de datos NoSQL diseñadas para almacenar, recuperar y administrar datos en un formato de documento similar a JSON o BSON. Son ideales para aplicaciones que requieren **flexibilidad, escalabilidad y rendimiento** en el manejo de datos semiestructurados.  

### **📌 ¿Qué es una Base de Datos basada en Documentos?**  
📌 **Definición:**  
Las bases de datos basadas en documentos almacenan la información en **documentos estructurados**, en lugar de filas y columnas como en bases de datos relacionales. Cada documento es una unidad independiente de datos, similar a un objeto JSON, y puede contener estructuras anidadas.  

🔹 **Ejemplo de un documento en formato JSON:**  
```json
{
  "_id": "001",
  "nombre": "Juan Pérez",
  "edad": 30,
  "correo": "juan.perez@example.com",
  "dirección": {
    "ciudad": "Bogotá",
    "país": "Colombia"
  },
  "compras": ["Laptop", "Teléfono", "Tablet"]
}
```

### **📌 Características Principales**
✅ **Flexibilidad en la estructura:** No requiere esquemas rígidos como en bases de datos SQL.  
✅ **Escalabilidad horizontal:** Pueden distribuirse fácilmente en múltiples servidores.  
✅ **Optimización para consultas rápidas:** Indexación eficiente de documentos.  
✅ **Datos semiestructurados:** Permiten almacenar información con estructuras anidadas y sin necesidad de relaciones complejas.

### **📌 ¿Cuándo Usar Bases de Datos basadas en Documentos?**
📌 **Casos de Uso Comunes:**  
🔹 **Aplicaciones Web y Móviles:** Manejo de perfiles de usuario, contenido dinámico.  
🔹 **E-commerce:** Catálogos de productos con especificaciones variadas.  
🔹 **Big Data y Análisis en Tiempo Real:** Almacenar y procesar grandes volúmenes de datos JSON.  
🔹 **Sistemas de Recomendación:** Personalización de contenido en plataformas como Netflix o Spotify.  

### **📌 Ejemplos de Bases de Datos basadas en Documentos**  
🔹 **MongoDB:** Una de las más populares, usa BSON (una versión binaria de JSON).  
🔹 **CouchDB:** Usa JSON y permite replicación distribuida.  
🔹 **Firebase Firestore:** Usada en aplicaciones móviles por Google.  
🔹 **Amazon DynamoDB:** Solución NoSQL de AWS para alta escalabilidad.

### **📌 Comparación con Bases de Datos Relacionales (SQL vs. NoSQL)**
| **Característica**        | **SQL (Relacional)**  | **NoSQL (Documentos)**  |
|--------------------------|----------------------|-------------------------|
| **Estructura de Datos**   | Tablas con filas/columnas | Documentos JSON/BSON |
| **Esquema Rígido**        | Sí                   | No (flexible) |
| **Escalabilidad**         | Vertical (servidores más potentes) | Horizontal (varios nodos) |
| **Relaciones Complejas**  | Sí                   | No recomendado |
| **Velocidad en Lecturas** | Menos eficiente para datos anidados | Más rápida con datos semiestructurados |

### **📌 Ejemplo de Operaciones en MongoDB**
✅ **Insertar un documento (`insertOne`)**  
```js
db.usuarios.insertOne({
  "nombre": "María García",
  "edad": 25,
  "correo": "maria.garcia@example.com"
});
```

✅ **Buscar documentos (`find`)**  
```js
db.usuarios.find({ "edad": { "$gte": 18 } });
```

✅ **Actualizar un documento (`updateOne`)**  
```js
db.usuarios.updateOne(
  { "nombre": "María García" },
  { "$set": { "edad": 26 } }
);
```

✅ **Eliminar un documento (`deleteOne`)**  
```js
db.usuarios.deleteOne({ "nombre": "María García" });
```
### **📌 Conclusión**
Las bases de datos basadas en documentos ofrecen **gran flexibilidad** y **rendimiento**, especialmente para aplicaciones que manejan datos semiestructurados. Son una excelente opción cuando se necesita escalabilidad y rapidez en consultas sin la complejidad de las relaciones SQL.  

### Resumen

Las bases de datos documentales son una de las formas más populares de bases de datos NoSQL. Como su nombre lo indica, estas bases de datos almacenan documentos, y estos documentos contienen los datos. Los datos generalmente están en un formato clave-valor, es decir, el nombre del campo y el valor que contiene. Una ventaja significativa de este sistema es la capacidad de guardar distintos tipos de datos, como strings, enteros, fechas, elementos geográficos, y más.

###¿Qué tipos de datos podemos almacenar en una base de datos documental? 

- Strings
- Enteros
- Fechas
- Elementos geográficos
- Objetos dentro de objetos
- Arrays con valores numéricos o de string
- Arrays de objetos

### ¿Cómo se manejan los documentos jerárquicos?

Los documentos en estas bases de datos permiten una estructura jerárquica donde ciertos elementos están por encima de otros, dependiendo de la profundidad en la que se encuentren. Esta jerarquía proporciona una gran flexibilidad, una de las mayores ventajas de las bases de datos documentales.

### ¿Qué ventajas tiene la flexibilidad de las bases de datos documentales?

- Permite cambios fáciles en los datos almacenados sin necesidad de modificar el esquema completo.
- Facilita la adaptación a cambios en la visión o funcionalidades de una empresa.
- Soporta documentos antiguos y nuevos simultáneamente, permitiendo consultas en colecciones con documentos de diferentes estructuras.

### ¿Cómo se benefician las startups de las bases de datos documentales?

Las startups, que a menudo enfrentan cambios rápidos en sus productos y visiones, se benefician enormemente de la flexibilidad de las bases de datos documentales. A diferencia de las bases de datos SQL, donde los cambios de esquema pueden ser costosos y complejos, las bases de datos documentales permiten cambios rápidos y eficientes.

### ¿Qué aplicaciones prácticas tienen las bases de datos documentales?

Un ejemplo práctico es el uso en campos de paneles solares, donde los paneles miden constantemente la cantidad de luz recibida para convertirla en electricidad. Estos datos pueden variar entre distintos modelos de paneles y pueden incluir diferentes tipos de sensores. Las bases de datos documentales permiten almacenar y consultar estos datos diversos y numerosos de manera eficiente.

### ¿Qué es una aplicación de series temporales en bases de datos documentales?

La aplicación de series temporales es una funcionalidad que permite manejar millones de datos introducidos constantemente con distintos formatos y estructuras. Esta capacidad es especialmente útil en escenarios como el monitoreo continuo de sensores en paneles solares, donde las mediciones pueden variar.

## Introducción a las Bases de Datos Clave-Valor

Las **bases de datos clave-valor** son un tipo de bases de datos NoSQL diseñadas para almacenar datos en un formato simple pero altamente eficiente: **pares clave-valor**. Son ideales para aplicaciones que requieren accesos rápidos y escalabilidad masiva.

### **📌 ¿Qué es una Base de Datos Clave-Valor?**  

Una **base de datos clave-valor** almacena información en una estructura sencilla donde **cada valor está asociado a una clave única**. Esto permite acceder rápidamente a los datos utilizando la clave como identificador.  

🔹 **Ejemplo de un par clave-valor:**  
```json
"user_001": { "nombre": "Juan Pérez", "edad": 30, "correo": "juan.perez@example.com" }
```

🔹 **Ejemplo en Redis (sintaxis de almacenamiento):**  
```sh
SET user_001 "{'nombre': 'Juan Pérez', 'edad': 30, 'correo': 'juan.perez@example.com'}"
```

### **📌 Características Principales**  

✅ **Alto rendimiento:** Lecturas y escrituras extremadamente rápidas.  
✅ **Escalabilidad horizontal:** Se pueden distribuir fácilmente en múltiples servidores.  
✅ **Modelo flexible:** No requiere un esquema fijo.  
✅ **Operaciones eficientes:** Acceso directo a los valores mediante la clave sin necesidad de búsquedas complejas.  
✅ **Uso de diferentes formatos de almacenamiento:** JSON, cadenas de texto, binarios, etc.

### **📌 Casos de Uso Comunes**  

🔹 **Almacenamiento en caché:** Bases de datos como **Redis** y **Memcached** se usan para mejorar la velocidad de aplicaciones web.  
🔹 **Manejo de sesiones de usuario:** Almacenar sesiones de usuario en aplicaciones web.  
🔹 **Sistemas de configuración:** Guardar configuraciones globales de aplicaciones.  
🔹 **Colas de mensajes:** Implementar colas de tareas distribuidas.  
🔹 **Carritos de compras:** Guardar datos temporales en aplicaciones de comercio electrónico.

### **📌 Ejemplos de Bases de Datos Clave-Valor**  

🔹 **Redis:** Base de datos en memoria con soporte para estructuras avanzadas como listas y hashes.  
🔹 **Memcached:** Almacenamiento en caché distribuido y de alta velocidad.  
🔹 **Amazon DynamoDB:** Base de datos NoSQL altamente escalable.  
🔹 **Riak KV:** Base de datos distribuida diseñada para disponibilidad y tolerancia a fallos.

### **📌 Comparación con Otros Modelos de Bases de Datos**  

| **Característica**        | **SQL (Relacional)**  | **Clave-Valor (NoSQL)** |
|--------------------------|----------------------|------------------------|
| **Estructura de Datos**   | Tablas con filas/columnas | Claves y valores simples |
| **Esquema Rígido**        | Sí                   | No (dinámico) |
| **Escalabilidad**         | Vertical (servidores más potentes) | Horizontal (varios nodos) |
| **Consultas Complejas**   | Sí                   | No (sólo acceso por clave) |
| **Velocidad de Lectura**  | Menos eficiente para consultas masivas | Extremadamente rápida |

### **📌 Ejemplo de Uso en Redis**  

✅ **Guardar un valor en Redis:**  
```sh
SET usuario:1001 "Juan Pérez"
```

✅ **Obtener el valor almacenado:**  
```sh
GET usuario:1001
```
🔹 Salida: `"Juan Pérez"`

✅ **Almacenar estructuras más complejas con JSON:**  
```sh
SET user:2001 '{"nombre": "María García", "edad": 28, "correo": "maria@example.com"}'
```

✅ **Eliminar una clave en Redis:**  
```sh
DEL usuario:1001
```

### **📌 Conclusión**  

Las bases de datos **clave-valor** son ideales para aplicaciones que requieren acceso ultrarrápido a datos, escalabilidad y almacenamiento flexible. Son ampliamente utilizadas en sistemas de caché, gestión de sesiones y almacenamiento de configuraciones.  

### Resumen

Las bases de datos NoSQL, especialmente las de tipo clave-valor, ofrecen una simplicidad que resulta en un rendimiento excepcional. Este tipo de bases de datos, donde cada entrada se compone de una clave única y un valor, permite almacenar y recuperar información rápidamente, lo que es ideal para aplicaciones donde la velocidad es crucial.

### ¿Qué son las bases de datos clave-valor?

Las bases de datos clave-valor se basan en un modelo simple: cada dato se asocia a una clave única y un valor, que puede variar desde un texto o un número hasta estructuras más complejas. Esta simplicidad facilita la rapidez en las operaciones de lectura y escritura.

### ¿Cómo se ordenan y consultan las claves?

Las claves en estas bases de datos pueden ordenarse de diversas maneras, como alfabéticamente o numéricamente. Este ordenamiento permite consultas extremadamente rápidas y facilita el particionamiento y la escalabilidad horizontal, lo que es esencial para manejar grandes volúmenes de datos.

### ¿Cuáles son las ventajas principales de las bases de datos clave-valor?

- **Alto rendimiento**: Gracias a su estructura simple, las operaciones son rápidas y eficientes.
- **Escalabilidad**: Fácil de escalar horizontalmente para manejar grandes cantidades de datos.
- **Flexibilidad**: Adecuada para diversos tipos de datos y aplicaciones.

### ¿En qué casos de uso son más útiles?

- **Videojuegos**: Permiten almacenar rápidamente eventos y acciones de los jugadores, donde milisegundos pueden marcar la diferencia entre ganar o perder.
- **Sesiones de usuarios**: Almacenan tokens de autenticación y datos de sesión para un acceso rápido y eficiente.

### ¿Qué ejemplos de tecnologías clave-valor existen?

Algunas tecnologías populares que utilizan este modelo incluyen DynamoDB y CouchDB. Estas bases de datos también cumplen con las propiedades ACID (Atomicidad, Consistencia, Aislamiento y Durabilidad), asegurando la integridad de los datos a pesar de su simpleza aparente.

### ¿Por qué la simplicidad es una ventaja?

Aunque las bases de datos clave-valor son simples, esta simplicidad es una de sus mayores ventajas, ya que facilita el rendimiento y la escalabilidad, haciendo que sean una elección popular para aplicaciones que requieren rapidez y eficiencia.

**Lecturas recomendadas**

[What is Amazon DynamoDB? - Amazon DynamoDB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html)

[ACID - Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/ACID)

## Introducción a las Bases de Datos Vectoriales

Las **bases de datos vectoriales** son un tipo de bases de datos optimizadas para almacenar, indexar y recuperar información basada en representaciones numéricas de alto nivel conocidas como **vectores**. Son fundamentales en aplicaciones de inteligencia artificial, búsqueda semántica y machine learning.

### **📌 ¿Qué es una Base de Datos Vectorial?**  

En lugar de almacenar datos en filas y columnas como una base de datos relacional, una base de datos vectorial almacena **vectores de alta dimensión** que representan entidades como imágenes, texto o audio.  

🔹 **Ejemplo de representación vectorial de palabras (Word Embeddings):**  
```python
"perro" → [0.12, 0.85, -0.45, ..., 0.67]
"gato"  → [0.14, 0.80, -0.40, ..., 0.70]
```
Aquí, las palabras *perro* y *gato* tienen vectores similares porque sus significados están relacionados.

### **📌 Características Principales**  

✅ **Optimización para búsqueda por similitud** (Nearest Neighbor Search - NNS).  
✅ **Soporte para alto volumen de datos no estructurados** (imágenes, texto, audio, video).  
✅ **Escalabilidad y eficiencia** en consultas sobre grandes cantidades de vectores.  
✅ **Uso de índices eficientes** como HNSW (Hierarchical Navigable Small World) o FAISS.  

### **📌 Casos de Uso Comunes**  

🔹 **Búsqueda de imágenes similar** (Google Reverse Image Search).  
🔹 **Recomendaciones personalizadas** en plataformas de streaming y comercio electrónico.  
🔹 **Búsqueda semántica en documentos** (Chatbots avanzados, recuperación de información).  
🔹 **Reconocimiento facial y biometría**.  
🔹 **Análisis de sentimientos y NLP** (Procesamiento del lenguaje natural).  

### **📌 Ejemplo de Bases de Datos Vectoriales**  

| **Base de Datos**  | **Descripción** |
|--------------------|----------------|
| **FAISS** (Facebook AI Similarity Search) | Altamente optimizado para búsquedas de similitud en grandes volúmenes de datos. |
| **Milvus** | Plataforma escalable y distribuida para almacenamiento de vectores. |
| **Pinecone** | Solución en la nube para búsqueda vectorial en AI y NLP. |
| **Weaviate** | Base de datos orientada a recuperación semántica con integración en ML. |
| **Annoy** (Approximate Nearest Neighbors Oh Yeah) | Biblioteca rápida para búsqueda de vecinos más cercanos. |

### **📌 Comparación con Bases de Datos Tradicionales**  

| **Característica**       | **SQL Relacional**    | **Base de Datos Vectorial** |
|-------------------------|----------------------|---------------------------|
| **Estructura de Datos**  | Tablas con filas/columnas | Espacios multidimensionales |
| **Búsqueda Exacta**      | Índices basados en B-Trees | Similitud basada en distancia (coseno, euclidiana, etc.) |
| **Escalabilidad**        | Vertical             | Horizontal |
| **Aplicaciones**        | Transacciones, CRUD  | IA, Búsqueda semántica, ML |

### **📌 Ejemplo de Uso con FAISS**  

**1️⃣ Instalar FAISS:**  
```sh
pip install faiss-cpu
```

**2️⃣ Crear y almacenar vectores:**  
```python
import faiss
import numpy as np

# Crear una base de datos de 1000 vectores de 128 dimensiones
dimension = 128
num_vectors = 1000
data = np.random.rand(num_vectors, dimension).astype('float32')

# Construir el índice FAISS
index = faiss.IndexFlatL2(dimension)
index.add(data)

# Buscar el vector más cercano a uno nuevo
query_vector = np.random.rand(1, dimension).astype('float32')
distances, indices = index.search(query_vector, k=5)  # Encuentra los 5 más cercanos

print("Índices más cercanos:", indices)
print("Distancias:", distances)
```

### **📌 Conclusión**  

Las **bases de datos vectoriales** están revolucionando la forma en que interactuamos con la información, permitiendo búsquedas más rápidas y precisas en entornos de IA, NLP y visión por computadora. Su capacidad para manejar datos no estructurados las convierte en una herramienta esencial en la era de la inteligencia artificial.

### Resumen

Las bases de datos vectoriales son esenciales para resolver problemas complejos como recomendaciones personalizadas y preguntas frecuentes con variaciones de lenguaje. Estos sistemas utilizan representaciones matemáticas para almacenar y procesar información de manera eficiente.

### ¿Qué es un vector?

Un vector es la representación de un array, un elemento con una estructura de datos que contiene varios valores específicos. Estos valores generalmente son números que van de -1 a 1, y representan información como texto, imágenes, sonido o video.

### ¿Cómo se generan los valores de un vector?

Los valores de un vector son generados por un encoder, una herramienta de machine learning que transforma la información original en valores numéricos. Este proceso crea lo que se llama un embedding, esencial para el procesamiento de imágenes, sonidos o lenguaje natural.

### ¿Qué es un valor semántico?

El valor semántico de un vector refleja el significado de la información que representa. Por ejemplo, en procesamiento de lenguaje natural, se identifican palabras clave, artículos y palabras poco frecuentes, asignando diferentes pesos según su importancia en el contexto. Esto permite que los vectores representen de manera efectiva la intención y el significado del texto.

### ¿Cómo se agrupan los vectores según su valor semántico?

Los vectores con valores semánticos similares se agrupan cercanamente. Por ejemplo, las palabras “king” y “queen” estarán cerca en el espacio vectorial debido a sus similitudes semánticas. Del mismo modo, “man” y “woman” estarán cerca entre sí y mostrarán relaciones de similitud con “king” y “queen” según su contexto semántico.

### ¿Qué implicaciones tiene la dirección de un vector?

La dirección de un vector indica su similitud con otros vectores. Vectores que apuntan en direcciones similares comparten características semánticas. Este principio es fundamental para algoritmos de recomendación y sistemas de búsqueda que dependen de las relaciones entre diferentes tipos de información.

**Lecturas recomendadas**

[Curso de Embeddings y Bases de Datos Vectoriales para NLP - Platzi](https://platzi.com/cursos/embeddings-nlp/)

[Machine Learning Engineer](https://platzi.com/ruta/mlengineer/)

## Alcances y Beneficios de NoSQL

Las bases de datos **NoSQL** (Not Only SQL) han surgido como una alternativa a las bases de datos relacionales, ofreciendo mayor flexibilidad, escalabilidad y eficiencia para manejar grandes volúmenes de datos no estructurados. 

### **📌 Alcances de NoSQL**  

### 1️⃣ **Escalabilidad Horizontal**  
✅ Permiten agregar más servidores en lugar de actualizar uno solo (escalabilidad horizontal).  
✅ Ideales para aplicaciones que manejan grandes cantidades de datos distribuidos.  

### 2️⃣ **Flexibilidad en el Modelo de Datos**  
✅ Soportan datos **estructurados, semiestructurados y no estructurados**.  
✅ No requieren un esquema fijo, permitiendo cambios sin afectar la base de datos.  

### 3️⃣ **Alto Rendimiento en Lectura y Escritura**  
✅ Diseñadas para manejar operaciones a gran escala en tiempo real.  
✅ Usadas en aplicaciones como redes sociales, big data y analítica en tiempo real.  

### 4️⃣ **Disponibilidad y Tolerancia a Fallos**  
✅ Distribuyen datos en múltiples nodos, reduciendo riesgos de pérdida.  
✅ Sistemas como **Cassandra** usan replicación automática.

### **📌 Beneficios de NoSQL**  

| **Beneficio**       | **Descripción** |
|---------------------|----------------|
| **Escalabilidad** | Soporta grandes volúmenes de datos sin perder rendimiento. |
| **Modelo Flexible** | Permite almacenar JSON, XML, binarios, entre otros formatos. |
| **Alta Disponibilidad** | Replicación de datos para evitar caídas del sistema. |
| **Consultas Rápidas** | Optimizado para búsquedas y análisis en tiempo real. |
| **Desarrollo Ágil** | No requiere esquemas rígidos, lo que facilita la iteración rápida. |

### **📌 Tipos de Bases de Datos NoSQL y Ejemplos**  

| **Tipo** | **Descripción** | **Ejemplos** |
|----------|---------------|-------------|
| **Clave-Valor** | Datos almacenados como un diccionario (clave → valor). | Redis, DynamoDB |
| **Documentos** | Usa documentos JSON o BSON para almacenar datos. | MongoDB, CouchDB |
| **Columnar** | Optimiza grandes cantidades de datos en columnas. | Apache Cassandra, HBase |
| **Grafos** | Almacena relaciones entre datos como nodos y aristas. | Neo4j, ArangoDB |

### **📌 Casos de Uso de NoSQL**  

🔹 **Big Data y Analítica** – Almacenamiento y procesamiento de datos masivos.  
🔹 **E-Commerce** – Manejo de catálogos flexibles y recomendaciones personalizadas.  
🔹 **Redes Sociales** – Soporte para interacciones en tiempo real y grandes volúmenes de usuarios.  
🔹 **IoT y Sensores** – Gestión de datos en dispositivos conectados.  
🔹 **Búsquedas Semánticas** – Indexación rápida para motores de búsqueda.  

### **📌 Conclusión**  

Las bases de datos **NoSQL** son ideales para aplicaciones modernas que requieren **escalabilidad, flexibilidad y velocidad**. Aunque no reemplazan completamente las bases de datos SQL, su uso es esencial en entornos con **datos dinámicos y en crecimiento constante**.

### Resumen

El uso de bases de datos NoSQL ha ganado terreno gracias al auge de la inteligencia artificial, aunque las bases de datos SQL aún dominan el mercado. Sin embargo, la flexibilidad y eficiencia de NoSQL son cada vez más relevantes, especialmente cuando se trata de integrar inteligencia artificial en nuevas aplicaciones o mejorar las ya existentes.

###¿Cómo impacta la inteligencia artificial en la elección de bases de datos?

La inteligencia artificial no solo facilita la creación de software, sino que también mejora aplicaciones existentes y da lugar a nuevas compañías. La incorporación de bases de datos vectoriales, por ejemplo, permite agregar contexto a la inteligencia artificial, reduciendo las alucinaciones de datos imprecisos y proporcionando respuestas más precisas y eficaces.

### ¿Qué tipo de datos requiere la inteligencia artificial?

El entrenamiento de inteligencia artificial y el feedback de los resultados generan una gran cantidad de datos. Aquí es donde la flexibilidad de las bases de datos basadas en documentos es crucial. Estas bases de datos pueden manejar de manera eficiente el procesamiento del lenguaje natural y otras tareas relacionadas con inteligencia artificial.

### ¿Qué ventajas ofrece NoSQL en este contexto?

NoSQL proporciona varias ventajas:

- **Flexibilidad**: Manejo de datos no estructurados y semi-estructurados.
- **Escalabilidad**: Fácil expansión horizontal.
- **Desempeño**: Mejora en el manejo de grandes volúmenes de datos en tiempo real.

### ¿Qué tipo de base de datos deberías utilizar?

La elección de la base de datos adecuada depende del caso de uso específico. Si trabajas con inteligencia artificial y necesitas manejar grandes volúmenes de datos no estructurados o en tiempo real, NoSQL puede ser la mejor opción. Considera siempre las necesidades particulares de tu proyecto para seleccionar la herramienta más eficaz.