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

