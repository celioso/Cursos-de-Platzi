# Curso de Bases de Datos con SQL

## ¡El poder de los datos!

Ese es el verdadero poder de los datos: **tomar algo aparentemente simple (como caramelos, autos o transacciones de banco)** y, con técnicas como regresión, clustering o visualización, encontrar **patrones ocultos** que nos ayudan a decidir mejor.

🔑 En otras palabras:
👉 **Los datos son la materia prima.**
👉 **La estadística, el machine learning y la programación son las herramientas.**
👉 **El conocimiento y las decisiones inteligentes son el resultado.**

### Resumen

#### ¿Cómo influye la inteligencia artificial en la fotografía?

La fotografía ha dado un giro revolucionario gracias a la inteligencia artificial generativa y la fotografía computacional. ¿Recuerdas la última vez que tomaste una foto? Es probable que, sin darte cuenta, la tecnología de IA haya optimizado colores, nitidez, y detalles para crear una imagen que podría incluso superar lo que perciben tus propios ojos. Esta tecnología no solo edita imágenes que hemos capturado, sino que también puede generar elementos que nunca estuvieron allí.

#### ¿Qué papel juega la IA en el contenido que consumimos?

Las plataformas de streaming como Netflix han adoptado la IA no solo para recomendarte qué película ver a continuación, sino para planificar futuras producciones. Al analizar tus preferencias, estos servicios pueden anticipar tus gustos antes de que se filme un nuevo contenido. Estarás siempre a un clic de distancia de una experiencia personalizada, gracias a la inteligencia de las bases de datos que gestionan la información sobre tus intereses.

#### ¿Por qué son fundamentales las bases de datos en nuestra vida cotidiana?

Cuando piensas en bases de datos, es probable que te imagines interminables filas y columnas. Sin embargo, su impacto va más allá. Siempre que realizas una compra en línea, verificas el clima o encuentras pareja por medio de una aplicación, estás interactuando con una base de datos. Estos sistemas recopilan, organizan y utilizan la información con el fin de mejorar nuestra experiencia diaria.

#### ¿Qué riesgos conllevan las bases de datos mal gestionadas?

El manejo inapropiado de los datos puede acarrear graves problemas. En 2015, Google se enfrentó a una fuerte crítica al mal identificar a personas afroamericanas en su algoritmo de reconocimiento facial. Similarmente, un chatbot de Microsoft se involucró en controversias por producir comentarios ofensivos basados en entradas generadas por usuarios. Estos ejemplos muestran que no es suficiente tener grandes volúmenes de datos; es vital gestionarlos éticamente para evitar consecuencias negativas.

#### ¿Cómo están transformando los datos nuestra interacción diaria?

Prácticamente todas nuestras interacciones tecnológicas están basadas en datos: desde el desbloqueo facial de tu móvil hasta las rutas sugeridas por apps de mapas. Todo se origina y se actualiza en tiempo real gracias a la vasta cantidad de datos generados globalmente.

El entenderlos y utilizarlos adecuadamente no solo nos permite cumplir tareas diarias, sino también descubrir nuevas formas de innovación y creatividad. Recuerda, siempre debemos manejar los datos con responsabilidad para aprovechar su potencial al máximo.

Estás a punto de iniciar un viaje para explorar las bases de datos y entender cómo interactuar con ellas mediante SQL. Conoce las maravillas de los datos con la orientación de Carolina Castañeda, una ingeniera de software líder en su campo. ¡Adéntrate en el fascinante mundo del análisis de datos con determinación y creatividad!

## Sistema de Gestión de Bases de Datos

Un **Sistema de Gestión de Bases de Datos (SGBD o DBMS por sus siglas en inglés, Database Management System)** es un **software especializado** que permite **crear, organizar, administrar y manipular** bases de datos de forma eficiente y segura.

### 🔹 ¿Qué es un SGBD?

Es la capa intermedia entre el **usuario/aplicación** y los **datos almacenados**, que facilita el acceso, garantiza la integridad, maneja la seguridad y optimiza el rendimiento de las consultas.

Ejemplo de SGBD muy usados:

* Relacionales: **MySQL, PostgreSQL, Oracle Database, SQL Server**
* NoSQL: **MongoDB, Cassandra, Redis**

### 🔹 Funciones principales

1. **Definición de datos** → Crear y modificar estructuras de tablas, índices, relaciones.
2. **Manipulación de datos** → Consultar, insertar, actualizar y eliminar información.
3. **Control de acceso y seguridad** → Gestión de usuarios, roles y permisos.
4. **Integridad de datos** → Garantizar que la información sea válida y consistente.
5. **Respaldo y recuperación** → Copias de seguridad y restauración en caso de fallos.
6. **Optimización** → Uso de índices, planes de ejecución y caché para consultas rápidas.

### 🔹 Tipos de SGBD

* **Relacionales (RDBMS):** Organizan la información en tablas relacionadas (SQL).
* **Jerárquicos:** Datos estructurados en forma de árbol.
* **Redes:** Los datos se representan como nodos conectados.
* **Orientados a objetos:** Manejan datos complejos como objetos.
* **NoSQL:** Manejan datos no estructurados o semiestructurados (documentos, grafos, clave-valor).

### 🔹 Ejemplo en SQL (SGBD Relacional)

```sql
-- Crear una tabla de clientes
CREATE TABLE clientes (
    id INT PRIMARY KEY,
    nombre VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    ciudad VARCHAR(50)
);

-- Insertar un cliente
INSERT INTO clientes (id, nombre, email, ciudad)
VALUES (1, 'Mario Vargas', 'mario@email.com', 'Bogotá');

-- Consultar clientes de Bogotá
SELECT * FROM clientes WHERE ciudad = 'Bogotá';
```

📌 En pocas palabras:
Un **SGBD** es como el "cerebro organizador" que se encarga de que los datos estén **seguros, accesibles y ordenados**, mientras los usuarios y aplicaciones se concentran en **usar la información**, no en preocuparse por cómo se guarda.

### Resumen

#### ¿Qué es un sistema de gestión de bases de datos?

Un sistema de gestión de bases de datos (SGBD) es una herramienta indispensable para manejar cualquier operación relacionada con bases de datos, como consultar, modificar y almacenar información. Además, facilita la recuperación de objetos dentro de la base de datos. Entre sus funciones principales, se encuentra el manejo de transacciones seguras y confiables, gracias al soporte ACID (atomicidad, consistencia, aislamiento y durabilidad), que garantiza la integridad de las operaciones realizadas.

#### ¿Cómo se manejan múltiples usuarios en un SGBD?

El control de la concurrencia es vital en un SGBD, pues permite que varios usuarios realicen diferentes operaciones en simultáneo sobre la misma base de datos. Por ejemplo, mientras un usuario inserta registros, otro puede estar creando procedimientos almacenados o modificando información, sin que esto genere inconvenientes en el desarrollo de datos.

#### ¿Qué lenguaje y herramientas se utilizan en los SGBD?

El éxito de un SGBD depende de la correcta interpretación y ejecución de lenguajes de consulta, adaptados según el tipo de dato a manipular. Esto incluye:

- **Lenguaje de consulta**: Es necesario para la interpretación y ejecución de acciones en la base de datos.
- **Optimización de consultas**: Los SGBD utilizan optimizadores, como índices, para mejorar la velocidad y eficacia del procesamiento de la información.
- **Manejo de dato**s: Permite almacenar elementos como tablas, vistas, procedimientos y relaciones en un solo lugar.

#### ¿Cómo se garantiza la seguridad y autenticación en base de datos?

La seguridad en un SGBD se asegura mediante:

- **Gestión de usuarios y roles**: Determina qué permisos tiene cada usuario, tales como lectura, escritura y modificación.
- **Cifrado de datos**: Protege la información almacenada.
- **Auditoría de transacciones**: Permite rastrear ejecuciones y sintaxis utilizadas por cada usuario o rol, especialmente útil para realizar un rollback de la base de datos o consultar transacciones específicas.

#### ¿Cuáles son las características de escalabilidad y rendimiento de un SGBD?

Los motores de bases de datos pueden escalar de forma horizontal o vertical, adaptándose a las necesidades específicas. Además, incorporan mecanismos como la caché, la replicación y el particionamiento para optimizar el manejo de datos. Este tipo de escalabilidad es crucial para responder de manera eficiente ante el aumento del volumen de datos o la cantidad de usuarios.

#### ¿Cómo se asegura la integridad y consistencia de los datos en un SGBD?

Se utilizan restricciones y disparadores para mantener la integridad y consistencia de los datos. Los disparadores son acciones automáticas que responden a eventos especificados, como la inserción de un nuevo registro que activa tareas subsecuentes.

#### ¿Qué es la compatibilidad y extensión en un SGBD?

La compatibilidad y extensión se refieren a las capacidades de interoperabilidad y extensibilidad del SGBD:

- **Interoperabilidad**: Facilita la integración con otros sistemas mediante conexiones específicas, como APIs o controladores nativos.
- **Extensibilidad**: Permite añadir nuevas funciones mediante módulos o plugins externos, incluyendo nuevos tipos de datos o funciones personalizadas.

Entender estas funcionalidades y características te permitirá gestionar bases de datos de manera eficiente y segura, facilitando la interacción y manipulación de grandes volúmenes de información dentro de diferentes entornos tecnológicos.

## ¿Qué es una base de datos?

Una **base de datos** es un sistema organizado para **almacenar, gestionar y recuperar información** de manera eficiente.

En palabras simples:
📂 Imagina un gran archivador digital en el que guardas datos (nombres, números, fechas, imágenes, etc.), pero en lugar de estar en papeles, está en un sistema que permite buscar, ordenar, relacionar y actualizar esa información rápidamente.

### 🔹 Características principales de una base de datos

* **Organización estructurada** → Los datos siguen un modelo (tablas, documentos, grafos, etc.).
* **Acceso rápido** → Se pueden consultar y modificar mediante lenguajes especializados (como **SQL**).
* **Consistencia** → Mantienen reglas para evitar errores o duplicaciones.
* **Seguridad** → Controlan quién puede ver o cambiar la información.
* **Escalabilidad** → Pueden manejar desde pocos datos hasta millones de registros.

### 🔹 Ejemplos cotidianos de bases de datos

* 📱 **Agenda de contactos** en tu celular (nombre, número, correo).
* 🛒 **Sistema de inventario** de un supermercado (productos, precios, stock).
* 💳 **Registros bancarios** (cuentas, movimientos, clientes).
* 🌐 **Redes sociales** (usuarios, publicaciones, comentarios).

### 🔹 Tipos de bases de datos

1. **Relacionales (SQL)** → Organizan datos en **tablas** con filas y columnas (ej. MySQL, PostgreSQL, Oracle).
2. **No relacionales (NoSQL)** → Usan estructuras más flexibles como documentos, grafos o pares clave-valor (ej. MongoDB, Redis, Cassandra).
3. **En memoria** → Optimizadas para rapidez, manteniendo datos directamente en RAM (ej. Redis).
4. **Distribuidas** → Reparten la información en varios servidores en la nube (ej. Google BigQuery, Amazon DynamoDB).

👉 Te hago un **mini ejemplo**:
Si tienes una **tabla de estudiantes** en una base de datos relacional:

| ID | Nombre     | Edad | Carrera    |
| -- | ---------- | ---- | ---------- |
| 1  | Ana Pérez  | 20   | Ingeniería |
| 2  | Luis Gómez | 22   | Medicina   |
| 3  | Carla Díaz | 19   | Derecho    |

Con una consulta SQL como:

```sql
SELECT Nombre FROM Estudiantes WHERE Edad > 20;
```

El sistema te devolvería: **Luis Gómez** ✅

### Resumen

#### ¿Qué son las bases de datos?

Las bases de datos son sistemas esenciales en la gestión de información que nos permiten almacenar, consultar, modificar y eliminar datos con eficiencia. Su relevancia en el análisis de información es tal que podríamos compararlas con un archivo de Excel, donde las filas representan registros y las columnas atributos. Sin embargo, cuando hablamos de bases de datos, estas se dividen principalmente en dos categorías: relacionales y no relacionales. Comprender las diferencias entre estos tipos de bases de datos es crucial para utilizar el tipo correcto en cada aplicación.

#### ¿Cuáles son las diferencias entre bases de datos relacionales y no relacionales?

Las bases de datos relacionales, conocidas como RDBMS (Relational Database Management Systems), están estructuradas principalmente en tablas que contienen filas y columnas. Las columnas representan atributos como nombre, edad o dirección, mientras que las filas contienen los registros. Este tipo de bases de datos sigue un esquema rígido, lo que significa que la estructura debe estar definida y acorde a la información que se desea almacenar. Esto también implica que no se pueden agregar atributos nuevos sin modificar el esquema existente. Las bases de datos relacionales también se destacan por el manejo de relaciones entre tablas, utilizando claves primarias y foráneas para garantizar la integridad y consistencia de los datos.

Por otro lado, las bases de datos no relacionales, también conocidas como NoSQL, presentan una estructura mucho más flexible. En estas bases, la información se puede almacenar en diversos formatos, como grafos, archivos JSON, o sistemas de clave-valor, y se pueden agregar atributos nuevos sin modificar su estructura subyacente. Además, las relaciones no son explícitas como en las relacionales, lo que conlleva a una gestión diferente del manejo de conexiones y reglas a nivel de aplicación.

#### ¿Cómo manejan la escalabilidad y la integridad los dos tipos?

En términos de escalabilidad, las bases de datos relacionales escalan principalmente de manera vertical, es decir, mejorando el hardware del servidor. Esto puede incluir actualizaciones de memoria, procesamiento o almacenamiento. No obstante, las bases de datos no relacionales son óptimas para la escalabilidad horizontal, agregando nodos o servidores adicionales.

La integridad es otro aspecto clave. Las bases de datos relacionales garantizan la consistencia de la información mediante restricciones estrictas como claves únicas, reglas de negocio, y relaciones entre tablas. En las no relacionales, por su flexibilidad y capacidad para manejar datos no estructurados o semi-estructurados, se prioriza la habilidad de manejar grandes volúmenes de información sin las mismas restricciones.

#### ¿Cuáles son los casos de uso para cada tipo de base de datos?

Las bases de datos relacionales son ideales para aplicaciones que requieren un manejo estructurado y consistente de datos, como los sistemas ERP, la gestión de inventarios, y la gestión de información financiera. Ejemplos de motores de bases de datos relacionales son MySQL, PostgreSQL, Oracle, Microsoft SQL Server y SQLite.

En contraste, las bases de datos no relacionales son adecuadas para el almacenamiento de datos no estructurados o semi-estructurados, como los que generan las aplicaciones web, redes sociales y algunos proyectos de Big Data. Estas bases sobresalen en el manejo de datos vectoriales y otros formatos que requieren flexibilidad. Algunos motores de bases de datos no relacionales populares son MongoDB, Cassandra, Redis y DynamoDB.

¡Ahora que hemos explorado las diferencias, características y aplicaciones de ambos tipos de bases de datos, estás listo para profundizar en las bases de datos relacionales y en el lenguaje de consulta SQL! Sigue aprendiendo y perfeccionando tus habilidades para sacar el máximo provecho a las bases de datos y su amplia aplicación en el mundo digital.

## ¿Qué es SQL?

SQL significa **Structured Query Language** o **Lenguaje de Consulta Estructurada**.

Es un **lenguaje de programación estándar** utilizado para **gestionar y manipular bases de datos relacionales** 📊.

En otras palabras:
👉 SQL te permite **hablar con una base de datos** para decirle qué datos quieres ver, agregar, modificar o eliminar.

### 🔹 ¿Para qué sirve SQL?

Con SQL puedes:

* **Crear** bases de datos y tablas.
* **Insertar** datos (ej. registrar un nuevo cliente).
* **Consultar** información (ej. buscar todos los productos con precio menor a \$100).
* **Actualizar** datos (ej. cambiar el correo de un usuario).
* **Eliminar** registros (ej. borrar un pedido cancelado).

### 🔹 Ejemplos básicos en SQL

1. **Crear una tabla:**

```sql
CREATE TABLE Estudiantes (
    ID INT PRIMARY KEY,
    Nombre VARCHAR(100),
    Edad INT,
    Carrera VARCHAR(50)
);
```

2. **Insertar datos:**

```sql
INSERT INTO Estudiantes (ID, Nombre, Edad, Carrera)
VALUES (1, 'Ana Pérez', 20, 'Ingeniería');
```

3. **Consultar datos:**

```sql
SELECT Nombre, Edad 
FROM Estudiantes 
WHERE Edad > 18;
```

4. **Actualizar un registro:**

```sql
UPDATE Estudiantes 
SET Edad = 21 
WHERE ID = 1;
```

5. **Eliminar un registro:**

```sql
DELETE FROM Estudiantes 
WHERE ID = 1;
```

## 🔹 Motores de bases de datos que usan SQL

* **MySQL** → muy popular en aplicaciones web.
* **PostgreSQL** → robusto y de código abierto.
* **SQLite** → ligero, usado en apps móviles.
* **SQL Server (Microsoft)** → usado en empresas.
* **Oracle Database** → muy usado en bancos y grandes compañías.

📌 Resumen:
**SQL es el idioma universal para trabajar con bases de datos relacionales.**

### Resumen

#### ¿Qué es un esquema en una base de datos?

Al adentrarnos en el mundo de las bases de datos, toca abordar varios conceptos fundamentales. Uno de ellos es el esquema. Un "esquema" en una base de datos se refiere a una estructura que puede existir dependiendo del área de negocio en el que se trabaja. Es decir, puedes tener múltiples esquemas dentro de una misma base de datos, como uno para contabilidad, otro para facturación, etc. Cada esquema puede contener diferentes objetos, que aprenderemos a continuación.

#### ¿Cuáles son los objetos de base de datos?

Los objetos en una base de datos son componentes esenciales que permiten almacenar y organizar la información. Los principales objetos son:

- **Tablas**: Son la base para almacenar datos, compuestas por filas y columnas. Las filas contienen registros, y las columnas, atributos.
- **Claves primarias**: Son identificadores únicos en las tablas, esenciales para diferenciar registros.
- **Claves foráneas**: Permiten establecer relaciones entre tablas, utilizando identificadores de tabla externa.
- **Vistas**: Pueden ser temporales o materializadas, funcionan como tablas virtuales para consultar datos.
- **Procedimientos almacenados**: Son bloques de código SQL que ejecutan tareas específicas, como consulta o modificación de datos.

#### ¿Qué es la terminología CRUD y cómo se aplica?

CRUD es un acrónimo ampliamente utilizado en la programación y gestión de datos, especialmente en bases de datos relacionales. Significa:

1. **Create**: Crear nuevos registros o estructuras.
2. **Read**: Leer o consultar información almacenada.
3. **Update**: Modificar registros existentes.
4. **Delete**: Eliminar registros o estructuras.

En SQL, estos se traducen, respectivamente, como `CREATE`, `SELECT`, `UPDATE` y `DELETE`. Todos son fundamentales para el manejo efectivo de bases de datos, permitiendo llevar a cabo operaciones básicas de mantenimiento y gestión de datos.

### ¿Cómo se estructuran las bases de datos?

Al hablar de bases de datos, es crucial comprender su estructura jerárquica:

1. **Motor de base de datos**: Es el software encargado de gestionar bases de datos.
2. **Bases de datos**: Cada motor puede contener varias bases de datos, cada una con su propósito específico.
3. **Esquemas**: Dentro de cada base de datos, los esquemas organizan los diversos objetos que se utilizan.

#### ¿Qué tipos de comandos SQL existen?

Para manejar una base de datos de manera efectiva, SQL tiene diferentes tipos de comandos que enseñaremos más adelante:

1. **DDL (Data Definition Language)**: Se utiliza para definir la estructura de la base de datos, por ejemplo, crear tablas.
2. **DML (Data Manipulation Language)**: Permite manipular los datos dentro de las tablas, utilizando los comandos INSERT, UPDATE, y DELETE.
3. **DCL (Data Control Language)**: Gestiona los permisos de acceso a los datos.
4. **TCL (Transaction Control Language)**: Maneja las transacciones, asegurando la consistencia y confiabilidad.
5. **DQL (Data Query Language)**: Interactúa principalmente con el comando `SELECT` para recuperar datos.

Con esta estructura de comandos, puedes realizar operaciones necesarias para gestionar, consultar y mantener datos en cualquier base de datos relacional. ¡Sigue explorando y perfeccionando tu uso de SQL!

## Los comandos SQL más importantes

Aquí tienes una lista con los **comandos SQL más importantes** (los que más se usan en bases de datos relacionales como MySQL, PostgreSQL, SQL Server, Oracle, etc.):

### 📌 **Comandos DDL (Data Definition Language) – Definen la estructura**

* **CREATE** → Crear bases de datos, tablas, vistas, índices.

  ```sql
  CREATE TABLE clientes (
      id INT PRIMARY KEY,
      nombre VARCHAR(100),
      correo VARCHAR(100)
  );
  ```
* **ALTER** → Modificar tablas ya existentes (agregar/eliminar columnas, cambiar tipos de datos).

  ```sql
  ALTER TABLE clientes ADD telefono VARCHAR(20);
  ```
* **DROP** → Eliminar tablas, bases de datos o vistas.

  ```sql
  DROP TABLE clientes;
  ```

### 📌 **Comandos DML (Data Manipulation Language) – Manipulan datos**

* **INSERT** → Insertar datos en una tabla.

  ```sql
  INSERT INTO clientes (id, nombre, correo) VALUES (1, 'Ana', 'ana@mail.com');
  ```
* **SELECT** → Consultar datos.

  ```sql
  SELECT nombre, correo FROM clientes WHERE id = 1;
  ```
* **UPDATE** → Actualizar registros existentes.

  ```sql
  UPDATE clientes SET correo = 'nuevo@mail.com' WHERE id = 1;
  ```
* **DELETE** → Eliminar registros.

  ```sql
  DELETE FROM clientes WHERE id = 1;
  ```

### 📌 **Comandos DCL (Data Control Language) – Seguridad**

* **GRANT** → Otorgar permisos a un usuario.

  ```sql
  GRANT SELECT, INSERT ON clientes TO usuario1;
  ```
* **REVOKE** → Revocar permisos.

  ```sql
  REVOKE INSERT ON clientes FROM usuario1;
  ```

### 📌 **Comandos TCL (Transaction Control Language) – Transacciones**

* **COMMIT** → Guardar los cambios.
* **ROLLBACK** → Revertir cambios en caso de error.
* **SAVEPOINT** → Crear un punto de restauración dentro de una transacción.

👉 En resumen:

* **Definición:** `CREATE, ALTER, DROP`
* **Manipulación:** `INSERT, SELECT, UPDATE, DELETE`
* **Control de seguridad:** `GRANT, REVOKE`
* **Transacciones:** `COMMIT, ROLLBACK, SAVEPOINT`

### Resumen

#### ¿Cuáles son los tipos de agrupación de comandos en bases de datos?

Manipular la estructura y la información de una base de datos es esencial para cualquier desarrollador o administrador de datos. Existen cinco tipos de agrupaciones de comandos con los que puedes interactuar y controlar una base de datos: el Lenguaje de Consulta (DQL), el Lenguaje de Definición de Datos (DDL), el Lenguaje de Manipulación de Datos (DML), el Control de Accesos (DCL), y el Manejo de Transacciones. Estos comandos permiten desde consultar información hasta gestionar transacciones complejas. Aprender a manejarlos te dará más autoridad y control sobre la gestión de datos.

#### ¿Cómo se implementa el lenguaje de consulta de datos?

El Lenguaje de Consulta de Datos, conocido como DQL, permite formular solicitudes de información en una base de datos. Su sintaxis más común es el comando `SELECT`, acompañado de `FROM` y el nombre de la tabla que estás consultando. Además, puedes integrar condicionales y funciones avanzadas para refinar tus consultas. Asimismo, el DQL no solo facilita la recuperación de datos sino que simplifica el proceso de uso de funciones complejas.

##### Ejemplo de sintaxis en SQL:

```sql
SELECT columna1, columna2 
FROM nombre_tabla 
WHERE condición;
```

#### ¿Qué es el lenguaje de definición de estructura?

El Lenguaje de Definición de Datos, o DDL, se enfoca en la estructura de una base de datos. Esto implica crear, modificar o eliminar tablas, procedimientos almacenados, vistas y otros objetos dentro de la base. Emplea varias palabras reservadas que permiten manejar las estructuras de datos al nivel más básico.

#### Sintaxis común para crear y modificar tablas en SQL:

- **Creación de tablas**:

```sql
CREATE TABLE nombre_tabla (
    columna1 tipo_dato,
    columna2 tipo_dato
);
```

- **Modificación de tablas**:

```sql
ALTER TABLE nombre_tabla
ADD nueva_columna tipo_dato;

```
- **Eliminar tablas**:

```sql
DROP TABLE nombre_tabla;
```

#### ¿Qué es el lenguaje de manipulación de datos?

El Lenguaje de Manipulación de Datos, o DML, está diseñado para interactuar con la información interna de las estructuras de base de datos ya creadas. Esto incluye la inserción, la actualización y la eliminación de registros dentro de las tablas.

#### Operaciones comunes en SQL:

- **Insertar datos en una tabla**:

```sql
INSERT INTO nombre_tabla (columna1, columna2) VALUES (valor1, valor2);
```

- **Actualizar datos en una tabla**:

```sql
UPDATE nombre_tabla
SET columna1 = nuevo_valor
WHERE condición;
```

- **Eliminar registros de una tabla**:

```sql
DELETE FROM nombre_tabla WHERE condición;
```

Recuerda que sin un `WHERE`, los comandos `UPDATE` y `DELETE` afectan a todos los registros de la tabla.

#### ¿Cómo se gestionan los controles de acceso en bases de datos?

El Control de Accesos, conocido como DCL, se refiere a cómo otorgar y revocar permisos sobre una base de datos. Esto es crucial para proteger los datos y asegurar que solo los usuarios autorizados puedan acceder y modificar información específica.

#### Ejemplo de comandos en SQL:

- **Otorgar permisos**:

`GRANT SELECT ON nombre_tabla TO usuario;`

- **Revocar permisos**:

`REVOKE SELECT ON nombre_tabla FROM usuario;`

#### ¿Qué es el lenguaje de control de transacciones?

El Lenguaje de Control de Transacciones está diseñado para manejar operaciones complejas dentro de una base de datos. Es vital para operaciones que requieren un alto control, permitiendo definir puntos de referencia, retroceder cambios o confirmar transacciones usando `SAVEPOINT`, `ROLLBACK` y `COMMIT`.

Conocer y dominar estos comandos no solo te proporciona herramientas esenciales para trabajar con bases de datos, sino que también optimiza esfuerzos y asegura precisión en la gestión de datos.

## Operaciones básicas en SQL

Las **operaciones básicas en SQL** se enfocan en el manejo de los datos dentro de una base de datos. Son las que más se usan en el día a día y corresponden al **CRUD**:

### 🔹 **1. SELECT – Consultar datos**

Permite **leer información** de una o varias tablas.

```sql
-- Obtener todas las columnas
SELECT * FROM clientes;

-- Obtener columnas específicas
SELECT nombre, correo FROM clientes;

-- Filtrar con condiciones
SELECT * FROM clientes WHERE id = 1;

-- Ordenar resultados
SELECT * FROM clientes ORDER BY nombre ASC;
```

### 🔹 **2. INSERT – Insertar datos**

Sirve para **agregar registros nuevos** a una tabla.

```sql
INSERT INTO clientes (id, nombre, correo) 
VALUES (1, 'Ana', 'ana@mail.com');
```

### 🔹 **3. UPDATE – Actualizar datos**

Permite **modificar registros existentes**.

```sql
UPDATE clientes 
SET correo = 'nuevo@mail.com' 
WHERE id = 1;
```

### 🔹 **4. DELETE – Eliminar datos**

Se utiliza para **borrar registros**.

```sql
DELETE FROM clientes WHERE id = 1;
```

### 🔹 **5. CREATE / DROP – Crear o eliminar tablas**

Son básicos cuando creamos estructuras.

```sql
-- Crear tabla
CREATE TABLE clientes (
    id INT PRIMARY KEY,
    nombre VARCHAR(100),
    correo VARCHAR(100)
);

-- Eliminar tabla
DROP TABLE clientes;
```

✅ En resumen, las **operaciones básicas en SQL** son:

* **SELECT** → Leer datos
* **INSERT** → Insertar datos
* **UPDATE** → Modificar datos
* **DELETE** → Borrar datos
* **CREATE / DROP** → Crear o eliminar tablas

### Resumen

#### ¿Cómo manipular datos en una base de datos?

Cuando se trata de gestionar datos en una base de datos, es esencial dominar diversas operaciones y funciones. Estas herramientas no solo permiten organizar y analizar eficazmente la información, sino que también ofrecen la flexibilidad de realizar consultas complejas y personalizadas. Vamos a explorar algunas de las operaciones más comunes que puedes implementar para maximizar la eficiencia de tu base de datos y extraer datos de acuerdo a tus necesidades específicas.

#### ¿Qué son las funciones de agregación y cómo se utilizan?

Las funciones de agregación son operaciones cruciales que nos permiten resumir y analizar datos. Algunas de las funciones más utilizadas incluyen:

- **SUMA**: Calcula la suma total de un conjunto de valores. Utilizado comúnmente para sumar salarios, ingresos, etc.
- **PROMEDIO**: Determina el promedio de un conjunto de datos, útil para calcular el salario medio en un departamento.
- **CONTEO**: Cuenta el número de registros en una tabla. Útil para saber cuántos empleados hay en una empresa.
- **MÍNIMO y MÁXIMO**: Extraen el valor mínimo o máximo de un conjunto, respectivamente.

Estas funciones se integran en consultas estructuradas con la sintaxis SQL. Por ejemplo, al utilizar la cláusula `SELECT`, podemos ejecutar una consulta que agrupe empleados por departamento y calcule el total de salarios:

```sql
SELECT departamento, SUM(salario) AS total_salario
FROM empleados
WHERE salario > 40000
GROUP BY departamento;
```

Además, estas funciones se pueden utilizar junto con condiciones adicionales, como fechas o rangos específicos.

#### ¿Cómo aplicar reglas condicionales avanzadas?

El uso de condicionales nos da la flexibilidad de aplicar diferentes reglas de negocio en la manipulación de datos. El uso del `CASE` es una metodología avanzada que nos permite gestionar datos según ciertas condiciones. Por ejemplo, podemos clasificar salarios como junior o senior:

```sql
SELECT 
  CASE 
    WHEN salario < 50000 THEN 'Junior'
    ELSE 'Senior'
  END AS nivel_salarial
FROM empleados;
```

Este ejemplo clasifica a los empleados basado en si sus salarios son menores o iguales a una cierta cantidad. De esta manera, podemos crear nuevas columnas a partir de decisiones lógicas.

#### ¿Qué son las uniones (joins) y cómo se aplican?

Las **uniones** son herramientas poderosas para combinar datos de diferentes tablas. Algunos de los tipos más comunes de `JOIN` son:

- **INNER JOIN**: Retorna las filas que tienen coincidencias en ambas tablas.
- **LEFT JOIN**: Muestra todas las filas de la tabla izquierda y las coincidencias de la tabla derecha.
- **RIGHT JOIN**: Muestra todas las filas de la tabla derecha y las coincidencias de la tabla izquierda.
- **FULL JOIN**: Combina todas las filas de ambas tablas, mostrando coincidencias donde las haya.

Ejemplo de `INNER JOIN`:

```sql
SELECT e.nombre, d.nombre_departamento
FROM empleados e
INNER JOIN departamentos d ON e.departamento_id = d.id;
```

Este consulta retorna los nombres de los empleados junto con sus departamentos cuando hay una coincidencia entre ambas tablas.

#### ¿Cómo implementar condicionales y filtrados avanzados?

Además de las uniones, podemos aplicar condicionales para filtrar datos más precisamente. Operadores como `AND`, `OR`, `NOT`, `BETWEEN`, `IN` y `LIKE` son fundamentales para manejar condiciones complejas:

- Usando `BETWEEN` para buscar un rango de valores:

`SELECT * FROM empleados WHERE salario BETWEEN 40000 AND 60000;`

- Usando `LIKE` para buscar patrones en cadenas de texto:

`SELECT * FROM empleados WHERE nombre LIKE 'J%';`

Estas operaciones nos proporcionan el control necesario para extraer y manipular datos según criterios específicos.

Explorar estas técnicas y funciones es clave para cualquier profesional que desee aprovechar al máximo su habilidad de gestionar bases de datos. Persiste en tu aprendizaje y práctica para poder implementar consultas eficientes y efectivas. Siempre hay nuevas formas de optimizar y personalizar la búsqueda y manipulación de información.

## Modelo Entidad Relación (ER)

¡Perfecto! 🚀 Vamos a ver qué es un **Modelo Entidad-Relación (ER)**, ya que es una de las bases de la **modelación de bases de datos**.

### 📌 **Modelo Entidad–Relación (ER)**

El **Modelo Entidad-Relación (ER)** es una forma gráfica de representar cómo se estructuran y se relacionan los datos dentro de una base de datos.
Fue propuesto por **Peter Chen en 1976** y se usa como **primer paso antes de implementar la base en SQL**.

### 🔹 **Elementos principales del modelo ER**

1. **Entidad** 🟦

   * Representa un objeto real o concepto que queremos almacenar en la base de datos.
   * Ejemplos: **Cliente**, **Producto**, **Empleado**, **Factura**.
   * Se dibujan como **rectángulos**.

2. **Atributo** 🟡

   * Son las propiedades o características de las entidades.
   * Ejemplo: un **Cliente** puede tener *id, nombre, correo, teléfono*.
   * Se dibujan como **óvalos** unidos a la entidad.

3. **Clave primaria** 🔑

   * Es un atributo (o conjunto de atributos) que **identifica de forma única** a cada entidad.
   * Ejemplo: `id_cliente`.

4. **Relación** 🔺

   * Representa cómo interactúan las entidades entre sí.
   * Ejemplo: **Cliente** "realiza" **Compra**, o **Empleado** "atiende" **Factura**.
   * Se dibujan como **rombos** conectados a las entidades.

5. **Cardinalidad** 🔄

   * Indica cuántos elementos de una entidad se asocian con cuántos de otra.
   * Tipos principales:

     * **1 a 1 (1:1)** → Un empleado tiene un solo usuario de acceso.
     * **1 a N (1\:N)** → Un cliente puede hacer muchas compras.
     * **N a M (N\:M)** → Un estudiante puede estar en muchos cursos y un curso puede tener muchos estudiantes.

### 🔹 **Ejemplo de diagrama ER**

📌 Supongamos que modelamos un sistema de **ventas**:

* **Cliente** (*id\_cliente, nombre, correo*)
* **Producto** (*id\_producto, nombre, precio*)
* **Factura** (*id\_factura, fecha*)

**Relaciones**:

* Un **Cliente** genera **muchas Facturas** → (1\:N).
* Una **Factura** puede contener **muchos Productos** y un **Producto** puede estar en muchas Facturas → (N\:M).

📊 El diagrama ER quedaría (resumido en texto):

```
Cliente (id_cliente, nombre, correo) ───< Factura (id_factura, fecha) >─── Producto (id_producto, nombre, precio)
```

✅ **En resumen**:
El **Modelo Entidad–Relación (ER)** sirve para diseñar cómo estarán organizados los datos antes de implementarlos en SQL, mostrando **entidades, atributos, relaciones y cardinalidades**.

### Resumen

#### ¿Cómo diseñar una base de datos usando el modelo entidad-relación?

El diseño de bases de datos es un proceso esencial para estructurar el almacenamiento y la gestión de datos de manera eficiente. Un enfoque popular que ayuda a los desarrolladores a este propósito es el **modelo entidad-relación (ER)**. Este modelo proporciona una representación gráfica que facilita el entendimiento de la estructura y funcionalidad de la base de datos antes de su implementación. A través del uso de elementos visuales como rectángulos y líneas, este modelo ilustra las entidades implicadas y las relaciones entre ellas, optimizando el diseño en etapas tempranas.

#### ¿Qué son las entidades y sus atributos?

En el contexto de un modelo ER, las entidades representan objetos concretos o conceptuales presentes en una base de datos. Estas entidades pueden ser tangibles, como estudiantes o aulas, o abstractas, como asignaturas. Una entidad se representa gráficamente mediante un rectángulo y, en la base de datos, se corresponde con una tabla. Cada tabla contiene una serie de atributos que describen propiedades específicas de la entidad.

- **Entidades concretas**: Representan objetos físicos tangibles.

 - Ejemplo: Estudiantes, aulas.

- **Entidades abstractas**: No tienen una existencia física.

 - Ejemplo: Asignaturas.

#### ¿Cuáles son los tipos de atributos?
Los atributos son las propiedades que delinean una entidad y en una base de datos forman las columnas de la tabla.

1. **Atributos simples:** No pueden subdividirse.
 - Ejemplo: Estatura.
2. **Atributos compuestos**: Pueden dividirse en varios sub-atributos.
 - Ejemplo: Dirección (país, región, ciudad, calle).
3. **Atributos monovalorados o clave**: Actúan como identificadores únicos, conocidos también como claves primarias.
 - Ejemplo: ID estudiante, ID profesor.
4. **Atributos multivalorados**: Pueden contener múltiples valores asociados.
 - Ejemplo: Correos electrónicos de un estudiante.
5. **Atributos derivados**: Se calculan a partir de otros atributos.
 - Ejemplo: Edad (derivada de la fecha de nacimiento).

#### ¿Cómo se representan las relaciones entre las entidades?

Las **relaciones** en un modelo ER permiten establecer cómo interactúan las diferentes entidades. Para definir estas interacciones, se utiliza el concepto de cardinalidad, que especifica el número de asociaciones posibles entre entidades.

1. **Cardinalidad uno a uno**: Cada entidad se asocia con una y solo una entidad complementaria.

2. **Cardinalidad uno a muchos**: Una entidad puede relacionarse con múltiples entidades complementarias.

3. **Cardinalidad muchos a muchos**: Varias entidades pueden asociarse con muchas otras entidades distintas.

Visualmente, las relaciones se representan con líneas que conectan las entidades y pueden incluir símbolos que indican la obligatoriedad o la opcionalidad de las mismas. La cardinalidad se representa mediante un sistema que incluye:

- **Uno obligatorio**: Una línea directa con dos cruces.
- **Muchos obligatorios**: Línea acompañada de un triángulo de líneas.
- **Opcional uno**: Línea con un óvalo que indica que es opcional.
- **Opcional muchos**: Línea con un óvalo acompañada de un triángulo de líneas.

#### ¿Cómo interpretar un diagrama de entidad-relación?

La correcta interpretación de un diagrama ER es crítica para la implementación exitosa de una base de datos. Los rectángulos representan las entidades fundamentales, mientras que los rombos denotan las relaciones o acciones que una entidad puede realizar en relación con otra. Comprender las representaciones gráficas y los componentes permite a los diseñadores identificar correctamente las estructuras necesarias y anticipar el comportamiento de la base de datos en la práctica.

En conclusión, el modelo ER es una herramienta esencial que permite estructurar el diseño de bases de datos comprensiblemente, abarcando tanto las entidades como sus relaciones, facilitando así una implementación más eficaz y óptima.

## Normalización

La **normalización** en bases de datos es un proceso fundamental para organizar los datos de manera eficiente, evitando redundancias y anomalías.

### 📌 **Normalización en Bases de Datos**

La **normalización** es el proceso de **estructurar las tablas y sus relaciones** con el fin de:

✅ Eliminar redundancia de datos.
✅ Mejorar la integridad de los datos.
✅ Evitar problemas al insertar, actualizar o eliminar información.
✅ Facilitar el mantenimiento de la base de datos.

Fue propuesta por **Edgar F. Codd**, el padre del modelo relacional.

### 🔹 **Formas Normales (FN)**

Existen varios niveles de normalización llamados **formas normales**. Cada forma aplica reglas más estrictas:

### 🔸 **Primera Forma Normal (1FN)**

* Cada columna debe contener valores **atómicos** (no divisibles).
* No se permiten **listas ni grupos repetidos** en una misma columna.
* Ejemplo ❌:

  ```
  Cliente(id, nombre, teléfonos)
  ```

  (si un cliente tiene varios teléfonos, esto rompe 1FN).

  ✅ Corrección:

  ```
  Cliente(id, nombre)
  Teléfono(id_cliente, teléfono)
  ```

### 🔸 **Segunda Forma Normal (2FN)**

* Cumple 1FN.
* Todos los atributos dependen de la **clave primaria completa**, no de una parte de ella.
* Aplica solo cuando la clave primaria es **compuesta**.
* Ejemplo ❌:

  ```
  Pedido(id_pedido, id_producto, cantidad, nombre_cliente)
  ```

  → Aquí, `nombre_cliente` depende solo de `id_pedido`, no de la clave compuesta `(id_pedido, id_producto)`.

  ✅ Corrección:

  ```
  Pedido(id_pedido, id_producto, cantidad)
  Cliente(id_cliente, nombre_cliente)
  ```

### 🔸 **Tercera Forma Normal (3FN)**

* Cumple 2FN.
* No debe haber **dependencias transitivas** (un atributo no clave depende de otro atributo no clave).
* Ejemplo ❌:

  ```
  Empleado(id_empleado, nombre, id_departamento, nombre_departamento)
  ```

  → `nombre_departamento` depende de `id_departamento`, no directamente de la clave `id_empleado`.

  ✅ Corrección:

  ```
  Empleado(id_empleado, nombre, id_departamento)
  Departamento(id_departamento, nombre_departamento)
  ```

## 🔹 Formas más avanzadas

* **BCNF (Boyce-Codd Normal Form)** → Variante más estricta de 3FN.
* **4FN y 5FN** → Elimina dependencias multivaluadas y de unión.

### 🔹 Ejemplo práctico de normalización

### Tabla NO normalizada:

```
Factura(id_factura, cliente, producto1, producto2, producto3, total)
```

### 1FN (datos atómicos, sin repeticiones):

```
Factura(id_factura, cliente, producto, total)
```

### 2FN (dependencia total de la clave):

```
Factura(id_factura, id_cliente, total)
DetalleFactura(id_factura, id_producto, cantidad)
```

### 3FN (sin dependencias transitivas):

```
Cliente(id_cliente, nombre_cliente)
Producto(id_producto, nombre, precio)
Factura(id_factura, id_cliente, total)
DetalleFactura(id_factura, id_producto, cantidad)
```

✅ **En resumen**:
La **normalización** mejora la calidad del diseño de la base de datos al eliminar redundancia y garantizar la consistencia, aplicando reglas paso a paso (1FN, 2FN, 3FN, …).

### Resumen

#### ¿Qué es la normalización en bases de datos?

La normalización es una técnica crucial en la creación de bases de datos que busca minimizar la redundancia de datos y garantizar su integridad. Permite dividir una tabla grande con una estructura variada en múltiples tablas siguiendo ciertas reglas. Este proceso es esencial para mejorar la eficiencia de las consultas y asegurar la calidad de los datos.

#### ¿En qué consiste la primera forma normal?

La primera regla de normalización incluye tres puntos clave:

1. **Eliminar grupos repetitivos**: Asegurarse de que cada columna en una tabla contenga valores atómicos, es decir, no divisibles.
2. **Garantizar registros únicos**: Cada fila debe ser única.
3. **Esquema de tabla lógica**: Dividir la información en tablas específicas según su tipo.

Por ejemplo, consideremos una tabla que almacena información de estudiantes y los cursos en los que están matriculados. Una mala práctica sería tener una columna 'cursos' con entradas como "matemáticas, física". Aquí se rompe la primera regla porque no es un dato atómico. Se puede resolver al crear tablas separadas para alumnos y matrículas, vinculando estudiantes con cursos de manera adecuada.

#### ¿Cómo aplicamos la segunda forma?

La segunda forma normal se basa en los preceptos de la primera, añadiendo la eliminación de dependencias parciales. Es imprescindible:

- Cumplir con la primera norma.
- Asegurar que cada atributo no clave dependa completamente de la clave primaria.

Por ejemplo, si una universidad almacena la calificación de estudiantes por curso, y en la columna 'profesor' solo depende del curso, no de la clave compuesta del estudiante ID, se está violando esta forma normal. Una manera de solucionarlo es crear dos tablas: una para matrículas con 'estudiante ID', 'curso' y 'grado'; otra para 'cursos' con 'profesor', eliminando dichas dependencias parciales.

#### ¿Qué garantiza la tercera forma?

La tercera forma normal requiere el cumplimiento de la segunda norma y, además, la eliminación de dependencias transitivas. Esto significa que los atributos no claves no deben depender de otros atributos no claves.

Imaginemos una tabla que contiene la dirección del profesor y una columna de 'cursos' y 'profesor'. Aquí, la solución es crear una tabla separada para la información del profesor, incluyendo detalles como nombre, identificación, dirección, y otra tabla para los cursos, logrando una separación de responsabilidades adecuada en las tablas.

#### ¿Qué es la Forma Normal de Boyce-Codd (BCNF)?

La BCNF es un refinamiento de la tercera forma normal que garantiza que cada determinante sea una clave candidata. Por ejemplo, si tenemos una tabla con 'curso', 'profesor', y 'clase', y un curso depende del salón y el profesor, pero este último no es clave, se infringe esta norma. La solución es dividir en una tabla con 'curso' y 'profesor', y otra con 'salón' y 'curso', eliminando relaciones no directas.

#### ¿Cómo se aplica la cuarta y quinta forma normal?

La cuarta y quinta formas normales llevan a un nivel más profundo la separación y simplificación de tablas:

- **Cuarta forma normal**: Evitar dependencias multivaluadas. Si un 'curso' tiene múltiples 'profesores' y 'salones', crearemos tablas separadas para evitar redundancias, como una para los cursos, otra para cursos y profesores, y una más para las relaciones con el salón.

- **Quinta forma normal**: Implica dividir tablas en otras más pequeñas para reconstruir los datos originales sin duplicar información. Así se garantiza que las combinaciones entre 'cursos', 'profesores', y 'salones' se reconstruyan de manera clara y eficiente.

La aplicación de estas formas asegura bases de datos robustas, evitando duplicados y manteniendo integridad, crucial para la calidad y eficiencia de sistemas de gestión de datos.

## Tipos de datos en SQL

¡Perfecto! 🚀 Te hago un resumen de los **tipos de datos en SQL** más comunes, organizados por categorías:

### 📌 1. **Números**

* **INT / INTEGER** → Enteros (ej: 1, 200, -15)
* **SMALLINT** → Enteros pequeños, ocupa menos memoria
* **BIGINT** → Enteros grandes
* **DECIMAL(p,s) / NUMERIC(p,s)** → Números exactos con decimales (ej: 123.45)

  * `p` = precisión (número total de dígitos)
  * `s` = escala (dígitos después del punto decimal)
* **FLOAT / REAL / DOUBLE** → Números con decimales pero aproximados (para cálculos científicos)

### 📌 2. **Texto**

* **CHAR(n)** → Cadena de longitud fija (ej: CHAR(5) → "Mario ")
* **VARCHAR(n)** → Cadena de longitud variable (ej: VARCHAR(50))
* **TEXT** → Texto largo (usado en descripciones, comentarios, etc.)

### 📌 3. **Fechas y tiempos**

* **DATE** → Solo fecha (AAAA-MM-DD)
* **TIME** → Solo hora (HH\:MM\:SS)
* **DATETIME** → Fecha y hora
* **TIMESTAMP** → Fecha y hora con zona horaria (se actualiza automáticamente en algunos motores como MySQL)
* **YEAR** → Solo el año

### 📌 4. **Booleanos**

* **BOOLEAN** → Verdadero (TRUE) o falso (FALSE)

### 📌 5. **Binarios**

* **BLOB** → Almacena datos binarios (imágenes, audio, archivos)
* **BINARY / VARBINARY** → Datos binarios fijos o variables

### 📌 6. **Otros (dependen del motor de SQL)**

* **ENUM** → Lista de valores predefinidos (ej: ENUM('Pequeño','Mediano','Grande'))
* **JSON** → Para almacenar datos en formato JSON (PostgreSQL, MySQL, SQL Server)

👉 Ejemplo en SQL:

```sql
CREATE TABLE empleados (
    id INT PRIMARY KEY,
    nombre VARCHAR(50),
    salario DECIMAL(10,2),
    fecha_contratacion DATE,
    activo BOOLEAN
);
```

### Resumen

#### ¿Qué son los tipos de datos y por qué son importantes?

Los tipos de datos son fundamentales para la gestión y optimización de bases de datos, ya que determinan el tipo de contenido que puede entrar en una columna, variable, o parámetro de un objeto. Sin una correcta definición, podríamos enfrentar desafíos en la eficiencia y calidad de los datos. Los tipos de datos no solo ayudan a definir la estructura y formato necesarios, sino que también juegan un papel crucial en la reducción de errores durante el procesamiento de datos.

#### ¿Cómo se clasifican los tipos de datos?
#### ¿Cuáles son los tipos de datos numéricos más comunes?

Los datos numéricos son esenciales para manejar cantidades y valores. Estos son los más utilizados:

- **`int` (entero)**: Con una capacidad de 4 bytes, es ideal para claves primarias o conteos simples, como el número de productos o personas.

- `smallint`: Similar al `int`, pero con capacidad de 2 bytes, adecuado para cifras más pequeñas.

- `bigint`: Para grandes números, con una capacidad de 8 bytes.

- `decimal`: Usa la sintaxis `decimal(p,s)`, determinando precisión hasta `s` lugares después de la coma. Muy útil para valores con precisión fija.

- `float`: Similar al decimal, pero con precisión ilimitada, útil cuando la precisión exacta no es crítica.

#### ¿Cuáles son los tipos de datos de texto más utilizados?

En el manejo textual, estos son los tipos de datos principales:

- `char(n)`: Define una longitud fija de texto. Ocupa el espacio completo definido, independientemente de la longitud real del texto almacenado.

- `varchar(n)`: Muy similar a `char`, pero almacena solo el tamaño real del texto, con un máximo de `n`. Más eficiente en espacio que `char`.

- `text`: Utilizado para textos largos, como descripciones de productos. No requiere especificar una longitud máxima.

#### ¿Qué tipos de datos de tiempo existen y cuándo usarlos?

Los datos de tiempo permiten manejar información temporal, esenciales para muchas aplicaciones:

- `time`: Para almacenar solo la hora.

- `date`: Exclusivamente para fechas.

- `datetime`: Combina fecha y hora, ideal para registros de eventos.

- `timestamp`: Similar a `datetime` pero también almacena la zona horaria UTC, crucial para aplicaciones distribuidas globalmente.

#### ¿Por qué es vital seleccionar correctamente el tipo de datos?

La elección del tipo de datos impacta directamente en la eficiencia y calidad del procesamiento de datos. Asignaciones incorrectas pueden llevar a:

- **Problemas de limpieza**: Prolongan el tiempo de transformación y limpieza de datos.
- **Ineficiencias en almacenamiento**: Desperdicio de espacio y recursos.
- **Errores en procesamiento**: Dificultad para realizar cálculos y reportes precisos.

Definir adecuadamente el tipo de datos desde la fase inicial de diseño, ya sea en la creación de tablas o procedimientos almacenados, es esencial. Esto garantiza la correcta interpretación y manipulación de cualquier información solicitada. También se anima a exploradores de datos a comentar y aportar sobre tipos de datos no mencionados o dudas sobre los explicados. Esta interacción es fundamental para enriquecer el aprendizaje y comprensión global del manejo de datos.

## ¿Cómo crear una base de datos en SQL?

En SQL, crear una base de datos es muy sencillo. Te muestro paso a paso:

### 📌 1. Crear una base de datos

El comando básico es:

```sql
CREATE DATABASE nombre_de_la_base;
```

Ejemplo:

```sql
CREATE DATABASE empresa;
```

Esto crea una base de datos llamada **empresa**.

### 📌 2. Usar la base de datos

Después de crearla, debes seleccionarla para trabajar en ella:

```sql
USE empresa;
```

### 📌 3. Crear tablas dentro de la base

Una base de datos vacía no tiene información. Necesitas crear tablas:

```sql
CREATE TABLE empleados (
    id INT PRIMARY KEY AUTO_INCREMENT,
    nombre VARCHAR(50) NOT NULL,
    puesto VARCHAR(50),
    salario DECIMAL(10,2),
    fecha_contratacion DATE
);
```

### 📌 4. Insertar datos en la tabla

```sql
INSERT INTO empleados (nombre, puesto, salario, fecha_contratacion)
VALUES ('Juan Pérez', 'Ingeniero', 3500.50, '2025-08-22');
```

### 📌 5. Consultar los datos

```sql
SELECT * FROM empleados;
```

Esto devuelve todos los registros de la tabla **empleados**.

👉 Diferencia por motor:

* **MySQL / MariaDB**: Usa `CREATE DATABASE` y `USE`.
* **PostgreSQL**: Se usa igual, pero en vez de `USE` debes conectarte con `\c nombre_base` en la terminal.
* **SQL Server**: También usa `CREATE DATABASE`, pero la conexión se hace con `USE nombre_base`.

¿Quieres que te haga un **ejemplo completo paso a paso** con:

1. Crear base de datos
2. Crear tabla
3. Insertar datos
4. Consultar

### Resumen

#### ¿Por qué es crucial seguir buenas prácticas al crear bases de datos?

La correcta creación de bases de datos, tablas y sus relaciones es fundamental para un ingeniero de datos. Estas habilidades no solo facilitan la organización de la información, sino que también optimizan el rendimiento y la ejecución de los procesos. Al considerar ciertos principios, como las reglas de normalización y el uso adecuado de claves primarias y foráneas, se garantiza integridad y eficiencia. Además, es esencial comprender los conceptos de claves de negocio y subrogadas, junto con la adecuada identificación de tipos de datos para mantener la calidad de la información.

#### ¿Cómo crear una base de datos y tablas en SQL?

Cuando se trabaja en SQL, la sintaxis y el formato son cruciales. Comienza creando una base de datos con el comando `CREATE DATABASE` seguido del nombre deseado, recordando que el estilo de nombres (mayúsculas o minúsculas) debe ser coherente para todas las bases de datos, tablas y atributos.

#### Sintaxis básica para crear una tabla
Para crear tablas, utiliza el comando `CREATE TABLE` seguido del nombre de la tabla y una lista de sus atributos:

```sql
CREATE TABLE Estudiantes (
    ID_Estudiante INT PRIMARY KEY,
    Nombre VARCHAR(50),
    Apellido VARCHAR(50),
    Edad INT,
    Correo VARCHAR(100),
    Fecha_Carga DATE,
    Fecha_Modificacion DATE
);
```

Es importante comenzar con las claves primarias, subrogadas o de negocio. Por ejemplo, en un comercio digital, un producto con un ID interno sería mejor gestionado con una clave subrogada dentro de la base de datos, mientras que la clave de negocio podría estar más relacionada con la identificación externa del producto.

#### Buenas prácticas adicionales

1. Incluye atributos de fecha de carga y modificación en tus tablas para control de versiones y soporte.
2. Define los nombres de tablas y atributos en un solo idioma para prevenir errores de interpretación.
3. Dependiendo del motor de base de datos, realiza ajustes necesarios como el uso de `IDENTITY` o `AUTOINCREMENT` para las claves primarias numéricas.

#### ¿Cómo gestionar las relaciones entre tablas con foreign keys?

La clave foránea o foreign key es fundamental para relacionar tablas. Al definir estos vínculos, debes especificar qué atributo se relaciona con otra tabla. Utiliza la sentencia `FOREIGN KEY` para establecer estas conexiones.

#### Ejemplo de relación entre tablas

Imaginemos que deseamos relacionar la tabla de estudiantes con la de instructores utilizando el atributo `ID_Instructor`:

```sql
CREATE TABLE Cursos (
    ID_Curso INT PRIMARY KEY,
    Nombre_Curso VARCHAR(100),
    ID_Instructor INT,
    FOREIGN KEY (ID_Instructor) REFERENCES Instructores(ID_Instructor)
);
```

Esta declaración asegura que cada registro de curso tenga asignado un instructor existente en la tabla `Instructores`.

#### ¿Qué considerar al usar distintos motores de bases de datos?

No todos los comandos son compatibles con todos los motores de bases de datos. Por ejemplo:

- En motores como SQL Server se puede usar `IDENTITY` para autoincrementar claves.
- En MySQL se utiliza `AUTO_INCREMENT`.
- SQLite no soporta directamente estas funcionalidades, pero se pueden implementar mediante procedimientos almacenados.

Es fundamental adaptar el código según el motor de base de datos usado y saber que algunas funcionalidades pueden variar o requerir soluciones alternativas.

#### Práctica recomendada

Te desafío a crear una tabla para instructores siguiendo estas pautas, y a compartir tus hallazgos o dudas sobre la sintaxis y tipos de datos en los comentarios. También, intenta crear una tabla de relación entre estudiantes y cursos, identificando cómo conectarás las claves foráneas, y revisa qué otros campos podrías incluir. Esta práctica fortalecerá tus habilidades y profundidad de conocimiento en gestión de bases de datos.

**Lecturas recomendadas**

[GitHub - platzi/curso-sql: Platzi curso de Bases de Datos con SQL 💚 | Selecciones, Joins, Agrupaciones | Domina 👩‍💻 herramientas clave desde cero hasta crear bases de datos robustas y eficientes](https://github.com/platzi/curso-sql)

[Edito usado en clase - SQL Online Compiler - Next gen SQL Editor](https://sqliteonline.com/)

## Buenas practicas de bases de datos con SQL

Las **buenas prácticas en bases de datos con SQL** son clave para que tu sistema sea **escalable, seguro y fácil de mantener**. Aquí te dejo una lista organizada:

### 🔹 1. Diseño y modelado

* **Normalización**: al menos hasta 3FN (Tercera Forma Normal) para evitar duplicación de datos.
* **Denormalización consciente**: solo si necesitas velocidad y sabes que el costo en duplicidad es manejable.
* **Claves primarias**: siempre define una `PRIMARY KEY` clara y única.
* **Claves externas (FOREIGN KEY)**: mantienen integridad entre tablas (ejemplo: `students` → `courses`).
* **Tipos de datos correctos**: usa `INT` para números, `DATE/TIMESTAMP` para fechas, `BOOLEAN` para valores lógicos.
* **Longitudes adecuadas**: no pongas `VARCHAR(500)` si solo necesitas `VARCHAR(50)`.

### 🔹 2. Rendimiento

* **Índices**:

  * Crea índices en columnas que usas mucho en `WHERE`, `JOIN` o `ORDER BY`.
  * No abuses: demasiados índices ralentizan `INSERT`/`UPDATE`.
* **Consultas eficientes**:

  * Usa `SELECT columnas` en vez de `SELECT *`.
  * Evita subconsultas innecesarias, prefiere `JOIN`.
* **Particionamiento y sharding** (cuando hay muchos datos).

### 🔹 3. Seguridad

* **Principio de privilegios mínimos**:

  * Crea usuarios con permisos limitados (`SELECT`, `INSERT`, etc.).
* **Parámetros en consultas**: evita concatenar strings para prevenir **SQL Injection**.

  * Ejemplo en Python con SQLite:

    ```python
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    ```
* **Encriptación de datos sensibles** (ejemplo: contraseñas → `bcrypt`, nunca en texto plano).
* **Backups automáticos**: siempre configura copias de seguridad periódicas.

### 🔹 4. Escalabilidad y mantenimiento

* **Nombres claros**:

  * Tablas en plural: `students`, `courses`.
  * Columnas descriptivas: `created_at`, `updated_at`, no solo `date1`.
* **Documentación**: escribe comentarios sobre relaciones y triggers.
* **Migraciones controladas**: usa herramientas como Alembic (Python) o Liquibase para mantener versiones.
* **Monitoreo**: revisa logs y rendimiento (`EXPLAIN` para analizar queries).

### 🔹 5. Ejemplo de buenas prácticas en tabla

```sql
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    age INTEGER CHECK (age BETWEEN 0 AND 120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

👉 Estas son prácticas que te harán ver como un **buen programador de bases de datos** en entrevistas.

### Resumen

#### ¿Qué es la normalización y por qué es importante?

La normalización es una práctica crucial en la manipulación de bases de datos. Su objetivo es garantizar la integridad y organización de los datos, permitiendo la adecuada estructuración de la información y el correcto relacionamiento entre tablas. Imaginemos una mudanza como metáfora: organizamos por áreas (como cocina o habitación) y no almacenamos un electrodoméstico donde estarían objetos de dormitorio. Con esta misma lógica, la normalización ayuda a identificar las áreas de negocio y atribuir de manera precisa cada objeto en una base de datos, evitando incongruencias y asegurando datos atómicos y bien relacionados.

#### ¿Cómo se aplica la primera forma normal?

La primera forma normal (1NF) se centra en la atomicidad de los datos y en la especificación de una clave primaria para cada registro. Los pasos a seguir incluyen:

- Asegurar que toda la información sea atómica, es decir, indivisible.
- Utilizar una clave primaria para diferenciar cada registro.
- Centralizar la información por columnas, evitando combinar diferentes tipos de información en un solo campo.

Por ejemplo, en una tabla de estudiantes y cursos, si un campo contiene múltiples cursos de forma conjunta, como "A, B, C", se estaría violando este principio ya que los datos no son indivisibles.

**Ejemplo**:

Si Marco está inscrito en los cursos A, B y C, cada inscripción debería ser un registro separado:

```bash
ID | Estudiante | Curso
1  | Marco      | A
1  | Marco      | B
1  | Marco      | C
```

#### ¿Qué implica la segunda forma normal?

La segunda forma normal (2NF) requiere que todos los atributos no clave dependan de la clave primaria. Esto significa que no debe haber dependencias parciales de la clave.

- Cumplir previamente con la 1NF.
- Crear tablas separadas para grupos de datos relacionados y establecer relaciones entre ellas mediante claves foráneas.

Por ejemplo, una tabla de cursos separada con su propia clave primaria elimina la redundancia de datos de curso en la tabla de estudiantes:

```bash
Tabla Cursos
ID | Curso
1  | A
2  | B
3  | C

Tabla Inscripciones
EstudianteID | CursoID
1            | 1
1            | 2
1            | 3
```

#### ¿Cómo se configura la tercera forma normal?

La tercera forma normal (3NF) afirma que los atributos no clave deben ser completamente independientes de los demás. Esto se traduce en:

- Asegurar que no haya dependencias funcionales transitivas.
- Mantener independiente toda información no relacionada directamente con la clave primaria.

Para nuestro ejemplo, las tablas de estudiantes y cursos deben contener solo información pertinente, evitando que un atributo dependa de otro que no sea la clave primaria.

**Ejemplo de tablas**:

1. **Tabla Estudiantes**: Contendrá solo detalles relevantes del estudiante:

- Nombre
- Apellido
- Identificación

2. **Tabla Cursos**: Incluirá solo información acerca de los cursos:

- Descripción
- Capacidad

3. **Tabla Intermedia**: Conectará estudiantes y cursos:

- EstudianteID

- CursoID

#### ¿Cómo organizar información de direcciones?

En el caso de las direcciones, abundar detalles atómicos es esencial.

- Separar el país, departamento y ciudad.
- Relacionar cada elemento mediante claves foráneas para mantener integridad y reducción de redundancias.

Cada empresa puede optar por estructuras diferentes, pero la justificación en base a las formas normales asegura un desarrollo más sencillo y mayor calidad de datos.

La normalización no solo maximiza la eficiencia del almacenamiento de datos, sino que también mejora las operaciones de transformación, análisis y calidad de los mismos, otorgando un sistema robusto y eficiente.

## Insertando Datos con SQL (INSERT)

Vamos a practicar cómo insertar datos en SQL (`INSERT INTO`).
Siguiendo con tu ejemplo de **Instructor** y **Courses**, te muestro cómo hacerlo paso a paso:

### 🔹 Insertar en **Instructor**

```sql
INSERT INTO Instructor (firstname, lastname, age, email, assignature)
VALUES ('Carlos', 'Pérez', 40, 'carlos.perez@email.com', 'Matemáticas');

INSERT INTO Instructor (firstname, lastname, age, email, assignature)
VALUES ('Laura', 'Gómez', 35, 'laura.gomez@email.com', 'Física');
```

👉 Aquí no necesitas pasar `instructor_id` porque es **AUTOINCREMENT**.

### 🔹 Insertar en **Courses** (con relación)

Supongamos que `Carlos Pérez` quedó con `instructor_id = 1` y `Laura Gómez = 2` (SQLite los asigna automáticamente):

```sql
INSERT INTO Courses (coursename, description, instructor_id, durationhours)
VALUES ('Álgebra Básica', 'Curso introductorio de álgebra', 1, 40);

INSERT INTO Courses (coursename, description, instructor_id, durationhours)
VALUES ('Mecánica Clásica', 'Fundamentos de la física mecánica', 2, 60);
```

👉 Aquí sí debes indicar el `instructor_id` correcto para que la **FOREIGN KEY** sea válida.

### 🔹 Ver los datos insertados

```sql
SELECT * FROM Instructor;
SELECT * FROM Courses;
```

📌 **Tip:** Si no recuerdas el `instructor_id` al insertar en `Courses`, puedes buscarlo con:

```sql
SELECT instructor_id, firstname, lastname FROM Instructor;
```

### Resumen

####¿Cómo realizar inserciones de datos en bases de datos? 

La inserción de datos en bases de datos relacionales es una habilidad esencial para cualquier desarrollador o analista de datos. Vamos a profundizar en el uso de las declaraciones INSERT INTO y algunas de sus complejidades.

#### ¿Qué son las sentencias INSERT INTO?

Las sentencias `INSERT INTO` se utilizan para agregar nuevas filas a una tabla en una base de datos. Esta operación es básica y forma el núcleo de las operaciones de manipulación de datos (DML). Para utilizar `INSERT INTO`, sigamos estos pasos:

1. Especificar el nombre de la tabla a la cual deseamos añadir información.
2. Detallar los atributos o columnas receptores de estos nuevos datos.
3. Asignar los valores correspondientes a cada atributo.

**Ejemplo de código SQL**

Aquí hay un ejemplo sencillo de cómo luciría una inserción:

```sql
INSERT INTO nombre_tabla (columna1, columna2, columna3)
VALUES (valor1, valor2, valor3);
```

#### ¿Cómo manejar valores por defecto?

A menudo, las tablas tienen columnas configuradas con valores por defecto, como la fecha de carga o la fecha de actualización. Estas no necesitan ser explícitamente especificadas en la sentencia `INSERT INTO`, lo que simplifica el proceso:

```sql
INSERT INTO estudiantes (nombre, apellido, correo)
VALUES ('Carolina', 'Martínez', 'carolina@example.com');
```

#### ¿Cómo trabajar con claves foráneas?

El manejo de claves foráneas es un componente clave en las bases de datos relacionales porque permite vincular tablas diferentes. Al insertar datos que involucren claves foráneas, el contenido debe coincidir con una clave primaria en otra tabla.

En este ejemplo, supongamos que tenemos una tabla de relacionamiento entre estudiantes y cursos:

- Estudiantes tiene un ID que es clave primaria.
- Cursos tiene un ID que es clave primaria.
- La tabla de relacionamiento tiene ambos como claves foráneas.

**Ejemplo de inserción con claves foráneas**

Supongamos que Carolina, cuyo ID de estudiante es 1, va a ser registrada en un curso de francés cuyo ID es también 1:

```sql
INSERT INTO relacion_estudiante_curso (estudiante_id, curso_id, fecha_matricula)
VALUES (1, 1, '2023-10-01');
```

#### ¿Cómo verificar las inserciones?
Después de realizar inserciones, es vital validar que los datos se han registrado correctamente. Esto se puede hacer utilizando una consulta `SELECT`:

`SELECT * FROM relacion_estudiante_curso;`

#### ¿Qué hacer si se cometen errores?

Los errores son parte del aprendizaje. Intenta insertar información incorrecta para entender cómo el motor de base de datos maneja estos errores y qué feedback proporciona. Practica insertando datos erróneos y revisa los mensajes de error para mejorar tu comprensión.

#### Recomendaciones

1. **Practica constantemente**: No hay mejor manera de aprender que practicar. Cree una base de datos de prueba y trabaja con diferentes tipos de inserciones y consultas.
2. **Juega con los datos**: Experimenta con diferentes escenarios y relaciones dentro de tu base de datos.
3. **Explora errores**: Inserta datos inapropiados o en formatos incorrectos para ver cómo tu base de datos maneja los errores.

Avanza con confianza en el mundo de las bases de datos, ampliando tus habilidades y profundizando en cada funcionalidad de las mismas. ¡Sigue practicando y explorando nuevas formas de manipular y consultar tus datos!

## Consultas y Selecciones en SQL (SELECT)

Las **consultas SELECT** en SQL son la base para obtener datos de tus tablas. Te muestro paso a paso con ejemplos claros, suponiendo que ya tienes tu tabla `people`:

```sql
CREATE TABLE people (
  person_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  first_name VARCHAR(255) NOT NULL,
  last_name VARCHAR(255) NOT NULL,
  address VARCHAR(255) NOT NULL,
  city VARCHAR(255) NOT NULL
);
```

Y supongamos que insertamos algunos datos:

```sql
INSERT INTO people (first_name, last_name, address, city)
VALUES 
('Juan', 'Pérez', 'Calle 123', 'Bogotá'),
('Ana', 'Gómez', 'Av 45 #10', 'Medellín'),
('Luis', 'Martínez', 'Cra 7 #45', 'Cali');
```

### 🔹 Consultas básicas con `SELECT`

### 1. Seleccionar **todas las columnas**

```sql
SELECT * FROM people;
```

### 2. Seleccionar **columnas específicas**

```sql
SELECT first_name, city FROM people;
```

### 3. Usar **WHERE** para filtrar

```sql
SELECT * FROM people
WHERE city = 'Bogotá';
```

### 4. Ordenar resultados con `ORDER BY`

```sql
SELECT * FROM people
ORDER BY last_name ASC;   -- ascendente
```

### 5. Limitar cantidad de resultados

```sql
SELECT * FROM people
LIMIT 2;
```

### 6. Usar alias (para renombrar columnas)

```sql
SELECT first_name AS Nombre, last_name AS Apellido
FROM people;
```

### 7. Filtrar con condiciones lógicas

```sql
SELECT * FROM people
WHERE city = 'Cali' OR city = 'Medellín';
```

### 8. Buscar por coincidencia con `LIKE`

```sql
SELECT * FROM people
WHERE first_name LIKE 'A%';  -- nombres que comienzan con A
```

### Resumen

#### Nota al pie

Queremos detallarte la función `SELECT`.
En SQL, el comando `SELECT` se usa para recuperar datos de una base de datos. Puedes especificar las columnas que deseas obtener.
Si realizamos el siguiente comando:
`SELECT * FROM COURSES`
Podemos observar la información de la tabla completa, que habíamos creado en clases anteriores con la tabla `COURSES`.
Si solo queremos seleccionar una columna o columnas en específico, debemos hacer lo siguiente:
Con el comando `SELECT`, mencionamos el nombre de las columnas que queremos traer.
En nuestro ejemplo, podemos llamar a `coursename` y `description`:
SELECT coursename, description (Seleccionamos las columnas que queremos traer)
 FROM COURSES (Elegimos la tabla de donde queremos obtener la información)
Puedes ordenar la información obtenida con tu comando `SELECT` utilizando la función `ORDER`.
En el ejemplo, podemos organizarlo de forma ascendente, descendente o por fecha:
ORDER BY coursename ASC

#### ¿Cómo utilizar la sentencia SELECT * FROM en SQL?

La sentencia `SELECT * FROM` es uno de los comandos más esenciales y comunes que utilizarás en el campo del análisis de datos, ya sea como analista, ingeniero de datos o cualquier profesional en este ámbito. Esta consulta te permite acceder y visualizar de manera inmediata toda la información contenida en una tabla de tu base de datos. Vamos a profundizar en su uso y algunas de sus variaciones.

#### ¿Cómo funciona la sentencia SELECT * FROM?

La funcionalidad básica de la sentencia `SELECT * FROM` implica tres componentes principales:

1. **SELECT**: Una palabra reservada que indica que deseas seleccionar datos de la base de datos.
2. **Asterisco (*)**: Indica que quieres seleccionar todos los campos de la tabla.
3. **FROM**: Designa la tabla de la cual deseas obtener información.

Por ejemplo, si deseas consultar toda la información almacenada en la tabla llamada "cursos", la sentencia será:

`SELECT * FROM cursos;`

Al ejecutarla, verás todos los registros y columnas disponibles en la tabla, incluyendo el nombre del curso, descripción, instructor ID, duración del curso, fecha de carga y fecha de modificación.

#### ¿Cómo especificar campos en la consulta?

A veces no necesitas toda la información de la tabla; solo estás interesado en ciertos atributos. En vez de utilizar el asterisco para seleccionar todos los campos, puedes especificar los nombres de las columnas que deseas consultar. Por ejemplo:

`SELECT nombre_curso, descripcion FROM cursos;`

Este comando mostrará únicamente los campos "nombre_curso" y "descripcion", permitiéndote enfocarte en la información necesaria.

#### ¿Cómo ordenar los resultados de la consulta?

Ordenar los resultados es otra capacidad poderosa. Puedes ordenar la información de manera ascendente o descendente, usando la cláusula `ORDER BY`. Por defecto, el orden es ascendente.

Para ordenar por el nombre del curso, lo harías de la siguiente forma:

`SELECT * FROM cursos ORDER BY nombre_curso ASC;`

O para orden descendente:

`SELECT * FROM cursos ORDER BY nombre_curso DESC;`

#### ¿Qué evitar al escribir consultas SQL?

Al trabajar con SQL, es importante prestar atención a la sensibilidad de las claves. Algunos motores de bases de datos son "keyssensitive", lo que significa que debes escribir los nombres de los atributos exactamente como los registraste en tu base de datos. Los errores de sintaxis, como llamar a un campo de forma incorrecta, son comunes, por lo que debes ser cuidadoso con la escritura.

Si ves un error como un campo no existente, verifica que el nombre esté bien escrito y que coincida con la base de datos. Por ejemplo, si un campo se registró como "fecha_carga", no funcionará si lo escribes de la forma "fecha-carga".

#### ¿Cómo seguir mejorando tus habilidades en SQL?

Te animo a que explores otras tablas en tu base de datos y juegues con los datos que tienes almacenados. Practica agrupando información, utilizando claves adicionales y manipula los datos para obtener una comprensión más profunda y afinada de SQL. Cada error es una oportunidad para aprender y mejorar tus habilidades de consulta y manejo de bases de datos.

La práctica constante te ayudará a convertirte en un experto en el manejo de datos, dándote la confianza necesaria para abordar problemas más complejos y proponer soluciones innovadoras en el mundo de los datos. ¡Continúa tu aventura en el fascinante mundo de SQL!

## Eliminación de Datos con SQL (DELETE)

te explico cómo funciona la eliminación de datos en SQL con **DELETE**.

La sentencia básica es:

```sql
DELETE FROM nombre_tabla
WHERE condición;
```

### 🔹 Ejemplos con tu tabla `people`

1. **Eliminar un registro específico por ID**

```sql
DELETE FROM people
WHERE person_id = 3;
```

👉 Esto borra solo la persona con `person_id = 3`.

2. **Eliminar por coincidencia de nombre**

```sql
DELETE FROM people
WHERE first_name = 'Juan';
```

👉 Elimina a todas las personas llamadas Juan.

3. **Eliminar registros de una ciudad**

```sql
DELETE FROM people
WHERE city = 'Bogotá';
```

👉 Borra todos los que viven en Bogotá.

4. **Eliminar TODOS los registros (vaciar la tabla)**
   ⚠️ Cuidado, esto elimina todo el contenido:

```sql
DELETE FROM people;
```

5. **Eliminar con condición múltiple**

```sql
DELETE FROM people
WHERE city = 'Medellín' AND last_name = 'Pérez';
```

👉 Solo borra a quienes cumplan **ambas condiciones**.

📌 Recomendación:
Antes de borrar, puedes verificar qué filas se afectarían con un **SELECT**. Por ejemplo:

```sql
SELECT * FROM people WHERE city = 'Bogotá';
```

y si estás seguro, ejecutas el `DELETE`.

### Resumen

### ¿Cómo evitar desastres al eliminar datos en SQL?

Trabajar con bases de datos en SQL es una habilidad esencial para cualquier profesional de datos. Sin embargo, es igualmente crucial entender cómo ejecutar las sentencias correctamente para evitar problemas graves, como la eliminación accidental de toda la producción. Aquí exploraremos los errores comunes y mejores prácticas al usar la sentencia DELETE en SQL para evitar desastres.

#### ¿Cuál es la errata más común al utilizar DELETE?

En el mundo de la ingeniería de datos, un error crítico es olvidar la cláusula `WHERE` en un `DELETE` statement. Esto puede provocar la eliminación de todos los registros en una tabla, lo que podría llevar a perder información crucial.

- **Sintaxis del DELETE**: Debe contener la palabra reservada `DELETE` seguida de `FROM`, el nombre de la tabla y, finalmente, un WHERE que especifique las condiciones para eliminar los datos.

`DELETE FROM nombre_tabla WHERE condición;`

- **Importancia del WHERE**: Este es el elemento más importante de la sintaxis. Sin él, eliminas toda la información de tu tabla, arriesgando perder datos valiosos y causando fallos en producción.

#### ¿Cómo poner en práctica DELETE de manera segura?

Antes de ejecutar cualquier `DELETE`, es esencial consultar la información de la tabla con una `SELECT` query para verificar los datos que serán afectados. De este modo, puedes asegurarte de que solo se eliminen los registros correctos.

1. **Consulta previa**: Revisa la información de la tabla que deseas modificar antes de aplicar el `DELETE`.

`SELECT * FROM nombre_tabla WHERE condición;`

2. **Eliminar por clave primaria**: Es recomendable utilizar la clave primaria en la cláusula `WHERE`, ya que es única y reduce el riesgo de afectar más registros de los necesarios.

3. **Verificar después de DELETE**: Consulta de nuevo la tabla para garantizar que se eliminaron los registros adecuados.

```sql
DELETE FROM estudiante WHERE id = 2;
SELECT * FROM estudiante;
```

#### ¿Qué otras sentencias SQL debes conocer?

Además de `DELETE`, hay otras sentencias SQL importantes que debes manejar con precisión para una gestión eficaz de la base de datos.

- **SELECT**: Esta es una de las sentencias más fundamentales, permitiéndote manipular datos, agregar `WHERE`, `GROUP BY`, `ORDER BY`, entre otros.

`SELECT * FROM nombre_tabla WHERE condición;`

- **UPDATE**: Utilizada para modificar datos. Recuerda usar `WHERE` para especificar qué registros deseas actualizar.

`UPDATE nombre_tabla SET columna = valor WHERE condición;`

- **CREATE DATABASE y CREATE TABLE**: Esta sintaxis te permite crear bases de datos y tablas, especificando restricciones como claves primarias y tipos de datos.

```sql
CREATE DATABASE nombre_base_datos;
CREATE TABLE nombre_tabla (
    id INT PRIMARY KEY,
    nombre VARCHAR(255)
);
```

Con este conocimiento, podrás evitar errores críticos en tus proyectos de datos. Ten siempre presente revisar la sintaxis detalladamente y comprender el impacto de tus acciones. Esto no solo garantiza la integridad de los datos, sino que también eleva tu habilidad profesional en el manejo de bases de datos. ¡Sigue aprendiendo y perfeccionando tus capacidades para convertirte en un experto en el mundo de los datos!

## Actualización de Datos con SQL (UPDATE)

El comando **UPDATE** en SQL se usa para **modificar registros existentes** en una tabla.

La sintaxis general es:

```sql
UPDATE nombre_tabla
SET columna1 = valor1, columna2 = valor2, ...
WHERE condición;
```

⚠️ **Muy importante**: si omites el `WHERE`, actualizarás **TODOS** los registros de la tabla.

### Ejemplos con tu tabla **INSTRUCTORS**

#### 1. Cambiar el correo de un instructor específico:

```sql
UPDATE INSTRUCTORS
SET EMAIL = 'nuevo.email@example.com'
WHERE INSTRUCTORID = 1;
```

#### 2. Actualizar la edad de todos los instructores con apellido "Smith":

```sql
UPDATE INSTRUCTORS
SET AGE = AGE + 1
WHERE LASTNAME = 'Smith';
```

#### 3. Actualizar múltiples columnas:

```sql
UPDATE INSTRUCTORS
SET FIRSTNAME = 'Robert',
    EMAIL = 'robert.smith@example.com'
WHERE INSTRUCTORID = 3;
```

#### 4. Actualizar **toda la tabla** (ejemplo: reiniciar correos):

```sql
UPDATE INSTRUCTORS
SET EMAIL = 'pendiente@asignar.com';
```

*(Ojo, esto cambia todos los correos en la tabla).*

### Resumen

#### ¿Cómo gestionar errores en bases de datos?

Imagina que los datos en tu base de datos contienen un error. La buena noticia es que SQL, un lenguaje de consulta estructurado, te permite modificar registros sin la necesidad de reconstruir toda la tabla. Este proceso trae grandes beneficios, como ahorrar tiempo y recursos, además de evitar la pérdida de información valiosa.

#### ¿Qué comando utilizar para actualizar información?

Para actualizar información en una tabla, utilizamos el comando UPDATE. Con UPDATE, puedes cambiar los valores de uno o más campos en las filas existentes de tu tabla. Aquí un ejemplo sencillo en el que vamos a corregir un error tipográfico en una base de datos que almacena información de personas:

```sql
UPDATE personas 
SET nombre = 'Juana' 
WHERE nombre = 'Juna';
```

En este caso, estamos actualizando todas las filas en las que el nombre es "Juna" a "Juana". Asegúrate siempre de que el `WHERE` esté bien definido para no modificar registros que no deseas alterar.

#### ¿Cómo confirmar los cambios realizados?

Después de ejecutar un comando `UPDATE`, es importante confirmar que los cambios se han realizado correctamente. Para ello, puedes utilizar el comando `SELECT` y verificar los resultados:

`SELECT * FROM personas WHERE nombre = 'Juana';`

Esto te mostrará todas las filas en las que el nombre es ahora "Juana". Es una buena práctica comprobar siempre los resultados para garantizar que la actualización se haya implementado como se esperaba.

#### ¿Qué precauciones tomar al modificar datos?

Modificar información en una base de datos es una tarea sensible que conlleva algunas consideraciones:

- **Revisar los datos**: Antes de actualizar, asegúrate de que el dato nuevo es correcto para evitar errores posteriores.
- **Realizar copias de seguridad**: Siempre realiza una copia de seguridad de la base de datos antes de efectuar cambios significativos, en caso de que necesites volver a la versión anterior.
- **Pruebas en entornos seguros**: Cuando sea posible, realiza pruebas en un entorno de desarrollo o pruebas para verificar los cambios antes de aplicarlos en producción.

Un mantenimiento adecuado de la base de datos garantiza la integridad y confiabilidad de los datos, lo cual es crucial para cualquier organización que dependa de la información almacenada para su operación diaria. Mantente siempre actualizado y sigue aprendiendo para mejorar tus habilidades en manejo de bases de datos.

## 🛠️ Guía de instalación de MySQL y MySQL Workbench

Para continuar con las siguientes clases vamos a necesitar la instalación de MySQL y MySQL Workbench.

Instalación de MySQL
Instalación de MySQL en Windows
1. **Descargar el instalador**

- Visita: [https://dev.mysql.com/downloads/installer](https://dev.mysql.com/downloads/installer)

- Descarga MySQL Installer for Windows (puede ser la versión Full o Web).

2. **Ejecutar el instalador**

- Haz clic derecho y selecciona **"Ejecutar como administrador"**.

- Elige la opción **Developer Default** (instala cliente, servidor, Workbench y herramientas adicionales).

- Acepta los términos y espera a que se descarguen todos los componentes.

3. **Configurar el servidor MySQL**

- **Tipo de configuración**: *Standalone MySQL Server*

- **Puerto**: 3306

- **Método de autenticación**: *Use Legacy Authentication Method*

- **Contraseña**: Crea una contraseña para el usuario root

4. **Verificar la instalación**

Abre una terminal (CMD o PowerShell) y ejecuta:

`mysql -u root -p`

Introduce la contraseña. Si accedes correctamente, ¡está funcionando!

### 🍏 Instalación de MySQL en macOS
1. **Instalar Homebrew (si no lo tienes)**

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

2. **Instalar MySQL**

`brew install mysql`

3. **Iniciar el servidor**

`brew services start mysql`

4. **Conectarse al cliente**

`mysql -u root`

Si da error de autenticación, ejecuta:

`mysql_secure_installation`

🐿 **Instalación de MySQL en Ubuntu / Debian**

1. **Actualizar el sistema**

`sudo apt update sudo apt upgrade`

2. **Instalar MySQL Server**

`sudo apt install mysql-server`

3. **Asegurar la instalación**

`sudo mysql_secure_installation`

 - Establece una contraseña para el usuario root

 - Acepta las opciones de seguridad recomendadas

4. **Conectarse al cliente**

`sudo mysql -u root -p`

💻 **Instalación de MySQL Workbench**

#### ¿Qué es?

MySQL Workbench es una interfaz gráfica para trabajar con bases de datos MySQL: puedes visualizar, modelar, escribir queries y administrar usuarios de forma más visual.

🔽 **Descarga**

- Ve a: [https://dev.mysql.com/downloads/workbench/](https://dev.mysql.com/downloads/workbench/)

- Elige tu sistema operativo y descarga el instalador.

🧩 **Instalación**

- **Windows/macOS**: Ejecuta el instalador y sigue los pasos. En Windows, si instalaste con el MySQL Installer, ya deberías tener Workbench incluido.

- **Linux**: En sistemas basados en Debian:

`sudo apt install mysql-workbench`

✅ **Verifica conexión**

- Abre MySQL Workbench.

- Crea una nueva conexión con el usuario root y el puerto 3306.

- Si logras conectarte y ver las bases de datos, ¡todo está listo!

✅ **Probar la conexión**

Una vez dentro del cliente (CLI o Workbench), ejecuta:

`SHOW DATABASES;`

Si ves information_schema, mysql, etc., entonces la instalación fue exitosa. 🎉

## ¿Qué es la cláusula WHERE de SQL?

La **cláusula `WHERE` en SQL** sirve para **filtrar registros** en una consulta o instrucción, de manera que solo se afecten o muestren las filas que cumplen con una condición específica.

### 📌 Uso general:

```sql
SELECT columnas
FROM tabla
WHERE condición;
```

### 🔹 Ejemplos prácticos

1. **Filtrar resultados en un `SELECT`**

```sql
SELECT * 
FROM STUDENTS
WHERE edad > 25;
```

👉 Muestra solo los estudiantes cuya edad sea mayor a 25.

2. **Actualizar registros específicos (`UPDATE`)**

```sql
UPDATE STUDENTS
SET edad = 30
WHERE student_id = 2;
```

👉 Solo actualiza la edad del estudiante con `id = 2`.

3. **Eliminar registros (`DELETE`)**

```sql
DELETE FROM STUDENTS
WHERE edad < 18;
```

👉 Elimina únicamente los estudiantes menores de 18 años.

### ⚠️ Importante:

* Si **no usas `WHERE`** en `UPDATE` o `DELETE`, afectarás **todas las filas** de la tabla.
* Puedes usar operadores como `=`, `<>` (distinto), `<`, `>`, `BETWEEN`, `LIKE`, `IN`, `AND`, `OR`.

### Resumen

#### ¿Por qué utilizar MySQL para análisis de datos?

MySQL se destaca como una de las plataformas más populares en el ámbito del análisis de datos, gracias a su robustez y flexibilidad. A diferencia de SQL Lite, MySQL implementa reglas más estrictas, lo que ayuda a mantener la integridad y calidad de los datos. Al trabajar con MySQL, se evita, por ejemplo, dejar campos como el Primary Key y identificadores nulos, garantizando así bases de datos bien estructuradas y confiables.

Además, MySQL ofrece un entorno de trabajo en consola donde se pueden practicar consultas complejas y manipulaciones de datos, lo que resulta esencial para desarrolladores y analistas de datos. Es altamente recomendable configurar adecuadamente el entorno al instalar MySQL, crear bases de datos y tablas desde cero, y usar herramientas como ChatGPT para generar ejemplos de registros a insertar.

#### ¿Cómo utilizar la sentencia WHERE en MySQL?

La sentencia `WHERE` es una herramienta poderosa y versátil en MySQL, ya que nos permite filtrar datos de forma precisa en nuestras consultas. Se puede emplear para modificar, eliminar o simplemente consultar datos mediante diferentes operadores lógicos y de comparación.

#### ¿Qué operadores lógicos se pueden utilizar?

1. **Operador de igualdad (`=`)**:

- Permite obtener registros que coincidan exactamente con un valor específico. Por ejemplo, para consultar estudiantes con un instructor_id específico.

2. **Operador de desigualdad (`!=` o `<>`)**:

- Filtra los datos que no coinciden con el valor especificado. Excluye resultados que coincidan con criterios determinados y es útil para obtener conjuntos de datos más relevantes.

3. **Operador de comparación**:

Operadores como `<`, `>`, `<=`, y `>=` permiten realizar consultas basadas en rangos numéricos.

#### ¿Cómo se usan los operadores para manipular datos de texto?

Para datos de texto, los operadores comparativos también son útiles. Se pueden utilizar comillas simples para encerrar los valores de texto específicos que queremos filtrar, por ejemplo, filtrar por un nombre de instructor específico o por correo electrónico.

`SELECT * FROM instructores WHERE primer_nombre = 'John';`

#### ¿Qué es la cláusula BETWEEN?

El operador `BETWEEN` es ideal para definir rangos inclusivos entre dos valores, y es especialmente útil para datos numéricos. Es vital indicar primero el menor valor seguido por el mayor al utilizar este operador.

`SELECT * FROM instructores WHERE salario BETWEEN 50000 AND 90000;`

#### ¿Cómo se pueden optimizar las consultas SQL?

Optimizar consultas SQL es crucial para mantener un rendimiento eficiente en bases de datos MySQL, especialmente al manejar grandes volúmenes de datos.

- **Índices**: Implementar índices para columnas usadas frecuentemente en la cláusula `WHERE`, ya que aceleran el acceso a los datos.
- **Consultas específicas**: Evitar el uso de `SELECT *` en favor de especificar solo las columnas necesarias.
- **Limitar resultados**: Si se requieren menos registros,` LIMIT` puede reducir la carga de las consultas.

#### ¿Cómo practicar con estos conceptos en MySQL?

La práctica constante es clave para dominar MySQL. Usa prácticas interactivas, como las consultas de ejemplo aportadas y modificadas con diferentes operadores y datos. Además, cuestiona sobre diferentes escenarios, como modificar datos específicos, lo cual fortalece las habilidades adquiridas.

Finalmente, invita a los demás a compartir experiencias y resultados de las prácticas mediante comentarios en plataformas de aprendizaje o foros, ya que la colaboración y el feedback son sumamente valiosos en el proceso de aprendizaje.

## Filtrar y Ordenar Datos en SQL (LIKE)

En SQL puedes **filtrar y ordenar datos** usando `WHERE`, `LIKE` y `ORDER BY`.

### 🔎 **1. Filtrar con `LIKE`**

La cláusula `LIKE` se usa en el `WHERE` para buscar patrones de texto.
Los comodines principales son:

* `%` → Cualquier número de caracteres.
* `_` → Un solo carácter.

Ejemplos:

```sql
-- Buscar estudiantes cuyo nombre empieza con 'A'
SELECT * 
FROM STUDENTS
WHERE FIRSTNAME LIKE 'A%';

-- Buscar estudiantes cuyo apellido termina en 'z'
SELECT * 
FROM STUDENTS
WHERE LASTNAME LIKE '%z';

-- Buscar estudiantes cuyo correo contiene 'gmail'
SELECT * 
FROM STUDENTS
WHERE EMAIL LIKE '%gmail%';
```

### 📌 **2. Ordenar con `ORDER BY`**

Puedes ordenar los resultados **ascendente (ASC)** o **descendente (DESC)**:

```sql
-- Ordenar estudiantes por edad (menor a mayor)
SELECT * 
FROM STUDENTS
ORDER BY AGE ASC;

-- Ordenar estudiantes por salario de mayor a menor
SELECT * 
FROM STUDENTS
ORDER BY SALARY DESC;
```

### 🎯 **3. Combinar LIKE + ORDER BY**

```sql
-- Buscar estudiantes cuyo nombre empiece con 'M' y ordenar por apellido
SELECT * 
FROM STUDENTS
WHERE FIRSTNAME LIKE 'M%'
ORDER BY LASTNAME ASC;

-- Buscar correos de Gmail y ordenarlos por fecha de carga
SELECT * 
FROM STUDENTS
WHERE EMAIL LIKE '%gmail%'
ORDER BY LOADDATE DESC;
```

### Resumen

#### ¿Cómo filtrar datos usando la cláusula `WHERE` y la palabra reservada `LIKE` en SQL?

En este artículo, exploraremos cómo filtrar datos de manera avanzada utilizando la cláusula `WHERE` junto con la palabra reservada `LIKE` en SQL. Este método te permitirá depurar y limpiar datos con mayor eficiencia y precisión, mejorando el rendimiento de tus consultas. A partir de casos específicos, como encontrar nombres que comienzan o terminan con determinadas letras o que contienen caracteres específicos, aprenderás cómo aplicar estos operadores para obtener exactamente los resultados que necesitas.

#### ¿Cómo seleccionar nombres que comienzan con una letra específica?

Para encontrar nombres que comienzan con una letra particular, por ejemplo, la letra 'C', se usa el operador `LIKE` combinado con `WHERE`. Aquí te mostramos cómo estructurar tu consulta:

```sql
SELECT * FROM estudiantes
WHERE nombre LIKE 'C%';
```

1. `SELECT *` selecciona todas las columnas de la tabla.
2. `FROM` estudiantes indica la tabla de la que se extraen los datos.
3. `WHERE nombre LIKE 'C%'` especifica que buscamos nombres que comiencen con 'C'.

#### ¿Cómo encontrar apellidos que terminan en una letra específica?

Para buscar apellidos que terminan con una letra específica, digamos 'Z', modificamos la posición del porcentaje en nuestra sentencia SQL. Aquí está el ejemplo:

```sql
SELECT * FROM estudiantes
WHERE apellido LIKE '%Z';
```

El símbolo `%` se coloca delante de la 'Z', indicando que buscamos apellidos que finalicen con esta letra.

#### ¿Cómo mostrar únicamente las columnas necesarias en una consulta?

Es fundamental optimizar nuestras consultas al seleccionar solo los datos necesarios, lo cual es crucial en el análisis avanzado o cuando se manejan grandes cantidades de datos, como en procesos de Big Data.

Supongamos que deseas ver solo el primer nombre y el apellido de personas de 20 años. La consulta se vería así:

```sql
SELECT nombre, apellido FROM estudiantes
WHERE edad = 20;
```

Poner solo las columnas necesarias en el `SELECT` garantiza una consulta más eficiente. Aquí, incluso si no estamos mostrando la edad en el resultado, se usa en la cláusula `WHERE` para filtrar las filas.

#### ¿Cómo trabajar con múltiples filtros en una consulta?

A veces se requiere aplicar varios criterios simultáneamente. Imagina que necesitas personas cuyo nombre empiece con 'M', tengan 20 años, y su apellido contenga la letra 'O'. Aquí está cómo hacerlo:

```sql
SELECT nombre, apellido FROM estudiantes
WHERE nombre LIKE 'M%'
AND edad = 20
AND apellido LIKE '%O%';
```

- `LIKE 'M%'` filtra los nombres que comienzan con 'M'.
- `edad = 20` asegura que solo se seleccionen personas de 20 años.
- `apellido LIKE '%O%'` busca apellidos que contengan la letra 'O' en cualquier posición.

#### Consejos para optimizar tus consultas SQL

- **Usar índices**: Asegúrate de que las columnas utilizadas en `WHERE` estén indexadas para mejorar el rendimiento.
- **Seleccionar solo columnas necesarias**: Evita el uso de `SELECT *` para reducir la carga del servidor y el tiempo de respuesta.
- **Practicar con diferentes tablas**: Familiarízate con la diversidad de tus tablas para perfeccionar tus habilidades en SQL.
- **Pruebas continuas**: Realiza y modifica consultas según diferentes escenarios para validar resultados y eficiencia.

Al aplicar estos métodos y consejos, podrás manejar tus consultas SQL con eficacia, llevando tu análisis de datos a un nuevo nivel. Deja que tus resultados hablen por ti al optimizar y refinar cada filtrado de datos. ¡Sigue practicando y explorando nuevas posibilidades para mejorar tu eficiencia en SQL!

## Cláusulas de Comparación Textual en SQL (AND, NULL, IN, NOT)

¡Excelente tema! 🚀
En SQL, además de `LIKE`, se pueden usar **cláusulas de comparación textual** para filtrar resultados de forma más precisa. Te muestro los más usados con ejemplos claros:

### 🔹 **1. `AND`**

Sirve para combinar **dos o más condiciones**:

```sql
-- Estudiantes con nombre 'Carlos' y edad mayor a 20
SELECT * 
FROM STUDENTS
WHERE FIRSTNAME = 'Carlos' AND AGE > 20;
```

### 🔹 **2. `NULL`**

Se usa para verificar si un valor está vacío (**NULL**) o no (**IS NOT NULL**):

```sql
-- Buscar estudiantes que no tienen correo
SELECT * 
FROM STUDENTS
WHERE EMAIL IS NULL;

-- Buscar estudiantes que sí tienen correo
SELECT * 
FROM STUDENTS
WHERE EMAIL IS NOT NULL;
```

⚠️ Ojo: en SQL no se usa `= NULL`, siempre se usa `IS NULL`.

### 🔹 **3. `IN`**

Sirve para comprobar si un valor está dentro de una **lista de valores**:

```sql
-- Estudiantes con edades de 18, 20 o 25
SELECT * 
FROM STUDENTS
WHERE AGE IN (18, 20, 25);

-- Estudiantes con nombre en una lista
SELECT * 
FROM STUDENTS
WHERE FIRSTNAME IN ('Carlos', 'María', 'Ana');
```

### 🔹 **4. `NOT`**

Sirve para **negar condiciones**:

```sql
-- Estudiantes que NO se llaman 'Carolina'
SELECT * 
FROM STUDENTS
WHERE FIRSTNAME NOT LIKE 'Carolina';

-- Estudiantes cuya edad no esté en 18 o 20
SELECT * 
FROM STUDENTS
WHERE AGE NOT IN (18, 20);
```

✅ Ejemplo combinando varias cláusulas:

```sql
-- Estudiantes mayores de 18, con correo Gmail y cuyo apellido NO sea 'Lopez'
SELECT * 
FROM STUDENTS
WHERE AGE > 18 
  AND EMAIL LIKE '%gmail%'
  AND LASTNAME NOT LIKE 'Lopez';
```

### Resumen

#### ¿Cómo utilizar operadores lógicos en análisis de datos?

El uso de operadores lógicos es fundamental en el análisis de datos y constituye una habilidad esencial para un ingeniero de datos. Los operadores permiten establecer criterios específicos en la información que manejamos, ya sea en procedimientos almacenados, vistas o flujos de trabajo. Veamos cómo se aplican estos operadores en una base de datos, utilizando la consola de SQL.

#### ¿Qué es un operador lógico y cómo se utiliza?

Los operadores lógicos nos permiten combinar múltiples criterios en nuestras consultas para obtener resultados precisos. Por ejemplo, al trabajar con una tabla de instructores, podemos aplicar el siguiente criterio: "el salario debe ser mayor a cincuenta mil dólares".

`SELECT * FROM instructores WHERE salario > 50000;`

Este operador simple nos proporcionará una lista de instructores cuyo salario excede los 50,000 dólares.

#### ¿Cómo utilizar el operador AND y el operador OR?

El operador AND nos ayuda a combinar múltiples condiciones que deben cumplirse simultáneamente. Imaginemos que además queremos que el primer nombre del instructor comience con la letra "J":

`SELECT * FROM instructores WHERE salario > 50000 AND nombre LIKE 'J%';`

Como resultado, obtendremos una lista que cumple ambas condiciones.

Por otro lado, el operador **OR** se utiliza para condiciones excluyentes, cumpliendo al menos una de ellas. Si deseamos aplicar esta lógica, la consulta cambiaría a:

`SELECT * FROM instructores WHERE salario > 50000 OR nombre LIKE 'J%';`

En este caso, la lista incluirá instructores que cumplen al menos una de las condiciones establecidas, resultando en un conjunto más grande de datos.

#### ¿Cómo manejar varias condiciones de búsqueda?

La capacidad de mezclar operadores lógicos permite definir aún más nuestras consultas. Añadiendo un criterio adicional, como nombres que comienzan con "D", podríamos tener:

`SELECT * FROM instructores WHERE (salario > 50000 OR nombre LIKE 'J%') OR nombre LIKE 'D%';`

Esto arroja una lista más amplia, incluía aquellos instructores cuyo primer nombre empieza con "D", además de los criterios antes mencionados.

#### ¿Cómo trabajar con valores nulos en SQL?

Los valores nulos son una parte compleja del análisis de datos. Comprender cómo manejarlos correctamente puede optimizar nuestras consultas.

#### ¿Cómo eliminar los datos nulos de los resultados?

Para visualizar registros cuyos nombres no sean nulos, podemos usar la siguiente consulta:

`SELECT * FROM estudiantes WHERE nombre IS NOT NULL;`

Esto mostrará solo los registros donde el campo nombre contiene datos válidos.

#### ¿Y si queremos ver los datos nulos?

Invertir la lógica es sencillo:

`SELECT * FROM estudiantes WHERE nombre IS NULL;`

Con ello, listamos sólo aquellos registros donde el campo nombre no tiene un valor almacenado.

#### ¿Cómo aplicar filtros con NOT IN?

El operador NOT IN permite excluir ciertos valores específicos de nuestros resultados. Por ejemplo, si deseamos excluir estudiantes con una edad determinada:

`SELECT * FROM estudiantes WHERE edad NOT IN (20);`

Esta consulta devolverá información de todos los estudiantes, excepto aquellos que tengan exactamente veinte años.

Practicando este tipo de consultas y dominando el uso de operadores lógicos y filtros, podrás optimizar tus análisis de datos y lograr cumplir de manera eficiente con cualquier requerimiento propuesto. Explora, experimenta y sigue aprendiendo para fortalecer tus habilidades en el apasionante mundo del análisis de datos.

## Funciones de Aritmética Básica en SQL (COUNT)

La **función `COUNT` en SQL** se usa para **contar registros** dentro de una tabla, de acuerdo con un criterio. Es una de las funciones de agregación más usadas.

### 📌 Sintaxis básica:

```sql
SELECT COUNT(*)
FROM nombre_tabla;
```

👉 Explicación:

* `COUNT(*)` → cuenta **todas las filas** de la tabla, incluyendo valores nulos.
* `COUNT(columna)` → cuenta solo las filas donde la columna **NO es nula**.

### ✅ Ejemplos prácticos:

1. **Contar todos los estudiantes en la tabla `STUDENTS`:**

```sql
SELECT COUNT(*) AS total_estudiantes
FROM STUDENTS;
```

2. **Contar estudiantes con correo registrado (ignora `NULL`):**

```sql
SELECT COUNT(EMAIL) AS con_correo
FROM STUDENTS;
```

3. **Contar estudiantes mayores de 25 años:**

```sql
SELECT COUNT(*) AS mayores_25
FROM STUDENTS
WHERE AGE > 25;
```

4. **Contar estudiantes por cada apellido (agrupados con `GROUP BY`):**

```sql
SELECT LASTNAME, COUNT(*) AS cantidad
FROM STUDENTS
GROUP BY LASTNAME;
```

### Resumen

#### Aclaración:

Hola, estudiantes. 
Queríamos detallar un poco sobre la función **COUNT** mencionada en la clase.
Para aplicarla en el primer ejemplo que la profesora relacionó, utilizamos el siguiente comando:
**SELECT** courseid, **COUNT**(studentid)
En este caso, **SELECT** se utiliza para identificar el ID del curso donde se va a realizar el conteo de los estudiantes. Es decir:
**SELECT** courseid
Después de ello, utilizamos la función COUNT, la cual realizará la búsqueda de cuántos estudiantes se encuentran en cada uno de los cursos:
**COUNT**(studentid)

Adicionalmente, hemos dejado el query para la creación de la tabla STUDENT_COURSE y el INSERT que se realiza sobre la misma.

#### ¿Cómo generar informes eficaces con SQL en entornos de Business Intelligence?

El Business Intelligence es esencial para la toma de decisiones empresariales, pues proporciona herramientas y tecnologías que ayudan a transformar datos en información valiosa. SQL, una de las principales herramientas de manipulación de datos, permite la creación de informes detallados. Este contenido te guiará a través de un escenario práctico donde aprendemos a construir informes utilizando consultas SQL. Abordaremos desde la agrupación de estudiantes por curso hasta el cálculo de saldos promedios de instructores.

#### ¿Cómo contar estudiantes por curso?

Imagina que tu jefe necesita saber cuántos estudiantes están inscritos en cada curso. La solución es usar SQL para agrupar la información. Empezaremos usando la sentencia `GROUP BY`, que nos permite clasificar datos según columnas específicas.

```sql
SELECT curso_id, COUNT(estudiante_id) AS total_estudiantes
FROM inscripciones
GROUP BY curso_id;
```

Este ejemplo agrupa estudiantes por curso, contando cuántos hay en cada uno. Es clave especificar el campo de estudiante dentro del `COUNT` para obtener resultados precisos.

#### ¿Cómo filtrar estudiantes con más de dos cursos?

Supongamos que ahora solo quieres mostrar estudiantes con más de dos cursos registrados. Aquí entra `HAVING`, que actúa como un filtro posterior al `GROUP BY`.

```sql
SELECT estudiante_id, COUNT(curso_id) AS total_cursos
FROM inscripciones
GROUP BY estudiante_id
HAVING COUNT(curso_id) > 2;
```

Esta consulta proporciona resultados donde solo los estudiantes con más de dos cursos registrados son mostrados, demostrando la flexibilidad de `HAVING` en SQL para crear filtros avanzados.

#### ¿Cómo calcular salarios de instructores con operaciones aritméticas?

En la gestión administrativa, conocer el salario total de los empleados puede ser crucial. SQL ofrece la función de SUM para sumar valores de una columna, y los `alias` para mejorar la legibilidad del resultado.

```sql
SELECT SUM(salario) AS salario_total
FROM instructores;
```

Este comando suma todos los salarios de la tabla de instructores. Al usar un `alias`, nominalizamos la columna resultante para facilitar su interpretación en los informes.

#### ¿Cómo calcular el promedio de salarios?

Para conocer el promedio salarial de los instructores, utilizamos AVG, que calcula el promedio de una columna de números.

```sql
SELECT AVG(salario) AS salario_promedio
FROM instructores;
```

Al igual que con otras funciones aritméticas, los `alias` ayudan a mantener un estándar uniforme y profesional en la presentación de datos al personal directivo.

#### Buenas prácticas y recomendaciones

Mantener consistencia y limpieza en el nombre de las columnas es fundamental. Evita mezclar idiomas o usar mayúsculas y minúsculas indiscriminadamente al nombrar columnas o utilizar `aliases`. Estas prácticas garantizan que nuestras consultas sean no solo correctas, sino también profesionalmente presentadas.

Practicar y experimentar con diferentes combinaciones es esencial para desarrollar habilidades avanzadas en SQL y BI. Imagina que tu jefe te pide diferentes informes y utiliza estos métodos para resolver problemas reales en un entorno de datos dinámico.

Si tienes alguna duda o deseas explorar otros escenarios, ¡anímate a plantear tus preguntas y comparte tus experiencias!

## Agrupación de Datos en SQL: GROUP BY, HAVING y CASE para Análisis Avanzado

¡Excelente tema 🚀! Vamos a ver cómo funcionan **`GROUP BY`**, **`HAVING`** y **`CASE`** en SQL para hacer análisis de datos más avanzados.

### 🔹 1. `GROUP BY`

Sirve para **agrupar registros** en función de una o varias columnas. Generalmente se combina con funciones de agregación (`COUNT`, `SUM`, `AVG`, `MAX`, `MIN`).

### Ejemplo:

Contar estudiantes por apellido:

```sql
SELECT LASTNAME, COUNT(*) AS cantidad
FROM STUDENTS
GROUP BY LASTNAME;
```

👉 Esto devuelve cuántos estudiantes tienen el mismo apellido.

### 🔹 2. `HAVING`

Se usa para **filtrar resultados después de hacer un `GROUP BY`**.
(`WHERE` no funciona con funciones agregadas, por eso existe `HAVING`).

### Ejemplo:

Mostrar solo apellidos que tengan **más de 1 estudiante**:

```sql
SELECT LASTNAME, COUNT(*) AS cantidad
FROM STUDENTS
GROUP BY LASTNAME
HAVING COUNT(*) > 1;
```

### 🔹 3. `CASE`

Es como un **if-else en SQL**. Permite clasificar o crear categorías dinámicas.

### Ejemplo:

Clasificar estudiantes por rango de edad:

```sql
SELECT 
    FIRSTNAME,
    LASTNAME,
    AGE,
    CASE
        WHEN AGE < 18 THEN 'Menor de Edad'
        WHEN AGE BETWEEN 18 AND 30 THEN 'Joven Adulto'
        WHEN AGE BETWEEN 31 AND 50 THEN 'Adulto'
        ELSE 'Mayor'
    END AS categoria_edad
FROM STUDENTS;
```

### 🔹 4. Ejemplo completo (GROUP BY + HAVING + CASE)

Queremos saber cuántos estudiantes hay en cada **categoría de edad**, pero mostrar solo las categorías que tienen **2 o más estudiantes**:

```sql
SELECT 
    CASE
        WHEN AGE < 18 THEN 'Menor de Edad'
        WHEN AGE BETWEEN 18 AND 30 THEN 'Joven Adulto'
        WHEN AGE BETWEEN 31 AND 50 THEN 'Adulto'
        ELSE 'Mayor'
    END AS categoria_edad,
    COUNT(*) AS total_estudiantes
FROM STUDENTS
GROUP BY categoria_edad
HAVING COUNT(*) >= 2;
```

👉 Con esto puedes hacer análisis avanzados directamente en SQL, sin necesidad de sacar los datos a otra herramienta.

### Resumen

#### ¿Cómo manejar grandes volúmenes de datos en bases de datos multirrelacionales?

El manejo eficiente de los datos es esencial cuando se trabaja con bases de datos complejas que contienen múltiples relaciones. Este proceso, aunque desafiante, se puede optimizar mediante el uso de herramientas y conceptos avanzados como el GROUP BY y el HAVING COUNT. Con estas técnicas, podremos agrupar, filtrar y clasificar los datos con facilidad. Vamos a explorar cómo aplicar estos conceptos y cuál es su efectividad a través de ejemplos prácticos.

#### ¿Qué es el GROUP BY y cómo aplicarlo?

`GROUP BY` es fundamental cuando necesitamos contar registros únicos en una base de datos. Supongamos que tienes una tabla de estudiantes, algunos de los cuales están matriculados en varios cursos. Si solamente quieres saber cuántos estudiantes únicos hay sin contar la duplicación de cursos, `GROUP BY` te permite consultar la cantidad exacta.

Para lograrlo, ejecuta un conteo sobre el ID del estudiante, agrupando así la información:

```sql
SELECT student_id, COUNT(*) 
FROM students_courses
GROUP BY student_id;
```

Esto te mostrará cada estudiante solo una vez junto con la cantidad total de cursos en los que están matriculados.

#### ¿Cómo utilizar el HAVING COUNT para filtrar agrupaciones?

`HAVING COUNT` es una herramienta poderosa para aplicar filtros sobre datos agrupados. Por ejemplo, si tu objetivo es obtener los cursos que tengan más de tres estudiantes, podrías implementar la siguiente consulta:

```sql
SELECT course_id, COUNT(*) 
FROM students_courses
GROUP BY course_id
HAVING COUNT(*) > 3;
```

Este enfoque te ayuda a aplicar condiciones adicionales a tu agrupación inicial, asegurando que los resultados exhibidos correspondan exactamente a tus condiciones deseadas.

#### ¿Cómo clasificar categorías con CASE WHEN?

El uso del `CASE WHEN` es crucial cuando requieres clasificar datos según reglas específicas. Al evaluar un atributo como la edad de diferentes estudiantes y clasificarlos en equipos según su rango, el siguiente código SQL es útil:

```sql
SELECT student_name, 
  CASE 
    WHEN age BETWEEN 18 AND 20 THEN 'Team A' 
    ELSE 'Team B' 
  END as team
FROM students;
```

Este código te permite clasificar automáticamente a los estudiantes en `Team A` si se encuentran dentro del rango de 18 a 20 años y en Team B si no cumplen esta condición.

#### ¿Cómo expandir los criterios de clasificación?

Para escenarios más complejos, donde los criterios combinan atributos distintos, puedes ampliar el `CASE WHEN`:

```sql
SELECT student_name, 
  CASE 
    WHEN age BETWEEN 18 AND 20 THEN 'Team A' 
    WHEN student_name = 'María' THEN 'Team A' 
    ELSE 'Team C' 
  END as team
FROM students;
```

Aquí, los estudiantes llamados María también se clasificarán en `Team A`, mientras que todos los demás irán al `Team C`.

Utilizar entidades como `GROUP BY`, `HAVING COUNT`, y `CASE WHEN` permite manejar grandes datos con mayor precisión. Si te sientes atascado, no olvides que siempre puedes recurrir a herramientas de inteligencia artificial como Platzi Ada para obtener asistencia. ¡Mantén tus habilidades al día y acepta nuevos retos!

## Tipos de JOIN en SQL

Los **`JOIN` en SQL** permiten combinar registros de dos o más tablas según una condición de relación (normalmente con claves primarias y foráneas).

Te muestro los tipos principales con ejemplos:

### 🔹 1. **INNER JOIN**

Devuelve solo las filas que tienen coincidencia en ambas tablas.

### Ejemplo:

```sql
SELECT s.STUDENT_ID, s.FIRSTNAME, e.COURSE
FROM STUDENTS s
INNER JOIN ENROLLMENTS e
    ON s.STUDENT_ID = e.STUDENT_ID;
```

👉 Muestra solo los estudiantes que **sí están matriculados** en algún curso.

### 🔹 2. **LEFT JOIN** (o LEFT OUTER JOIN)

Devuelve **todas las filas de la tabla izquierda**, y solo las coincidencias de la tabla derecha. Si no hay coincidencia, pone `NULL`.

```sql
SELECT s.STUDENT_ID, s.FIRSTNAME, e.COURSE
FROM STUDENTS s
LEFT JOIN ENROLLMENTS e
    ON s.STUDENT_ID = e.STUDENT_ID;
```

👉 Muestra **todos los estudiantes**, incluso los que no tienen cursos.

### 🔹 3. **RIGHT JOIN** (o RIGHT OUTER JOIN)

Es lo contrario del LEFT JOIN: devuelve **todas las filas de la tabla derecha**, y solo las coincidencias de la izquierda.

```sql
SELECT s.STUDENT_ID, s.FIRSTNAME, e.COURSE
FROM STUDENTS s
RIGHT JOIN ENROLLMENTS e
    ON s.STUDENT_ID = e.STUDENT_ID;
```

👉 Muestra **todos los cursos con estudiantes**, e incluso cursos donde nadie se matriculó (`NULL` en datos de estudiante).

### 🔹 4. **FULL OUTER JOIN**

Devuelve todas las filas de ambas tablas, coincidan o no.
Cuando no hay coincidencia, aparecen valores `NULL`.

```sql
SELECT s.STUDENT_ID, s.FIRSTNAME, e.COURSE
FROM STUDENTS s
FULL OUTER JOIN ENROLLMENTS e
    ON s.STUDENT_ID = e.STUDENT_ID;
```

👉 Une los dos mundos: todos los estudiantes y todos los cursos, aunque no tengan relación.

### 🔹 5. **CROSS JOIN**

Devuelve el **producto cartesiano**: todas las combinaciones posibles entre filas de ambas tablas.

```sql
SELECT s.FIRSTNAME, e.COURSE
FROM STUDENTS s
CROSS JOIN ENROLLMENTS e;
```

👉 Si hay 10 estudiantes y 5 cursos, obtendrás **50 combinaciones**.

### 🔹 6. **SELF JOIN**

Es un join de una tabla **consigo misma**.
Útil para jerarquías o relaciones dentro de la misma tabla.

### Ejemplo:

Tabla `EMPLOYEES` con `EMPLOYEE_ID` y `MANAGER_ID`:

```sql
SELECT e1.NAME AS empleado, e2.NAME AS jefe
FROM EMPLOYEES e1
INNER JOIN EMPLOYEES e2
    ON e1.MANAGER_ID = e2.EMPLOYEE_ID;
```

👉 Relaciona cada empleado con su jefe.

📌 Resumen gráfico:

* **INNER JOIN** → Solo coincidencias.
* **LEFT JOIN** → Todos de la izquierda + coincidencias.
* **RIGHT JOIN** → Todos de la derecha + coincidencias.
* **FULL JOIN** → Todos de ambos lados.
* **CROSS JOIN** → Todas las combinaciones.
* **SELF JOIN** → La tabla contra sí misma.

### Resumen

#### ¿Qué son los tipos de "join" en SQL?

En el ámbito del manejo de bases de datos, especialmente al trabajar con SQL, los "join" son esenciales para relacionar y combinar información de diferentes tablas. Estos permiten obtener datos detallados y completos, enriquecer nuestras consultas y hacer un análisis más robusto de los datos. Conocer los diferentes tipos de "join" es crucial para aprovechar al máximo el potencial de SQL. Analicemos cada uno de ellos.

#### ¿Cómo funciona el Inner Join?

El Inner Join es uno de los tipos de "join" más utilizados en SQL. Su propósito es devolver únicamente los registros que tienen coincidencias en ambas tablas involucradas en la consulta. Por ejemplo, si tenemos dos tablas que contienen letras del abecedario, el Inner Join solo mostrará aquellas letras que están presentes en ambas tablas. Si las tablas contienen las letras A, B, C en una y A, C, D en la otra, el Inner Join mostraría A y C, al ser los únicos elementos comunes.

```sql
SELECT * FROM tabla1
INNER JOIN tabla2 ON tabla1.letra = tabla2.letra;
```

#### ¿Qué hace un Left Join?

El Left Join es útil cuando queremos obtener todos los registros de la tabla a la izquierda de nuestra consulta, junto con las coincidencias de la tabla derecha. Si no hay coincidencias, el resultado incluirá NULL para los registros de la tabla derecha que no coinciden. Siguiendo el ejemplo de las letras, al usar un Left Join con la tabla1 a la izquierda y tabla2 a la derecha, obtendremos todas las letras de tabla1 y las coincidencias con tabla2.

```sql
SELECT * FROM tabla1
LEFT JOIN tabla2 ON tabla1.letra = tabla2.letra;
```

#### ¿Cómo opera el Right Join?

El Right Join es casi idéntico al Left Join, pero enfocado en la tabla derecha. Devolverá todos los registros de la tabla derecha, y las coincidencias o NULL donde no existan en la tabla izquierda. Si se tiene la tabla2 a la derecha, el resultado incluirá todas las letras de tabla2, más las coincidencias de tabla1.

```sql
SELECT * FROM tabla1
RIGHT JOIN tabla2 ON tabla1.letra = tabla2.letra;
```

#### ¿Qué es el Full Outer Join?

El Full Outer Join combina los resultados del Left Join y del Right Join. Esto significa que mostrará todos los registros de ambas tablas, duplicando donde existan coincidencias y llenando con NULL donde no haya coincidencias. Esta función es potente para obtener una visión completa de los datos, ya que agrupa toda la información de ambas tablas, sin importar si hay coincidencias o no.

```sql
SELECT * FROM tabla1
FULL OUTER JOIN tabla2 ON tabla1.letra = tabla2.letra;
```

#### ¿Cómo utilizar el Cross Join?

El Cross Join crea un producto cartesiano de las dos tablas involucradas, lo que significa que devuelve cada fila de la primera tabla combinada con cada fila de la segunda tabla. Este "join" no necesita una condición ON, ya que simplemente toma todas las combinaciones posibles de registros.

Imaginemos una tabla de productos y otra de marcas. Al aplicar un Cross Join, cada producto se combina con cada marca, generando todas las combinaciones posibles.

```sql
SELECT productos.nombre, marcas.nombre 
FROM productos
CROSS JOIN marcas;
```

Los "joins" son herramientas poderosas en SQL que permiten relacionar distintos elementos de bases de datos, facilitando la creación de reportes detallados y entendimientos profundos de los datos. Con estos conocimientos, estás preparado para empezar a realizar combinaciones de tablas de manera efectiva y adecuada.

## ¿Cómo funciona INNER JOIN, LEFT JOIN, RIGHT JOIN y FULL JOIN?

Muy buena pregunta 🙌. Te explico de forma clara cómo funcionan los **JOINs** en SQL con un ejemplo sencillo.

### 📌 Supongamos que tenemos dos tablas:

### Tabla **STUDENTS**

| student\_id | name   | age |
| ----------- | ------ | --- |
| 1           | Ana    | 20  |
| 2           | Carlos | 22  |
| 3           | Juan   | 23  |

### Tabla **COURSES**

| course\_id | student\_id | course\_name |
| ---------- | ----------- | ------------ |
| 101        | 1           | Matemáticas  |
| 102        | 2           | Historia     |
| 103        | 4           | Biología     |

### 🔹 1. **INNER JOIN**

Muestra solo las filas que tienen coincidencias en **ambas tablas**.

```sql
SELECT s.name, c.course_name
FROM STUDENTS s
INNER JOIN COURSES c ON s.student_id = c.student_id;
```

📊 Resultado:

| name   | course\_name |
| ------ | ------------ |
| Ana    | Matemáticas  |
| Carlos | Historia     |

➡️ Juan no aparece porque no tiene curso, y Biología no aparece porque el `student_id=4` no está en STUDENTS.

### 🔹 2. **LEFT JOIN**

Muestra **todos los registros de la tabla izquierda (STUDENTS)** y solo los que coinciden de la derecha (COURSES).

```sql
SELECT s.name, c.course_name
FROM STUDENTS s
LEFT JOIN COURSES c ON s.student_id = c.student_id;
```

📊 Resultado:

| name   | course\_name |
| ------ | ------------ |
| Ana    | Matemáticas  |
| Carlos | Historia     |
| Juan   | NULL         |

➡️ Juan aparece aunque no tenga curso (con `NULL`).

### 🔹 3. **RIGHT JOIN**

Muestra **todos los registros de la tabla derecha (COURSES)** y solo los que coinciden de la izquierda (STUDENTS).

```sql
SELECT s.name, c.course_name
FROM STUDENTS s
RIGHT JOIN COURSES c ON s.student_id = c.student_id;
```

📊 Resultado:

| name   | course\_name |
| ------ | ------------ |
| Ana    | Matemáticas  |
| Carlos | Historia     |
| NULL   | Biología     |

➡️ Biología aparece aunque no tenga estudiante (con `NULL`).

### 🔹 4. **FULL JOIN** (o FULL OUTER JOIN)

Muestra **todos los registros de ambas tablas**, coincidan o no.
(No todos los motores SQL lo soportan directamente; en MySQL se simula con `UNION` de LEFT y RIGHT).

```sql
SELECT s.name, c.course_name
FROM STUDENTS s
FULL OUTER JOIN COURSES c ON s.student_id = c.student_id;
```

📊 Resultado:

| name   | course\_name |
| ------ | ------------ |
| Ana    | Matemáticas  |
| Carlos | Historia     |
| Juan   | NULL         |
| NULL   | Biología     |

➡️ Combina todo, con `NULL` en donde no hay coincidencia.

📌 **Resumen gráfico:**

* **INNER JOIN** → Solo intersección.
* **LEFT JOIN** → Todo de la izquierda + coincidencias.
* **RIGHT JOIN** → Todo de la derecha + coincidencias.
* **FULL JOIN** → Todo de ambas tablas.

### Resumen

#### ¿Cómo implementar lo aprendido sobre joins en consola?

El trabajo con bases de datos es esencial para cualquier profesional en el campo de la tecnología y la ciencia de datos. Utilizar los joins de manera eficiente puede ayudarte a conectar múltiples tablas rápidamente. Aquí aprenderás cómo replantear toda una base de datos y practicar la creación de tablas y la inserción de datos desde la consola.

#### ¿Cómo recrear la base de datos?

1. **Crear la base de datos**: Es necesario definir el esquema y estructura de la base de datos, incluyendo las tablas, sus columnas y los tipos de datos adecuados para cada columna.
2. **Identificar claves primarias**: Las claves primarias son esenciales para identificar de manera única cada registro dentro de una tabla.
3. **Registrar claves foráneas**: Permiten relacionar tablas y asegurar la integridad de los datos.

#### Ejemplo de Código SQL:

```sql
CREATE TABLE productos (
    id INT PRIMARY KEY,
    nombre VARCHAR(255),
    marca VARCHAR(255),
    precio DECIMAL(10, 2)
);

CREATE TABLE marcas (
    id INT PRIMARY KEY,
    nombre VARCHAR(255)
);

-- Insertar datos
INSERT INTO productos (id, nombre, marca, precio) VALUES (1, 'Producto1', 'MarcaA', 100.00);
INSERT INTO marcas (id, nombre) VALUES (1, 'MarcaA');
```

#### ¿Cómo ejecutar joins en consola?

La sentencia básica `SELECT * FROM` se utiliza para consultar todas las columnas de una tabla. Los Joins permiten combinar registros de dos o más tablas.

#### Ejemplo de Inner Join:

```sql
SELECT p.*, m.nombre 
FROM productos p
INNER JOIN marcas m ON p.marca = m.id;
```

- El INNER JOIN conecta las dos tablas usando las claves primarias y foráneas establecidas.

#### ¿Cómo realizar un join de tipo Right, Left o Full Outer?

La única diferencia entre los tipos de joins radica en la palabra clave utilizada en la sentencia SQL.

##### Cambios en el Join:

1. **Right Join**: Muestra todos los registros de la tabla a la derecha y los registros coincidentes de la tabla a la izquierda.

```sql
SELECT p.*, m.nombre 
FROM productos p
RIGHT JOIN marcas m ON p.marca = m.id;
```

2. **Left Join**: Proporciona todos los registros de la tabla a la izquierda y los coincidentes de la tabla a la derecha.

```sql
SELECT p.*, m.nombre 
FROM productos p
LEFT JOIN marcas m ON p.marca = m.id;
```

3. **Full Outer Join**: Combina el Right Join y el Left Join.

```sql
SELECT p.*, m.nombre 
FROM productos p
FULL OUTER JOIN marcas m ON p.marca = m.id;
```

#### ¿Por qué es importante el nombramiento adecuado de tablas en joins?

El uso de alias para tablas puede simplificar las consultas. Sin embargo, para procedimientos más complejos, el uso de nombres cortos o una sola letra como alias puede ser confuso. Es recomendable usar nombres significativos que denoten con claridad la conexión que se está realizando entre las tablas. Esto facilita la comprensión, el mantenimiento y el soporte del código.

- **Buenas prácticas**:
 - Usa nombres intuitivos y descriptivos para alias de tablas.
 - Evita nombres genéricos que no den contexto.
 - Asegúrate de que el nombramiento sea coherente en todo tu código.

Recuerda continuar practicando y refinando tus habilidades con bases de datos. ¡La experiencia y la dedicación son clave para dominar esta área y enfrentar procesos de desarrollo de software con éxito!

## Vistas Materializadas en SQL: Como optimizar tus consultas y reportes.

¡Tema clave para acelerar reportes! 🚀
Una **vista materializada (MV)** guarda en disco el **resultado** de una consulta (joins, agregados, filtros). Así tus reportes “pesados” se leen como si fueran una tabla, evitando recalcular cada vez.

### ¿Cuándo usarlas?

* Dashboards y reportes con **joins/aggregaciones costosas**.
* Consultas que leen mucho y escriben poco.
* Datos que pueden estar **ligeramente desactualizados** (consistencia eventual).

### Ventajas y costos

**+** Respuestas mucho más rápidas.
**+** Menos carga en tablas base.
**–** Ocupan almacenamiento.
**–** Hay que **refrescarlas** (pueden estar “stale”).
**–** Mantenimiento/índices extra.

### Patrones de uso (con ejemplos)

### PostgreSQL

```sql
-- 1) Crear MV
CREATE MATERIALIZED VIEW mv_sales_daily AS
SELECT
  date_trunc('day', s.created_at) AS day,
  p.category_id,
  COUNT(*) AS orders,
  SUM(s.amount) AS revenue
FROM sales s
JOIN products p ON p.id = s.product_id
GROUP BY 1, 2;

-- 2) Indexar para acelerar lecturas
CREATE INDEX ON mv_sales_daily (day, category_id);

-- 3) Refrescar (bloquea lecturas durante la reconstrucción)
REFRESH MATERIALIZED VIEW mv_sales_daily;

-- 4) Refrescar sin bloquear lectores (usa tabla intermedia)
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sales_daily;
```

> Nota: PostgreSQL **no** hace “fast refresh” nativo (incremental). Cada `REFRESH` recalcula completo; `CONCURRENTLY` evita bloquear consultas lectoras. (Si necesitas incrementales, existen extensiones como *pg\_ivm*).

**Cuándo refrescar**

* Programado (p. ej., cada hora/noche con `cron`/`pg_cron`).
* Bajo demanda (tras un ETL).
* Mixto: rápido (parcial) vía tabla de staging → `REFRESH` inmediatamente después del load.

### Oracle

```sql
-- Requiere logs para FAST REFRESH
CREATE MATERIALIZED VIEW LOG ON SALES WITH ROWID, SEQUENCE
  INCLUDING NEW VALUES;

CREATE MATERIALIZED VIEW mv_sales_daily
BUILD IMMEDIATE
REFRESH FAST ON COMMIT
AS
SELECT trunc(created_at) AS day, product_id,
       COUNT(*) orders, SUM(amount) revenue
FROM sales
GROUP BY trunc(created_at), product_id;
```

* **REFRESH FAST**: incremental, usa *materialized view logs*.
* **ON COMMIT**: se refresca al confirmar transacción.
* Alternativas: `ON DEMAND`, `REFRESH COMPLETE` (recalcula todo).

### SQL Server (equivalente: *Indexed Views*)

SQL Server no tiene “materialized views” por nombre, pero las **vistas indexadas** guardan físicamente el resultado.

```sql
-- Reglas: determinismo, SCHEMABINDING, COUNT_BIG(*), etc.
CREATE VIEW dbo.vwSalesDaily
WITH SCHEMABINDING
AS
SELECT
  CONVERT(date, created_at) AS day,
  product_id,
  COUNT_BIG(*) AS orders,
  SUM(amount) AS revenue
FROM dbo.sales
GROUP BY CONVERT(date, created_at), product_id;

-- Materializa creando un índice único
CREATE UNIQUE CLUSTERED INDEX IX_vwSalesDaily
  ON dbo.vwSalesDaily(day, product_id);
```

> Se mantiene automáticamente (incremental), a costa de **más costo de escritura** en las tablas base.

### MySQL/MariaDB

No traen MV nativas. Patrón común:

1. Crear tabla destino `mv_sales_daily`.
2. Poblarla con `INSERT ... SELECT` o `CREATE TABLE ... AS SELECT`.
3. Refrescarla con **EVENTOS** (MySQL) o un orquestador (Airflow/cron).

```sql
CREATE TABLE mv_sales_daily AS
SELECT ... -- tu consulta pesada

CREATE EVENT ev_refresh_mv_sales_daily
ON SCHEDULE EVERY 1 HOUR
DO
  REPLACE INTO mv_sales_daily
  SELECT ... ;
```

### Buenas prácticas para optimizar

1. **Diseña pensando en la consulta**

   * Deja solo columnas necesarias.
   * Pre-agrega al nivel que lea tu dashboard (día/semana/mes).

2. **Índices en la MV**

   * Indexa las columnas que usas en `WHERE`, `JOIN` y `ORDER BY`.
   * Ej.: `(day, category_id)` para filtros por fecha/categoría.

3. **Estrategia de refresh**

   * Define SLA de frescura: cada 5 min, hora, noche.
   * **PostgreSQL**: usa `CONCURRENTLY` para evitar bloquear lectores.
   * **Oracle/SQL Server**: aprovecha *fast refresh* / vistas indexadas si los writes lo permiten.

4. **Particiona por tiempo (si el motor lo permite)**

   * Facilita refresh por ventana (solo el último día/mes).

5. **Monitorea**

   * Tamaño, tiempos de refresh, latencia de frescura.
   * Métricas del plan de ejecución de las lecturas y del refresh.

6. **Control de staleness**

   * Guarda `last_refreshed_at` (otra tabla o comentario) y muéstralo en el dashboard.

### Checklist rápido

* [ ] ¿Qué latencia de datos aceptas (minutos/horas)?
* [ ] ¿Qué columnas/agrupaciones necesita el dashboard?
* [ ] ¿Hay índices adecuados en la MV?
* [ ] ¿Cómo y cuándo se refresca? ¿Bloquea lectores?
* [ ] ¿Cuánta carga extra añade el mantenimiento a las escrituras?

### Resumen

#### ¿Por qué son importantes las vistas en la transformación de datos?

Al abordar el análisis de datos, un paso crucial es la transformación, donde típicamente empleamos diversos recursos. Las vistas temporales, las tablas temporales, las vistas materializadas y los procedimientos almacenados son solo algunos de estos recursos esenciales. Estos recursos facilitan la transformación de datos como parte del proceso ETL (Extracción, Transformación y Carga). Además, las vistas nos ofrecen una ventaja significativa: limitan la cantidad de datos mostrada al usuario final, mejorando la eficiencia y rendimiento de nuestras consultas.

#### ¿Cómo se crea una vista?

Crear una vista dentro de la sintaxis de consola es sencillo pero requiere atención a ciertos detalles. Utilizamos la palabra reservada `CREATE VIEW`, seguida del nombre deseado para la vista. Posteriormente, empleamos el término `AS` para especificar la consulta que configurará la vista.

```sql
CREATE VIEW vista_estudiantes AS
SELECT *
FROM estudiantes;
```

Este ejemplo básico selecciona todos los campos de la tabla "estudiantes". Aunque las vistas son conceptos distintos a las tablas, el resultado se encuentra en una sección llamada "vistas" al actualizarlas.

### ¿Cómo se optimiza una vista para mostrar solo datos necesarios?

Normalmente, el propósito principal de una vista no es mostrar todos los campos de una tabla, sino limitar los datos mostrados sólo a lo relevante para el usuario.

Por razones de seguridad, muchas veces es necesario restringir la información accesible para ciertos usuarios. Podemos lograrlo llamando específicamente los campos deseados. Por ejemplo, si queremos mostrar solo la clave primaria, el primer nombre y el apellido de las personas, se podría modificar así:

```sql
CREATE OR REPLACE VIEW vista_estudiantes_ligera AS
SELECT clave_primaria, primer_nombre, apellido
FROM estudiantes;
```

Podemos recrear o modificar una vista usando `CREATE OR REPLACE`, que no requiere la palabra reservada `UPDATE` ya que está destinada solo a la manipulación de datos en tablas.

### ¿Cómo se elimina una vista?

Si una vista ya no es necesaria o fue creada por error, eliminarla es simple usando `DROP VIEW`. Este comando elimina la vista de la base de datos de forma rápida y efectiva.

`DROP VIEW vista_estudiantes_ligera;`

El comando `DROP` es el mismo utilizado para la eliminación de tablas, añadiendo la palabra reservada `VIEW` para indicarle que aplicaremos la acción sobre una vista.

#### ¿Cómo avanzar en la práctica de la creación de vistas?

Te invito a crear una vista usando otra tabla de tu base de datos y realizar una consulta más elaborada. Intenta aplicar lo que has aprendido, como los JOIN y condicionales. Comparte tu experiencia y tus consultas recreadas en los comentarios, ¡y hazme saber si tienes preguntas! La práctica es la clave para el dominio de la transformación de datos y cada intento te acerca más a la maestría en el manejo de bases de datos.

## Vistas Materializadas y Temporales en SQL

Perfecto 🚀, vamos a ver **Vistas Materializadas y Vistas Temporales en SQL**, qué son, sus diferencias y cómo se usan:

### 🔹 1. Vistas Temporales (o simplemente *Vistas*)

* Una **vista** es una consulta almacenada con un nombre, como si fuera una "tabla virtual".
* No guarda los datos, solo la instrucción SQL que se ejecuta cada vez que se consulta.
* Son útiles para simplificar consultas complejas y mejorar la seguridad (mostrando solo ciertos campos).

📌 **Ejemplo de Vista:**

```sql
CREATE VIEW VistaEstudiantesMayores AS
SELECT FIRSTNAME, LASTNAME, AGE
FROM STUDENTS
WHERE AGE >= 30;
```

📌 **Consulta desde la vista:**

```sql
SELECT * FROM VistaEstudiantesMayores;
```

⚠️ Características:

* Siempre muestran los **datos actualizados** de las tablas originales.
* No consumen almacenamiento extra (excepto metadatos).
* Son más lentas que las materializadas en consultas muy grandes, porque siempre recalculan.

### 🔹 2. Vistas Materializadas

* Son como una vista, pero **guardan físicamente el resultado de la consulta** en disco.
* Se comportan casi como una tabla, por eso permiten consultas más rápidas en reportes o análisis.
* Se deben **refrescar** (actualizar) cuando cambian los datos de las tablas base.

📌 **Ejemplo en Oracle o PostgreSQL:**

```sql
CREATE MATERIALIZED VIEW MV_EstudiantesResumen AS
SELECT COURSE_ID, COUNT(*) AS Total_Estudiantes
FROM STUDENT_COURSE
GROUP BY COURSE_ID;
```

📌 **Actualizar la vista materializada:**

```sql
REFRESH MATERIALIZED VIEW MV_EstudiantesResumen;
```

⚠️ Características:

* Ocupan **espacio en disco** porque almacenan datos.
* Son rápidas en consultas repetitivas (reportes, dashboards).
* Requieren estrategia de **refresco** (manual o automático).

### 🔹 Diferencias Clave

| Característica | Vista                             | Vista Materializada                 |
| -------------- | --------------------------------- | ----------------------------------- |
| Almacenamiento | No guarda datos, solo la consulta | Guarda datos físicamente            |
| Rendimiento    | Más lenta en datos grandes        | Más rápida en consultas repetitivas |
| Actualización  | Siempre en tiempo real            | Necesita `REFRESH`                  |
| Uso típico     | Simplificar queries, seguridad    | Reportes, BI, Data Warehousing      |

👉 En **MySQL** no existen *vistas materializadas nativas*, pero se pueden simular con **tablas temporales o triggers**.
👉 En **Oracle, PostgreSQL y SQL Server (Indexed Views)** sí existen oficialmente.

### Resumen

#### ¿Cómo optimizar los procesos de ETL mediante vistas temporales y materializadas?

Los procesos de ETL (Extracción, Transformación y Carga) son fundamentales en el ámbito del desarrollo de ingeniería de datos, ya que usualmente requieren un tiempo considerable para su ejecución. Un ingeniero de datos puede enfrentar la necesidad de consultar repetidamente la misma información, y en estas circunstancias, las vistas temporales y materializadas son herramientas útiles para optimizar flujos de trabajo y mejorar el rendimiento de las consultas. Hoy, profundizaremos en estos conceptos y te brindaremos el conocimiento necesario para implementarlos efectivamente en tus proyectos.

#### ¿Qué son las vistas temporales y cómo se utilizan?

Las vistas temporales son una solución sencilla y eficaz para escenarios donde se necesita consultar la misma información repetidamente durante una sesión de trabajo. A diferencia de las vistas permanentes, que almacenan los datos de manera indefinida, las vistas temporales existen solo mientras la sesión está activa. Al cerrar la sesión, estas vistas se eliminan automáticamente del sistema.

Para crear una vista temporal, se utiliza el siguiente comando SQL:

```sql
CREATE TEMPORARY VIEW nombre_vista AS
SELECT ...
```

Donde:

- `CREATE TEMPORARY VIEW`: indica que la vista que se creará será temporal.
- nombre_vista: es el nombre que deseas asignar a tu vista.
- `SELECT ...`: es la consulta que define qué datos se almacenarán en la vista.

Este enfoque es ideal cuando no necesitas conservar los datos permanentemente, aunque requieras consultarlos reiteradas veces en el mismo contexto operativo. Se recomienda para operaciones o análisis que son significativos solo por un tiempo limitado.

### ¿Por qué considerar las vistas materializadas?

En situaciones donde las consultas son extensas, complejas o el soporte de estas es demandante debido a la dificultad de sus operaciones, las vistas materializadas emergen como una herramienta potente. Estas vistas almacenan físicamente los resultados de la consulta en una base de datos, permitiendo un acceso más rápido a los datos procesados, reduciendo así el tiempo de procesamiento en futuras consultas.

Para crear una vista materializada, puedes utilizar el siguiente comando:

```sql
CREATE MATERIALIZED VIEW nombre_mat_vista AS
SELECT ...
```

Este comando es similar al de las vistas permanentes, pero la diferencia clave yace en el almacenamiento físico de los resultados, lo que no solo optimiza el rendimiento, sino también facilita la eficacia y capacidad de soporte técnico para operaciones futuras.

### Ventajas de las vistas materializadas en el soporte técnico

- **Reducción en el tiempo de ejecución**: Al almacenar los resultados ya calculados de consultas complejas, las vistas materializadas liberan recursos al evitar el recalculo en cada ejecución.
- **Mejora en la eficiencia del soporte**: Proporcionan una capa preprocesada que simplifica la solución de problemas y la implementación de cambios requeridos.
- **Optimización de procesos**: Especialmente útil en escenarios donde las transformaciones matemáticas o reglas de negocios exigen procesamiento inmediato y repetido.

#### Consejos finales para implementar y experimentar con vistas
Al realizar experimentos con vistas temporales y materializadas, te sugerimos seguir estos pasos:

- Prueba diferentes comandos de creación de vistas en el motor de base de datos que estés utilizando.
- Detecta y resuelve cualquier incompatibilidad de palabras reservadas específicas de tu sistema. Cada motor SQL (como PostgreSQL, MySQL, etc.) puede tener variaciones en el uso de palabras clave.
- Comparte hallazgos y desafíos con tus colegas para enriquecerte mutuamente con diferentes experiencias y conocimientos.

¡Sigue explorando y mejorando tus habilidades en ingeniería de datos! Implementar las herramientas correctas no solo aumentará tu eficiencia sino que también te posicionará como un profesional experto en la gestión avanzada de datos.

## Expresiones de Tablas Comunes (CTE) en SQL

Las **CTE (Common Table Expressions)** o **Expresiones de Tabla Común** en SQL son una forma de definir consultas temporales que solo existen durante la ejecución de una sentencia.

### 🔹 ¿Qué es una CTE?

* Es como una “vista temporal” que se define **dentro de una consulta**.
* Solo existe durante esa consulta (a diferencia de una vista normal que queda almacenada).
* Se escribe con la cláusula `WITH`.

### 🔸 Sintaxis general

```sql
WITH nombre_cte AS (
    SELECT ...
    FROM ...
    WHERE ...
)
SELECT *
FROM nombre_cte
WHERE ...;
```

### 🔹 Ejemplo 1: Simplificar consultas

Supongamos que queremos la lista de estudiantes mayores de 25 años y luego filtrar solo los que están en ingeniería:

```sql
WITH EstudiantesMayores AS (
    SELECT student_id, firstname, lastname, age, career
    FROM students
    WHERE age > 25
)
SELECT *
FROM EstudiantesMayores
WHERE career = 'Ingeniería';
```

✅ Aquí la CTE `EstudiantesMayores` evita repetir la subconsulta.

### 🔹 Ejemplo 2: Usar varias CTEs en cadena

Podemos definir varias y reutilizarlas en la consulta final:

```sql
WITH VentasPorCliente AS (
    SELECT cliente_id, SUM(total) AS total_compras
    FROM ventas
    GROUP BY cliente_id
),
ClientesVIP AS (
    SELECT cliente_id
    FROM VentasPorCliente
    WHERE total_compras > 10000
)
SELECT c.nombre, v.total_compras
FROM ClientesVIP vip
JOIN VentasPorCliente v ON vip.cliente_id = v.cliente_id
JOIN clientes c ON c.id = v.cliente_id;
```

### 🔹 Ejemplo 3: CTE Recursiva

Una de las grandes ventajas es que las CTE pueden ser **recursivas**, muy útiles para recorrer jerarquías (ej. empleados y jefes, categorías de productos, etc.).

📌 Encontrar toda la jerarquía de un empleado:

```sql
WITH RECURSIVE Jerarquia AS (
    -- Caso base
    SELECT empleado_id, nombre, jefe_id
    FROM empleados
    WHERE empleado_id = 1   -- jefe inicial

    UNION ALL

    -- Caso recursivo
    SELECT e.empleado_id, e.nombre, e.jefe_id
    FROM empleados e
    INNER JOIN Jerarquia j ON e.jefe_id = j.empleado_id
)
SELECT * FROM Jerarquia;
```

👉 Esto devuelve al empleado inicial y todos sus subordinados en todos los niveles.

### 🔹 Diferencias entre CTE y Vista

| Aspecto       | CTE                             | Vista                                          |
| ------------- | ------------------------------- | ---------------------------------------------- |
| Persistencia  | Solo existe durante la consulta | Persistente (queda creada en DB)               |
| Reutilización | Solo en la query actual         | Reutilizable en varias consultas               |
| Rendimiento   | Similar a subconsulta           | Puede mejorar con índices                      |
| Ideal para    | Consultas complejas, recursivas | Reportes, seguridad, simplificación permanente |

📌 **CTE se soporta en**: SQL Server, PostgreSQL, Oracle, MySQL (desde 8.0).

### Resumen

#### ¿Qué es el proceso de transformación de datos?

En el mundo de la manipulación de datos, las siglas ETR (extracción, transformación y carga) y ELT (extracción, carga y transformación) son fundamentales. Dentro de estos procesos, la transformación de datos juega un papel crucial. Se trata del paso donde limpiamos, depuramos y aplicamos reglas de negocio a los datos, generando indicadores y estadísticas conforme a las necesidades de la información. Sin embargo, este proceso puede complicarse ya que nuestras consultas se vuelven avanzadas, abarcando múltiples tablas y utilizando complejas funciones matemáticas y algorítmicas.

#### ¿Qué son las CTE en SQL?

Las CTE (Common Table Expressions) en SQL son una herramienta valiosa para mejorar la legibilidad y mantenimiento del código, facilitando el desarrollo y entendimiento del negocio. Las CTE permiten dividir una consulta de alto nivel en subconsultas más fáciles de comprender. Estas subconsultas luego se pueden relacionar en una consulta principal.

#### ¿Cómo se estructura una CTE?

Crear una CTE en SQL es sencillo. La sintaxis básica se inicia con la palabra reservada WITH, seguida de un nombre claro que refiera al proceso de negocio que representa la consulta. Este nombre es fundamental para la claridad y funcionalidad del código.

```sql
WITH nombre_de_la_cte AS (
    -- Aquí se coloca la subconsulta
)
SELECT * FROM nombre_de_la_cte;
```

#### Buenas prácticas al nombrar una CTE

- **Claridad**: El nombre debe hacer clara referencia al proceso de negocio que aborda.
- **Funcionalidad**: Ayuda a otros desarrolladores a entender la finalidad de la CTE.

#### ¿Cómo se usan las CTE con múltiples subconsultas?

Las CTE pueden contener una o varias subconsultas. En el caso de múltiples subconsultas, pueden relacionarse entre sí mediante `JOIN`, `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN` y otras herramientas que hemos aprendido en SQL.

#### Ejemplo práctico de CTE

Imaginemos que queremos calcular el salario promedio de los empleados y aplicar un condicionante a los resultados. Aquí, la CTE nos permitiría:

1. Calcular el salario promedio en una subconsulta.

2. Usar la consulta principal para filtrar estos resultados según criterios específicos, por ejemplo, aquellos cuyos salarios superen un valor determinado.

```sql
WITH promedio_salarios AS ( SELECT departamento_id, AVG(salario) AS promedio_salario FROM empleados GROUP BY departamento_id ) SELECT * FROM promedio_salarios WHERE promedio_salario > 65000;
```

#### Retos y consideraciones

Al usar CTE, es importante recordar que:

- Puedes referenciar múltiples subconsultas dentro de la consulta principal.
- Cada subconsulta puede depender de otras creando relaciones complejas.
- Misturas de distintas operaciones SQL como agrupamientos y joins enriquecen las consultas.

#### ¿Cómo enriquecer nuestras consultas usando CTE?
Además de simplificar nuestro código, el uso de CTE fomenta la experimentación con nuevas consultas combinando diversas subconsultas. Se invita a la comunidad de desarrolladores a interactuar con estas herramientas, creando consultas avanzadas y compartiendo sus soluciones para el enriquecimiento mutuo.

Las CTE representan un método eficiente y eficaz para trabajar con datos complejos en SQL, mejorando la claridad, el mantenimiento y el proceso de desarrollo en general. ¡Anímate a poner en práctica tus conocimientos y crea consultas innovadoras!

## Procedimientos Almacenados en SQL

Perfecto 🚀, vamos con los **Procedimientos Almacenados en SQL (Stored Procedures)**.

### 🔹 ¿Qué es un Procedimiento Almacenado?

Un **Procedimiento Almacenado** es un bloque de código SQL que se guarda en la base de datos y puede ejecutarse varias veces, como una función.

Sirven para:

* Reutilizar lógica de negocio en la base de datos.
* Centralizar validaciones o procesos.
* Mejorar seguridad (se da acceso al procedimiento, no a la tabla).
* Reducir tráfico entre cliente y servidor (la lógica corre en el servidor).

### 🔸 Sintaxis general

La sintaxis puede variar un poco según el motor de base de datos (**SQL Server, MySQL, PostgreSQL, Oracle**).

### ✅ SQL Server

```sql
CREATE PROCEDURE NombreProcedimiento
    @Parametro1 INT,
    @Parametro2 VARCHAR(50)
AS
BEGIN
    SELECT * 
    FROM empleados
    WHERE id = @Parametro1;
END;
```

Ejecutar:

```sql
EXEC NombreProcedimiento 5, 'valor';
```

---

### ✅ MySQL / MariaDB

```sql
DELIMITER //

CREATE PROCEDURE NombreProcedimiento (
    IN p_id INT,
    IN p_nombre VARCHAR(50)
)
BEGIN
    SELECT * 
    FROM empleados
    WHERE id = p_id AND nombre = p_nombre;
END //

DELIMITER ;
```

Ejecutar:

```sql
CALL NombreProcedimiento(5, 'Juan');
```

### ✅ PostgreSQL

```sql
CREATE OR REPLACE PROCEDURE NombreProcedimiento(p_id INT, p_nombre VARCHAR)
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE NOTICE 'ID: %, Nombre: %', p_id, p_nombre;
END;
$$;
```

Ejecutar:

```sql
CALL NombreProcedimiento(5, 'Juan');
```

#### 🔸 Tipos de Parámetros

Los procedimientos pueden recibir parámetros de tres formas (nombres cambian según el motor):

* **IN** → Entrada (reciben valores).
* **OUT** → Salida (devuelven valores).
* **INOUT** → Pueden recibir y devolver valores.

Ejemplo en MySQL:

```sql
CREATE PROCEDURE SumarNumeros(IN a INT, IN b INT, OUT resultado INT)
BEGIN
    SET resultado = a + b;
END;
```

Llamada:

```sql
CALL SumarNumeros(3, 5, @res);
SELECT @res; -- devuelve 8
```

#### 🔸 Ejemplo real

Supongamos que queremos un procedimiento que devuelva los clientes con compras mayores a cierto valor:

```sql
CREATE PROCEDURE ClientesVIP(IN limite DECIMAL(10,2))
BEGIN
    SELECT cliente_id, nombre, SUM(total) AS total_compras
    FROM ventas
    GROUP BY cliente_id, nombre
    HAVING SUM(total) > limite;
END;
```

Ejecutar:

```sql
CALL ClientesVIP(10000);
```

### 🔸 Ventajas de los Procedimientos Almacenados

✅ Reutilización de código.
✅ Seguridad (no das acceso directo a tablas).
✅ Mejor rendimiento (menos tráfico de consultas).
✅ Mantenimiento centralizado.

### Resumen

#### ¿Qué son los procedimientos almacenados y por qué son útiles?

Los procedimientos almacenados son una herramienta fundamental para optimizar procesos repetitivos en la gestión de bases de datos. En lugar de escribir y ejecutar manualmente las consultas SQL cada vez que se necesita, los procedimientos almacenados permiten definir un conjunto de instrucciones SQL que pueden ser ejecutadas automáticamente. Esto es especialmente útil para tareas que requieren inserción, modificación o creación de registros con regularidad, como actualizaciones diarias, semanales o mensuales.

- **Automatización**: Los procedimientos almacenados facilitan la automatización de procesos al permitir su ejecución predefinida e interactuar con otros procesos empresariales.
- **Optimización**: Ayudan a mejorar la eficiencia del sistema al reducir el tiempo y esfuerzo para realizar tareas repetitivas, manteniendo la lógica de programación ya establecida.
- **Interacción con otros sistemas**: Pueden ser diseñados para generar salidas que se utilizan como entradas en otros sistemas, mejorando la integración y la coherencia en los flujos de trabajo.

##### ¿Cómo se crean los procedimientos almacenados en SQL y MySQL?

#### Creación de un procedimiento almacenado en SQL Server

Para crear un procedimiento almacenado en SQL Server, se utiliza la palabra reservada `CREATE PROCEDURE` seguida del nombre del procedimiento. Es importante ser específico en la denominación para facilitar la identificación de la función del procedimiento. Aquí se definen las variables y parámetros necesarios, especificando los tipos de datos que se esperan recibir.

```sql
CREATE PROCEDURE InsertEmployee 
    @Name NVARCHAR(50), 
    @Surname NVARCHAR(50), 
    @DepartmentId INT, 
    @Salary DECIMAL(10, 2), 
    @HireDate DATE
AS
BEGIN
    INSERT INTO Employees (Name, Surname, DepartmentId, Salary, HireDate)
    VALUES (@Name, @Surname, @DepartmentId, @Salary, @HireDate);
END;
```

### Creación de un procedimiento almacenado en MySQL

De manera similar a SQL Server, en MySQL se usa `CREATE PROCEDURE` pero los parámetros se definen con la palabra reservada `IN` para indicar que son entradas al procedimiento.

```sql
DELIMITER //
CREATE PROCEDURE InsertEmployee (IN Name VARCHAR(50), IN Surname VARCHAR(50), IN DepartmentId INT, IN Salary DECIMAL(10,2), IN HireDate DATE)
BEGIN
    INSERT INTO Employees (Name, Surname, DepartmentId, Salary, HireDate)
    VALUES (Name, Surname, DepartmentId, Salary, HireDate);
END;
//
DELIMITER ;
```

#### ¿Cómo se ejecutan los procedimientos almacenados?

Al ejecutar un procedimiento almacenado, se utiliza una sintaxis específica según el lenguaje SQL que se esté empleando.

- **En SQL Server**, se puede invocar el procedimiento utilizando la palabra reservada `EXEC`:

`EXEC InsertEmployee 'John', 'Doe', 3, 55000, '2023-10-01';`

- **En MySQL**, se utiliza la palabra CALL:

`CALL InsertEmployee('John', 'Doe', 3, 55000, '2023-10-01');`

Es crucial recordar que los parámetros deben enviarse en el orden exacto en que fueron definidos en el procedimiento.

#### ¿Cómo empezar a practicar procedimientos almacenados?

La práctica es esencial para dominar la creación y uso de procedimientos almacenados. Se recomienda utilizar plataformas de práctica SQL para experimentar con ejemplos y resolver errores que puedan surgir en el proceso de codificación.

1. **Elige un entorno SQL**: Puedes escoger plataformas gratuitas como MySQL Workbench, SQL Server Management Studio, o entornos en línea como DB-Fiddle.
2. **Prueba diferentes consultas**: Inserta, actualiza o elimina registros utilizando procedimientos almacenados.
3. **Experimenta con condicionales**: Asegúrate de usar condiciones para evitar operar en datos incorrectos.

Amplía tus conocimientos integrando consultas más complejas que incorporen estadísticas, predicciones o analíticas avanzadas. Así, no solo optimizas el tiempo de codificación, sino que también contribuyes al desarrollo eficiente de soluciones empresariales con procedimientos almacenados. Sigue practicando y compartiendo tus experiencias para fortalecer tu habilidad en esta poderosa herramienta de bases de datos.

## Procedimientos Almacenados en SQL: Gestión de Variables y Manejo de Excepciones

¡Excelente tema! 🚀
Los **Procedimientos Almacenados (Stored Procedures)** en SQL no solo permiten encapsular lógica de negocio dentro de la base de datos, sino que también pueden **manejar variables** y **gestionar excepciones** para un control más robusto.

### 🔹 1. Gestión de Variables en Procedimientos Almacenados

En un **procedimiento almacenado**, puedes declarar, asignar y usar **variables locales** para cálculos o control de flujo.

📌 Ejemplo en **MySQL**:

```sql
DELIMITER $$

CREATE PROCEDURE GetStudentInfo(IN student_id INT)
BEGIN
    DECLARE student_name VARCHAR(100);
    DECLARE student_age INT;

    -- Asignar valores a las variables
    SELECT CONCAT(FIRSTNAME, ' ', LASTNAME), AGE
    INTO student_name, student_age
    FROM STUDENTS
    WHERE STUDENT_ID = student_id;

    -- Mostrar resultado
    SELECT student_name AS Nombre, student_age AS Edad;
END$$

DELIMITER ;
```

✔ Aquí declaramos variables (`student_name`, `student_age`) y les asignamos datos con `SELECT ... INTO`.

### 🔹 2. Manejo de Excepciones

En algunos motores de BD como **MySQL** y **Oracle**, puedes manejar errores con **handlers**.
En **MySQL**, usamos `DECLARE ... HANDLER`.

📌 Ejemplo en **MySQL**:

```sql
DELIMITER $$

CREATE PROCEDURE SafeInsertStudent(
    IN fname VARCHAR(50),
    IN lname VARCHAR(50),
    IN age INT,
    IN email VARCHAR(100)
)
BEGIN
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
    BEGIN
        -- Acción en caso de error
        SELECT 'Ocurrió un error durante la inserción' AS ErrorMessage;
    END;

    -- Intentar inserción
    INSERT INTO STUDENTS (FIRSTNAME, LASTNAME, AGE, EMAIL)
    VALUES (fname, lname, age, email);

    -- Confirmar éxito
    SELECT 'Estudiante insertado correctamente' AS SuccessMessage;
END$$

DELIMITER ;
```

✔ Aquí si ocurre un error (ejemplo: email duplicado, violación de clave), el handler captura la excepción y retorna un mensaje en vez de fallar.

### 🔹 3. Manejo de Excepciones en **Oracle PL/SQL**

Oracle tiene un bloque `EXCEPTION` muy potente:

```sql
CREATE OR REPLACE PROCEDURE SafeInsertStudent(
    p_fname IN VARCHAR2,
    p_lname IN VARCHAR2,
    p_age   IN NUMBER,
    p_email IN VARCHAR2
) AS
BEGIN
    INSERT INTO STUDENTS (FIRSTNAME, LASTNAME, AGE, EMAIL)
    VALUES (p_fname, p_lname, p_age, p_email);

    DBMS_OUTPUT.PUT_LINE('Estudiante insertado correctamente');

EXCEPTION
    WHEN DUP_VAL_ON_INDEX THEN
        DBMS_OUTPUT.PUT_LINE('Error: Email duplicado');
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error inesperado: ' || SQLERRM);
END;
/
```

✔ `EXCEPTION` permite capturar errores específicos (`DUP_VAL_ON_INDEX`) o generales (`OTHERS`).

✅ **En resumen**:

* **Variables** → sirven para almacenar resultados intermedios dentro del procedimiento.
* **Handlers/Exceptions** → aseguran que el procedimiento no falle ante errores inesperados y puedes dar mensajes personalizados.

### Resumen

#### ¿Cómo crear procedimientos almacenados con parámetros de entrada y salida?

El manejo de procederes almacenados en bases de datos como MySQL es crucial para optimizar transacciones complejas y asegurar la integridad de los datos. Aprender a utilizarlos eficazmente, incluyendo la interacción con parámetros de entrada y salida, puede mejorar significativamente tu competencia en el uso de bases de datos. En este instructivo, exploraremos un escenario práctico donde se emplean ambos tipos de parámetros, construyendo sobre la base de los procedimientos previamente almacenados. Al finalizar, tendrás una comprensión sólida de cómo gestionar transacciones y manejar excepciones.

#### ¿Qué son los parámetros de entrada y salida?

En un procedimiento almacenado, los parámetros de entrada permiten pasar información a la operación. Utilizamos la palabra reservada `IN` para definirlos, seguido por el nombre del parámetro y su tipo de dato. Por otro lado, los parámetros de salida, definidos con `OUT`, sirven para devolver datos después de completar el procedimiento. Ambos tipos de parámetros comparten una estructura similar, diferenciándose solo por la palabra clave inicial.

#### ¿Cómo influye el uso de variables y manejo de excepciones?

Las variables actúan como contenedores para resultados intermedios o finales dentro del procedimiento almacenado. Es importante no confundir parámetros y variables, pues en MySQL las variables se declaran con la palabra clave `DECLARE`. En nuestro ejemplo, se utilizará una variable para almacenar el resultado del promedio de salario. Además, el manejo de excepciones es vital para cancelar transacciones ante errores, evitando inconsistencias en tus datos. Se utiliza `ROLLBACK` para revertir la transacción fallida, junto con el uso de `SELECT` para mostrar mensajes de error informativos al usuario.

#### ¿Cuál es el proceso para crear un procedimiento almacenado?

1. **Definición de Parámetros y Variables**:

- Parámetros de entrada (primer nombre, apellido, departamento, salario, fecha).
- Parámetro de salida (resultado promedio).
- Declaración de la variable para almacenar el resultado del promedio.

2. **Iniciación de la Transacción**:

- Comenzar la transacción con `BEGIN`.
- Iniciar la declaración de operaciones que incluye, en este ejemplo, `INSERT INTO` y una consulta de promedio de salario.

3. Manejo de Excepciones:

- Utilizar `SELECT` para definir el mensaje de error usando `AS` para etiquetarlo.
- Establecer `ROLLBACK` para revertir operaciones en caso de error.

4. **Concluir Transacción Exitosa**:

- Asignar el resultado del cálculo de promedio a la variable usando el símbolo `=`.
- Confirmar la conclusión exitosa de la transacción con `COMMIT`.
- Finalizar el procedimiento con `END`.

#### Ejemplo de sintaxis en MySQL

```sql
CREATE PROCEDURE CalcularPromedioSalario (
    IN nombre VARCHAR(100),
    IN apellido VARCHAR(100),
    IN departamento VARCHAR(100),
    IN salario DECIMAL(10,2),
    IN fecha DATE,
    OUT promedio DECIMAL(10,2)
)
BEGIN
    DECLARE resultadoPromedio DECIMAL(10,2);

    START TRANSACTION;

    BEGIN
        -- Insertar nueva entrada en la tabla empleados
        INSERT INTO empleados (nombre, apellido, departamento, salario, fecha)
        VALUES (nombre, apellido, departamento, salario, fecha);

        -- Calcular el promedio de salarios
        SELECT AVG(salario) INTO resultadoPromedio FROM empleados;

        -- Asignar el promedio calculado al parámetro de salida
        SET promedio = resultadoPromedio;

        COMMIT;
    EXCEPTION
        BEGIN
            -- Mensaje de error y rollback si algo falla
            SELECT 'Error: no se pudo completar la operación.' AS mensajeError;
            ROLLBACK;
        END;
    END;
    END;
```

Este ejemplo ilustra un procedimiento almacenado que inserta datos y calcula un promedio. En caso de error, el rollback garantiza que no se realicen cambios parciales en los datos.

El compromiso con el aprendizaje de estas herramientas te habilitará para manejar transacciones complejas con confianza y eficiencia. Te animamos a experimentar y crear tus propios procedimientos, incrementando así tu conocimiento y habilidades en gestión de bases de datos. Si tienes preguntas o deseas contribuir al desarrollo de estas prácticas, ¡compártelo en los comentarios!

## Respaldos y Restauración de Bases de Datos

Los **respaldos (backups)** y la **restauración (restore)** son procesos críticos en la administración de bases de datos, ya que garantizan la **seguridad, disponibilidad e integridad de la información** en caso de fallos, errores humanos o desastres.

### 🔹 1. ¿Qué es un Respaldo (Backup)?

Un **backup** es una copia de seguridad de la base de datos o de partes de ella (tablas, esquemas, registros, logs).
Sirve para recuperar los datos si se pierden o dañan.

### Tipos de respaldos:

1. **Completo (Full Backup)**
   Copia toda la base de datos.

   ```sql
   -- Ejemplo en MySQL
   mysqldump -u usuario -p basededatos > backup.sql
   ```

2. **Diferencial**
   Copia solo lo que ha cambiado desde el último backup completo.

3. **Incremental**
   Copia únicamente lo que cambió desde el último backup (sea completo o incremental).

4. **En caliente (Hot Backup)**
   Se realiza mientras la base está en uso, sin detener operaciones.

5. **En frío (Cold Backup)**
   Se realiza cuando la base de datos está detenida.

### 🔹 2. ¿Qué es una Restauración (Restore)?

Es el proceso de **recuperar la base de datos** a partir de un respaldo.
Permite volver a un estado anterior conocido.

### Ejemplo en MySQL:

```bash
mysql -u usuario -p basededatos < backup.sql
```

### Ejemplo en SQL Server:

```sql
RESTORE DATABASE MiBase
FROM DISK = 'C:\Backups\mibase.bak'
WITH REPLACE;
```

### 🔹 3. Estrategias de Respaldo y Restauración

* **Regla 3-2-1**:

  * 3 copias de los datos
  * 2 en diferentes medios (disco, nube)
  * 1 fuera del sitio (remoto)

* **Automatización** con jobs (cron, SQL Agent, scripts).

* **Pruebas periódicas** de restauración para garantizar que el backup funciona.

* **Monitoreo y alertas** para detectar fallos en el respaldo.

### 🔹 4. Buenas Prácticas

✅ Planifica respaldos regulares (diarios, semanales).
✅ Usa cifrado para proteger datos sensibles.
✅ Mantén logs de transacciones para recuperación punto en el tiempo.
✅ Documenta los procedimientos de recuperación ante desastres (DRP).

📌 **En resumen:**

* **Backup** = prevención.
* **Restore** = recuperación.
  Ambos son pilares de la **administración de bases de datos segura**.

### Resumen

#### ¿Qué es la copia de seguridad y restauración de datos?

Imagínate que, por error, alguien ejecuta un comando `drop table` o `delete` sin utilizar el `where`, provocando pérdidas masivas de información en una base de datos. O que, simplemente, un servidor colapsa, y se pierde la información temporalmente. Para estos casos, el uso adecuado de copias de seguridad (backups) y la restauración de datos es crucial. El encargado de garantizar esta seguridad y disponibilidad de la información es el administrador de bases de datos.

#### ¿Cuándo se deberían realizar las copias de seguridad?

La periodicidad para realizar copias de seguridad puede variar dependiendo de la criticidad de la información. Por ejemplo:

- **Industrias de venta de productos**: Podría ser suficiente tomar un snapshot (copia de seguridad) diariamente.
- **Bancos**: Debido a la sensibilidad de los datos, es probable que se realicen copias en diversos momentos del día.

#### ¿Cómo se pueden automatizar estas tareas?

En la actualidad, muchas plataformas en la nube ofrecen servicios que automatizan la creación de copias de seguridad según un período determinado por el usuario. Esto reduce la carga de trabajo y minimiza el riesgo de errores humanos.

##### ¿Cómo exportar e importar datos?

Exportar e importar datos es fundamental para gestionar y restaurar información. A continuación, se presenta un proceso simple para realizar estas tareas utilizando herramientas de consola.

#### Pasos para la exportación de datos

1. Selecciona la opción DataExport en tu consola de base de datos.
2. Elige la base de datos cuyo esquema deseas respaldar.
3. Establece la ruta donde se almacenarán los archivos de backup.
4. Verifica qué elementos de la base de datos (tablas, rutinas, procedimientos, vistas) se están exportando.

#### Pasos para la importación de datos

1. Accede a las opciones del servidor y selecciona `DataImport`.
2. Especifica la ruta en la que guardaste los archivos de respaldo.
3. Carga el contenido seleccionando la base de datos de destino.
4. Inicia el proceso para restablecer los archivos.

Este proceso puede realizarse manualmente, pero también existen métodos para automatizarlo, lo que resulta especialmente útil para proyectos grandes o con alta demanda de cambios.

#### ¿Cómo mejorar la práctica de backups?

Para fortalecer tus habilidades en la gestión de copias de seguridad:

- **Practica regularmente**: Realiza ejercicios de backup y restauración con bases de datos creadas durante el curso u otras que desarrolles.
- **Explora nuevas alternativas**: Investiga y prueba diferentes soluciones de restauración de datos, tanto locales como en la nube.
- **Utiliza recursos adicionales**: Revisa archivos y materiales proporcionados durante el curso para afianzar tus conocimientos prácticos.

Si tienes alguna duda o conoces métodos innovadores para la restauración de datos, ¡no dudes en compartir tus comentarios! La colaboración es clave para mejorar continuamente nuestras habilidades técnicas y de manejo de datos.

## Potenciando los Datos en la Nube: Data Science, Big Data, ML e AI

Hoy en día los datos en la nube son el **petróleo digital** y combinarlos con **Data Science, Big Data, Machine Learning (ML) e Inteligencia Artificial (AI)** permite transformar información en **valor estratégico**.

### 🚀 Potenciando los Datos en la Nube

### 🔹 1. La Nube como Habilitador

La nube (**AWS, Azure, Google Cloud, Oracle Cloud**) ofrece:

* **Escalabilidad**: crecer o reducir recursos bajo demanda.
* **Almacenamiento flexible**: desde bases relacionales hasta *data lakes*.
* **Procesamiento distribuido**: análisis en tiempo real con clusters y pipelines.
* **Servicios administrados**: quitan carga operativa al equipo.

### 🔹 2. Data Science en la Nube

El **Científico de Datos** aprovecha la nube para:

* **Recolectar y limpiar datos** (ETL/ELT con herramientas como AWS Glue, Azure Data Factory, Google Dataflow).
* **Analizar** con entornos listos (Jupyter en SageMaker, Databricks, Colab).
* **Visualizar** con dashboards (Power BI, Tableau, Looker, QuickSight).

👉 Ejemplo: Usar **BigQuery (GCP)** para analizar millones de registros de transacciones en segundos y detectar patrones de consumo.

### 🔹 3. Big Data en la Nube

El **Big Data** se caracteriza por las 5V:

1. **Volumen** (terabytes/petabytes)
2. **Velocidad** (procesamiento en streaming)
3. **Variedad** (estructurados, no estructurados, IoT)
4. **Veracidad** (datos confiables)
5. **Valor** (generar insights reales)

Herramientas clave:

* **Data Lakes**: Amazon S3 + Lake Formation, Azure Data Lake, Google Cloud Storage.
* **Procesamiento distribuido**: Spark, Hadoop en EMR, Dataproc, HDInsight.
* **Streaming**: Kafka, Kinesis, Pub/Sub.

### 🔹 4. Machine Learning (ML) en la Nube

Permite crear modelos predictivos y prescriptivos sin necesidad de infra compleja.
Servicios destacados:

* **AWS SageMaker** (entrenar, implementar y monitorear modelos).
* **Azure Machine Learning**.
* **Google Vertex AI**.

Ejemplo de aplicación:

* **Predicción de demanda** en retail.
* **Detección de fraude** en banca en tiempo real.

### 🔹 5. Inteligencia Artificial (AI) en la Nube

La AI lleva los modelos ML a otro nivel, con **capacidades cognitivas**:

* **Procesamiento de Lenguaje Natural (NLP)**: análisis de sentimiento, chatbots.
* **Visión por Computador**: reconocimiento facial, conteo de objetos en video.
* **Modelos generativos (GenAI)**: crear texto, imágenes, código (ej. OpenAI en Azure, Gemini en GCP).

Ejemplo:
Un hospital usa **AI en la nube** para analizar radiografías y detectar enfermedades con más rapidez que un diagnóstico manual.

### 🔹 6. Sinergia de Todo el Ecosistema

Cuando combinamos estas piezas:

* **Big Data** → captura masiva de información.
* **Data Science** → análisis y extracción de conocimiento.
* **Machine Learning** → predicción y automatización.
* **AI** → decisiones inteligentes y generación de nuevo contenido.
* **Cloud** → escalabilidad, seguridad y velocidad.

### 🔹 7. Retos y Buenas Prácticas

⚠️ Retos:

* Gobernanza y calidad de datos.
* Privacidad y cumplimiento (GDPR, HIPAA).
* Costos si no se optimiza.

✅ Buenas prácticas:

* Definir un **Data Lake + Data Warehouse híbrido**.
* Aplicar **MLOps** para escalar modelos ML.
* Implementar **seguridad en capas** (encriptación, IAM, auditoría).
* Aprovechar **arquitecturas serverless** (BigQuery, Athena, Synapse).

📌 **En resumen:**
La **nube es el motor** que hace posible llevar **Data Science, Big Data, ML y AI** a escala, de forma ágil, segura y rentable.

### Resumen

#### ¿Qué es la copia de seguridad y restauración de datos?

Imagínate que, por error, alguien ejecuta un comando `drop table` o `delete` sin utilizar el `where`, provocando pérdidas masivas de información en una base de datos. O que, simplemente, un servidor colapsa, y se pierde la información temporalmente. Para estos casos, el uso adecuado de copias de seguridad (backups) y la restauración de datos es crucial. El encargado de garantizar esta seguridad y disponibilidad de la información es el administrador de bases de datos.

#### ¿Cuándo se deberían realizar las copias de seguridad?

La periodicidad para realizar copias de seguridad puede variar dependiendo de la criticidad de la información. Por ejemplo:

- **Industrias de venta de productos**: Podría ser suficiente tomar un snapshot (copia de seguridad) diariamente.
- **Bancos**: Debido a la sensibilidad de los datos, es probable que se realicen copias en diversos momentos del día.

#### ¿Cómo se pueden automatizar estas tareas?

En la actualidad, muchas plataformas en la nube ofrecen servicios que automatizan la creación de copias de seguridad según un período determinado por el usuario. Esto reduce la carga de trabajo y minimiza el riesgo de errores humanos.

##### ¿Cómo exportar e importar datos?

Exportar e importar datos es fundamental para gestionar y restaurar información. A continuación, se presenta un proceso simple para realizar estas tareas utilizando herramientas de consola.

#### Pasos para la exportación de datos

1. Selecciona la opción DataExport en tu consola de base de datos.
2. Elige la base de datos cuyo esquema deseas respaldar.
3. Establece la ruta donde se almacenarán los archivos de backup.
4. Verifica qué elementos de la base de datos (tablas, rutinas, procedimientos, vistas) se están exportando.

#### Pasos para la importación de datos

1. Accede a las opciones del servidor y selecciona `DataImport`.
2. Especifica la ruta en la que guardaste los archivos de respaldo.
3. Carga el contenido seleccionando la base de datos de destino.
4. Inicia el proceso para restablecer los archivos.

Este proceso puede realizarse manualmente, pero también existen métodos para automatizarlo, lo que resulta especialmente útil para proyectos grandes o con alta demanda de cambios.

#### ¿Cómo mejorar la práctica de backups?

Para fortalecer tus habilidades en la gestión de copias de seguridad:

- **Practica regularmente**: Realiza ejercicios de backup y restauración con bases de datos creadas durante el curso u otras que desarrolles.
- **Explora nuevas alternativas**: Investiga y prueba diferentes soluciones de restauración de datos, tanto locales como en la nube.
- **Utiliza recursos adicionales**: Revisa archivos y materiales proporcionados durante el curso para afianzar tus conocimientos prácticos.

Si tienes alguna duda o conoces métodos innovadores para la restauración de datos, ¡no dudes en compartir tus comentarios! La colaboración es clave para mejorar continuamente nuestras habilidades técnicas y de manejo de datos.

## SQL para Análisis de Datos: Primeros pasos con Power BI

Combinar **SQL** con **Power BI** es una de las formas más efectivas para realizar **análisis de datos** y crear **reportes interactivos**. Te muestro los **primeros pasos**:

### 1. Preparar la Base de Datos

Antes de conectar con Power BI, necesitas una fuente de datos SQL (por ejemplo, **SQL Server, MySQL o PostgreSQL**).

Ejemplo de consulta SQL simple:

```sql
SELECT 
    student_id,
    firstname,
    lastname,
    age,
    email
FROM students
WHERE age > 20;
```

### 2. Conectar Power BI con SQL

En Power BI Desktop:

1. Clic en **Obtener datos** → **SQL Server** (o MySQL, PostgreSQL según corresponda).
2. Ingresar el **nombre del servidor** y la **base de datos**.
3. Elegir entre **Importar datos** (trae los datos a Power BI) o **DirectQuery** (consulta en tiempo real).
4. Seleccionar las tablas o escribir tu propia consulta SQL.

Ejemplo:

```sql
SELECT 
    AGE,
    COUNT(*) AS total_estudiantes
FROM students
GROUP BY AGE;
```

Esto permitirá que Power BI reciba la información ya agregada.

### 3. Transformar Datos (Power Query)

* El editor de Power BI permite:

  * **Renombrar columnas**
  * **Quitar duplicados**
  * **Filtrar datos**
  * **Crear columnas calculadas**
* Esto complementa lo que no quieras hacer directamente en SQL.

### 4. Crear Visualizaciones

Con los datos cargados puedes:

* Hacer **gráficos de barras** para distribución por edad.
* Usar **tarjetas** para mostrar totales (ej. número de estudiantes).
* Construir **dashboards interactivos** con filtros dinámicos.

### 5. Casos Comunes de SQL + Power BI

* **Ventas**: Totales, promedios y comparaciones por región.
* **Educación**: Cantidad de estudiantes por curso y edad.
* **Finanzas**: Tendencias de gastos e ingresos con series de tiempo.

👉 Con esto ya tendrías un flujo completo: **SQL (extracción y filtrado) → Power BI (visualización e interacción)**.

### Resumen

### ¿Cómo visualizar datos de manera efectiva?

La visualización de datos es un paso crucial al presentar la información final al usuario. Involucra no solo el análisis exhaustivo de los resultados, sino también la identificación de KPIs (Indicadores Clave de Desempeño) que puedan tener un impacto significativo en las decisiones de la empresa.

Para una visualización efectiva, debes tener en cuenta:

- **Identificación de KPIs**: Descubre cuál es el valor de ciertos indicadores, como la trayectoria de ventas anuales, o alertas, como la caída en las ventas de un producto.
- **Uso de Dashboards**: Utiliza herramientas como Power BI para crear dashboards que faciliten la interpretación de la información.

#### ¿Qué plataformas se pueden utilizar?

Una plataforma muy recomendada para la visualización de datos es Power BI, que te permite conectarte a diversos orígenes de datos:

- **Fuentes de datos variadas**: Puedes conectarte a bases de datos, archivos físicos (Excel, CSV) o servicios en línea.
- **Integración con otros recursos**: Es posible conectar Power BI a una base de datos MySQL, o cargar archivos CSV transformándolos en tablas visualizables.

### ¿Cuáles son los componentes esenciales de un Dashboard en Power BI?

Un Dashboard en Power BI se compone de tres secciones cruciales:

1. **Filtros**: Mejoran la experiencia del usuario al permitirle enfocar su análisis en áreas específicas.
2. **Visualizaciones**: Ofrecen gráficos interactivos como diagramas de trayectoria para facilitar el análisis comparativo de información.
3. **Datos**: Muestra la información cargada, segmentada por variables como departamento, edad y ventas, y diferenciada por tipo de dato (texto, numérico, tiempo).

##### ¿Cómo se seleccionan y visualizan los datos?

Para crear visualizaciones efectivas en Power BI, sigue estos pasos:

1. **Conectar y cargar datos**: Importa datos desde tus archivos CSV o bases de datos directas.
2. **Elegir tipología de gráfico**: Explora y selecciona diferentes tipos de gráficos que mejor representen tus datos.
3. **Experimentación y creatividad**: Cambia ejes y visualizaciones para identificar nuevos insights y KPIs.

#### ¿Qué más puedo explorar en Power BI?

Power BI proporciona varias características adicionales para profundizar en tus análisis, como:

- **Crear KPIs avanzados**: Aplica reglas de cálculo porcentual para determinar el valor relativo de una métrica en comparación con otras.
- **Agregar filtros avanzados**: Mejora la interacción con tus datos filtrando información específica para un análisis más focalizado.

#### Riqueza de los datos y su impacto

Los datos son esenciales no solo en tecnología, sino en todos los campos comerciales. Manipular y analizar estos datos correctamente puede conducir a decisiones óptimas, mientras que un mal análisis puede llevar a errores significativos. Algunas consideraciones clave incluyen:

- **Interpretación adecuada**: Un análisis preciso requiere entender correctamente los datos y su contexto.
- **Calidad y valor**: Asegúrate de que la información sea real, relevante y de valor para la empresa.

En resumen, el manejo eficiente de datos te permite no solo entender el presente, sino prever tendencias futuras para una toma de decisiones más informada y estratégica. ¡Sigue explorando y fortaleciendo tu habilidad para interpretar datos con herramientas como Power BI!