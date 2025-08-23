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