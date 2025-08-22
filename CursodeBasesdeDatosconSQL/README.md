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