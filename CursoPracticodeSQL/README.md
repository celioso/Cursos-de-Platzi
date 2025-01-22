# Curso Práctico de SQL

## Breve historia de SQL

**SQL (Structured Query Language)** es un lenguaje de programación diseñado para interactuar con bases de datos. A lo largo de los años, ha evolucionado desde sus inicios hasta convertirse en uno de los estándares más utilizados para el manejo de datos.

#### Historia:

1. **1970**: Edgar F. Codd, un científico de IBM, propuso el modelo relacional como una forma de representar y manejar datos. Su trabajo sentó las bases para el desarrollo de SQL.

2. **1974**: IBM desarrolló el primer lenguaje para bases de datos relacionales, llamado **SEQUEL** (Structured English Query Language), que posteriormente evolucionó a SQL.

3. **1976**: El término SQL fue adoptado oficialmente por ANSI (American National Standards Institute) y se convirtió en un estándar para bases de datos.

4. **1986**: Oracle fue la primera compañía en implementar SQL en sus sistemas de bases de datos comerciales.

5. **1989**: SQL se convierte en un estándar reconocido internacionalmente con la publicación de la **SQL-89**.

6. **1992**: Se publica la especificación **SQL-92**, que define formalmente el lenguaje SQL como lo conocemos hoy en día. Este estándar es el más ampliamente utilizado.

7. **2003**: Se publica **SQL:2003**, una versión más avanzada que incorpora características como la programación en procedimientos almacenados y nuevos tipos de datos.

8. **2011**: Se lanza **SQL:2011**, incluyendo más funcionalidad para análisis, JSON, y acceso distribuido.

A lo largo de los años, SQL ha sido adoptado por múltiples sistemas de bases de datos, como PostgreSQL, MySQL, Oracle, SQL Server, y otros, consolidándose como el estándar clave para gestionar bases de datos relacionales.

![que es sql](images/queessql.jpg)

![SQL CHEAT SHEET](images/SQLCHEATSHEET.jpg)

## Álgebra relacional

El **álgebra relacional** es una teoría matemática utilizada para definir las operaciones que se pueden realizar en los modelos de bases de datos relacionales. Se basa en el conjunto de principios matemáticos y lógica que describe cómo se pueden combinar y manipular tablas (relaciones) para obtener resultados deseados.

#### Principales operaciones del Álgebra Relacional:

1. **Selección (σ)**:
   - Permite filtrar las filas (registros) de una tabla según una condición.
   - Sintaxis: `σ_condición(Tabla)`
   - Ejemplo: `σ[edad>30](Empleado)` selecciona todos los empleados mayores de 30 años.

2. **Proyección (π)**:
   - Permite seleccionar ciertas columnas de una tabla.
   - Sintaxis: `π_columnas(Tabla)`
   - Ejemplo: `π[nombre,apellido](Empleado)` selecciona sólo las columnas nombre y apellido.

3. **Unión (∪)**:
   - Combina los registros de dos tablas, eliminando los duplicados.
   - Sintaxis: `Tabla1 ∪ Tabla2`
   - Nota: Ambas tablas deben tener el mismo número de columnas y tipos de datos.

4. **Intersección (∩)**:
   - Devuelve los registros que están presentes en ambas tablas.
   - Sintaxis: `Tabla1 ∩ Tabla2`
   - Ambas tablas deben tener el mismo número de columnas y tipos de datos.

5. **Diferencia (−)**:
   - Devuelve los registros presentes en la primera tabla pero no en la segunda.
   - Sintaxis: `Tabla1 − Tabla2`

6. **Producto cartesiano (×)**:
   - Combina todas las filas de una tabla con todas las filas de otra tabla.
   - Sintaxis: `Tabla1 × Tabla2`

7. **Join (⋈)**:
   - Combina las filas de dos tablas en función de una condición de igualdad entre columnas.
   - Tipos de Join:
     - **Inner Join**: Combina las filas que coinciden en ambas tablas.
     - **Left (Outer) Join**: Combina las filas de la primera tabla con todas las filas de la segunda, colocando `NULL` en las columnas cuando no hay coincidencia.
     - **Right (Outer) Join**: Combina las filas de la segunda tabla con todas las filas de la primera, colocando `NULL` en las columnas cuando no hay coincidencia.
     - **Full (Outer) Join**: Combina las filas de ambas tablas, colocando `NULL` en las columnas cuando no hay coincidencia en ambas.

8. **Renombramiento (ρ)**:
   - Permite renombrar las columnas o tablas resultantes de una operación.
   - Sintaxis: `ρ_nombre_nueva(operacion)`

9. **Agrupación (GROUP BY)**:
   - Agrupa los registros según una o varias columnas y luego aplica funciones agregadas como `SUM`, `AVG`, `COUNT`, `MAX`, `MIN`.
   - Sintaxis: `GROUP BY columna`

10. **Ordenamiento (ORDER BY)**:
    - Ordena los registros de acuerdo con una o más columnas.
    - Sintaxis: `ORDER BY columna`

#### Ejemplo básico:
Queremos seleccionar los nombres y las edades de los empleados que trabajan en el departamento 10, ordenados por edad:

```sql
SELECT nombre, edad
FROM Empleado
WHERE departamento = 10
ORDER BY edad;
```

Este sería un ejemplo clásico de la operación en álgebra relacional. 

El álgebra relacional es fundamental para entender cómo funcionan las consultas SQL y cómo los datos pueden manipularse utilizando estas operaciones matemáticas básicas.

## Instalación de la BD de ejemplo

Como requisito previo es necesario instalar la BD siguiendo este [tutorial](https://platzi.com/clases/1480-postgresql/24177-instalacion-y-configuracion-de-la-base-de-datos/) del curso de PostgreSQL.

Archivos de datos SQL: descarga archivo [platzi-carreras.sql](https://static.platzi.com/media/public/uploads/platzi-carreras_65e74975-a728-481b-bbe4-c52483529661.sql) y archivo [platzi-alumnos.sql](https://static.platzi.com/media/public/uploads/platzi-alumnos_1f7f704d-2c8a-49c7-8c3d-e94365eb8d15.sql).

Una vez tienes instalado PostgreSQL y pgAdmin vamos a crear la estructura de datos que veremos a lo largo del curso.

Para hacerlo abre pgAdmin (normalmente está en la dirección: [http://127.0.0.1:63435/browser/](http://127.0.0.1:63435/browser/)), y expande el panel correspondiente a tu base de datos, en mi caso la he nombrado “prueba”.

![pgAdmin 1](pgAdmin1.png)

En la sección esquemas da click secundario y selecciona la opción Create > Schema…

![pgAdmin 2](pgAdmin2.png)

Al seleccionar la opción abrirá un cuadro de diálogo en donde debes escribir el nombre del esquema, en este caso será “platzi”. Si eliges un nombre distinto, asegúrate de seguir los ejemplos en el curso con el nombre elegido; por ejemplo si en el curso mencionamos la sentencia:

`SELECT * FROM platzi.alumnos`

Sustituye platzi por el nombre que elegiste.

Finalmente selecciona tu usuario de postgres en el campo Owner, esto es para que asigne todos los permisos del nuevo esquema a tu usuario.

![pgAdmin 3](pgAdmin3.png)

Revisa que tu esquema se haya generado de manera correcta recargando la página y expandiendo el panel Schemas en tu base de datos.

![pgAdmin 4](pgAdmin4.png)

Dirígete al menú superior y selecciona el menú Tools > Query Tool.

![pgAdmin 5](pgAdmin5.png)

Esto desplegará la herramienta en la ventana principal. Da click en el botón “Open File” ilustrado por un icono de folder abierto.

![pgAdmin 6](pgAdmin6.png)

Busca en tus archivos y selecciona el archivo platzi.alumnos.sql que descargaste de este curso, da click en el botón “Select”.

![pgAdmin 7](pgAdmin7.png)

Esto abrirá el código SQL que deberás ejecutar dando click en el botón ”Execute/Refresh” con el icono play.

![pgAdmin 8](pgAdmin8.png)

Al terminar debes ver un aviso similar al siguiente:

![pgAdmin 9](pgAdmin9.png)

Ahora repetiremos el proceso para la tabla platzi.carreras. Dirígete nuevamente al botón “Open File” y da click en él.

![pgAdmin 10](pgAdmin10.png)


Encuentra y selecciona el archivo platzi.carreras.sql y da click en el botón “Select”.

![pgAdmin 11](pgAdmin11.png)

Una vez abierto el archivo corre el script dando click en el botón “Execute/Refresh”

![pgAdmin 12](pgAdmin12.png)

Debes ver nuevamente un aviso como el siguiente:

![pgAdmin 13](pgAdmin13.png)

¡Felicidades! Ya tienes todo listo para realizar los ejercicios y retos del curso.

## Qué es una proyección (SELECT)

La **proyección** es un concepto del álgebra relacional que en SQL se implementa con el comando **`SELECT`**. Este se utiliza para seleccionar y devolver columnas específicas de una tabla, eliminando las columnas no deseadas y enfocándose únicamente en los datos relevantes.

#### Características de la Proyección:
1. **Selección de columnas**:
   - Permite especificar qué columnas se incluirán en el resultado.
   - Reduce el número de atributos en el conjunto de resultados.

2. **No elimina filas**:
   - La proyección afecta solo las columnas; las filas permanecen tal como están, a menos que se combine con una condición de filtro (**`WHERE`**).

3. **Sintaxis básica**:
   ```sql
   SELECT columna1, columna2, ...
   FROM nombre_tabla;
   ```

#### Ejemplo básico:
Dada la tabla **empleados**:

| id_empleado | nombre   | edad | salario |
|-------------|----------|------|---------|
| 1           | Ana      | 30   | 1500    |
| 2           | Luis     | 45   | 2000    |
| 3           | Marta    | 29   | 1800    |

Consulta para proyectar solo los nombres y edades:
```sql
SELECT nombre, edad
FROM empleados;
```

Resultado:
| nombre   | edad |
|----------|------|
| Ana      | 30   |
| Luis     | 45   |
| Marta    | 29   |

#### Variantes del **SELECT**:
1. **Seleccionar todas las columnas**:
   - Usando el carácter `*`, se obtienen todas las columnas de una tabla:
     ```sql
     SELECT *
     FROM empleados;
     ```

2. **Eliminar duplicados**:
   - Usando **`DISTINCT`**, se eliminan filas duplicadas en las columnas proyectadas:
     ```sql
     SELECT DISTINCT edad
     FROM empleados;
     ```
     Resultado:
     | edad |
     |------|
     | 30   |
     | 45   |
     | 29   |

3. **Crear nuevas columnas derivadas**:
   - Se pueden realizar cálculos o concatenar datos y proyectarlos como una nueva columna:
     ```sql
     SELECT nombre, salario * 12 AS salario_anual
     FROM empleados;
     ```
     Resultado:
     | nombre   | salario_anual |
     |----------|---------------|
     | Ana      | 18000         |
     | Luis     | 24000         |
     | Marta    | 21600         |

4. **Usar alias para las columnas**:
   - Se puede cambiar el nombre de las columnas en el resultado:
     ```sql
     SELECT nombre AS empleado, salario AS ingreso
     FROM empleados;
     ```

5. **Proyección condicional**:
   - Combinando con **`WHERE`**, selecciona filas específicas:
     ```sql
     SELECT nombre, salario
     FROM empleados
     WHERE edad > 30;
     ```

#### Importancia de la Proyección:
1. **Optimización**:
   - Reduce el volumen de datos transmitido y procesado al cliente o aplicación.
2. **Claridad**:
   - Facilita el entendimiento de los resultados al mostrar solo las columnas necesarias.
3. **Preparación de datos**:
   - Permite estructurar los datos para reportes, análisis o transformaciones posteriores.

#### Diferencia entre Proyección y Selección:
- **Proyección**: Se enfoca en columnas (atributos).
- **Selección**: Se enfoca en filas, usualmente mediante condiciones (**`WHERE`**). 

Ambos conceptos se pueden combinar en una consulta SQL para obtener datos específicos y relevantes.

![SELECT](images/SELECT.JPG)


## Origen (FROM)

El término **FROM** en SQL es una cláusula clave que especifica la fuente de los datos para una consulta. Es el punto de partida para cualquier operación de consulta en SQL, ya que indica la tabla o tablas de las que se extraerá la información.

#### Características principales del **FROM**:

1. **Especificación de tablas**:
   - Indica de qué tabla o tablas se obtendrán los datos.
   - Ejemplo:
     ```sql
     SELECT nombre, edad
     FROM empleados;
     ```
     En este caso, `empleados` es la tabla fuente.

2. **Combinar tablas**:
   - Permite realizar combinaciones de datos entre múltiples tablas utilizando **JOIN**.
   - Ejemplo (Inner Join):
     ```sql
     SELECT e.nombre, d.nombre_departamento
     FROM empleados e
     JOIN departamentos d
     ON e.id_departamento = d.id_departamento;
     ```
     Aquí, se combinan las tablas `empleados` y `departamentos`.

3. **Subconsultas**:
   - Se pueden usar subconsultas como fuentes de datos.
   - Ejemplo:
     ```sql
     SELECT *
     FROM (
         SELECT id, nombre
         FROM empleados
         WHERE edad > 30
     ) AS subconsulta;
     ```
     En este caso, la subconsulta actúa como una tabla temporal.

4. **Especificación de alias**:
   - Los alias facilitan la referencia a tablas y columnas con nombres largos.
   - Ejemplo:
     ```sql
     SELECT e.nombre, e.edad
     FROM empleados AS e;
     ```

5. **Uso con funciones**:
   - Se puede usar con funciones para generar tablas virtuales.
   - Ejemplo (uso de una función generadora de series):
     ```sql
     SELECT *
     FROM generate_series(1, 10) AS numeros;
     ```

#### Operaciones avanzadas con **FROM**:

1. **Unión de tablas**:
   - **CROSS JOIN**: Producto cartesiano.
     ```sql
     SELECT *
     FROM tabla1
     CROSS JOIN tabla2;
     ```
   - **LEFT JOIN**: Incluye todas las filas de la tabla izquierda.
   - **RIGHT JOIN**: Incluye todas las filas de la tabla derecha.
   - **FULL JOIN**: Combina todas las filas de ambas tablas.

2. **Consulta sobre vistas o vistas materializadas**:
   - El **FROM** también puede apuntar a vistas.
   - Ejemplo:
     ```sql
     SELECT *
     FROM vista_empleados_activos;
     ```

3. **Uso con tablas derivadas**:
   - Permite generar una "tabla en línea" con una subconsulta.
   - Ejemplo:
     ```sql
     SELECT sub.id, sub.total
     FROM (
         SELECT id_departamento AS id, COUNT(*) AS total
         FROM empleados
         GROUP BY id_departamento
     ) AS sub;
     ```

El **FROM** es el núcleo para cualquier consulta SQL, ya que establece la base sobre la que se construyen las demás cláusulas como **WHERE**, **GROUP BY**, **HAVING**, y **ORDER BY**.

![FROM](images/From.jpg)

## Productos cartesianos (JOIN)

El producto cartesiano, también conocido como **CROSS JOIN**, es una operación en SQL que combina cada fila de una tabla con cada fila de otra tabla. Es la base para otros tipos de combinaciones (**INNER JOIN**, **LEFT JOIN**, etc.), pero en su forma básica no aplica ninguna condición para emparejar filas. 

#### Características del Producto Cartesiano:
1. **Resultado**:
   - El número total de filas en el resultado es el producto del número de filas de las dos tablas.
   - Si la tabla A tiene \( n \) filas y la tabla B tiene \( m \) filas, el resultado del producto cartesiano tendrá \( n \times m \) filas.

2. **Sintaxis básica**:
   ```sql
   SELECT *
   FROM tabla1
   CROSS JOIN tabla2;
   ```

3. **Alias para simplificar referencias**:
   - Al trabajar con productos cartesianos, es útil asignar alias a las tablas:
     ```sql
     SELECT t1.columna1, t2.columna2
     FROM tabla1 AS t1
     CROSS JOIN tabla2 AS t2;
     ```

#### Ejemplo de un Producto Cartesiano:
Supongamos que tenemos las siguientes tablas:

**Tabla A** (empleados):
| id_empleado | nombre   |
|-------------|----------|
| 1           | Ana      |
| 2           | Luis     |

**Tabla B** (departamentos):
| id_departamento | nombre_departamento |
|-----------------|---------------------|
| 10              | Ventas             |
| 20              | Marketing          |

La consulta:
```sql
SELECT *
FROM empleados
CROSS JOIN departamentos;
```

**Resultado del Producto Cartesiano**:
| id_empleado | nombre   | id_departamento | nombre_departamento |
|-------------|----------|-----------------|---------------------|
| 1           | Ana      | 10              | Ventas             |
| 1           | Ana      | 20              | Marketing          |
| 2           | Luis     | 10              | Ventas             |
| 2           | Luis     | 20              | Marketing          |

#### Uso Práctico del Producto Cartesiano:
Aunque el producto cartesiano puro no es común en escenarios prácticos debido a su crecimiento exponencial, puede ser útil en ciertos casos:
1. **Generar combinaciones**:
   - Por ejemplo, todas las combinaciones posibles de productos y colores:
     ```sql
     SELECT p.nombre_producto, c.color
     FROM productos AS p
     CROSS JOIN colores AS c;
     ```

2. **Simular combinaciones sin relaciones explícitas**:
   - Útil en análisis exploratorio cuando no hay llaves foráneas entre las tablas.

3. **Fundamento para otros tipos de JOIN**:
   - El **INNER JOIN** o el **LEFT JOIN** se derivan del producto cartesiano con condiciones aplicadas para emparejar filas.

#### Consideraciones:
- El producto cartesiano puede generar resultados extremadamente grandes si las tablas involucradas son grandes.
- Es más eficiente usar **INNER JOIN** con condiciones específicas en lugar de **CROSS JOIN**, salvo que realmente se necesite combinar todas las filas.

#### Alternativa con Condición:
Para evitar un producto cartesiano completo y filtrar resultados, se puede usar un **INNER JOIN** con una condición:
```sql
SELECT e.nombre, d.nombre_departamento
FROM empleados AS e
INNER JOIN departamentos AS d
ON e.id_departamento = d.id_departamento;
```
Esto genera una intersección basada en la relación lógica entre las tablas.

![JOIN](images/join.jpg)

**Lecturas recomendadas**

[Introducción a los conjuntos en Curso de Matemáticas Discretas](https://platzi.com/clases/1319-discretas/12215-introduccion-a-los-conjuntos/)

[Curso de Matemáticas discretas](https://platzi.com/clases/discretas/)

## Selección (WHERE)

La **selección** es un concepto del álgebra relacional que en SQL se implementa mediante la cláusula **`WHERE`**. Se utiliza para filtrar las filas de una tabla, devolviendo solo aquellas que cumplen con una condición lógica específica.

#### Características de la Selección:
1. **Filtra filas**:
   - Permite definir condiciones para incluir solo los datos relevantes.
   - No afecta las columnas seleccionadas, sino el conjunto de filas.

2. **Sintaxis básica**:
   ```sql
   SELECT columnas
   FROM tabla
   WHERE condición;
   ```

3. **Operadores comunes en condiciones**:
   - **Comparación**: `=`, `!=`, `<`, `<=`, `>`, `>=`
   - **Lógicos**: `AND`, `OR`, `NOT`
   - **Inclusión**: `IN`, `BETWEEN`
   - **Patrones**: `LIKE`, `ILIKE`
   - **Nulos**: `IS NULL`, `IS NOT NULL`

#### Ejemplo básico:
Dada la tabla **empleados**:

| id_empleado | nombre   | edad | salario |
|-------------|----------|------|---------|
| 1           | Ana      | 30   | 1500    |
| 2           | Luis     | 45   | 2000    |
| 3           | Marta    | 29   | 1800    |

Consulta para seleccionar empleados con salario mayor a 1500:
```sql
SELECT nombre, salario
FROM empleados
WHERE salario > 1500;
```

Resultado:
| nombre   | salario |
|----------|---------|
| Luis     | 2000    |
| Marta    | 1800    |

#### Operadores en **`WHERE`**:
1. **Condiciones simples**:
   ```sql
   WHERE edad = 30;
   ```

2. **Condiciones compuestas**:
   - **AND**: Ambas condiciones deben cumplirse.
     ```sql
     WHERE edad > 30 AND salario > 1800;
     ```
   - **OR**: Al menos una condición debe cumplirse.
     ```sql
     WHERE edad > 30 OR salario > 1800;
     ```
   - **NOT**: Excluye filas que cumplen la condición.
     ```sql
     WHERE NOT (edad > 30);
     ```

3. **Rango de valores**:
   - Usando **`BETWEEN`**:
     ```sql
     WHERE salario BETWEEN 1500 AND 2000;
     ```

4. **Inclusión en listas**:
   - Usando **`IN`**:
     ```sql
     WHERE nombre IN ('Ana', 'Marta');
     ```

5. **Búsqueda de patrones**:
   - Usando **`LIKE`**:
     ```sql
     WHERE nombre LIKE 'M%'; -- Nombres que comiencen con "M"
     ```
   - Usando comodines:
     - `%`: Cualquier número de caracteres.
     - `_`: Un solo carácter.

6. **Valores nulos**:
   - Para verificar si una columna tiene un valor `NULL`:
     ```sql
     WHERE salario IS NULL;
     ```

#### Selección combinada con Proyección:
Puedes usar **`WHERE`** junto con **`SELECT`** para filtrar y seleccionar columnas específicas:
```sql
SELECT nombre, salario
FROM empleados
WHERE edad > 30;
```

Resultado:
| nombre   | salario |
|----------|---------|
| Luis     | 2000    |

#### Importancia de la Selección:
1. **Eficiencia**:
   - Reduce el número de filas procesadas.
   - Optimiza las consultas al trabajar con datos relevantes.
   
2. **Precisión**:
   - Proporciona control sobre los resultados al filtrar datos según necesidades específicas.

3. **Preparación de datos**:
   - Útil para análisis, reportes o integraciones con otras aplicaciones.

#### Diferencia entre Selección y Proyección:
- **Selección (`WHERE`)**: Filtra filas.
- **Proyección (`SELECT`)**: Selecciona columnas.

Ambos conceptos son fundamentales en SQL y se combinan para obtener datos específicos de una tabla.

![WHERE](images/RESUMEN_WHERE.jpg)

## Ordenamiento (ORDER BY)

La cláusula **`ORDER BY`** se utiliza en SQL para ordenar los resultados de una consulta en función de una o más columnas. Este orden puede ser ascendente o descendente, dependiendo de las necesidades del usuario.

### Características del **`ORDER BY`**:

1. **Control del orden**:
   - **Ascendente (ASC)**: Ordena los resultados de menor a mayor (por defecto).
   - **Descendente (DESC)**: Ordena los resultados de mayor a menor.

2. **Ordenación múltiple**:
   - Puedes especificar varias columnas para ordenar los resultados.
   - El orden de prioridad sigue el orden en que se escriben las columnas.

3. **Compatibilidad con funciones**:
   - Puede ordenar por expresiones o funciones, como concatenaciones, cálculos o agregados.

### Sintaxis:
```sql
SELECT columnas
FROM tabla
[WHERE condición]
ORDER BY columna1 [ASC | DESC], columna2 [ASC | DESC];
```

### Ejemplo básico:

Dada la tabla **empleados**:

| id_empleado | nombre   | edad | salario |
|-------------|----------|------|---------|
| 1           | Ana      | 30   | 1500    |
| 2           | Luis     | 45   | 2000    |
| 3           | Marta    | 29   | 1800    |

#### Ordenar por salario de menor a mayor:
```sql
SELECT nombre, salario
FROM empleados
ORDER BY salario ASC;
```

Resultado:
| nombre   | salario |
|----------|---------|
| Ana      | 1500    |
| Marta    | 1800    |
| Luis     | 2000    |

#### Ordenar por salario de mayor a menor:
```sql
SELECT nombre, salario
FROM empleados
ORDER BY salario DESC;
```

Resultado:
| nombre   | salario |
|----------|---------|
| Luis     | 2000    |
| Marta    | 1800    |
| Ana      | 1500    |

### Ordenamiento múltiple:

#### Ordenar por edad y, en caso de empate, por salario:
```sql
SELECT nombre, edad, salario
FROM empleados
ORDER BY edad ASC, salario DESC;
```

Resultado:
| nombre   | edad | salario |
|----------|------|---------|
| Marta    | 29   | 1800    |
| Ana      | 30   | 1500    |
| Luis     | 45   | 2000    |

### Ordenar por expresiones o funciones:
Puedes usar funciones o expresiones directamente en **`ORDER BY`**.

#### Ordenar por la longitud del nombre:
```sql
SELECT nombre, salario
FROM empleados
ORDER BY LENGTH(nombre) ASC;
```

Resultado:
| nombre   | salario |
|----------|---------|
| Ana      | 1500    |
| Luis     | 2000    |
| Marta    | 1800    |

### Ordenar por columnas que no están en la lista de selección:
Aunque una columna no esté en la lista de columnas seleccionadas, puedes usarla en **`ORDER BY`**.

```sql
SELECT nombre
FROM empleados
ORDER BY salario DESC;
```

Resultado:
| nombre   |
|----------|
| Luis     |
| Marta    |
| Ana      |

### Importancia del **`ORDER BY`**:
1. **Presentación ordenada**:
   - Mejora la legibilidad de los datos, especialmente para reportes o tablas grandes.

2. **Preparación de datos**:
   - Facilita análisis posteriores al ordenar datos de forma significativa.

3. **Compatibilidad con otras operaciones**:
   - Combina bien con funciones de agregación, particionamiento y paginación.

### Notas importantes:
- Si no especificas **ASC** o **DESC**, el orden será **ASC** por defecto.
- El rendimiento de **`ORDER BY`** puede depender del tamaño de la tabla y de la existencia de índices.

El ordenamiento es una herramienta poderosa para estructurar los resultados y facilitar su comprensión y análisis.

![order by](images/ordenby.jpg)