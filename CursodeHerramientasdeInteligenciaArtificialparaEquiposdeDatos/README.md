# Curso de Herramientas de Inteligencia Artificial para Equipos de Datos

## Usar IA para programar es inevitable

Es cierto que el uso de la inteligencia artificial (IA) en la programación está creciendo rápidamente, y en muchos casos, se ha vuelto una herramienta esencial. La IA tiene el potencial de transformar el proceso de desarrollo de software de diversas maneras, incluyendo:

1. **Automatización de tareas repetitivas**: Los algoritmos de IA pueden encargarse de tareas rutinarias como la generación de código básico, la detección de errores comunes, y el formateo de código, lo que permite a los desarrolladores concentrarse en problemas más complejos.

2. **Mejora de la productividad**: Herramientas como GitHub Copilot y ChatGPT, integradas en editores de código, ofrecen sugerencias en tiempo real, ayudan a escribir código más rápido y proporcionan fragmentos de código para tareas comunes. Esto acelera el desarrollo y reduce el tiempo necesario para resolver problemas.

3. **Depuración y optimización**: Los sistemas de IA pueden analizar grandes volúmenes de código y datos para detectar patrones de fallos, optimizar el rendimiento y sugerir correcciones antes de que el problema se manifieste.

4. **Accesibilidad para programadores novatos**: Los desarrolladores sin mucha experiencia pueden usar IA para aprender, generar ideas y escribir código, facilitando el aprendizaje y la entrada en la industria tecnológica.

5. **Generación automática de pruebas**: La IA puede generar casos de prueba y validar el código automáticamente, lo que mejora la calidad del software y reduce el tiempo que lleva realizar pruebas manualmente.

6. **Programación colaborativa**: Las IA pueden facilitar el trabajo en equipo, sugiriendo mejoras en el código y promoviendo buenas prácticas de programación entre los miembros del equipo.

Aunque es inevitable que la IA continúe integrándose en el proceso de programación, sigue siendo esencial que los programadores humanos mantengan el control creativo y estratégico sobre el desarrollo. La IA es una herramienta complementaria que amplifica la capacidad de los desarrolladores, pero no sustituye el pensamiento crítico ni la resolución creativa de problemas.

```--SQL POSTGRES  
--Genera un modelo de datos con las tablas customer, orders, orders_detail, product, utilizando fireign keys 

CREATE TABLE customer (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(50) NOT NULL,
    phone VARCHAR(50) NOT NULL,
    address VARCHAR(50) NOT NULL,
    city VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    zip_code VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    order_date DATE NOT NULL,
    order_status VARCHAR(50) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
);

CREATE TABLE product (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(50) NOT NULL,
    product_price DECIMAL(10,2) NOT NULL
);

CREATE TABLE orders_detail (
    order_detail_id SERIAL PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES product(product_id)
);

--- crea la tabla category [description, status, updated_at, created_at] y product_category con las foreign keys keys necesarias 

CREATE TABLE category (
    category_id SERIAL PRIMARY KEY,
    description VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    updated_at DATE NOT NULL,
    created_at DATE NOT NULL
);

CREATE TABLE product_category (
    product_category_id SERIAL PRIMARY KEY,
    product_id INT NOT NULL,
    category_id INT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES product(product_id),
    FOREIGN KEY (category_id) REFERENCES category(category_id)
);

-- crea una copia de la tabla product_category con el nombre product_category_backup

CREATE TABLE product_category_backup AS TABLE product_category;

-- elimina la tabla product_category_backup 

DROP TABLE product_category_backup;
```

Para mejorar la velocidad de las consultas en la tabla `orders`, puedes crear índices compuestos en función de las consultas más comunes. Por ejemplo, si consultas frecuentemente las órdenes por el `customer_id` y la `order_date`, puedes crear un índice compuesto en estos campos.

A continuación se muestran algunos ejemplos de índices compuestos útiles para la tabla `orders`:

```sql
-- Índice compuesto en customer_id y order_date
CREATE INDEX idx_orders_customer_date
ON orders (customer_id, order_date);

-- Índice compuesto en status y order_date (para filtrar por estado y fecha)
CREATE INDEX idx_orders_status_date
ON orders (status, order_date);

-- Índice compuesto en customer_id y status (para consultas por cliente y estado del pedido)
CREATE INDEX idx_orders_customer_status
ON orders (customer_id, status);
```

### ¿Cómo mejoran las consultas estos índices?

1. **`idx_orders_customer_date`**: Optimiza las consultas que buscan órdenes de un cliente específico y las ordenan o filtran por fecha.
   ```sql
   SELECT * FROM orders
   WHERE customer_id = 123 AND order_date > '2023-01-01';
   ```

2. **`idx_orders_status_date`**: Acelera las consultas que filtran las órdenes por estado (por ejemplo, "pendiente" o "enviado") y fecha.
   ```sql
   SELECT * FROM orders
   WHERE status = 'Pending' AND order_date BETWEEN '2023-01-01' AND '2023-06-01';
   ```

3. **`idx_orders_customer_status`**: Mejora las consultas que buscan órdenes por cliente y estado.
   ```sql
   SELECT * FROM orders
   WHERE customer_id = 123 AND status = 'Shipped';
   ```

### Recomendaciones adicionales:
- Analiza las consultas más comunes que se ejecutan en tu base de datos antes de crear índices para asegurarte de que mejoren el rendimiento.
- No añadas demasiados índices, ya que afectarán el rendimiento de las inserciones y actualizaciones.

Para agregar las columnas `created_at` y `updated_at` a la tabla `orders_detail`, puedes usar la siguiente consulta en SQL para Postgres:

```sql
-- Agregar columnas created_at y updated_at a la tabla orders_detail
ALTER TABLE orders_detail
ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
```

### Explicación:
- **`created_at`**: Almacena la fecha y hora en que se creó la fila, con un valor predeterminado de la hora actual (`CURRENT_TIMESTAMP`).
- **`updated_at`**: Almacena la fecha y hora en que se actualizó la fila, también con un valor predeterminado de la hora actual.

Para que la columna `updated_at` se actualice automáticamente cada vez que se modifica un registro, podrías usar un `TRIGGER`. Aquí tienes un ejemplo de cómo hacerlo:

```sql
-- Crear función para actualizar updated_at en cambios
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Crear el trigger para orders_detail
CREATE TRIGGER update_orders_detail_timestamp
BEFORE UPDATE ON orders_detail
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();
```

Este `TRIGGER` asegura que cada vez que se actualice una fila en `orders_detail`, la columna `updated_at` se actualice automáticamente.

**Lecturas recomendadas**

[ia-data/SQL/DDL_SQL.pgsql at main · platzi/ia-data · GitHub](https://github.com/platzi/ia-data/blob/main/SQL/DDL_SQL.pgsql "ia-data/SQL/DDL_SQL.pgsql at main · platzi/ia-data · GitHub")

## Consultas de SQL con GitHub Copilot y ChatGPT

Tanto GitHub Copilot como ChatGPT pueden ser herramientas útiles para generar consultas SQL, pero lo hacen de diferentes maneras. Aquí te muestro cómo puedes usarlas en conjunto para obtener lo mejor de ambas:

### 1. **Uso de GitHub Copilot para SQL**:
GitHub Copilot funciona dentro de tu entorno de desarrollo (por ejemplo, VS Code) y sugiere código en tiempo real mientras escribes. Al trabajar con SQL, Copilot puede sugerir consultas basadas en el contexto de tu código.

#### Ejemplo:
Cuando estás creando una tabla o escribiendo una consulta, Copilot puede completar automáticamente las instrucciones SQL. Si estás en un archivo `.sql` y empiezas a escribir `SELECT`, Copilot puede sugerir algo como:

```sql
SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '30 days';
```

### Ventajas de GitHub Copilot:
- **Sugerencias rápidas y contextuales**: Copilot puede adivinar lo que necesitas en función de tu código actual.
- **Integración fluida**: Funciona dentro de tu IDE, lo que te permite trabajar sin interrupciones.
- **Autocompletado**: Sugiere consultas y estructuras de tablas basadas en tus necesidades.

### 2. **Uso de ChatGPT para SQL**:
ChatGPT es excelente cuando necesitas entender, refinar o generar consultas más complejas desde cero. También puede explicarte el propósito de una consulta o sugerirte optimizaciones. 

#### Ejemplo:
Si tienes un caso más específico, como querer optimizar una consulta SQL o crear una relación compleja entre varias tablas, puedes preguntarle a ChatGPT algo como:

```sql
-- ¿Cómo puedo optimizar esta consulta para que sea más rápida?

SELECT p.product_name, SUM(od.quantity) 
FROM orders o
JOIN orders_detail od ON o.id = od.order_id
JOIN product p ON od.product_id = p.id
GROUP BY p.product_name;
```

Y ChatGPT podría sugerirte cosas como:
- Crear índices en las columnas que se usan en las uniones (`JOIN`) o en las cláusulas `WHERE`.
- Agregar índices compuestos en las columnas más consultadas.
- Reestructurar la consulta para mejorar el rendimiento.

### Ventajas de ChatGPT:
- **Explicaciones y soporte**: ChatGPT puede explicarte por qué una consulta funciona de una manera determinada y cómo mejorarla.
- **Generación de consultas desde cero**: Puedes pedirle a ChatGPT que genere consultas o procedimientos almacenados sin escribir nada.
- **Optimización**: ChatGPT puede sugerir mejoras en rendimiento o estructura de datos.

### Ejemplo de uso combinado:
1. **Escribiendo SQL con Copilot**:
   - Empiezas escribiendo la estructura básica de la consulta o tabla.
   - GitHub Copilot sugiere automáticamente la sintaxis SQL.

2. **Refinando y optimizando con ChatGPT**:
   - Tomas la consulta generada por Copilot y la ingresas en ChatGPT para pedir optimizaciones, sugerencias de índices, o explicaciones.
   - ChatGPT te da una respuesta detallada para ajustar o mejorar la consulta.

### Resumen:
- **GitHub Copilot** es más rápido para generar código SQL en tiempo real dentro de tu IDE, especialmente útil para escribir consultas simples o estructuras básicas de tablas.
- **ChatGPT** te permite obtener explicaciones, sugerencias de optimización, y consultas más avanzadas que Copilot quizás no sugiere automáticamente.

Usar ambos en conjunto es una forma poderosa de mejorar tu flujo de trabajo con SQL.

**Lecturas recomendadas**

[ia-data/SQL/consultas_sql.pgsql at main · platzi/ia-data · GitHub](https://github.com/platzi/ia-data/blob/main/SQL/consultas_sql.pgsql "ia-data/SQL/consultas_sql.pgsql at main · platzi/ia-data · GitHub")
[ia-data/SQL/consultas_sql_2.pgsql at main · platzi/ia-data · GitHub](https://github.com/platzi/ia-data/blob/main/SQL/consultas_sql_2.pgsql "ia-data/SQL/consultas_sql_2.pgsql at main · platzi/ia-data · GitHub")
[ia-data/sources/sales.csv at main · platzi/ia-data · GitHub](https://github.com/platzi/ia-data/blob/main/sources/sales.csv "ia-data/sources/sales.csv at main · platzi/ia-data · GitHub")