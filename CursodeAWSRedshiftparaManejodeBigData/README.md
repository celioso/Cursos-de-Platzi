# Curso de AWS Redshift para Manejo de Big Data

## Objetivos y presentación del proyecto

**Amazon Redshift** es un servicio de almacenamiento de datos (data warehouse) en la nube provisto por Amazon Web Services (AWS). Está diseñado para analizar grandes volúmenes de datos de manera rápida y escalable. Es ideal para empresas que necesitan realizar análisis complejos y generar informes sobre datos almacenados.

### **Características principales de Amazon Redshift**
1. **Almacenamiento columnar**:
   - Redshift utiliza un modelo de almacenamiento columnar que optimiza las consultas analíticas, ya que permite leer solo las columnas relevantes en lugar de todas las filas.

2. **Escalabilidad**:
   - Redshift puede escalar horizontalmente añadiendo nodos al clúster, o verticalmente aumentando el tamaño de los nodos.

3. **Compatibilidad con SQL**:
   - Es compatible con SQL estándar, lo que facilita a los analistas y científicos de datos realizar consultas sin aprender un nuevo lenguaje.

4. **Integración con AWS**:
   - Se integra de forma nativa con servicios de AWS como S3, Kinesis, Glue, y QuickSight, facilitando la carga y análisis de datos.

5. **Compresión y particionamiento**:
   - Redshift aplica compresión y particionamiento automático de datos para mejorar el rendimiento y reducir costos.

6. **Distribución de datos**:
   - Usa estrategias como distribución por claves, rondas o todo nodo para optimizar la distribución de datos entre nodos.

### **Casos de uso comunes**
1. **Análisis empresarial**:
   - Generación de reportes, dashboards y visualizaciones con herramientas como Tableau, Power BI o Amazon QuickSight.

2. **Big Data Analytics**:
   - Procesamiento de grandes volúmenes de datos para identificar patrones, tendencias y obtener insights.

3. **Integración con sistemas ETL**:
   - Redshift se utiliza como destino de datos extraídos, transformados y cargados (ETL) desde diversas fuentes.

4. **Machine Learning**:
   - Preprocesamiento y almacenamiento de grandes conjuntos de datos para entrenar modelos ML.

### **Componentes de Amazon Redshift**

1. **Clúster**:
   - Es el entorno principal que contiene uno o más nodos donde se almacenan los datos y se procesan las consultas.

2. **Nodos**:
   - **Nodo líder**: Coordina las consultas y distribuye tareas a los nodos de cómputo.
   - **Nodos de cómputo**: Ejecutan consultas y almacenan los datos.

3. **Redshift Spectrum**:
   - Permite realizar consultas directamente en datos almacenados en S3 sin necesidad de cargarlos en Redshift.

4. **Conexión JDBC/ODBC**:
   - Redshift soporta conectores estándar para integrarse con herramientas de análisis y BI.

### **Arquitectura básica de Redshift**

1. **Carga de datos**:
   - Los datos se cargan desde diversas fuentes como:
     - Bases de datos relacionales (usando AWS DMS).
     - Archivos CSV, JSON, Parquet en S3.
     - Streams en tiempo real con Kinesis.

2. **Procesamiento**:
   - Redshift optimiza las consultas utilizando índices, almacenamiento columnar y estrategias de distribución.

3. **Consulta y análisis**:
   - Los usuarios acceden a los datos utilizando SQL o herramientas BI conectadas al clúster.

### **Ventajas de Amazon Redshift**

1. **Rendimiento**:
   - Diseño optimizado para consultas analíticas complejas y grandes volúmenes de datos.

2. **Costo-efectivo**:
   - Sistema de pago por uso con precios ajustados para almacenamiento y cómputo.

3. **Facilidad de uso**:
   - Configuración sencilla y escalabilidad sin interrupciones.

4. **Seguridad**:
   - Soporta cifrado de datos en reposo y en tránsito, integración con AWS IAM, y auditorías.

### **Ejemplo de uso con Python**

Amazon Redshift se integra fácilmente con Python usando bibliotecas como `psycopg2` o `SQLAlchemy`. A continuación, un ejemplo de conexión:

```python
import psycopg2

# Configuración de conexión
host = 'redshift-cluster.endpoint.amazonaws.com'
port = 5439
dbname = 'mydatabase'
user = 'myuser'
password = 'mypassword'

# Conexión a Redshift
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    print("Conexión exitosa")
    
    # Ejecutar consulta
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mi_tabla LIMIT 10;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
        
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Error en la conexión: {e}")
```

**Archivos de la clase**

[curso-de-redshift.pdf](https://static.platzi.com/media/public/uploads/curso-de-redshift_cdaf8dc8-3cdf-4e77-b1e5-a3390075746c.pdf)

**Lecturas recomendadas**

[Fundamentos de Bases de Datos](https://platzi.com/clases/bd/)

[Curso de SQL y MySQL | Platzi](https://platzi.com/clases/sql-mysql)

[Curso de Big Data](https://platzi.com/clases/big-data/)

[Curso de Fundamentos de AWS Cloud](https://platzi.com/clases/aws-cloud/)

[¿Cómo puedo empezar a usar AWS? en Curso de Fundamentos de AWS Cloud](https://platzi.com/clases/1323-aws-cloud/12574-como-puedo-empezar-a-usar-aws/)

## Aprende qué es un Data Warehouse

Un **Data Warehouse** (almacén de datos) es una base de datos especializada que centraliza y organiza información de múltiples sistemas o fuentes con el objetivo de apoyar la toma de decisiones empresariales. Está optimizado para consultas y análisis, en lugar de procesamiento de transacciones. 

### Características principales:
1. **Integración**: Combina datos de diversas fuentes (bases de datos transaccionales, archivos externos, sistemas ERP, CRM, etc.).
2. **Orientación a temas**: Los datos están organizados en torno a temas o áreas de interés, como ventas, clientes o finanzas.
3. **Consistencia temporal**: Los datos son históricos y se almacenan con marcadores de tiempo, permitiendo análisis a lo largo del tiempo.
4. **No volátil**: Los datos no se actualizan ni eliminan una vez almacenados, solo se agregan para conservar el historial.

### Componentes clave:
- **ETL (Extract, Transform, Load)**: Procesos que extraen datos de fuentes, los transforman según necesidades específicas y los cargan en el Data Warehouse.
- **Base de datos del Data Warehouse**: El repositorio central para almacenar datos organizados.
- **Herramientas de análisis y visualización**: Software que permite a los usuarios explorar los datos y generar reportes.

### Usos comunes:
- Generar informes ejecutivos y dashboards.
- Realizar análisis predictivo y minería de datos.
- Tomar decisiones estratégicas basadas en patrones históricos.

**Lecturas recomendadas**

[¿Qué es un Data Warehouse?](https://platzi.com/blog/ques-un-data-warehouse/)

[¿Qué es un ETL?](https://platzi.com/blog/que-es-un-etl/)

## Bases de datos columnares y arquitectura orientada a optimización de consultas

### Bases de Datos Columnares

Las bases de datos columnares están diseñadas para manejar eficientemente grandes volúmenes de datos en columnas en lugar de filas. A diferencia de las bases de datos tradicionales orientadas a filas, las columnares almacenan y procesan datos de forma optimizada por columnas, lo que permite:

- **Optimización en consultas analíticas**: Las bases de datos columnares están especialmente optimizadas para consultas analíticas, ya que permiten realizar operaciones sobre columnas específicas, lo que reduce la cantidad de datos leídos y procesados.
- **Eficiencia en almacenamiento**: Almacenamiento más eficiente debido a la compresión columnar, lo que resulta en un menor uso de disco y una mejor utilización de recursos.
- **Mejora del rendimiento en lecturas**: Son ideales para consultas de agregación, sumas, promedios, y otros cálculos que operan en grandes conjuntos de datos.

### Arquitectura Orientada a Optimización de Consultas

La arquitectura orientada a la optimización de consultas es un diseño de bases de datos que prioriza el rendimiento en la ejecución de consultas. Esta arquitectura incluye varios componentes clave:

1. **Indexación Avanzada**: Utilización de índices específicos para acelerar las búsquedas y filtrados.
2. **Técnicas de particionado**: División de grandes conjuntos de datos en particiones más pequeñas para acelerar consultas específicas.
3. **Caché**: Almacenamiento temporal de resultados para consultas repetidas, reduciendo el tiempo de acceso a los datos.
4. **Optimización de Consulta**: Técnicas como predicción de índices, agrupamiento de datos y ejecución paralela para mejorar el rendimiento.
5. **Compresión y almacenamiento eficiente**: Reducción del espacio ocupado por los datos mediante técnicas avanzadas de almacenamiento, tanto físicas como lógico-físicas.

En conjunto, estas características ayudan a optimizar el rendimiento de las bases de datos y a satisfacer demandas de procesamiento masivo de datos.

**Lecturas recomendadas**

[¿Qué es una base de datos columnar? – AWS](https://aws.amazon.com/es/nosql/columnar/)

[Características de Amazon Redshift - Almacén de datos en la nube - Amazon Web Services](https://aws.amazon.com/es/redshift/features/)

## ¿Cómo funciona AWS Redshift?

### Bases de Datos Columnares

Las bases de datos columnares están diseñadas para manejar eficientemente grandes volúmenes de datos en columnas en lugar de filas. A diferencia de las bases de datos tradicionales orientadas a filas, las columnares almacenan y procesan datos de forma optimizada por columnas, lo que permite:

- **Optimización en consultas analíticas**: Las bases de datos columnares están especialmente optimizadas para consultas analíticas, ya que permiten realizar operaciones sobre columnas específicas, lo que reduce la cantidad de datos leídos y procesados.
- **Eficiencia en almacenamiento**: Almacenamiento más eficiente debido a la compresión columnar, lo que resulta en un menor uso de disco y una mejor utilización de recursos.
- **Mejora del rendimiento en lecturas**: Son ideales para consultas de agregación, sumas, promedios, y otros cálculos que operan en grandes conjuntos de datos.

### Arquitectura Orientada a Optimización de Consultas

La arquitectura orientada a la optimización de consultas es un diseño de bases de datos que prioriza el rendimiento en la ejecución de consultas. Esta arquitectura incluye varios componentes clave:

1. **Indexación Avanzada**: Utilización de índices específicos para acelerar las búsquedas y filtrados.
2. **Técnicas de particionado**: División de grandes conjuntos de datos en particiones más pequeñas para acelerar consultas específicas.
3. **Caché**: Almacenamiento temporal de resultados para consultas repetidas, reduciendo el tiempo de acceso a los datos.
4. **Optimización de Consulta**: Técnicas como predicción de índices, agrupamiento de datos y ejecución paralela para mejorar el rendimiento.
5. **Compresión y almacenamiento eficiente**: Reducción del espacio ocupado por los datos mediante técnicas avanzadas de almacenamiento, tanto físicas como lógico-físicas.

En conjunto, estas características ayudan a optimizar el rendimiento de las bases de datos y a satisfacer demandas de procesamiento masivo de datos.

## Creando nuestro entorno de trabajo en AWS

Crear tu entorno de trabajo en AWS implica varios pasos para configurar los recursos necesarios para almacenar, procesar y analizar datos. Aquí tienes una guía básica para configurar un entorno en AWS:

### 1. **Crear una cuenta en AWS**
   - Ve al [sitio web de AWS](https://aws.amazon.com/) y crea una cuenta. Asegúrate de verificar tu identidad si es necesario.

### 2. **Configurar el acceso a AWS**
   - Configura las credenciales para acceder a AWS a través de la CLI o SDK. Puedes generar claves desde la consola de AWS (IAM -> Users -> Policies).

### 3. **Crear un Bucket de S3 para Almacenamiento de Datos**
   - S3 es un servicio de almacenamiento en la nube que puedes usar para guardar datos y acceder a ellos fácilmente.
   
   **Pasos**:
   - Navega a S3 en AWS.
   - Haz clic en "Crear Bucket".
   - Configura las opciones de almacenamiento, como el nombre del bucket, región, permisos y otras configuraciones como la región de datos y la clase de almacenamiento.

### 4. **Configurar una Base de Datos Redshift**
   - **AWS Redshift** se utiliza para el análisis de datos utilizando almacenamiento columnar.
   
   **Pasos**:
   - Navega a Redshift en AWS.
   - Crea un nuevo clúster de Redshift.
   - Define la configuración como tipo de nodo, número de nodos, tipo de almacenamiento, y configuraciones de seguridad.
   - Una vez creado, accede al clúster y carga datos desde S3 o fuentes externas.

### 5. **Instalar y Configurar AWS CLI**
   - La CLI de AWS permite la gestión y automatización de recursos a través de comandos en línea.
   
   **Pasos**:
   - Descarga e instala la CLI desde la [página oficial de AWS CLI](https://aws.amazon.com/cli/).
   - Configura tus credenciales (`aws configure`) para autenticarse en AWS desde la CLI.

### 6. **Automatización con AWS Lambda**
   - AWS Lambda permite ejecutar funciones en respuesta a eventos, como nuevos datos almacenados en S3 o la finalización de consultas en Redshift.
   
   **Pasos**:
   - Crea una nueva función Lambda.
   - Escribe tu código en Python o cualquier otro lenguaje compatible.
   - Configura triggers para ejecutar Lambda automáticamente en eventos específicos.

### 7. **Monitoreo y Seguridad**
   - Configura el monitoreo utilizando servicios como CloudWatch para supervisar el rendimiento de tus recursos.
   - Establece políticas de seguridad usando IAM para controlar quién tiene acceso a qué recursos.

### 8. **Optimización y Escalabilidad**
   - Ajusta las configuraciones de almacenamiento y procesamiento según tus necesidades.
   - Escala automáticamente los recursos usando Elastic Load Balancing o configuraciones manuales para aumentar o disminuir recursos.

### 9. **Documentación y Buenas Prácticas**
   - Mantén documentadas todas las configuraciones, políticas y configuraciones de seguridad.
   - Utiliza herramientas como Terraform o CloudFormation para automatizar la creación y gestión de infraestructura en AWS.

Con estos pasos básicos, puedes configurar un entorno en AWS que incluye almacenamiento, procesamiento de datos y análisis, creando una solución robusta para manejo de datos en la nube.

**Lecturas recomendadas**

[¿Cómo puedo empezar a usar AWS? en Curso de Fundamentos de AWS Cloud](https://platzi.com/clases/1323-aws-cloud/12574-como-puedo-empezar-a-usar-aws/)

[Curso de Fundamentos de AWS Cloud](https://platzi.com/clases/aws-cloud/)

[Curso de Cloud Computing con AWS](https://platzi.com/clases/aws-computo/)

## Configura tu primer cluster

Para configurar tu primer clúster en AWS Redshift, sigue estos pasos:

### 1. **Iniciar sesión en AWS**
   - Accede a la consola de AWS en [AWS Management Console](https://aws.amazon.com/console/).

### 2. **Navegar a AWS Redshift**
   - Ve a la sección de **Redshift** en la consola de AWS desde la barra de búsqueda o selecciona la categoría "Bases de datos" y luego selecciona **Redshift**.

### 3. **Crear un Nuevo Clúster Redshift**
   - Haz clic en **"Crear clúster"**.

### 4. **Configurar los Parámetros del Clúster**

   **Nombre del Clúster**: Define un nombre único para tu clúster.

   **Conjunto de Parámetros**: Elige entre diferentes conjuntos predefinidos o personaliza las configuraciones según las necesidades de rendimiento (ej., almacenamiento, nodo, tipo de nodo).

   **Tipo de Nodo**: Elige el tipo de nodo que deseas usar (Small, Medium, Large, etc.). Los nodos más grandes ofrecen mejor rendimiento, pero también mayor costo.

   **Número de Nodos**: Define cuántos nodos quieres utilizar. Por ejemplo, si estás comenzando, puedes comenzar con un solo nodo (Single Node) o un clúster de múltiples nodos si se requiere procesamiento distribuido.

   **Configuración de Almacenamiento**: Ajusta la cantidad de almacenamiento (GB) en función de tus necesidades. El almacenamiento se ajusta automáticamente según la cantidad de datos que procesas.

### 5. **Configuraciones Avanzadas**

   - **Conexión**: Configura el puerto, enrutamiento y red según tus necesidades de seguridad.
   - **Seguridad**: Añade grupos de seguridad para permitir acceso seguro desde otras aplicaciones o redes específicas.
   - **Seguridad de Datos**: Encriptación del clúster para asegurar los datos.

### 6. **Creación del Clúster**

   - Una vez configurado todo, haz clic en **"Crear Clúster"**. AWS comenzará a crear tu clúster Redshift.

### 7. **Conectar al Clúster**

   - Una vez creado, tendrás un endpoint para conectarte al clúster desde una herramienta como SQL client o desde aplicaciones que interactúan con bases de datos.

### 8. **Importar Datos**

   - Usa **Amazon S3** para cargar datos en tu clúster Redshift. También puedes conectarte a bases de datos externas o fuentes de datos a través de conexiones JDBC o otras.

### 9. **Monitoreo y Optimización**

   - Utiliza **CloudWatch** para monitorizar métricas del clúster.
   - Ajusta configuraciones conforme el uso y rendimiento del clúster.

Este proceso crea un entorno básico para trabajar con AWS Redshift.

**Lecturas recomendadas**

[Prueba gratuita de Amazon Redshift - Almacén de datos en la nube - Amazon Web Services](https://aws.amazon.com/es/redshift/free-trial/)

## Consumiendo Redshift: empieza la magia

Consumir datos desde un clúster de **AWS Redshift** implica interactuar con las tablas que has cargado y ejecutar consultas SQL optimizadas para obtener insights o alimentar aplicaciones downstream. Aquí te explico cómo empezar a consumir Redshift:

### **1. Configuración inicial**
Asegúrate de que:
- Tu clúster de Redshift está creado y funcionando.
- Los datos están cargados en Redshift (puedes usar Amazon S3 para cargar datos).
- Las herramientas de consulta están configuradas para acceder al clúster.

### **2. Conectar a Redshift**
Puedes conectarte al clúster de Redshift mediante:
- **Cliente SQL**: Herramientas como DBeaver, SQL Workbench/J o cualquier herramienta compatible con JDBC/ODBC.
- **AWS Query Editor**: Disponible directamente en la consola de AWS Redshift.
- **Librerías de programación**: Usando librerías como `boto3` (Python) o conectores JDBC/ODBC en lenguajes como Java o C#.

#### **Pasos para conectarte**
1. **Obtén el endpoint del clúster** desde la consola de AWS.
2. **Configura tu cliente SQL**:
   - Endpoint (ejemplo: `redshift-cluster-name.cluster-abc123xyz.us-west-2.redshift.amazonaws.com`).
   - Usuario y contraseña configurados al crear el clúster.
   - Puerto (por defecto: `5439`).
   - Nombre de la base de datos.
3. **Proporciona credenciales**:
   - Al usar herramientas externas, asegúrate de que las credenciales coincidan con las configuradas.

### **3. Escribe y ejecuta consultas SQL**

#### **Consulta básica**
```sql
SELECT * FROM employees LIMIT 10;
```
- Extrae los primeros 10 registros de la tabla `employees`.

#### **Filtrar datos**
```sql
SELECT name, department 
FROM employees 
WHERE department = 'Sales';
```
- Filtra empleados cuyo departamento sea "Ventas".

#### **Agrupar datos**
```sql
SELECT department, COUNT(*) AS num_employees 
FROM employees 
GROUP BY department;
```
- Cuenta el número de empleados por departamento.

#### **Optimizar con SORTKEY**
Si tienes una tabla configurada con una clave de ordenación (SORTKEY), aprovecha este diseño en tus consultas para mejorar el rendimiento.

### **4. Integración con herramientas**
Puedes consumir datos desde Redshift para alimentar dashboards, sistemas de análisis o aplicaciones usando:
- **Amazon QuickSight**: Visualiza datos directamente desde tu clúster Redshift.
- **ETL Tools**: Conecta Redshift con herramientas como Apache Airflow o Glue para mover datos entre sistemas.
- **Python (psycopg2)**:
  ```python
  import psycopg2

  conn = psycopg2.connect(
      dbname='your_dbname',
      host='your_endpoint',
      port='5439',
      user='your_username',
      password='your_password'
  )
  cur = conn.cursor()
  cur.execute("SELECT * FROM employees LIMIT 10;")
  rows = cur.fetchall()
  for row in rows:
      print(row)
  conn.close()
  ```

### **5. Monitorear y ajustar**
- Usa la consola de **Redshift Performance** para identificar consultas que consumen demasiados recursos.
- Aplica estrategias como la partición de tablas, claves de distribución (DISTKEY) y claves de ordenación (SORTKEY) para mejorar tiempos de consulta.

### **6. Automatización y despliegue**
- **Amazon EventBridge**: Programa la ejecución de consultas.
- **Airflow**: Automatiza la extracción, transformación y carga desde y hacia Redshift.

**Lecturas recomendadas**

[Curso de Fundamentos de AWS Cloud](https://platzi.com/clases/aws-cloud/)

[Installation - Dbeaver](https://dbeaver.com/docs/wiki/Installation/)

[GitHub - alarcon7a/redshift_course: Data repository of redshift course](https://github.com/alarcon7a/redshift_course)

## Sentencias SQL en Redshift

Amazon Redshift utiliza una variante de PostgreSQL para procesar consultas SQL. Puedes interactuar con Redshift escribiendo sentencias SQL para tareas como creación de tablas, manipulación de datos, y consultas. Aquí están algunos ejemplos comunes de sentencias SQL utilizadas en Redshift:

### 1. **Creación de tablas**
```sql
CREATE TABLE empleados (
    id_empleado INT IDENTITY(1, 1),
    nombre VARCHAR(50),
    apellido VARCHAR(50),
    fecha_ingreso DATE,
    salario DECIMAL(10, 2)
);
```

### 2. **Cargar datos desde S3**
Redshift permite cargar datos desde archivos almacenados en Amazon S3:
```sql
COPY empleados
FROM 's3://mi-bucket/datos/empleados.csv'
CREDENTIALS 'aws_access_key_id=TU_ACCESS_KEY;aws_secret_access_key=TU_SECRET_KEY'
CSV
IGNOREHEADER 1;
```

### 3. **Consultas**
- **Consulta básica**:
  ```sql
  SELECT nombre, apellido, salario
  FROM empleados
  WHERE salario > 50000;
  ```

- **Ordenar resultados**:
  ```sql
  SELECT * 
  FROM empleados
  ORDER BY fecha_ingreso DESC;
  ```

- **Agrupación y funciones de agregación**:
  ```sql
  SELECT fecha_ingreso, COUNT(*) AS total_empleados
  FROM empleados
  GROUP BY fecha_ingreso
  HAVING COUNT(*) > 1;
  ```

### 4. **Actualizar datos**
```sql
UPDATE empleados
SET salario = salario * 1.10
WHERE fecha_ingreso < '2020-01-01';
```

### 5. **Eliminar datos**
```sql
DELETE FROM empleados
WHERE fecha_ingreso < '2010-01-01';
```

### 6. **Uniones (JOIN)**
```sql
SELECT e.nombre, e.apellido, d.nombre AS departamento
FROM empleados e
JOIN departamentos d
ON e.id_departamento = d.id_departamento;
```

---

### 7. **Creación de vistas**
```sql
CREATE VIEW vista_empleados_activos AS
SELECT id_empleado, nombre, apellido
FROM empleados
WHERE estado = 'activo';
```

### 8. **Funciones analíticas**
```sql
SELECT nombre, apellido, salario,
       RANK() OVER (ORDER BY salario DESC) AS rank_salario
FROM empleados;
```

### 9. **Optimización de consultas**
Para optimizar tus consultas en Redshift:
- **Sort Keys**: Define claves de ordenamiento para optimizar consultas frecuentes.
  ```sql
  CREATE TABLE empleados (
      id_empleado INT,
      nombre VARCHAR(50),
      salario DECIMAL(10, 2)
  )
  SORTKEY (salario);
  ```
- **Dist Keys**: Define claves de distribución para optimizar operaciones que implican varias nodos.
  ```sql
  CREATE TABLE ventas (
      id_venta INT,
      id_producto INT,
      total DECIMAL(10, 2)
  )
  DISTKEY (id_producto);
  ```

**Lecturas recomendadas**

[Base de datos de muestra - Amazon Redshift](https://docs.aws.amazon.com/es_es/redshift/latest/dg/c_sampledb.html)

[redshift_course/Tickit_db at master · alarcon7a/redshift_course · GitHub](https://github.com/alarcon7a/redshift_course/tree/master/Tickit_db)

