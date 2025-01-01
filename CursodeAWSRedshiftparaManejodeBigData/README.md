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