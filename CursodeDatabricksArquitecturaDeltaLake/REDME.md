# Curso de Databricks Arquitectura Delta Lake

## Databricks como solución integral

**Databricks** es una plataforma de análisis de datos basada en la nube que permite a las empresas aprovechar el poder de **Apache Spark** para el procesamiento de datos en paralelo a gran escala. Diseñada para ser una solución integral, Databricks combina funcionalidades para la ingeniería de datos, la ciencia de datos, el aprendizaje automático (Machine Learning) y el análisis empresarial, todo dentro de un entorno colaborativo y gestionado. 

### **Características principales de Databricks como solución integral**

#### **1. Plataforma unificada**
Databricks integra diversas disciplinas de trabajo en torno a los datos:
- **Ingeniería de datos**: Con Apache Spark como núcleo, permite ETL eficiente, procesamiento de datos en tiempo real y manipulación de grandes volúmenes de datos.
- **Ciencia de datos**: Facilita el desarrollo y entrenamiento de modelos de Machine Learning con herramientas avanzadas.
- **Business Intelligence**: Compatible con herramientas como Power BI, Tableau, y generación de dashboards interactivos.

#### **2. Escalabilidad**
- Procesa grandes volúmenes de datos en paralelo, distribuidos entre múltiples nodos.
- Escalado automático que ajusta recursos según la carga de trabajo, optimizando costos.

#### **3. Integración con almacenamiento y datos**
- Compatible con **data lakes** como **Azure Data Lake**, **Amazon S3**, y **Google Cloud Storage**.
- Uso de **Delta Lake**, una capa de almacenamiento transaccional sobre un data lake, que combina:
  - Almacenamiento barato y escalable.
  - Garantías ACID (Atomicidad, Consistencia, Aislamiento, Durabilidad).

#### **4. Capacidades de colaboración**
- **Notebooks compartidos**: Los equipos pueden trabajar juntos en un entorno interactivo que admite Python, Scala, SQL y R.
- Funcionalidades de control de versiones y visualización integrada.

#### **5. Machine Learning y AI**
- Soporte para frameworks como **TensorFlow**, **PyTorch**, **XGBoost** y bibliotecas de Python como **scikit-learn**.
- Pipelines de Machine Learning integrados para automatizar el entrenamiento, validación y despliegue de modelos.

#### **6. Seguridad y gobernanza**
- Compatible con estándares de seguridad como GDPR, HIPAA y SOC 2.
- Administración de accesos basada en roles (RBAC).
- Auditoría y monitoreo de actividades.

#### **7. Multi-cloud y opciones gestionadas**
- Disponible en **AWS**, **Azure** y **Google Cloud**.
- Servicios gestionados que eliminan la necesidad de configurar y administrar clusters manualmente.

### **Ventajas de usar Databricks**
1. **Ahorro de costos**: Uso eficiente de recursos con escalado automático y pay-as-you-go.
2. **Mayor productividad**: Ambientes colaborativos que reducen la fricción entre equipos de ingeniería y análisis.
3. **Versatilidad**: Admite múltiples lenguajes de programación, motores de datos y herramientas externas.
4. **Integración con ecosistemas existentes**: Soporte para flujos de trabajo en la nube, herramientas de visualización y soluciones de terceros.

### **Casos de uso típicos**
- **Ingeniería de datos**: Ingesta y limpieza de datos a gran escala para crear un data lake.
- **Análisis de Big Data**: Generación de reportes interactivos y análisis exploratorio.
- **Machine Learning**: Desarrollo de modelos predictivos y experimentación en tiempo real.
- **IoT y procesamiento en tiempo real**: Ingesta de flujos de datos en tiempo real para análisis instantáneos.

Databricks se posiciona como una herramienta integral para empresas que buscan democratizar el acceso a los datos y facilitar la colaboración entre equipos diversos.

## ¿Qué es Databricks y para qué sirve?

**Databricks** es una plataforma de análisis de datos basada en la nube que combina procesamiento de datos, ciencia de datos, Machine Learning (ML) y análisis empresarial en un entorno colaborativo y escalable. Fue desarrollada inicialmente por los creadores de Apache Spark, por lo que se basa en esta tecnología para ofrecer procesamiento en paralelo y procesamiento distribuido.

### **¿Qué es Databricks?**

Databricks proporciona una solución completa para manejar datos a gran escala, desde la ingesta de datos hasta el análisis avanzado y la generación de modelos de Machine Learning. Está diseñado para ser accesible y fácil de usar tanto para científicos de datos como para ingenieros de datos.

### **Funciones principales de Databricks**

1. **Procesamiento de Datos a Escala**
   - Soporta procesamiento distribuido usando Apache Spark, lo que permite manejar grandes volúmenes de datos de manera eficiente.
   - Ideal para tareas como ETL (Extract, Transform, Load), análisis de datos y Machine Learning.

2. **Ambiente Colaborativo**
   - Integra notebooks colaborativos donde diferentes equipos pueden trabajar juntos utilizando Python, Scala, SQL, R y otros lenguajes.
   - Facilita el desarrollo ágil con integración en tiempo real, análisis interactivo y colaboración en equipo.

3. **Integración con Servicios de Nube**
   - Compatible con plataformas como **AWS**, **Azure** y **Google Cloud**.
   - Permite la integración con almacenes de datos como **S3**, **Azure Data Lake**, y otras soluciones de almacenamiento de datos.

4. **Machine Learning y Ciencia de Datos**
   - Facilita el desarrollo de modelos de Machine Learning mediante pipelines automatizados, entrenamiento de modelos y despliegue de modelos.
   - Soporte para frameworks de Machine Learning como **TensorFlow**, **PyTorch**, **scikit-learn**, y más.

5. **Governanza y Seguridad**
   - Cumple con estándares de seguridad como **GDPR**, **HIPAA**, y SOC 2.
   - Gestión basada en roles (RBAC) y auditoría integrada para el control de acceso.

6. **Análisis y Business Intelligence**
   - Permite crear dashboards interactivos y generar insights a través de herramientas avanzadas de visualización y análisis.

### **¿Para qué sirve Databricks?**

Databricks es útil para una amplia variedad de casos de uso, incluyendo:

- **Análisis de Big Data**: Procesamiento y análisis de grandes volúmenes de datos de manera eficiente.
- **Ciencia de Datos**: Exploración de datos, modelado predictivo y descubrimiento de patrones mediante técnicas avanzadas.
- **Machine Learning**: Automatización del ciclo de vida del Machine Learning, desde el desarrollo hasta la implementación.
- **IoT y procesamiento en tiempo real**: Gestión y análisis de datos provenientes de dispositivos IoT en tiempo real.
- **Data Warehousing y ETL**: Simplificación de procesos ETL y gestión de data lakes.
- **Integración y colaboración**: Facilita la colaboración entre equipos técnicos y de negocio en proyectos de datos.

### **Beneficios de Databricks**

- **Escalabilidad**: Manejo eficiente de grandes volúmenes de datos utilizando múltiples nodos.
- **Productividad**: Interfaz intuitiva y soporte para múltiples lenguajes permiten una fácil adopción y flexibilidad.
- **Costo-eficiencia**: Optimización automática de recursos para reducir costos al mínimo.
- **Innovación**: Proporciona acceso a las últimas tecnologías y marcos de datos.

### **Casos de uso específicos**

- **Industria Financiera**: Modelos de predicción de riesgos crediticios y análisis financiero.
- **Retail**: Optimización de inventarios, análisis de tendencias de consumo y personalización del cliente.
- **Salud**: Procesamiento de datos clínicos y análisis epidemiológico.

**Lecturas recomendadas**

[Databricks: Qué es y características | OpenWebinars](https://openwebinars.net/blog/databricks-que-es-y-caracteristicas/)

[Sign up for Databricks Community edition | Databricks on AWS](https://docs.databricks.com/en/getting-started/community-edition.html)

[Try Databricks free](https://www.databricks.com/try-databricks?scid=7018Y000001Fi0QQAS&utm_medium=programmatic&utm_source=google&utm_campaign=21398313547&utm_adgroup=&utm_content=trial&utm_offer=try-databricks&utm_ad=&utm_term=&gad_source=1&gclid=CjwKCAiA-Oi7BhA1EiwA2rIu22pH7dB663MtraS0aGCxZdVos5Jdyw13aG6gv7I6ZCtutw5p-O0qTxoCjhgQAvD_BwE#account)

[Lectura - Creacion de cuenta Databricks.pdf - Google Drive](https://drive.google.com/file/d/1wwWF8Xki7NFTyRszhkUDHpzbbO0Xj1Bt/view?usp=sharing)

## Infraestructura de almacenamiento y procesamiento en Databricks

Databricks es una plataforma basada en la nube que proporciona un entorno unificado para análisis de datos, ciencia de datos, Machine Learning (ML) y procesamiento en tiempo real. La infraestructura en Databricks se basa en el almacenamiento y procesamiento distribuidos utilizando tecnología como Apache Spark.

### **Infraestructura de almacenamiento en Databricks**

1. **Almacenamiento en Databricks**:
   - Databricks utiliza diferentes servicios de almacenamiento integrados para manejar datos, como:
     - **Databricks File System (DBFS)**: Un sistema de archivos distribuido basado en la nube que permite a los usuarios almacenar, leer y escribir archivos en el entorno de Databricks. Es una capa de almacenamiento temporal y persistente para el trabajo en notebooks.
     - **Amazon S3**: Integración con sistemas de almacenamiento en la nube como S3 para el manejo de datos a gran escala. Databricks permite trabajar directamente con archivos alojados en S3, facilitando la ingestión y procesamiento de datos.
     - **Azure Blob Storage**: Similar a S3, proporciona almacenamiento de objetos en Azure.
     - **Google Cloud Storage**: Integración con el almacenamiento en la nube de Google para el manejo de grandes volúmenes de datos.

2. **Tipos de Datos**:
   - Datos estructurados (tablas, CSV, JSON).
   - Datos semi-estructurados (archivos Avro, Parquet, ORC).
   - Datos no estructurados (imágenes, videos, logs).

### **Infraestructura de procesamiento en Databricks**

1. **Apache Spark**:
   - Databricks está construido sobre Apache Spark, que permite el procesamiento distribuido en paralelo. Esto permite ejecutar tareas intensivas en recursos como procesamiento de datos masivos, Machine Learning y análisis en tiempo real.
   - Las capacidades de procesamiento incluyen:
     - **Transformaciones masivas de datos**: Operaciones como filtrado, agrupación, unión, sumas parciales, etc.
     - **Modelos de Machine Learning**: Entrenamiento de modelos en paralelo utilizando Spark MLlib.
     - **Procesamiento de Streams**: Procesamiento de datos en tiempo real utilizando Spark Streaming.

2. **Niveles de procesamiento**:
   - **Computación general**: Un entorno para tareas analíticas estándar y procesamiento de datos.
   - **Clusters optimizados**: Clusters con configuraciones específicas para Machine Learning (GPU, CPU optimizados), procesamiento de datos a gran escala y rendimiento máximo.

3. **Tareas Distribuidas**:
   - En Databricks, las tareas se dividen en múltiples trabajos y ejecutan operaciones de manera simultánea en distintos nodos, reduciendo tiempos de procesamiento.

### **Capa de Gestión y Orquestación**

- Databricks proporciona una orquestación integrada para la gestión de flujos de trabajo y pipelines de datos.
- Soporte para tareas agendadas y automatización de tareas a través de Apache Airflow o Delta Live Tables para flujos de datos en tiempo real.

### **Seguridad y Escalabilidad**

- **Seguridad**: Databricks ofrece integración con servicios de seguridad en la nube como IAM (Identity and Access Management), SSO (Single Sign-On), y encriptación de datos en tránsito y reposo.
- **Escalabilidad**: Los clústeres en Databricks pueden escalar horizontalmente o verticalmente según las necesidades de procesamiento y almacenamiento.

### **Beneficios de la Infraestructura en Databricks**

- Escalabilidad masiva.
- Procesamiento distribuido optimizado.
- Integración con servicios de almacenamiento en nube líderes.
- Seguridad y gobernanza avanzada.

## Spark como motor de procesamiento Big Data

Apache Spark es un motor de procesamiento Big Data diseñado para manejar grandes volúmenes de datos de manera rápida y eficiente. A continuación te explico cómo funciona Spark y sus principales características:

### Características principales de Spark:

1. **Rendimiento rápido**: Spark puede procesar datos en memoria, lo que significa que las operaciones suelen ser más rápidas que en soluciones basadas solo en discos.

2. **Procesamiento en paralelo**: Spark permite distribuir las tareas entre múltiples nodos en un clúster para procesar datos de manera distribuida.

3. **Modelos de trabajo**: Spark admite diversos modelos de procesamiento, incluyendo:
   - **Batch**: Procesamiento por lotes de grandes volúmenes de datos.
   - **Streaming**: Procesamiento de datos en tiempo real provenientes de fuentes como redes sociales, sensores, o logs.
   - **Machine Learning**: Modelos de aprendizaje automático para análisis predictivo.
   - **SQL y DataFrames**: Acceso directo a datos estructurados mediante consultas SQL o manipulación de DataFrames.

4. **Interoperabilidad**: Spark puede integrarse con otros sistemas como Hadoop, Hive, Cassandra, HBase, y otros frameworks comunes en Big Data.

5. **Flexibilidad**: Soporta múltiples lenguajes de programación como Scala, Java, Python, R y SQL, lo que facilita el desarrollo para diversos usuarios.

6. **Resiliencia**: Spark proporciona mecanismos para manejar errores y recuperarse de fallos en los nodos, asegurando que los datos procesados no se pierdan.

7. **Optimización**: Tiene optimizaciones integradas como reescritura de consultas, ejecución optimizada en memoria, y otras técnicas para maximizar el rendimiento.

### Usos comunes de Spark:

- **Análisis de datos**: Procesamiento y análisis de grandes volúmenes de datos.
- **Machine Learning**: Implementación y entrenamiento de modelos de machine learning.
- **Procesamiento de datos en tiempo real**: Integración de datos en tiempo real con Apache Kafka, Flink, o similar.
- **Visualización**: Generación de reportes y análisis en tiempo real con gráficos interactivos.

Spark es ampliamente utilizado en grandes empresas para soluciones de Big Data, dado su alto rendimiento y capacidad para manejar múltiples tipos de datos y tareas de procesamiento.

**Lecturas recomendadas**

[Hadoop y Spark: diferencia entre los marcos de Apache. AWS](https://aws.amazon.com/es/compare/the-difference-between-hadoop-vs-spark/#:~:text=Apache%20Hadoop%20permite%20agrupar%20varios,en%20datos%20de%20cualquier%20tama%C3%B1o)

[Lectura - Repaso de Arquitecturas basicas de Big DATA (3).pdf - Google Drive](https://drive.google.com/file/d/1-e43pi_2NJ-yyTBsNiiVf1JZ_DXO5U7q/view?usp=sharing)

[¿Qué es Apache Spark?](https://cloud.google.com/learn/what-is-apache-spark?hl=es)

## Preparación de cluster de procesamiento

La preparación de un cluster de procesamiento, como en Apache Spark o Databricks, implica varios pasos para configurarlo, optimizarlo y garantizar su buen funcionamiento. A continuación se describen algunos aspectos clave:

### 1. **Configuración Inicial del Cluster**:
   - **Elección del Tipo de Cluster**: Determinar el tamaño del cluster (número de nodos), tipo de máquina (tamaño del nodo) y recursos necesarios como CPU, memoria y almacenamiento.
   - **Configuración de Recursos**: Asegurarse de asignar recursos adecuados según el tipo de tareas a realizar (computación intensiva, procesamiento de datos, análisis, etc.).

### 2. **Instalación y Configuración de Software**:
   - **Instalación de Spark o Databricks**: Instalación de Apache Spark o Databricks en los nodos del cluster.
   - **Configuración de Clusters**: Definir configuraciones como cantidad de núcleos, memoria, almacenamiento temporal, etc.

### 3. **Optimización del Rendimiento**:
   - **Distribución de Datos**: Distribuir los datos equitativamente entre los nodos para evitar cuellos de botella y mejorar el rendimiento.
   - **Compresión de Datos**: Usar técnicas de compresión adecuadas (Run-length, Delta, etc.) para reducir el uso de espacio y mejorar el procesamiento.
   - **Configuración de Particiones y Slices**: Configurar adecuadamente las particiones y slices para asegurar un procesamiento eficiente.

### 4. **Seguridad y Accesos**:
   - **Credenciales y Permisos**: Configurar adecuadamente credenciales y permisos de acceso para gestionar la seguridad en el cluster.

### 5. **Monitoreo y Mantenimiento**:
   - **Monitoreo de Rendimiento**: Implementar herramientas de monitoreo para seguir el rendimiento del cluster y tomar acciones correctivas si es necesario.
   - **Actualizaciones y Escalabilidad**: Asegurar que el cluster pueda escalar y adaptarse a cargas de trabajo cambiantes.

### 6. **Optimización de Consultas**:
   - **Optimización de Consultas**: Utilizar herramientas como "EXPLAIN" o "Analyze" para entender cómo se ejecutan las consultas y optimizarlas.

Estos pasos son fundamentales para garantizar un rendimiento eficiente y efectivo en la ejecución de tareas de procesamiento de datos en un cluster.

## Preparación de cluster de almacenamiento

La preparación de un cluster de almacenamiento implica configurar un entorno para gestionar, distribuir y almacenar datos de manera eficiente en un sistema distribuido. Aquí están los pasos clave para preparar un cluster de almacenamiento:

### **1. Elección del Sistema de Almacenamiento**
   - **Sistema Distribuido**: Seleccionar una solución como HDFS (Hadoop Distributed File System), Amazon S3, Azure Data Lake, o Databricks File System (DBFS).
   - **Tipo de Datos**: Considerar los tipos de datos a almacenar (estructurados, semi-estructurados o no estructurados).
   - **Escalabilidad y Redundancia**: Evaluar la capacidad de escalar y garantizar la disponibilidad mediante la replicación de datos.

### **2. Configuración de Nodos de Almacenamiento**
   - **Tipos de Nodos**:
     - *Master Node*: Gestiona la metadata del almacenamiento (nombres de archivos, ubicaciones de bloques, etc.).
     - *Worker Nodes*: Almacenan los bloques de datos y los procesan según sea necesario.
   - **Asignación de Recursos**:
     - Espacio en disco.
     - Memoria para gestionar operaciones locales de almacenamiento.

### **3. Configuración del Almacenamiento Distribuido**
   - **Replicación de Datos**: Configurar niveles de replicación para garantizar la redundancia y alta disponibilidad.
   - **Tamaño de Bloques**: Determinar el tamaño óptimo de bloques (por ejemplo, 128 MB o 256 MB en HDFS) para optimizar la lectura y escritura.
   - **Distribución de Datos**: Establecer políticas para distribuir datos uniformemente entre los nodos, evitando sobrecargas.

### **4. Seguridad y Control de Acceso**
   - **Encriptación**: Habilitar encriptación para datos en tránsito y en reposo.
   - **Autenticación y Autorización**:
     - Configurar servicios como Kerberos (para HDFS) o IAM roles (para S3).
     - Definir permisos granulares para usuarios y aplicaciones.

### **5. Integración con Procesamiento**
   - **Conexión con Motores de Procesamiento**:
     - Integrar el cluster de almacenamiento con herramientas como Spark, Hive, o Databricks.
   - **Optimización para Consultas**:
     - Usar formatos de datos optimizados como Parquet, ORC o Avro.
     - Crear particiones para mejorar el rendimiento de consultas.

### **6. Supervisión y Mantenimiento**
   - **Monitoreo de Salud**: Implementar herramientas para supervisar el uso del disco, estado de los nodos y replicación de datos (como Ambari para HDFS o CloudWatch para S3).
   - **Mantenimiento Preventivo**:
     - Reequilibrar datos entre nodos si es necesario.
     - Asegurar que los nodos tengan espacio suficiente para nuevas cargas.

### **7. Escalabilidad y Optimización**
   - **Escalado Horizontal**: Agregar nodos adicionales según aumenten las necesidades de almacenamiento.
   - **Compresión de Datos**: Aplicar técnicas de compresión (como Snappy o Gzip) para ahorrar espacio y optimizar el rendimiento.

Estos pasos aseguran que un cluster de almacenamiento esté preparado para manejar grandes volúmenes de datos, brindar alta disponibilidad y funcionar de manera eficiente junto con sistemas de procesamiento.

**Lecturas recomendadas**

[persona.data - Google Drive](https://drive.google.com/file/d/1X0Cy_v_ayQZjtz-JaWx4eYyb30DOSvH-/view?usp=sharing)

[2015-summary.csv - Google Drive](https://drive.google.com/file/d/1f-At0wsEkn3qHGRyoXLYFb1CVyHAIi_b/view?usp=sharing)

[transacciones.json - Google Drive](https://drive.google.com/file/d/1Jp-AgClWeD2ypLueNfHMuLQIUIE3MVGi/view?usp=sharing)

[transacciones.xml - Google Drive](https://drive.google.com/file/d/10atuc7Ke_to5NMfcudT2LCjhbk1Tgf3E/view?usp=sharing)

## ¿Qué son las transformaciones y acciones en Spark?

En Apache Spark, las **transformaciones** y **acciones** son los dos tipos principales de operaciones que se pueden realizar sobre un conjunto de datos distribuido (RDD, DataFrame o Dataset). Estas operaciones son fundamentales para realizar análisis y procesamiento de datos. 

---

### **Transformaciones**
- **Definición**: Operaciones que crean un nuevo conjunto de datos a partir de uno existente, sin ejecutarse de inmediato. Son **perezosas** (lazy), lo que significa que Spark no las evalúa hasta que se llama a una acción.
- **Propósito**: Especificar cómo transformar los datos, pero no realizar la transformación hasta que sea necesario.
- **Ejemplos**:
  1. **`map(función)`**: Aplica una función a cada elemento del conjunto de datos y devuelve un nuevo RDD o DataFrame.
  2. **`filter(función)`**: Devuelve un nuevo conjunto de datos que contiene solo los elementos que cumplen con una condición.
  3. **`flatMap(función)`**: Similar a `map`, pero puede devolver múltiples valores por elemento (aplana los resultados).
  4. **`groupByKey()`**: Agrupa los datos por clave.
  5. **`reduceByKey(función)`**: Combina los valores de cada clave usando una función.
  6. **`distinct()`**: Elimina duplicados del conjunto de datos.
  7. **`join(otherDataset)`**: Une dos conjuntos de datos.

- **Resultado**: Un nuevo conjunto de datos (RDD, DataFrame o Dataset).

---

### **Acciones**
- **Definición**: Operaciones que devuelven un resultado final o un efecto secundario (como guardar datos). Son **evaluadas inmediatamente** y desencadenan la ejecución de las transformaciones acumuladas.
- **Propósito**: Recuperar datos o realizar tareas concretas, como almacenar resultados o calcular estadísticas.
- **Ejemplos**:
  1. **`collect()`**: Devuelve todos los elementos como una colección al controlador (driver).
  2. **`count()`**: Devuelve el número total de elementos en el conjunto de datos.
  3. **`take(n)`**: Recupera los primeros `n` elementos.
  4. **`saveAsTextFile(path)`**: Guarda los datos en un archivo de texto en el sistema de archivos.
  5. **`reduce(función)`**: Aplica una función para combinar los elementos y devolver un único valor.
  6. **`foreach(función)`**: Aplica una función a cada elemento del conjunto de datos (no devuelve nada).

---

### **Diferencias Clave**
| Aspecto              | Transformaciones                  | Acciones                          |
|-----------------------|------------------------------------|------------------------------------|
| **Evaluación**        | Perezosa (lazy).                 | Inmediata.                        |
| **Resultado**         | Nuevo conjunto de datos.         | Resultado final o efecto secundario. |
| **Propósito**         | Definir la lógica del cálculo.    | Obtener o guardar datos.          |
| **Ejemplos comunes**  | `map`, `filter`, `reduceByKey`.   | `collect`, `count`, `saveAsTextFile`. |

---

### **Cómo funcionan juntas**
- Las transformaciones se encadenan para definir un flujo de procesamiento.
- Una acción desencadena la ejecución del flujo completo.
  
**Ejemplo:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TransformacionesYAcciones").getOrCreate()

# Crear un RDD
data = [1, 2, 3, 4, 5]
rdd = spark.sparkContext.parallelize(data)

# Transformaciones (no ejecutadas aún)
rdd_filtered = rdd.filter(lambda x: x % 2 == 0)  # Filtrar números pares
rdd_squared = rdd_filtered.map(lambda x: x**2)  # Elevar al cuadrado

# Acción (se ejecuta todo el flujo)
result = rdd_squared.collect()
print(result)  # Output: [4, 16]
```

En este ejemplo:
- Las transformaciones (`filter` y `map`) definen cómo procesar los datos.
- La acción (`collect`) desencadena la ejecución y devuelve el resultado.

**Lecturas recomendadas**

[¿Cuáles son las Transformaciones y Acciones en Spark?](https://keepcoding.io/blog/transformaciones-y-acciones-en-spark/)

