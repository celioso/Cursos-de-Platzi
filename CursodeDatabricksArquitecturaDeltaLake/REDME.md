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

## ¿Qué son los RDD en Apache Spark?

### **Resilient Distributed Dataset (RDD)** en Apache Spark

Los **RDD** son la unidad básica de datos en Apache Spark y representan un conjunto distribuido, inmutable y tolerante a fallos de datos que puede procesarse en paralelo. Fueron la primera API de abstracción de datos introducida en Spark y se utilizan para realizar cálculos distribuidos de manera eficiente.

---

### **Características principales de los RDD**
1. **Inmutabilidad**: Una vez creado, un RDD no puede modificarse, pero puede derivarse uno nuevo aplicando transformaciones.
2. **Distribución**: Los datos están divididos en particiones que se distribuyen entre los nodos del clúster para su procesamiento paralelo.
3. **Tolerancia a fallos**: Spark registra las operaciones realizadas sobre los datos (línea de tiempo de transformación) y puede reconstruir las particiones perdidas en caso de fallos.
4. **Evaluación perezosa**: Las transformaciones sobre un RDD no se ejecutan de inmediato, sino hasta que se realiza una acción.
5. **Operaciones de alto nivel**: Soportan operaciones como `map`, `filter`, `reduce`, y más, lo que permite construir complejos flujos de datos de forma sencilla.

---

### **Cómo se crean los RDD**
1. **Desde datos existentes**:
   - Desde un archivo (como texto, CSV, etc.).
   - Desde una colección en el programa principal.
   - Ejemplo:
     ```python
     rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
     ```
     ```python
     rdd = spark.sparkContext.textFile("ruta/al/archivo.txt")
     ```

2. **A partir de transformaciones**:
   - Aplicando transformaciones (como `map` o `filter`) a un RDD existente.

3. **Generado a partir de datos externos**:
   - Desde bases de datos, sistemas de almacenamiento como HDFS, S3, etc.

---

### **Operaciones en RDD**
Las operaciones sobre los RDD se dividen en dos categorías:

1. **Transformaciones**:
   - Crean un nuevo RDD a partir de otro.
   - Ejemplos:
     - `map(función)`: Aplica una función a cada elemento.
     - `filter(función)`: Filtra elementos que cumplen una condición.
     - `flatMap(función)`: Aplica una función y aplana los resultados.
     - `union()`: Combina dos RDD.
     - `reduceByKey(función)`: Combina valores con la misma clave.
   - **Evaluación**: Perezosa (no se ejecutan hasta que se llame a una acción).

2. **Acciones**:
   - Ejecutan las transformaciones y devuelven un resultado.
   - Ejemplos:
     - `collect()`: Recupera todos los elementos.
     - `count()`: Cuenta los elementos.
     - `take(n)`: Recupera los primeros `n` elementos.
     - `saveAsTextFile(path)`: Guarda los datos en un archivo de texto.

---

### **Ventajas de los RDD**
1. **Procesamiento paralelo**: Los datos se dividen en particiones para procesarse simultáneamente.
2. **Tolerancia a fallos**: Spark puede reconstruir datos automáticamente a partir de la secuencia de transformaciones.
3. **Flexibilidad**: Los RDD admiten varios tipos de operaciones y datos.
4. **Integración con Hadoop**: Pueden usar HDFS, HBase y otras fuentes de datos.

---

### **Limitaciones de los RDD**
1. **Complejidad**: La API de RDD requiere escribir más código para tareas comunes, comparado con APIs más modernas como DataFrames y Datasets.
2. **Optimización limitada**: Los RDD no aprovechan las optimizaciones automáticas de Spark SQL y Catalyst.
3. **Eficiencia**: Operaciones como agrupamientos o filtrados pueden ser menos eficientes que las realizadas con DataFrames o Datasets.

---

### **Ejemplo de uso de RDD**
```python
from pyspark import SparkContext

# Crear un contexto de Spark
sc = SparkContext("local", "EjemploRDD")

# Crear un RDD desde una lista
datos = [1, 2, 3, 4, 5]
rdd = sc.parallelize(datos)

# Aplicar transformaciones
rdd_filtrado = rdd.filter(lambda x: x % 2 == 0)  # Filtrar números pares
rdd_cuadrado = rdd_filtrado.map(lambda x: x**2)  # Elevar al cuadrado

# Ejecutar una acción
resultado = rdd_cuadrado.collect()
print(resultado)  # Salida: [4, 16]

# Detener el contexto
sc.stop()
```

En este ejemplo:
1. Se crea un RDD desde una lista.
2. Se aplican transformaciones (`filter` y `map`).
3. Se ejecuta una acción (`collect`) para obtener el resultado.

**Lecturas recomendadas**

[¿Qué es RDD (Resilient Distributed Datasets)?](https://keepcoding.io/blog/rdd-resilient-distributed-datasets/)

## Apache Spark: acciones

En Apache Spark, las **acciones** son operaciones que ejecutan el flujo de transformaciones definido sobre un RDD, DataFrame o Dataset, devolviendo un resultado al controlador (driver) o guardando los datos en almacenamiento externo. 

A diferencia de las transformaciones, que son **evaluadas de forma perezosa**, las acciones **desencadenan la ejecución** de todas las transformaciones previas en el pipeline.

---

### **Características de las Acciones**
1. **Inicia el cálculo**: Producen un resultado final o guardan datos en un sistema externo.
2. **Devuelve un valor**: El resultado puede ser un valor al controlador, un conteo, una lista de elementos o datos almacenados.
3. **Forzan la evaluación**: Ejecutan todas las transformaciones acumuladas hasta ese punto.
4. **Consume recursos del clúster**: Las acciones generan cargas en los nodos al procesar los datos.

---

### **Principales Acciones en Spark**
A continuación, se describen las acciones más comunes, junto con ejemplos en Python y Scala.

#### 1. **`collect()`**
   - Recupera todos los elementos del RDD o DataFrame y los devuelve al controlador como una lista.
   - **Uso**:
     - Ideal para conjuntos de datos pequeños, ya que todos los datos deben caber en la memoria del controlador.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([1, 2, 3, 4, 5])
     resultado = rdd.collect()
     print(resultado)  # Salida: [1, 2, 3, 4, 5]
     ```

#### 2. **`count()`**
   - Cuenta el número total de elementos en un RDD o DataFrame.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([1, 2, 3, 4, 5])
     total = rdd.count()
     print(total)  # Salida: 5
     ```

#### 3. **`take(n)`**
   - Recupera los primeros `n` elementos del RDD o DataFrame.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([10, 20, 30, 40, 50])
     primeros = rdd.take(3)
     print(primeros)  # Salida: [10, 20, 30]
     ```

#### 4. **`reduce(función)`**
   - Aplica una función de reducción a los elementos del RDD o DataFrame y devuelve un único valor.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([1, 2, 3, 4])
     suma = rdd.reduce(lambda x, y: x + y)
     print(suma)  # Salida: 10
     ```

#### 5. **`first()`**
   - Devuelve el primer elemento del RDD o DataFrame.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([100, 200, 300])
     primero = rdd.first()
     print(primero)  # Salida: 100
     ```

#### 6. **`saveAsTextFile(path)`**
   - Guarda el RDD como un archivo de texto en la ruta especificada. Cada partición se almacena como un archivo separado.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize(["Hola", "Mundo", "Spark"])
     rdd.saveAsTextFile("ruta/salida.txt")
     ```

#### 7. **`saveAsSequenceFile(path)`**
   - Guarda los datos como un archivo de secuencia (usado para claves y valores).
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([(1, "A"), (2, "B"), (3, "C")])
     rdd.saveAsSequenceFile("ruta/salida-secuencia")
     ```

#### 8. **`countByValue()`**
   - Devuelve un mapa con la frecuencia de cada elemento en el RDD.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([1, 2, 2, 3, 3, 3])
     frecuencias = rdd.countByValue()
     print(frecuencias)  # Salida: {1: 1, 2: 2, 3: 3}
     ```

#### 9. **`foreach(función)`**
   - Aplica una función a cada elemento del RDD sin devolver un resultado al controlador.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([1, 2, 3])
     rdd.foreach(lambda x: print(x))  # Imprime cada elemento
     ```

#### 10. **`takeSample(withReplacement, num, seed=None)`**
   - Devuelve una muestra de elementos del RDD.
   - **Parámetros**:
     - `withReplacement`: Indica si los elementos pueden repetirse en la muestra.
     - `num`: Tamaño de la muestra.
     - `seed`: Semilla opcional para reproducibilidad.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize([1, 2, 3, 4, 5])
     muestra = rdd.takeSample(False, 3, seed=42)
     print(muestra)  # Salida: [1, 4, 5]
     ```

#### 11. **`saveAsObjectFile(path)`**
   - Serializa el RDD y lo guarda como un archivo binario en la ruta especificada.
   - **Ejemplo**:
     ```python
     rdd = sc.parallelize(["a", "b", "c"])
     rdd.saveAsObjectFile("ruta/archivo_objeto")
     ```

---

### **Consideraciones importantes**
- Usar acciones como `collect()` o `take()` con datos grandes puede saturar la memoria del controlador. En esos casos, es mejor usar acciones que guarden los datos en almacenamiento externo.
- Las acciones desencadenan todas las transformaciones pendientes, por lo que es esencial diseñar flujos eficientes.

--- 

### **Ejemplo de flujo completo**
```python
from pyspark import SparkContext

# Crear un contexto de Spark
sc = SparkContext("local", "AccionesEjemplo")

# Crear un RDD
rdd = sc.parallelize([1, 2, 3, 4, 5, 6])

# Transformaciones
rdd_pares = rdd.filter(lambda x: x % 2 == 0)
rdd_cuadrados = rdd_pares.map(lambda x: x**2)

# Acción
resultado = rdd_cuadrados.collect()
print(resultado)  # Salida: [4, 16, 36]

# Detener el contexto
sc.stop()
```

En este flujo:
1. Se crean transformaciones (`filter` y `map`) para filtrar números pares y elevarlos al cuadrado.
2. Se ejecuta la acción `collect()` para obtener el resultado final.

**Lecturas recomendadas**

[Practica extra RDD (1).pdf - Google Drive](https://drive.google.com/file/d/1Ocy3sydSBhmmVjJWIDY1D446C2MRFiF8/view?usp=sharing)

[Resolucion - RDD.ipynb - Google Drive](https://drive.google.com/file/d/1Hhd7-oyjswG6FiALMajxZQIGSyT-iS59/view?usp=sharing)

## Lectura de datos con Spark

La lectura de datos en Spark es una de las operaciones iniciales más comunes. Spark puede leer datos desde múltiples fuentes, como archivos de texto, CSV, JSON, Parquet, bases de datos, sistemas de almacenamiento distribuido (como HDFS, S3, y Azure Blob Storage), entre otros.

A continuación, se detallan los pasos y ejemplos para leer datos usando Spark en Python.

---

### **1. Configuración inicial**
Primero, necesitas importar las bibliotecas necesarias y configurar un **SparkSession**:

```python
from pyspark.sql import SparkSession

# Crear una sesión de Spark
spark = SparkSession.builder \
    .appName("Lectura de datos") \
    .getOrCreate()
```

---

### **2. Tipos de datos que puedes leer**

#### **a. Archivos de texto**
Para leer archivos de texto, Spark genera un RDD o DataFrame donde cada línea del archivo es un registro.

```python
rdd = spark.sparkContext.textFile("ruta/al/archivo.txt")
print(rdd.collect())  # Muestra el contenido del archivo
```

#### **b. CSV**
Spark soporta la lectura de archivos CSV con opciones como encabezados, separadores personalizados y manejo de tipos de datos.

```python
# Leer un archivo CSV con encabezado
df_csv = spark.read.csv("ruta/al/archivo.csv", header=True, inferSchema=True)

# Mostrar las primeras filas
df_csv.show()
```

**Parámetros comunes:**
- `header=True`: Indica si la primera fila contiene nombres de columnas.
- `inferSchema=True`: Infieren los tipos de datos automáticamente.
- `sep=',':` Define el separador del archivo (por defecto, coma).

#### **c. JSON**
Spark puede leer datos en formato JSON, que pueden ser simples o anidados.

```python
# Leer un archivo JSON
df_json = spark.read.json("ruta/al/archivo.json")

# Mostrar la estructura del DataFrame
df_json.printSchema()
df_json.show()
```

#### **d. Parquet**
Parquet es un formato columnar altamente eficiente y compatible con Spark.

```python
# Leer un archivo Parquet
df_parquet = spark.read.parquet("ruta/al/archivo.parquet")

# Mostrar las primeras filas
df_parquet.show()
```

#### **e. JDBC (Bases de datos)**
Puedes conectar Spark a bases de datos relacionales mediante JDBC.

```python
df_jdbc = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:mysql://host:puerto/nombre_base") \
    .option("driver", "com.mysql.jdbc.Driver") \
    .option("dbtable", "nombre_tabla") \
    .option("user", "usuario") \
    .option("password", "contraseña") \
    .load()

df_jdbc.show()
```

---

### **3. Opciones adicionales**
Spark proporciona diversas opciones para ajustar cómo se leen los datos:

```python
df_csv = spark.read \
    .option("header", "true") \
    .option("sep", ";") \
    .option("inferSchema", "true") \
    .csv("ruta/al/archivo.csv")
df_csv.show()
```

---

### **4. Guardar datos después de leerlos**
Después de leer datos, puedes procesarlos y guardarlos en otros formatos.

```python
# Guardar en Parquet
df_csv.write.parquet("ruta/salida/parquet")

# Guardar en JSON
df_csv.write.json("ruta/salida/json")
```

---

### **Ejemplo completo**

```python
from pyspark.sql import SparkSession

# Crear la sesión de Spark
spark = SparkSession.builder \
    .appName("Ejemplo de lectura de datos") \
    .getOrCreate()

# Leer un archivo CSV
df_csv = spark.read.csv("ruta/al/archivo.csv", header=True, inferSchema=True)

# Mostrar información del DataFrame
df_csv.printSchema()
df_csv.show()

# Filtrar y guardar en formato Parquet
df_filtrado = df_csv.filter(df_csv['columna'] > 10)
df_filtrado.write.parquet("ruta/salida/filtrado.parquet")

# Finalizar la sesión de Spark
spark.stop()
```

---

### **Conclusión**
- Spark permite leer datos desde una amplia variedad de fuentes.
- Puedes usar parámetros como `header`, `inferSchema`, y `sep` para personalizar la lectura.
- Una vez cargados, los datos pueden transformarse y guardarse en diferentes formatos. 

Esto facilita trabajar con grandes volúmenes de datos en un flujo de trabajo de análisis o procesamiento distribuido.

**Lecturas recomendadas**

[Clase_Lectura_de_Datos.ipynb - Google Drive](https://drive.google.com/file/d/1AynZXI_u1KNp9czQX7ww1Nl2sqU_01xn/view?usp=sharing)

## ¿Qué es la Spark UI?

La **Spark UI (Spark User Interface)** es una interfaz gráfica de usuario integrada en Apache Spark que proporciona información detallada sobre la ejecución de trabajos, etapas y tareas en un clúster de Spark. Es una herramienta esencial para monitorear, depurar y optimizar aplicaciones en Spark.

### **Funciones principales de Spark UI**
1. **Monitoreo en tiempo real**:
   - Muestra el progreso de los trabajos (jobs) y etapas (stages) que se están ejecutando.
   - Proporciona detalles de las tareas individuales que componen cada etapa.

2. **Optimización de rendimiento**:
   - Ayuda a identificar cuellos de botella, como tareas lentas o desbalanceo en la distribución de datos.
   - Permite analizar el uso de recursos como CPU y memoria.

3. **Depuración de errores**:
   - Proporciona información detallada sobre errores en trabajos fallidos.
   - Muestra la cantidad de datos procesados, el tiempo de ejecución y las métricas relacionadas.

4. **Historial de aplicaciones**:
   - Permite revisar el historial de aplicaciones completadas, si se ha configurado un almacenamiento para el historial de eventos.

### **Componentes principales de Spark UI**
1. **Jobs**:
   - Lista todos los trabajos ejecutados en la aplicación, con detalles como estado (en ejecución, completado, fallido), duración y etapas asociadas.

2. **Stages**:
   - Muestra todas las etapas del trabajo, incluyendo métricas como tiempo de ejecución, shuffle read/write y número de tareas.

3. **Tasks**:
   - Detalla el progreso de las tareas dentro de cada etapa, con información sobre el tiempo de ejecución y recursos utilizados.

4. **Storage**:
   - Muestra información sobre los datos almacenados en caché o persistentes, como el tamaño de las particiones y el nivel de almacenamiento.

5. **Environment**:
   - Proporciona detalles sobre el entorno de ejecución, incluyendo configuraciones de Spark, variables del sistema y propiedades del clúster.

6. **SQL**:
   - Si se ejecutan consultas SQL, esta pestaña muestra los planes de ejecución físicos y lógicos, ayudando a optimizar las consultas.

7. **Streaming** (si se usa Spark Streaming):
   - Muestra métricas relacionadas con micro-batches, como la latencia y el tamaño de los lotes procesados.

### **Acceso a Spark UI**
- Para acceder a Spark UI, utiliza la URL proporcionada por tu entorno de Spark. Por ejemplo:
  - En un clúster local: `http://localhost:4040`
  - En un entorno como Databricks, Spark UI se integra con la interfaz de Databricks.

- Si Spark se ejecuta en un clúster distribuido, cada nodo maestro o administrador puede tener una URL específica para la Spark UI.

### **Beneficios de usar Spark UI**
- **Diagnóstico rápido**: Identifica problemas en la aplicación sin necesidad de examinar grandes cantidades de registros.
- **Optimización continua**: Mejora el rendimiento mediante el análisis de patrones de ejecución.
- **Simplicidad**: Proporciona una visión clara de procesos complejos en un formato visual.

### **Limitaciones**
- No ofrece soporte directo para depuración a nivel de código fuente.
- En aplicaciones largas o complejas, el análisis de grandes cantidades de datos en la interfaz puede ser tedioso.
- La interfaz local se pierde cuando se cierra la aplicación, a menos que se configure un almacenamiento de eventos para mantener el historial.

La Spark UI es una herramienta clave para trabajar con Apache Spark, ofreciendo insights poderosos para desarrolladores, administradores de clúster y analistas de datos.

**Lecturas recomendadas**

[Entendiendo la interfaz de usuario de Spark. Web UI Spark I | by Brayan Buitrago | iWannaBeDataDriven | Medium](https://medium.com/iwannabedatadriven/entendiendo-la-interfaz-de-usuario-de-spark-web-ui-spark-i-d03c6bd562a5#:~:text=Interfaz%20de%20usuario%20de%20Spark%20%E2%80%94%20Jobs&amp;text=Apache%20Spark%20proporciona%20una%20interfaz,varios%20monitores%20para%20diferentes%20prop%C3%B3sitos.)

## ¿Cómo instalar una librería en Databricks?

