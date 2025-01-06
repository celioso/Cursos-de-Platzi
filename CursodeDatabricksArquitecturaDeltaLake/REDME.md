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

En Databricks, instalar librerías es un proceso simple y directo. Puedes instalar librerías en un clúster desde diferentes fuentes como PyPI, Maven, CRAN, archivos locales, o incluso archivos de librerías personalizados almacenados en DBFS o S3.

### **Pasos para instalar una librería en Databricks**

#### **1. Desde el menú del clúster**
1. **Accede al clúster**:
   - En el menú lateral, ve a **Compute** y selecciona el clúster donde deseas instalar la librería.

2. **Selecciona "Libraries"**:
   - Haz clic en la pestaña **Libraries** en la interfaz del clúster.

3. **Instala la librería**:
   - Haz clic en **Install New**.
   - Selecciona la fuente de la librería:
     - **PyPI**: Para librerías de Python (e.g., `pandas`, `numpy`).
     - **Maven**: Para librerías de Java/Scala.
     - **CRAN**: Para librerías de R.
     - **Local Jar/Library**: Para archivos locales.
     - **Custom Library**: Para librerías en almacenamiento remoto (e.g., DBFS, S3).

4. **Especifica la librería**:
   - Si es de **PyPI**, escribe el nombre (e.g., `requests`) o incluye una versión específica (`requests==2.26.0`).
   - Haz clic en **Install**.

#### **2. Desde un notebook**
Puedes instalar una librería directamente desde un notebook usando comandos mágicos:

##### **Para PyPI:**
```python
%pip install nombre_libreria
```
Ejemplo:
```python
%pip install matplotlib
```

##### **Para librerías de Maven:**
```python
%scala
spark.jars.packages += "grupo:nombre:versión"
```
Ejemplo:
```scala
spark.jars.packages += "org.apache.spark:spark-avro_2.12:3.4.0"
```

##### **Para librerías R:**
```R
install.packages("nombre_libreria")
```
Ejemplo:
```R
install.packages("ggplot2")
```

### **Verificación**
1. Si la instalación fue desde el clúster, la librería estará disponible en todos los notebooks asociados al clúster.
2. Si fue desde un notebook, estará disponible únicamente en ese notebook.

### **Consideraciones importantes**
- **Reinicio del clúster**: Algunas instalaciones pueden requerir un reinicio del clúster para que los cambios surtan efecto.
- **Versiones compatibles**: Asegúrate de instalar versiones de librerías compatibles con tu versión de Spark y Databricks.
- **Ámbito**: Las librerías instaladas a través de `%pip` están limitadas al ámbito del notebook, mientras que las instaladas a nivel de clúster son globales para todos los notebooks del clúster.

Con estos pasos, podrás instalar cualquier librería en Databricks y empezar a usarla en tus análisis o flujos de trabajo.

**Lecturas recomendadas**

[Clase - Cómo instalar una librería en Databricks_.ipynb - Google Drive](https://drive.google.com/file/d/1mobRLK6j5PHKL09k2XrVt7lcByAGjNyZ/view?usp=sharing)

[MVN Repository](https://mvnrepository.com/)

## Spark en local vs. en la nube

**Spark en local** y **Spark en la nube** tienen diferencias significativas en términos de configuración, escalabilidad, rendimiento y accesibilidad. A continuación se comparan ambos entornos:

### **Spark en local**

#### Ventajas:
- **Costo**: Es gratuito para uso personal o de desarrollo (requieres tener un entorno configurado en tu máquina).
- **Flexibilidad**: Puedes personalizar y optimizar Spark de acuerdo a tu hardware.
- **Desarrollo y Pruebas**: Ideal para pruebas locales o desarrollo rápido.

#### Desventajas:
- **Escalabilidad limitada**: Dependiente del hardware de la máquina local, lo que puede limitar el procesamiento a conjuntos de datos más pequeños.
- **Mantención**: Necesitas gestionar la configuración, los recursos y cualquier problema relacionado con el hardware.

#### Uso típico:
- Pequeñas cantidades de datos.
- Desarrollo, pruebas y demostraciones.

### **Spark en la nube (como Databricks o AWS EMR)**

#### Ventajas:
- **Escalabilidad**: Puedes escalar según tus necesidades utilizando clústeres distribuidos en la nube, desde pocos hasta miles de nodos.
- **Acceso compartido**: Varias instancias pueden trabajar simultáneamente en el mismo clúster sin preocuparte por el estado de recursos físicos.
- **Optimización**: Servicios optimizados específicamente para Spark en la nube, lo que facilita la gestión de recursos.
- **Seguridad**: Integraciones robustas de IAM (gestión de identidad y acceso) y políticas empresariales.

#### Desventajas:
- **Costo**: Usualmente más caro que ejecutar Spark en local, especialmente a gran escala.
- **Dependencia externa**: Requiere acceso a internet y servicios en la nube, que puede generar costos adicionales por tráfico o almacenamiento.

#### Uso típico:
- Grandes conjuntos de datos.
- Procesamiento en tiempo real.
- Colaboración y trabajo en equipo.
- Modelos complejos o machine learning.

### **Comparación General**

| Aspecto              | Local                            | Cloud                           |
|---------------------|---------------------------------|---------------------------------|
| **Escalabilidad**   | Limitada por el hardware local.    | Prácticamente ilimitada mediante clústeres distribuidos. |
| **Rendimiento**    | Menor rendimiento para conjuntos de datos grandes. | Alto rendimiento mediante recursos optimizados. |
| **Configuración**  | Configuración manual y compleja.  | Configuración simplificada con servicios gestionados. |
| **Costo**          | Bajo o gratuito para pequeñas pruebas. | Puede ser más caro, pero escalable y manejado por proveedores. |
| **Seguridad**      | Gestionado manualmente.            | Integraciones robustas y políticas empresariales. |

### **Recomendación**

- **Desarrollo y pruebas**: Inicia en local para entender la herramienta y ajustar configuraciones.
- **Producción y grandes datasets**: Usa Spark en la nube para garantizar escalabilidad, optimización y seguridad.

**Lecturas recomendadas**

[Spark local mode vs Cluster mode](https://ar.video.search.yahoo.com/search/video?fr=mcafee&ei=UTF-8&p=spark+local+vs+nube&type=E210AR885G0#id=3&vid=7a3598f010ed006b93d0565d10324662&action=click)

## ¿Qué son los Dataframes en Apache Spark?

En Apache Spark, un DataFrame es una estructura de datos distribuida que se organiza en filas y columnas. Es una abstracción de datos tabulares que se asemeja a una tabla en una base de datos relacional o a un DataFrame en lenguajes de programación como R o Python.

Resulta importante mencionar que un DataFrame en Spark puede ser creado a partir de diversas fuentes de datos como archivos CSV, JSON, Parquet, bases de datos SQL, entre otros. Por supuesto, también lo podemos crear desde 0.

En esencia, un dataframe proporciona una interfaz de programación para manipular y procesar datos de manera distribuida en un clúster de computadoras. A este tipo de objeto también es posible aplicarle transformaciones y acciones en Spark.

Algunas características clave de los DataFrames en Apache Spark son:

1. **Inmutabilidad**: Al igual que en otros contextos de Spark los DataFrames son inmutables, lo que significa que no se pueden modificar directamente después de su creación. Las transformaciones en un Dataframe crean otro nuevo.
2. **Optimización de consultas**: Spark optimiza automáticamente las consultas en los DataFrames utilizando su motor de ejecución Catalyst. Esto permite realizar operaciones de manera eficiente y optimizada.
3. **Soporte para varios lenguajes**: Los DataFrames en Spark están disponibles en varios lenguajes de programación como Scala, Java, Python y R, lo que facilita su uso en diferentes entornos.
4. **Integración con fuentes de datos externas**: Pueden leer y escribir datos desde y hacia una variedad de fuentes de datos, lo que facilita la integración con diferentes sistemas y formatos.
5. API rica: Los DataFrames proporcionan una API rica que permite realizar operaciones complejas de manipulación y análisis de datos de manera declarativa, lo que facilita la expresión de la lógica de procesamiento.

Por lo tanto, podemos concluir que los DataFrames en Apache Spark ofrecen una interfaz de alto nivel para el procesamiento de datos distribuidos, facilitando la manipulación y el análisis de grandes conjuntos de datos de manera eficiente en clústeres de computadoras.

**Escritura y lectura de dataframes en Spark**

En Apache Spark, la lectura y escritura de DataFrames se realiza utilizando las API correspondientes proporcionadas por Spark.

Lectura de DataFrames:

**A. Desde un archivo CSV:**

 ```python
from pyspark.sql import SparkSession

 spark = SparkSession.builder.appName("example").getOrCreate()

 df = spark.read.csv("ruta/del/archivo.csv", header=True, inferSchema=True)
```
 
En este ejemplo, **header=True** indica que la primera fila del archivo CSV contiene nombres de columnas, y **inferSchema=True** intentará inferir automáticamente los tipos de datos de las columnas.

**B. Desde un archivo Parquet:**

```python
 df = spark.read.parquet("ruta/del/archivo.parquet")
```

Parquet es un formato de almacenamiento eficiente para datos estructurados y es compatible con Spark.

**C. Desde una tabla de base de datos SQL:**

```python
df = spark.read.jdbc(url="jdbc:mysql://localhost:3306/base_de_datos", table="nombre_tabla", properties={"user": "usuario", "password": "contraseña"})
```

En este ejemplo se lee desde una tabla de una base de datos SQL.

Escritura de DataFrames:

1. **A un archivo CSV:**

`df.write.csv(“ruta/de/destino”, header=True)`

Esto escribirá el DataFrame en formato CSV en la ruta especificada.

**2. A un archivo Parquet:**

`df.write.parquet(“ruta/de/destino”)`

Escribir en formato Parquet es eficiente y puede ser una buena opción para el almacenamiento a largo plazo.

**3. A una tabla de base de datos SQL:**

```python
df.write.jdbc(url=“jdbc:mysql://localhost:3306/base_de_datos”, table=“nombre_tabla”, mode=“overwrite”, properties={“user”: “usuario”, “password”: “contraseña”})
```

En este ejemplo, mode=“overwrite” indica que se sobrescribirán los datos si la tabla ya existe. Otras opciones incluyen “append” y “ignore”.

Más información: [https://spark.apache.org/docs/latest/sql-programming-guide.html](https://spark.apache.org/docs/latest/sql-programming-guide.html "https://spark.apache.org/docs/latest/sql-programming-guide.html")

## Laboratorio - PySpark SQL - Parte 1

¿Te gustaría que cree un laboratorio práctico en formato paso a paso para trabajar con PySpark SQL? Puedo incluir ejercicios que cubran desde la configuración inicial hasta consultas más avanzadas con ejemplos prácticos. Por ejemplo:

1. **Configuración inicial**: Instalación de PySpark y creación de un entorno en Databricks o localmente.
2. **Carga de datos**: Lectura de archivos CSV, JSON o Parquet.
3. **Transformaciones básicas**: Uso de operaciones como `select`, `filter`, y `groupBy`.
4. **Consultas SQL**: Registro de DataFrames como tablas temporales y ejecución de consultas SQL.
5. **Acciones y exportación de resultados**: Uso de `show`, `collect` y escritura de resultados en archivos.

## Laboratorio - PySpark SQL - Parte 2

¡Entendido! Si estás trabajando en un laboratorio de PySpark SQL, seguramente estás explorando cómo usar las capacidades SQL en PySpark para consultas, análisis y manipulación de datos.

### Parte 2: Ejemplo estructurado para PySpark SQL

A continuación, te guiaré a través de una estructura típica para trabajar con PySpark SQL en un laboratorio:

#### 1. **Crear o cargar un DataFrame**
Cargamos datos desde un archivo CSV o creamos un DataFrame manualmente.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Crear una SparkSession
spark = SparkSession.builder.appName("PySpark SQL Lab").getOrCreate()

# Crear un DataFrame manualmente
data = [
    (1, "Alice", 5000),
    (2, "Bob", 4000),
    (3, "Charlie", 7000),
    (4, "David", 10000),
]
columns = ["id", "name", "salary"]

df = spark.createDataFrame(data, columns)

# Mostrar los datos iniciales
df.show()
```

#### 2. **Registrar una vista temporal**
Registrar una tabla temporal para ejecutar consultas SQL.

```python
# Registrar el DataFrame como una tabla temporal
df.createOrReplaceTempView("employees")
```

#### 3. **Ejecutar consultas SQL**
Usar Spark SQL para ejecutar consultas sobre los datos.

```python
# Ejemplo 1: Seleccionar todos los registros
result = spark.sql("SELECT * FROM employees")
result.show()

# Ejemplo 2: Filtrar empleados con salario mayor a 5000
result = spark.sql("SELECT * FROM employees WHERE salary > 5000")
result.show()

# Ejemplo 3: Agregar una columna calculada (bonus)
result = spark.sql("SELECT *, salary * 0.1 AS bonus FROM employees")
result.show()

# Ejemplo 4: Calcular la suma total de salarios
result = spark.sql("SELECT SUM(salary) AS total_salary FROM employees")
result.show()
```

#### 4. **Realizar transformaciones adicionales**
Combinar operaciones SQL y funciones PySpark.

```python
# Agregar una columna 'bonus' al DataFrame original y calcular el total
df = df.withColumn("bonus", col("salary") * 0.1)
df = df.withColumn("total", col("salary") + col("bonus"))

# Mostrar el DataFrame actualizado
df.show()
```

#### 5. **Guardar los resultados**
Guardar el resultado de las consultas o transformaciones en diferentes formatos.

```python
# Guardar el DataFrame actualizado como un archivo CSV
df.write.csv("output/employees_with_bonus", header=True)
```

**Lecturas recomendadas**

[Spark SQL - Consigna.ipynb - Google Drive](https://drive.google.com/file/d/17pfMCkUSoWT1vlx0BEs2Vi6X9LzsDaBa/view?usp=sharing)

[Spark SQL - Resolucion.ipynb - Google Drive](https://drive.google.com/file/d/1_WSDtU96W_Mhlm5Ne6k-Apk7qxeTBogK/view?usp=sharing)

[Clase - Laboratorio - PySpark SQL - Parte 2.ipynb - Google Drive](https://drive.google.com/file/d/1dt4NGhhlwvm8Ur1cdrCUcw0HMXuYgTKB/view?usp=sharing)

## UDF en Apache Spark

En Apache Spark, una **UDF (User Defined Function)** es una función definida por el usuario que puedes utilizar para realizar transformaciones personalizadas en los datos. Las UDF permiten aplicar lógica compleja a columnas de un DataFrame que no está disponible con las funciones integradas de Spark.

### Características principales de las UDF:
1. **Personalización**: Permiten agregar lógica específica del negocio.
2. **Compatibilidad**: Se pueden usar con diferentes lenguajes como Python, Scala, Java y R.
3. **Desempeño**: Las UDF pueden ser más lentas que las funciones integradas de Spark porque se ejecutan en el intérprete del lenguaje (por ejemplo, Python) y no están optimizadas para la ejecución distribuida.

### 1. Crear una UDF en Python
Puedes registrar una función personalizada como una UDF y usarla en transformaciones.

#### Paso 1: Importar las funciones necesarias
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
```

#### Paso 2: Definir la función personalizada
```python
# Función Python normal
def categorize_salary(salary):
    if salary < 5000:
        return "Low"
    elif salary <= 8000:
        return "Medium"
    else:
        return "High"
```

#### Paso 3: Registrar la función como UDF
```python
categorize_salary_udf = udf(categorize_salary, StringType())
```

#### Paso 4: Aplicar la UDF al DataFrame
```python
data = [
    (1, "Alice", 4000),
    (2, "Bob", 5000),
    (3, "Charlie", 9000),
]
columns = ["id", "name", "salary"]

df = spark.createDataFrame(data, columns)

# Usar la UDF para transformar los datos
df = df.withColumn("salary_category", categorize_salary_udf(df["salary"]))
df.show()
```

**Salida esperada:**
```
+---+-------+------+---------------+
| id|   name|salary|salary_category|
+---+-------+------+---------------+
|  1|  Alice|  4000|            Low|
|  2|    Bob|  5000|         Medium|
|  3|Charlie|  9000|           High|
+---+-------+------+---------------+
```

### 2. UDF con SQL
Puedes registrar una UDF y usarla en consultas SQL.

```python
# Registrar la UDF en el contexto SQL
spark.udf.register("categorize_salary", categorize_salary, StringType())

# Registrar la tabla temporal
df.createOrReplaceTempView("employees")

# Ejecutar una consulta SQL con la UDF
result = spark.sql("SELECT id, name, salary, categorize_salary(salary) AS salary_category FROM employees")
result.show()
```

### 3. Tipos de datos soportados
Cuando defines una UDF, necesitas especificar el tipo de dato de salida usando las clases de `pyspark.sql.types`:

- **StringType**: Para cadenas.
- **IntegerType**: Para enteros.
- **DoubleType**: Para números decimales.
- **ArrayType**: Para listas o arreglos.
- **StructType**: Para estructuras anidadas.

Ejemplo con un tipo complejo:
```python
from pyspark.sql.types import ArrayType

def split_name(name):
    return name.split(" ")

split_name_udf = udf(split_name, ArrayType(StringType()))
df = df.withColumn("name_parts", split_name_udf(df["name"]))
df.show()
```

### 4. Consideraciones de rendimiento
- **Uso de funciones integradas**: Siempre que sea posible, usa las funciones integradas de Spark en lugar de UDF, ya que están optimizadas para la ejecución distribuida.
- **Vectorización con Pandas UDF**: Para mejorar el rendimiento, puedes usar Pandas UDFs, que están optimizadas para operaciones vectorizadas y aprovechan mejor los recursos de Spark.

Ejemplo de Pandas UDF:
```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

@pandas_udf(StringType())
def categorize_salary_pandas(salary_series):
    return salary_series.apply(categorize_salary)

df = df.withColumn("salary_category", categorize_salary_pandas(df["salary"]))
```

**Lecturas recomendadas**

[¿Qué son las funciones definidas por el usuario (UDF)? - Azure Databricks | Microsoft Learn](https://learn.microsoft.com/es-es/azure/databricks/udf/)

[Clase - Ejemplo practico de UDF.ipynb - Google Drive](https://drive.google.com/file/d/1SYF9CzzrAys6aALSbXibNPmHIMx4ensw/view?usp=sharing)

[Ejercicios.pdf - Google Drive](https://drive.google.com/file/d/1_QyDZaZO9TNI6abWk1_33YSx-iknR9yA/view?usp=sharing)

[Resolucion - UDF.ipynb - Google Drive](https://drive.google.com/file/d/1wl51xG3t6KG4C18bKb0R80yzdbqlNLdD/view?usp=sharing)

## Arquitectura Data Lake vs Delta Lake

**Data Lake**:

1. **Definición**: Un Data Lake es un repositorio centralizado para almacenar grandes volúmenes de datos estructurados, semi-estructurados y no estructurados a gran escala. No requiere un esquema previo, permitiendo que los datos se carguen en bruto.

2. **Características**:
   - **Esquema opcional**: Los datos pueden ingresar en cualquier formato sin necesidad de un esquema definido inicialmente.
   - **Flexibilidad**: Ideal para almacenar grandes cantidades de datos variados.
   - **Desventajas**: Falta de rendimiento en consultas masivas o transformaciones complejas, ya que no se optimiza el almacenamiento.

3. **Ejemplo**: Amazon S3, Azure Data Lake, Google Cloud Storage.

**Delta Lake**:

1. **Definición**: Delta Lake es una solución de almacenamiento que combina las ventajas de un Data Lake con las capacidades de un almacenamiento optimizado y altamente eficiente. Es un formato de datos abierto, compatible con Spark, que permite operaciones ACID (Atomicidad, Consistencia, Aislamiento, Durabilidad).

2. **Características**:
   - **Optimización**: Delta Lake ofrece una mejor optimización de las operaciones mediante el uso de metadata, almacenamiento optimizado y eliminación de duplicados.
   - **Transacciones ACID**: Soporte para operaciones seguras y consistentes, con rollback de transacciones.
   - **Integración con Spark**: Compatible con Apache Spark, permitiendo procesamiento rápido y eficiente.
   - **Historial de datos**: Soporte para la evolución del esquema con control de versiones y registros de cambios.

3. **Ventajas**:
   - **Rendimiento** mejorado en consultas y transformaciones.
   - **Optimización de almacenamiento** a través de particiones y eliminación de archivos innecesarios.
   - **Alta seguridad** y gestión del acceso a datos.

4. **Ejemplo**: Delta Lake en Databricks o en servicios compatibles como Azure Synapse y Google Cloud BigLake.

### Comparación:

- **Data Lake** es una solución genérica de almacenamiento de datos, mientras que **Delta Lake** agrega capacidades avanzadas de optimización, versionamiento, y transacciones.

**Lecturas recomendadas**

[What is the Difference Between a Data Lake and a Delta Lake? | by Ansam Yousry | Technology Hits | Medium](https://medium.com/technology-hits/what-is-the-difference-between-a-data-lake-and-a-delta-lake-2ff64d85758b)

[Lectura - Parquet vs Delta.pdf - Google Drive](https://drive.google.com/file/d/1cK1MbwM8eEZScfUWaOKg2LPWlQrRGaLh/view?usp=sharing)

## Características y beneficios del Delta Lake

Como ya hemos visto, Delta Lake es una solución de almacenamiento de datos basada en un sistema de archivos distribuido diseñado para mejorar la calidad, la confiabilidad y el rendimiento de los datos en entornos de big data.

**Características de Delta Lake:**

- *Transacciones ACID:*
Delta Lake proporciona soporte nativo para transacciones ACID (atomicidad, consistencia, aislamiento y durabilidad), lo que garantiza un rendimiento fluido de lectura y escritura, y consistente incluso en entornos distribuidos.

- *Control de versiones:*
Admite un historial de versiones completo de los datos almacenados, lo que le permite analizar los cambios a lo largo del tiempo y volver a versiones anteriores si es necesario.

- *Operaciones de fusión y actualización:*
Facilita las operaciones de fusión y actualización de datos, lo que simplifica el procesamiento y la edición de datos.

- *Optimizaciones de lectura y escritura:*
Contiene optimizaciones que aceleran las operaciones de lectura y escritura, como la indexación y la gestión de estadísticas que mejoran el rendimiento en comparación con el uso del sistema de archivos directamente sin estas optimizaciones.

- *Compatibilidad con Apache Spark:*
Delta Lake es totalmente compatible con Apache Spark lo que facilita la integración en el ecosistema Spark y aprovecha las capacidades de procesamiento distribuido.

- *Evolución del esquema:*
Facilita la evolución del esquema de datos a lo largo del tiempo, permitiendo cambios en la estructura de datos sin afectar la compatibilidad con versiones anteriores.

- *Gestión de metadatos:*
Delta Lake almacena metadatos internos que proporcionan información sobre los datos, facilitando la gestión y el control de los datos.

**Beneficios del Delta Lake**

- *Integridad y coherencia de los datos:*
La gestión de transacciones ACID garantiza la integridad y la coherencia de los datos, lo cual es fundamental en entornos donde la precisión de los datos es fundamental.

- *Mejor rendimiento:*
Las optimizaciones internas, como la indexación y la gestión de estadísticas, mejoran el rendimiento de las operaciones de lectura y escritura y permiten un acceso más eficiente a los datos.

- *Historial de versiones para revisión:*
El historial de versiones le permite monitorear y analizar los cambios de datos a lo largo del tiempo y proporciona una descripción detallada de la evolución de los conjuntos de datos.

- *Flexibilidad en el desarrollo de esquemas:*
La capacidad de evolucionar sin problemas el esquema de datos facilita una adaptación perfecta a los cambios comerciales.

- *Operaciones simplificadas:*
Delta Lake simplifica operaciones como la fusión y la actualización lo que facilita el trabajo con datos.

- *Compatibilidad con herramientas de big data:*
Al admitir Apache Spark, Delta Lake se puede utilizar fácilmente con otras herramientas de big data, ampliando su aplicabilidad en entornos distribuidos.

Para concluir, estas características y beneficios hacen que Delta Lake sea una solución poderosa para el almacenamiento y la gestión de datos en entornos de big data, proporcionando un conjunto de funcionalidades avanzadas para abordar desafíos comunes en la administración de grandes volúmenes de datos.

## Medallion architecture

**Medallion Architecture** es un enfoque moderno para la arquitectura de almacenamiento de datos que utiliza capas definidas para gestionar diferentes tipos de datos y casos de uso en un solo sistema de datos. Es ampliamente utilizado en entornos de datos en la nube y sistemas distribuidos.

#### Componentes de la Medallion Architecture:

1. **Bronze Layer (Capa Bronce)**:
   - **Descripción**: Capa de entrada de datos sin procesar. Aquí se cargan datos en bruto desde sistemas fuentes como Data Lakes, S3, o bases de datos.
   - **Objetivo**: Almacenamiento de datos originales, sin transformación ni limpieza.
   - **Usos**: Datos históricos, raw data, logs, o registros sin procesar.

2. **Silver Layer (Capa Plata)**:
   - **Descripción**: Capa de procesamiento y transformación de datos. Los datos en esta capa son más estructurados y limpios que en la capa bronce.
   - **Objetivo**: Procesamiento y transformación de datos, aplicación de esquemas, y creación de datasets que pueden ser utilizados en análisis.
   - **Usos**: Datos transformados para reporting, Machine Learning, y análisis en tiempo real.

3. **Gold Layer (Capa Oro)**:
   - **Descripción**: Capa de datos finales optimizada para consumo de aplicaciones analíticas o modelos predictivos. Esta capa almacena datos altamente procesados y enriquecidos.
   - **Objetivo**: Crear conjuntos de datos preprocesados optimizados para dashboards, visualizaciones, o modelos analíticos avanzados.
   - **Usos**: Datos listos para su uso en BI, análisis ad-hoc, inteligencia empresarial (BI), y visualizaciones.

#### Ventajas de la Medallion Architecture:

- **Flexibilidad**: Permite gestionar diferentes tipos de datos y transformaciones en un solo flujo de trabajo.
- **Optimización**: Mejor rendimiento y escalabilidad gracias a la separación de capas.
- **Seguridad**: Datos brutos en la capa bronce, procesados en la capa plata y altamente estructurados en la capa oro.
- **Historial y evolución**: Cada capa puede evolucionar su esquema y agregar datos históricos.

#### Implementación en Databricks:

- **Bronze**: Almacenamiento inicial en DBFS o Delta Lake sin transformaciones.
- **Silver**: Transformaciones, limpieza y procesamiento usando PySpark o Databricks Notebooks.
- **Gold**: Datos altamente optimizados para uso analítico y reporting.

**Lecturas recomendadas**

[What is the medallion lakehouse architecture? - Azure Databricks | Microsoft Learn](https://learn.microsoft.com/en-us/azure/databricks/lakehouse/medallion)

[Lectura - Arquitectura Medallion.pdf - Google Drive](https://drive.google.com/file/d/1CB473K1shy0ulgclYHMi6uGoKJ0x1Pze/view?usp=sharing)