# Curso de Fundamentos de ETL con Python y Pentaho

## ¿Qué es un ETL en ingeniería de datos?

Un **ETL** (Extract, Transform, Load) es un proceso en la ingeniería de datos que se utiliza para mover y transformar datos desde diferentes fuentes hacia un destino centralizado, como un **data warehouse** o un **data lake**, para su análisis o procesamiento. Este proceso consta de tres fases principales:

### **1. Extract (Extracción):**
- **Objetivo:** Obtener datos de una o más fuentes heterogéneas.
- **Fuentes típicas:** Bases de datos SQL, archivos (CSV, JSON, XML), APIs, sistemas ERP, hojas de cálculo, entre otros.
- **Desafíos comunes:** Manejo de datos inconsistentes, conexiones lentas y formatos variados.

### **2. Transform (Transformación):**
- **Objetivo:** Procesar y convertir los datos para que sean consistentes, limpios y adecuados para el análisis.
- **Ejemplos de transformaciones:**
  - Normalización de formatos (ejemplo: fechas o monedas).
  - Enriquecimiento de datos (agregar datos adicionales desde otra fuente).
  - Limpieza de datos (eliminar duplicados, corregir errores).
  - Agregaciones o cálculos (ejemplo: sumar ingresos mensuales).
- **Beneficio:** Garantiza que los datos sean útiles y coherentes antes de su almacenamiento.

### **3. Load (Carga):**
- **Objetivo:** Transferir los datos transformados al sistema de destino.
- **Tipos de carga:**
  - **Carga completa:** Se transfiere todo el conjunto de datos.
  - **Carga incremental:** Solo se cargan los datos nuevos o modificados.
- **Destinos típicos:** Bases de datos analíticas, herramientas de BI (Business Intelligence), o sistemas de almacenamiento en la nube.

### **Importancia del ETL:**
- Proporciona una base sólida para el análisis de datos y la toma de decisiones basada en datos confiables.
- Permite integrar información de múltiples fuentes en un único sistema.
- Mejora la calidad y accesibilidad de los datos.

### Herramientas ETL populares:
- **Open Source:** Apache Nifi, Talend Open Studio, Pentaho.
- **Comerciales:** Informatica, Microsoft SQL Server Integration Services (SSIS).
- **Basadas en la nube:** AWS Glue, Google Dataflow, Azure Data Factory.

Aquí tienes un ejemplo básico de un proceso ETL en Python utilizando la biblioteca `pandas`. Este script leerá datos de un archivo CSV, realizará algunas transformaciones básicas y luego los cargará en un archivo de salida:

### Ejemplo de ETL con Python y pandas

```python
import pandas as pd

# ETL Pipeline

# --- Extract ---
# Leer datos desde un archivo CSV
input_file = 'datos_entrada.csv'  # Archivo de entrada
try:
    data = pd.read_csv(input_file)
    print("Datos cargados exitosamente:")
    print(data.head())
except FileNotFoundError:
    print(f"El archivo {input_file} no existe. Por favor, verifica la ruta.")

# --- Transform ---
# Transformaciones básicas:
# 1. Renombrar columnas
data.rename(columns={"Nombre": "name", "Edad": "age", "País": "country"}, inplace=True)

# 2. Convertir la edad a números enteros
data['age'] = pd.to_numeric(data['age'], errors='coerce')

# 3. Filtrar filas donde la edad no sea nula y mayores de 18 años
data = data[data['age'] >= 18]

# 4. Capitalizar nombres de país
data['country'] = data['country'].str.capitalize()

print("\nDatos después de la transformación:")
print(data.head())

# --- Load ---
# Guardar datos transformados en un nuevo archivo
output_file = 'datos_salida.csv'
data.to_csv(output_file, index=False)
print(f"\nDatos procesados guardados en {output_file}")
```

### ¿Qué hace este script?

1. **Extract (Extracción):**
   - Lee un archivo CSV llamado `datos_entrada.csv`.
   - Muestra las primeras filas para verificar el contenido.

2. **Transform (Transformación):**
   - Renombra columnas para estandarizar los nombres.
   - Convierte la columna de edad (`Edad`) en números enteros.
   - Filtra registros donde la edad sea mayor o igual a 18.
   - Ajusta los nombres de los países para tener formato capitalizado.

3. **Load (Carga):**
   - Guarda el resultado transformado en un nuevo archivo llamado `datos_salida.csv`.

### Estructura de entrada esperada (`datos_entrada.csv`):
```csv
Nombre,Edad,País
Juan,25,Colombia
Maria,17,perú
Carlos,30,Chile
Ana,,Brasil
```

### Resultado (`datos_salida.csv`):
```csv
name,age,country
Juan,25,Colombia
Carlos,30,Chile
```
### **ETL vs. ELT: Propósito y Usos**

La elección entre **ETL (Extract, Transform, Load)** y **ELT (Extract, Load, Transform)** depende del caso de uso, la infraestructura y los requisitos del negocio. Ambos son enfoques utilizados para mover y procesar datos, pero tienen diferencias clave en cómo y dónde se realiza la transformación de los datos.

### **1. ETL (Extract, Transform, Load)**

#### **¿Qué es?**
- En ETL, los datos se extraen de las fuentes, se transforman en un entorno intermedio (como un servidor o herramienta dedicada), y luego se cargan en el sistema de destino.

#### **¿Para qué se utiliza?**
- **Sistemas tradicionales de análisis de datos** como **Data Warehouses** (almacenes de datos).
- Cuando el sistema de destino tiene capacidades limitadas para procesar datos o se requieren transformaciones complejas antes de la carga.
- **Casos donde se prioriza la calidad de los datos antes del análisis.**

#### **Ventajas:**
1. **Transformación temprana:** Los datos llegan al destino ya listos para el análisis.
2. **Control sobre la calidad de datos:** Permite realizar validaciones estrictas y limpiezas antes de la carga.
3. **Compatible con sistemas antiguos:** Ideal para bases de datos tradicionales que no manejan grandes volúmenes de datos.

#### **Limitaciones:**
- Puede ser más lento, especialmente con volúmenes grandes de datos.
- Requiere una infraestructura intermedia para realizar las transformaciones.

### **2. ELT (Extract, Load, Transform)**

#### **¿Qué es?**
- En ELT, los datos se extraen de las fuentes, se cargan directamente en el sistema de destino (por ejemplo, un Data Lake o un Data Warehouse moderno), y luego se transforman utilizando el poder computacional del sistema de destino.

#### **¿Para qué se utiliza?**
- **Sistemas modernos en la nube**, como **Data Lakes** (Google BigQuery, Amazon Redshift, Snowflake).
- Cuando los volúmenes de datos son masivos y los sistemas de destino tienen alta capacidad de procesamiento.
- **Análisis en tiempo real o cuasi-tiempo real**, donde los datos deben estar rápidamente disponibles.

#### **Ventajas:**
1. **Velocidad de carga inicial:** Los datos se mueven rápidamente al destino.
2. **Escalabilidad:** Aprovecha la potencia de procesamiento de sistemas en la nube.
3. **Flexibilidad:** Permite explorar y transformar los datos según las necesidades posteriores.

#### **Limitaciones:**
- Requiere un sistema de destino robusto con capacidades avanzadas.
- Puede cargar datos "sucios" inicialmente, lo que podría generar problemas si no se transforman correctamente.

### **Comparación Directa**

| Característica              | **ETL**                           | **ELT**                          |
|-----------------------------|------------------------------------|-----------------------------------|
| **Momento de transformación** | Antes de cargar los datos          | Después de cargar los datos       |
| **Sistema de destino**       | Data Warehouse tradicional         | Data Lake o Data Warehouse moderno |
| **Procesamiento de datos**   | Externo al destino                 | Interno en el destino            |
| **Volúmenes de datos**       | Moderados                          | Grandes o masivos                |
| **Velocidad inicial**        | Más lenta                          | Más rápida                       |
| **Limpieza y calidad inicial**| Alta calidad antes de la carga     | Requiere procesamiento posterior |

### **¿Cuándo usar ETL?**
- Cuando los datos requieren transformaciones complejas o detalladas antes de ser útiles.
- Cuando trabajas con sistemas antiguos o limitados en capacidad.
- Cuando los datos necesitan estar listos para análisis inmediatamente después de la carga.

### **¿Cuándo usar ELT?**
- Cuando se manejan grandes volúmenes de datos no estructurados.
- Si el sistema de destino tiene gran capacidad de procesamiento y escalabilidad (por ejemplo, BigQuery, Snowflake).
- Cuando se necesita flexibilidad para analizar o transformar datos en diferentes momentos.

## Conceptos base de ETL

### Conceptos Base de ETL (Extract, Transform, Load)

ETL (Extraer, Transformar, Cargar) es un proceso fundamental en la ingeniería de datos que permite trasladar y procesar datos desde múltiples fuentes hacia un destino final para análisis o almacenamiento. A continuación, se describen los conceptos base:

### **1. Extracción (Extract)**
**Definición:** Es el proceso de recopilar datos desde una o varias fuentes heterogéneas. Las fuentes pueden incluir bases de datos relacionales, archivos planos (CSV, JSON, XML), APIs, logs, o sistemas ERP.  

**Características:**
- **Variedad de fuentes:** Datos estructurados (tablas SQL) y no estructurados (archivos de texto, imágenes).
- **Objetivo:** Obtener datos sin alterar su formato original.
- **Herramientas comunes:** Conectores de bases de datos, APIs REST, scripts personalizados.

**Ejemplo:**  
Conectar a una base de datos SQL para extraer una tabla de usuarios:
```sql
SELECT * FROM usuarios;
```

### **2. Transformación (Transform)**
**Definición:** Es la etapa donde los datos se limpian, estandarizan, enriquecen o transforman para adaptarse a las necesidades del negocio o del sistema de destino.  

**Operaciones típicas:**
- **Limpieza:** Eliminar valores nulos, duplicados o inconsistentes.
- **Normalización:** Cambiar formatos de fecha o convertir unidades de medida.
- **Cálculos:** Crear nuevas columnas (por ejemplo, calcular ingresos anuales a partir de ingresos mensuales).
- **Enriquecimiento:** Combinar datos de múltiples fuentes.
- **Validación:** Asegurarse de que los datos cumplen con reglas de negocio específicas.

**Ejemplo:**  
Convertir un archivo CSV de ventas en un formato estandarizado:
```python
import pandas as pd

# Cargar datos
data = pd.read_csv("ventas.csv")

# Limpiar y transformar
data['fecha'] = pd.to_datetime(data['fecha'])
data['total'] = data['cantidad'] * data['precio_unitario']
data = data.dropna()  # Eliminar valores nulos
```

### **3. Carga (Load)**
**Definición:** Es el proceso de mover los datos transformados al sistema de destino, como un almacén de datos (Data Warehouse), base de datos, o sistema de análisis.  

**Tipos de carga:**
- **Carga completa:** Sobrescribe los datos existentes en cada ejecución.
- **Carga incremental:** Solo se cargan los datos nuevos o modificados.
- **Carga en tiempo real:** Los datos se envían continuamente al destino.

**Herramientas comunes:** 
- SQL para bases de datos relacionales.
- APIs o conectores específicos para sistemas en la nube como Amazon S3 o Google BigQuery.

**Ejemplo:**  
Insertar los datos procesados en una tabla de SQL:
```sql
INSERT INTO ventas_procesadas (fecha, producto, cantidad, total)
VALUES ('2024-01-01', 'Laptop', 10, 15000);
```

### **Objetivo del Proceso ETL**
El propósito principal de ETL es consolidar datos dispersos en un solo lugar, procesarlos para que sean útiles y garantizar que estén listos para el análisis o la toma de decisiones. Esto incluye:
- **Integración:** Combinar datos de diferentes fuentes.
- **Consistencia:** Proveer datos limpios y estructurados.
- **Eficiencia:** Reducir la complejidad del acceso y análisis.

### **ETL vs. ELT**
Aunque ETL es el enfoque tradicional, **ELT (Extract, Load, Transform)** es una variación que carga los datos directamente en el almacén de datos antes de transformarlos. Esto se utiliza especialmente en sistemas modernos basados en la nube.

## Consideraciones de ETL

### **Consideraciones Clave en un Proceso ETL**

El éxito de un proyecto ETL (Extract, Transform, Load) depende de la planificación cuidadosa, la comprensión de las necesidades del negocio y la calidad de la ejecución. Aquí tienes las principales consideraciones al implementar un proceso ETL:

### **1. Comprensión de los Requisitos**
- **Objetivos del negocio:** Define claramente qué se espera lograr con el proceso ETL (reportes, análisis, monitoreo, etc.).
- **Volumen de datos:** Considera la cantidad de datos a procesar y su frecuencia (diaria, semanal, en tiempo real).
- **Fuente de datos:** Identifica todas las fuentes de datos (bases de datos relacionales, APIs, archivos CSV, etc.) y su formato.

### **2. Calidad de los Datos**
- **Integridad de datos:** Verifica que los datos de las fuentes sean completos y precisos.
- **Consistencia:** Asegúrate de que los datos tengan un formato estándar (por ejemplo, fechas y monedas).
- **Manejo de datos erróneos:** Implementa estrategias para tratar datos faltantes, duplicados o corruptos.

### **3. Escalabilidad**
- **Crecimiento futuro:** Diseña el sistema para manejar un incremento en el volumen y variedad de datos.
- **Escalabilidad horizontal:** Utiliza herramientas capaces de procesar datos en paralelo para mantener el rendimiento.

### **4. Rendimiento**
- **Tiempo de procesamiento:** Minimiza el tiempo necesario para extraer, transformar y cargar datos, especialmente para procesos críticos.
- **Optimización de consultas:** Utiliza índices y otras técnicas para acelerar las operaciones en bases de datos.

### **5. Seguridad**
- **Encriptación:** Protege los datos sensibles durante la transferencia (en tránsito) y el almacenamiento (en reposo).
- **Control de acceso:** Implementa políticas de seguridad para limitar quién puede acceder a los datos y realizar cambios.
- **Regulaciones:** Cumple con normativas como GDPR, HIPAA o CCPA, según sea necesario.

### **6. Mantenimiento**
- **Monitoreo:** Configura alertas para detectar fallos en tiempo real.
- **Registro de errores:** Implementa un sistema de logging para rastrear problemas en el flujo de datos.
- **Actualizaciones:** Asegúrate de que las herramientas ETL puedan actualizarse sin interrupciones significativas.

### **7. Herramientas y Tecnología**
- **Elección de herramientas:** Decide entre herramientas comerciales (Informatica, Talend) o plataformas en la nube (AWS Glue, Google Dataflow).
- **Compatibilidad:** Asegúrate de que la herramienta seleccionada pueda conectarse a todas las fuentes de datos necesarias.

### **8. Transformaciones de Datos**
- **Complejidad:** Evalúa qué tan complejas son las transformaciones necesarias (filtros, agregaciones, cambios de formato).
- **Flexibilidad:** Diseña transformaciones modulares y reutilizables.
- **Pruebas:** Verifica que las transformaciones produzcan resultados correctos antes de cargarlas en el destino.

### **9. Procesos de Carga**
- **Tipo de carga:** Define si será completa o incremental (solo datos nuevos o modificados).
- **Manejo de fallos:** Implementa mecanismos para reiniciar cargas fallidas sin duplicar datos.
- **Orden de carga:** Asegúrate de que las dependencias entre tablas se respeten.

### **10. Documentación**
- **Mapeo de datos:** Documenta cómo se transforman los datos desde las fuentes hasta el destino.
- **Guías operativas:** Proporciona instrucciones claras para administrar y solucionar problemas del flujo ETL.
- **Versionado:** Registra cambios en el diseño del flujo para facilitar auditorías y mantenimiento.

### **11. Consideraciones Adicionales**
- **ETL vs. ELT:** Evalúa si un enfoque ELT podría ser más adecuado para el caso de uso específico.
- **Costos:** Considera los costos de licencias, infraestructura y mantenimiento de las herramientas ETL.
- **Pruebas:** Realiza pruebas exhaustivas antes de implementar en producción.

Al abordar cada una de estas áreas, puedes garantizar que el proceso ETL sea robusto, eficiente y alineado con las necesidades del negocio. 

## Servicios y herramientas para ETL

### **Servicios y Herramientas para ETL**

El éxito de los procesos de ETL depende en gran medida de las herramientas y servicios que facilitan la extracción, transformación y carga de datos. A continuación, se describen las principales opciones divididas en categorías clave:

### **1. Herramientas ETL Tradicionales**
Estas herramientas están diseñadas específicamente para procesos ETL en entornos locales o híbridos.

- **Informatica PowerCenter**  
  Una de las herramientas más populares y robustas para ETL. Ofrece funciones avanzadas para transformar y gestionar grandes volúmenes de datos.

- **Talend Data Integration**  
  Plataforma de código abierto que incluye conectores para diversas fuentes de datos y capacidades avanzadas de transformación.

- **IBM DataStage**  
  Herramienta empresarial para grandes proyectos ETL, ideal para integraciones complejas y procesamiento de big data.

- **Microsoft SQL Server Integration Services (SSIS)**  
  Parte de Microsoft SQL Server, ofrece capacidades ETL para usuarios que trabajan con bases de datos SQL.

### **2. Herramientas ETL en la Nube**
Diseñadas para aprovechar la escalabilidad y flexibilidad de la nube, estas herramientas integran flujos ETL con servicios en la nube.

- **AWS Glue**  
  Servicio ETL totalmente administrado en AWS que permite ejecutar transformaciones de datos basadas en Python (PySpark).

- **Google Cloud Dataflow**  
  Ofrece capacidades de procesamiento en tiempo real y por lotes para pipelines de datos en la nube de Google.

- **Azure Data Factory**  
  Solución de integración de datos de Microsoft Azure que permite mover y transformar datos entre múltiples orígenes y destinos.

- **Snowflake + Matillion**  
  Snowflake es un almacén de datos en la nube, y Matillion es una herramienta ETL diseñada específicamente para integrarse con Snowflake.

### **3. Herramientas Open Source**
Opciones gratuitas que ofrecen flexibilidad y personalización para desarrolladores y pequeños equipos.

- **Apache Nifi**  
  Herramienta de integración de datos visual para flujos ETL. Ideal para flujos en tiempo real y automatización.

- **Apache Airflow**  
  Aunque no es una herramienta ETL tradicional, permite programar y orquestar pipelines ETL.

- **Pentaho Data Integration (PDI)**  
  Herramienta de código abierto que proporciona un enfoque visual para construir y ejecutar flujos ETL.

### **4. Herramientas de Orquestación de Datos**
Estas herramientas gestionan pipelines de datos más complejos, combinando ETL con otras funcionalidades.

- **Fivetran**  
  Automatiza la extracción de datos y los carga en destinos populares como BigQuery, Snowflake o Redshift.

- **Stitch**  
  Una herramienta ligera para mover datos rápidamente hacia almacenes de datos.

- **dbt (Data Build Tool)**  
  Aunque es más una herramienta ELT, ayuda a gestionar transformaciones SQL en almacenes de datos modernos.

### **5. Herramientas de Big Data y Procesamiento en Tiempo Real**
Diseñadas para manejar volúmenes masivos de datos y ofrecer capacidades en tiempo real.

- **Apache Spark**  
  Plataforma de análisis distribuido que permite realizar ETL a gran escala con alta velocidad.

- **Kafka + Kafka Streams**  
  Para flujos ETL en tiempo real con mensajes entre sistemas distribuidos.

- **Databricks**  
  Plataforma basada en Apache Spark que permite construir pipelines ETL avanzados.

### **6. Servicios ETL Especializados**
Soluciones diseñadas para necesidades específicas de sectores o casos de uso.

- **Alteryx**  
  Enfocada en la integración y análisis de datos, ideal para usuarios que requieren análisis avanzado sin codificación extensa.

- **SAP Data Services**  
  Herramienta ETL orientada a la integración de datos empresariales en entornos SAP.

- **Boomi (Dell Boomi)**  
  Solución basada en la nube que facilita la integración de datos entre aplicaciones SaaS y locales.

### **7. Factores para Elegir una Herramienta ETL**
1. **Volumen de datos:** ¿Es un entorno pequeño o big data?  
2. **Compatibilidad:** ¿Se conecta fácilmente a tus fuentes y destinos?  
3. **Rendimiento:** ¿Puede manejar la frecuencia y carga de datos?  
4. **Facilidad de uso:** ¿Es necesario programar o es más visual?  
5. **Costo:** Considera herramientas gratuitas frente a licencias comerciales.  
6. **Escalabilidad:** ¿Se adapta al crecimiento futuro de los datos?  

**Lecturas recomendadas**

[pandas - Python Data Analysis Library](https://pandas.pydata.org/)

[Curso Básico de Manipulación y Transformación de Datos con Pandas y NumPy - Platzi](https://platzi.com/cursos/pandas-numpy/)

[Tutorial desde cero para dominar Pandas [Python]](https://platzi.com/blog/pandas/)

[Pentaho from Hitachi Vantara - Browse Files at SourceForge.net](https://sourceforge.net/projects/pentaho/files/)

[Pentaho Data Integration - Hitachi Vantara Lumada and Pentaho Documentation](https://help.hitachivantara.com/Documentation/Pentaho/7.0/0D0/Pentaho_Data_Integration)

[Descargar Pentaho](https://pentaho.com/pentaho-developer-edition/)

## Sources

El término "sources" en el contexto de ETL y procesamiento de datos se refiere a las **fuentes de datos**. Estas fuentes son los orígenes de la información que se extrae para ser procesada y transformada dentro de los sistemas ETL. Pueden provenir de diferentes tipos de sistemas o bases de datos, tanto estructurados como no estructurados. A continuación, te explico algunos conceptos clave sobre las fuentes de datos en un proceso ETL, en español:

### Fuentes de Datos en un Proceso ETL:
1. **Bases de Datos Relacionales (RDBMS)**:
   Las bases de datos como **MySQL**, **PostgreSQL**, **Oracle** o **SQL Server** suelen ser fuentes comunes para los procesos ETL. Los datos extraídos de estas fuentes generalmente están estructurados y organizados en tablas, lo que facilita su extracción.

2. **Archivos de Texto y CSV**:
   Archivos planos como **CSV**, **JSON**, **XML** o **TXT** son comunes en muchos procesos ETL. Estos archivos pueden contener datos en formato tabular o jerárquico, pero requieren procesamiento para ser transformados en un formato adecuado para el análisis.

3. **APIs**:
   Las **APIs (Interfaces de Programación de Aplicaciones)** permiten acceder a datos de aplicaciones externas, como redes sociales, plataformas de comercio electrónico o sistemas de información. Los datos extraídos a través de una API generalmente están en formato JSON o XML.

4. **Sistemas de Almacenamiento en la Nube**:
   Fuentes como **Amazon S3**, **Google Cloud Storage**, o **Azure Blob Storage** son muy utilizadas, ya que permiten almacenar grandes volúmenes de datos no estructurados que se pueden extraer para su procesamiento ETL.

5. **Sistemas NoSQL**:
   Bases de datos NoSQL como **MongoDB**, **Cassandra**, o **CouchDB** son comunes cuando los datos no siguen una estructura rígida de tablas y relaciones. Estos sistemas pueden ser fuentes para datos semi-estructurados o no estructurados.

6. **Flujos de Datos en Tiempo Real**:
   Los sistemas que generan datos en tiempo real, como los sensores IoT, o las plataformas de streaming como **Apache Kafka**, pueden ser fuentes de datos para procesos ETL de transmisión continua (streaming ETL), donde los datos son procesados en tiempo real en lugar de ser extraídos en lotes.

### Explicación de las Fuentes de Datos en el Contexto ETL:
Las **fuentes de datos** son un componente crucial en los procesos ETL (Extract, Transform, Load), ya que son el primer paso para obtener la información necesaria para su análisis o almacenamiento. En este proceso:

- **Extract (Extracción)**: Se extraen los datos de las fuentes. Aquí es donde se encuentran las "sources" que alimentan el proceso.
- **Transform (Transformación)**: Los datos extraídos se limpian, se validan y se estructuran según sea necesario. 
- **Load (Carga)**: Finalmente, los datos transformados se cargan en un sistema de destino, como un **Data Warehouse** o un **Data Lake**, donde pueden ser utilizados para análisis y reportes.

### Ejemplos de Fuentes Comunes en la Industria:
- En la **industria financiera**, las fuentes de datos pueden incluir bases de datos de transacciones bancarias, sistemas de tarjetas de crédito, APIs de pagos, etc.
- En la **industria de salud**, las fuentes pueden ser registros médicos electrónicos, dispositivos médicos, o incluso datos de investigación científica.
- En la **industria petrolera**, las fuentes de datos pueden ser sensores en los pozos de petróleo, sistemas de monitoreo de maquinaria, y bases de datos que contienen registros de producción.

En resumen, las **sources** en un proceso ETL son todas aquellas plataformas, bases de datos, archivos, APIs o sistemas que proveen los datos que se van a extraer, transformar y cargar en el flujo de trabajo ETL. Estas fuentes son fundamentales porque determinan la calidad, cantidad y tipo de datos que se utilizarán en los análisis posteriores.

## Configuración de base de datos source y entorno para ETL en Python

¡Hola! En esta clase conocerás cómo configurar una base de datos con SQL, que será una de las 3 fuentes para extraer datos en el proyecto de ETL. Las otras dos fuentes son un archivo JSON y otro CSV que conocerás en clases posteriores.

Además, conocerás cómo conectarte a esta base de datos OLTP con un software de administración de bases de datos. Puede ser DataSpell, DBeaver o el de tu preferencia.

Te sugiero usar DataSpell. Más adelante de este tutorial verás cómo configurarlo.

💡Algo que tenemos que destacar es que la base de datos SQL source no se tendría que crear en un proceso de ETL. Esta base de datos ya estaría creada en algún lado de la infraestructura de los sistemas y aplicaciones de la empresa donde estés colaborando.

En este caso lo estamos haciendo por fines educativos para que tengas una base de datos de donde tomar datos y conozcas el proceso de extracción.

Para la configuración de nuestra base de datos source usaremos PostgreSQL. Podemos utilizarlo de dos formas, una instalación local de PostgreSQL o una configuración por Docker. Te sugiero hacerlo por Docker.

### 1. Crear container en Docker

Recordemos que Docker es un entorno de gestión de contenedores, de manera que usaremos una imagen base con toda la configuración que requerimos sin instalar necesariamente en nuestra máquina. Solo utilizando los recursos del sistema para correr dicha imagen, algo similar a una máquina virtual.

Por ahora, solo necesitas haber tomado el [Curso de Python: PIP y Entornos Virtuales](https://platzi.com/cursos/python-pip/ "Curso de Python: PIP y Entornos Virtuales") para conocer lo esencial de cómo usar esta herramienta con Python. En ese curso encontrarás la[ clase para saber cómo instalarlo en tu computador](https://platzi.com/clases/4261-python-pip/55136-instalacion-de-docker/ " clase para saber cómo instalarlo en tu computador").

Una vez que tengas instalado Docker en tu computador, ejecuta este comando en tu terminal:

WSL 2, Linux o macOS

```bash
sudo docker run -d --name=postgres -p 5432:5432 -v postgres-volume:/var/lib/postgresql/data -e POSTGRES_PASSWORD=mysecretpass postgres
```

Windows

```bash
docker run -d --name=postgres -p 5432:5432 -v postgres-volume:/var/lib/postgresql/data -e POSTGRES_PASSWORD=mysecretpass postgres
```

Como podrás notar, en este comando se específico lo siguiente para la creación de la base de datos con Docker:

- Nombre del container: `--name=postgres`
- Puerto a compartir con la máquina local: `-p 5432:5432`
- Volumen para el manejo de disco e información: `-v postgres-volume:/var/lib/postgresql/data`
- Password en PostgreSQL: `POSTGRES_PASSWORD=mysecretpass`

### 1.5 Instalación local de PostgreSQL (opcional)

De no usar Docker podrías ver la clase del curso de PostgreSQL en donde aprendes a instalarlo localmente, pero te sugiero intentarlo con Docker ya que puede agilizar tu flujo de trabajo. 😉

### 2. Validar container creado

Una vez que hayas creado el container de Docker usa el comando `docker ps` en tu terminal. Podrás ver todos los contenedores que se encuentran en ejecución actualmente y una descripción.

Deberás ver la IMAGE postgres.

![IMAGE postgres](images/etl01.png)

### 3. Configurar DataSpell

Para conectarte a la base de datos usarás un software de administración de bases de datos. Existen varios que puedes utilizar. Para el seguimiento del curso te sugiero utilizar **DataSpell** o, en su defecto, **DBeaver**.

DataSpell es un **IDE** completo para ciencia de de datos donde, además de conectarte y hacer consultas a bases de datos, podrás ejecutar Jupyter Notebooks. ¡Todo en el mismo lugar! 💪🏽

![DataSpell](images/dataspell.png)

💡 Una de sus desventajas es que es de pago, pero tiene un período de prueba de 30 días para que lo pruebes con este curso. Además existen ciertas opciones para obtener [licencias para estudiantes de bachillerato y universidad](https://www.jetbrains.com/community/education/#students "licencias para estudiantes de bachillerato y universidad").

⚠️🦫 En caso de que decidas usar DBeaver en lugar de DataSpell, utiliza tu entorno local de Jupyter Notebooks con Anaconda para la ejecución del código Python de las siguientes clases. 🐍

### Instalación de DataSpell

1. Para instalar DataSpell ve a [su sitio web aquí](https://www.jetbrains.com/dataspell/ "su sitio web aquí") y descarga la versión para tu sistema operativo.📥

3. Instálalo siguiendo las instrucciones que te aparezcan en el instalador.

⚠️ Cuando te solicite actualizar PATH Variable acepta marcando la opción que te indique. Esto es para evitar errores de ambientes en el futuro. En Windows se ve así:

![PATH](images/PATH.png)

Al finalizar te pedirá reiniciar el computador:

![restar dataspell](images/restar_dataspell.png)

4. Abre DataSpell ya que se haya instalado. Al hacer esto por primera vez te pedirá iniciar sesión. Elige la versión free trial registrando tu cuenta para ello.

5. Una vez que tengas tu cuenta configurada te pedirá elegir un intérprete de Python 🐍.

Previamente deberás tener instalado **Anaconda** en tu sistema operativo. Te recomiendo que crees un **ambiente de Anaconda** (**Conda environment**) único para el proyecto del curso. Llama al ambiente `fundamentos-etl`.

Elige el ambiente de Anaconda que usarás para el proyecto y presiona el botón Launch DataSpell.

![Welcome to DataSpel](images/WelcometoDataSpell.png)

Elegir un intérprete de Anaconda servirá para ejecutar Jupyter Notebooks en DataSpell.

6. Crea un nuevo Workspace en **DataSpell**. Presiona el botón File en la barra superior y luego elige la opción New Workspace Directory.

![new workspace directory](images/new_workspace_directory.png)

Llama` fundamentos-etl` al workspace y presiona el botón azul **Create**.

![new etl workspace](images/new_etl_workspace.png)

### Elegir ambiente de WSL2 (opcional si usas WSL)

Si quieres usar DataSpell con tu entorno en Windows con WSL 2, deberás conectar DataSpell al ambiente de Anaconda que tenga tu WSL.🐍

0. Crea un ambiente de Anaconda en tu WSL dedicado al proyecto de tu curso si todavía no lo has hecho. Llámalo fundamentos-etl

`conda create --name fundamentos-etl python=3.9`

1. Después ve a DataSpell en su parte inferior donde aparece el intérprete. Presiona la dirección que aparece y elige la opción **Interpreter Settings**.

![elegir interprete dataspell](images/elegir_interprete_dataspell.png)

2. Escoge el workspace `fundamentos-etl` creado anteriormente en DataSpell.

⚠️OJO: el workspace y el Anaconda Environment no son lo mismo. El Anaconda Environment lo vamos a cargar dentro del Workspace de DataSpell.

Después presiona el botón **Add Interpreter** e inmediatamente selecciona la opción **On WS**L.

![elegir_interprete_wsl_dataspell.png](images/elegir_interprete_wsl_dataspell.png)

3. Elige la distribución de Linux a usar y da clic en el botón Next cuando aparezca el mensaje "Instrospection completed succesfully!

![ws instrospection](images/wsl instrospection.png)

4. Elige el intérprete a usar. Este puede ser un Virtualvenv Environment, el System Interpreter o un Conda Environment. Elige la opción de Conda Environment.

![select interpreter.png](images/select_interpreter.png)

5. Mara la casilla **Use existing environment**. Elige el Conda Environment de WSL que usarás para tu proyecto. Anteriormente debiste crearlo desde tu terminal en WSL y llamarlo `fundamentos-etl`.

Finalmente, presiona el botón azul **Create**.

![conda fundamentos etl environment](images/conda-fundamentos-etl-environment.png)

6. Para terminar el proceso presiona el botón azul OK en la parte inferior.

![workspace](images/workspace.png)

7. Listo, ya deberá aparecer tu entorno de Anaconda en WSL cargado en la parte inferior de DataSpell.

![ambiente cargado](images/ambiente-cargado.png)

⚠️Si te aparece un error que indique que el ambiente no puede ser usado como el intérprete del workspace es porque estás intentando cargar el ambiente en el workspace general y no en un workspace de DataSpell que creaste.

[Aquí](https://www.jetbrains.com/help/dataspell/using-wsl-as-a-remote-interpreter.html "Aquí") encuentras la guía oficial de cómo conectar tu DataSpell al intérprete de Python o Anaconda en WSL, por si necesitas aprender a configurarlo a detalle.

Recuerda que otra alternativa en Windows es instalar [Anaconda para Windows](https://www.anaconda.com/products/distribution "Anaconda para Windows") y conectar DataSpell directamente a esta versión.

### 4. Conexión a la base de datos PostgreSQL

Sigue estos pasos para conectarte a la base de datos postgres desde DataSpell.

1. Abre DataSpell en tu computador.

![data spell](images/dataspells.png)

2. Ve a la pestaña de **Database** y en ella da clic en el **botón de signo de +**.

![database dataspell.png](images/database_dataspell.png)

3. Selecciona la opción de **Data Source** y dentro del menú desplegable elige la opción de **PostgreSQL**.

![workspace](images/workspace25.png)

4. Introduce los datos siguientes en la conexión:

- **Name**: local_postgres
- **Host**: localhost
- **Port**: 5432
- **User**: postgres
- **Database**: postgres
- **Url (opcional)**: jdbc:postgresql://localhost:5432/postgres
- **Password**: mysecretpass

5. Da clic en el botón de Test Connection para probar la conexión. Puede que te solicite actualizar unos drivers, acéptalos. Una vez que indique que la conexión es exitosa, da clic en el botón OK.

![etl 02](images/etl02.png)

6. Listo, ya tienes tu base de datos conectada en DataSpell.

![postgres data spell](images/postgresdataspell.png)

### 4. Cargar datos en la base de datos Postgres

Dentro de DataSpell, ya con la conexión a la base de datos previamente creada, ejecutarás el script ***postgres_public_trades.sql***.

Descárgalo [aquí de Google Drive](https://drive.google.com/file/u/2/d/19U7l0kp3mEh8SYYG6BjoDp0kVPYWDsqI/view?usp=share_link "aquí de Google Drive"). 📥

⚠️Este archivo pesa cerca de 500 MB, por lo que puede demorar su descarga. Contiene la creación de una tabla llamada trades y los insert de registros de la tabla.

⚠️Es posible que al intentar correr este script en **DBeaver** no sea posible por falta de memoria. Te sugerimos cortarlo en varias partes y cargar cada script independientemente.

![etl 03](images/etl031.png)

Una vez descargado el archivo **postgres_public_trades.sql** sigue estos pasos para cargar los datos con DataSpell:

1. Da clic derecho sobre la base de datos de PostgreSQL.

![etl 04](images/etl04.png)

2. Posteriormente da clic en SQL Script y luego en Run SQL Scripts.

![dataspell run sql script](images/dataspell_run_sql_script.png)

3. Ubica el script descargado dentro de tu computador y da clic en OK.

![etl 06](images/etl06.png)

⚠️La creación de la tabla y la carga de datos puede demorar cerca de 15-20 minutos en DataSpell.

![script sql done](images/script_sql_done.png)

### 5. Prueba la tabla trades

Una vez terminada la ejecución del script, consulta la tabla Trades ya cargada. Abre el editor de queries desde tu base de datos en DataSpell e ingresa la siguiente consulta:

`SELECT * FROM trades;`

![etl 07](images/etl07.png)

¡Listo! Ya tienes lo esencial para comenzar a extraer datos de una base de datos OLTP y correr tus notebooks de Python.

Avanza a la siguiente clase. ⚙️

## Extracción de datos con Python y Pandas

La **extracción de datos** con Python y Pandas es una práctica común en el análisis de datos y en procesos ETL (Extract, Transform, Load). **Pandas** es una biblioteca poderosa que permite manipular y analizar datos estructurados fácilmente. A continuación, se describen los métodos más utilizados para extraer datos con Pandas:

---

## 1. **Extracción desde archivos comunes**

### a) Archivos CSV
```python
import pandas as pd

# Cargar un archivo CSV
df = pd.read_csv('archivo.csv')

# Mostrar las primeras filas
print(df.head())
```

### b) Archivos Excel
```python
# Cargar un archivo Excel
df = pd.read_excel('archivo.xlsx', sheet_name='Hoja1')

# Mostrar resumen de datos
print(df.info())
```

### c) Archivos de texto delimitados
```python
# Cargar archivo con delimitadores personalizados (ejemplo: tabulación)
df = pd.read_csv('archivo.txt', delimiter='\t')

# Mostrar estadísticas descriptivas
print(df.describe())
```

---

## 2. **Extracción desde Bases de Datos**

Pandas puede conectarse a bases de datos relacionales como **MySQL**, **PostgreSQL** y otras utilizando bibliotecas como `SQLAlchemy`.

### Ejemplo con SQLite
```python
import sqlite3

# Conectar a la base de datos
conn = sqlite3.connect('base_de_datos.db')

# Ejecutar una consulta y cargar los datos en un DataFrame
query = "SELECT * FROM tabla"
df = pd.read_sql_query(query, conn)

# Cerrar conexión
conn.close()

print(df)
```

### Ejemplo con MySQL y SQLAlchemy
```python
from sqlalchemy import create_engine

# Crear conexión
engine = create_engine('mysql+pymysql://usuario:contraseña@host/nombre_base_datos')

# Ejecutar consulta
query = "SELECT * FROM tabla"
df = pd.read_sql(query, engine)

print(df)
```

---

## 3. **Extracción desde APIs**

Pandas puede trabajar con datos obtenidos de APIs, que usualmente están en formato **JSON**.

### Ejemplo con `requests`
```python
import pandas as pd
import requests

# Realizar la solicitud
response = requests.get('https://api.ejemplo.com/data')
data = response.json()

# Convertir a DataFrame
df = pd.json_normalize(data)

print(df.head())
```

---

## 4. **Extracción desde fuentes en la nube**

### a) Desde Amazon S3
```python
import boto3
import pandas as pd

# Configurar cliente S3
s3 = boto3.client('s3')

# Descargar archivo
s3.download_file('mi-bucket', 'ruta/al/archivo.csv', 'archivo_local.csv')

# Leer el archivo descargado
df = pd.read_csv('archivo_local.csv')

print(df)
```

### b) Desde Google Sheets
Utilizando la API de Google Sheets.
```python
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configurar las credenciales
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('credenciales.json', scope)
client = gspread.authorize(credentials)

# Obtener hoja de cálculo
sheet = client.open("nombre_hoja").sheet1

# Convertir a DataFrame
data = sheet.get_all_records()
df = pd.DataFrame(data)

print(df.head())
```

---

## 5. **Extracción desde flujos en tiempo real**

Pandas no está diseñado para datos en tiempo real, pero puede integrarse con herramientas como **Kafka** o **Spark** para leer datos en tiempo real y transformarlos.

---

## Recomendaciones:
1. **Validación de datos:** Verificar datos faltantes o inconsistencias después de la extracción usando métodos como `df.isnull().sum()`.
2. **Eficiencia:** Si trabajas con datos grandes, considera usar `chunksize` al leer archivos o bases de datos.
   ```python
   for chunk in pd.read_csv('archivo_grande.csv', chunksize=1000):
       print(chunk.head())
   ```

Con estas técnicas, puedes extraer datos desde diversas fuentes y trabajar con ellos eficientemente en tus proyectos de ingeniería de datos. 😊