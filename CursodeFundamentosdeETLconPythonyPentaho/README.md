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

**Lecturas recomendadas**

[template_ETL_OEC.ipynb - Google Drive](https://drive.google.com/file/d/1P5kQo5_0bkzLNakbBwlf-9evT9PYtae_/view?usp=share_link)

Curso de Python: PIP y Entornos Virtuales - Platzi

[JetBrains DataSpell: The IDE for Data Scientists](https://www.jetbrains.com/dataspell/)

[country_data.json - Google Drive](https://drive.google.com/file/d/19QM_nHhUZ4s3yZcV7ePis5mD1dTtCl9Z/view?usp=share_link)

[hs_codes.csv - Google Drive](https://drive.google.com/file/d/1C6EwxoQmROiC27gvTsUNBVPhrtfH8mDI/view?usp=share_link)

[postgres_public_trades.sql - Google Drive](https://drive.google.com/file/d/19U7l0kp3mEh8SYYG6BjoDp0kVPYWDsqI/view?usp=share_link)

[guia_ETL_OEC.ipynb - Google Drive](https://drive.google.com/file/d/1LjmlMnpajBsNnTqWozh8T-5NpbetIzO_/view?usp=share_link)

## Transformación

En el contexto de **ETL (Extract, Transform, Load)**, la **Transformación** es la etapa intermedia del proceso. Su propósito principal es convertir los datos extraídos desde las fuentes en un formato adecuado para ser almacenados y utilizados en el sistema de destino, como un Data Warehouse.  

### Principales tareas en la Transformación:
1. **Limpieza de datos**:  
   - Eliminar duplicados.
   - Gestionar valores faltantes.
   - Corregir errores en los datos.

2. **Estandarización**:  
   - Asegurar que los datos tienen el mismo formato y estructura.
   - Por ejemplo, unificar formatos de fechas o convertir unidades.

3. **Enriquecimiento de datos**:  
   - Combinar datos de múltiples fuentes para agregar más contexto o información útil.

4. **Cálculos y derivaciones**:  
   - Crear nuevas columnas o métricas a partir de los datos existentes, como calcular ingresos netos o márgenes de ganancia.

5. **Filtrado**:  
   - Seleccionar solo los datos relevantes o necesarios para la aplicación objetivo.

6. **Validación**:  
   - Verificar que los datos transformados cumplen con las reglas de negocio y estándares requeridos.

7. **Agrupación y agregación**:  
   - Resumir datos, como calcular promedios, totales, o conteos por categorías.

### Ejemplo en Python con Pandas:
Supongamos que extraemos datos de ventas de un archivo CSV y necesitamos realizar transformaciones básicas.

```python
import pandas as pd

# Cargar datos
data = pd.read_csv("ventas.csv")

# Limpieza: Eliminar duplicados
data = data.drop_duplicates()

# Estandarización: Convertir fechas al mismo formato
data['fecha'] = pd.to_datetime(data['fecha'], format='%Y-%m-%d')

# Enriquecimiento: Calcular total de ventas
data['total_ventas'] = data['precio_unitario'] * data['cantidad']

# Filtrar: Seleccionar datos relevantes
data = data[data['total_ventas'] > 1000]

print(data.head())
```

En este ejemplo, se realizan tareas comunes de transformación antes de cargar los datos transformados en un destino.  

La transformación es crucial porque asegura que los datos sean consistentes, precisos y útiles para el análisis.

## Transformación de datos de transacciones

La **transformación de datos de transacciones** es un paso clave dentro de un proceso de **ETL** (Extracción, Transformación y Carga) en el que los datos se procesan para que sean más útiles y adecuados para su análisis posterior. En el contexto de las transacciones financieras o de ventas, este paso implica la conversión de datos crudos provenientes de diferentes fuentes en un formato más estandarizado, limpio y estructurado. Aquí te dejo algunos aspectos clave y ejemplos de cómo se realiza la transformación de datos de transacciones:

### 1. **Limpieza de Datos**
   La limpieza es fundamental para asegurar que no haya errores en los datos antes de cargarlos a la base de datos o al sistema de análisis.

   - **Eliminar registros duplicados**: Si tienes registros de transacciones duplicados, necesitarás eliminarlos.
   - **Rellenar valores nulos**: Algunas transacciones pueden tener valores faltantes, como un monto o una fecha. Dependiendo de las reglas del negocio, podrías decidir rellenar estos valores con un valor predeterminado o eliminarlos.
   - **Formato de fechas**: Es posible que las fechas de las transacciones vengan en diferentes formatos (por ejemplo, `DD/MM/YYYY` o `MM-DD-YYYY`). Se deben estandarizar en un formato único.

   **Ejemplo en Python (Pandas)**:
   ```python
   import pandas as pd

   # Eliminar duplicados
   df = df.drop_duplicates(subset=["transaction_id"])

   # Rellenar valores nulos
   df["transaction_amount"].fillna(0, inplace=True)

   # Convertir fechas
   df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d")
   ```

### 2. **Agregación de Datos**
   Las transacciones a menudo deben agregarse para obtener métricas clave como el total de ventas por día, el total de transacciones por usuario, etc. Esto se hace mediante operaciones como la suma, el promedio, el conteo, etc.

   - **Total de ventas diarias**: Si cada transacción tiene un monto asociado, puedes agregar las ventas por día.
   - **Transacciones por usuario**: Puedes contar el número de transacciones realizadas por cada cliente o usuario.

   **Ejemplo en Python (Pandas)**:
   ```python
   # Total de ventas por día
   df_daily_sales = df.groupby("transaction_date")["transaction_amount"].sum().reset_index()

   # Número de transacciones por cliente
   df_transactions_per_user = df.groupby("user_id")["transaction_id"].count().reset_index()
   ```

### 3. **Normalización y Estandarización**
   Los datos de transacciones pueden tener diferentes unidades o escalas. Es importante estandarizar estos valores para que sean consistentes.

   - **Normalización de montos**: Si tienes transacciones en diferentes monedas, deberías convertirlas a una moneda común.
   - **Transformar categorías**: Las categorías de productos o servicios pueden tener diferentes etiquetas (por ejemplo, "Electrónica", "Electrodomésticos", "Tech"). Puedes agruparlos bajo categorías estandarizadas.

   **Ejemplo en Python (Pandas)**:
   ```python
   # Convertir montos a una moneda común (suponiendo una tasa de cambio de 1 USD = 0.85 EUR)
   df["transaction_amount_usd"] = df["transaction_amount"] * 0.85

   # Estandarizar categorías de productos
   category_map = {"Electrodomésticos": "Electronics", "Tech": "Electronics"}
   df["product_category"] = df["product_category"].replace(category_map)
   ```

### 4. **Cálculo de Métricas Derivadas**
   Las métricas derivadas son cálculos adicionales basados en los datos de transacciones que pueden ayudar a tomar decisiones o hacer análisis.

   - **Monto de transacciones por usuario**: Calcular cuánto ha gastado cada usuario.
   - **Margen de beneficio**: Si tienes datos sobre el costo y el precio de los productos, puedes calcular el margen de beneficio.

   **Ejemplo en Python (Pandas)**:
   ```python
   # Calcular el monto total por usuario
   df_user_spending = df.groupby("user_id")["transaction_amount"].sum().reset_index()

   # Calcular margen de beneficio
   df["profit_margin"] = (df["transaction_amount"] - df["cost_amount"]) / df["transaction_amount"]
   ```

### 5. **Enriquecimiento de Datos**
   A veces es necesario enriquecer los datos de transacciones con información adicional que provenga de otras fuentes. Esto podría incluir detalles sobre el usuario, ubicación, productos o promociones.

   - **Datos de cliente**: Puedes agregar información sobre los clientes, como el nombre, la ubicación o su nivel de fidelidad.
   - **Categorías de productos**: Si tienes una lista de productos con su categoría, puedes añadirla a cada transacción.

   **Ejemplo en Python (Pandas)**:
   ```python
   # Suponiendo que tienes un DataFrame con información de clientes
   df_customers = pd.DataFrame({"user_id": [1, 2], "customer_name": ["Alice", "Bob"]})

   # Unir la información de los clientes con las transacciones
   df = pd.merge(df, df_customers, on="user_id", how="left")
   ```

### 6. **Formato de Salida**
   Finalmente, los datos de transacciones transformados deben estar en el formato adecuado para su almacenamiento o análisis posterior. Pueden almacenarse en bases de datos, archivos CSV, formatos como JSON o Parquet, entre otros.

   **Ejemplo en Python (Pandas)**:
   ```python
   # Guardar los datos transformados en un archivo CSV
   df.to_csv("transacciones_transformadas.csv", index=False)
   ```

### Resumen del Proceso de Transformación de Datos de Transacciones:
1. **Limpieza de datos**: Eliminar duplicados, rellenar valores nulos, convertir fechas.
2. **Agregación de datos**: Sumar transacciones por fecha, contar transacciones por usuario.
3. **Normalización**: Convertir unidades (como moneda) y estandarizar categorías.
4. **Cálculo de métricas derivadas**: Calcular métricas adicionales como el gasto total por cliente o el margen de beneficio.
5. **Enriquecimiento de datos**: Agregar datos adicionales como la información del cliente.
6. **Exportación y almacenamiento**: Guardar los datos en el formato deseado (CSV, base de datos, etc.).

Este proceso de transformación es clave para preparar los datos para su análisis o para generar informes de negocio confiables.

**Lecturas recomendadas**

[country_data.json - Google Drive](https://drive.google.com/file/d/19QM_nHhUZ4s3yZcV7ePis5mD1dTtCl9Z/view?usp=share_link)

[template_ETL_OEC.ipynb - Google Drive](https://drive.google.com/file/d/1P5kQo5_0bkzLNakbBwlf-9evT9PYtae_/view?usp=share_link)

## Carga

En el contexto de un proceso ETL (Extract, Transform, Load), la **carga (Load)** se refiere al paso final en el que los datos transformados se cargan en el sistema de destino o en el almacén de datos (Data Warehouse). Este paso es crucial porque los datos deben estar disponibles para su análisis o uso en el sistema al que se dirigen.

### Tipos de carga de datos:
1. **Carga completa (Full Load)**:
   - Se cargan todos los datos desde cero. Es útil cuando el dataset es pequeño o cuando se necesita reemplazar completamente los datos existentes en el sistema de destino.
   
2. **Carga incremental (Incremental Load)**:
   - Solo se cargan los datos nuevos o modificados desde la última carga. Esto es eficiente para datasets grandes, ya que solo se agregan cambios, no todo el conjunto de datos.

3. **Carga en tiempo real (Real-Time Load)**:
   - Los datos se cargan en tiempo real, lo que significa que se actualizan casi inmediatamente después de que los datos son transformados. Es útil en aplicaciones que requieren datos actualizados constantemente.

### Consideraciones en la carga de datos:
- **Consistencia**: Asegurarse de que los datos en el sistema de destino estén completos y sin errores.
- **Performance**: La carga de grandes volúmenes de datos debe realizarse de manera eficiente para evitar bloqueos o cuellos de botella en el sistema de destino.
- **Automatización**: La carga debe ser un proceso automatizado, ejecutado a intervalos regulares o en función de cambios en los datos.

## Configuración de clúster en AWS Redshift

¡Hola, te doy la bienvenida! Me da gusto encontrarte en este curso. Soy [Alexis](https://platzi.com/profes/alexinaraujo/ "Alexis"), profesor de AWS en Platzi.

Te acompañaré para crear un clúster y su base de datos en **AWS Redshift** y un **bucket en S3**. Los usaremos para el target del ETL de este curso.

Antes de continuar, quiero recordarte que es importante que tengas el conocimiento de lo que aprendemos en estos dos cursos donde tengo el gusto de ser tu profe:

- [Curso de Introducción a AWS: Fundamentos de Cloud Computing](https://platzi.com/cursos/aws-fundamentos/ "Curso de Introducción a AWS: Fundamentos de Cloud Computing")
- [Curso de Introducción a AWS: Cómputo, Almacenamiento y Bases de Datos](https://platzi.com/cursos/aws-computo/ "Curso de Introducción a AWS: Cómputo, Almacenamiento y Bases de Datos")

Te servirá para que este proceso te sea mucho más familiar y para que ya tengas creada tu cuenta de AWS.

Si ya lo tienes sigue estos pasos para configurar el target de tu ETL.

### Entra a AWS Free Tier

1. Abre el sitio web de AWS Free Tier aquí.
⚠️Es importante recordar que el free tier tiene un período de 1 año para que no te cobre AWS.

2. Dentro del buscador de detalles de nivel gratuito busca “**redshift**”.

![aws free](images/aws_free.png)

3. Observa que podrás tener una prueba gratuita de 2 meses de AWS Redshift al elegir un nodo tipo DC2.Large

![redshift free](images/redshift_free.png)

![DC2 large aws](images/DC2_large_aws.png)

⚠️⚠️⚠️ Recuerda que es muy muy importante que elijas ese tipo de nodo DC2.Large al crear el clúster de AWS Redshift. Para que sea gratuito por 2 meses, de lo contrario se te harán cobros a tu tarjeta.

Ya que sabes esto, avanza a crear el clúster en los siguientes pasos.

### Creación de clúster en AWS Redshift

### 1. Inicia sesión en la consola de AWS

Da clic en iniciar sesión en la parte superior derecha o [aquí](http://console.aws.amazon.com/ "aquí").

![inicia sesion aws](images/inicia_sesion_aws.png)

### 2. Elige tu región

Después de iniciar sesión en tu cuenta de AWS elige la región donde crearás el clúster en la parte superior derecha de la consola de AWS.

![consola region aws](images/consola_region_aws.png)

Te sugiero elegir la región de Oregon (us-west-2) que es la que usamos en este curso, pero puedes seleccionar la que mejor te convenga. Solo recuerda cual es para que la pongas al cargar los datos desde la notebook de Python.

### 3. Buscar el servicio de Redshift

Busca “redshift” en el cuadro de búsqueda de servicios y da clic en la opción **Amazon Redshift** que aparece.

![buscar redshift](images/buscar_redshift.png)

Esto te llevará a la consola de Redshift.

![redshift console](images/redshift_console.png)

### 4. Ve al panel de clústeres

Dentro de la consola de Redshift ve al panel de la izquierda que se despliega con el ícono de las tres rayas horizontales. Selecciona la opción de **Clústeres**.

![clusteres redshift](images/clusteres_redshift.png)

### 5. Crea el clúster

Dentro del panel de clústeres da clic en el botón naranja **Crear clúster**.

![rear cluster redshift](images/rear_cluster_redshift.png)

En la configuración del clúster da nombre al clúster. Puedes llamarlo demo-platzi-curso-etl. Elige la prueba gratuita.

En el resumen de configuración calculada deberás ver que el tipo de nodo sea dc2.large. 
⚠️Recuerda que esto es importante de verificar para que sea gratuito el uso de tu Redshift durante 2 meses.

![free cluster](images/free_cluster.png)

Una vez seleccionados estos campos, desciende para establecer el usuario y contraseña del clúster de Redshift.

⚠️Recuerda guardar en un lugar seguro estas credenciales, como en un gestor de contraseñas, para que puedas conectarte al clúster.

Nombra **demoplatzi** al **Nombre de usuario** y pon una contraseña segura.

![admin pass cluster](images/admin_pass_cluster.png)

Finalmente, da clic en el botón naranja **Crear clúster**.

⌛Espera hasta que el estado de creación del clúster lo marque en verde como Available. Esto puede demorar varios minutos, para revisar el estado da clic en el botón refrescar de la flecha en círculo.

![cluster listo](images/cluster_listo.png)

### 6. Modificar las reglas de seguridad del clúster

Entra al clúster dando clic en el nombre dentro del panel de clústeres.

![cluster listo](images/cluster_listo1.png)

Entra a la pestaña de **Propiedades**.

![cluster listo 1](images/propiedades.png)

Baja hasta la sección de Configuración de **red** y **seguridad** y da clic sobre el **Grupo de seguridad de la VPC**.

![VPC sec](images/VPC_sec.png)

Selecciona el **ID del grupo de seguridad**.

![id grupo seguridad](images/id_grupo_seguridad.png)

Baja y da clic en **Editar reglas de entrada**.

![reglas_de_entrada.png](images/reglas_de_entrada.png)

Da clic en el botón inferior **Agregar regla**

![agregar regla](images/agregar_regla.png)

En tipo elige Redshift y en **origen** elige 0.0.0.0/0. Finalmente, da clic en el botón naranja **Guardar** reglas.

![nueva regla entrada](images/nueva_regla_entrada.png)

Regresa al panel del clúster a la zona de configuración de red y seguridad. Da clic en el botón editar.

![editar seguridad](images/editar_seguridad.png)

Desciende y en la parte inferior marca la casilla **Activar accesibilidad pública**. Da clic en el botón naranja **Guardar cambios**.

![red seguridad publica](images/red_seguridad_publica.png)

⚠️Esto es algo que no debe hacerse en entornos de producción. En este caso lo harás al ser un demo con fines educativos, para evitar complicaciones de configuración adicional de accesos de seguridad.

### 7. Conéctate a Redshift desde tu gestor de bases de datos.

Abre tu gestor, ya sea DataSpell, DBeaver o pgAdmin.

Dentro de tu gestor crea una nueva conexión a una base de datos de tipo Redshift. Es muy importante buscar la opción de Redshift.

En el caso de DataSpell ve a la pestaña de Database y en ella da clic en el botón de signo de +.

![database dataspell1](images/database_dataspell1.png)

Selecciona la opción de **Data Source** y dentro del menú desplegable busca y elige la opción de Amazon Redshift.

workspace.png
![workspace 1](images/workspace1.png)

Regresa al panel del administrador del clúster de Redshift en la consola de AWS y copia el punto de enlace.

![copiar punto enlace](images/copiar-punto-enlace.png)

Regresa a la interfaz de tu gestor de bases de datos e ingresa los los siguientes datos para conectar a la base de datos:

- **Host**: es la url del punto de enlace que copiaste eliminando la parte final “:5439/dev”. Tendrá una forma como [server.redshift.amazonaws.com](http://server.redshift.amazonaws.com/ "server.redshift.amazonaws.com")
- **Port**: 5439
- **User**: demoplatzi o el que hayas puesto.
- **Password**: la que le hayas puesto a tu clúster de AWS Redshift cuando lo creaste en el paso 5.
- **Database**: dev

![database con redshift](images/database_con_redshift.png)

Da clic en el botón de **Test Connection** o su correspondiente para probar la conexión. Si estás en DataSpell te pedirá actualizar unos drivers, dile que OK.

Una vez que indique que la conexión es exitosa, da clic en el **botón OK**, o en el botón disponible de otro gestor, para aceptar la conexión.

![conexion redshift ok](images/conexion_redshift_ok.png)

¡Listo tienes creado tu clúster con una base de datos dev en Redshift y te has conectado a ella! 🚀

![conexión redshift](images/conexión_redshift.png)

### Creación de bucket de almacenamiento en AWS S3

Para el ETL crearás un bucket en S3 donde temporalmente almacenarás unos archivos CSV donde guardarás los datos de las tablas que has creado en el proceso de transformación.

### 1. Entra al panel de S3 desde la consola de AWS.

Busca “**S3**” en el buscador de la consola y selecciona la opción de S3.

![s3](images/s3.png)

Da clic en el botón naranja **Crear bucket** en el panel de S3.

![panel s3](images/panel-s3.png)

Da un nombre al bucket. Este nombre debe ser único, ya que no puede haber más de un bucket de S3 con el mismo nombre. También asegúrate que la región de AWS sea la misma que hayas elegido para tu clúster de Redshift, en nuestro caso fue **us-west-2**.

![crear bucket 3](images/crear_bucket_s3.png)

Desciende hasta abajo y da clic en el botón naranja **Crear bucket**.

![boton crear bucket](images/boton-crear-bucket.png)

¡Listo, ya deberá aparecer que tienes tu bucket creado! 👏🏽

![bucket creado](images/bucket_creado.png)

Avanza a la siguiente clase para crear las tablas donde cargarás los datos de los archivos CSV que crearás durante el proceso de carga y donde configurarás ciertas variables de entorno por seguridad. ⚙️➡️

Recomiendo utilizar variables de entorno con nombres personalizados, ya que por ejemplo la variable de entorno USER es el nombre de usuario de tu sistema operativo en el caso de linux. Yo lo Personalicé así:

```bash
export AWS_ACCESS_KEY_ID=********
export AWS_SECRET_ACCESS_KEY=******
export AWS_HOST="tu host"
export AWS_DATABASE=dev
export AWS_USER=demoplatzi
export AWS_PASSWORD=******
```

## Carga de datos con Python

La **carga de datos con Python** es el paso final en un proceso de ETL (Extracción, Transformación y Carga). En este proceso, los datos ya transformados se insertan en el sistema de destino, como un Data Warehouse, una base de datos relacional o un servicio en la nube. Aquí te explico cómo realizar la carga utilizando Python y la librería `pandas` en combinación con `SQLAlchemy` y `psycopg2`.

### Pasos para cargar datos:

#### 1. **Preparar los datos**
Los datos deben estar en un formato estructurado (como un DataFrame de pandas) y listos para cargarse en la base de datos destino.

```python
import pandas as pd

# Datos de ejemplo
data = {
    'product_id': [1, 2, 3],
    'product_name': ['Laptop', 'Mouse', 'Keyboard'],
    'sales': [1200, 300, 450]
}

df = pd.DataFrame(data)
```

#### 2. **Configurar la conexión**
Usa `SQLAlchemy` para conectarte a la base de datos. Configura correctamente el **string de conexión** con las credenciales y el host de tu base de datos.

```python
from sqlalchemy import create_engine

# String de conexión para una base de datos PostgreSQL o Amazon Redshift
engine = create_engine("postgresql+psycopg2://username:password@host:port/database")
```

Reemplaza:
- `username`: Nombre de usuario
- `password`: Contraseña
- `host`: Dirección del servidor
- `port`: Puerto (por ejemplo, `5439` para Redshift)
- `database`: Nombre de la base de datos

#### 3. **Escribir los datos en la base**
Utiliza el método `to_sql` de pandas para insertar el DataFrame en una tabla.

```python
# Cargar los datos en la base de datos
df.to_sql(
    name='sales_data',         # Nombre de la tabla destino
    con=engine,                # Conexión a la base de datos
    if_exists='replace',       # Qué hacer si la tabla ya existe ('replace', 'append', 'fail')
    index=False                # Si no quieres cargar el índice del DataFrame como columna
)

print("Datos cargados correctamente.")
```

#### 4. **Verificar la carga**
Consulta la base de datos para asegurarte de que los datos se han cargado correctamente.

```python
# Leer los datos para verificar
df_loaded = pd.read_sql("SELECT * FROM sales_data", con=engine)
print(df_loaded)
```

### Ejemplo completo:

```python
import pandas as pd
from sqlalchemy import create_engine

# Preparar los datos
data = {
    'product_id': [1, 2, 3],
    'product_name': ['Laptop', 'Mouse', 'Keyboard'],
    'sales': [1200, 300, 450]
}
df = pd.DataFrame(data)

# Configurar conexión
engine = create_engine("postgresql+psycopg2://username:password@host:port/database")

# Cargar datos en la base de datos
df.to_sql(
    name='sales_data',
    con=engine,
    if_exists='replace',
    index=False
)

# Verificar los datos cargados
df_loaded = pd.read_sql("SELECT * FROM sales_data", con=engine)
print(df_loaded)
```

### Recomendaciones:
1. **Validación previa**: Antes de cargar los datos, revisa que no contengan valores nulos o inconsistencias.
2. **Estrategia de carga**:
   - `replace`: Sobrescribe la tabla si ya existe.
   - `append`: Agrega los datos nuevos sin eliminar los existentes.
3. **Manejo de errores**: Implementa bloques `try-except` para capturar errores en la conexión o la carga.

Con este enfoque puedes integrar datos a bases de datos de manera confiable en el contexto de un proceso ETL.

**Lecturas recomendadas**

[Examples of using the Amazon Redshift Python connector - Amazon Redshift](https://docs.aws.amazon.com/redshift/latest/mgmt/python-connect-examples.html)

[template_ETL_OEC.ipynb - Google Drive](https://drive.google.com/file/d/1P5kQo5_0bkzLNakbBwlf-9evT9PYtae_/view?usp=share_link)

[Curso de AWS Redshift para Manejo de Big Data - Platzi](https://platzi.com/cursos/redshift-big-data/)

[country_data.json - Google Drive](https://drive.google.com/file/d/19QM_nHhUZ4s3yZcV7ePis5mD1dTtCl9Z/view?usp=share_link)

[hs_codes.csv - Google Drive](https://drive.google.com/file/d/1C6EwxoQmROiC27gvTsUNBVPhrtfH8mDI/view?usp=share_link)

[postgres_public_trades.sql - Google Drive](https://drive.google.com/file/d/19U7l0kp3mEh8SYYG6BjoDp0kVPYWDsqI/view?usp=share_link)

[guia_ETL_OEC.ipynb - Google Drive](https://drive.google.com/file/d/1LjmlMnpajBsNnTqWozh8T-5NpbetIzO_/view?usp=share_link)

### Nota: Seguridad .env

Un archivo **`.env`** se utiliza para almacenar configuraciones sensibles o variables de entorno de un proyecto, como credenciales, configuraciones de conexión y otros parámetros que no deseas incluir directamente en tu código fuente. Es una práctica común para mantener tu código seguro y facilitar la configuración de entornos (desarrollo, pruebas, producción).

---

### ¿Qué puede contener un archivo `.env`?

Un archivo `.env` suele tener pares clave-valor separados por un signo igual `=`. Por ejemplo:

```plaintext
DB_HOST=localhost
DB_PORT=5432
DB_NAME=my_database
DB_USER=admin
DB_PASSWORD=securepassword
SECRET_KEY=mysecretkey123
DEBUG=True
```

---

### ¿Cómo usar un archivo `.env` en Python?

Python puede manejar archivos `.env` utilizando librerías como **`python-dotenv`**. Esta librería carga automáticamente las variables definidas en el archivo `.env` y las hace accesibles a través de `os.environ`.

#### 1. **Instalar `python-dotenv`**
Primero, instala la librería usando pip:

```bash
pip install python-dotenv
```

---

#### 2. **Crear un archivo `.env`**
Guarda tus credenciales y configuraciones en un archivo `.env` en el directorio raíz de tu proyecto:

```plaintext
DB_HOST=localhost
DB_PORT=5432
DB_NAME=my_database
DB_USER=admin
DB_PASSWORD=securepassword
```

---

#### 3. **Cargar y usar las variables en Python**

Carga las variables en tu código utilizando `python-dotenv` y el módulo `os`:

```python
from dotenv import load_dotenv
import os

# Cargar las variables del archivo .env
load_dotenv()

# Obtener las variables
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

# Usar las variables (ejemplo con SQLAlchemy)
from sqlalchemy import create_engine

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
print("Conexión exitosa.")
```

---

### Ventajas de usar `.env`:
1. **Seguridad**: Evita exponer credenciales sensibles en el código fuente.
2. **Flexibilidad**: Facilita la configuración de entornos diferentes (desarrollo, pruebas, producción).
3. **Mantenibilidad**: Centraliza las configuraciones del proyecto.

---

### Buenas prácticas:
1. **No incluir `.env` en el repositorio**: Agrega el archivo `.env` a tu `.gitignore` para evitar subirlo a GitHub o a cualquier otro sistema de control de versiones:
   ```plaintext
   # .gitignore
   .env
   ```
2. **Usar variables descriptivas**: Nombres claros como `DB_HOST` o `API_KEY` ayudan a entender su propósito.

Con estas configuraciones, puedes manejar datos sensibles de manera segura y profesional en tus proyectos Python.

## Estructurar sentence para carga de datos de S3 a Redshift

¡Hola! En la clase anterior definimos esta función de carga de archivos en nuestra notebook de Python:

![sentence](images/sentence.png)

💡Dentro de esta función esta la variable sentence, que es una secuencia de texto con una instrucción para copiar los registros de los archivos CSV, que están dentro del bucket de S3, hacia la base de datos dev que está en un schema.

Esta sentencia debe configurarse específicamente para el schema de tu base de datos y la dirección de tu bucket en S3.

Revisitemos tres componentes principales dentro de la sentencia:

![sentence partes](images/sentence_partes.png)

- **Schema en DB en Redshift**: es el schema de nuestra base de datos donde están las tablas a donde se copiarán los registros de los archivos CSV. Este debe ser remplazado por ese schema, que en tu caso debería ser public en lugar de `etl_test`

- **Nombre de bucket de S3**: dirección del bucket donde se guardan temporalmente los archivos CSV. En tu caso debe remplazarse `s3://platzi-etl/course_etl_target/{}` por el nombre único de tu bucket. Si tu bucket se llamara **nombre-unico-bucket-curso-platzi** en la sentencia debería estar `s3://nombre-unico-bucket-curso-platzi{}`

- **Región elegida de AWS**: es esa región donde creaste el clúster de Redshift. En caso de que no lo hayas hecho en la región `us-west-2` escribe el código de la región que elegiste.

Tu sentence debería verse de forma similar a la siguiente, pero recuerda que no será idéntica porque el nombre de tu bucket es único en el mundo:

```python
sentence = '''copy public.{} from 's3://nombre-unico-bucket-curso-platzi/{}' credentials 'aws_access_key_id={};aws_secret_access_key={}' csv delimiter '|' region 'us-west-2' ignoreheader 1'''.format(table_name, file_name, os.environ.get('AWS_ACCESS_KEY_ID'), os.environ.get('AWS_SECRET_ACCESS_KEY'))
```

¡Listo! Aplica estos ajustes a tu sentence y úsalos durante la próxima clase donde usaremos la función load_file() para cargar los registros en Redshift. ⚙️👏🏽

## Carga de datos: subida de archivos a AWS Redshift

Subir datos a **AWS Redshift** implica un proceso que usualmente consta de los siguientes pasos:  

1. **Preparar los datos**: Asegúrate de que los datos están en un formato compatible, como archivos CSV, JSON, Parquet, o Avro.  
2. **Subir los datos a S3**: AWS Redshift carga datos desde Amazon S3, así que primero necesitas colocar tus archivos en un bucket de S3.  
3. **Cargar datos a Redshift**: Usa la instrucción `COPY` para transferir los datos desde S3 a tu tabla de Redshift.

---

### Prerrequisitos
1. **Crear una base de datos y tablas en Redshift**:
   - Necesitas un clúster configurado y una tabla creada en Redshift donde cargar los datos.
2. **Configurar un bucket en S3**:
   - Define un bucket donde alojarás los datos que serán importados.
3. **Credenciales AWS**:
   - Necesitarás una IAM Role o Access Key/Secret Access Key con permisos para acceder a S3 y Redshift.

---

### Proceso detallado

#### 1. **Preparar los datos y subirlos a S3**

Puedes usar **AWS CLI** para subir datos a tu bucket de S3:
```bash
aws s3 cp /path/to/your/data.csv s3://your-bucket-name/data.csv
```

O, si estás usando Python, puedes emplear la librería `boto3`:
```python
import boto3

# Configurar el cliente de S3
s3 = boto3.client('s3')

# Subir archivo
s3.upload_file('data.csv', 'your-bucket-name', 'data.csv')
print("Archivo subido exitosamente a S3.")
```

---

#### 2. **Configurar conexión a Redshift**

Usa `psycopg2` o SQLAlchemy para conectarte al clúster de Redshift:
```python
from sqlalchemy import create_engine

# Datos de conexión
DATABASE = "your_database"
USER = "your_user"
PASSWORD = "your_password"
HOST = "your-cluster-endpoint"
PORT = "5439"

# Crear la conexión
engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
connection = engine.connect()
print("Conexión exitosa a Redshift.")
```

---

#### 3. **Crear la tabla en Redshift**

Define la estructura de tu tabla para que coincida con los datos que cargarás:
```sql
CREATE TABLE your_table (
    column1 VARCHAR(50),
    column2 INT,
    column3 DATE
);
```

---

#### 4. **Cargar datos desde S3 a Redshift**

Usa la instrucción `COPY` para transferir datos desde S3 a tu tabla en Redshift:
```python
# Comando SQL para copiar los datos
copy_command = """
    COPY your_table
    FROM 's3://your-bucket-name/data.csv'
    IAM_ROLE 'arn:aws:iam::your-account-id:role/your-redshift-role'
    FORMAT AS CSV
    IGNOREHEADER 1;
"""

# Ejecutar el comando
connection.execute(copy_command)
print("Datos cargados exitosamente en Redshift.")
```

---

### Consideraciones importantes
1. **Rol IAM**:
   - El rol asociado a tu clúster Redshift debe tener permisos para acceder al bucket S3.
   - Política mínima necesaria:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": "s3:*",
           "Resource": "arn:aws:s3:::your-bucket-name/*"
         }
       ]
     }
     ```

2. **Validación**:
   - Después de cargar los datos, verifica que se cargaron correctamente:
     ```sql
     SELECT COUNT(*) FROM your_table;
     ```

3. **Formato del archivo**:
   - Asegúrate de que el archivo en S3 está correctamente formateado y coincide con las columnas de la tabla.

4. **Errores comunes**:
   - **Permisos insuficientes**: Verifica que el rol IAM tiene acceso al bucket S3 y permisos COPY en Redshift.
   - **Formato incorrecto**: Si el archivo CSV tiene delimitadores inconsistentes, podrías recibir errores.

---

### Alternativa: Usar AWS Data Wrangler

La librería `awswrangler` simplifica el proceso de carga desde Pandas DataFrame a Redshift:
```python
import awswrangler as wr
import pandas as pd

# Crear DataFrame
df = pd.read_csv('data.csv')

# Cargar el DataFrame a Redshift
wr.redshift.copy_from_files(
    paths=["s3://your-bucket-name/data.csv"],
    con=engine.raw_connection(),
    schema="public",
    table="your_table",
    iam_role="arn:aws:iam::your-account-id:role/your-redshift-role",
    format="csv",
    mode="overwrite"
)
```

Con este enfoque, puedes integrar Redshift con tus flujos de datos en Python de manera más sencilla.

**Lecturas recomendadas**

[Examples of using the Amazon Redshift Python connector - Amazon Redshift](https://docs.aws.amazon.com/redshift/latest/mgmt/python-connect-examples.html)

[Curso de AWS Redshift para Manejo de Big Data - Platzi](https://platzi.com/cursos/redshift-big-data/)

[template_ETL_OEC.ipynb - Google Drive](https://drive.google.com/file/d/1P5kQo5_0bkzLNakbBwlf-9evT9PYtae_/view?usp=share_link)