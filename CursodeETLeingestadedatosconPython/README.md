# Curso de ETL e ingesta de datos con Python
OneDrive/Escritorio/programación/platzi/CursodeETLeingestadedatosconPython/notebook/dataapi.ipynb
## ETL con Jupyter Notebook y Python

Realizar un proceso ETL (Extract, Transform, Load) en **Jupyter Notebook** utilizando **Python** es una forma práctica de trabajar con datos de manera interactiva. A continuación, te detallo los pasos básicos para implementar un proceso ETL, junto con ejemplos de código.

### **1. Extract (Extracción)**
La etapa de extracción implica obtener datos desde diversas fuentes, como bases de datos, archivos CSV, APIs, etc.

#### Ejemplo: Extracción desde un archivo CSV
```python
import pandas as pd

# Extraer datos desde un archivo CSV
data = pd.read_csv('data.csv')

# Vista previa de los datos
print(data.head())
```

#### Ejemplo: Extracción desde una base de datos SQL
```python
import sqlalchemy

# Configuración de la conexión a la base de datos
engine = sqlalchemy.create_engine('sqlite:///database.db')

# Consulta a la base de datos
query = "SELECT * FROM table_name"
data = pd.read_sql(query, engine)

# Vista previa de los datos
print(data.head())
```

### **2. Transform (Transformación)**
En esta etapa, se limpian y transforman los datos para prepararlos para el análisis o el almacenamiento.

#### Ejemplo: Limpieza de datos
```python
# Eliminar valores nulos
data.dropna(inplace=True)

# Renombrar columnas
data.rename(columns={'old_name': 'new_name'}, inplace=True)

# Convertir tipos de datos
data['column'] = data['column'].astype(float)
```

#### Ejemplo: Agregaciones y transformaciones
```python
# Crear una nueva columna
data['new_column'] = data['existing_column'] * 2

# Agrupar y calcular estadísticas
summary = data.groupby('category_column').agg({'value_column': ['mean', 'sum']})
print(summary)
```

### **3. Load (Carga)**
La etapa de carga implica guardar los datos transformados en un destino, como una base de datos o un archivo.

#### Ejemplo: Cargar en un archivo CSV
```python
data.to_csv('transformed_data.csv', index=False)
```

#### Ejemplo: Cargar en una base de datos
```python
# Cargar datos transformados a la base de datos
data.to_sql('new_table', engine, if_exists='replace', index=False)
```

### **4. Integración en Jupyter Notebook**
Puedes dividir el proceso ETL en celdas independientes en el notebook para trabajar con cada etapa de manera modular. Por ejemplo:

- **Celda 1**: Extracción de datos.
- **Celda 2**: Transformación de datos.
- **Celda 3**: Carga de datos.

### **5. Buenas prácticas**
- **Documentación**: Usa celdas Markdown para explicar cada paso.
- **Pruebas**: Agrega validaciones y asegúrate de que los datos sean consistentes en cada etapa.
- **Visualización**: Utiliza librerías como `matplotlib` o `seaborn` para explorar y verificar los datos transformados.
  
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data['new_column'], bins=30)
plt.show()
```

### **Resumen**

### ¿Cómo puede Python simplificar el manejo de datos desorganizados?

En el dinámico mundo empresarial, los datos llegan de diversas fuentes como hojas de Excel, bases de datos y APIs. Sin embargo, a menudo están desordenados, incompletos y caóticos. Para enfrentar este desafío, Python surge como una herramienta poderosa y versátil que facilita el manejo y organización de estos datos. Con solo unas pocas líneas de código, no solo puedes procesar datos de manera efectiva sino también prepararlos para un análisis más profundo. Vamos a explorar cómo.

### ¿Cómo empezar con Pandas en Python para gestionar datos?

Para comenzar a trabajar eficazmente con datos en Python, es esencial familiarizarse con la biblioteca Pandas. Esta herramienta es crucial para manipular y estructurar conjuntos de datos de una forma sencilla y directa. La manera más efectiva de ilustrar su uso es mediante Jupyter Notebook, un entorno muy utilizado para el análisis de datos.

En el siguiente ejemplo, veremos el proceso básico para importar un archivo CSV usando Pandas:

```Python
import pandas as pd

data = pd.read_csv('ruta/del/archivo/ventas.csv')  
print(data.head())
```

1. **Importar Pandas**: La primera línea de código importa la librería Pandas con el alias pd, permitiendo su uso directo y rápido en nuestro código.
2. **Cargar el archivo CSV**: Utilizamos la función read_csv() de Pandas para leer el archivo de ventas. Aquí, data es un objeto que guarda el DataFrame resultante del archivo CSV.
3. **Visualizar los datos**: Con data.head(), podemos observar las primeras filas del DataFrame, lo que nos da una representación inicial de cómo están estructurados nuestros datos.

### ¿Qué habilidades aprenderás en el curso de analítica de datos?

El curso no solo te instruye a trabajar con archivos CSV, sino que también abarca una variedad de otras fuentes de datos. La idea es capacitarte para construir flujos de ETL (Extract, Transform, Load) que conviertan datos desorganizados en información clara y lista para el análisis.

1. **Manejo de diversas fuentes de datos**: Aprenderás a integrar y procesar datos de archivos Excel, bases de datos y APIs, añadiendo versatilidad a tus habilidades de gestión de datos.
2. **Construcción de flujos de ETL**: Descubrirás cómo desarrollar pipelines de ETL robustos que automatizan la transformación y limpieza de datos, preparándolos para el análisis y almacenamiento eficiente.
3. **Buenas prácticas**: Incorporarás principios esenciales de manejo de datos que son demandados en el competitivo mundo profesional, asegurando que estás bien preparado para retos futuros.

### ¿Por qué Python es la mejor opción para transformar el caos en conocimiento?

Python, con su enfoque en la simplicidad y la claridad del código, es la elección preferida por analistas de datos para transformar datos desordenados en conocimiento valioso. Aquí algunos de los beneficios de usar Python para el manejo de datos:

- **Flexibilidad**: Capaz de trabajar con múltiples tipos de datos y formatos, adaptándose a cualquier necesidad empresarial.
- **Comunidad activa**: La vasta comunidad de usuarios proporciona recursos y bibliotecas adicionales que mejoran continuamente sus capacidades.
- **Automatización y eficiencia**: Permite crear procesos automatizados que ahorran tiempo y minimizan errores humanos, incrementando la productividad.

Con Python y Pandas, tienes a tu disposición herramientas precisas para convertir el desorden en una estructura clara y útil. Si estás listo para transformar el caos de datos en información valiosa, vamos a comenzar este emocionante camino en el análisis de datos.

## ¿Qué es ETL?

**ETL** significa **Extract, Transform, Load** (Extraer, Transformar, Cargar). Es un proceso esencial en la ingeniería de datos que consiste en tomar datos de diversas fuentes, transformarlos según las necesidades del negocio y almacenarlos en un destino para su análisis o uso posterior.

### **Etapas del proceso ETL**
1. **Extract (Extracción):**
   - Se obtienen datos de diferentes fuentes, como bases de datos, archivos planos, APIs o servicios web.
   - Los datos pueden estar estructurados (tablas), semiestructurados (JSON, XML) o no estructurados (texto, imágenes).
   - Ejemplo: Descargar datos de un archivo CSV, una base de datos SQL o una API.

2. **Transform (Transformación):**
   - Se limpian, enriquecen y transforman los datos para adaptarlos a los requisitos del negocio.
   - Algunas operaciones comunes incluyen:
     - Eliminación de valores nulos.
     - Conversión de tipos de datos.
     - Normalización o estandarización.
     - Agregación o cálculo de métricas.
     - Enriquecimiento con datos adicionales.
   - Ejemplo: Cambiar los nombres de columnas, calcular nuevas métricas o convertir fechas en un formato estándar.

3. **Load (Carga):**
   - Los datos transformados se cargan en un sistema de almacenamiento o destino, como:
     - Bases de datos relacionales.
     - Data Warehouses (almacenes de datos).
     - Archivos o plataformas en la nube.
   - Ejemplo: Guardar los datos procesados en una tabla de PostgreSQL o en un sistema como Amazon Redshift.

---

### **¿Para qué se usa ETL?**
- Preparar datos para análisis y generación de informes.
- Migrar datos de un sistema antiguo a uno nuevo.
- Consolidar datos provenientes de diferentes fuentes.
- Alimentar un **Data Warehouse** para realizar análisis avanzados o integrar procesos de **Business Intelligence (BI)**.

---

### **Beneficios de ETL**
- **Centralización de datos**: Combina datos de múltiples fuentes en un solo lugar.
- **Calidad de datos**: Garantiza que los datos estén limpios y listos para su uso.
- **Flexibilidad**: Facilita la integración de nuevas fuentes de datos.
- **Automatización**: Reduce el tiempo manual necesario para preparar datos.

---

### **Ejemplo práctico de ETL**
Imagina una empresa de comercio electrónico:
- **Extract**: Recoge datos de ventas de una base de datos, datos de clientes de un CRM y datos de tráfico web de Google Analytics.
- **Transform**: Limpia los datos eliminando duplicados, calcula la retención de clientes y genera métricas como el ingreso promedio por usuario (ARPU).
- **Load**: Almacena los datos procesados en un **Data Warehouse** como Snowflake para que los analistas puedan usarlos en dashboards.

### Resumen

### ¿Qué es el proceso de ETL y cuál es su relación con la ingeniería de datos?

El proceso de ETL es fundamental en la práctica de la ingeniería de datos, siendo la columna vertebral para transformar datos brutos en información útil y accesible. Consta de tres fases: extracción, transformación y carga. La fase de extracción implica acceder a datos desde diversas fuentes. Luego, en la transformación, se aplican técnicas como la eliminación de duplicados, manejo de valores faltantes y más, para convertir los datos en formatos más útiles. Finalmente, la carga involucra almacenar los datos transformados en un repositorio accesible para su posterior análisis y uso.

Este proceso permite a las empresas conectar diversas fuentes de información con su infraestructura analítica, posibilitando la integración de datos que estén listos para su consumo analítico, y es crucial para la toma de decisiones basadas en datos precisos y actualizados.

### ¿Por qué es importante la gestión de datos y cuáles son sus aplicaciones?

La gestión de datos adecuada mediante procesos de ETL es esencial para el proceso de toma de decisiones, permitiendo obtener análisis en tiempo real que son críticos para muchas empresas. Además, permite integrar herramientas complementarias como el big data para el manejo de grandes volúmenes de información, o la ciencia de datos para crear modelos avanzados vinculados a la inteligencia artificial.

Estos procesos mejoran la eficiencia y efectividad de las operaciones empresariales, asegurando que los datos están listos para satisfacer las necesidades específicas de las empresas.

### ¿Qué consideraciones se deben tener al crear flujos de ETL?

Existen varias consideraciones clave al desarrollar flujos de ETL:

- **Definición de objetivos**: Clarificar el propósito y el objetivo que se desea lograr con el ETL.
- **Selección de herramientas**: Elegir las herramientas tecnológicas apropiadas según el contexto y necesidades de la empresa.
- **Carga de datos**: Decidir si la carga de datos será incremental o completa y si es necesario el particionado de los datos.
- **Documentación**: Mantener una documentación detallada y estructurada para facilitar la comprensión y mantenimiento del flujo.
Estas consideraciones son fundamentales para asegurar la eficiencia y eficacia de los procesos de ETL, ayudando a lograr que estos flujos sean sostenibles y alineados con las necesidades empresariales.

### ¿Cuáles son las herramientas de ETL y su clasificación?

Las herramientas de ETL pueden clasificarse en tres categorías principales:

- **On-premise**: Tecnologías instaladas localmente, como Informática PowerCenter, SQL Server Integration Services y Talend.

- **Custom**: Soluciones a medida desarrolladas con lenguajes de programación como Python, Java o SQL.

- **On-cloud**: Herramientas que se ejecutan en la nube, como AWS Glue, Google Cloud Dataflow y Azure Data Factory.

Cada tipo ofrece diferentes ventajas, y la elección de una herramienta específica dependerá del contexto particular y las necesidades empresariales.

### ¿Cómo integrar Python en los procesos de ETL?

Python es altamente valorado en los procesos de ETL por su flexibilidad, capacidad para manipular grandes volúmenes de datos y amplia gama de bibliotecas especializadas. Permite personalizar los flujos de ETL al ofrecer librerías como pandas para el manejo de datos, SQLAlchemy para el trabajo con bases de datos, Apache Airflow para la orquestación de flujos de datos, BeautifulSoup para web scraping y Apache Spark para el procesamiento distribuido de datos.

### ¿Cuáles son las buenas prácticas al usar Python para ETL?

Para obtener el máximo rendimiento al usar Python en ETL, se recomienda:

- **Modularización del código**: Facilita el mantenimiento y la prueba de segmentos específicos del flujo.
- **Manejo de errores y excepciones**: Garantiza que los procesos sean robustos y menos propensos a fallos.
- **Validación y limpieza de datos**: Asegura que los datos sean precisos y útiles.
- **Optimización del rendimiento**: Mejora la eficiencia del proceso.
- **Documentación**: Mantiene el proceso claro y accesible para futuras referencias.

Estas prácticas asegurarán que los procesos de ETL sean sólidos, eficientes y adaptables a cambios futuros en la organización.

Python continúa siendo una herramienta clave en el arsenal de un ingeniero de datos, proporcionando las funcionalidades necesarias para manejar flujos de datos complejos y de gran volumen con facilidad y precisión.

## Cómo identificar y conectar fuentes de datos para ETL

Identificar y conectar fuentes de datos para un proceso ETL es un paso crucial para garantizar que los datos necesarios estén disponibles, sean accesibles y puedan integrarse de manera eficiente. A continuación, se describen las etapas clave para este proceso:

### **1. Identificación de fuentes de datos**
1. **Entender los requisitos del negocio**:
   - Pregunta: ¿Qué datos son necesarios? ¿Para qué propósito se utilizarán?
   - Ejemplo: Si estás analizando ventas, necesitas datos de facturación, clientes y productos.

2. **Tipos comunes de fuentes de datos**:
   - **Bases de datos relacionales**: MySQL, PostgreSQL, SQL Server, Oracle.
   - **Bases de datos NoSQL**: MongoDB, Cassandra, Redis.
   - **Archivos**: CSV, Excel, JSON, XML.
   - **APIs**: APIs REST/GraphQL para extraer datos desde servicios externos.
   - **Plataformas en la nube**: AWS S3, Google Cloud Storage, Azure Blob.
   - **Streams de datos en tiempo real**: Kafka, Flink, RabbitMQ.

3. **Mapeo de datos disponibles**:
   - Realiza un inventario de las fuentes existentes en tu organización o en servicios externos.
   - Documenta detalles como tipo de fuente, ubicación, acceso y formato.

### **2. Evaluación de accesibilidad**
1. **Conexión a la fuente**:
   - Asegúrate de que puedes acceder a las fuentes utilizando credenciales válidas.
   - Pregunta: ¿Se necesita autenticación? (usuario, contraseña, token de API).

2. **Permisos y seguridad**:
   - Verifica los permisos necesarios para leer datos.
   - Implementa medidas de seguridad, como cifrado de conexiones (por ejemplo, SSL/TLS).

3. **Protocolos de acceso**:
   - Bases de datos: Usa conectores como `pyodbc`, `sqlalchemy`, `psycopg2`.
   - Archivos locales: Asegúrate de que las rutas sean accesibles.
   - APIs: Usa librerías como `requests` o `http.client`.

### **3. Prueba de conexión**
Realiza una conexión de prueba a cada fuente para verificar que puedes extraer datos correctamente.

#### Ejemplo: Conexión a una base de datos PostgreSQL
```python
import sqlalchemy

# Cadena de conexión
engine = sqlalchemy.create_engine('postgresql://user:password@host:port/dbname')

# Probar la conexión con una consulta
query = "SELECT * FROM table_name LIMIT 5"
data = pd.read_sql(query, engine)
print(data.head())
```

#### Ejemplo: Conexión a una API REST
```python
import requests

url = "https://api.example.com/data"
headers = {"Authorization": "Bearer YOUR_API_TOKEN"}
response = requests.get(url, headers=headers)

# Verificar respuesta
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")
```

### **4. Validación de datos**
1. **Estructura de los datos**:
   - ¿Están en un formato compatible? Ejemplo: JSON, CSV, tabla SQL.
   - Verifica que las columnas, nombres de campos y tipos de datos coincidan con las expectativas.

2. **Calidad de los datos**:
   - Evalúa si los datos tienen valores nulos, duplicados o inconsistencias.

### **5. Planificación de integración**
1. **Frecuencia de extracción**:
   - ¿Los datos deben extraerse en tiempo real, diariamente, semanalmente?

2. **Herramientas de integración**:
   - Usa frameworks de ETL como:
     - **Apache Airflow** para tareas programadas.
     - **Talend** o **Informatica** para ETL visual.
     - **Python** con librerías como `pandas` y `sqlalchemy` para ETL ad-hoc.

### **Consejos adicionales**
- **Centralización**: Si tienes muchas fuentes, usa un Data Warehouse (BigQuery, Snowflake, Redshift).
- **Documentación**: Mantén un registro de las fuentes, formatos y configuraciones.
- **Escalabilidad**: Asegúrate de que las conexiones soporten el volumen de datos requerido.


## Solicitud desde una API
```python
import pandas as pd
import requests

url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
print(df.head())
```

## Instalando Anaconda y Jupyter Notebooks

Aquí tienes una guía paso a paso para instalar **Anaconda** y comenzar a usar **Jupyter Notebooks** en tu sistema.

### **1. Descargar Anaconda**
Anaconda es una distribución de Python que incluye herramientas como Jupyter Notebook, Spyder, y muchas librerías útiles para ciencia de datos y machine learning.

1. Ve al sitio oficial de Anaconda: [https://www.anaconda.com](https://www.anaconda.com).
2. Descarga la versión de Anaconda para tu sistema operativo (Windows, macOS o Linux). Elige la versión que incluye Python 3.x (la más reciente).

### **2. Instalar Anaconda**
1. **Windows**:
   - Ejecuta el instalador descargado (`.exe`).
   - Acepta los términos de la licencia.
   - Elige "Instalar para mí" si eres el único usuario.
   - Marca la opción para añadir Anaconda a la variable PATH (opcional, pero útil para usar `conda` en la terminal).
   - Finaliza la instalación.

2. **macOS**:
   - Abre el instalador (`.pkg`) descargado.
   - Sigue las instrucciones en pantalla.
   - Asegúrate de añadir `conda` a tu PATH si se solicita.

3. **Linux**:
   - Abre la terminal.
   - Navega al directorio donde descargaste el archivo.
   - Ejecuta el comando:
     ```bash
     bash Anaconda3-latest-Linux-x86_64.sh
     ```
   - Sigue las instrucciones del instalador.

### **3. Verificar la instalación**
Una vez instalado, abre una terminal (o el Anaconda Prompt en Windows) y escribe:

```bash
conda --version
```

Si muestra la versión de `conda`, la instalación fue exitosa.

### **4. Iniciar Jupyter Notebook**
Para abrir Jupyter Notebook desde Anaconda:
1. Abre **Anaconda Navigator** (interfaz gráfica).
2. Haz clic en el botón **Launch** debajo de "Jupyter Notebook".

O bien, desde la terminal (o Anaconda Prompt):

```bash
jupyter notebook
```

Esto abrirá Jupyter Notebook en tu navegador web por defecto. Desde allí, puedes crear y editar archivos `.ipynb`.

### **5. Crear un nuevo entorno (opcional)**
Si deseas trabajar en un entorno limpio o separado:
1. Crea un entorno nuevo:
   ```bash
   conda create --name mi_entorno python=3.9
   ```
2. Activa el entorno:
   - **Windows**:
     ```bash
     conda activate mi_entorno
     ```
   - **macOS/Linux**:
     ```bash
     conda activate mi_entorno
     ```
3. Instala Jupyter Notebook en el entorno:
   ```bash
   conda install jupyter
   ```

### **6. Buenas prácticas**
- Mantén tus entornos organizados. Usa `conda list` para ver las librerías instaladas en tu entorno.
- Actualiza Anaconda regularmente:
  ```bash
  conda update conda
  conda update anaconda
  ```

### Resumen

### ¿Cómo instalar las herramientas necesarias para el curso de procesamiento de datos?

¡Hola! ¿Listo para adentrarte en el fascinante mundo del procesamiento de datos? Este curso te guiará en el uso de herramientas fundamentales como Anaconda y Jupyter Notebook. Aquí te explicaré paso a paso cómo instalar todo lo necesario para que tu experiencia de aprendizaje sea fluida.

### ¿Qué necesitas saber sobre Anaconda?

Anaconda es una distribución integral que incluye varias tecnologías esenciales para el manejo de datos. Es compatible con diferentes sistemas operativos, y además, viene con Python 3.2 preinstalado. Para comenzar, simplemente descarga Anaconda desde su [página oficial](https://www.anaconda.com/products/distribution#download-section) sin necesidad de registrarte con tu email.

El proceso de instalación de Anaconda es sencillo:

1. Elige la versión de Anaconda que corresponda a tu sistema operativo.
2. Descarga el instalador. Este procedimiento puede tomar algunos minutos.
3. Sigue el asistente de instalación intuitivamente pulsando "Siguiente" hasta finalizar.

Una vez instalado, utiliza el Anaconda Navigator para acceder a múltiples herramientas, entre ellas Jupyter Notebook, la cual es crucial para este curso.

### ¿Cómo usar Jupyter Notebook para instalar librerías?

Jupyter Notebook será tu fiel compañero a lo largo del curso. Puedes instalar librerías necesarias directamente en un cuaderno de Jupyter Notebook o a través de la terminal de Anaconda. Aquí vamos a resaltar cómo hacerlo desde Jupyter, dado que es la opción más amigable para principiantes.

### Instalación de la librería `requests`

Esta librería es fundamental para realizar solicitudes a páginas web. Su instalación es directa. Puedes hacerlo copiando y ejecutando el siguiente código en una celda de Jupyter Notebook:

`!pip install requests`

### Verificando la disponibilidad de `JSON`

No es necesario instalar JSON si ya posees Python 3.5 o superior, ya que está integrado. Sin embargo, si usas una versión anterior, considera actualizar Python para evitar problemas de compatibilidad.

### Instalación de `SQLite`

`SQLite` te permitirá conectarte a bases de datos. Instálalo fácilmente usando este código:

`!pip install pysqlite3`

### Instalación de `SQLAlchemy`

Otra librería importante es `SQLAlchemy`, útil para manipular bases de datos. Instálalo de manera similar:

`!pip install sqlalchemy`

### ¿Qué otras librerías van a ser utilizadas?

Aunque muchas librerías como Pandas, NumPy, Matplotlib y Seaborn generalmente vienen preinstaladas con Anaconda, es importante asegurarse de que todas estén listas para usarse. Estas herramientas serán fundamentales para los laboratorios prácticos que están por venir.

### Consejos útiles y solución de problemas

Usar Anaconda y Jupyter Notebook garantiza la compatibilidad y funcionalidad óptima para los ejercicios del curso. En el raro caso de que surjas problemas durante la instalación, no dudes en dejar tus preguntas en los comentarios para obtener ayuda.

Ahora que tienes el entorno configurado, estás listo para comenzar la emocionante parte práctica del curso. Sigue explorando, aprendiendo y preparándote para dominarlos análisis de datos. ¡Adelante y mucho éxito!

**Lecturas recomendadas**

[Download Anaconda Distribution | Anaconda](https://www.anaconda.com/download)

## Ingesta de Datos desde Archivos CSV

La ingesta de datos desde archivos CSV es una de las tareas más comunes en proyectos de análisis de datos y ciencia de datos. Aquí tienes una guía detallada para realizarla utilizando Python y la librería **pandas**.

### **1. Instalación de Pandas**
Si aún no tienes pandas instalado, puedes instalarlo con pip:

```bash
pip install pandas
```

### **2. Importar Pandas**
Antes de comenzar, importa la librería en tu script o notebook:

```python
import pandas as pd
```

### **3. Leer un archivo CSV**
Usa el método `pd.read_csv` para cargar datos desde un archivo CSV a un **DataFrame**.

```python
# Cargar un archivo CSV
df = pd.read_csv('ruta/del/archivo.csv')

# Mostrar las primeras 5 filas
print(df.head())
```

### **4. Opciones comunes de `read_csv`**
1. **Especificar el delimitador**:
   Si el archivo usa un delimitador diferente (por ejemplo, `;`):
   ```python
   df = pd.read_csv('archivo.csv', delimiter=';')
   ```

2. **Especificar la codificación**:
   Si el archivo tiene caracteres especiales:
   ```python
   df = pd.read_csv('archivo.csv', encoding='utf-8')
   ```

3. **Columnas específicas**:
   Si solo necesitas ciertas columnas:
   ```python
   df = pd.read_csv('archivo.csv', usecols=['columna1', 'columna2'])
   ```

4. **Nombres de columnas personalizados**:
   Si deseas renombrar las columnas al cargar:
   ```python
   df = pd.read_csv('archivo.csv', names=['A', 'B', 'C'], header=0)
   ```

5. **Manejo de valores nulos**:
   Para reemplazar ciertos valores con `NaN`:
   ```python
   df = pd.read_csv('archivo.csv', na_values=['NA', '?'])
   ```

6. **Tamaño del archivo (chunking)**:
   Para leer el archivo en partes si es muy grande:
   ```python
   chunk_size = 10000  # Número de filas por bloque
   for chunk in pd.read_csv('archivo_grande.csv', chunksize=chunk_size):
       print(chunk.head())
   ```

### **5. Ejemplo práctico**
Supongamos que tienes un archivo llamado `ventas.csv` con el siguiente contenido:

| Fecha       | Producto  | Ventas |
|-------------|-----------|--------|
| 2023-01-01  | A         | 100    |
| 2023-01-02  | B         | 200    |
| 2023-01-03  | C         | 150    |

Cargarlo y explorarlo sería así:

```python
# Leer el archivo CSV
df = pd.read_csv('ventas.csv')

# Mostrar información básica
print(df.info())

# Resumen estadístico
print(df.describe())

# Filtrar por una columna
ventas_a = df[df['Producto'] == 'A']
print(ventas_a)
```

### **6. Guardar el DataFrame nuevamente en un archivo CSV**
Si necesitas guardar los datos procesados en un nuevo archivo CSV:

```python
# Guardar en un archivo CSV
df.to_csv('ventas_procesadas.csv', index=False)
print("Archivo guardado exitosamente.")
```

### **Errores comunes al cargar archivos CSV**
1. **Archivo no encontrado**:
   - Asegúrate de que la ruta del archivo sea correcta.
   - Usa rutas absolutas o verifica el directorio de trabajo.

2. **Problemas de codificación**:
   - Especifica la codificación adecuada (`utf-8`, `latin1`, etc.).

3. **Formato incorrecto**:
   - Asegúrate de que el delimitador sea el correcto (`delimiter=','`).

## Ingesta de Datos desde Archivos Excel

La ingesta de datos desde archivos **Excel** en Python es muy común y se realiza principalmente con la librería **`pandas`**. A continuación te explico cómo leer archivos Excel utilizando **`pandas`** y algunas opciones importantes para manejar estos archivos.

### **1. Instalación de Pandas y openpyxl (si es necesario)**
Asegúrate de tener instaladas las bibliotecas necesarias. Si no tienes `pandas` o `openpyxl`, instálalas usando `pip`:

```bash
pip install pandas openpyxl
```

### **2. Importar Pandas**
Primero, asegúrate de importar **pandas** en tu script:

```python
import pandas as pd
```

### **3. Leer un archivo Excel**
Puedes usar el método `read_excel` para leer un archivo Excel (.xls o .xlsx).

```python
# Leer un archivo Excel
df = pd.read_excel("archivo_excel.xlsx")

# Mostrar las primeras 5 filas
print(df.head())
```

### **4. Opciones comunes de `read_excel`**

1. **Especificar el nombre de la hoja**:
   Si el archivo tiene varias hojas, puedes especificar cuál quieres leer:

   ```python
   df = pd.read_excel("archivo_excel.xlsx", sheet_name="Hoja1")
   ```

2. **Manejo de índices**:
   Puedes especificar en qué columna o filas se deben usar como índices:

   ```python
   df = pd.read_excel("archivo_excel.xlsx", index_col='ColumnaIndice')
   ```

3. **Seleccionar columnas específicas**:
   Si solo necesitas algunas columnas:

   ```python
   df = pd.read_excel("archivo_excel.xlsx", usecols=['Columna1', 'Columna2'])
   ```

4. **Manejo de valores nulos**:
   Puedes especificar cómo se deben manejar los valores nulos:

   ```python
   df = pd.read_excel("archivo_excel.xlsx", na_values=['NA', 'N/A'])
   ```

5. **Tamaño del archivo (chunking)**:
   Si el archivo es muy grande y deseas leerlo por partes:

   ```python
   chunk_size = 1000  # Número de filas por bloque
   for chunk in pd.read_excel("archivo_excel.xlsx", chunksize=chunk_size):
       print(chunk.head())
   ```

### **5. Ejemplo práctico:**
Supongamos que tienes un archivo Excel con datos similares a esto:

| Nombre      | Edad | Ciudad     |
|-------------|------|------------|
| Juan        | 28   | Bogotá     |
| Ana         | 35   | Medellín   |
| Carlos      | 40   | Cali       |

Tu script podría verse así:

```python
# Leer el archivo Excel
df = pd.read_excel("AB_NYC_2019.xlsx")

# Mostrar las primeras 5 filas
print(df.head())

# Información general del DataFrame
print(df.info())

# Estadísticas básicas
print(df.describe())
```

### **6. Guardar el DataFrame en un archivo Excel**
Si deseas guardar el DataFrame procesado en un nuevo archivo Excel:

```python
# Guardar el DataFrame en un archivo Excel
df.to_excel("archivo_procesado.xlsx", index=False)
print("Archivo guardado exitosamente.")
```

### **Errores comunes al leer archivos Excel**
1. **Archivo no encontrado**:
   - Asegúrate de que la ruta del archivo sea correcta.

2. **Problemas de codificación**:
   - Especifica la codificación adecuada (`utf-8`, `latin1`, etc.).

3. **Formatos de archivo Excel incorrectos**:
   - Puede ser útil revisar el archivo para ver si tiene combinaciones de celdas, valores nulos o encabezados mal configurados.

**Lecturas recomendadas**

[Ingesta de Datos desde Archivos Excel.ipynb](https://github.com/platzi/etl-python/blob/main/6.%20Ingesta%20de%20Datos%20desde%20Archivos%20Excel/Ingesta%20de%20Datos%20desde%20Archivos%20Excel.ipynb)

[data.xlsx](https://github.com/platzi/etl-python/blob/main/6.%20Ingesta%20de%20Datos%20desde%20Archivos%20Excel/data.xlsx)

[Qué es visualización de datos y para qué sirve](https://platzi.com/blog/visualizacion-de-datos/)

## Ingesta de Datos desde APIs

La ingesta de datos desde **APIs** es una tarea común en el trabajo con datos, especialmente cuando necesitas acceder a información en tiempo real o a través de servicios en línea. Aquí te dejo una guía básica para consumir APIs utilizando **Python**.


### **1. Instalación de las bibliotecas necesarias**
Primero, asegúrate de tener instaladas las bibliotecas que necesitas:

- **`requests`**: Para hacer solicitudes HTTP.
- **`json`**: Para procesar la respuesta en formato JSON.

```bash
pip install requests
```

### **2. Importar las bibliotecas necesarias**
```python
import requests
import json
```

### **3. Realizar una solicitud HTTP**
Para consumir datos desde una API, necesitas hacer una **`GET`** o **`POST`** (dependiendo del endpoint) y recibir la respuesta:

```python
# URL de la API (ejemplo)
url = "https://api.example.com/data"

# Hacer la solicitud HTTP GET
response = requests.get(url)

# Verificar el estado de la solicitud
if response.status_code == 200:
    # Obtener los datos como un diccionario JSON
    data = response.json()
    print(data)
else:
    print("Error en la solicitud:", response.status_code)
```

### **4. Opciones adicionales al trabajar con APIs**
- **Pasar parámetros o headers**:
  Si la API requiere autenticación o parámetros específicos:

  ```python
  headers = {
      'Authorization': 'Bearer TOKEN'
  }

  params = {
      'key1': 'value1',
      'key2': 'value2'
  }

  response = requests.get("https://api.example.com/data", headers=headers, params=params)

  if response.status_code == 200:
      data = response.json()
      print(data)
  ```

- **Realizar una solicitud POST**:
  En caso de necesitar enviar datos a la API:

  ```python
  payload = {
      'param1': 'value1',
      'param2': 'value2'
  }

  response = requests.post("https://api.example.com/data", json=payload)

  if response.status_code == 201:
      data = response.json()
      print(data)
  ```

### **5. Procesar la respuesta JSON**
El formato más común de respuesta desde una API suele ser **JSON**. Puedes procesar los datos para extraer lo que necesitas:

```python
# Obtener datos de ejemplo (respuesta en JSON)
response = requests.get("https://api.example.com/data")
data = response.json()

# Extraer ciertos valores
for item in data['items']:
    print("Nombre:", item['name'])
    print("Valor:", item['value'])
```

### **6. Manejo de errores comunes**
1. **400 - Bad Request**:
   - Asegúrate de que los parámetros enviados son correctos.

2. **401 - Unauthorized**:
   - Revisa la autenticación o los tokens que se están utilizando.

3. **404 - Not Found**:
   - Verifica que el endpoint o la URL sea correcta.

4. **500 - Internal Server Error**:
   - Es posible que el servicio esté teniendo problemas.

### **7. Guardar los datos en un archivo (opcional)**
Si necesitas guardar los datos recuperados de la API en un archivo JSON:

```python
with open('datos_api.json', 'w') as file:
    json.dump(data, file)
```

### **8. Leer los datos desde un archivo JSON**
Si ya tienes un archivo JSON guardado, puedes leerlo nuevamente:

```python
with open('datos_api.json', 'r') as file:
    data = json.load(file)
    print(data)
```

### **Ejemplo práctico usando API pública de ejemplo**:

```python
import requests

# URL de ejemplo de API pública
url = "https://jsonplaceholder.typicode.com/posts"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    for post in data:
        print(f'Título: {post["title"]}')
        print(f'Cuerpo: {post["body"]}')
        print('-' * 40)
else:
    print("Error en la solicitud:", response.status_code)
```

**Lecturas recomendadas**

[API con Python - Ejemplo 1.ipynb](https://github.com/platzi/etl-python/blob/main/7.%20Ingesta%20de%20Datos%20desde%20APIs/API%20con%20Python%20-%20Ejemplo%201.ipynb)

[API con Python - Ejemplo 2.ipynb](https://github.com/platzi/etl-python/blob/main/7.%20Ingesta%20de%20Datos%20desde%20APIs/API%20con%20Python%20-%20Ejemplo%202.ipynb)

## Ingesta de Datos desde Bases de Datos

La ingesta de datos desde bases de datos es una tarea esencial en proyectos de **ETL** y análisis de datos. Con Python, puedes conectarte a bases de datos populares como **MySQL**, **PostgreSQL**, **SQLite**, **SQL Server**, entre otras, utilizando bibliotecas como **`pymysql`**, **`psycopg2`**, o **`sqlite3`**, junto con **`pandas`** para manipular los datos.

### **1. Instalación de bibliotecas necesarias**
Primero, asegúrate de instalar las bibliotecas necesarias según el tipo de base de datos:

- **MySQL**: `pip install pymysql`
- **PostgreSQL**: `pip install psycopg2`
- **SQL Server**: `pip install pyodbc`
- **SQLite**: Biblioteca integrada en Python

Adicionalmente, instala **`sqlalchemy`** y **`pandas`** para trabajar con conexiones y manipulación de datos.

```bash
pip install sqlalchemy pandas
```

### **2. Crear una conexión a la base de datos**
El primer paso es establecer una conexión utilizando el **string de conexión** adecuado para tu base de datos.

#### **a. MySQL**
```python
from sqlalchemy import create_engine

# String de conexión: MySQL
engine = create_engine("mysql+pymysql://usuario:contraseña@host/nombre_base_datos")

# Probar conexión
connection = engine.connect()
print("Conexión exitosa a MySQL")
```

#### **b. PostgreSQL**
```python
from sqlalchemy import create_engine

# String de conexión: PostgreSQL
engine = create_engine("postgresql+psycopg2://usuario:contraseña@host/nombre_base_datos")

# Probar conexión
connection = engine.connect()
print("Conexión exitosa a PostgreSQL")
```

#### **c. SQLite**
```python
from sqlalchemy import create_engine

# Conexión a un archivo SQLite
engine = create_engine("sqlite:///nombre_base_datos.sqlite")

# Probar conexión
connection = engine.connect()
print("Conexión exitosa a SQLite")
```

### **3. Consultar datos desde la base de datos**
Una vez establecida la conexión, puedes ejecutar una consulta SQL para obtener datos y cargarlos en un **DataFrame** con **pandas**.

```python
import pandas as pd

# Consulta SQL
query = "SELECT * FROM nombre_tabla"

# Leer datos desde la base de datos a un DataFrame
df = pd.read_sql(query, engine)

# Mostrar los primeros registros
print(df.head())
```

### **4. Guardar los datos procesados (opcional)**
Puedes guardar los datos procesados en un nuevo archivo:

```python
# Guardar los datos en un archivo CSV
df.to_csv("datos_procesados.csv", index=False)

# Guardar los datos en un archivo Excel
df.to_excel("datos_procesados.xlsx", index=False)
```

### **5. Manejo de errores comunes**
1. **Error de conexión**:
   - Verifica el string de conexión (usuario, contraseña, host, puerto, nombre de la base de datos).
   - Revisa si el servidor de la base de datos está activo.

2. **Error en la consulta SQL**:
   - Asegúrate de que la tabla y las columnas mencionadas en la consulta existan.
   - Valida la sintaxis SQL.

3. **Dependencias faltantes**:
   - Instala las bibliotecas necesarias (`sqlalchemy`, `pymysql`, etc.).

### **6. Ejemplo práctico**
Supongamos que tienes una base de datos MySQL con una tabla llamada `empleados`:

```python
from sqlalchemy import create_engine
import pandas as pd

# Conexión a MySQL
engine = create_engine("mysql+pymysql://usuario:contraseña@localhost/mi_base_datos")

# Consulta SQL
query = "SELECT nombre, edad, salario FROM empleados"

# Leer datos en un DataFrame
df = pd.read_sql(query, engine)

# Mostrar los datos
print(df)

# Guardar en CSV
df.to_csv("empleados.csv", index=False)
print("Datos guardados en empleados.csv")
```

### **7. Cerrar la conexión**
Es importante cerrar la conexión después de procesar los datos:

```python
connection.close()
print("Conexión cerrada.")
```

**Lecturas recomendadas**

[Sqlite3 + SQLAlchemy.ipynb](https://github.com/platzi/etl-python/blob/main/8.%20Ingesta%20de%20Datos%20desde%20Bases%20de%20Datos/Sqlite3%20%2B%20SQLAlchemy.ipynb)

[nba_salary.sqlite](https://github.com/platzi/etl-python/blob/main/8.%20Ingesta%20de%20Datos%20desde%20Bases%20de%20Datos/nba_salary.sqlite)

[Documentación para instalar: MySQL Workbench](https://dev.mysql.com/doc/workbench/en/)

## Procesa datos con Pandas

¡Claro! **Pandas** es una de las bibliotecas más utilizadas en Python para procesar, analizar y manipular datos. A continuación, te muestro cómo realizar las operaciones más comunes utilizando Pandas.

### **1. Importar Pandas**
Asegúrate de tener Pandas instalado y luego importarlo:

```python
import pandas as pd
```

### **2. Cargar datos**
Puedes cargar datos desde varios formatos: CSV, Excel, JSON, etc.

#### a) Desde un archivo CSV
```python
df = pd.read_csv("archivo.csv")
print(df.head())  # Mostrar las primeras 5 filas
```

#### b) Desde un archivo Excel
```python
df = pd.read_excel("archivo.xlsx")
print(df.info())  # Mostrar información general del DataFrame
```

#### c) Desde un archivo JSON
```python
df = pd.read_json("archivo.json")
print(df.describe())  # Estadísticas descriptivas de columnas numéricas
```

### **3. Inspeccionar datos**
Estas funciones son útiles para explorar un DataFrame:

```python
print(df.head())         # Primeras 5 filas
print(df.tail())         # Últimas 5 filas
print(df.shape)          # Dimensiones del DataFrame (filas, columnas)
print(df.columns)        # Nombres de las columnas
print(df.dtypes)         # Tipos de datos de las columnas
```

### **4. Selección y filtrado de datos**
#### Seleccionar columnas
```python
df['columna']  # Seleccionar una columna
df[['columna1', 'columna2']]  # Seleccionar varias columnas
```

#### Filtrar filas
```python
filtro = df['edad'] > 30
df_filtrado = df[filtro]
```

### **5. Limpieza de datos**
#### Eliminar filas/columnas con valores nulos
```python
df_limpio = df.dropna()  # Eliminar filas con valores nulos
```

#### Rellenar valores nulos
```python
df['columna'].fillna(0, inplace=True)  # Rellenar nulos con 0
```

#### Renombrar columnas
```python
df.rename(columns={'antigua': 'nueva'}, inplace=True)
```

#### Eliminar duplicados
```python
df = df.drop_duplicates()
```

### **6. Operaciones con columnas**
#### Crear una nueva columna
```python
df['nueva_columna'] = df['columna1'] + df['columna2']
```

#### Aplicar una función a una columna
```python
df['nueva_columna'] = df['columna'].apply(lambda x: x * 2)
```

### **7. Agrupación y agregación**
#### Agrupar datos
```python
df_agrupado = df.groupby('categoria')['valor'].sum()
print(df_agrupado)
```

#### Estadísticas básicas
```python
print(df['columna'].mean())  # Promedio
print(df['columna'].median())  # Mediana
```

### **8. Exportar datos**
Guarda el DataFrame procesado en varios formatos:

#### a) CSV
```python
df.to_csv("archivo_limpio.csv", index=False)
```

#### b) Excel
```python
df.to_excel("archivo_limpio.xlsx", index=False)
```

#### c) JSON
```python
df.to_json("archivo_limpio.json", orient='records')
```

### Ejemplo completo
```python
import pandas as pd

# Cargar datos
df = pd.read_csv("archivo.csv")

# Inspeccionar
print(df.info())

# Limpiar datos
df = df.dropna()  # Eliminar filas con nulos

# Crear una nueva columna
df['precio_total'] = df['cantidad'] * df['precio_unitario']

# Agrupar y sumar
resumen = df.groupby('categoria')['precio_total'].sum()

# Exportar el resultado
resumen.to_csv("resumen_categorias.csv")
```

**Lecturas recomendadas**

[Repaso de Pandas.ipynb](https://github.com/platzi/etl-python/blob/main/9.%20Repaso%20de%20Pandas/Repaso%20de%20Pandas.ipynb)

## Métricas de Calidad y Perfilado de Datos

El análisis de calidad y el perfilado de datos son pasos fundamentales en cualquier flujo de trabajo de datos. **Pandas** ofrece herramientas poderosas para calcular métricas de calidad y realizar el perfilado inicial. Aquí hay una guía sobre cómo puedes lograrlo.

## **1. Métricas de Calidad de Datos**

### a) **Valores Nulos**
Identificar y manejar valores nulos es clave para mantener la calidad de los datos.

```python
# Total de valores nulos por columna
print(df.isnull().sum())

# Porcentaje de valores nulos por columna
print(df.isnull().mean() * 100)
```

### b) **Duplicados**
Detectar registros duplicados ayuda a evitar inconsistencias.

```python
# Identificar duplicados
duplicados = df.duplicated()

# Contar registros duplicados
print("Registros duplicados:", duplicados.sum())

# Eliminar duplicados
df_sin_duplicados = df.drop_duplicates()
```

### c) **Consistencia de tipos de datos**
Verificar si las columnas tienen los tipos correctos.

```python
print(df.dtypes)  # Tipos de datos por columna
```

Si hay inconsistencias, puedes convertir tipos:

```python
df['columna'] = df['columna'].astype(float)  # Cambiar tipo a float
```

### d) **Rangos de valores**
Comprobar valores atípicos (outliers) y rangos incorrectos.

```python
# Resumen estadístico
print(df.describe())

# Filtrar valores fuera de un rango esperado
outliers = df[df['columna'] > 1000]
print(outliers)
```

## **2. Perfilado de Datos**

### a) **Resúmenes básicos**
Obtener una visión general de las estadísticas de las columnas.

```python
# Resumen estadístico general
print(df.describe(include='all'))  # Incluir datos no numéricos
```

### b) **Distribución de datos**
Ver la distribución de valores únicos y sus frecuencias.

```python
# Contar valores únicos por columna
print(df['columna'].value_counts())

# Proporción de cada valor
print(df['columna'].value_counts(normalize=True) * 100)
```

### c) **Detección de cardinalidad**
Identificar columnas con alta o baja cardinalidad.

```python
# Número de valores únicos
print(df.nunique())
```

## **3. Visualización de Calidad y Perfilado**
Usar gráficos ayuda a detectar problemas rápidamente.

### a) **Valores Nulos**
Visualizar columnas con muchos valores faltantes.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Mapa de Valores Nulos")
plt.show()
```

### b) **Distribución de Datos**
Examinar la distribución de las columnas.

```python
df['columna'].hist(bins=30)
plt.title("Distribución de la columna")
plt.xlabel("Valores")
plt.ylabel("Frecuencia")
plt.show()
```

## **4. Herramientas Avanzadas de Perfilado**

Si deseas un perfilado automático y detallado de los datos, puedes usar bibliotecas especializadas.

### a) **Pandas Profiling**
Genera reportes interactivos del perfilado de datos.

```bash
pip install pandas-profiling
```

```python
from pandas_profiling import ProfileReport

# Crear el reporte
profile = ProfileReport(df, title="Reporte de Perfilado de Datos", explorative=True)

# Mostrar el reporte en Jupyter
profile.to_notebook_iframe()

# Guardar el reporte en un archivo HTML
profile.to_file("reporte.html")
```

### b) **Sweetviz**
Otra herramienta para el análisis exploratorio de datos.

```bash
pip install sweetviz
```

```python
import sweetviz as sv

# Crear y mostrar el reporte
reporte = sv.analyze(df)
reporte.show_html("sweetviz_reporte.html")
```

## **5. Ejemplo Completo**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("archivo.csv")

# Inspeccionar valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# Eliminar duplicados
df = df.drop_duplicates()

# Resumen estadístico
print("Estadísticas descriptivas:")
print(df.describe())

# Visualizar valores nulos
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Valores Nulos")
plt.show()

# Crear un reporte con Pandas Profiling
from pandas_profiling import ProfileReport
reporte = ProfileReport(df, title="Perfilado de Datos", explorative=True)
reporte.to_file("perfilado.html")

# Validate data

df['data_column'].apply(lambda x: validate_data(x))
```

## Técnicas de Limpieza de Datos

La limpieza de datos es esencial para garantizar que los datos sean precisos, consistentes y adecuados para el análisis. Aquí te explico algunas de las técnicas más comunes para limpiar datos con **Pandas** en Python.

## **1. Identificar y Manejar Valores Nulos**
Los valores nulos son comunes en los datos y pueden manejarse de diferentes maneras.

### a) Identificar valores nulos
```python
# Contar valores nulos por columna
print(df.isnull().sum())

# Visualizar valores nulos
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Mapa de Valores Nulos")
plt.show()
```

### b) Eliminar valores nulos
```python
# Eliminar filas con valores nulos
df = df.dropna()

# Eliminar columnas con muchos valores nulos
df = df.dropna(axis=1)
```

### c) Rellenar valores nulos
```python
# Rellenar con un valor fijo
df['columna'] = df['columna'].fillna(0)

# Rellenar con la media, mediana o moda
df['columna'] = df['columna'].fillna(df['columna'].mean())
df['columna'] = df['columna'].fillna(df['columna'].median())
df['columna'] = df['columna'].fillna(df['columna'].mode()[0])
```

## **2. Manejar Duplicados**
### a) Identificar duplicados
```python
# Identificar filas duplicadas
print(df.duplicated().sum())

# Ver duplicados específicos
print(df[df.duplicated()])
```

### b) Eliminar duplicados
```python
df = df.drop_duplicates()
```

## **3. Corregir Tipos de Datos**
Los datos pueden estar mal clasificados y necesitan conversión.

### a) Cambiar el tipo de una columna
```python
df['columna'] = df['columna'].astype(float)  # Convertir a flotante
df['fecha'] = pd.to_datetime(df['fecha'])    # Convertir a fecha
```

### b) Detectar errores en los tipos
```python
# Encontrar filas no convertibles
errores = df[~df['columna'].str.isnumeric()]
print(errores)
```

## **4. Corregir Valores Atípicos (Outliers)**
### a) Detectar valores atípicos
```python
# Usar el rango intercuartílico (IQR)
Q1 = df['columna'].quantile(0.25)
Q3 = df['columna'].quantile(0.75)
IQR = Q3 - Q1

# Filtrar outliers
outliers = df[(df['columna'] < Q1 - 1.5 * IQR) | (df['columna'] > Q3 + 1.5 * IQR)]
print(outliers)
```

### b) Manejar valores atípicos
```python
# Reemplazar outliers con la mediana
mediana = df['columna'].median()
df.loc[(df['columna'] < Q1 - 1.5 * IQR) | (df['columna'] > Q3 + 1.5 * IQR), 'columna'] = mediana
```

## **5. Normalizar y Estandarizar Datos**
### a) Normalización (escala de 0 a 1)
```python
df['columna_normalizada'] = (df['columna'] - df['columna'].min()) / (df['columna'].max() - df['columna'].min())
```

### b) Estandarización (media 0, desviación estándar 1)
```python
df['columna_estandarizada'] = (df['columna'] - df['columna'].mean()) / df['columna'].std()
```

## **6. Corregir Datos Categóricos**
### a) Normalizar texto
```python
# Convertir a minúsculas
df['columna'] = df['columna'].str.lower()

# Eliminar espacios extra
df['columna'] = df['columna'].str.strip()
```

### b) Reemplazar valores incorrectos
```python
df['columna'] = df['columna'].replace({'valor_incorrecto': 'valor_correcto'})
```

## **7. Manejar Rangos y Valores Inválidos**
### a) Filtrar valores fuera de rango
```python
df = df[(df['columna'] >= 0) & (df['columna'] <= 100)]
```

### b) Corregir valores inválidos
```python
# Reemplazar valores negativos con NaN
df['columna'] = df['columna'].apply(lambda x: x if x >= 0 else None)
```


## **8. Ejemplo Completo**
```python
import pandas as pd

# Cargar datos
df = pd.read_csv("archivo.csv")

# Identificar y manejar valores nulos
df['columna'] = df['columna'].fillna(df['columna'].mean())

# Eliminar duplicados
df = df.drop_duplicates()

# Corregir tipos de datos
df['fecha'] = pd.to_datetime(df['fecha'])

# Manejar valores atípicos
Q1 = df['columna'].quantile(0.25)
Q3 = df['columna'].quantile(0.75)
IQR = Q3 - Q1
df.loc[(df['columna'] < Q1 - 1.5 * IQR) | (df['columna'] > Q3 + 1.5 * IQR), 'columna'] = df['columna'].median()

# Guardar datos limpios
df.to_csv("archivo_limpio.csv", index=False)
```

### Resumen 
### ¿Cómo limpiar datos en Python de manera efectiva? 

La limpieza de datos es un paso crucial en el análisis de datos. Permite no solo garantizar la calidad y precisión de los análisis subsecuentes, sino que también puede prevenir errores en algoritmos y modelos predictivos. Usar Python y la librería Pandas ofrece herramientas poderosas que facilitan este proceso. A continuación, exploraremos diversas técnicas y métodos para llevar a cabo una limpieza eficiente de datos.

### ¿Cómo manejar valores nulos en un DataFrame?

Al analizar datos, es común encontrarse con valores nulos o "mising". Estos pueden tratarse de distintas maneras:

- **Detección**: Utiliza `isnull()` para identificar valores nulos en cada columna. Posteriormente, con `.sum()`, se puede contar cuántos valores nulos existen por columna.

  `valores_nulos = df.isnull().sum()`

- **Eliminación**: Con `dropna()`, eliminamos filas que contengan valores nulos. Aunque eficaz, puede no siempre ser lo ideal si se pierden datos valiosos.

  `df_limpio = df.dropna()`

- **Imputación**: Se refiere a llenar valores nulos con un dato específico. Por ejemplo, podríamos usar `fillna()` para imputar ceros en variables numéricas o colocar "desconocido" en una columna de texto.

  `df_rellenado = df.fillna({'salario': 0, 'nombre': 'desconocido'})`

### ¿Cómo corregir errores en los tipos de datos?

Una tarea habitual en la limpieza de datos es asegurar que cada columna tenga el tipo de dato correcto. Esto se puede lograr fácilmente con Pandas:

- **Convertir a numérico**: `to_numeric()` transforma columnas en números, útil cuando datos se almacenan como texto.

  `df['edad'] = pd.to_numeric(df['edad'], errors='coerce')`

- **Transformaciones estadísticas para imputación**: Más allá de simples constantes, podemos usar métodos estadísticos como la media para la imputación.

  `df['salario'].fillna(df['salario'].mean(), inplace=True)`

### ¿Cómo transformar variables categóricas?

Frecuentemente es necesario convertir variables categóricas a numéricas, especialmente al prepararlas para modelos de aprendizaje automático:

- **Mapeo binario**: Con `map()` es posible transformar variables binarias, como género, a 0 y 1, facilitando su uso en modelos.

  `df['género'] = df['género'].map({'femenino': 0, 'masculino': 1})`

- **Variables ficticias**: `get_dummies()` genera columnas binarias para cada categoría de una variable. Al usarlo, el parámetro `drop_first=True` ayuda a evitar redundancias.

  `df_dummies = pd.get_dummies(df['departamento'], drop_first=True)`

### ¿Cómo manejar variables categóricas no binarias?

Manejar categorías no binarias introduce complejidad, pero también brinda más información. Por ejemplo, al tratar con géneros no binarios podríamos:

- **Ampliar categorías en mapeo**: Ajustar `map()` para incluir más categorías.

- **Uso de variables ficticias**: `get_dummies()` permite incluir múltiples categorías sin perder información.

### Recomendaciones para mejoras prácticas

- Antes de eliminar datos, considera el impacto en los análisis.
- Imputa con métodos estadísticos cuando sea posible para mantener integridad en datos.
- Revisa valores tipo datos después de conversiones usando `.info()`.

En definitiva, estos métodos proveen las bases necesarias para una limpieza de datos efectiva. Viajar por el mundo de los datos bien preparados no solo incrementará la eficiencia de tus análisis, sino que también te permitirá sacar conclusiones más precisas y significativas. Este es solo el comienzo, sigue explorando y perfeccionando tus habilidades con prácticas y nuevos desafíos.

## Transformaciones y Filtrado Esencial de Datos

Las transformaciones y el filtrado son fundamentales para procesar y analizar datos. Aquí tienes un resumen de las técnicas esenciales para transformar y filtrar datos usando **Pandas** en Python.

---

## **1. Transformaciones de Datos**

### a) **Seleccionar y Renombrar Columnas**
```python
# Seleccionar columnas específicas
df = df[['columna1', 'columna2']]

# Renombrar columnas
df = df.rename(columns={'columna_antigua': 'columna_nueva'})
```

---

### b) **Crear o Modificar Columnas**
```python
# Crear una nueva columna basada en otras
df['nueva_columna'] = df['columna1'] + df['columna2']

# Aplicar una transformación a una columna
df['columna'] = df['columna'].apply(lambda x: x**2)
```

---

### c) **Transformar Formato de Datos**
```python
# Convertir una columna a mayúsculas
df['columna'] = df['columna'].str.upper()

# Extraer partes de una fecha
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
```

---

### d) **Agrupación y Operaciones Agregadas**
```python
# Agrupar por una columna y calcular la media
agrupado = df.groupby('categoria')['valor'].mean()

# Varias operaciones en el mismo grupo
agrupado = df.groupby('categoria').agg({'valor': ['mean', 'sum']})
```

---

### e) **Pivot Tables**
```python
# Crear una tabla pivote
tabla_pivote = df.pivot_table(values='valor', index='categoria', columns='tipo', aggfunc='sum')
```

---

## **2. Filtrado de Datos**

### a) **Filtrar Filas por Condiciones**
```python
# Filtrar filas donde una columna sea mayor que un valor
df_filtrado = df[df['columna'] > 50]

# Filtrar usando varias condiciones
df_filtrado = df[(df['columna1'] > 50) & (df['columna2'] < 100)]
```

---

### b) **Filtrar por Valores en una Lista**
```python
# Filtrar filas con valores específicos
df_filtrado = df[df['columna'].isin(['valor1', 'valor2'])]
```

---

### c) **Filtrar Valores Nulos o No Nulos**
```python
# Filtrar filas sin valores nulos en una columna
df_sin_nulos = df[df['columna'].notnull()]

# Filtrar filas con valores nulos
df_con_nulos = df[df['columna'].isnull()]
```

---

### d) **Filtrar por Texto**
```python
# Filtrar filas donde una columna contiene un texto específico
df_texto = df[df['columna'].str.contains('texto')]

# Filtrar filas donde una columna empieza o termina con un texto
df_empieza = df[df['columna'].str.startswith('inicio')]
df_termina = df[df['columna'].str.endswith('final')]
```

---

## **3. Reordenar y Reestructurar Datos**

### a) **Ordenar Datos**
```python
# Ordenar por una columna
df_ordenado = df.sort_values('columna')

# Ordenar por varias columnas
df_ordenado = df.sort_values(['columna1', 'columna2'], ascending=[True, False])
```

---

### b) **Reindexar Filas**
```python
# Reordenar el índice
df = df.reset_index(drop=True)

# Establecer una columna como índice
df = df.set_index('columna')
```

---

### c) **Unir y Combinar Datos**
```python
# Combinar dos DataFrames por una columna común (merge)
df_combinado = pd.merge(df1, df2, on='clave', how='inner')

# Concatenar DataFrames
df_concatenado = pd.concat([df1, df2], axis=0)  # Filas
df_concatenado = pd.concat([df1, df2], axis=1)  # Columnas
```

---

## **4. Ejemplo Completo**

```python
import pandas as pd

# Crear un DataFrame de ejemplo
data = {
    'producto': ['A', 'B', 'C', 'A', 'B'],
    'ventas': [100, 200, 300, 150, 250],
    'fecha': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
}
df = pd.DataFrame(data)

# Filtrar productos con ventas mayores a 150
df_filtrado = df[df['ventas'] > 150]

# Crear una columna con el mes de la venta
df['mes'] = df['fecha'].dt.month

# Agrupar por producto y sumar ventas
resumen = df.groupby('producto')['ventas'].sum()

# Ordenar por ventas totales
resumen = resumen.sort_values(ascending=False)

# Imprimir resultados
print("Datos Filtrados:\n", df_filtrado)
print("\nResumen por Producto:\n", resumen)
```

---

## **5. Herramientas Adicionales**

- **Filtros Avanzados**: Puedes usar expresiones regulares con `str.contains` para filtrados más complejos.
- **Transformaciones Escalables**: Usa `apply` o `map` para aplicar funciones personalizadas.
- **Integración con otras librerías**: Combina estas técnicas con librerías como **NumPy** o **SciPy** para cálculos avanzados.

**Lecturas recomendadas**

[Transformaciones Básicas.ipynb](https://github.com/platzi/etl-python/blob/main/12.%20Transformaciones%20basicas/Transformaciones%20B%C3%A1sicas.ipynb)

[datos_ejemplo.csv](https://github.com/platzi/etl-python/blob/main/12.%20Transformaciones%20basicas/datos_ejemplo.csv)

## Agrupaciones y Resumen de Datos

Las **agrupaciones** y los **resúmenes** son técnicas esenciales en el análisis de datos. Con Pandas, puedes agrupar datos por una o varias columnas y luego calcular resúmenes como la suma, la media, la cuenta, o cualquier función agregada. Aquí te explico cómo hacerlo paso a paso.

## **1. Agrupar Datos por una o Varias Columnas**

### a) **Agrupar por una sola columna**
```python
import pandas as pd

# Crear un DataFrame de ejemplo
data = {
    'producto': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'ventas': [100, 200, 300, 150, 250, 350, 400, 300],
}
df = pd.DataFrame(data)

# Agrupar por la columna 'producto' y calcular la suma de ventas
agrupado = df.groupby('producto')['ventas'].sum()
print(agrupado)
```
### Salida:
```
producto
A    600
B    750
C    650
Name: ventas, dtype: int64
```

### b) **Agrupar por varias columnas**
```python
# Agrupar por 'producto' y 'categoria' y calcular la suma de ventas
agrupado = df.groupby(['producto', 'categoria'])['ventas'].sum()
print(agrupado)
```
### Salida:
```
producto  categoria
A         electronicos    400
          muebles          200
B         electronicos    500
          muebles          300
C         electronicos    350
          muebles          300
Name: ventas, dtype: int64
```

## **2. Aplicar Agregados sobre Agrupaciones**

### a) **Suma**
```python
# Calcular la suma por producto
suma_ventas = df.groupby('producto')['ventas'].sum()
print(suma_ventas)
```

### b) **Media (promedio)**
```python
# Calcular la media de ventas por producto
media_ventas = df.groupby('producto')['ventas'].mean()
print(media_ventas)
```

### c) **Cuenta**
```python
# Contar el número de registros por producto
cuenta_productos = df.groupby('producto').size()
print(cuenta_productos)
```

### d) **Mínimo y Máximo**
```python
# Calcular el mínimo y máximo de ventas por producto
min_max_ventas = df.groupby('producto')['ventas'].agg(['min', 'max'])
print(min_max_ventas)
```

### e) **Función personalizada**
```python
# Aplicar una función personalizada
def diferencia_min_max(x):
    return x.max() - x.min()

diferencia_ventas = df.groupby('producto')['ventas'].agg(diferencia_min_max)
print(diferencia_ventas)
```

## **3. Pivot Tables para Agrupaciones**
La función `pivot_table` es útil para crear tablas pivoteadas donde puedes agrupar y resumir datos.

```python
# Crear una tabla pivote con la suma de ventas por producto y categoría
tabla_pivote = df.pivot_table(values='ventas', index='producto', columns='categoria', aggfunc='sum', fill_value=0)
print(tabla_pivote)
```
### Salida:
```
categoria  electronicos  muebles
producto                     
A                  400      200
B                  500      300
C                  350      300
```

## **4. Uso de .agg() para múltiples funciones**
Con `agg()` puedes aplicar múltiples funciones a las columnas en un solo paso.

```python
# Aplicar múltiples funciones al mismo tiempo
resumen_multiples = df.groupby('producto')['ventas'].agg(['sum', 'mean', 'max'])
print(resumen_multiples)
```
### Salida:
```
           sum       mean  max
producto                         
A          600  200.000000  400
B          750  250.000000  500
C          650  216.666667  350
```

## **5. Visualización con Agrupaciones**
```python
import matplotlib.pyplot as plt

# Agrupar por producto y calcular la suma de ventas
agrupado = df.groupby('producto')['ventas'].sum()

# Graficar las ventas agrupadas
agrupado.plot(kind='bar')
plt.title("Ventas por Producto")
plt.xlabel("Producto")
plt.ylabel("Ventas")
plt.show()
```

### **Ejemplo Completo**
```python
import pandas as pd

# Crear DataFrame de ejemplo
data = {
    'producto': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'categoria': ['electronicos', 'muebles', 'electronicos', 'electronicos', 'muebles', 'electronicos', 'muebles', 'muebles'],
    'ventas': [100, 200, 300, 150, 250, 350, 400, 300],
}
df = pd.DataFrame(data)

# Agrupar y calcular sumas de ventas
agrupado = df.groupby('producto')['ventas'].sum()
print("Ventas Totales por Producto:\n", agrupado)

# Aplicar múltiples funciones
resumen = df.groupby('producto')['ventas'].agg(['sum', 'mean', 'max'])
print("\nResumen Detallado:\n", resumen)

# Tabla pivote
tabla_pivote = df.pivot_table(values='ventas', index='producto', columns='categoria', aggfunc='sum', fill_value=0)
print("\nTabla Pivote:\n", tabla_pivote)

# Visualización
agrupado.plot(kind='bar')
plt.title("Ventas por Producto")
plt.xlabel("Producto")
plt.ylabel("Ventas")
plt.show()
```

### **Conclusión**
Las **agrupaciones** te permiten trabajar con datos organizados, y los **resúmenes** facilitan extraer insights significativos como sumas, promedios, o contar elementos. Con Pandas, puedes manejar esto de manera eficiente y clara.

**Lecturas recomendadas**

[Agrupaciones y Resumen de Datos.ipynb at main](https://github.com/platzi/etl-python/blob/main/13.%20Agrupaciones%20y%20Resumen%20de%20Datos/Agrupaciones%20y%20Resumen%20de%20Datos.ipynb)

[datos_ejemplo.csv](https://github.com/platzi/etl-python/blob/main/13.%20Agrupaciones%20y%20Resumen%20de%20Datos/datos_ejemplo.csv)

## Transformaciones Avanzadas

Las **transformaciones avanzadas** en Pandas te permiten modificar, transformar y limpiar los datos de una manera más compleja. Esto puede incluir el manejo de datos faltantes, combinaciones, segmentaciones, recodificaciones, o la creación de nuevas variables basadas en cálculos.

A continuación, te muestro algunas **transformaciones avanzadas** que puedes realizar con Pandas:

### **1. Manejo de Valores Faltantes**

```python
import pandas as pd
import numpy as np

# Crear DataFrame de ejemplo con NaN valores
data = {
    'producto': ['A', 'B', 'C', 'A', 'B', None, 'A', 'B'],
    'ventas': [100, 200, 300, 150, 250, None, 400, 300],
}
df = pd.DataFrame(data)

# Rellenar los valores faltantes con la media de la columna
df['ventas'].fillna(df['ventas'].mean(), inplace=True)
print("Valores faltantes rellenados:\n", df)

# Eliminar filas con valores faltantes
df_sin_nan = df.dropna()
print("\nFilas sin valores faltantes:\n", df_sin_nan)
```

### **2. Filtrar Datos por Condiciones**
```python
# Filtrar productos con ventas mayores a 200
productos_filtrados = df[df['ventas'] > 200]
print("\nProductos con ventas mayores a 200:\n", productos_filtrados)
```

### **3. Recodificación de Variables**

```python
# Recodear una columna de texto
df['producto_categoria'] = df['producto'].replace({'A': 'Electrónica', 'B': 'Muebles'})
print("\nRecodificación de categoría:\n", df)
```

### **4. Aplicar Funciones a Datos**
```python
# Aplicar una función personalizada para calcular el precio por cantidad
df['precio_por_cantidad'] = df.apply(lambda x: x['ventas'] / 2 if pd.notnull(x['ventas']) else 0, axis=1)
print("\nPrecio por cantidad:\n", df)
```

### **5. Crear Nuevas Variables Basadas en Condiciones**
```python
# Crear una nueva columna basada en condiciones
df['ventas_categorias'] = np.where(df['ventas'] > 250, 'Alto', 'Bajo')
print("\nVentas categorizadas:\n", df)
```

### **6. Agrupaciones y Transformaciones Combinadas**
```python
# Agrupar por producto y calcular la suma de ventas por producto
grupo_suma = df.groupby('producto')['ventas'].sum()
print("\nSuma de ventas por producto:\n", grupo_suma)

# Crear una nueva columna con el porcentaje de ventas por cada producto
df['porcentaje_ventas'] = df['ventas'] / df['ventas'].sum() * 100
print("\nPorcentaje de ventas:\n", df)
```

### **7. Redefinir Índices**
```python
# Cambiar el índice del DataFrame a la columna 'producto'
df.set_index('producto', inplace=True)
print("\nÍndice cambiado:\n", df)
```

### **8. Ordenar Datos**
```python
# Ordenar el DataFrame por la columna de ventas en orden descendente
df_ordenado = df.sort_values(by='ventas', ascending=False)
print("\nDataFrame ordenado:\n", df_ordenado)
```

### **9. Combinación de Datos**
```python
# Crear un DataFrame adicional para la combinación
data_extra = {
    'producto': ['A', 'B', 'C', 'D'],
    'precio_unitario': [10, 20, 15, 25]
}
df_extra = pd.DataFrame(data_extra)

# Fusionar ambos DataFrames por la columna 'producto'
df_combinado = df.merge(df_extra, on='producto', how='left')
print("\nDataFrame combinado:\n", df_combinado)
```

### **10. Transformación de Fechas**
```python
# Crear un DataFrame con fechas
data_fecha = {
    'fecha': ['2025-01-01', '2025-02-15', '2025-03-10'],
    'ventas': [100, 200, 300]
}
df_fecha = pd.DataFrame(data_fecha)

# Convertir la columna 'fecha' a tipo datetime
df_fecha['fecha'] = pd.to_datetime(df_fecha['fecha'])
print("\nFechas transformadas:\n", df_fecha)
```

### **11. Normalización de Datos**
```python
from sklearn.preprocessing import MinMaxScaler

# Normalizar la columna de ventas
scaler = MinMaxScaler()
df['ventas_normalizadas'] = scaler.fit_transform(df[['ventas']])
print("\nDatos normalizados:\n", df)
```

### **12. Resampling y Agregaciones por Fechas**
```python
# Crear un DataFrame adicional con fechas
data_fecha = {
    'fecha': pd.date_range(start='2025-01-01', periods=6, freq='M'),
    'ventas': [100, 200, 300, 150, 250, 350]
}
df_fecha = pd.DataFrame(data_fecha)

# Resamplear para calcular la suma mensual de ventas
df_resampleado = df_fecha.set_index('fecha').resample('M').sum()
print("\nResampleado mensual:\n", df_resampleado)
```

Estas **transformaciones avanzadas** son clave para trabajar con datos de manera eficiente y extraer información valiosa para los análisis.

### Resumen

### ¿Cómo aplicar técnicas avanzadas de manipulación de datos en Python?

La manipulación de datos es una habilidad fundamental para cualquier profesional que trabaje con grandes volúmenes de información. Con Python y sus bibliotecas potentes como Pandas, puedes llevar estas técnicas al siguiente nivel. En esta clase, descubrirás cómo aplicar transformaciones avanzadas utilizando funciones personalizadas, pivot tables, y diferentes métodos de 'join'. ¡Sumérgete en este fascinante mundo y descubre el poder de los datos!

### ¿Cómo leer y analizar múltiples archivos CSV?

Comencemos con una tarea básica pero crucial: leer archivos CSV. Para este ejemplo, trabajaremos con dos archivos CSV: uno con información de empleados y otro con datos de bonificaciones.

La estructura de ambos data frames puede visualizarse de la siguiente manera:

- **Empleados**:

- ID empleado

- Nombre

- Departamento

- Salario

- Fecha de ingreso

- **Bonificaciones**:

- ID empleado

- Bonificación

Es esencial recordar que ambos archivos tienen una columna en común: el ID del empleado. Esta columna nos servirá para futuras operaciones de 'join' entre data frames.

```python
import pandas as pd

empleados_df = pd.read_csv('empleados.csv')
bonificaciones_df = pd.read_csv('bonificaciones.csv')

print(empleados_df.head())
print(bonificaciones_df.head())
```

### ¿Cómo usar funciones personalizadas con el método apply?

El método `apply` te permite aplicar funciones personalizadas a las columnas de un data frame. Empezaremos creando una función sencilla para calcular el salario anual multiplicando el salario mensual por doce.

```python
def salario_anual(salario):
    return salario * 12

empleados_df['Salario Anual'] = empleados_df['Salario'].apply(salario_anual)
```

Un ejemplo más avanzado es calcular la antigüedad de un empleado basándose en la fecha de ingreso. Creamos una función para verificar si un empleado tiene más de cinco años de antigüedad.

```python
from datetime import datetime

def antiguedad_cinco_anos(fecha_ingreso):
    hoy = pd.to_datetime('today')
    antiguedad = (hoy - pd.to_datetime(fecha_ingreso)).days / 365
    return antiguedad > 5

empleados_df['Antigüedad > 5 Años'] = empleados_df['Fecha de ingreso'].apply(antiguedad_cinco_anos)
```

### ¿Qué son y cómo usar las tablas pivote (Pivot Tables)?

Las tablas pivote te permiten reorganizar los datos para obtener información agregada. En este contexto, generaremos una tabla que muestre el salario promedio por departamento.

```python
tabla_pivote = empleados_df.pivot_table(values='Salario', index='Departamento', aggfunc='mean')
print(tabla_pivote)
```

### ¿Cómo integrar data frames usando Merge y Join?

Finalmente, exploraremos cómo combinar información de múltiples fuentes. Utilizaremos `merge` para combinar los archivos de empleados y bonificaciones usando el ID de empleado y el método 'left'.

`df_merged = pd.merge(empleados_df, bonificaciones_df, on='ID empleado', how='left')`

Podemos también configurar índices y realizar 'joins' adicionales usando la columna Departamento para integrar información de ubicaciones.

```python
departamentos = {'Departamento': ['Ventas', 'IT', 'Recursos Humanos'],
                 'Ubicación': ['Madrid', 'Barcelona', 'Valencia']}
df_departamentos = pd.DataFrame(departamentos).set_index('Departamento')

df_combined = empleados_df.set_index('Departamento').join(df_departamentos)
```

### Desafío práctico: ¿Cómo aplicar lo aprendido a un contexto real?

Ahora que entiendes cómo manipular data frames con técnicas avanzadas en Python, te lanzo un desafío. Supón que tienes datos de productos en una tienda. Crea una función que calcule el total de ventas multiplicando cantidad por precio, y otra que clasifique productos como baratos, medios o caros. Aplica el método `apply` para poner en práctica lo aprendido y comparte tus soluciones.

La manipulación de datos es un mundo con infinitas posibilidades y desafíos. ¡Sigue así, promoviendo y perfeccionando tus habilidades en este viaje extraordinario a través de los datos!

**Lecturas recomendadas**

[Transformaciones Avanzadas.ipynb](https://github.com/platzi/etl-python/blob/main/14.%20Transformaciones%20Avanzadas/Transformaciones%20Avanzadas.ipynb)

[Empleados.csv](https://github.com/platzi/etl-python/blob/main/14.%20Transformaciones%20Avanzadas/empleados.csv)

[Bonificaciones.csv](https://github.com/platzi/etl-python/blob/main/14.%20Transformaciones%20Avanzadas/bonificaciones.csv)

## Carga de Datos en Archivos CSV

La **carga de datos en archivos CSV** es un paso clave para trabajar con datos utilizando Pandas en Python. Aquí te dejo una guía básica para cargar y manejar estos archivos:

### **Pasos para cargar datos desde un archivo CSV:**

```python
import pandas as pd

# 1. Cargar el archivo CSV
df = pd.read_csv("AB_NYC_2019.csv")  # Ajusta la ruta a tu archivo CSV

# 2. Verificar el contenido del DataFrame
print(df.head())  # Muestra las primeras 5 filas del DataFrame

# 3. Revisar los tipos de datos
print(df.dtypes)

# 4. Verificar información general del DataFrame
print(df.info())

# 5. Revisar los primeros 5 registros
print(df.head())

# 6. Revisar las columnas y el índice
print(df.columns)
print(df.index)
```

### **Errores comunes al cargar CSV**
- **Errores de delimitador**: Si ves errores como "C error: Expected 1 fields in line 37, saw 2", revisa el parámetro `sep`. El delimitador por defecto en muchos CSV es `,`, pero algunos usan `;`, `\t` o `|`.

### **Solución cuando el CSV tiene valores nulos**
- **Cómo manejar valores nulos**:
  - **1**: **Llenar valores faltantes**: Puedes usar `fillna()`. Por ejemplo:
    ```python
    df.fillna(0, inplace=True)  # Reemplaza valores nulos con 0
    ```
  - **2**: **Eliminar filas con nulos**:
    ```python
    df.dropna(inplace=True)  # Elimina todas las filas que tengan nulos
    ```
- **Verificar valores nulos**:
  ```python
  print(df.isnull().sum())  # Muestra la cantidad de valores nulos por columna
  ```

### **Conversión de tipos de datos**:
Si ves columnas que no tienen el tipo de dato deseado, puedes convertirlas con `astype()`:
```python
df['columna_numerica'] = df['columna_numerica'].astype(float)  # Convertir a float
df['columna_fecha'] = pd.to_datetime(df['columna_fecha'])  # Convertir a fecha
```

### **Guardar el DataFrame en otro archivo CSV**:
Una vez procesado el DataFrame, puedes guardarlo en un nuevo CSV:
```python
df.to_csv('processed_AB_NYC_2019.csv', index=False)  # Guardar sin el índice
```

Esta es la manera básica de manejar y cargar archivos CSV en Pandas. 

## Carga completa e Incremental en CSV

La **carga completa e incremental en CSV** son estrategias utilizadas para manejar la entrada de datos dependiendo de las necesidades y la cantidad de nuevos datos. Aquí te explico ambos enfoques:

### **1. Carga Completa de Datos**  
Este enfoque implica cargar todos los datos cada vez que actualizas tu DataFrame. Es útil cuando los datos cambian periódicamente y necesitas tener una copia completa del archivo CSV.

#### **Pasos para realizar una Carga Completa:**

```python
import pandas as pd

# 1. Cargar el CSV completo
df_nueva_carga = pd.read_csv("AB_NYC_2019.csv")  # Ajusta la ruta a tu archivo CSV

# 2. Verificar el contenido del nuevo DataFrame
print(df_nueva_carga.head())

# 3. Guardar el DataFrame completo en un nuevo CSV
df_nueva_carga.to_csv('AB_NYC_2019_completo.csv', index=False)  # Guardar los nuevos datos
```

### **2. Carga Incremental de Datos**  
Este enfoque es útil cuando solo necesitas cargar los nuevos datos que se han agregado desde la última carga, evitando cargar todo nuevamente. Esto es ideal cuando tienes un archivo CSV grande y solo quieres procesar los nuevos registros.

#### **Pasos para realizar una Carga Incremental:**

```python
import pandas as pd

# 1. Cargar el CSV existente
df_existente = pd.read_csv("AB_NYC_2019_completo.csv")

# 2. Cargar los nuevos datos del archivo CSV
df_nueva_carga = pd.read_csv("AB_NYC_2019.csv")

# 3. Identificar nuevos registros (esto puede hacerse por ejemplo, comparando por el campo de fecha)
df_nuevos = df_nueva_carga[~df_nueva_carga['id'].isin(df_existente['id'])]  # 'id' sería un campo único para comparar

# 4. Combinar los nuevos registros con el DataFrame existente
df_actualizado = pd.concat([df_existente, df_nuevos], ignore_index=True)

# 5. Guardar el DataFrame actualizado
df_actualizado.to_csv('AB_NYC_2019_incremental.csv', index=False)  # Guardar solo los nuevos datos
```

### **Consejos para manejar la Carga Incremental:**

- **Campo único**: Usa un campo único como `id`, `timestamp`, o `fecha` para identificar qué registros ya están en el dataset.
- **Validación de registros**: Asegúrate de realizar una validación adecuada para no perder registros duplicados ni dejar fuera datos relevantes.
- **Optimización**: Mantén la última versión del CSV comprimida para mejorar el rendimiento al cargar y almacenar los datos.

Con estas estrategias, puedes manejar eficientemente tanto la carga completa como incremental de tus archivos CSV, según las necesidades de tu análisis. 

## Particionado de datos en Python

El **particionado de datos** es una técnica fundamental en Machine Learning, especialmente cuando se trabaja con datasets grandes. El objetivo es dividir los datos en conjuntos de entrenamiento, validación y prueba para evaluar el rendimiento de los modelos. En Python, `sklearn` es la librería más utilizada para realizar esta tarea.

### **Métodos Comunes de Particionado de Datos:**

### **1. Partición Aleatoria (Random Split)**

Este método divide los datos en subconjuntos de forma aleatoria.

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Datos ficticios
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Características
y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # Etiquetas

# Partición aleatoria en 70% de entrenamiento y 30% de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Entrenamiento X:", X_train)
print("Prueba X:", X_test)
```

### **2. Partición Basada en Tiempo**

Este método es útil cuando los datos tienen una secuencia temporal y quieres mantener la relación temporal en los datos de entrenamiento y prueba.

```python
from sklearn.model_selection import TimeSeriesSplit

# Datos ficticios (con una secuencia temporal)
time_series_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Crear particiones basadas en el tiempo
tscv = TimeSeriesSplit(n_splits=3)  # 3 particiones

for train_index, test_index in tscv.split(time_series_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = time_series_data[train_index], time_series_data[test_index]
    print("X_train:", X_train, "X_test:", X_test)
```

### **3. Partición K-Folds**

Este método divide los datos en `k` partes iguales y realiza validación cruzada.

```python
from sklearn.model_selection import KFold

# Datos ficticios
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Partición K-Folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    print("X_train:", X_train, "X_test:", X_test)
```

### **4. Partición Proporcional a un grupo o clase**

Este método divide los datos según una proporción deseada, como por ejemplo las clases.

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Datos ficticios con clases desequilibradas
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])  # Etiquetas binarias

# Partición estratificada
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X_train:", X_train, "y_train:", y_train)
    print("X_test:", X_test, "y_test:", y_test)
```

### **Beneficios del Particionado de Datos**:
- **Entrenamiento**: Crear modelos robustos utilizando sólo los datos del conjunto de entrenamiento.
- **Validación**: Evaluar el rendimiento del modelo usando datos no vistos previamente.
- **Prueba**: Verificar el rendimiento final del modelo en datos completamente nuevos.

Con estos métodos puedes elegir el adecuado según las necesidades de tu proyecto, ya sea para modelos supervisados, análisis temporal, validación cruzada, etc.

**Lecturas recomendadas**

[Particionado de Datos en CSV.ipynb](https://github.com/platzi/etl-python/blob/main/17.%20Particionado%20de%20Datos%20en%20CSV/Particionado%20de%20Datos%20en%20CSV.ipynb)

[data_completo.csv](https://github.com/platzi/etl-python/blob/main/17.%20Particionado%20de%20Datos%20en%20CSV/data_completo.csv)

## Carga de Datos en Archivos Excel

La **carga de datos desde archivos Excel** en Python se realiza comúnmente con la biblioteca `pandas`. Esta ofrece métodos potentes y sencillos para leer y escribir datos en archivos Excel.

### **1. Instalación de Dependencias**
Asegúrate de tener instalada la biblioteca necesaria para manejar archivos Excel. Puedes instalarla con:

```bash
pip install pandas openpyxl
```

- `pandas`: Necesario para trabajar con datos estructurados.
- `openpyxl`: Requerido para leer y escribir archivos Excel con formato `.xlsx`.

### **2. Leer Datos desde un Archivo Excel**

#### **a) Cargar un Archivo Excel Sencillo**
```python
import pandas as pd

# Leer el archivo Excel
df = pd.read_excel("archivo.xlsx")  # Por defecto, lee la primera hoja
print(df.head())  # Mostrar las primeras filas del DataFrame
```

#### **b) Leer una Hoja Específica**
Si tu archivo Excel tiene múltiples hojas, puedes especificar cuál cargar:
```python
df = pd.read_excel("archivo.xlsx", sheet_name="Hoja1")
print(df.head())
```

#### **c) Leer Varias Hojas al Mismo Tiempo**
Para cargar múltiples hojas, puedes usar:
```python
dfs = pd.read_excel("archivo.xlsx", sheet_name=None)  # Carga todas las hojas como un diccionario
print(dfs.keys())  # Muestra los nombres de las hojas
print(dfs['Hoja1'])  # Accede a una hoja específica
```

#### **d) Seleccionar Columnas Específicas**
```python
df = pd.read_excel("archivo.xlsx", usecols=["Columna1", "Columna2"])
print(df)
```

### **3. Escribir Datos a un Archivo Excel**

#### **a) Guardar un DataFrame en Excel**
```python
# Crear un DataFrame de ejemplo
data = {
    "Nombre": ["Ana", "Luis", "Carlos"],
    "Edad": [28, 35, 40],
    "Ciudad": ["Madrid", "Bogotá", "México"]
}
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo Excel
df.to_excel("salida.xlsx", index=False)  # index=False evita guardar el índice
```

#### **b) Escribir en Múltiples Hojas**
```python
# Crear otro DataFrame
data2 = {
    "Producto": ["A", "B", "C"],
    "Precio": [10, 20, 30]
}
df2 = pd.DataFrame(data2)

# Guardar en un archivo Excel con múltiples hojas
with pd.ExcelWriter("salida_multiple.xlsx") as writer:
    df.to_excel(writer, sheet_name="Personas", index=False)
    df2.to_excel(writer, sheet_name="Productos", index=False)
```

### **4. Consejos Comunes**
- **Formato del Archivo**: Asegúrate de que el archivo tenga una extensión válida (`.xlsx`, `.xls`, `.xlsm`).
- **Dependencias**: Si tienes problemas, confirma que `openpyxl` esté correctamente instalado.
- **Errores Comunes**: Si obtienes un error de codificación o permisos, verifica que el archivo no esté abierto o en uso por otra aplicación.

### Resumen

### ¿Cómo guardar datos en archivos de Excel usando Python y Pandas?

Al manipular grandes cantidades de información, la capacidad para almacenar eficientemente los datos es invaluable. Usar Python junto con la librería Pandas puede simplificar esta tarea, permitiéndonos guardar nuestros datos en archivos de Excel, un formato ampliamente utilizado en el entorno empresarial. En este artículo, aprenderemos cómo generar archivos de Excel con nuestros datos, y exploraremos un ejemplo práctico de carga incremental de datos utilizando Python.

### ¿Cómo se generan archivos de Excel con Pandas?

Para empezar, la generación de archivos de Excel con Pandas implica crear un DataFrame a partir de nuestros datos. En este caso, trabajamos con diccionarios que convertimos en un DataFrame, el cual posteriormente exportamos a un archivo de Excel. Sigamos este proceso paso a paso:

1. **Configuración Inicial**: Importamos la librería Pandas, la cual es esencial para el manejo de datos en estructuras DataFrame.

3. **Generación de Datos**: Creamos un diccionario con los datos deseados, lo convertimos a DataFrame y lo almacenamos en un archivo de Excel.

```python
import pandas as pd

# Datos como un diccionario
datos = {
    "ID": [1, 2, 3],
    "Nombre": ["Karla", "Laura", "Luis"]
}

# Convertir a DataFrame
data_nueva1 = pd.DataFrame(datos)

# Exportar a un archivo Excel
data_nueva1.to_excel('data_nueva1.xlsx', index=False)
```

Este mismo método se repite para crear un segundo archivo data_nueva2.xlsx con otros datos.

3. **Verificación de Resultados**: Recomendamos utilizar un gestor de archivos para verificar que se hayan generado correctamente los archivos de Excel.

### ¿Qué es una carga incremental en archivos de Excel?

La carga incremental es un proceso eficiente para manejar grandes volúmenes de información en el que se actualizan gradualmente los registros existentes en un archivo. Este proceso asegura que solo se añadan al archivo los nuevos datos que se generen, sin duplicar información existente. Implementemos esta técnica:

1. **Definición de la Función**: Crearemos una función denominada cargaincremental_excel, encargada de administrar la lógica de la carga incremental.

```python
def cargaincremental_excel(data_nueva, data_completa_ruta):
    if os.path.exists(data_completa_ruta):
        data_completa = pd.read_excel(data_completa_ruta)
        data_completa = pd.concat([data_completa, data_nueva]).drop_duplicates()
    else:
        data_completa = data_nueva

    data_completa.to_excel(data_completa_ruta, index=False)
    print("Carga incremental realizada correctamente.")
```

2. **Ejecución de la Función**: Proveemos los datos nuevos y la ruta del archivo completo donde deseamos realizar la carga incremental.

```python
# Ejecutar Carga Incremental
cargaincremental_excel(data_nueva1, 'data_completa.xlsx')
```

### ¿Cómo manejar la actualización y combinación de datos?

Ahora que entendemos la carga incremental, el siguiente paso es actualizar nuestros datos. Combinaremos data existente con nuevos registros usando la función creada previamente. Un detalle importante a tener en cuenta es cerrar los archivos de Excel antes de modificarlos, ya que pueden presentarse errores de escritura si están abiertos.

- **Actualización de Datos**: Añadimos nuevos registros mediante la lectura de `data_nueva2.xlsx` y combinamos esta data con el archivo completo.

```python
# Nuevos datos para actualizar
data_nueva2 = pd.read_excel('data_nueva2.xlsx')

# Actualizar archivo completo
cargaincremental_excel(data_nueva2, 'data_completa.xlsx')
```

A medida que vamos avanzando, es crucial verificar siempre los resultados de nuestra manipulación de datos revisando los archivos generados. Nuestra carga incremental debe mostrar exitosamente todos los registros tanto de data_nueva1 como de data_nueva2.

Al manejar eficientemente la carga incremental, no solo optimizamos el manejo de nuestros datos, sino que también mitigamos los riesgos asociados con la duplicación de registros, asegurando una gestión ordenada y precisa de la información.

### ¿Cómo desafiarte a ti mismo con un ejercicio práctico?

Te invito a un reto emocionante: intenta replicar el mismo proceso que hemos realizado con archivos de Excel, pero usando otro formato de archivo como CSV. Utiliza las herramientas aprendidas para experimentar y expandir tus capacidades. Comparte tus experiencias y logros en los comentarios.

¡Ábrete camino en el amplio mundo del manejo de datos y continúa desarrollando tus habilidades que te llevarán a un nuevo nivel de destreza!

**Lecturas recomendadas**

[Carga de Datos en Archivos Excel.ipynb](https://github.com/platzi/etl-python/blob/main/18.%20Carga%20de%20Datos%20en%20Archivos%20Excel/Carga%20de%20Datos%20en%20Archivos%20Excel.ipynb)

## Configuración de MySQL y Python para el Proyecto ETL

La configuración de MySQL con Python es fundamental para proyectos ETL, ya que permite conectar y manipular datos desde una base de datos relacional. A continuación, se presenta un paso a paso para configurar MySQL y Python para tu proyecto ETL.

### **1. Instalación de MySQL**
Si no tienes MySQL instalado, sigue estos pasos:
1. Descarga MySQL desde [MySQL Downloads](https://dev.mysql.com/downloads/).
2. Instala MySQL Server y elige una contraseña para el usuario root durante la configuración.
3. (Opcional) Instala MySQL Workbench para gestionar la base de datos gráficamente.

### **2. Configuración del Entorno en Python**
Asegúrate de instalar las bibliotecas necesarias:

```bash
pip install mysql-connector-python pandas
```

- **`mysql-connector-python`**: Controlador para conectarte a MySQL desde Python.
- **`pandas`**: Necesario para manejar datos en Python.

### **3. Crear una Base de Datos en MySQL**
Puedes usar MySQL Workbench o la línea de comandos:

```sql
CREATE DATABASE proyecto_etl;
USE proyecto_etl;

CREATE TABLE ventas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    producto VARCHAR(50),
    cantidad INT,
    precio DECIMAL(10, 2),
    fecha DATE
);
```

### **4. Conectar Python a MySQL**
A continuación, se muestra cómo establecer una conexión básica:

```python
import mysql.connector

# Conexión a la base de datos
conexion = mysql.connector.connect(
    host="localhost",
    user="root",  # Cambia a tu usuario
    password="tu_contraseña",  # Cambia a tu contraseña
    database="proyecto_etl"
)

if conexion.is_connected():
    print("Conexión exitosa a la base de datos")
conexion.close()
```

### **5. Insertar Datos en la Base de Datos**
Puedes insertar datos manualmente o desde un archivo como CSV.

#### **a) Insertar Datos Manualmente**
```python
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="tu_contraseña",
    database="proyecto_etl"
)

cursor = conexion.cursor()

# Insertar datos
datos = ("Laptop", 5, 700.50, "2025-01-16")
query = "INSERT INTO ventas (producto, cantidad, precio, fecha) VALUES (%s, %s, %s, %s)"
cursor.execute(query, datos)

# Confirmar la transacción
conexion.commit()
print(f"Filas insertadas: {cursor.rowcount}")

cursor.close()
conexion.close()
```

#### **b) Insertar Datos desde un Archivo CSV**
```python
import pandas as pd

# Leer el archivo CSV
df = pd.read_csv("ventas.csv")

# Conectar a la base de datos
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="tu_contraseña",
    database="proyecto_etl"
)

cursor = conexion.cursor()

# Insertar filas del DataFrame
for _, row in df.iterrows():
    query = "INSERT INTO ventas (producto, cantidad, precio, fecha) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, tuple(row))

conexion.commit()
print(f"Filas insertadas: {cursor.rowcount}")

cursor.close()
conexion.close()
```

### **6. Leer Datos desde la Base de Datos**
Esto es útil para extraer datos en la fase inicial del ETL.

```python
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="tu_contraseña",
    database="proyecto_etl"
)

query = "SELECT * FROM ventas"
df = pd.read_sql(query, conexion)
print(df)

conexion.close()
```

### **7. Buenas Prácticas**
1. **Archivos de Configuración**: Guarda las credenciales de la base de datos en un archivo separado o usa variables de entorno para mayor seguridad.
2. **Control de Errores**: Implementa bloques `try-except` para manejar errores de conexión o consulta.
3. **Optimización**: Usa técnicas de carga por lotes para insertar grandes volúmenes de datos.

### Resumen

### ¿Cómo comenzar con el proyecto final setup?

Iniciar con el setup del proyecto final es una tarea crucial que te ayudará a consolidar cada aprendizaje del curso. Aquí, te guiaré en los pasos iniciales necesarios, asegurándome de que tengas toda la información necesaria para configurarlo exitosamente. El objetivo es trabajar con la base de datos Sakyla y establecer tu entorno de trabajo.

### ¿Dónde encontrar los recursos necesarios?

Para empezar, dirígete a la página de MySQL, donde encontrarás bases de datos que se utilizan en un entorno académico. La que necesitamos es **Sakyla Database**, que cuenta con información sobre clientes y alquileres de películas, entre otros datos. Será fundamental descargar la versión comprimida (zip) del archivo. También puedes encontrar este archivo en los recursos del curso facilitado por el instructor.

#### ¿Cómo descomprimir y previsualizar los elementos?

Una vez descargado el archivo de base de datos, descomprímelo. Al hacerlo, verás tres elementos esenciales:

- Datos a utilizar en las tablas.
- El esquema de la base de datos.
- La estructura de la base de datos.

### ¿Cómo realizar el setup en MySQL Workbench?

Configurar tu entorno de trabajo en MySQL Workbench es esencial. Aquí te explico cómo hacerlo paso a paso:

### ¿Cómo instalar MySQL Workbench?

Si aún no tienes instalado MySQL Workbench, es vital hacerlo. Puedes instalarlo directamente desde la página oficial de MySQL.

### ¿Cómo importar el esquema de la base de datos?

El siguiente paso es importar el esquema de la base de datos:

- Dentro de MySQL Workbench, selecciona la opción server data import.
- Opta por la opción de importar desde un archivo.
- Navega hasta el archivo esquema, schema.sql, en la base de datos Sakyla y ábrelo.
- Inicia el proceso de importación presionando start.

Se completará sin errores, estableciendo la primera parte del setup al tener el esquema de metadatos importado correctamente.

### ¿Cómo importar los datos de la base de datos?

Para importar los datos:

1. Selecciona `file` y luego open sql script.
2. Abre el script `data.sql`.
3. Ejecuta el script para que las inserciones sean procesadas.

Actualiza las tablas en tu base de datos para verificar que se hayan creado correctamente.

### ¿Cómo validar la importación y configurar el entorno?

### ¿Cómo verificar que los datos sean correctos?

La validación es una tarea crucial para confirmar que todo está funcionando adecuadamente. Aquí te explico cómo hacerlo:

- Crea un script SQL.
- Configura Sakyla como la base de datos por default.
- Ejecuta una consulta sencilla como `SELECT * FROM actor;`

`SELECT * FROM actor;`

Presiona el botón de ejecución para validar que los datos están correctos, revisando detalles como el nombre y apellido de los actores.

### ¿Cómo conectar MySQL con Python?

Instalar la librería de MySQL Connector para Python es el último paso del setup:

- Asegúrate de tener instalado MySQL Connector con el siguiente comando en Jupyter Notebook: 

`!pip install mysql-connector-python `

Esta librería es crucial para establecer la conexión desde Jupyter Notebook hacia nuestro entorno de MySQL.

### ¿Qué hacer en caso de problemas?

Si encuentras algún problema durante la instalación o algún paso del setup, no dudes en dejar un comentario. Estamos aquí para ayudar y asegurarnos de que puedas continuar sin inconvenientes con tu proyecto final. ¡Adelante, estás un paso más cerca de completar este emocionante proyecto de aprendizaje!

**Lecturas recomendadas**

[SetUp Proyecto](https://github.com/platzi/etl-python/tree/main/19.%20SetUp%20Proyecto)

## Planificación y Extracción de Datos desde MySQL

La planificación y extracción de datos desde MySQL en un proyecto ETL requiere una buena estructura para garantizar que los datos se extraigan de manera eficiente y segura. A continuación, te muestro los pasos esenciales:

## **1. Planificación de la Extracción**

### **a. Definir los Requerimientos de Datos**
- **¿Qué datos necesitas?** Identifica las tablas y columnas requeridas.
- **¿Con qué frecuencia necesitas los datos?** Determina si la extracción será completa (todos los datos) o incremental (solo los datos nuevos o actualizados).
- **¿Qué filtros aplicarás?** Especifica condiciones como rangos de fechas o valores específicos.

### **b. Evaluar el Volumen de Datos**
- **¿Cuántos registros?** Estima el tamaño de los datos.
- **Optimización**: Considera índices en las tablas para acelerar las consultas.

### **c. Asegurar la Conexión y Seguridad**
- Verifica las credenciales y permisos de acceso.
- Asegúrate de que el usuario de MySQL tenga los privilegios necesarios para realizar consultas.

## **2. Configuración del Entorno en Python**

### **a. Instalación de Librerías**
Asegúrate de tener instaladas las bibliotecas necesarias para conectarte a MySQL y trabajar con datos:

```bash
pip install mysql-connector-python pandas
```

## **3. Extracción de Datos desde MySQL**

### **a. Conexión a la Base de Datos**
Crea una conexión a MySQL desde Python:

```python
import mysql.connector

# Conectar a MySQL
conexion = mysql.connector.connect(
    host="localhost",       # Dirección del servidor MySQL
    user="etl_user",        # Usuario con acceso
    password="tu_password", # Contraseña del usuario
    database="proyecto_etl" # Base de datos a trabajar
)

if conexion.is_connected():
    print("Conexión exitosa a la base de datos")
conexion.close()
```

### **b. Consulta para Extracción de Datos**
Usa consultas SQL para extraer los datos que necesitas.

#### **Extracción Completa**
```python
import pandas as pd

# Conexión
conexion = mysql.connector.connect(
    host="localhost",
    user="etl_user",
    password="tu_password",
    database="proyecto_etl"
)

# Consulta SQL
query = "SELECT * FROM ventas;"
df = pd.read_sql(query, conexion)

# Mostrar datos
print(df.head())

conexion.close()
```

#### **Extracción Incremental**
Extrae datos nuevos o actualizados basándote en una columna como `fecha`.

```python
query = """
SELECT * 
FROM ventas
WHERE fecha >= '2025-01-01';
"""
df = pd.read_sql(query, conexion)
```

## **4. Optimización de la Extracción**

### **a. Filtrar los Datos en SQL**
Realiza filtros en la consulta SQL para reducir el volumen de datos extraídos.

```sql
SELECT producto, cantidad, precio, fecha
FROM ventas
WHERE cantidad > 5 AND fecha >= '2025-01-01';
```

### **b. Trabajar con Lotes**
Si el volumen de datos es grande, extrae los datos en partes.

```python
cursor = conexion.cursor()

# Consulta con límite
query = "SELECT * FROM ventas LIMIT 1000 OFFSET 0;"
cursor.execute(query)

for row in cursor.fetchall():
    print(row)

cursor.close()
```

## **5. Validación y Registro**
Asegúrate de registrar cada extracción:
- **Validación**: Comprueba que no haya datos corruptos o faltantes.
- **Registro**: Guarda un log con información de la extracción: fecha, registros extraídos, tiempo de ejecución.

## **6. Ejemplo Completo**
Combina todo en un flujo básico:

```python
import mysql.connector
import pandas as pd

# Configuración de conexión
conexion = mysql.connector.connect(
    host="localhost",
    user="etl_user",
    password="tu_password",
    database="proyecto_etl"
)

# Extracción de datos
query = "SELECT producto, cantidad, precio, fecha FROM ventas WHERE cantidad > 5;"
df = pd.read_sql(query, conexion)

# Mostrar resultados
print("Registros extraídos:", len(df))
print(df.head())

# Guardar en un archivo CSV
df.to_csv("ventas_filtradas.csv", index=False)

conexion.close()
```

## **7. Siguientes Pasos**
- **Transformación**: Limpia y prepara los datos.
- **Carga**: Inserta los datos procesados en otro sistema, como un Data Warehouse.

### Resumen

### ¿Qué vamos a hacer en este proyecto de ETL?

Prepara tus herramientas y conocimientos de Python y MySQL, porque en este proyecto llevaremos a cabo un proceso completo de Extracción, Transformación y Carga (ETL). Utilizaremos como base nuestra base de datos existente llamada "Shakira". Las transformaciones a realizar son únicas y permitirán optimizar y reorganizar los datos de la tabla de actores. Al finalizar, todo lo que transformemos se transferirá a una nueva base de datos, "Shakira ETL". Descubramos los emocionantes pasos que realizaremos a continuación.

### ¿Cuáles son las transformaciones clave del proyecto?

1. **Filtrado de actores**: Identificaremos aquellos actores cuyo nombre empieza con la letra 'a'.
2. **Creación de una columna completa para el nombre**: Concatenaremos las columnas `first name` y `last name` para formar la columna `full name`.
3. **Calcular la longitud del nombre**: Crearemos **namelends**, que reflejará la longitud total del nombre.
4. **Conversiones a mayúsculas**: Todos los nombres se convertirán a mayúsculas.
5. **Filtrado por longitud del nombre**: Mantendremos actores cuyo `full name` sea mayor de diez caracteres.
6. **Cálculo del año de registro**: Determinaremos el año en el cual cada actor se registró.
7. **Agrupación por apellido**: Agruparemos los datos por `last name` y contaremos cuántos actores comparten el mismo apellido.
8. **Indicador de nombre único**: Añadiremos una columna que marca si un actor tiene un nombre único.
9. **Estatus del actor**: Se agregará un estatus para indicar si el actor tiene más de diez años en la base de datos.
10. **Eliminación de una columna innecesaria**: Finalmente, se eliminará la columna last update.

### ¿Cómo nos conectamos a MySQL con Python?

Para comenzar, debemos garantizar una conexión estable con MySQL. Aquí están los pasos para conectarnos exitosamente:

1. **Importación de librerías necesarias**:

- `MySQL Connector`: Para manejar la conexión con MySQL.

- `Pandas`: Para realizar transformaciones de datos.

- `SQLAlchemy`: Para ayudar en la carga de datos.

- `datetime`: Para trabajar con campos de fecha.

```python
import mysql.connector
import pandas as pd
import sqlalchemy
from datetime import datetime
```

2. **Configuración de la conexión**:

- Se utiliza "localhost" para servidores locales.

- Proporciona tus credenciales como usuario y contraseña.

- Conéctate a la base de datos "Shakira".

```python
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='yourpassword',
    database='Shakira'
)
cursor = connection.cursor()
```

3. **Extracción de datos**:

- Utilizamos una consulta SQL simple para obtener todos los campos de la tabla de actores.

- Transformamos los resultados en un data frame de Pandas para su manipulación.

```python
query = "SELECT * FROM actor"
cursor.execute(query)
results = cursor.fetchall()
columns = [col[0] for col in cursor.description]
df_actors = pd.DataFrame(results, columns=columns)
```

4. **Seguridad de conexiones**:

- Siempre cierra la conexión cuando no es necesaria para evitar problemas de seguridad.

`connection.close()`

Ahora que hemos establecido la conexión y extraído los datos iniciales de MySQL usando Python, tenemos la base perfecta para iniciar las transformaciones. Si encontraste algún obstáculo, no dudes en compartirlo y juntos encontraremos la solución. Adelante, queda mucho por explorar y aprender en próximas etapas. ¡Nos vemos en el siguiente reto ETL!

## Transformación de datos con Python

La **transformación de datos** es un paso esencial en un flujo ETL. En este proceso, se limpian, modifican y estructuran los datos para que sean útiles y consistentes para su análisis o carga en otro sistema. Python, con su ecosistema de bibliotecas como **Pandas**, facilita este proceso.

## **1. Etapas Principales de la Transformación de Datos**

### **a. Limpieza de Datos**
- Manejo de valores nulos.
- Eliminación de duplicados.
- Corrección de valores erróneos.

### **b. Modificación y Estandarización**
- Cambios en el formato de columnas (fechas, texto, etc.).
- Creación de nuevas columnas basadas en datos existentes.
- Conversión de tipos de datos.

### **c. Filtrado y Reducción**
- Filtrar filas o columnas irrelevantes.
- Agrupar y resumir datos.

### **d. Enriquecimiento**
- Combinación de datos con otras fuentes.
- Cálculo de métricas adicionales.

## **2. Herramientas en Python para Transformación**

La biblioteca **Pandas** es la herramienta más popular para la manipulación y transformación de datos en Python.

```bash
pip install pandas
```

## **3. Ejemplo Práctico de Transformación**

Supongamos que tienes un conjunto de datos con las ventas de productos en un archivo CSV:

```plaintext
producto,cantidad,precio_unitario,fecha
Laptop,2,700,2025-01-01
Mouse,10,20,2025-01-02
Teclado,,50,2025-01-03
Monitor,5,200,2025-01-01
```

### **a. Cargar los Datos**
```python
import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("ventas.csv")

# Mostrar el DataFrame
print(df)
```

### **b. Limpieza de Datos**
#### **Manejo de Valores Nulos**
1. Reemplazar valores nulos con un valor por defecto:
```python
df['cantidad'].fillna(0, inplace=True)
```

2. Eliminar filas con valores nulos:
```python
df.dropna(inplace=True)
```

#### **Eliminar Duplicados**
```python
df.drop_duplicates(inplace=True)
```

### **c. Creación de Nuevas Columnas**
Agregar una columna `total_venta` calculada a partir de `cantidad` y `precio_unitario`:

```python
df['total_venta'] = df['cantidad'] * df['precio_unitario']
```

### **d. Conversión de Tipos de Datos**
Convertir la columna `fecha` a formato de fecha:

```python
df['fecha'] = pd.to_datetime(df['fecha'])
```

### **e. Filtrado de Datos**
Filtrar productos con un `total_venta` mayor a 500:

```python
df_filtrado = df[df['total_venta'] > 500]
```

### **f. Agrupación y Resumen**
Calcular las ventas totales por producto:

```python
ventas_por_producto = df.groupby('producto')['total_venta'].sum()
print(ventas_por_producto)
```

### **g. Ordenar Datos**
Ordenar el DataFrame por `total_venta` en orden descendente:

```python
df.sort_values(by='total_venta', ascending=False, inplace=True)
```

## **4. Transformación Avanzada**

### **a. Aplicar Funciones Personalizadas**
Transformar texto, por ejemplo, convertir los nombres de productos a mayúsculas:

```python
df['producto'] = df['producto'].apply(lambda x: x.upper())
```

### **b. Combinar DataFrames**
Unir datos de dos DataFrames (ejemplo: categorías de productos):

```python
categorias = pd.DataFrame({
    'producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor'],
    'categoria': ['Electrónica', 'Accesorios', 'Accesorios', 'Electrónica']
})

df = pd.merge(df, categorias, on='producto', how='left')
```

### **c. Pivot Tables**
Crear una tabla dinámica para analizar las ventas por fecha:

```python
pivot = df.pivot_table(
    values='total_venta',
    index='fecha',
    columns='producto',
    aggfunc='sum',
    fill_value=0
)
print(pivot)
```

## **5. Guardar los Datos Transformados**
Guarda el DataFrame transformado en un archivo CSV o Excel:

```python
df.to_csv("ventas_transformadas.csv", index=False)
df.to_excel("ventas_transformadas.xlsx", index=False)
```

## **6. Código Completo**
```python
import pandas as pd

# Cargar datos
df = pd.read_csv("ventas.csv")

# Limpieza
df['cantidad'].fillna(0, inplace=True)
df['fecha'] = pd.to_datetime(df['fecha'])

# Transformaciones
df['total_venta'] = df['cantidad'] * df['precio_unitario']
df.sort_values(by='total_venta', ascending=False, inplace=True)

# Agrupación
ventas_por_producto = df.groupby('producto')['total_venta'].sum()

# Guardar
df.to_csv("ventas_transformadas.csv", index=False)
```

## **7. Siguientes Pasos**
- **Integración**: Integra las transformaciones en un flujo ETL completo.
- **Optimización**: Usa herramientas como **Dask** para manejar grandes volúmenes de datos.
- **Automatización**: Configura tareas programadas con Python para automatizar la transformación.

### Resumen

### ¿Cómo transformar datos en Python y MySQL?

Transformar datos es un proceso esencial en la manipulación de bases de datos. A menudo, es necesario manipular los datos mediante diversas transformaciones para adaptarse a los requerimientos específicos del análisis. En esta guía exploramos cómo llevar a cabo estas transformaciones tanto en Python como en MySQL, resaltando la importancia de obtener resultados coherentes y precisos en ambas plataformas.

### ¿Cómo se realiza el filtrado de datos?

La primera transformación que realizaremos es filtrar los actores cuyo primer nombre comienza con la letra 'A'. Este es un excelente ejemplo de la facilidad y potencia de las funciones de filtrado en Python.

```python
# Filtrando actores con primer nombre que comienza con 'A'
actores_filtrados = datos_actores[datos_actores['first_name'].str.startswith('A')].copy()
```

En MySQL, el equivalente es usar la cláusula `LIKE` para seleccionar registros relevantes:

`SELECT * FROM actor WHERE first_name LIKE 'A%';`

Este proceso es crucial para verificar que las transformaciones logren el mismo resultado en ambas plataformas, lo cual es vital para la consistencia de los datos.

### ¿Cómo concatenar y calcular longitudes?

Para concatenar los nombres (first name y last name) y crear una nueva columna llamada `full_name` en Python, simplemente usamos:

```python
# Concatenar nombre y apellido
datos_actores['full_name'] = datos_actores['first_name'] + " " + datos_actores['last_name']
```

En MySQL, esta tarea se realizaría con la función `CONCAT`:

`SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM actor;`

Además, podemos calcular la longitud de `full_name` de la siguiente manera:

```python
# Calcular la longitud del nombre completo
datos_actores['name_length'] = datos_actores['full_name'].apply(len)
```

Mientras que en MySQL usamos:

`SELECT LENGTH(CONCAT(first_name, ' ', last_name)) AS name_length FROM actor;`

### ¿Cómo aplicar transformaciones adicionales?

Existen diversas transformaciones que podemos realizar, tales como cambiar los nombres a mayúsculas o filtrar según la longitud del nombre completo. Estas transformaciones nos permiten manipular y entender mejor nuestros datos:

```python
# Convertir a mayúsculas
datos_actores['first_name'] = datos_actores['first_name'].str.upper()
datos_actores['last_name'] = datos_actores['last_name'].str.upper()

# Filtrar por longitud del nombre
actores_filtrados = datos_actores[datos_actores['name_length'] > 10]
```

Para MySQL, utilizamos las funciones `UPPER` y cláusulas adicionales para el filtrado:

```python
SELECT UPPER(first_name) AS first_name, UPPER(last_name) AS last_name FROM actor;
SELECT * FROM actor WHERE LENGTH(CONCAT(first_name, ' ', last_name)) > 10;
```

### ¿Cómo manejar datos temporales y agrupamientos?

Los métodos de transformación permiten manejar datos temporales, como calcular el año de registro de un actor:

```python
# Calcular el año de registro
datos_actores['registration_year'] = pd.to_datetime(datos_actores['last_update']).dt.year
```

En MySQL, esto se consigue usando la función `YEAR`:

`SELECT YEAR(last_update) AS registration_year FROM actor;`

El agrupamiento y conteo por apellidos es otra transformación útil para analizar los datos:

```python
# Agrupar por apellido y contar
apellido_conteo = datos_actores.groupby('last_name').size().reset_index(name='count')
```

Y en MySQL, utilizamos GROUP BY:

`SELECT last_name, COUNT(actor_id) as count FROM actor GROUP BY last_name;`

### ¿Cómo evaluar datos únicos y estado de actores?

Identificar si un nombre es único y evaluar el estado de un actor basado en su tiempo en la base de datos son transformaciones que pueden ofrecer información valiosa:

```python
# Verificar nombres únicos
datos_actores['unique_name'] = datos_actores['full_name'].duplicated(False)

# Añadir estado del actor
datos_actores['actor_status'] = datos_actores['registration_year'].apply(lambda x: 'Activo' if 2023 - x > 10 else 'Inactivo')
```

Estos scripts demuestran cómo aprovechar las capacidades tanto de Python como de MySQL para gestionar y transformar datos de forma efectiva. Esta práctica nos permite tener un entendimiento más profundo de los datos y asegurarnos de que están listos para análisis y reportes. ¡Sigue practicando y observa cómo tus habilidades continúan desarrollándose!

**Lecturas recomendadas**

[Códigos de SQL.sql](https://github.com/platzi/etl-python/blob/main/20%20y%2021.%20Proyecto%20de%20ETL/Codigos%20de%20SQL.sql)

[Ejercicio ETL - Sakila.ipynb](https://github.com/platzi/etl-python/blob/main/20%20y%2021.%20Proyecto%20de%20ETL/Ejercicio%20ETL%20-%20Sakila.ipynb)

## Manejo de errores y buenas prácticas de ETL

El manejo de errores y la implementación de buenas prácticas son fundamentales para garantizar que los procesos ETL (Extracción, Transformación y Carga) sean confiables, eficientes y escalables. A continuación, se describen estrategias y recomendaciones para abordar ambos aspectos:

## **1. Manejo de Errores en Procesos ETL**

Los errores en ETL pueden ocurrir en cualquiera de las tres etapas: **Extracción**, **Transformación** o **Carga**. Aquí se detallan los más comunes y cómo gestionarlos:

### **a. Extracción de Datos**
**Errores comunes:**
- Fuentes de datos inaccesibles (API caídas, bases de datos no conectadas).
- Archivos mal formateados o corruptos.
- Inconsistencia en los esquemas de datos.

**Buenas prácticas:**
1. **Validar la conexión antes de extraer:**
   ```python
   try:
       conn = create_connection()
   except Exception as e:
       print(f"Error al conectar a la base de datos: {e}")
   ```

2. **Comprobar el formato de los datos:**
   ```python
   try:
       df = pd.read_csv("archivo.csv")
   except pd.errors.ParserError as e:
       print(f"Error al leer el archivo CSV: {e}")
   ```

3. **Manejo de APIs:**
   - Reintentar solicitudes en caso de fallo.
   - Establecer límites de tiempo para evitar bloqueos.
   ```python
   import requests
   from requests.exceptions import RequestException

   try:
       response = requests.get("https://api.example.com/data", timeout=5)
       response.raise_for_status()
   except RequestException as e:
       print(f"Error al extraer datos de la API: {e}")
   ```

### **b. Transformación de Datos**
**Errores comunes:**
- Valores nulos o inconsistentes.
- Problemas con el tipo de datos (e.g., texto en una columna numérica).
- Operaciones mal definidas (e.g., divisiones por cero).

**Buenas prácticas:**
1. **Manejo de valores nulos:**
   ```python
   df['columna'].fillna(valor_por_defecto, inplace=True)
   ```

2. **Validación de tipos de datos:**
   ```python
   try:
       df['columna'] = df['columna'].astype(float)
   except ValueError as e:
       print(f"Error al convertir tipos de datos: {e}")
   ```

3. **Evitar divisiones por cero:**
   ```python
   df['resultado'] = df['numerador'] / df['denominador'].replace(0, 1)
   ```

### **c. Carga de Datos**
**Errores comunes:**
- Restricciones en la base de datos (duplicados, tipos de datos no válidos).
- Conexiones fallidas.
- Tamaño excesivo de los datos.

**Buenas prácticas:**
1. **Validar antes de insertar:**
   ```python
   if not df.empty:
       insertar_datos(df)
   else:
       print("El DataFrame está vacío, no se cargaron datos.")
   ```

2. **Control de duplicados:**
   ```python
   df.drop_duplicates(subset='id', inplace=True)
   ```

3. **Carga en lotes para datos grandes:**
   ```python
   for chunk in pd.read_csv("archivo.csv", chunksize=1000):
       cargar_datos(chunk)
   ```

## **2. Buenas Prácticas Generales para ETL**

### **a. Modularidad**
Divide el proceso en funciones o clases bien definidas:
- `extraer_datos()`
- `transformar_datos()`
- `cargar_datos()`

Esto facilita la depuración, el mantenimiento y la reutilización del código.

### **b. Validación y Pruebas**
1. **Pruebas unitarias:** 
   Prueba funciones individuales para cada etapa del ETL.
   ```python
   def test_cargar_datos():
       assert cargar_datos(datos_prueba) == "Carga exitosa"
   ```

2. **Verificaciones de integridad:**
   - Validar la consistencia de los datos después de cada transformación.
   - Comprobar si los datos cargados coinciden con los esperados.

### **c. Registro de Errores (Logging)**
Utiliza el módulo `logging` para registrar errores y eventos clave:
```python
import logging

logging.basicConfig(
    filename='etl.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    df = pd.read_csv("archivo.csv")
except Exception as e:
    logging.error(f"Error al cargar el archivo: {e}")
```

### **d. Automatización y Programación**
- Usa **cron jobs** o bibliotecas como **APScheduler** para ejecutar el ETL automáticamente en intervalos definidos.
- Ejemplo con `APScheduler`:
   ```python
   from apscheduler.schedulers.blocking import BlockingScheduler

   def ejecutar_etl():
       extraer_datos()
       transformar_datos()
       cargar_datos()

   scheduler = BlockingScheduler()
   scheduler.add_job(ejecutar_etl, 'interval', hours=1)
   scheduler.start()
   ```

### **e. Monitorización**
1. **Alertas:** Configura notificaciones por correo o Slack para errores críticos.
2. **Dashboard de estado:** Usa herramientas como Grafana para visualizar métricas de ETL.

## **3. Código Completo de Ejemplo**
```python
import pandas as pd
import logging

# Configuración de logging
logging.basicConfig(filename='etl.log', level=logging.INFO)

def extraer_datos():
    try:
        df = pd.read_csv("ventas.csv")
        logging.info("Datos extraídos correctamente.")
        return df
    except Exception as e:
        logging.error(f"Error al extraer datos: {e}")
        return None

def transformar_datos(df):
    try:
        df['total_venta'] = df['cantidad'] * df['precio_unitario']
        df.dropna(inplace=True)
        logging.info("Datos transformados correctamente.")
        return df
    except Exception as e:
        logging.error(f"Error al transformar datos: {e}")
        return None

def cargar_datos(df):
    try:
        # Simula una carga
        print("Datos cargados exitosamente.")
        logging.info("Datos cargados correctamente.")
    except Exception as e:
        logging.error(f"Error al cargar datos: {e}")

# Ejecución del ETL
datos = extraer_datos()
if datos is not None:
    datos_transformados = transformar_datos(datos)
    if datos_transformados is not None:
        cargar_datos(datos_transformados)
```

## **4. Conclusión**
Implementar un manejo robusto de errores y seguir buenas prácticas garantiza que los procesos ETL sean:
- **Confiables:** Menos susceptibles a fallos.
- **Escalables:** Fáciles de adaptar a mayores volúmenes de datos.
- **Mantenibles:** Sencillos de entender y modificar.

### Resumen

### ¿Cuáles son las mejores prácticas en procesos ETL?

Cuando te aventuras en el mundo del procesamiento de datos, una de las primeras paradas que encontrarás será el proceso ETL (Extract, Transform, Load). Este proceso es crucial para la correcta gestión e integración de datos. Hoy te traigo algunas buenas prácticas indispensables para optimizar tus flujos de trabajo ETL. ¡Vamos a ello!

1. **Modularización y documentación**: Una de las mejores maneras de organizar tus procesos ETL es modularizando el código. Divide tu código en secciones manejables y asegúrate de documentarlo de tal forma que cualquier miembro del equipo pueda entender qué se está haciendo y por qué. Herramientas como Jupyter Notebook pueden ser de gran ayuda, ya que permiten integrar código y documentación en un solo archivo.

2. **Optimización del rendimiento**: Al trabajar con consultas, es crucial evaluar el tiempo de ejecución como un parámetro de referencia. Al medir la eficiencia de una query, podrás identificar áreas de mejora para optimizar el rendimiento del sistema.

3. **Automatización de procesos**: Automatiza tanto como sea posible para mejorar la efectividad y eliminar errores humanos. Esto garantizará que el proceso ETL se ejecute de manera consistente.

4. **Validación de datos**: Realiza validaciones tanto antes como después de la carga. Esto te permitirá interceptar errores potenciales y asegurar que los datos que manejas son precisos.

### ¿Cómo evitar errores comunes en procesos ETL?

A pesar de las mejores intenciones, los errores en los procesos ETL son comunes y pueden afectar gravemente la calidad de los datos. Aquí tienes una lista de los más frecuentes y cómo afrontarlos:

- **Formato y tipos de datos incorrectos**: Después de las transformaciones, asegúrate de que los formatos y tipos de datos de tus columnas sean los adecuados. Esto puede evitar sorpresas desagradables más adelante.

- **Datos duplicados**: Es muy fácil que se dupliquen los datos durante el proceso ETL. Vigila y elimina los duplicados para mantener la integridad referencial de tus datos.

- **Datos nulos o faltantes**: Los cruces de datos pueden provocar la aparición de valores nulos. Implementa transformaciones necesarias para corregir este problema y asegurar que los conjuntos de datos estén completos.

### ¿Cómo documentar correctamente un flujo ETL?

La documentación es una piedra angular en cualquier flujo de datos. Estos son los aspectos que jamás deberías olvidar a la hora de documentar tus procesos ETL:

- **Roles y responsabilidades**: Define claramente quién es responsable de cada etapa del proceso. Esto asegura transparencia y facilita la asignación de tareas.

- **Descripción del flujo de datos**: Asegúrate de que haya una descripción clara de qué consiste el flujo de datos, qué tablas se usan, y cuál es el contexto general.

- **Especificaciones de transformaciones**: Documenta por qué se realizan ciertas transformaciones y el propósito detrás de ellas. Esto te ayudará a justificar las decisiones tomadas durante el proceso.

- **Manejo de errores y auditoría**: Define cómo se manejarán las excepciones y cómo incorporar aspectos de auditoría y control de versiones. Esto te permite rastrear cambios y errores de manera eficiente.

Seguridad y gestión de accesos: Establece lineamientos claros para la seguridad de los datos y el control de acceso a los mismos, asegurando la protección de la información sensitiva.

## Carga de datos en ETL

En el contexto de un proceso ETL (Extracción, Transformación y Carga), la etapa de **Carga de Datos** se refiere al paso final en el que los datos procesados y transformados se insertan en su destino final. Este destino puede ser un almacén de datos (data warehouse), una base de datos, o incluso un archivo.

## **1. Métodos Comunes de Carga de Datos**

### **a. Carga Completa**
Se carga todo el conjunto de datos en el destino, sobrescribiendo cualquier dato existente. Este método se usa generalmente cuando:
- Se realiza la carga inicial del sistema.
- Los datos cambian completamente en cada ciclo de ETL.

**Ejemplo:**
```python
import pandas as pd
from sqlalchemy import create_engine

# Crear una conexión a la base de datos
engine = create_engine("mysql+pymysql://usuario:contraseña@localhost/base_de_datos")

# Cargar todo el DataFrame
df.to_sql("tabla_destino", con=engine, if_exists="replace", index=False)
```

### **b. Carga Incremental**
Solo se cargan los datos nuevos o actualizados desde la última carga. Es más eficiente y adecuado para sistemas con grandes volúmenes de datos.

**Pasos comunes:**
1. Identificar registros nuevos o modificados (por ejemplo, con una columna `timestamp` o un identificador único).
2. Insertar solo esos registros en el destino.

**Ejemplo:**
```python
# Seleccionar registros modificados después de una fecha específica
nuevos_datos = df[df['fecha_modificacion'] > ultima_fecha_carga]

# Agregar los datos nuevos a la tabla existente
nuevos_datos.to_sql("tabla_destino", con=engine, if_exists="append", index=False)
```

## **2. Estrategias de Carga por Tipo de Destino**

### **a. Almacén de Datos (Data Warehouse)**
Los datos procesados se cargan en un esquema optimizado para análisis, como un esquema estrella o copo de nieve. Herramientas como Apache Airflow y Talend son útiles para este tipo de cargas.

### **b. Bases de Datos Relacionales**
1. **Carga directa con Python:**
   Utilizando bibliotecas como `SQLAlchemy` o `pymysql`:
   ```python
   import pymysql

   connection = pymysql.connect(
       host="localhost",
       user="usuario",
       password="contraseña",
       database="base_de_datos"
   )

   cursor = connection.cursor()
   for _, row in df.iterrows():
       query = f"""
       INSERT INTO tabla (col1, col2, col3)
       VALUES ({row['col1']}, {row['col2']}, {row['col3']})
       """
       cursor.execute(query)
   connection.commit()
   cursor.close()
   ```

2. **Validación antes de la carga:**
   - Asegúrate de que los datos cumplan con las restricciones de la base de datos.
   - Evita duplicados si es necesario.

### **c. Archivos**
1. **Formato CSV:**
   ```python
   df.to_csv("archivo_salida.csv", index=False)
   ```

2. **Formato Excel:**
   ```python
   df.to_excel("archivo_salida.xlsx", index=False)
   ```

3. **JSON:**
   ```python
   df.to_json("archivo_salida.json", orient="records", lines=True)
   ```

## **3. Buenas Prácticas en la Carga de Datos**

### **a. Uso de Transacciones**
- Garantiza que la carga sea atómica y se pueda revertir en caso de error.
```python
with engine.begin() as connection:
    connection.execute("INSERT INTO tabla ...")
```

### **b. Control de Errores**
Registra y maneja los errores durante la carga:
```python
try:
    df.to_sql("tabla_destino", con=engine, if_exists="append", index=False)
except Exception as e:
    print(f"Error durante la carga: {e}")
```

### **c. Monitorización**
- Usa herramientas de registro (`logging`) para rastrear el progreso.
- Implementa métricas para validar la carga (e.g., número de registros cargados).

## **4. Código Completo de Ejemplo**

```python
import pandas as pd
from sqlalchemy import create_engine

# Configuración de la base de datos
engine = create_engine("mysql+pymysql://usuario:contraseña@localhost/base_de_datos")

# Extracción de datos (simulada)
df = pd.read_csv("datos.csv")

# Transformación de datos (simulada)
df['total'] = df['cantidad'] * df['precio_unitario']

# Carga de datos
try:
    df.to_sql("ventas", con=engine, if_exists="replace", index=False)
    print("Datos cargados exitosamente.")
except Exception as e:
    print(f"Error durante la carga: {e}")
```

## **5. Conclusión**
La etapa de **carga** en un proceso ETL debe ser eficiente, segura y confiable para garantizar la integridad y disponibilidad de los datos en el destino final. La elección entre carga completa o incremental, junto con el uso de buenas prácticas, determinará el éxito de este proceso.

### Resumen

El proceso de creación de una base de datos y sus tablas es crucial en el manejo de datos, especialmente cuando se trata de proyectos ETL (Extract, Transform, Load). En este ejercicio, hemos usado MySQL junto con Python para demostrar cómo se puede automatizar esta tarea.

Primero, verificamos si la base de datos existe y, si no es así, la creamos. Usamos el siguiente comando SQL:

```python
CREATE DATABASE IF NOT EXISTS Akila;
USE Akila;
```

Una vez creada la base de datos, procedemos a la creación de las tablas. Por ejemplo, la tabla `actor transformado` con su llave primaria y campos personalizados:

```python
CREATE TABLE IF NOT EXISTS actor_transformado (
  actor_id INT PRIMARY KEY,
  first_name VARCHAR(50),
  last_name VARCHAR(50),
  description TEXT
);
```

Esta tabla almacenará transformaciones hechas sobre nuestros datos de actores, especificando campos como `first_name` y `last_name`.

### ¿Cómo ejecutamos las consultas SQL usando Python?

Para garantizar la integridad del proceso, Python ofrece herramientas que facilitan la conexión con bases de datos SQL, como `MySQL Connector` y `SQLAlchemy`. Aquí se explica cómo manejar estas operaciones en Python.

1. **Conexión al servidor**: Primeramente, establecemos la conexión al servidor de base de datos y creamos un cursor.

```python
import mysql.connector

conn = mysql.connector.connect( host="tu_host", user="tu_usuario", password="tu_contraseña", database="Akila" ) cursor = conn.cursor()
```

2. **Ejecutar las consultas**: Tanto para crear la base de datos como las tablas, ejecutamos las consultas con el cursor.

```python
cursor.execute("CREATE DATABASE IF NOT EXISTS Akila;") cursor.execute("USE Akila;") cursor.execute(""" CREATE TABLE IF NOT EXISTS actor_transformado ( actor_id INT PRIMARY KEY, first_name VARCHAR(50), last_name VARCHAR(50), description TEXT ); """)
```

3. **Confirmar cambios y cerrar conexión**: Después de ejecutar las consultas, confirmamos los cambios y cerramos la conexión.

`conn.commit() cursor.close() conn.close()`

### ¿Cómo validar la carga y transformación de los datos con Python?

Una vez creadas las tablas, la carga de datos se realiza usando Pandas y SQLAlchemy. Este proceso recopila los datos, los transforma según necesidades y los guarda en las tablas SQL.

Conectar y cargar datos: Usamos pandas para leer y transformar datos, luego los cargamos a nuestras tablas con SQLAlchemy.

```python
from sqlalchemy import create_engine import pandas as pd

engine = create_engine('mysql+mysqlconnector://tu_usuario:tu_contraseña@tu_host/Akila') df = pd.DataFrame({ 'actor_id': [1, 2], 'first_name': ['John', 'Jane'], 'last_name': ['Doe', 'Doe'], 'description': ['Actor principal', 'Actor secundario'] }) df.to_sql('actor_transformado', engine, if_exists='replace', index=False)
```

2. **Validar carga de datos**: Validamos usando queries SQL para asegurar que los datos estén correctamente cargados.

```python
query = "SELECT * FROM actor_transformado;" df_result = pd.read_sql(query, engine) print(df_result)
```

Este proceso muestra cómo utilizar la programación para gestionar bases de datos de manera eficiente. Además, al terminar el ejercicio, es una excelente práctica verificar manualmente los datos en MySQL para garantizar que todo esté en su lugar. ¿Has logrado ejecutar este proyecto? ¿Qué retos enfrentaste? ¡Cuéntanos en los comentarios para que podamos ayudarte! Sigue adelante, cada paso cuenta en tu proceso de aprendizaje.

## Ética y Privacidad en la Gestión de Datos.

### ¿Cómo se crea una base de datos y tablas en MySQL?

El proceso de creación de una base de datos y sus tablas es crucial en el manejo de datos, especialmente cuando se trata de proyectos ETL (Extract, Transform, Load). En este ejercicio, hemos usado MySQL junto con Python para demostrar cómo se puede automatizar esta tarea.

Primero, verificamos si la base de datos existe y, si no es así, la creamos. Usamos el siguiente comando SQL:

`CREATE DATABASE IF NOT EXISTS Akila; USE Akila;`

Una vez creada la base de datos, procedemos a la creación de las tablas. Por ejemplo, la tabla actor transformado con su llave primaria y campos personalizados:
`CREATE TABLE IF NOT EXISTS actor_transformado ( actor_id INT PRIMARY KEY, first_name VARCHAR(50), last_name VARCHAR(50), description TEXT );`

Esta tabla almacenará transformaciones hechas sobre nuestros datos de actores, especificando campos como first_name y last_name.

### ¿Cómo ejecutamos las consultas SQL usando Python?

Para garantizar la integridad del proceso, Python ofrece herramientas que facilitan la conexión con bases de datos SQL, como MySQL Connector y SQLAlchemy. Aquí se explica cómo manejar estas operaciones en Python.

Conexión al servidor: Primeramente, establecemos la conexión al servidor de base de datos y creamos un cursor.
import mysql.connector

`conn = mysql.connector.connect( host="tu_host", user="tu_usuario", password="tu_contraseña", database="Akila" ) cursor = conn.cursor()`

Ejecutar las consultas: Tanto para crear la base de datos como las tablas, ejecutamos las consultas con el cursor.
`cursor.execute("CREATE DATABASE IF NOT EXISTS Akila;") cursor.execute("USE Akila;") cursor.execute(""" CREATE TABLE IF NOT EXISTS actor_transformado ( actor_id INT PRIMARY KEY, first_name VARCHAR(50), last_name VARCHAR(50), description TEXT ); """)`

Confirmar cambios y cerrar conexión: Después de ejecutar las consultas, confirmamos los cambios y cerramos la conexión.
conn.commit()
cursor.close()
conn.close()

###¿Cómo validar la carga y transformación de los datos con Python?

Una vez creadas las tablas, la carga de datos se realiza usando Pandas y SQLAlchemy. Este proceso recopila los datos, los transforma según necesidades y los guarda en las tablas SQL.

Conectar y cargar datos: Usamos pandas para leer y transformar datos, luego los cargamos a nuestras tablas con `SQLAlchemy.
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(‘mysql+mysqlconnector://tu_usuario:tu_contraseña@tu_host/Akila’)
df = pd.DataFrame({
‘actor_id’: [1, 2],
‘first_name’: [‘John’, ‘Jane’],
‘last_name’: [‘Doe’, ‘Doe’],
‘description’: [‘Actor principal’, ‘Actor secundario’]
})
df.to_sql(‘actor_transformado’, engine, if_exists=‘replace’, index=False)`

Validar carga de datos: Validamos usando queries SQL para asegurar que los datos estén correctamente cargados.
query = "SELECT * FROM actor_transformado;" df_result = pd.read_sql(query, engine) print(df_result)
```

Este proceso cautiva cómo utilizar la programación para gestionar bases de datos de manera eficiente. Además, al terminar el ejercicio, es una excelente práctica verificar manualmente los datos en MySQL para garantizar que todo esté en su lugar. ¿Has logrado ejecutar este proyecto? ¿Qué retos enfrentaste? ¡Cuéntanos en los comentarios para que podamos ayudarte! Sigue adelante, cada paso cuenta en tu proceso de aprendizaje.

**Lecturas recomendadas**

[Curso de Ética y Manejo de Datos para Data Science e Inteligencia Artificial](https://platzi.com/cursos/etica-ia/)