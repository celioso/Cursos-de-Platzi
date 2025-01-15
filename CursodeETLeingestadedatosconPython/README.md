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