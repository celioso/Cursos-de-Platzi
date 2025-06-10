# Curso de Introducción a Machine Learning

## Algoritmos de Machine Learning: Preparación y Aplicación de Datos

¡Perfecto! El tema **"Algoritmos de Machine Learning: Preparación y Aplicación de Datos"** es fundamental para crear modelos efectivos. A continuación te doy una guía resumida y clara sobre lo más importante que debes saber:

### 🧪 1. Preparación de Datos (Preprocesamiento)

Antes de aplicar cualquier algoritmo de Machine Learning, es esencial preparar los datos:

### ✅ Limpieza de Datos

* **Eliminar valores nulos** o imputarlos (`mean`, `median`, `mode`)
* **Eliminar duplicados**
* **Detectar y corregir errores** (como valores fuera de rango)

### ✅ Codificación de Variables Categóricas

* **One-Hot Encoding** (para variables nominales)
* **Label Encoding** (para variables ordinales)

### ✅ Normalización / Escalado

* **StandardScaler**: transforma a media 0 y desviación estándar 1
* **MinMaxScaler**: escala entre 0 y 1

### ✅ Dividir Datos

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🤖 2. Aplicación de Algoritmos de ML

Aquí algunos algoritmos comunes según el tipo de problema:

### 🔢 Regresión (cuando el resultado es numérico)

* **Linear Regression**
* **Ridge/Lasso Regression**
* **Decision Tree Regressor**
* **Random Forest Regressor**

### 📊 Clasificación (cuando el resultado es una categoría)

* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**
* **Random Forest Classifier**
* **Gradient Boosting (XGBoost, LightGBM)**

### 🧬 Clustering (aprendizaje no supervisado)

* **K-Means**
* **DBSCAN**
* **Hierarchical Clustering**

### 📈 3. Evaluación del Modelo

### Para Clasificación:

* **Accuracy**
* **Precision, Recall, F1-score**
* **Matriz de Confusión**

```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Para Regresión:

* **MAE**, **MSE**, **RMSE**
* **R² (coeficiente de determinación)**

### 🛠️ 4. Herramientas y Bibliotecas Usuales

* `pandas`, `numpy` → manipulación de datos
* `scikit-learn` → modelos de ML, métricas, preprocesamiento
* `matplotlib`, `seaborn` → visualización
* `xgboost`, `lightgbm` → modelos avanzados de boosting

### ⚙️ Flujo Básico de un Proyecto de ML

1. Cargar y explorar los datos
2. Preprocesar los datos
3. Dividir en entrenamiento y prueba
4. Aplicar modelos
5. Evaluar resultados
6. Ajustar hiperparámetros si es necesario
7. Guardar modelo (`joblib`, `pickle`) y documentar

### Resumen

#### ¿Qué es el machine learning y por qué es importante?

En un mundo que avanza a pasos agigantados hacia la inteligencia artificial, el machine learning se destaca como una herramienta esencial para convertir datos en conocimiento. Soy Natasha, jefa de investigación en AI/ML en MindsDB, y estoy aquí para guiarte en esta emocionante travesía por el aprendizaje automático. Vamos a explorar los algoritmos que permiten sacar el máximo provecho de tus datos, cómo implementarlos y qué modelos elegir para tus necesidades específicas.

#### ¿Cómo prepararse para aprender machine learning?

Para aprovechar al máximo el aprendizaje de machine learning, es fundamental contar con algunas bases previas que te ayudaran a seguir de manera fluida:

- **Conocimiento de Python**: Dado que muchas de las herramientas de machine learning están escritas en este lenguaje, familiarizarte con Python te brindará una ventaja significativa.
- **Experiencia con pandas**: Este paquete de Python es crucial para manipular y analizar datos. Te ayudará a gestionar y preparar los conjuntos de datos eficientemente.
- **Uso de Matplotlib**: Esta herramienta de trazado te permitirá visualizar los datos, facilitando la comprensión de sus relaciones y características antes de aplicar modelos.
- **Intuición en probabilidad y estadístic**a: Conocer los fundamentos te permitirá entender las decisiones detrás de los modelos y mejorarás tu capacidad para interpretar sus predicciones.

Te recomiendo explorar los cursos ofrecidos en Platzi, donde puedes adquirir o fortalecer estos conocimientos esenciales.

#### ¿Cuáles son los pasos clave para trabajar con machine learning?

La preparación y visualización de datos son pasos previos fundamentales para enfrentar problemas de machine learning con éxito. Este proceso se puede dividir principalmente en tres objetivos:

1. **Preparación de Dato**s:

- Es crucial manejar los datos de forma adecuada, asegurando que estén limpios y estructurados antes de realizar cualquier análisis.
- La visualización de relaciones dentro de los datos facilita la identificación de patrones que podrían ser útiles para entrenar modelos.

2. **Comprender los algoritmos de machine learning**:

- Una vez que los datos están listos, es momento de seleccionar el algoritmo adecuado. Conocer cómo estos algoritmos operan detrás del telón y cómo hacen sus predicciones amplía significativamente la comprensión y efectividad de tus modelos.

3. **Exploración del Deep Learning**:

- Este subcampo del machine learning se centra en redes neuronales complejas, que son particularmente efectivas para abordar problemas complejos debido a su arquitectura inspirada en el cerebro humano.

#### ¿Cómo seguir aprendiendo y aplicando machine learning?

El camino hacia la maestría en machine learning es continuo y siempre está evolucionando, con nuevas tecnologías y técnicas emergiendo regularmente. Aquí hay algunas recomendaciones para seguir creciendo:

- Participa en comunidades de aprendizaje y foros donde puedes compartir conocimientos y resolver dudas junto a otros entusiastas.
- Experimenta con proyectos personales o contribuciones a proyectos de código abierto para ganar experiencia práctica.
- Mantente actualizado con las últimas tendencias y prácticas en machine learning mediante la lectura de artículos, investigación y contenido especializado.

El machine learning ofrece un vasto campo de oportunidades y desafíos. Al mejorar tus habilidades y aplicar tus conocimientos, posiblemente serás un actor clave en la implementación de soluciones inteligentes en tus entornos de trabajo o proyectos personales. ¡Continúa explorando y aprendiendo para liberar todo el potencial de tus datos en el mundo digital!

**Archivos de la clase**

[slides-espanol-curso-introduccion-machine-learning-por-mindsdb.pdf](https://static.platzi.com/media/public/uploads/slides-espanol-curso-introduccion-machine-learning-por-mindsdb_8c5ff985-0581-4977-9ecf-53dd1817fc3f.pdf)

**Lecturas recomendadas**

[Machine Learning in your Database using SQL - MindsDB](https://mindsdb.com/)

[Curso de Jupyter Notebook - Platzi](https://platzi.com/cursos/jupyter-notebook/)

[Curso de Python [Empieza Gratis] - Platzi](https://platzi.com/cursos/python/)

[Curso de Python Intermedio - Platzi](https://platzi.com/cursos/python-intermedio/)

[Curso de Estadística Descriptiva - Platzi](https://platzi.com/cursos/estadistica-descriptiva/)

[Curso de Matemáticas para Data Science: Cálculo Básico - Platzi](https://platzi.com/cursos/calculo-data-science/)

[Curso de Matemáticas para Data Science: Probabilidad - Platzi](https://platzi.com/cursos/ds-probabilidad/)

[Curso de Fundamentos de Álgebra Lineal con Python - Platzi](https://platzi.com/cursos/algebra-lineal/)

[Curso de Visualización de Datos para Business Intelligence - Platzi](https://platzi.com/cursos/visualizacion-datos/)

[Curso de Pandas con Python [Empieza Gratis] - Platzi](https://platzi.com/cursos/pandas/)

[Curso de Álgebra Lineal para Machine Learning - Platzi](https://platzi.com/cursos/algebra-ml/)

[Curso Práctico de Regresión Lineal con Python - Platzi](https://platzi.com/cursos/regresion-lineal/)

## Introducción al Machine Learning: Historia y Conceptos Básicos

¡Excelente elección! Una **introducción al Machine Learning** (ML) debe cubrir tanto la historia como los conceptos clave para entender cómo y por qué se usa esta disciplina. Aquí tienes un resumen claro y didáctico:

### 🧠 ¿Qué es el Machine Learning?

**Machine Learning (aprendizaje automático)** es una rama de la inteligencia artificial (IA) que permite que las computadoras aprendan **a partir de datos** sin ser programadas explícitamente para cada tarea.

> “ML es el campo de estudio que da a las computadoras la habilidad de aprender sin ser explícitamente programadas.” – Arthur Samuel (1959)

### 📜 Breve Historia del Machine Learning

| Año      | Hito                                    | Descripción                                                               |
| -------- | --------------------------------------- | ------------------------------------------------------------------------- |
| **1950** | Test de Turing                          | Alan Turing propone una prueba para medir la inteligencia artificial.     |
| **1959** | Arthur Samuel                           | Crea uno de los primeros programas de ML: un juego de damas que aprende.  |
| **1986** | Backpropagation                         | Se populariza el algoritmo de retropropagación en redes neuronales.       |
| **1997** | IBM Deep Blue                           | Vence al campeón mundial de ajedrez Garry Kasparov.                       |
| **2012** | AlexNet                                 | Red neuronal que gana el concurso ImageNet, revoluciona el Deep Learning. |
| **Hoy**  | IA generativa, autoML, IA en producción | Grandes modelos como GPT, BERT, DALL·E, etc.                              |

### 🧩 Tipos de Aprendizaje en ML

1. **Aprendizaje Supervisado**

   * Entrenamiento con **datos etiquetados**.
   * Ejemplos: regresión lineal, árboles de decisión, SVM, redes neuronales.
   * Problemas típicos: **clasificación** y **regresión**.

2. **Aprendizaje No Supervisado**

   * No hay etiquetas; se busca **estructura** o **patrones** en los datos.
   * Ejemplos: K-means, PCA, clustering jerárquico.

3. **Aprendizaje por Refuerzo**

   * Un agente **aprende por prueba y error** mediante recompensas.
   * Usado en juegos, robótica, optimización.

### 📦 Componentes Clave de un Sistema de ML

* **Datos**: El combustible del modelo.
* **Modelo**: La estructura matemática que aprende.
* **Algoritmo de entrenamiento**: Cómo aprende el modelo (p. ej., regresión, redes neuronales).
* **Métrica de evaluación**: Cómo sabemos si el modelo funciona bien (accuracy, MSE, etc.).

### 🧪 Ejemplos de Aplicaciones

| Campo      | Aplicación                                 |
| ---------- | ------------------------------------------ |
| Salud      | Diagnóstico de enfermedades                |
| Finanzas   | Detección de fraudes                       |
| Marketing  | Recomendaciones personalizadas             |
| Transporte | Rutas inteligentes, coches autónomos       |
| Lenguaje   | Traducción automática, generación de texto |

### 🛠️ Herramientas Populares

* **Python** (lenguaje líder en ML)
* **scikit-learn**, **TensorFlow**, **PyTorch**, **XGBoost**
* **Jupyter Notebooks** para prototipado y visualización

### Resumen

#### ¿Qué es el machine learning?

El **machine learning** es la ciencia que explora el uso de algoritmos para identificar patrones en conjuntos de datos y resolver tareas específicas. Esta disciplina se centra en tomar descriptores o características de los datos—como X-uno y X-dos en los ejemplos mencionados—y descubrir relaciones significativas que nos permitan responder a preguntas críticas. No es sólo un concepto abstracto; tiene aplicaciones prácticas en nuestra vida diaria. Un ejemplo palpable es el filtro de spam en tu correo electrónico, donde sofisticados algoritmos determinan cuáles mensajes evitar.

#### ¿Cómo se aplica el machine learning en nuestro día a día?

La utilización de algoritmos de machine learning no es limitada a contextos académicos o de investigación; está profundamente integrada en la tecnología que usamos cotidianamente:

- **Filtros de correo spam**: Empresas han invertido miles de millones para mejorar la detección de spam, alcanzando niveles de precisión impresionantes. En 2015, por ejemplo, Google logró que un algoritmo identificara el spam con un 99.9% de efectividad.

- **Asistentes personales y dispositivos inteligentes**: Desde asistentes en nuestros teléfonos hasta robots de limpieza como Roombas, estas tecnologías emplean machine learning para mejorar su desempeño y adaptarse mejor a nuestras necesidades.

- **Juegos de estrategia**: El algoritmo tras AlphaGo, que superó a jugadores humanos en complejos juegos de mesa, muestra la potencia de machine learning en la toma de decisiones estratégicas.

#### ¿Cuál es la historia del machine learning?

Aunque muchas veces se percibe el machine learning como un fenómeno reciente, sus raíces datan de los años 50. Desde entonces, ha evolucionado significativamente, impulsado por avances en recursos computacionales. Esta evolución ha permitido su aplicación en una amplia gama de áreas, desde programación para juegos sencillos hasta tecnologías avanzadas que impactan nuestro entorno cotidiano.

#### ¿Por qué el machine learning es importante en el ámbito tecnológico actual?

El machine learning se ha convertido en un pilar dentro del mundo tecnológico por varias razones:

- **Crecimiento y relevancia**: Es un campo en rápido crecimiento, reflejado en el aumento de inversión en startups dedicadas al machine learning y su presencia en las habilidades más demandadas dentro del sector tecnológico.

- **Nuevas oportunidades**: Cada vez son más las oportunidades para que nuevos colaboradores contribuyan al crecimiento del campo. La encuesta de 2020 destacó a Python, machine learning y deep learning como algunas de las habilidades tecnológicas más buscadas.

En conclusión, no solo es relevante aprender sobre machine learning por sus diversas aplicaciones prácticas, sino también porque ofrece una plataforma para la innovación continua en tecnología. Si estás considerando involucrarte en este fascinante campo, ¡ahora es el momento perfecto para hacerlo! En las próximas etapas, exploraremos más sobre herramientas esenciales en ciencia de datos para fortalecer tu comprensión y aplicación del machine learning.

## Introducción a la Ciencia de Datos: Carga y Visualización de Conjuntos

¡Perfecto! Comenzar con la **ciencia de datos** implica familiarizarse con **la carga, exploración y visualización de conjuntos de datos**. Aquí tienes una introducción práctica y clara para iniciarte:

### 🧠 ¿Qué es la Ciencia de Datos?

La **ciencia de datos** combina estadística, programación y conocimiento del dominio para extraer valor a partir de datos. Un paso clave es **cargar los datos correctamente y visualizarlos para comprenderlos**.

### 📥 1. Carga de Conjuntos de Datos

Usamos **pandas**, una biblioteca de Python, para trabajar con datos en forma de tablas (DataFrames).

### 📌 Ejemplo básico de carga:

```python
import pandas as pd

# Cargar un archivo CSV
df = pd.read_csv("datos.csv")

# Ver las primeras filas
print(df.head())
```

### 📌 También puedes cargar desde:

* Excel: `pd.read_excel("archivo.xlsx")`
* JSON: `pd.read_json("archivo.json")`
* URLs: `pd.read_csv("https://archivo.csv")`

### 🔎 2. Exploración Rápida del Dataset

```python
# Ver la forma del dataset (filas, columnas)
print(df.shape)

# Nombres de columnas
print(df.columns)

# Tipos de datos
print(df.dtypes)

# Resumen estadístico
print(df.describe())

# Valores nulos
print(df.isnull().sum())
```

### 📊 3. Visualización de Datos

Usamos bibliotecas como **Matplotlib** y **Seaborn** para representar los datos gráficamente.

### 📌 Gráficos básicos:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma
df["edad"].hist()
plt.title("Distribución de Edad")
plt.show()

# Gráfico de dispersión
sns.scatterplot(x="ingresos", y="gastos", data=df)
plt.title("Ingresos vs Gastos")
plt.show()

# Boxplot por categoría
sns.boxplot(x="genero", y="ingresos", data=df)
plt.title("Distribución de Ingresos por Género")
plt.show()
```

### 📁 Datasets Recomendados para Práctica

Puedes comenzar con datasets famosos como:

* `titanic` (sobrevivientes del Titanic)
* `iris` (características de flores)
* `tips` (propinas en restaurantes)

Se pueden cargar directamente desde Seaborn:

```python
df = sns.load_dataset("titanic")
```

### 🔄 Flujo Típico de un Proyecto

1. **Cargar** los datos
2. **Explorar** la estructura y los valores
3. **Limpiar** datos nulos o inconsistentes
4. **Visualizar** para encontrar patrones
5. **Modelar** si aplica (Machine Learning)

### Recursos

#### ¿Cuál es la importancia de comprender tus datos antes de entrenar un modelo?

Para entrenar modelos de machine learning exitosos, es crítico inspeccionar y comprender los datos de manera exhaustiva. Los modelos solo serán tan efectivos como la calidad de los datos que los alimentan. La ciencia de datos nos proporciona las herramientas necesarias para profundizar en los datos, comprender sus características y resolver problemas antes de avanzar al modelado. Esta exploración inicial incluye la identificación de features, filas, columnas y la detección de valores atípicos.

#### ¿Qué terminología clave deberías conocer?

- **Datos**: Son unidades de información obtenidas de diferentes observaciones, desde encuestas simples hasta complejas bases de datos financieras.
- **Features**: Son descriptores de las cualidades o propiedades de los datos, como altura, género o niveles de glucosa.
- **Filas y columnas**: Las filas representan instancias individuales dentro del conjunto de datos, mientras que las columnas describen las características o features de cada instancia.
- **Valores atípicos**: Pueden ser desviaciones estadísticas o valores incorrectos, y su inclusión o exclusión debe ser evaluada cuidadosamente.
- **Preprocesamiento**: Consiste en preparar los datos para maximizar el aprovechamiento por los modelos, mediante la eliminación, imputación de valores perdidos o la escalación de datos.

#### ¿Qué tipos de datos se suelen manejar?

La clasificación adecuada de los datos es fundamental para la preparación de los mismos. Los tipos de datos comunes incluyen:

- **Datos numéricos**: Estos pueden ser valores discretos o continuos, como la cantidad de monedas o la temperatura.
- **Datos categóricos**: Son etiquetados e incluyen variables como formas de objetos o tipos de clima. Estos deben ser convertidos a formatos numéricos para el modelado, usando técnicas como el "one hot encoding".

Datos más complejos, como imágenes y texto, requieren preprocesamiento avanzado y uso de técnicas de machine learning especializadas, aunque estos no se abordan en este contexto.

#### ¿Cómo se precarga y visualiza un conjunto de datos?

A la hora de trabajar con conjuntos de datos, herramientas como Pandas en Python ofrecen funcionalidad poderosa para cargar y explorar datos. Se utilizan formatos como CSV para organizar y acceder a la información. Aquí algunos comandos útiles:

- `read_CSV`: Se utiliza para cargar un conjunto de datos desde un archivo CSV.
- `head`: Permite inspeccionar las primeras filas del dataset para asegurar que se haya cargado correctamente.
- `dtypes`: Infiere los tipos de datos de cada columna, asistiendo en su correcta categorización.

#### ¿Cómo se visualizan las relaciones y distribuciones?

Una vez cargados los datos, la visualización es clave para entender relaciones entre features y detectar posibles anomalías. Dos técnicas populares son:

- **Histogramas**: Ayudan a visualizar la distribución de un feature específico, como la cantidad de monedas que una persona podría tener en su bolsillo. Los datos se agrupan en "bins" representando frecuencias dentro de un rango determinado.

- **Gráficos de dispersión**: Son útiles para explorar relaciones entre dos features, como la correlación entre la presión arterial y la edad. Estos gráficos revelan tendencias y posibles errores en los datos, como valores atípicos.

En conclusión, asegurar una comprensión sólida de nuestros datos iniciales y realizar una exploración exhaustiva mediante preprocesamiento y visualización es esencial antes de sumergirse en el entrenamiento de modelos de machine learning. Esto optimiza la fiabilidad y precisión de las predicciones del modelo.

## Algoritmos Supervisados y No Supervisados en Machine Learning

¡Genial! Entender la diferencia entre **algoritmos supervisados y no supervisados** en Machine Learning (ML) es fundamental para aplicar la técnica adecuada según el tipo de datos y problema que tengas.

### 🧠 ¿Qué son los Algoritmos Supervisados y No Supervisados?

### 📌 **Aprendizaje Supervisado**

El modelo aprende a partir de **datos etiquetados**, es decir, el conjunto de entrenamiento incluye tanto los **inputs** (X) como las **respuestas esperadas** (y).

🔍 **Objetivo**: Predecir una salida basada en ejemplos conocidos.

#### Ejemplos de algoritmos:

| Tipo          | Algoritmo                                                 | Uso común                                        |
| ------------- | --------------------------------------------------------- | ------------------------------------------------ |
| Clasificación | `Logistic Regression`, `Random Forest`, `SVM`, `KNN`      | Predecir categorías (spam/no spam, diagnóstico)  |
| Regresión     | `Linear Regression`, `Decision Tree Regressor`, `XGBoost` | Predecir valores numéricos (precio, temperatura) |

### Ejemplo en código:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)  # Entrena con datos etiquetados
```

### 📌 **Aprendizaje No Supervisado**

El modelo **no tiene etiquetas**. Se utiliza para descubrir **patrones ocultos**, **grupos** o **estructura** en los datos.

🔍 **Objetivo**: Entender la distribución o agrupar datos sin respuestas previas.

#### Ejemplos de algoritmos:

| Algoritmo                                   | Uso común                     |
| ------------------------------------------- | ----------------------------- |
| `K-Means`                                   | Agrupar clientes en segmentos |
| `DBSCAN`                                    | Detección de anomalías        |
| `PCA` (Análisis de Componentes Principales) | Reducción de dimensionalidad  |

### Ejemplo en código:

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)  # Solo necesita los datos, no etiquetas
```

### ⚖️ Comparación Rápida

| Característica        | Supervisado                     | No Supervisado                          |
| --------------------- | ------------------------------- | --------------------------------------- |
| Requiere etiquetas    | ✅ Sí                            | ❌ No                                    |
| Tipos de problemas    | Clasificación y regresión       | Clustering, reducción de dimensiones    |
| Ejemplo típico        | Predecir si un cliente comprará | Segmentar clientes según comportamiento |
| Ejemplo de algoritmos | SVM, Random Forest, XGBoost     | K-Means, PCA, DBSCAN                    |

### 🧪 ¿Cuál elegir?

* Usa **supervisado** cuando tienes datos etiquetados y quieres **predecir**.
* Usa **no supervisado** cuando tienes solo características y quieres **explorar** o **agrupar**.

## Algoritmos Supervisados y No Supervisados en Machine Learning

¡Genial! Entender la diferencia entre **algoritmos supervisados y no supervisados** en Machine Learning (ML) es fundamental para aplicar la técnica adecuada según el tipo de datos y problema que tengas.

### 🧠 ¿Qué son los Algoritmos Supervisados y No Supervisados?

### 📌 **Aprendizaje Supervisado**

El modelo aprende a partir de **datos etiquetados**, es decir, el conjunto de entrenamiento incluye tanto los **inputs** (X) como las **respuestas esperadas** (y).

🔍 **Objetivo**: Predecir una salida basada en ejemplos conocidos.

#### Ejemplos de algoritmos:

| Tipo          | Algoritmo                                                 | Uso común                                        |
| ------------- | --------------------------------------------------------- | ------------------------------------------------ |
| Clasificación | `Logistic Regression`, `Random Forest`, `SVM`, `KNN`      | Predecir categorías (spam/no spam, diagnóstico)  |
| Regresión     | `Linear Regression`, `Decision Tree Regressor`, `XGBoost` | Predecir valores numéricos (precio, temperatura) |

### Ejemplo en código:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)  # Entrena con datos etiquetados
```

### 📌 **Aprendizaje No Supervisado**

El modelo **no tiene etiquetas**. Se utiliza para descubrir **patrones ocultos**, **grupos** o **estructura** en los datos.

🔍 **Objetivo**: Entender la distribución o agrupar datos sin respuestas previas.

#### Ejemplos de algoritmos:

| Algoritmo                                   | Uso común                     |
| ------------------------------------------- | ----------------------------- |
| `K-Means`                                   | Agrupar clientes en segmentos |
| `DBSCAN`                                    | Detección de anomalías        |
| `PCA` (Análisis de Componentes Principales) | Reducción de dimensionalidad  |

### Ejemplo en código:

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)  # Solo necesita los datos, no etiquetas
```

### ⚖️ Comparación Rápida

| Característica        | Supervisado                     | No Supervisado                          |
| --------------------- | ------------------------------- | --------------------------------------- |
| Requiere etiquetas    | ✅ Sí                            | ❌ No                                    |
| Tipos de problemas    | Clasificación y regresión       | Clustering, reducción de dimensiones    |
| Ejemplo típico        | Predecir si un cliente comprará | Segmentar clientes según comportamiento |
| Ejemplo de algoritmos | SVM, Random Forest, XGBoost     | K-Means, PCA, DBSCAN                    |

### 🧪 ¿Cuál elegir?

* Usa **supervisado** cuando tienes datos etiquetados y quieres **predecir**.
* Usa **no supervisado** cuando tienes solo características y quieres **explorar** o **agrupar**.

### Resumen

#### ¿Qué tipos de algoritmos y modelos de machine learning existen?

En el fascinante mundo del machine learning, los algoritmos y modelos juegan un papel crucial al abordar problemas complejos y ayudar a obtener insights valiosos de los datos. Existen diferentes tipos de enfoques y algoritmos, cada uno diseñado para resolver tipos específicos de problemas. En esta guía, exploraremos las características distintivas de los enfoques supervisados y no supervisados, dos formas predominantes en este ámbito.

#### ¿Qué es el aprendizaje supervisado?

El aprendizaje supervisado se centra en usar características de entrada para predecir una variable de salida objetivo. Este enfoque es útil cuando queremos que un modelo aprenda de datos etiquetados para hacer predicciones precisas. El aprendizaje supervisado se divide principalmente en dos categorías:

1. **Regresión**:

- **Objetivo**: Predecir un valor numérico continuo.
- **Ejemplo**: Estimar la temperatura exterior basándose en diversas features como la hora del día, la ubicación y la humedad.
- **Técnicas comunes**: Regresión lineal, que analiza las relaciones entre las variables dependientes y una o más variables independientes.

2. **Clasificación**:

- **Objetivo**: Predecir una etiqueta o categoría.
- **Ejemplo**: Determinar la retención de un cliente o la validez de una transacción.
- **Técnicas comunes**: Regresión logística y bosque aleatorio, que son poderosas herramientas para investigar conjuntos de datos complejos.

#### ¿Qué es el aprendizaje no supervisado?

El aprendizaje no supervisado se aplica cuando no se tiene una variable objetivo clara y se busca descubrir patrones o estructuras inherentes en los datos. Este enfoque es fundamental para identificar agrupamientos o reducir la dimensionalidad de los datos.

1. **Agrupación**:

- **Objetivo**: Encontrar grupos naturales en los datos.
- **Ejemplo**: Segmentación de clientes en marketing basado en comportamientos de navegación o productos vistos.
- **Técnicas comunes**: K-means y agrupación jerárquica, que ayudan a identificar relaciones latentes en los datos.

2. **Reducción de dimensionalidad**:

- **Objetivo**: Simplificar los datos mientras se mantienen las características más informativas.
- **Ejemplo**: Transformar grandes conjuntos de datos en representaciones más manejables sin perder información crucial.
- **Técnicas comunes**: Análisis de componentes principales (PCA) y T-SNE, que son esenciales para tratar con big data.

#### ¿Qué algoritmos específicos son populares en machine learning?

Para enfrentar los diversos desafíos en machine learning, varios algoritmos han ganado popularidad debido a su eficacia y robustez. A continuación, se describen algunos de los más utilizados:

- **Aprendizaje supervisado**:

 - **Regresión lineal**: Usado para predecir valores continuos y explorar relaciones entre variables.
 - **Regresión logística y bosque aleatorio**: Aptos para problemas de clasificación donde el objetivo es etiquetar observaciones.

- **Aprendizaje no supervisado**:

 - **K-means**: Ideal para identificar clusters en conjuntos de datos sin etiquetar.
 - **Análisis de componentes principales (PCA) y T-SNE**: Útiles en la reducción de dimensionalidad, permitiendo visualizar datos complejos en espacios más reducidos.

El dominio de estos conceptos fundamentales y la comprensión de cuándo y cómo aplicar estos algoritmos es crucial para cualquier persona que busque aventurarse en el mundo del machine learning. ¡Sigue explorando y practicando para desentrañar todo el potencial que estos métodos ofrecen!

## Procesamiento y Análisis de Datos para Machine Learning

¡Hola! Te doy la bienvenida a esta clase donde comenzaremos a poner a prueba lo que has aprendido en los cursos previos de ciencia de datos e inteligencia artificial de Platzi y en este.

Recuerda que para avanzar con esta clase deberás haber tomado los siguientes cursos:

- [Curso de Entorno de Trabajo para Ciencia de Datos con Jupyter Notebooks y Anaconda](https://platzi.com/cursos/jupyter-notebook/)
- [Curso Básico de Python](https://platzi.com/cursos/python/)
- [Curso de Python: Comprehensions, Lambdas y Manejo de Errores](https://platzi.com/cursos/python-intermedio/)
- [Curso de Matemáticas para Data Science: Estadística Descriptiva](https://platzi.com/cursos/estadistica-descriptiva/)
- [Curso Práctico de Regresión Lineal con Python](https://platzi.com/cursos/regresion-python/)
- [Curso de Matemáticas para Data Science: Cálculo Básico](https://platzi.com/cursos/calculo-diferencial-ds/)
- [Curso de Matemáticas para Data Science: Probabilidad](https://platzi.com/cursos/ds-probabilidad/)
- [Curso de Fundamentos de Álgebra Lineal con Python](https://platzi.com/cursos/algebra-lineal/)
- [Curso de Principios de Visualización de Datos para Business Intelligence](https://platzi.com/cursos/visualizacion-datos/)
- [Curso de Manipulación y Análisis de Datos con Pandas y Python](https://platzi.com/cursos/pandas/)
- [Curso de Álgebra Lineal Aplicada para Machine Learning](https://platzi.com/cursos/algebra-ml/)

Te reitero que es muy importante que conozcas estos temas y ya tengas las habilidades para que puedas aprender con facilidad y seguir con el curso hasta el final. Aprender machine learning en un principio no es una tarea sencilla, pero con la preparación adecuada y dedicación podemos obtener este conocimiento de forma trascendental.

Let’s go for it! 💪

#### Nuestra notebook de ejercicios

Para esta clase tendrás una [notebook en Google Colab](https://colab.research.google.com/drive/1u9ps-c_u0SbMh07pA5pKYOIeLbACSAx1?usp=sharing "notebook en Google Colab") donde encontrarás piezas de código con explicaciones sobre **el paso a paso para procesar y analizar un dataset** antes de comenzar a aplicar algoritmos de machine learning.

[Accede al notebook aquí.](https://colab.research.google.com/drive/1u9ps-c_u0SbMh07pA5pKYOIeLbACSAx1?usp=sharing "Accede al notebook aquí.")

Crea una copia de este notebook en tu Google Drive o utilizalo en el entorno de Jupyter notebook que prefieras.

En el notebook también encontrarás ejercicios que deberás **resolver por tu cuenta**. Sigue las instrucciones dentro del notebook y comparte tus resultados en los comentarios de esta clase.

En dado caso de que tengas alguna duda o no puedas completar alguno de los ejercicios, al final del notebook encontrarás una sección con las **respuestas**, pero antes de revisarlas da el máximo esfuerzo para realizar los ejercicios. Así aprenderás mucho más.

De igual forma te invito a que dejes en comentarios cualquier duda, dificultad o pregunta que tengas al momento de seguir el notebook y realizar los ejercicios. Con mucho gusto la comunidad de Platzi te ayudará.

¡Te deseo mucho éxito y nos vemos en el próximo módulo! Comenzaremos a detallar los diferentes modelos que existen de machine learning. 🧠

## Modelos de Machine Learning: Uso, Implementación y Evaluación

¡Excelente! Aquí tienes una guía clara sobre los **modelos de Machine Learning**, su **uso**, **implementación** y **evaluación**, pensada especialmente para quienes trabajan en ciencia de datos y proyectos prácticos.

### 🤖 ¿Qué es un Modelo de Machine Learning?

Un **modelo de Machine Learning (ML)** es una función que aprende de los datos para hacer predicciones o clasificaciones sin estar explícitamente programado para cada tarea.

### 1️⃣ **Uso de los Modelos**

Se utilizan para resolver diferentes tipos de problemas:

| Tipo de problema       | Ejemplos comunes                       | Tipo de ML            |
| ---------------------- | -------------------------------------- | --------------------- |
| Clasificación          | Diagnóstico médico, spam, fraude       | Supervisado           |
| Regresión              | Predicción de precios, clima           | Supervisado           |
| Clustering             | Segmentación de clientes               | No supervisado        |
| Reducción de dimensión | Visualización, compresión              | No supervisado        |
| Series temporales      | Predicción de ventas, bolsa de valores | Supervisado o híbrido |

### 2️⃣ **Implementación en Python (con scikit-learn)**

### 🔸 Ejemplo: Clasificación con `RandomForestClassifier`

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Cargar dataset
X, y = load_iris(return_X_y=True)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Crear y entrenar modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)
```

### 3️⃣ **Evaluación de Modelos**

Es crucial medir el desempeño de un modelo. Las métricas varían según el tipo de problema.

### 🔍 Para clasificación:

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 🔍 Para regresión:

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}, R2: {r2}")
```

### 🔍 Para clustering (no supervisado):

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, model.labels_)
print(f"Silhouette Score: {score}")
```

### 🛠️ Buenas Prácticas

* **Normaliza** o **escalas** los datos cuando uses modelos sensibles (como SVM o KNN).
* Usa **cross-validation** para evitar sobreajuste.
* Guarda tus modelos con `joblib` o `pickle` para usarlos en producción.
* Aplica **grid search** o **random search** para optimizar hiperparámetros.

### 📌 Herramientas populares

* **scikit-learn**: modelos clásicos de ML
* **XGBoost / LightGBM**: modelos potentes para tabulares
* **TensorFlow / PyTorch**: modelos de deep learning

### Resumen

#### ¿Qué ingredientes componen un modelo de machine learning?
Al adentrarnos en el fascinante mundo del Machine Learning es crucial comprender los elementos que hacen posible que los algoritmos realicen predicciones precisas. Estos modelos no son magia; funcionan gracias a tres ingredientes fundamentales que determinan su éxito.

1. **Proceso de decisión**: En el corazón de cada modelo de Machine Learning se encuentran los parámetros, que son como las agujas en las brújulas de un explorador. Estos parámetros, ajustables y tratables, ayudan al modelo a generar predicciones al guiarlo a través de un vasto paisaje de datos.

2. **Función de error**: Se conoce también como función de pérdida o costo. Esta función actúa como un crítico constructivo, señalando qué tan lejos está el modelo de alcanzar el objetivo. Ayuda a los desarrolladores a entender las decisiones que toma el modelo y a ajustar el rumbo para mejorar la precisión.

3. **Regla de actualización**: Aquí es donde reside la verdadera magia del aprendizaje. Una vez que el modelo realiza una predicción, la regla de actualización interviene para mejorar los parámetros, asegurando que con cada iteración, las predicciones sean más precisas. Este proceso de retroalimentación es esencial para refinar el modelo y alcanzar un rendimiento óptimo.

#### ¿Cómo preparar los datos para los modelos de ML?

Antes de alimentar los datos a un modelo, es esencial asegurarse de que estén preparados adecuadamente. Dos pasos críticos garantizan que los datos sean utilizados de manera eficiente y efectiva.

#### ¿Por qué es importante la normalización de datos?

La normalización es una práctica común en Machine Learning, especialmente cuando se trabaja con optimización. El objetivo es asegurar que los modelos no enfrenten problemas de estabilidad numérica. La normalización implica transformar los datos numéricos para que tengan una media de cero y una desviación estándar de uno. De esta forma, aunque no se altera la información contenida en la columna, se ajusta la escala, permitiendo que el modelo procese los valores dentro de un contexto uniforme.

#### ¿Cómo dividir los datos para evaluar los modelos?

Dividir el conjunto de datos es fundamental para evaluar los modelos de manera efectiva. Se emplean comúnmente tres divisiones:

- **Entrenamiento**: Constituye generalmente el 80% del conjunto de datos. Estos datos son el entrenamiento ideal para enseñar al modelo a reconocer patrones.

- **Validación**: Se utiliza para evaluar la precisión del modelo y ajustar sus parámetros.

- **Prueba**: Este conjunto se mantiene apartado, fuera del alcance del modelo, y representa del 0 al 20% de los datos. Sirve para realizar una evaluación final objetiva y determinar si el modelo funciona tal como se espera.

Cada uno de estos conjuntos tiene un propósito específico y adaptarlos según el problema en cuestión puede ser crucial para obtener resultados óptimos.

#### ¿Qué tipo de algoritmos supervisados existen?

Dentro del vasto ecosistema del machine learning, los algoritmos supervisados juegan un rol preponderante. Estos algoritmos son entrenados con datos etiquetados, facilitando la predicción de resultados precisos. Vamos a destacar tres modelos significativos que ilustran su utilidad.

#### ¿Cómo funciona la regresión lineal?

La regresión lineal es una herramienta fundamental en el análisis predictivo. Su objetivo es modelar la relación entre una variable dependiente y una o más variables independientes, permitiendo hacer predicciones continuas. Este modelo es ampliamente reconocido por su simplicidad y eficacia en numerosos ámbitos.

#### ¿Qué logra la regresión logística?

A pesar de su nombre, la regresión logística se centra en la clasificación, no en la regresión. Utiliza una función logística para modelar la probabilidad de un conjunto de clases. Ideal para problemas de clasificación binaria, la regresión logística descifra patrones complejos para categorizar datos de manera precisa.

#### ¿En qué consiste el bosque aleatorio?

El bosque aleatorio es una técnica de aprendizaje de conjunto que combina muchos árboles de decisión para realizar predicciones más precisas y robustas. Esta metodología es especialmente útil en tareas de clasificación, ofreciendo una defensa sólida contra el sobreajuste y mejorando la capacidad del modelo para generalizar.

#### ¿Qué diferencia hay en los enfoques no supervisados?

A diferencia del aprendizaje supervisado, los algoritmos no supervisados no dependen de datos etiquetados. Estos modelos exploran patrones ocultos sin supervisión previa. Uno de los métodos más destacados en este ámbito es "K-means".

#### ¿Qué es "K-means"?

"K-means" es un algoritmo de agrupamiento que organiza los datos en "K" grupos según las características internas. Es eficiente para identificar estructuras dentro de grandes conjuntos de datos, ayudando a descubrir patrones o segmentaciones valiosas sin un propósito guiado. Esta técnica es esencial para tareas como la segmentación de mercado o la agrupación de documentos.

La educación es una puerta hacia el futuro. Las herramientas del machine learning, y la comprensión de sus fundamentos, son esenciales para enfrentar los retos del mañana. ¡Sigue explorando y aprendiendo!

## Regresión Lineal: Predicción y Evaluación de Modelos Numéricos

¡Perfecto! La **regresión lineal** es uno de los algoritmos más utilizados y sencillos en Machine Learning para predecir valores **numéricos continuos**. A continuación, te explico cómo se usa, cómo implementarla en Python y cómo evaluarla correctamente.

### 📘 ¿Qué es la Regresión Lineal?

Es un modelo supervisado que busca una relación **lineal** entre una o más variables independientes (X) y una variable dependiente (y).
La fórmula básica es:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon
$$

* **y**: valor a predecir
* **x**: variables independientes
* **β**: coeficientes del modelo
* **ε**: error

### 🧠 ¿Cuándo usar regresión lineal?

Usa este modelo cuando:

* Quieres predecir un **número real** (precio, edad, ingreso, etc.).
* Hay **relación lineal** entre variables.
* Necesitas **interpretabilidad** (coeficientes claros).

### 🛠️ Implementación en Python

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generar datos sintéticos
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear y entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)
```

### 📊 Evaluación del Modelo

Usamos varias métricas para medir qué tan buenas son las predicciones:

### 1. **Error Cuadrático Medio (MSE)**

Mide cuánto se desvían las predicciones del valor real.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")
```

### 2. **Coeficiente de Determinación (R²)**

Indica qué porcentaje de la varianza es explicada por el modelo (1.0 es perfecto).

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.2f}")
```

### 3. **Visualización del ajuste**

```python
plt.scatter(X_test, y_test, color='blue', label='Real')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción')
plt.legend()
plt.title('Regresión lineal')
plt.show()
```

### 📌 Buenas prácticas

✅ Verifica la **linealidad** con gráficos de dispersión.
✅ Escala los datos si los rangos varían mucho (con `StandardScaler`).
✅ Usa **regresión regularizada** (Ridge/Lasso) si hay sobreajuste o muchas variables.

### Resumen

####¿Qué es la regresión lineal? 

La regresión lineal nos permite predecir un número basándonos en las características de un conjunto de datos. Imagina que dibujas una línea que conecta las características con el objetivo de salida. Este modelo puede identificar relaciones positivas: cuando aumenta el valor de X, también lo hace Y. O relaciones negativas: a medida que X crece, Y disminuye. Sin embargo, la regresión lineal puede no ser adecuada para datos complejos. A continuación, exploraremos en profundidad los conceptos clave de este enfoque.

#### ¿Cuáles son los elementos del proceso de decisión en regresión lineal?

El proceso de decisión en regresión lineal se centra en los parámetros: pesos y sesgos. Estos parámetros ayudan a determinar cómo cada característica de entrada influye en el objetivo de salida. Puedes imaginar los pesos como una hipótesis que mide la relación entre la característica de entrada y el objetivo de salida "Y".

- Pesos (W-one): Indican la relación entre la característica de entrada y el objetivo de salida. Comprender el peso te permite anticipar cómo un cambio en X afecta a Y.
- Sesgo (W-cero): Esto nos dice qué esperar en el objetivo final cuando la característica de entrada no existe (X-cero). Esencialmente, es el valor predicho cuando todas las características de entrada son cero.

#### ¿Cómo funciona la función de coste?

La función de coste mide qué tan bien predice el modelo la salida correcta. Comparando los resultados predichos con los reales del conjunto de entrenamiento, tratamos de minimizar la diferencia entre ambos. En otras palabras, buscamos acortar esas líneas verticales entre los puntos de datos reales y nuestra línea de predicción.

#### ¿Cómo se implementa la regla de actualización?

La regla de actualización ajusta los valores de los pesos y sesgos para minimizar dicha diferencia. Utiliza técnicas de optimización numérica para encontrar la línea que mejor se ajuste a los datos. De esta forma, se optimiza el modelo para predecir con mayor exactitud.

#### ¿Cuándo es efectiva una regresión lineal?

La eficiencia de un modelo de regresión lineal se evalúa usando métricas como el error cuadrático medio o "R cuadrado". Estas métricas indican el grado de correlación entre las variables:

- **Error cuadrático medio**: Mide la diferencia promedio entre los valores predichos y los reales.
- **R cuadrado**: Evaluado entre 0 y 1, indica la correlación existente. Un valor cercano a 1 sugiere una fuerte correlación, mientras que un valor cerca de 0 indica lo contrario.

#### ¿Cómo optimizar un modelo de regresión lineal?

Para optimizar un modelo de regresión lineal, se deben seguir tres pasos:

1. **Definir parámetros**: Ajustar los pesos y sesgos para analizar la influencia de cada característica de entrada en la salida.
2. **Minimizar la función de coste**: Reducir el error para mejorar la precisión del modelo.
3. **Aplicar la regla de actualización**: Ajustar los parámetros utilizando métodos de optimización numérica para mejorar la predicción.

#### ¿Es la regresión lineal adecuada para todas las situaciones?

No siempre. Si bien es efectiva para datasets sencillos y con relaciones lineales claras, se puede quedar corta con datos más complejos. En la próxima lección exploraremos la regresión logística, una técnica que ayuda a clasificar y etiquetar datos, ofreciendo así una perspectiva diferente para enfrentar otros tipos de problemas de predicción.

¡Continúa aprendiendo y mejorando tus habilidades en machine learning! Con cada lección dominas nuevas herramientas para abordar mejor tus desafíos analíticos.

## Regresión Logística: Clasificación y Predicción de Probabilidades

¡Claro! La **Regresión Logística** es una técnica fundamental de **clasificación supervisada** en Machine Learning. Aunque su nombre contiene “regresión”, su objetivo principal no es predecir valores continuos, sino **clasificar** observaciones y estimar **probabilidades**.

### 📘 ¿Qué es la Regresión Logística?

La regresión logística estima la **probabilidad** de que una observación pertenezca a una clase específica.
La salida del modelo está en el rango $[0, 1]$, gracias a la función **sigmoide (logística)**:

$$
P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \dots + \beta_nx_n)}}
$$

### 🔍 ¿Cuándo usarla?

Cuando tu variable objetivo (**y**) es **categórica binaria** (0 o 1), como:

* Email spam o no spam
* Cliente comprará o no comprará
* Diagnóstico positivo o negativo

### 🛠️ Implementación en Python

### ✅ Ejemplo básico con scikit-learn

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Cargar dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)
```

### 📊 Evaluación del Modelo

### 1. **Reporte de clasificación**

Precisión, recall, F1-score.

```python
print(classification_report(y_test, y_pred))
```

### 2. **Matriz de confusión**

Para ver verdaderos positivos/negativos y errores.

```python
print(confusion_matrix(y_test, y_pred))
```

### 3. **Probabilidades**

```python
# Ver probabilidades de pertenecer a la clase 1
y_proba = model.predict_proba(X_test)[:, 1]
```

### 📈 Visualización de la función sigmoide

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)
plt.title('Función Sigmoide')
plt.xlabel('z')
plt.ylabel('P(y=1)')
plt.grid(True)
plt.show()
```

### ✅ Ventajas

* Rápido y eficiente con datasets linealmente separables
* Interpretable: puedes ver los coeficientes de impacto
* Permite calibrar probabilidades reales

### ⚠️ Limitaciones

* No funciona bien con relaciones no lineales (usa SVM o árboles en ese caso).
* Supone independencia entre predictores (puede violarse en práctica).
* Sensible a valores atípicos y multicolinealidad.

### 🧠 Variantes avanzadas

* **Regresión logística multinomial**: para clasificación con más de dos clases (`multi_class='multinomial'`)
* **Regularización (L1/L2)**: para evitar sobreajuste (`penalty='l1'` o `'l2'`)

### Resumen

#### ¿Qué es la regresión logística y cómo funciona?

La regresión logística es una técnica poderosa utilizada en problemas de clasificación. Aunque su nombre sugiere una similitud con la regresión lineal, su propósito principal es dividir o clasificar datos en diferentes categorías. En la misma línea, se ajusta una función que busca separar dos clases distintas dentro de un conjunto de datos. Esta metodología es fundamental cuando se trata de predecir la probabilidad de un evento binario, como aprobar o no un examen.

#### ¿Cómo se aplica la regresión logística en un ejemplo de educación?

Imagina que eres un profesor que busca recomendar cuántas horas deben estudiar los estudiantes para aprobar un examen. Para esto, podrías realizar una encuesta que pregunte a cada estudiante cuántas horas estudiaron y si aprobaron o no. Aquí, el objetivo de la regresión logística es encontrar una fórmula que permita predecir la probabilidad de que un estudiante pase. Si el resultado de la fórmula es 0,5 o más, consideraríamos que el estudiante probablemente aprobará. Esta técnica es muy eficiente para optimizar predicciones en situaciones similares.

#### ¿Cómo funciona la función de coste en la regresión logística?

La función de coste es crucial para evaluar si la predicción es precisa en términos de probabilidades de aprobar o reprobar. Se trata de una función que mide la diferencia entre las predicciones del modelo y los resultados reales, buscando minimizar el error. Este concepto se puede aplicar a diferentes tipos de problemas, no solo binarios, mediante el ajuste de parámetros que mejoren la separación entre clases.

#### ¿Cómo se mide la precisión de los modelos de regresión logística?

La precisión de un modelo de regresión logística se puede evaluar mediante una matriz de confusión. Esta herramienta evalúa si las predicciones del modelo reflejan la realidad al categorizar correctamente los resultados. Especialmente útil cuando hay un desequilibrio en los datos (más aprobados que reprobados, por ejemplo), ayuda a comprender cómo el modelo está fallando en sus predicciones. Si el conjunto de datos está equilibrado, medir la precisión, es decir, la proporción de predicciones correctas, es una técnica común para evaluar el rendimiento.

#### ¿Cuáles son los pasos clave del proceso de regresión logística?

1. **Proceso de decisión**: Busca predecir la línea que mejor divide las clases, estimando la probabilidad de pertenencia a una clase en particular.

2. **Función de coste**: Evaluar un conjunto de pesos que permita predecir de manera más precisa si una observación pertenece a un grupo o no.

3. **Regla de actualización**: Ajustar los pesos para optimizar la probabilidad de predicción, refinando la línea divisoria dentro del conjunto de datos.

Conocer estos pilares te ayudará a aplicar la regresión logística eficazmente en diversas situaciones prácticas. Deberás recordar que, como en la matemática o la programación, ensayo y error son parte del proceso. ¡No te desanimes, sigue aprendiendo y dominando esta técnica!

**Lecturas recomendadas**

[Regresión logística](https://platzi.com/clases/2081-ds-probabilidad/33070-regresion-logistica/)

## Clasificadores de Bosque Aleatorio: Conceptos y Aplicaciones

¡Con gusto! Vamos a explorar los **Clasificadores de Bosque Aleatorio (Random Forest Classifiers)**, una herramienta muy potente y versátil en Machine Learning.

### 🌳 ¿Qué es un Bosque Aleatorio (Random Forest)?

Un **Random Forest** es un modelo de *aprendizaje supervisado* que combina múltiples **árboles de decisión** (decision trees) para mejorar la precisión y controlar el sobreajuste (overfitting).
Funciona tanto para **clasificación** como para **regresión**, pero aquí nos enfocamos en **clasificación**.

### 🧠 ¿Cómo funciona?

1. Crea múltiples árboles de decisión usando distintos subconjuntos aleatorios de los datos (**bootstrap**).
2. Cada árbol da una predicción.
3. Para clasificación, el bosque elige la clase **más votada** (mayoría de votos).

### 🔑 Características Clave

* **Robusto** al sobreajuste (mejor que un solo árbol).
* **No lineal**: puede modelar relaciones complejas.
* **Tolerante a datos faltantes y ruidosos**.
* **Proporciona importancia de variables** automáticamente.

### ✅ ¿Cuándo usarlo?

Usa Random Forest cuando:

* Necesitas un modelo preciso sin mucho ajuste.
* Tienes muchas características (features).
* Quieres saber qué variables son más importantes.
* Tus datos son ruidosos o tienen valores atípicos.

### 🛠️ Implementación en Python

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar datos de ejemplo
data = load_iris()
X = data.data
y = data.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predicción
y_pred = rf.predict(X_test)

# Evaluación
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
```

### 📊 Importancia de las Características

```python
import matplotlib.pyplot as plt
import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.title("Importancia de las variables")
plt.show()
```

### 📌 Hiperparámetros comunes

* `n_estimators`: número de árboles (más = más robusto, pero más lento)
* `max_depth`: profundidad máxima de cada árbol
* `max_features`: número de características a considerar en cada split
* `bootstrap`: si se usa muestreo con reemplazo

### ⚠️ Consideraciones

* Puede ser más lento y consumir más memoria que modelos simples.
* Poca interpretabilidad (no es un modelo explicativo).
* Los árboles no se pueden visualizar fácilmente como en un `DecisionTreeClassifier`.

### 🔁 Casos de uso

* Clasificación de textos o correos electrónicos (spam vs. no spam)
* Diagnóstico médico (enfermo vs. sano)
* Predicción de abandono de clientes (churn)
* Sistemas de detección de fraude

### Resumen

#### ¿Qué es un clasificador de bosque aleatorio?

Un clasificador de bosque aleatorio es una herramienta poderosa en el ámbito del aprendizaje automático, específicamente diseñada para etiquetar datos de manera precisa y eficiente. Se basa en unir múltiples árboles de decisión para mejorar la precisión en las predicciones y evitar errores comunes, como etiquetar incorrectamente datos nuevos. Este enfoque es especialmente útil cuando se necesita tomar decisiones rápidas y fundamentadas, como al determinar si un juguete es seguro para un niño.

#### ¿Cómo funciona un árbol de decisión?

El árbol de decisión es el componente básico del clasificador de bosque aleatorio. Imagina que debes decidir si un juguete es seguro. Comienzas formulando preguntas basadas en las características del juguete, como su color o forma. Cada pregunta divide tus datos en categorías, separando los elementos peligrosos de los seguros. Los nodos de decisión corresponden a estas preguntas, mientras que los nodos hoja representan el resultado final de las preguntas realizadas.

#### Ejemplo práctico: Clasificación de juguetes

Supongamos que encuentras dos nuevos juguetes: un círculo rosa y un círculo azul. Puedes realizar preguntas similares para determinar su seguridad. Si preguntas "¿es un círculo?" y la respuesta es afirmativa, el juguete se considera seguro. Sin embargo, si el modelo predice incorrectamente que un círculo azul es peligroso solo porque es azul, podrías necesitar ajustar tus criterios. Aquí es donde entra en juego el bosque aleatorio.

#### ¿Por qué utilizar un bosque aleatorio?

El bosque aleatorio ayuda a corregir errores de clasificación al incluir múltiples árboles de decisión que "votan" por la respuesta correcta. Cada árbol proporciona una respuesta basada en diferentes divisiones de datos, y la respuesta más votada es la que se adopta. Esto garantiza un etiquetado más preciso y reduce el riesgo de sesgos en las predicciones.

#### Componentes clave del bosque aleatorio

- **Número de árboles**: La cantidad de árboles de decisión que tienes. A mayor cantidad, tu modelo será más robusto, pero también requerirá más recursos computacionales.
- **Número máximo de features**: Las características que eliges para clasificar y predecir resultados.
- **Profundidad máxima**: El número máximo de preguntas que un árbol puede hacer antes de llegar a una conclusión. Profundizar permite realizar análisis más complejos.
- **Parámetros "n split" y "n min"**: Controlan la cantidad mínima de datos necesarios para hacer una división en un nodo y el número mínimo de puntos de datos en un nodo hoja antes de detener el proceso de decisión.

#### ¿Cómo evaluar el rendimiento de un bosque aleatorio?

Para medir la efectividad de un bosque aleatorio, se utilizan métricas de clasificación y regresión. En clasificación, la matriz de confusión es una herramienta esencial, ya que permite comparar las predicciones con los valores reales para determinar la precisión del modelo. En tareas de regresión, se puede trazar la correlación entre los valores predichos y los reales para observar cómo se alinean.

#### Consideraciones finales sobre el proceso de decisión

El proceso de decisión en un bosque aleatorio involucra seleccionar conjuntos de características y determinar cuál es la mejor manera de dividir los datos. La función de coste busca el umbral óptimo para estas divisiones, mientras que la regla de actualización dicta continuar o detenerse en función de los valores mínimos presentes en los nodos hoja.

Con esta comprensión clara y detallada del bosque aleatorio, podrás integrar este potente modelo de aprendizaje automático en tus proyectos de forma efectiva. Te animamos a seguir explorando en el fascinante mundo del machine learning y a experimentar con diferentes configuraciones para obtener los mejores resultados. ¡El conocimiento es poder, y en tus manos está aprovecharlo al máximo!

## Aprendizaje No Supervisado: Clustering con K-means

¡Perfecto! El **Clustering con K-means** es una técnica clásica y poderosa de **aprendizaje no supervisado** utilizada para **agrupar datos sin etiquetas** previas.

### 🧠 ¿Qué es el Aprendizaje No Supervisado?

Es un tipo de aprendizaje automático donde **no se conoce la etiqueta o categoría de los datos**. El algoritmo intenta **descubrir estructuras** ocultas o patrones en los datos.

### 🔵 ¿Qué es K-means?

**K-means** es un algoritmo que:

* Agrupa datos en **K grupos (clusters)**.
* Cada grupo está representado por el **centroide (promedio)** de sus puntos.
* Los puntos se asignan al cluster **más cercano** (distancia euclidiana, generalmente).

### 🔁 ¿Cómo funciona?

1. Se eligen **K centros** aleatoriamente.
2. Cada punto se asigna al **centro más cercano**.
3. Se actualizan los centros como el **promedio** de los puntos asignados.
4. Se repiten los pasos 2 y 3 hasta que los centros no cambian significativamente (**convergencia**).

### 🧪 Ejemplo en Python

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Crear datos simulados
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Visualizar resultados
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centros = kmeans.cluster_centers_
plt.scatter(centros[:, 0], centros[:, 1], c='red', s=200, alpha=0.75)
plt.title("Clustering con K-means")
plt.show()
```

### 📐 ¿Cómo elegir el número K?

Se usa el método del **codo (elbow method)**:

* Se prueba con distintos valores de K.
* Se calcula la **suma de errores cuadrados** (inertia).
* Se elige el K donde la mejora se estabiliza (el "codo").

```python
inercia = []
for k in range(1, 10):
    modelo = KMeans(n_clusters=k)
    modelo.fit(X)
    inercia.append(modelo.inertia_)

plt.plot(range(1, 10), inercia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.show()
```

### ✅ Ventajas de K-means

* Fácil de entender y rápido de implementar.
* Funciona bien con grandes conjuntos de datos.
* Resultados interpretables si K está bien elegido.

### ⚠️ Limitaciones

* Hay que especificar **K** manualmente.
* Asume clusters esféricos y del mismo tamaño.
* Sensible a la escala de los datos y a valores atípicos.
* Puede converger en **mínimos locales**.

### 🧰 Casos de Uso

* Agrupamiento de clientes por comportamiento.
* Segmentación de mercado.
* Compresión de imágenes.
* Detección de patrones o anomalías.

### Resumen

#### ¿Qué es K-means en el aprendizaje no supervisado?

El aprendizaje no supervisado es una fascinante rama de la inteligencia artificial enfocada en encontrar estructuras ocultas en los datos sin la necesidad de etiquetarlos previamente. Un ejemplo destacado dentro de este enfoque es el algoritmo K-means, utilizado con frecuencia en tareas de agrupamiento. ¿Por qué? Está diseñado para identificar y asignar puntos de datos a grupos o "clusters", permitiendo una análisis de patrones de manera efectiva.

#### ¿Cómo funciona K-means?

El corazón del K-means yace en el concepto de "centroide", que actúa como líder o representante de un cluster particular. Estos centroides pueden colocarse inicialmente al azar en el espacio de datos, pero el proceso luego se encarga de ajustarlos para representar mejor a los datos.

Los pasos generales del algoritmo son:

1. **Inicialización aleatoria**: Se eligen posiciones al azar para los centroides.
2. **Asignación de pertenencia**: Cada punto de datos se asocia al centroide más cercano, formando así un cluster.
3. **Actualización de centroides**: Los centroides se reubican calculando la media de los puntos dentro de cada cluster.
4. **Repetición de los pasos**: Se repiten los pasos 2 y 3 hasta que los centroides estabilicen su posición o las asignaciones de clusters no cambien más.

#### ¿Cuáles son los parámetros clave en K-means?

El parámetro más crítico en K-means es el valor "K", que representa el número de clusters deseados. Al variar "K", se pueden obtener agrupamientos con diferentes formas y estructuras, lo que hace fundamental elegir un valor adecuado.

**Ejemplo práctico y visualización**

Imaginemos ejecutar K-means con diferentes valores de "K" para un mismo conjunto de datos. Al aumentar "K" desde 2 hasta 4, se observa cómo las agrupaciones cambian tanto en forma como en número de puntos por cada cluster. Para refinar este proceso, se utilizan métricas de rendimiento que ayudan a determinar si el número de "K" es ideal para un modelo específico.

#### ¿Cuál es la función de coste en K-means?

El objetivo principal de K-means es optimizar la posición de los centroides de manera que los puntos de datos estén lo más cerca posible a su centroides asignado. En otras palabras, minimiza la suma de las distancias al cuadrado desde cada punto hasta su centroide correspondiente.

Este proceso garantiza que los grupos de datos resultantes sean lo más compactos y diferenciados posibles.

#### ¿Cómo se actualizan los centroides?

**Regla de actualización**: Los centroides se recalculan basándose en las medias de los puntos del cluster. Este nuevo cálculo redefine la posición de los centroides para reflejar mejor su cluster. El ciclo de recalculación continúa hasta que:

- La posición de los centroides cambia de manera insignificante,
- O no hay cambios en las asignaciones de los puntos a los clusters.

#### ¿Cómo determinar el valor adecuado de "K"?

Seleccionar el "K" correcto puede ser desafiante pero crucial para un modelo exitoso. Algunas técnicas comunes incluyen:

- **Inercia**: Evalúa cuán agrupados están los puntos a su centroide; se busca que este valor sea lo más bajo posible.
- **Puntuación de silueta**: Mide la separación entre clusters; un valor cercano a uno indica una buena separación.
- **Elbow plot (gráfico de codo)**: Traza la inercia en función de "K". La curva resultante ayudará a identificar el "K" óptimo, donde añadir más clusters no mejora significativamente la agrupación.

#### Exploración mediante el conjunto de datos Iris

Una manera práctica y entretenida de asimilar estos conceptos es probando K-means con el conjunto de datos Iris, famoso en el mundo del machine learning. Contar con diferentes características de las flores permite no solo agruparlas efectivamente, sino experimentar con distintas configuraciones del algoritmo.

Los datos de Iris se utilizan para predecir la categorización basada en características como el ancho y largo de los sépalos y pétalos.

Estas características convierten a K-means en una herramienta poderosa para estructurar y entender datos sin etiquetar. Su aplicación en diversas áreas del análisis de datos lo hace esencial para los científicos de datos y analistas. Si te animas, te invito a experimentar con tus propios conjuntos de datos y explorar el mundo visual y dinámico de K-means.

## Guía práctica de algoritmos de machine learning con scikit-learn

¡Hola! Te doy la bienvenida a esta clase con un nuevo reto.

Antes que nada espero que te haya ido excelente con el notebook del reto anterior y que hayas completado todos los ejercicios. Recuerda que cualquier duda puedes dejarla en comentarios de la clase para que toda la comunidad de Platzi pueda apoyarte.

**Notebook de algoritmos de machine learning**

En la [notebook en Google Colab](https://colab.research.google.com/drive/1OwBLVJmV-xdwwSdEIINgyioetB_5jjVQ?usp=sharing "notebook en Google Colab") de esta clase encontrarás una guía para probar **algoritmos de machine learning** en código. Desde la **carga de datos** hasta e**ntrenar el modelo** y **verificar su performance**.

Para estos ejemplos utilizaremos la librería de scikit-learn, una de las librerías con las que podemos comenzar a aprender el uso de algoritmos de machine learning de manera más sencilla.

En los siguientes cursos de la ruta de la Escuela de Data Science profundizarás en el uso de [scikit-lear](https://scikit-learn.org/stable/ "scikit-lear")n y otras librerías de machine learning. ¡Comencemos con la base! 🧠

[Accede al notebook aquí.](https://colab.research.google.com/drive/1OwBLVJmV-xdwwSdEIINgyioetB_5jjVQ?usp=sharing "Accede al notebook aquí.")

Crea una copia de este notebook en tu Google Drive o utilízalo en el entorno de Jupyter notebook que prefieras. Recuerda instalar las librerías necesarias para ejecutar el código si ejecutas tu notebook en un entorno local.

Esta notebook no tiene ejercicios adicionales como la anterior, pero este el **reto** que tienes para esta clase:
- Identifica en qué partes del código aplicamos los diferentes conceptos teóricos aprendidos en las clases anteriores.
¡Leo tus anotaciones en los comentarios y nos vemos en el próximo módulo!

## Fundamentos de Redes Neuronales y Deep Learning

¡Vamos a ello! Aquí tienes una introducción clara y estructurada sobre los **Fundamentos de Redes Neuronales y Deep Learning**, esenciales en el campo del Machine Learning avanzado.

### 🧠 ¿Qué es una Red Neuronal?

Una **Red Neuronal Artificial (RNA)** es un modelo inspirado en el cerebro humano que se compone de **nodos (neuronas)** organizados en **capas**. Su objetivo es **aprender patrones complejos** a partir de datos.

### 🏗️ Estructura Básica

1. **Capa de entrada (input)**: recibe los datos.
2. **Capas ocultas (hidden layers)**: transforman los datos mediante operaciones matemáticas.
3. **Capa de salida (output)**: da la predicción final.

Cada **neurona**:

* Recibe entradas.
* Multiplica por pesos.
* Suma un sesgo.
* Aplica una función de activación no lineal.

```text
       Entrada       →       Neuronas ocultas       →    Salida
  [X1, X2, X3, ...]          (con pesos + bias)          [Predicción]
```

### 🔢 Matemáticamente

Para una neurona:

```python
z = w1*x1 + w2*x2 + ... + wn*xn + b
a = f(z)
```

* `w`: pesos
* `x`: entradas
* `b`: sesgo
* `f`: función de activación (ReLU, Sigmoid, etc.)
* `a`: activación (salida)

### 🔄 ¿Cómo aprenden las redes?

**Aprenden mediante entrenamiento**, ajustando los pesos para minimizar un error (pérdida):

1. **Forward pass**: calcula la salida y el error.
2. **Backward pass (backpropagation)**: ajusta los pesos en dirección contraria al error.
3. **Optimización**: se usa un algoritmo como **Gradient Descent**.

### 🔧 Funciones de Activación Comunes

| Función | Fórmula                           | Uso principal                    |
| ------- | --------------------------------- | -------------------------------- |
| Sigmoid | 1 / (1 + exp(-x))                 | Clasificación binaria            |
| ReLU    | max(0, x)                         | Capas ocultas en redes profundas |
| Softmax | exp(x\_i)/Σexp(x\_j)              | Clasificación multiclase         |
| Tanh    | (exp(x)-exp(-x))/(exp(x)+exp(-x)) | Escala entre -1 y 1              |

### 🧠 ¿Qué es Deep Learning?

**Deep Learning** = redes neuronales con **muchas capas ocultas** (deep = profundo).
Permiten aprender **representaciones jerárquicas** de los datos (por ejemplo, de píxeles a caras).

### 📊 Aplicaciones de Deep Learning

* Visión por computadora (reconocimiento facial, objetos)
* Procesamiento de lenguaje natural (traducción, chatbots)
* Audio (reconocimiento de voz)
* Juegos (AlphaGo)
* Medicina (diagnóstico por imágenes)

### 🛠️ Herramientas y Librerías

* **TensorFlow** (Google)
* **PyTorch** (Meta/Facebook)
* Keras (interfaz de alto nivel para TensorFlow)

### 🧪 Ejemplo Simple en PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Red con una capa oculta
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Datos de ejemplo
X = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# Pérdida y optimizador
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Entrenamiento simple
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

print("Entrenado, pérdida final:", loss.item())
```

### 📌 Conceptos Importantes Relacionados

* **Overfitting**: cuando el modelo memoriza en lugar de generalizar.
* **Regularización (L2, Dropout)**: ayuda a evitar overfitting.
* **Batch size, epochs, learning rate**: hiperparámetros que afectan el entrenamiento.

### Resumen

#### ¿Qué es una red neuronal?

Las redes neuronales son fundamentales en el campo del deep learning, utilizadas en diversos sectores para resolver problemas complejos como el etiquetado de imágenes o la generación de texto. Funcionan mediante la simulación de las conexiones neuronales del cerebro humano, permitiendo una transformación profunda en cómo las máquinas entienden y procesan información.

#### ¿Cómo está estructurada una red neuronal?

Una red neuronal típica está formada por tres componentes clave:

- **Capa de entrada**: Recepción de datos procesados que se transmiten a la capa oculta. Estos datos suelen incluir información relevante, como características demográficas o anteriores compras en un modelo que pretende predecir el comportamiento de compra de los clientes.

- **Capa oculta**: Este es el núcleo de las operaciones complejas. Oficia como un centro de procesamiento donde se realizan módulos y funciones complejas que permiten clasificar y analizar datos, como reconocer si una imagen es de un perro o un gato o incluso generar texto imitando un tuit. Esta capa está formada por unidades ocultas interconectadas, manipulando los datos a través de ajustes en parámetros como pesos y funciones de activación.

- **Capa de salida**: Es el componente decisivo de la red que proporciona la respuesta o predicción esperada. A menudo, esta capa se utiliza para emitir una probabilidad u otra medida que ayude en la toma de decisiones.

#### ¿Cuáles son los tipos de funciones de activación?

Las funciones de activación son esenciales para determinar el comportamiento de las salidas en cada neurona de las redes. Actúan como filtros que permiten o bloquean el paso de ciertos valores. Aquí algunos ejemplos:

- **Función ReLU (Rectified Linear Unit)**: Esta función introduce una no linealidad, aplicada habitualmente en capas ocultas, permitiendo la propagación de valores positivos y bloqueando los negativos.

- **Softmax**: Común en salidas de clasificación, convierte un vector de números en una distribución de probabilidad sobre las diferentes clases, útil para predecir la clase correcta de un ítem.

- **Sigmoid**: Genera un output de probabilidad entre 0 y 1, ofreciendo una visión clara para distinguir entre categorías binarias.

#### ¿Qué es el deep learning?

El deep learning se refiere a incrementar la profundidad de nuestras redes entendiendo dos aspectos principales:

- **Profundidad**: Añadir más capas ocultas, facilitando representaciones de mayor complejidad de los datos de entrada.

- **Anchura**: Incluir más unidades ocultas dentro de una capa, diversificando la manera en que se procesa la información.

La manipulación experta de estos parámetros es vital para resolver problemas específicos, y en el futuro, comprenderemos cómo entrenar estas redes para optimizar su rendimiento.

#### ¿Cómo entrenar una red neuronal?

El entrenamiento de una red neuronal implica ajustar sus parámetros para mejorar en la tarea que se desea realizar. Este proceso suele incluir la aplicación de métodos como el descenso del gradiente para optimizar los pesos de las conexiones entre neuronas. En la próxima serie de clases nos sumergiremos en estos mecanismos, explorando cómo actualizarlos y evaluarlos de manera exhaustiva. ¡No pierdas la oportunidad de avanzar en este fascinante campo!

## Mejora de Redes Neuronales: Ajuste, Overfitting y Dropout

¡Claro! Aquí tienes una guía clara y concisa sobre la **mejora de redes neuronales**, enfocándonos en **ajuste de modelos, overfitting y dropout**, aspectos fundamentales en Deep Learning.

### 🎯 Objetivo: Mejorar el Rendimiento del Modelo

Cuando entrenamos una red neuronal, el objetivo es que **aprenda patrones generales** del conjunto de datos, no que los **memorice**. Aquí es donde entran conceptos como **ajuste del modelo**, **overfitting**, **underfitting** y **regularización** (como **Dropout**).

### 🔧 1. Ajuste del Modelo (Model Tuning)

El **ajuste del modelo** implica encontrar la mejor combinación de:

* Número de capas y neuronas
* Función de activación
* Tasa de aprendizaje (`learning rate`)
* Épocas de entrenamiento (`epochs`)
* Optimizador (SGD, Adam, RMSProp…)

También incluye:

* Normalización de los datos
* Elección de la arquitectura adecuada
* Tamaño del lote (`batch size`)

### 🛠 Técnicas comunes:

* **Búsqueda aleatoria o grid search** para hiperparámetros
* **Early Stopping**: detener el entrenamiento si la validación ya no mejora

### ⚠️ 2. Overfitting y Underfitting

| Situación        | Descripción                                  | Consecuencia                                    |
| ---------------- | -------------------------------------------- | ----------------------------------------------- |
| **Underfitting** | El modelo no aprende lo suficiente           | Mala precisión en entrenamiento y prueba        |
| **Overfitting**  | El modelo aprende demasiado (memoriza datos) | Alta precisión en entrenamiento, baja en prueba |

🔍 **Cómo detectarlo**:

* **Overfitting**: pérdida (loss) de entrenamiento baja, pero validación alta.
* **Underfitting**: ambas pérdidas (entrenamiento y validación) son altas.

### 🧯 3. Dropout: Técnica de Regularización

**Dropout** es una técnica para **prevenir el overfitting**. Durante el entrenamiento, **desactiva aleatoriamente algunas neuronas** de la red (con probabilidad *p*) en cada batch.

### 🎲 ¿Qué hace?

* Evita que las neuronas dependan excesivamente unas de otras.
* Fuerza al modelo a generalizar mejor.

### 💡 Ejemplo en PyTorch:

```python
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # Dropout del 50%
    nn.Linear(64, 10),
    nn.Softmax(dim=1)
)
```

### 🤖 Dropout solo se aplica durante el **entrenamiento**, no en la inferencia (evaluación).

### ✅ Otras Técnicas para Mejorar Generalización

* **Regularización L2** (`weight_decay`): penaliza pesos grandes.
* **Aumento de datos (data augmentation)**: útil en imágenes y texto.
* **Batch Normalization**: estabiliza y acelera el entrenamiento.
* **Reducir la complejidad del modelo** (menos capas o neuronas).

### 📊 Visualización de Overfitting

```text
Epochs →
│
│   🟥 Entrenamiento ↓       ← pérdida baja
│   🟦 Validación ↑           ← pérdida alta
│
└─────────────────────────
        Overfitting detectado
```

### 🧠 En resumen:

| Técnica           | Propósito                          |
| ----------------- | ---------------------------------- |
| Dropout           | Evitar overfitting                 |
| Early Stopping    | Detener antes de que empeore       |
| L2 Regularización | Controlar tamaño de los pesos      |
| Data Augmentation | Aumentar la diversidad del dataset |

### Resumen

#### ¿Cómo mejorar las redes neuronales para obtener predicciones robustas?

En el apasionante mundo de las redes neuronales, dominar la habilidad de crear modelos robustos y precisos es esencial. Este articulo buscará guiarte a través de los pasos necesarios para mejorar tus redes neuronales, asegurando que hagan predicciones estables y confiables al evaluar correctamente las preguntas planteadas. La clave está en encontrar el equilibrio adecuado entre diferentes metodologías y entender cuándo un modelo está haciendo un buen trabajo o cuándo necesita ajustes.

#### ¿Qué significa el ajuste de modelos en redes neuronales?

En primer lugar, comprender cómo los modelos se ajustan a los datos es crucial. Existen tres situaciones distintas respecto a esto:

- Bajo ajuste (Underfitting): Ocurre cuando el modelo no ha captado correctamente el patrón de los datos de entrenamiento, lo que compromete el potencial de hacer predicciones precisas.
- Ajuste ideal: El modelo identifica adecuadamente los patrones subyacentes en los datos, logrando una predicción efectiva.
- Sobreajuste (Overfitting): Aquí, el modelo memoriza los datos de entrenamiento sin comprender realmente los patrones, lo que limita su capacidad de generalizar a nuevos datos.

#### ¿Cómo evitar el sobreajuste?

Un desafío común en redes neuronales, dada su gran cantidad de parámetros, es el sobreajuste. Afortunadamente, contamos con técnicas como el dropout para mitigar este problema. Durante el entrenamiento, el dropout actúa desactivando temporalmente algunos nodos ocultos, lo que previene que el modelo adquiera demasiada información y se limite a memorizar.

#### ¿Cómo determinar el número óptimo de épocas?

El procedimiento de entrenamiento en redes neuronales implica un ciclo repetitivo de pases hacia adelante, cálculo de pérdidas y retropropagación. Un ciclo completo de este proceso, para cada dato, se denomina época. La clave está en encontrar el balance adecuado de épocas para garantizar que la red generalice bien.

- **Uso de conjuntos de validación**: Esta técnica ayuda a evaluar si los patrones aprendidos en los datos de entrenamiento son aplicables al conjunto de validación. El objetivo es seleccionar un modelo donde el rendimiento de validación alcance su punto máximo antes de estancarse.

#### ¿Qué hemos aprendido sobre la estructura de las redes neuronales?

Hasta ahora, hemos explorado las partes esenciales de una red neuronal:

- **Capas de la red**:

 - **Capa de entrada**: Procesa los features iniciales del problema.
 - **Capas ocultas**: Manipulan las características para abordar problemas complejos mediante operaciones internas.
 - Capa de salida: Realiza la predicción final, ya sea de tipo regresión o clasificación.

- **Activación**: Presente en las capas ocultas y de salida, permite obtener representaciones más complejas y detalladas de los datos de entrada.

#### ¿Cómo se optimiza el entrenamiento de una red neuronal?

El entrenamiento eficiente de una red neuronal es un proceso continuo y dinámico que implica:

1. **Paso hacia adelante**: Proyección inicial de los datos a través de la red.
2. **Cálculo de pérdidas**: Determina qué tan efectiva es la predicción actual.
3. **Retropropagación**: Actualiza los pesos y ajustes de la red en base al error calculado, afinando así el modelo.

Estas prácticas combinadas con el manejo adecuado del dropout y la evaluación del rendimiento de validación pueden guiarte hacia la creación de modelos más robustos. Si bien has aprendido mucho sobre ciencia de datos y redes neuronales hasta este punto, recuerda que siempre hay más por descubrir en el campo del machine learning. ¡Anímate a seguir explorando para expandir tus conocimientos! Nos encontraremos nuevamente en próximas oportunidades para profundizar aún más en este fascinante espacio.

## Entrenamiento Efectivo de Redes Neuronales: Arquitectura y Tasa de Aprendizaje

¡Perfecto! Vamos a ver cómo lograr un **entrenamiento efectivo de redes neuronales** centrándonos en dos factores críticos: **la arquitectura** y **la tasa de aprendizaje (learning rate)**. Ambos son clave para obtener modelos precisos, eficientes y que generalicen bien.

### 🏗️ 1. Arquitectura de Redes Neuronales

La **arquitectura** define **cómo está construida la red**: número de capas, tipo de capas, cuántas neuronas por capa, funciones de activación, etc.

### 🔹 Componentes comunes:

* **Capas densas (Fully connected)**: típicas en redes simples.
* **Capas convolucionales (CNNs)**: visión por computadora.
* **Capas recurrentes (RNNs, LSTM)**: procesamiento de secuencias.
* **Capas de normalización (BatchNorm)**: estabilizan el aprendizaje.
* **Capas de regularización (Dropout)**: evitan overfitting.

### 📌 Buenas prácticas:

* **Empieza simple**: pocas capas, pocas neuronas.
* **Profundiza gradualmente**: si el modelo underfitea.
* **No uses más parámetros de los necesarios**: puede sobreajustar.

```python
# Arquitectura sencilla en PyTorch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(10, 64),  # Capa de entrada
    nn.ReLU(),
    nn.Linear(64, 32),  # Capa oculta
    nn.ReLU(),
    nn.Linear(32, 1),   # Capa de salida
    nn.Sigmoid()
)
```

### 📉 2. Tasa de Aprendizaje (Learning Rate)

La **tasa de aprendizaje** determina cuánto se ajustan los pesos en cada paso del entrenamiento.

### 🔢 Valores típicos:

* `0.1` → muy alto (puede saltarse el mínimo)
* `0.01` → común
* `0.001` o menos → más lento, pero más preciso

### ⚠️ Problemas frecuentes:

| Problema              | Síntoma                             |
| --------------------- | ----------------------------------- |
| Tasa muy alta         | La pérdida oscila o nunca disminuye |
| Tasa muy baja         | Aprendizaje extremadamente lento    |
| Tasa variable (ideal) | Disminuye al acercarse al óptimo    |

### 📌 Soluciones avanzadas:

* **Learning rate decay**: reducir la tasa durante el entrenamiento.
* **Schedulers** en PyTorch: `StepLR`, `ReduceLROnPlateau`, `ExponentialLR`, etc.
* **Warm-up**: comenzar con tasa baja e ir subiendo.

```python
import torch.optim as optim

optimizer = optim.Adam(modelo.parameters(), lr=0.01)

# Programador de tasa de aprendizaje
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```

### 📊 Comparación visual (conceptual):

```text
                Tasa de aprendizaje

 Pérdida
   ▲
   │  ╭╮       ← tasa muy alta: oscilación
   │ ╱╲╱╲
   │   ╲__     ← tasa correcta: descenso suave
   │      ╲
   │       ╲__ ← tasa muy baja: lento o estancado
   └──────────────► Épocas
```

### 🧠 Consejos Finales para Entrenamiento Efectivo

✅ **Normaliza los datos** antes de entrenar.
✅ Usa **validación cruzada** para evaluar generalización.
✅ Controla el **overfitting** con Dropout o Early Stopping.
✅ Ajusta la arquitectura y tasa de aprendizaje **con experimentación controlada**.
✅ Usa **gráficos de pérdida y precisión** para guiar decisiones.

### Resumen

#### ¿Cómo entrenar redes neuronales efectivamente?

El entrenamiento de redes neuronales es un elemento crucial en su implementación y éxito. Nos encontramos en una era donde la inteligencia artificial avanza rápidamente, y comprender cómo optimizar estas poderosas herramientas es vital. Vamos a explorar las mejores prácticas para asegurarnos de que nuestras redes neuronales estén funcionando óptimamente, desde la elección de la arquitectura hasta el manejo de la tasa de aprendizaje.

#### ¿Qué tipos de arquitecturas de redes neuronales existen?

Seleccionar la arquitectura adecuada para una red neuronal es el primer paso esencial en su entrenamiento. Cada tipo de arquitectura tiene características únicas que la hacen más adecuada para ciertos problemas.

1. **Redes neuronales profundas**: Usan funciones de activación y son ideales para resolver problemas complejos no lineales. Son especialmente útiles donde no se aplican modelos lineales.

2. **Redes neuronales convolucionales**: Utilizan operadores convolucionales y mecanismos de agrupación, y son excelentes para captar motivos y escalas en datos visuales, como imágenes y genomics.

3. **Redes neuronales recurrente**s: Estas redes implementan un concepto de memoria, permitiéndoles recordar secuencias largas. Se emplean principalmente en modelos lingüísticos, donde es crucial retener contexto a lo largo de una secuencia de frases o palabras.

#### ¿Cuál es la receta de entrenamiento para redes neuronales?

Una vez que tenemos la arquitectura adecuada, el siguiente paso es seguir una receta de entrenamiento efectiva. Este proceso generalmente incluye tres etapas:

1. **Cálculo de avance (feed forward)**: Partimos desde la entrada y avanzamos hasta la capa de salida, utilizando funciones de activación lineales o no lineales para evaluar el valor de predicción.

2. **Función de pérdida**: Mide qué tan bien una red neuronal predice un valor comparado con el valor real. Para problemas de regresión se utiliza la pérdida de error cuadrático medio, mientras que para problemas de clasificación, se podrían usar funciones de pérdida como la entropía cruzada binaria.

3. **Propagación hacia atrás (backpropagation)**: Este paso evalúa los pesos desde la capa de salida a la capa de entrada, ajustando los pesos para minimizar la función de pérdida.

#### ¿Cómo mejorar el desempeño de las redes neuronales?

A medida que avanza el entrenamiento, es importante monitorear la pérdida y el desempeño general del modelo para evitar el sobreajuste, un fenómeno donde la red aprende demasiado específicamente de los datos de entrenamiento. Algunas estrategias para mejorar el desempeño incluyen:

- **Uso de datos de validación**: Ayuda a asegurarse de que el modelo está verdaderamente generalizando lo aprendido, en lugar de memorizar los ejemplos de entrenamiento.

- **Optimización de la tasa de aprendizaje**: Ajustar adecuadamente la tasa de aprendizaje es crucial. Una tasa muy baja provocará un entrenamiento lento, mientras que una tasa muy alta puede causar inestabilidad en el modelo.

En resumen, el entrenamiento efectivo de redes neuronales requiere una planificación cuidadosa y ajustes constantes. Con paciencia y práctica, podemos aprovechar al máximo el potencial de estas herramientas poderosas. A medida que continúas explorando este fascinante campo, recuerda que cada reto es una oportunidad para aprender y mejorar.

## Curso de Fundamentos Prácticos de Machine Learning

Antes de que te vayas quiero contarte algo más. Durante este curso has aprendido las bases teóricas de machine learning y has comenzado a jugar con código de modelos con tu primera librería. Pero esto solo es el principio, es la introducción.

Como te comenté en la clase anterior, todavía hay mucho más por aprender en machine learning, deep learning e inteligencia artificial en general. Por ello quiero compartirte la ruta para continuar aprendiendo:

**Machine learning**
### Curso de Fundamentos Prácticos de Machine Learning

Profundiza en el uso práctico de regresiones, árboles de decisión, clusterización con k-means e incluso toca las bases del deep learning.

Tómalo [aquí](https://platzi.com/cursos/fundamentos-ml/ "aquí").

### Curso Profesional de Machine Learning con Scikit-Learn

Con este curso aprenderás a implementar los principales algoritmos disponibles en scikit-learn de manera profesional. Visitarás temas como optimización de features, optimización paramétrica, salida a producción y más.

[Tómalo aquí](https://platzi.com/cursos/scikitlearn/ "Tómalo aquí").

**Deep learning con redes neuronales**

### Curso de Fundamentos de Redes Neuronales con Python y Keras

Conoce cómo funcionan las redes neuronales creando una red neuronal con Python y Numpy. Aprende a utilizar Keras, la librería esencial para aprender el uso de redes neuronales.

Tómalo [aquí](https://platzi.com/cursos/redes-neuronales/ "aquí").

**Procesamiento de lenguaje natural**

### Curso de Fundamentos de Procesamiento de Lenguaje Natural con Python y NLTK

Aprende cómo los algoritmos pueden aprender a procesar el lenguaje humano con Python y NLTK y entrena tus primeros modelos de procesamiento de lenguaje natural.

Tómalo [aquí](https://platzi.com/cursos/python-lenguaje-natural/ "aquí").

Curso de Algoritmos de Clasificación de Texto
Descubre las aplicaciones del procesamiento de lenguaje natural en clasificación de textos. Comprende y usa estos algoritmos con Python y la librería NLTK.

Tómalo [aquí](https://platzi.com/cursos/clasificacion-texto/ "aquí").

Para terminar te invito a que regreses a este curso cada vez que tengas dudas sobre las bases que vimos y necesites recordarlas. Te ayudará mucho mientras sigues aprendiendo con los siguientes cursos que te recomendé.

¡Ahora sí nos vemos en la próxima!

## Resumen del Curso de Machine Learning y Habilidades Avanzadas

### ¿Qué habilidades se han desarrollado en este curso de machine learning?
En este curso hemos hecho un recorrido exhaustivo por los conceptos fundamentales del machine learning. Partimos desde los principios básicos de la ciencia de datos, explorando cómo cargar y trabajar con datos efectivamente. También abordamos la visualización, que nos ayuda a entender las relaciones entre las distintas características de un conjunto de datos.

A lo largo del curso, profundizamos en distintos enfoques de aprendizaje:

- **Aprendizaje supervisad**o: Estudiamos cómo predecir objetivos en contextos de regresión y clasificación, dotándonos de herramientas para enfrentar una amplia gama de problemas.
- **Aprendizaje no supervisado**: Aprendimos a descubrir la estructura de los datos sin etiquetas preconcebidas, permitiéndonos revelar patrones ocultos e insights valiosos.
- **Redes neuronales**: Exploramos estos modelos avanzados que permiten realizar predicciones de funciones complejas, aprendiendo a entrenarlos y evaluar su rendimiento.

### ¿Qué otras habilidades pueden ampliar tu comprensión en machine learning?

El campo del machine learning es vasto y en constante evolución. Afortunadamente, en Platzi puedes seguir ampliando tus conocimientos con una variedad de cursos diseñados para diferentes niveles y necesidades. Aquí hay algunas recomendaciones sobre lo que puedes estudiar a continuación:

- **Machine Learning práctico y avanzado**: Profundizar en el aprendizaje automático y explorar casos de uso en la vida cotidiana te ayudará a aplicar lo aprendido en situaciones reales.
- **Uso avanzado de bibliotecas de ML**: Conocer bibliotecas avanzadas como TensorFlow es crucial para desplegar técnicas de machine learning efectivas.
- **Despliegue de aplicaciones de ML**: Aprender a llevar tus modelos a producción es una habilidad altamente demandada en la industria.
- **Procesamiento del lenguaje natural (NLP)**: Entender las aplicaciones especializadas de machine learning, como el NLP, te permitirá abordar desafíos específicos con un enfoque adecuado.

#### ¿Cómo continuar aprovechando al máximo los recursos de aprendizaje?

Es esencial mantener el impulso tras terminar el curso. Aquí te dejamos algunos consejos para seguir desarrollando tus habilidades:

- **Participa en la comunidad**: Interactuar con otros estudiantes y expertos te proporcionará diferentes perspectivas y soluciones a los desafíos que encuentres.
- **Realiza los retos**: Los ejercicios prácticos son fundamentales para consolidar el conocimiento teórico y desarrollar una actitud resolutiva.
- **Evalúa tu progreso**: No olvides completar el examen para evaluar tu comprensión del tema y detectar áreas de mejora.
- **Deja una reseña**: Si disfrutaste del curso, compartir tus opiniones no solo ayuda a otros estudiantes a tomar decisiones informadas sino que también respalda a los instructores y la plataforma.

Recuerda que la clave para dominar el machine learning es una combinación de estudio constante y práctica, así que sigue explorando, experimentando y aprendiendo. ¡Nos vemos en el siguiente curso!