# Curso de Fundamentos de Machine Learning

## Tipos de modelos de machine learning para analizar equipos deportivos

### 🧠 1. **Modelos Supervisados**

Se usan cuando tienes datos **etiquetados** (es decir, con resultados conocidos).

### 📊 a) **Clasificación**

Predicen categorías discretas (ganar, perder, empatar, etc.)

**Ejemplos:**

* Predecir si un equipo ganará el próximo partido.
* Clasificar si un jugador está en forma o lesionado.

**Modelos comunes:**

* Regresión logística
* Árboles de decisión
* Random Forest
* SVM (Máquinas de vectores soporte)
* Redes neuronales

### 📈 b) **Regresión**

Predicen un **valor numérico continuo**.

**Ejemplos:**

* Predecir cuántos goles marcará un equipo.
* Estimar el rendimiento de un jugador.

**Modelos comunes:**

* Regresión lineal / ridge / lasso
* XGBoost
* Redes neuronales

### 🧠 2. **Modelos No Supervisados**

Se usan cuando **no hay etiquetas** conocidas.

### 🔍 a) **Clustering (agrupamiento)**

Agrupa datos similares automáticamente.

**Ejemplos:**

* Agrupar jugadores según estilo de juego.
* Detectar patrones en los rivales (equipos ofensivos vs. defensivos).

**Modelos comunes:**

* K-means
* DBSCAN
* Modelos de mezcla gaussiana

### 🧮 b) **Análisis de Componentes Principales (PCA)**

Reduce dimensiones para visualizar y analizar mejor los datos complejos (como estadísticas por partido).

### 🧠 3. **Modelos de Series de Tiempo**

Usados para **analizar datos secuenciales**, como resultados por fecha o evolución de un jugador.

**Ejemplos:**

* Predecir la evolución de un equipo en la tabla.
* Estimar el rendimiento físico de un atleta a lo largo del tiempo.

**Modelos comunes:**

* ARIMA
* LSTM (redes neuronales recurrentes)
* Prophet (de Facebook)
* Exponential Smoothing

### 🧠 4. **Aprendizaje por Refuerzo (Reinforcement Learning)**

Útil para **estrategia y toma de decisiones** en tiempo real o simulaciones.

**Ejemplos:**

* Optimizar tácticas de equipo (con simuladores).
* Decidir cambios de jugadores en tiempo real.
* Analizar trayectorias óptimas en deportes como fútbol, básquet, etc.

### ⚽ Aplicaciones Concretas en Deportes

| Área                     | Modelo típico                | Ejemplo práctico                                |
| ------------------------ | ---------------------------- | ----------------------------------------------- |
| Predicción de resultados | Clasificación / regresión    | ¿Ganará el partido el equipo A?                 |
| Evaluación de jugadores  | Clustering / PCA             | ¿Qué estilo de juego tiene este jugador?        |
| Análisis de desempeño    | Series de tiempo / regresión | ¿Cómo evolucionó el rendimiento este mes?       |
| Scouting / reclutamiento | Clasificación                | ¿Encajará un jugador en cierto perfil táctico?  |
| Estrategia táctica       | Aprendizaje por refuerzo     | ¿Cuál es la mejor jugada en una situación dada? |

### Resumen

¿Sabías que el *machine learning* permite mejorar significativamente el rendimiento deportivo de un equipo? En el caso del equipo Cebollitas, que perdió cuatro de sus últimos cinco partidos, la incorporación de datos como estadísticas de jugadores, registros de entrenamientos y videos de partidos podría marcar una diferencia importante. Descubramos juntos cómo los modelos de machine learning pueden ser la clave para optimizar resultados deportivos.

#### ¿Qué modelos de machine learning son útiles en el deporte?

El aprendizaje automático presenta diversas modalidades útiles en contextos deportivos. A continuación, repasamos las principales que podrían aplicarse eficientemente al equipo Cebollitas:

#### ¿Qué son los modelos supervisados en machine learning?

Estos modelos aprenden mediante ejemplos específicos, diferenciando escenarios positivos y negativos. Aplicado al fútbol, podrían analizar:

- Factores determinantes en los resultados de partidos específicos.
- Variaciones que inciden en victorias o derrotas.
- Patrones recurrentes asociados con bajos rendimientos.

De esta forma, facilitan predecir resultados futuros basados en condiciones anteriores.

### ¿Cómo funcionan los modelos no supervisados?

Por otra parte, los modelos no supervisados analizan datos sin valoraciones previas. Algunas aplicaciones prácticas son:

- Identificar grupos de jugadores con hábitos de entrenamiento similares.
- Reconocer atletas que presentan mayor riesgo de lesiones.
- Descubrir comportamientos y tendencias internas de los jugadores.

Estos hallazgos permiten mejorar decisiones técnicas y estratégicas del equipo.

#### ¿Qué ventajas ofrecen los modelos de refuerzo?

Los modelos por refuerzo mejoran mediante prueba y error obteniendo recompensas. Esto es especialmente útil en:

- La simulación de partidos para diseñar tácticas efectivas.
- Recomendación de jugadas específicas adaptadas al rival.
- Optimización de rutinas y entrenamientos para maximizar el rendimiento.

#### ¿Qué herramientas se utilizan comúnmente para aplicar machine learning en el deporte?

Existen diversas herramientas tecnológicas que facilitan la aplicación del aprendizaje automático en la actividad física y deportes, por ejemplo:

- Scikit learn, útil para algoritmos predictivos.
- TensorFlow y PyTorch, que se emplean tanto en análisis predictivo como en aprendizaje por refuerzo.
- Python, lenguaje comúnmente empleado debido a su versatilidad e integración con estas herramientas.

Estas tecnologías permiten aplicar directamente modelos al contexto particular del equipo y medir su impacto en rendimiento.

#### ¿Por qué elegir correctamente un modelo de *machine learning* importa?

Más que conocer múltiples algoritmos de manera abstracta, lo realmente importante radica en saber elegir el modelo adecuado según las necesidades específicas del equipo. Implementando estas estrategias, Cebollitas podrá:

- Analizar datos reales del equipo.
- Medir mejoras en tiempo real.
- Tomar decisiones basadas en evidencia estadística.

¿Te gustaría formar parte de este reto y convertirte en el próximo analista de datos o ingeniero de machine learning para Cebollitas? ¡Comparte tus ideas en comentarios y comencemos el partido!

**Lecturas recomendadas**

[scikit-learn: machine learning in Python — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/)

## Modelos supervisados de machine learning para análisis deportivo

### 🧠 ¿Qué es un modelo supervisado?

Un modelo supervisado **aprende a partir de datos con etiquetas conocidas**, es decir, ejemplos donde se conoce el resultado (ganó/perdió, rendimiento numérico, etc.).

👉 Se divide en dos tipos principales:

* **Clasificación** (salida categórica)
* **Regresión** (salida numérica continua)

### ⚽ Aplicaciones deportivas comunes

| Objetivo                                | Tipo de modelo supervisado |
| --------------------------------------- | -------------------------- |
| Predecir si un equipo ganará            | Clasificación              |
| Clasificar lesiones como leves o graves | Clasificación              |
| Estimar goles, puntos o asistencias     | Regresión                  |
| Predecir tiempo de recuperación         | Regresión                  |
| Clasificar tipo de jugada (pase, tiro)  | Clasificación              |

### 🔍 1. **Modelos de Clasificación**

### 📌 ¿Cuándo usar?

Cuando el resultado pertenece a **clases discretas**: ganar/perder, bajo/medio/alto, A/B/C...

### 📊 Modelos comunes:

* **Regresión logística**: simple y eficaz para 2 clases.
* **K-Nearest Neighbors (KNN)**: asigna la clase más frecuente entre vecinos cercanos.
* **Árboles de decisión y Random Forest**: modelos interpretables y robustos.
* **Support Vector Machines (SVM)**: eficaz con datos complejos y pocas dimensiones.
* **Redes neuronales**: útiles si tienes muchos datos y no linealidades complejas.

### 🏟 Ejemplo:

**¿Ganará el equipo A contra el B?**

Datos de entrada:

* posesión
* tiros a puerta
* faltas
* goles previos

Etiqueta:

* 1 = ganó, 0 = no ganó

### 📏 2. **Modelos de Regresión**

### 📌 ¿Cuándo usar?

Cuando el resultado es un **número real**: goles, puntuación, tiempo, distancia, etc.

### 📊 Modelos comunes:

* **Regresión lineal**: para relaciones simples.
* **Ridge y Lasso**: versiones regularizadas.
* **Árboles de regresión / Random Forest Regressor**
* **XGBoost**: potente y eficiente para predicción.
* **Redes neuronales**: si hay muchas variables o relaciones no lineales.

### 🏟 Ejemplo:

**¿Cuántos goles marcará el equipo X en el próximo partido?**

Datos de entrada:

* promedio de goles por partido
* defensa del rival
* condición de local o visitante

Etiqueta:

* Número de goles (0, 1, 2, …)

### 🧪 Mini ejemplo en Python

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Datos simulados
data = pd.DataFrame({
    'posesión': [52, 47, 60, 55],
    'tiros': [8, 5, 11, 7],
    'local': [1, 0, 1, 0],
    'gana': [1, 0, 1, 0]  # etiqueta
})

X = data[['posesión', 'tiros', 'local']]
y = data['gana']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

predicciones = clf.predict(X_test)
print("Precisión:", accuracy_score(y_test, predicciones))
```

### 🧠 ¿Cómo elegir el mejor modelo?

Depende de:

* Tamaño y calidad de los datos.
* Tipo de problema (clasificación vs regresión).
* ¿Quieres explicabilidad o solo precisión?
* ¿Necesitas que funcione en tiempo real?

### Resumen

¿Te imaginas poder anticiparte al resultado de un partido antes siquiera de jugarlo? Esta idea, más cercana de lo que crees, es posible mediante los modelos supervisados de inteligencia artificial. Estos modelos analizan resultados pasados, como los del equipo Cebollitas, que ha perdido más del 60% de sus partidos como visitante, para prever futuros resultados con sorprendente precisión. Pero, ¿cómo funcionan realmente estos modelos y cuál es su aplicación concreta en el análisis deportivo?

#### ¿Qué datos usan los modelos supervisados para predecir partidos?

Para realizar una predicción efectiva, estos modelos necesitan ejemplos claros de datos etiquetados. Específicamente, se analizan detalles precisos de partidos disputados anteriormente, como:

- Cantidad de goles anotados.
- Posición y desempeño del equipo.
- Número de tiros o remates al arco.
- Precisión en los tiros, como los de Tara Álvarez.
- Pases completados, como los que realiza Carol McClain.

Con esta información valiosa, los modelos "aprenden" patrones consistentes que permiten anticipar con mayor exactitud futuros resultados deportivos.

##### ¿Cuáles son los modelos supervisados más utilizados?
#### ¿Qué es y cómo funciona la regresión lineal?

La regresión lineal ofrece un cálculo simple, pero efectivo, para anticipar valores numéricos. Por ejemplo, si se conoce el número de remates al arco del equipo, este método permitirá estimar cuántos goles marcarán con base en dichos datos, proporcionando una relación gráfica fácil de interpretar.

#### ¿Qué diferencia hay con la regresión logística?

A diferencia de la lineal, la regresión logística no predice directamente cifras específicas, sino una probabilidad. Así, podrías saber que existe un 80% de probabilidad de ganar el próximo partido. Aunque se denomina regresión, su papel real es realizar una clasificación clara de posibilidades.

#### ¿Por qué usar árboles de decisión?

¿Quieres ver cómo tomar decisiones al estilo de un director técnico? El árbol de decisión opera dividiendo criterios claros para saber cuál es la siguiente mejor jugada. Por ejemplo, si tienes más del 60% de posesión y más de 10 disparos al arco, irás directamente al ataque. Aun así, es crucial ser cuidadoso, porque pueden memorizar en vez de generalizar.

#### ¿Qué ventajas aporta el random forest?

Para evitar la tendencia del árbol de decisión a memorizar, el random forest emplea múltiples árboles, donde cada uno vota por un resultado probable. Esto da como resultado más robustez, mayor precisión y menor tasa de errores, haciendo este método muy confiable.

#### ¿Dónde destacan las máquinas de soporte vectorial (SVM)?

Las SVM son herramientas excepcionales para trazar límites de clasificación precisos en un conjunto de datos, diferenciando claramente partidos ganados y perdidos en relación a características específicas como tiros al arco o posesión del balón.

#### ¿Cuál es el rol de las redes neuronales?

Cuando los datos se vuelven especialmente complejos, entran en juego las redes neuronales. Estas técnicas avanzadas encuentran combinaciones ocultas y patrones no lineales, ideales para detectar jugadas imperceptibles a simple vista, aunque requieren grandes cantidades de información y una alta potencia de computación.

#### ¿Qué necesitan estos modelos para ofrecer buenos resultados?

La clave de un modelo supervisado exitoso radica en la calidad de sus datos etiquetados. Como analistas deportivos de alto nivel, nuestro trabajo consiste precisamente en proporcionar estos datos cuidadosamente seleccionados, fundamentales para que los modelos aprendan correctamente y puedan anticiparse a resultados precisos.

Ahora que conoces cómo funcionan estos modelos supervisados, cuéntanos, ¿cuál elegirías para mejorar el rendimiento del equipo Cebollitas en su próximo partido? ¡Comparte tus comentarios!

**Lecturas recomendadas**

[1. Supervised learning — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/supervised_learning.html)

[1.1. Linear Models — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares%20https://towardsdatascience.com/understanding-linear-regression-output-in-r-7a9cbda948b3/)

## Modelos no supervisados para análisis futbolístico

### 🔷 ¿Qué es un modelo no supervisado?

Es un tipo de algoritmo que **aprende de los datos sin conocer las respuestas correctas**. Su objetivo es:

* Agrupar observaciones similares.
* Reducir la complejidad de los datos.
* Encontrar patrones o estructuras internas.

### ⚽ Aplicaciones en análisis futbolístico

| Aplicación                                    | Técnica recomendada                           |
| --------------------------------------------- | --------------------------------------------- |
| Agrupar jugadores por estilo de juego         | **Clustering (K-Means, DBSCAN)**              |
| Detectar formaciones tácticas automáticamente | **Clustering o reducción de dimensión**       |
| Reducir variables redundantes en estadísticas | **PCA (Análisis de Componentes Principales)** |
| Análisis de scouting (segmentar talento)      | **Clustering + análisis de distancias**       |
| Análisis posicional basado en tracking de GPS | **Modelos de densidad, GMM**                  |
### 🔹 1. Clustering (Agrupamiento)

### ✔ ¿Qué hace?

Agrupa jugadores, partidos o jugadas **similares** entre sí, sin que tú definas los grupos previamente.

### 🧠 Algoritmos populares:

* **K-Means**: divide datos en K grupos definidos por distancia.
* **DBSCAN**: detecta grupos de puntos densos sin definir K.
* **Gaussian Mixture Models (GMM)**: agrupa por distribuciones probabilísticas.

### ⚽ Ejemplo en fútbol:

Agrupar jugadores según estas estadísticas:

* Pases completados
* Intercepciones
* Disparos al arco
* Minutos jugados

Así puedes descubrir roles reales: creadores, defensores puros, atacantes móviles, etc.

### 🔹 2. PCA (Análisis de Componentes Principales)

### ✔ ¿Qué hace?

Reduce **dimensiones** de un conjunto de datos manteniendo la mayor parte de la **variabilidad**.

### ⚽ En fútbol:

* Simplificar datos de rendimiento (decenas de métricas por jugador).
* Visualizar en 2D o 3D las "similitudes" entre jugadores.
* Analizar tendencias generales del equipo.

### 🔹 3. Modelos de Detección de Anomalías

### ✔ ¿Qué hace?

Detecta **comportamientos fuera de lo común** (anomalías).

### ⚽ En fútbol:

* Detectar partidos atípicos (para scouting o apuestas).
* Identificar lesiones o rendimientos inusuales.
* Señalar jugadas raras en el tracking del balón.

### 🧠 Algoritmos:

* Isolation Forest
* One-Class SVM

### 🧪 Ejemplo: Clustering de jugadores con K-Means en Python

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Datos simulados: estadísticas por jugador
data = pd.DataFrame({
    'pases': [55, 70, 65, 20, 30, 25],
    'disparos': [2, 1, 3, 5, 4, 6],
    'intercepciones': [3, 2, 4, 6, 7, 5]
})

# Normalización
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Agrupar en 2 clústeres
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(data_scaled)

data['rol_estimado'] = labels
print(data)
```

### 🧠 Conclusión

| Técnica               | ¿Para qué sirve?                                      |
| --------------------- | ----------------------------------------------------- |
| **K-Means, DBSCAN**   | Agrupar jugadores, jugadas o partidos                 |
| **PCA**               | Reducir dimensión y encontrar estructura en los datos |
| **Anomaly Detection** | Detectar eventos o desempeños fuera de lo común       |

### Resumen

El fútbol actual va más allá de goles y asistencias. Cuando las estadísticas básicas no son suficientes para evaluar completamente a un jugador, recurrimos a los **modelos no supervisados**, técnicas que permiten revelar patrones ocultos en el juego sin depender exclusivamente de etiquetas tradicionales.

#### ¿Qué es el clustering y cómo ayuda a evaluar jugadores?

El *clustering* es una técnica estadística que agrupa elementos que comparten características similares. En fútbol, esto nos permite identificar jugadores que desempeñan roles específicos o tienen rendimientos parecidos sin necesidad de etiquetas previas como goles marcados o asistencias realizadas.

#### ¿Cómo funciona el algoritmo K-means?

K-means es uno de los algoritmos más populares del clustering. Funciona dividiendo los datos en "k" grupos, asegurándose de que cada jugador esté lo más cercano posible al centro de su respectivo grupo. Así, logramos identificar roles específicos como:

- Jugadores carrileros incansables.
- Motores de recuperación.
- Delanteros fantasmas.

#### ¿Qué hacer cuando los datos no tienen formas claras de agrupación?

No siempre los datos se organizan claramente. En estos casos, utilizamos algoritmos más especializados como DBSCAN, que agrupa a los jugadores según la densidad de los datos. Este método detecta grupos aunque no tengan formas geométricas explícitas, examinando cómo están distribuidos los datos en conjunto.

#### ¿En qué consiste el Clustering jerárquico y cuándo usarlo?

El clustering jerárquico organiza los datos en una estructura en árbol, conocida como dendrograma. Este método es ideal cuando analizamos jugadas ofensivas o estilos de juego sin etiquetas definidas como goles, permitiendo observar cómo jugadores o jugadas específicas se agrupan en estructuras más amplias de características similares.

#### ¿Cómo visualizar la información cuando hay muchas variables?

En ocasiones, manejar una gran cantidad de variables es abrumador. Para estos escenarios, la reducción de dimensionalidad nos brinda herramientas prácticas para resumir información destacada sin perder detalles relevantes.

#### ¿Qué es PCA (Análisis de Componentes Principales)?

PCA reduce la cantidad de variables creando nuevas dimensiones que capturan la mayoría de la información original. Es similar a observar un partido desde un dron: perdemos algunos detalles específicos, pero obtenemos una visión general del rendimiento y estilo de los jugadores.

#### ¿Hay otras opciones para visualizar datos complejos?

Sí, técnicas avanzadas como **t-SNE** o **UMAP** permiten representar datos complejos en gráficos bidimensionales o tridimensionales, revelando patrones menos obvios y facilitando la interpretación del desempeño futbolístico.

#### ¿Cómo evaluar la efectividad del clustering sin etiquetas?

Evaluar resultados sin etiquetas tradicionales como victorias o goles presenta desafíos especiales. Usamos métricas específicas para confirmar la validez del agrupamiento:

- **Inercia**: mide qué tan compactos son los grupos.
- **Coeficiente de Silhouette**: evalúa qué tan bien separados están los grupos entre sí.

Estas herramientas no ofrecen respuestas definitivas, pero son útiles para verificar que las agrupaciones tengan sentido desde el punto de vista analítico.
 
**Lecturas recomendadas**

[2. Unsupervised learning — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/unsupervised_learning.html)

## Configuración de Python y Jupyter para análisis deportivo

### ⚙️ 1. Instalar Python

### Opción recomendada: **Anaconda**

Anaconda incluye Python + Jupyter + librerías para ciencia de datos.

🔗 Descarga desde: [https://www.anaconda.com/download](https://www.anaconda.com/download)

### Alternativa: **Python + pip**

* Instala Python desde: [https://www.python.org/downloads](https://www.python.org/downloads/)
* Luego instala Jupyter y bibliotecas:

```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```

### 📓 2. Abrir Jupyter Notebook

Una vez instalado:

```bash
jupyter notebook
```

Esto abrirá tu navegador con un entorno interactivo donde puedes escribir código, visualizar gráficos y explorar datos.

### 📦 3. Bibliotecas esenciales para análisis deportivo

| Biblioteca     | Uso principal                                 |
| -------------- | --------------------------------------------- |
| `pandas`       | Manipulación de datos (CSV, Excel, JSON...)   |
| `numpy`        | Cálculo numérico y álgebra lineal             |
| `matplotlib`   | Gráficos básicos (líneas, barras, dispersión) |
| `seaborn`      | Gráficos estadísticos más elegantes           |
| `scikit-learn` | Machine learning (modelos supervisados y no)  |
| `statsmodels`  | Modelado estadístico y regresión              |
| `plotly`       | Gráficos interactivos (opcional)              |
| `xgboost`      | Modelos predictivos potentes (opcional)       |

Instalación:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels plotly xgboost
```

### 🏟️ 4. Bibliotecas opcionales para deportes

* [`football-data`](https://pypi.org/project/football-data-api/): para acceder a APIs de fútbol.
* [`fifa-api`](https://pypi.org/project/fut/): acceso a estadísticas de FIFA.
* [`mplsoccer`](https://mplsoccer.readthedocs.io/): gráficos avanzados tipo "pizza charts", radar y análisis de fútbol.

```bash
pip install mplsoccer
```

### 🧪 5. Ejemplo básico de análisis deportivo

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar estadísticas simuladas de jugadores
df = pd.DataFrame({
    'Jugador': ['A', 'B', 'C', 'D'],
    'Goles': [10, 5, 7, 2],
    'Asistencias': [3, 7, 2, 4],
    'Minutos': [900, 850, 1000, 700]
})

# Relación goles-minutos
df['Goles por minuto'] = df['Goles'] / df['Minutos']

# Gráfico
sns.barplot(x='Jugador', y='Goles por minuto', data=df)
plt.title('Goles por minuto jugado')
plt.show()
```

### 🧠 Consejos adicionales

✅ Organiza tus notebooks por proyectos: `datos/`, `modelos/`, `gráficas/`

✅ Usa `Google Colab` si no deseas instalar nada localmente.

✅ Guarda tus datasets en CSV y cárgalos con `pandas.read_csv("archivo.csv")`.

✅ Si usas datos de páginas como [FBref.com](https://fbref.com/) o [Understat](https://understat.com/), puedes integrarlos fácilmente.

### Resumen

La frustración en el fútbol es común cuando los resultados no reflejan claramente el rendimiento del equipo. Esta situación ocurre incluso cuando se tienen más posesión y pases completados. Para solventar este inconveniente, utilizaremos Python y técnicas de machine learning supervisado que nos permitirán analizar datos precisos y entender mejor lo que pasa dentro del campo.

#### ¿Por qué usar Python para análisis deportivo?

Python es ideal debido a que es simple, poderoso y cuenta con las herramientas adecuadas para el análisis y procesamiento de datos deportivos. Para facilitar nuestro trabajo, usaremos Jupyter Notebook, que permite escribir código, hacer justificaciones y visualizar gráficos cómodamente en un mismo entorno.

#### ¿Cómo preparar tu entorno para análisis?

Es importante tener todo listo antes de comenzar a analizar resultados deportivos con datos. Para ello, generaremos un ambiente virtual completo instalando Python, Jupyter Notebook e incluirá algunas librerías esenciales.

- Instalar Python y Jupyter Notebook.
- Crear un archivo notebook con extensión punto IPYNB.
- Nombrar tu archivo notebook como “cebollitas día uno”.

Este ambiente será tu campo digital para entrenar modelos analíticos, probar nuevas ideas y evaluar resultados.

#### ¿Cuál será tu primera línea de código?

Tu introducción a Python inicia con algo sencillo, pero significativo, como escribir tu primera línea de código. Puedes comenzar con un breve mensaje de bienvenida:

`print("Bienvenidos al Cebollita FC")`

Este sencillo ejercicio representa el pitazo inicial para futuras predicciones estadísticas basadas en datos reales. A partir del próximo entrenamiento, se comenzará a trabajar con información auténtica del club, comprendiendo estadísticas, desempeño y descubriendo patrones útiles para anticipar resultados.

Este nuevo enfoque con *machine learning* y Python permitirá que analizar el fútbol sea mucho más eficiente, brindándote las herramientas necesarias para comprender mejor cada situación del partido.

**Archivos de la clase**

[partidos-cebollitas.csv](https://static.platzi.com/media/public/uploads/partidos_cebollitas_fe82a1a4-e109-41b1-8b78-d9b4341dacaf.csv "partidos-cebollitas.csv")

## Limpieza de datos deportivos con Pandas en Python

### 🧼 ¿Qué es la limpieza de datos?

La limpieza de datos incluye:

1. Cargar datos correctamente.
2. Identificar y tratar **valores nulos**.
3. Corregir **tipos de datos**.
4. Eliminar **duplicados** o errores.
5. Renombrar columnas y estandarizar nombres.
6. Filtrar registros no válidos o inconsistentes.
7. Crear columnas útiles (goles por minuto, etc.).

### 🏟️ Supongamos que tienes este dataset (CSV)

```csv
jugador, goles, minutos_jugados, equipo, fecha
Messi, 3, 90, Inter Miami, 2023-09-12
Mbappe, , 85, PSG, 2023-09-12
Messi, 3, 90, Inter Miami, 2023-09-12
Lewandowski, 1, 78, FC Barcelona, 2023-09-12
Falcao, 0, , Rayo Vallecano, 2023-09-12
```

### 🐼 Paso a paso con Pandas

```python
import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('estadisticas.csv')

# Mostrar las primeras filas
print(df.head())
```

### 🔍 1. Verificar valores nulos

```python
print(df.isnull().sum())  # ¿Dónde hay valores faltantes?
```

➡️ Solución:

```python
df['goles'].fillna(0, inplace=True)  # Rellenar con 0
df['minutos_jugados'].fillna(df['minutos_jugados'].mean(), inplace=True)  # Promedio
```

### 📋 2. Eliminar duplicados

```python
df = df.drop_duplicates()
```

### 🔢 3. Corregir tipos de datos

```python
df['goles'] = df['goles'].astype(int)
df['minutos_jugados'] = df['minutos_jugados'].astype(int)
df['fecha'] = pd.to_datetime(df['fecha'])
```

### ✏️ 4. Renombrar columnas (opcional)

```python
df.rename(columns={'minutos_jugados': 'min_jugados'}, inplace=True)
```

### ➗ 5. Crear columnas nuevas (ej. eficiencia)

```python
df['goles_por_minuto'] = df['goles'] / df['min_jugados']
```

### 🧹 6. Filtrar registros no válidos

```python
df = df[df['min_jugados'] > 0]  # Eliminar jugadores sin minutos jugados
```

### 📊 Resultado limpio

```python
print(df)
```

### 🧠 Consejos extra

| Tarea                        | Función útil de Pandas                 |
| ---------------------------- | -------------------------------------- |
| Ver resumen general          | `df.info()`                            |
| Estadísticas básicas         | `df.describe()`                        |
| Ver valores únicos por campo | `df['equipo'].unique()`                |
| Filtrar por condición        | `df[df['equipo'] == 'PSG']`            |
| Exportar CSV limpio          | `df.to_csv('limpio.csv', index=False)` |

### Resumen

Analizar y limpiar los datos es clave antes de utilizar modelos predictivos en deportes. Al usar Python con la librería pandas, es posible manipular fácilmente grandes volúmenes de información, pasando de conjuntos de datos confusos y desorganizados a información clara y útil que pueda ayudar a prever futuros resultados del club.

#### ¿Qué herramientas se necesitan para la limpieza de datos deportivos?

Usamos principalmente Python y la librería pandas. Dentro del entorno de desarrollo de Visual Studio Code, importamos y utilizamos:

- pandas (`pd`), una herramienta robusta para manipulación y análisis de datos.
- Notebooks en Python para facilitar el análisis interactivo de cada etapa del proceso de limpieza.

#### ¿Cómo cargar y visualizar inicialmente los datos deportivos?

Los datos se cargan mediante la función `pd.read_csv()`, que permite acceder directamente al archivo CSV. La función `head()` muestra las primeras filas de nuestros datos, proporcionando una vista rápida y fundamental del estado inicial del dataset. Esta visión previa es semejante a observar los primeros minutos de juego para seleccionar la estrategia adecuada.

#### ¿Cuáles son los pasos fundamentales para preparar los datos?

La preparación de datos requiere una serie de técnicas específicas en Python:

#### ¿Cómo evaluar la calidad general de los datos del equipo?

La función info() es clave para obtener una visión general de los datos:

- Identifica columnas disponibles.
- Detecta valores ausentes o errores.
- Indica tipos de datos presentes (integer, objetos, etc.).
- Presenta consumo de memoria, útil para futuras optimizaciones.

#### ¿Cómo lidiar con datos faltantes en los registros deportivos?

Para los valores nulos, particularmente en columnas críticas como los goles anotados, se utiliza:

- `isnull().sum()` para identificar faltantes.
- Se rellenan estos valores con el promedio en lugar de eliminarlos, manteniendo la integridad del dataset sin introducir registros falsos.

#### ¿Qué es el One-Hot Encoding y cómo se aplica en equipos deportivos?

Se aplica a variables categóricas como los nombres de equipos usando:

- `pd.get_dummies()` transforma estas categorías en columnas binarias de ceros y unos.
- Facilitando que los algoritmos comprendan y procesen estos datos numéricamente.

#### ¿Cómo gestionar duplicados y evitar su impacto en los resultados?

Con la función `duplicate()`, eliminamos registros idénticos que podrían sesgar el entrenamiento de los modelos predictivos, asegurando aprendizaje claro y único en cada caso.

#### ¿Por qué y cómo se ajustan los formatos de fecha?

Las fechas inconsistentes o mal formateadas se ajustan usando `pd.to_datetime()`. Esta transformación permite realizar análisis temporales útiles para detectar patrones como rachas positivas o negativas en ciertas épocas del año.

#### ¿Cómo evaluar finalmente el estado de nuestro dataset limpio?

Luego de completar estas tareas, se recomienda verificar nuevamente con `info()` y `head()` el estado final de los datos. Además, confirmar con funciones como `shape` el tamaño y la estructura en filas y columnas.

- Confirmar características generales del dataset.
- Asegurar ausencia de nulos o duplicados.
- Validar columnas y registros disponibles.

Este método provee claridad total antes de avanzar a modelar resultados con técnicas más avanzadas.

¿Qué otras técnicas recomiendas para preparar datos antes de predecir resultados en deportes? ¡Comparte tu opinión en los comentarios!

**Archivos de la clase**

[partidos-cebollitas.csv](https://static.platzi.com/media/public/uploads/partidos_cebollitas_9eada58c-fb57-4224-a3f5-6d9efc881c2e.csv "partidos-cebollitas.csv")

**Lecturas recomendadas**

[pandas - Python Data Analysis Library](https://pandas.pydata.org/ "pandas - Python Data Analysis Library")

[machine-learning/02_preparacion.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/02_preparacion.ipynb "machine-learning/02_preparacion.ipynb at main · platzi/machine-learning · GitHub")

[machine-learning/02_preparacion.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/02_preparacion.ipynb "machine-learning/02_preparacion.ipynb at main · platzi/machine-learning · GitHub")

## Análisis de rendimiento deportivo con estadística descriptiva

### ⚽ ¿Qué es la estadística descriptiva en deportes?

Es el **análisis numérico y gráfico** que describe y resume un conjunto de datos deportivos sin hacer inferencias más allá de la muestra.

### 📊 Ejemplos de variables analizadas

| Variable              | Tipo         | Ejemplo                         |
| --------------------- | ------------ | ------------------------------- |
| Goles                 | Cuantitativa | 0, 1, 2, 3…                     |
| Minutos jugados       | Cuantitativa | 90, 85, 75…                     |
| Posición en el campo  | Cualitativa  | Delantero, Defensa, Mediocampo… |
| Resultado del partido | Cualitativa  | Victoria, Derrota, Empate       |

### 🧪 Medidas comunes

### 1. **Tendencia central**

* **Media (promedio)**: Valor medio de rendimiento.
* **Mediana**: Valor central (útil con datos sesgados).
* **Moda**: Valor más frecuente (por ejemplo, goles más comunes).

### 2. **Dispersión**

* **Rango**: Diferencia entre el valor más alto y el más bajo.
* **Desviación estándar**: Qué tanto varían los datos respecto a la media.
* **Varianza**: Medida cuadrática de dispersión.

### 3. **Resumen estadístico rápido**

En Python:

```python
df.describe()
```

### 🧱 Ejemplo práctico con Pandas (fútbol)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset simulado
df = pd.DataFrame({
    'Jugador': ['A', 'B', 'C', 'D', 'E'],
    'Goles': [10, 5, 8, 0, 3],
    'Asistencias': [4, 7, 2, 0, 1],
    'Minutos': [900, 800, 950, 400, 600]
})

# Estadísticas descriptivas
print(df.describe())

# Agregar columna de eficiencia
df['Goles por 90 min'] = df['Goles'] / df['Minutos'] * 90

# Gráfico: comparación de goles
sns.barplot(x='Jugador', y='Goles', data=df)
plt.title('Goles por Jugador')
plt.show()

# Gráfico: eficiencia de gol
sns.barplot(x='Jugador', y='Goles por 90 min', data=df)
plt.title('Goles por 90 Minutos')
plt.show()
```

### 📈 ¿Qué puedes hacer con estos análisis?

| Aplicación                        | Ejemplo concreto                           |
| --------------------------------- | ------------------------------------------ |
| Comparar jugadores                | ¿Quién tiene mejor promedio de goles?      |
| Evaluar consistencia              | ¿Quién es más regular en su rendimiento?   |
| Visualizar rendimiento por equipo | Barras, cajas, dispersión, radar, etc.     |
| Detectar valores atípicos         | ¿Hay jugadores con estadísticas inusuales? |

### 🧠 Conclusión

La estadística descriptiva permite:

* Evaluar el **nivel y consistencia de desempeño**.
* Facilita la **toma de decisiones deportivas** (alineaciones, fichajes).
* Es base para aplicar modelos predictivos posteriormente.

### Resumen

¿Te has preguntado por qué algunos partidos se ganan con facilidad y otros se complican más de lo normal? La respuesta puede encontrarse en los datos, específicamente en cómo se analizan utilizando métodos de estadística descriptiva y visualización gráfica. Antes de predecir los resultados futuros, es esencial comprender en detalle los resultados del pasado.

#### ¿Qué herramientas necesitamos para analizar los partidos?

Para entender qué está pasando en el campo, necesitamos algunas herramientas técnicas fundamentales:

- **Pandas**: manipulación y preparación inicial de nuestros datos.
- **Matplotlib y Seaborn**: para crear visualizaciones intuitivas que faciliten la interpretación de información compleja.

Estas herramientas nos permiten extraer información crucial, como promedio de goles, desviaciones estándar, máximo y mínimo de goles marcados.

#### ¿Cómo identificar tendencias y patrones clave?

Las siguientes preguntas nos guían en el análisis del rendimiento:

#### ¿Es mejor nuestro desempeño como local o como visitante?

Al observar los promedios encontramos algo inesperado: el equipo tiene en promedio **2.2 goles como local** y **2.6 goles como visitante**, indicando un rendimiento sólido incluso fuera de casa. Esto sugiere que nuestro rendimiento no depende únicamente del apoyo local.

#### ¿Qué revelan los histogramas sobre nuestros goles?

Los histogramas, realizados con Seaborn, muestran frecuencias reales:

- Localmente, solemos marcar regularmente 2 a 4 goles.
- De visitante, aunque hay partidos con cero goles, sorprendentemente también hay varios encuentros en los cuales llegamos a 3 o 5 goles anotados.

Estos gráficos clarifican si nuestro rendimiento es consistente o variable partido a partido.

#### ¿Cómo detectar resultados inusuales con Boxplots?

Los boxplots destacan fácilmente valores extremos:

- Señalan claramente los partidos excepcionales, tales como goleadas o resultados adversos.
- Revelan que es frecuente hacer entre 1 y 4 goles cuando jugamos en casa, con picos ocasionales de 5 goles.

Aunque la mayoría de resultados están concentrados en un rango limitado, estos valores ayudan a entender nuestras capacidades en escenarios extraordinarios.

#### ¿Tener más posesión siempre significa más goles?

Probamos esta hipótesis usando un gráfico de dispersión:

- A pesar de una posesión del 45%, la cantidad de goles anotados varía considerablemente.
- Diversidad en resultados, desde partidos sin goles hasta encuentros con cinco goles, indica que la posesión sola no determina nuestro rendimiento goleador.

Esto establece una relación compleja entre la posesión del balón y la efectividad goleadora del equipo.

#### ¿Qué revela el mapa de calor sobre correlaciones importantes?

Utilizamos un mapa de calor para detectar conexiones ocultas entre diferentes acciones durante el partido. La correlación más alta registrada entre variables, como goles locales y posesión, es relativamente baja (0.17), indicando que las relaciones entre estas métricas no son extremadamente fuertes.

No obstante, aun siendo significativas, su influencia es moderada, lo cual es útil para modelar predicciones futuras.

¿Qué correlaciones crees que debemos priorizar al momento de entrenar nuestros futuros modelos predictivos? Danos tu opinión en los comentarios sobre qué variables te parecen más determinantes para el rendimiento del equipo.

**Archivos de la clase**

[partidos-cebollitas.csv](https://static.platzi.com/media/public/uploads/partidos_cebollitas_e88b84b1-059b-4559-b1b5-78838f9f7ccc.csv "partidos-cebollitas.csv")

**Lecturas recomendadas**

[Tutorials — Matplotlib 3.10.3 documentation](https://matplotlib.org/stable/tutorials/index "Tutorials — Matplotlib 3.10.3 documentation")

[User guide and tutorial — seaborn 0.13.2 documentation](https://seaborn.pydata.org/tutorial "User guide and tutorial — seaborn 0.13.2 documentation")

[machine-learning/03_exploracion_datos_cebollitas.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/03_exploracion_datos_cebollitas.ipynb "machine-learning/03_exploracion_datos_cebollitas.ipynb at main · platzi/machine-learning · GitHub")

## Normalización y estandarización de datos para modelos predictivos

### 📏 ¿Por qué normalizar o estandarizar?

Porque:

* Muchas técnicas de ML **son sensibles a la escala de los datos**.
* Si tus variables tienen **unidades distintas** (por ejemplo, “goles” y “minutos”), una puede dominar a la otra si no las escalas.
* Mejora la **velocidad de entrenamiento** y la **precisión del modelo**.

### 🔄 Diferencia entre normalización y estandarización

| Técnica                       | Qué hace                                                                           | Rango típico       |
| ----------------------------- | ---------------------------------------------------------------------------------- | ------------------ |
| **Normalización** (Min-Max)   | Escala los datos a un **rango fijo**, normalmente entre **0 y 1**                  | \[0, 1] o \[-1, 1] |
| **Estandarización** (Z-score) | Convierte los datos a una distribución con **media = 0 y desviación estándar = 1** | Media = 0, Std = 1 |

### ⚽ Ejemplo en análisis deportivo

Supón que tienes:

| Jugador | Goles | Minutos | Pases Completos |
| ------- | ----- | ------- | --------------- |
| A       | 5     | 900     | 300             |
| B       | 2     | 750     | 250             |
| C       | 7     | 1100    | 500             |

👉 Estos valores están en **escalas diferentes** → necesitas escalarlos.

### 🧪 En Python con `sklearn`

### 🔹 Normalización (Min-Max Scaling)

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Datos simulados
df = pd.DataFrame({
    'Goles': [5, 2, 7],
    'Minutos': [900, 750, 1100],
    'Pases': [300, 250, 500]
})

scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Normalizados:\n", df_norm)
```

### 🔸 Estandarización (Z-score)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Estandarizados:\n", df_std)
```

### 🧠 ¿Cuándo usar cada uno?

| Si vas a usar...                   | Recomendación                                        |
| ---------------------------------- | ---------------------------------------------------- |
| KNN, SVM, Redes Neuronales, PCA    | ✅ Escala tus datos (normalización o estandarización) |
| Árboles de decisión, Random Forest | ❌ No es obligatorio escalar                          |
| Visualización (radar, scatter...)  | ✅ Normalizar para comparación clara                  |

### 🎯 Conclusión

* **Normalizar o estandarizar** mejora el rendimiento del modelo.
* Elige el método según el algoritmo que uses.
* Usa `sklearn.preprocessing` para escalar fácilmente.

### Resumen

Para asegurar predicciones precisas en modelos de aprendizaje automático, es clave **nivelar la escala de las variables**. Cuando los datos presentan rangos muy diversos, las variables con números mayores pueden afectar injustamente los resultados. Aquí aprenderemos cómo corregir esto con dos técnicas fundamentales: **Min-Max Scaling** para normalización y Standard Scaler para estandarización.

#### ¿Por qué es necesario escalar los datos antes de entrenar modelos?

Los modelos de aprendizaje automático pueden confundirse cuando las variables manejan escalas muy distintas. Por ejemplo, una variable como tiros al arco (valores entre 0 y 15) podría parecerle menos relevante al modelo que la posesión (del 0 al 100), simplemente por tener números más pequeños. Esto no es realista, ya que ambas podrían tener igual importancia.

#### ¿Qué ocurre al no escalar correctamente?

Cuando usamos datos sin escalar:

- El modelo asigna incorrectamente mayor importancia a magnitudes mayores.
- Se generan sesgos en las predicciones.
- Puede afectar negativamente la precisión del modelo.

Al escalar los datos, logramos una medida justa que permite al modelo aprender con mayor precisión.

#### ¿Qué diferencia hay entre Min-Max Scaling y Standard Scaler?

Existen dos métodos claves para escalar tus datos, cada uno ideal para ciertas circunstancias:

#### ¿Cuándo usar Min-Max Scaling?

El Min-Max Scaling es ideal cuando tus datos no siguen una distribución normal. Este método transforma cualquier valor numérico a un rango cerrado entre 0 y 1. Por ejemplo, si un futbolista tiene 12 tiros al arco y el máximo registrado es 15, al aplicar Min-Max Scaling obtendremos:

- Resultado escalado: 12/15 = 0,8.

Este proceso se realiza usando la herramienta MinMaxScaler de las bibliotecas de procesamiento de datos.

#### ¿En qué casos se aplica Standard Scaler?

El Standard Scaler, por otro lado, es más apropiado cuando deseas preservar la distribución original de los datos. Este método centra la información alrededor de cero, con desviación estándar de uno. Es especialmente útil con algoritmos que necesitan datos centrados como regresión lineal o PCA. Los resultados obtenidos tendrán una distribución estandarizada, eliminando cualquier sesgo por magnitud original.

#### ¿Cómo aplicar estas técnicas paso a paso?

Veamos brevemente cómo aplicar ambos métodos utilizando herramientas comunes en Python como pandas para datos y las clases MinMaxScaler y StandardScaler para escalar.

#### Aplicando Min-Max Scaling a tus datos

- Primero, crea una instancia de MinMaxScaler.
- Utiliza la función fit_transform para calcular el mínimo y máximo de cada columna y aplicar la escala a la vez.
- Guarda estos datos normalizados en columnas nuevas:

```python
from sklearn.preprocessing import MinMaxScaler
scaler_norm = MinMaxScaler()
datos_normalizados = scaler_norm.fit_transform(datos[['tiros_al_arco_local', 'tiros_al_arco_visitante']])
datos[['tiros_al_arco_local_norm', 'tiros_al_arco_visitante_norm']] = datos_normalizados
```

#### Aplicando Standard Scaler correctamente

Para estandarizar:

- Genera una instancia de StandardScaler.
- Nuevamente, usa fit_transform para calcular media y desviación estándar y aplicar la transformación.
- Guarda los resultados en nuevas columnas:

```python
from sklearn.preprocessing import StandardScaler
scaler_std = StandardScaler()
datos_estandarizados = scaler_std.fit_transform(datos[['posesion_local', 'posesion_visitante']])
datos[['posesion_local_std', 'posesion_visitante_std']] = datos_estandarizados
```

#### ¿Cómo saber si el escalado fue exitoso?

Para visualizar si tu escalado fue exitoso y efectivo, usa histogramas, una visualización que revisa cómo se distribuyen tus datos escalados. Para ello, emplea bibliotecas como Matplotlib o Seaborn:

```python
import matplotlib.pyplot as plt 
import seaborn as sns

fig, axes = plt.subplots(1, 2)
sns.histplot(datos['tiros_al_arco_local_norm'], ax=axes[0])
axes[0].set_title('Tiros al Arco Local Normalizados')

sns.histplot(datos['posesion_local_std'], ax=axes[1], color='orange')
axes[1].set_title('Posesión Local Estandarizada')

plt.show()
```

Dichas gráficas mostrarán claramente si los datos tienen una distribución adecuada y equilibrada. Cuéntanos, ¿qué inferencias puedes sacar de tu visualización?

**Lecturas recomendadas**

[7.3. Preprocessing data — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/preprocessing "7.3. Preprocessing data — scikit-learn 1.7.0 documentation")

[Importance of Feature Scaling — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance "Importance of Feature Scaling — scikit-learn 1.7.0 documentation")

[machine-learning/04_escalado_datos_cebollitas.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/04_escalado_datos_cebollitas.ipynb "machine-learning/04_escalado_datos_cebollitas.ipynb at main · platzi/machine-learning · GitHub")

## Ingeniería de características para mejorar modelos de machine learning

### 🧠 ¿Qué es la ingeniería de características?

Es el proceso de:

1. **Crear nuevas variables** a partir de datos existentes.
2. **Seleccionar las más relevantes.**
3. **Transformar o escalar datos** para que los modelos aprendan mejor.
4. **Codificar datos categóricos.**

### ⚽ Ejemplo en datos deportivos

Supón que tienes este conjunto de datos:

| Jugador     | Goles | Minutos | Pases Completos |
| ----------- | ----- | ------- | --------------- |
| Messi       | 3     | 90      | 70              |
| Lewandowski | 1     | 60      | 35              |

### 🔧 Podemos crear nuevas características como:

| Nueva variable            | Fórmula                           | Significado                             |
| ------------------------- | --------------------------------- | --------------------------------------- |
| Goles por minuto          | `Goles / Minutos`                 | Eficiencia goleadora                    |
| Pases por minuto          | `Pases / Minutos`                 | Participación en juego                  |
| Participación total       | `Goles + Asistencias`             | Impacto ofensivo                        |
| Pases acertados (%)       | `Pases / Pases intentados * 100`  | Precisión de pase (si se tiene el dato) |
| Diferencia de rendimiento | `Goles por minuto - media global` | Comparación con otros                   |

### 🧪 Ejemplo en Python

```python
import pandas as pd

# Datos básicos
df = pd.DataFrame({
    'Jugador': ['Messi', 'Lewandowski'],
    'Goles': [3, 1],
    'Minutos': [90, 60],
    'Pases': [70, 35]
})

# Ingeniería de características
df['Goles_por_minuto'] = df['Goles'] / df['Minutos']
df['Pases_por_minuto'] = df['Pases'] / df['Minutos']

print(df)
```

### 🧠 Técnicas comunes de ingeniería de características

### 📌 1. **Escalado**

* Usar `MinMaxScaler` o `StandardScaler` para igualar escalas.

### 📌 2. **Codificación de variables categóricas**

* `OneHotEncoder`, `LabelEncoder` para posiciones, equipos, etc.

### 📌 3. **Extracción de tiempo**

* Separar "fecha del partido" en "día de la semana", "mes", "temporada".

### 📌 4. **Cruces de variables**

* Multiplicar o dividir variables para encontrar relaciones (por ejemplo: **posesión × tiros al arco**).

### 📌 5. **Transformaciones estadísticas**

* Logaritmo, raíz cuadrada, z-score… para normalizar distribuciones.

### 📈 Beneficios

✅ Mejora el rendimiento del modelo.
✅ Permite modelos más simples con mejores resultados.
✅ Reduce la necesidad de redes neuronales profundas en problemas sencillos.
✅ Mejora la **interpretabilidad** de los modelos.

### 🧠 Consejo clave

> "Un modelo simple con buenas características supera a un modelo complejo con malas características."

### Resumen

La ingeniería de características, también conocida como *feature engineering*, es una herramienta fundamental en el ámbito del *machine learning*. Su objetivo principal consiste en crear nuevas variables a partir de datos ya existentes. Esto permite a los modelos identificar patrones más profundos y útiles, mejorando así notablemente su desempeño predictivo.

#### ¿Por qué es valiosa la ingeniería de características?

Esta técnica transforma datos simples en información más relevante para los modelos. Sabemos que los algoritmos no son capaces de detectar relaciones ocultas automáticamente; sin embargo, un analista puede crear variables estratégicas que aporten nuevo contexto al algoritmo:

- **Diferencia de goles**: goles del equipo local menos goles del visitante, útil para predecir si un equipo ganó, perdió o empató.
- **Ratio de tiros sobre posesión**: relaciona los disparos realizados con la posesión durante el juego, midiendo eficiencia ofensiva.

Ambas variables aportan información valiosa que no está explícitamente escrita en los datos originales.

#### ¿Qué pasos seguir para crear nuevas variables?

El proceso para implementar estas nuevas variables se desglosa en los siguientes bloques prácticos:

#### Bloque número uno: importar datos

Aquí es clave importar las bibliotecas necesarias, principalmente pandas, para tener acceso organizado y eficiente a los datos completos.

#### Bloque número dos: calcular diferencia de goles

Esta nueva columna se crea restando los goles del equipo visitante a los del local, ayudándonos a entender rápidamente el desempeño de los equipos.

#### Bloque número tres: evaluar la eficiencia ofensiva

Se crea un ratio dividiendo los tiros al arco local sobre la posesión del equipo local. Este índice mide directamente la capacidad de aprovechar la posesión del balón para generar tiros al arco.

#### Bloque número cuatro: visualizar con histogramas

Visualizar los datos es crucial. Aquí se recomienda utilizar hist plot de seaborne para observar la distribución de la diferencia de goles:

- Importar bibliotecas (seaborne y matplotlib).
- Usar hist plot evaluando gráfica y visualmente la distribución.

Esto permite una interpretación rápida sobre cómo se comporta esta nueva variable: ¿el equipo tiende más a empatar, perder o ganar?

#### Bloque número cinco: establecer correlaciones

Finalmente, se utiliza un mapa de calor (heat map) para evaluar la correlación entre variables originales y aquellas recientemente creadas. Esto es determinante para comprobar la utilidad real de estas características añadidas.

Por ejemplo, una fuerte correlación encontrada en el análisis fue entre goles locales y la diferencia de goles, indicando una relación sólida que puede mejorar los modelos predictivos.

¿Y tú qué opinas de estas nuevas variables? ¿En qué situaciones crees que podrían aportarte un mayor valor predictivo? Cuéntanos tu experiencia en los comentarios.

**Lecturas recomendadas**

[User Guide — pandas 2.3.0 documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html)

[machine-learning/05_ingenieria_caracteristicas_cebollitas.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/05_ingenieria_caracteristicas_cebollitas.ipynb "machine-learning/05_ingenieria_caracteristicas_cebollitas.ipynb at main · platzi/machine-learning · GitHub")

## Selección de variables relevantes para modelos predictivos

La **selección de variables relevantes** (también llamada **feature selection**) es una etapa crucial en machine learning porque:

* Mejora la **precisión** del modelo.
* Reduce el **overfitting**.
* Aumenta la **velocidad de entrenamiento**.
* Facilita la **interpretabilidad** del modelo.

### 🧠 ¿Qué es la selección de variables?

Es el proceso de **elegir solo las características más útiles** para predecir una variable objetivo y **descartar las irrelevantes o redundantes**.

### ⚽ Ejemplo en análisis deportivo

Imagina que tienes estas variables para predecir si un equipo ganará:

* Posesión del balón
* Número de tiros
* Goles recibidos
* Minutos jugados
* Día de la semana
* Nombre del equipo rival

👉 Algunas de estas **no ayudan** al modelo o incluso lo **confunden**. El objetivo es quedarte solo con las más relevantes, como posesión y tiros.


### 🔍 Técnicas para seleccionar variables

### 1. **Análisis de correlación**

Ideal para variables numéricas.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de correlación
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlación entre variables")
plt.show()
```

> Elimina variables muy correlacionadas entre sí (multicolinealidad) o no relacionadas con la variable objetivo.

### 2. **Método de importancia de características (feature importance)**

Usado con modelos como Random Forest, XGBoost o árboles de decisión.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

importancia = pd.Series(model.feature_importances_, index=X.columns)
importancia.sort_values(ascending=False).plot(kind='bar')
```

### 3. **Selección automática (`SelectKBest`)**

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)  # Top 5 variables
X_new = selector.fit_transform(X, y)
```

### 4. **Eliminación recursiva (`RFE`)**

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()
rfe = RFE(modelo, n_features_to_select=3)
fit = rfe.fit(X, y)

# Ver qué variables fueron seleccionadas
print(X.columns[fit.support_])
```

### ✅ Buenas prácticas

| Práctica                                  | Descripción                                                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Establece una **variable objetivo** clara | Ej: `ganó = 1, no ganó = 0`                                                                                         |
| Escala los datos si es necesario          | Mejora precisión en modelos sensibles a magnitudes                                                                  |
| Usa técnicas combinadas                   | Correlación + RFE, o Feature Importance + PCA                                                                       |
| Evita **leakage**                         | No incluyas variables que solo se conocen después del evento (ej. goles anotados si estás prediciendo el resultado) |

### 🧠 Conclusión

La selección de variables:

* **Hace tus modelos más precisos y rápidos**.
* Evita usar datos que **no aportan valor**.
* Es esencial en datasets deportivos con muchas estadísticas por partido o jugador.

### Resumen

Optimizar modelos predictivos no siempre implica utilizar más variables, sino seleccionar correctamente aquellas que realmente aportan valor a tu análisis. Este método, conocido como selección de características, ayuda a reducir ruido, simplificar modelos y prevenir sobreajustes, resultando en predicciones más acertadas.

#### ¿Qué es la selección de características en modelado predictivo?

La selección de características consiste en identificar y retener únicamente aquellas variables que presenten mayor relevancia estadística con el objetivo a predecir. Al aplicar esta práctica:

- Se simplifica el modelo, haciendo su interpretación más sencilla.
- Se eliminan variables irrelevantes que actúan como ruido.
- Mejora la capacidad de generalización y precisión del modelo.

#### ¿Cómo realizar una selección univariada con SelectKBest?

Utilizando herramientas del paquete *scikit-learn*, puedes aplicar SelectKBest para seleccionar aquellas variables con mejor rendimiento individual:

- Importa `SelectKBest` y `f_regression`.
- Usa `f_regression` para calcular la puntuación F, evaluando qué tanto influye cada variable en tu objetivo.
- Define tu matriz de características X y tu vector objetivo Y.
- Selecciona las variables con mejores puntajes individuales y visualiza en orden decreciente cuales aportan más:

```python
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=2)
selector.fit(X, y)
```

#### ¿Por qué utilizar árboles de decisión para evaluar importancia de variables?

Los árboles de decisión te permiten evaluar no solo correlaciones lineales, sino también posibles interacciones o relaciones no lineales. Siguiendo estos pasos puedes beneficiarte de esta técnica:

- Importa `DecisionTreeRegressor`.
- Instancia el modelo, define una semilla (`random_state`) para resultados reproducibles.
- Ajusta el modelo usando tus datos X e Y.
- Evalúa la importancia relativa de cada variable mediante la reducción total del error:

```python
from sklearn.tree import DecisionTreeRegressor
modelo_arbol = DecisionTreeRegressor(random_state=0)
modelo_arbol.fit(X, y)
```

#### ¿Qué aporta la visualización comparativa de técnicas?

Una comparación visual, mediante gráficas de barras con seaborn y matplotlib, aclara la diferencia en aportaciones que cada variable tiene según distintos métodos utilizados. Esto te permite validar conclusiones rápidamente y detectar variables consistentemente relevantes:

- Las técnicas utilizadas fueron claramente comparadas:
- SelectKBest analiza correlaciones lineales.
- DecisionTreeRegressor evalúa relaciones complejas, incluyendo no lineales.

Aunque ambas técnicas generaron valores específicos diferentes, hubo coherencia en variables destacadas como posesión local y ratio tiros sobre posesión.

#### ¿Realmente se necesitan todas las variables?

Este ejercicio práctico evidenció que menos variables pueden conseguir resultados equivalentes o superiores cuando están correctamente elegidas:

- **Ratio de tiros sobre posesión**.
- **Porcentaje de posesión local**.

Estas variables han demostrado ser cruciales en la predicción eficaz de partidos.

**Lecturas recomendadas**

[1.13. Feature selection — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/feature_selection.html "1.13. Feature selection — scikit-learn 1.7.0 documentation")

[machine-learning/06_seleccion_caracteristicas_cebollitas.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/06_seleccion_caracteristicas_cebollitas.ipynb "machine-learning/06_seleccion_caracteristicas_cebollitas.ipynb at main · platzi/machine-learning · GitHub")

[How to Choose a Feature Selection Method For Machine Learning](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/ "How to Choose a Feature Selection Method For Machine Learning")