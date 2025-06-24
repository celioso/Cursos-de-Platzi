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

## División de datos en entrenamiento y prueba con scikit-learn

Dividir tus datos en **entrenamiento y prueba** es una parte fundamental en cualquier proyecto de **machine learning**. Con **scikit-learn**, puedes hacerlo fácilmente usando la función `train_test_split`.

### 🧠 ¿Por qué dividir los datos?

* **Entrenamiento (train)**: se usa para ajustar (entrenar) el modelo.
* **Prueba (test)**: se usa para evaluar qué tan bien generaliza el modelo a datos nuevos.
* Evita que el modelo aprenda "de memoria" los datos (sobreajuste).

### ✅ Ejemplo en Python con `scikit-learn`

Supongamos que tienes un conjunto de datos con características `X` y etiquetas `y`:

```python
from sklearn.model_selection import train_test_split

# Supongamos que X y y ya están definidos
# X = características (variables independientes)
# y = etiqueta (variable objetivo)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🔍 Parámetros:

* `test_size=0.2`: 20% de los datos se usan para prueba, 80% para entrenamiento.
* `random_state=42`: asegura que la división sea **reproducible** (siempre igual).

### 📊 Visualizando tamaños:

```python
print("Tamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)
```

### 📌 Tip adicional:

Si tu conjunto es muy **desbalanceado**, puedes usar:

```python
train_test_split(X, y, stratify=y, test_size=0.2)
```

Esto mantiene la **proporción de clases** tanto en entrenamiento como en prueba.

### Resumen

Evaluar correctamente un modelo de machine learning es crucial para asegurar su utilidad con datos que nunca ha visto anteriormente. Una buena manera de lograrlo es mediante la técnica conocida como **train test split**. Esta técnica consiste en dividir el conjunto de datos en dos partes principales: **entrenamiento**, donde el modelo aprende, y **prueba o test**, donde comprobamos su eficacia en escenarios nuevos.

##### ¿Por qué es importante dividir los datos en entrenamiento y prueba?

Para evitar que nuestro modelo simplemente memorice o se adapte en exceso a los datos con los que fue entrenado (problema conocido como overfitting), es esencial verificar cómo se comporta frente a datos nuevos. Esta división nos permite evaluar objetivamente su capacidad de generalizar lo aprendido:

- **Dato de entrenamiento**: Aquí, el modelo aprende patrones y características esenciales.
- **Dato de prueba (test)**: Conjunto nuevo utilizado para validar la eficacia real y la generalización del modelo.

#### ¿Cómo implementar la división de datos con scikit-learn?

Para llevar a cabo esta división, usamos la librería scikit-learn, específicamente la función train_test_split:

`from sklearn.model_selection import train_test_split`

Se configuran los parámetros como:

- **test_size** (tamaño del conjunto de prueba), habitualmente recomendado en 20%.
- **random_state**, para tener resultados consistentes en repeticiones.

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

#### ¿Cómo puedo experimentar con diferentes tamaños del conjunto prueba?

Existen herramientas como los widgets interactivos de Jupyter que pueden facilitar la comprensión del impacto:

```python
import ipywidgets as widgets
widgets.FloatSlider(value=0.2, min=0.1, max=0.4, step=0.05, continuous_update=False)
```

Usando controles dinámicos, experimentamos visualmente diferentes divisiones y examinamos cómo afecta al conjunto:

- Más datos en entrenamiento implicará potencialmente mejor aprendizaje.
- Más datos en prueba permitirá validar más robustamente su predicción.

#### ¿Cuál es la recomendación estándar para dividir los datasets?

Lo habitual es utilizar una proporción de 80-20, manteniendo el 80% para el entrenamiento y el 20% restante para el test. Esta distribución ha demostrado ser efectiva en la mayoría de escenarios, equilibrando aprendizaje y validación.

Ahora estás preparado para implementar esta práctica recomendada: dividir eficientemente los datos de tu modelo, garantizando así resultados confiables en nuevos conjuntos de información. ¿Listo para avanzar y aplicar regresión lineal en tus predicciones? Cuéntanos en comentarios cómo te fue con tu nueva implementación.

**Lecturas recomendadas**

[train_test_split — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html "train_test_split — scikit-learn 1.7.0 documentation")

[machine-learning/07_division_datos_interactiva.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/07_division_datos_interactiva.ipynb "machine-learning/07_division_datos_interactiva.ipynb at main · platzi/machine-learning · GitHub")

## Regresión lineal para predecir goles en fútbol

Vamos a ver cómo aplicar **regresión lineal** para predecir goles en fútbol usando **Python y scikit-learn**. Este modelo es ideal si quieres explorar relaciones como:

📊 **¿Cuántos goles marcará un equipo según sus tiros al arco, posesión, pases, etc.?**

### ⚽ Ejemplo: Regresión Lineal para predecir goles

### 📁 1. Datos de ejemplo (`pandas`)

Supongamos que tienes un DataFrame con estas columnas:

```python
import pandas as pd

# Datos ficticios de partidos
data = {
    'tiros_arco': [5, 3, 8, 6, 7],
    'posesion': [60, 45, 70, 55, 65],
    'pases': [500, 300, 700, 450, 600],
    'goles': [2, 1, 3, 2, 3]
}

df = pd.DataFrame(data)
```

### 🧪 2. División en entrenamiento y prueba

```python
from sklearn.model_selection import train_test_split

X = df[['tiros_arco', 'posesion', 'pases']]
y = df['goles']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🧠 3. Entrenar el modelo

```python
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)
```

### 🔍 4. Hacer predicciones

```python
y_pred = modelo.predict(X_test)
print("Predicciones de goles:", y_pred)
```

### 📈 5. Evaluar el modelo

```python
from sklearn.metrics import mean_squared_error, r2_score

print("Error cuadrático medio:", mean_squared_error(y_test, y_pred))
print("R² score:", r2_score(y_test, y_pred))
```

---

### 🧮 6. Interpretar los coeficientes

```python
coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo.coef_
})
print(coeficientes)
```

### ✅ ¿Qué te permite hacer esto?

* Ver **qué variables influyen más** en los goles.
* Usar el modelo para predecir goles futuros de equipos nuevos.
* Crear visualizaciones con `matplotlib` o `seaborn`.es?

### Resumen

¿Te imaginas poder predecir los goles de tu equipo favorito con métodos precisos? La regresión lineal, un modelo clásico en *machine learning*, ofrece una herramienta poderosa para asociar variables clave, como posesión del balón y tiros al arco, con la diferencia de goles en cada partido.

#### ¿Qué significa predecir goles usando regresión lineal?

La regresión lineal busca la mejor fórmula matemática para identificar cómo ciertas variables influyen en un resultado específico, como la diferencia de goles. Para esto, se usan datos precisos y claros:

- Posición de balón.
- Número de tiros al arco.

Estos datos permiten predecir la diferencia de goles, es decir, cuántos goles más marcará uno de los equipos sobre el contrincante.

#### ¿Cómo preparar los datos para entrenar el modelo?

El proceso es sencillo y directo:

- Se importa la biblioteca pandas y la función train_test_split.
- Creación de variable objetivo "diferencia de goles", definida por goles locales menos visitantes.
- Selección de variables predictoras, que incluyen posesión local y cantidad de tiros al arco.
- División del conjunto de datos en entrenamiento (80%) y evaluación (20%), manteniendo la consistencia en resultados mediante el parámetro random state.

#### ¿Qué resultados ofrece el modelo de regresión lineal?

Tras entrenar el modelo con la clase Linear Regression de la biblioteca scikit-learn, se obtienen dos elementos clave:

- **Intercepto (beta cero)**: predicción cuando las variables independientes son cero.
- **Coeficientes (betas)**: muestran cómo cambia la predicción al incrementar cada variable en una unidad.

Por ejemplo:

- Incrementar 1 unidad la posesión local cambia en promedio 0.06 la diferencia de goles.
- Incrementar 1 unidad los tiros al arco locales cambia en promedio -0.05 la diferencia de goles.

Esto ayuda a comprender qué variables merecen atención especial en la estrategia del equipo.

#### ¿Cómo evaluar y visualizar las predicciones?

Luego de hacer las predicciones con modelo_rl.predict, es fundamental visualizar los resultados:

- Uso de gráficos de dispersión comparando goles reales frente a predichos.
- Identificación rápida de qué predicciones coinciden mejor con la realidad y cuáles necesitan ajustes.

#### ¿Qué herramientas interactivas aportan valor adicional?

Los controles dinámicos, mediante sliders interactivos, permiten explorar cómo diferentes escenarios en posesión y tiros afectan la predicción final. Esto resulta especialmente útil para demostraciones prácticas y planificación estratégica con el entrenador o jugadores clave del equipo.

¿Has probado previamente una herramienta similar? ¿Cuál ha sido tu experiencia utilizando modelos estadísticos en deportes? ¡Comparte tus opiniones y expectativas sobre estos métodos!
 
**Lecturas recomendadas**

[machine-learning/08_regresion_lineal_cebollitas.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/08_regresion_lineal_cebollitas.ipynb "machine-learning/08_regresion_lineal_cebollitas.ipynb at main · platzi/machine-learning · GitHub")

## Métricas de evaluación para modelos de machine learning

Las **métricas de evaluación** son fundamentales para medir el **rendimiento real de un modelo de machine learning**, tanto en tareas de **regresión** como de **clasificación**. Aquí te presento las más comunes y útiles según el tipo de problema:


### 🧮 Para **Regresión** (predicción de valores numéricos, como goles)

### 1. **MSE – Error Cuadrático Medio**

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

* Penaliza los errores grandes.
* Entre menor, mejor.

### 2. **RMSE – Raíz del Error Cuadrático Medio**

```python
import numpy as np
rmse = np.sqrt(mse)
```

* Más interpretable, ya que está en la misma escala que la variable de salida.

### 3. **MAE – Error Absoluto Medio**

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

* Promedia las diferencias absolutas.
* Más robusto a valores atípicos que el MSE.

### 4. **R² – Coeficiente de Determinación**

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

* Cuánto del comportamiento de `y` es explicado por el modelo.
* Valores entre 0 y 1 (o negativos si el modelo es malo).
* Idealmente cercano a 1.

### 🧪 Para **Clasificación** (como predecir si un equipo gana, empata o pierde)

### 1. **Accuracy (Precisión global)**

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```

* Proporción de predicciones correctas.

### 2. **Precision, Recall, F1-score**

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

* **Precisión (Precision)**: cuántos positivos predichos realmente lo eran.
* **Recall (Sensibilidad)**: cuántos positivos reales fueron detectados.
* **F1-score**: balance entre precision y recall.

### 3. **Matriz de Confusión**

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred))
```

* Muestra aciertos y errores por clase.

### 4. **ROC AUC (para clasificación binaria)**

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_proba)
```

* Mide la capacidad del modelo para diferenciar clases.

### 📊 Visualizaciones útiles

* `sklearn.metrics.plot_confusion_matrix()`
* `seaborn.heatmap()` para la matriz de confusión
* Gráficas de ROC y Precision-Recall

### Resumen

Evaluar un modelo de *machine learning* es tan importante como entrenarlo. Al utilizar métricas específicas, es posible determinar qué tan bien está desempeñándose el modelo y si realmente puede usarse para tomar decisiones informadas. En este contexto, consideraremos cuatro métricas fundamentales: **Error Cuadrático Medio (MSE), Raíz del Error Cuadrático Medio (RMSE), Error Absoluto Medio (MAE) y Coeficiente de Determinación (R²)**.

#### ¿Qué métricas existen para evaluar modelos predictivos?

Cada métrica brinda información particular sobre el rendimiento del modelo:

- **MSE** penaliza fuertemente errores significativos al amplificarlos al cuadrado, lo que ayuda a detectar desviaciones considerables aunque la interpretación directa en términos prácticos (por ejemplo, goles) es complicada.
- **RMSE** convierte el MSE nuevamente a la escala original, proporcionando una interpretación más intuitiva y fácil de comunicar; muy útil para presentaciones a personas no especializadas técnicamente.
- **MAE** calcula el promedio directo de los errores absolutos, siendo robusto frente a valores extremos o outliers, con una interpretación clara y directa.
- **Coeficiente R²** muestra cuánto de la variación en los datos logra explicar el modelo, indicando su capacidad general para captar tendencias.

#### ¿Cómo implementar estas métricas en Python?

Con bibliotecas como *pandas*, *NumPy* y funciones específicas de evaluación, se realiza un cálculo riguroso. Previamente, dividimos nuestros datos entre entrenamiento y validación con *train test split*, y ajustamos un modelo de regresión lineal usando

```python
from sklearn.linear_model import LinearRegression
modelo_RL = LinearRegression()
modelo_RL.fit(X_train, y_train)
y_pred = modelo_RL.predict(X_test)
```

Luego, aplicamos las métricas mencionadas:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
```

Estas medidas aportan claridad al Informe Técnico sobre el rendimiento del modelo y facilitan la comunicación efectiva con distintos públicos interesados, como entrenadores o directivos.

#### ¿Por qué usar múltiples métricas de evaluación?

Combinar varias métricas es clave pues así obtenemos un panorama integral del modelo:

- MSE y RMSE: Detectan desviaciones importantes.
- MAE: Presenta el error típico claramente.
- R²: Indica la proporción de la variabilidad explicada por el modelo.

Usadas conjuntamente, estas métricas proveen un diagnóstico robusto sobre la utilidad práctica del modelo y ayudan a decidir próximos pasos para ajustes y mejoras.

Te invito a compartir tus experiencias evaluando modelos o cualquier inquietud sobre las métricas mencionadas.

**Lecturas recomendadas**

[Overfitting in Machine Learning: What It Is and How to Prevent It](https://elitedatascience.com/overfitting-in-machine-learning "Overfitting in Machine Learning: What It Is and How to Prevent It")

[machine-learning/09_evaluacion_modelo_cebollitas.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/09_evaluacion_modelo_cebollitas.ipynb "machine-learning/09_evaluacion_modelo_cebollitas.ipynb at main · platzi/machine-learning · GitHub")

## Evaluación de métricas en regresión lineal para datos deportivos

Evaluar un modelo de **regresión lineal aplicado a datos deportivos** (como predecir goles, tiros al arco, puntos, etc.) es esencial para entender qué tan bien está funcionando tu modelo.

### ⚽ Escenario típico

Supón que tienes datos deportivos y estás prediciendo una variable como:

> **`y = goles`**
> A partir de variables como: tiros al arco, posesión, pases, faltas, etc.

### ✅ Métricas clave para evaluación de regresión lineal

### 1. 📉 **MSE – Error Cuadrático Medio**

* Penaliza fuertemente los errores grandes.

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

### 2. 📊 **MAE – Error Absoluto Medio**

* Promedia las diferencias absolutas. Más interpretable y robusto.

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

### 3. 📈 **RMSE – Raíz del Error Cuadrático Medio**

* Muestra el error promedio en la **misma escala que la variable objetivo**.

```python
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

### 4. 🧮 **R² – Coeficiente de Determinación**

* Indica qué proporción de la varianza es explicada por el modelo.
* Valor entre 0 y 1 (mejor si se acerca a 1).

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

### 🔧 Ejemplo en código:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# Supongamos que tienes datos deportivos
data = {
    'tiros_arco': [5, 3, 8, 6, 7],
    'posesion': [60, 45, 70, 55, 65],
    'pases': [500, 300, 700, 450, 600],
    'goles': [2, 1, 3, 2, 3]
}
df = pd.DataFrame(data)

X = df[['tiros_arco', 'posesion', 'pases']]
y = df['goles']

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R\u00b2: {r2:.2f}")
```

### 📌 Interpretación de resultados

| Métrica | ¿Qué indica?                                                  |
| ------- | ------------------------------------------------------------- |
| MSE     | Qué tan grandes son los errores (penaliza errores grandes).   |
| MAE     | Promedio del error absoluto (más fácil de interpretar).       |
| RMSE    | Error típico (en unidades de goles, por ejemplo).             |
| R²      | Qué tan bien el modelo explica la variabilidad del resultado. |

### Resumen

La creación de un modelo de regresión lineal aplicado a datos deportivos, específicamente para analizar goles en partidos de fútbol, implica evaluar su efectividad mediante métricas clave como el R cuadrado y el error cuadrático medio (RMC). Al importar nuestros datos y el modelo entrenado, observamos cómo estos indicadores nos informan claramente sobre el desempeño del modelo y su utilidad práctica.

#### ¿Qué información obtenemos al evaluar nuestro modelo?

Al aplicar métricas como el R cuadrado (R dos), determinamos rápidamente si nuestro modelo de regresión lineal explica adecuadamente la variabilidad observada en los datos:

- Cuando el valor es negativo, indica que el modelo es incluso menos acertado que simples suposiciones aleatorias.
- Si el valor está entre cero y 0.3, el nivel explicativo es insuficiente, señalando potencial under fitting.
- Valores superiores a 0.3 sugieren un grado aceptable de explicación de los datos.

En este caso, al encontrar un R cuadrado negativo, confirmamos que nuestro modelo actual no capta correctamente los patrones necesarios para explicar las variaciones en diferencia de goles.

#### ¿Son adecuadas las variables utilizadas?

Es fundamental cuestionarnos sobre la elección y relevancia de las variables usadas. ¿Están capturando realmente los factores decisivos que marcan la diferencia en goles? Algunas variables importantes, como la localía o el desempeño rival en tiros al arco, podrían estar ausentes. Considerar estas dimensiones del juego puede aportar mejores insights y elevar significativamente la precisión del modelo.

#### ¿Existen limitaciones concretas al usar regresión lineal en fútbol?

La regresión lineal presenta ciertas limitaciones importantes al aplicarla a situaciones complejas como partidos de fútbol:

- Supone relaciones lineales entre variables, condición que no necesariamente refleja la dinámica real de un partido.
- No captura adecuadamente interacciones o efectos no lineales frecuentes en contextos deportivos.

Estas limitaciones invitan a explorar otros modelos más adecuados.

#### ¿Es suficiente este modelo para la toma de decisiones deportivas?

Debido al bajo desempeño identificado, este modelo en específico no podría considerarse suficiente para fundamentar decisiones deportivas estratégicas. Su reducido poder explicativo limita la fiabilidad de las predicciones realizadas, aconsejando buscar alternativas que aporten una visión más robusta y confiable.

#### ¿Qué alternativas podemos considerar para mejorar el modelo?

Tenemos diversas opciones de mejora y optimización:

- Incorporación de nuevas variables relevantes, tales como la localía, características del rival o estadísticas adicionales (por ejemplo, tiros al arco).
- Aplicación de distintos modelos predictivos más sofisticados y flexibles, como árboles de decisión, random forest o algoritmos como XGBoost.
- Implementación de validación cruzada para evaluar con mayor precisión la capacidad predictiva.
- Filtrado y transformación de datos para mejorar métricas predictivas.

Mantener una mente abierta hacia estos enfoques diferentes podría resultar clave en la obtención de modelos más efectivos, asegurando decisiones estratégicas enraizadas en análisis sólidos y precisos.

¿Y tú qué opinas sobre estos enfoques adicionales? Esta reflexión es parte clave del aprendizaje continuo.
 
**Lecturas recomendadas**

[Overfitting in Machine Learning: What It Is and How to Prevent It](https://elitedatascience.com/overfitting-in-machine-learning "Overfitting in Machine Learning: What It Is and How to Prevent It")

[machine-learning/10_reflexion_modelo_regresion_cebollitas.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/10_reflexion_modelo_regresion_cebollitas.ipynb "machine-learning/10_reflexion_modelo_regresion_cebollitas.ipynb at main · platzi/machine-learning · GitHub")

[Bonus: machine-learning/11_Bonus.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/11_Bonus.ipynb "Bonus: machine-learning/11_Bonus.ipynb at main · platzi/machine-learning · GitHub")

## Reflexión Crítica y Conclusión

La comprensión de modelos de aprendizaje automático requiere no solo implementarlos, sino también saber evaluar su rendimiento y adaptarse cuando no funcionan como esperamos. En el mundo real, es común tener que pivotar entre diferentes algoritmos hasta encontrar el que mejor se ajusta a nuestros datos y al problema que intentamos resolver. Aprender a interpretar métricas y tomar decisiones basadas en ellas es una habilidad fundamental para cualquier científico de datos.

#### ¿Por qué falló el modelo de regresión lineal?

Al analizar el rendimiento de nuestro modelo de regresión lineal, nos encontramos con resultados poco alentadores. Las métricas revelan un panorama claro:

- El modelo presenta un **R² negativo**, lo que indica que su desempeño es peor que simplemente predecir el valor promedio de los datos.
- Los errores (RMSE y MAE) son **bastante altos**, demostrando una pobre capacidad predictiva.

Estos resultados sugieren fuertemente que la relación entre nuestras variables no es lineal. Cuando intentamos forzar una relación lineal en datos que siguen patrones no lineales, el modelo no puede captar adecuadamente estos patrones, resultando en predicciones deficientes.

#### ¿Qué alternativas tenemos frente a un modelo que no funciona?

Cuando un modelo no cumple con nuestras expectativas, es momento de explorar alternativas. En este caso, el árbol de decisión emerge como una opción prometedora:

- Los árboles de decisión pueden capturar relaciones no lineales entre variables.
- Son capaces de modelar interacciones complejas sin asumir una forma específica en los datos.

Al implementar este nuevo enfoque, observamos mejoras significativas en todas las métricas:

- **Reducción en RMSE y MAE**: Los errores de predicción disminuyeron notablemente.
- R² positivo: A diferencia del modelo lineal, el árbol demuestra capacidad para explicar la variabilidad en los datos.

Estas mejoras confirman nuestra hipótesis: estamos tratando con datos que presentan relaciones no lineales.

#### ¿Qué hemos aprendido hasta ahora?

Este ejercicio nos ha proporcionado valiosas lecciones:

1. **Preparación de datos y construcción de modelos básicos**: Hemos aprendido a procesar datos y crear modelos iniciales para abordar problemas.
2. **Evaluación mediante métricas**: Ahora sabemos interpretar diferentes métricas y utilizarlas para evaluar el rendimiento de nuestros modelos.
3. **No todos los algoritmos sirven para todos los problemas**: Quizás la lección más importante es comprender que debemos adaptar nuestro enfoque según la naturaleza de los datos.

#### ¿Cómo rediseñar nuestra estrategia a partir de estos hallazgos?

Con base en los resultados obtenidos, podemos replantear nuestra aproximación al problema:

1. **Redefinir un pipeline más adecuado**: Utilizar el árbol de decisión como modelo base e iterar sobre él.
2. **Mejorar las visualizaciones**: Crear representaciones visuales que nos ayuden a entender mejor la estructura no lineal de nuestros datos.
3. **Explorar modelos más robustos**: Considerar algoritmos más avanzados que puedan capturar patrones complejos, como:
- Random Forest
- Gradient Boosting
- Redes neuronales

Este nuevo enfoque marca un comienzo más realista y alineado con el comportamiento real de nuestros datos. La capacidad de pivotar y adaptarse cuando los resultados no son los esperados es una habilidad crucial en ciencia de datos.

El camino del aprendizaje automático está lleno de iteraciones y ajustes. Cada "fracaso" nos acerca más a una comprensión profunda de nuestros datos y a soluciones más efectivas. ¿Qué otros modelos crees que podrían funcionar bien con datos no lineales? ¿Has tenido experiencias similares donde tuviste que cambiar completamente tu enfoque?

**Archivos de la clase**

[jugadores-cebollitas.csv](https://static.platzi.com/media/public/uploads/jugadores_cebollitas_33ecea5c-f6f0-44ca-9ff4-e4135408bc04.csv "jugadores-cebollitas.csv")

**Lecturas recomendadas**

[3.4. Metrics and scoring: quantifying the quality of predictions — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/model_evaluation.html "3.4. Metrics and scoring: quantifying the quality of predictions — scikit-learn 1.7.0 documentation")

[machine-learning/12_Clase_Reflexion_Critica_Conclusion.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/12_Clase_Reflexion_Critica_Conclusion.ipynb "machine-learning/12_Clase_Reflexion_Critica_Conclusion.ipynb at main · platzi/machine-learning · GitHub")

## Clasificación automatizada de jugadores con algoritmo K-means

El algoritmo **K-Means** es una técnica de **machine learning no supervisado** muy útil para **agrupar jugadores automáticamente** según su rendimiento, estilo o características físicas, sin necesidad de conocer de antemano sus posiciones o roles.

### ⚽ Ejemplo práctico: Clasificación de jugadores con K-Means

### 📌 Objetivo:

Agrupar jugadores en **clusters** similares con base en estadísticas como:

* Goles
* Asistencias
* Pases completados
* Recuperaciones
* Velocidad, etc.

### 🧰 Paso a paso con Python:

#### 1. 📥 Cargar datos de ejemplo

```python
import pandas as pd

# Datos ficticios
data = {
    'nombre': ['Jugador A', 'Jugador B', 'Jugador C', 'Jugador D', 'Jugador E'],
    'goles': [10, 2, 5, 0, 7],
    'asistencias': [5, 1, 2, 0, 3],
    'pases_completos': [300, 100, 200, 150, 250]
}

df = pd.DataFrame(data)
```

#### 2. 🎯 Seleccionar variables y escalar

```python
from sklearn.preprocessing import StandardScaler

X = df[['goles', 'asistencias', 'pases_completos']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 3. 🤖 Aplicar K-Means

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

#### 4. 📊 Ver los resultados

```python
print(df[['nombre', 'cluster']])
```

### 🎨 (Opcional) Visualización con `matplotlib`

```python
import matplotlib.pyplot as plt

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('Goles (escalado)')
plt.ylabel('Asistencias (escalado)')
plt.title('Clasificación de jugadores con K-Means')
plt.grid(True)
plt.show()
```

### 🧠 ¿Qué puedes hacer con esto?

* Identificar **tipos de jugadores** (ofensivos, creativos, defensivos, etc.).
* Sugerir **roles dentro del equipo** automáticamente.
* Analizar cómo se agrupan tus jugadores vs. los de otros equipos.

### 🧪 Tip:

Si no sabes cuántos grupos (clusters) elegir, usa el **método del codo (elbow method)** para determinar el mejor valor de `k`.

### Resumen

¿Sabías que puedes utilizar Machine Learning no solo para predecir resultados, sino también para entender mejor a los jugadores de tu equipo? Una herramienta potente es el algoritmo K-means, eficaz para agrupar atletas según sus estadísticas individuales. Con K-means, identificamos perfiles estratégicos como delanteros goleadores, volantes creativos o defensas equilibrados sin asignar etiquetas previas.

#### ¿Qué es el aprendizaje no supervisado y cómo se utiliza con jugadores de fútbol?

El aprendizaje no supervisado implica enseñar al modelo sin ejemplos específicos, permitiendo que encuentren patrones por su cuenta. A diferencia del aprendizaje supervisado, aquí no decimos qué es correcto o incorrecto desde el inicio. Con algoritmos como K-means, los jugadores se agrupan automáticamente en base a características compartidas como:

- Goles realizados.
- Asistencias otorgadas.
- Pases completados.
- Tiros al arco.

Esto ayuda a revelar semejanzas que quizás hasta el momento habían pasado desapercibidas.

#### ¿Cómo funciona el algoritmo K-means para clasificar jugadores?

K-means agrupa jugadores según características numéricas específicas. Sigue estos pasos clave:

1. Selecciona un número predefinido de clusters o grupos.
2. Asigna inicialmente los jugadores a un grupo basándose en cercanía matemática.
3. Ajusta iterativamente hasta lograr grupos estables.

De esta forma, jugadores con perfiles similares se agrupan entre sí, facilitando la interpretación de sus desempeños.

#### ¿Qué ofrece explorar estos clusters en un entorno interactivo como Jupyter Notebook?

Cuando usamos K-means dentro de un notebook, podemos realizar procesos como:

#### Importar datos

Usando `pandas`, cargamos métricas individuales desde un archivo:

```python
import pandas as pd
df = pd.read_csv('jugadores.csv')
df.head()
```

#### Visualizar relaciones estadísticas

Con librerías como `seaborn` y `matplotlib`, visualizamos patrones y correlaciones fácilmente:

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)
plt.show()
```

#### Aplicar el clustering con K-means

Implementamos el algoritmo indicando el número de grupos deseado:

```python
from sklearn.cluster import KMeans
modelo = KMeans(n_clusters=3, random_state=0)
df['cluster'] = modelo.fit_predict(df[['goles', 'asistencias', 'pases', 'tiros_al_arco']])
df.head()
```

#### ¿Cómo interpretar los grupos generados por K-means?

Con gráficos y estadísticas podemos definir perfiles claros. Por ejemplo, un grupo con alta cifra de goles y tiros, pero pocas asistencias, sugiere delanteros ofensivos. Así mismo, un grupo predominante en asistencias y pases podría indicar volantes creativos.

#### Visualización gráfica de clusters

Un gráfico de dispersión o scatter plot permite visualizar rápidamente estos perfiles:

```python
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='goles', y='asistencias', hue='cluster', palette='Set1')
plt.title('Grupos de Jugadores según Goles y Asistencias')
plt.xlabel('Goles')
plt.ylabel('Asistencias')
plt.show()
```

#### Exploración interactiva

La interactividad permite ajustar dinámicamente el número de grupos para identificar la cantidad ideal de perfiles útiles:

```python
import ipywidgets as widgets
from ipywidgets import interact

def cluster_interactivo(num_clusters):
    modelo = KMeans(n_clusters=num_clusters, random_state=0)
    df['cluster'] = modelo.fit_predict(df[['goles', 'asistencias', 'pases', 'tiros_al_arco']])
    sns.scatterplot(data=df, x='goles', y='asistencias', hue='cluster', palette='Set1')
    plt.show()

interact(cluster_interactivo, num_clusters=(2,6,1))
```

#### ¿Qué beneficios aporta clasificar jugadores mediante K-means?

El análisis automatizado permite: - Entrenar a jugadores según su perfil específico. - Tomar decisiones tácticas fundamentadas en estadísticas reales. - Identificar necesidades claras para futuros fichajes.

Ahora cuentas con una herramienta fiable para conocer en profundidad a tus jugadores, optimizar entrenamientos e implementar tácticas efectivas. ¿Te animas a probarla en tu equipo y contarnos qué patrones o grupos nuevos encontraste?
 
**Lecturas recomendadas**

[2.3. Clustering — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means "2.3. Clustering — scikit-learn 1.7.0 documentation")

[machine-learning/16_clustering_kmeans_jugadores.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/16_clustering_kmeans_jugadores.ipynb "machine-learning/16_clustering_kmeans_jugadores.ipynb at main · platzi/machine-learning · GitHub")

## Interpretación de clusters K-Means para perfiles de jugadores

Una vez que aplicaste **K-Means** y tienes los jugadores agrupados en **clusters**, el siguiente paso es interpretar esos grupos. Es decir: **¿qué significa cada cluster?** ¿Qué perfil de jugador representa?

### 🎯 ¿Qué es interpretar los clusters?

Interpretar un **cluster** es descubrir **qué tienen en común los jugadores dentro de ese grupo**. Esto se hace observando las **características promedio** de cada grupo.

### 🧠 Pasos para interpretar clusters de jugadores

### 1. ✅ **Agregar el número de cluster a tu DataFrame**

Si no lo has hecho:

```python
df['cluster'] = kmeans.labels_
```

### 2. 📊 **Agrupar por cluster y obtener estadísticas**

```python
perfil_cluster = df.groupby('cluster').mean(numeric_only=True)
print(perfil_cluster)
```

Esto te dirá, por ejemplo:

| cluster | goles | asistencias | pases\_completos |
| ------- | ----- | ----------- | ---------------- |
| 0       | 7.5   | 3.2         | 280              |
| 1       | 1.0   | 0.3         | 120              |

👉 Aquí puedes decir:

* Cluster 0 = **jugadores ofensivos** (marcan más, asisten más).
* Cluster 1 = **jugadores defensivos o con menor participación ofensiva**.

### 3. 🧩 **Etiquetar clusters con perfiles intuitivos**

Puedes usar la media de los datos o visualizaciones para decidir etiquetas como:

| Cluster | Perfil sugerido           |
| ------- | ------------------------- |
| 0       | "Atacantes creativos"     |
| 1       | "Defensores o suplentes"  |
| 2       | "Mediocampistas de apoyo" |

### 4. 📈 (Opcional) **Visualiza los clusters**

```python
import matplotlib.pyplot as plt

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel("Goles (escalado)")
plt.ylabel("Asistencias (escalado)")
plt.title("Clusters de jugadores - KMeans")
plt.show()
```

Si tienes más de 2 dimensiones, puedes usar PCA para reducir a 2D:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='tab10')
plt.title("Clusters de jugadores (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

### 🧪 Consejo profesional

* Añade variables **contextuales**: minutos jugados, posición, equipo, etc.
* Compara tu interpretación con la realidad (¿el modelo detecta bien a los delanteros, volantes, etc.?).

### Resumen

El análisis del rendimiento deportivo mediante técnicas avanzadas como el clustering K-Means permite identificar rápidamente perfiles claros dentro de un equipo de fútbol. Aprenderás cómo interpretar clusters generados a partir de estadísticas clave, como goles, asistencias, pases completados y tiros al arco, identificando quiénes destacan en ataque, defensa o juego colectivo. Esta habilidad constituye una valiosa herramienta estratégica y táctica para cualquier cuerpo técnico.

#### ¿Qué es el análisis por clustering usando K-Means?

El algoritmo K-Means es una técnica del aprendizaje no supervisado usada para agrupar individuos, como jugadores de fútbol, según ciertas características estadísticas. Su objetivo es identificar perfiles o grupos homogéneos para facilitar la toma de decisiones tácticas y estratégicas.

#### ¿Cómo preparar los datos para K-Means?

En primer lugar, se importan y visualizan los datos mediante la librería pandas en Python, asegurando observar bien las columnas disponibles. Los datos seleccionados típicamente para este análisis incluyen:

- Goles.
- Asistencias.
- Pases completados.
- Tiros al arco.

La ejecución del algoritmo K-Means requiere únicamente estas variables específicas para crear grupos útiles basados en estadísticas reales de juego.

```python
import pandas as pd
jugadores = pd.read_csv('jugadores.csv')
print(jugadores.columns)
```

#### ¿Cómo interpretar los resultados del análisis?

Una vez creados los clusters por K-Means, utilizamos el método `.groupby()` junto a `.mean()` para calcular promedios de cada métrica dentro de cada cluster. Este proceso revela perfiles promedio muy claros de cada grupo. Observando así:

- Qué jugadores anotan más.
- Quiénes asisten más frecuentemente.
- Cuáles completan más pases o rematan más al arco.

#### ¿Cómo se visualizan y comparan los clusters?

La visualización mediante Boxplots permite examinar con claridad y rapidez la distribución interna y los valores atípicos (outliers) por cada grupo. Las gráficas obtenidas destacan de inmediato las diferencias estadísticamente significativas entre clusters.

Mediante estas visualizaciones podemos confirmar hipótesis, por ejemplo:

- El cluster 0 presenta más goles.
- El cluster 1 destaca en asistencias.
- El cluster con más tiros al arco posiblemente representa delanteros.

Esto ayuda mucho al cuerpo técnico a entender claramente dónde sobresale cada jugador.

#### ¿Cómo utilizar widgets para una exploración dinámica?

La utilización de widgets de selección rápido permite filtrar datos visualmente, lo cual es extremadamente útil en reuniones técnicas. Mediante Python, se pueden ver jugadores específicos por cluster junto a sus métricas destacadas. Esto permite una interacción en tiempo real, mejorando la comprensión y facilitando la planificación técnica.

```python
import ipywidgets as widgets
from IPython.display import display

cluster_selector = widgets.Dropdown(options=[0,1,2])

def mostrar_jugadores(cluster):
    display(jugadores[jugadores['cluster'] == cluster])

widgets.interact(mostrar_jugadores, cluster=cluster_selector)
```

#### ¿Cómo aplicar estos resultados en decisiones reales?

El análisis avanzado mediante clustering es directamente aplicable en decisiones tácticas, ayudando a los entrenadores a definir roles específicos dentro del campo de juego:

- Organizar alineaciones óptimas según las fortalezas estadísticas.
- Diseñar entrenamientos personalizados, enfocados en el perfil real.
- Evaluar el potencial fichaje basados en necesidades concretas del equipo.

Este enfoque claro y objetivo basado en datos puede ser crucial para implementar una gestión táctica moderna que conduzca a mejores resultados deportivos.

Finalmente, la próxima técnica a revisar será PCA (Principal Component Analysis), utilizada para simplificar visualizaciones complejas sin perder información relevante.
 
**Lecturas recomendadas**

[Las 12 mejores herramientas y software UX para perfeccionar la experiencia de usuario](https://www.hotjar.com/es/diseno-ux/herramientas/ "Las 12 mejores herramientas y software UX para perfeccionar la experiencia de usuario")

[Plotting Time Series Boxplots](https://towardsdatascience.com/plotting-time-series-boxplots-5a21f2b76cfe/ "Plotting Time Series Boxplots")

[machine-learning/17_interpretacion_clusters_jugadores.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/17_interpretacion_clusters_jugadores.ipynb "machine-learning/17_interpretacion_clusters_jugadores.ipynb at main · platzi/machine-learning · GitHub")

## Análisis PCA para agrupar jugadores según rendimiento

El **Análisis de Componentes Principales (PCA)** es una técnica muy útil para:

🔹 **Reducir la dimensionalidad** de tus datos
🔹 **Visualizar grupos (clusters) de jugadores** en 2D o 3D
🔹 **Mantener la mayor varianza posible** de los datos originales

### ⚽ Escenario: Agrupar jugadores según rendimiento

Supón que tienes un DataFrame `df_jugadores` con estadísticas como:

* Goles
* Asistencias
* Tiros al arco
* Pases completados
* Recuperaciones
* ... y una columna `cluster` asignada por `KMeans`.

### 🧰 Paso a paso: Análisis PCA en Python

### 1. 📦 Importar librerías necesarias

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. ⚙️ Preparar los datos

```python
# Selecciona solo las columnas numéricas de rendimiento
features = ['goles', 'asistencias', 'pases_completados (%)', 'tiros_al_arco']

# Estandarizar para que todas las variables tengan media 0 y varianza 1
X = StandardScaler().fit_transform(df_jugadores[features])
```

### 3. 🧠 Aplicar PCA

```python
pca = PCA(n_components=2)  # Para visualización en 2D
X_pca = pca.fit_transform(X)

# Agregar los componentes al DataFrame
df_jugadores['PC1'] = X_pca[:, 0]
df_jugadores['PC2'] = X_pca[:, 1]
```

### 4. 📊 Visualizar los clusters en el espacio PCA

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='cluster',
    palette='Set2',
    data=df_jugadores,
    s=100,
    edgecolor='k'
)
plt.title('Clusters de jugadores en espacio PCA')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.grid(True)
plt.show()
```

### ✅ ¿Qué interpretas de este gráfico?

* Los **puntos cercanos** representan jugadores similares en sus estadísticas.
* Cada **color** representa un **cluster de K-Means**.
* Si los clusters están **bien separados**, significa que tu segmentación tiene **sentido y valor analítico**.
* Puedes analizar qué **variables contribuyen más a cada componente** usando:

```python
pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
```

### Resumen

El análisis de datos en fútbol puede complicarse cuando manejamos múltiples variables por jugador, como goles, asistencias o precisión. La técnica PCA, o Análisis de Componentes Principales, simplifica este proceso reduciendo múltiples variables a solo dos o tres componentes principales, ofreciendo una visualización gráfica clara e intuitiva del rendimiento y agrupación natural de jugadores.

#### ¿Qué es PCA y cómo simplifica el análisis futbolístico?

El PCA es una técnica matemática que transforma variables complejas en unas pocas nuevas llamadas componentes principales. Estas nuevas variables son combinaciones de las originales y retienen la mayor parte de la información inicial. Esto permite:

- Visualizar datos complejos en gráficos 2D o 3D.
- Crear resúmenes efectivos del rendimiento individual y grupal.
- Identificar grupos naturales de jugadores según estadísticas específicas.

#### ¿Cómo beneficia el PCA al scouting y decisiones tácticas?

Realizar un análisis mediante PCA tiene múltiples ventajas:

- Facilita la identificación visual rápida de grupos de jugadores (delanteros, volantes, defensivos).
- Permite detectar jugadores atípicos o outliers, con habilidades únicas en comparación con el equipo.
- Simplifica la comparación directa entre jugadores.
- Apoya la toma de decisiones tácticas efectivas y selección de refuerzos ideales.

#### Visualización práctica del rendimiento con PCA

En el ejemplo práctico, tomando en cuenta variables como goles, asistencias o tiros, el PCA revela diferentes grupos claros en un solo gráfico:

- Los delanteros destacan en una esquina por su alta cantidad de goles y tiros.
- Mediocampistas aparecen centralizados, combinando diversas capacidades.
- Volantes resaltan por asistencias y pases, ubicados generalmente en otra área del gráfico.

#### Integración con clustering (K-means)

Se complementa PCA con K-means clustering para asignar etiquetas visuales claras a cada jugador según su estilo de juego:

- Cada color identifica un perfil futbolístico particular.
- Sirve para planificación, entrenamientos específicos, fichajes y scouting.
- Facilita la exposición visual sencilla y rápida de perfiles técnicos al cuerpo encargado.

#### Visualización interactiva del PCA

Con herramientas interactivas, como widgets dropdown, se puede:

- Explorar interactivamente combinaciones de componentes principales.
- Presentar al cuerpo técnico visualizaciones dinámicas y personalizadas.
- Facilitar el análisis detallado del rendimiento en tiempo real.

#### ¿Qué se logra al implementar PCA en el análisis futbolístico?

Implementar PCA implica obtener:

- Reducción efectiva de la complejidad de los datos.
- Visualización rápida y clara de agrupamientos naturales y perfiles específicos de jugadores.
- Una herramienta ágil que respalde decisiones técnicas inteligentes en tiempo real.

Te invito a comentar qué otros usos prácticos considerarías para PCA dentro de tu equipo.
 
**Lecturas recomendadas**

[2.5. Decomposing signals in components (matrix factorization problems) — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca "2.5. Decomposing signals in components (matrix factorization problems) — scikit-learn 1.7.0 documentation")

[machine-learning/18_pca_visualizacion_jugadores.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/18_pca_visualizacion_jugadores.ipynb "machine-learning/18_pca_visualizacion_jugadores.ipynb at main · platzi/machine-learning · GitHub")

## Pipeline integrado de Machine Learning para análisis deportivo

Un **pipeline de Machine Learning** bien diseñado para **análisis deportivo** te permite automatizar y optimizar todo el flujo de trabajo, desde los datos hasta las predicciones.

### ⚽ ¿Qué es un pipeline de ML en análisis deportivo?

Es un **flujo estructurado** que:

1. Recibe y limpia datos de rendimiento deportivo.
2. Extrae o transforma variables (features).
3. Aplica escalamiento o normalización.
4. Entrena un modelo (regresión, clasificación, clustering...).
5. Evalúa el desempeño del modelo.
6. Aplica el modelo a nuevos datos.

### 🔄 Ejemplo de pipeline con `scikit-learn`

### 🎯 Caso práctico:

Predecir la cantidad de goles de un jugador a partir de sus estadísticas (tiros, asistencias, pases, etc.).

### ✅ 1. Importar librerías

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
```

### ✅ 2. Datos de ejemplo

```python
df = pd.DataFrame({
    'tiros_al_arco': [30, 12, 45, 10, 33],
    'asistencias': [5, 2, 7, 1, 3],
    'pases_completados': [300, 150, 400, 120, 280],
    'goles': [12, 3, 15, 2, 10]  # variable objetivo
})

X = df.drop('goles', axis=1)
y = df['goles']
```

### ✅ 3. Dividir en entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ 4. Crear el pipeline

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),              # Escalamiento de datos
    ('regresor', LinearRegression())           # Modelo de regresión
])
```

### ✅ 5. Entrenar el modelo

```python
pipeline.fit(X_train, y_train)
```

### ✅ 6. Evaluar el modelo

```python
y_pred = pipeline.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))
```

### 🚀 ¿Qué más se puede integrar al pipeline?

* **Selección de variables** (`SelectKBest`, `RFE`)
* **Reducción de dimensionalidad** (`PCA`)
* **Modelos avanzados** (Random Forest, XGBoost)
* **Cross-validation**
* **Exportación automática con `joblib`**

### 🧠 ¿Por qué usar pipelines?

* 💡 Reproducibilidad
* 🔁 Reutilización del flujo
* ✅ Evitas errores entre etapas
* 📦 Es fácil de integrar con **GridSearchCV** y producción

### Resumen

La inteligencia deportiva a través de modelos de *machine learning* está transformando la forma en que equipos y entrenadores ejecutan sus estrategias. Al combinar eficazmente modelos supervisados y no supervisados en un pipeline integrado, logramos predicciones precisas sobre resultados de partidos y análisis automático de perfiles de jugadores.

#### ¿Qué es un pipeline integrado avanzado?

Un pipeline integrado avanzado en el ámbito deportivo permite automatizar todo un flujo de trabajo, desde la preparación de datos hasta la generación automática de predicciones. Esta herramienta reúne modelos supervisados y no supervisados para ofrecer resultados coherentes, claros y escalables en tiempo real.

Este pipeline presenta dos funciones centrales:

- **Modelo supervisado (Regresión Ridge)**: predice diferencias esperadas de goles teniendo en cuenta estadísticas clave como posesión de balón y cantidad de tiros.
- **Modelo no supervisado (Clúster K-Means)**: clasifica automáticamente a los jugadores en grupos claramente definidos según estadísticas individuales tales como goles, asistencias y pases concretados.

La combinación de ambos modelos constituye un poderoso motor analítico:

- Escala automáticamente datos en ambos modelos.
- Genera predicciones claras y fáciles de interpretar.
- Facilita decisiones rápidas y efectivas sobre estrategias a seguir durante los partidos.

#### ¿Cómo funciona la integración entre modelos supervisados y no supervisados?

La aplicación funciona en varios pasos bien definidos:

1. Carga y preparación de datasets sobre partidos y jugadores.
2. Implementación del pipeline supervisado con regresión Ridge para predecir resultados.
3. Uso de K-Means en un pipeline no supervisado para clasificar a los jugadores en perfiles de acuerdo a su desempeño.
4. Análisis integrado para visualizar resultados esperados y perfiles de jugadores disponibles, proporcionando una base sólida para decisiones tácticas en tiempo real.

#### ¿Cómo esta herramienta beneficia al cuerpo técnico?

Contar con esta herramienta predictiva es como tener un asistente inteligente 24/7. Permite:

- Visualizar rápidamente cómo puede desarrollarse un partido, prediciendo diferencias de goles basadas en escenarios ajustables de posesión y tiros al arco.
- Identificar claramente tipos específicos de jugadores según perfiles individuales clasificados previamente.
- Ajustar en tiempo real las tácticas según las predicciones generadas.
- Crear análisis personalizados para recomendaciones tácticas, fichajes o entrenamientos específicos.

Este sistema también incluye widgets interactivos que permiten una interacción dinámica con los modelos predictivos, ofreciendo al equipo técnico una plataforma intuitiva y accesible para evaluar escenarios cambiantes.

¿Qué ajustes tácticos sugerirías para optimizar resultados en el próximo partido? Te invito a comentar tus ideas basadas en estas herramientas predictivas.

**Lecturas recomendadas**

[Pipeline — scikit-learn 1.7.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html "Pipeline — scikit-learn 1.7.0 documentation")

[machine-learning/19_pipeline_avanzado_presentacion.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/19_pipeline_avanzado_presentacion.ipynb "machine-learning/19_pipeline_avanzado_presentacion.ipynb at main · platzi/machine-learning · GitHub")

## Redes neuronales artificiales con PyTorch para clasificación binaria

Las **redes neuronales artificiales (ANN)** con **PyTorch** son una herramienta poderosa para tareas como **clasificación binaria**, por ejemplo:

> ¿Un equipo gana (1) o no gana (0) un partido?

### ⚙️ ¿Qué cubriremos?

* Estructura de una red neuronal para clasificación binaria
* Código en PyTorch paso a paso
* Entrenamiento, evaluación y predicción

### ✅ Paso 1: Librerías necesarias

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
```

### ✅ Paso 2: Datos de ejemplo

Supongamos que tienes estadísticas de partidos:

```python
# X = tiros al arco, posesión, pases, etc.
X = np.array([
    [5, 60, 300],
    [2, 45, 150],
    [8, 70, 400],
    [3, 40, 100],
    [6, 65, 280]
])

# y = 1 si ganó el equipo, 0 si no
y = np.array([1, 0, 1, 0, 1])
```

### ✅ Paso 3: Preprocesamiento

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

### ✅ Paso 4: Red neuronal

```python
class RedBinaria(nn.Module):
    def __init__(self):
        super(RedBinaria, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8),    # 3 features de entrada
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()        # Activación para clasificación binaria
        )

    def forward(self, x):
        return self.net(x)
```

### ✅ Paso 5: Entrenamiento

```python
modelo = RedBinaria()
criterio = nn.BCELoss()  # Binary Cross Entropy
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

# Entrenar
for epoch in range(200):
    modelo.train()
    salida = modelo(X_train)
    loss = criterio(salida, y_train)
    
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### ✅ Paso 6: Evaluación

```python
modelo.eval()
with torch.no_grad():
    pred = modelo(X_test)
    pred_labels = (pred >= 0.5).float()

accuracy = accuracy_score(y_test, pred_labels)
print(f"Accuracy: {accuracy:.2f}")
```

### ✅ ¿Qué puedes ajustar?

* Cantidad de capas o neuronas
* Activaciones (ReLU, Tanh)
* Métricas (F1, precisión, recall)
* Función de pérdida (por ejemplo `BCEWithLogitsLoss` sin `Sigmoid`)

### Resumen

Comprender el funcionamiento de las **redes neuronales artificiales (RNA)** es fundamental para avanzar en el área del aprendizaje profundo o *deep learning*. Estas redes, capaces de aprender patrones de forma automática, constituyen la base de muchos sistemas de inteligencia artificial, especialmente en aplicaciones deportivas como el fútbol. Usando PyTorch, una herramienta de programación y análisis, podemos definir, entrenar y evaluar nuestras propias redes neuronales con facilidad y precisión.

#### ¿Qué herramientas necesitamos para crear una red neuronal?

Para comenzar el trabajo con RNA en PyTorch, es fundamental contar con las bibliotecas adecuadas:

- **Torch**: proporciona soporte básico para tensores y autograd.
- **Torch NN**: incluye elementos esenciales para crear las diferentes capas de una red.
- **Torch Optim**: usado para actualizar los parámetros y pesos.
- **NumPy**: facilita la manipulación de matrices en Python.

Estas herramientas nos permiten definir redes, manejar datos y realizar entrenamiento supervisado de manera efectiva.

#### ¿Cómo crear y entrenar una red neuronal sencilla para clasificación binaria?

El proceso inicia con un dataset sintético sencillo:

- Creamos datos con 100 muestras y 4 características cada una.
- Convertimos estos datos y sus etiquetas en tensores, utilizando `Torch.from_numpy`.

Luego definimos una red neuronal compuesta por:

- Una capa oculta de 8 neuronas con activación ReLU.
- Una capa de salida con activación sigmoide, útil para problemas de clasificación binaria.

La pérdida en clasificación se mide mediante la función BCLoss, adecuada para este tipo de activación. Para actualizar los pesos y optimizar la red se utiliza Adam Optimizer, lo cual facilita la convergencia y mejora el rendimiento.

El entrenamiento implica calcular predicciones, evaluar errores, obtener gradientes y ajustar pesos automáticamente. Cada época del entrenamiento verifica la pérdida para determinar cómo progresa el aprendizaje.

#### ¿Cómo evaluar y ajustar la estructura de la red neuronal?

Para medir resultados, se transforma la salida de la red en predicciones binarias usando un umbral de 0.5 y se evalúa la precisión mediante el porcentaje de aciertos:

- Se desactivan los cálculos de gradientes para rapidez.
- Se obtiene la precisión final del modelo, observando cuántas predicciones acierta frente al total del dataset.

Además, se incluye una herramienta interactiva para modificar la arquitectura de red de manera dinámica:

- Podemos ajustar fácilmente el número de capas ocultas entre 1 y 5.
- Cada capa modifica cómo aprende y generaliza el modelo.
- Este ejercicio ilustra la importancia del balance entre la capacidad del modelo para aprender y evitar sobreajustes.

Experimentar con diferentes configuraciones te permitirá comprender claramente cómo se correlacionan la arquitectura de la red y su eficacia en los resultados prácticos del fútbol, al mejorar desde soluciones de detección de jugadas hasta el análisis detallado de imágenes y videos.
 
**Lecturas recomendadas**

[Tutorials  |  TensorFlow Core](https://www.tensorflow.org/tutorials "Tutorials  |  TensorFlow Core")

[PyTorch](https://pytorch.org/ "PyTorch")

[machine-learning/20_intro_redes_neuronales.ipynb at main · platzi/machine-learning · GitHub](https://github.com/platzi/machine-learning/blob/main/20_intro_redes_neuronales.ipynb "machine-learning/20_intro_redes_neuronales.ipynb at main · platzi/machine-learning · GitHub")

## Análisis de sentimientos en comentarios deportivos con NLP

El **análisis de sentimientos** con **NLP (Procesamiento de Lenguaje Natural)** es ideal para interpretar comentarios de fans, periodistas o redes sociales sobre eventos deportivos, jugadores o equipos.

### 🎯 ¿Qué es el análisis de sentimientos?

Es una técnica de NLP que **detecta la opinión emocional** detrás de un texto:

* **Positivo** → elogios, entusiasmo, apoyo
* **Negativo** → críticas, decepción
* **Neutral** → información objetiva o sin carga emocional

### 🛠️ Herramientas comunes para hacerlo en Python

* **NLTK / TextBlob** → fácil para empezar
* **Hugging Face Transformers** (modelos preentrenados como BERT)
* **scikit-learn** con TF-IDF y regresores
* **spaCy** para tareas de NLP general + extensiones

### ✅ Pipeline típico de análisis de sentimientos deportivo

### 1. 🧾 Recolectar comentarios

Ejemplo:

```python
comentarios = [
    "¡Qué gran partido jugó Messi!",
    "Fue una vergüenza el arbitraje.",
    "El equipo no mostró nada hoy.",
    "Increíble atajada del arquero.",
    "Un empate justo, buen nivel de ambos."
]
```

### 2. 🧽 Preprocesamiento (con `nltk` o `re`)

```python
import re

def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar signos
    return texto

comentarios_limpios = [limpiar(c) for c in comentarios]
```

### 3. 📦 Análisis rápido con `TextBlob`

```python
from textblob import TextBlob

for c in comentarios_limpios:
    blob = TextBlob(c)
    print(f"Comentario: {c}")
    print(f"Polaridad: {blob.sentiment.polarity:.2f} → {'Positivo' if blob.sentiment.polarity > 0 else 'Negativo' if blob.sentiment.polarity < 0 else 'Neutral'}")
    print()
```

### 🧠 ¿Qué hace TextBlob?

* `polarity`: valor entre -1 (negativo) y 1 (positivo)
* `subjectivity`: qué tan subjetivo u objetivo es el texto (opcional para otras tareas)

### 📈 ¿Y si quiero usar un modelo más potente como BERT?

```python
from transformers import pipeline

clasificador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

resultados = clasificador(comentarios)
for comentario, res in zip(comentarios, resultados):
    print(f"{comentario} → {res['label']}, score: {res['score']:.2f}")
```

Este modelo entrega predicciones del 1 al 5 🌟.

#3# 🔍 Aplicaciones en deportes

* 🏟️ **Monitorear reacciones** en tiempo real durante partidos
* 👥 **Evaluar percepción** de fans sobre jugadores o decisiones tácticas
* 📊 **Visualizar tendencias** emocionales en redes o foros
* 📢 **Segmentar audiencia** por tono de opinión

### Resumen

¿Te imaginas qué decisiones podrías tomar si supieras exactamente lo que sienten los fanáticos de tu equipo? Eso es lo que propone el procesamiento de lenguaje natural (NLP), una poderosa rama de la inteligencia artificial (IA) que permite a las máquinas entender, interpretar y analizar textos humanos, desde comentarios en redes hasta reportes de prensa.

#### ¿Qué es NLP y cómo puede aplicarse en análisis deportivos?

El NLP (*Natural Language Processing*) es una tecnología clave en sistemas conocidos como Siri, Google o ChatGPT. Gracias a esta tecnología, puedes extraer información clave de opiniones escritas por seguidores y medios, transformándolas en decisiones basadas en datos emocionales concretos.

En el contexto deportivo, esto significa:

- Medir la moral de la hinchada luego de partidos clave.
- Identificar críticas y alabanzas hacia distintas áreas del equipo, como la defensa o el ataque.
- Tomar decisiones estratégicas que estén conectadas con la realidad emocional del club.

#### ¿Cómo preparar los datos textuales para analizarlos con NLP?

La efectividad del análisis NPL depende fundamentalmente de cómo prepares tus datos. El proceso inicial es sencillo y directo:

1. **Carga de datos**: Importar tus comentarios deportivos desde archivos CSV utilizando pandas y asegurarte que cada comentario sea tratado como texto.
2. **Limpieza de texto**: Crear una función sencilla en Python que utilice expresiones regulares para:
3. Convertir todas las letras a minúsculas.
4. Eliminar espacios excesivos, signos de puntuación y caracteres especiales.
5. **Inspección inicial**: Visualizar los primeros resultados limpios para confirmar que el proceso fue exitoso.

Aquí un breve ejemplo en Python:

```python
import re

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Zñáéíóúü0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto
```

Esta función simplifica considerablemente los comentarios, volviéndolos más fáciles de analizar y comprender.

#### ¿Cómo visualizar fácilmente los resultados del análisis emocional?

Las representaciones visuales son herramientas potentes para entender rápidamente grandes volúmenes de información cualitativa:

- **Nube de palabras**: Generar gráficos visuales del vocabulario más repetido en los comentarios, permitiendo identificar fácilmente preocupaciones recurrentes o temas valiosos para el equipo técnico.
- **Distribución de sentimientos**: Representar gráficamente cuántos comentarios son positivos, negativos o neutros ayuda a detectar tendencias generales entre los seguidores.

Ejemplo para generar una nube de palabras:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, colormap='viridis').generate(texto_total)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

Utilizando Seaborn se genera una distribución visual de sentimientos:

```python
import seaborn as sns
sns.countplot(data=df_comentarios, x='sentimiento')
plt.title('Distribución de Sentimientos')
plt.xlabel('Sentimiento')
plt.ylabel('Frecuencia')
plt.show()
```

#### ¿Cómo explorar comentarios específicos mediante interactividad?

Integrar funciones interactivas permite examinar comentarios puntuales basados en su categoría emocional. Esto puede lograrse mediante widgets en Jupyter Notebook, facilitando la exploración cualitativa del contenido.

Un sencillo ejemplo de implementación interactiva:

```python
import ipywidgets as widgets
from IPython.display import display

seleccion = widgets.Dropdown(options=df_comentarios['sentimiento'].unique())

def mostrar_comentarios(categoria):
    display(df_comentarios[df_comentarios['sentimiento'] == categoria].sample(5))

widgets.interactive(mostrar_comentarios, categoria=seleccion)
```

Este enfoque mejora enormemente la interacción con los datos y aporta claridad en la toma de decisiones deportivas guiadas por las emociones reales de seguidores y prensa.
 
**Archivos de la clase**

[comentarios-deportivos.csv](https://static.platzi.com/media/public/uploads/comentarios_deportivos_a58295c0-8866-44e6-b80e-1ca5f7c2342b.csv "comentarios-deportivos.csv")

**Lecturas recomendadas**

[spaCy 101: Everything you need to know · spaCy Usage Documentation](https://spacy.io/usage/spacy-101 "spaCy 101: Everything you need to know · spaCy Usage Documentation")

[NLTK Book](https://www.nltk.org/book/ "NLTK Book")

[Rate limit · GitHub](https://github.com/platzi/machine-learning/blob/main/22_intro_nlp_deportivo.ipynb "Rate limit · GitHub")