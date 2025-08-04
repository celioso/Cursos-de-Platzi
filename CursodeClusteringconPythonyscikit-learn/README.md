# Curso de Clustering con Python y scikit-learn

## ¿Qué es el clustering en machine learning?

**Clustering** en *machine learning* es una técnica de **aprendizaje no supervisado** que consiste en **agrupar datos similares** entre sí en grupos llamados **clústers** (*clusters*, en inglés), **sin usar etiquetas o categorías predefinidas**.

### 🔍 ¿Qué significa eso?

* El algoritmo **no sabe a qué clase pertenece cada dato**.
* En lugar de eso, **descubre patrones o estructuras ocultas** en los datos.
* Los puntos que son **más similares entre sí** (según una medida de distancia o similitud) se agrupan en el mismo clúster.

### ✅ Ejemplos de uso de clustering:

* Agrupar clientes con comportamientos de compra similares (segmentación de mercado).
* Identificar patrones en imágenes o documentos.
* Detección de anomalías o fraudes.
* Agrupación de genes con funciones similares en biología.

### 🔧 Algoritmos comunes de clustering:

| Algoritmo                   | Característica principal                                                  |
| --------------------------- | ------------------------------------------------------------------------- |
| **K-Means**                 | Divide los datos en **K grupos** basados en distancias.                   |
| **DBSCAN**                  | Agrupa por **densidad de puntos** (detecta clústers de forma arbitraria). |
| **Hierarchical Clustering** | Crea una **jerarquía de clústers** (en forma de árbol o dendrograma).     |

### 🎯 Resumen:

> **Clustering** es un método de aprendizaje no supervisado que busca **descubrir grupos naturales** dentro de un conjunto de datos **sin necesidad de etiquetas previas**.

**Archivos de la clase**

[slides-curso-clustering-con-python-scikit-learn.pdf](https://static.platzi.com/media/public/uploads/slides-curso-clustering-con-python-scikit-learn_835791cf-0955-4487-8d59-987d91044699.pdf)

**Lecturas recomendadas**

[Curso de Introducción a Machine Learning por MindsDB - Platzi](https://platzi.com/cursos/machine-learning/)

## Tu primer clustering con scikit-learn

¡Perfecto! Aquí tienes un ejemplo básico de cómo hacer tu **primer clustering** usando **`scikit-learn`** en Python, utilizando el algoritmo **K-Means**, uno de los más comunes y fáciles de usar.

### 📦 Requisitos previos

Primero, asegúrate de tener las siguientes librerías instaladas:

```bash
pip install scikit-learn matplotlib seaborn
```

### ✅ Ejemplo básico: Clustering con K-Means

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns

# 1. Generamos datos de ejemplo
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

# 2. Aplicamos KMeans con 3 clústers
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 3. Obtenemos las predicciones
y_kmeans = kmeans.predict(X)

# 4. Visualizamos el resultado
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette="Set1", s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroides')
plt.title("Clustering con KMeans")
plt.legend()
plt.show()
```

### 🔍 ¿Qué hace este código?

1. **Crea datos sintéticos** con 3 clústers.
2. Aplica el algoritmo **K-Means** para agrupar los datos.
3. **Visualiza** los grupos asignados y los centroides de cada clúster.

### ¿Y luego?

Puedes intentar con tus propios datos o probar otros algoritmos de clustering como:

* `DBSCAN` (detecta clústers de forma irregular)
* `AgglomerativeClustering` (jerárquico)
* `MeanShift`

## ¿Cuándo usar clustering?

Puedes usar **clustering** cuando quieres **descubrir grupos naturales o estructuras ocultas** en tus datos **sin tener etiquetas previas**. Es una técnica de **aprendizaje no supervisado**, útil en muchas situaciones donde necesitas **explorar, segmentar o reducir complejidad**.

### 🧠 ¿Cuándo usar clustering? (Casos comunes)

#### 1. **Segmentación de clientes**

* Para agrupar clientes con comportamientos similares.
* Ejemplo: Marketing personalizado según hábitos de compra.

#### 2. **Análisis exploratorio de datos (EDA)**

* Para **detectar patrones desconocidos** antes de aplicar modelos supervisados.
* Útil para descubrir **subgrupos naturales**.

#### 3. **Agrupación de documentos o textos**

* Agrupar artículos, reseñas o noticias por tema sin etiquetarlos previamente.
* Ejemplo: Agrupar reseñas similares de productos.

#### 4. **Reducción de complejidad**

* Cuando tienes muchos datos y quieres entender su **estructura interna**.
* Puedes usar clustering como paso previo a modelos más complejos.

#### 5. **Detección de anomalías**

* Algunos outliers no pertenecen a ningún clúster y pueden ser **anomalías o fraudes**.

#### 6. **Agrupamiento de imágenes**

* Por similitud de colores, formas o patrones visuales.
* Ejemplo: Clasificar fotos similares en galerías.

#### 7. **Agrupación geográfica**

* Agrupar ubicaciones por cercanía (ej. zonas de entrega, clientes cercanos).

### ❗ Cuándo **NO** es recomendable

* Cuando **ya tienes etiquetas claras** para cada clase → usa modelos **supervisados**.
* Si los datos **no tienen patrones claros o separables**.
* Si necesitas resultados **explicables y consistentes**: algunos algoritmos de clustering pueden ser sensibles a la inicialización (como K-Means).

### 🧩 ¿Qué necesitas para aplicar clustering?

* Datos **sin etiquetas**.
* Alguna idea de **cuántos grupos** esperas (aunque hay algoritmos que lo infieren).
* Una **métrica de distancia o similitud** que tenga sentido en tu dominio (euclidiana, coseno, etc.).

## ¿Cómo evaluar modelos de clustering?

Evaluar modelos de **clustering** puede ser un reto porque **no hay etiquetas verdaderas** (en aprendizaje no supervisado), pero existen métricas **internas** y **externas** que nos ayudan a medir qué tan buenos son los grupos que genera el algoritmo.

### ✅ ¿Cómo evaluar un modelo de clustering?

### 🔹 1. **Métricas internas**

Usadas **cuando no hay etiquetas reales**. Evalúan la **cohesión** (qué tan compactos son los clústers) y la **separación** (qué tan distintos son entre sí).

| Métrica                     | ¿Qué mide?                                                   | Valor ideal        |
| --------------------------- | ------------------------------------------------------------ | ------------------ |
| **Silhouette Score**        | Qué tan cerca está cada punto de su propio clúster vs otros. | Cerca de **1**     |
| **Davies-Bouldin Index**    | Ratio de dispersión intra-clúster / distancia inter-clúster. | **Más bajo mejor** |
| **Calinski-Harabasz Index** | Variación entre clústers comparada con la interna.           | **Más alto mejor** |

### 🔹 2. **Métricas externas**

Usadas **cuando tienes etiquetas reales** (como en benchmarks).

| Métrica                                 | ¿Qué compara?                                           |
| --------------------------------------- | ------------------------------------------------------- |
| **Adjusted Rand Index (ARI)**           | Compara similitud entre clústers y etiquetas reales.    |
| **Normalized Mutual Information (NMI)** | Mide información compartida entre clústers y etiquetas. |
| **Fowlkes-Mallows Score**               | Evalúa precisión entre pares de puntos.                 |

### 🔹 3. **Visualización**

Aunque no es una métrica numérica, **visualizar los clústers** ayuda a:

* Ver si hay **solapamientos** o agrupaciones claras.
* Detectar **outliers**.
* Usar reducción de dimensiones como **PCA** o **t-SNE** para representar datos en 2D/3D.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print("Silhouette Score:", score)
```

### 🧠 Consejo:

Usa **Silhouette Score** cuando no tienes etiquetas, y si puedes comparar resultados con una verdad conocida, incluye también **ARI** o **NMI**.

**Lecturas recomendadas**

[Selecting the number of clusters with silhouette analysis on KMeans clustering — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

## ¿Qué es el algoritmo de K-means y cómo funciona?

El **algoritmo de K-Means** es uno de los algoritmos de **clustering más populares** en *machine learning no supervisado*. Su objetivo es **dividir un conjunto de datos en *K* grupos (clústers)**, donde cada grupo contiene puntos que son similares entre sí y diferentes de los de otros grupos.

### 🧠 ¿Qué hace K-Means?

Dado un número **K** (el número de clústers deseado), el algoritmo agrupa los datos de forma que se **minimice la distancia** de cada punto a su centroide (el "centro" del clúster).

### ⚙️ ¿Cómo funciona K-Means? (Pasos)

1. **Inicialización**:

   * Escoge **K puntos aleatorios** como **centroides iniciales** (uno por clúster).

2. **Asignación de clústers**:

   * Cada punto se asigna al clúster **más cercano** (usando, por ejemplo, distancia euclidiana).

3. **Actualización de centroides**:

   * Se recalcula el **centroide** de cada clúster como el **promedio** de los puntos asignados a él.

4. **Repetir**:

   * Repite los pasos 2 y 3 hasta que:

     * Los centroides **ya no cambian significativamente**, o
     * Se alcanza un número máximo de iteraciones.

### 📈 Ejemplo visual

Imagina un conjunto de puntos dispersos. K-Means:

* Coloca 3 puntos aleatorios como centroides (si K = 3).
* Agrupa los puntos según cercanía a esos centroides.
* Recalcula los centroides.
* Repite hasta que las posiciones de los centroides se estabilicen.

### ✅ Ventajas

* **Fácil de entender e implementar.**
* Funciona bien con clústers **esféricos y separados**.
* **Rápido** incluso con grandes conjuntos de datos.

### ⚠️ Limitaciones

* Tienes que **definir K** previamente.
* Es sensible a:

  * **Outliers**
  * **Inicialización aleatoria** (puede converger a soluciones subóptimas).
* Asume que los clústers son de forma redonda y de tamaño similar.

### 🔍 ¿Cuándo usarlo?

* Segmentación de clientes.
* Agrupamiento de imágenes.
* Análisis exploratorio de datos.

**Lecturas recomendadas** 

[Visualizing K-Means Clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

[Clustering comparison | Cartography Playground](https://cartography-playground.gitlab.io/playgrounds/clustering-comparison/)

## ¿Cuándo usar K-means?

**K-means** es un algoritmo de aprendizaje no supervisado usado principalmente para **clustering** (agrupamiento). Es útil cuando se desea agrupar elementos en subconjuntos similares **sin etiquetas previas**.

### 📌 Cuándo usar K-means:

1. **Cuando no hay etiquetas (unsupervised learning)**:
   Tienes datos sin categorías asignadas y deseas encontrar patrones naturales en ellos.

2. **Cuando esperas grupos esféricos y bien separados**:
   K-means funciona mejor cuando los clusters tienen forma circular o esférica (por la forma como calcula distancias).

3. **Cuando conoces o puedes estimar el número de clusters (k)**:
   Es necesario definir cuántos grupos esperas obtener.

4. **Cuando los datos no tienen muchos outliers**:
   K-means es sensible a valores atípicos porque usa medias para agrupar.

5. **Para segmentación de clientes, compresión de imágenes, detección de patrones, etc.**

### ❌ No se recomienda K-means cuando:

* Los clusters tienen formas irregulares o distintos tamaños.
* Hay muchos valores atípicos.
* No tienes idea del número adecuado de grupos (aunque puedes usar métodos como el codo/elbow para estimarlo).

## Implementando K-means

A continuación te muestro un ejemplo paso a paso de **cómo implementar K-means en Python** usando `scikit-learn`.

### ✅ **Paso 1: Importar librerías necesarias**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```

### ✅ **Paso 2: Generar datos de ejemplo**

```python
# Creamos un conjunto de datos sintético con 3 clusters
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Visualizamos los datos
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Datos sin clasificar")
plt.show()
```

### ✅ **Paso 3: Aplicar K-means**

```python
# Creamos y ajustamos el modelo
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Obtenemos las predicciones (a qué grupo pertenece cada punto)
y_kmeans = kmeans.predict(X)

# Centros de los clusters
centros = kmeans.cluster_centers_
```

### ✅ **Paso 4: Visualizar los resultados**

```python
# Graficamos los datos clasificados por cluster
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Mostramos los centros de cada cluster
plt.scatter(centros[:, 0], centros[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("Resultados del clustering con K-means")
plt.show()
```

### ✅ **Paso 5 (Opcional): Evaluar el número óptimo de clusters (método del codo)**

```python
inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()
```

**Archivos de la clase**

[clustering-kmeans.ipynb](https://static.platzi.com/media/public/uploads/clustering_kmeans_1aa7190f-894b-4ea5-b457-24fb419c7226.ipynb)

**Lecturas recomendadas**

[sklearn.cluster.KMeans — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

[clustering_kmeans.ipynb - Google Drive](https://drive.google.com/file/d/1MqZPEldivmSm0l3coh9APCDrA267sjKj/view?usp=sharing)

## Encontrando K

Para **encontrar el número óptimo de clusters (K)** al usar K-Means, existen varios métodos. Aquí te explico los principales y te doy ejemplos prácticos para que los implementes fácilmente:

### ✅ 1. **Elbow Method (Método del Codo)**

### ¿Qué hace?

Mide la **inercia** (dentro del grupo de errores cuadrados - SSE) para distintos valores de K y busca el "codo" donde la mejora se estabiliza.

### Código:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Datos simulados para el ejemplo
X, _ = make_blobs(n_samples=300, centers=5, random_state=42)

inertia = []

K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()
```

> 🔍 Busca el punto donde la curva se "dobla", como un codo.

### ✅ 2. **Silhouette Score**

### ¿Qué hace?

Mide qué tan bien separados están los clusters. Cuanto mayor el valor (cerca de 1), mejor separados y definidos están.

### Código:

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

K_range = range(2, 11)  # Comienza desde 2 porque Silhouette no está definido para k=1
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Puntaje de Silhouette')
plt.show()
```

> ✅ El valor óptimo de `k` es donde el **Silhouette Score es más alto**.

### ✅ 3. **Usando Yellowbrick (si lo logras instalar)**

```python
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(1,10), metric='silhouette', timings=False)
visualizer.fit(X)
visualizer.show()
```

### ¿Qué método elegir?

| Método           | Ideal cuando...                                                             |
| ---------------- | --------------------------------------------------------------------------- |
| Elbow            | Quieres ver el punto donde el "beneficio" se estabiliza                     |
| Silhouette Score | Quieres evaluar la calidad de los clusters                                  |
| Yellowbrick      | Quieres automatizar visualmente el análisis (requiere instalación correcta) |

**Archivos de la clase**

[clustering-kmeans.ipynb](https://static.platzi.com/media/public/uploads/clustering_kmeans_e52e3f11-b422-4522-86b8-693f0cd189b8.ipynb)

**Lecturas recomendadas**

[clustering_kmeans.ipynb - Google Drive](https://drive.google.com/file/d/1MqZPEldivmSm0l3coh9APCDrA267sjKj/view?usp=sharing)

## Evaluando resultados de K-means

Una vez que ejecutas el algoritmo K-Means y encuentras el valor óptimo de **K**, es fundamental **evaluar los resultados** para asegurarte de que los clusters tengan sentido. Aquí tienes **formas efectivas de evaluar K-Means**:

### ✅ 1. **Silhouette Score (Evaluación numérica)**

### ¿Qué mide?

* Qué tan **bien separados** y **cohesivos** están los clusters.
* Va de `-1` (malo) a `+1` (muy bueno).

### Código:

```python
from sklearn.metrics import silhouette_score

# Asumiendo que ya tienes `X` (datos) y `kmeans.labels_` (etiquetas)
score = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {score:.3f}")
```

> ✅ Un score cerca de 1 es excelente. Cerca de 0 indica clusters solapados. Negativo es mal clustering.

### ✅ 2. **Visualización de Clusters (2D)**

### ¿Para qué sirve?

* Visualmente valida si los grupos son distinguibles.
* Solo se puede usar si tienes 2 o 3 dimensiones, o aplicas reducción como PCA.

### Código:

```python
import matplotlib.pyplot as plt

# Asumiendo que usaste kmeans = KMeans(n_clusters=k)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroides')
plt.title("Visualización de Clusters K-Means")
plt.legend()
plt.show()
```

### ✅ 3. **Matriz de confusión (si tienes etiquetas reales)**

> Si estás haciendo clustering sobre datos **etiquetados (supervisado)**, puedes evaluar con exactitud y métricas de clasificación:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Supongamos que tienes y_true (etiquetas reales) y y_pred (kmeans.labels_)
sns.heatmap(confusion_matrix(y_true, kmeans.labels_), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Etiquetas de KMeans")
plt.ylabel("Etiquetas Reales")
plt.title("Matriz de Confusión")
plt.show()
```

### ✅ 4. **Silhouette Visualizer (Yellowbrick - opcional)**

```python
from yellowbrick.cluster import SilhouetteVisualizer

model = KMeans(n_clusters=k, random_state=42)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(X)
visualizer.show()
```

> 📌 Esto muestra la forma y consistencia de cada cluster.

### ✅ 5. **Comparación Intra vs Inter Clúster**

* **Intra-cluster distance**: qué tan compactos son los puntos de cada cluster.
* **Inter-cluster distance**: qué tan separados están los centroides.

Si lo deseas, puedo ayudarte a programar un pequeño análisis de estas métricas.

## ¿Qué es hierarchical clustering y cómo funciona?

El **Hierarchical Clustering** (o **clustering jerárquico**) es un algoritmo de agrupamiento **no supervisado** que construye una jerarquía de clusters, es decir, una estructura en forma de **árbol** que muestra cómo se agrupan los datos.

### 🔍 ¿Cómo funciona?

Existen dos enfoques principales:

#### 1. **Agglomerative (aglomerativo)** – de abajo hacia arriba 👇

* **El más común**.
* Cada punto de datos comienza como su propio cluster.
* En cada paso, los **dos clusters más cercanos se fusionan**.
* Esto se repite hasta que todos los puntos están en un solo cluster o se cumple un criterio.

#### 2. **Divisive (divisivo)** – de arriba hacia abajo 👆

* Todos los datos comienzan en un único gran cluster.
* En cada paso, se **divide el cluster más grande** hasta que cada punto es su propio cluster.

### 🧠 ¿Cómo decide qué clusters unir o dividir?

El algoritmo usa una **métrica de distancia** entre clusters. Algunas comunes son:

| Método     | Qué mide                                                      |
| ---------- | ------------------------------------------------------------- |
| `single`   | Distancia mínima entre puntos de clusters                     |
| `complete` | Distancia máxima entre puntos de clusters                     |
| `average`  | Promedio de todas las distancias                              |
| `ward`     | Minimiza la varianza total dentro de los clusters (muy usada) |

### 📈 Representación: Dendrograma

El resultado se visualiza como un **dendrograma**, un diagrama en forma de árbol que muestra cómo se fusionan (o dividen) los clusters a medida que cambia la distancia.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

linked = linkage(X, method='ward')  # o 'single', 'average', etc.
dendrogram(linked)
plt.title("Dendrograma de Clustering Jerárquico")
plt.xlabel("Puntos de datos")
plt.ylabel("Distancia")
plt.show()
```

### ✅ ¿Cuándo usarlo?

* Cuando quieres **ver la estructura jerárquica** de los datos.
* Cuando el número de clusters **no es conocido de antemano**.
* Para datos con **pocos puntos** (porque no escala bien con muchos datos).

### ❌ Desventajas

* **Lento** en datasets grandes (`O(n^3)`).
* Puede ser **sensible a ruido y outliers**.
* No permite reorganizar clusters una vez creados.

## ¿Cuándo usar hierarchical clustering?

Usar **clustering jerárquico** (`Hierarchical Clustering`) es recomendable en los siguientes escenarios:

### ✅ **CUÁNDO USARLO**

#### 1. **Cuando no sabes cuántos clusters hay**

* El dendrograma te permite **explorar la estructura** y **elegir el número de clusters visualmente** (cortando el árbol).
* Ideal para **descubrimiento exploratorio**.

#### 2. **Cuando quieres una visión jerárquica de los datos**

* Si te interesa **ver relaciones padre-hijo** entre grupos (subgrupos dentro de grupos más grandes), este método es ideal.
* Ejemplo: taxonomía biológica, estructura de carpetas, segmentación de clientes multinivel.

#### 3. **Para datasets pequeños o medianos**

* Funciona bien con **menos de \~1,000-5,000 puntos**. Más allá de eso, puede volverse muy lento y demandante en memoria.
* Es mejor para análisis en profundidad que para producción a gran escala.

#### 4. **Cuando los clusters no son esféricos ni del mismo tamaño**

* A diferencia de K-means, que asume clusters circulares del mismo tamaño, el clustering jerárquico **no impone esa suposición**.

#### 5. **Cuando necesitas interpretar los resultados**

* El dendrograma es **intuitivo y visualmente explicativo**, muy útil en reportes o análisis descriptivos.

### ❌ **CUÁNDO EVITARLO**

| Situación                                | Por qué evitarlo                                                      |
| ---------------------------------------- | --------------------------------------------------------------------- |
| Dataset muy grande                       | Tiene **complejidad O(n³)** y **memoria O(n²)**.                      |
| Necesitas clasificar nuevos datos rápido | No es **incremental** ni rápido para nuevos puntos.                   |
| Tienes muchos outliers                   | Puede generar **clusters distorsionados** si no hay preprocesamiento. |

### 🔄 Alternativas en esos casos

| Necesidad                            | Alternativa                    |
| ------------------------------------ | ------------------------------ |
| Escalabilidad                        | K-means, MiniBatchKMeans       |
| Detección de formas arbitrarias      | DBSCAN, HDBSCAN                |
| Robustez a ruido                     | DBSCAN                         |
| Clasificación rápida de nuevos datos | K-means o modelos supervisados |

## Implementando hierarchical clustering

Aquí tienes una implementación completa de **Hierarchical Clustering (Clustering Jerárquico)** con `scikit-learn`, usando un dataset sintético y visualizando tanto los clusters como el dendrograma:

### ✅ PASO A PASO: Hierarchical Clustering con `scikit-learn`

#### 1. Instalar librerías (si no las tienes):

```bash
pip install matplotlib seaborn scipy scikit-learn
```
#### 2. Código completo:

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Configurar estilo gráfico
sns.set(style="whitegrid")

# Generar datos artificiales
X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=1.0, random_state=42)

# Estandarizar (muy recomendable)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear y ajustar modelo de clustering jerárquico
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
clusters = hc.fit_predict(X_scaled)

# Visualizar los clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Clusters jerárquicos")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()
```

#### 3. Visualizar el dendrograma (opcional, pero muy útil):

```python
# Calcular matriz de enlaces
linked = linkage(X_scaled, method='ward')

# Graficar dendrograma
plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode='level', p=5)  # p=5 muestra los niveles superiores
plt.title("Dendrograma")
plt.xlabel("Puntos de datos")
plt.ylabel("Distancia")
plt.show()
```

### 🔍 Opciones importantes en `AgglomerativeClustering`:

| Parámetro    | Descripción                                                     |
| ------------ | --------------------------------------------------------------- |
| `n_clusters` | Número de clusters finales                                      |
| `linkage`    | Tipo de enlace: `'ward'`, `'complete'`, `'average'`, `'single'` |
| `affinity`   | Métrica de distancia (solo si `linkage ≠ 'ward'`)               |

**Archivos de la clase**

[hierachical-clustering.ipynb](https://static.platzi.com/media/public/uploads/hierachical_clustering_e27a7035-5973-4cf5-be54-b7d27bd9f351.ipynb)

**Lecturas recomendadas**

[sklearn.cluster.AgglomerativeClustering — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

[hierachical_clustering.ipynb - Google Drive](https://drive.google.com/file/d/1luOzJwPqgSfEz2PHTg3PufcOyHP0imnO/view?usp=sharing)

## Evaluando resultados de hierarchical clustering

Para **evaluar los resultados del clustering jerárquico (hierarchical clustering)**, puedes aplicar métricas internas similares a las usadas en otros algoritmos como K-means. Aquí te muestro cómo hacerlo y con qué herramientas:

### 📊 **1. Silhouette Score**

Mide qué tan bien están separados los clústeres:

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, y_hc)
print(f"Silhouette Score: {score:.3f}")
```

* **Rango:** de -1 a 1. Mientras más cercano a 1, mejor definidos están los grupos.

### 🧮 **2. Calinski-Harabasz Index**

Mide la dispersión entre clústeres frente a la dispersión dentro del clúster:

```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X_scaled, y_hc)
print(f"Calinski-Harabasz Index: {score:.2f}")
```

* **Interpretación:** cuanto mayor el valor, mejor separados están los clústeres.

### 🔢 **3. Davies-Bouldin Index**

Mide la similitud entre clústeres; valores más bajos son mejores:

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X_scaled, y_hc)
print(f"Davies-Bouldin Index: {score:.3f}")
```

### 🌳 **4. Visualización con dendrograma**

Si usas `scipy`, puedes generar un dendrograma para inspección visual:

```python
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrograma')
plt.xlabel('Muestras')
plt.ylabel('Distancia')
plt.show()
```

### ✅ Conclusión

| Métrica                 | ¿Qué mide?                    | Ideal       |
| ----------------------- | ----------------------------- | ----------- |
| Silhouette Score        | Separación y compactación     | Cercano a 1 |
| Calinski-Harabasz Index | Dispersión entre/intra grupos | Alto        |
| Davies-Bouldin Index    | Similitud entre clústeres     | Bajo        |

**Lecturas recomendadas**

[Selecting the number of clusters with silhouette analysis on KMeans clustering — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

[hierachical_clustering.ipynb - Google Drive](https://drive.google.com/file/d/1luOzJwPqgSfEz2PHTg3PufcOyHP0imnO/view?usp=sharing)

## ¿Qué es DBSCAN y cómo funciona?

**DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) es un algoritmo de clustering basado en densidad. A diferencia de K-means o el clustering jerárquico, **no necesitas especificar el número de clústeres de antemano**, y es muy eficaz para encontrar **clústeres de forma arbitraria y detectar ruido (outliers)**.

### 🔧 ¿Cómo funciona DBSCAN?

Se basa en dos parámetros principales:

1. **ε (epsilon):** el radio para considerar vecinos cercanos.
2. **minPts:** el número mínimo de puntos para formar un clúster denso.

### 🧱 Clasificación de puntos

DBSCAN clasifica los puntos en tres tipos:

* **Punto núcleo:** tiene al menos `minPts` vecinos dentro de un radio `ε`.
* **Punto frontera:** está dentro del radio `ε` de un punto núcleo, pero no tiene suficientes vecinos para ser núcleo.
* **Ruido (outlier):** no es núcleo ni frontera.

### 🔄 Algoritmo paso a paso:

1. Elige un punto no visitado.
2. Si tiene suficientes vecinos dentro de `ε`, crea un nuevo clúster.
3. Expande el clúster agregando todos los puntos densamente conectados.
4. Si no tiene suficientes vecinos, márcalo como **ruido**.
5. Repite hasta visitar todos los puntos.

### 🟢 Ventajas de DBSCAN

* No necesita saber el número de clústeres.
* Puede detectar **formas complejas** y **outliers**.
* Robusto al ruido.

### 🔴 Desventajas

* Difícil de elegir los parámetros óptimos `ε` y `minPts`.
* No funciona bien si los clústeres tienen **densidades muy diferentes**.
* Menos eficiente en conjuntos de datos muy grandes o de alta dimensión.

### 📌 Ejemplo visual

Supón un conjunto de datos con dos clústeres curvados y algo de ruido. K-means probablemente divida mal los clústeres porque supone formas circulares. DBSCAN, en cambio, los detecta correctamente y marca el ruido.

**Lecturas recomendadas**

[Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

## ¿Cuándo usar DBSCAN?

Puedes usar **DBSCAN** cuando:

### ✅ **1. No conoces el número de clústeres de antemano**

* DBSCAN detecta automáticamente cuántos clústeres hay (siempre que la densidad lo permita).
* Ideal para exploración de datos sin conocimiento previo.

### ✅ **2. Quieres detectar *outliers* o ruido**

* DBSCAN clasifica naturalmente los puntos aislados como **ruido**, lo que es útil en análisis de anomalías o fraudes.

### ✅ \**3. Tus clústeres tienen formas **no circulares** o **arbitrarias***

* A diferencia de K-means, que funciona bien solo con formas esféricas, DBSCAN puede detectar clústeres **curvados, alargados o irregulares**.

### ✅ **4. Tus datos tienen densidades consistentes**

* Si los clústeres tienen **densidades similares**, DBSCAN puede separarlos bien.

### 🚫 **Cuándo **no** usar DBSCAN**

* Si los clústeres tienen **densidades muy diferentes** → DBSCAN puede fallar en separar bien los grupos.
* En **altas dimensiones**, la distancia euclidiana pierde significado → puede que `ε` no funcione bien.
* Si necesitas explicaciones claras de los clústeres → K-means o clustering jerárquico pueden ser más interpretables.

### 📌 Ejemplos de aplicación:

* Detección de fraudes bancarios (puntos atípicos).
* Análisis de tráfico GPS (trayectorias densas vs. aisladas).
* Agrupación de tweets o noticias similares (previo uso de reducción de dimensionalidad).
* Imágenes satelitales donde los objetos tienen formas irregulares.

## Implementando DBSCAN

Aquí tienes una implementación práctica de **DBSCAN** en Python usando `scikit-learn`, con visualización incluida:

### ✅ **Paso 1: Cargar y preparar los datos**

Usaremos datos simulados con formas no circulares para mostrar la ventaja de DBSCAN.

```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Crear un conjunto de datos de ejemplo (dos medias lunas)
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# Visualizar
plt.scatter(X[:, 0], X[:, 1])
plt.title("Datos de ejemplo: dos medias lunas")
plt.show()
```

### ✅ **Paso 2: Aplicar DBSCAN**

```python
from sklearn.cluster import DBSCAN

# Crear modelo DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)  # Puedes ajustar estos parámetros

# Ajustar modelo
y_dbscan = dbscan.fit_predict(X)

# Visualizar resultado
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='plasma')
plt.title("Clústeres encontrados con DBSCAN")
plt.show()
```

### ⚙️ **Parámetros importantes:**

* `eps`: distancia máxima entre dos puntos para que uno sea considerado vecino del otro.
* `min_samples`: número mínimo de puntos para formar un clúster denso (incluye el punto central).

### ✅ **Paso 3 (opcional): Identificar *outliers***

```python
# Los puntos con etiqueta -1 son considerados ruido
import numpy as np

outliers = X[y_dbscan == -1]

plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='plasma')
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Outliers')
plt.legend()
plt.title("DBSCAN con detección de outliers")
plt.show()
```

**Archivos de la clase**

[dbscan.ipynb](https://static.platzi.com/media/public/uploads/dbscan_f460079b-4fed-429f-9fdd-58b6aad8eb9b.ipynb)

**Lecturas recomendadas**

[sklearn.cluster.DBSCAN — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

[dbscan.ipynb - Google Drive](https://drive.google.com/file/d/1e0LRhK3k00yxHZy5Q9eXS0jTU0GdnYTq/view?usp=sharing)

## Encontrar híper-parámetros

Para encontrar los **hiperparámetros óptimos de DBSCAN**, especialmente `eps` (radio de vecindad), podemos usar un gráfico de **distancias de los vecinos más cercanos**. Esto ayuda a detectar el valor de `eps` a partir de una "rodilla" o cambio brusco en la curva.

### ✅ **Paso 1: Graficar las distancias del vecino más cercano**

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Ajustar vecinos
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Ordenar las distancias del 4º vecino (porque n_neighbors=5)
distances = np.sort(distances[:, 4])  # el índice es n_neighbors - 1

# Graficar
plt.plot(distances)
plt.ylabel("Distancia al 4º vecino más cercano")
plt.xlabel("Puntos ordenados por distancia")
plt.title("Curva de codo para encontrar eps")
plt.grid(True)
plt.show()
```

🧠 **Interpretación**: Busca el punto donde la pendiente del gráfico cambia bruscamente ("codo"). Ese valor es una buena estimación para `eps`.

### ✅ **Paso 2: Aplicar DBSCAN con eps hallado**

Supón que el gráfico muestra un codo en `eps ≈ 0.25`:

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.25, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Visualizar
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='plasma')
plt.title("DBSCAN con eps ajustado")
plt.show()
```

### ✅ **Paso 3 (opcional): Evaluar calidad con Silhouette Score**

```python
from sklearn.metrics import silhouette_score

# Asegúrate de que haya más de 1 clúster
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
if n_clusters > 1:
    score = silhouette_score(X, y_dbscan)
    print(f"Silhouette Score: {score:.3f}")
else:
    print("DBSCAN no encontró suficientes clústeres para calcular Silhouette Score.")
```

**Lecturas recomendadas**

[sklearn.cluster.DBSCAN — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

[dbscan.ipynb - Google Drive](https://drive.google.com/file/d/1e0LRhK3k00yxHZy5Q9eXS0jTU0GdnYTq/view?usp=sharing)

## Evaluando resultados de DBSCAN

Para **evaluar los resultados de DBSCAN**, puedes utilizar diversas métricas y visualizaciones. Aquí te explico los enfoques más comunes:

### ✅ 1. **Visualización de los clústeres**

La forma más directa de evaluar DBSCAN es visualizar los clústeres:

```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='plasma')
plt.title("Resultados de DBSCAN")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.grid(True)
plt.show()
```

* Los **ruidos** serán etiquetados como `-1`.
* Colores diferentes representan diferentes clústeres.

### ✅ 2. **Silhouette Score**

Evalúa qué tan bien está cada punto dentro de su clúster:

```python
from sklearn.metrics import silhouette_score

labels = y_dbscan
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

if n_clusters > 1:
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.3f}")
else:
    print("No se pueden calcular métricas: hay menos de 2 clústeres.")
```

> 🎯 **Interpretación**: Cuanto más cercano a 1, mejor. Valores < 0 indican mala asignación.

### ✅ 3. **Número de clústeres y ruido**

Puedes revisar cuántos clústeres encontró DBSCAN y cuántos puntos consideró como ruido:

```python
import numpy as np

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Número de clústeres encontrados: {n_clusters}")
print(f"Número de puntos de ruido: {n_noise}")
```

### ✅ 4. **Confusión con etiquetas reales (si existen)**

Si tienes etiquetas verdaderas (`y_true`), puedes usar métricas como *Adjusted Rand Index (ARI)* o *Homogeneity Score*:

```python
from sklearn.metrics import adjusted_rand_score, homogeneity_score

print("ARI:", adjusted_rand_score(y_true, y_dbscan))
print("Homogeneidad:", homogeneity_score(y_true, y_dbscan))
```

### ✅ 5. **Silhouette Visualizer (opcional)**

Si tienes instalado `yellowbrick`, puedes usar un gráfico de silueta:

```python
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import DBSCAN

visualizer = SilhouetteVisualizer(DBSCAN(eps=0.3, min_samples=5))
visualizer.fit(X)
visualizer.show()
```

> Asegúrate de que haya más de un clúster para que esto funcione.

**Lecturas recomendadas**

[Selecting the number of clusters with silhouette analysis on KMeans clustering — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

[dbscan.ipynb - Google Drive](https://drive.google.com/file/d/1e0LRhK3k00yxHZy5Q9eXS0jTU0GdnYTq/view?usp=sharing)