# Curso de Decision Trees y Random Forest con Python y scikit-learn

## ¿Qué son los árboles de decisión?

Los **árboles de decisión** son modelos de aprendizaje supervisado que se utilizan para resolver problemas de **clasificación** y **regresión**. Su estructura se asemeja a un árbol, donde cada **nodo interno** representa una pregunta o condición sobre una característica (feature), cada **rama** representa el resultado de esa condición, y cada **hoja** representa una predicción final (una clase o un valor numérico).

### 🔍 ¿Cómo funcionan?

1. **División del conjunto de datos**:
   En cada nodo, el algoritmo selecciona la característica que mejor divide los datos según algún criterio (como Gini, Entropía o MSE).

2. **Construcción del árbol**:
   El proceso se repite de forma recursiva dividiendo el conjunto en subconjuntos hasta que:

   * Todos los datos en un nodo pertenecen a la misma clase.
   * Se alcanza una profundidad máxima.
   * Otras condiciones de parada.

3. **Predicción**:
   Para predecir con un árbol, se sigue una ruta desde la raíz hasta una hoja, tomando decisiones según los valores de entrada.

### ✳️ Ventajas

* Fácil de entender e interpretar.
* No requiere escalado de variables.
* Puede manejar datos tanto categóricos como numéricos.
* Permite visualizar cómo se toman las decisiones.

### ⚠️ Desventajas

* Pueden **sobreajustarse** fácilmente si no se podan.
* Sensibles a pequeñas variaciones en los datos.
* No suelen ser tan precisos como modelos más complejos (aunque se pueden combinar en **Random Forests** o **Gradient Boosting**).

### 🔧 Ejemplo en Python con Scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
```

**Archivos de la clase**

[decision-trees-random-forest-slides.pdf](https://static.platzi.com/media/public/uploads/decision-trees-random-forest-slides_a67df04d-0c2a-45ec-9ff0-68222424cc81.pdf)

## Tu primer árbol de decisión con scikit-learn

Aquí tienes un ejemplo completo de **tu primer árbol de decisión con Scikit-learn**, usando el clásico conjunto de datos **Iris**. Es ideal para aprender porque tiene 4 características y 3 clases de flores.

### 🌱 Paso a paso: Árbol de Decisión en Scikit-learn

### 1. 📦 Importar librerías necesarias

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
```

### 2. 🌸 Cargar dataset Iris

```python
iris = load_iris()
X = iris.data
y = iris.target
```

### 3. ✂️ Dividir en entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 4. 🌳 Crear y entrenar el árbol

```python
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
```

### 5. 🔮 Hacer predicciones

```python
y_pred = modelo.predict(X_test)
```

### 6. ✅ Evaluar el modelo

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 7. 📈 Visualizar el árbol de decisión

```python
plt.figure(figsize=(12, 8))
plot_tree(modelo, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
```

### 🧠 ¿Qué hace este árbol?

* Separa las flores (Setosa, Versicolor, Virginica) según el largo y ancho del sépalo y del pétalo.
* En cada nodo, hace una pregunta como “¿petal length (cm) <= 2.45?” para clasificar.

**Archivos de la clase**

[primer-arbol-decision-template.ipynb](https://static.platzi.com/media/public/uploads/primer_arbol_decision_template_84c03d57-915e-4533-852c-f596f47c81e8.ipynb)
[primer-arbol-decision-completed.ipynb](https://static.platzi.com/media/public/uploads/primer_arbol_decision_completed_c6330ed6-aa2e-4c32-b499-2480b6a5b78f.ipynb)

**Lecturas recomendadas**

[CS109](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)

[primer_arbol_decision_completed.ipynb - Google Drive](https://drive.google.com/file/d/1t8iSZcNn5Th9I7B_FrzrVXh-68jPwsol/view?usp=sharing)

[primer_arbol_decision_template.ipynb - Google Drive](https://drive.google.com/file/d/1wsxzoUcyLHCJ55YrFW5wMHqYhthWuckR/view?usp=sharing)

## Análisis de datos para tu primer árbol de decisión

Aquí tienes un análisis básico de los datos antes de construir tu primer árbol de decisión. Usaremos el dataset **Iris** como ejemplo, que es ideal para comenzar porque:

* Es pequeño y limpio.
* Tiene 150 observaciones.
* Tiene 4 características numéricas.
* Su objetivo (target) es predecir el tipo de flor (Setosa, Versicolor o Virginica).

### 🔍 1. Cargar y entender los datos

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

X['species'] = y
X.head()
```

**Salida esperada (primeras filas):**

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | species |
| ----------------- | ---------------- | ----------------- | ---------------- | ------- |
| 5.1               | 3.5              | 1.4               | 0.2              | 0       |
| 4.9               | 3.0              | 1.4               | 0.2              | 0       |
| ...               | ...              | ...               | ...              | ...     |

* `species`: 0 = Setosa, 1 = Versicolor, 2 = Virginica

### 📊 2. Estadísticas descriptivas

```python
X.describe()
```

Esto te dará un resumen de los valores mínimos, máximos, medias, etc. Sirve para:

* Ver si hay valores extremos (outliers).
* Ver si las escalas son muy distintas.
* Observar la distribución de cada variable.

### 📈 3. Distribución de clases

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='species', data=X)
plt.title('Distribución de clases')
plt.xticks([0, 1, 2], iris.target_names)
plt.show()
```

### 🔗 4. Correlaciones

```python
sns.heatmap(X.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.show()
```

Esto ayuda a ver qué características están más relacionadas entre sí.

### 🧠 5. Observaciones clave

* **Setosa** es fácil de separar por su pétalo corto y delgado.
* Las clases **Versicolor** y **Virginica** son más parecidas.
* Las características más importantes suelen ser el **largo y ancho del pétalo**.

**Archivos de la clase**

[primer-arbol-decision-template.ipynb](https://static.platzi.com/media/public/uploads/primer_arbol_decision_template_84c03d57-915e-4533-852c-f596f47c81e8.ipynb)
[primer-arbol-decision-completed.ipynb](https://static.platzi.com/media/public/uploads/primer_arbol_decision_completed_c6330ed6-aa2e-4c32-b499-2480b6a5b78f.ipynb)

**Lecturas recomendadas**

[CS109](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)

[Curso de Análisis Exploratorio de Datos - Platzi](https://platzi.com/cursos/analisis-exploratorio-datos/)

[primer_arbol_decision_completed.ipynb - Google Drive](https://drive.google.com/file/d/1t8iSZcNn5Th9I7B_FrzrVXh-68jPwsol/view?usp=sharing)

[primer_arbol_decision_template.ipynb - Google Drive](https://drive.google.com/file/d/1wsxzoUcyLHCJ55YrFW5wMHqYhthWuckR/view?usp=sharing)

## Entrenamiento y evaluación de árbol de decisión con scikit-learn

Aquí tienes un ejemplo **completo** y **explicado paso a paso** de cómo entrenar y evaluar un **árbol de decisión** usando `scikit-learn` con el dataset del **Titanic**:

### 🧪 1. Cargar y preparar los datos

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el dataset (puede ser desde seaborn, kaggle o archivo local)
titanic = pd.read_csv('titanic.csv')  # o usa seaborn.load_dataset('titanic')

# Selección de variables
features = ['Pclass', 'Sex', 'Age', 'Fare']
target = 'Survived'

# Eliminar nulos simples (sólo para simplificar el ejemplo)
titanic = titanic.dropna(subset=features + [target])

# Convertir variables categóricas a numéricas
titanic = pd.get_dummies(titanic, columns=['Sex'], drop_first=True, dtype=int)

# Variables predictoras y variable objetivo
X = titanic[['Pclass', 'Age', 'Fare', 'Sex_male']]
y = titanic[target]
```

### 🧠 2. Dividir en conjunto de entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 🌲 3. Entrenar el árbol de decisión

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 📊 4. Evaluación del modelo

```python
# Predicciones
y_pred = model.predict(X_test)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### 🌳 5. (Opcional) Visualizar el árbol

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
```

### ✅ Resultado

Con esto obtendrás:

* Un modelo de árbol de decisión entrenado.
* Métricas de precisión (`accuracy`, `precision`, `recall`, `f1-score`).
* Visualización clara de cómo el árbol toma decisiones.

## ¿Cómo funcionan los árboles de decisión?

Los **árboles de decisión** son algoritmos de aprendizaje supervisado que se utilizan para **clasificación** o **regresión**. Funcionan dividiendo los datos en ramas basadas en preguntas simples sobre las características (features), hasta llegar a una predicción.

### 🌳 ¿Cómo funciona un árbol de decisión?

1. **Inicio (nodo raíz)**:
   El árbol comienza con todos los datos en un nodo inicial.

2. **División (nodos internos)**:
   Se elige la **característica (feature)** que mejor separa los datos según un criterio (por ejemplo, **entropía**, **índice Gini** o **reducción de varianza**).

3. **Ramas**:
   Se crean ramas según los valores de esa característica. Cada rama lleva a un subconjunto del dataset.

4. **Repetición**:
   Este proceso se repite recursivamente en cada subconjunto, formando un árbol.

5. **Fin (hojas)**:
   Cuando no se puede dividir más (por ejemplo, los datos están completamente separados o se llega a un límite de profundidad), se hace una predicción basada en la mayoría de clase (clasificación) o en el promedio (regresión).

### 🧠 Ejemplo simple (Clasificación):

**¿Sobrevivió una persona en el Titanic?**
Variables:

* Edad
* Sexo
* Clase del boleto

El árbol podría hacer:

* **¿Sex\_female == 1?**

  * Sí → ¿Pclass <= 2?

    * Sí → Probabilidad alta de sobrevivir
    * No → Probabilidad media
  * No → ¿Age <= 10?

    * Sí → Probabilidad media
    * No → Probabilidad baja

### ⚙️ Criterios de división comunes:

* **Gini** (por defecto en `sklearn`): mide impureza
* **Entropía**: mide la cantidad de información necesaria para clasificar
* **MSE** (para regresión): error cuadrático medio

### ✅ Ventajas:

* Fácil de entender e interpretar
* No requiere normalización de datos
* Puede trabajar con variables categóricas y numéricas

### ❌ Desventajas:

* Puede sobreajustar (overfitting) si no se poda o se limita la profundidad
* Poca estabilidad: cambios pequeños en los datos pueden cambiar mucho el árbol

## ¿Cuándo usar árboles de decisión?

Debes considerar usar **árboles de decisión** cuando:

### ✅ **1. Quieres interpretabilidad**

* Los árboles son **fáciles de visualizar y entender**. Puedes explicar decisiones con reglas simples del tipo “si... entonces...”.
* Ideal cuando necesitas **explicar el modelo a personas no técnicas**.

### ✅ **2. Tus datos incluyen variables categóricas y numéricas**

* Los árboles manejan bien **ambos tipos** sin necesidad de normalización ni transformación compleja.

### ✅ **3. Tienes relaciones no lineales**

* A diferencia de la regresión lineal, los árboles **capturan interacciones y no linealidades** entre variables automáticamente.

### ✅ **4. Quieres saber qué variables son más importantes**

* El modelo calcula automáticamente **importancia de características**, lo cual es útil para **selección de variables** o interpretación.

### ✅ **5. Los datos tienen valores faltantes o están mal escalados**

* Los árboles son **resistentes** a valores faltantes (algunos algoritmos los manejan bien) y **no necesitan normalización**.

### ✅ **6. Tu problema es de clasificación o regresión**

* Puedes usar árboles para:

  * **Clasificación** (ej. detección de spam, predicción de enfermedad)
  * **Regresión** (ej. predicción de precios, demanda, consumo)

### ❌ **¿Cuándo evitar árboles de decisión?**

* Cuando necesitas **altísima precisión**: suelen tener peor desempeño que métodos como Random Forest o XGBoost.
* Cuando el dataset es **muy pequeño y complejo**: puede sobreajustar.
* Cuando necesitas **predicciones muy estables**: los árboles simples pueden variar bastante ante pequeños cambios en los datos.

### 📌 En resumen:

Usa árboles de decisión cuando necesitas un modelo **rápido, interpretable y versátil**, especialmente en las primeras etapas del análisis o cuando el entendimiento del modelo es prioritario.

**Lecturas recomendadas**

[Árbol de decisión en valoración de inversiones | 2023 | Economipedia](https://economipedia.com/definiciones/arbol-de-decision-en-valoracion-de-inversiones.html)

## Conociendo problema a resolver y dataset de clasificación

Para **resolver un problema de clasificación** con **machine learning**, como identificar el tipo de automóvil (por ejemplo: **fósil**, **eléctrico** o **híbrido**), necesitas tener claro dos cosas fundamentales:

### ✅ 1. **Conocer el problema a resolver**

Esto implica entender:

* Qué **queremos predecir** → En este caso, el **tipo de automóvil**.
* Qué **tipo de problema es** → Es un **problema de clasificación multiclase**.
* Qué datos tenemos disponibles para predecir → Variables como peso, potencia, tipo de motor, consumo, emisiones, etc.

### ✅ 2. **Conocer y preparar el dataset**

Esto incluye:

#### a. **Variable objetivo (target)**

Es la que vamos a predecir, en este caso:

```python
'y_tipo'  → ['fósil', 'eléctrico', 'híbrido']
```

Esta variable debe ser **convertida a valores numéricos**, por ejemplo usando `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['tipo_auto'] = le.fit_transform(df['tipo_auto'])
```

Esto puede traducir:

* fósil → 0
* eléctrico → 1
* híbrido → 2

#### b. **Variables predictoras (features)**

Estas son las columnas que usará el modelo para aprender. Por ejemplo:

```python
X = df[['peso', 'potencia', 'consumo', 'emisiones']]
y = df['tipo_auto']  # variable objetivo codificada
```

### ✅ 3. **División del dataset**

Siempre se divide el dataset en datos de entrenamiento y prueba:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ 4. **Entrenamiento del modelo**

Por ejemplo, con un árbol de decisión:

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
```

### ✅ 5. **Evaluación**

Evaluamos qué tan bien predice:

```python
from sklearn.metrics import accuracy_score

y_pred = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))
```

**Archivos de la clase**

[decision-tree-random-forest-project-template.ipynb](https://static.platzi.com/media/public/uploads/decision_tree_random_forest_project_template_fd59d141-d477-4f10-b39e-8127e0fbccb8.ipynb)
[decision-tree-random-forest-project-completed.ipynb](https://static.platzi.com/media/public/uploads/decision_tree_random_forest_project_completed_ede3a4ea-5f79-4ed7-9fbc-bdfd3b16b017.ipynb)

**Lecturas recomendadas**

[Car Evaluation Data Set | Kaggle](https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set)

[decision_tree_random_forest_project_completed.ipynb - Google Drive](https://drive.google.com/file/d/1Ck8R2GXK_ZeW9oYRIdXgVhBqi3ibt_QJ/view?usp=sharing)

[decision_tree_random_forest_project_template.ipynb - Google Drive](https://drive.google.com/file/d/1PFP6e4YfAI8nXq31kzRC8NquONJqLqeK/view?usp=sharing)

## Análisis exploratorio de datos para árbol de decisión

El **análisis exploratorio de datos (EDA)** es un paso clave antes de aplicar un algoritmo como un **árbol de decisión**, ya que te permite:

* Entender la estructura y calidad del dataset.
* Detectar valores faltantes o atípicos.
* Visualizar relaciones entre variables.
* Evaluar qué variables podrían ser importantes para la predicción.

### ✅ Pasos del Análisis Exploratorio de Datos (EDA) para Árbol de Decisión

### 1. **Cargar los datos y revisar estructura**

```python
import pandas as pd

df = pd.read_csv('autos.csv')  # ejemplo
print(df.head())
print(df.info())
print(df.describe())
```

### 2. **Identificar la variable objetivo (target)**

Verifica si es una variable **categórica** (clasificación) o **numérica** (regresión).

```python
print(df['tipo_auto'].value_counts())
```

Ejemplo de categorías: `['fósil', 'eléctrico', 'híbrido']`

### 3. **Visualizar la distribución de la variable target**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='tipo_auto', data=df)
plt.title('Distribución del tipo de automóvil')
plt.show()
```

### 4. **Revisar valores nulos**

```python
print(df.isnull().sum())
```

Soluciones:

* Imputar valores nulos (media, moda, etc.)
* Eliminar columnas o filas con muchos nulos

### 5. **Revisar correlaciones entre variables numéricas**

Aunque los árboles no necesitan variables escaladas ni normalizadas, es útil conocer la relación entre variables.

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### 6. **Codificar variables categóricas**

Los árboles pueden trabajar con etiquetas numéricas, así que debes convertir las categorías:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['tipo_auto'] = le.fit_transform(df['tipo_auto'])
```

Aplica lo mismo a otras columnas categóricas si es necesario.

### 7. **Detectar outliers**

Puedes usar diagramas de caja (boxplots):

```python
sns.boxplot(x=df['potencia'])
plt.title('Potencia - detección de outliers')
plt.show()
```

### 8. **Análisis bivariado**

Estudia cómo se relacionan las variables predictoras con la variable objetivo:

```python
sns.boxplot(x='tipo_auto', y='consumo', data=df)
```

### 9. **Feature importance (opcional luego del modelo)**

Los árboles te permiten saber qué variables son más importantes después del entrenamiento:

```python
from sklearn.tree import DecisionTreeClassifier

X = df.drop('tipo_auto', axis=1)
y = df['tipo_auto']

modelo = DecisionTreeClassifier()
modelo.fit(X, y)

importances = modelo.feature_importances_
for col, imp in zip(X.columns, importances):
    print(f"{col}: {imp:.3f}")
```

## Procesamiento de datos para el entrenamiento de árbol de decisión

El **procesamiento de datos** para entrenar un **árbol de decisión** implica preparar tu dataset de forma que el algoritmo pueda aprender patrones de manera efectiva. Aunque los árboles de decisión son muy flexibles (no requieren escalado de variables, por ejemplo), **sí necesitan ciertos pasos clave** para funcionar correctamente.

### ✅ Pasos de Procesamiento de Datos para Árbol de Decisión (Clasificación o Regresión)

### 1. **Separar variables predictoras y objetivo**

```python
X = df.drop('target', axis=1)  # variables predictoras
y = df['target']               # variable objetivo
```

### 2. **Codificar variables categóricas**

Los árboles requieren valores numéricos.

**Opción A: Label Encoding (útil si hay orden implícito o pocas categorías)**

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X['transmision'] = le.fit_transform(X['transmision'])
```

**Opción B: One-Hot Encoding (útil para categorías sin orden)**

```python
X = pd.get_dummies(X, columns=['marca', 'modelo'])
```

### 3. **Manejo de valores nulos**

Los árboles no manejan valores faltantes por sí solos.

```python
X = X.fillna(X.median())  # o usar X.dropna() si es apropiado
```

### 4. **División de datos en entrenamiento y prueba**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 5. **(Opcional) Balancear clases si es clasificación**

Si tienes un problema de clasificación desbalanceada:

```python
from sklearn.utils import resample

# Combinar X e y
df_train = pd.concat([X_train, y_train], axis=1)
minority = df_train[df_train['target'] == 'clase_rara']
majority = df_train[df_train['target'] == 'clase_común']

minority_upsampled = resample(minority,
                              replace=True,
                              n_samples=len(majority),
                              random_state=42)

df_balanced = pd.concat([majority, minority_upsampled])
X_train = df_balanced.drop('target', axis=1)
y_train = df_balanced['target']
```

### 6. **Entrenamiento del árbol**

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
```

### 7. **Evaluación**

```python
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 8. **(Opcional) Visualizar el árbol**

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
```

## Entrenamiento de modelo de clasificación con árbol de decisión

Aquí tienes un ejemplo completo y claro de **entrenamiento de un modelo de clasificación con árbol de decisión**, usando Python y `scikit-learn`.

### ✅ **Ejemplo paso a paso: Clasificación con Árbol de Decisión**

Usaremos el dataset `Iris` como ejemplo.

#### **1. Importar librerías**

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
```

#### **2. Cargar datos**

```python
iris = load_iris()
X = iris.data                  # variables predictoras
y = iris.target                # variable objetivo (0, 1, 2)
```

#### **3. Dividir en entrenamiento y prueba**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### **4. Entrenar el árbol de decisión**

```python
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)
```

#### **5. Predecir y evaluar**

```python
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
```

#### **6. (Opcional) Visualizar el árbol**

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

### 🔍 ¿Qué puedes modificar?

* Cambia `max_depth`, `criterion`, `min_samples_split`, etc. para ver cómo afecta al rendimiento.
* Usa tus propios datos (`pd.read_csv(...)`) y reemplaza `X` e `y`.

**Lecturas recomendadas**

[1.10. Decision Trees — scikit-learn 1.2.1 documentation](https://scikit-learn.org/stable/modules/tree.html)

[sklearn.tree.DecisionTreeClassifier — scikit-learn 1.2.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[decision_tree_random_forest_project_completed.ipynb - Google Drive](https://drive.google.com/file/d/1Ck8R2GXK_ZeW9oYRIdXgVhBqi3ibt_QJ/view?usp=sharing)

[decision_tree_random_forest_project_template.ipynb - Google Drive](https://drive.google.com/file/d/1PFP6e4YfAI8nXq31kzRC8NquONJqLqeK/view?usp=sharing)

## ¿Cómo evaluar un modelo de árbol de decisión?

Evaluar un modelo de **árbol de decisión** implica analizar qué tan bien predice sobre datos nuevos. Aquí tienes los pasos más importantes para hacerlo:

### ✅ **1. Dividir los datos**

Antes de entrenar, debes separar tu dataset:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ **2. Entrenar el modelo**

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
```

### ✅ **3. Realizar predicciones**

```python
y_pred = modelo.predict(X_test)
```

### ✅ **4. Evaluar el rendimiento**

Usa métricas de clasificación:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
```

### 🔍 ¿Qué significan estas métricas?

| Métrica                   | Significado                                              |
| ------------------------- | -------------------------------------------------------- |
| **Accuracy**              | Porcentaje de predicciones correctas                     |
| **Precisión (precision)** | Qué tan precisas son las predicciones positivas          |
| **Recall (sensibilidad)** | Qué tanto recupera el modelo de las clases verdaderas    |
| **F1-score**              | Balance entre precisión y recall                         |
| **Confusion Matrix**      | Muestra predicciones correctas vs. incorrectas por clase |

### ✅ **5. Importancia de variables (opcional)**

Para saber qué variables son más útiles:

```python
import pandas as pd

importancia = modelo.feature_importances_
print(pd.DataFrame({'Feature': feature_names, 'Importancia': importancia}))
```

### ✅ **6. Validación cruzada (opcional)**

Para tener una mejor idea del rendimiento general:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(modelo, X, y, cv=5)
print("Accuracy promedio:", scores.mean())
```

## Evaluación de resultados del modelo de árbol de decisión

La **evaluación de resultados** de un modelo de árbol de decisión se realiza para determinar qué tan bien generaliza a nuevos datos. A continuación te explico las principales herramientas y cómo interpretarlas:

### ✅ 1. **Predicción del modelo**

Después de entrenar el modelo:

```python
y_pred = modelo.predict(X_test)
```

### ✅ 2. **Métricas comunes de evaluación**

### 📊 a) **Accuracy (exactitud)**

Mide el porcentaje de predicciones correctas.

```python
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
```

> 🧠 Útil si las clases están balanceadas. No confiable si una clase domina.

### 📉 b) **Matriz de confusión**

Muestra cuántas predicciones fueron correctas o incorrectas por clase.

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
```

> 📌 Cada fila representa la clase real, cada columna la clase predicha.

### 📄 c) **Reporte de clasificación**

Incluye precisión, recall y F1-score por clase:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

* **Precision:** % de predicciones positivas correctas.
* **Recall (sensibilidad):** % de positivos reales bien clasificados.
* **F1-score:** Promedio armónico de precisión y recall.

### 📈 d) **Curva ROC y AUC (para clasificación binaria)**

Mide rendimiento del modelo en distintas probabilidades de corte.

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_proba = modelo.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
print("AUC:", auc)
```

> ⚠️ Solo aplicable para problemas binarios (2 clases).

### ✅ 3. **Evaluación con validación cruzada (opcional)**

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(modelo, X, y, cv=5)
print("Accuracy promedio:", scores.mean())
```

### ✅ 4. **Importancia de características**

Permite interpretar qué variables influyeron más:

```python
import pandas as pd

pd.DataFrame({
    'Característica': feature_names,
    'Importancia': modelo.feature_importances_
}).sort_values(by='Importancia', ascending=False)
```

**Lecturas recomendadas**

[sklearn.metrics.accuracy_score — scikit-learn 1.2.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

[decision_tree_random_forest_project_completed.ipynb - Google Drive](https://drive.google.com/file/d/1Ck8R2GXK_ZeW9oYRIdXgVhBqi3ibt_QJ/view?usp=sharing)

[decision_tree_random_forest_project_template.ipynb - Google Drive](https://drive.google.com/file/d/1PFP6e4YfAI8nXq31kzRC8NquONJqLqeK/view?usp=sharing)

## ¿Qué son los random forest o bosques aleatorios?

Los **Random Forest** o **Bosques Aleatorios** son un **algoritmo de aprendizaje automático supervisado** que se utiliza tanto para **clasificación** como para **regresión**.

### 🌲 ¿Qué son?

Un **Random Forest** es un **conjunto (ensamble)** de muchos **árboles de decisión** que trabajan juntos. En lugar de confiar en un solo árbol de decisión, este método construye varios árboles y **combina sus resultados** para obtener una predicción más precisa y robusta.

### 🔍 ¿Cómo funciona?

1. **Se crean muchos árboles de decisión**, cada uno entrenado con una **muestra aleatoria del dataset** (con reemplazo, técnica llamada *bootstrap*).
2. Para cada árbol, al momento de decidir una división, **solo se consideran aleatoriamente un subconjunto de las variables**.
3. Para clasificación: se toma el **voto mayoritario** de todos los árboles.

   * Para regresión: se toma el **promedio** de las predicciones de todos los árboles.

### 🎯 Ventajas:

* ✅ Reduce el **sobreajuste (overfitting)** de un solo árbol.
* ✅ Es más **preciso y estable** que un árbol de decisión individual.
* ✅ Funciona bien incluso sin ajustes finos de hiperparámetros.
* ✅ Puede manejar **datos faltantes y variables categóricas**.

### ⚠️ Desventajas:

* ❌ Es más lento y consume más memoria que un solo árbol.
* ❌ Difícil de interpretar (a diferencia de un solo árbol de decisión).

### 📊 Ejemplo de uso:

Si estás clasificando tipos de vehículos (eléctrico, híbrido, gasolina) a partir de datos como peso, potencia, y consumo, un modelo **Random Forest** puede ofrecer muy buena precisión al combinar decisiones de múltiples árboles.

## Tu primer random forest con scikit-learn

¡Perfecto! Vamos a crear tu **primer Random Forest con `scikit-learn`**, paso a paso. Usaremos un dataset clásico: **Iris**, que es ideal para clasificación.

### ✅ 1. Importar librerías necesarias

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### ✅ 2. Cargar el dataset y dividir en entrenamiento y prueba

```python
# Cargar el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ 3. Crear y entrenar el modelo Random Forest

```python
# Crear el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)
```

### ✅ 4. Realizar predicciones y evaluar

```python
# Predicciones
y_pred = model.predict(X_test)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
```

### ✅ 5. Visualizar la importancia de las características

```python
# Importancia de las variables
importances = model.feature_importances_
features = iris.feature_names

# Mostrar con Seaborn
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Importancia de características en Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Característica")
plt.show()
```

### ✅ Resultado Esperado:

* Una **alta precisión** (accuracy > 90%) en este dataset.
* Gráfico con las variables más relevantes para el modelo.
* Matriz de confusión para visualizar errores (falsos positivos/negativos).

**Archivos de la clase**

[primer-random-forest-completed.ipynb](https://static.platzi.com/media/public/uploads/primer_random_forest_completed_e98709b3-f62e-4b17-befb-02e3cdf7b810.ipynb)
[primer-random-forest-template.ipynb](https://static.platzi.com/media/public/uploads/primer_random_forest_template_a4b4eae3-7fbe-49c5-9f32-66bb8aca4f38.ipynb)

**Lecturas recomendadas**

[pima-indians-diabetes.csv | Kaggle](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv)

[primer_random_forest_completed.ipynb - Google Drive](https://drive.google.com/file/d/19eaWQEU_9OH7Hy3VnQErGiRClUVLzNEK/view?usp=sharing)

[primer_random_forest_template.ipynb - Google Drive](https://drive.google.com/file/d/1ksrGgYppmGM8U8uNErNKWM5pn5lP9sd1/view?usp=sharing)

## Entrenamiento de tu primer modelo de random forest con scikit-learn

¡Perfecto! Aquí tienes una guía clara y práctica para el **entrenamiento de tu primer modelo de Random Forest con `scikit-learn`** usando un dataset real:

### 🔧 Paso 1: Importar librerías necesarias

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

### 📥 Paso 2: Cargar y explorar los datos

```python
# Dataset clásico: Iris
iris = load_iris()
X = iris.data
y = iris.target

# Opcional: ver forma de los datos
print("Shape X:", X.shape)
print("Shape y:", y.shape)
```

### ✂️ Paso 3: Dividir en entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🌲 Paso 4: Crear y entrenar el modelo Random Forest

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### 📈 Paso 5: Realizar predicciones

```python
y_pred = rf_model.predict(X_test)
```

### ✅ Paso 6: Evaluar el modelo

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### 🔍 Paso 7: Visualizar importancia de características

```python
feature_importances = rf_model.feature_importances_
feature_names = iris.feature_names

sns.barplot(x=feature_importances, y=feature_names)
plt.title("Importancia de las características")
plt.xlabel("Importancia")
plt.ylabel("Característica")
plt.show()
```

### 🎉 Resultado:

Con el dataset **Iris**, deberías obtener una precisión bastante alta (cerca o superior al 95%), y podrás ver qué variables fueron más importantes para la clasificación.

**Lecturas recomendadas**

[pima-indians-diabetes.csv | Kaggle](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv)

[primer_random_forest_completed.ipynb - Google Drive](https://drive.google.com/file/d/19eaWQEU_9OH7Hy3VnQErGiRClUVLzNEK/view?usp=sharing)

[primer_random_forest_template.ipynb - Google Drive](https://drive.google.com/file/d/1ksrGgYppmGM8U8uNErNKWM5pn5lP9sd1/view?usp=sharing)

## Evaluación de tu primer modelo de random forest con scikit-learn

¡Perfecto! Una vez entrenado tu primer modelo de **Random Forest** con `scikit-learn`, es fundamental **evaluarlo correctamente** para entender su rendimiento. Aquí te explico cómo hacerlo paso a paso:

### ✅ **1. Predicción en datos de prueba**

```python
y_pred = rf_model.predict(X_test)
```

También puedes obtener las **probabilidades de clase** si lo necesitas:

```python
y_proba = rf_model.predict_proba(X_test)
```

### 📊 **2. Métricas de evaluación**

#### a. **Exactitud (Accuracy)**

```python
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### b. **Reporte de Clasificación (precision, recall, f1-score)**

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

#### c. **Matriz de confusión**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
```

### 🧠 **3. Importancia de características**

Para entender qué variables son más relevantes:

```python
import pandas as pd

importances = pd.Series(rf_model.feature_importances_, index=iris.feature_names)
importances.sort_values(ascending=True).plot(kind='barh')
plt.title('Importancia de las características')
plt.show()
```

### 🧪 **4. Validación cruzada (opcional)**

Para evaluar el modelo de forma más robusta:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X, y, cv=5)
print("Accuracy promedio con CV:", scores.mean())
```

### 🔍 Resultado esperado

Con un dataset como el **Iris**, podrías obtener:

* Accuracy > 0.95
* Una matriz de confusión clara
* Las características más importantes (ej. `petal length` o `petal width`)

## ¿Cómo funcionan los random forest?

Los **Random Forest** (o **bosques aleatorios**) son un algoritmo de **aprendizaje automático supervisado** basado en **árboles de decisión**, pero mejorado gracias a la combinación de muchos árboles. Su objetivo es hacer predicciones más **precisas**, **robustas** y **menos propensas al sobreajuste** que un único árbol.

### 🌳 ¿Cómo funciona un Random Forest?

1. **Creación de múltiples árboles de decisión (forest):**

   * En lugar de construir un solo árbol, el Random Forest construye **muchos árboles** (por ejemplo, 100 o 500).
   * Cada árbol se entrena con una **muestra aleatoria** del conjunto de datos original (con reemplazo, es decir, **bootstrap**).

2. **Selección aleatoria de características:**

   * Cuando un árbol va a hacer una división en un nodo, **no mira todas las características**, sino que selecciona un subconjunto aleatorio.
   * Esto aumenta la **diversidad** entre los árboles y mejora la generalización.

3. **Votación o promedio de predicciones:**

   * Para **clasificación**, cada árbol "vota" por una clase, y gana la clase más votada.
   * Para **regresión**, se promedia la predicción de todos los árboles.

### 🎯 Ejemplo simple (clasificación)

Supongamos que quieres predecir si un cliente comprará o no un producto.

* Entrenas 100 árboles, cada uno con diferentes subconjuntos de datos y características.
* Un cliente nuevo llega. Cada árbol da su predicción (sí o no).
* El **resultado final** será el que tenga **más votos**.

### ⚖️ Ventajas del Random Forest

✅ Reduce el **sobreajuste** comparado con un solo árbol
✅ Funciona bien en la mayoría de los problemas (clasificación y regresión)
✅ Puede manejar **datos faltantes** y **variables categóricas o numéricas**
✅ Da una **medida de importancia** de las variables (`feature_importances_`)

### ⚠️ Desventajas

❌ Menos interpretables que un solo árbol
❌ Consumen más recursos (RAM/tiempo)
❌ El entrenamiento puede ser más lento con muchos árboles o muchos datos

### 🔧 Parámetros clave en Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(
    n_estimators=100,       # número de árboles
    max_depth=None,         # profundidad máxima del árbol
    max_features='sqrt',    # número de features aleatorios por split
    bootstrap=True,         # si usar muestreo con reemplazo
    random_state=42
)
```

## ¿Cuándo utilizar random forest?

Puedes utilizar **Random Forest** cuando necesitas un modelo robusto y preciso para **clasificación** o **regresión**, especialmente en situaciones donde:

### ✅ **Cuándo usar Random Forest**

#### 1. **Tienes muchos datos y no sabes qué modelo usar**

* Random Forest es un buen modelo **por defecto**: funciona bien sin necesidad de demasiada configuración.
* Puede manejar **datos con muchas características** y detectar cuáles son realmente importantes.

#### 2. **Tus datos tienen ruido o relaciones no lineales**

* Random Forest no asume relaciones lineales entre variables.
* Es resistente al **sobreajuste**, especialmente comparado con árboles de decisión individuales.

#### 3. **Quisieras una estimación de la importancia de las variables**

* Random Forest te da una medida clara de la **importancia de cada feature**, útil para interpretar el modelo o reducir la dimensionalidad.

#### 4. **No te importa que el modelo sea poco interpretable**

* A diferencia de una regresión lineal o un solo árbol, el modelo es como una "caja negra".
* Pero si **la precisión es más importante que la explicación**, es una buena opción.

#### 5. **Tienes datos faltantes o mezcla de datos categóricos y numéricos**

* Random Forest puede tolerar **cierto nivel de datos faltantes**.
* Maneja datos categóricos codificados (por ejemplo, con One-Hot Encoding) sin problemas.

#### 6. **Tu problema es de clasificación multiclase o multietiqueta**

* Funciona bien en escenarios donde hay más de dos clases o múltiples etiquetas.

### ❌ **Cuándo evitar Random Forest**

* Cuando necesitas un modelo **muy interpretable** (por ejemplo, en medicina o leyes).
* Si tienes **poco poder de cómputo**: entrenar muchos árboles puede ser costoso.
* Si tu conjunto de datos es **muy pequeño**, un árbol de decisión o un modelo más simple puede ser mejor.

### 🧠 Ejemplos de uso en la vida real

* **Banca**: detectar fraudes o evaluar el riesgo crediticio.
* **Medicina**: predecir enfermedades a partir de datos clínicos.
* **Marketing**: segmentar clientes o predecir abandono.
* **Finanzas**: predecir el precio de una acción.
* **Ingeniería**: detectar fallos en sensores o equipos.

**Lecturas recomendadas**

[Random Forest (Bosque Aleatorio): combinando árboles - IArtificial.net](https://www.iartificial.net/random-forest-bosque-aleatorio/)

## Entrenamiento de modelo de clasificación de carros con random forest

Aquí tienes un ejemplo completo de **entrenamiento de un modelo de clasificación de carros usando Random Forest en Python con `scikit-learn`**, desde los datos hasta la predicción:

### ✅ Supongamos que tienes un dataset con las siguientes columnas:

* `marca`, `anio`, `cilindraje`, `tipo_combustible`, `precio_categoria` (donde esta última es la variable **objetivo**: "alto", "medio", "bajo")

### 📦 1. Importar librerías

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
```

### 📄 2. Datos de ejemplo

```python
# Datos simulados
data = {
    'marca': ['Toyota', 'Mazda', 'Renault', 'Chevrolet', 'Kia'],
    'anio': [2015, 2018, 2020, 2017, 2016],
    'cilindraje': [1.6, 2.0, 1.2, 1.4, 1.6],
    'tipo_combustible': ['Gasolina', 'Gasolina', 'Gasolina', 'Diesel', 'Gasolina'],
    'precio_categoria': ['medio', 'alto', 'bajo', 'medio', 'bajo']
}
df = pd.DataFrame(data)
```

### 🧹 3. Preprocesamiento

```python
# Codificación de variables categóricas
le_marca = LabelEncoder()
le_comb = LabelEncoder()
le_target = LabelEncoder()

df['marca'] = le_marca.fit_transform(df['marca'])
df['tipo_combustible'] = le_comb.fit_transform(df['tipo_combustible'])
df['precio_categoria'] = le_target.fit_transform(df['precio_categoria'])  # Etiquetas 0, 1, 2
```

### ✂️ 4. División en train/test

```python
X = df.drop('precio_categoria', axis=1)
y = df['precio_categoria']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### 🌲 5. Entrenamiento del modelo Random Forest

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 🧪 6. Evaluación del modelo

```python
y_pred = model.predict(X_test)

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
```

### 🧠 7. Predecir nuevos autos

```python
nuevo_auto = pd.DataFrame({
    'marca': le_marca.transform(['Toyota']),
    'anio': [2022],
    'cilindraje': [1.8],
    'tipo_combustible': le_comb.transform(['Gasolina'])
})

pred = model.predict(nuevo_auto)
print(f"Categoría de precio predicha: {le_target.inverse_transform(pred)[0]}")
```

**Archivos de la clase**

[decision-tree-random-forest-project-completed.ipynb](https://static.platzi.com/media/public/uploads/decision_tree_random_forest_project_completed_562b6c49-de86-4fb3-9a11-cdc2f5c37678.ipynb)
[decision-tree-random-forest-project-template.ipynb](https://static.platzi.com/media/public/uploads/decision_tree_random_forest_project_template_8dcf18db-e157-4d47-85a9-ef449f26de4c.ipynb)

**Lecturas recomendadas**

[sklearn.ensemble.RandomForestClassifier — scikit-learn 1.2.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
[decision_tree_random_forest_project_completed.ipynb - Google Drive](https://drive.google.com/file/d/1Ck8R2GXK_ZeW9oYRIdXgVhBqi3ibt_QJ/view?usp=sharing)
[decision_tree_random_forest_project_template.ipynb - Google Drive](https://drive.google.com/file/d/1PFP6e4YfAI8nXq31kzRC8NquONJqLqeK/view?usp=sharing)

## Evaluación de resultados del modelo de clasificación con random forest

La **evaluación de resultados de un modelo de clasificación con Random Forest** en scikit-learn se realiza principalmente con métricas como:

### ✅ 1. **Accuracy (exactitud)**

Mide el porcentaje de predicciones correctas sobre el total.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy:.2f}")
```

### ✅ 2. **Matriz de confusión**

Muestra cuántos ejemplos se clasificaron correctamente y cuáles fueron confundidos entre clases.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
```

### ✅ 3. **Reporte de clasificación**

Incluye precisión, recall y F1-score para cada clase:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=le_target.classes_))
```

🔹 **Precision**: De los que predije como clase X, ¿cuántos eran realmente X?
🔹 **Recall**: De todos los que eran clase X, ¿cuántos los detecté correctamente?
🔹 **F1-score**: Promedio ponderado de precisión y recall.

### ✅ 4. **Importancia de características**

Para saber qué variables influyen más en el modelo:

```python
importances = model.feature_importances_
for col, imp in zip(X.columns, importances):
    print(f"{col}: {imp:.4f}")
```

### 🎯 Ejemplo de salida esperada (si se usó el código anterior)

```text
Accuracy del modelo: 0.80

              precision    recall  f1-score   support

        alto       0.75      1.00      0.86         1
        bajo       1.00      0.50      0.67         2
       medio       1.00      1.00      1.00         1

    accuracy                           0.80         4
   macro avg       0.92      0.83      0.84         4
weighted avg       0.88      0.80      0.79         4
```

**Archivos de la clase**

[decision-tree-random-forest-project-completed.ipynb](https://static.platzi.com/media/public/uploads/decision_tree_random_forest_project_completed_2b8cc33a-1f4f-47f4-8f0b-8f881066cd3d.ipynb)
[decision-tree-random-forest-project-template.ipynb](https://static.platzi.com/media/public/uploads/decision_tree_random_forest_project_template_f5d5ec26-0355-4844-8cc1-2e016d08c6e9.ipynb)

**Lecturas recomendadas**

[sklearn.metrics.accuracy_score — scikit-learn 1.2.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
[decision_tree_random_forest_project_completed.ipynb - Google Drive](https://drive.google.com/file/d/1Ck8R2GXK_ZeW9oYRIdXgVhBqi3ibt_QJ/view?usp=sharing)
[decision_tree_random_forest_project_template.ipynb - Google Drive](https://drive.google.com/file/d/1PFP6e4YfAI8nXq31kzRC8NquONJqLqeK/view?usp=sharing)

**Lecturas recomendadas**

[UCI Machine Learning Repository: Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income)

[https://www.linkedin.com/in/laylascheli/](https://www.linkedin.com/in/laylascheli/)