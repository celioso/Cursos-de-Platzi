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