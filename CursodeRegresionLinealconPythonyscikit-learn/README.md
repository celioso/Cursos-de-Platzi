# Curso de Regresión Lineal con Python y scikit-learn

## Tu primera regresión lineal con scikit-learn

Vamos a hacer **tu primera regresión lineal** usando `scikit-learn`, una de las bibliotecas más populares de Python para machine learning.

### 📈 ¿Qué es una regresión lineal?

Es un modelo matemático que **relaciona una variable independiente (X)** con **una variable dependiente (y)** a través de una **recta**:

$$
y = mX + b
$$

Donde:

* `m` es la **pendiente**
* `b` es la **intersección** o término independiente

### ✅ Ejemplo paso a paso con código

### 1. Importar bibliotecas necesarias

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```

### 2. Crear datos de ejemplo

```python
# Datos artificiales
X = np.array([[1], [2], [3], [4], [5]])  # variable independiente
y = np.array([2, 4, 5, 4, 5])           # variable dependiente
```

### 3. Crear y entrenar el modelo

```python
modelo = LinearRegression()
modelo.fit(X, y)
```

### 4. Ver los coeficientes del modelo

```python
print("Pendiente (coef_):", modelo.coef_)
print("Intersección (intercept_):", modelo.intercept_)
```

### 5. Hacer predicciones

```python
X_nuevos = np.array([[6], [7]])
predicciones = modelo.predict(X_nuevos)
print("Predicciones:", predicciones)
```

### 6. Visualizar la regresión

```python
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, modelo.predict(X), color='red', label='Línea de regresión')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Regresión Lineal con scikit-learn")
plt.grid(True)
plt.show()
```

### 🎓 ¿Qué aprendiste?

Con esto lograste:

* Crear un modelo de regresión lineal
* Ajustarlo con datos reales
* Predecir nuevos valores
* Visualizar la línea de mejor ajuste

**Archivos de la clase**

[2-slides-1.pdf](https://static.platzi.com/media/public/uploads/2-slides-1_86e64012-0cea-4015-8896-adb2f9d8c966.pdf)
[primera-regresion-lineal.ipynb](https://static.platzi.com/media/public/uploads/primera_regresion_lineal_755953b4-3c16-42de-a9e3-78fe9742244f.ipynb)
[primera-regresion-lineal-template.ipynb](https://static.platzi.com/media/public/uploads/primera_regresion_lineal_template_42c71d99-6fdb-4af7-98f2-e3f8e2e76b7a.ipynb)
[housing.data](https://static.platzi.com/media/public/uploads/housing_edf82b23-51e2-4f57-887d-6be67da34d27.data)

## Análisis de datos para tu primera regresión lineal

vamos a realizar un **análisis de datos** para nuestra primera **regresión lineal** utilizando el famoso dataset de **Housing de Boston** (aunque oficialmente retirado de `sklearn` por temas éticos, aún puede usarse con cuidado desde UCI).

Este dataset contiene 506 filas y 14 columnas. La variable objetivo (`target`) es el precio medio de las viviendas en miles de dólares.

### 🔹 Paso 1: Cargar los datos

```python
import pandas as pd

# Cargar el dataset desde UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
columnas = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
data = pd.read_csv(url, header=None, sep=r'\s+', names=columnas)
```

### 🔹 Paso 2: Inspeccionar el dataset

```python
# Ver primeras filas
print(data.head())

# Resumen estadístico
print(data.describe())

# Ver si hay valores nulos
print(data.isnull().sum())
```

### 🔹 Paso 3: Visualización de correlaciones

Podemos usar `seaborn` para ver cómo se relacionan las variables con la variable objetivo `MEDV`.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Mapa de calor de correlaciones
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()
```

### 🔹 Paso 4: Selección de una variable para regresión simple

Vamos a elegir la variable más correlacionada con `MEDV`. Por lo general, `LSTAT` (porcentaje de población con bajos ingresos) tiene una fuerte correlación negativa con el precio.

```python
sns.scatterplot(x='LSTAT', y='MEDV', data=data)
plt.title("Relación entre LSTAT y MEDV")
plt.xlabel("LSTAT (% población de bajos ingresos)")
plt.ylabel("Precio medio (MEDV)")
plt.show()
```

### ¿Qué sigue?

Con esta base ya podemos:

1. Dividir el dataset en entrenamiento y prueba.
2. Crear un modelo de regresión lineal con `scikit-learn`.
3. Entrenar el modelo.
4. Hacer predicciones y evaluar el desempeño.

## Entrenando un modelo de regresión lineal con scikit-learn

Entrenar un modelo de regresión lineal con **Scikit-learn** es sencillo. A continuación te muestro un ejemplo completo, paso a paso, con explicaciones:

### ✅ 1. Importar las librerías necesarias

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

### ✅ 2. Crear o cargar los datos

Aquí vamos a crear datos sintéticos como ejemplo:

```python
# Datos de ejemplo (relación lineal con algo de ruido)
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

### ✅ 3. Dividir en entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ 4. Entrenar el modelo

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### ✅ 5. Ver los coeficientes del modelo

```python
print(f"Intercepto: {model.intercept_}")
print(f"Coeficientes: {model.coef_}")
```

### ✅ 6. Realizar predicciones

```python
y_pred = model.predict(X_test)
```

### ✅ 7. Evaluar el modelo

```python
print(f"Error cuadrático medio (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coeficiente de determinación (R²): {r2_score(y_test, y_pred):.2f}")
```

### ✅ 8. (Opcional) Visualizar resultados

```python
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Modelo de Regresión Lineal')
plt.show()
```

**Lecturas recomendadas**

[scikit-learn: machine learning in Python — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/)

[sklearn.linear_model.LinearRegression — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

[sklearn.preprocessing.StandardScaler — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

[Primera_regresión_lineal.ipynb - Google Drive](https://drive.google.com/file/d/1wh1T5AE1AgmgDcTaiziaeKYSuovVFQPk/view?usp=sharing)

[Primera_regresión_lineal_Template.ipynb - Google Drive](https://drive.google.com/file/d/1NkcOQg-BTEAD0ttGDcZcAZwYpD0jXPJd/view?usp=sharing)

## ¿Qué es la regresión lineal?

La **regresión lineal** es un método estadístico y de aprendizaje automático que se utiliza para **modelar la relación entre una variable dependiente (o de salida) y una o más variables independientes (o de entrada)**.

### 🔹 ¿Qué hace?

Intenta **ajustar una línea recta** (en el caso simple) que **mejor explique** cómo cambia la variable dependiente a medida que cambian las independientes.

### 🔸 Ejemplo simple (una sola variable):

Supón que tienes datos de estudio:

| Horas de estudio | Nota en el examen |
| ---------------- | ----------------- |
| 1                | 55                |
| 2                | 65                |
| 3                | 70                |
| 4                | 75                |

La regresión lineal busca encontrar una **línea** del tipo:

$$
y = b_0 + b_1 x
$$

Donde:

* $y$ es la nota (variable dependiente),
* $x$ son las horas de estudio (variable independiente),
* $b_0$ es el **intercepto** (valor cuando $x = 0$),
* $b_1$ es el **coeficiente** (pendiente de la línea, cuánto cambia $y$ por cada unidad de $x$).

### 🔸 ¿Para qué sirve?

* **Predicción**: Estimar valores futuros.
* **Interpretación**: Entender qué variables afectan a otra.
* **Reducción**: En modelos complejos, ayuda a simplificar relaciones.

### 🔹 Tipos de regresión lineal:

1. **Simple**: Una variable independiente.
2. **Múltiple**: Varias variables independientes.

### 🔸 Ejemplo gráfico (regresión simple):

Imagina que trazas una línea sobre un conjunto de puntos dispersos. Esa línea representa la **mejor estimación promedio** del comportamiento de esos datos.

## Cuándo utilizar un modelo de regresión lineal

Puedes utilizar un **modelo de regresión lineal** cuando se cumplen las siguientes condiciones o se busca alguno de estos objetivos:

### ✅ **CUÁNDO USARLO:**

#### 1. **Relación lineal entre variables**

* Cuando crees que hay una **relación lineal** (aproximadamente recta) entre la variable dependiente y una o más independientes.
* Ejemplo: A más horas de estudio → mayor nota.

#### 2. **Variables numéricas**

* Es ideal cuando las **variables de entrada (independientes)** y la **variable objetivo (dependiente)** son **numéricas continuas**.

#### 3. **Pocos datos y modelo interpretable**

* Cuando necesitas un modelo **sencillo, rápido y fácil de interpretar**.
* Puedes ver claramente qué variable tiene más influencia en el resultado.

#### 4. **El objetivo es predecir o explicar**

* Puedes usarlo tanto para:

  * **Predecir** valores (como ingresos, temperatura, precios).
  * **Explicar** cómo influye cada variable sobre otra.

#### 5. **No hay demasiada colinealidad**

* Las variables independientes **no deben estar altamente correlacionadas** entre sí (porque confunden al modelo).

#### 6. **Errores con distribución normal (idealmente)**

* Aunque no es obligatorio para predecir, si vas a hacer inferencia estadística (como tests de hipótesis), los **errores (residuos)** deben seguir una **distribución normal**.

### ❌ **CUÁNDO NO USARLO:**

* Cuando la relación entre variables **no es lineal**.
* Si tienes **muchas variables categóricas** y no las has convertido correctamente (one-hot encoding, etc.).
* Cuando hay **outliers extremos** que afectan mucho la pendiente.
* Si hay **relaciones complejas o no lineales** entre las variables → mejor usar árboles, redes neuronales, etc.

### 📌 Ejemplos de uso real:

| Caso                  | Variable dependiente | Variables independientes                  |
| --------------------- | -------------------- | ----------------------------------------- |
| Predicción de precios | Precio de una casa   | Tamaño, ubicación, número de habitaciones |
| Medicina              | Nivel de colesterol  | Edad, peso, dieta                         |
| Negocios              | Ventas mensuales     | Gasto en publicidad, precio del producto  |

## Función de pérdida y optimización: mínimos cuadrados

En **regresión lineal**, la **función de pérdida** más común es la de **mínimos cuadrados**. Aquí te explico qué es y cómo se usa para la **optimización del modelo**:

### 🎯 ¿Qué es la función de pérdida de mínimos cuadrados?

Es una función que **mide el error** entre los valores predichos por el modelo y los valores reales. La idea es **minimizar ese error** durante el entrenamiento.

### 📐 Definición matemática

Dada una muestra de datos con `n` observaciones:

$$
\text{Pérdida} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Donde:

* $y_i$ = valor real
* $\hat{y}_i$ = valor predicho por el modelo
* La diferencia $y_i - \hat{y}_i$ se llama **residuo**
* Se eleva al cuadrado para:

  * Penalizar más los errores grandes
  * Evitar que errores positivos y negativos se cancelen

Esta pérdida también se conoce como **Error Cuadrático Total (SSE)** o **Suma de los errores al cuadrado**.

### 🛠 ¿Cómo se optimiza?

El modelo de regresión lineal busca los **coeficientes (pendientes y término independiente)** que **minimizan esta función de pérdida**.

Esto se puede hacer con:

* **Solución analítica (ecuación normal)**:
  Para modelos pequeños o simples.
* **Descenso del gradiente**:
  Método iterativo que ajusta los coeficientes paso a paso en la dirección que reduce el error.

### 📉 ¿Por qué mínimos cuadrados?

Porque es:

* Rápido y computacionalmente eficiente.
* Fácil de interpretar.
* Funciona bien si los **errores siguen una distribución normal**.

### 📌 En Python con `scikit-learn`:

Cuando usas:

```python
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(X, y)
```

Internamente se está minimizando la función de **mínimos cuadrados** para encontrar los mejores coeficientes.

**Lecturas recomendadas**

[Regresión Lineal | Aprende Machine Learning](https://www.aprendemachinelearning.com/tag/regresion-lineal/)

[¿Qué es el descenso del gradiente? - Platzi](https://platzi.com/clases/2155-calculo-data-science/35480-que-es-el-descenso-del-gradiente/)

## Evaluando el modelo: R^2 y MSE

Al evaluar un modelo de **regresión lineal**, es fundamental medir qué tan bien predice los valores. Dos métricas ampliamente utilizadas son:

### 📊 1. Coeficiente de Determinación: **R² (R-squared)**

### ¿Qué es?

* R² mide la **proporción de la varianza** en la variable dependiente $y$ que es explicada por las variables independientes $X$.
* Es una métrica de **bondad de ajuste**.

### Fórmula:

$$
R^2 = 1 - \frac{SSE}{SST}
$$

Donde:

* $SSE = \sum (y_i - \hat{y}_i)^2$: **Suma de errores al cuadrado (residuos)**
* $SST = \sum (y_i - \bar{y})^2$: **Varianza total del modelo**

### Interpretación:

* $R^2 = 1$: predicción perfecta.
* $R^2 = 0$: el modelo no explica nada mejor que la media.
* Puede ser **negativo** si el modelo es peor que predecir con la media.

### 📉 2. Error Cuadrático Medio: **MSE (Mean Squared Error)**

### ¿Qué es?

* Mide el **promedio de los errores al cuadrado** entre los valores reales y los predichos.

### Fórmula:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Interpretación:

* Cuanto menor sea el MSE, **mejor el rendimiento** del modelo.
* Tiene las **mismas unidades** al cuadrado que la variable objetivo.
* Penaliza fuertemente los errores grandes.

### 🧪 Ejemplo práctico con scikit-learn:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)
```

**Lecturas recomendadas**

[3.3. Metrics and scoring: quantifying the quality of predictions — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

[Difference between Adjusted R-squared and R-squared](https://www.listendata.com/2014/08/adjusted-r-squared.html)

[Interpreting Residual Plots to Improve Your Regression](https://www.qualtrics.com/support/stats-iq/analyses/regression-guides/interpreting-residual-plots-improve-regression/)

## Regresión lineal multivariable

La **regresión lineal multivariable** (o **regresión lineal múltiple**) es una extensión de la regresión lineal simple. En lugar de tener una sola variable independiente (input), se tienen **dos o más variables independientes** para predecir una variable dependiente.

### 🧮 Forma general del modelo

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \varepsilon
$$

* $y$: variable objetivo (dependiente)
* $x_1, x_2, ..., x_n$: variables predictoras (independientes)
* $\beta_0$: intercepto (bias)
* $\beta_1, ..., \beta_n$: coeficientes de cada variable
* $\varepsilon$: error o ruido aleatorio

### ✅ ¿Cuándo usar regresión lineal multivariable?

* Cuando tienes **más de una característica** (feature) que afecta la variable que deseas predecir.
* Ejemplo: predecir el precio de una casa usando el número de habitaciones, superficie, ubicación, etc.

### 🛠️ ¿Cómo se entrena con `scikit-learn`?

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Supón que df tiene columnas: 'habitaciones', 'metros_cuadrados', 'precio'
X = df[['habitaciones', 'metros_cuadrados']]  # variables independientes
y = df['precio']  # variable dependiente

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print(modelo.coef_)      # Coeficientes de las variables
print(modelo.intercept_) # Intercepto
```

### 📈 Evaluación

Se puede evaluar usando:

* $R^2$: coeficiente de determinación
* MSE: error cuadrático medio

```python
from sklearn.metrics import r2_score, mean_squared_error

y_pred = modelo.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

**Archivos de la clase**

[ejercicio-multivariable.ipynb](https://static.platzi.com/media/public/uploads/ejercicio_multivariable_32b1d1b6-256d-48ce-9bcc-10bb8e108079.ipynb)

**Lecturas recomendadas**

[Ejercicio_Multivariable.ipynb - Google Drive](https://drive.google.com/file/d/1ONeY0MvSBuSml6zp_g1un_febp1g4NgL/view?usp=sharing)

## Regresión lineal para predecir los gastos médicos de pacientes

La **regresión lineal** es una técnica muy útil para predecir los **gastos médicos** de pacientes si cuentas con variables numéricas relevantes como:

* Edad (`age`)
* IMC (`bmi`)
* Número de hijos (`children`)
* Sexo (`sex`)
* Fumador (`smoker`)
* Región (`region`)

Estas variables se pueden usar como características (`X`) para predecir el gasto médico (`charges`).

### 🧠 ¿Por qué usar regresión lineal?

Porque es una forma de modelar cómo distintas características influyen en el resultado (en este caso, los gastos médicos). Por ejemplo:

* Fumar puede aumentar el gasto.
* Mayor edad también suele estar asociada a mayores gastos.
* Un IMC alto podría correlacionarse con más problemas de salud.

### ✅ Ejemplo básico con Python y scikit-learn

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Carga de datos (puedes usar un dataset como el de 'insurance.csv')
df = pd.read_csv('insurance.csv')

# Variables numéricas y categóricas
numeric = ['age', 'bmi', 'children']
categorical = ['sex', 'smoker', 'region']

# Separar variables predictoras y objetivo
X = df[numeric + categorical]
y = df['charges']

# Preprocesamiento: estandarizar numéricas y one-hot encoding a categóricas
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(drop='first'), categorical)
])

# Pipeline con regresión lineal
model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento
model.fit(X_train, y_train)

# Evaluación
from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test)
print('R²:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
```

### 📊 Interpretación

* `R²` indica qué tan bien el modelo explica los datos (más cerca a 1 = mejor).
* `MSE` indica el error medio al predecir los gastos (menor = mejor).

**Lecturas recomendadas**

[Medical Cost Personal Datasets | Kaggle](https://www.kaggle.com/mirichoi0218/insurance)

## Exploración y preparación de datos

Para aplicar **regresión lineal a gastos médicos**, primero debes hacer una **exploración y preparación de datos** adecuada. Aquí te muestro paso a paso cómo hacerlo con el famoso dataset `insurance.csv`:

### 🗂️ 1. **Cargar los datos**

```python
import pandas as pd

df = pd.read_csv('insurance.csv')
print(df.head())
```

### 📋 2. **Exploración inicial (EDA: Exploratory Data Analysis)**

```python
# Tamaño del dataset
print(df.shape)

# Tipos de datos
print(df.dtypes)

# Estadísticas básicas
print(df.describe())

# Ver si hay valores nulos
print(df.isnull().sum())
```

### 📊 3. **Visualizaciones útiles**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Distribución de la variable objetivo
sns.histplot(df['charges'], kde=True)
plt.title('Distribución de gastos médicos')
plt.show()

# Relación entre edad e IMC con gastos
sns.scatterplot(x='age', y='charges', data=df)
plt.title('Edad vs Gastos médicos')
plt.show()

sns.scatterplot(x='bmi', y='charges', data=df)
plt.title('IMC vs Gastos médicos')
plt.show()

# Boxplots para variables categóricas
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Gastos según hábito de fumar')
plt.show()

sns.boxplot(x='region', y='charges', data=df)
plt.title('Gastos por región')
plt.show()
```

### 🔧 4. **Codificación de variables categóricas**

```python
# Codificación one-hot para 'sex', 'smoker', 'region'
df_encoded = pd.get_dummies(df, drop_first=True)
print(df_encoded.head())
```

### ⚖️ 5. **Escalado de variables numéricas (opcional pero recomendado)**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = ['age', 'bmi', 'children']
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
```

### ✅ 6. **Separación de variables predictoras y objetivo**

```python
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
```

### 🧪 7. **División en entrenamiento y prueba**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Este preprocesamiento te deja listo para entrenar un modelo de **regresión lineal** u otro más avanzado.

## Análisis de correlación de los datos

El **análisis de correlación** es una técnica fundamental para comprender las relaciones entre variables numéricas en un conjunto de datos. Es especialmente útil antes de aplicar un modelo de regresión lineal, como el que estás construyendo para predecir los gastos médicos.

Aquí tienes una guía paso a paso usando `pandas`, `seaborn` y `matplotlib` para un análisis de correlación claro y visual:

### ✅ Paso 1: Cargar los datos

```python
import pandas as pd

# Por ejemplo, si usas el dataset de gastos médicos
df = pd.read_csv('insurance.csv')
```

### ✅ Paso 2: Convertir variables categóricas (si es necesario)

```python
# Convertir columnas categóricas con one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)
```

### ✅ Paso 3: Calcular la matriz de correlación

```python
correlation_matrix = df_encoded.corr()
print(correlation_matrix)
```

### ✅ Paso 4: Visualizar la matriz de correlación con un heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()
```

### ✅ ¿Qué observar?

* Busca **correlaciones altas (cercanas a +1 o -1)** entre la variable objetivo (`charges`) y las otras.
* Por ejemplo:

  * Si `bmi`, `smoker_yes`, o `age` tienen una correlación fuerte con `charges`, son buenos candidatos como predictores.
* Ten cuidado con **multicolinealidad** (cuando dos predictores están fuertemente correlacionados entre sí).

## Entrenamiento del modelo

¡Perfecto! Una vez explorados y preparados los datos, el siguiente paso es **entrenar el modelo de regresión lineal**. A continuación te explico cómo hacerlo paso a paso con `scikit-learn`:

### ✅ Paso 1: Importar librerías necesarias

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

### ✅ Paso 2: Cargar y preparar los datos

```python
# Cargar el dataset
df = pd.read_csv('insurance.csv')

# Codificar variables categóricas (por ejemplo: sexo, fumador, región)
df_encoded = pd.get_dummies(df, drop_first=True)

# Separar variables independientes (X) y variable objetivo (y)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
```

### ✅ Paso 3: Dividir el conjunto de datos

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ Paso 4: Crear y entrenar el modelo

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### ✅ Paso 5: Ver los coeficientes del modelo

```python
# Mostrar los coeficientes junto con los nombres de las variables
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
print(coeff_df)
```

### ✅ Paso 6: Realizar predicciones

```python
y_pred = model.predict(X_test)
```

### ✅ Paso 7: Evaluar el modelo

Puedes usar R² y el Error Cuadrático Medio (MSE):

```python
from sklearn.metrics import r2_score, mean_squared_error

print('R²:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
```

**Archivos de la clase**

[proyecto-final.ipynb](https://static.platzi.com/media/public/uploads/proyecto_final_4a65bcf4-a423-4ec8-b844-614430462fb4.ipynb)

[proyecto-final-template.ipynb](https://static.platzi.com/media/public/uploads/proyecto_final_template_4b44140a-4417-42eb-b260-378da1179ab3.ipynb)

**Lecturas recomendadas**

[sklearn.linear_model.LinearRegression — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

[Proyecto_Final.ipynb - Google Drive](https://drive.google.com/file/d/1UDV8y9eyZuKEDSIRKY7w93Bdo7EBqz5Z/view?usp=sharing)

[Proyecto_Final_Template.ipynb - Google Drive](https://drive.google.com/file/d/1PrSPTAVz6lXKAqNcTlUvWc_de8TRILOx/view?usp=sharing)

## Evaluando el modelo

¡Perfecto! Una vez entrenado el modelo de regresión lineal, es fundamental **evaluar su rendimiento** para saber qué tan bien está prediciendo los gastos médicos.

Aquí tienes una guía completa para evaluar el modelo:

### ✅ 1. Importar métricas de evaluación

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
```

### ✅ 2. Calcular métricas de desempeño

```python
# Predicciones
y_pred = model.predict(X_test)

# R² (coeficiente de determinación)
r2 = r2_score(y_test, y_pred)

# Error absoluto medio
mae = mean_absolute_error(y_test, y_pred)

# Error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

# Raíz del error cuadrático medio
rmse = np.sqrt(mse)

# Mostrar resultados
print(f'R²: {r2:.4f}')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
```

### ✅ 3. Interpretar resultados

| Métrica  | Interpretación                                                                                                                                    |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R²**   | Mide qué porcentaje de la variabilidad en los gastos se explica por las variables independientes. 1 es perfecto, 0 significa que no explica nada. |
| **MAE**  | Promedio del error absoluto entre las predicciones y los valores reales.                                                                          |
| **MSE**  | Promedio de los errores al cuadrado. Penaliza más los errores grandes.                                                                            |
| **RMSE** | Raíz cuadrada del MSE. Más interpretable que el MSE, está en las mismas unidades que `charges`.                                                   |

### ✅ 4. (Opcional) Gráfico de comparación

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valores reales (charges)')
plt.ylabel('Predicciones (charges)')
plt.title('Predicciones vs Valores reales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Línea ideal
plt.grid()
plt.show()
```

Si deseas mejorar la precisión del modelo, podrías probar con **modelos más avanzados** como:

* Ridge, Lasso (regresión regularizada)
* Random Forest o Gradient Boosting
* XGBoost

**Archivos de la clase**

[proyecto-final-template.ipynb](https://static.platzi.com/media/public/uploads/proyecto_final_template_ae2e550d-3eb9-46b0-83f7-9d83ede0a8e7.ipynb)
[proyecto-final.ipynb](https://static.platzi.com/media/public/uploads/proyecto_final_ca80d638-ee4b-42ae-9ddf-6881f2c7a43a.ipynb)

**Lecturas recomendadas**

[3.3. Metrics and scoring: quantifying the quality of predictions — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

[Proyecto_Final_Template.ipynb - Google Drive](https://drive.google.com/file/d/1PrSPTAVz6lXKAqNcTlUvWc_de8TRILOx/view?usp=sharing)

[Proyecto_Final.ipynb - Google Drive](https://drive.google.com/file/d/1UDV8y9eyZuKEDSIRKY7w93Bdo7EBqz5Z/view?usp=sharing)

## Mejorando el modelo

Para **mejorar tu modelo de regresión lineal para predecir los gastos médicos de pacientes**, puedes aplicar varias estrategias. Aquí te dejo un flujo de trabajo guiado, con ejemplos en código, asumiendo que ya tienes los datos listos (`X_train`, `X_test`, `y_train`, `y_test`):

### 🧠 1. **Agregar características polinómicas**

Puede ayudar si la relación entre variables no es lineal:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Grado 2 (cuadrático)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_poly, y_train)

y_pred_poly = model_poly.predict(X_test_poly)
print("R² (polinómica):", r2_score(y_test, y_pred_poly).round(4))
```

### 🔄 2. **Escalar las variables (si no lo has hecho)**

Muy útil si combinas variables con diferentes escalas (por ejemplo, edad y número de hijos):

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_scaled = model.predict(X_test_scaled)
print("R² (escalado):", r2_score(y_test, y_pred_scaled).round(4))
```

### 🔥 3. **Probar modelos más potentes (como regularización)**

Para controlar el sobreajuste o mejorar con variables no relevantes:

#### a) **Ridge Regression** (L2)

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print("R² Ridge:", r2_score(y_test, y_pred_ridge).round(4))
```

#### b) **Lasso Regression** (L1 - hace selección de variables)

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
print("R² Lasso:", r2_score(y_test, y_pred_lasso).round(4))
```

### 📊 4. **Ingeniería de características**

* Convertir variables categóricas a dummies con `pd.get_dummies`
* Probar interacciones entre variables
* Agregar transformaciones no lineales (log, raíz, etc.)

Ejemplo:

```python
import numpy as np
X_train["bmi_log"] = np.log(X_train["bmi"])
X_test["bmi_log"] = np.log(X_test["bmi"])
```

### 🧪 5. **Validación cruzada para comparar**

Te permite evaluar cuál modelo generaliza mejor:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print("R² promedio (CV):", scores.mean().round(4))
```

## ¿Qué hay más allá de la linealidad?

Más allá de la **linealidad** en los modelos de regresión, existen métodos que permiten capturar **relaciones más complejas y no lineales** entre las variables. Aquí te explico las principales alternativas y conceptos clave:

### 🔹 1. **Regresión Polinómica**

* Transforma las variables originales en potencias (cuadrado, cubo, etc.).
* Ejemplo: en vez de ajustar una línea recta, ajusta una **curva**.
* Útil cuando la relación entre X e Y es curvilínea.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```

### 🔹 2. **Modelos No Paramétricos**

* **Árboles de Decisión**, **Random Forest**, **Gradient Boosting** y **XGBoost**: modelan relaciones complejas sin suponer una forma funcional explícita.
* Flexibles, pero pueden sobreajustar si no se regulan.

### 🔹 3. **Regresión con Splines**

* Divide el dominio de la variable en tramos y ajusta funciones diferentes (por ejemplo, polinomios) en cada tramo.
* Es suave y flexible para capturar formas no lineales.

### 🔹 4. **Modelos Basados en Kernels**

* **SVM con kernel RBF**, **Kernel Ridge Regression**, entre otros.
* Usan transformaciones no lineales implícitas para separar/predicir datos en espacios de mayor dimensión.

### 🔹 5. **Redes Neuronales**

* Capturan relaciones altamente no lineales.
* Útiles con muchos datos y relaciones complejas.
* Requieren más recursos y tiempo de entrenamiento.

### 🔹 6. **Transformaciones de Variables**

* Aplicar funciones como logaritmos, raíces, exponenciales a las variables para linealizar relaciones no lineales.

### 📌 En resumen:

Cuando la relación entre variables no es lineal, puedes:

* Usar **regresión polinómica**.
* Aplicar **transformaciones**.
* O directamente cambiar a **modelos más flexibles**, como árboles o redes neuronales.