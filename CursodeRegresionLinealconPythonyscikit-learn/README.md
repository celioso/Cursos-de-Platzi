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