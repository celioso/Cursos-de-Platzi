# Curso de Regresión Logística con Python y scikit-learn

## Regresión Logística con Python y Scikit Learn

Claro, aquí tienes una guía clara y concisa para aplicar **regresión logística con Python y Scikit-Learn**, ideal para resolver problemas de **clasificación binaria** (por ejemplo, si un cliente comprará o no, si un email es spam o no, etc.).

### ✅ **1. Importar librerías necesarias**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

### ✅ **2. Cargar y preparar los datos**

Para el ejemplo, usaremos el dataset de Scikit-Learn de cáncer de mama (binario: maligno/benigno).

```python
from sklearn.datasets import load_breast_cancer

# Cargar dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
```

### ✅ **3. Dividir los datos en entrenamiento y prueba**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ✅ **4. Entrenar el modelo de regresión logística**

```python
model = LogisticRegression(max_iter=10000)  # Ajustamos el número máximo de iteraciones
model.fit(X_train, y_train)
```

### ✅ **5. Hacer predicciones**

```python
y_pred = model.predict(X_test)
```

### ✅ **6. Evaluar el modelo**

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### ✅ **Resultado esperado**

Deberías ver un **accuracy bastante alto (\~95%-98%)**, ya que el dataset de cáncer de mama es limpio y bien balanceado.

### ✅ ¿Qué sigue?

Puedes explorar más:

* Normalización de datos (`StandardScaler`)
* Curvas ROC y AUC
* Regresión logística multiclase (`multi_class='multinomial'`)
* Regularización (`penalty='l1'`, `l2`, etc.)

### Resumen

#### ¿Qué es la regresión logística y cómo se aplica en Machine Learning?

La regresión logística es una herramienta esencial dentro del ámbito de la ciencia de datos y Machine Learning que, por su nombre, puede generar cierta confusión. A pesar de llamarse "regresión", realmente se utiliza para tareas de clasificación. En un mundo donde los datos son el nuevo petróleo, comprender la regresión logística te otorga una ventaja competitiva crucial. Es muy valorada por su capacidad para clasificar datos binarios a partir de un enfoque probabilístico.

#### ¿Cuál es su función en Machine Learning?

En Machine Learning, los algoritmos se dividen en varios tipos, incluyendo los supervisados, no supervisados y de refuerzo. La regresión logística pertenece a la categoría de algoritmos supervisados, específicamente en la familia de clasificación. Su objetivo no es proporcionar un valor continuo, sino prever una clase binaria representada con 0 o 1, verdadero o falso.

#### ¿Cómo funciona la función sigmoidal?

El corazón de la regresión logística es la función sigmoidal. Caracterizada por su forma en "S", esta función transforma valores continuos en la probabilidad de pertenecer a una clase determinada:

- **Rango de la sigmoidal**: De 0 a 1, lo que la alinea perfectamente con los fundamentos de probabilidad.
- **Clasificación binaria**: Si el valor resultante está igual o por encima de 0.5, se clasifica como 1; de lo contrario, como 0.

Este mecanismo de funcionamiento es esencial en la predicción de resultados binarios, como puede ser la aprobación de un examen en función de las horas de estudio dedicadas.

**Ejemplo práctico de regresión logística**

Para ilustrar este concepto, consideremos un escenario educativo. Imagínate que estás evaluando la probabilidad de que un estudiante apruebe un examen basado en las horas de estudio:

1. **0 horas de estudio**: Es probable que no aprueben (clase 0).
2. **Mucho tiempo de estudio**: Es probable que aprueben (clase 1).

Dibujando los datos en un gráfico, las horas de estudio se representan como puntos que, al ser procesados por la función sigmoidal, generan un modelo que predice si el estudiante aprobará o no.

#### Interpretación de la probabilidad

La máxima contribución de la regresión logística es su cualidad de interpretación basada en probabilidades, proporcionando una perspectiva más comprensible de los resultados:

- **Mayor o igual a 0.5**: El estudiante aprueba.
- **Menor a 0.5**: El estudiante no aprueba.

#### Aplicaciones y recomendaciones para el aprendizaje

Aprender a utilizar la regresión logística requiere ciertos conocimientos previos:

- **Matemáticas para Machine Learning**: Comprender cómo las matemáticas se aplican a los modelos de datos.
- **Análisis de datos con Python y Pandas**: Habilidades en el manejo y análisis de datos.
- **Visualización de datos con Matplotlib y Seaborn**: Facilitar la interpretación mediante gráficos.
- **Algoritmos de Machine Learning**: Bases sólidas en regresión lineal y otros conceptos de aprendizaje automático.

Desarrollar una comprensión profunda y práctica de la regresión logística no solo refuerza tu formación en ciencia de datos, sino que también te prepara para abordar problemas reales con confianza y creatividad. ¡Sigue explorando y creciendo en tu camino en el mundo de Machine Learning!

**Archivos de la clase**

[slides-curso-regresion-logistica.pdf](https://static.platzi.com/media/public/uploads/slides-curso-regresion-logistica_9cd293b7-6432-4524-8eb1-e94a707bb9ec.pdf)

**Lecturas recomendadas**

[Classifier Playground](http://www.ccom.ucsd.edu/~cdeotte/programs/classify.html)

## Regresión Logística con Python y Scikit Learn

¡Perfecto! Aquí tienes un ejemplo **completo y comentado** de cómo implementar una **Regresión Logística con Python y Scikit-Learn** para resolver un problema de **clasificación binaria** paso a paso.

### 🧠 Objetivo

Utilizar regresión logística para predecir si una persona tiene o no diabetes, usando el conjunto de datos `Pima Indians Diabetes`.

### 📦 1. Instalar e importar librerías necesarias

```bash
pip install pandas scikit-learn matplotlib seaborn
```

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
```

### 📊 2. Cargar el dataset

Supongamos que tienes el archivo `diabetes.csv` (puedes descargarlo desde [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)):

```python
df = pd.read_csv('diabetes.csv')
print(df.head())
```

### 🧹 3. Preparar los datos

Separar variables independientes (`X`) y la variable objetivo (`y`):

```python
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```

### 🔀 4. Dividir en entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🤖 5. Crear y entrenar el modelo

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### 🧪 6. Realizar predicciones

```python
y_pred = model.predict(X_test)
```

### 📈 7. Evaluar el modelo

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
```

### 📉 8. Visualizar matriz de confusión (opcional)

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
```

### ✅ Resultado esperado

Un `accuracy` entre **70% y 80%**, dependiendo del modelo y del dataset, es común en este problema.

### ¿Quieres avanzar más?

Puedes añadir:

* Regularización (`penalty='l1'`, `'l2'`)
* Evaluación con curva ROC
* Balanceo de clases (`class_weight='balanced'`)
* Escalado de datos con `StandardScaler`

### Resumen

#### ¿Cómo se configura el entorno y se cargan los datos en un modelo de regresión logística?

La regresión logística es una técnica poderosa y versátil en el campo del Machine Learning, especialmente para la clasificación de datos. El uso de Python y Scikit Learn facilita su implementación, permitiéndonos abordar tareas complejas con relativa sencillez. Comenzaremos discutiendo cómo configurar el entorno y cargar eficientemente los datos necesarios.

#### ¿Qué librerías se necesitan?

Para este proyecto, necesitamos varias librerías que nos ayudarán en distintos aspectos del proceso:

- **Scikit Learn**: Esencial para manipular datasets y aplicar regresión logística.
- **Pandas**: Para la manipulación y análisis de datos estructurados.
- **Matplotlib y Seaborn**: Para la visualización de datos.
- **NumPy**: Utilizado para efectuar operaciones sobre matrices y arrays.

Estas librerías, ya precargadas en el entorno, nos permiten trabajar sin complicaciones. El dataset específico que usaremos son imágenes de dígitos escritos a mano, disponibles mediante `LogDigit` desde `Scikit Learn.dataset`.

#### ¿Cómo cargamos los datos?

Iniciamos cargando los datos en un objeto llamado Digits:

```python
from sklearn.datasets import load_digits
digits = load_digits()
```

El objeto `Digits` contiene varias propiedades relevantes, incluyendo los datos (`data`), los nombres de las columnas o características (`feature_names`), y una variable Target, que indica qué dígito está representado en cada imagen.

#### ¿Cómo se visualizan los datos?

Para ver de forma más clara estas imágenes de dígitos, hacemos uso de NumPy para reestructurarlas en un formato de 8x8, que es la estructura documentada en el dataset original.

```python
import numpy as np

image = np.reshape(digits.data[0], (8, 8))
```

Podemos visualizar la imagen utilizando Matplotlib:

```python
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')
plt.show()
```

Esta visualización nos permite entender mejor los datos que estamos manipulando, ofreciendo una base firme para el aprendizaje del modelo.

#### ¿Cómo dividir los datos en entrenamiento y prueba?

Dividir adecuadamente nuestros datos entre conjuntos de entrenamiento y prueba es crucial para validar y evaluar el desempeño de nuestro modelo. Esta responsabilidad no solo sustenta los resultados obtenidos, sino que asegura la fiabilidad del algoritmo ante datos no vistos previamente.

#### ¿Por qué es importante esta división?

La separación de los datos en entrenamiento y prueba permite:

- Asegurar que nuestro modelo no está "aprendiendo de memoria" el dataset completo.
- Validar el modelo con datos que no ha visto antes, permitiéndonos evaluar su precisión de forma objetiva.

#### ¿Cómo hacemos el split de datos?

La función `train_test_split` de Scikit Learn se utiliza para dividir los datos:

from sklearn.model_selection import train_test_split

```python
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)
```

Aquí, `test_size=0.2` indica que el 20% del dataset se utilizará para pruebas. El `random_state` asegura que la división sea reproducible en futuras ejecuciones.

#### ¿Cómo se entrena y evalúa un modelo de regresión logística?

Una vez que los datos están listos y divididos, el siguiente paso es entrenar el modelo de regresión logística. Aquí, se destacará cómo configurar un modelo, entrenarlo, predecir resultados, y finalmente, evaluar su rendimiento.

#### ¿Cómo configurar y entrenar el modelo?

La configuración y entrenamiento del modelo son extremadamente sencillos:

```python
from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression(max_iter=200)
logistic_reg.fit(x_train, y_train)
```

La función `fit` entrena el modelo utilizando el conjunto de entrenamiento.

#### ¿Cómo se realizan predicciones?

Con el modelo entrenado, podemos obtener predicciones en el conjunto de prueba:

`predictions = logistic_reg.predict(x_test)`

Estas predicciones nos permitirán evaluar el desempeño del modelo comparándolas con los valores reales de `y_test`.

#### ¿Cómo evaluamos el modelo?

Para evaluar la efectividad del modelo, utilizamos una matriz de confusión:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
```

Y para visualizarlo:

```python
import seaborn as sns

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, linewidths=0.5, square=True, cmap='coolwarm')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
```

Esta matriz permite identificar mayormente los aciertos y errores del modelo; los valores en la diagonal indican el número de predicciones correctas.

Explorar la regresión logística utilizando Python y Scikit Learn es un excelente punto de partida para adentrarse en el mundo del machine learning. La simplicidad del código y la precisión en la clasificación demuestran la efectividad de esta técnica. Invito a seguir indagando y practicando con modelos más complejos, siguiendo este curso o explorando otros datasets y algoritmos. ¡El aprendizaje nunca termina!

**Archivos de la clase**

[mi-primera-regresion-logistica.ipynb](https://static.platzi.com/media/public/uploads/mi_primera_regresion_logistica_cc427d4e-ac0d-4f86-b38d-dee909ec9aa2.ipynb)

**Lecturas recomendadas**

[MNIST classification using multinomial logistic + L1 — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html)

[Recognizing hand-written digits — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

[The Digit Dataset — scikit-learn 1.1.3 documentation](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html)

[Mi_primera_regresion_logistica.ipynb - Google Drive](https://drive.google.com/file/d/1HQ8jnYgJsXScPo6TJ8vS_6SMd52hEFzh/view?usp=sharing)

## Cuándo usar la regresión logística en modelos de clasificación

La **regresión logística** se usa cuando tienes un problema de **clasificación**, es decir, cuando la variable que quieres predecir (variable dependiente) **es categórica**, como:

* Sí o No
* 0 o 1
* Aprobado o Reprobado
* Enfermo o Sano

### ✅ **Cuándo usar regresión logística**

1. **Cuando el objetivo es clasificar en dos clases (binaria):**

   * Ej: Predecir si un correo es *spam* o *no spam*.

2. **Cuando la relación entre las variables independientes (X) y la probabilidad del evento se puede modelar como una curva sigmoide.**

3. **Cuando los valores de salida deben ser interpretables como probabilidades.**

   * Por ejemplo, la probabilidad de que un paciente tenga una enfermedad.

4. **Cuando se necesita un modelo sencillo, eficiente y rápido de entrenar.**

### 🔢 Tipos de regresión logística

* **Binaria:** Solo dos clases (0 o 1).
* **Multinomial:** Tres o más clases sin orden.
* **Ordinal:** Tres o más clases con orden (p. ej., *bajo, medio, alto*).

### ⚠️ No usar regresión logística cuando...

* La variable objetivo es **continua** (usa regresión lineal u otro modelo).
* Hay relaciones **altamente no lineales** que no pueden ser bien modeladas con una transformación logística (en ese caso, modelos como Random Forest, SVM o redes neuronales pueden funcionar mejor).

### Resumen

#### ¿Cuándo usar la regresión logística?

La regresión logística es una herramienta poderosa para tareas de clasificación y es crucial entender cuándo es apropiado utilizarla. Con su fácil implementación y la capacidad de interpretar coeficientes, es una opción valiosa en el arsenal de modelos de aprendizaje automático. A continuación, descubriremos las ventajas, limitaciones y momentos más adecuados para aplicar este algoritmo.

#### ¿Cuáles son las ventajas de la regresión logística?

Este modelo presenta diferentes beneficios que lo convierten en una opción atractiva:

- **Facilidad de implementación**: Como vimos anteriormente, se puede entrenar un modelo de regresión logística con solo unas pocas líneas de código.
- **Coeficientes interpretables**: Al igual que en la regresión lineal, los resultados que arroja el modelo son comprensibles y se pueden traducir a la realidad.
- **Inferencia de características**: Permite identificar cuán influyentes son las diferentes características en el resultado final de la clasificación.
- **Clasificaciones con niveles de certeza**: No solo indica si el resultado es 0 o 1, sino que aporta un porcentaje de seguridad en dicha clasificación.
- **Excelentes resultados con dataset linealmente separables**: Funciona óptimamente cuando las variables tienen un comportamiento lineal.

#### ¿Qué limitaciones tiene la regresión logística?

A pesar de sus numerosas ventajas, la regresión logística también tiene ciertas limitaciones:

- **Asume linealidad**: Supone que existe una relación lineal entre las variables dependientes, lo cual no siempre ocurre en la práctica.
- **Overfitting en alta dimensionalidad**: Posee tendencia al overfitting cuando se enfrenta a datasets con muchas características.
- **Problemas con la multicolinearidad**: La presencia de características altamente correlacionadas puede afectar negativamente el rendimiento del modelo.
- **Requiere datasets grandes para mejores resultados**: Los datasets pequeños pueden no proporcionar la cantidad suficiente de información para un modelo preciso.

#### ¿Cuándo es ideal utilizar la regresión logística?

Este modelo es particularmente útil en las siguientes situaciones:

- Cuando se buscan soluciones sencillas y rápidas.
- Para estimar probabilidades de ocurrencia de un evento (clasificación binaria).
- En datasets que son linealmente separables y tienen grandes volúmenes de datos.
- Ideal si el dataset está balanceado, con proporciones similares de las clases a estudiar.

#### ¿Por qué no utilizar la regresión lineal para clasificación?

Mientras que la regresión lineal pretende encontrar una recta que explique el comportamiento de los datos de forma continua, para datos que necesitan clasificaciones de verdaderos y falsos, este no es el caso. Al trazar una línea recta, podría no discernir adecuadamente entre las clases que se solapan, lo que llevaría a un mal desempeño. La regresión logística, en cambio, transforma la línea recta en una sigmoide que permite mejorar la clasificación al gestionar probabilidades, sirviendo así a su propósito de categorización.

La regresión logística surge como un recurso altamente valioso cuando se busca la clasificación con certeza y simplicidad. Con sus ventajas y desventajas claramente delineadas, es crucial saber cuándo elegir y aplicar este método para obtener los resultados deseados. ¡Sigue investigando y ampliando tu conocimiento en esta fascinante área!

## Regresión Logística: Fórmula y Aplicación en Python

¡Claro, Mario! Vamos a ver la **fórmula de la Regresión Logística** y cómo aplicarla en **Python usando Scikit-learn**. Este modelo es muy utilizado para **clasificación binaria** (por ejemplo, predecir si un correo es *spam* o *no spam*).

### 📌 Fórmula de la Regresión Logística

La regresión logística modela la **probabilidad** de que un ejemplo pertenezca a una clase (por ejemplo, clase 1):

$$
P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Donde:

* $z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n$
* $\sigma(z)$: función sigmoide
* $w_0$: intercepto (bias)
* $w_i$: pesos de los atributos $x_i$

### 🧪 Ejemplo en Python con Scikit-learn

Supongamos que queremos predecir si un estudiante pasará un examen basándonos en sus horas de estudio.

### ✅ Paso 1: Importar librerías

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
```

### ✅ Paso 2: Crear un dataset simple

```python
# Horas de estudio y resultado (1=aprobado, 0=reprobado)
data = {
    'horas_estudio': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'resultado': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['horas_estudio']]
y = df['resultado']
```

### ✅ Paso 3: Dividir datos y entrenar modelo

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)
```

### ✅ Paso 4: Predecir y evaluar

```python
y_pred = modelo.predict(X_test)

print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
```

### ✅ Paso 5: Predecir probabilidad

```python
nuevas_horas = np.array([[4.5]])
probabilidad = modelo.predict_proba(nuevas_horas)
print(f"Probabilidad de aprobar con 4.5 horas: {probabilidad[0][1]:.2f}")
```

### Resumen

¿Cómo funciona la fórmula de la regresión logística?
La regresión logística es un algoritmo crucial para la clasificación de datos, permitiéndonos predecir la probabilidad de un evento binario, como "sí" o "no", "verdadero" o "falso", "positivo" o "negativo". Para lograrlo, utilizamos la función sigmoide. Esta función, representada por la fórmula ( P = \frac{1}{1 + e^{-\zeta}} ), convierte cualquier valor en una probabilidad comprendida entre 0 y 1. Pero, ¿cómo se lleva a cabo este proceso y cuál es la base matemática detrás de esta operación?

#### ¿Qué es la función sigmoide?

La función sigmoide es una función matemática que transforma cualquier valor real en un valor comprendido entre 0 y 1, adquiriendo una forma de "S" al graficarse. Esta función es particularmente útil en regresión logística, pues nos permite trabajar con probabilidades:

```python
import numpy as np
import matplotlib.pyplot as plt

# Definir una función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Crear un rango de datos entre -10 y 10
z = np.linspace(-10, 10, 100)

# Calcular la función sigmoide
sigmoid_values = sigmoid(z)

# Graficar la función
plt.plot(z, sigmoid_values)
plt.title('Función Sigmoide')
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.grid(True)
plt.show()
```

Al aplicar la función sigmoide, cualquier dato recibido, sin importar su magnitud, se transformará en un valor entre 0 y 1, ideal para representar probabilidades y hacer predicciones.

#### ¿Cómo los "odds" y los "log odds" contribuyen a la regresión logística?

Un concepto fundamental en regresión logística es el de los "odds", que expresan la probabilidad del éxito de un evento sobre la probabilidad de su fracaso. Por ejemplo, si tenemos una probabilidad de éxito de 80%, los "odds" serían:

[ \text{odds} = \frac{0.80}{1 - 0.80} = 4 ]

Los "log odds" se emplean para manejar mejor los infinitos, ya que al aplicar el logaritmo natural a los "odds", toda la información se centra alrededor del cero, permitiendo a los algoritmos procesar estos valores de forma más efectiva:

[ \text{log odds} = \ln(\text{odds}) ]

#### ¿Cuál es la relación entre la regresión lineal y la regresión logística?

La regresión logística se basa en las mismas premisas que la regresión lineal, aunque con un objetivo diferente: predecir una probabilidad en lugar de un valor continuo. Utilizamos una fórmula similar a la de la regresión lineal:

[ \beta_0 + \beta_1 \cdot x ]

Aquí, (\beta_0) representa el intercepto y (\beta_1) la pendiente. En regresión logística, este modelo lineal se introduce en la función sigmoide para obtener probabilidades.

Para ilustrar cómo estas piezas se integran, veamos cómo se transforma la fórmula de la regresión lineal en una fórmula de regresión logística:

[ P = \frac{e^{\beta_0 + \beta_1 \cdot x}}{1 + e^{\beta_0 + \beta_1 \cdot x}} ]

#### ¿Por qué es útil la regresión logística?

La regresión logística permite abordar problemas de clasificación binaria de manera eficiente y precisa. Al convertir valores continuos en probabilidades, facilita la toma de decisiones basada en datos. Esta capacidad de asignar una probabilidad a cada caso nos permite clasificar con certeza eventos como un diagnóstico médico, la aprobación de un crédito, o el resultado de un partido deportivo.

A medida que amplíes tus conocimientos en machine learning, descubrirás que la regresión logística es solo la punta del iceberg. Existen numerosos algoritmos y métodos para abordar problemas de clasificación y predicción. Sin embargo, entender las bases de la regresión logística te brindará una ventaja significativa en el mundo del análisis de datos. ¡Sigue explorando y construyendo habilidades valiosas en este campo!

**Lecturas recomendadas**

[Classifier Playground](http://www.ccom.ucsd.edu/~cdeotte/programs/classify.html)

## Regresión Logística Aplicada a Dataset Binomial de Churn

La **Regresión Logística aplicada a un dataset binomial de Churn** (abandono de clientes) es una técnica muy común en análisis de datos para predecir si un cliente **se quedará (0)** o **se irá (1)**, usando variables como edad, ingresos, uso del servicio, etc.

Aquí tienes una guía clara y concisa con **explicación + código en Python con Scikit-Learn**:

### ✅ 1. ¿Qué es la regresión logística?

Es un modelo de clasificación supervisada usado cuando el **output es binario** (por ejemplo: `0` = se queda, `1` = se va).

La fórmula general es:

$$
P(y = 1 | X) = \frac{1}{1 + e^{-(w_0 + w_1x_1 + w_2x_2 + \cdots + w_nx_n)}}
$$

### 📊 2. Dataset de Churn (ejemplo simulado)

Supón que tienes un dataset `churn.csv` con columnas como:

* `edad`
* `ingresos`
* `uso_mensual`
* `tiempo_en_meses`
* `churn` (0 = se queda, 1 = se va)

### 🧪 3. Aplicación en Python

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar el dataset
df = pd.read_csv("churn.csv")

# 2. Separar características y etiqueta
X = df[["edad", "ingresos", "uso_mensual", "tiempo_en_meses"]]
y = df["churn"]

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 5. Predicciones y evaluación
y_pred = modelo.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 📈 4. ¿Cómo interpretar?

* **Confusion Matrix**: Muestra los verdaderos positivos, negativos, falsos positivos y negativos.
* **Precisión / Recall / F1-Score**: Evaluación de la calidad del modelo.

### Resumen

#### ¿Cómo aplicar la regresión logística desde cero?

La regresión logística es una poderosa herramienta dentro del aprendizaje automático y la inteligencia artificial utilizada principalmente para problemas de clasificación. Este proceso, que empieza desde la preparación de los datos hasta la implementación del modelo, es fundamental para obtener resultados precisos y confiables. Descubramos cómo aplicar la regresión logística en un proyecto desde cero.

#### ¿Qué es la regresión logística y cómo se clasifica?

La regresión logística es un tipo de modelo estadístico que se utiliza para predecir resultados binarios en una muestra de datos. A este tipo de problemas se les llama comúnmente "dataset binomiales". Un ejemplo clásico es predecir si un cliente de una compañía hará "churn" (es decir, cancelará su suscripción) o no. En general, la regresión logística se especializa en:

- **Datasets binomiales**: con solo dos resultados posibles (0 o 1, verdadero o falso, sí o no).
- **Datasets multinomiales**: con más de dos posibles clasificaciones, aunque la especialidad de la regresión logística es con datasets binomiales.

#### ¿Cómo preparar los datos efectivamente?

Una parte crítica del proyecto es la preparación de los datos. Un buen procesamiento te ayudará a obtener resultados más precisos y eficientes. Aquí te presento los pasos esenciales del proceso:

1. **Eliminar duplicados** y procesar valores nulos para evitar sesgos en el modelo.
2. **Remover columnas innecesarias** que no aporten valor a la clasificación.
3. **Convertir datos categóricos en numéricos**, ya que los algoritmos de machine learning funcionan mejor con números.
4. **Escalar los datos** para facilitar el manejo del algoritmo.

### ¿Qué dataset se utiliza para este proyecto?

Para este proyecto, se utiliza un dataset de "churn" de Kaggle, que se relaciona con el evento en el que un cliente da de baja los servicios de una compañía. Las características del dataset incluyen:

- **Servicios contratados**: como teléfono, línea de internet, seguridad online, etc.
- **Información del cliente**: tipo de contrato, método de pago, facturación, etc.
- **Datos demográficos**: género, edad, rango salarial, entre otros.

#### ¿Cómo implementar la limpieza y transformación de datos en Python?

A continuación, se presenta un extracto del código en Python necesario para la preparación de datos usando librerías comunes como Pandas y NumPy:

```python
# Importar librerías necesarias
import pandas as pd
import numpy as np

# Cargar los datos
df_data = pd.read_csv('ruta/al/dataset.csv')

# Verificar y transformar columnas numéricas
df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')

# Manejar valores nulos
df_data.dropna(inplace=True)

# Eliminar columnas innecesarias
df_data.drop('customerID', axis=1, inplace=True)

# Convertir la variable objetivo a numérica
df_data['Churn'] = df_data['Churn'].replace({'Yes': 1, 'No': 0})

# Aplicar One-Hot Encoding a variables categóricas
df_data = pd.get_dummies(df_data)
```

#### ¿Qué sigue después de la limpieza de datos?

Después de la limpieza y transformación inicial de los datos, el siguiente paso es lidiar con la multicolinealidad y escalar los datos. Estos pasos son cruciales para asegurar que el modelo de regresión logística funcione de manera coherente y con mayor precisión.

Este enfoque metódico asegura resultados sólidos en cualquier proyecto de aprendizaje automático. ¡Sigue aprendiendo y profundizando en cada paso de este proceso! Explorando y convirtiendo datos a su forma más conducente para los algoritmos, establecerás una base robusta para posteriores análisis y modelos predictivos.

**Archivos de la clase**

[regresion-logistica-binomial.ipynb](https://static.platzi.com/media/public/uploads/regresion_logistica_binomial_87729390-4a2c-4332-9fee-d8f5397f550c.ipynb)

**Lecturas recomendadas**

[Telco Customer Churn | Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

[regresion_logistica_binomial.ipynb - Google Drive](https://drive.google.com/file/d/1q7QYevfV-hfGPaiSFnAdxAUbxwhvSmIG/view?usp=sharing)

## Análisis de Correlación y Escalado de Datos en Pandas

Para realizar un **análisis de correlación** y aplicar **escalado de datos** usando `pandas` (y bibliotecas complementarias como `seaborn`, `scikit-learn` y `matplotlib`), puedes seguir estos pasos clave:

### 📌 1. **Importar librerías necesarias**

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

### 📌 2. **Cargar el dataset**

```python
df = pd.read_csv("ruta/dataset.csv")  # Cambia la ruta por la tuya
print(df.head())
```

### 📌 3. **Análisis de correlación**

#### 📊 Matriz de correlación

```python
correlation_matrix = df.corr(numeric_only=True)  # Solo numéricos
print(correlation_matrix)
```

#### 🔍 Visualización con mapa de calor

```python
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlación")
plt.show()
```

### 📌 4. **Escalado de datos**

#### ➕ **Estandarización (media = 0, desviación estándar = 1)**

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
scaled_df = pd.DataFrame(scaled_data, columns=df.select_dtypes(include=['float64', 'int64']).columns)
```

#### 📈 **Normalización (valores entre 0 y 1)**

```python
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
normalized_df = pd.DataFrame(normalized_data, columns=df.select_dtypes(include=['float64', 'int64']).columns)
```

### 📌 5. **Correlación después del escalado (opcional)**

```python
sns.heatmap(scaled_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlación tras Escalado")
plt.show()
```

### 🧠 ¿Por qué escalar?

* El escalado es útil **antes de aplicar modelos como regresión logística, SVM, KNN o PCA**, que son sensibles a las magnitudes de los datos.
* **La correlación no se ve afectada por el escalado estándar**, ya que mide relación, no magnitud.

### Resumen

#### ¿Cómo realizar un análisis de correlación de datos?

Para entender mejor las relaciones entre los datos y la variable objetivo, el análisis de correlación es vital. En este caso, se trata de comprender cómo las diferentes variables de un conjunto de datos se vinculan con el "churn".

#### ¿Qué es la correlación y cómo se calcula en Pandas?

La correlación mide qué tan cercanas o lejanas están dos variables. Utilizando Pandas, calculamos estas correlaciones con el comando corr(), aplicándolo a las columnas que más interesan, como el churn.

```python
correlation = dataframe.corr()["churn"].sort_values(ascending=True)
correlation.plot(kind='bar')
plt.show()
```

En el ejemplo, se utiliza un gráfico de barras para visualizar las correlaciones, que hemos ordenado de manera ascendente para facilitar su interpretación.

#### ¿Cuáles son las observaciones del análisis de correlación?

Algunas variables, como tener un contrato mes a mes, están altamente correlacionadas con el churn. Si un cliente tiene un contrato mensual, es más probable que abandone el servicio. Sin embargo, otras características, como el género del cliente o tener un servicio telefónico, no tienen relación significativa con el churn.

Además, las características como cuánto tiempo lleva un cliente con el contrato o si tiene un contrato a dos años, están inversamente correlacionadas. Esto indica que mientras más tiempo y mayor dureza tenga el contrato, menor es la probabilidad de churn.

#### ¿Cómo se pueden escalar los datos?

La escalabilidad de los datos es crucial para preparar el dataset para modelos de machine learning. Esto se debe a que las variables están en diferentes escalas y deben ser ajustadas para evitar que el modelo le otorgue una mayor importancia a una sobre otra.

#### ¿Qué es y cómo se usa MinMaxScaler?

MinMaxScaler es una herramienta de `SciKit Learn` destinada a escalar variables a un rango común, usualmente de 0 a 1. Esto se logra fácilmente con el siguiente código:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataframe)

scaled_dataframe = pd.DataFrame(scaled_data, columns=dataframe.columns)
```

#### ¿Cómo llevar los datos escalados a un DataFrame?

Tras escalar los datos, queda un array que debe convertirse nuevamente en un DataFrame para mantener la estructura de columnas:

`scaled_dataframe = pd.DataFrame(scaled_data, columns=dataframe.columns)`

Así, los datos están listos para pasarse al modelo de machine learning, como la regresión logística, que evaluará la probabilidad de churn con mayor precisión.

Este proceso no solo ayuda a mantener la consistencia de los datos, sino también a mejorar la interpretación y el rendimiento del algoritmo de clasificación. Es un paso esencial en el preprocesamiento de los datos en un proyecto de ciencia de datos.

## Análisis Exploratorio de Datos con Visualización usando Seaborn y Matplotlib

Aquí tienes una guía clara para realizar un **Análisis Exploratorio de Datos (EDA)** utilizando **Seaborn** y **Matplotlib**, dos de las bibliotecas más populares en Python para visualización de datos.

### 🧪 Análisis Exploratorio de Datos (EDA) con Seaborn y Matplotlib

### 📦 Paso 1: Importar librerías necesarias

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Opcional para estilos más bonitos
sns.set(style="darkgrid")
```

### 📂 Paso 2: Cargar tus datos

Ejemplo con el dataset de *Titanic*:

```python
df = sns.load_dataset('titanic')
df.head()
```

Si usas un CSV:

```python
df = pd.read_csv('ruta/dataset.csv')
```

### 📊 Paso 3: Visualización Univariada

#### a. Distribuciones numéricas

```python
sns.histplot(data=df, x='age', kde=True)
plt.title('Distribución de Edad')
plt.show()
```

#### b. Variables categóricas

```python
sns.countplot(data=df, x='class')
plt.title('Conteo por Clase')
plt.show()
```

### 📈 Paso 4: Visualización Bivariada

#### a. Categórica vs numérica

```python
sns.boxplot(data=df, x='class', y='age')
plt.title('Boxplot de Edad por Clase')
plt.show()
```

#### b. Numérica vs numérica

```python
sns.scatterplot(data=df, x='age', y='fare', hue='sex')
plt.title('Edad vs Tarifa')
plt.show()
```

### 🧩 Paso 5: Correlaciones

```python
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlaciones')
plt.show()
```

### 📋 Paso 6: Insights y Conclusiones

Después de las visualizaciones, puedes responder preguntas como:

* ¿Qué variables están más correlacionadas con el objetivo?
* ¿Existen valores atípicos?
* ¿Qué grupos presentan comportamientos distintos?

### ✅ Extras útiles

* **Pairplot** para relaciones entre múltiples variables numéricas:

  ```python
  sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue='survived')
  plt.show()
  ```

* **Gráficos de violín** para comparar distribuciones:

  ```python
  sns.violinplot(x='class', y='age', data=df)
  plt.title('Distribución de Edad por Clase')
  plt.show()
  ```

  ### Resumen

#### ¿Cómo se realiza un análisis exploratorio de datos?

El análisis exploratorio de datos (EDA) es un componente crucial en el proceso de análisis de datos. Nos permite comprender mejor las variables de nuestro conjunto de datos y cómo se relacionan entre sí. Para realizar este análisis utilizaremos herramientas de visualización de datos como Seaborn y Matplotlib. Estos son componentes esenciales dentro del ecosistema de Python para análisis de datos y visualización.

Primero, asegurémonos de tener importadas las librerías necesarias. El objetivo es analizar los datos desde su origen y no aquellos que han sido preprocesados. Esto ofrece una visión más clara del comportamiento original de los datos.

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

#### ¿Cómo se comparan las variables categóricas?

El siguiente paso tras importar nuestras librerías es identificar las variables categóricas y visualizarlas. Estas visualizaciones permiten observar cómo las variables categóricas están relacionadas con nuestra variable de interés, en este caso, el churn.

```python
def plotCategorical(column):
    plt.figure(figsize=(10, 10))
    sns.countplot(data=dfdata, x=column, hue='churn')
    plt.show()

categorical_columns = dfdata.select_dtypes(include='object').columns

for column in categorical_columns:
    plotCategorical(column)
```

- Se analiza si hay bias o sesgos en los datos basado en variables como género, partners, dependientes, servicios telefónicos, etc.
- Se observa que, por ejemplo, las personas sin partners tienen un mayor churn, lo cual puede tener sentido dado el contexto de estudio.

#### ¿Cómo se analizan las variables numéricas?

Después de explorar las variables categóricas, es crucial analizar las variables numéricas para entender tendencias o correlaciones dentro de los datos, utilizando gráficos de dispersión y diagramas KDE.

```python
sns.pairplot(dfdata, hue='churn', palette='bright', diag_kind='kde')
plt.figure(figsize=(10, 10))
plt.show()
```

- Los gráficos nos mostraron que las personas que realizan churn suelen tener cargos mensuales altos y poco tiempo en la compañía.
- La variable "tiempo en la compañía" en conjunto con "cargo mensual" mostró que personas con poco tiempo y costos elevados tienden a hacer churn.

#### ¿Qué reveló el análisis sobre la variable 'churn'?

El análisis destacó el impacto significativo de algunas variables en la probabilidad de churn:

- **Cargo mensual**: Tiene una fuerte correlación con churn; cargos más altos están asociados con mayores tasas de churn.
- **Contrato mensual**: Los clientes con contrato mes a mes son más propensos a churn, algo observable en los datos categóricos.
- **Género**: No parece ser una variable determinante en el comportamiento de churn.

Nuestra exploración del dataset ha sido enriquecedora, permitiendo identificar variables clave que contribuyen al churn. Esta información será vital cuando apliquemos algoritmos de regresión logística para solucionar problemas de clasificación binomial en siguientes etapas. ¡Continúa con tu aprendizaje para lograr un modelo predictivo acertado!

## Regresión Logística para Clasificación Binomial

La **Regresión Logística para Clasificación Binomial** es una técnica estadística y de machine learning utilizada cuando el objetivo es **predecir una variable categórica binaria**, es decir, que solo tiene dos posibles resultados, como por ejemplo:

* **Sí / No**
* **0 / 1**
* **Cliente se va / Cliente se queda**
* **Enfermo / Sano**

### ✅ ¿Cuándo usarla?

Usa **regresión logística binomial** cuando:

* Tu variable objetivo es binaria (solo dos clases).
* Quieres estimar **la probabilidad** de que una observación pertenezca a una de esas dos clases.
* Los predictores pueden ser continuos o categóricos.

### 🧪 Fórmula matemática

La fórmula general de la regresión logística es:

$$
P(y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_n X_n)}}
$$

Donde:

* $P(y = 1 | X)$: Probabilidad de que la variable dependiente sea 1.
* $\beta_0$: Intercepto.
* $\beta_i$: Coeficientes del modelo.
* $X_i$: Variables predictoras.

### 🐍 Implementación en Python con Scikit-Learn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Cargar dataset (ejemplo: churn)
df = pd.read_csv('telco_churn.csv')

# Preprocesamiento (ejemplo simple)
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 📈 Métricas de evaluación típicas:

* **Accuracy**: Qué tan seguido acierta el modelo.
* **Precision y Recall**: Especialmente útiles si hay desbalance de clases.
* **ROC AUC**: Área bajo la curva para comparar clasificaciones probabilísticas.

### Resumen

#### ¿Cómo aplicar la regresión logística binomial para resolver problemas de clasificación?

La regresión logística binomial es un poderoso algoritmo usado para problemas de clasificación, como determinar si un cliente dejará de usar un servicio (churn) o no. Aprender a implementarla y entender sus resultados es esencial para todo apasionado de la ciencia de datos. En este artículo, exploraremos un ejemplo práctico paso a paso utilizando bibliotecas populares de Python como Scikit-Learn.

#### ¿Cómo prepararse para la regresión logística?

El primer paso al implementar la regresión logística es preparar los datos adecuadamente. En nuestro ejemplo, separamos las variables independentes (X) y la variable dependiente (y) en un dataset, asegurándonos de excluír la columna objetivo (la que queremos predecir).

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Supongamos que `df` es nuestro DataFrame inicial.
X = df.drop(columns=['churn'])  # Eliminar columna objetivo
y = df['churn'].values          # Variable objetivo
```

#### ¿Cómo dividir los datos para entrenamiento y pruebas?

Dividir tus datos en subconjuntos de entrenamiento y prueba es crucial para asegurar que tu modelo se desempeña bien en datos no conocidos. El 70% de los datos normalmente se utiliza para entrenamiento y el 30% restante para pruebas.

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)`

#### ¿Cómo entrenar el modelo de regresión logística?

Utilizando Scikit-Learn, entrenar un modelo de regresión logística es directo y eficiente. Después de crear el objeto del modelo, simplemente aplicamos el método fit con nuestros conjuntos de entrenamiento.

```python
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
```

#### ¿Cómo hacer predicciones y evaluar resultados?

El siguiente paso es hacer predicciones utilizando nuestro modelo entrenado y evaluar su precisión.

```python
# Hacer predicciones sobre el conjunto de prueba
predicciones = modelo.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, predicciones)
print(f'Precisión del modelo: {precision * 100:.2f}%')
```

En nuestro ejemplo, logramos una precisión del 79%. Este valor puede variar dependiendo de diversos factores, como ajustes en el preprocesamiento de datos o variaciones en los datos mismos.

#### ¿Qué significa la 'accuracy' y cómo interpretarla?

La 'accuracy' o precisión es un indicador de cuántas de nuestras predicciones fueron correctas en comparación con el total de casos. Aunque una precisión alta sugiere un buen rendimiento, es vital considerar:

- **Desbalanceo de clases**: En problemas donde una clase es mucho más prevalente que otras, la precisión por sí sola podría no ser suficiente para evaluar el modelo.
- **Contexto del problema**: Diferentes áreas pueden tener requisitos de precisión distintos. Un 79% puede ser excelente en ciertos contextos y aceptable en otros.

Al finalizar este proceso, no solo hemos aprendido a aplicar la regresión logística binomial, sino también a interpretar resultados y ajustar nuestros enfoques basados en la comprensión del contexto del problema. ¡Continúa profundizando y mejorando tus habilidades!

## Regresión Logística: Evaluación y Optimización de Modelos

La **regresión logística** es un modelo estadístico ampliamente utilizado para problemas de **clasificación binaria** (por ejemplo: aprobar/reprobar, enfermedad/sano, fraude/no fraude). Aquí tienes una guía completa con los puntos clave para su **evaluación y optimización**:

### 📘 1. Fundamentos de la Regresión Logística

* **Objetivo**: Predecir la probabilidad de que una observación pertenezca a una clase.
* **Función principal**:

  $$
  P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_n X_n)}}
  $$
* **Salida**: Valores entre 0 y 1 → Probabilidades → Se clasifican en clases usando un umbral (por defecto, 0.5).

### 🧪 2. Evaluación del Modelo

### ✅ Métricas más importantes:

| Métrica                   | Descripción                                                               |
| ------------------------- | ------------------------------------------------------------------------- |
| **Accuracy**              | Proporción de predicciones correctas. Peligrosa en clases desbalanceadas. |
| **Precision**             | TP / (TP + FP) → ¿Qué tan precisas son las predicciones positivas?        |
| **Recall (Sensibilidad)** | TP / (TP + FN) → ¿Qué tan bien detecta los positivos?                     |
| **F1 Score**              | Media armónica entre precision y recall. Útil cuando hay desbalance.      |
| **ROC-AUC**               | Área bajo la curva ROC. Evalúa desempeño a todos los umbrales posibles.   |
| **Matriz de Confusión**   | Tabla 2×2 con TP, TN, FP, FN.                                             |

#### Ejemplo en código (usando `sklearn`):

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
```

### 🚀 3. Optimización del Modelo

### 🔍 a) Selección de Variables

* Usa **análisis univariado**, **correlación**, o técnicas automáticas como:

  * **RFE (Recursive Feature Elimination)**
  * **L1 Regularization (Lasso)**

### 🛠 b) Regularización

* **Evita sobreajuste** penalizando coeficientes grandes:

  * L1 (Lasso): fuerza coeficientes a cero → selección de variables.
  * L2 (Ridge): encoge coeficientes sin eliminarlos.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1.0)  # menor C = más regularización
```

### 🔄 c) Validación Cruzada

* Divide el dataset en múltiples particiones para evaluar estabilidad.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(scores.mean())
```

### 📊 d) Optimización del Umbral de Clasificación

* Por defecto es 0.5, pero puedes ajustarlo con base en la curva ROC o maximizando F1.

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
# Escoge umbral que maximice F1, por ejemplo
```

### 🧠 4. Diagnóstico de Errores

* **Revisar casos mal clasificados (FP y FN)** para mejorar el modelo.
* Usar herramientas como **SHAP** o **LIME** para interpretar decisiones del modelo.

### 📌 Conclusión

Una regresión logística **bien evaluada y optimizada** puede ser muy poderosa y robusta, incluso frente a modelos más complejos. La clave está en:

* Elegir buenas variables.
* Aplicar regularización.
* Evaluar con métricas completas (no solo accuracy).
* Ajustar el umbral para el contexto del problema.
* Validar con datos nuevos o cruzados.

### Resumen

#### ¿Cómo la regresión logística evalúa el modelo?

La regresión logística posee una poderosa capacidad para evaluar modelos, utilizando su distintiva forma de S para proyectar los puntos de datos y obtener probabilidades. Pero, ¿cómo logra realmente obtener esos buenos resultados? En este artículo, profundizaremos en esos detalles esenciales para entender por qué la regresión logística es tan eficaz en modelar datos.

#### ¿Cómo utiliza el estimador de máxima verosimilitud (MLE)?

El Estimador de Máxima Verosimilitud (Maximum Likelihood Estimator, MLE) es un algoritmo crucial en la evaluación de modelos de regresión logística. Su función es simple: tomar todas las probabilidades calculadas y realizar una suma ponderada de ellas. Además, se aplica el logaritmo a esta suma, técnica que optimiza el proceso de predicción:

- Las probabilidades positivas se utilizan tal cual, mientras que para las negativas se aplica 1 menos la probabilidad.
- Se obtiene así un rate continuo que indica qué tan bien se hacen las predicciones: cuanto más alto, mejor es la calidad de la predicción.

#### ¿Qué rol juega la función de costo en Machine Learning?

En el ámbito de la inteligencia artificial, no solo se busca optimizar un modelo, sino minimizar el error o la función de costo. Aquí es donde entra en juego el descenso del gradiente, diminuyendo el rate de la función de costo. El objetivo es claro: mejorar la precisión de predicción.

#### ¿Cómo funciona el descenso del gradiente?

- La función de costo es matemática y mide la diferencia entre la predicción del modelo y el valor real.
- A través de derivadas parciales repetidas, se busca el punto más bajo de esta función.
- Al alcanzar el mínimo de la función de costo, se optimizan las predicciones.

#### ¿Cómo calcular la función de costo para una predicción?

El cálculo de esta función implica la diferencia entre las predicciones del modelo y los resultados reales. Supongamos que:

- Para un resultado real de 1, dejamos la probabilidad predicha; si es 0, aplicamos 1 menos la probabilidad.
- Aplica el logaritmo para obtener un valor depurado de la función de costo.

Esto se puede ejemplificar así:

1. Predicción de probabilidad = 0.8, valor real = 1:
- Aplicando el logaritmo, se obtiene un valor de -0.2231.

2. Probabilidad de 0.95, pero valor real = 0:
- Resultado del cálculo da -2.9957.

Finalmente, sumando estos valores y calculando el promedio, se obtiene el valor de la función de costo. Cuanto más bajo sea este valor, mejor será la precisión de las predicciones.

#### ¿Por qué es fundamental entender estos conceptos en Machine Learning?

Dominar estos conceptos es crucial en el ámbito de la inteligencia artificial y el deep learning. Comprender la mecánica detrás de la regresión logística y la optimización del descenso del gradiente permitirá implementar modelos más eficientes. Para aquellos interesados en profundizar, se recomienda cursos en redes neuronales, donde estos temas se abordan con mayor detalle y desde cero, usando herramientas como NumPy.

La comprensión de estos procesos no solo acrecentará el conocimiento técnico, sino que también potenciará la habilidad para implementar modelos predictivos efectivos en el mundo real. ¡Continúa aprendiendo y perfecciona tus habilidades!

## Análisis de Resultados en Modelos de Regresión Logística

El **análisis de resultados en modelos de regresión logística** es clave para interpretar qué tan bien está funcionando tu modelo, especialmente cuando estás resolviendo un problema de **clasificación binaria** (como predecir si un cliente se irá o no, si hay fraude o no, etc.).

Aquí tienes una guía clara con los pasos esenciales y ejemplos en Python:

### 🔍 Análisis de Resultados en Regresión Logística

### 1. **Entrenar el modelo**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Dataset de ejemplo
data = load_breast_cancer()
X, y = data.data, data.target

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 2. **Predicción**

```python
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

### 3. **Métricas de Evaluación**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
```

### 4. **Matriz de Confusión**

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

### 5. **Curva ROC**

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

### 6. **Análisis de Coeficientes**

```python
import pandas as pd
coefs = pd.Series(model.coef_[0], index=data.feature_names)
coefs.sort_values(ascending=False).plot(kind='bar', figsize=(10, 4), title="Coeficientes del Modelo")
plt.tight_layout()
plt.show()
```

Los **coeficientes positivos** indican una mayor probabilidad de pertenecer a la clase positiva, y los negativos a la clase negativa.

### ✅ Conclusión

Con este análisis puedes:

* Evaluar la **precisión general** y el **riesgo de errores tipo I y II**.
* Ver **qué variables tienen más impacto** en la predicción.
* Ajustar tu modelo para mejorar su **capacidad predictiva**.

### Resumen

#### ¿Cuáles son las ventajas de la regresión logística para predicciones?

La regresión logística es una herramienta valiosa para tratar problemas de clasificación binaria en el campo del Machine Learning. Su principal atractivo es su capacidad para no solo predecir clasificaciones binarias como 0 o 1, sino también estimar las probabilidades y el nivel de certeza de cada predicción. Una ventaja significativa es su facilidad para entender la importancia de diferentes características, reflejada en los coeficientes, que indica qué predictores son más relevantes para el resultado esperado.

#### ¿Cómo se interpretan los coeficientes en una regresión logística?

Cuando trabajamos con modelos de regresión logística, los coeficientes nos proporcionan información crucial sobre la importancia de cada variable en la predicción.

- **Coeficientes positivos**: Indican que, a medida que esta característica incrementa, también lo hace la probabilidad de que el resultado sea "1".
- **Coeficientes negativos**: Indican lo contrario, es decir, una disminución en ese predictor aumenta la probabilidad de obtener un resultado de "0".

Por ejemplo, si el "total shares" y el "contract month to month" tienen coeficientes relevantes positivamente, se entiende que estos factores contribuyen a que el usuario decida no continuar con el servicio (churn). Esto se puede visualizar de manera efectiva mediante gráficos de barras que resalten estas correlaciones.

#### ¿Cuál es el papel de la matriz de confusión en la evaluación del modelo?

La matriz de confusión es una herramienta visual clave que ayuda a comprender cómo está funcionando un modelo de clasificación. Proporciona no solo un indicador de la exactitud del modelo, sino también una visión clara de sus errores.

- **True Positives (TP)** y **True Negatives (TN)**: Las predicciones correctas realizadas por el modelo. En el dataset del ejemplo, las veces que el valor real era 0 o 1 y el modelo predijo correctamente.
- **False Positives (FP)** y **False Negatives (FN)**: Errores, donde el valor predicho no coincide con el valor real.

Conocer estas métricas permite calcular otras como el precision, recall, y el F1 score, brindando una evaluación más completa sobre la efectividad del modelo.

#### ¿Cómo mejorar la precisión de un modelo de regresión logística?

Con una comprensión más clara de las características que afectan la predicción, es posible mejorar la exactitud del modelo. Aquí hay algunos consejos prácticos:

1. **Análisis de coeficientes**: Identificar las variables que no aportan significativamente y considerar su eliminación puede ser clave. Unas variables sin relevancia pueden agregar ruido y reducir la calidad de las predicciones.
2. **Balanceo de datos**: Asegurar que el dataset esté balanceado, especialmente en problemas de clasificación binaria, mejora el rendimiento del modelo.
3. **Optimización de hiperparámetro**s: Ajustar adecuadamente los parámetros del modelo puede significar mejoras sustanciales en su capacidad de predicción.

Fomenta a los estudiantes a continuar experimentando, eliminando variables no esenciales y ajustando parámetros para obtener resultados más precisos. Con cada iteración, la comprensión del modelo y la habilidad para mejorar sus predicciones crecen, lo que es un verdadero testimonio del poder del aprendizaje y la práctica continua en Machine Learning.

## Regularizadores L1 y L2 en Regresión Logística

En **regresión logística**, los regularizadores **L1** y **L2** se usan para evitar el **sobreajuste** del modelo al penalizar coeficientes demasiado grandes. Cada uno actúa de manera diferente sobre los parámetros del modelo.

### 🔍 ¿Qué son los Regularizadores?

Cuando entrenas un modelo de regresión logística, estás optimizando una función de pérdida (log-loss) para encontrar los mejores coeficientes (pesos).
Si no se regulariza, el modelo puede ajustarse demasiado a los datos de entrenamiento y generalizar mal a los nuevos.

La regularización agrega una penalización a la función de pérdida:

* **L1 (Lasso):** Penaliza la **suma de los valores absolutos** de los coeficientes.
* **L2 (Ridge):** Penaliza la **suma de los cuadrados** de los coeficientes.

### ⚖️ Diferencias clave

| Característica      | L1 (Lasso)                        | L2 (Ridge)                          |    |             |
| ------------------- | --------------------------------- | ----------------------------------- | -- | ----------- |
| Penalización        | \`λ \* ∑                          | wᵢ                                  | \` | `λ * ∑ wᵢ²` |
| Efecto en los pesos | Fuerza a algunos coeficientes a 0 | Reduce pero no elimina coeficientes |    |             |
| Ideal para          | Selección de variables (sparse)   | Cuando todas las variables importan |    |             |
| Interpretabilidad   | Alta (modelo más simple)          | Menor (modelo más complejo)         |    |             |

### 🧠 En Regresión Logística

La función objetivo regularizada sería:

* **L1:**
  `Loss = LogLoss + α * ∑ |wᵢ|`
* **L2:**
  `Loss = LogLoss + α * ∑ wᵢ²`

### 🧪 Ejemplo en Python

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Cargar datos
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Modelo con L1 (Lasso)
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)  # C es inverso de α
model_l1.fit(X_train, y_train)
print("Accuracy (L1):", accuracy_score(y_test, model_l1.predict(X_test)))

# Modelo con L2 (Ridge)
model_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=1.0)
model_l2.fit(X_train, y_train)
print("Accuracy (L2):", accuracy_score(y_test, model_l2.predict(X_test)))
```

### 📌 Nota sobre el parámetro `C`

* `C` en `LogisticRegression` es el **inverso de la regularización** (`C = 1/λ`)
* **Valores pequeños de `C`** → **mayor regularización**
* **Valores grandes de `C`** → **menor regularización**

### ✅ Conclusión

* Usa **L1** si quieres seleccionar automáticamente las variables más importantes (coeficientes 0).
* Usa **L2** si todas las variables aportan y quieres evitar sobreajuste.
* También puedes usar una **combinación de ambas**: **Elastic Net** (`penalty='elasticnet'` con `l1_ratio`).

### Resumen

#### ¿Qué son los regularizadores en la regresión logística?

Los regularizadores son herramientas fundamentales en el mundo del aprendizaje automático y la ciencia de datos. Su propósito es ayudar a reducir la complejidad de los modelos y, en consecuencia, minimizar el problema del sobreajuste o overfitting. El sobreajuste ocurre cuando un modelo es tan complejo que se ajusta demasiado a los datos de entrenamiento, perdiendo su capacidad para generalizar a datos nuevos.

En esencia, los regularizadores introducen una penalización a la función de costo del modelo, ajustando la intensidad o el peso de los parámetros. Esto se logra mediante los regularizadores L1 y L2, dos de las opciones más comunes en la implementación de regresiones logísticas. Vamos a desglosar cómo funcionan estos métodos y cómo puedes configurarlos en tus modelos.

#### ¿Cómo funcionan los regularizadores L1 y L2? 

#### Regularizador L1

El regularizador L1 añade el peso de la suma de los valores absolutos de todos los parámetros en la regresión logística. La fórmula incluye un término multiplicativo llamado lambda (λ), que es completamente parametrizable:

- **Ventaja**: Este tipo de regularización induce a una mayor probabilidad de que los pesos de muchos de los parámetros sean exactamente cero, lo que efectivamente reduce la complejidad del modelo manteniendo solo los parámetros más significativos.

#### Regularizador L2

Por otro lado, el regularizador L2 utiliza la suma de los valores cuadrados de los pesos de los parámetros. Al igual que el L1, también incluye el parámetro lambda (λ):

- **Ventaja**: Esto tiende a distribuir los errores de manera más uniforme entre los parámetros, lo que puede ser útil en casos donde se necesita una representación más equilibrada de los datos.

#### Lambda (λ) y su importancia

Elegir un valor adecuado para lambda es crucial. Los valores bajos de λ aportan poca penalización y pueden no reducir significativamente el overfitting. En cambio, valores altos pueden llevar al modelo hacia el infravalor o underfitting, donde el modelo es demasiado simple. Ajustar este parámetro es, por lo tanto, esencial para encontrar el balance adecuado.

####¿Cómo configurar los regularizadores en tu modelo?
#### Uso por defecto en regresiones logísticas

Por defecto, las regresiones logísticas suelen utilizar el regularizador L2, aplicando una penalización estándar. Sin embargo, existen otras opciones disponibles, como no usar ninguna penalización o elegir L1, dependiendo de las necesidades específicas del modelo.

#### Configuración de la constante C

La constante C es inversa al valor de λ y determina la fuerza de la penalización. Por defecto, C vale 1. Este valor se puede modificar para afinar el comportamiento del regularizador en tu modelo, repitiendo esta configuración hasta obtener resultados óptimos.

Para aplicar y ajustar estos regularizadores, se recomienda explorar herramientas prácticas como notebooks de Jupyter, donde puedes implementar estas técnicas y observar su efecto en tiempo real.

Recuerda, la clave está en experimentar y ajustar hasta encontrar el correcto balance que minimice el sobreajuste sin comprometer la capacidad del modelo para generalizar. ¡Continúa explorando y mejorando tus modelos!

## Regresión Logística Multiclase: Estrategias y Solvers Efectivos

La **regresión logística multiclase** (o **multinomial**) es una extensión de la regresión logística binaria que permite predecir más de dos clases. Es común en problemas de clasificación como reconocimiento de dígitos, categorías de texto, tipos de enfermedades, etc.

### 🧠 Conceptos Clave

### 📌 1. **Estrategias para clasificación multiclase**

#### a) **One-vs-Rest (OvR)**

* Se entrena un clasificador binario por cada clase contra el resto.
* Ventaja: rápido, simple.
* Desventaja: menos preciso cuando las clases están correlacionadas.
* Usado por defecto en muchos algoritmos, incluido `LogisticRegression` de `sklearn`.

#### b) **Multinomial (Softmax)**

* Modela directamente la probabilidad de cada clase con una función softmax.
* Más preciso cuando hay muchas clases bien diferenciadas.
* Requiere solvers que soporten la opción `multi_class='multinomial'`.

### 📌 2. **Solvers disponibles en `scikit-learn`**

| Solver        | OvR | Multinomial | L1 | L2 | ElasticNet |
| ------------- | --- | ----------- | -- | -- | ---------- |
| **liblinear** | ✅   | ❌           | ✅  | ✅  | ❌          |
| **newton-cg** | ✅   | ✅           | ❌  | ✅  | ❌          |
| **lbfgs**     | ✅   | ✅           | ❌  | ✅  | ❌          |
| **sag**       | ✅   | ✅           | ❌  | ✅  | ❌          |
| **saga**      | ✅   | ✅           | ✅  | ✅  | ✅          |

* ✅ **Recomendado para multiclase multinomial:** `lbfgs`, `newton-cg`, `saga`
* ⚠️ `liblinear` **no** sirve para softmax multiclase.

### 📌 Ejemplo práctico en Python con `scikit-learn`

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Datos: clasificación de flores Iris (3 clases)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Modelo: Regresión Logística Multiclase con softmax
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
clf.fit(X_train, y_train)

# Predicción y evaluación
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### ✅ Buenas Prácticas

* **Escalar los datos**: `StandardScaler` ayuda al entrenamiento eficiente.
* **Evaluar varias métricas**: precisión, recall, F1-score por clase.
* **Evitar `liblinear`** si necesitas softmax verdadero.
* **Usar `saga`** si quieres combinar L1 y L2 (ElasticNet).

### Resumen

#### ¿Qué es la regresión logística multiclase?

La regresión logística multiclase es una extensión de la regresión logística tradicional que se utiliza cuando hay más de dos clases a predecir. Este tipo de regresión se convierte en una herramienta poderosa para clasificar problemas donde las categorías no son simplemente cero o uno, sino que pueden incluir múltiples valores, como triángulos, equis y cuadros, o colores como verde, azul y rojo. Esto es especialmente útil en situaciones donde se requiere una clasificación más precisa y detallada.

#### ¿Cómo funciona la técnica "One vs Rest"?

La técnica "One vs Rest" es una estrategia simple pero eficaz para manejar problemas de clasificación multiclase convirtiéndolos en problemas binomiales. Se realiza evaluando cada categoría posible frente al resto de las categorías, reduciendo así el problema a uno de clasificación binomial.

- Ejemplo: Si tienes tres clases posibles, como triángulos, equis y cuadros, el proceso sería:
 - Determinar si es un triángulo o no (cero o uno).
 - Luego, verificar si es un cuadrado o no.
 - Finalmente, comprobar si es una equis o no.

Al final, elegimos la clase con mayor probabilidad de ser la correcta. Este enfoque simplifica el problema de clasificación múltiple al convertirlo temporalmente en múltiples problemas más sencillos.

#### ¿Qué es la multinominal logistic regression?

La multinominal logistic regression aprovecha la función softmax para evaluar las probabilidades de cada clase posible de manera simultánea. Este método evalúa todas las clases juntas, no separadamente como "One vs Rest", y busca maximizar la probabilidad de la clase correcta.

- **Softmax**: Es una función que convierte las salidas de la red, conocidas como "logits", en probabilidades. Estas probabilidades suman uno y la clase con el mayor valor de probabilidad es elegida para la predicción.

- **Logits**: Estos son valores continuos que representan las salidas antes de convertirlas en probabilidades reales, y permiten calcular la clase probable.

Por ejemplo, si tres clases tienen probabilidades de 0.7, 0.2 y 0.1 respectivamente, softmax seleccionaría la clase con 0.7 como la predicción final al ser la de mayor probabilidad.

#### ¿Cómo se elige el solver adecuado?

Los "solvers" son algoritmos que optimizan el descenso de gradiente para minimizar la función de costo. Elegir el solver incorrecto puede resultar en errores o resultados no óptimos al aplicar regresiones logísticas multiclase.

- **Tipos de solvers**: Incluyen liblinear, lbfgs, y newton-cg, cada uno con sus ventajas y limitaciones.
- **Compatibilidad**: No todos los solvers son compatibles con todas las configuraciones de regresión. Por ejemplo, "liblinear" no puede usarse con multinominal logistic regression y una regularización L2.
- **Datasets grandes**: Solvers como "sag" o "saga" son preferibles para datasets largos debido a su capacidad para manejar efficiently volúmenes grandes de datos.

Elegir el solver adecuado no solo optimiza los resultados de nuestras clasificaciones, sino que también evita errores durante la ejecución de la regresión.

#### Recomendaciones y prácticas al trabajar con regresión logística multiclase

- **Ejecute varias pruebas**: Dada la variedad de técnicas y opciones, probar varias configuraciones puede ayudar a identificar la estrategia que mejor se adecúa a su conjunto de datos particular.
- **Use tablas de compatibilidad**: Las tablas de referencia proporcionan orientación sobre qué combinaciones de técnicas, solvers y regularizadores son viables. Esto ayuda a evitar errores de ejecución y garantiza configuraciones óptimas.
- **Optimización en datasets grandes**: Priorice solvers eficientes como "sag" o "saga" cuando trabaje con grandes volúmenes de datos para mejorar tiempos de procesamiento y resultados.

La regresión logística multiclase es una herramienta versátil y poderosa. Dominar sus métodos y comprender la selección adecuada de solvers te capacita para abordar problemas complejos de clasificación con confianza y eficiencia.

## Clasificación Multiclase con Regresión Logística en Python

Aquí tienes una guía completa y práctica para realizar **clasificación multiclase con regresión logística en Python**, usando `scikit-learn`:

### ✅ Paso 1: Importar librerías necesarias

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
```

### ✅ Paso 2: Cargar y preparar los datos

Usamos el dataset **Iris** (3 clases: Setosa, Versicolor, Virginica):

```python
# Cargar dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### ✅ Paso 3: Entrenar modelo de regresión logística multiclase

```python
# Modelo multiclase con softmax (multinomial)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)
```

### ✅ Paso 4: Evaluar el modelo

```python
# Predicción
y_pred = model.predict(X_test_scaled)

# Reporte de clasificación
print(classification_report(y_test, y_pred, target_names=target_names))

# Matriz de confusión
ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, display_labels=target_names)
plt.title("Matriz de Confusión")
plt.show()
```

### ✅ Paso 5: Visualización (opcional)

Si deseas visualizar en 2D (reduciendo dimensiones), puedes usar PCA:

```python
from sklearn.decomposition import PCA

# Reducir a 2D para graficar
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred, cmap='viridis', edgecolor='k')
plt.title("Predicciones de Regresión Logística Multiclase (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
plt.grid(True)
plt.show()
```

### 📌 Conclusión

Esta implementación demuestra:

* Cómo usar regresión logística para clasificación multiclase.
* Cómo escalar datos y aplicar softmax (`multi_class='multinomial'`).
* Cómo evaluar el modelo con reportes y visualizaciones.

### Resumen

#### ¿Qué es la clasificación múltiple utilizando regresión logística?

La clasificación múltiple es un proceso fundamental en el aprendizaje automático donde se pretende clasificar datos en más de dos categorías diferentes. En el caso de una regresión logística, que se utiliza principalmente para problemas de clasificación binaria, se extiende para abordar problemas de clasificación múltiple. Un ejemplo práctico de esto es el dataset Dry Beans, donde el objetivo es clasificar diferentes tipos de frijoles secos utilizando varias variables numéricas, como el área, el perímetro y la longitud.

#### ¿Cómo preparar un dataset para la regresión logística?

Preparar un dataset de manera adecuada es crucial para el éxito de cualquier modelo de aprendizaje automático. Aquí te presentamos una guía paso a paso sobre la preparación del dataset usado en la regresión logística para múltiples clases:

1. **Carga de Librerías Necesarias**: Se requiere el uso de diversas librerías de Python como Pandas para la manipulación de datos, NumPy para cálculos algebraicos, Matplotlib y Seaborn para la visualización de datos, y Scikit-learn para dividir los datos y aplicar la regresión logística.

2. **Carga y Visualización de Datos**:

```python
import pandas as pd
df = pd.read_csv('ruta/dataset.csv')
print(df.head())
```

3. **Limpieza de Datos**:

- **Eliminación de Duplicados**:

`df.drop_duplicates(inplace=True)`

- **Detección de Valores Nulos**:

`print(df.isnull().sum())`

- Análisis de Outliers:

`df.describe()`

4. **Balanceo del Datase**t: Mediante la técnica de undersampling, se ajustan las clases al tamaño de la clase minoritaria para evitar sesgos.

```python
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(random_state=42)
X_res, y_res = undersample.fit_resample(X, y)
```

¿Cómo transformar variables categóricas a numéricas?

En la regresión logística, es esencial que todas las variables sean numéricas. Las variables categóricas deben transformarse de la siguiente manera:

```python
import numpy as np

# Transformación de variables categóricas a numéricas
unique_classes = list(np.unique(y_res))
y_res.replace(unique_classes, list(range(1, len(unique_classes)+1)), inplace=True)
```

#### ¿Por qué es importante el balanceo de datasets?

Un dataset balanceado es crucial para evitar que el modelo se incline hacia las clases más representativas, lo que podría llevar a un sesgo en las predicciones. Este balanceo se puede lograr mediante técnicas como el undersampling o oversampling.

#### ¿Qué sigue después de preparar el dataset?

Luego de realizar la limpieza y el balanceo del dataset, es importante estandarizar las características del mismo. La estandarización asegura que todas las características tengan una media de cero y una desviación estándar de uno. Este paso se abordará más a fondo junto con el análisis exploratorio en clases posteriores. ¡Te invitamos a continuar explorando y aprendiendo sobre estas técnicas apasionantes en el mundo del aprendizaje automático!

**Lecturas recomendadas**

[regresion_logistica_multiclase.ipynb - Google Drive](https://drive.google.com/file/d/1M1ty-KZ601Kejdpe8mbnI28kPUu9xfWq/view?usp=sharing)

[Dry Bean Dataset | Kaggle](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset)

## Análisis Exploratorio y Escalamiento de Datos para Regresión Logística

Claro, aquí tienes una **guía completa y práctica** para realizar **Análisis Exploratorio de Datos (EDA)** y **escalamiento** antes de aplicar **Regresión Logística** en Python usando `pandas`, `matplotlib`, `seaborn` y `scikit-learn`.

### 📊 1. Análisis Exploratorio de Datos (EDA)

### Paso 1: Cargar los datos

```python
import pandas as pd

df = pd.read_csv('tu_archivo.csv')  # o usar un dataset de sklearn
df.head()
```

### Paso 2: Revisión general

```python
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Paso 3: Distribución de clases (para clasificación)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='target')
plt.title('Distribución de Clases')
plt.show()
```

### Paso 4: Análisis de correlación

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()
```

### Paso 5: Análisis univariado y multivariado

```python
for col in df.select_dtypes(include='number').columns:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.show()
```

```python
sns.pairplot(df, hue='target')
plt.show()
```

### 🔧 2. Preprocesamiento y Escalamiento

### Paso 1: Separar variables

```python
X = df.drop(columns='target')
y = df['target']
```

### Paso 2: Escalar variables numéricas

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### ⚙️ 3. Aplicar Regresión Logística

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 🧪 Opcional: Manejo de datos desbalanceados

Si tu `target` está desbalanceado:

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_scaled, y)
```

### Resumen

#### ¿Por qué es importante realizar un análisis exploratorio de datos?

El análisis exploratorio de datos es crucial para identificar patrones relevantes y posibles correlaciones entre las variables de un dataset. Esto no solo ayuda a mejorar la comprensión de los datos, sino que también optimiza el rendimiento de los modelos predictivos al identificar y eliminar variables que podrían inducir ruido o colinearidad en los datos.

#### ¿Cómo analizamos la correlación entre variables?

En esta lección, se realizó un análisis de correlación visualizando un mapa de calor (heatmap) de las correlaciones entre los atributos del dataset. En este contexto, las correlaciones pueden variar entre -1 y 1:

- **1 o cercanas a 1**: Altamente correlacionadas.
- **0 o cercanas a 0**: No correlacionadas.
- **-1 o cercanas a -1**: Correlación inversa.

El objetivo es descubrir variables altamente correlacionadas que podrían afectar el modelo y decidir si eliminarlas.

#### Ejemplo de código del análisis de correlación:

```python
plt.figure(figsize=(15, 10))
sns.heatmap(dtf.corr(), annot=True)
plt.show()
```

#### ¿Cuáles variables eliminamos y por qué?

A partir del análisis, se decidió eliminar las variables `convex_area` y `equidiameter` debido a su alta correlación con otras variables como `area`, `perimeter`, `length`, y `width`, que podrían conducir a un sobreajuste del modelo.

#### Ejemplo de código para eliminar variables:

`xOver.drop(['convex_area', 'equidiameter'], axis=1, inplace=True)`

#### ¿Cómo visualizamos la distribución de nuestras variables y clases?

La visualización es una herramienta poderosa en el análisis exploratorio. Mediante la creación de diagramas de dispersión y Kernel Density Estimation (KDE), se puede evaluar si las clases dentro de los datos son linealmente separables. Esto facilita entender la estructura de los datos y la selección del método de clasificación.

#### Ejemplo de código para visualización:

`sns.pairplot(df, hue="class")`

#### ¿Por qué realizar el escalamiento y la división del dataset?

El escalamiento de los datos y su posterior división en conjuntos de entrenamiento y prueba son pasos fundamentales para estandarizar los datos, asegurar que el modelo obtenga resultados replicables, y generalice correctamente en nuevos datos que no ha visto.

#### Ejemplo de código para escalamiento y división:

```python
X_train, X_test, y_train, y_test = train_test_split(XOver, YOver, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Conclusiones prácticas

Al aplicar estos pasos, no solo se mejora la calidad del dataset, sino que también se fortalece el conocimiento sobre el negocio y los datos en los que se basa el modelo. Estos conocimientos permiten ajustar las decisiones a lo largo del proceso de modelado para obtener predicciones más precisas y eficaces. ¿Listo para seguir aprendiendo? ¡Avancemos en el próximo módulo para continuar mejorando nuestras habilidades en ciencia de datos!

## Optimización de Modelos de Regresión Logística Multiclase

La **optimización de modelos de regresión logística multiclase** busca mejorar el rendimiento del modelo ajustando sus parámetros, seleccionando características relevantes y evaluando adecuadamente su desempeño. A continuación, te explico los pasos clave con ejemplos en Python:

### 🔢 1. ¿Qué es Regresión Logística Multiclase?

Es una extensión de la regresión logística binaria para problemas con más de dos clases. En `scikit-learn`, se maneja con las estrategias:

* `one-vs-rest` (por defecto): ajusta un clasificador por clase.
* `multinomial`: considera todas las clases al mismo tiempo (requiere solvers específicos).

### 🧰 2. Preparación y Entrenamiento del Modelo

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Dataset de ejemplo
data = load_iris()
X, y = data.data, data.target

# Escalado y split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo base
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)
```

### ⚙️ 3. Optimización con Validación Cruzada y Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10],  # regularización
    'solver': ['newton-cg', 'lbfgs', 'saga'],
    'multi_class': ['multinomial']
}

grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Mejores parámetros:", grid.best_params_)
print("Mejor precisión en validación:", grid.best_score_)
```

### 📈 4. Evaluación del Modelo

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = grid.predict(X_test)

print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
```

### 🧪 5. Consideraciones Avanzadas

* **Regularización**: controla el sobreajuste. Usa `C` más pequeños para mayor penalización.
* **Solvers** recomendados:

  * `lbfgs`: rápido y eficiente para datos pequeños/medianos.
  * `newton-cg`: buena para problemas multiclase.
  * `saga`: compatible con `L1` y grandes volúmenes.
* **Regularización L1 vs L2**:

  * L1 (Lasso): puede eliminar variables irrelevantes.
  * L2 (Ridge): reduce complejidad del modelo sin eliminar variables.

### ✅ Recomendaciones

* Estandariza tus datos antes de entrenar.
* Usa validación cruzada para evitar overfitting.
* Considera `StratifiedKFold` si las clases están desbalanceadas.
* Evalúa con precisión, recall, F1-score y matriz de confusión.

### Resumen

#### ¿Cómo entrenar un modelo de regresión logística multiclase?

La regresión logística es una de las técnicas más utilizadas en la clasificación de datos. Permite categorizar de manera eficaz un conjunto de datos en varias clases, facilitando la comprensión del comportamiento de los mismos. En este sentido, vamos a explicar cómo entrenar un modelo de regresión logística multiclase usando LogisticRegression de la librería Scikit-learn de Python mediante el uso de parámetros como solver, multi_class, y C, así como la iteración sobre diferentes combinaciones para obtener el mejor modelo posible.

#### ¿Qué pasos se siguen para crear el modelo?

Para comenzar, es necesario definir las variables y parámetros que se usarán en el entrenamiento del modelo. Los pasos son:

1. **Definir el modelo**: Utilizamos LogisticRegression especificando parámetros clave. Un ejemplo es el random state para asegurar resultados repetibles.

```python
from sklearn.linear_model import LogisticRegression

logistic_regression_model = LogisticRegression(
    random_state=42,
    solver='saga',
    multi_class='multinomial',
    n_jobs=-1,
    C=1.0
)
```

2. **Crear una función**: Para gestionar de forma dinámica los parámetros, podemos crear una función que acepte los parámetros `C`, `solver` y `multi_class`.

```python
def logistic_model(C, solver, multi_class):
    return LogisticRegression(
        C=C,
        solver=solver,
        multi_class=multi_class,
        n_jobs=-1,
        random_state=42
    )
```

2. **Entrenar al modelo**: Una vez definido, entrenar al modelo con los datos de entrenamiento y realizar predicciones.

```python
model = logistic_model(1, 'saga', 'multinomial')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

4. Evaluar resultados: Es crucial evaluar la precisión del modelo utilizando métricas como la matriz de confusión y el `accuracy score`.

```python
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
print('Confusion Matrix:\n', cm)
print('Accuracy:', accuracy)

```

#### ¿Cómo mejorar el modelo?

Una buena práctica para optimizar el modelo es probar distintas combinaciones de `solver` y `multi_class` y ver cuál proporciona mejores resultados.

1. **Iteración sobre combinaciones**: Utilizar bucles para iterar a través de posibles valores para `multi_class` y `solver`.

```python
multiclass_options = ['ovr', 'multinomial']
solver_list = ['newton-cg', 'saga', 'liblinear', 'sag']

best_score = 0
best_params = {}

for mc in multiclass_options:
    for solver in solver_list:
        try:
            model = logistic_model(1, solver, mc)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            if accuracy > best_score:
                best_score = accuracy
                best_params = {'solver': solver, 'multi_class': mc}

        except Exception as e:
            # Handle exceptions for incompatible configurations
            continue

print('Best Score:', best_score)
print('Best Params:', best_params)
```

2. **Visualizar los resultados**: Utilizar gráficos para analizar los resultados obtenidos y así seleccionar el modelo más adecuado.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=best_params.keys(), y=best_params.values())
plt.title('Scores with different solvers and multi_class options')
plt.xticks(rotation=90)
plt.show()
```

Este proceso puede parecer exhaustivo, pero es crucial para entender el rendimiento de cada configuración y seleccionar el mejor modelo para la clasificación múltiple.

#### ¿Por qué es importante el ajuste de hiperparámetros?

Ajustar los hiperparámetros permite:

- **Obtener un modelo más preciso**: Incrementando la tasa de clasificación correcta.
- **Mejorar la eficiencia computacional**: Adaptando los recursos al problema.
- **Aumentar la robustez del modelo**: Frente a ruido y datos atípicos.

La clave para el éxito en la regresión logística multiclase reside en realizar análisis minuciosos de los resultados y en ajustar los parámetros adecuadamente. De esta manera, podremos garantizar la implementación de un modelo que no solo cumpla con la tarea de clasificación, sino que lo haga con un alto grado de precisión. ¡Continúa explorando y mejorando tus modelos para lograr mejores desempeños en tus proyectos de machine learning!

## Proyecto Final: Diagnóstico de Cáncer de Seno con Regresión Logística

**Lecturas recomendadas**

[Breast Cancer Wisconsin (Diagnostic) Data Set | Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)