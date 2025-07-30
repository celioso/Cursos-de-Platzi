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