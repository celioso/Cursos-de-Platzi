# Curso de Álgebra Lineal Aplicada para Machine Learning

## Descomposición de Matrices y Su Aplicación en Machine Learning

La **descomposición de matrices** es una herramienta fundamental en **álgebra lineal aplicada al Machine Learning**, ya que permite simplificar cálculos complejos, reducir la dimensionalidad y extraer información estructural de los datos. A continuación te explico los **principales tipos**, sus **aplicaciones** y ejemplos prácticos.

### ✅ ¿Qué es la Descomposición de Matrices?

Consiste en **descomponer una matriz en varios factores** o submatrices con propiedades especiales que facilitan:

* La solución de sistemas de ecuaciones
* La reducción de dimensiones
* La compresión de datos
* La mejora del rendimiento en modelos de machine learning

### 🔍 Tipos Principales de Descomposición

### 1. **Descomposición LU (Lower-Upper)**

* Descompone una matriz cuadrada A en:

  $$
  A = L \cdot U
  $$

  donde `L` es triangular inferior y `U` es triangular superior.

* ✅ **Aplicaciones**:

  * Resolver sistemas de ecuaciones lineales.
  * Acelerar algoritmos numéricos.

### 2. **Descomposición QR**

* Descompone A en:

  $$
  A = Q \cdot R
  $$

  donde `Q` es ortogonal (o unitario) y `R` es triangular superior.

* ✅ **Aplicaciones**:

  * Soluciones numéricas estables.
  * Regresión lineal.

### 3. **Descomposición SVD (Singular Value Decomposition)**

* Factoriza A en:

  $$
  A = U \cdot \Sigma \cdot V^T
  $$

  donde `U` y `V` son ortogonales y `Σ` contiene los valores singulares.

* ✅ **Aplicaciones**:

  * **Reducción de dimensionalidad (PCA)**
  * **Recomendadores**
  * **Compresión de imágenes**
  * **Detección de patrones latentes**

### 4. **Descomposición Eig (de autovalores)**

* Para matrices cuadradas:

  $$
  A = V \cdot D \cdot V^{-1}
  $$

  donde `D` es diagonal (autovalores) y `V` contiene los autovectores.

* ✅ **Aplicaciones**:

  * Análisis de componentes principales (PCA)
  * Estabilidad de sistemas
  * Métodos espectrales

### 💡 Ejemplo Práctico en Python (SVD con NumPy)

```python
import numpy as np

A = np.array([[3, 2], [2, 3]])
U, S, VT = np.linalg.svd(A)

print("Matriz U:\n", U)
print("Valores singulares Σ:\n", S)
print("Matriz V^T:\n", VT)
```

### 🤖 Aplicaciones en Machine Learning

| Tipo de descomposición | Aplicación ML destacada                          |
| ---------------------- | ------------------------------------------------ |
| **SVD**                | Recomendadores, PCA                              |
| **LU / QR**            | Solución eficiente de sistemas, regresión lineal |
| **Eig**                | PCA, clustering espectral                        |
| **NMF (No Negativa)**  | Modelado de temas (topic modeling)               |

### Resumen

#### ¿Por qué es importante entender las matrices en data science?

Comprender el uso de las matrices en data science es fundamental para abordar problemas complejos y optimizar procesos. Las matrices permiten realizar transformaciones lineales, facilitando la manipulación y el análisis de datos en gran escala. En muchos casos, especialmente en áreas como machine learning, entender las matrices es clave para mejorar la eficiencia computacional debido a la reducción de dimensiones y al manejo de datos de alta densidad.

#### ¿Qué conceptos previos necesitas?

Es crucial recordar ciertos conceptos que serán tu base para avanzar en este curso. Entre estos:

- **Matrices e Identidad**: Comprender qué es una matriz y las operaciones básicas que puedes realizar.
- **Inversa de una matriz cuadrada**: Saber cómo calcularla y las condiciones bajo las cuales existe.

Estos fundamentos te permitirán ir más allá y aventurarte en el cálculo de autovalores y autovectores, y cómo estos permiten descomponer una matriz. Además, entenderás qué es el SVD y la descomposición en valores singulares.

#### ¿Cómo se relaciona el Álgebra Lineal con Machine Learning?

La relación del álgebra lineal con el machine learning es directa, ya que muchos de los algoritmos utilizados en esta área requieren manipular y transformar grandes volúmenes de datos. Aquí algunos puntos clave:

- **Reducción de dimensionalidad:** Disminuir el número de dimensiones puede llevar a procesos más eficientes sin perder información significativa.
- **Optimización de algoritmos**: Al reducir dimensionalidades, disminuye el tiempo computacional necesario, lo cual es esencial cuando se manejan grandes conjuntos de datos.
- **Transformaciones lineales**: Las matrices permiten transformar y manipular datos eficazmente, lo que es crucial para entrenar modelos de machine learning.

Trabajar con matrices y entender su aplicación práctica te dará ventaja al manejar sistemas de machine learning más complejos, asegurando que tu enfoque sea tanto preciso como eficiente.

#### ¿Qué más aprenderás en este curso?

El propósito de este curso es ir más allá de los fundamentos y explorar temas avanzados de álgebra lineal aplicados a data science. Esto incluye:

- **Cálculo de Pseudo-inversas**: O inversas generalizadas, útiles en sistemas que no tienen una solución única o bien definida.
- **Algoritmo PCA (Análisis de Componentes Principales)**: Este es un método muy utilizado para la reducción de dimensionalidad y análisis exploratorio de datos.
- **Aplicaciones prácticas**: Implementación de estos conceptos en problemas reales, que te permitirá ver en acción las técnicas aprendidas.

Este curso está diseñado no solo para enriquecer tu conocimiento teórico, sino para empoderarte a aplicar estas herramientas de manera efectiva en tus proyectos de ciencia de datos. ¡Sigue adelante y descubre el potencial del álgebra lineal en el mundo del machine learning y data science!

## Transformaciones Lineales con Matrices en Python: Visualización y Análisis

Vamos a abordar el tema **Transformaciones Lineales con Matrices en Python** con un enfoque práctico: entender, visualizar y analizar cómo una matriz puede transformar vectores en el plano.


### 🧠 ¿Qué es una transformación lineal?

Una **transformación lineal** es una función que lleva vectores de un espacio a otro respetando suma y multiplicación escalar. Se representa mediante **multiplicación de una matriz por un vector**.

Si $A$ es una matriz y $\vec{v}$ un vector, entonces:

$$
\text{Transformación: } T(\vec{v}) = A \cdot \vec{v}
$$

### 🛠️ Herramientas en Python

Usaremos:

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 🧪 Ejemplo 1: Transformación en 2D

```python
# Vectores originales (cuadrado unitario)
original = np.array([
    [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
]).T

# Matriz de transformación
A = np.array([
    [2, 1],
    [1, 3]
])

# Aplicar la transformación
transformado = A @ original

# Visualizar
plt.figure(figsize=(6,6))
plt.plot(*original, label='Original', color='blue')
plt.plot(*transformado, label='Transformado', color='red')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Transformación Lineal con Matriz A')
plt.show()
```

### 🔍 Análisis

* La matriz **A** cambia la forma del cuadrado unitario.
* Esta transformación puede **escalar, rotar, reflejar o sesgar** los vectores originales dependiendo de los valores de la matriz.

### 🧭 Ejemplo 2: Escalamiento y rotación

```python
from math import cos, sin, pi

# Escalamiento
S = np.array([
    [2, 0],
    [0, 0.5]
])

# Rotación 45 grados
theta = pi / 4
R = np.array([
    [cos(theta), -sin(theta)],
    [sin(theta), cos(theta)]
])

# Combinar: escalar y luego rotar
T = R @ S
resultado = T @ original

plt.figure(figsize=(6,6))
plt.plot(*original, label='Original', color='blue')
plt.plot(*resultado, label='Escala + Rotación', color='green')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Transformación Lineal: Escala y Rotación')
plt.show()
```

### 📚 Aplicaciones

* **Compresión de imágenes** (PCA).
* **Animación y gráficos por computadora**.
* **Simulación física** y geometría computacional.
* **Machine learning** (transformaciones en espacios latentes).

### Resumen

#### ¿Cómo entendemos las matrices como transformaciones lineales?

Las matrices pueden entenderse como transformaciones lineales que, al aplicarse a un espacio o un vector, generan una transformación. Cuando aplicamos una matriz, podemos afectar a un vector modificando su tamaño o incluso rotándolo. En el mundo de la programación, podemos llevar esto a la práctica utilizando Python y librerías como NumPy y Matplotlib para representar gráficamente estos cambios.

#### ¿Cómo configuramos nuestro entorno en Python para visualizaciones?

Para empezar, necesitamos importar las librerías necesarias. Aquí va un pequeño fragmento de código en Python que nos permitirá ver los gráficos debajo de cada celda de nuestro notebook:

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

Posteriormente, definimos nuestras matrices y vectores usando `numpy`.

#### ¿Cómo definimos y aplicamos una transformación con matrices?

Supongamos que tenemos la siguiente matriz:

`A = np.array([[-1, 3], [2, -2]])`

Queremos investigar qué transformación genera esta matriz al aplicarla al siguiente vector:

`v = np.array([2, 1])`

La transformación de un vector `v` usando una matriz `A` se realiza a través del producto interno de la matriz y el vector. Pero antes de eso, definamos una función para graficar los vectores.

#### ¿Cómo graficamos vectores en Python?

Es útil tener una función versátil para graficar múltiples vectores. Aquí hay una base de cómo podemos definir y utilizar esta función:

```python
def graficar_vectores(vectores, colores, alpha=1):
    plt.figure()
    plt.axvline(x=0, color='grey', lw=1)
    plt.axhline(y=0, color='grey', lw=1)
    for i in range(len(vectores)):
        x = np.concatenate([[0, 0], vectores[i]])
        plt.quiver(*x[::2], *x[1::2], angles='xy', scale_units='xy', scale=1, color=colores[i], alpha=alpha)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()

v_flaten = v.flatten()
graficar_vectores([v_flaten], ['blue'])
```

Esta función ayuda a visualizar cómo han cambiado los vectores al ser transformados por la matriz. Los ejes cruzan en `x=0` y el color de las líneas es `gris`.

#### ¿Cómo se transforman los vectores utilizando matrices?

Cuando aplicamos la matriz `A` al vector `v`, podemos ver el cambio que se produce. Primero, realizamos el cálculo del producto interno:

```python
v_transformado = A.dot(v)
graficar_vectores([v.flatten(), v_transformado.flatten()], ['blue', 'orange'])
```

Aquí, graficamos el vector original `v` junto con el vector `v_transformado` para observar la transformación visualmente, comparando sus posiciones y direcciones.

#### ¿Por qué es importante entender estas transformaciones en Aprendizaje Automático?

Las transformaciones de matrices son fundamentales en el aprendizaje automático, especialmente cuando trabajamos con imágenes o datos que tienen representaciones matriciales. Las matrices permiten transformar estos datos de manera que pueden ser procesados más eficientemente por algoritmos de Deep Learning o Machine Learning.

Entender las representaciones de vectores y matrices y cómo podemos alargar, rotar o modificar su escala es clave para manipular datos estructurados como imágenes, en las cuales cada pixel puede ser parte de una matriz mayor. Cuando llevamos matrices a vectores, este proceso se llama "flatten" y es crucial para el tratamiento de datos en modelos computacionales.

#### ¿Qué papel juegan los determinantes en estas transformaciones?

El determinante de una matriz nos ofrece información valiosa sobre la transformación. Un determinante negativo (como el `-4` en nuestro ejemplo) puede indicarnos que la transformación involucra una inversión o un giro. Por otro lado, si las normas de los vectores antes y después de la transformación se mantienen iguales, puede señalar que hay vectores en el espacio que no cambian su longitud.

```python
determinante = np.linalg.det(A)
print(f"Determinante de la matriz A: {determinante}")
```

En esta exploración, continuamos profundizando en cómo las operaciones de matriz y producto escalar nos ayudan a dar forma a datos y patrones, sentando las bases para descubrimientos más intrincados en el vasto campo del análisis de datos y Machine Learning.