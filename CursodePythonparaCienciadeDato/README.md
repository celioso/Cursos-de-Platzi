# Curso de Python para Ciencia de Dato

## Fundamentos para Análisis de Datos en NumPy y Pandas

Instalar Numpy y Pandas `pip install numpy pandas`

**Lecturas recomendadas**

[NumPy -](https://numpy.org/ "NumPy -")

[pandas - Python Data Analysis Library](https://pandas.pydata.org/ "pandas - Python Data Analysis Library")

[Curso de Python - Platzi](https://platzi.com/cursos/python/ "Curso de Python - Platzi")

## Dimensiones en NumPy y Pandas: De Escalares a Tensors

Las dimensiones juegan un papel fundamental en las estructuras de datos y cálculos en las bibliotecas como NumPy y Pandas. En este contexto, se hace referencia a estructuras de datos que pueden ir desde **escalares** (valores únicos) hasta **tensores** (estructuras multidimensionales). Vamos a recorrer las dimensiones, desde lo más simple hasta lo más complejo.

### 1. **Escalares (Dimensión 0)**

Un **escalar** es un valor único, sin dimensiones ni estructura de datos adicional. En NumPy, un escalar sería un simple número.

- **NumPy**: Un escalar sería un número independiente, como `5`, `3.14` o `True`.
  
  ```python
  import numpy as np
  scalar = np.array(5)
  ```

  Aquí `scalar` es un escalar, un arreglo de dimensión 0 (`ndim=0`).

- **Pandas**: Un valor escalar en Pandas también es un único valor, como un número almacenado en una celda de una Serie o un DataFrame.

### 2. **Vectores (1D: Una Dimensión)**

Un **vector** es una estructura unidimensional, o de una sola fila o columna de datos. En NumPy, esto es conocido como un arreglo de una sola dimensión.

- **NumPy**: Un arreglo unidimensional o vector puede contener múltiples valores en una sola "fila".
  
  ```python
  vector = np.array([1, 2, 3, 4])
  print(vector.shape)  # (4,)
  ```

  Aquí `vector` tiene una forma de `(4,)`, lo que significa que es un vector con 4 elementos.

- **Pandas**: En Pandas, el equivalente a un vector es una **Serie** (`pd.Series`), que es una estructura unidimensional.

  ```python
  import pandas as pd
  series = pd.Series([1, 2, 3, 4])
  print(series.shape)  # (4,)
  ```

  Las Series son indexadas, por lo que cada valor tiene un índice asociado.

### 3. **Matrices (2D: Dos Dimensiones)**

Una **matriz** es una estructura bidimensional, generalmente representada como filas y columnas. Es equivalente a una tabla de datos.

- **NumPy**: En NumPy, una matriz es un arreglo bidimensional.

  ```python
  matrix = np.array([[1, 2], [3, 4], [5, 6]])
  print(matrix.shape)  # (3, 2)
  ```

  La forma `(3, 2)` significa que tiene 3 filas y 2 columnas.

- **Pandas**: En Pandas, el equivalente a una matriz es un **DataFrame** (`pd.DataFrame`), que contiene filas y columnas con etiquetas.

  ```python
  dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
  print(dataframe.shape)  # (3, 2)
  ```

  Un DataFrame tiene etiquetas tanto para las filas como para las columnas.

### 4. **Tensores (3D o Más: Tres Dimensiones o Más)**

Un **tensor** es una estructura multidimensional que puede tener tres o más dimensiones. Es útil para representar datos complejos, como imágenes (3D) o videos (4D).

- **NumPy**: NumPy soporta tensores de cualquier dimensión.

  Ejemplo de un tensor 3D, como una "pila" de matrices:

  ```python
  tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
  print(tensor.shape)  # (2, 2, 2)
  ```

  Aquí, el tensor tiene la forma `(2, 2, 2)`, que significa que contiene 2 matrices, cada una de tamaño 2x2.

- **Pandas**: Pandas generalmente no maneja tensores de forma directa, ya que está diseñado para trabajar principalmente con datos tabulares (hasta 2D). Sin embargo, es posible representar estructuras más complejas utilizando múltiples DataFrames o anidando DataFrames dentro de listas.

### Resumen de Dimensiones en NumPy y Pandas:

| Dimensión | NumPy | Pandas |
| --------- | ------| ------ |
| 0D (Escalar) | `np.array(5)` | Un valor en una celda |
| 1D (Vector) | `np.array([1, 2, 3])` | `pd.Series([1, 2, 3])` |
| 2D (Matriz) | `np.array([[1, 2], [3, 4]])` | `pd.DataFrame(...)` |
| 3D (Tensor) | `np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])` | No disponible directamente |

### Diferencia clave entre NumPy y Pandas:
- **NumPy** es más flexible en términos de dimensiones, permitiendo la manipulación de tensores multidimensionales (3D, 4D, etc.).
- **Pandas** está orientado a la manipulación de datos tabulares, y generalmente se limita a dos dimensiones (filas y columnas).

## Arrays en NumPy

### Métodos de creación de Arrays de NumPy

NumPy proporciona diversas maneras de crear arrays, facilitando la realización de cálculos numéricos y análisis de datos de manera eficiente en Python.

1. Creación de Arrays a partir de Listas

Podemos crear un array a partir de una lista o una lista de listas:

```python
import numpy as np

# Array unidimensional
array_1d = np.array([1, 2, 3, 4])

# Array bidimensional
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
```

2. Creación de Arrays con Funciones Predefinidas

NumPy proporciona funciones predefinidas para crear arrays de manera más rápida y conveniente:

- **np.zeros()**: Crea un array lleno de ceros.

`zeros_array = np.zeros((3, 3))`

- **np.ones()**: Crea un array lleno de unos.

`ones_array = np.ones((2, 4))`

- **np.arange()**: Crea un array con una secuencia de números.

`range_array = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]`

- **np.linspace()**: Crea un array con números igualmente espaciados.

`linspace_array = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]`

3. Especificando Tipos de Datos (Datatypes)

Al crear un array, podemos especificar el tipo de datos que contendrá utilizando el parámetro `dtype`. Esta especificación es crucial para la eficiencia y precisión en cálculos numéricos. Aquí se detallan algunos de los tipos de datos más comunes:

- `int32`: Entero de 32 bits.
- `float32`: Número de punto flotante de 32 bits.
- `float64`: Número de punto flotante de 64 bits (por defecto para números flotantes en NumPy).
- `bool`: Valores booleanos (True o False).
- `complex64`: Número complejo de 64 bits.
- `complex128`: Número complejo de 128 bits.
- `str`: Cadenas de texto.

Podemos especificar estos tipos de datos al crear el array utilizando el parámetro `dtype`:

```python
pythonCopiar código
# Array de enteros
int_array = np.array([1, 2, 3], dtype='int32')

# Array de flotantes
float_array = np.array([1.0, 2.0, 3.0], dtype='float32')

# Array de booleanos
bool_array = np.array([True, False, True], dtype='bool')

# Array de números complejos
complex_array = np.array([1+2j, 3+4j], dtype='complex64')

# Array de cadenas de texto
str_array = np.array(['a', 'b', 'c'], dtype='str')
```

Algunos de estos tipos también pueden ser especificados con abreviaturas en el parámetro `dtype`. Por ejemplo, `'d'` es equivalente a `float64`, que es el tipo de datos de punto flotante de 64 bits en NumPy:

```python
# Creando un array con dtype 'd' (equivalente a float64)
array_float64 = np.array([1, 2, 3], dtype='d')
print(array_float64)
```

4. NaN (Not a Number)

NaN es un valor especial utilizado para representar datos que no son números, especialmente en el contexto de operaciones matemáticas que no tienen un resultado definido. Por ejemplo, la división de cero por cero (0/0) o la raíz cuadrada de un número negativo.

```python
nan_array = np.array([1, 2, np.nan, 4])
print(nan_array)
```

El valor NaN es muy útil para manejar datos faltantes o resultados indefinidos en cálculos numéricos.

NumPy proporciona varios métodos para crear arrays, lo que facilita la generación de diferentes tipos de arrays según las necesidades del usuario. A continuación, te explico los principales métodos para crear arrays en NumPy:

### 1. **Creación a partir de listas o tuplas**

La forma más básica de crear un array es a partir de una lista o tupla de Python. Simplemente se convierte el objeto en un array de NumPy.

```python
import numpy as np

# A partir de una lista
array_from_list = np.array([1, 2, 3, 4, 5])

# A partir de una tupla
array_from_tuple = np.array((1, 2, 3, 4, 5))

print(array_from_list)  # [1 2 3 4 5]
```

### 2. **Arrays de ceros: `np.zeros()`**

Este método crea un array de cualquier dimensión, pero inicializado con ceros.

```python
# Array unidimensional de 5 ceros
zeros_array = np.zeros(5)

# Matriz de 3x3 de ceros
zeros_matrix = np.zeros((3, 3))

print(zeros_matrix)
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]
```

### 3. **Arrays de unos: `np.ones()`**

Similar a `np.zeros()`, pero en este caso el array se llena con unos.

```python
# Array unidimensional de 5 unos
ones_array = np.ones(5)

# Matriz de 3x3 de unos
ones_matrix = np.ones((3, 3))

print(ones_matrix)
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]
```

### 4. **Arrays vacíos: `np.empty()`**

Crea un array sin inicializar sus valores, por lo que contendrá los valores aleatorios de la memoria.

```python
# Matriz de 2x2 vacía
empty_array = np.empty((2, 2))

print(empty_array)
# Los valores son impredecibles, ya que no se inicializan
```

### 5. **Array con un rango de números: `np.arange()`**

Genera un array con números en un rango específico. Similar a la función `range()` en Python, pero devuelve un array de NumPy.

```python
# Array con números del 0 al 9
arange_array = np.arange(10)

# Array con números desde 5 hasta 15 con un paso de 2
arange_with_step = np.arange(5, 15, 2)

print(arange_with_step)  # [ 5  7  9 11 13]
```

### 6. **Array con números equidistantes: `np.linspace()`**

Crea un array de números equidistantes en un intervalo definido, con un número específico de elementos.

```python
# Array con 5 números entre 0 y 1
linspace_array = np.linspace(0, 1, 5)

print(linspace_array)  # [0.   0.25 0.5  0.75 1. ]
```

### 7. **Matriz identidad: `np.eye()`**

Crea una **matriz identidad**, que es una matriz cuadrada con unos en la diagonal y ceros en las posiciones restantes.

```python
# Matriz identidad de 3x3
identity_matrix = np.eye(3)

print(identity_matrix)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 8. **Arrays de números aleatorios: `np.random`**

NumPy proporciona varias funciones dentro del módulo `random` para generar arrays de números aleatorios.

- **`np.random.rand()`**: Genera números aleatorios entre 0 y 1 con una distribución uniforme.

```python
random_array = np.random.rand(3, 3)

print(random_array)
# [[0.1036817  0.87872515 0.33837883]
#  [0.42231567 0.21185037 0.86598657]
#  [0.58776519 0.15980242 0.04907681]]
```

- **`np.random.randint()`**: Genera números enteros aleatorios dentro de un rango.

```python
random_int_array = np.random.randint(0, 10, size=(3, 3))

print(random_int_array)
# [[9 2 6]
#  [4 1 8]
#  [7 3 5]]
```

### 9. **Array con un patrón repetido: `np.tile()`**

Crea un nuevo array repitiendo el array de entrada a lo largo de un número específico de repeticiones.

```python
array = np.array([1, 2, 3])

# Repetir el array 2 veces en filas y 3 veces en columnas
tiled_array = np.tile(array, (2, 3))

print(tiled_array)
# [[1 2 3 1 2 3 1 2 3]
#  [1 2 3 1 2 3 1 2 3]]
```

### 10. **Array de un archivo: `np.loadtxt()` y `np.genfromtxt()`**

NumPy permite cargar arrays desde archivos de texto, CSV u otros formatos. Dos métodos comunes son:

- **`np.loadtxt()`**: Carga datos desde un archivo de texto, asumiendo que todos los valores tienen el mismo tipo.

```python
# Cargar datos desde un archivo de texto
# data = np.loadtxt('data.txt')
```

- **`np.genfromtxt()`**: Similar a `loadtxt()`, pero puede manejar valores faltantes.

```python
# Cargar datos desde un archivo CSV
# data = np.genfromtxt('data.csv', delimiter=',')
```

### Resumen de los métodos de creación de arrays:

| Método | Descripción |
| --- | --- |
| `np.array()` | Convierte listas o tuplas en arrays. |
| `np.zeros()` | Crea un array de ceros. |
| `np.ones()` | Crea un array de unos. |
| `np.empty()` | Crea un array vacío (sin inicialización). |
| `np.arange()` | Genera un array con un rango de números. |
| `np.linspace()` | Genera un array con números equidistantes. |
| `np.eye()` | Crea una matriz identidad. |
| `np.random.rand()` | Genera un array de números aleatorios uniformemente distribuidos entre 0 y 1. |
| `np.random.randint()` | Genera un array de enteros aleatorios. |
| `np.tile()` | Repite un array un número específico de veces. |
| `np.loadtxt()` | Carga un array desde un archivo de texto. |
| `np.genfromtxt()` | Carga un array desde un archivo con manejo de valores faltantes. |

Estos son algunos de los métodos más comunes y útiles para crear arrays en NumPy. ¿Te gustaría explorar alguno en más detalle o ver ejemplos más avanzados?

**Lecturas recomendadas**

[Curso de Estadística y Probabilidad - Platzi](https://platzi.com/cursos/estadistica-probabilidad/ "Curso de Estadística y Probabilidad - Platzi")

[numpy.array — NumPy v2.0 Manual](https://numpy.org/doc/stable/reference/generated/numpy.array.html "numpy.array — NumPy v2.0 Manual")

## Introducción al álgebra lineal con NumPy

![Introducción al álgebra lineal](./images/algebraLinea.png "Introducción al álgebra lineal")

El álgebra lineal es una rama fundamental de las matemáticas que se centra en el estudio de los vectores, las matrices y las transformaciones lineales. NumPy proporciona herramientas para resolver sistemas de ecuaciones lineales, realizar transformaciones geométricas y modelar problemas en diversas áreas de la ciencia y la ingeniería. Su aplicación es tan amplia que se encuentra en el corazón de múltiples disciplinas científicas y tecnológicas, facilitando desde la simulación de fenómenos físicos hasta la optimización de sistemas complejos.

Los vectores y las matrices, los bloques de construcción del álgebra lineal, nos permiten representar y manipular datos de manera eficiente, un vector puede representar una lista de valores que podrían ser coordenadas espaciales, mientras que una matriz puede representar una transformación que afecta a estos vectores. Las operaciones básicas del álgebra lineal, como la suma, la multiplicación y la transposición de matrices, forman la base de muchas técnicas avanzadas en la física, la ingeniería, la economía y la informática.

### Conceptos básicos de álgebra lineal

1. **Vectores**: Son objetos que tienen magnitud y dirección. Se pueden representar como una lista de números, que son las coordenadas del vector.
2. **Matrices**: Son arreglos bidimensionales de números que representan transformaciones lineales. Una matriz puede transformar un vector en otro vector.
3. **Transformaciones Lineales**: Son funciones que toman vectores como entrada y producen otros vectores como salida, respetando las operaciones de suma y multiplicación por un escalar.
4. **Espacios Vectoriales**: Conjuntos de vectores que pueden sumarse entre sí y multiplicarse por escalares, siguiendo ciertas reglas.

### Ejemplos aplicativos

1. **Gráficos por Computadora**: Las transformaciones lineales se utilizan para rotar, escalar y traducir objetos en la pantalla.
2. **Procesamiento de Imágenes**: Las matrices de convolución (kernels) se usan para aplicar filtros a las imágenes, mejorando su calidad o destacando características específicas.
3. **Aprendizaje Automático**: Los algoritmos de regresión lineal, redes neuronales y otros modelos dependen en gran medida de las operaciones matriciales.

### Operaciones principales en álgebra lineal

Vamos a ver algunas de las operaciones más comunes en álgebra lineal utilizando matrices.

### Suma de matrices

La suma de matrices se realiza elemento por elemento. Por ejemplo, si tenemos dos matrices A y B:

![Suma de matrices](./images/sumaMatrices.png "Suma de matrices")

Código en NumPy para la suma de matrices:

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

suma = A + B
print("Suma de matrices:\n", suma)
```

### Multiplicación de matrices

La multiplicación de matrices combina filas de una matriz con columnas de otra. Por ejemplo, si tenemos las mismas matrices A y B:

![Multiplicación de matrices](./images/MultMatrices.png "Multiplicación de matrices")

Código en NumPy para la multiplicación de matrices:

```python
producto = np.dot(A, B)
print("Producto de matrices:\n", producto)
```

### Transposición de Matrices

La transposición de una matriz intercambia sus filas y columnas. Por ejemplo, la transposición de la matriz A es:

![Transposición de Matrices](./images/TransMatrices.png "Transposición de Matrices")

### Determinante de una matriz

El determinante es un valor único que puede calcularse a partir de una matriz cuadrada. Por ejemplo, el determinante de la matriz AAA es:

![Determinante de una matriz](./images/DeterMatrices "Determinante de una matriz")

Código en NumPy para el determinante de una matriz:

```python
determinante = np.linalg.det(A)
print("Determinante de A:", determinante)
```

### Más operaciones de álgebra lineal con NumPy

NumPy ofrece una variedad de funciones que facilitan el trabajo con álgebra lineal. Aquí hay algunas más:

### Inversa de una matriz

La matriz inversa de A:

![Inversa de una matriz](./images/InverMatris.webp "Inversa de una matriz")

```python
inversa = np.linalg.inv(A)
print("Inversa de A:\n", inversa)
```

### Valores y vectores propios

Los valores propios y los vectores propios son fundamentales en muchas aplicaciones, como en la compresión de datos y el análisis de sistemas dinámicos.

```python
valores_propios, vectores_propios = np.linalg.eig(A)
print("Valores propios de A:\n", valores_propios)
print("Vectores propios de A:\n", vectores_propios)
```

### Resolución de sistemas de ecuaciones lineales

Para resolver un sistema de ecuaciones lineales AX=BAX = BAX=B:

```python
B = np.array([1, 2])
X = np.linalg.solve(A, B)
print("Solución del sistema AX = B:\n", X)
```

NumPy es una herramienta poderosa para manejar cálculos numéricos y operaciones de álgebra lineal en Python. Su eficiencia y facilidad de uso la convierten en una biblioteca indispensable para científicos de datos, ingenieros y desarrolladores. Desde la creación de arrays hasta la manipulación de imágenes, NumPy abre un mundo de posibilidades en diversas aplicaciones del mundo real.

Espero que este blog haya despertado tu interés y te haya dado una visión clara de cómo usar NumPy para realizar operaciones de álgebra lineal y mucho más. ¡Sigue explorando y experimentando con NumPy para descubrir todo su potencial! ⚡️

## Indexación y Slicing

En NumPy, la **indexación** y el **slicing** permiten acceder y modificar elementos de arrays de manera muy flexible. Estas técnicas son esenciales para trabajar con arrays de múltiples dimensiones.

### **1. Indexación**

La indexación en NumPy es similar a la de listas de Python, pero se extiende a arrays de múltiples dimensiones. Los índices comienzan en `0`.

#### **1.1. Indexación en arrays unidimensionales**
```python
import numpy as np

# Array unidimensional
arr = np.array([10, 20, 30, 40, 50])

# Acceder al primer elemento
print(arr[0])  # 10

# Acceder al último elemento
print(arr[-1])  # 50
```

#### **1.2. Indexación en arrays multidimensionales**
En arrays multidimensionales, se usa una tupla de índices, donde cada índice corresponde a una dimensión del array.

```python
# Array bidimensional
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Acceder al elemento en la fila 1, columna 2
print(arr_2d[1, 2])  # 6

# Acceder a toda la fila 0
print(arr_2d[0])  # [1 2 3]

# Acceder a toda la columna 1
print(arr_2d[:, 1])  # [2 5 8]
```

#### **1.3. Modificación mediante indexación**
Se puede cambiar el valor de un elemento específico accediendo a él mediante su índice.

```python
# Cambiar el valor en la fila 1, columna 2
arr_2d[1, 2] = 10
print(arr_2d)
# [[ 1  2  3]
#  [ 4  5 10]
#  [ 7  8  9]]
```

### **2. Slicing (Corte de arrays)**

El slicing en NumPy permite extraer una parte del array (subarrays). La sintaxis general para slicing es `arr[inicio:fin]`, donde se incluye `inicio` y se excluye `fin`.

#### **2.1. Slicing en arrays unidimensionales**
```python
# Array unidimensional
arr = np.array([10, 20, 30, 40, 50])

# Extraer elementos desde el índice 1 hasta el 3 (no incluye el 3)
print(arr[1:3])  # [20 30]

# Todos los elementos desde el índice 2 en adelante
print(arr[2:])  # [30 40 50]

# Todos los elementos hasta el índice 3 (sin incluirlo)
print(arr[:3])  # [10 20 30]
```

#### **2.2. Slicing en arrays multidimensionales**
En arrays multidimensionales, se puede aplicar slicing en cada dimensión.

```python
# Array bidimensional
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Extraer un subarray: filas 0 y 1, columnas 1 y 2
subarray = arr_2d[0:2, 1:3]
print(subarray)
# [[2 3]
#  [5 6]]

# Extraer todas las filas y solo la columna 0
print(arr_2d[:, 0])  # [1 4 7]
```

#### **2.3. Slicing con paso**
Se puede definir un paso en el slicing, que indica la cantidad de elementos que se deben saltar.

```python
# Array unidimensional
arr = np.array([10, 20, 30, 40, 50])

# Extraer elementos cada dos posiciones
print(arr[::2])  # [10 30 50]

# Array bidimensional
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Extraer subarray con salto en las filas
print(arr_2d[::2, :])  # [[1 2 3] [7 8 9]]
```

### **3. Indexación booleana**

NumPy permite seleccionar elementos del array en función de una condición. Esto se conoce como **indexación booleana**.

```python
# Array unidimensional
arr = np.array([10, 20, 30, 40, 50])

# Condición: seleccionar elementos mayores a 20
cond = arr > 20
print(arr[cond])  # [30 40 50]
```

### **4. Indexación avanzada**

En NumPy, también puedes seleccionar elementos específicos utilizando listas o arrays de índices.

```python
# Array unidimensional
arr = np.array([10, 20, 30, 40, 50])

# Seleccionar elementos en los índices 0, 2 y 4
print(arr[[0, 2, 4]])  # [10 30 50]

# Array bidimensional
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Seleccionar elementos específicos de varias filas
print(arr_2d[[0, 2], [1, 2]])  # [2 9]
```

### **5. Asignación mediante slicing**

También es posible modificar partes de un array utilizando slicing.

```python
# Array unidimensional
arr = np.array([10, 20, 30, 40, 50])

# Cambiar los primeros tres elementos
arr[:3] = [1, 2, 3]
print(arr)  # [ 1  2  3 40 50]

# Array bidimensional
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Cambiar toda la primera fila
arr_2d[0, :] = [0, 0, 0]
print(arr_2d)
# [[0 0 0]
#  [4 5 6]
#  [7 8 9]]
```

### Resumen de indexación y slicing en NumPy:

- **Indexación**: Se usa para acceder a elementos individuales de arrays de cualquier dimensión.
- **Slicing**: Extrae subarrays especificando rangos (inicio, fin, paso).
- **Indexación booleana**: Selecciona elementos basados en condiciones.
- **Indexación avanzada**: Permite seleccionar varios elementos específicos a la vez.

Estas técnicas son muy poderosas para manipular y acceder a datos de manera eficiente en NumPy. ¿Te gustaría profundizar en algún aspecto específico?