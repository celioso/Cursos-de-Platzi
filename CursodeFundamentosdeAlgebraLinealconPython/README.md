# Curso de Fundamentos de Álgebra Lineal con Python

## Profesor: Sebastian Sosa
## Version: 09/2019

## Basado en el Segundo Capitulo "2 Algebra Lineal" del libro http://www.deeplearningbook.org/

@book{Goodfellow-et-al-2016,
    title={Deep Learning},
    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
    publisher={MIT Press},
    note={\url{http://www.deeplearningbook.org}},
    year={2016}
}

## Álgebra Lineal para Análisis de Datos y Algoritmos

¡Perfecto! El **álgebra lineal** es fundamental en el análisis de datos y el desarrollo de algoritmos, especialmente en **ciencia de datos**, **machine learning** y **visión computacional**. Aquí tienes una introducción clara y útil:

### 🔷 ¿Qué es Álgebra Lineal?

Es una rama de las matemáticas que estudia **vectores**, **matrices**, **espacios vectoriales** y **transformaciones lineales**. Se utiliza para representar y resolver sistemas de ecuaciones lineales y para manejar datos multivariados de forma estructurada.

### 🔑 Aplicaciones en Ciencia de Datos y Algoritmos

1. **Representación de Datos**

   * Los datos tabulares (filas/columnas) se representan como **matrices**.
   * Cada fila puede ser un vector de características de una observación.

2. **Transformaciones Lineales**

   * Escalado, rotación, reducción de dimensión (PCA), etc., se describen con multiplicación de matrices.

3. **Modelos de Machine Learning**

   * Regresión lineal, redes neuronales, y SVM dependen del cálculo con vectores y matrices.
   * Por ejemplo, una red neuronal aplica pesos (matrices) a vectores de entrada.

4. **Descomposición de Matrices**

   * Técnicas como **SVD (Singular Value Decomposition)** y **eigenvalores** se usan para compresión, recomendadores y reducción de dimensiones.

5. **Solución de Sistemas de Ecuaciones**

   * Resolver $AX = B$ es clave para entender el comportamiento de modelos lineales.

### 🔧 Álgebra Lineal en Python (con NumPy)

```python
import numpy as np

# Vectores
v1 = np.array([1, 2])
v2 = np.array([3, 4])

# Suma y producto punto
print("Suma:", v1 + v2)
print("Producto punto:", np.dot(v1, v2))

# Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# Multiplicación de matrices
print("Multiplicación de matrices:\n", np.dot(A, B))

# Inversa y transpuesta
print("Inversa:\n", np.linalg.inv(A))
print("Transpuesta:\n", A.T)
```

### 🔍 Conceptos Clave

| Concepto          | Uso en Ciencia de Datos                    |
| ----------------- | ------------------------------------------ |
| Vector            | Representa un dato o variable multivariada |
| Matriz            | Conjunto de vectores (dataset)             |
| Producto escalar  | Similaridad entre vectores                 |
| Norma             | Longitud o magnitud del vector             |
| Autovalores       | Variabilidad en reducción de dimensión     |
| Inversa de matriz | Resolver ecuaciones lineales               |

### Resumen

### ¿Por qué es fundamental aprender álgebra lineal?

El álgebra lineal es una herramienta esencial para cualquier persona interesada en ciencias, ingeniería y tecnología. Es la base para entender las operaciones entre matrices y vectores, componentes clave en la mayoría de los algoritmos modernos. Sebastián Sosa, con su experiencia en ciencias matemáticas y análisis de datos, ilustra cómo este saber matemático se aplica en el aprendizaje automático y proyectos de la vida real. Su experiencia resalta la importancia de dominar conceptos que, aunque parecen estáticos, son atemporales en su aplicación.

#### ¿Cómo se aplica el álgebra lineal en proyectos?

Sebastián ha trabajado en proyectos fascinantes que demuestran la aplicabilidad del álgebra lineal. Uno de estos proyectos es un aplicativo que identifica especies de árboles a partir de una foto de una hoja, utilizando algoritmos que requieren de un profundo entendimiento de las matrices y vectores. En otro proyecto, trabajó en un sistema que recomienda canciones basadas en sus letras, una aplicación perfecta del álgebra lineal para procesar grandes volúmenes de datos textuales.

#### ¿Cuáles son los conceptos esenciales del álgebra lineal a estudiar?

En álgebra lineal, es crucial familiarizarse con:

- **Matrices**: Tablas de números que representan datos o transformaciones.
- **Vectores**: Elementos que tienen dirección y magnitud, usados para representar datos espaciales.
- T**ensor**: Una generalización de matrices y vectores, crucial en las ciencias computacionales modernas.

Estos conceptos permiten realizar operaciones que trascienden las matemáticas ordinarias y son esenciales para algoritmos avanzados.

#### ¿Qué beneficios ofrece conocer álgebra lineal a largo plazo?

El conocimiento de álgebra lineal no solo es útil en la actualidad, sino también prepara el camino para adaptarse a las tecnologías del futuro. A medida que los algoritmos y las tecnologías evolucionan, el entendimiento de estos fundamentos matemáticos te permitirá adoptar nuevos avances sin dificultad. Al igual que las operaciones básicas de aritmética permanecen invariables, los principios del álgebra lineal otorgarán solidez a tus habilidades analíticas en cualquier contexto tecnológico.

#### ¿Cómo puedes prepararte para aprender álgebra lineal?

Para iniciar tu viaje en el álgebra lineal, es importante configurar tu entorno de aprendizaje de manera adecuada. Asegúrate de tener acceso a herramientas y recursos que te permitan practicar y aplicar tus conocimientos en casos prácticos y proyectos. Este inicio te proporcionará una base sólida que, como señala Sebastián Sosa, te ayudará a mantenerte relevante y competitivo en un mundo donde los datos son más valiosos que nunca.

Embárcate en el aprendizaje del álgebra lineal con entusiasmo y curiosidad. Desarrollar una comprensión sólida de estos conceptos matemáticos abrirá un mundo de posibilidades en la ciencia y la tecnología.

## Uso de Jupyter Notebook para Análisis de Datos Reproducibles

El **uso de Jupyter Notebook** para análisis de datos reproducibles es una práctica fundamental en ciencia de datos, ya que permite combinar código, visualizaciones y explicaciones en un mismo entorno. Aquí te explico cómo usarlo paso a paso con enfoque en **reproducibilidad**:

### ✅ ¿Qué es un Análisis Reproducible?

Es aquel que puede ser **repetido por otras personas** (o por ti en el futuro) y **obtener los mismos resultados**, siempre que los datos y el entorno sean iguales.

### 🧰 1. Instalar y Abrir Jupyter Notebook

Si usas `conda`:

```bash
conda install notebook
jupyter notebook
```

Con `pip`:

```bash
pip install notebook
jupyter notebook
```

### 📝 2. Estructura Recomendada de un Notebook Reproducible

#### 📌 Secciones típicas:

1. **Título y objetivo**

   ```markdown
   # Análisis de Ventas - Enero 2025
   Este notebook analiza las ventas mensuales para detectar tendencias y patrones.
   ```

2. **Importación de librerías**

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```

3. **Fijar semilla aleatoria** (para reproducibilidad en modelos o simulaciones)

   ```python
   np.random.seed(42)
   ```

4. **Carga de datos**

   ```python
   df = pd.read_csv('ventas.csv')
   ```

5. **Exploración de datos**

   ```python
   df.head()
   df.describe()
   ```

6. **Visualizaciones**

   ```python
   sns.histplot(df['ventas'], bins=10)
   ```

7. **Modelado o cálculos**

   ```python
   promedio = df['ventas'].mean()
   ```

8. **Conclusiones**

   ```markdown
   El promedio de ventas fue de $X. Se observó un pico en la semana 2, relacionado con promociones especiales.
   ```

### 🔁 3. Buenas Prácticas para Reproducibilidad

* Ejecuta todo desde el principio con **Kernel > Restart & Run All**.
* Usa rutas relativas para cargar archivos (`./data/archivo.csv`).
* Anota supuestos, decisiones y pasos clave con celdas de **Markdown**.
* Guarda un `requirements.txt` o `environment.yml` para el entorno.

### 🧪 Ejemplo de Reproducibilidad Técnica

```bash
pip freeze > requirements.txt
```

o con conda:

```bash
conda env export > environment.yml
```

Esto permite que otros creen el mismo entorno con:

```bash
pip install -r requirements.txt
```

o

```bash
conda env create -f environment.yml
```

### Resumen

#### ¿Qué es Jupyter Notebook y por qué es esencial para el análisis de datos?

Jupyter Notebook es una herramienta poderosa en el mundo del análisis de datos que permite mantener código, documentación y gráficos en un único lugar interactivo. Esto facilita la creación de análisis reproducibles y la documentación simultánea del proceso. No solo ahorra tiempo, sino que también garantiza que los resultados se puedan compartir y revisar fácilmente. Para los científicos de datos y analistas, es una plataforma esencial que hace que el flujo de trabajo sea más eficiente y colaborativo.

#### ¿Cómo iniciar Jupyter Notebook con Anaconda?

Iniciar Jupyter Notebook es bastante sencillo si utilizas Anaconda. Este entorno ya viene preconfigurado con las herramientas necesarias para comenzar a trabajar:

1. **Abre Anaconda Navigator**: Busca la opción de abrir Jupyter Notebook desde el dashboard.
2. **Selecciona el entorno adecuad**o: Cambia al entorno que hayas configurado, en este caso, Platzi Fundamentos de AI.
3. **Abrir el explorador de carpetas**: Jupyter abrirá automáticamente una ventana en tu navegador mostrando las carpetas accesibles en tu sistema donde puedes guardar tus datos.
4. **Crear nuevas carpetas**: Puedes crear nuevas carpetas para organizar mejor tu trabajo usando la opción de generar nueva carpeta y luego renombrarla según tus necesidades.

#### ¿Cómo crear y utilizar un notebook en Jupyter?

Una vez que hayas configurado tu entorno y estés en la interfaz de Jupyter, puedes comenzar a crear y utilizar notebooks:

- **Generar un notebook nuevo de Python**:

`# Esto se hace desde la interfaz web de Jupyter.`

- **Renombrar el notebook**: Haz clic en el nombre por defecto para renombrarlo, lo cual ayuda a mantener tu trabajo organizado.

- **Escribir y ejecutar código**:

```python
# Ejemplo de código para conocer la versión de Python
from platform import python_version
print(python_version())
```

Puedes ejecutar este bloque de código presionando `Shift` + `Enter.`

#### ¿Cómo realizar comentarios en tu código?

Hacer comentarios en tu código dentro de Jupyter Notebook es muy práctico y te permite explicar lo que estás haciendo directamente en el entorno. Para comentarios simples, utiliza:

`# Este es un comentario sobre el código`

Si necesitas comentar varias líneas a la vez, selecciona el bloque y presiona `Ctrl + /` para comentar todas las líneas seleccionadas, o para descomentarlas si ya están comentadas.

#### ¿Cómo interactuar con múltiples navegadores en Jupyter?

Jupyter Notebook generalmente abre en tu navegador predeterminado. Si necesitas usar otro navegador, sigue estos pasos:

1. Copia el enlace proporcionado por Jupyter en el navegador abierto.
2. Pega el enlace en la barra de direcciones del navegador que deseas usar.

#### ¿Cuáles son las mejores prácticas para gestionar celdas en Jupyter?

Es importante no exceder las 100 celdas por notebook para mantener un rendimiento óptimo. Si tu análisis requiere más, considera dividir tu trabajo en múltiples notebooks.

#### ¿Cómo reutilizar código de otros notebooks?

Para optimizar tus análisis, puedes crear funciones auxiliares en un notebook y reutilizarlas en otros. Usa el comando de ejecución mágica para ejecutar el código de otro notebook:

`%run path/to/your_script.ipynb`

Este enfoque no solo incrementa la eficiencia, sino que también hace que tus análisis parezcan más profesionales.

¡Sigue explorando y aprovechando al máximo las capacidades de Jupyter Notebook para tus proyectos de análisis de datos! Su reproducción efectiva del análisis y la capacidad de compartir trabajos francamente mejora la colaboración y la comprobación de resultados en equipo.

**Lecturas recomendadas**

[Project Jupyter | Home](https://jupyter.org/)

## Elementos Básicos de Álgebra Lineal en Python: Escalares a Tensores

¡Perfecto! Aquí tienes una introducción clara y práctica sobre los **Elementos Básicos de Álgebra Lineal en Python**, desde **escalares** hasta **tensores**, usando **NumPy**, que es la librería estándar para operaciones numéricas en Python.

### 🔢 Elementos Básicos de Álgebra Lineal en Python

### 1. **Escalares (0D)**

Un escalar es simplemente un número. En NumPy:

```python
import numpy as np

escalar = np.array(5)
print(escalar)         # 5
print(escalar.ndim)    # 0 (dimensión escalar)
```

### 2. **Vectores (1D)**

Un vector es una lista de números (una dimensión):

```python
vector = np.array([1, 2, 3])
print(vector)
print(vector.ndim)     # 1
```

### 3. **Matrices (2D)**

Una matriz es una tabla de números (dos dimensiones):

```python
matriz = np.array([[1, 2], [3, 4]])
print(matriz)
print(matriz.ndim)     # 2
```

### 4. **Tensores (3D o más)**

Un tensor es una estructura de datos de más de 2 dimensiones:

```python
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(tensor)
print(tensor.ndim)     # 3
```

### 🧮 Operaciones básicas

### 🔹 Suma y Resta

```python
a = np.array([1, 2])
b = np.array([3, 4])
print(a + b)  # [4 6]
print(a - b)  # [-2 -2]
```

### 🔹 Producto Escalar

```python
np.dot(a, b)  # 1*3 + 2*4 = 11
```

### 🔹 Multiplicación de Matrices

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
print(np.dot(A, B))
```

### 🧠 Visualizar Dimensiones

```python
print(f'Escalar: {escalar.shape}')
print(f'Vector: {vector.shape}')
print(f'Matriz: {matriz.shape}')
print(f'Tensor: {tensor.shape}')
```

### 📌 En resumen:

| Estructura | Dimensión | Ejemplo                 |
| ---------- | --------- | ----------------------- |
| Escalar    | 0D        | `5`                     |
| Vector     | 1D        | `[1, 2, 3]`             |
| Matriz     | 2D        | `[[1, 2], [3, 4]]`      |
| Tensor     | 3D o más  | `[[[1,2], [3,4]], ...]` |

### Resumen

#### ¿Cuáles son los elementos básicos de la matemática en álgebra lineal?

El aprendizaje de los fundamentos de álgebra lineal es esencial para emprender cualquier estudio de algoritmos avanzados como Machine Learning, Deep Learning y análisis de datos. Los conceptos básicos que exploraremos incluyen: escalar, vector, matriz y tensor.

#### ¿Qué es un escalar?

En matemática, un escalar es simplemente un número único. Esto puede ser un número entero, un punto flotante (número con decimales), o un número complejo. No obstante, en Python, un escalar puede ser más flexible: además de ser cualquier tipo de número, también puede ser un string, o incluso una variable nula conocida como None.

Si quieres profundizar en cómo Python maneja los escalares, Platzi ofrece cursos en los que puedes explorar más sobre las estructuras de datos en Python.

#### ¿Cómo reconocemos y definimos un vector?

Un vector es un conjunto de números ordenados. Imagina una caja donde puedes colocar múltiples números organizados de una manera particular. En Python, los vectores pueden ser creados usando la librería NumPy, un paquete esencial para cualquier persona interesada en cálculos numéricos complejos.

```python
import numpy as np
vector = np.array([1, 2, 3, 4])
```

#### ¿Qué es una matriz?

La matriz es un paso más allá del vector, ya que da un mayor grado de libertad – podemos movernos a través de filas y columnas, creando así un sistema bidimensional de números.

En Python, una matriz también se crea mediante `NumPy`, pero contiene múltiples vectores alineados.

`matriz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`

#### ¿Cómo definimos un tensor?

Finalmente, avanzamos hacia los tensores. Un tensor expande aún más la complejidad al añadir una tercera dimensión (o más) que agrupa múltiples matrices. Un tensor es esencial para realizar cálculos más complejos y para el trabajo con datos de múltiples dimensiones, como las imágenes.

```python
tensor = np.array([
    [[1, 2], [3, 4]], 
    [[5, 6], [7, 8]], 
    [[9, 10], [11, 12]]
])
```

#### ¿Cómo se grafican los tensores?

Es interesante visualizar un tensor cuando contiene datos que podrían corresponder a imágenes, lo cual es común en `Deep Learning.` Usamos la librería `Matplotlib` para su representación visual.

```python
import matplotlib.pyplot as plt

# Configurar para que los gráficos aparezcan debajo de la celda
%matplotlib inline

# Crear y mostrar una visualización del tensor
plt.imshow(tensor[0], interpolation='nearest')
plt.show()
```

#### Consejos prácticos para practicar

Ahora que comprendes los conceptos clave, te animo a practicar creando distintas estructuras:

1. **Escalar**: Crea un escalar en Python con el número 42.
2. **Vector**: Define un vector que contenga los números primos 2, 3, 5 y 7.
3. **Matriz**: Genera una matriz de tamaño 3x2.
4. **Tensor**: Representa un tensor donde la primera fila sea blanca, la segunda negra, y la tercera gris.

Comparte tus resultados y cualquier duda en el sistema de discusiones de la plataforma. ¡Tu aportación enriquece la comunidad y afianci lecho lo aprendido!

## Dimensiones de Escalares, Vectores, Matrices y Tensores en Python

En Python, especialmente usando la biblioteca **NumPy**, podemos representar **escalares, vectores, matrices y tensores** fácilmente, y también observar sus **dimensiones**. Aquí te explico cada uno con ejemplos:

### 🔹 1. **Escalar**

Un escalar es un solo número, sin dimensiones.

```python
import numpy as np

escalar = np.array(5)
print("Escalar:", escalar)
print("Dimensión:", escalar.ndim)  # 0 dimensiones
```

🔸 **Dimensión:** `0`
🔸 **Forma:** `()`
🔸 **Ejemplo de uso:** una constante como temperatura, velocidad, etc.

### 🔹 2. **Vector**

Un vector es una secuencia ordenada de números (una lista unidimensional).

```python
vector = np.array([1, 2, 3])
print("Vector:", vector)
print("Dimensión:", vector.ndim)  # 1 dimensión
```

🔸 **Dimensión:** `1`
🔸 **Forma:** `(3,)`
🔸 **Ejemplo de uso:** coordenadas, características de una instancia, etc.

### 🔹 3. **Matriz**

Una matriz es una tabla de números (bidimensional).

```python
matriz = np.array([[1, 2, 3], [4, 5, 6]])
print("Matriz:\n", matriz)
print("Dimensión:", matriz.ndim)  # 2 dimensiones
```

🔸 **Dimensión:** `2`
🔸 **Forma:** `(2, 3)`
🔸 **Ejemplo de uso:** datos de un dataset, imágenes en escala de grises, etc.

### 🔹 4. **Tensor**

Un tensor es una estructura con más de dos dimensiones.

```python
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("Tensor:\n", tensor)
print("Dimensión:", tensor.ndim)  # 3 dimensiones
```

🔸 **Dimensión:** `3`
🔸 **Forma:** `(2, 2, 2)`
🔸 **Ejemplo de uso:** imágenes RGB, datos de series temporales con múltiples variables, etc.

### 🧠 Resumen

| Tipo    | Ejemplo                   | Dimensión | Forma       |
| ------- | ------------------------- | --------- | ----------- |
| Escalar | `5`                       | 0D        | `()`        |
| Vector  | `[1, 2, 3]`               | 1D        | `(3,)`      |
| Matriz  | `[[1, 2, 3], [4, 5, 6]]`  | 2D        | `(2, 3)`    |
| Tensor  | `[[[1, 2], [3, 4]], ...]` | 3D o más  | `(2, 2, 2)` |

### Resumen

#### ¿Qué son las dimensiones y por qué son importantes en matrices y tensores?

Las dimensiones de los elementos como escalares, vectores, matrices y tensores son un concepto fundamental en el manejo de datos en Python. La comprensión de las dimensiones ayuda a realizar cálculos correctamente y a interpretar los resultados esperados. A menudo, aunque una operación matemática no esté definida estrictamente, Python puede ejecutarla, lo que puede generar confusión. Por ello, entender cómo funcionan las dimensiones es crucial para garantizar que las operaciones que ejecutamos son las correctas.

#### ¿Cómo representamos las dimensiones en Python?

En Python, utilizamos librerías para manejar las dimensiones y las operaciones sobre escalar, vector, matriz y tensor. Veamos algunos ejemplos:

#### Escalar

Un escalar es un elemento que no tiene dimensiones. Es simplemente un valor individual. Cuando verificamos su dimensión utilizando ciertas funciones en Python, el sistema nos indica que no posee este atributo.

#### Vector

Un vector se representa normalmente como un conjunto unidimensional de números. Por ejemplo, un vector con elementos `[1, 2, 3]` se considera que tiene una dimensión con tres elementos. Cuando usamos la función `len` para determinar su tamaño, obtendremos el número de elementos en la primera dimensión.

#### Matriz

Una matriz es una colección bidimensional de números. Por ejemplo, una matriz 2x2 tiene dos elementos en las filas y dos en las columnas. Utilizando la función `len`, se retorna el número de filas, mientras que `shape` proporciona una visión más completa, retornando un arreglo tal como `[2, 2]` que indica filas y columnas.

#### Código de ejemplo para matrices:

```python
import numpy as np

matriz = np.array([[1, 2], [3, 4]])
print("Tamaño del arreglo:", matriz.shape)  # Devuelve (2, 2)
```

#### Tensor

Los tensores son extensiones de las matrices a más dimensiones. Un tensor 3x3x3 es un arreglo tridimensional, donde le primero '3' representa el número de matrices en las primeras dos dimensiones. Este se utiliza comúnmente para representar datos complejos, como una serie de imágenes que cambian con el tiempo, siendo útil en la representación de videos.

#### Ejemplo de manejo de tensores:

```python
tensor = np.ones((3, 3, 3))
print("Forma del tensor:", tensor.shape)  # Devuelve (3, 3, 3)
```

#### Herramientas y funciones útiles para trabajar con dimensiones

Al trabajar con Python, existen múltiples funciones que son útiles para obtener información relacionada a las dimensiones:

- `len()`: Devuelve la cantidad de elementos en la primera dimensión.
- `shape`: Proporciona todas las dimensiones de un objeto como una tupla.
- `size`: Calcula el total de elementos multiplicando todas las dimensiones, útil para entender el tamaño total de un objeto.

#### Diferencias entre `len()` y `shape`

Mientras que `len()` solo retiene el número de elementos en la primera dimensión, `shape` ofrece un panorama completo de todas las dimensiones del objeto. Esto es crucial al tratar con objetos más complejos como matrices y tensores, donde conocer todas las dimensiones nos permite asegurar que las operaciones que implementamos son las correctas.

#### Consejos prácticos para estudiantes

Al comenzar con matrices y tensores en Python:

1. Experimenta creando varios vectores, matrices y tensores para practicar la interpretación de sus dimensiones.
2. Usa la función `shape` en lugar de `len` para obtener información detallada sobre las dimensiones.
3. Realiza operaciones con estos elementos y nota cómo cambian las dimensiones.
4. Ejercítate en casos de uso, como la representación de un video mediante tensores, para familiarizarte con su aplicación práctica.

Continúa explorando y no dudes en experimentar. Las matemáticas y las definiciones pueden parecer complejas, pero con práctica y determinación, dominarás el arte de las dimensiones en Python. ¡Sigue adelante y descubre el fascinante mundo de la programación y el procesamiento de datos!

## Transposición y Suma de Matrices en Python

¡Perfecto! Vamos a ver cómo realizar la **transposición** y la **suma de matrices** en Python utilizando `NumPy`, que es la librería más común para trabajar con matrices y álgebra lineal.

### ✅ 1. Transposición de una matriz

La **transposición** de una matriz invierte sus filas y columnas. En Python, puedes usar `.T` para hacer esto fácilmente.

### 📌 Ejemplo:

```python
import numpy as np

# Definimos una matriz 2x3
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Transpuesta de A
A_T = A.T

print("Matriz A:")
print(A)
print("\nTranspuesta de A:")
print(A_T)
```

### ✅ 2. Suma de matrices

Para **sumar dos matrices**, estas deben tener las mismas dimensiones.

### 📌 Ejemplo:

```python
import numpy as np

# Matrices 2x2
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Suma de A y B
C = A + B

print("Matriz A:")
print(A)
print("\nMatriz B:")
print(B)
print("\nSuma A + B:")
print(C)
```

### ⚠️ Notas importantes:

* Ambas matrices deben tener **las mismas dimensiones** para poder sumarse.
* La transposición no cambia los valores, solo su **ubicación** en filas y columnas.

## Suma de Matrices con Dimensiones Distintas usando Broadcasting

En **NumPy**, puedes sumar matrices con dimensiones distintas **usando broadcasting**, pero **solo si sus formas son compatibles**. Veamos cómo funciona y luego un ejemplo claro.

### 📘 Reglas de Broadcasting en NumPy

Dos dimensiones son **compatibles** si:

1. Son **iguales**, o
2. **Una de ellas es 1**

NumPy "extiende" automáticamente la dimensión con tamaño 1 para que coincida con la otra.

### 🧮 Ejemplo práctico

Supón que tienes:

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])        # Forma (2, 3)

B = np.array([[10],
              [20]])             # Forma (2, 1)
```

La forma de `A` es `(2, 3)`
La forma de `B` es `(2, 1)`

✅ Las dimensiones son compatibles:

* A: (2, **3**)
* B: (2, **1**) → se "expande" a (2, 3)

Entonces puedes hacer:

```python
C = A + B
print(C)
```

**Resultado:**

```text
[[11 12 13]
 [24 25 26]]
```

### ❌ Ejemplo no compatible

```python
A = np.array([[1, 2, 3]])
B = np.array([[1, 2]])

# A.shape = (1, 3)
# B.shape = (1, 2)

# Esto generará un error porque las dimensiones 3 y 2 no son compatibles.
C = A + B  # ❌ ValueError
```

### ✅ Usos típicos de Broadcasting

* Sumar un **vector a una matriz** fila por fila o columna por columna.
* Operaciones entre **matrices y escalares**.
* Ajuste de dimensiones para datos en ML y ciencia de datos.

### Resumen

#### ¿Cuándo es posible sumar matrices de dimensiones distintas?

En el apasionante mundo de la programación y el análisis numérico, sumar matrices de distintas dimensiones a menudo trae consigo desafíos intrigantes. En esta clase, exploramos cuándo y cómo es posible llevar a cabo esta operación, aplicando las reglas del broadcasting en arrays de Numpy, una librería muy útil en Python. Comprender el broadcasting es clave para expandir tus habilidades de manipulación de matrices.

#### ¿Qué es el broadcasting en Numpy?

El broadcasting es un concepto esencial en Numpy que permite realizar operaciones aritméticas en matrices de diferentes dimensiones. En esencia, Numpy extiende la matriz de menor dimensión para que coincida con la de mayor dimensión, siempre que ello sea posible siguiendo ciertas reglas. Esta capacidad aumenta significativamente la flexibilidad y eficiencia al trabajar con arrays.

#### ¿Por qué numpy arroja error al sumar matrices y vectores no compatibles?

En ocasiones, nos vemos con la tarea de sumar un vector a una matriz. Supongamos que tenemos una matriz de dimensiones 3x2 y un vector de longitud 3. Numpy mostrará un error porque las dimensiones no coinciden para el broadcasting. La matriz de 3x2 necesita, al menos, un vector compatible que sea de longitud 2 (el número de columnas) para extender el vector por filas.

#### ¿Cómo solucionar errores de dimensiones en la suma?

Una solución a este error es transponer la matriz cuando es posible. Cambiar las dimensiones de la matriz a 2x3, por ejemplo, permite sumar un vector de tres elementos. Este proceso funciona ya que, al transponer, la matriz y el vector se vuelven compatibles, permitiendo a Numpy extender el vector para sumar cada uno de sus elementos a las columnas de la matriz.

**Ejemplos de broadcasting**

#### ¿Cómo funciona la adición de una matriz y un escalar?

Una aplicación común de broadcasting es sumar un escalar a todas las posiciones de una matriz. Por ejemplo, sumar 42 a una matriz de 3x2 implica que Numpy internamente replica el escalar, efectuando la suma como si fuese un array de la misma dimensión que la matriz. El resultado es sencillo: cada elemento de la matriz original incrementa en valor.

Aquí te dejamos un ejemplo práctico:

```python
import numpy as np

# Definimos una matriz y un escalar
matriz = np.array([[1, 2], [3, 4], [5, 6]])
escalar = 42

# Suma de la matriz con el escalar aplicando broadcasting
resultado = matriz + escalar
print(resultado)
```

El código demostrará cómo cada elemento de la matriz se incrementa por el valor del escalar.

#### ¿Qué pasa cuando sumamos un vector a una matriz traspuesta?

Consideremos una matriz de 2x3 y un vector de longitud 3. Siguiendo las reglas del broadcasting, Numpy extenderá el vector para que coincida con las dimensiones de la matriz al sumar los tres elementos del vector a cada fila de la matriz.

**Ejemplo en acción:**

```python
# Definimos una matriz traspuesta y un vector
matriz_t = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

# Suma de la matriz traspuesta con el vector usando broadcasting
resultado = matriz_t + vector
print(resultado)
```

Aquí vemos cómo, de manera efectiva, cada elemento del vector se adiciona a sus respectivas filas.

#### ¿Cómo practicar el concepto de broadcasting?

Un buen ejercicio para consolidar lo aprendido es tomar un vector de cinco elementos y sumarlo a una matriz de 5x5. Esto no solo fortalecerá tu comprensión del broadcasting, sino que también mejorará tu destreza en el uso de Numpy. ¡No dudes en experimentar y verás que las posibilidades son inmensas!

La exploración constante te permitirá profundizar tu conocimiento y dominar este potente recurso. Sigue practicando y adentrándote en el análisis de datos con confianza y curiosidad. ¡Estoy seguro de que lo lograrás con éxito!

## Producto Interno: Definición y Ejemplos Prácticos

### 📘 ¿Qué es el Producto Interno?

El **producto interno** (también llamado **producto punto** o **dot product**) es una operación algebraica entre dos **vectores del mismo tamaño** que produce un **escalar**.

### 🧮 Fórmula

Si tienes dos vectores:

$$
\vec{a} = [a_1, a_2, ..., a_n] \quad \text{y} \quad \vec{b} = [b_1, b_2, ..., b_n]
$$

El **producto interno** se calcula como:

$$
\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n
$$

### ✅ Ejemplo Manual

$$
\vec{a} = [2, 3] \quad \vec{b} = [4, 1]
$$

$$
\vec{a} \cdot \vec{b} = 2 \cdot 4 + 3 \cdot 1 = 8 + 3 = 11
$$

### 🐍 Ejemplo en Python

```python
import numpy as np

a = np.array([2, 3])
b = np.array([4, 1])

producto_interno = np.dot(a, b)
print(producto_interno)  # Output: 11
```

También puedes usar:

```python
producto_interno = a @ b
```

### 📐 Interpretación Geométrica

El producto interno también se puede interpretar como:

$$
\vec{a} \cdot \vec{b} = \|\vec{a}\| \|\vec{b}\| \cos(\theta)
$$

Donde:

* $\|\vec{a}\|$ y $\|\vec{b}\|$ son las normas (magnitudes) de los vectores.
* $\theta$ es el ángulo entre ellos.

🔸 Si el producto interno = 0 → los vectores son **ortogonales** (perpendiculares).

### 📊 Usos Prácticos

* **Álgebra Lineal:** proyecciones, ortogonalidad.
* **Machine Learning:** cálculo de similitud de vectores (por ejemplo, en **cosine similarity**).
* **Gráficos 3D:** detección de ángulos entre vectores de fuerza, dirección, etc.

### Resumen

#### ¿Qué es el producto interno y cómo se diferencia de la multiplicación de matrices?

El producto interno es fundamental en álgebra lineal y se distingue de la simple multiplicación de una matriz por un vector, ya que implica no solo una operación matemática sino también una comprensión más profunda de cómo interactúan los vectores en el espacio. Mientras que la multiplicación estándar de matrices y vectores proporciona una nueva matriz, el producto interno resulta en un vector de una dimensión específica. ¿Quieres conocer más a fondo estas operaciones y entender cómo utilizarlas en tus proyectos? ¡Acompáñame mientras desglosamos el proceso!

#### Multiplicación de matriz por vector

Cuando multiplicamos una matriz por un vector, el resultado es otra matriz si se cumplen las condiciones de tamaño. Por ejemplo, considere una matriz de tamaño 3x2 y un vector de dimensión 2. Al multiplicarlos, obtienes una nueva matriz que mantiene ciertas propiedades originales mientras multiplica cada componente del vector con las columnas correspondientes de la matriz.

```python
# Ejemplo código para multiplicación de matriz por vector
import numpy as np

# Definiendo la matriz y el vector
matriz = np.array([[1, 2], [3, 4], [5, 6]])
vector = np.array([2, 3])

# Multiplicación de matriz por vector
resultado_matriz = np.dot(matriz, vector)
print(resultado_matriz)
```

En este ejemplo, el proceso de "broadcasting" toma lugar al multiplicar la primera columna por 2 y la segunda columna por 3, obteniendo resultados individuales que se suman.

#### ¿Cómo se realiza el producto interno de una matriz y un vector?

El producto interno, también conocido como "dot product" o "producto escalar", implica multiplicar las correspondientes entradas de matriz y vector y luego sumar estos productos. A diferencia de la multiplicación de matriz, el resultado es un vector donde cada elemento es una suma de productos diferentes.

```python
# Ejemplo código para producto interno
resultado_producto_interno = np.inner(matriz, vector)
print(resultado_producto_interno)
```

Aquí, cada elemento del resultado es calculado multiplicando los elementos correspondientes del vector y la matriz, luego sumándolos. Esto da un resultado específico basado en las dimensiones del vector y del número de filas en la matriz.

#### Consideraciones y ejercicios para profundizar

- **Diferentes métodos**: Hay dos formas principales de realizar el producto interno: calculándolo directamente al multiplicar el vector y la matriz, y utilizando funciones de bibliotecas como `numpy` que ofrecen funciones definidas como `np.dot()`.
- **Prueba interactividad**: Multiplica una matriz por un vector y luego invierte el orden: multiplicar el vector por la matriz. Observando los resultados, notarás cómo varían o permanecen iguales bajo ciertas condiciones.
- **Participación en discusiones**: Aprovecha los foros de discusión para resolver dudas y compartir hallazgos con otros estudiantes. La interacción y el intercambio de ideas enriquece el aprendizaje.

Este conocimiento no solo es esencial para quienes quieren adentrarse en campos como el machine learning o data science, sino que también es una habilidad clave en la programación y la resolución de problemas matemáticos complejos. ¡Sigue explorando y no dudes en preguntar lo que necesites! Nos vemos en la próxima clase.

## Producto Interno entre Dos Matrices: Definición y Cálculo

### 📘 ¿Qué es el Producto Interno entre Dos Matrices?

En álgebra lineal, el **producto interno entre matrices** generalmente **no se aplica directamente como en vectores**. Sin embargo, existen dos conceptos similares:

### 1. 🔁 **Producto Matricial (Multiplicación de Matrices)**

Si tienes dos matrices $A \in \mathbb{R}^{m \times n}$ y $B \in \mathbb{R}^{n \times p}$, el **producto matricial** $C = A \cdot B$ es una **matriz $m \times p$**.

Este no es un producto interno clásico, pero es el producto más común entre matrices.

📌 Ejemplo:

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = np.dot(A, B)
print(C)
```

🧮 Resultado:

```
[[19 22]
 [43 50]]
```

### 2. 🔸 **Producto Interno como Escalar (Frobenius Inner Product)**

Cuando **ambas matrices son del mismo tamaño**, el producto interno puede definirse como:

$$
\langle A, B \rangle = \sum_{i,j} A_{ij} \cdot B_{ij}
$$

Este es el **producto interno de Frobenius**, y da como resultado un **escalar**.

📌 Ejemplo:

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

producto_interno = np.sum(A * B)
print(producto_interno)
```

🧮 Resultado:

```
70  # = 1*5 + 2*6 + 3*7 + 4*8
```

### 📊 ¿Cuándo usar cada uno?

| Contexto                                 | Método                                          |
| ---------------------------------------- | ----------------------------------------------- |
| Transformación lineal o redes neuronales | `np.dot(A, B)` o `A @ B`                        |
| Similitud o comparación de matrices      | `np.sum(A * B)` (producto interno de Frobenius) |

### Resumen

#### ¿Cómo se define el producto interno entre matrices?

El producto interno entre matrices es uno de los conceptos más útiles y fascinantes de la álgebra lineal. Nos permite combinar matrices de manera que se puedan aprovechar sus propiedades para cálculos más complejos. En particular, el producto interno entre dos matrices está definido cuando el número de columnas de la primera matriz es idéntico al número de filas de la segunda. Esto se traduce en que las dimensiones de las matrices involucradas deben estar alineadas adecuadamente.

Esta operación es esencial en campos como la computación, donde se utilizan matrices para representar datos y para realizar cálculos avanzados de manera eficiente. Veamos cómo se aplica este concepto a través de un ejemplo práctico.

#### ¿Cómo aplicamos el producto interno a dos matrices?

Para ilustrar el producto interno, consideremos las matrices `A` y `B`. La matriz `A` es de dimensiones 4x3 y la matriz `B` es de dimensiones 3x2. En este caso, el producto interno (`A \cdot B`) es posible, pero el producto (`B \cdot A`) no lo es.

#### Paso a paso del producto (A \cdot B)

La operación (A \cdot B) es posible porque el número de columnas de `A` coincide con el número de filas de `B`, es decir, 3. Esta coincidencia nos permite realizar el producto interno. Echemos un vistazo al proceso:

1. **Multi-step Calculation**: Se multiplica cada elemento de la fila de `A` por el elemento correspondiente de la columna de `B`, y luego se suman estos productos.

3. **Iteración por filas y columnas**: Se repite el proceso para cada fila de `A` y cada columna de `B` hasta llenar una nueva matriz resultante. La matriz resultante tendrá las dimensiones ((4x2)), derivadas del número de filas de A y el número de columnas de B.

El resultado nos da una nueva matriz cuya dimensión es 4x2, que resulta de esta multiplicación de matrices individuales.

```python
# Ejemplo de matrices A y B
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
B = np.array([[2, 3], [5, 7], [11, 13]])

# Realizando el producto interno
C = np.dot(A, B)

print(C)
```

Este código en Python ilustra cómo se calcula el producto interno usando la biblioteca `NumPy`, que simplifica el manejo de operaciones matriciales. Este ejemplo resulta en una matriz resultante de 4x2, conforme a nuestras expectativas.

#### ¿Qué sucede cuando el producto interno no está definido?

Intentar calcular el producto interno de (B \cdot A) nos lleva a un error porque las dimensiones correspondientes no coinciden. En lugar de corregir el cálculo, detengamos un momento para entender la razón detrás del error.

El problema radica en que el número de columnas de `B` (que es 2) no coincide con el número de filas de `A` (que es 4). Esto implica que el producto no está definido bajo las reglas del álgebra lineal, lo cual suele conducir a errores en los sistemas de programación, indicándonos que las dimensiones no son compatibles.

#### Consejos prácticos para evitar errores

- **Verificar dimensiones**: Siempre verifica que las dimensiones de las matrices estén alineadas antes de intentar calcular el producto interno.
- **Uso de herramientas**: Emplear herramientas como NumPy, ya que proporcionan funciones para realizar estas comprobaciones automáticamente y manejar matrices grandes de manera eficiente.
- **Comprender los resultados**: Asegúrate de que el tamaño de la matriz resultante tenga sentido dentro del contexto de tu problema.

Te animo a que pongas en práctica este conocimiento calculando el resultado del producto interno de las matrices proporcionadas y completando los valores faltantes en el ejercicio propuesto. ¡Continúa explorando y desarrollando tus habilidades de álgebra lineal!

## Propiedades del Producto Interno en Álgebra Lineal

Claro, aquí tienes un resumen claro y útil de las **propiedades del producto interno** en álgebra lineal:

### 🔷 Propiedades del Producto Interno

Sea $\vec{u}, \vec{v}, \vec{w} \in \mathbb{R}^n$ y $\alpha \in \mathbb{R}$, el **producto interno** (también llamado **producto punto**) se denota:

$$
\langle \vec{u}, \vec{v} \rangle = \sum_{i=1}^n u_i v_i
$$

### ✅ 1. **Conmutatividad**

$$
\langle \vec{u}, \vec{v} \rangle = \langle \vec{v}, \vec{u} \rangle
$$

### ✅ 2. **Bilinealidad** (Linealidad en cada componente)

$$
\langle \alpha \vec{u}, \vec{v} \rangle = \alpha \langle \vec{u}, \vec{v} \rangle
$$

$$
\langle \vec{u} + \vec{w}, \vec{v} \rangle = \langle \vec{u}, \vec{v} \rangle + \langle \vec{w}, \vec{v} \rangle
$$

### ✅ 3. **Positividad**

$$
\langle \vec{v}, \vec{v} \rangle \geq 0
$$

### ✅ 4. **Definida Positiva**

$$
\langle \vec{v}, \vec{v} \rangle = 0 \iff \vec{v} = \vec{0}
$$

### ✅ 5. **Relación con la Norma**

La **norma (longitud)** de un vector es:

$$
\| \vec{v} \| = \sqrt{ \langle \vec{v}, \vec{v} \rangle }
$$

### ✅ 6. **Ortogonalidad**

Dos vectores son **ortogonales** (perpendiculares) si:

$$
\langle \vec{u}, \vec{v} \rangle = 0
$$

### ✅ 7. **Desigualdad de Cauchy-Schwarz**

$$
|\langle \vec{u}, \vec{v} \rangle| \leq \| \vec{u} \| \cdot \| \vec{v} \|
$$

### ✅ 8. **Proyección Ortogonal**

La proyección de $\vec{u}$ sobre $\vec{v}$ es:

$$
\text{proj}_{\vec{v}} \vec{u} = \frac{ \langle \vec{u}, \vec{v} \rangle }{ \langle \vec{v}, \vec{v} \rangle } \vec{v}
$$

### Resumen

#### ¿Cómo entender las propiedades del producto interno?

El producto interno es una operación fundamental en el álgebra lineal que no solo nos permite calcular magnitudes, sino también entender relaciones geométricas entre vectores. Resulta enriquecedor descubrir sus propiedades, ya que nos facilitan el manejo y manipulación de matrices y vectores. A través de ejemplos prácticos, podemos visualizar cómo estas propiedades se manifiestan, lo que fortalece nuestra comprensión teórica y práctica. Analicemos a fondo estas propiedades esenciales del producto interno.

#### ¿Qué es la propiedad asociativa?

La propiedad asociativa es una característica crucial en operaciones matemáticas que nos indica que el orden en el cual agrupamos las operaciones no afecta el resultado. En el contexto del producto interno, si tenemos matrices A, B y C, esta propiedad nos asegura que:

`[ (A \cdot B) \cdot C = A \cdot (B \cdot C) ]`

Ahora, aunque en el álgebra lineal la propiedad asociativa tal cual se aplica a operaciones de escalar y no directamente a matrices, sí aplica en la combinación de productos internos, como una forma de redistribuir la multiplicación en operaciones más complejas.

En código, comprobamos la propiedad asociativa así:

```python
import numpy as np

# Definimos matrices A, B, C
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 10], [11, 12]])

# Verificando la propiedad asociativa
resultado_1 = np.dot(np.dot(A, B), C)
resultado_2 = np.dot(A, np.dot(B, C))

print(resultado_1)
print(resultado_2)
```

Las salidas deberían coincidir, lo cual nos muestra la propiedad asociativa en acción.

#### ¿Cómo comprobar la propiedad distributiva?

La propiedad distributiva nos indica que distribuir un producto sobre una suma es equivalente a realizar cada operación por separado y luego sumar los resultados. Matemáticamente, para el producto interno, se expresa como:

`[ A \cdot (B + C) = A \cdot B + A \cdot C ]`

Este concepto se aplica al distribuir la multiplicación de una matriz por la suma de dos matrices, otorgando fluidez y flexibilidad en los cálculos.

Veamos cómo probar esta propiedad usando código de Python:

```python
D = B + C

# Verificar propiedad distributiva
distributiva_1 = np.dot(A, D)
distributiva_2 = np.dot(A, B) + np.dot(A, C)

print(distributiva_1)
print(distributiva_2)
```

Si ambas matrices resultantes son idénticas, hemos confirmado la propiedad distributiva.

#### ¿El producto interno es conmutativo?

Podríamos estar tentados a asumir que la conmutatividad, una propiedad esencial en la multiplicación escalar, se aplica igualmente al producto interno. Sin embargo, en álgebra de matrices, esto no siempre es el caso. Para matrices, en general no es verdad que:

`[ A \cdot B = B \cdot A ]`

No obstante, en el caso específico de productos de vectores, podemos observar esta propiedad. Consideremos un ejemplo con vectores:

```python
# Definición de vectores
v1 = np.array([2, 7])
v2 = np.array([3, 5])

# Cálculo del producto interno
producto_1 = np.dot(v1, v2)
producto_2 = np.dot(v2, v1)

print(producto_1) # Debe imprimir 41
print(producto_2) # Debe imprimir 41
```

Aquí, vemos que el producto interno de vectores sí es conmutativo, lo cual no es lo mismo en matrices generales.

#### ¿Qué ejercicios prácticos puedo hacer?

Practicar con ejemplos es vital para internalizar estos conceptos. Te animo a pensar en dos matrices de dimensiones distintas para las cuales puedas calcular el Producto Interno, pero no puedas invertir sus roles. Como una pista, podemos recordar cómo las dimensiones deben ser compatibles para que la multiplicación sea válida. Comparte tus hallazgos en los foros de discusión y sigue explorando estas fascinantes propiedades matemáticas en tus estudios de álgebra lineal.

## Transposición y Producto Interno de Matrices

¡Claro! Vamos a ver **cómo se realiza la transposición y el producto interno de matrices**, tanto desde el punto de vista teórico como práctico en Python.

### 🔁 Transposición de Matrices

La **transposición** de una matriz $A$ se denota como $A^T$ y consiste en convertir las filas en columnas y viceversa.

### Ejemplo:

Si:

$$
A = \begin{bmatrix}
1 & 2 \\
3 & 4 \\
\end{bmatrix}
\quad \Rightarrow \quad
A^T = \begin{bmatrix}
1 & 3 \\
2 & 4 \\
\end{bmatrix}
$$

### En Python:

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

A_transpuesta = A.T
print("Transpuesta de A:\n", A_transpuesta)
```

### 🔷 Producto Interno de Matrices (Producto Matricial)

El **producto interno (o producto punto)** entre matrices también puede interpretarse como **producto matricial** cuando se cumple la condición de dimensiones.

Si $A$ es de tamaño $m \times n$ y $B$ de tamaño $n \times p$, entonces:

$$
C = A \cdot B \quad \text{tendrá tamaño} \quad m \times p
$$

### Ejemplo:

$$
A = \begin{bmatrix}1 & 2\end{bmatrix},\quad B = \begin{bmatrix}3 \\ 4\end{bmatrix}
\quad \Rightarrow \quad A \cdot B = 1\cdot3 + 2\cdot4 = 11
$$

### En Python:

```python
A = np.array([[1, 2]])
B = np.array([[3],
              [4]])

producto_interno = np.dot(A, B)
print("Producto interno de A y B:\n", producto_interno)
```

### 🧠 Nota:

* Si estás trabajando con **vectores**, `np.dot()` o `np.inner()` puede ser usado.
* Para **matrices**, también puedes usar `@` o `np.matmul()` para claridad y compatibilidad.

### Resumen

#### ¿Qué es la transposición de un producto de matrices?

La transposición de matrices es un concepto central en álgebra lineal. Al operar con matrices, tanto la forma en que se multiplican como su transposición juegan roles cruciales. Hoy exploramos cómo se interrelacionan estas transposiciones dentro del producto interno de matrices. La propiedad que definiremos es: si tenemos una matriz A multiplicada internamente con B, su transposición es igual al producto de B transpuesta con A transpuesta. Esto es expresado matemáticamente como:

`(AB)^T = B^T A^T`

Esta propiedad es vital al trabajar en álgebra lineal, ya que permite una manipulación más intuitiva y versátil de las matrices, tratándolas casi como si fueran números.

#### ¿Qué implicaciones tiene esta propiedad en el trabajo con matrices?

La versatilidad que otorga esta propiedad en operaciones matriciales es significativa:

- **Flexibilidad Operativ**a: Podemos organizar las operaciones de una manera que facilite los cálculos, porque la transposición nos permite cambiar el orden sin alterar el resultado final.

- **Optimización de Cálculos**: En lugar de recalcular completamente los sistemas de ecuaciones, el uso correcto de transposiciones ahorra tiempo y esfuerzo, simplificando el proceso.

- **Jugabilidad Matemática**: Al aplicar dos veces la transposición a una matriz, la devolvemos a su forma original. Esto permite experimentar con las transformaciones y, de ser necesario, revertirlas con facilidad.

```python
# Ejemplo básico de transposición en Python usando NumPy
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])  # Matriz 3x2
B = np.array([[7, 8], [9, 10]])         # Matriz 2x2

# Producto interno y su transposición
product_transposed = np.matmul(A, B).T
B_transpose_A_transpose = np.matmul(B.T, A.T)

print("Transposición del producto interno:", product_transposed)
print("Producto de transposiciones:", B_transpose_A_transpose)
```

#### ¿Cómo verificar la igualdad de estas operaciones?

Verificar la equivalencia entre ( (AB)^T ) y ( B^T A^T ) te acerca un paso más a tratar matrices como si fueran números. Con la ayuda de herramientas como NumPy en Python, el proceso se facilita:

- **Comprobación de Igualdad**: Mediante funciones que comparan matrices, como numpy.allclose(), puedes certificar que el resultado final de ambas operaciones es el mismo.

- **Exploración Práctica**: Al comenzar con matrices A y B de dimensiones adecuadas (como A de 3x2 y B de 2x2 en el ejemplo), puedes aplicar las operaciones y verificar la igualdad de resultados, reforzando la comprensión teórica con la práctica.

- **Aplicación en Sistemas Lineales**: Esta propiedad resulta útil al resolver sistemas de ecuaciones lineales, reduciendo la complejidad al manejar las matrices involucradas.

```python
# Verificación de igualdad entre las operaciones
equal_check = np.allclose(product_transposed, B_transpose_A_transpose)

print("Las operaciones son iguales:", equal_check)
```

Continuar explorando y practicando estos conceptos profundizará tu comprensión y habilidad en álgebra lineal, preparándote para abordar sistemas de ecuaciones complejos con confianza y eficiencia. ¡Te animo a sumergirte más en este apasionante mundo matemático!

## Comprobación gráfica de sistemas de ecuaciones lineales

¡Perfecto! Visualizar **sistemas de ecuaciones lineales** gráficamente es una excelente forma de entender su solución. Te mostraré cómo hacerlo paso a paso con un ejemplo usando **Python** y **Matplotlib**.

### 🎯 Objetivo:

Graficar dos ecuaciones lineales y mostrar su **punto de intersección**, que representa la **solución del sistema**.

### 📌 Ejemplo de Sistema de Ecuaciones:

$$
\text{Ecuación 1: } y = 2x + 1  
$$

$$
\text{Ecuación 2: } y = -x + 4
$$

### 🧮 Paso a Paso en Python:

```python
import numpy as np
import matplotlib.pyplot as plt

# Valores de x
x = np.linspace(-10, 10, 200)

# Ecuaciones
y1 = 2 * x + 1      # y = 2x + 1
y2 = -1 * x + 4     # y = -x + 4

# Intersección (resolviendo el sistema de ecuaciones)
# Igualamos: 2x + 1 = -x + 4 => 3x = 3 => x = 1, y = 2(1) + 1 = 3
x_intersect = 1
y_intersect = 3

# Gráfico
plt.figure(figsize=(8,6))
plt.plot(x, y1, label='y = 2x + 1')
plt.plot(x, y2, label='y = -x + 4')
plt.plot(x_intersect, y_intersect, 'ro', label='Intersección (1,3)')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title('Solución Gráfica de un Sistema de Ecuaciones Lineales')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 🧠 Interpretación:

* Cada línea representa una ecuación lineal.
* El **punto rojo** muestra la solución del sistema: el punto donde ambas rectas se cruzan.
* Si no se cruzan → el sistema **no tiene solución (inconsistente)**.
* Si son la misma recta → **infinitas soluciones**.

## Matrices Identidad, Inversa y Singular: Propiedades y Cálculo

¡Perfecto! Este es un tema central en **álgebra lineal**, con aplicaciones directas en sistemas de ecuaciones, optimización, computación gráfica y machine learning. Vamos a revisar las **propiedades clave** y cómo se calculan:

### 🔷 1. **Matriz Identidad (𝐼)**

### 🔹 Definición:

Una **matriz cuadrada** donde todos los elementos de la **diagonal principal son 1** y los demás son 0.

### 🔹 Propiedades:

* $A \cdot I = A$
* $I \cdot A = A$
* $I^{-1} = I$
* Es **la unidad** de la multiplicación matricial.

### 🔹 Ejemplo en Python:

```python
import numpy as np

I = np.eye(3)
print(I)
```

### 🔷 2. **Matriz Inversa (𝐴⁻¹)**

### 🔹 Definición:

La matriz inversa de $A$ es otra matriz $A^{-1}$ tal que:

$$
A \cdot A^{-1} = I
$$

### 🔹 Solo existe si:

* $A$ es **cuadrada**.
* $\det(A) \ne 0$ → No es singular.

### 🔹 Propiedades:

* $(A^{-1})^{-1} = A$
* $(AB)^{-1} = B^{-1}A^{-1}$
* $(A^T)^{-1} = (A^{-1})^T$

### 🔹 En Python:

```python
A = np.array([[2, 1], [5, 3]])
A_inv = np.linalg.inv(A)
print(A_inv)
```

### 🔷 3. **Matriz Singular**

### 🔹 Definición:

Una matriz cuadrada que **no tiene inversa**.

### 🔹 Ocurre cuando:

* $\det(A) = 0$
* Sus filas o columnas son **linealmente dependientes**.

### 🔹 Ejemplo:

```python
A = np.array([[2, 4], [1, 2]])  # Segunda fila es múltiplo de la primera
print(np.linalg.det(A))  # Resultado: 0 → matriz singular
```

### 🔷 4. **Cálculo Rápido del Determinante**

Usado para saber si la matriz tiene inversa.

```python
np.linalg.det(A)  # Si es 0, es singular
```

### 📘 Extra: Resolver un Sistema con la Inversa

$$
AX = B \Rightarrow X = A^{-1}B
$$

```python
A = np.array([[2, 1], [5, 3]])
B = np.array([[5], [13]])
X = np.linalg.inv(A).dot(B)
print(X)
```

### Resumen

#### ¿Qué son las matrices especiales y sus características?

Las matrices especiales juegan un papel crucial en el álgebra lineal y poseen propiedades únicas que las hacen destacarse. Entre ellas, encontramos la matriz identidad, la matriz inversa y la matriz singular. Entender las peculiaridades de cada una es esencial para diversos cálculos y aplicaciones en matemáticas avanzadas.

#### ¿Qué es la matriz identidad?

La matriz identidad es una transformación neutra dentro del contexto de las matrices. En esencia, es una matriz cuadrada en la que todos los elementos de la diagonal principal son unos, y todos los otros elementos son ceros. La función eye de bibliotecas como NumPy nos permite generarla fácilmente. Su peculiaridad es que, al multiplicarla por cualquier vector, este permanece inalterado, similar a como el número uno es el elemento neutro en la multiplicación de números.

```python
import numpy as np

# Generamos una matriz identidad de dimensión 3x3
identidad = np.eye(3)
print(identidad)
```

#### ¿Qué representa la matriz inversa?

La matriz inversa cumple una función similar al concepto de inverso en la multiplicación usual: cuando una matriz ( A ) se multiplica por su inversa ( A^{-1} ), obtenemos la matriz identidad. Para calcularla, utilizamos funciones específicas, como `np.linalg.inv` de NumPy.

```python
# Definimos una matriz 3x3
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Calculamos la inversa de la matriz A
A_inversa = np.linalg.inv(A)
print(A_inversa)
```

Sin embargo, es importante destacar que no todas las matrices tienen una inversa. Generalmente, solo las matrices cuadradas que cumplen ciertas condiciones pueden tener esta propiedad.

#### ¿Por qué algunas matrices son singulares?

La matriz singular es especial porque carece de inversa. Esto sucede cuando determinantes de la matriz son cero, indicando que la matriz es "degenerada". Intentar calcular la inversa de una matriz singular generará un error, como en el siguiente ejemplo:

```python
# Intentamos calcular la inversa de una matriz singular
try:
    singular = np.array([[1, 2], [2, 4]])
    singular_inversa = np.linalg.inv(singular)
except np.linalg.LinAlgError:
    print("Error: Matrix is singular")
```

#### ¿Cómo resolver sistemas de ecuaciones lineales usando matrices?

Las matrices son fundamentales en la resolución de sistemas de ecuaciones lineales. Al tener la matriz inversa de un sistema ( A \times X = B ), podemos multiplicar ambos lados por ( A^{-1} ) para resolver ( X ).

```python
# Supongamos que tenemos un sistema de ecuaciones
A = np.array([[3, 1], [1, 2]])
B = np.array([9, 8])

# Calculamos la inversa de A
A_inversa = np.linalg.inv(A)

# Resolvemos para X
X = np.dot(A_inversa, B)
print(X)
```

Contar con la matriz inversa permite implementar algoritmos eficientes para soluciones exactas de sistemas lineales, lo que es de gran utilidad en campos como la ingeniería y las ciencias computacionales. Con estas herramientas, tu entendimiento y manejo del álgebra lineal se vuelve potente y versátil. ¡Continúa explorando y profundizando en el fascinante mundo de las matemáticas!

## Solución de Sistemas Lineales usando la Inversa de una Matriz

¡Excelente! Resolver sistemas de ecuaciones lineales usando la **inversa de una matriz** es una técnica clásica del álgebra lineal cuando el sistema se puede representar como:

$$
AX = B
$$

Donde:

* $A$ es la **matriz de coeficientes** (cuadrada).
* $X$ es el **vector columna** de incógnitas.
* $B$ es el **vector columna** de resultados.

### ✅ Paso a Paso para Resolver $AX = B$ con la Inversa

### 📌 1. Verifica que $A$ sea **cuadrada** y **no singular** (determinante ≠ 0)

### 📌 2. Calcula la **inversa de A**: $A^{-1}$

### 📌 3. Multiplica ambos lados por $A^{-1}$:

$$
A^{-1}AX = A^{-1}B \Rightarrow IX = A^{-1}B \Rightarrow X = A^{-1}B
$$

### 🔢 Ejemplo Práctico en Python

Supongamos el siguiente sistema:

$$
\begin{cases}
2x + y = 5 \\
5x + 3y = 13
\end{cases}
$$

Representamos el sistema:

```python
import numpy as np

# Matriz de coeficientes A
A = np.array([[2, 1],
              [5, 3]])

# Vector de resultados B
B = np.array([[5],
              [13]])

# Verificamos si A tiene inversa
if np.linalg.det(A) != 0:
    A_inv = np.linalg.inv(A)
    X = A_inv @ B
    print("Solución del sistema (x, y):")
    print(X)
else:
    print("La matriz A es singular. No tiene inversa.")
```

### 📌 Resultado

El código te devuelve el valor de $x$ y $y$ como solución del sistema.

### ⚠️ Consideraciones

* Este método **no es el más eficiente** computacionalmente para grandes sistemas.
* Es ideal para **análisis teórico o sistemas pequeños**.
* Para sistemas grandes o mal condicionados, se prefiere `np.linalg.solve(A, B)`.

### 🔄 Alternativa más eficiente:

```python
X = np.linalg.solve(A, B)  # Resuelve directamente sin calcular la inversa
```

### Resumen

#### ¿Cómo utilizar la matriz inversa para resolver un sistema de ecuaciones lineales?

Las matrices inversas son herramientas poderosas en álgebra lineal que nos permiten encontrar soluciones a sistemas de ecuaciones lineales. Imagina que tienes un sistema matemático que resolver y deseas utilizar una matriz inversa; hacerlo podría simplificar mucho el proceso. Vamos a profundizar en cómo se hace esto paso a paso usando Python.

#### ¿Cómo definir matrices y vectores en Python?

Primero, definimos nuestras matrices y vectores utilizando la biblioteca NumPy, que es muy útil para el manejo de datos numéricos en Python. Para un sistema de ecuaciones sencillo, donde tenemos las ecuaciones (3x + y = 1) y (2x + y = 1), organizamos la matriz de coeficientes y el vector de resultados así:

```python
import numpy as np

# Definición de la matriz A
A = np.array([[3, 1], [2, 1]])

# Definición del vector B
B = np.array([1, 1])
```

#### ¿Cómo calcular la matriz inversa?

El siguiente paso es calcular la matriz inversa de (A). En álgebra lineal, si una matriz ( A ) tiene una inversa, significa que podemos multiplicarla por su inversa para obtener la matriz identidad. En Python:

```python
# Calcular la matriz inversa de A
inversa_A = np.linalg.inv(A)
```

#### ¿Cómo resolver el sistema de ecuaciones?

Una vez obtenida la matriz inversa, podemos encontrar la solución ( X ) multiplicando esta inversa por el vector ( B ):

```python
# Calcular el vector solución X
X = np.dot(inversa_A, B)
```

El resultado te dará los valores de ( x ) y ( y ) que solucionan el sistema. En nuestro caso, el vector ( X ) debería ser muy similar a ([0, 1]), lo que corresponde a ( x = 0 ) y ( y = 1 ).

#### ¿Qué pasa si cambiamos el vector de resultados?

Si cambias ( B ) para ver si la misma matriz inversa puede ayudarnos a resolver otra configuración de resultados, tendrías algo así:

```python
# Nuevo vector B
B_nuevo = np.array([3, 7])

# Calcular el nuevo vector solución X usando la misma inversa
X_nuevo = np.dot(inversa_A, B_nuevo)
```

Este enfoque te proporciona la solución para cualquier vector ( B ) dado, siempre que los coeficientes de las variables en las ecuaciones permanezcan iguales.

#### ¿Cuáles son las limitaciones del uso de matrices inversas?

Aunque resolver sistemas de ecuaciones lineales usando matrices inversas es conveniente, no es siempre eficiente debido a problemas numéricos que pueden surgir, especialmente cuando lidias con matrices grandes o mal condicionadas. A menudo, otras técnicas como la eliminación Gaussiana o métodos numéricos de aproximación pueden ser más adecuados.

#### ¿Por qué es importante la práctica de métodos numéricos?

Los métodos numéricos se utilizan para encontrar soluciones aproximadas a ecuaciones y no dependen de las ineficiencias inherentes a las matrices inversas en representaciones computacionales. Saber cuándo y cómo utilizar diferentes métodos es esencial para quienes trabajan con álgebra lineal y problemas matemáticos complejos en la práctica.

¿Te interesa seguir explorando esta rica área de las matemáticas computacionales? ¡Sigue practicando, afina tus habilidades y aprovecha al máximo estas herramientas fascinantes!

## Sistemas de Ecuaciones: Soluciones Únicas, Múltiples o Ninguna

En **Álgebra**, un **sistema de ecuaciones** es un conjunto de dos o más ecuaciones con dos o más incógnitas. Las soluciones del sistema representan los valores de las incógnitas que **satisfacen todas las ecuaciones simultáneamente**. Según las características de las rectas (en el caso de sistemas lineales de dos variables), los sistemas pueden tener:

### 🔹 1. **Una Solución Única** (Sistema Compatible Determinado)

* Las rectas **se cortan en un solo punto**.
* Las ecuaciones representan **rectas con distintas pendientes**.
* **Geometría**: Las rectas se cruzan.
* **Solución**: Un único par ordenado $(x, y)$.

**Ejemplo:**

$$
\begin{cases}
x + y = 5 \\
x - y = 1
\end{cases}
$$

✅ Tiene una única solución: $x = 3$, $y = 2$

### 🔹 2. **Infinitas Soluciones** (Sistema Compatible Indeterminado)

* Las rectas **son coincidentes** (la misma recta).
* Una ecuación es **múltiplo de la otra**.
* **Geometría**: Todas sus soluciones son compartidas.
* **Solución**: Infinitos pares ordenados que cumplen ambas ecuaciones.

**Ejemplo:**

$$
\begin{cases}
2x + 4y = 6 \\
x + 2y = 3
\end{cases}
$$

✅ Las dos ecuaciones representan la **misma recta**.
Infinitas soluciones: cualquier par $(x, y)$ que cumpla la ecuación.

### 🔹 3. **Ninguna Solución** (Sistema Incompatible)

* Las rectas **son paralelas** y **nunca se cruzan**.
* Tienen **la misma pendiente** pero **diferentes ordenadas al origen**.
* **Geometría**: Nunca se tocan.
* **Solución**: ❌ No existe ningún par ordenado que satisfaga ambas ecuaciones.

**Ejemplo:**

$$
\begin{cases}
x + 2y = 4 \\
x + 2y = 7
\end{cases}
$$

❌ No hay solución. Las rectas son paralelas.

### 🔎 ¿Cómo determinar el tipo de solución?

Si el sistema es de la forma:

$$
\begin{cases}
a_1x + b_1y = c_1 \\
a_2x + b_2y = c_2
\end{cases}
$$

Revisa los cocientes:

| Comparación                                             | Tipo de Sistema      |
| ------------------------------------------------------- | -------------------- |
| $\frac{a_1}{a_2} \ne \frac{b_1}{b_2}$                   | Una solución única   |
| $\frac{a_1}{a_2} = \frac{b_1}{b_2} = \frac{c_1}{c_2}$   | Infinitas soluciones |
| $\frac{a_1}{a_2} = \frac{b_1}{b_2} \ne \frac{c_1}{c_2}$ | Ninguna solución     |

### Resumen

#### ¿Cuáles son los tipos de sistemas de ecuaciones lineales?

En el estudio de sistemas de ecuaciones lineales, es fundamental comprender los diferentes tipos de soluciones que pueden existir. Estos sistemas pueden clasificarse en tres categorías: sin solución, con una solución única o con infinitas soluciones. Cada una de estas situaciones se comporta de manera particular y presentan características únicas que es crucial identificar.

#### ¿Qué ocurre cuando un sistema no tiene solución?

Un sistema de ecuaciones no tiene solución cuando es sobre-determinado, es decir, cuando hay más ecuaciones que variables. Esto se traduce gráficamente en que las líneas o planos representando las ecuaciones no se cruzan en un punto común.

Por ejemplo, consideremos el sistema de ecuaciones:

- ( y_1 = 3x + 5 )
- ( y_2 = -x + 3 )
- ( y_3 = 2x + 1 )

En este caso, al graficar estas ecuaciones, notamos que no existe ningún punto donde las tres líneas se intersecten. Esta ausencia de intersección confirma que no hay solución al sistema, lo que ilustra el concepto de un sistema sobre-determinado.

```python
# Ejemplo en Python usando una biblioteca de gráficos
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-6, 6, 400)  
y1 = 3 * x + 5
y2 = -x + 3
y3 = 2 * x + 1

plt.plot(x, y1, label='y1 = 3x + 5')
plt.plot(x, y2, label='y2 = -x + 3')
plt.plot(x, y3, label='y3 = 2x + 1')

plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.axhline(0, color='grey', lw=0.8)
plt.axvline(0, color='grey', lw=0.8)
plt.legend()
plt.show()
```

#### ¿Cómo identificamos una solución única?

Un sistema de ecuaciones tiene una solución única cuando las ecuaciones se intersectan en un punto específico. Gráficamente, esto se refleja en una sola intersección entre las líneas de las ecuaciones, lo que destaca que existe una única combinación de valores que satisface todas las ecuaciones a la vez.

Por ejemplo, consideremos las ecuaciones:

- ( y_2 = -x + 3 )
- ( y_3 = 2x + 1 )

Si graficamos estas ecuaciones, bajando una tiene una pendiente diferente a la otra, resultando en una intersección en un punto exacto. Este punto de intersección representa que el sistema tiene una solución específica y única.

```python
# Gráfica de un sistema con una solución única
plt.plot(x, y2, label='y2 = -x + 3')
plt.plot(x, y3, label='y3 = 2x + 1')

plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.axhline(0, color='grey', lw=0.8)
plt.axvline(0, color='grey', lw=0.8)
plt.legend()
plt.show()
```

#### ¿Cuándo un sistema tiene infinitas soluciones?

Un sistema de ecuaciones tiene infinitas soluciones cuando las ecuaciones son dependientes, es decir, una ecuación es una múltiplo o derivado de la otra. Esto implica que cualquier solución válida para una ecuación es automáticamente válida para la otra. Gráficamente, esto se representa por una sola línea que se superpone a otra, indicando que hay un grado de libertad.

Por ejemplo, en el caso de sólo usar la ecuación ( y_3 = 2x + 1 ), existe un grado de libertad desde el momento en que cualquier valor de ( x ) encuentra un correspondiente valor de ( y ), siguiendo esta ecuación.

```python
# Sistema con infinitas soluciones
plt.plot(x, y3, label='y3 = 2x + 1')

plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.axhline(0, color='grey', lw=0.8)
plt.axvline(0, color='grey', lw=0.8)
plt.legend()
plt.show()
```

En resumen, los sistemas de ecuaciones lineales pueden presentar escenarios de cero, una o infinitas soluciones, reflejando la rica diversidad de comportamientos en las configuraciones algebraicas y sus soluciones. ¡Continúa indagando en estos conceptos para fortalecer tu entendimiento y aplicación!

## Visualización de Vectores y Funciones Reutilizables en Python

Visualizar vectores en Python puede ayudarte a comprender mejor los conceptos de álgebra lineal, especialmente cuando se trata de sumas, productos escalares, transformaciones, etc. Para ello, también es muy útil construir **funciones reutilizables**, que faciliten repetir el trabajo sin duplicar código.

Aquí tienes una guía práctica para **visualizar vectores en 2D y 3D**, junto con funciones reutilizables en Python usando `matplotlib` y `numpy`.

### 🧩 1. Importación de Bibliotecas

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Solo necesario para vectores 3D
```

### 🧮 2. Funciones Reutilizables para Visualizar Vectores

### 🔹 2D: Vectores en el plano

```python
def graficar_vectores_2d(vectores, colores=None):
    plt.figure()
    ax = plt.gca()

    # Asignar colores por defecto si no se pasan
    if colores is None:
        colores = ['r', 'b', 'g', 'y', 'm']
    
    for i, v in enumerate(vectores):
        color = colores[i % len(colores)]
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color)
        plt.text(v[0]*1.1, v[1]*1.1, f'{v}', fontsize=12, color=color)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Visualización de Vectores 2D")
    plt.show()
```

### 🔹 3D: Vectores en el espacio

```python
def graficar_vectores_3d(vectores, colores=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colores is None:
        colores = ['r', 'b', 'g', 'y', 'm']
    
    for i, v in enumerate(vectores):
        color = colores[i % len(colores)]
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color)
        ax.text(v[0]*1.1, v[1]*1.1, v[2]*1.1, f'{v}', color=color)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Visualización de Vectores 3D")
    plt.show()
```

### 🔁 3. Ejemplo de Uso

### ✳️ En 2D:

```python
v1 = np.array([4, 2])
v2 = np.array([-1, 5])
graficar_vectores_2d([v1, v2])
```

### ✳️ En 3D:

```python
v1 = np.array([2, 4, 3])
v2 = np.array([-3, 1, 5])
graficar_vectores_3d([v1, v2])
```

### 🧠 Bonus: Función para Generar Vectores Aleatorios

```python
def generar_vectores(n=2, dim=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return [np.random.randint(-10, 10, dim) for _ in range(n)]
```

### Resumen

#### ¿Cómo visualizar vectores con Python?

Cuando trabajamos con combinaciones lineales, es crucial poder visualizar los vectores de manera efectiva. Esto no solo nos ayuda a comprender mejor el problema, sino que también facilita la interpretación de resultados. Sigamos un proceso paso a paso para crear una función que nos permita graficar vectores utilizando Python.

#### ¿Qué herramientas utilizamos para graficar?

Para abordar esta tarea, emplearemos `NumPy` y `Matplotlib`, dos bibliotecas fundamentales en el ecosistema de Python para manejo de datos y graficación. A continuación, asegurémonos de importar las librerías necesarias:

```python
import numpy as np
import matplotlib.pyplot as plt
```

#### ¿Cómo creamos una función para graficar vectores?

Pensando en la reutilización y claridad del código, lo mejor es encapsular la lógica de graficación de vectores en una función. Así, podemos llamar a esa función cada vez que la necesitemos. Vamos a crear la función `graficar_vectores`:

```python
def graficar_vectores(vectores, colores, alpha=1):
    plt.figure()
    plt.axvline(x=0, color='grey', zorder=0)
    plt.axhline(y=0, color='grey', zorder=0)
    for i, vector in enumerate(vectores):
        plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=colores[i], alpha=alpha)
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.show()
```

#### ¿Cómo preparar los datos para graficar?

Definimos los vectores que deseamos visualizar. Aquí, creamos dos vectores `v1` y `v2`:

```python
v1 = np.array([2, 5])
v2 = np.array([3, -2])
```

#### ¿Cómo llamamos a la función para graficar?

Una vez que hemos definido nuestros vectores y la función graficar_vectores, podemos proceder a graficar:

`graficar_vectores([v1, v2], ['orange', 'blue'])`

Este comando generará una gráfica donde los vectores se distinguen claramente por los colores especificados.

#### ¿Cómo guardar la función para reutilizarla?

Cuando tienes funciones útiles como ésta, es ventajoso guardarlas en un notebook separado para que puedas reutilizarlas en diferentes proyectos sin perder eficiencia. Creamos un archivo para guardar la función:

1. Crear una carpeta llamada `Funciones_auxiliares`.
2. Dentro, un notebook llamado `Graficar_vectores.ipynb`.
3. Copiar y pegar la función `graficar_vectores` en este nuevo notebook.

Luego, para importarla en nuestros notebooks principales:

`%run '../Funciones_auxiliares/Graficar_vectores.ipynb'`

Esto garantiza que cualquier actualización que hagamos a la función se reflejará automáticamente en todos los análisis donde la utilizamos.

#### ¿Cuáles son los beneficios de esta organización del código?

Tener funciones reutilizables y bien organizadas trae numerosos beneficios:

- `Mantenimiento eficiente del código`: Si necesitamos actualizar la función, podemos hacerlo en un solo lugar.
- `Claridad y profesionalismo`: Un código estructurado es más fácil de entender, compartir y escalar.
- `Productividad incrementada`: Ahorra tiempo al evitar reescribir la misma lógica en diferentes partes de un proyecto.

¡Continúa aprendiendo y explorando nuevas formas de optimizar tus análisis! Cada herramienta y técnica que domines te acercará a resultados más precisos y eficientes.

## Combinaciones Lineales de Vectores: Concepto y Aplicaciones Prácticas

Claro, aquí tienes una explicación clara y útil sobre **combinaciones lineales de vectores**, tanto en teoría como con ejemplos prácticos en Python:

### 🧠 ¿Qué es una Combinación Lineal?

Una **combinación lineal** de vectores es una expresión como:

$$
\vec{v} = a_1 \vec{v}_1 + a_2 \vec{v}_2 + \dots + a_n \vec{v}_n
$$

Donde:

* $\vec{v}_1, \vec{v}_2, \dots, \vec{v}_n$ son **vectores base**.
* $a_1, a_2, \dots, a_n$ son **escalares** (números reales).
* El resultado $\vec{v}$ es otro vector.

👉 El conjunto de **todas las combinaciones lineales posibles** de un conjunto de vectores forma un **subespacio vectorial**.

### ✍️ Ejemplo Conceptual

Sean:

$$
\vec{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix},\quad
\vec{v}_2 = \begin{bmatrix} 3 \\ -1 \end{bmatrix}
$$

Una combinación lineal podría ser:

$$
\vec{v} = 2\vec{v}_1 + (-1)\vec{v}_2 = 2 \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 3 \\ -1 \end{bmatrix} = \begin{bmatrix} -1 \\ 5 \end{bmatrix}
$$


### 🔧 Aplicación Práctica en Python

```python
import numpy as np

# Definir vectores
v1 = np.array([1, 2])
v2 = np.array([3, -1])

# Escalares
a = 2
b = -1

# Combinación lineal
v = a * v1 + b * v2
print("Combinación lineal:", v)
```

### 📊 Visualización con Matplotlib

```python
import matplotlib.pyplot as plt

def graficarVectores(vecs, colores=None):
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')

    if colores is None:
        colores = ['r', 'b', 'g', 'orange']
    
    for i in range(len(vecs)):
        plt.quiver(0, 0,
                   vecs[i][0],
                   vecs[i][1],
                   angles='xy', scale_units='xy', scale=1, color=colores[i])
        plt.text(vecs[i][0]*1.1, vecs[i][1]*1.1, str(vecs[i]))

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.title("Combinación Lineal de Vectores")
    plt.show()

# Visualizar
graficarVectores([v1, v2, v], ['blue', 'green', 'red'])
```

### 🧪 Aplicaciones Reales

| Aplicación                  | Descripción                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------- |
| **Gráficos por Computador** | Mezcla de colores y transformaciones se modelan como combinaciones lineales         |
| **Ingeniería**              | Fuerzas que actúan sobre un objeto se suman vectorialmente                          |
| **Machine Learning**        | Los modelos lineales como la regresión usan combinaciones lineales de variables     |
| **Robótica**                | El movimiento de un brazo robótico puede representarse con vectores y combinaciones |
| **Economía**                | Modelos de producción o portafolios financieros involucran combinaciones lineales   |

### 🚩 ¿Cómo Saber si un Vector Está en el Espacio Generado?

### Ejemplo:

¿El vector $\vec{w} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}$ está en el subespacio generado por $\vec{v}_1$ y $\vec{v}_2$?

Planteamos:

$$
a \cdot \vec{v}_1 + b \cdot \vec{v}_2 = \vec{w}
$$

Resolvemos el sistema lineal. En Python:

```python
from numpy.linalg import solve

# Queremos encontrar a, b tal que: a*v1 + b*v2 = w
w = np.array([4, 3])
A = np.column_stack((v1, v2))  # Matriz con v1 y v2 como columnas

# Resolver el sistema A * [a, b] = w
solucion = solve(A, w)
print("Coeficientes a, b:", solucion)
```

Si tiene solución, está en el espacio generado. Si no, no.

### Resumen

#### ¿Qué es una combinación lineal y cuál es su importancia?

El concepto de combinación lineal es clave en matemáticas y física, especialmente en el álgebra lineal. Una combinación lineal se refiere a la combinación de vectores mediante la multiplicación de cada uno por un escalar seguido de la suma de los resultados. La importancia radica en su capacidad para generar nuevos vectores a partir de otros existentes y describir espacios completos, como es el caso de \( \mathbb{R}^2 \).

#### ¿Cómo se realiza una combinación lineal de vectores?

Para ilustrar el proceso de combinación lineal de vectores, te mostramos el siguiente ejemplo:

Imagina dos vectores \( \mathbf{v1} = (1, 2) \) y \( \mathbf{v2} = (5, -2) \). Una combinación lineal de \( \mathbf{v1} \) y \( \mathbf{v2} \) podría ser calcular \( 2 \cdot \mathbf{v1} + 3 \cdot \mathbf{v2} \). En este caso:

- Multiplicamos \( \mathbf{v1} \) por 2: \( 2 \cdot (1, 2) = (2, 4) \).
- Multiplicamos \( \mathbf{v2} \) por 3: \( 3 \cdot (5, -2) = (15, -6) \).
- Sumamos ambos resultados: \( (2, 4) + (15, -6) = (17, -2) \).

El vector resultante \( (17, -2) \) es la combinación lineal de \( \mathbf{v1} \) y \( \mathbf{v2} \).

#### ¿Cómo visualizar combinaciones lineales gráficamente?

Para representar gráficamente combinaciones lineales, se puede utilizar una programación en Python con bibliotecas como Matplotlib y NumPy. El proceso implica definir los vectores originales, calcular la combinación lineal y, finalmente, utilizar una función de graficación para visualizar estos vectores.

Aquí tienes un ejemplo de cómo realizarlo:

```python
import matplotlib.pyplot as plt
import numpy as np

# Función para graficar los vectores
def graficar_vectores(vectores, colores):
    plt.figure()
    plt.quiver(0, 0, vectores[:, 0], vectores[:, 1], angles='xy', scale_units='xy', scale=1, color=colores)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Definición de los vectores
v1 = np.array([1, 2])
v2 = np.array([5, -2])

# Cálculo de la combinación lineal
comb_lineal = 2*v1 + 3*v2

# Graficar los vectores y su combinación lineal
graficar_vectores(np.array([v1, v2, comb_lineal]), ['orange', 'blue', 'red'])
```

Al ejecutar el código anterior, se podrán visualizar los vectores iniciales en naranja y azul, y su combinación lineal en rojo.

#### ¿Por qué son importantes las combinaciones lineales?

Las combinaciones lineales permiten una comprensión profunda de la estructura de los espacios vectoriales. Al manipular vectores adecuadamente, es posible representar y definir dominios completos. Por ejemplo, en \( \mathbb{R}^2 \), con la combinación lineal adecuada de dos vectores no colineales, puedes describir cualquier vector del plano. Esto tiene aplicaciones significativas en computación gráfica, optimización y más áreas.

#### ¿Describen siempre un espacio completo?

No todas las combinaciones lineales de vectores describen un espacio completo. La clave para que esto suceda reside en la elección de los vectores iniciales. Deben ser linealmente independientes, es decir, no deben ser múltiplos el uno del otro. Esto garantiza la cobertura completa del espacio deseado, como cuando se usan vectores base en un espacio de dimensión mayor.

En resumen, las combinaciones lineales son no solo un componente matemático esencial, sino también una herramienta poderosa para resolver un sinfín de problemas en ciencias aplicadas. Así que, ¡sigue explorando y experimentando con ellas!

## Combinaciones Lineales y Espacios Vectoriales en R2 y R3

¡Perfecto! Vamos a conectar los conceptos de **combinaciones lineales** con los **espacios vectoriales** en $\mathbb{R}^2$ y $\mathbb{R}^3$, combinando teoría clara con ejemplos prácticos en Python.

### 🔷 1. Combinaciones Lineales en $\mathbb{R}^2$ y $\mathbb{R}^3$

### 📌 Definición

Una **combinación lineal** de vectores $\vec{v}_1, \vec{v}_2, ..., \vec{v}_n$ es:

$$
\vec{v} = a_1 \vec{v}_1 + a_2 \vec{v}_2 + \dots + a_n \vec{v}_n
$$

donde $a_i \in \mathbb{R}$ son escalares.

### 📌 ¿Qué es el "espacio generado"?

El conjunto de **todas** las combinaciones lineales posibles de un conjunto de vectores se llama el **espacio generado** (o **subespacio generado**).

### 🟦 2. Combinaciones Lineales en $\mathbb{R}^2$

### 🔹 Caso 1: Dos vectores linealmente independientes

Sean:

$$
\vec{v}_1 = \begin{bmatrix}1 \\ 0\end{bmatrix}, \quad
\vec{v}_2 = \begin{bmatrix}0 \\ 1\end{bmatrix}
$$

Cualquier combinación lineal de ellos puede cubrir **todo $\mathbb{R}^2$**.

### 🔹 Caso 2: Dos vectores linealmente dependientes

$$
\vec{v}_1 = \begin{bmatrix}2 \\ 4\end{bmatrix}, \quad
\vec{v}_2 = \begin{bmatrix}1 \\ 2\end{bmatrix}
$$

Como $\vec{v}_1 = 2 \cdot \vec{v}_2$, **generan una línea recta**: no cubren todo el plano.

### 🟥 3. Combinaciones Lineales en $\mathbb{R}^3$

### 🔹 Caso 1: Tres vectores en el mismo plano

Si los vectores están en el mismo plano (uno es combinación lineal de los otros dos), el espacio generado es un **plano** en $\mathbb{R}^3$.

### 🔹 Caso 2: Tres vectores linealmente independientes

Entonces generan todo el **espacio tridimensional $\mathbb{R}^3$**.

### 🔧 4. Implementación en Python

### 📍 Visualización en 2D

```python
import numpy as np
import matplotlib.pyplot as plt

def graficar_vectores_2d(vectores, colores=None):
    plt.figure()
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')

    if colores is None:
        colores = ['r', 'g', 'b']

    for i, v in enumerate(vectores):
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=colores[i])
        plt.text(v[0]*1.1, v[1]*1.1, str(v), fontsize=12)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.title("Vectores en R2")
    plt.show()

# Ejemplo
v1 = np.array([1, 0])
v2 = np.array([0, 1])
graficar_vectores_2d([v1, v2])
```

### 📍 Visualización en 3D

```python
from mpl_toolkits.mplot3d import Axes3D

def graficar_vectores_3d(vectores, colores=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colores is None:
        colores = ['r', 'g', 'b']

    for i, v in enumerate(vectores):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=colores[i])
        ax.text(v[0]*1.1, v[1]*1.1, v[2]*1.1, str(v), fontsize=10)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Vectores en R3")
    plt.show()

# Ejemplo
v1 = np.array([1, 2, 0])
v2 = np.array([-1, 1, 0])
v3 = np.array([0, 0, 1])
graficar_vectores_3d([v1, v2, v3])
```

### 🧪 5. Verificación con Álgebra Lineal

### ¿Están los vectores en el mismo plano?

```python
from numpy.linalg import matrix_rank

A = np.column_stack([v1, v2, v3])
print("Rango de los vectores:", matrix_rank(A))
```

* Si el rango es 2: los vectores están en un plano.
* Si es 3: generan todo $\mathbb{R}^3$.

### 🚀 6. Aplicaciones Prácticas

| Campo                | Aplicación                                               |
| -------------------- | -------------------------------------------------------- |
| **Física**           | Suma de fuerzas, descomposición de vectores              |
| **Robótica**         | Posición y movimiento en el espacio                      |
| **Gráficos 3D**      | Transformaciones y modelado en entornos tridimensionales |
| **Machine Learning** | Espacios de características, PCA, modelos lineales       |
| **Economía**         | Combinaciones de activos en un portafolio                |

### Resumen

#### ¿Cómo podemos generar espacios en sí mismos a partir de vectores?

En la investigación de álgebra lineal, comprender cómo los vectores pueden generar espacios es fundamental. Todo comienza con la combinación lineal de vectores, una técnica poderosa que permite crear espacios en sí mismos, como se vio en la clase anterior. Este proceso implica utilizar combinaciones específicas de vectores para formar un espacio determinado. Vamos a explorar cómo esto se realiza, utilizando vectores en diferentes espacios y cómo el resultado puede variar dependiendo de los vectores elegidos.

#### ¿Cómo graficar el espacio generado por vectores?

Para visualizar el espacio generado por vectores dados, usamos herramientas de programación como NumPy y Matplotlib, que permiten crear gráficos interactivos. El enfoque general es el siguiente:

1. **Definir los vectores**: Se comienza definiendo los vectores que se usarán para generar el espacio. Por ejemplo, tenemos:

- ( v_1 = \begin{bmatrix} -1 \ 1 \end{bmatrix} )
- ( v_2 = \begin{bmatrix} -1 \ -1 \end{bmatrix} )

2. **Implementar combinaciones lineales**: Utilizamos combinaciones lineales para visualizar el espacio formado. Esto implica integrar el código necesario para realizar las operaciones matemáticas y gráficas.

3. **Definir límites gráficos**: Establecemos límites para los ejes del gráfico, permitiendo así una visualización clara del espacio.

4. **Interpretar resultados**: En este caso, observamos que la combinación de estos vectores resulta en una recta, debido a la interdependencia de los vectores.

#### ¿Qué ocurre al modificar los vectores iniciales?

Los vectores que usamos para generar el espacio tienen un impacto directo en el tipo de espacio que podemos crear. Por ejemplo, cambiemos los vectores iniciales a:

- ( v_1 = \begin{bmatrix} 1 \ 0 \end{bmatrix} )
- ( v_2 = \begin{bmatrix} 2 \ -3 \end{bmatrix} )

Esto nos lleva a una diferente configuración. Al seguir los pasos para graficar este nuevo conjunto, nos damos cuenta de que, ahora, se puede generar el espacio ( \mathbb{R}^2 ) en su totalidad. Este tipo de transformaciones resaltan cómo cambiar los vectores altera dramáticamente el espacio resultante.

#### ¿Cómo se relacionan los subespacios en espacios de mayor dimensión?

Es usual en álgebra lineal trabajar en espacios de diferentes dimensiones. Por ejemplo, ( \mathbb{R}^3 ) puede contener subespacios como ( \mathbb{R}^2 ), y queremos observar cómo los subespacios interactúan en espacios de mayor dimensión. Para ilustrarlo:

1. **Definir vectores en (\mathbb{R}^3)**: Usamos vectores como:

- ( v_1 = \begin{bmatrix} 1 \ 0 \ 0 \end{bmatrix} )
- ( v_2 = \begin{bmatrix} 2 \ -3 \ 0 \end{bmatrix} )

2. **Configurar gráficos en 3D**: Utilizamos opciones gráficas en tres dimensiones, aportando una visualización más rica de las interacciones entre vectores.

3. **Comprender hiperplanos**: Un hiperplano en (\mathbb{R}^3) es un espacio de dimensión menos uno. Si trabajamos en (\mathbb{R}^3) y los vectores solo generan (\mathbb{R}^2), se dice que han generado un hiperplano.

#### ¿Cómo aplicar el conocimiento adquirido?

En el contexto de cualquier estudio matemático o de física, aplicar este conocimiento es esencial. Se pueden modelar situaciones complejas donde determinar el espacio generado por ciertas fuerzas (vectores) es vital. Esto también se aplica en economía y ciencias de la computación, donde el modelado espacial a menudo implica descomponer problemas en vectores y sus combinaciones para prever resultados o optimizar soluciones.

La comprensión y aplicación práctica de combinaciones lineales y subespacios prepara a los estudiantes y profesionales para desafíos en múltiples disciplinas. Entender cómo los vectores interactúan para formar espacios es crucial en el desarrollo de habilidades técnicas avanzadas, y representa la base para investigaciones más profundas en matemáticas aplicadas y teoría de sistemas.

## Relación entre combinaciones lineales y sistemas de ecuaciones lineales

¡Buena pregunta! La **relación entre combinaciones lineales y sistemas de ecuaciones lineales** es **fundamental** en álgebra lineal. Te explico con claridad y ejemplos visuales y algebraicos.

### ✅ ¿Qué es una combinación lineal?

Dado un conjunto de vectores $\vec{v}_1, \vec{v}_2, ..., \vec{v}_n$, una **combinación lineal** es cualquier vector de la forma:

$$
\vec{w} = a_1\vec{v}_1 + a_2\vec{v}_2 + \cdots + a_n\vec{v}_n
$$

donde $a_1, a_2, ..., a_n \in \mathbb{R}$.

### 🧮 ¿Cómo se relaciona con sistemas de ecuaciones?

Cuando **preguntas si un vector $\vec{w}$** puede escribirse como una combinación lineal de otros vectores, estás resolviendo **un sistema de ecuaciones lineales**.

### 🎯 Ejemplo concreto en $\mathbb{R}^2$

Supón que tienes:

$$
\vec{v}_1 = \begin{bmatrix}1\\2\end{bmatrix},\quad
\vec{v}_2 = \begin{bmatrix}3\\1\end{bmatrix},\quad
\vec{w} = \begin{bmatrix}9\\5\end{bmatrix}
$$

¿Existe una combinación lineal de $\vec{v}_1$ y $\vec{v}_2$ que dé $\vec{w}$?
Es decir:

$$
a\begin{bmatrix}1\\2\end{bmatrix} + b\begin{bmatrix}3\\1\end{bmatrix} = \begin{bmatrix}9\\5\end{bmatrix}
$$

Esto se traduce en un sistema de ecuaciones:

$$
\begin{cases}
a + 3b = 9 \\
2a + b = 5
\end{cases}
$$

Resolver este sistema te da los coeficientes $a$ y $b$ de la combinación lineal.

### 🧠 Conclusión: la relación clave

* **Combinación lineal**: buscar si un vector es "alcanzable" por otros vectores escalados y sumados.
* **Sistema de ecuaciones**: herramienta algebraica para encontrar los coeficientes de esa combinación.

### 📌 Entonces:

> Resolver un sistema lineal **es lo mismo que** encontrar una combinación lineal específica de vectores.

### 🚀 Aplicaciones prácticas

* **Geometría**: Determinar si un punto está en un plano o línea.
* **Álgebra lineal**: Ver si un conjunto genera un espacio (espacio generado).
* **Análisis de datos**: Modelos lineales en regresión.
* **Programación gráfica**: Movimiento y transformaciones en coordenadas.

### Resumen

#### ¿Cómo entender las combinaciones lineales en sistemas de ecuaciones?

Las combinaciones lineales nos permiten entender la relación entre vectores dentro de un espacio matemático, pero ¿cómo se relaciona esto con la resolución de sistemas de ecuaciones lineales? Imagina que tienes una matriz denominada A y deseas encontrar una solución B. La pregunta clave aquí es si podemos expresar B como una combinación lineal de dos vectores dados, generalmente denominados V1 y V2. Si podemos, encontraremos los valores de X1 y X2 que satisfacen esta ecuación.

#### ¿Cómo funcionan las matrices como sistemas generadores?

Una matriz puede ser vista como un sistema generador que forma espacios a partir de los vectores que la componen. Si consideramos dos vectores, por ejemplo, V1 y V2 que originan una línea recta en R², estos vectores generan un subespacio dentro de ese plano. Al multiplicar estos vectores por algunos valores, se exploran todas las combinaciones lineales posibles dentro de ese subespacio.

#### Ejemplo práctico con gráficos

Imagina que tienes dos vectores:

- V1 = [1, 1]
- V2 = [-1, -1]

Estos dos vectores generan una línea en el espacio R². Podemos visualizar este espacio utilizando gráficos. Para hacerlo:

1. Asignamos un valor ( a ) en el rango de (-10) a (10).
2. Asignamos un valor ( b ) también en el rango de (-10) a (10).
3. Trazamos la primera coordenada de V1 multiplicada por ( a ) y sumada a la primera coordenada de V2 multiplicada por ( b ).
4. Hacemos lo mismo para la segunda coordenada.

Esta representación gráfica muestra efectivamente el espacio generado por las combinaciones lineales de V1 y V2 con estos rangos de valores.

#### ¿Existen soluciones para cualquier vector B?

Cuando intentamos resolver un sistema de ecuaciones con una matriz generadora y un vector determinado B, debemos plantearnos si B puede ser expresado como una combinación lineal de los vectores de la matriz. Si B puede ser escrito de esa manera, entonces en principio hay una solución posible. Pero esto no siempre es posible, especialmente si el vector B vive fuera del espacio generado por los vectores en la matriz.

#### Análisis de soluciones

Por ejemplo, si proponemos B como los valores [-10, 10], podemos notar que este vector está fuera del espacio generado por V1 y V2. Dado que ambos vectores son linealmente dependientes, no podemos expresar B como una combinación lineal de estos vectores. Este es el resultado típico cuando una matriz no tiene vectores linealmente independientes suficientes para abarcar el subespacio necesario.

#### ¿Qué significa la dependencia lineal en un sistema?

La dependencia lineal ocurre cuando uno de los vectores se puede escribir como múltiplo de otro. Esto significa que su influencia en la generación de un espacio es redundante y no aporta dimensiones adicionales. En el caso ejemplificado, V1 es el negativo de V2 ([1, 1] y [-1, -1]), lo que indica que no hay vectores adicionales en el espacio y, por lo tanto, la matriz efectivamente tiene menos dimensiones de las aparentes.

#### Consejos prácticos

- **Analiza la independencia**: Asegúrate de que los vectores en tu matriz son linealmente independientes para garantizar que puedes abarcar el subespacio necesario.
- **Visualiza el espacio**: Utiliza herramientas como gráficos para visualizar el espacio generado por los vectores, esto puede ofrecer una perspectiva valiosa que facilita la comprensión.
- **Comprende las limitaciones**: No todas las matrices pueden resolver para cualquier vector B; entiende las limitaciones de tus sistemas de ecuaciones.

Con esta comprensión podrás abordar problemas de álgebra lineal con mayor confianza y habilidad. ¡Sigue explorando y practicando para fortalecer tu dominio sobre las matrices y las combinaciones lineales en los sistemas de ecuaciones!

## Matrices y Dependencia Lineal en Sistemas de Ecuaciones

Las **matrices** son la forma más compacta de representar y resolver sistemas de ecuaciones lineales, y a su vez nos permiten estudiar de manera eficiente la **dependencia lineal** de sus filas o columnas. A continuación te explico ambos conceptos y te muestro ejemplos en Python.

### 1. Representación matricial de un sistema

Un sistema de $m$ ecuaciones lineales con $n$ incógnitas:

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1\\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2\\
\quad\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

se escribe como

$$
A\,\mathbf{x} = \mathbf{b},
\quad
A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n}\\
a_{21} & a_{22} & \cdots & a_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix},\ 
\mathbf{x}=\begin{pmatrix}x_1\\\vdots\\x_n\end{pmatrix},\ 
\mathbf{b}=\begin{pmatrix}b_1\\\vdots\\b_m\end{pmatrix}.
$$

### 2. Dependencia lineal y rango de la matriz

* Un conjunto de vectores (filas o columnas de $A$) es **linealmente dependiente** si al menos uno de ellos puede expresarse como combinación lineal de los demás.
* El **rango** de $A$, $\mathrm{rank}(A)$, es el número máximo de filas (o columnas) linealmente independientes.

### 🔑 Hechos clave

1. **Sistema compatible determinado** (una única solución) ↔ $\mathrm{rank}(A) = \mathrm{rank}([A\,|\,\mathbf b]) = n$.
2. **Sistema compatible indeterminado** (infinitas soluciones) ↔ $\mathrm{rank}(A) = \mathrm{rank}([A\,|\,\mathbf b]) < n$.
3. **Sistema incompatible** (sin solución) ↔ $\mathrm{rank}(A) < \mathrm{rank}([A\,|\,\mathbf b])$.

### 3. Ejemplo en Python

```python
import numpy as np
from numpy.linalg import matrix_rank, solve, lstsq

# Matriz de coeficientes A y vector b
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 1, 1]], dtype=float)
b = np.array([6, 12, 3], dtype=float)

# 1) Cálculo de rangos
rA  = matrix_rank(A)
rAb = matrix_rank(np.c_[A, b])   # matriz aumentada [A | b]

print("rank(A) =", rA)
print("rank([A|b]) =", rAb)

# 2) Dependencia lineal entre columnas
#    Observamos que la 2ª columna = 2 × 1ª columna, por tanto rango < 3
print("¿Columnas dependientes?", rA < A.shape[1])

# 3) Resolver sistema
if rA == rAb == A.shape[1]:
    x = solve(A, b)
    print("Solución única:", x)
elif rA == rAb < A.shape[1]:
    # Infinitas soluciones: obtenemos una mínima-norma con lstsq
    x, residuals, _, _ = lstsq(A, b, rcond=None)
    print("Solución mínima-norma:", x)
else:
    print("Sistema incompatible (sin solución)")
```

**Salida esperada:**

```
rank(A) = 2
rank([A|b]) = 2
¿Columnas dependientes? True
Solución mínima-norma: [0.42857143 0.85714286 0.        ]
```

* `rank(A)=2<3` nos dice que las 3 columnas de $A$ son dependientes y generan un subespacio de dimensión 2.
* Como `rank(A)=rank([A|b])=2 < 3`, hay infinitas soluciones; `lstsq` da la de norma mínima.

### 4. Geometría de la dependencia lineal

* En $\mathbb{R}^3$, tres columnas dependientes significan que todos los puntos $A\mathbf{x}$ caen en un **plano** o **línea** (subespacio de dimensión 2 o 1).
* Si fueran independientes (rango 3), generarían todo $\mathbb{R}^3$.

### 5. Visualización rápida (2D)

Para ver un caso simple en $\mathbb{R}^2$, donde dos columnas dependientes generan una línea:

```python
import matplotlib.pyplot as plt

# Dos vectores dependientes en R2
u = np.array([1, 2])
v = 2*u       # dependiente

# Genero combinaciones a·u + b·v
pts = [a*u + b*v for a in range(-3,4) for b in range(-3,4)]
X, Y = zip(*pts)

plt.scatter(X, Y, s=10, alpha=0.6)
plt.axhline(0,color='gray'); plt.axvline(0,color='gray')
plt.gca().set_aspect('equal')
plt.title("Línea generada por vectores dependientes")
plt.show()
```

Verás que todos los puntos están alineados: **la dependencia lineal aparece como una “línea” en 2D**.

### 📝 Resumen

1. **Matriz** = forma compacta de un sistema lineal.
2. **Rango** identifica cuántas filas/columnas son independientes.
3. **Dependencia lineal** ↔ columnas (o filas) “sobran” y generan un subespacio de menor dimensión.
4. El **rango** y el **rango aumentado** determinan si el sistema tiene única, infinitas o ninguna solución.

### Resumen

#### ¿Qué condiciones debe cumplir una matriz para que un sistema de ecuaciones lineales tenga solución?

Para que un sistema de ecuaciones lineales tenga solución, es esencial que la matriz ( A ) que representa el sistema tenga ciertas características. La matriz debe ser cuadrada y todos sus vectores deben ser linealmente independientes. Esto significa que ninguno de los vectores que componen la matriz puede ser expresado como una combinación lineal de otros vectores. Ahora, veamos un ejemplo práctico.

#### ¿Cómo identificar matrices linealmente dependientes?

Utilizar herramientas como NumPy en Python facilita la identificación de vectores linealmente dependientes en una matriz. Comencemos importando la biblioteca NumPy y definiendo nuestra matriz ( A ).

```python
import numpy as np

A = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
])
```

A primera vista, esta matriz parece cuadrada, puesto que tiene tantas filas como columnas. No obstante, es importante verificar que todos sus vectores sean linealmente independientes.

#### ¿Cómo se determina la dependencia lineal en una matriz?

Una forma eficaz de identificar dependencias lineales es mediante el cálculo de los autovalores y autovectores de la matriz. Los autovalores iguales a cero son indicativos de dependencia lineal.

Utilicemos NumPy para calcular estos valores.

```python
from numpy.linalg import eig

valores, vectores = eig(A)
# Detectamos los autovalores que son cero
```

Podemos observar que la tercera fila de la matriz ( A ), que se expresa como ([0, 1, 1, 0]), es linealmente dependiente, ya que puede escribirse como la suma de los vectores ([0, 1, 0, 0]) y ([0, 0, 1, 0]).

#### ¿Qué implicaciones tiene la dependencia lineal en una matriz?

La presencia de vectores linealmente dependientes en una matriz tiene consecuencias significativas. Principalmente, esto implica que no se puede calcular la inversa de dicha matriz, y es conocida como una matriz singular. Probemos calcular la inversa de nuestra matriz ( A ).

```python
from numpy.linalg import LinAlgError

try:
    A_inv = np.linalg.inv(A)
except LinAlgError:
    print("La matriz es singular y no tiene inversa.")
```

Esta singularidad se debe a la presencia de al menos un vector que es una combinación lineal de otros vectores de la matriz. Si removemos los vectores dependientes, la matriz resultante perdería su forma cuadrada, al no tener la misma cantidad de filas y columnas.

#### Estrategias para identificar vectores dependientes

Otra estrategia es analizar las columnas de la matriz. En el ejemplo presentado, observamos que la primera y la cuarta columna son idénticas, indicando que una depende de la otra. La eliminación de estas similitudes puede facilitar la conversión de la matriz en una versión cuadrada y funcional para encontrar soluciones a los sistemas de ecuaciones.

Conocer estas técnicas no solo es útil para las matemáticas teóricas, sino que también se aplica en diversos campos donde los sistemas de ecuaciones lineales juegan un papel fundamental, como la ingeniería, la economía y las ciencias computacionales. ¡Continúa explorando este fascinante mundo de las matrices y descubre cómo puedes aplicar estos conocimientos!

## Propiedades y Cálculo de la Norma de un Vector

¡Claro! Vamos a ver en detalle qué es la **norma de un vector**, cómo se **calcula**, y qué **propiedades** importantes tiene. Además, te muestro cómo implementarlo en **Python**.

### ✅ ¿Qué es la norma de un vector?

La **norma** de un vector mide su **magnitud** o “longitud” en el espacio. Se denota por:

$$
\|\vec{v}\|
$$

Para un vector $\vec{v} = (v_1, v_2, ..., v_n)$, la **norma euclidiana (L2)** se define como:

$$
\|\vec{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

### 📏 Tipos de normas comunes

| Nombre           | Fórmula                             | Notación   |   |                       |
| ---------------- | ----------------------------------- | ---------- | - | --------------------- |
| Norma L2         | $\|\vec{v}\|_2 = \sqrt{\sum v_i^2}$ | Euclidiana |   |                       |
| Norma L1         | ( \|\vec{v}\|\_1 = \sum             | v\_i       | ) | Manhattan / Taxicab   |
| Norma $L_\infty$ | ( \|\vec{v}\|\_\infty = \max        | v\_i       | ) | Máximo valor absoluto |

### 🧠 Propiedades de la norma

Sea $\vec{u}, \vec{v} \in \mathbb{R}^n$ y $\alpha \in \mathbb{R}$, entonces:

1. **Positividad**:

   $$
   \|\vec{v}\| \geq 0,\quad \text{y} \quad \|\vec{v}\| = 0 \iff \vec{v} = \vec{0}
   $$

2. **Homogeneidad (escalar)**:

   $$
   \|\alpha \vec{v}\| = |\alpha| \cdot \|\vec{v}\|
   $$

3. **Desigualdad triangular**:

   $$
   \|\vec{u} + \vec{v}\| \leq \|\vec{u}\| + \|\vec{v}\|
   $$

### 🧮 Cálculo de la norma en Python

```python
import numpy as np

# Vector en R3
v = np.array([3, 4, 12])

# Norma Euclidiana (L2)
norma_l2 = np.linalg.norm(v)

# Norma L1
norma_l1 = np.linalg.norm(v, ord=1)

# Norma infinito
norma_inf = np.linalg.norm(v, ord=np.inf)

print("Norma L2 (Euclidiana):", norma_l2)
print("Norma L1 (Manhattan):", norma_l1)
print("Norma Infinito:", norma_inf)
```

**Salida esperada:**

```
Norma L2 (Euclidiana): 13.0
Norma L1 (Manhattan): 19
Norma Infinito: 12
```

### 📌 Aplicaciones prácticas

* 🔍 **Análisis de errores**: distancia entre predicciones y datos reales.
* 🧠 **Normalización de datos** en machine learning.
* 🧭 **Dirección y magnitud** en física.
* 💻 **Reducción de dimensiones** y compresión de información.
* 📉 **Medida de similitud** entre vectores.

### 🎯 Extra: Normalizar un vector

Para obtener un vector **unitario** (longitud 1) en la misma dirección:

```python
v_unitario = v / np.linalg.norm(v)
print("Vector normalizado:", v_unitario)
print("Norma del vector normalizado:", np.linalg.norm(v_unitario))  # Siempre 1
```

### Resumen

#### ¿Qué es la Norma de un vector y por qué es importante?

La Norma de un vector es una herramienta matemática clave para medir el tamaño de un vector. Esta medida se representa mediante un número que siempre es cero o positivo. La Norma ayuda a determinar aspectos críticos, como el error en aproximaciones o la efectividad en clasificaciones. En este contexto, es vital conocer las propiedades de la Norma para aplicarlas correctamente.

#### ¿Cuáles son las propiedades de la Norma?

1. **Nunca negativa**: La Norma de cualquier vector nunca es negativa. Puede ser cero si el vector se encuentra exactamente en el origen, y este es el único caso en que la Norma será cero.

2. **Desigualdad triangular**: La suma de los vectores tiene una Norma que es siempre menor o igual a la suma de sus Normas individuales. Esto refleja el principio de que la distancia más corta entre dos puntos es una línea recta.

3. **Escalar por un vector**: Cuando multiplicamos un vector por un escalar, la Norma del resultado es igual al valor absoluto del escalar multiplicado por la Norma del vector original.

#### ¿Cómo calcular la Norma en Python?

Calcular la Norma de un vector en Python es sencillo con la librería `numpy`. A continuación, mostramos cómo realizar este cálculo utilizando un ejemplo práctico.

```python
import numpy as np

# Definimos los vectores
B1 = np.array([2, 7])
B2 = np.array([3, 5])

# Calculamos la suma de los vectores
B1_B2 = B1 + B2  # Resultado: array([5, 12])

# Calculamos la Norma de cada vector usando la función `np.linalg.norm`
norma_B1 = np.linalg.norm(B1)
norma_B2 = np.linalg.norm(B2)
norma_B1_B2 = np.linalg.norm(B1_B2)

# Verificamos la desigualdad triangular
assert norma_B1_B2 <= norma_B1 + norma_B2
```

Este código ayuda a visualizar la aplicación de la desigualdad triangular y la medida de Normas individuales y conjuntas.

#### ¿Cómo graficar vectores y su Norma en un plano?

Podemos visualizar la Norma y sus propiedades geométricas en Python utilizando `matplotlib` para gráficos y `seaborn` para opciones de color. Aquí se presenta una guía básica para graficar vectores y comprender la desigualdad triangular visualmente.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración básica para gráficos
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

# Definición de vectores y su origen
origen_B1 = np.array([0, 0]), B1
origen_B2 = np.array([0, 0]), B2
origen_suma = np.array([0, 0]), B1_B2

# Graficar vectores
plt.quiver(*origen_B1, angles='xy', scale_units='xy', scale=1, color=sns.color_palette("husl", 8)[1])
plt.quiver(*origen_B2, angles='xy', scale_units='xy', scale=1, color=sns.color_palette("husl", 8)[2])
plt.quiver(*origen_suma, angles='xy', scale_units='xy', scale=1, color=sns.color_palette("husl", 8)[0])

# Ajustar límites de los gráficos
plt.xlim(-0.5, 6)
plt.ylim(-0.5, 15)

# Mostrar gráfico
plt.show()
```

Visualizar los vectores y su Norma permite una comprensión más intuitiva de cómo operan estas matemáticas en el espacio bidimensional. Cada vector y su suma se hacen evidentes, destacando la aplicación de la desigualdad triangular.

¡Continúa explorando y experimentando con más ejemplos para dominar estos conceptos fundamentales!

## Cálculo de Normas en Python para Aprendizaje Automático

¡Perfecto! El cálculo de **normas de vectores** es fundamental en **aprendizaje automático (machine learning)**, ya que se usa para medir distancias, errores, regularización y más. A continuación te explico cómo y **por qué** se usan, y te muestro cómo implementarlas en Python con ejemplos prácticos.

### 🧠 ¿Por qué usamos normas en Machine Learning?

1. **Distancia entre puntos**: para clasificar o agrupar datos (e.g. k-NN, clustering).
2. **Regularización**: para evitar sobreajuste en modelos (L1 y L2).
3. **Normalización de datos**: para escalar características y mejorar el entrenamiento.
4. **Evaluación de errores**: en funciones de pérdida como MSE o MAE.

### 📏 Normas más utilizadas

| Norma               | Fórmula                             | Uso en ML                        |   |                               |
| ------------------- | ----------------------------------- | -------------------------------- | - | ----------------------------- |
| **L2 (Euclidiana)** | $\|\vec{x}\|_2 = \sqrt{\sum x_i^2}$ | Regularización Ridge, distancias |   |                               |
| **L1 (Manhattan)**  | ( \|\vec{x}\|\_1 = \sum             | x\_i                             | ) | Regularización Lasso, errores |
| **L∞ (máximo)**     | ( \|\vec{x}\|\_\infty = \max        | x\_i                             | ) | Detección de outliers         |

### 🧪 Ejemplo 1: Comparación de normas en vectores

```python
import numpy as np

x = np.array([3, -4, 5])

l1 = np.linalg.norm(x, ord=1)
l2 = np.linalg.norm(x)            # por defecto ord=2
linf = np.linalg.norm(x, ord=np.inf)

print(f"L1: {l1:.2f} | L2: {l2:.2f} | L∞: {linf:.2f}")
```

**Salida:**

```
L1: 12.00 | L2: 7.07 | L∞: 5.00
```

### 🤖 Ejemplo 2: Distancia entre vectores (e.g., k-NN)

```python
from sklearn.metrics import pairwise_distances

a = np.array([[1, 2]])
b = np.array([[4, 6]])

dist_euclid = pairwise_distances(a, b, metric='euclidean')
dist_manhat = pairwise_distances(a, b, metric='manhattan')

print("Distancia Euclidiana:", dist_euclid[0][0])
print("Distancia Manhattan:", dist_manhat[0][0])
```

### 🧰 Ejemplo 3: Regularización en regresión

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Datos sintéticos
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Regresión con regularización L2 (Ridge)
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train, y_train)
print("Coeficientes Ridge:", modelo_ridge.coef_)

# Regresión con regularización L1 (Lasso)
modelo_lasso = Lasso(alpha=0.1)
modelo_lasso.fit(X_train, y_train)
print("Coeficientes Lasso:", modelo_lasso.coef_)
```

* **Ridge** tiende a reducir los coeficientes.
* **Lasso** tiende a hacer que algunos sean exactamente **0** → selección de características.

### ✏️ Ejemplo 4: Normalizar vectores (unit norm)

```python
from sklearn.preprocessing import Normalizer

X = np.array([[3, 4], [1, -1], [0, 5]])

normalizador = Normalizer(norm='l2')  # también acepta 'l1' o 'max'
X_norm = normalizador.fit_transform(X)

print("Vectores normalizados:\n", X_norm)
```

### ✅ Conclusión

Las normas son herramientas esenciales en ML para:

* **Medir similitud o distancia** (clustering, k-NN).
* **Reducir el sobreajuste** (regularización L1/L2).
* **Escalar y preparar datos** (normalización).

### Resumen

#### ¿Qué son las normas y cómo se utilizan en aprendizaje automático?

Las normas son herramientas fundamentales en el aprendizaje automático y otras áreas de la ciencia de datos utilizadas para medir diversas propiedades de los vectores. Existen diferentes tipos de normas que se emplean para calcular errores, distancias y más. En este artículo, exploraremos las normas más comunes y discutiremos cómo se pueden implementar utilizando la biblioteca NumPy en Python. Las normas que abordaremos incluyen L0, L1, L2 y la norma infinita.

#### ¿Cómo calcular la norma L0?

La norma L0 es la más sencilla de entender: calcula la cantidad de elementos distintos de cero en un vector. Es útil para determinar elementos no nulos, por ejemplo, al evaluar la cantidad de compras realizadas por usuarios, donde cada componente del vector representa una compra. Este es el procedimiento para calcular la norma L0 en Python con NumPy:

```python
import numpy as np

# Definimos un vector
vector = np.array([1, 2, 0, 5, 6, 0])

# Calculamos la norma L0
norma_l0 = np.linalg.norm(vector, ord=0)

print(norma_l0)  # Devuelve 4, hay 4 elementos distintos de cero.
```

#### ¿Cómo se calcula la norma L1?

La norma L1, también conocida como norma de suma absoluta, entrega la suma de los valores absolutos de los componentes del vector. Esta norma cobra relevancia en situaciones donde necesitamos una medida que dependa linealmente de cada componente del vector:

```python
# Definimos un vector con valores positivos y negativos
vector = np.array([1, -1, 1, -1, 1])

# Calculamos la norma L1
norma_l1 = np.linalg.norm(vector, ord=1)

print(norma_l1)  # Devuelve 5, la suma de valores absolutos.
```

#### ¿Por qué es importante la norma L2?

La norma L2 es probablemente la más conocida. Está relacionada con la distancia euclidiana, la medida estándar en geometría para calcular la distancia entre dos puntos en un espacio. Se utiliza ampliamente en aprendizaje automático debido a su simplicidad y eficacia computacional. Al elevar los componentes al cuadrado en lugar de tomar la raíz cuadrada, es posible optimizar algoritmos para mejorar el rendimiento:

```python
# Definimos un vector
vector = np.array([1, 1])

# Calculamos la norma L2
norma_l2 = np.linalg.norm(vector)

print(norma_l2)  # Devuelve aproximadamente 1.41, la raíz cuadrada de 2.

# Calculamos la norma L2 al cuadrado
norma_l2_squared = np.linalg.norm(vector) ** 2

print(norma_l2_squared)  # Devuelve 2.

# También se puede calcular usando el producto interno
norma_l2_squared_internal = np.dot(vector, vector)

print(norma_l2_squared_internal)  # Devuelve 2.
```

#### ¿Qué es la norma infinita y cómo se calcula?

La norma infinita proporciona el valor absoluto más grande de un vector. Es útil en situaciones en las que necesitamos detectar valores extremos que puedan ser significativos para un análisis más detallado. Su cálculo en Python es sencillo usando NumPy:

```python
# Definimos un vector con un valor prominente
vector = np.array([1, 2, 3, -100])

# Calculamos la norma infinita
norma_inf = np.linalg.norm(vector, ord=np.inf)

print(norma_inf)  # Devuelve 100, el valor absoluto máximo del vector.
```

Las normas son herramientas versátiles y potentes en el aprendizaje automático, desempeñando un papel crucial para evaluar diferentes aspectos de los datos de entrada. Su correcta aplicación puede mejorar significativamente la eficiencia de los algoritmos. A medida que avances en tus estudios y aplicaciones de machine learning, comprender y utilizar estas normas te será cada vez más indispensable. ¡Sigue aprendiendo y explorando el vasto mundo del aprendizaje automático!