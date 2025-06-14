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