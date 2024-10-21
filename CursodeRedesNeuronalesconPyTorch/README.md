# Curso de Redes Neuronales con PyTorch

## ¿Qué necesitas para aprender PyTorch?

Para aprender PyTorch de manera efectiva, necesitarás lo siguiente:

### 1. **Fundamentos de Python y programación**:
   - Familiaridad con la sintaxis de Python y conceptos como bucles, funciones, clases, y listas es fundamental.
   - Conocimiento en bibliotecas como **NumPy** y **Pandas** es útil, ya que estas se usan frecuentemente junto a PyTorch para manipulación de datos.

### 2. **Bases de Álgebra Lineal y Cálculo**:
   - Conocimientos sobre **matrices**, **vectores**, **producto punto**, y operaciones matemáticas como derivadas y gradientes, ya que PyTorch se utiliza para construir redes neuronales que dependen de estos conceptos.

### 3. **Conocer Redes Neuronales y Deep Learning**:
   - Comprender cómo funcionan las **redes neuronales** y **el backpropagation**.
   - Familiarizarse con conceptos de deep learning como **funciones de activación**, **función de pérdida**, **optimización con gradiente descendente**, **dropout**, **regularización**, etc.

### 4. **Entorno de trabajo**:
   - Instalar **PyTorch** en tu computadora o entorno virtual. PyTorch tiene una instalación sencilla con `pip install torch`.
   - **Jupyter Notebooks** o **Google Colab** son buenas herramientas para empezar a trabajar con PyTorch, ya que permiten probar el código de forma interactiva.

### 5. **Práctica con ejemplos básicos**:
   - Implementar redes neuronales sencillas, como una red **Feedforward** o **Red Neuronal Convolucional (CNN)**.
   - Trabajar en problemas clásicos de clasificación (como **MNIST** o **CIFAR-10**) para ver cómo entrenar, evaluar y mejorar un modelo con PyTorch.

### 6. **Documentación y tutoriales**:
   - Aprovechar la **documentación oficial de PyTorch**: https://pytorch.org/docs/
   - Seguir tutoriales básicos y ejemplos para familiarizarte con la API de PyTorch.

### 7. **Ejemplos avanzados**:
   - Una vez que tengas los conceptos básicos, puedes profundizar en temas más avanzados como **redes neuronales recurrentes (RNNs)**, **transformers**, **GANs**, o **aprendizaje por refuerzo**.

En resumen, dominar los conceptos de redes neuronales y tener una base sólida en Python son los primeros pasos. Luego, aprender PyTorch se puede hacer progresivamente, comenzando con ejemplos básicos y luego avanzando hacia implementaciones más complejas.

**Lecturas recomendadas**

- [Curso de Redes Neuronales con Python y Keras - Platzi](https://platzi.com/cursos/redes-neuronales/)

- [Platzi: Cursos online profesionales de tecnología](https://platzi.com/ruta/deep-learning-python/)

- [Conceptos de Redes Neuronales - Material Extra.pdf - Google Drive](https://drive.google.com/file/d/1n5A1sKMHt4yDd50b016cBSJbqtDSamGA/view)

- [template_3_Clasificación_de_datos_con_TorchText.ipynb - Google Drive](https://drive.google.com/file/d/1Rtdp-2uppNiw7wMaVgp4UpKYjRvYWBHG/view)

- [3_Clasificación_de_datos_con_TorchText.ipynb - Google Drive](https://drive.google.com/file/d/1NlRhIvV4RNG0CBT_6TPjrExRB7df-ok3/view)

- [Curso de Redes Neuronales con PyTorch.pdf - Google Drive](https://drive.google.com/file/d/1rOpEsYvH5BEjSdeuOII_gRD4BU0iyB56/view)

## ¿Por qué usar PyTorch?

Usar **PyTorch** tiene varias ventajas que lo han hecho muy popular en el campo del **deep learning** y la investigación en **inteligencia artificial**. Aquí te explico por qué:

### 1. **Facilidad de uso y diseño intuitivo**:
   - PyTorch tiene una **interfaz sencilla y amigable** para desarrollar modelos. Su estructura es muy **pythonica**, lo que lo hace intuitivo para quienes ya están familiarizados con Python. A diferencia de otros frameworks, la sintaxis de PyTorch se siente natural al programar y sigue los principios de Python.
   - Además, el **modelo imperativo** (también llamado "define-by-run") permite construir redes neuronales sobre la marcha, lo que facilita depurar el código de forma dinámica.

### 2. **Dinamismo en los gráficos computacionales**:
   - PyTorch utiliza gráficos **dinámicos** de computación, lo que significa que puedes modificar el modelo en tiempo real, ejecutar código de control de flujo complejo (como `if-else`, bucles) y depurar fácilmente.
   - Esto contrasta con otros frameworks que utilizan gráficos estáticos (como TensorFlow 1.x), en los que debes definir todo el gráfico computacional antes de ejecutarlo.

### 3. **Soporte fuerte para investigación**:
   - Debido a su flexibilidad y facilidad para escribir código dinámico, PyTorch es **muy popular en el ámbito de la investigación** en deep learning. Muchos artículos y prototipos de investigaciones usan PyTorch porque permite probar ideas rápidamente.
   - También está bien documentado y tiene una **gran comunidad** de investigadores y desarrolladores que contribuyen y comparten ejemplos.

### 4. **Autograd**:
   - PyTorch tiene un sistema llamado **Autograd**, que automáticamente calcula los gradientes necesarios para entrenar una red neuronal, manejando de manera eficiente el cálculo de derivadas parciales y propagación hacia atrás. Esto facilita el entrenamiento y la optimización de los modelos, incluso en redes complejas.

### 5. **Soporte para GPU**:
   - PyTorch facilita el uso de **GPU** (con soporte para CUDA) para acelerar el entrenamiento y la inferencia de modelos. Puedes mover tus tensores y modelos fácilmente entre CPU y GPU con unas pocas líneas de código (`.cuda()` y `.cpu()`).
   - Esta capacidad hace que PyTorch sea muy eficiente para trabajar con grandes cantidades de datos o modelos complejos.

### 6. **Amplia comunidad y recursos**:
   - PyTorch tiene una **gran comunidad activa** que proporciona recursos como tutoriales, ejemplos y documentación extensa.
   - Hay soporte para muchas arquitecturas avanzadas de deep learning (por ejemplo, CNNs, RNNs, transformers), lo que facilita la implementación de algoritmos recientes en la investigación y aplicaciones industriales.

### 7. **Integración con otros frameworks**:
   - PyTorch se integra bien con otros frameworks y herramientas de machine learning, como **ONNX (Open Neural Network Exchange)**, lo que facilita la interoperabilidad con otros frameworks como TensorFlow. También puedes exportar modelos entrenados a formatos compatibles con otras plataformas para despliegue en producción.

### 8. **PyTorch Lightning y otras extensiones**:
   - Herramientas como **PyTorch Lightning** facilitan la estructura y modularidad del código, permitiendo enfocarse en el contenido y la lógica del modelo, mientras que los detalles de entrenamiento (manejo de GPUs, optimizadores, etc.) se abstraen.

### 9. **Ecosistema para producción**:
   - Aunque inicialmente se consideraba más adecuado para investigación, con la introducción de **TorchServe** y la mejora de soporte para **JIT (Just-in-Time)**, PyTorch es ahora una opción sólida para **despliegue en producción**, facilitando tanto la experimentación como el paso a producción.

### 10. **Compatibilidad con el desarrollo moderno de IA**:
   - PyTorch es compatible con arquitecturas modernas, como **Transformers** y redes neuronales profundas avanzadas. Muchas de las bibliotecas de NLP, como **Hugging Face Transformers**, se construyen sobre PyTorch.

En resumen, PyTorch es excelente por su facilidad de uso, flexibilidad, potencia y soporte para investigación avanzada y despliegue en producción, lo que lo convierte en una herramienta robusta para proyectos de **deep learning** en cualquier etapa del desarrollo.

**Lecturas recomendadas**

[State of AI Report 2022](https://www.stateof.ai/)

[PyTorch](https://pytorch.org/)

[All Things AI - The Complete Resource Of Artificial Intelligence Tools & Services](https://allthingsai.com/)

[PyTorch Foundation | PyTorch](https://pytorch.org/foundation)

[Hugging Face – The AI community building the future.](https://drive.google.com/file/d/1n5A1sKMHt4yDd50b016cBSJbqtDSamGA/view)

## Hola, mundo en PyTorch

Aquí tienes un ejemplo básico de "Hola, mundo" en PyTorch, que simplemente crea un tensor y lo imprime:

```python
import torch

# Crear un tensor
hola_mundo = torch.tensor([1, 2, 3])

# Imprimir el tensor
print("Hola, mundo en PyTorch:", hola_mundo)
```

Este código crea un tensor simple con los valores `[1, 2, 3]` utilizando PyTorch y lo imprime en la consola junto con el mensaje "Hola, mundo en PyTorch". Es una forma simple de verificar que PyTorch está funcionando correctamente en tu entorno.

## Creación de Tensores en PyTorch

En PyTorch, los tensores son estructuras de datos similares a los arrays de NumPy, pero con la ventaja de que pueden ser utilizados en GPUs para acelerar los cálculos. A continuación, te muestro varias formas de crear tensores en PyTorch:

### 1. **Crear un tensor a partir de una lista**
```python
import torch

tensor = torch.tensor([1, 2, 3, 4, 5])
print(tensor)
```

### 2. **Crear un tensor de ceros**
```python
tensor = torch.zeros(3, 4)  # Un tensor de 3 filas y 4 columnas lleno de ceros
print(tensor)
```

### 3. **Crear un tensor de unos**
```python
tensor = torch.ones(3, 4)  # Un tensor de 3 filas y 4 columnas lleno de unos
print(tensor)
```

### 4. **Crear un tensor con valores aleatorios**
```python
tensor = torch.rand(3, 4)  # Un tensor de 3 filas y 4 columnas con valores aleatorios entre 0 y 1
print(tensor)
```

### 5. **Crear un tensor con una secuencia de números**
```python
tensor = torch.arange(0, 10, step=2)  # Un tensor que contiene los valores [0, 2, 4, 6, 8]
print(tensor)
```

### 6. **Crear un tensor con una distribución normal**
```python
tensor = torch.randn(3, 4)  # Tensor con distribución normal estándar (media 0, desviación estándar 1)
print(tensor)
```

### 7. **Crear un tensor vacío**
```python
tensor = torch.empty(2, 3)  # Un tensor vacío de 2x3 (sin valores inicializados)
print(tensor)
```

### 8. **Convertir un array de NumPy a un tensor**
Si ya tienes un array en NumPy y deseas convertirlo a un tensor de PyTorch:
```python
import numpy as np

array = np.array([1, 2, 3, 4])
tensor = torch.from_numpy(array)
print(tensor)
```

### 9. **Especificar el tipo de datos del tensor**
Puedes crear un tensor de un tipo específico, por ejemplo, un tensor de enteros (`torch.int`) o de punto flotante (`torch.float`):
```python
tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # Tensor de tipo float
print(tensor)
```

### 10. **Crear un tensor en una GPU (si está disponible)**
Si tienes una GPU y deseas que el tensor sea creado directamente en la GPU, puedes especificarlo con `device='cuda'`:
```python
if torch.cuda.is_available():
    tensor = torch.tensor([1, 2, 3], device='cuda')  # Crear el tensor en la GPU
    print(tensor)
```

Estos son algunos de los métodos más comunes para crear tensores en PyTorch.

## Debugging de operaciones con tensores

El proceso de depuración ("debugging") de operaciones con tensores en PyTorch implica revisar el estado y las transformaciones que sufren los tensores durante el proceso de cálculo. Aquí te ofrezco algunas técnicas y herramientas útiles para depurar operaciones con tensores:

### 1. **Verificar las dimensiones de los tensores (shape)**
Un error común al trabajar con tensores es que no coincidan sus dimensiones para las operaciones. Usar `tensor.shape` te permite revisar la forma (dimensión) de los tensores.

```python
import torch

tensor = torch.rand(3, 4)
print(tensor.shape)  # Salida: torch.Size([3, 4])
```

Si una operación falla, lo primero que deberías hacer es verificar que las dimensiones sean compatibles.

### 2. **Imprimir el tensor en varios puntos del código**
Cuando algo no sale como esperas, puedes imprimir el tensor en diferentes etapas del cálculo para verificar su contenido.

```python
tensor = torch.rand(3, 4)
print(tensor)  # Imprime el contenido del tensor
```

También puedes verificar propiedades como:
- **dtype** (tipo de dato del tensor)
- **device** (si está en CPU o GPU)

```python
print(tensor.dtype)  # Tipo de dato del tensor (float, int, etc.)
print(tensor.device)  # Verificar si el tensor está en CPU o GPU
```

### 3. **Usar `assert` para validar condiciones**
Puedes usar `assert` para validar que ciertas propiedades del tensor sean las correctas antes de realizar una operación.

```python
tensor = torch.rand(3, 4)
assert tensor.shape == (3, 4), "El tensor no tiene la forma correcta"
```

### 4. **Tener cuidado con la asignación en GPU**
Si trabajas con GPU y ocurre un error, asegúrate de que los tensores estén en el mismo dispositivo. No puedes realizar operaciones entre tensores en dispositivos diferentes.

```python
if torch.cuda.is_available():
    tensor_cpu = torch.rand(3, 4)
    tensor_gpu = tensor_cpu.to('cuda')
    print(tensor_gpu.device)  # Verifica que el tensor esté en la GPU
```

Si intentas operar entre un tensor en CPU y otro en GPU, obtendrás un error, así que asegúrate de moverlos al mismo dispositivo:

```python
# tensor_cpu + tensor_gpu  -> Esto generará un error
tensor_cpu = tensor_cpu.to('cuda')  # Movemos ambos tensores a la GPU
result = tensor_cpu + tensor_gpu
```

### 5. **Comprobar errores numéricos (NaN, Inf)**
En ocasiones, los valores de los tensores pueden convertirse en `NaN` o `Inf` debido a cálculos mal condicionados (como divisiones por cero o logaritmos de valores negativos).

Puedes verificar si un tensor contiene estos valores:

```python
tensor = torch.tensor([float('inf'), -float('inf'), float('nan'), 1.0])

# Comprobar si hay NaNs
print(torch.isnan(tensor))  # Salida: tensor([False, False,  True, False])

# Comprobar si hay Infs
print(torch.isinf(tensor))  # Salida: tensor([ True,  True, False, False])
```

### 6. **Trazado con `autograd` para identificar errores en el cálculo de gradientes**
Si estás utilizando autograd y los gradientes no se calculan como esperas, puedes revisar el flujo de cálculo del gradiente mediante `torch.autograd`.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()

print(x.grad)  # Imprime el gradiente de `x`
```

Si en algún momento pierdes el gradiente o se genera un error en el flujo de cálculo, puedes depurar revisando el historial de operaciones.

### 7. **Uso de `torch.set_printoptions` para mejorar la visualización**
A veces los tensores grandes no se muestran completamente, lo cual puede dificultar la depuración. Puedes ajustar las opciones de impresión para mostrar más detalles del tensor:

```python
torch.set_printoptions(precision=3, edgeitems=2, linewidth=75)

tensor = torch.rand(100, 4)
print(tensor)
```

### 8. **Depurar operaciones en GPU**
Depurar tensores en GPU puede ser un desafío, ya que los errores son menos descriptivos. Para aislar el problema, puedes copiar el tensor de vuelta a la CPU y revisar su valor:

```python
tensor_gpu = torch.rand(3, 4).to('cuda')
tensor_cpu = tensor_gpu.cpu()
print(tensor_cpu)  # Inspecciona el tensor en CPU
```

### 9. **Comparación con NumPy**
A veces, puedes comparar resultados de operaciones en PyTorch con NumPy para asegurarte de que todo funcione como esperas.

```python
import numpy as np

tensor = torch.rand(3, 4)
array = tensor.numpy()

# Comparar una operación
assert np.allclose(tensor.sum().item(), array.sum()), "Los resultados no coinciden"
```

### 10. **Uso de herramientas de depuración como PDB**
Si necesitas una depuración más profunda, puedes usar el depurador interactivo de Python, `pdb`. Por ejemplo, puedes pausar el programa y examinar el estado de las variables en ese punto:

```python
import pdb

tensor = torch.rand(3, 4)
pdb.set_trace()  # Inicia el depurador interactivo aquí
```

Esto te permitirá ejecutar comandos para inspeccionar variables y entender el flujo del programa.

Estas técnicas te ayudarán a depurar operaciones con tensores y detectar problemas en tus redes neuronales o cualquier cálculo que estés haciendo en PyTorch.

**Lecturas recomendadas**

[Tensor Attributes — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/tensor_attributes.html)

[torch.Tensor.type — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.type.html)

[torch.cuda — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/cuda.html)

## Conversión y operación de tensores con PyTorch

La **conversión** y **operación** de tensores en PyTorch es esencial para manipular y procesar datos de manera eficiente. A continuación, te explico los conceptos clave y cómo realizar estas tareas.

### 1. **Creación de Tensores**

En PyTorch, los tensores son similares a los arrays de NumPy, pero con la ventaja de que pueden ser utilizados en GPU para realizar cálculos más rápidos.

```python
import torch

# Crear un tensor vacío
tensor_vacio = torch.empty(3, 4)  # Crea un tensor vacío de 3x4

# Crear un tensor con ceros
tensor_ceros = torch.zeros(3, 4)  # Tensor de 3x4 lleno de ceros

# Crear un tensor con unos
tensor_unos = torch.ones(3, 4)  # Tensor de 3x4 lleno de unos

# Crear un tensor con valores aleatorios
tensor_aleatorio = torch.rand(3, 4)  # Valores aleatorios en [0, 1)
```

### 2. **Conversiones entre Tensores y NumPy**

Los tensores de PyTorch pueden convertirse fácilmente en arrays de NumPy y viceversa. Esto es útil cuando ya tienes código que utiliza NumPy y quieres aprovechar PyTorch.

- **De Tensor a NumPy:**
  ```python
  tensor = torch.rand(3, 3)
  array_numpy = tensor.numpy()
  print(array_numpy)  # Conversión de tensor a array NumPy
  ```

- **De NumPy a Tensor:**
  ```python
  import numpy as np
  
  array_numpy = np.array([[1, 2], [3, 4]])
  tensor = torch.from_numpy(array_numpy)
  print(tensor)  # Conversión de array NumPy a tensor PyTorch
  ```

### 3. **Operaciones con Tensores**

PyTorch permite realizar operaciones matemáticas con tensores, similares a las que se pueden hacer con arrays de NumPy.

- **Suma de Tensores:**
  ```python
  x = torch.rand(2, 3)
  y = torch.rand(2, 3)

  suma = x + y  # Suma elemento por elemento
  print(suma)
  ```

- **Multiplicación Elemento por Elemento (Hadamard):**
  ```python
  mult_elemento = x * y
  print(mult_elemento)
  ```

- **Producto Matricial (Producto Punto):**
  ```python
  x = torch.rand(3, 2)
  y = torch.rand(2, 4)

  producto_matriz = torch.matmul(x, y)  # Producto matricial
  print(producto_matriz)
  ```

### 4. **Modificación del Tipo de Datos (`dtype`)**

Los tensores en PyTorch pueden ser de varios tipos, como `float32`, `int64`, etc. Puedes convertir el tipo de un tensor usando el método `to()` o cambiando el `dtype` directamente.

- **Especificar el Tipo de Dato al Crear el Tensor:**
  ```python
  tensor_flotante = torch.ones(3, 3, dtype=torch.float32)
  ```

- **Convertir el Tipo de Dato Después de Crear el Tensor:**
  ```python
  tensor = torch.rand(3, 3)
  tensor_entero = tensor.to(torch.int32)  # Convertir a enteros
  ```

### 5. **Mover Tensores entre CPU y GPU**

PyTorch soporta operaciones en GPU, lo cual acelera significativamente los cálculos. Puedes mover tensores entre CPU y GPU con `.to()`.

- **Mover un Tensor a la GPU:**
  ```python
  if torch.cuda.is_available():
      tensor_gpu = tensor.to('cuda')
      print(tensor_gpu.device)  # Verifica que el tensor esté en la GPU
  ```

- **Mover un Tensor de la GPU a la CPU:**
  ```python
  tensor_cpu = tensor_gpu.to('cpu')
  ```

### 6. **Operaciones de Redimensionado y Transposición**

A veces necesitas cambiar la forma de un tensor o reordenar sus dimensiones. Estas operaciones son muy comunes en procesamiento de datos.

- **Cambiar la Forma (`view`):**
  ```python
  tensor = torch.rand(4, 4)
  tensor_reshaped = tensor.view(2, 8)  # Redimensionar a 2x8
  print(tensor_reshaped.shape)  # Verifica la nueva forma
  ```

- **Transponer un Tensor (`transpose`):**
  ```python
  tensor = torch.rand(2, 3)
  tensor_transpose = torch.transpose(tensor, 0, 1)  # Intercambia las dimensiones 0 y 1
  print(tensor_transpose.shape)  # Verifica la forma después de transponer
  ```

### 7. **Operaciones de Agregación**

Puedes calcular estadísticas como sumas, medias, mínimos y máximos a lo largo de un eje del tensor.

- **Suma de Todos los Elementos:**
  ```python
  tensor = torch.rand(3, 3)
  suma_total = tensor.sum()
  print(suma_total)  # Imprime la suma de todos los elementos del tensor
  ```

- **Media por Ejes Específicos:**
  ```python
  media_filas = tensor.mean(dim=1)  # Media a lo largo de las filas (dimensión 1)
  print(media_filas)
  ```

### 8. **Operaciones de Corte e Indexado**

Al igual que con los arrays de NumPy, puedes acceder a subconjuntos de un tensor o modificar partes específicas de él.

- **Acceder a un Elemento:**
  ```python
  tensor = torch.rand(4, 4)
  print(tensor[1, 2])  # Acceder al elemento en la fila 1, columna 2
  ```

- **Modificar un Elemento:**
  ```python
  tensor[1, 2] = 10  # Asignar un valor al elemento en la fila 1, columna 2
  ```

- **Cortar Tensores:**
  ```python
  tensor_slice = tensor[:, 1]  # Seleccionar la segunda columna de todas las filas
  ```

### 9. **Operaciones de Concatenación**

Puedes unir varios tensores a lo largo de un eje específico.

- **Concatenar Tensores:**
  ```python
  tensor_a = torch.rand(2, 3)
  tensor_b = torch.rand(2, 3)

  tensor_concat = torch.cat((tensor_a, tensor_b), dim=0)  # Concatenar a lo largo de las filas
  print(tensor_concat)
  ```

### 10. **Operaciones de Comparación**

Puedes comparar tensores para obtener un tensor booleano (True/False) en cada posición.

- **Comparar Elementos:**
  ```python
  tensor = torch.rand(2, 3)
  comparacion = tensor > 0.5  # Comparar si los elementos son mayores que 0.5
  print(comparacion)
  ```

### Conclusión

PyTorch ofrece una gran flexibilidad para trabajar con tensores y realizar operaciones numéricas de manera eficiente. Las operaciones de conversión entre NumPy y PyTorch, junto con la posibilidad de mover tensores entre CPU y GPU, permiten un flujo de trabajo robusto y acelerado para el desarrollo de modelos de aprendizaje profundo y otras tareas.

**Lecturas recomendadas** 

[Conceptos de Redes Neuronales - Material Extra.pdf - Google Drive](https://drive.google.com/file/d/1n5A1sKMHt4yDd50b016cBSJbqtDSamGA/view)

## Generación y split de datos para entrenamiento de modelo

La generación y el split de datos son pasos cruciales en el proceso de entrenamiento de modelos de machine learning. Aquí te explico los conceptos clave y cómo se implementan:

### Generación de Datos
La generación de datos puede referirse a varios métodos, incluyendo:

1. **Recopilación de Datos:**
   - Obtener datos de fuentes existentes (bases de datos, archivos CSV, APIs, etc.).
   - Generar datos sintéticos usando técnicas como la simulación o algoritmos generativos.

2. **Preprocesamiento:**
   - Limpieza de datos: Eliminar duplicados, manejar valores nulos, corregir errores en los datos.
   - Transformación de datos: Normalización, escalado, codificación de variables categóricas, etc.
   - División en características (features) y etiquetas (labels).

### Split de Datos
Dividir el conjunto de datos en diferentes subsets es esencial para evaluar el rendimiento del modelo. Los splits más comunes son:

1. **Training Set (Conjunto de Entrenamiento):**
   - Usado para entrenar el modelo. Generalmente, este conjunto representa el 70-80% de los datos.

2. **Validation Set (Conjunto de Validación):**
   - Usado para ajustar los hiperparámetros del modelo y evitar el overfitting. Comúnmente representa un 10-15% de los datos.

3. **Test Set (Conjunto de Prueba):**
   - Usado para evaluar el rendimiento final del modelo. Similar al conjunto de validación, representa el 10-15% de los datos.

### Implementación en Python
Aquí te muestro un ejemplo de cómo puedes dividir un conjunto de datos usando `train_test_split` de `scikit-learn`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar datos
data = pd.read_csv('dataset.csv')

# Separar características y etiquetas
X = data.drop('target', axis=1)  # Características
y = data['target']                # Etiquetas

# Dividir el conjunto de datos
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Tamaños de los conjuntos
print(f'Tamaño del conjunto de entrenamiento: {X_train.shape[0]}')
print(f'Tamaño del conjunto de validación: {X_val.shape[0]}')
print(f'Tamaño del conjunto de prueba: {X_test.shape[0]}')
```

### Consideraciones
- **Aleatoriedad:** Siempre es recomendable establecer una semilla (usando `random_state`) para garantizar que los splits sean reproducibles.
- **Estratificación:** Si el conjunto de datos es desequilibrado, es recomendable usar `stratify` en `train_test_split` para mantener la proporción de clases en los conjuntos.

## Estructura de modelo en PyTorch con torch.nn

Claro, la estructura de un modelo en PyTorch se define comúnmente utilizando el módulo `torch.nn`, que proporciona herramientas para construir redes neuronales de manera eficiente. A continuación, te explico cómo se construye un modelo y te doy un ejemplo.

### Estructura de un Modelo en PyTorch

1. **Definición de la Clase del Modelo:**
   - Se crea una clase que hereda de `torch.nn.Module`.
   - En el constructor `__init__`, se definen las capas de la red neuronal (como `Linear`, `Conv2d`, etc.).
   - En el método `forward`, se definen las operaciones que se aplican a las entradas a través de las capas.

2. **Método `forward`:**
   - Este método toma la entrada y la pasa a través de las diferentes capas de la red en el orden que se definieron en `__init__`.

3. **Instanciación y Uso:**
   - Una vez definido el modelo, se puede instanciar y utilizar para realizar predicciones o entrenarlo con datos.

### Ejemplo de un Modelo Simple

Aquí tienes un ejemplo de un modelo de red neuronal simple con una capa oculta:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Definición del modelo
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # Definir las capas
        self.fc1 = nn.Linear(input_size, hidden_size)  # Capa oculta
        self.fc2 = nn.Linear(hidden_size, output_size)  # Capa de salida
        self.relu = nn.ReLU()  # Función de activación ReLU

    def forward(self, x):
        # Pasar la entrada a través de las capas
        x = self.fc1(x)  # Capa oculta
        x = self.relu(x)  # Activación
        x = self.fc2(x)  # Capa de salida
        return x

# Parámetros del modelo
input_size = 10  # Número de características de entrada
hidden_size = 5  # Número de neuronas en la capa oculta
output_size = 1  # Número de neuronas en la capa de salida (por ejemplo, para regresión)

# Crear una instancia del modelo
model = SimpleNN(input_size, hidden_size, output_size)

# Ver la arquitectura del modelo
print(model)

# Ejemplo de uso
# Crear un tensor de entrada aleatorio
input_tensor = torch.randn(1, input_size)  # Tamaño del batch = 1
output_tensor = model(input_tensor)  # Pasar la entrada a través del modelo
print("Output:", output_tensor)
```

### Explicación del Ejemplo

1. **Definición de la Clase `SimpleNN`:**
   - La clase `SimpleNN` hereda de `nn.Module`.
   - Se definen dos capas lineales (`fc1` y `fc2`) y una función de activación ReLU.

2. **Método `forward`:**
   - En este método, la entrada se pasa primero a través de la capa oculta y luego se aplica la activación, seguida de la capa de salida.

3. **Instanciación y Uso:**
   - Se crea una instancia del modelo y se imprime su arquitectura.
   - Un tensor de entrada aleatorio se crea y se pasa a través del modelo para obtener la salida.

### Compilación y Entrenamiento
Para entrenar el modelo, deberás definir una función de pérdida y un optimizador, y luego ejecutar un bucle de entrenamiento que actualice los pesos del modelo en función de los datos.

**Lecturas recomendadas**

[Parameter — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

[torch.nn — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/nn.html)

[Module — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

## Entrenamiento, funciones de pérdida y optimizadores

En PyTorch, el proceso de entrenamiento de un modelo implica definir una **función de pérdida** (loss function) y un **optimizador**, que se encargan de actualizar los pesos del modelo en función del error entre las predicciones del modelo y los valores reales. Aquí te explico cada uno de estos elementos y te doy un ejemplo práctico.

### Entrenamiento en PyTorch: Conceptos Clave

1. **Función de Pérdida (Loss Function):**
   La función de pérdida mide la diferencia entre las predicciones del modelo y los valores verdaderos. Su valor se minimiza durante el entrenamiento. Algunas funciones comunes son:
   - `nn.MSELoss`: Para problemas de regresión (Minimiza el error cuadrático medio).
   - `nn.CrossEntropyLoss`: Para problemas de clasificación múltiple.
   - `nn.BCELoss`: Para problemas de clasificación binaria.

2. **Optimizador:**
   El optimizador es el algoritmo que ajusta los pesos del modelo para reducir la función de pérdida. Un optimizador popular es **Stochastic Gradient Descent (SGD)**, pero PyTorch también ofrece otros optimizadores como **Adam**.
   - `torch.optim.SGD`: Descenso de gradiente estocástico.
   - `torch.optim.Adam`: Un optimizador más avanzado, que a menudo funciona mejor en redes más complejas.

3. **Ciclo de Entrenamiento:**
   - **Paso 1:** Pasar los datos de entrada a través del modelo.
   - **Paso 2:** Calcular la pérdida entre las predicciones y los valores reales.
   - **Paso 3:** Retropropagar el error (backpropagation).
   - **Paso 4:** Actualizar los pesos utilizando el optimizador.

### Ejemplo Completo

Este es un ejemplo de entrenamiento de un modelo simple con una función de pérdida y un optimizador.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Definición del modelo (similar al ejemplo anterior)
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Capa oculta
        self.fc2 = nn.Linear(hidden_size, output_size)  # Capa de salida
        self.relu = nn.ReLU()  # Función de activación ReLU

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Parámetros del modelo
input_size = 10
hidden_size = 5
output_size = 1
learning_rate = 0.01

# Crear una instancia del modelo
model = SimpleNN(input_size, hidden_size, output_size)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()  # Pérdida para regresión
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Optimizador SGD

# Datos ficticios para entrenamiento
inputs = torch.randn(100, input_size)  # 100 ejemplos, 10 características
targets = torch.randn(100, output_size)  # 100 etiquetas de salida

# Entrenamiento del modelo
num_epochs = 100

for epoch in range(num_epochs):
    # Paso hacia adelante (Forward pass): calcular predicciones
    outputs = model(inputs)
    
    # Calcular la pérdida
    loss = criterion(outputs, targets)
    
    # Paso hacia atrás (Backward pass): retropropagación
    optimizer.zero_grad()  # Resetear los gradientes
    loss.backward()  # Calcular los gradientes
    
    # Actualizar los pesos
    optimizer.step()  # Actualizar los parámetros del modelo
    
    # Imprimir la pérdida en intervalos
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predicción con el modelo entrenado
test_input = torch.randn(1, input_size)
predicted_output = model(test_input)
print(f'Predicted Output: {predicted_output}')
```

### Explicación Paso a Paso

1. **Modelo:** Usamos una red neuronal simple con una capa oculta y una función de activación ReLU.
2. **Función de pérdida (`criterion`):** Se utiliza `MSELoss` (Error Cuadrático Medio) para este ejemplo de regresión.
3. **Optimizador (`optimizer`):** Se usa SGD (Stochastic Gradient Descent) con una tasa de aprendizaje de 0.01.
4. **Datos de Entrenamiento:** Generamos datos ficticios aleatorios para los ejemplos y etiquetas.
5. **Ciclo de Entrenamiento:**
   - Se calculan las predicciones del modelo.
   - Se calcula la pérdida comparando las predicciones con las etiquetas reales.
   - Se retropropaga el error para calcular los gradientes.
   - Se actualizan los pesos del modelo con el optimizador.
6. **Impresión del Progreso:** Se imprime la pérdida cada 10 épocas.

### Otras Funciones de Pérdida y Optimizadores

#### Funciones de Pérdida Comunes:
- `nn.CrossEntropyLoss`: Para clasificación múltiple.
- `nn.BCELoss`: Para clasificación binaria.

#### Optimizadores Comunes:
- `torch.optim.Adam`: Un optimizador avanzado que usa tasas de aprendizaje adaptativas.
- `torch.optim.RMSprop`: Similar a Adam, pero con algunas diferencias en cómo actualiza los parámetros.

Este es el flujo básico del entrenamiento de modelos en PyTorch. ¿Te gustaría ver más ejemplos o detalles sobre algún otro optimizador o tipo de función de pérdida?

**Lecturas recomendadas**

[torch.optim — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/optim.html#algorithms)

[torch.nn — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)

## Entrenamiento y visualización de pérdida

El proceso de entrenamiento de un modelo y la visualización de la pérdida son esenciales para verificar si el modelo está aprendiendo correctamente. A continuación te muestro cómo puedes entrenar el modelo y luego visualizar la pérdida de entrenamiento y prueba con una gráfica, usando `matplotlib`.

### Entrenamiento del modelo

El proceso de entrenamiento ya lo hemos ajustado en el ejemplo anterior, pero ahora vamos a mejorar el código para que podamos visualizar las pérdidas en forma de gráfica.

### Código para entrenamiento con visualización de la pérdida

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Asegura la reproducibilidad
torch.manual_seed(42)

# Establezca cuántas veces el modelo pasará por los datos de entrenamiento
epocas = 100

# Listas para realizar un seguimiento de la pérdida durante el entrenamiento y la prueba
entrenamiento_loss = []
test_loss = []

for epoca in range(epocas):
    ### Entrenamiento

    # Pon el modelo en modo entrenamiento
    model_1.train()

    # 1. Pase hacia adelante los datos usando el método forward()
    y_predc = model_1(X_prueba)

    # 2. Calcula la pérdida (Cuán diferentes son las predicciones de nuestros modelos)
    perdida = fn_perd(y_predc, y_entrada)

    # 3. Gradiente cero del optimizador
    optimizador.zero_grad()

    # 4. Pérdida al revés (backward)
    perdida.backward()

    # 5. Progreso del optimizador
    optimizador.step()

    # Agregar la pérdida de entrenamiento a la lista
    entrenamiento_loss.append(perdida.item())

    ### Evaluación (sin cálculo de gradientes)
    model_1.eval()
    with torch.no_grad():
        # 1. Reenviar datos de prueba
        prueba_predc = model_1(X_prueba)

        # 2. Calcular la pérdida en datos de prueba
        prueba_perdida = fn_perd(prueba_predc, y_prueba.type(torch.float))

        # Agregar la pérdida de prueba a la lista
        test_loss.append(prueba_perdida.item())

    # Imprimir cada 10 épocas para monitorear el progreso
    if (epoca+1) % 10 == 0:
        print(f'Epoca [{epoca+1}/{epocas}], Pérdida entrenamiento: {perdida.item():.4f}, Pérdida prueba: {prueba_perdida.item():.4f}')

### Visualización de la pérdida
plt.plot(entrenamiento_loss, label="Pérdida entrenamiento")
plt.plot(test_loss, label="Pérdida prueba")
plt.title("Pérdida durante el entrenamiento y la prueba")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()
```

### Explicación:

1. **Entrenamiento:** Cada época realiza un pase hacia adelante con los datos de entrenamiento, calcula la pérdida, retropropaga los gradientes y actualiza los parámetros del modelo.

2. **Evaluación:** En el modo de evaluación, calculamos la pérdida en el conjunto de prueba, pero sin retropropagación ni actualización de gradientes (esto ahorra memoria y tiempo).

3. **Almacenamiento de pérdidas:** Al final de cada época, almacenamos la pérdida tanto del entrenamiento como de la prueba en sus respectivas listas: `entrenamiento_loss` y `test_loss`.

4. **Visualización:** Utilizamos `matplotlib` para visualizar cómo las pérdidas disminuyen durante las épocas, lo que te permitirá ver si el modelo está aprendiendo correctamente o si está ocurriendo algún problema como **overfitting** (cuando la pérdida de entrenamiento es baja, pero la de prueba no mejora).

### Interpretación de la gráfica:
- Si ambas curvas (entrenamiento y prueba) descienden a lo largo del tiempo, el modelo está aprendiendo correctamente.
- Si la curva de entrenamiento disminuye, pero la de prueba comienza a estancarse o aumentar, esto podría ser señal de **overfitting**.

## Predicción con un modelo de PyTorch entrenado

Una vez que has entrenado un modelo en PyTorch, puedes utilizarlo para realizar predicciones sobre nuevos datos o sobre datos de prueba. Para ello, es fundamental cambiar el modelo a modo de evaluación utilizando `model.eval()` y asegurarte de que no se están calculando los gradientes con `torch.no_grad()`, ya que durante la predicción no es necesario el retropropagación.

Aquí te muestro cómo hacer predicciones con un modelo de PyTorch ya entrenado:

### Ejemplo de código para hacer predicciones

```python
import torch

# Pon el modelo en modo de evaluación
model_1.eval()

# Datos de entrada para la predicción (puede ser cualquier tensor nuevo o de prueba)
# Asegúrate de que X_nuevos_datos tiene la misma estructura que los datos de entrenamiento
X_nuevos_datos = torch.tensor([[0.5, 0.8], [0.3, 0.9]])  # Ejemplo de nuevos datos (cambia según tu caso)

# Realizar la predicción sin calcular gradientes
with torch.no_grad():
    # Pase hacia adelante para hacer predicciones
    predicciones = model_1(X_nuevos_datos)

# Si las predicciones son logits (para clasificación), puedes convertirlas a probabilidades
# Por ejemplo, si la última capa de tu modelo no tiene una función softmax, puedes aplicar una:
probs = torch.softmax(predicciones, dim=1)

# O si usas una regresión, puedes imprimir directamente los valores predichos
print("Predicciones:")
print(predicciones)

# Si es un problema de clasificación, puedes obtener la clase predicha
clases_predichas = torch.argmax(probs, dim=1)
print("Clases predichas:", clases_predichas)
```

### Explicación del código:

1. **`model.eval()`**: Cambiamos el modelo al modo de evaluación. Esto asegura que ciertas capas como `Dropout` o `BatchNorm` se comporten de manera adecuada durante la predicción.

2. **Datos de entrada**: Usamos `X_nuevos_datos`, que debe tener el mismo formato y dimensiones que los datos de entrenamiento. Si usas normalización o preprocesamiento durante el entrenamiento, asegúrate de aplicarlo también a los datos de predicción.

3. **`torch.no_grad()`**: Esto desactiva el cálculo de gradientes, lo que hace que la predicción sea más rápida y eficiente en términos de memoria.

4. **Predicción**: Usamos el modelo para hacer una predicción (`model_1(X_nuevos_datos)`). Si es un modelo de clasificación, los resultados pueden estar en forma de **logits**, por lo que podemos aplicar `torch.softmax()` para convertirlos en probabilidades.

5. **Clases predichas** (si es clasificación): Si el modelo es de clasificación, `torch.argmax()` puede ayudarte a obtener la clase predicha para cada muestra.

### Consideraciones adicionales:

- **Clasificación**: En un problema de clasificación, el modelo típicamente devuelve logits (valores sin procesar antes de la función de activación final), y se pueden convertir a probabilidades con `softmax`. Luego, las clases predichas se obtienen usando `torch.argmax()`.

- **Regresión**: En problemas de regresión, la salida será el valor predicho directamente, por lo que no necesitas aplicar `softmax`.

- **Preprocesamiento**: Asegúrate de aplicar el mismo preprocesamiento a los datos de entrada de predicción que aplicaste durante el entrenamiento (como la normalización de características).

## Datos para clasificación de texto

Para realizar una clasificación de texto en machine learning, el proceso comienza con la preparación de datos, que involucra varios pasos, como la recolección de los datos de texto, su preprocesamiento, la representación en un formato adecuado para modelos (normalmente como vectores numéricos) y la división de los datos en conjuntos de entrenamiento y prueba. Aquí te doy un resumen de los pasos clave:

### 1. **Recolección de datos:**
   - Datos etiquetados son fundamentales para tareas de clasificación. Algunos datasets populares para la clasificación de texto incluyen:
     - **IMDB Reviews** (clasificación de sentimientos).
     - **20 Newsgroups** (clasificación de noticias).
     - **SpamAssassin** (detección de spam).

### 2. **Preprocesamiento de texto:**
   - **Limpieza**: Eliminar puntuación, dígitos, URLs, y otros elementos irrelevantes.
   - **Tokenización**: Separar el texto en palabras o tokens (puede ser palabra o n-gramas).
   - **Lematización/Stemming**: Reducir palabras a su forma base o raíz.
   - **Stop words removal**: Eliminar palabras comunes (como "el", "de", "la" en español) que no aportan mucho a la clasificación.

### 3. **Representación de texto (Features):**
   El texto necesita ser convertido a un formato numérico que el modelo pueda procesar. Algunas de las técnicas más utilizadas son:
   - **Bag of Words (BoW)**: Representación de los textos en forma de conteos de palabras.
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Ajusta los conteos de palabras en función de su frecuencia en otros documentos, ponderando la importancia.
   - **Word embeddings**: Representar las palabras como vectores de una manera que capture el contexto, como en **Word2Vec** o **GloVe**.
   - **Modelos preentrenados** como **BERT** o **GPT** que generan representaciones numéricas sofisticadas.

### 4. **División de datos:**
   Dividir los datos en conjuntos de:
   - **Entrenamiento**: Aproximadamente el 80% de los datos se usa para entrenar el modelo.
   - **Prueba**: El otro 20% se usa para evaluar el modelo.
   - **Validación (opcional)**: A veces se reserva un 10% de los datos de entrenamiento para ajuste de hiperparámetros.

### 5. **Entrenamiento y evaluación:**
   - **Entrenamiento**: Aplicar el modelo de clasificación de texto (como Naive Bayes, SVM, Redes Neuronales, etc.).
   - **Evaluación**: Usar métricas como la **precisión**, **recall**, **f1-score** y la **matriz de confusión** para medir el desempeño del modelo.

### Ejemplo en PyTorch
Si estás usando PyTorch, podrías emplear una red neuronal o un modelo más simple con embeddings preentrenados para resolver la clasificación.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Supongamos que tienes tus datos listos en X (textos) y y (etiquetas)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aquí puedes usar un modelo de embeddings preentrenado o construir una red simple
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        out = self.fc(embeds.mean(1))  # Agregar pooling si es necesario
        return out

# Crear el modelo, función de pérdida y optimizador
model = TextClassifier(vocab_size=5000, embed_dim=64, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento simplificado
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)  # Asegúrate que X_train esté tokenizado y vectorizado
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

**Lecturas recomendadas**

[torchtext.datasets — Torchtext 0.15.0 documentation](https://pytorch.org/text/stable/datasets.html#dbpedia)

[torchtext.datasets — Torchtext 0.15.0 documentation](https://pytorch.org/text/stable/datasets.html)

## Procesamiento de datos: tokenización y creación de vocabulario

El procesamiento de datos, especialmente en tareas de procesamiento de lenguaje natural (NLP), implica pasos como la tokenización y la creación de vocabularios. Aquí te explico ambos conceptos y cómo se implementan, particularmente en el contexto de PyTorch.

### 1. Tokenización

La **tokenización** es el proceso de dividir un texto en unidades más pequeñas llamadas "tokens". Estos pueden ser palabras, subpalabras o caracteres. La tokenización permite que los modelos entiendan el texto en un formato que pueden procesar.

Existen diferentes enfoques de tokenización:

- **Tokenización por palabras**: Divide el texto en palabras.
- **Tokenización por subpalabras**: Utiliza algoritmos como Byte Pair Encoding (BPE) para dividir palabras en subunidades, lo que es útil para manejar palabras desconocidas y reducir el vocabulario.
- **Tokenización por caracteres**: Cada carácter se convierte en un token, lo que puede ser útil para ciertos tipos de modelos.

### Ejemplo de Tokenización en PyTorch

Aquí hay un ejemplo básico de cómo realizar la tokenización utilizando `torchtext`:

```python
import torch
from torchtext.data.utils import get_tokenizer

# Texto de ejemplo
text = "Hola, esto es un ejemplo de tokenización."

# Crear un tokenizador
tokenizer = get_tokenizer("basic_english")

# Tokenizar el texto
tokens = tokenizer(text)
print(tokens)
```

### 2. Creación de Vocabulario

La **creación de vocabulario** implica construir un conjunto de todos los tokens únicos que aparecen en tu conjunto de datos. Esto es fundamental porque el modelo necesita mapear cada token a un número entero (índice) que puede utilizar durante el entrenamiento.

Los vocabularios pueden ser simples o pueden incluir mapeos adicionales, como las frecuencias de palabras.

### Ejemplo de Creación de Vocabulario en PyTorch

A continuación, se muestra un ejemplo de cómo crear un vocabulario a partir de los tokens:

```python
from collections import Counter
from torchtext.vocab import Vocab

# Contar la frecuencia de los tokens
counter = Counter(tokens)

# Crear el vocabulario
vocab = Vocab(counter)

# Ver el vocabulario
print(vocab.stoi)  # Muestra el índice de cada token
print(vocab.itos)  # Muestra el token correspondiente a cada índice
```

### Integración en el Entrenamiento del Modelo

Una vez que tienes los tokens y el vocabulario, puedes convertir tus textos en secuencias de índices y alimentar estos índices a tu modelo de PyTorch para el entrenamiento.

### Ejemplo Completo

Aquí hay un flujo de trabajo más completo que incluye la tokenización y la creación de vocabulario:

```python
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

# Texto de ejemplo
corpus = [
    "Hola, esto es un ejemplo de tokenización.",
    "Este es otro ejemplo para crear vocabulario."
]

# Crear un tokenizador
tokenizer = get_tokenizer("basic_english")

# Tokenizar y contar la frecuencia de tokens
tokens = []
for line in corpus:
    tokens.extend(tokenizer(line))

# Crear el vocabulario
counter = Counter(tokens)
vocab = Vocab(counter)

# Convertir texto a índices
text_indices = [[vocab[token] for token in tokenizer(line)] for line in corpus]

# Mostrar los resultados
print("Vocabulario:", vocab.stoi)
print("Índices del texto:", text_indices)
```

### Conclusión

La tokenización y la creación de vocabulario son pasos críticos en el procesamiento de datos para modelos de NLP. Usar bibliotecas como `torchtext` simplifica mucho estos procesos, permitiendo concentrarse en el diseño y entrenamiento de modelos en lugar de preocuparse por el preprocesamiento de datos.

## Procesamiento de datos: preparación del DataLoader()

Para preparar un `DataLoader` en PyTorch, primero necesitas un conjunto de datos adecuado y luego crear un `DataLoader` que pueda iterar sobre ese conjunto de datos en mini-batches. El `DataLoader` es una herramienta muy útil que permite manejar la carga de datos, la aleatorización y el agrupamiento de muestras.

Aquí te muestro cómo puedes hacerlo utilizando un conjunto de datos de texto, como el de AG News, y cómo crear un `DataLoader`:

### Paso 1: Importar las librerías necesarias
```python
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
```

### Paso 2: Cargar el conjunto de datos
```python
# Cargar el conjunto de datos AG News
train_iter = AG_NEWS(split='train')
```

### Paso 3: Crear un tokenizador y construir el vocabulario
```python
# Crear un tokenizador
tokenizador = get_tokenizer('basic_english')

# Función para generar tokens
def yield_tokens(data_iter):
    for _, texto in data_iter:
        yield tokenizador(texto)

# Construir el vocabulario
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
```

### Paso 4: Preparar los datos para el DataLoader
Para usar el `DataLoader`, necesitas definir cómo quieres convertir cada texto en una secuencia de índices basada en el vocabulario. Esto puede implicar la conversión de textos en tensores.

```python
# Cargar de nuevo el conjunto de datos para que esté fresco
train_iter = AG_NEWS(split='train')

# Función para convertir texto en índices de vocabulario
def process_text(text):
    return torch.tensor([vocab[token] for token in tokenizador(text)], dtype=torch.int64)

# Crear una lista de tuplas (texto procesado, etiqueta)
data = [(process_text(text), label) for label, text in train_iter]
```

### Paso 5: Crear el DataLoader
```python
# Crear un DataLoader
batch_size = 16  # Puedes ajustar el tamaño del batch
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Ejemplo de iterar sobre el DataLoader
for batch in data_loader:
    texts, labels = batch
    print("Batch de textos:", texts)
    print("Batch de etiquetas:", labels)
    break  # Solo mostramos el primer batch
```

### Resumen
1. **Cargar el conjunto de datos:** Puedes utilizar cualquier conjunto de datos de texto compatible.
2. **Tokenizar y construir vocabulario:** Convierte los textos en índices que el modelo puede entender.
3. **Preparar los datos:** Asegúrate de que cada texto está representado como un tensor.
4. **Crear el DataLoader:** Esto facilita el procesamiento por lotes y la aleatorización.

Esto te permitirá gestionar fácilmente tus datos durante el entrenamiento del modelo en PyTorch.

**Lecturas recomendadas**

[torch.utils.data — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/data.html)

[torch.cumsum — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.cumsum.html)

## Creación de modelo de clasificación de texto con PyTorch

Para crear un modelo de clasificación de texto utilizando PyTorch, puedes seguir un enfoque basado en redes neuronales. Aquí te mostraré cómo construir un modelo simple de clasificación de texto utilizando una red neuronal totalmente conectada (fully connected neural network) con `torch.nn`. Este ejemplo utilizará el conjunto de datos AG News, pero puedes adaptarlo a cualquier conjunto de datos que estés utilizando.

### Paso 1: Importar las librerías necesarias
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
```

### Paso 2: Cargar y preparar el conjunto de datos
```python
# Cargar el conjunto de datos AG News
train_iter = AG_NEWS(split='train')

# Crear un tokenizador
tokenizador = get_tokenizer('basic_english')

# Función para generar tokens
def yield_tokens(data_iter):
    for _, texto in data_iter:
        yield tokenizador(texto)

# Construir el vocabulario
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Función para convertir texto en índices de vocabulario
def process_text(text):
    return torch.tensor([vocab[token] for token in tokenizador(text)], dtype=torch.int64)

# Cargar de nuevo el conjunto de datos
train_iter = AG_NEWS(split='train')

# Crear una lista de tuplas (texto procesado, etiqueta)
data = [(process_text(text), label) for label, text in train_iter]

# Crear un DataLoader
batch_size = 16  # Puedes ajustar el tamaño del batch
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
```

### Paso 3: Definir el modelo
Aquí definimos un modelo simple de red neuronal:
```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: tensor de índices
        x = self.embedding(x)  # Obtiene las representaciones de las palabras
        x = x.mean(dim=1)  # Promedia los embeddings (puedes usar otras técnicas de agregación)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
```

### Paso 4: Inicializar el modelo, la función de pérdida y el optimizador
```python
# Parámetros
vocab_size = len(vocab)
embed_dim = 64  # Dimensión de los embeddings
num_classes = 4  # Número de clases en AG News

# Inicializar el modelo
model = TextClassifier(vocab_size, embed_dim, num_classes)

# Definir la función de pérdida y el optimizador
criterion = nn.NLLLoss()  # Pérdida negativa logarítmica
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Paso 5: Entrenamiento del modelo
```python
# Definir el número de épocas
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    for texts, labels in data_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(texts)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}")
```

### Paso 6: Evaluación del modelo
Para evaluar el modelo, puedes usar un conjunto de validación o prueba. A continuación se muestra un ejemplo básico:
```python
def evaluate_model(model, data_loader):
    model.eval()  # Cambiar a modo evaluación
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# Puedes evaluar usando el mismo DataLoader de entrenamiento o uno diferente
evaluate_model(model, data_loader)
```

### Resumen
1. **Importar librerías y cargar el conjunto de datos.**
2. **Definir el modelo de clasificación.**
3. **Inicializar el modelo, la función de pérdida y el optimizador.**
4. **Entrenar el modelo.**
5. **Evaluar el modelo.**

Esto te dará una buena base para crear un modelo de clasificación de texto en PyTorch.