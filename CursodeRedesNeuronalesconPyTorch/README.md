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

