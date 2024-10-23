# Curso de Redes Neuronales Convolucionales con Python y Keras

## La importancia del computer vision

La **computer vision** (visión por computadora) es una rama de la inteligencia artificial que se ocupa de cómo las computadoras pueden ser programadas para entender y procesar imágenes y videos de manera similar a como lo hacen los humanos. Su importancia radica en varios aspectos:

1. **Automatización y Eficiencia**: La visión por computadora permite la automatización de tareas que requieren interpretación visual, como el reconocimiento de objetos, la inspección de calidad en fábricas, y la lectura de señales. Esto aumenta la eficiencia y reduce la necesidad de intervención humana.

2. **Mejora en la Toma de Decisiones**: En campos como la medicina, la visión por computadora puede analizar imágenes médicas (como radiografías o resonancias magnéticas) para detectar enfermedades, lo que ayuda a los médicos a tomar decisiones más informadas y rápidas.

3. **Interacción Hombre-Máquina**: La visión por computadora mejora la interacción entre humanos y máquinas, permitiendo interfaces más intuitivas, como el reconocimiento facial, el seguimiento de gestos y la realidad aumentada.

4. **Análisis de Datos Visuales**: Con el aumento del contenido visual en la web y las redes sociales, la visión por computadora permite analizar grandes volúmenes de datos visuales para entender tendencias, comportamientos del usuario y patrones.

5. **Seguridad y Vigilancia**: La visión por computadora se utiliza en sistemas de seguridad y vigilancia para reconocer caras, detectar comportamientos sospechosos y analizar videos en tiempo real.

6. **Innovaciones en Transporte**: En el ámbito de los vehículos autónomos, la visión por computadora es fundamental para interpretar el entorno, reconocer señales de tráfico y evitar obstáculos.

7. **Agricultura y Medio Ambiente**: Se usa para monitorear cultivos, detectar plagas, y evaluar la salud de las plantas mediante el análisis de imágenes aéreas y satelitales.

8. **Creatividad y Entretenimiento**: En la industria del entretenimiento, la visión por computadora se aplica en la creación de efectos visuales, en videojuegos y en la mejora de la experiencia del usuario en plataformas de streaming.

9. **Desarrollo de Nuevas Tecnologías**: La visión por computadora está en el corazón de muchas tecnologías emergentes, como la inteligencia artificial en robótica, que necesita entender su entorno para operar eficazmente.

En resumen, la visión por computadora no solo tiene aplicaciones prácticas en diversas industrias, sino que también está impulsando la innovación y el desarrollo de nuevas tecnologías que transforman la forma en que interactuamos con el mundo digital y físico.

## ¿Qué herramientas usaremos para redes neuronales convolucionales?

Para trabajar con **redes neuronales convolucionales** (CNNs) en el desarrollo de proyectos de **deep learning**, existen varias herramientas y bibliotecas clave que te facilitarán el proceso de construcción, entrenamiento y evaluación de modelos. Las principales herramientas que se suelen usar para CNNs incluyen:

### 1. **PyTorch**
   - **Descripción**: PyTorch es una de las bibliotecas más populares para crear y entrenar redes neuronales. Proporciona una flexibilidad notable, lo que lo convierte en una excelente opción para trabajar con CNNs.
   - **Funciones clave**: 
     - Soporte para GPU con `torch.cuda`.
     - Módulo `torch.nn` para definir capas convolucionales (`nn.Conv2d`) y operaciones comunes como la normalización por lotes (`nn.BatchNorm2d`).
     - Herramientas para visualización de gráficos computacionales dinámicos.
     - API flexible para la creación de arquitecturas personalizadas.

### 2. **TensorFlow/Keras**
   - **Descripción**: TensorFlow, junto con su API de alto nivel Keras, es otra opción ampliamente utilizada para redes neuronales convolucionales. TensorFlow es conocido por su escalabilidad y uso en aplicaciones de producción, mientras que Keras simplifica la construcción de modelos.
   - **Funciones clave**: 
     - Definición sencilla de capas convolucionales con `Conv2D` y capas de pooling con `MaxPooling2D`.
     - Funcionalidad de visualización de gráficos con TensorBoard.
     - Compatibilidad con TPU y GPU para acelerar el entrenamiento.
     - Keras facilita la creación de modelos complejos con pocas líneas de código.

### 3. **OpenCV**
   - **Descripción**: OpenCV es una biblioteca poderosa para el procesamiento de imágenes, que complementa a las redes neuronales convolucionales en tareas de visión por computadora.
   - **Funciones clave**: 
     - Preprocesamiento de imágenes: detección de bordes, escalado, normalización, conversión de color.
     - Procesamiento de imágenes en tiempo real para tareas como la detección de objetos y seguimiento.
     - Se puede usar para aumentar los datos de entrenamiento de CNNs.

### 4. **FastAI**
   - **Descripción**: FastAI es una biblioteca construida sobre PyTorch que facilita el entrenamiento rápido de CNNs con técnicas de aprendizaje profundo de última generación.
   - **Funciones clave**: 
     - Entrenamiento simplificado de CNNs utilizando clases como `cnn_learner`.
     - Métodos avanzados como entrenamiento discriminativo, transferencia de aprendizaje y ajuste de hiperparámetros.
     - Amplia documentación y comunidad de soporte.

### 5. **Torchvision**
   - **Descripción**: Torchvision es una biblioteca complementaria de PyTorch diseñada para el manejo de datos de visión por computadora, como imágenes y videos. Facilita la carga, transformación y uso de datos para el entrenamiento de CNNs.
   - **Funciones clave**: 
     - Carga de conjuntos de datos populares como CIFAR-10, MNIST y ImageNet.
     - Transformaciones de datos (escalado, rotación, normalización).
     - Modelos preentrenados para transferencia de aprendizaje (ResNet, VGG, AlexNet, etc.).

### 6. **Matplotlib/Seaborn**
   - **Descripción**: Estas bibliotecas de visualización se utilizan para representar gráficamente el entrenamiento de CNNs y visualizar sus resultados.
   - **Funciones clave**:
     - Visualización de precisión, pérdida y otras métricas durante el entrenamiento.
     - Visualización de imágenes y mapas de características de las capas convolucionales.

### 7. **Horovod**
   - **Descripción**: Si tu proyecto implica el entrenamiento de modelos en grandes cantidades de datos y necesitas escalar tu entrenamiento a múltiples GPUs o nodos, Horovod es una excelente herramienta para el entrenamiento distribuido.
   - **Funciones clave**:
     - Permite el entrenamiento de redes neuronales convolucionales en entornos distribuidos.
     - Compatible con TensorFlow, PyTorch y Keras.

### 8. **Scikit-learn**
   - **Descripción**: Aunque Scikit-learn no se utiliza directamente para redes neuronales convolucionales, se emplea para preprocesamiento, división de datos y evaluación de modelos.
   - **Funciones clave**:
     - División de datos en conjuntos de entrenamiento y prueba.
     - Normalización y estandarización de características.
     - Evaluación de métricas de rendimiento (precisión, recall, F1).

### 9. **Albumentations**
   - **Descripción**: Esta biblioteca está orientada a la **aumentación de datos** (data augmentation) para imágenes. Es extremadamente útil para CNNs que requieren grandes cantidades de datos variados para generalizar bien.
   - **Funciones clave**:
     - Aplicación de transformaciones geométricas y de color a las imágenes de manera eficiente.
     - Complemento perfecto para Torchvision en el preprocesamiento de imágenes.

### 10. **Hugging Face**
   - **Descripción**: Hugging Face no es solo para modelos de procesamiento de lenguaje natural, también cuenta con modelos preentrenados para visión por computadora, como Vision Transformers (ViT).
   - **Funciones clave**:
     - Descarga y utilización de modelos preentrenados en visión por computadora.
     - Capacidades de transferencia de aprendizaje y evaluación de modelos avanzados.

Estas herramientas permiten crear, entrenar y desplegar redes neuronales convolucionales de manera eficiente y flexible, cubriendo tanto la parte de preprocesamiento de imágenes como el diseño de modelos avanzados y el manejo de grandes volúmenes de datos.

**Lecturas recomendadas**

[Kaggle: Your Machine Learning and Data Science Community](https://kaggle.com/)

## ¿Qué son las redes convolucionales?

Las **redes neuronales convolucionales** (o CNN, por sus siglas en inglés: **Convolutional Neural Networks**) son un tipo especial de red neuronal diseñada específicamente para procesar datos que tienen una estructura de rejilla, como imágenes. Son ampliamente utilizadas en tareas de visión por computadora, como el reconocimiento de objetos, la detección de rostros y la clasificación de imágenes.

### Características clave de las redes convolucionales:

1. **Capas convolucionales (Convolutional Layers):** 
   Estas capas son el corazón de las CNN. En lugar de conectar cada neurona con todas las entradas, como en las redes completamente conectadas, las capas convolucionales aplican un **filtro** (o **kernel**) que se desplaza sobre la imagen (o cualquier entrada estructurada), detectando patrones locales como bordes, texturas o formas específicas.

   - El resultado de aplicar el filtro se llama **mapa de características** o **feature map**.
   - Los filtros son **aprendibles**, lo que significa que durante el entrenamiento, la red ajusta estos filtros para captar características importantes de la imagen.

2. **Pooling (Submuestreo):**
   Después de las capas convolucionales, suele aplicarse una capa de **pooling**, que reduce la dimensionalidad de los mapas de características. El tipo más común es el **max-pooling**, que selecciona el valor máximo de una región del mapa de características, ayudando a reducir el tamaño de la imagen y a mantener las características más importantes.

3. **ReLU (Rectified Linear Unit):**
   Es una función de activación que introduce no linealidades en la red. En las CNN, la función **ReLU** (Rectified Linear Unit) se usa después de la operación de convolución para reemplazar todos los valores negativos por cero, ayudando a acelerar el proceso de aprendizaje.

4. **Capas completamente conectadas (Fully Connected Layers):**
   Después de varias capas convolucionales y de pooling, las características extraídas se pasan a una o más capas completamente conectadas. Estas capas son similares a las utilizadas en las redes neuronales tradicionales y combinan todas las características extraídas para realizar la tarea final, como la clasificación.

5. **Aprendizaje de características jerárquicas:**
   Las CNN tienen la capacidad de aprender características jerárquicas. Las primeras capas aprenden a detectar bordes simples, las capas intermedias capturan formas más complejas y, finalmente, las últimas capas capturan representaciones abstractas de alto nivel de los objetos.

### Ejemplos de aplicaciones de redes convolucionales:
- **Clasificación de imágenes:** Reconocimiento de objetos en imágenes, como en el famoso dataset ImageNet.
- **Detección de objetos:** Identificación y localización de objetos en una imagen.
- **Segmentación de imágenes:** Dividir una imagen en regiones basadas en diferentes clases de objetos.
- **Reconocimiento facial:** Identificación y verificación de rostros en imágenes o videos.
- **Diagnóstico médico:** Análisis de imágenes médicas como rayos X, resonancias magnéticas o tomografías.

### Ventajas:
- **Capacidad de detección automática de características:** Las CNN no requieren que las características sean diseñadas manualmente, ya que aprenden directamente a partir de los datos de entrada.
- **Reducción de parámetros:** Gracias al uso de filtros compartidos y la reducción de dimensionalidad a través de pooling, las CNN son más eficientes que las redes totalmente conectadas cuando se trata de procesar datos como imágenes.

Las CNN han revolucionado la visión por computadora y han permitido grandes avances en el reconocimiento y procesamiento de imágenes.

Lecturas recomendadas

[https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect8.pdf](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect8.pdf)

## Creando nuestra primera red convolucional

Crear tu primera **red convolucional** (CNN) en PyTorch es un gran paso para aprender sobre el uso de las CNN en la visión por computadora. A continuación te mostraré cómo se puede construir una CNN simple para la clasificación de imágenes usando PyTorch.

Este ejemplo usará el conjunto de datos **CIFAR-10**, que es un conjunto estándar para la clasificación de imágenes en 10 categorías (aviones, autos, gatos, etc.).

### 1. Preparar el entorno
Primero, asegúrate de tener las bibliotecas necesarias instaladas. Si no las tienes, puedes instalarlas utilizando `pip`:

```bash
pip install torch torchvision
```

### 2. Importar las bibliotecas necesarias

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

### 3. Cargar el conjunto de datos (CIFAR-10)

Aquí usamos `torchvision` para cargar el conjunto de datos CIFAR-10, que se dividirá en conjuntos de entrenamiento y prueba, y se aplicarán algunas transformaciones como la normalización y el redimensionamiento.

```python
# Definir transformaciones para normalizar y convertir a tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Descargar el conjunto de datos CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Cargar los datos en mini-lotes
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 4. Definir la arquitectura de la red convolucional

A continuación, definimos nuestra primera red convolucional simple, que incluye varias capas convolucionales, capas de activación ReLU, max pooling y capas completamente conectadas.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Primera capa convolucional: Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # CIFAR-10 images are 32x32, so after pooling, they're 8x8
        self.fc2 = nn.Linear(512, 10)  # 10 output classes (for CIFAR-10)
    
    def forward(self, x):
        # Aplicar la primera capa convolucional seguida de ReLU y max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Segunda capa convolucional seguida de ReLU y max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Aplanar los datos para conectarlos a las capas completamente conectadas
        x = x.view(-1, 64 * 8 * 8)
        
        # Capa completamente conectada seguida de ReLU
        x = F.relu(self.fc1(x))
        
        # Capa de salida (no necesitamos una activación aquí, ya que usaremos CrossEntropyLoss más adelante)
        x = self.fc2(x)
        return x
```

### 5. Definir el criterio de pérdida y el optimizador

Usaremos la **pérdida de entropía cruzada** (`CrossEntropyLoss`) para la clasificación, y el optimizador **Adam**, que es eficiente para entrenar redes neuronales.

```python
model = SimpleCNN()

# Criterio de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 6. Entrenar el modelo

Ahora, escribimos el bucle de entrenamiento para optimizar el modelo sobre el conjunto de datos de entrenamiento.

```python
# Entrenamiento
num_epochs = 10  # Número de épocas de entrenamiento

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### 7. Evaluar el modelo

Después de entrenar, evaluamos el rendimiento del modelo en el conjunto de datos de prueba.

```python
# Evaluación
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
```

### 8. Ejecutar todo el proceso

Este código cargará el conjunto de datos CIFAR-10, entrenará una red convolucional simple durante 10 épocas y calculará la precisión del modelo en los datos de prueba. Con esto, ya tienes una CNN básica implementada en PyTorch.

### Posibles mejoras:
- **Aumentar la profundidad de la red**: Se pueden agregar más capas convolucionales.
- **Ajustar hiperparámetros**: Modificar el tamaño del lote, el número de épocas, o usar técnicas de regularización como el dropout.
- **Entrenar en GPU**: Si tienes acceso a una GPU, puedes mover el modelo y los datos a la GPU usando `.to('cuda')`.

Este es solo el comienzo, y las redes convolucionales tienen muchas aplicaciones avanzadas que puedes explorar conforme te familiarices más con ellas.

**Nota**: sale un error y cambien uan parte del codigo co ChatGPT

```python
el codigo enterior me arroja este error
/opt/conda/lib/python3.10/site-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an input_shape/input_dim argument to a layer. When using Sequential models, prefer using an Input(shape) object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
```

El error que estás viendo es una advertencia que indica que es preferible usar un objeto `Input(shape)` en lugar de pasar el argumento `input_shape` directamente en la primera capa de tu modelo secuencial.

Para corregirlo, puedes cambiar la primera capa `Conv2D` utilizando `tf.keras.layers.Input` de la siguiente manera:

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

# Definir el modelo secuencial
model = tf.keras.Sequential()

# Añadir la capa de entrada explícita
model.add(Input(shape=(28, 28, 1)))  # Definir explícitamente la capa de entrada

# Añadir las capas convolucionales y densas
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Resumen del modelo
model.summary()
```

Este cambio asegura que la capa de entrada se defina explícitamente antes de las capas convolucionales, eliminando la advertencia. 

**Lecturas recomendadas**

[Fashion MNIST dataset, an alternative to MNIST](https://keras.io/api/datasets/fashion_mnist/v)

[Mi primera red neuronal convolucional | Kaggle](https://www.kaggle.com/alarcon7a/mi-primera-red-neuronal-convolucional)

## Entrenando nuestra primera red convolucional

¡Claro! Vamos a crear un ejemplo de entrenamiento de una red neuronal convolucional (CNN) utilizando TensorFlow y el conjunto de datos Fashion MNIST. Este conjunto de datos contiene imágenes de ropa en 10 categorías diferentes. Te proporcionaré el código y una explicación detallada del proceso.

### Ejemplo de Código

Aquí tienes un ejemplo de cómo construir y entrenar una CNN en TensorFlow:

```python
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocesar las imágenes
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0  # Normalizar y redimensionar
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0  # Normalizar y redimensionar

# Convertir etiquetas a formato categórico
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Definir el modelo
model = Sequential()
model.add(Input(shape=(28, 28, 1)))  # Capa de entrada
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy:.4f}')

# Graficar la pérdida y precisión durante el entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Épocas')
plt.legend()
plt.show()
```

### Explicación del Código

1. **Carga de Datos**:
   - Utilizamos `fashion_mnist.load_data()` para cargar el conjunto de datos. Contiene imágenes de ropa, divididas en un conjunto de entrenamiento (60,000 imágenes) y un conjunto de prueba (10,000 imágenes).

2. **Preprocesamiento**:
   - Redimensionamos las imágenes a un formato adecuado para la red convolucional (28x28x1), ya que son imágenes en escala de grises.
   - Normalizamos los valores de los píxeles dividiendo por 255.0 para que estén en el rango [0, 1].
   - Convertimos las etiquetas en formato categórico utilizando `to_categorical()`, para que el modelo pueda interpretar las clases de manera adecuada.

3. **Definición del Modelo**:
   - Usamos `Sequential()` para construir el modelo de manera lineal.
   - **Capas Convolucionales** (`Conv2D`):
     - La primera capa convolucional tiene 64 filtros de tamaño 3x3 y usa la activación ReLU.
     - La segunda capa tiene 32 filtros, también de tamaño 3x3.
   - **Max Pooling** (`MaxPooling2D`): Se utiliza para reducir las dimensiones de la imagen, preservando características importantes.
   - **Dropout**: Se agrega para prevenir el sobreajuste al desactivar un porcentaje de las neuronas durante el entrenamiento.
   - **Capa densa** (`Dense`): Se utiliza para la clasificación final. La última capa tiene 10 neuronas (una por cada categoría) con activación softmax.

4. **Compilación del Modelo**:
   - Usamos el optimizador Adam y la función de pérdida de entropía cruzada categórica. La métrica de precisión se utiliza para evaluar el rendimiento.

5. **Entrenamiento**:
   - Entrenamos el modelo durante 10 épocas con un tamaño de lote de 64. Usamos una fracción del conjunto de entrenamiento (20%) para validación.

6. **Evaluación**:
   - Evaluamos el modelo en el conjunto de prueba y mostramos la precisión.

7. **Visualización**:
   - Graficamos la precisión de entrenamiento y validación a lo largo de las épocas para observar el rendimiento del modelo.

### Conclusión

Este código proporciona un punto de partida para trabajar con redes neuronales convolucionales en tareas de clasificación de imágenes. Puedes experimentar con diferentes arquitecturas, parámetros y técnicas de regularización para mejorar el rendimiento del modelo en el conjunto de datos de Fashion MNIST.

## Consejos para el manejo de imágenes

- Una imagen es una composición de pixeles.

- Los pixeles poseen un rango número que va del 0 al 255, siendo 0 el más oscuro y 255 es más claro.

- Las maquinas lo que realmente interpreta es una composición matricial la cual representan las instrucciones de composición de un color.

**Recomendaciones al trabajar con imágenes**

- No todas las imágenes manejan escala de grises.

- La mayoría de las imágenes manejan escala de colores que les añade una complejidad extra que se representa con la composición de 3 canales que , además, añade tiempo de cómputo: del inglés “red”, “green”, “blue” siendo la base de la combinación RGB rojo, verde y azul.

- Trabaja las imágenes primero y de ser posible (siempre y cuando el color no importe) en escala de grises para así ahorrar tiempo de cómputo.

Si por ejemplo, trabajas en un algoritmo que se encarga de distinguir y clasificar ciertas partes de una carretera, pues en este escenario el color tiene una relevancia para optimizar o no importe en el proceso.

Siempre manejar una escala de dimensiones definidas para tus imágenes. Con esto, se busca que el algoritmo pre-entrenado no entre en conflicto al momento de recibir canales extras y no puede llegar a comprenderlos.

## Manejo de imágenes con Python

El manejo de imágenes en Python es fundamental en muchas aplicaciones, especialmente en el aprendizaje automático y la visión por computadora. A continuación, te presento una explicación teórica y un ejemplo práctico de cómo trabajar con imágenes en Python utilizando bibliotecas como **PIL** (Pillow) y **OpenCV**.

### Teoría

#### 1. **Carga de Imágenes**
   - Las imágenes se pueden cargar desde archivos utilizando bibliotecas como Pillow o OpenCV. Al cargar una imagen, se convierte en un objeto que puede ser manipulado.

#### 2. **Manipulación de Imágenes**
   - **Redimensionamiento**: Cambiar el tamaño de la imagen a dimensiones específicas.
   - **Recorte**: Extraer una región específica de la imagen.
   - **Rotación**: Girar la imagen a un ángulo específico.
   - **Cambio de Formato**: Convertir imágenes entre diferentes formatos (por ejemplo, JPG a PNG).

#### 3. **Transformaciones**
   - Las transformaciones como el escalado, la inversión de colores, y la aplicación de filtros son comunes en el preprocesamiento de imágenes.

#### 4. **Visualización**
   - Usar bibliotecas como Matplotlib para mostrar imágenes.

#### 5. **Guardado de Imágenes**
   - Se pueden guardar imágenes manipuladas en el disco en diferentes formatos.

### Ejemplo Práctico

A continuación se muestra un ejemplo utilizando Pillow y Matplotlib para cargar, manipular y mostrar imágenes.

#### Instalación de Bibliotecas

Asegúrate de tener instaladas las bibliotecas necesarias. Puedes instalarlas utilizando pip:

```bash
pip install pillow matplotlib
```

#### Código Ejemplo

```python
from PIL import Image
import matplotlib.pyplot as plt

# Cargar una imagen
image_path = 'ruta/a/tu/imagen.jpg'  # Cambia esta ruta a tu imagen
image = Image.open(image_path)

# Mostrar la imagen original
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Imagen Original")
plt.axis('off')  # Oculta los ejes
plt.show()

# Redimensionar la imagen
resized_image = image.resize((100, 100))  # Cambiar a 100x100 píxeles

# Recortar la imagen (izquierda, arriba, derecha, abajo)
cropped_image = image.crop((50, 50, 200, 200))  # Recorta la región deseada

# Rotar la imagen
rotated_image = image.rotate(45)  # Rota 45 grados

# Mostrar las imágenes manipuladas
plt.figure(figsize=(12, 12))

plt.subplot(1, 3, 1)
plt.imshow(resized_image)
plt.title("Imagen Redimensionada")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cropped_image)
plt.title("Imagen Recortada")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(rotated_image)
plt.title("Imagen Rotada")
plt.axis('off')

plt.show()

# Guardar la imagen redimensionada
resized_image.save('imagen_redimensionada.jpg')
```

### Explicación del Código

1. **Carga de Imagen**: Utilizamos `Image.open()` para cargar la imagen desde el disco.
2. **Visualización**: Usamos `matplotlib.pyplot` para mostrar la imagen original.
3. **Redimensionamiento**: Utilizamos el método `resize()` para cambiar el tamaño de la imagen.
4. **Recorte**: Utilizamos el método `crop()` para extraer una región específica de la imagen.
5. **Rotación**: Utilizamos el método `rotate()` para girar la imagen.
6. **Guardar Imagen**: Finalmente, guardamos la imagen redimensionada utilizando el método `save()`.

### Conclusión

Este ejemplo te da una introducción a cómo manejar imágenes en Python utilizando Pillow y Matplotlib. Puedes explorar más funcionalidades en la documentación de [Pillow](https://pillow.readthedocs.io/en/stable/) y [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/). ¡Experimenta con diferentes operaciones y transforma tus imágenes!

**Lecturas recomendadas**

[cnn-data-sources | Kaggle](https://www.kaggle.com/alarcon7a/cnn-data-sources)

[Manejo de imágenes con Python | Kaggle](https://www.kaggle.com/alarcon7a/manejo-de-im-genes-con-python)

## Kernel en redes neuronales

### Teoría del Kernel en Redes Neuronales

En el contexto de las redes neuronales y el aprendizaje profundo, el término "kernel" puede referirse a dos conceptos principales:

1. **Kernel en Convoluciones**: En redes neuronales convolucionales (CNN), un kernel (o filtro) es una pequeña matriz de pesos que se desplaza sobre la entrada para realizar operaciones de convolución. Los kernels ayudan a detectar características específicas en los datos, como bordes, texturas, y patrones. Cada kernel se entrena para aprender diferentes características durante el proceso de entrenamiento.

2. **Kernel en Aprendizaje de Máquinas**: En métodos de aprendizaje de máquinas, como las máquinas de soporte vectorial (SVM), un kernel es una función que transforma los datos de entrada en un espacio de características de mayor dimensión. Esto permite encontrar patrones no lineales en los datos al facilitar la separación de diferentes clases.

### Ejemplo: Uso de Kernels en Redes Neuronales Convolucionales

A continuación, se presenta un ejemplo básico de una red neuronal convolucional que utiliza kernels para procesar imágenes. Usaremos TensorFlow y Keras para construir y entrenar un modelo simple que clasifique imágenes de dígitos de MNIST.

#### Paso 1: Importar las bibliotecas necesarias

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
```

#### Paso 2: Cargar y preprocesar los datos

```python
# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar las imágenes a un rango de [0, 1]
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
```

#### Paso 3: Construir el modelo

```python
# Definir la arquitectura del modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Primer kernel
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Segundo kernel
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Capa de salida

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

#### Paso 4: Entrenar el modelo

```python
# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
```

#### Paso 5: Evaluar el modelo

```python
# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')
```

### Explicación del Ejemplo

1. **Kernels en la Convolución**:
   - En la línea `model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))`, se crea un kernel de tamaño 3x3 que se aplica a la entrada de la imagen. Este kernel se desplaza sobre la imagen para extraer características relevantes.
   - El número de filtros (en este caso, 32) indica cuántos kernels se están utilizando para detectar características diferentes.

2. **MaxPooling**: Después de cada capa convolucional, se utiliza la operación de MaxPooling para reducir la dimensionalidad de las características y hacer que el modelo sea más eficiente.

3. **Capa de Salida**: Finalmente, se agrega una capa densa que utiliza la activación softmax para clasificar las imágenes en 10 clases diferentes (dígitos del 0 al 9).

### Conclusión

Los kernels son fundamentales en las redes neuronales convolucionales, ya que permiten detectar características importantes en las imágenes. Este ejemplo ilustra cómo construir una red neuronal convolucional simple y cómo los kernels se utilizan para extraer características de las imágenes.

**Lecturas recomendadas**

[Image Kernels explained visually](https://setosa.io/ev/image-kernels/)

[Convolutions: Image convolution examples - AI Shack](https://aishack.in/tutorials/image-convolution-examples/)