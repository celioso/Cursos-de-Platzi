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

## El kernel en acción

El **kernel** es un concepto clave en el campo de las redes neuronales, particularmente en las **redes neuronales convolucionales (CNNs)**. En este contexto, un kernel (o filtro) es una matriz pequeña de números que se desplaza sobre una imagen o matriz de entrada para realizar una operación de convolución. Este proceso permite extraer características importantes como bordes, texturas, o patrones dentro de los datos.

### Conceptos clave:

1. **Kernel (Filtro)**:
   - Es una pequeña matriz de pesos que se aplica a la entrada (como una imagen) a través de una operación de convolución. Normalmente, los kernels en las CNNs tienen un tamaño pequeño, como 3x3 o 5x5, y se inicializan con valores aleatorios que luego se ajustan durante el entrenamiento.
   
2. **Operación de Convolución**:
   - Durante la convolución, el kernel se desliza sobre la imagen de entrada y realiza una multiplicación punto a punto entre los elementos del kernel y una sección correspondiente de la entrada, generando una nueva matriz (mapa de características) que representa las características detectadas en la imagen.

3. **Aprendizaje de Características**:
   - En una CNN, los kernels son aprendidos automáticamente durante el entrenamiento del modelo a través de la retropropagación, ajustando los pesos para detectar características relevantes.

4. **Stride y Padding**:
   - El *stride* es el número de píxeles que el kernel se desplaza en cada paso. Un *stride* mayor reduce el tamaño del mapa de características.
   - El *padding* añade un borde de píxeles a la entrada, permitiendo que el kernel procese los píxeles de los bordes de la imagen.

### Ejemplo básico de convolución con un kernel 3x3

Supongamos que tenemos una imagen en escala de grises de 5x5 píxeles y un kernel de 3x3 que detecta bordes verticales.

#### Imagen de entrada (5x5 píxeles):
```
[1, 2, 3, 0, 1]
[4, 5, 6, 2, 1]
[7, 8, 9, 1, 2]
[1, 3, 2, 1, 3]
[2, 4, 1, 3, 0]
```

#### Kernel de detección de bordes (3x3):
```
[-1, 0, 1]
[-1, 0, 1]
[-1, 0, 1]
```

Este kernel se desliza sobre la imagen, y en cada paso realiza una multiplicación entre los valores correspondientes de la imagen y el kernel, sumando los productos para obtener un valor para la posición actual del mapa de características.

#### Ejemplo de operación de convolución en Python:

```python
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Imagen de entrada (5x5)
image = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 2, 1],
    [7, 8, 9, 1, 2],
    [1, 3, 2, 1, 3],
    [2, 4, 1, 3, 0]
])

# Kernel de detección de bordes (3x3)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Aplicar la convolución
output = convolve2d(image, kernel, mode='valid')

# Mostrar la imagen de entrada y el resultado de la convolución
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Imagen de entrada')
ax[1].imshow(output, cmap='gray')
ax[1].set_title('Resultado de la convolución')
plt.show()
```

### Explicación del código:
1. **Imagen**: Se define una matriz 2D de 5x5 para representar una imagen en escala de grises.
2. **Kernel**: El filtro de detección de bordes es una matriz de 3x3 diseñada para detectar bordes verticales.
3. **Convolución**: Usamos la función `convolve2d` de `scipy.signal` para aplicar la operación de convolución entre la imagen y el kernel.
4. **Visualización**: Usamos `matplotlib` para mostrar la imagen de entrada y el resultado del proceso de convolución.

### Salida:

La operación de convolución genera un nuevo mapa de características que resalta los bordes verticales en la imagen de entrada.

### Importancia del kernel en redes convolucionales:
- Los **kernels** permiten que las redes neuronales convolucionales sean especialmente eficaces en tareas de **visión por computadora** como el reconocimiento de imágenes.
- A través de múltiples capas convolucionales, las CNN pueden aprender a detectar características simples como bordes y texturas en las primeras capas, y características más complejas en las capas posteriores.

Este enfoque es uno de los motivos por los que las CNN son tan efectivas para el procesamiento de imágenes y han revolucionado campos como la visión por computadora, la segmentación de imágenes y el reconocimiento facial.

**Lecturas recomendadas**

[Image Kernels explained visually](https://setosa.io/ev/image-kernels/)

[cnn-data-sources | Kaggle](https://www.kaggle.com/alarcon7a/cnn-data-sources)

[Aplicación de kernel | Kaggle](https://www.kaggle.com/alarcon7a/aplicaci-n-de-kernel)

## Padding y Strides

### **Padding** y **Strides** en Redes Neuronales Convolucionales (CNN)

Las **redes neuronales convolucionales (CNN)** utilizan dos conceptos clave para controlar cómo se aplican los filtros a las entradas: **padding** y **strides**. Ambos conceptos influyen en el tamaño de las salidas y en cómo se mueven los filtros a lo largo de la imagen de entrada.

### **1. Padding**
El **padding** se refiere al proceso de añadir píxeles adicionales alrededor del borde de la imagen de entrada antes de aplicar el filtro convolucional. Esto se hace para controlar el tamaño de la salida y prevenir la pérdida de información en los bordes.

#### Tipos de Padding:
- **Sin padding (valid padding):** No se añaden píxeles adicionales. Esto reduce el tamaño de la imagen de salida, ya que el filtro solo se aplica a las áreas donde puede encajar completamente.
- **Padding (same padding):** Se añaden píxeles para asegurarse de que la salida tenga el mismo tamaño que la entrada. Esto es útil para mantener las dimensiones originales de la imagen.

#### Ejemplo:
Si tienes una imagen de tamaño 5x5 y aplicas un filtro de 3x3 sin padding, la imagen de salida será de 3x3 (menor que la original). Con padding, la imagen de salida podría ser de 5x5.

**Ejemplo de código:**
```python
import tensorflow as tf

# Imagen de entrada de 5x5x1
input_image = tf.random.normal([1, 5, 5, 1])

# Aplicamos una convolución sin padding (válido)
conv_layer_no_padding = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='valid')
output_no_padding = conv_layer_no_padding(input_image)

print("Tamaño de salida sin padding:", output_no_padding.shape)  # Salida de 3x3

# Aplicamos una convolución con padding (mismo tamaño)
conv_layer_with_padding = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')
output_with_padding = conv_layer_with_padding(input_image)

print("Tamaño de salida con padding:", output_with_padding.shape)  # Salida de 5x5
```

### **2. Strides**
El **stride** define cuántos píxeles se mueve el filtro a lo largo de la imagen de entrada cada vez que se aplica. El valor por defecto del stride es 1, lo que significa que el filtro se mueve un píxel a la vez. Un **stride** mayor permite mover el filtro más rápido y reduce el tamaño de la imagen de salida.

#### Ejemplo:
Si aplicas un filtro de 3x3 con un **stride de 1**, el filtro se moverá un píxel a la vez y generará una salida más detallada. Si aplicas un **stride de 2**, el filtro se moverá dos píxeles a la vez, reduciendo el tamaño de la salida.

**Ejemplo de código:**
```python
# Imagen de entrada de 5x5x1
input_image = tf.random.normal([1, 5, 5, 1])

# Aplicamos una convolución con stride de 1
conv_layer_stride_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='valid')
output_stride_1 = conv_layer_stride_1(input_image)

print("Tamaño de salida con stride 1:", output_stride_1.shape)  # Salida de 3x3

# Aplicamos una convolución con stride de 2
conv_layer_stride_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, padding='valid')
output_stride_2 = conv_layer_stride_2(input_image)

print("Tamaño de salida con stride 2:", output_stride_2.shape)  # Salida de 2x2
```

### **Combinando Padding y Strides**
Puedes combinar **padding** y **strides** para ajustar tanto el tamaño como la forma en que se procesan las imágenes. El **padding** te ayuda a controlar si la salida tendrá el mismo tamaño que la entrada, mientras que los **strides** ajustan cómo se aplica el filtro.

### **Conclusión**
- **Padding** ayuda a mantener el tamaño de la imagen y prevenir la pérdida de información en los bordes.
- **Strides** controlan la velocidad con la que el filtro se mueve a través de la imagen, afectando el tamaño de la salida.

El correcto uso de **padding** y **strides** es crucial para controlar las dimensiones y el rendimiento de los modelos de CNN.

**Lecturas recomendadas**

[https://arxiv.org/pdf/2010.02178.pdf](https://arxiv.org/pdf/2010.02178.pdf)

[Conv2D layer](https://keras.io/api/layers/convolution_layers/convolution2d/)

## Capa de pooling

### Capa de Pooling en Redes Neuronales Convolucionales (CNN)

Las capas de **pooling** son un componente fundamental en las **redes neuronales convolucionales (CNN)**. Su función principal es **reducir las dimensiones espaciales** (anchura y altura) de los mapas de características generados por las capas convolucionales, manteniendo las características más importantes. Al hacer esto, se reduce la cantidad de parámetros en la red, lo que ayuda a **prevenir el sobreajuste** y mejora la eficiencia computacional.

### Tipos de Pooling

Existen dos tipos principales de capas de pooling:

1. **Max Pooling (Agrupación máxima):**
   Selecciona el valor máximo de una ventana de entrada, destacando las características más importantes.

2. **Average Pooling (Agrupación promedio):**
   Calcula el valor promedio de los valores dentro de la ventana, generalizando la información.

### Max Pooling vs Average Pooling:
- **Max Pooling** es más común en la práctica, ya que permite que la red se concentre en las características más prominentes.
- **Average Pooling** es útil cuando se quiere retener más información general, en lugar de solo los picos de activación.

### ¿Cómo funciona la capa de Pooling?

En una capa de pooling, se aplica una ventana (normalmente de 2x2) sobre el mapa de características, y se desplaza por la imagen de acuerdo con el **stride** (normalmente stride=2). En el caso de **max pooling**, dentro de cada ventana se selecciona el valor máximo, reduciendo así el tamaño de la imagen de entrada a la mitad (si el stride es 2).

### Ejemplo de Max Pooling

Supongamos que tienes un mapa de características de entrada de tamaño 4x4 y aplicas una operación de **max pooling** con una ventana de 2x2 y un stride de 2.

Mapa de características de entrada (4x4):
```
[[1, 3, 2, 4],
 [5, 6, 1, 2],
 [8, 9, 4, 1],
 [3, 7, 5, 6]]
```

Aplicando max pooling (tomamos el valor máximo en cada ventana 2x2):
```
[[6, 4],
 [9, 6]]
```

### Ejemplo en código con TensorFlow/Keras

```python
import tensorflow as tf
import numpy as np

# Creamos una imagen de ejemplo de 4x4 con un solo canal
input_image = np.array([[[[1], [3], [2], [4]],
                         [[5], [6], [1], [2]],
                         [[8], [9], [4], [1]],
                         [[3], [7], [5], [6]]]], dtype=np.float32)

# Definimos el tensor de entrada
input_tensor = tf.constant(input_image)

# Aplicamos la capa de MaxPooling2D con una ventana de 2x2 y un stride de 2
max_pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

# Aplicamos la capa de pooling a la entrada
output = max_pool_layer(input_tensor)

# Mostramos la salida
print("Entrada:")
print(input_image.reshape(4, 4))  # Formateamos para visualizar mejor

print("\nSalida después de MaxPooling2D:")
print(output.numpy().reshape(2, 2))  # Salida tras aplicar max pooling
```

### Salida del código:
```
Entrada:
[[1. 3. 2. 4.]
 [5. 6. 1. 2.]
 [8. 9. 4. 1.]
 [3. 7. 5. 6.]]

Salida después de MaxPooling2D:
[[6. 4.]
 [9. 6.]]
```

### Explicación:
- La entrada es un tensor de 4x4 con un canal.
- La operación de **max pooling** con una ventana de 2x2 y un stride de 2 selecciona el valor máximo en cada sub-matriz 2x2 de la imagen.
- El tamaño de la imagen de salida es reducido de 4x4 a 2x2, conservando las características más destacadas.

### Beneficios de la Capa de Pooling

1. **Reducción de dimensionalidad:** Al reducir el tamaño de los mapas de características, la red es más eficiente y requiere menos memoria.
2. **Menor riesgo de sobreajuste:** Con menos parámetros, hay menor posibilidad de que la red se ajuste excesivamente a los datos de entrenamiento.
3. **Retención de características importantes:** La agrupación máxima, en particular, conserva las características más destacadas de cada región.

### Conclusión

Las capas de pooling, especialmente el **Max Pooling**, son esenciales en las CNN para reducir el tamaño de los datos, preservar características importantes y mejorar la eficiencia computacional.

## Arquitectura de redes convolucionales

### Arquitectura de Redes Neuronales Convolucionales (CNN)

Las **Redes Neuronales Convolucionales (CNN)** son un tipo de red neuronal especialmente diseñadas para trabajar con datos con estructuras de tipo grid, como imágenes. Estas redes son altamente eficaces en tareas de visión por computadora, como clasificación de imágenes, detección de objetos y segmentación.

Una CNN consta de varias capas organizadas de forma jerárquica, en las cuales cada capa procesa características cada vez más complejas. Las capas principales que forman parte de una CNN incluyen capas **convolucionales**, **de activación**, **de pooling** y **completamente conectadas**.

### Principales Componentes de una CNN

1. **Capa de Convolución (Conv Layer)**:
   La capa convolucional es el bloque básico de una CNN. Aplica filtros (o kernels) sobre la imagen de entrada para extraer características locales. Cada filtro aprende a detectar características específicas, como bordes, texturas o patrones.

2. **Capa de Activación (ReLU)**:
   Después de cada capa convolucional, se aplica una función de activación no lineal, como **ReLU (Rectified Linear Unit)**, que introduce no linealidades al modelo, permitiendo que la red aprenda relaciones más complejas.

3. **Capa de Pooling**:
   Las capas de pooling (por lo general **Max Pooling**) se encargan de reducir las dimensiones espaciales de los mapas de características, lo que disminuye la cantidad de parámetros y computación en la red. También ayudan a hacer las características más invariantes a la traslación.

4. **Capa Completamente Conectada (Fully Connected Layer)**:
   Después de las capas convolucionales y de pooling, la información procesada se aplana en un vector y pasa por una o varias capas completamente conectadas. Estas capas actúan como un clasificador final.

5. **Función de Activación Softmax**:
   En la última capa, normalmente se utiliza la activación **softmax** (para clasificación multiclase), que convierte las salidas en probabilidades.

### Ejemplo de una Arquitectura CNN Básica

A continuación, se presenta una arquitectura CNN típica para clasificación de imágenes con TensorFlow y Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Definir el modelo secuencial
model = models.Sequential()

# Primera capa convolucional
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))  # Max Pooling

# Segunda capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  # Max Pooling

# Tercera capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanar los datos y conectar a las capas completamente conectadas
model.add(layers.Flatten())  # Aplana el mapa de características
model.add(layers.Dense(64, activation='relu'))  # Capa completamente conectada
model.add(layers.Dense(10, activation='softmax'))  # Capa de salida para clasificación (10 clases)

# Resumen de la arquitectura del modelo
model.summary()
```

### Explicación de la Arquitectura

1. **Primera Capa Convolucional (Conv2D):**
   - Aplica 32 filtros de tamaño 3x3 a la imagen de entrada (28x28x1) con activación ReLU.
   - **Max Pooling** reduce el tamaño espacial de 28x28 a 14x14.

2. **Segunda Capa Convolucional:**
   - Aplica 64 filtros de tamaño 3x3, seguidos de otra operación de Max Pooling, que reduce las dimensiones a 7x7.

3. **Tercera Capa Convolucional:**
   - Aplica 64 filtros adicionales, detectando características más complejas.

4. **Capa de Flatten:**
   - Aplana los datos (7x7x64) en un vector de características de 3136 elementos para conectarse a la capa densa.

5. **Capa Completamente Conectada:**
   - Esta capa densamente conectada tiene 64 unidades, seguidas por la capa de salida **softmax** para la clasificación en 10 clases (en este caso, suponiendo un problema de clasificación con 10 clases).

### Flujo de Datos en una CNN

Cuando una imagen entra en la red:

1. **Capa Convolucional**: Los filtros convolucionales se aplican a la imagen, y se generan mapas de características que resaltan patrones locales, como bordes, esquinas y texturas.
2. **ReLU**: Se eliminan las activaciones negativas en los mapas de características.
3. **Max Pooling**: Se reduce el tamaño de los mapas de características, reteniendo solo la información más importante.
4. **Capas adicionales**: A medida que avanzamos en la red, las capas convolucionales capturan características más abstractas y complejas.
5. **Capa Completamente Conectada**: Finalmente, la información se transforma en una salida de clasificación.

### Visualización de la Arquitectura del Modelo

El resumen del modelo proporcionado por `model.summary()` produce una tabla similar a la siguiente:

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
flatten_1 (Flatten)          (None, 576)               0         
dense_1 (Dense)              (None, 64)                36928     
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

### Aplicaciones de las CNN

- **Clasificación de Imágenes:** Etiquetar imágenes en diferentes categorías (como en el caso de MNIST, clasificación de dígitos).
- **Detección de Objetos:** Encontrar la ubicación y la clase de objetos dentro de una imagen.
- **Segmentación Semántica:** Clasificar cada píxel de una imagen en una categoría.
- **Reconocimiento Facial:** Identificar o verificar rostros en imágenes o videos.
  
### Conclusión

Las CNN son una arquitectura poderosa que permite procesar y clasificar datos visuales de manera eficiente. La combinación de capas convolucionales, pooling y completamente conectadas permite a las CNN aprender características jerárquicas en las imágenes, lo que es esencial para su rendimiento en tareas complejas de visión por computadora.

**Lecturas recomendadas**

[cifar10  |  TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cifar10)

[cifar - clasification | Kaggle](https://www.kaggle.com/alarcon7a/cifar-clasification)

[CIFAR10 small images classification dataset](https://keras.io/api/datasets/cifar10/)

## Clasificación con redes neuronales convolucionales

### Clasificación con Redes Neuronales Convolucionales (CNN)

La **clasificación de imágenes** mediante **Redes Neuronales Convolucionales (CNN)** es una de las aplicaciones más comunes en el campo de la visión por computadora. Las CNN son especialmente eficaces porque capturan patrones y características espaciales de las imágenes mediante la convolución, lo que las hace ideales para tareas como el reconocimiento de objetos, clasificación de imágenes, y más.

#### ¿Qué es la Clasificación de Imágenes?

En la clasificación de imágenes, el objetivo es asignar una etiqueta a una imagen de acuerdo con las características visuales que presenta. Por ejemplo, en el conjunto de datos de **MNIST**, la tarea consiste en clasificar imágenes de dígitos escritos a mano en categorías de 0 a 9.

### ¿Cómo funcionan las CNN en la Clasificación de Imágenes?

Las CNN utilizan varias capas, como convolucionales, de activación y de pooling, para extraer características relevantes de las imágenes de entrada. La información procesada pasa luego por capas completamente conectadas (fully connected), que funcionan como un clasificador y generan la probabilidad de que la imagen pertenezca a una determinada clase.

### Arquitectura de una CNN para Clasificación

- **Entrada (Input Layer):** Imagen en formato de píxeles, con dimensiones típicas como 28x28 píxeles en el caso del dataset MNIST.
- **Capas Convolucionales (Conv Layer):** Aplican filtros que detectan características locales, como bordes y patrones.
- **Capas de Pooling (Max Pooling Layer):** Reducen las dimensiones espaciales de los mapas de características, preservando la información más relevante.
- **Capas Completamente Conectadas (Dense Layer):** Conectan todas las neuronas de la capa anterior y producen una salida.
- **Capa de Salida (Output Layer):** Utiliza la activación **softmax** para producir una distribución de probabilidades sobre las clases posibles.

### Ejemplo Práctico de Clasificación de Imágenes con CNN usando TensorFlow y Keras

Vamos a construir una CNN básica para clasificar imágenes del conjunto de datos **MNIST** (dígitos escritos a mano).

#### Paso 1: Importar las librerías necesarias

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Redimensionar y normalizar los datos
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
```

- Las imágenes de entrada se redimensionan a 28x28 píxeles y se les añade una dimensión de canal (1 en este caso porque es en escala de grises).
- Se normalizan dividiendo entre 255 para que los valores estén en el rango [0, 1].

#### Paso 2: Construir el modelo CNN

```python
# Definir el modelo
model = models.Sequential()

# Primera capa convolucional
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Segunda capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Tercera capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanar las características y conectarlas a una capa densa
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Capa de salida con 10 neuronas (10 clases) y softmax
model.add(layers.Dense(10, activation='softmax'))

# Resumen del modelo
model.summary()
```

### Explicación del Modelo

1. **Primera Capa Convolucional:**
   - Aplica 32 filtros de 3x3 sobre la imagen de entrada. La activación `ReLU` introduce no linealidad, y la entrada es de tamaño 28x28x1.

2. **MaxPooling2D:** 
   - Reduce la dimensionalidad espacial de la imagen con un tamaño de ventana 2x2, lo que genera una reducción del tamaño a la mitad.

3. **Segunda y Tercera Capa Convolucional:** 
   - Detectan características más complejas al aplicar más filtros, 64 en este caso, con convoluciones de 3x3.

4. **Capa de Flatten:** 
   - Aplana los datos de la red para pasarlos a la capa completamente conectada.

5. **Capas Densas:** 
   - Primera capa completamente conectada con 64 neuronas, seguida de la capa de salida con 10 neuronas, donde cada neurona representa una clase.

6. **Capa de salida:** 
   - Utiliza la activación **softmax** para obtener una distribución de probabilidades sobre las 10 posibles clases (dígitos del 0 al 9).

#### Paso 3: Compilar el modelo

```python
# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

- El optimizador **Adam** se utiliza para minimizar la función de pérdida.
- La función de pérdida es **sparse_categorical_crossentropy**, que se usa para clasificación de múltiples clases.

#### Paso 4: Entrenar el modelo

```python
# Entrenar el modelo
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
```

Este código entrena el modelo CNN con los datos de entrenamiento durante 5 épocas y evalúa su rendimiento en los datos de validación (el conjunto de prueba).

#### Paso 5: Evaluar el modelo

```python
# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc:.4f}')
```

- Al final, se evalúa la precisión del modelo en el conjunto de datos de prueba.

#### Paso 6: Visualización de las métricas de entrenamiento

Podemos visualizar la precisión y la pérdida durante el entrenamiento:

```python
# Graficar la precisión del entrenamiento y la validación
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Graficar la pérdida del entrenamiento y la validación
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
```

### Explicación General

1. **Entrenamiento:** El modelo CNN se entrena ajustando los pesos en las capas convolucionales y completamente conectadas para minimizar la pérdida (sparse_categorical_crossentropy). La red ajusta los filtros para aprender a detectar características específicas como bordes o formas en las imágenes de entrada.

2. **Clasificación:** Después del entrenamiento, el modelo predice la clase más probable para una imagen de entrada basándose en las características aprendidas.

3. **Precisión:** A medida que se entrena el modelo, esperamos que tanto la precisión en el conjunto de entrenamiento como en el conjunto de validación aumenten.

### Conclusión

Las CNN son extremadamente eficaces para la clasificación de imágenes debido a su capacidad para aprender características jerárquicas de las imágenes. En este ejemplo, se construyó y entrenó un modelo CNN utilizando el dataset MNIST, mostrando cómo la red puede clasificar dígitos escritos a mano.

**Lecturas recomendadas**

[cifar10  |  TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cifar10)

[cifar - clasification | Kaggle](https://www.kaggle.com/alarcon7a/cifar-clasification)

[CIFAR10 small images classification dataset](https://keras.io/api/datasets/cifar10/)

## Creación de red convolucional para clasificación

### Creación de una Red Neuronal Convolucional (CNN) para Clasificación

Una **Red Neuronal Convolucional (CNN)** es un tipo de red neuronal diseñada específicamente para procesar datos con una estructura de cuadrícula, como las imágenes. Se utiliza principalmente en tareas de visión por computadora, como clasificación, detección de objetos y segmentación de imágenes.

### Componentes Clave de una CNN

1. **Capa Convolucional (Conv Layer):** La CNN aplica filtros (kernels) sobre la imagen de entrada, que extraen características locales como bordes, texturas o patrones.
2. **Capa de Activación (ReLU):** Introduce no linealidades en la red, lo que permite que la red aprenda funciones más complejas.
3. **Capa de Pooling (Max Pooling):** Reduce las dimensiones de las características extraídas para hacer el modelo más eficiente y evitar sobreajuste.
4. **Capas Completamente Conectadas (Dense Layers):** Después de las capas convolucionales, los datos se aplanan y se conectan a neuronas para clasificar las características extraídas.

### Ejemplo: Creación de una CNN para Clasificar el Dataset **CIFAR-10**

El dataset CIFAR-10 contiene imágenes de 10 clases diferentes, como aviones, automóviles, pájaros, gatos, etc.

#### Paso 1: Importar las librerías necesarias

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Cargar el dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar las imágenes entre 0 y 1
train_images, test_images = train_images / 255.0, test_images / 255.0
```

- **CIFAR-10:** Consiste en 60,000 imágenes de tamaño 32x32 y 3 canales (RGB).
- Las imágenes se normalizan para que los valores de los píxeles estén entre 0 y 1, facilitando el entrenamiento del modelo.

#### Paso 2: Visualizar algunas imágenes del dataset

```python
# Definir las clases de CIFAR-10
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Visualizar las primeras 9 imágenes del conjunto de entrenamiento
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])  # Quitar marcas de los ejes x
    plt.yticks([])  # Quitar marcas de los ejes y
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```

#### Paso 3: Definir la arquitectura de la CNN

```python
# Definir el modelo CNN
model = models.Sequential()

# Primera capa convolucional
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Segunda capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Tercera capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanar los datos
model.add(layers.Flatten())

# Añadir una capa completamente conectada
model.add(layers.Dense(64, activation='relu'))

# Capa de salida con 10 neuronas (una para cada clase)
model.add(layers.Dense(10, activation='softmax'))

# Ver el resumen del modelo
model.summary()
```

### Explicación del Modelo

1. **Capas Convolucionales:** 
   - Se utilizan tres capas convolucionales, cada una aplicando un filtro sobre la imagen.
   - La primera capa convolucional usa 32 filtros de tamaño 3x3. Las siguientes dos capas utilizan 64 filtros de 3x3.
   - Cada capa es seguida por una capa de **MaxPooling2D**, que reduce las dimensiones de las características.

2. **Capa de Aplanamiento (Flatten):** 
   - Aplana las características aprendidas por las capas convolucionales para conectarlas a una capa densa.

3. **Capas Densas:** 
   - La capa completamente conectada tiene 64 neuronas, que detectan combinaciones de las características extraídas.
   - La capa de salida tiene 10 neuronas, cada una correspondiente a una clase de CIFAR-10, y usa la activación **softmax** para generar probabilidades de clasificación.

#### Paso 4: Compilar y entrenar el modelo

```python
# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```

- **Optimizador Adam:** Se utiliza para actualizar los pesos del modelo.
- **Función de pérdida:** `sparse_categorical_crossentropy` porque estamos tratando con clasificación de múltiples clases.
- El modelo se entrena durante 10 épocas.

#### Paso 5: Evaluar el rendimiento del modelo

```python
# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")
```

- El modelo se evalúa en el conjunto de prueba, y se imprime la precisión.

#### Paso 6: Visualizar las métricas de entrenamiento

```python
# Graficar la precisión del entrenamiento y validación
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.show()

# Graficar la pérdida del entrenamiento y validación
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')
plt.show()
```

### Conclusión

En este ejercicio hemos creado una **red neuronal convolucional** para clasificar imágenes del dataset CIFAR-10. El modelo consiste en capas convolucionales y de pooling que extraen características de las imágenes y capas densas que clasifican las características aprendidas. Este proceso es fundamental para las aplicaciones de visión por computadora, como el reconocimiento de objetos o clasificación automática de imágenes.

## Entrenamiento de un modelo de clasificación con redes convolucionales

Entrenar un modelo de clasificación con redes neuronales convolucionales (CNN) implica varios pasos clave: desde la definición de la arquitectura de la red, la preparación de los datos, el entrenamiento del modelo y finalmente la evaluación de su rendimiento.

Aquí te explico cómo realizar este proceso paso a paso con un ejemplo simple que utiliza el conjunto de datos **CIFAR-10** (un conjunto de imágenes pequeñas de 10 clases):

### 1. **Importar las librerías necesarias**
Primero, asegúrate de tener instaladas las librerías necesarias como `TensorFlow`, `NumPy`, y `Matplotlib`.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
```

### 2. **Cargar y preparar los datos**

El conjunto de datos CIFAR-10 contiene 60,000 imágenes de 32x32 píxeles clasificadas en 10 categorías.

```python
# Cargar los datos de CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar los valores de píxeles entre 0 y 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convertir las etiquetas a formato one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

### 3. **Definir la arquitectura de la red convolucional**

Aquí definimos una red sencilla con varias capas convolucionales, de pooling y totalmente conectadas.

```python
# Definir el modelo secuencial
model = Sequential()

# Primera capa convolucional
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda capa convolucional
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercera capa convolucional
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar las capas
model.add(Flatten())

# Añadir una capa densa
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(10, activation='softmax'))

# Mostrar el resumen del modelo
model.summary()
```

### 4. **Compilar el modelo**

Especificamos el optimizador, la función de pérdida y las métricas de evaluación.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 5. **Entrenar el modelo**

Entrenamos el modelo utilizando los datos de entrenamiento. Ajusta las épocas y el tamaño del batch según tus necesidades.

```python
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
```

### 6. **Evaluar el modelo**

Después del entrenamiento, evaluamos el modelo con los datos de prueba.

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
```

### 7. **Visualizar los resultados**

Puedes visualizar la precisión y la pérdida durante el entrenamiento para ver cómo ha evolucionado el modelo.

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
```

### Explicación del modelo

1. **Capas convolucionales (Conv2D):** Extraen características importantes de las imágenes como bordes y texturas.
2. **Pooling (MaxPooling2D):** Reduce el tamaño de las características manteniendo la información más relevante.
3. **Aplanamiento (Flatten):** Convierte los datos 2D en un vector para pasar a la capa densa.
4. **Capa totalmente conectada (Dense):** Realiza la clasificación final.

Este modelo es simple y puede mejorar con ajustes más avanzados, como añadir más capas, usar diferentes optimizadores o aplicar técnicas de regularización.

**Lecturas recomendadas**

[cifar10  |  TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cifar10)

[cifar - clasification | Kaggle](https://www.kaggle.com/alarcon7a/cifar-clasification)

[CNN Explainer](https://poloclub.github.io/cnn-explainer/)