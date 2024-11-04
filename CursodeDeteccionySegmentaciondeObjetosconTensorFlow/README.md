# Curso de Detección y Segmentación de Objetos con TensorFlow

## ¿Qué es la visión computarizada y cuáles son sus tipos?

La visión computarizada es una rama de la inteligencia artificial (IA) que permite a las máquinas interpretar y comprender el contenido de imágenes y videos de la misma forma que lo haría un ser humano. Utiliza algoritmos de procesamiento de imágenes, redes neuronales y técnicas de aprendizaje profundo para analizar datos visuales y extraer información útil.

¿Para qué se usa la visión computarizada?
La visión computarizada se aplica en numerosos campos, como la seguridad, la conducción autónoma, la robótica, la medicina, la industria manufacturera, el análisis de imágenes satelitales, y muchos más. Sus aplicaciones buscan automatizar tareas que requieren la capacidad de visión, como la detección de objetos, el reconocimiento facial, la segmentación de imágenes, etc.

Tipos de visión computarizada
La visión computarizada abarca varias áreas y subcampos, entre los que se incluyen:

Detección de objetos:

Identificación y localización de objetos específicos en una imagen o video.
Ejemplos: detección de peatones en sistemas de conducción autónoma, identificación de productos en almacenes.
Reconocimiento de imágenes:

Clasificación de imágenes en categorías predefinidas.
Ejemplos: clasificación de imágenes de perros y gatos, diagnóstico médico basado en imágenes.
Segmentación de imágenes:

Dividir una imagen en segmentos o regiones para identificar los límites de los objetos.
Ejemplos: segmentación semántica para reconocer píxeles específicos de un objeto, como en imágenes médicas.
Reconocimiento facial:

Identificación y verificación de identidades a través de características faciales.
Ejemplos: sistemas de autenticación biométrica en teléfonos inteligentes, vigilancia de seguridad.
Análisis de video:

Procesamiento y análisis de secuencias de video para detectar movimiento, actividades o cambios.
Ejemplos: detección de actividad sospechosa en videovigilancia, monitoreo del tráfico vehicular.
Visión estéreo:

Uso de dos o más cámaras para captar diferentes ángulos de una escena y recrear la percepción de profundidad.
Ejemplos: mapeo en 3D y reconstrucción de entornos.
Reconocimiento de patrones:

Identificación de patrones o características en imágenes para analizar datos de manera más eficiente.
Ejemplo: detección de anomalías en imágenes industriales para control de calidad.
Reconocimiento óptico de caracteres (OCR):

Conversión de texto en imágenes o documentos escaneados a texto digital editable.
Ejemplos: lectura automática de matrículas de vehículos, digitalización de documentos.
Herramientas y técnicas utilizadas
La visión computarizada se implementa mediante el uso de diversas tecnologías y algoritmos, como:

Redes neuronales convolucionales (CNN): Modelos de aprendizaje profundo especialmente diseñados para analizar datos visuales.
Procesamiento de imágenes: Técnicas para mejorar y manipular imágenes antes de la interpretación.
Modelos pre-entrenados: Redes como ResNet, VGG, MobileNet, entre otros, que se utilizan para aplicaciones rápidas con pocas modificaciones.
Frameworks: Herramientas como OpenCV, TensorFlow, PyTorch, y Keras son fundamentales para desarrollar soluciones de visión computarizada.
En resumen, la visión computarizada es una disciplina que está transformando múltiples industrias al automatizar y mejorar la comprensión de los datos visuales, y existen varios tipos que se enfocan en diferentes tareas y aplicaciones.

## Introducción a object detection: sliding window y bounding box

La **detección de objetos** es una técnica clave en la visión computarizada que no solo reconoce qué objetos están presentes en una imagen, sino que también los localiza mediante **bounding boxes** (cajas delimitadoras). Este proceso es esencial para aplicaciones como la conducción autónoma, la vigilancia, la robótica, y muchas otras.

### 1. ¿Qué es la detección de objetos?
La detección de objetos consiste en identificar instancias de objetos de una o más clases (como personas, automóviles, animales, etc.) en una imagen o video y marcarlos con una **caja delimitadora** que define la posición y el tamaño de cada objeto. 

### 2. Técnicas básicas de detección de objetos

#### a. Sliding Window
El método de **ventana deslizante (sliding window)** es uno de los enfoques más básicos y antiguos en la detección de objetos. Este método implica:

- **Dividir la imagen**: La imagen se divide en secciones más pequeñas utilizando una ventana de tamaño fijo que se desliza sobre toda la imagen en diferentes posiciones y escalas.
- **Extracción de características**: Para cada posición de la ventana, se extraen características de la imagen que luego se envían a un clasificador (por ejemplo, SVM o una red neuronal simple).
- **Clasificación**: El clasificador determina si la región de la imagen contiene o no el objeto de interés.
- **Escaneo a múltiples escalas**: Para detectar objetos de diferentes tamaños, la ventana se redimensiona y se vuelve a escanear la imagen.

**Ventajas**:
- Simplicidad en su implementación y comprensión.
- Compatible con cualquier clasificador o algoritmo de extracción de características.

**Desventajas**:
- **Computacionalmente costoso**: Escanear la imagen a diferentes escalas y posiciones requiere mucho tiempo de procesamiento, especialmente en imágenes de alta resolución.
- **Ineficiencia**: No es práctico para aplicaciones en tiempo real.

#### b. Bounding Box
Una **bounding box** es un rectángulo que rodea el objeto detectado y se utiliza para marcar su ubicación en la imagen. Las coordenadas de una bounding box se suelen definir mediante cuatro valores: `(x_min, y_min, x_max, y_max)`, donde `(x_min, y_min)` representa la esquina superior izquierda y `(x_max, y_max)` la esquina inferior derecha.

Las bounding boxes se utilizan junto con los algoritmos de detección de objetos para presentar visualmente dónde se encuentra un objeto detectado dentro de la imagen.

### 3. Limitaciones del enfoque tradicional
El enfoque de sliding window, aunque funcional, es ineficiente para tareas complejas debido a la alta carga computacional y al procesamiento redundante de áreas de la imagen. Esto llevó al desarrollo de métodos más avanzados como:

- **Redes convolucionales (CNN)** para una extracción de características más eficiente.
- **R-CNN (Regions with CNN features)**: Mejora la selección de regiones al aplicar una red convolucional en cada propuesta de región.
- **YOLO (You Only Look Once)** y **SSD (Single Shot MultiBox Detector)**: Técnicas modernas que procesan la imagen completa en un solo paso para detectar objetos en tiempo real.

### 4. ¿Cómo mejoran las técnicas modernas la detección de objetos?
Las técnicas modernas, como YOLO y SSD, reemplazan el enfoque de sliding window con redes neuronales que procesan la imagen en su totalidad y predicen simultáneamente múltiples bounding boxes y las clases de los objetos. Esto resulta en una detección de objetos mucho más rápida y precisa, ideal para aplicaciones en tiempo real.

**Ventajas de los métodos avanzados**:
- **Detección en tiempo real**: Las redes como YOLO pueden procesar múltiples cuadros por segundo.
- **Precisión mejorada**: Mejor rendimiento en términos de precisión y localización de los objetos.
- **Escalabilidad**: Capacidad para trabajar con múltiples clases de objetos de diferentes tamaños.

En conclusión, la **detección de objetos** ha evolucionado significativamente desde enfoques tradicionales como la **sliding window** hasta métodos más avanzados que emplean **bounding boxes** y técnicas de deep learning para mejorar la eficiencia y precisión.

## Generando video de sliding window

Para generar un video de un proceso de `sliding window`, puedes seguir los pasos a continuación. Vamos a asumir que ya tienes las imágenes generadas en una carpeta (por ejemplo, `/content/sliding_windows/`).

### 1. Usar OpenCV para crear un video

Puedes usar la biblioteca `cv2` de OpenCV para leer las imágenes y escribir un archivo de video.

```python
import cv2
import os

# Directorio de las imágenes generadas
image_folder = '/content/sliding_windows/'
video_name = '/content/sliding_window_video.avi'

# Obtener una lista de todas las imágenes
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # Asegurarse de que estén en el orden correcto

# Leer la primera imagen para obtener el tamaño del video
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Crear el objeto de escritura de video
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))

# Escribir cada imagen en el video
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Liberar el objeto de escritura de video
video.release()

print(f"Video guardado como {video_name}")
```

### Explicación:
- **`cv2.VideoWriter_fourcc(*'XVID')`**: Codificación del video (puedes cambiarlo por `*'mp4v'` para `.mp4`).
- **`10`**: Número de cuadros por segundo (puedes ajustar el valor para cambiar la velocidad del video).
- **`(width, height)`**: Tamaño del video, que debe coincidir con el tamaño de las imágenes.

### Convertir el video a otro formato (opcional):
Si prefieres un archivo `.mp4`, puedes usar herramientas como `ffmpeg`:

```bash
!ffmpeg -i /content/sliding_window_video.avi /content/sliding_window_video.mp4
```

Este proceso convierte el video `.avi` a `.mp4`.

### Resultado:
Después de ejecutar el código, tendrás un archivo de video que muestra el proceso de `sliding window`.

## Introducción a object detection: backbone, non-max suppression y métricas

La detección de objetos es una técnica clave en la visión por computadora que permite identificar y localizar objetos en imágenes o videos. Para comprender su funcionamiento, es importante conocer algunos conceptos básicos como el **backbone**, **non-max suppression (NMS)** y las **métricas de evaluación**.

### 1. **Backbone**
El *backbone* es una red neuronal pre-entrenada que se utiliza como base para extraer características de las imágenes. Esta red actúa como un extractor de características profundo y convierte una imagen de entrada en un conjunto de mapas de características que se pueden procesar para detectar objetos.

**Ejemplos de backbones comunes:**
- **ResNet**: Una red residual que facilita el entrenamiento de redes muy profundas gracias a sus conexiones residuales.
- **VGG**: Una red más sencilla pero eficaz para la extracción de características, aunque es más costosa computacionalmente.
- **MobileNet**: Ideal para aplicaciones en dispositivos con recursos limitados debido a su eficiencia.
- **EfficientNet**: Un backbone más moderno que ofrece un buen equilibrio entre precisión y rendimiento computacional.

### 2. **Non-Max Suppression (NMS)**
El *Non-Max Suppression* es un proceso utilizado para reducir la cantidad de cajas delimitadoras superpuestas. Cuando un modelo de detección de objetos predice múltiples cajas para un mismo objeto, NMS selecciona la caja con la puntuación más alta y elimina las otras si su área de superposición excede un umbral definido.

**Funcionamiento de NMS:**
1. Se ordenan todas las cajas predictivas por la puntuación de confianza en orden descendente.
2. Se selecciona la caja con la puntuación más alta y se añade al resultado final.
3. Se eliminan las cajas que tienen un IoU (Intersección sobre Unión) mayor que un umbral predefinido con la caja seleccionada.
4. Se repite el proceso hasta que no queden más cajas por evaluar.

**Ventaja**: NMS ayuda a reducir las detecciones redundantes, permitiendo un resultado más claro y preciso.

### 3. **Métricas de Evaluación**
Para medir el rendimiento de los modelos de detección de objetos, se utilizan varias métricas, siendo las más comunes:

- **IoU (Intersección sobre Unión)**: Es la métrica clave para determinar la precisión de las predicciones. Mide la superposición entre la caja predictiva y la caja de verdad de campo. Se calcula como el área de intersección dividido por el área de unión de ambas cajas. Un IoU de 1 significa que la predicción es perfecta.

- **mAP (mean Average Precision)**:
  - **AP (Average Precision)**: Calcula el área bajo la curva de precisión-recall para una clase específica.
  - **mAP**: Es el promedio de APs calculado para todas las clases. Una métrica de evaluación común en la detección de objetos que proporciona una idea del rendimiento general del modelo.

- **Precisión y Recall**:
  - **Precisión**: Mide la cantidad de verdaderos positivos sobre todos los positivos predichos. Una alta precisión indica pocas detecciones falsas positivas.
  - **Recall**: Mide la cantidad de verdaderos positivos sobre todos los positivos reales. Un alto recall indica que se detectan la mayoría de los objetos presentes.

**Ejemplo de Evaluación**:
Un modelo con un mAP alto es capaz de detectar y localizar objetos con precisión y consistencia, lo cual es esencial en aplicaciones como la vigilancia, el análisis de imágenes médicas, la conducción autónoma y más.

### **Resumen**
- **Backbone**: Extrae características de las imágenes.
- **NMS**: Reduce las detecciones redundantes seleccionando las cajas más significativas.
- **Métricas de evaluación**: IoU, mAP, precisión y recall ayudan a medir el rendimiento de los modelos de detección.

Estos conceptos son la base para comprender y mejorar los modelos de detección de objetos y son fundamentales para quienes trabajan en aplicaciones de visión por computadora.

**Trabajando con métricas de object detection**

**Intersección entre uniones (IOU)**: la verdadera posición del objecto sobre la predicción del modelo
**Non-max supression**: Selecciona un bounding box correcto de los múltiples bounding boxes superpuestos
**Métricas importantes**:
- mean average precision (mAp): métrica de accuracy
- número de frames por segundo: Es cuántos frames es capaz de procesar nuestro modelo por segundo de manera efectiva. Es muy importante para evaluar el modelo en la vida real

## Visualización de IoU en object detection

Para visualizar el IoU (Intersection over Union) en detección de objetos, necesitas superponer las cajas (ground truth y predicción) sobre una imagen y mostrar la región de intersección. Aquí te explico cómo hacerlo paso a paso usando `matplotlib`.

### Código para Visualizar IoU

1. **Dibuja las cajas delimitadoras (bounding boxes)**.
2. **Calcula la región de intersección y dibújala**.
3. **Muestra el valor de IoU en la imagen**.

Aquí tienes un ejemplo de código para visualizar el IoU:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def bb_intersection_over_union(boxA, boxB):
    # Coordenadas de la intersección
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Área de la intersección
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    
    # Áreas de las cajas
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Cálculo del IoU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    
    return iou, (xA, yA, xB, yB)

# Cajas de ejemplo [x_min, y_min, x_max, y_max]
boxA = [50, 50, 150, 150]  # Ground truth
boxB = [80, 80, 180, 180]  # Predicción

iou, intersection_box = bb_intersection_over_union(boxA, boxB)

# Visualización
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.set_title(f"IoU: {iou:.2f}")
ax.imshow(np.ones((200, 200, 3)))  # Imagen de fondo (por simplicidad, es un rectángulo blanco)

# Dibujar la caja ground truth
rectA = patches.Rectangle((boxA[0], boxA[1]), boxA[2] - boxA[0], boxA[3] - boxA[1],
                          linewidth=2, edgecolor='g', facecolor='none', label="Ground Truth")
ax.add_patch(rectA)

# Dibujar la caja de predicción
rectB = patches.Rectangle((boxB[0], boxB[1]), boxB[2] - boxB[0], boxB[3] - boxB[1],
                          linewidth=2, edgecolor='b', facecolor='none', label="Prediction")
ax.add_patch(rectB)

# Dibujar la región de intersección
if intersection_box[2] > intersection_box[0] and intersection_box[3] > intersection_box[1]:
    intersection_rect = patches.Rectangle((intersection_box[0], intersection_box[1]),
                                          intersection_box[2] - intersection_box[0],
                                          intersection_box[3] - intersection_box[1],
                                          linewidth=2, edgecolor='r', facecolor='red', alpha=0.3, label="Intersection")
    ax.add_patch(intersection_rect)

plt.legend(loc='upper right')
plt.show()
```

### Explicación del Código:
- **`bb_intersection_over_union`**: Calcula el área de intersección entre dos cajas y devuelve el IoU.
- **`matplotlib.patches.Rectangle`**: Se utiliza para dibujar las cajas sobre la imagen.
- **Intersección**: Se dibuja con un color diferente para destacar la región de superposición.

### Resultado:
Este código superpone las cajas y muestra visualmente la intersección en rojo, junto con el valor del IoU en el título de la gráfica. Esto es útil para entender visualmente cómo se solapan las predicciones y los valores reales en la detección de objetos.

## Tipos de arquitecturas en detección de objetos

La detección de objetos es una tarea clave en visión por computadora que implica identificar y localizar objetos dentro de imágenes o videos. Existen diferentes arquitecturas y enfoques para realizar esta tarea, cada uno con sus propias ventajas y desventajas. Aquí te presento algunos de los tipos más comunes de arquitecturas en detección de objetos:

### 1. **Modelos Basados en Regiones (Region-based Models)**

- **R-CNN (Region-based Convolutional Neural Networks):**
  - Utiliza un algoritmo de propuesto de regiones para generar regiones de interés (proposals) y luego aplica una CNN para clasificar estas regiones.
  - Se compone de tres partes: generación de propuestas, extracción de características mediante una CNN y clasificación de las propuestas.

- **Fast R-CNN:**
  - Mejora el R-CNN al aplicar la CNN a toda la imagen primero y luego extraer características para las regiones propuestas.
  - Usa una red compartida, lo que reduce el tiempo de computación.

- **Faster R-CNN:**
  - Introduce una red de propuestas de región (RPN) que comparte características con la CNN, mejorando la velocidad y precisión en comparación con Fast R-CNN.

### 2. **Modelos Basados en Cuadros (Single Shot Models)**

- **YOLO (You Only Look Once):**
  - Un modelo de detección en tiempo real que divide la imagen en una cuadrícula y predice bounding boxes y probabilidades para cada celda en un solo paso.
  - Es conocido por su velocidad y eficiencia, lo que lo hace adecuado para aplicaciones en tiempo real.

- **SSD (Single Shot MultiBox Detector):**
  - Similar a YOLO, pero utiliza diferentes escalas de características para detectar objetos de diferentes tamaños.
  - Predice bounding boxes y probabilidades de clase a partir de múltiples capas de la red.

### 3. **Modelos Basados en Cuadrículas (Grid-based Models)**

- **RetinaNet:**
  - Combina características de modelos de un solo disparo y modelos basados en regiones.
  - Introduce un enfoque de "focal loss" para abordar el problema del desbalance en clases durante el entrenamiento, enfocándose más en los ejemplos difíciles de clasificar.

### 4. **Modelos de Detección Basados en Redes de Puntos (Point-based Models)**

- **CenterNet:**
  - Detecta objetos mediante la identificación de sus centros y luego predice el tamaño y otras propiedades a partir de esta información.
  - Se basa en una red de puntos que identifica las ubicaciones de los objetos.

### 5. **Modelos de Detección con Redes Generativas (Generative Models)**

- **DETR (Detection Transformer):**
  - Un enfoque más reciente que utiliza arquitecturas de transformadores, eliminando la necesidad de técnicas de propuesto de regiones.
  - Utiliza una arquitectura de atención para directamente detectar objetos a partir de la representación de la imagen.

### 6. **Redes Convolucionales para Segmentación (Segmentation Models)**

- **Mask R-CNN:**
  - Una extensión de Faster R-CNN que agrega una rama para la segmentación de instancias, permitiendo la detección y la segmentación simultáneamente.
  - Es útil para tareas donde se necesita no solo la ubicación, sino también la forma precisa de los objetos.

### 7. **Modelos Híbridos**

- **Cascade R-CNN:**
  - Mejora la precisión a través de un enfoque en cascada que utiliza múltiples etapas de detección, refinando las predicciones en cada etapa.

### Consideraciones Finales

Cada una de estas arquitecturas tiene sus propias características y es adecuada para diferentes aplicaciones y requisitos, como velocidad, precisión y complejidad del modelo. La elección del modelo adecuado dependerá de factores como la naturaleza de los datos, la disponibilidad de recursos computacionales y los requisitos de la aplicación específica.

## Arquitecturas relevantes en object detection

Aquí tienes una lista de algunas de las arquitecturas más relevantes y populares en detección de objetos, junto con una breve descripción de cada una:

### 1. **R-CNN (Region-based Convolutional Neural Networks)**
- **Descripción:** Introdujo la idea de utilizar una CNN para extraer características de regiones de interés propuestas. Este enfoque utiliza técnicas como Selective Search para generar propuestas de regiones, que luego son clasificadas por una CNN.
- **Ventaja:** Alta precisión en la detección.

### 2. **Fast R-CNN**
- **Descripción:** Mejora de R-CNN que procesa toda la imagen en la CNN una sola vez y utiliza estas características compartidas para clasificar todas las propuestas de región, lo que reduce significativamente el tiempo de inferencia.
- **Ventaja:** Mayor velocidad y eficiencia en comparación con R-CNN.

### 3. **Faster R-CNN**
- **Descripción:** Introduce una Red de Propuestas de Región (RPN) que trabaja en paralelo con la red principal, lo que permite generar propuestas de forma más rápida y eficiente.
- **Ventaja:** Combina velocidad y precisión.

### 4. **YOLO (You Only Look Once)**
- **Descripción:** Aborda la detección de objetos como un problema de regresión en una sola red. Divide la imagen en una cuadrícula y predice bounding boxes y clases de objetos para cada celda en un solo paso.
- **Ventaja:** Extremadamente rápido y adecuado para aplicaciones en tiempo real.

### 5. **SSD (Single Shot MultiBox Detector)**
- **Descripción:** Similar a YOLO, pero utiliza múltiples capas para detectar objetos de diferentes escalas, lo que mejora la precisión en objetos pequeños.
- **Ventaja:** Balance entre velocidad y precisión.

### 6. **RetinaNet**
- **Descripción:** Combina la arquitectura de un solo disparo con un nuevo tipo de pérdida llamada "focal loss", que ayuda a manejar el desbalance entre clases en el entrenamiento.
- **Ventaja:** Alta precisión en objetos difíciles de detectar, manteniendo una velocidad razonable.

### 7. **Mask R-CNN**
- **Descripción:** Extensión de Faster R-CNN que agrega una rama para la segmentación de instancias, permitiendo no solo la detección de objetos, sino también la segmentación de sus contornos.
- **Ventaja:** Capacidad de realizar detección y segmentación simultáneamente.

### 8. **CenterNet**
- **Descripción:** Enfocado en detectar el centro de los objetos y predecir sus dimensiones. Utiliza un enfoque de detección basado en puntos en lugar de propuestas de región.
- **Ventaja:** Simplicidad y efectividad en la detección de objetos.

### 9. **DETR (Detection Transformer)**
- **Descripción:** Un enfoque más reciente que utiliza arquitecturas de transformadores para detectar objetos directamente a partir de la representación de la imagen, eliminando la necesidad de propuestas de región.
- **Ventaja:** Innovador, con resultados competitivos en precisión.

### 10. **Cascade R-CNN**
- **Descripción:** Mejora la precisión utilizando múltiples etapas de detección en cascada, refinando las predicciones en cada etapa.
- **Ventaja:** Precisión mejorada a través de un enfoque estructurado.

### 11. **EfficientDet**
- **Descripción:** Basado en EfficientNet, optimiza la arquitectura para detección de objetos. Utiliza una combinación de técnicas de escalado para mejorar la eficiencia.
- **Ventaja:** Alta precisión con menor costo computacional.

### 12. **Focal Loss**
- **Descripción:** No es una arquitectura en sí, pero es una técnica que se utiliza en modelos como RetinaNet para abordar el problema del desbalance en la detección de objetos.
- **Ventaja:** Mejora el rendimiento en clases difíciles de detectar.

### Consideraciones Finales
La elección de la arquitectura adecuada depende de varios factores, incluyendo la precisión requerida, la velocidad de inferencia, y los recursos computacionales disponibles. Las arquitecturas más recientes, como DETR y EfficientDet, están ganando popularidad debido a su rendimiento competitivo y eficiencia.

**Lecturas recomendadas**

[Fast R-CNN](https://arxiv.org/abs/1504.08083)

[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)