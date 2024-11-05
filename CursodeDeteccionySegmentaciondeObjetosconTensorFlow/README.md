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

## Utilizando un dataset de object detection

Para trabajar con un dataset de detección de objetos, necesitas cargar, preprocesar y preparar el dataset en un formato adecuado para que tu modelo de detección pueda usarlo. Aquí te explico cómo hacerlo de manera general:

### 1. Selección y carga de un dataset
Algunos de los datasets de detección de objetos más populares incluyen COCO, Pascal VOC y Open Images. Para fines prácticos, supongamos que estás trabajando con el dataset COCO.

**COCO Dataset**: Es un dataset que contiene imágenes con anotaciones de objetos en formato JSON.

### 2. Preparación del entorno
Asegúrate de tener instaladas las bibliotecas necesarias:

```bash
pip install tensorflow opencv-python matplotlib pycocotools
```

### 3. Cargar el dataset en tu script de Python
Puedes utilizar `pycocotools` para trabajar con las anotaciones de COCO:

```python
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Ruta al archivo de anotaciones y las imágenes
annotation_path = 'ruta/a/annotations/instances_train2017.json'
image_dir = 'ruta/a/images/train2017/'

# Cargar anotaciones de COCO
coco = COCO(annotation_path)

# Obtener una imagen por su ID y mostrarla
image_id = 123456  # Ejemplo de ID de imagen
image_info = coco.loadImgs(image_id)[0]
image_path = f"{image_dir}/{image_info['file_name']}"

# Leer y mostrar la imagen
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')
plt.show()

# Obtener anotaciones para la imagen seleccionada
ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
annotations = coco.loadAnns(ann_ids)

# Dibujar las cajas delimitadoras
for ann in annotations:
    bbox = ann['bbox']  # Formato [x_min, y_min, width, height]
    x, y, width, height = bbox
    plt.gca().add_patch(plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none'))

plt.imshow(image)
plt.show()
```

### 4. Preprocesamiento de datos
Es importante ajustar las imágenes y las cajas delimitadoras al tamaño de entrada requerido por tu modelo (por ejemplo, 300x300 para algunos modelos pre-entrenados).

```python
def preprocess_image(image, target_size=(300, 300)):
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0  # Normalizar a [0, 1]
    return image_normalized
```

### 5. Entrenamiento del modelo
Utiliza modelos pre-entrenados como SSD, Faster R-CNN o YOLO y adapta la última capa a tu dataset. Puedes usar bibliotecas como `TensorFlow Object Detection API` o `Detectron2`.

### 6. Métricas de evaluación
Evalúa tu modelo con métricas como **mAP (mean Average Precision)** para medir la precisión de la detección de objetos.

### Consideraciones adicionales
- **Aumentación de datos**: Aplica técnicas de aumentación como rotación, cambio de brillo, recortes y espejado para mejorar la capacidad de generalización de tu modelo.
- **Non-Max Suppression (NMS)**: Implementa NMS para eliminar cajas delimitadoras redundantes y quedarte con la detección más relevante.

Esta es una introducción a cómo trabajar con datasets de detección de objetos en Python. Si necesitas ejemplos más específicos o trabajar con otros datasets como Pascal VOC.

Datasets Generales:
COCO (Common Objects in Context):

Descripción: Es uno de los datasets más populares y ampliamente utilizados para la detección de objetos, segmentación y etiquetado de imágenes.
Contenido: Más de 330,000 imágenes con más de 2 millones de instancias de objetos y más de 80 categorías.
Aplicaciones: Detección de objetos, segmentación de instancias, keypoints (puntos clave para reconocimiento de poses).
Formato de anotación: JSON (con detalles de bounding boxes y segmentaciones).
PASCAL VOC (Visual Object Classes):

Descripción: Dataset clásico que ha sido fundamental en el desarrollo de modelos de detección de objetos.
Contenido: Imágenes de 20 clases de objetos, con anotaciones de bounding boxes y segmentación.
Aplicaciones: Ideal para entrenar y evaluar modelos de detección y clasificación.
Formato de anotación: XML.
ImageNet:

Descripción: Dataset masivo utilizado inicialmente para la clasificación de imágenes, pero extendido para la detección de objetos en ImageNet Object Detection Challenge.
Contenido: Millones de imágenes clasificadas en miles de categorías, y una parte de ellas etiquetada con cajas delimitadoras.
Aplicaciones: Clasificación de imágenes y detección de objetos.
Formato de anotación: XML (similar a PASCAL VOC).
Datasets Especializados:
KITTI:

Descripción: Dataset especializado en visión por computadora para vehículos autónomos.
Contenido: Incluye imágenes y anotaciones capturadas desde vehículos con sensores como cámaras y LIDAR. Categorías de objetos incluyen autos, peatones, ciclistas, etc.
Aplicaciones: Detección de objetos, segmentación de escenas urbanas, SLAM (localización y mapeo simultáneo).
Formato de anotación: Texto plano con datos de bounding boxes y coordenadas 3D.
nuScenes:

Descripción: Dataset avanzado para vehículos autónomos, con anotaciones más detalladas que KITTI.
Contenido: Contiene imágenes, datos de sensores múltiples (cámaras, LIDAR, radar), y anotaciones en 3D.
Aplicaciones: Percepción en vehículos autónomos, detección y seguimiento de objetos en 3D.
Formato de anotación: JSON.
VisDrone:

Descripción: Dataset orientado a imágenes y videos capturados por drones.
Contenido: Imágenes y videos de escenas al aire libre con diversas condiciones de iluminación y ángulos.
Aplicaciones: Detección de personas, autos y otros objetos en imágenes aéreas.
Formato de anotación: Texto plano, con detalles de bounding boxes y tipos de objetos.
Diferencias y Usos:
Generales: Estos datasets se utilizan para entrenar modelos que pueden generalizar en una amplia variedad de escenarios y objetos.
Especializados: Diseñados para aplicaciones concretas, como la visión en vehículos autónomos o imágenes aéreas, lo que permite un entrenamiento más preciso en esos contextos.
Elección del Dataset:
La elección del dataset depende del problema específico que quieras resolver. Por ejemplo:

COCO y PASCAL VOC son ideales para comenzar en la detección de objetos.
KITTI y nuScenes son excelentes para proyectos de vehículos autónomos.
VisDrone es útil si trabajas con imágenes capturadas desde drones.
Cada uno de estos datasets tiene su propia estructura de anotación, lo que implica adaptar tu código de preprocesamiento a su formato específico.

**Lecturas recomendadas**

[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/)

[Car Object Detection | Kaggle](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)

[Self-Driving Cars | Kaggle](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

[object-detection-II.ipynb - Google Drive](https://drive.google.com/file/d/1JGhTnZEYZoXKjkXfYTX7x2dh8EEQAgP4/view?usp=sharing)

[COCO - Common Objects in Context](https://cocodataset.org/#home)

[The PASCAL Visual Object Classes Homepage](http://host.robots.ox.ac.uk/pascal/VOC/)

[The latest in Machine Learning | Papers With Code](https://paperswithcode.com/)

[The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)

## Carga de dataset de object detection

Para cargar un dataset de detección de objetos en Python, puedes usar bibliotecas como `TensorFlow`, `PyTorch`, o herramientas específicas de carga de datos. Aquí hay una guía para cargar datasets comunes:

### 1. Usando `TensorFlow` y `tf.data.Dataset`:
Puedes cargar imágenes y anotaciones con `tf.data.Dataset` para crear un pipeline de datos eficiente.

```python
import tensorflow as tf
import os

# Ruta del dataset (imágenes y anotaciones)
image_dir = '/ruta/a/imagenes'
annotation_dir = '/ruta/a/anotaciones'

def parse_image_and_annotation(image_path):
    # Carga de la imagen
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Aquí puedes cargar y parsear las anotaciones correspondientes
    # Por ejemplo, usando un formato como COCO o Pascal VOC.

    return image, annotation  # Devolver la imagen y la anotación

# Crear un Dataset de TensorFlow
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(parse_image_and_annotation)
```

### 2. Usando `PyTorch` y `torchvision`:
La biblioteca `torchvision` proporciona métodos para cargar datasets como COCO y PASCAL VOC.

```python
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Transformaciones de imagen (puedes agregar más según tus necesidades)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Carga del dataset COCO
coco_dataset = torchvision.datasets.CocoDetection(
    root='/ruta/a/imagenes',
    annFile='/ruta/a/anotaciones/annotations.json',
    transform=transform
)

# Crear un DataLoader para iterar sobre el dataset
dataloader = DataLoader(coco_dataset, batch_size=4, shuffle=True)

# Iterar sobre el dataloader
for images, targets in dataloader:
    print("Imágenes:", images.shape)
    print("Anotaciones:", targets)
```

### 3. Usar bibliotecas especializadas como `FiftyOne`:
`FiftyOne` es una biblioteca poderosa para explorar y cargar datasets de visión por computadora.

```python
import fiftyone as fo

# Cargar el dataset COCO
dataset = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_dir="/ruta/a/dataset/coco"
)

# Visualizar el dataset en FiftyOne
session = fo.launch_app(dataset)
```

### Tips:
- **Formatos de anotación**: Asegúrate de que el formato de anotación sea compatible (COCO, PASCAL VOC, etc.).
- **Transformaciones**: Aplica transformaciones (como redimensionar o normalizar) para ajustar el dataset a tu modelo.
- **Data Augmentation**: Puedes usar bibliotecas como `Albumentations` para aumentar el dataset y mejorar la robustez del modelo.

Esto debería proporcionarte un punto de partida sólido para cargar y utilizar datasets de detección de objetos.

**Lecturas recomendadas**

[object-detection-II.ipynb - Google Drive](https://drive.google.com/file/d/1JGhTnZEYZoXKjkXfYTX7x2dh8EEQAgP4/view?usp=sharing)

[Self-Driving Cars | Kaggle](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

## Exploración del dataset de object detection

La exploración de un dataset de detección de objetos es un paso fundamental para comprender la estructura de los datos, las características de las imágenes y las etiquetas asociadas. Aquí hay una guía para explorar un dataset de detección de objetos:

### 1. Cargar el Dataset
El primer paso es cargar el dataset en tu entorno. Dependiendo del formato, podrías necesitar librerías como `pandas`, `PIL`, `matplotlib`, y `xml.etree.ElementTree` para trabajar con datos y anotaciones.

### 2. Visualizar la Estructura de los Datos
Verifica cómo se organizan las imágenes y las anotaciones:
- **COCO**: Normalmente tiene un archivo JSON que contiene las anotaciones.
- **PASCAL VOC**: Usa archivos XML para cada imagen.
- **KITTI**: Tiene archivos de texto con información sobre los objetos en las imágenes.

#### Ejemplo de exploración de un archivo COCO:
```python
import json

# Cargar archivo de anotaciones
with open('path/to/annotations.json', 'r') as f:
    coco_data = json.load(f)

# Ver la estructura básica de las anotaciones
print(coco_data.keys())
print("Número de imágenes:", len(coco_data['images']))
print("Número de anotaciones:", len(coco_data['annotations']))
```

### 3. Visualizar Imágenes con Anotaciones
Puedes usar `matplotlib` y `PIL` para mostrar imágenes y superponer las cajas delimitadoras.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Cargar imagen y anotaciones
image_path = 'path/to/image.jpg'
image = Image.open(image_path)

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

# Dibujar una caja delimitadora (ejemplo)
bbox = [xmin, ymin, width, height]  # reemplaza con los valores del dataset
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.show()
```

### 4. Análisis Estadístico del Dataset
Puedes explorar estadísticas básicas como el tamaño promedio de las cajas, el número de objetos por imagen, etc.

```python
import numpy as np

# Calcular el área de las cajas
areas = [(ann['bbox'][2] * ann['bbox'][3]) for ann in coco_data['annotations']]
print("Área promedio de las cajas:", np.mean(areas))
print("Área máxima de las cajas:", np.max(areas))
print("Área mínima de las cajas:", np.min(areas))
```

### 5. Visualización de Distribuciones
Utiliza `matplotlib` o `seaborn` para graficar distribuciones de características como el número de objetos por imagen o las clases más comunes.

```python
import seaborn as sns

num_objects = [len([ann for ann in coco_data['annotations'] if ann['image_id'] == img['id']]) for img in coco_data['images']]
sns.histplot(num_objects, kde=True)
plt.title('Distribución de objetos por imagen')
plt.show()
```

### 6. Consideraciones Adicionales
- **Etiquetas de clase**: Es útil contar la frecuencia de cada clase para detectar desequilibrios en el dataset.
- **Resolución de las imágenes**: Examina si las imágenes tienen resoluciones consistentes o variables.

Esta exploración te ayudará a comprender mejor las características del dataset, preparar el pipeline de preprocesamiento y ajustar los modelos de detección de objetos para un rendimiento óptimo.

**Lecturas recomendadas**

[object-detection-II.ipynb - Google Drive](https://drive.google.com/file/d/1JGhTnZEYZoXKjkXfYTX7x2dh8EEQAgP4/view?usp=sharing)

[Self-Driving Cars | Kaggle](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

## Visualización de bounding boxes en el dataset de object detection

Para visualizar las *bounding boxes* en un dataset de detección de objetos, necesitas cargar las imágenes junto con sus anotaciones y superponer las cajas sobre las imágenes. Aquí tienes un ejemplo general de cómo hacerlo en Python usando `matplotlib` y `PIL`:

### Paso a paso para la visualización

1. **Cargar las imágenes y las anotaciones**: Dependiendo del dataset, las anotaciones pueden estar en diferentes formatos (JSON para COCO, XML para PASCAL VOC, etc.).
2. **Dibujar las *bounding boxes* en las imágenes**: Usar `matplotlib.patches.Rectangle` para superponer las cajas en las imágenes.

### Ejemplo de código para visualizar *bounding boxes*

Este ejemplo asume que tienes un archivo JSON con anotaciones (como en el formato COCO) y que las imágenes están en una carpeta.

```python
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Ruta del archivo de anotaciones y de las imágenes
annotations_path = 'path/to/annotations.json'
images_path = 'path/to/images/'

# Cargar archivo de anotaciones
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Diccionario para mapear IDs de imágenes a rutas
image_id_to_path = {img['id']: img['file_name'] for img in annotations['images']}

# Visualizar una imagen con sus anotaciones de *bounding boxes*
image_id = annotations['annotations'][0]['image_id']  # ID de ejemplo
image_file = image_id_to_path[image_id]
image_path = images_path + image_file

# Cargar la imagen
image = Image.open(image_path)

# Dibujar la imagen y las cajas
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(image)

# Agregar las *bounding boxes*
for ann in annotations['annotations']:
    if ann['image_id'] == image_id:
        bbox = ann['bbox']  # Formato [x_min, y_min, ancho, alto]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

plt.show()
```

### Explicación de las anotaciones
- **COCO**: Las cajas están en el formato `[x_min, y_min, width, height]`.
- **PASCAL VOC**: Generalmente usan `[x_min, y_min, x_max, y_max]`, por lo que deberías adaptar el código para calcular el ancho y alto si es necesario.

### Personalización
- **Colores**: Puedes cambiar el color de las cajas con `edgecolor`.
- **Etiquetas**: Puedes agregar texto a las cajas para mostrar la clase de cada objeto usando `ax.text`.

### Visualización mejorada
Si trabajas con imágenes grandes o muchas cajas, considera agregar una leyenda o ajustar la transparencia (`alpha`) de las cajas para facilitar la visualización.

Este método es útil para verificar visualmente si las anotaciones están correctamente alineadas con los objetos en las imágenes antes de entrenar un modelo de detección de objetos.

**Lecturas recomendadas**

[object-detection-II.ipynb - Google Drive](https://drive.google.com/file/d/1JGhTnZEYZoXKjkXfYTX7x2dh8EEQAgP4/view?usp=sharing)

[Self-Driving Cars | Kaggle](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

## Aumentado de datos con Albumentation

Para aumentar datos con la biblioteca **Albumentations** en un dataset de detección de objetos, puedes utilizar transformaciones que mantengan las anotaciones de las cajas delimitadoras (bounding boxes). Albumentations es muy útil porque tiene soporte específico para bounding boxes, de modo que las transformaciones aplicadas a las imágenes se reflejan correctamente en las coordenadas de las cajas. Aquí tienes un ejemplo básico de cómo hacerlo:

1. **Instalación de Albumentations**: Primero, asegúrate de que tienes Albumentations instalado.

```bash
pip install albumentations
```

2. **Aplicación de aumentos**: Aquí te muestro cómo configurar algunas transformaciones y aplicarlas a imágenes con cajas delimitadoras.

### Ejemplo de Aumento de Datos para Detección de Objetos

```python
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Configuración de las transformaciones con Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Volteo horizontal
    A.VerticalFlip(p=0.5),    # Volteo vertical
    A.RandomBrightnessContrast(p=0.2),  # Variación de brillo y contraste
    A.Rotate(limit=30, p=0.5), # Rotación con un límite de 30 grados
    A.RandomSizedBBoxSafeCrop(width=300, height=300, p=0.5),  # Recorte de tamaño
    A.Resize(width=400, height=400),  # Redimensionamiento de la imagen
    ToTensorV2()                      # Conversión a tensor
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# Imagen de ejemplo y cajas delimitadoras
image = cv2.imread('ruta/a/imagen.jpg')  # Carga tu imagen
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Bounding boxes en formato Pascal VOC (x_min, y_min, x_max, y_max)
bboxes = [[50, 50, 200, 200], [150, 150, 300, 300]]  # Coordenadas de ejemplo
category_ids = [1, 2]  # Etiquetas de las categorías correspondientes a cada bbox

# Aplicación de las transformaciones
augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
augmented_image = augmented['image']
augmented_bboxes = augmented['bboxes']

# Visualización de la imagen aumentada y las bounding boxes
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(augmented_image.permute(1, 2, 0).cpu().numpy())  # Reordenar si es tensor
for bbox in augmented_bboxes:
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
```

### Explicación del Código

- **Transformaciones**: Definimos varias transformaciones comunes para el aumento de datos, como el volteo horizontal, ajustes de brillo y contraste, rotación, y recorte seguro.
- **Formato de Bounding Boxes**: Es importante especificar `bbox_params` con el formato adecuado, que en este caso es `'pascal_voc'` (x_min, y_min, x_max, y_max). Otros formatos disponibles son `coco` (x_min, y_min, ancho, alto) y `yolo`.
- **Visualización**: Se dibujan las bounding boxes en la imagen aumentada para verificar que las coordenadas son correctas.

Este enfoque permite que las transformaciones se reflejen tanto en las imágenes como en las coordenadas de las cajas, asegurando que el modelo entienda las variaciones en los datos sin perder precisión en las etiquetas.

**Lecturas recomendadas**

[object-detection-II.ipynb - Google Drive](https://drive.google.com/file/d/1JGhTnZEYZoXKjkXfYTX7x2dh8EEQAgP4/view?usp=sharing)

[Albumentations Documentation](https://albumentations.ai/docs/)

[Self-Driving Cars | Kaggle](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

[Self-Supervised Monocular Depth Estimation](https://www.linkedin.com/pulse/self-supervised-monocular-depth-estimation-santosh-sawant/)

## Implementando Albumentation en object detection

Aquí tienes un ejemplo detallado de cómo implementar Albumentations en un flujo de trabajo de detección de objetos. Este código simula la preparación y aumento de datos en un conjunto de datos de detección de objetos, aplicando transformaciones y asegurando que las bounding boxes se ajusten correctamente.

### Paso a Paso para Implementar Albumentations en Detección de Objetos

1. **Definir Transformaciones de Aumento**: Configura varias transformaciones de Albumentations que aplicaremos a las imágenes y a sus bounding boxes.
2. **Aplicar Transformaciones a Cada Imagen**: Usa un bucle para procesar cada imagen en el conjunto de datos, aplicando las transformaciones y manteniendo las etiquetas de las cajas delimitadoras.
3. **Visualizar**: Muestra las imágenes con bounding boxes para verificar el resultado.

### Ejemplo de Código

```python
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Configuración de transformaciones de Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.RandomSizedBBoxSafeCrop(width=256, height=256, p=0.5),
    A.Resize(width=300, height=300)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# Ejemplo de imagen y bounding boxes en formato Pascal VOC (x_min, y_min, x_max, y_max)
image = cv2.imread('ruta/a/imagen.jpg')  # Reemplaza con la ruta de tu imagen
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB si es necesario

# Definir bounding boxes y etiquetas de categorías
bboxes = [[30, 40, 200, 210], [50, 90, 150, 170]]  # Bounding boxes de ejemplo
category_ids = [1, 2]  # Etiquetas de las clases correspondientes

# Aplicar las transformaciones a la imagen y bounding boxes
augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
augmented_image = augmented['image']
augmented_bboxes = augmented['bboxes']

# Función para visualizar la imagen con bounding boxes
def visualize_bbox(img, bboxes, category_ids, category_id_to_name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    for bbox, cat_id in zip(bboxes, category_ids):
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, category_id_to_name[cat_id], color='red', 
                fontsize=12, backgroundcolor='white')
    plt.axis('off')
    plt.show()

# Diccionario para convertir IDs de categorías a nombres (opcional)
category_id_to_name = {1: 'Clase 1', 2: 'Clase 2'}

# Visualizar imagen aumentada con bounding boxes
visualize_bbox(augmented_image, augmented_bboxes, category_ids, category_id_to_name)
```

### Explicación del Código

- **Transformaciones**: Las transformaciones aplicadas incluyen rotaciones, volteos, ajuste de brillo y contraste, y un recorte seguro (mantiene las bounding boxes).
- **Bounding Boxes**: `A.BboxParams` se configura con el formato `'pascal_voc'` para que Albumentations interprete las cajas correctamente. Además, `label_fields=['category_ids']` asegura que las etiquetas de clase se mantengan junto a las bounding boxes.
- **Visualización**: La función `visualize_bbox` dibuja las bounding boxes en la imagen aumentada y, opcionalmente, muestra la clase en cada una. Esto es útil para verificar que las transformaciones mantengan la integridad de las etiquetas.

### Notas

1. **Formato de Bounding Boxes**: Albumentations admite varios formatos de bounding boxes (`'pascal_voc'`, `'coco'`, `'yolo'`). Asegúrate de usar el que corresponda a tu dataset.
2. **Uso en Lote**: Este código procesa una sola imagen, pero puedes integrarlo en un bucle para aplicarlo a todo un lote o conjunto de datos.

Este pipeline facilita la creación de variaciones en los datos de entrenamiento, lo que ayuda a mejorar la generalización del modelo en tareas de detección de objetos.

**Lecturas recomendadas**

[object-detection-II.ipynb - Google Drive](https://drive.google.com/file/d/1JGhTnZEYZoXKjkXfYTX7x2dh8EEQAgP4/view?usp=sharing)