# Curso de Detección y Segmentación de Objetos con TensorFlow

1. **¿Cómo se utiliza un bounding box?**
   
**R//=** Engloba cada uno de los objetos en la imagen con un recuadro, indicando la clase a la que pertenece el objeto con un texto.

2. **¿Cómo puedes realizar un sliding window?**
 
**R//=** Dada una imagen, se crea una ventana que va recorriendo deslizándose todo el espacio de la imagen.

3. **¿Qué utilidad tenía la red de la que se crea el backbone para el modelo de object detection?**
   
**R//=** El modelo del que se saca el backbone es una red de clasificación pre-entrenada para una gran cantidad de clases.

4. **¿En qué rango se encuentra el valor de IoU?**
   
**R//=** Entre 0 y 1, siendo 1 el valor perfecto.

5. **¿Qué tipo de arquitectura debería ser en principio más rápida: la arquitectura multietapa o la arquitectura de una etapa?**
    
**R//=** La de una etapa, porque realiza todo el procesamiento en un solo paso hacia delante de la red.

6. **De las arquitecturas presentadas de object detection de una etapa, ¿cuál tiene diferentes versiones con mejoras incrementales (v3, v4...) y sigue aun siendo mejorada por la comunidad?**
    
**R//=** YOLO, con sus diferentes versiones. YOLOv3 es una de las más famosas, aunque se ha seguido mejorando (v3, v4, v5...)

7. **¿Para qué sirve utilizar la API de object detection de TensorFlow?**
    
**R//=** Nos ofrece una serie de métodos ya desarrollados para facilitar el trabajo con datasets de object detection y problemas de object detection en general.

8. **¿Cómo podemos trabajar con el archivo de anotaciones que nos indica dónde están los bounding boxes en las imágenes?**
    
**R//=** Podemos utilizar Pandas para poder leer y trabajar con el archivo .csv de forma más cómoda, gracias a sus métodos ya implementados.

9. **¿Cómo se muestra un bounding box dentro de una imagen mediante código?**
    
**R//=** Utilizamos las coordenadas de la caja para definir sus puntos y además incluimos la clase y el porcentaje de confianza. Todo esto lo ofrece el método visualize_boxes_and_labels_on_image_array.

10. **¿Albumentations es compatible con diferentes frameworks de deep learning?**
    
**R//=** Verdadero. Tiene integración tanto con TensorFlow como con PyTorch, lo que hace que su integración sea muy sencilla.

11. **¿Cómo crearíamos un pipeline de Albumentations para generar las transformaciones?**
    
**R//=** Utilizamos la clase Compose y le pasamos las diferentes transformaciones que queramos aplicar, con sus parámetros.

12. **¿Para qué le pasamos los bounding boxes a nuestro objeto de Albumentations?**
    
**R//=** Le pasamos los bounding boxes para que actualice las coordenadas de los bounding boxes acordes a los cambios aplicados.

13. **¿Cómo podemos acceder a modelos pre-entrenados de object detection?**
    
**R//=** TensorFlow tiene una gran variedad de modelos pre-entrados, que se pueden descargar comprimidos. Cada uno de los modelos contiene tanto sus pesos, como la configuración de la red para realizar la carga.

14. **¿Qué clases de objetos puede detectar un modelo pre-entrenado de object detection?**
    
**R//=** El modelo pre-entrenado puede detectar objetos de las clases para las que ha sido entrenado. Por ejemplo, si se ha entrenado con COCO, podrá detectar los objetos de ese dataset.

15. **¿Cómo podemos realizar la carga de nuestro modelo pre-entrenado para hacer fine-tuning?**

**R//=** Para hacer fine-tuning debemos realizar la carga del modelo pre-entrenado restaurando solo la cabeza de regresión e inicializando a 0 la de clasificación.
    
16. **¿Qué parte de nuestro dataset de object detection utilizamos para realizar el entrenamiento?**
    
**R//=** Para el entrenamiento utilizamos la parte de training y dejamos otra parte de test para, una vez terminado el entrenamiento, comprobar el rendimiento.

17. **¿Cómo podemos añadir data augmentation a nuestro pipeline de entrenamiento en object detection?**
    
**R//=** El data augmentation se hace en cada paso de entrenamiento, realizando transformaciones diferentes en las imágenes por cada paso.

18. **¿Cómo podemos ajustar el entrenamiento según nuestras necesidades en un problema de object detection?**
    
**R//=** Podemos realizar un tuning de los hiperparámetros para poder encontrar la mejor solución.

19. **¿Puede detectar un modelo de object detection con fine-tuning las clases previas que ya conocía?**
    
**R//=** Falso, 

20. **¿Cuál es la diferencia fundamental entre segmentation y object detection?**
    
**R//=** En segmentación queremos tener todo el contexto de la imagen a nivel de pixel. Por eso debemos conocer qué hay en cada uno de ellos. En object detection la precisión no es tanta y agrupamos los objetos con bounding boxes.

21. **¿Qué tipo de segmentación utilizarías para un problema en el que queremos diferenciar 5 clases diferentes?**
    
**R//=** Segmentación semántica, porque no nos interesan los objetos diferenciados, solo las clases.

22. **¿Qué partes clave suelen tener las arquitecturas de segmentación?**
    
**R//=** Suelen tener un módulo encoder que genera las características y otro decoder que proyecta esas características.

23. **¿Cómo podría ser un dataset de segmentación?**
    
**R//=** Este dataset debería de tener un conjunto de imágenes con sus máscaras y clases asociadas.

24. **¿En qué recursos podemos descubrir datasets de segmentación?**
    
**R//=** Podemos utilizar Kaggle, TensorFlow datasets, Papers with Code y las webs oficiales para descubrir datasets.

25. **¿Cómo podemos visualizar las máscaras de segmentación correspondientes a una imagen?**
    
**R//=** Debemos utilizar un módulo especial de TensorFlow para poder realizar la visualización de las máscaras de segmentación.  no es la respuesta

26. **¿Cómo podríamos utilizar la información que se va recogiendo en el encoder para utilizarla en el decoder en un problema de segmentación?**
    
**R//=** Utilizamos skip connections para pasar información de las capas previas a las capas posteriores y así poder utilizarla.

27. **Para obtener buenos resultados en un problema de segmentación, ¿es obligatorio tener un gran tiempo de entrenamiento porque la precisión es pixel a pixel?**
    
**R//=** Falso. Depende del tipo de problema, el dataset y la arquitectura de red. Es posible que con un entrenamiento corto se puedan obtener resultados buenos si el dataset es pequeño y el conjunto de test no es muy diferente.

28. **¿Cómo podemos predecir las máscaras de segmentación una vez tenemos nuestro modelo entrenado?**
    
**R//=** Podemos utilizar el método típico predict que viene dentro del API de TensorFlow.