# Curso de Redes Neuronales Convolucionales con Python y Keras

1. **De todas las opciones esta NO es un campo de computer vision:**
   
**R//=** Regresión

2. **¿Qué es una red neuronal convolucional?**
 
**R//=** Una red neuronal artificial capaz de simular la corteza visual de nuestro cerebro.

3. **Keras maneja un tipo específico de capa para realizar convoluciones en su librería layers. ¿Cuál es esta capa?**
   
**R//=** Conv2D

4. **Una máquina puede interpretar una imagen porque:**
   
**R//=** Los pixeles de la imagen se encuentran en forma numérica, lo que permite ver a la máquina una matriz de datos.

5. **¿Cuál es la razón por la que es preferible manejar imágenes en escala de grises siempre y cuando el color no importe?**
    
**R//=** Se maneja una única capa de color lo que hace que la red sea menos compleja.

6. **¿Cuántas capas manejan las imágenes a color?**
    
**R//=** 3

7. **Los pixeles que manejan las imágenes tiene un rango de colores de:**
    
**R//=** 0-255

8. **¿En esencia qué es el kernel o filtro?**
    
**R//=** Una matriz que se desliza a través de la imagen o input de la red neuronal.

9. **¿Cual seria un filtro adecuado para detectar bordes horizontales?**
    
**R//=** [[-1,-1,-1], [0,0,0], [1,1,1]]

10. **¿A qué hace referencia el padding?**
    
**R//=** Agregar un borde externo alrededor del input completado con ceros.

11. **¿Cuál es el objetivo de padding?**
    
**R//=** Mantener todos los detalles de la imagen y conservar su tamaño (ancho x largo).

12. **A mayor número en strides, menor es la dimensión resultante (ancho x largo) en el output de la convolución. Esto es:**
    
**R//=** Verdadero

13. **Los únicos valores posibles en la capa de convolución de Keras asociados a padding son “valid” y “same”. Esto es:**
    
**R//=** Verdadero

14. **¿Cuál es el objetivo de la capa de pooling?**
    
**R//=** Reducir el número de parámetros de la red y a su vez su complejidad.

15. **¿En esencia qué hace la capa de pooling?**
    
**R//=** Reducir el tamaño de la imagen (ancho x largo) sin perder las características que la componen.

16. **La capa de flatten sirve para:**
    
**R//=** Llevar el input de múltiples dimensiones a una salida de una única dimensión.

17. **Podemos decir que comúnmente entre la red neuronal sea más profunda, aumenta la dimensión (ancho x largo). Esto es:**
    
**R//=** Falso

18. **¿En qué consiste data augmentation?**
    
**R//=** Crear nuevas imágenes basadas en el set de entrenamiento con leves cambios como rotación, zoom, brillo, etc.

19. **NO es una opción válida para fill_mode en data augmentation:**
    
**R//=** black

20. **¿En qué consiste el early stopping?**
    
**R//=** Detener el entrenamiento después de n épocas donde no hubo mejoras.

21. **El checkpoint funciona para guardar los pesos del modelo de la época con mejor desempeño en un archivo. Esto es:**
    
**R//=** Verdadero

22. **¿Normalizar los datos es una técnica para que cada píxel tenga una misma distribución?**
    
**R//=** Verdadero

23. **Con normalizar nuestros datos de entrada se busca:**
    
**R//=** Que la red converja más rápido al estar todos los datos en una misma distribución.

24. **Aplicar batch normalization con Keras se logra con:**
    
**R//=** tf.keras.layers.BatchNormalization()

25. **¿Para redes neuronales es posible leer las imágenes directamente de un directorio sin necesidad de cargarlas a un tensor previamente?**
    
**R//=** Verdadero

26. **¿A qué objeto se aplican los métodos de checkpoints y early stopping en Keras?**
    
**R//=** callbacks

27. **¿Qué sistema usa como backend la API de Keras?**
    
**R//=** TensorFlow