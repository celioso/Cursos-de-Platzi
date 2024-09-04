### Curso de Python para Ciencia de Datos

1. **¿Qué es la imputación de valores faltantes??**

**R/:** Estimar los valores ausentes con base en los valores válidos de otras variables y/o casos de muestra o modelos.

2. **Como profesional en data, podrás crear múltiples modelos para entender el comportamiento de tus datos. ¿Qué problemas puedes enfrentar al crear modelos con datos con valores faltantes?**

**R/:** Todas las opciones son correctas.

3. **¿Es posible eliminar los valores faltantes sin una revisión previa y que esto NO introduzca sesgos?**

**R/:** No, es necesario revisar los valores faltantes para saber si es posible eliminarlos.

4. **¿Cuál de las siguientes afirmaciones NO es una implicación de los datos Missing Completely At Random (MCAR)?**

**R/:** La imputación es necesaria.

5. **¿Cuál de las siguientes afirmaciones SÍ es una implicación de los datos Missing Not At Random (MNAR)?**

**R/:** Todas las opciones son correctas.

6. **Después de examinar tus datos con valores faltantes, tus colegas te sugieren mejorar tu diseño experimental o realizar un análisis de sensibilidad. ¿Qué tipo de mecanismo de valores faltantes creen que actúa sobre tus datos?**

**R/:** MNAR

7. **¿Qué es la codificación ordinal?**

**R/:** Una codificación ordinal implica mapear cada etiqueta (categoría) única a un valor entero.

8. **Una vez ajustamos y transformamos nuestros datos con codificador, podemos hacer uso de su atributo `____` para obtener las categorías únicas de nuestro conjunto de datos.**

**R/:** encoder.categories_

9. **¿Qué es la codificación one-hot encoding?**

**R/:** Convertirá cada valor categórico en una nueva columna categórica y le asignará un valor binario de 1 o 0 a esas columnas.

10. **¿Cuál es una ventaja de realizar one-hot encoding utilizando sklearn en lugar de Pandas?**

**R/:** Realizar one-hot encoding puede llegar a ser más robusto debido a que guarda la información necesaria de las categorías involucradas e incluye una forma de realizar transformaciones inversas.

11. **¿Cuándo utilizarías una codificación one-hot en lugar de una codificación ordinal?**

**R/:** Cuando mis variables categóricas no tienen un orden natural.

12. **¿Qué son las imputaciones basadas en donantes?**

**R/:** Completa los valores que faltan para una unidad dada copiando los valores observados de otra unidad, el donante.

13. **¿Cuál de las siguientes afirmaciones NO es una desventaja de la imputación por media, mediana y moda?**

**R/:** No afectará el estadístico en cuestión ni el tamaño de muestra.

14. **¿Por cuál de las siguientes razones podrías utilizar la imputación por media, mediana y moda?**

**R/:** Todas las opciones son correctas.

15. **Al realizar una imputación por llenado hacia atrás y hacia adelante, existen trucos que pueden ayudarte a enfrentar el problema de que las relaciones multivariables puedan ser distorsionadas. ¿Cuál de las siguientes opciones NO es uno de estos trucos?**

**R/:** Realizar una imputación con dominios creados completamente al azar.

16. **¿Cuál de las siguientes afirmaciones es una desventaja de la imputación por interpolación?**

**R/:** Puede introducir valores fuera de rango.

17. **Recibes datos de series temporales, ¿cuál método de Pandas podría ayudarte a probar distintos modelos de interpolación?**

**R/:** df.interpolate()

18. **¿Cuál de las siguientes afirmaciones NO es una desventaja de la imputación por K-vecinos más cercanos (KNN)?**

**R/:** Buen rendimiento con conjuntos de datos pequeños.

19. **El algoritmo de imputación por K-vecinos más cercanos (KNN) es un método de imputación con base en modelos. Esto es:**

**R/:** Falso, es basada en donante.

20. **Necesitas cuantificar la distancia únicamente entre variables numéricas. ¿Cuál de las siguientes métricas de distancia podría ser más útil?**

**R/:** Euclidiana.

21. **Necesitas cuantificar la distancia únicamente entre variables categóricas sin un orden natural. ¿Cuál de las siguientes métricas de distancia podría ser más útil?**

**R/:** Hamming

22. **Necesitas cuantificar la distancia entre variables de distintos tipos (númericas, categóricas). ¿Cuál de las siguientes métricas de distancia podría ser más útil?**

**R/:** Gower

23. **¿Cuál de las siguientes afirmaciones es una ventaja de la imputación con base en modelos?**

**R/:** Todas las opciones son correctas.

24. **¿Cuál de las siguientes afirmaciones es una desventaja de la imputación con base en modelos?**

**R/:** Todas las opciones son correctas.

25. **¿Cuál de las siguientes afirmaciones es una desventaja de Imputaciones Múltiples por Ecuaciones Encadenadas (MICE)?**

**R/:** Para funcionar bien necesitas pensar en el modelo de imputación y el modelo de análisis.

26. **¿Cuál función te permite visualizar la nulidad de tu conjunto de datos en forma de matriz?**

**R/:** missingno.matrix()

27. **¿Cuál es el papel de la función `sklearn.compose.make_column_transformer()`?**

**R/:** Transformar cada columna con un transformador asociado a esta y guardar la información necesaria para realizar el procesamiento inverso.

28. **¿Cuáles son las imputaciones basadas en modelos?**

**R/:** Encuentran un modelo predictivo para cada variable objetivo en el conjunto de datos que contiene valores faltantes.

29. **.¿La siguiente frase es verdadera o falsa?**

***Realizar transformaciones a tus datos es un paso fundamental en cada análisis. Por ejemplo, cambio de escalas, cambio de codificaciones, entre otras. No obstante, siempre es útil regresar los datos a su estado natural, ya sea para continuar explorando o comunicar los resultados de forma original.***

**R/:** Verdadera