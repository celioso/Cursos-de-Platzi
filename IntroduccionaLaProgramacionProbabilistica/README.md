### Curso de Estadística Computacional con Python

# Curso de Introducción al Pensamiento Probabilístico

### Probabilidad condicional

La probabilidad condicional es una medida de la probabilidad de que ocurra un evento, dado que otro evento ha ocurrido. Se denota como P(A∣B), que se lee como "la probabilidad de A dado B". La fórmula para calcular la probabilidad condicional es:

![Ecuacion ](https://www.probabilidadyestadistica.net/wp-content/ql-cache/quicklatex.com-1657be0e0d63499dd7b0603acaa6855c_l3.svg "Ecuacion ")

Donde:

- P(A∩B) es la probabilidad de que ocurran ambos eventos A y B simultáneamente.

- P(B) es la probabilidad de que ocurra el evento B.

La probabilidad condicional es útil en diversos contextos, como la estadística, la teoría de la información, y en problemas de toma de decisiones donde se necesita considerar la información previa para hacer predicciones o tomar acciones adecuadas.

### Teorema de Bayes

El Teorema de Bayes es un concepto fundamental en la teoría de la probabilidad que describe cómo actualizar nuestras creencias sobre la probabilidad de un evento dado después de observar nueva evidencia relevante. Se formula de la siguiente manera:

![Ecuacion de bayes](https://wikimedia.org/api/rest_v1/media/math/render/svg/9e246bd8f652b1317907a108b8cb0215977ad798 "Ecuacion de bayes")

Donde:

- P(A∣B) es la probabilidad de que ocurra el evento A dado que ha ocurrido el evento B (probabilidad condicional de A dado B).

- P(B∣A) es la probabilidad de que ocurra el evento B dado que ha ocurrido el evento A (probabilidad condicional de B dado A).

- P(A) es la probabilidad previa de que ocurra el evento A (probabilidad a priori de A).

- P(B) es la probabilidad de que ocurra el evento B.

El Teorema de Bayes es especialmente útil cuando se trata de actualizar nuestras creencias sobre la probabilidad de un evento después de observar nueva información. Es ampliamente utilizado en campos como la estadística, la inteligencia artificial, el aprendizaje automático, la medicina, entre otros.

### Análisis de síntomas

![](https://static.platzi.com/media/user_upload/Teoremas%20de%20Bayes-4d9e1a78-07f1-4452-aae5-9f76c182f3d3.jpg)

### Aplicaciones del Teorema de Bayes

El Teorema de Bayes es uno de los mecanismos matemáticos más importantes en la actualidad. A grandes rasgos, nos permite medir nuestra certidumbre con respecto a un suceso tomando en cuenta nuestro conocimiento previo y la evidencia que tenemos a nuestra disposición. El Teorema de Bayes permea en tu vida diaria, desde descubrimientos científicos hasta coches autónomos, el Teorema de Bayes es el motor conceptual que alimenta mucho de nuestro mundo moderno.

En esta lectura me gustaría darte ejemplos de cómo se utiliza en la vida moderna para que puedas comenzar a implementarlo en tus proyectos, análisis y hasta en
tu vida personal.


#### Turing y el código enigma de los Nazis

Casi todos sabemos que Alan Turing es uno de los padres del cómputo moderno; pocos saben que fue gracias a él que los aliados pudieron tener una ventaja decisiva cuando Turing logró descifrar el código enigma que encriptaba todas las comunicaciones nazis; pero aún menos saben que para romper este código utilizó el Teorema de Bayes.

Lo que hizo Turing fue aplicar el Teorema para descifrar un segmento de un mensaje, calcular las probabilidades iniciales y actualizar las probabilidades
de que el mensaje era correcto cuando nueva evidencia (pistas) era presentada.

#### Finanzas

Una de las decisiones más difíciles cuando estás manejando un portafolio de inversión es determinar si un instrumento financiero (acciones, valores, bonos, etc.) se va a apreciar en el futuro y por cuánto, o si, por el contrario se debe vender el instrumento. Los portafolios managers más exitosos utilizan el Teorema de Bayes para analizar sus portafolios.

En pocas palabras, puedes determinar las probabilidades iniciales basándote en el rendimiento previo de tu portafolio o en el rendimiento de toda la bolsa y
luego añadir evidencia (estados financieros, proyecciones del mercado, etc.) para tener una mayor confianza en las decisiones de venta o compra.

#### Derecho

El Derecho es uno de los campos más fértiles para aplicar pensamiento bayesiano. Cuando un abogado quiere defender a su cliente, puede comenzar a evaluar una probabilidad de ganar (basada en su experiencia previa, o en estadísticas sobre el número de juicios y condenados con respecto del tema legal que competa) y actualiza su probabilidad conforme vayan sucediendo los eventos del proceso jurisdiccional.
Cada nueva notificación, cada prueba y evidencia que encuentre, etc. sirve para actualizar la confianza del abogado.

#### Inteligencia artificial

El Teorema de Bayes es central en el desarrollo de sistemas modernos de inteligencia artificial. Cuando un coche autónomo se encuentra navegando en las calles, tiene que identificar todos los objetos que se encuentran en su "campo de visión" y determinar cuál es la probabilidad de tener una colisión. Esta probabilidad se actualiza con cada movimiento de cada objeto y con el propio movimiento del vehículo autónomo. Esta constante actualización de probabilidades es lo que permite que los vehículos autónomos tomen decisiones
acertadas que eviten accidentes.

En esta rama existen muchos ejemplos como para cubrirlos todos, pero quiero por lo menos mencionar algunos casos de uso: filtros de spam, reconocimiento de voz, motores de búsqueda, análisis de riesgo crediticio, ofertas automáticas, y un largo etcétera.

Para terminar, me gustaría compartir una cita del famoso economista John Maynard Keynes que resume perfectamente el tipo de pensamiento que quiero que desarrolles: "Cuando los hechos cambian, yo cambio mi opinión. ¿Qué hace usted, señor?"


# Mentiras estadísticas

## Garbage in, garbage out

La calidad de nuestros datos es igual de fundamental que la precisión de nuestros cómputos. Cuando los datos son errados, aunque tengamos un cómputo prístino nuestro resultado serán erróneos.

En pocas palabras: con datos errados las conclusiones serán erradas.

## Imágenes engañosas

Las visualizaciones son muy importantes para entender un conjunto de datos. Sin embargo, cuando se juega con la escala se puede llegar a conclusiones incorrectas.

Nunca se debe confiar en una gráfica sin escalas o etiquetas.

## Cum Hoc Ergo Propter Hoc

Dos variables están positivamente correlacionadas cuando se mueven en la misma dirección y negativamente correlacionadas cuando se mueven en direcciones opuestas. Esta correlación no implica causalidad.

Puede existir variables escondidas que generen la correlación. _Después de esto, eso; entonces a consecuencia de esto, eso._

## Prejuicio en el muestreo

Para que un muestreo pueda servir como base para la inferencia estadística tiene que ser aleatorio y representativo.

El prejuicio en el muestreo elimina la representatividad de las muestras.

A veces conseguir muestras es difícil, por lo que se utiliza a la población de más fácil acceso (caso estudios universitarios).

## Falacia del francotirador de Texas

Esta falacia se da cuando no se toma la aleatoriedad en consideración. También sucede cuando uno se enfoca en las similitudes e ignora las diferencias.

Cuando fallamos al tener una hipótesis antes de recolectar datos estamos en alto riesgo de car en esta falacia (muy común en Data Science).

## Porcentajes confusos

Cuando no sabemos la cuenta total del cual se obtiene un porcentaje tenemos el riesgo de concluir falsos resultados, siempre es importante ver el contexto, y los porcentajes, en vacio, no significan mucho.

## Falacia de regresión

Muchos eventos fluctúan naturalmente, por ejemplo, la temperatura promedio de una ciudad, el rendimiento de un atleta, los rendimientos de un portafolio de inversión, etc.

Cuando algo fluctúa y se aplican medidas correctivas se puede creer que existe un vínculo de causalidad en lugar de una regresión a la media.

# Introducción a Machine Learning

_"Es el campo de estudio que le da a las computadoras la habilidad de aprender sin ser explícitamente programadas."_ - Arthur Samuel, 1959.

- Machine learning se utiliza cuando:
    - Programar un algoritmo es imposible.
    - El problema es muy complejo o no se conocen altoritmos para resolverlo.
    - Ayuda a los humanos a entender patrones (data mining).

- Aprendizaje supervisado vs no supervisado vs semisupervisado.

- Batch vs online learning.

## Feature vectors

Se utilizan para representar características simbólicas o numéricas llamadas _features._ Permiten analizar un objeto desde una perspectiva matemática.

Los algoritmos de machine learning típicamente requieren representaciones numéricas para poder ejecutar el cómputo.

Uno de los _feature vectors_ más conocidos es la representación del color a través de RGB
- color = [R, G, B]
- Procesamiento de imágenes: Gradientes, bordes, áreas, colores, etc.
- Reconocimiento de voz: Distancia de sonidos, nivel de ruido, razón ruido / señal, etc.
- Spam: Dirección IP, estructura del texto, frecuencia de palabras, encabezados, etc.

## Métricas de distancia

Muchos de los algoritmos de machine learning pueden clasificarse como algoritmos de optimización. Lo que desean optimizar es una función que en muchas ocasiones se refiere a la distancia entre features.

# Agrupamiento

## Introducción al agrupamiento

Es un proceso mediante el cual se agrupan objetos similares en clusters que los identifican. Se clasifican como aprendizaje no supervisado, ya que no requiere la utilización de etiquetas.

Permite entender la estructura de los datos y la similitud entre los mismos.

Es utilizado en motores de recomendación, análisis de redes sociales, análisis de riesgo crediticio, clasificación de genes, riesgos médicos, etc.

## Agrupamiento jerárquico

Es un algoritmo que agrupa objetos similares en grupos llamados _clusters_. El algoritmo comienza tratando a cada objeto como un cluster individual y luego realiza los siguientes pasos de manera recursiva:
- Identifica los 2 clusters con menor distancia (lo más similares).
- Agrupa los 2 clusters en 1 nuevo.

El _output_ final es un dendrograma que muestra la relación entre objetos y grupos.

Es importante determinar qué medida de distancia vamos a utilizar y los puntos a utilizar en cada cluster (linkage criteria).

## Agrupamiento K means

Es un algoritmo que agrupa utilizando centroides. El algoritmo funciona asignando puntos al azar (K define el número inicial de clusters) y después:
- En cada iteración el punto se ajusta a su nuevo centroide y cada punto se recalcula con la distancia con respecto de los centroides.
- Los puntos se reasignan al nuevo centro.
- El algoritmo se repite de manera iterativa hasta que ya no existen mejoras.

## Otras técnicas de agrupamiento

El agrupamiento es una técnica de Machine Learning que consiste, en pocas palabras, en dividir una población en grupos con la consecuencia de que los datos en un grupo son más similares entre ellos que entre los otros grupos.

Imagina que eres el dueño de una startup que hace ecommerce y quieres tener estrategias de venta para tus clientes. Es casi imposible diseñar una estrategia por cada individuo, pero se puede utilizar el agrupamiento para dividir a los clientes en grupos que tengan similitudes relevantes y así reducir el problema a unas cuantas estrategias.

Existen dos tipos de agrupamiento:
- **Agrupamiento estricto (hard clustering):** en el cual cada dato pertenece a un grupo u otro. No hay puntos medios.

- **Agrupamiento laxo (soft clustering):** en el cual en lugar de asignar un dato a un grupo, se asigna probabilidades a cada dato de pertenecer o no a un grupo.

Un punto muy importante que debes considerar cuando ejecutas técnicas de agrupamiento es que debes definir muy claro a qué te refieres cuando hablas de similitud entre puntos, porque esto puede ayudarte a definir el algoritmo correcto para tus necesidades particulares.

A grandes rasgos existen cuatro aproximaciones para definir similitud:

- **Modelos conectivos:** Estos modelos asumen que los puntos más similares son los que se encuentran más cercanos en el espacio de búsqueda. Recuerda que este espacio puede ser altamente dimensional cuando tus _feature vectors_ definen muchas características a analizar. Una desventaja de este tipo de modelos es que no escalan para conjuntos de datos grandes (aunque es posible utilizar una muestra y aplicar técnicas de estadística inferencial para obtener resultados).

- **Modelos de centroide:** Este tipo de modelos definen similitud en términos de cercanía con el centroide del grupo. Los datos se agrupan al determinar cuál es el centroide más cercano.

- **Modelos de distribución:** Este tipo de modelos trata de asignar probabilidades a cada dato para determinar si pertenecen a una distribución específica o no (por ejemplo, normal, binomial, Poisson, etc.).

- **Modelos de densidad:** Estos modelos analizan la densidad de los datos en diferentes regiones y dividen el conjunto en grupos. Luego asignan los puntos de acuerdo a las áreas de densidad en las que se haya dividido el dataset.

Acuérdate que no tienes que casarte con un modelo específico. Muchos de los mejores Ingenieros de Machine Learning y Científicos de Datos utilizan varios modelos con el mismo conjunto de datos para analizar el rendimiento de los diversos algoritmos que tienen a su disposición. Así que experimenta y siempre compara tus resultados antes de tomar una decisión.

# Clasificación

## Introducción a la clasificación

Es el proceso mediante el cual se predice la clase de cierto dato. Es un tipo de aprendizaje supervisado, ya que para que funcione, se necesitan etiquetas con los datos (labels).

Se utiliza en muchos dominios, incluyendo la medicina, aprobación crediticia, reconocimiento de imágenes, vehículos autónomos, entre otros.

Sigue dos pasos: aprendizaje (creación del modelo) y clasificación.

## Clasificación K nearest neighbors

Parte del supuesto de que ya tenemos un conjunto de datos clasificado. Trata de encontrar los "vecinos más cercanos".

K se refiere a la cantidad de vecinos que se utilizarán para clasificar un ejemplo que aún no ha sido clasificado.

Es sencillo de implementar y tiene aplicaciones en medicina, finanzas, agricultura, etc.

Es computacionalmente muy costoso y no sirve con datos de alta dimensionalidad.

## Otras tecnicas de clasificación

La clasificación es un tipo de Machine Learning supervisado. Esto significa que para entrenar un modelo necesitamos un conjunto de datos (_dataset_) que ya tenga etiquetas (_labels_) para poder entrenar nuestros modelos.

La mejor forma de pensar en algoritmos de clasificación es pensar en el sombrero clasificador de Harry Potter. Cuando un nuevo alumno de Hogwarts entra a la escuela es necesario asignarlo/clasificarlo en una de las 4 casas. El sombrero obtiene los datos cuando se lo coloca el alumno y define cuál es el mejor match para su caso particular. Aquí estamos asumiendo que el sombrero es un algoritmo que ya ha sido entrenado y que los alumnos son nuevos _data_ points que tienen que ser clasificados.

### Clasificadores lineales

Estos tipos de clasificadores se distinguen porque dividen el conjunto de datos con una línea (que puede ser multidimensional dependiendo de la cantidad de _features_ que hemos utilizado para definir a nuestros datos). Esto genera áreas dentro de nuestro espacio de búsqueda para que cuando coloquemos un nuevo dato podamos clasificarlo fácilmente.
​
El problema con este tipo de modelos es que son pocos flexibles cuando el conjunto de datos no puede ser separado fácilmente con una simple línea; por ejemplo, cuando necesitáramos una figura más compleja para dividirlo (como un polígono).

### Regresión logística

Estos algoritmos se parecen mucho a los clasificadores lineales, con la diferencia de que no se divide simplemente con una línea, sino con un gradiente que determina la probabilidad de que un punto pertenezca a una categoría u otra. Es decir, la gradiente determina la probabilidad de que un punto sea asignado a una categoría y mientras un dato se aleje más en una dirección será mayor la probabilidad de que pertenezca a una categoría.

Imagina que estos algoritmos generan un área difusa en la que no estamos seguros de la clasificación y un área clara en la que tenemos un alto grado de certeza
en cuanto a la categoría que pertenece un punto.
​
### Nearest neighbor

Los modelos que utilizan nearest neighbor se apoyan de los datos que ya han sido clasificados para determinar la distancia entre sus “vecinos más cercanos.” El algoritmo más común que utiliza esta técnica se llama K-nearest neighbors y la K representa el número de vecinos que se utilizarán para clasificar los datos. En pocas palabras, se identifican los datos más cercanos y en el caso más sencillo se hace una votación simple (por ejemplo, 5 azules, 2 rojos, por lo tanto azul).

Una característica de estos modelos es que “dibujan” una línea que se asemeja a una costa para clasificar los datos. Mientras K sea más grande la “línea costera” se alisa y se asemeja más y más a una línea simple. Por lo tanto, la definición de K tiene un impacto importante en el desempeño de nuestro algoritmo de clasificación.

### Support Vector Machines

Estos algoritmos se diferencian por tener la habilidad de generar figuras complejas (polígonos) que pueden agrupar datos. Si la figura que tendríamos que dibujar para dividir nuestros datos es diferente a una línea (círculos, polígonos, etc.), entonces estos modelos son una buena opción.

### Árboles de decisión

Este tipo de algoritmos nos permiten generar una árbol que tenemos que recorrer y tomar decisiones cada vez que avanzamos en un nivel. Por ejemplo:
- Si un feature en análisis es mayor a 5, dibuja la línea y=2x+3, de lo contrario dibuja y=-3x+5
- Si el feature siguiente es menor a 2, dibuja otra línea y así sucesivamente.
