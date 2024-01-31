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