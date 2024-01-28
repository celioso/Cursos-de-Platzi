### Curso de Estadística Computacional con Python

**Programación Dinámica**

**Introducción a la Programación Dinámica**

En la década de los 50s Richard Bellman necesitaba financiamiento del gobierno para poder continuar con sus investigaciones, por lo que necesitaba un nombre rimbombante para que no fueran capaz de rechazar su solicitud, por lo que eligió **programación dinámica**. Las propias palabras de Bellman fueron:

“[El nombre] *Programación Dinámica se escogió para esconder a patrocinadores gubernamentales el hecho que en realidad estaba haciendo Matemáticas. La frase Programación Dinámica es algo que ningún congresista puede oponerse.”* - Richard Bellman.

Ya sabiendo que *Programación Dinámica* no esta relacionado con su nombre, lo cierto es que si es una de las técnicas mas poderosas para poder optimizar cierto tipos de problemas.

Los problemas que puede optimizar son aquellos que tienen una **subestructura óptima**, esto significa que una **solución óptima global** se puede encontrar al combinar **soluciones óptimas de subproblemas locales**.

También nos podemos encontrar con los **problemas empalmados**, los cuales implican resolver el mismo problema en varias ocasiones para dar con una solución óptima.

Una técnica para obtener una alta velocidad en nuestro programa es la Memorización, el cual consiste en guardar cómputos previos y evitar realizarlos nuevamente. Normalmente se utiliza un diccionario, donde las consultas se pueden hacer en `O(1)`, y para ello hacemos un cambio de tiempo por espacio.

### Caminos Aleatorios

Los *caminos aleatorios* son modelos matemáticos que describen el movimiento de una partícula u objeto que se desplaza en pasos sucesivos de manera aleatoria. Estos modelos se utilizan en una amplia variedad de campos, incluyendo la física, la biología, la economía y la informática, entre otros. Aquí te doy una breve descripción y algunas aplicaciones de los caminos aleatorios:

#### Descripción básica:

- **Movimiento Aleatorio:** En un camino aleatorio, la partícula u objeto se mueve en cada paso en una dirección aleatoria, determinada por alguna distribución de probabilidad.

- **Pasos Discretos o Continuos:** Dependiendo del contexto, los caminos aleatorios pueden modelar movimientos en pasos discretos (por ejemplo, caminar sobre una cuadrícula) o movimientos continuos (por ejemplo, el movimiento browniano en física).

- **Evolución Temporal:** La evolución del camino aleatorio se describe a lo largo del tiempo, mostrando cómo se mueve la partícula en función de los pasos anteriores.

#### Aplicaciones:
1. **Física:** Los caminos aleatorios son utilizados para modelar el movimiento browniano de partículas en un fluido, la difusión de moléculas en una solución, y otros fenómenos relacionados con la aleatoriedad en la física estadística.

2. **Biología:** Se aplican en estudios de movimiento animal, migración de especies, búsqueda de alimento, y difusión de moléculas en sistemas biológicos.

3. **Economía y Finanzas:** En finanzas, los caminos aleatorios se utilizan para modelar la evolución de los precios de los activos financieros en mercados que se suponen eficientes y aleatorios.

4. **Ciencias de la Computación:** Los caminos aleatorios son fundamentales en algoritmos de simulación, optimización estocástica, y en problemas de muestreo aleatorio.

5. **Redes Complejas:** Se aplican en la teoría de grafos para modelar el comportamiento de caminos o trayectorias aleatorias en redes complejas como redes sociales, redes de comunicación, etc.

6. **Procesos Estocásticos**: Los caminos aleatorios son un ejemplo de un proceso estocástico, y como tal, son fundamentales en la teoría de probabilidad y en la modelización de sistemas dinámicos.

En resumen, los caminos aleatorios son una herramienta poderosa y versátil que se utiliza en una amplia gama de disciplinas para modelar y entender el comportamiento de sistemas complejos y estocásticos.

#### ¿Qué son los caminos aleatorios?

Los caminos aleatorios son un tipo de simulación que elige aleatoriamente una decisión dentro de un conjunto de decisiones válidas. Se utiliza en muchos campos del conocimiento cuando los sistemas no son deterministas e incluyen elementos de aleatoriedad.

### programación estocástica

La *"programación estocástica"* es un término que puede referirse a un enfoque de programación que incorpora elementos de aleatoriedad o incertidumbre en sus procesos y decisiones. Este enfoque es especialmente útil cuando se trabaja con sistemas que involucran comportamientos aleatorios o variables inciertas.

Aquí hay algunas áreas donde se utiliza la programación estocástica:

1. **Simulación Estocástica:** Se utilizan modelos estocásticos para simular sistemas y procesos que involucran variables aleatorias. Esto puede ser útil para prever el comportamiento de sistemas complejos en los que no todas las variables son completamente conocidas.

2. **Optimización Estocástica:** En lugar de trabajar con problemas de optimización determinista, donde todas las variables están definidas con precisión, la optimización estocástica aborda problemas donde algunas variables pueden ser aleatorias o inciertas. Se pueden emplear técnicas como la programación estocástica para encontrar soluciones óptimas o cercanas a óptimas en tales situaciones.

3. **Procesos Estocásticos:** Estudio de sistemas que evolucionan en el tiempo de manera probabilística. Esto incluye áreas como la teoría de colas, procesos de Markov, cadenas de Markov, entre otros, donde se modelan sistemas que cambian de estado de manera aleatoria.

4. **Aprendizaje Automático Estocástico**: En muchos algoritmos de aprendizaje automático, como el descenso de gradiente estocástico utilizado en el entrenamiento de redes neuronales, se introducen elementos de aleatoriedad para hacer más eficiente el proceso de aprendizaje o para abordar problemas con grandes volúmenes de datos.

En resumen, la programación estocástica es un enfoque flexible y poderoso que permite modelar y resolver una amplia gama de problemas en los que la aleatoriedad o la incertidumbre juegan un papel crucial.

### Cálculo de Probabilidades

La **probabilidad** es una medida de la certidumbre asociada a un evento o suceso futuro y suele expresarse como un número entre 0 y 1. Una *probabilidad* de 0 significa que un suceso jamás sucederá, y en su contraparte una *probabilidad* de 1 significa que está garantizado que sucederá.

Al hablar de *probabilidad* preguntamos qué fracción de todos los posibles eventos tiene la propiedad que buscamos, por eso es importante poder calcular todas las posibilidades de un evento para entender su probabilidad. La probabilidad de que un *evento suceda* y de que* no suceda* es siempre **1**.

- **Ley del complemento:**
 - P(A) + P(~A) = 1
- **Ley multiplicativa:**
 - P(A y B) = P(A) * P(B)
- **Ley aditiva:**
 - Mutuamente exclusivos: P(A o B) = P(A) + P(B)
 - No exclusivos: P(A o B) = P(A) + P(B) - P(A y B)

Para ver un ejemplo práctico de las leyes anteriores vamos a realizar un ejercicio de el lanzamiento de un dado de 6 caras:

- La probabilidad de que salga el número **1**:
Tenemos **6** posibilidades y el número **1** es una de ellas, por lo que la probabilidad es **1/6**.

- La probabilidad de que nos toque el numero **1** o **2**:
Tenemos **6** posibilidades y el número **1** es una de ellas y el **2** es otra. El que nos toque un número es **mutuamente exclusivo**, ya que no podemos obtener 2 números al mismo tiempo. Bajo esta premisa utilizaremos la **ley aditiva mutuamente exclusiva**.

`P(1 o 2) = P(1) + P(2)` 
`P(1 o 2) = 1/6 + 1/6`
`P(1 o 2) = 2/6`

- La probabilidad de que nos salga el número **1** al menos **1 vez** en **10** **lanzamientos**:

Para cada lanzamiento tenemos la posibilidad de **1/6** de que nos toque **1**, por lo que utilizamos la **ley multiplicativa**.

`(1/6)^10 = 0.8333`

### inferencia estadística

La *inferencia estadística* es una rama de la estadística que se centra en el uso de muestras para hacer conclusiones o inferencias sobre una población más amplia. Implica el uso de técnicas y métodos estadísticos para estimar parámetros desconocidos, probar hipótesis y tomar decisiones basadas en los datos observados en la muestra.

Aquí hay algunos conceptos clave relacionados con la inferencia estadística:

#### Estimación de parámetros:

- **Estimación puntual:** Consiste en proporcionar un único valor como estimación del parámetro desconocido de la población, utilizando estadísticas descriptivas como la media muestral, la mediana o la moda.

- **Intervalo de confianza:** Proporciona un rango de valores dentro del cual se cree que se encuentra el parámetro de la población con cierto nivel de confianza. Se calcula a partir de la muestra y tiene en cuenta la variabilidad de los datos.

### Pruebas de hipótesis:
- **Hipótesis nula y alternativa:** En una prueba de hipótesis, se establece una hipótesis nula (H0), que generalmente representa la afirmación de que no hay efecto o diferencia, y una hipótesis alternativa (H1), que afirma lo contrario.

- **Estadístico de prueba:** Es una medida calculada a partir de los datos de la muestra que se utiliza para evaluar la veracidad de la hipótesis nula.

- **Valor p:** Es la probabilidad de obtener un estadístico de prueba al menos tan extremo como el observado, si la hipótesis nula es verdadera. Se compara con un nivel de significancia predefinido para decidir si se rechaza o no la hipótesis nula.

### Tipos de inferencia estadística:
- **Inferencia paramétrica:** Se basa en supuestos sobre la distribución de los datos en la población, como la normalidad, y utiliza parámetros poblacionales para hacer inferencias.

- **Inferencia no paramétrica:** No requiere supuestos específicos sobre la distribución de los datos en la población y se basa en métodos que no dependen de parámetros poblacionales.

### Aplicaciones:

- La inferencia estadística se utiliza en una amplia gama de campos, incluyendo la medicina, la economía, la ciencia política, la ingeniería, entre otros, para tomar decisiones informadas basadas en datos observados en muestras.

- En medicina, por ejemplo, se utilizan pruebas de hipótesis para evaluar la efectividad de nuevos tratamientos o para determinar si hay una asociación entre ciertos factores y enfermedades.

- En negocios, la inferencia estadística se utiliza para tomar decisiones sobre precios, marketing, gestión de recursos humanos y planificación estratégica, entre otros aspectos.

La inferencia estadística es una herramienta fundamental en el análisis de datos y la toma de decisiones en un entorno de incertidumbre, permitiendo a los investigadores y profesionales obtener conclusiones significativas a partir de muestras limitadas de datos.

### distribución normal


La *distribución normal*, también conocida como distribución gaussiana, es una de las distribuciones de probabilidad más importantes en estadística y teoría de la probabilidad. Se caracteriza por su forma de campana simétrica y su centro en el valor medio. La función de densidad de probabilidad de una distribución normal está dada por la fórmula:

![Ecuacion de distribucion normal](data:image/jpeg;base64,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 "Ecuacion de distribucion normal")

Donde:


- x es la variable aleatoria.

- μ es la media (valor esperado) de la distribución.

- σ es la desviación estándar, que determina la dispersión de los datos alrededor de la media.

La distribución normal es importante en estadística porque muchos fenómenos naturales y sociales tienden a seguir este patrón. Además, tiene propiedades matemáticas útiles, como la propiedad de ser completamente definida por su media y desviación estándar, y la forma en que se relaciona con otras distribuciones estadísticas en el contexto del teorema del límite central.

### Muestreo

El *muestreo* es muy importante cuando no tenemos acceso a toda la población que queremos explorar. Uno de los grandes descubrimientos de la estadística es que las muestras aleatorias tienden a mostrar las mismas propiedades de la población objetivo. Hasta este punto todos los **muestreos** que hemos hecho son de tipo **probabilísticos**.

En un **muestreo aleatorio** cualquier miembro de la población tiene la misma probabilidad de ser escogido.

En un **muestreo estratificado** tomamos en consideración las características de la población para partirla en subgrupos y luego tomamos muestras de cada subgrupo, esto incrementa la probabilidad de que el muestreo sea representativo de la población.