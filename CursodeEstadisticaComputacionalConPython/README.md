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

