# Curso de Análisis Exploratorio de Datos

## Por que debería hacer un análisis exploratorio de datos?
-  **Organizar y entender las variables**: podrás identificar los diferentes tipos de variables, las categorías a la que pertenecen y el tipo de análisis que puedes realizar sobre ellas.
- **Establecer relaciones entre las variables**
- **Encontrar patrones ocultos en los datos:** podrás encontrar información o comportamientos relevantes cuando hagas el EDA.
- **Ayuda a escoger el modelo correcto para la necesidad correcta:** una vez encuentres como están relacionadas las variables podrás descubrir las variables que mas se ajustan a un tipo de modelo y de esta manera eligiras el correcto
- **Ayuda a tomar decisiones informadas:** decisiones basadas en los datos, en las relaciones que encuentres entre variables, en patrones ocultos y en los modelos que generes a través de la EDA
### Pasos de una Análisis Exploratorio de Datos
1. **Hacer preguntas sobe los datos**.  Hazte las siguientes preguntas para guiar el EDA:
	- Que te gustaria encontrar?
	- Que quisieras saber de los datos?
	- Cual es la razon para realizar el analisis?
1. ** Determinar el tamaño de los datos.** Debes responder preguntas como:
	- Cuantas observaciones existen?
	- Cuantas variables hay?
	- Necesito todas las observaciones?
	- Necesito todas las variables?
1. **Categorizar las variables**. Debes preguntarte:
	- Cuantas variables categóricas existen?
	- Cuantas variables continuas existen?
	- Como puedo explorar cada variable dependiendo de su categoría?
1. **Limpieza y validación de los datos**. En ese paso debes preguntarte:
	- Tengo valores faltantes?
	- Cual es la proporción de datos faltantes?
	- Como puedo tratar a los datos faltantes?
	- Cual es la distribución de los datos?
	- Tengo valores atipicos?
1. **Establecer relaciones entre los datos.** Responde preguntas como:
	- Existe algun tipo de relacion entre mi variable X y Y?
	- Que pasa ahora si considero la variable Z en el analisis?
	- Que significa que las observaciones se agrupen?
	- Que significa el patron que se observa?

    ## Tipos de Datos Y Análisis de variables
**Tipos de análisis:**

- **Análisis Univariado:** analiza a cada variable por separado, entender sus característica.
- **Análisis Bivariado:** analiza la relacion de cada par de variables. Permite buscar relaciones intrinsecas entre los datos
- **Análisis Multivariado:** analiza el efecto simultaneo de multiples variables. Analiza la variables como un conjunto

## Cualitativos
**Categóricos**
Este tipo de datos representa las características de un objeto; por ejemplo, género, estado civil, tipo de dirección o categorías de las películas. Estos datos a menudo se denominan conjuntos de datos cualitativos en estadística.

Una variable que describe datos categóricos se denomina variable categórica. Estos tipos de variables pueden tener uno de un número limitado de valores. Es más fácil para los estudiantes de informática entender los valores categóricos como tipos enumerados o enumeraciones de variables. Hay diferentes tipos de variables categóricas:

- **Ordina**l
En las escalas ordinales, el orden de los valores es un factor significativo.
Una encuesta donde se me muestran 5 valores y debo de escoger uno de ellos

- **Nominal**
Estos se practican para etiquetar variables sin ningún valor cuantitativo. Las escalas se conocen generalmente como etiquetas. Y estas escalas son mutuamente excluyentes y no tienen ninguna importancia numérica. Veamos algunos ejemplos:
Genero, los idiomas que se hablan en un país en particular, Especies biológicas, Partes de la oración en gramática, Rangos taxonómicos en biología
Las escalas nominales se consideran escalas cualitativas y las medidas se toman utilizando las escalas cualitativas.
Ejemplo, podría ser una escala para evaluar, de cinco valores ordinales diferentes: totalmente de acuerdo / de acuerdo / neutral / en desacuerdo / totalmente en desacuerdo.
Este tipo de escala son llamadas Likert., para este tipo de datos, se permite aplicar la mediana como medida de tendencia central; sin embargo, el promedio no esta permitido.

- **Interval**

En las escalas de intervalo, tanto el orden como las diferencias exactas entre los valores son significativos. Las escalas de intervalo se utilizan ampliamente en estadística, por ejemplo, en la medida de las tendencias centrales: media, mediana, moda y desviaciones estándar. Los ejemplos incluyen la ubicación en coordenadas cartesianas y la dirección medida en grados desde el norte magnético. La media, la mediana y la moda están permitidas en datos de intervalo.

- **Ratio**
Contienen orden, valores exactos y cero absoluto, lo que permite su uso en estadísticas descriptivas e inferenciales. Estas escalas ofrecen numerosas posibilidades para el análisis estadístico. Las operaciones matemáticas, la medida de las tendencias centrales y la medida de la dispersión y el coeficiente de variación también se pueden calcular a partir de tales escalas.
Los ejemplos incluyen una medida de energía, masa, longitud, duración, energía eléctrica, ángulo plano y volumen.

## Cuantitativos
**Numéricos**
Estos datos tienen un sentido de medición involucrado en ellos; por ejemplo, la edad, la altura, el peso, la presión arterial, la frecuencia cardíaca, la temperatura, el número de dientes, el número de huesos y el número de miembros de la familia de una persona. Estos datos a menudo se denominan datos cuantitativos en las estadísticas. El conjunto de datos numérico puede ser de tipo discreto o continuo.

- **Discreto** : Altura, peso, longitud, volumen, temperatura, humedad, edad.
Estos son datos que son contables y sus valores se pueden enumerar. Por ejemplo, si lanzamos una moneda, el número de caras en 200 lanzamientos de moneda puede tomar valores de 0 a 200 casos (finitos).
Una variable que representa un conjunto de datos discreto se denomina variable discreta. La variable discreta toma un número fijo de valores distintos. Por ejemplo, la variable País puede tener valores como Nepal, India, Noruega y Japón. La variable Rango de un alumno en un aula puede tomar valores de 1, 2, 3, 4, 5, etc.

- **Continuo :** número de amigos, calificación.
Una variable que puede tener un número infinito de valores numéricos dentro de un rango específico se clasifica como datos continuos. Una variable que describe datos continuos es una variable continua. Por ejemplo, ¿cuál es la temperatura de tu ciudad hoy?

## Recolección de datos, limpieza y validación
“Forma de recolección de información que permite obtener conocimiento de primera mano e ideas originales sobre el problema o investigación”

**Tipos de recolección de datos**

1. Primaria: Datos colectados de primera mano a través de encuestas, entrevistas, experimentos y otros.

3. Secundaria: Datos previamente recolectados por una fuente primaria externa al usuario primario.Por ejemplo, datos de departamentos de gobierno o empresas similares a la del usuario primario.

5. Terciaria: Son datos que se adquieren de fuentes completamente externas al usuario primario. Por ejemplo, a través de proveedores de datos.

**¿Qué es la validación de datos?**

“El proceso de asegurar la consistencia y precisión dentro de un conjunto de datos.”
“Si los datos no son precisos desde el comienzo, los resultados definitivamente no serán precisos.”

**¿Qué se debe validar para asegurar consistencia?**

	- Modelo de datos.
	- Seguimiento de formato estándar de archivos.
	- Tipos de datos.
	- Rango de variables.
	- Unicidad.
	- Consistencia de expresiones.
	- Valores nulos.

    ## Estadistica Descriptiva Aplicada: Medidas de Dispersion
- **Rango**: La diferencia entre el valor maximo y minimo de los datos. Da una idea de que tan dispersos estan los datos
- **Rango Intercuartilico:** Comprende el 25%, tanto arriba como abajo, de los datos respecto a la mediana. Divide el rango de los datos en 4 partes iguales y considera solo el 50% de los datos
- **Desviacion Estandar:** Ofrece la dispersion media de una variable. Si a la media de una distribucion Normal se le suma, por arriba y por debajo, la desviacion estandar se obtiene un rango que contiene el 65% de los datos. Si se suma dos desviaciones estandar se obtiene el 95% de los datos. Si se suma tres desviaciones estandar se obtiene el 99% de los datos
![](https://static.platzi.com/media/user_upload/graph20-2da3c616-b5cf-4afe-b7d8-781d02f28d08.jpg)
## Asimetría Estadística

Esta relacionado con la simetria de la distribucion
- Si media = mediana = moda implica que la distribucion es simetrica.
- Si media > mediana > moda, entonces La distribución esta sesgada hacia la izquierda. (Sesgo positivo)
- Si media < mediana < moda, entonces la distribucion esta sesgada hacia la derecha. (Sesgo negativo)

![Asimetia estadistica](https://static.platzi.com/media/user_upload/graph21-d21e5c5d-7810-498e-a7d5-b763e8ebc152.jpg "Asimetia estadistica")

## Cutorsis
Es un estadístico de que tan juntos o que tan dispersos están los datos respecto a la media.

- Si Cutorsis = 0, los datos estan distribuidos homogeneamente alrededor de la media (Distribucion Mesocurtica)
- Si Cutorsis > 0, los datos estan concentrados alrededor de la media (Distribucion Leptocurtica)
- Si Cutorsis < 0, los datos estan alejados de la media (Distribucion Platicurtica)

![Cutorsis](https://static.platzi.com/media/user_upload/graph22-c7755fff-5b0f-42cb-83f1-70ce0c869cb9.jpg "Cutorsis")

## Analisis de Regresion Simple
Permite medir la fuerza del efecto en los datos mediante el ajuste de una linea recta. Es mucho mas efectivo para interpretar el comportamiento de los datos.

**Valores obtenidos en el análisis de regresión simple:**

- **Slope** (es el efecto que tiene la correlación en caso de que exista).
- Intercept (indica en donde se corta el eje de las y, porque estamos ajustando una línea).
- **Rvalue **(indica cuanto de nuestra variabilidad de los datos estamos capturando con la regresión lineal, podemos ver representado el coeficiente de correlación en este valor aunque con un poco de ruido).
- **Pvalue** (indica si la regresión lineal es significativa)
- Y otra serie de parámetros que no forman parte del objeto de estudio de esta clase

## Que hacer cuando tengo muchas variables?
Cuando se tiene muchas variables, un analisis de pares de variables puede ser confuso por lo que tenemos que recurrir a tecnicas que nos ayudan a entender la variacion de todos los datos de manera simple: Reduciendo las dimensiones para obtener un unico espacio (Pasar de 10 variables a solo 2). Algunas de estas tecnicas son:

- **Analisis de Componentes Principales (PCA):** un ejemplo de utilidad es la demostracion de que los genes reflejan la geografia de Europa
- **TSNE (T - Distributed Stochastic Neighbor Embedding):** Separacion de todos los tipos de cancer
- **UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction):** intenta capturar la estructura global preservando la estructura local de los datos utlizando proyecciones en un plano
- **Comparacion:** algoritmo de reduccion de dimension vs conjunto de datos