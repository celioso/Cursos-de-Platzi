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