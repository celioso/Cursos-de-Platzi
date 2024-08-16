# Curso de Python

## ¿Por qué aprender Python?

La NASA, con su programa Artemisa, planea enviar al próximo hombre a la luna en 2025, apoyándose en proyectos como la clasificación de rocas usando Python e inteligencia artificial.

### Fundación de Python

Python es un lenguaje de programación de alto nivel que fue creado por Guido van Rossum y su primera versión fue lanzada en 1991.

Van Rossum comenzó a trabajar en Python a finales de los años 80 como un proyecto de tiempo libre para suceder al lenguaje ABC, con la intención de crear un lenguaje que fuera fácil de leer y escribir.

Python se ha desarrollado bajo una filosofía de diseño que enfatiza la legibilidad del código y la sintaxis que permite a los programadores expresar conceptos en menos líneas de código en comparación con otros lenguajes como C++ o Java.

A lo largo de los años, Python ha ganado una enorme popularidad y se ha convertido en uno de los lenguajes de programación más utilizados en el mundo. Sus aplicaciones son vastas y variadas, incluyendo desarrollo web (con frameworks como Django y Flask), análisis de datos y machine learning (con bibliotecas como Pandas, NumPy, y TensorFlow), automatización de tareas, desarrollo de software, y más.

La versatilidad y la facilidad de uso de Python lo han convertido en una herramienta esencial tanto para principiantes como para desarrolladores experimentados.

## ¿Cuáles son las ventajas de Python para los nuevos programadores?

- Simplicidad: Python sigue una filosofía de simplicidad, con una sintaxis clara y sencilla.
- Accesibilidad: Es fácil de aprender, permitiendo a los programadores enfocarse en los fundamentos de la programación.
- Versatilidad: Soporta múltiples paradigmas de programación, incluyendo la programación orientada a objetos y funcional.
- Popularidad: Usado por millones de personas, Python permite realizar proyectos complejos con pocas líneas de código.

[GitHub - platzi/python: Repositorio de ejercicios del Curso de Python de Platzi](https://github.com/platzi/python)

[Welcome to Python.org](https://www.python.org/)

[Introducción a Python para la exploración del espacio - Training | Microsoft Learn](https://learn.microsoft.com/es-es/training/modules/introduction-python-nasa/)

[3.12.4 Documentati](https://docs.python.org/3/)

## Introducción a Python

Python es uno de los lenguajes más amigables para iniciar en la programación. En comparación con otros lenguajes, en Python podemos escribir un “Hola Mundo” con una sola línea de código.

## ¿Cómo instalar Python en Windows?

- Ve al navegador y escribe “Python”.
- Asegúrate de entrar a [python.org](http://python.org/) y haz clic en la sección de descargas.
- Descarga el instalador y ábrelo.
- Marca la opción “Add Python to PATH” y selecciona “Install Now”.
- Una vez instalado, abre la terminal o cmd y escribe “python” para comprobar la instalación.

### ¿Cómo instalar Python en Mac?

- Abre tu navegador y escribe “Python”.
- Asegúrate de ingresar a [python.org](http://python.org/).
- En la sección de descargas, la página detectará tu sistema operativo MacOS automáticamente.
- Descarga el instalador, ábrelo y sigue las instrucciones: “Continuar”, “Aceptar términos y condiciones” e “Instalar”.
- Confirma que Python está instalado abriendo la terminal y escribiendo “python.

### ¿Cómo escribir el “Hola Mundo” en Python?

- Abre la terminal en tu sistema.
- Escribe python para acceder al intérprete de Python.
- Ingresa el siguiente código: print("Hola Mundo") y presiona Enter.
- Verás el mensaje “Hola Mundo” impreso en la consola.

### ¿Cómo usar Visual Studio Code para Python?

- Descarga Visual Studio Code desde [code.visualstudio.com](http://code.visualstudio.com/ "code.visualstudio.com").
- Instala el editor y ábrelo.
- Crea una nueva carpeta para tus proyectos, por ejemplo, “Curso Python”.
- Abre la carpeta en Visual Studio Code y crea un archivo nuevo con la extensión .py (por ejemplo, hola.py).
- Escribe print("Hola Mundo") en el archivo.
- Guarda el archivo (Ctrl+S) y ejecuta el código usando el botón de ejecutar en Visual Studio Code.
- Asegúrate de tener instalada la extensión de Python para una mejor experiencia de codificación.

Recuerda que Python es un lenguaje interpretado, lo que significa que las instrucciones se ejecutan directamente sin necesidad de compilación previa, facilitando la visualización inmediata de resultados.

[Our Documentation | Python.org](https://www.python.org/doc/)

[Visual Studio Code - Code Editing. Redefined](https://code.visualstudio.com/)

## Conceptos Básicos de Programación

Comprender los conceptos de semántica y sintaxis es crucial en programación. La semántica da sentido al código, asegurando consistencia y coherencia en las operaciones, mientras que la sintaxis dicta cómo escribir correctamente el código, siguiendo reglas específicas.

### ¿Qué es la semántica en programación?

La semántica en programación se refiere al significado y consistencia del código. Si trabajamos con números, tanto las entradas como las salidas deben ser números para mantener coherencia. Al nombrar variables, debemos hacerlo de manera lógica y significativa para que el código tenga sentido.

### ¿Cómo afecta la sintaxis al código?

La sintaxis en programación es el conjunto de reglas que dicta cómo escribir el código correctamente. Cada lenguaje tiene su propia sintaxis, y seguir estas reglas es esencial para evitar errores. Por ejemplo, si iniciamos con un paréntesis, debemos cerrarlo. No hacerlo resultará en errores de sintaxis que el compilador o intérprete señalará.

### ¿Cómo se manejan errores de sintaxis?

Cuando hay errores de sintaxis, el compilador o el entorno de desarrollo, como Visual Studio Code, nos indicarán dónde está el problema. Por ejemplo, al olvidar cerrar un paréntesis, obtendremos un mensaje de error que señala esta omisión. Corregir estos errores es esencial para que el código se ejecute correctamente.

### ¿Qué papel juegan las variables en la semántica?

Las variables son contenedores de información y su correcto uso es fundamental para la semántica. Al crear una variable, debemos seguir la sintaxis correcta: nombre de la variable, signo de asignación y el valor. Usar nombres descriptivos y relevantes para las variables asegura que el código sea comprensible y lógico.

### ¿Qué errores comunes pueden ocurrir con las variables?

- Usar nombres de variables que no reflejan su contenido.
- Intentar utilizar palabras reservadas del lenguaje como nombres de variables.
- Iniciar nombres de variables con números, lo cual no está permitido.
- No declarar una variable antes de su uso, lo que generará errores de tipo NameError.

### ¿Cómo nombrar correctamente las variables?

- Utilizar nombres descriptivos y significativos.
- Iniciar con letras y seguir con números o caracteres permitidos.
- Evitar palabras reservadas.
- Usar guiones bajos para separar palabras en nombres de variables.

### ¿Qué sucede al redefinir variables?

Redefinir una variable sobrescribirá su valor anterior. Python ejecuta las líneas de código de arriba hacia abajo y de izquierda a derecha, por lo que el último valor asignado a una variable será el que prevalezca.

### ¿Cómo evitar errores de semántica al nombrar variables?

Usar nombres de variables que reflejen claramente su propósito. Por ejemplo, si una variable almacena un nombre, debería llamarse nombre y no edad. Esto evita confusiones y asegura que el código sea fácil de entender y mantener.

[PEP 8: The Style Guide for Python Code](https://pep8.org/)

## Manipulación de Cadenas de Texto en Python

Entender cómo trabajar con las cadenas en Python es fundamental para la manipulación de textos y datos en muchos proyectos. Desde definir variables hasta aplicar métodos específicos, el uso de strings es una habilidad básica pero poderosa que se utiliza en áreas avanzadas como el procesamiento del lenguaje natural (NLP).

### ¿Cómo se definen las cadenas en Python?

Para crear una cadena en Python, puedes utilizar comillas simples, dobles o triples. Por ejemplo:

- Comillas simples: name = 'Carly'
- Comillas dobles: name = "Carly"
- Comillas triples: name = '''Carly'''

Las comillas triples permiten incluir saltos de línea y espacios en blanco.

### ¿Cómo se imprime y verifica el tipo de dato de una variable?

Para imprimir el valor de una variable y verificar su tipo, puedes utilizar la función print junto con la función type:

**name** = 'Carly'
**print(name)**  # Imprime 'Carly'
**print(type(name))**  # Imprime

### ¿Cómo se indexan las cadenas en Python?

Las cadenas son colecciones ordenadas y accesibles por índices. Puedes acceder a un carácter específico utilizando corchetes:

**name** = 'Carly'
**print(name[0])**  # Imprime 'C'
**print(name[-1])**  # Imprime 'y'

### ¿Qué pasa si intentas acceder a un índice que no existe en Python?

Si intentas acceder a un índice fuera del rango de la cadena, Python arrojará un `IndexError`:

**print(name[20])**  # Genera IndexError

### ¿Cómo se concatenan cadenas?

Puedes concatenar cadenas utilizando el operador `+` y repetirlas con el operador `*`:

**first_name** = 'Carly'
**last_name** = 'Marcela'
**full_name** = first_name + ' ' + last_name
**print(full_name)**  # Imprime 'Carly Marcela'

**print(name * 5)**  # Imprime 'CarlyCarlyCarlyCarlyCarly'

### ¿Cómo se consultan la longitud y otras operaciones en cadenas?

Para obtener la longitud de una cadena, se usa la función `len`:

**print(len(name))**  # Imprime 5

Además, las cadenas tienen métodos específicos como `lower()`, `upper()`, y `strip()`:

**print(name.lower())**  # Imprime 'carly'
**print(name.upper())**  # Imprime 'CARLY'
**print(last_name.strip())**  # Elimina espacios en blanco al principio y al final

### ¿Qué importancia tienen las cadenas en áreas avanzadas como el NLP?

El manejo de cadenas es esencial en NLP, donde grandes cantidades de texto deben ser limpiadas y procesadas. Métodos como `strip()`,` lower()`, y `replace()` ayudan a preparar los datos para análisis más complejos.

Aquí tienes una explicación de cada uno de estos métodos en Python, que se utilizan comúnmente con cadenas de texto y en el caso de eval() y exec(), para ejecutar código:

**Métodos de Cadenas:**

1. `.count(substring)`

 - **Descripción**: Cuenta el número de veces que aparece el substring en la cadena.
 - Ejemplo:

```python
texto = "hola hola hola"
print(texto.count("hola"))  # Salida: 3
```

2. .capitalize()

 - **Descripción**: Devuelve una copia de la cadena con el primer carácter en mayúscula y el resto en minúscula.
 - **Ejemplo:**

```python
texto = "hola mundo"
print(texto.capitalize())  # Salida: "Hola mundo"
```

3. `.title()`
 - **Descripción**: Devuelve una copia de la cadena con el primer carácter de cada palabra en mayúscula.
 - **Ejemplo:**

```python
texto = "hola mundo"
print(texto.title())  # Salida: "Hola Mundo"
```

4. `.swapcase()`

- **Descripción**: Devuelve una copia de la cadena con los caracteres en mayúscula convertidos a minúscula y viceversa.
- **Ejemplo:**

```python
texto = "Hola Mundo"
print(texto.swapcase())  # Salida: "hOLA mUNDO"
```

5. .replace(old, new)

 - **Descripción**: Devuelve una copia de la cadena con todas las ocurrencias del old reemplazadas por new.
 - **Ejemplo:**

```python
texto = "hola mundo"
print(texto.replace("mundo", "universo"))  # Salida: "hola universo"
```

6. `.split(separator)`

 - **Descripción:** Divide la cadena en una lista de subcadenas usando el separator. Si no se proporciona separator, se usa cualquier espacio en blanco como separador.
 - **Ejemplo:**

```python
texto = "hola mundo"
print(texto.split())  # Salida: ['hola', 'mundo']
```

7. `.strip()`

 - **Descripción:** Devuelve una copia de la cadena sin los espacios en blanco al principio y al final.
 - **Ejemplo:**

```python
texto = "   hola mundo   "
print(texto.strip())  # Salida: "hola mundo"
```

8. `.lstrip()`

 - **Descripción:** Devuelve una copia de la cadena sin los espacios en blanco al principio.
 - **Ejemplo:**

```python
texto = "   hola mundo"
print(texto.lstrip())  # Salida: "hola mundo"
```

9. `.rstrip()`

 - **Descripción:** Devuelve una copia de la cadena sin los espacios en blanco al final.
 - **Ejemplo:**

```python
texto = "hola mundo   "
print(texto.rstrip())  # Salida: "hola mundo"
```

10. `.find(substring)`

 - **Descripción**: Devuelve el índice de la primera aparición del substring en la cadena. Devuelve -1 si no se encuentra.
 - **Ejemplo:**

texto = "hola mundo"
print(texto.find("mundo"))  # Salida: 5

11. `.index(substring)`

 - **Descripción:** Similar a find(), pero lanza una excepción (ValueError) si el substring no se encuentra en la cadena.
 - **Ejemplo:**

```python
texto = "hola mundo"
print(texto.index("mundo"))  # Salida: 5
```

### Métodos de Evaluación de Código:

12. `eval(expression)`

 - **Descripción:** Evalúa la expresión expression que se pasa como una cadena de texto y devuelve el resultado. Es potencialmente peligroso si se ejecuta código no confiable, ya que puede ejecutar cualquier código Python.
 - **Ejemplo:**

```python
x = 1
print(eval("x + 1"))  # Salida: 2
```
13. `exec(code)`

 - **Descripción:** Ejecuta el código Python que se pasa como una cadena de texto. Similar a eval(), pero puede ejecutar múltiples líneas de código y no devuelve un valor. También es potencialmente peligroso.
 - **Ejemplo:**

```python
code = """
```
```python
def saludo(nombre):
	return f"Hola {nombre}"

print(saludo("Mundo"))
exec(code)
# Salida: Hola Mundo
```

Estos métodos proporcionan varias funcionalidades útiles para el manejo y procesamiento de cadenas en Python y para la ejecución dinámica de código.

[Métodos de las cadenas — documentación de Python - 3.12.4](https://docs.python.org/es/3/library/stdtypes.html#string-methods "Lecturas recomendadas Métodos de las cadenas — documentación de Python - 3.12.4")

## Enteros, Flotantes y Booleanos

Comprender los diferentes tipos de datos en Python es crucial para la programación eficiente. En Python, cada variable pertenece a una clase o “class”, y su tipo puede ser identificado usando la función `type()`.

¿Qué es un tipo de dato en Python?
En Python, un tipo de dato se refiere a la clase de datos que una variable puede contener. Esto se puede verificar con la función `type()`, que devuelve la clase del valor contenido en la variable. Por ejemplo, `type('Hello')` devuelve `class 'str'`, indicando que el dato es una cadena de texto.

### ¿Cómo se manejan los números enteros en Python?

Los números enteros en Python pertenecen a la clase `int`, que se utiliza para representar números sin parte decimal. Al declarar una variable como `x = 5`, se puede comprobar su tipo con `type(x)`, que devolverá `int`. Esta clase es ideal para operaciones aritméticas básicas.

### ¿Qué son los números flotantes y cómo se usan?

Los números flotantes pertenecen a la clase `float` y se utilizan para representar números con decimales. Por ejemplo, `y = 5.0` es un número flotante. Para operaciones con números muy grandes o pequeños, se puede usar la notación científica, como `z = 1e6`, que representa 1 multiplicado por 10 elevado a 6. Python maneja automáticamente las operaciones aritméticas con flotantes devolviendo resultados en `float`.

### ¿Cómo se utiliza la notación científica en Python?

La notación científica se emplea para representar números muy grandes o muy pequeños de manera compacta. Por ejemplo, `1e6` representa 1,000,000 y `1e-6` representa 0.000001. Esta notación es útil en cálculos científicos y financieros, donde los números pueden variar significativamente en magnitud.

### ¿Cómo se manejan los booleanos en Python?

Los booleanos en Python pertenecen a la clase `bool` y pueden tomar uno de dos valores: `True` o `False`. Estos valores son fundamentales para las operaciones lógicas y las estructuras de control de flujo, como las condicionales. Al declarar una variable como `isTrue = True`, se puede comprobar su tipo con `type(isTrue)`, que devolverá `bool`.

### ¿Qué importancia tienen los tipos de datos en las operaciones matemáticas?

Es crucial entender los tipos de datos al realizar operaciones matemáticas en Python. Las operaciones entre enteros (`int`) y flotantes (`float`) devuelven resultados en `float`. Por ejemplo, sumar un entero y un flotante, como `x + y`, devolverá un resultado en `float`. Este comportamiento es importante para evitar errores cuando el usuario ingresa un tipo de dato inesperado.

### ¿Cómo se usan los comentarios en Python?

Los comentarios en Python se crean utilizando el símbolo `#`, y se usan para anotar y explicar el código. Las líneas comentadas no son ejecutadas por Python. Por ejemplo, `# Este es un comentario` es una línea que Python ignora durante la ejecución.

### ¿Qué operaciones matemáticas básicas se pueden hacer en Python?

En Python se pueden realizar operaciones matemáticas básicas como suma, resta, multiplicación y división. Por ejemplo, se puede sumar dos variables `x` e `y` con `x + y`. Si se utilizan dos números flotantes, el resultado será también un número flotante.

## Todo lo que Debes Saber sobre print en Python

La función incorporada print puede parecer básico al principio, pero ten en cuenta que será una herramienta que usarás de múltiples maneras a lo largo de tu código. Desde el icónico “Hola mundo” hasta mensajes de depuración y presentación de resultados, `print` es la puerta de entrada a la comunicación de tus programas con el mundo exterior.

Iniciar con un simple “Hola mundo” no solo es una tradición en la programación, sino también un momento crucial donde tu código cobra vida. Es la primera línea de código que demuestra que tu entorno de desarrollo está configurado correctamente y que estás listo para empezar a crear.

Aprenderás a aprovechar al máximo la función incorporada print en Python. Desde formatos avanzados hasta el manejo de caracteres especiales y secuencias de escape, descubrirás cómo print puede ser una herramienta poderosa y versátil en tu caja de herramientas de programación.

1. Uso básico de `print`

El uso más sencillo de print consiste en pasar el texto que deseas mostrar entre comillas. Este código imprimirá “Nunca pares de aprender” en la consola, siendo una excelente forma de probar si tu entorno de Python está configurado correctamente.

`print("Nunca pares de aprender")`

Resultado:

`Nunca pares de aprender`

2. Uso de la coma en `print`

La coma dentro de la función `print` se usa para separar varios argumentos. Al hacerlo, Python añade automáticamente un espacio entre los argumentos. Esto es diferente a concatenar cadenas con el operador `+`, que no añade espacios adicionales.

`print("Nunca", "pares", "de", "aprender")`

Resultado:

`Nunca pares de aprender`

Por otro lado, al concatenar cadenas con el operador `+`, los elementos se unen sin ningún espacio adicional, a menos que lo añadas explícitamente.

`print("Nunca" + "pares" + "de" + "aprender")`

Resultado:

`Nuncaparesdeaprender`

Para añadir un espacio explícitamente cuando concatenas cadenas, debes incluirlo dentro de las comillas.

`print("Nunca" + " " + "pares" + " " + "de" + " " + "aprender")`

Resultado:

`Nunca pares de aprender`

3. Uso de `sep`

El parámetro `sep` permite especificar cómo separar los elementos al imprimir. En este ejemplo, los elementos “Nunca”, “pares”, “de” y “aprender” se imprimirán con una coma y un espacio entre ellos, resultando en “Nunca, pares, de, aprender”. Puedes cambiar sep por cualquier cadena de caracteres que desees usar como separador.

`print("Nunca", "pares", "de", "aprender", sep=", ")`

Resultado:

`Nunca, pares, de, aprender`

4. Uso de `end`

El parámetro end cambia lo que se imprime al final de la llamada a print. En lugar de imprimir cada mensaje en una nueva línea, end="" asegura que “Nunca” y “pares” se impriman en la misma línea, resultando en “Nunca pares”. Por defecto, end es un salto de línea ("\n"), lo que hace que cada llamada a print comience en una nueva línea.

```python
print("Nunca", end=" ")
print("pares de aprender")
```

Resultado:

`Nunca pares de aprender`

5. Impresión de variables

Puedes usar print para mostrar el valor de las variables. En este ejemplo, imprimirá “Frase: Nunca pares de aprender” y “Autor: Platzi”. Esto es útil para depurar y ver los valores de las variables en diferentes puntos de tu programa.

```python
frase = "Nunca pares de aprender"
author = "Platzi"
print("Frase:", frase, "Autor:", author)
```

Resultado:

`Frase: Nunca pares de aprender Autor: Platzi`

6. Uso de formato con f-strings

Las f-strings permiten insertar expresiones dentro de cadenas de texto. Al anteponer una `f` a la cadena de texto, puedes incluir variables directamente dentro de las llaves `{}`. En este ejemplo, frase y author se insertarán en la cadena, resultando en “Frase: Nunca pares de aprender, Autor: Platzi”. Esto hace que el código sea más legible y fácil de escribir.

```python
frase = "Nunca pares de aprender"
author = "Platzi"
print(f"Frase: {frase}, Autor: {author}")
```

Resultado:

`Frase: Nunca pares de aprender, Autor: Platzi`

7. Uso de formato con `format`

El método format es otra forma de insertar valores en cadenas de texto. Usando `{}` como marcadores de posición, puedes pasar los valores que quieres insertar como argumentos de `format`. En este ejemplo, se imprimirá “Frase: Nunca pares de aprender, Autor: Platzi”. Es una forma flexible y poderosa de formatear cadenas, aunque las f-strings son más concisas.

```python
frase = "Nunca pares de aprender"
author = "Platzi"
print("Frase: {}, Autor: {}".format(frase, author))
```

Resultado:

`Frase: Nunca pares de aprender, Autor: Platzi`

8. Impresión con formato específico

Puedes controlar el formato de los números al imprimir. En este ejemplo, `:.2f` indica que el número debe mostrarse con dos decimales. Así, imprimirá “Valor: 3.14”, redondeando el número a dos decimales. Esto es especialmente útil cuando trabajas con datos numéricos y necesitas un formato específico.

```python
valor = 3.14159
print("Valor: {:.2f}".format(valor))
```

Resultado:

`Valor: 3.14`

9. Saltos de línea y caracteres especiales

Los saltos de línea en Python se indican con la secuencia de escape `\n`. Por ejemplo, para imprimir “Hola\nmundo”, que aparecerá en dos líneas:

`print("Hola\nmundo")`

Resultado:

```python
Hola
mundo
```

Para imprimir una cadena que contenga comillas simples o dobles dentro de ellas, debes usar secuencias de escape para evitar confusiones con la sintaxis de Python. Por ejemplo, para imprimir la frase “Hola soy ‘Carli’”:

`print('Hola soy \'Carli\'')`

Resultado:

`Hola soy 'Carli'`

Si necesitas imprimir una ruta de archivo en Windows, que incluya barras invertidas, también necesitarás usar secuencias de escape para evitar que Python interprete las barras invertidas como parte de secuencias de escape. Por ejemplo:

`print("La ruta de archivo es: C:\\Users\\Usuario\\Desktop\\archivo.txt")`

Resultado:

`La ruta de archivo es: C:\Users\Usuario\Desktop\archivo.txt`

En Python, estas secuencias de escape te permiten manejar caracteres especiales y estructurar la salida de texto según sea necesario, asegurando que la salida se formatee correctamente en la consola o en cualquier otro medio donde se imprima.

Con estos ejemplos y explicaciones adicionales, tendrás una comprensión más completa sobre cómo manejar saltos de línea y caracteres especiales en Python al usar la función `print`.

## Operaciones Matemáticas en Python

En el mundo de la programación con Python, las operaciones matemáticas básicas como la suma, resta, multiplicación y división son fundamentales. Sin embargo, Python ofrece operaciones adicionales que expanden nuestras posibilidades.

### ¿Cómo realizamos las operaciones matemáticas básicas en Python?

Primero, creamos dos variables: `a` con valor 10 y `b` con valor 3. Usamos comentarios para indicar que estamos trabajando con operadores numéricos. Imprimimos los resultados de las cuatro operaciones básicas:

```python
a = 10
b = 3
print("Suma:", a + b)
print("Resta:", a - b)
print("Multiplicación:", a * b)
print("División:", a / b)
```

### ¿Qué es el operador módulo y cómo se usa?

El operador módulo (`%`) obtiene el residuo de una división. Por ejemplo, `13 % 5` devuelve 3. En Python, se usa así:

```python
a = 13
b = 5
print("Módulo:", a % b)
```

### ¿Qué sucede al dividir por cero en Python?

Dividir por cero genera un error. Para ilustrarlo:

```python
a = 10
b = 0
try:
    print(a / b)
except ZeroDivisionError:
    print("Error: División por cero")
```

### ¿Qué es la división de enteros y cómo se implementa?

La división de enteros (`//`) devuelve solo la parte entera de una división. Por ejemplo:

```python
a = 10
b = 3
print("División Entera:", a // b)
```

### ¿Cómo se realiza la potenciación en Python?

La potenciación se representa con `**`. Para elevar 10 al cubo, usamos:

```python
a = 10
b = 3
print("Potenciación:", a ** b)
```

### ¿Qué es PEMDAS y cómo afecta nuestras operaciones?

PEMDAS es la regla de prioridad de operaciones: Paréntesis, Exponentes, Multiplicación y División (de izquierda a derecha), Adición y Sustracción (de izquierda a derecha). Veamos un ejemplo:

```python
operation = (2 + 3) * 4
print(operation)  # Resultado: 20
```

### ¿Cómo se manejan los operadores booleanos en Python?

Los operadores booleanos comparan valores y devuelven `True` o `False`. Ejemplos:

```python
a = 10
b = 3
print(a > b)  # True
print(a < b)  # False
print(a == b)  # False
print(a != b)  # True
```

Estos operadores nos ayudan a tomar decisiones en el código, permitiendo crear condiciones y bucles efectivos.

[Orden de evaluación - Wikipedia, la enciclopedia libre](https://es.wikipedia.org/wiki/Orden_de_evaluaci%C3%B3n "dfdf")

## Operaciones de Entrada/Salida en Consola

En Python, cuando trabajamos con proyectos que requieren interacción del usuario, es común solicitar datos como correo o contraseña para ejecutar acciones específicas. Este mismo enfoque es útil para entender la función input.

¿Cómo se recibe información del usuario en Python?
Para recibir información del usuario desde la consola, creamos una variable y asignamos el resultado de la función input. Por ejemplo, para pedir el nombre del usuario:

```python
nombre = input("Ingrese su nombre: ")
print(nombre)
```

Al ejecutar este código, se habilita una sección para introducir información. Ingresamos un nombre, presionamos Enter y se imprime el valor guardado en la variable nombre.

### ¿Qué ocurre si eliminamos la función `print`?

Si eliminamos print y ejecutamos el código, el nombre ingresado no se mostrará en la consola:

`nombre = input("Ingrese su nombre: ")`

Para ver el resultado, es imprescindible usar print.

Podemos solicitar la edad del usuario creando una variable `edad` y utilizando `input`, luego imprimimos ambos valores:

```python
nombre = input("Ingrese su nombre: ")
edad = input("Ingrese su edad: ")
print(nombre)
print(edad)
```

Al ejecutar, ingresamos el nombre y la edad, y ambos valores se muestran en pantalla.

### ¿Cuál es el tipo de dato devuelto por input?

El resultado de input es siempre un string, incluso si ingresamos un número. Podemos verificar el tipo de dato usando `type`:

```python
name = input("Ingrese su nombre: ")
age = input("Ingrese su edad: ")
print(type(name))
print(type(age))
```
Al ejecutar, se mostrará que ambos valores son de tipo str.

### ¿Cómo se convierte el tipo de dato (casting)?

Si queremos que la edad sea un número entero en lugar de un string, usamos el casting:

`age = int(input("Ingrese su edad: "))`

Ejecutamos y verificamos que `age` ahora es un entero. También podemos convertir a otros tipos de datos, como flotantes:

`age = float(input("Ingrese su edad: "))`

### ¿Qué sucede si ingresamos un dato inesperado?

Si el código espera un entero, pero ingresamos un string, se produce un `ValueError`. Es importante manejar el tipo de datos correctamente para evitar errores:

En Python, las operaciones de entrada y salida (I/O) en consola son fundamentales para interactuar con el usuario. Estas operaciones se realizan principalmente utilizando las funciones `input()` para recibir datos del usuario y `print()` para mostrar información en la consola.

### Operación de Entrada: `input()`

La función `input()` se utiliza para leer una línea de texto ingresada por el usuario desde la consola. El texto ingresado se devuelve como una cadena (string).

**Sintaxis:**

```python
variable = input("Texto para el usuario: ")
```

**Ejemplo:**

```python
nombre = input("¿Cómo te llamas? ")
print("Hola, " + nombre + "!")
```

- **Explicación:** En este ejemplo, se solicita al usuario que ingrese su nombre. Luego, se utiliza `print()` para saludar al usuario con el nombre ingresado.

**Convertir la Entrada:**

Como `input()` devuelve una cadena, si necesitas un tipo de dato diferente (por ejemplo, un número entero o un número de punto flotante), deberás convertir la entrada usando funciones como `int()` o `float()`.

**Ejemplo:**

```python
edad = int(input("¿Cuántos años tienes? "))
print("Tendrás " + str(edad + 1) + " el próximo año.")
```

- **Explicación:** Aquí, `input()` recoge la edad del usuario como una cadena, y `int()` la convierte en un número entero para poder realizar operaciones aritméticas.

### Operación de Salida: `print()`

La función `print()` se utiliza para mostrar datos en la consola. Es una de las funciones más utilizadas en Python para generar salidas.

**Sintaxis:**

```python
print(objeto1, objeto2, ..., sep=' ', end='\n')
```

- **`objeto1, objeto2, ...`**: Los objetos que deseas imprimir. Puedes pasar múltiples objetos, y `print()` los separará automáticamente con un espacio.
- **`sep`**: Especifica el separador entre los objetos. Por defecto es un espacio (`' '`).
- **`end`**: Especifica lo que se añade al final de la salida. Por defecto es un salto de línea (`'\n'`).

**Ejemplo Básico:**

```python
print("Hola, mundo!")
```

**Ejemplo con Múltiples Objetos:**

```python
nombre = "Alice"
edad = 30
print("Nombre:", nombre, "Edad:", edad)
```

**Personalizar Separadores y Finales:**

```python
print("A", "B", "C", sep="-")  # Salida: A-B-C
print("Hola,", end=" ")
print("mundo!")  # Salida: Hola, mundo!
```

### Resumen de Operaciones de Entrada/Salida

1. **Entrada con `input()`**:
   - Se utiliza para capturar datos del usuario.
   - Siempre devuelve una cadena, que puede ser convertida a otros tipos de datos según sea necesario.

2. **Salida con `print()`**:
   - Muestra datos en la consola.
   - Permite personalizar el formato de la salida mediante argumentos opcionales como `sep` y `end`.

Estas funciones forman la base de la interacción con el usuario en aplicaciones de consola en Python.

## Listas

Listas en Python nos facilita la tarea de permitir la manipulación y almacenamiento de datos diversos de manera estructurada y eficiente.

### ¿Cómo crear una lista en Python?

Para iniciar, se crea una variable llamada todo utilizando corchetes para indicar que se trata de una lista. Dentro de los corchetes, se añaden los elementos separados por comas, por ejemplo:

`todo = ["Dirigirnos al hotel", "Almorzar", "Visitar un museo", "Volver al hotel"]`

### ¿Qué tipos de datos se pueden almacenar en una lista?

Las listas en Python pueden almacenar múltiples tipos de datos, incluyendo cadenas, números enteros, números flotantes y valores booleanos. También pueden contener otras listas. Ejemplo:

```python
mix = ["string", 1, 2.5, True, [3, 4]]
print(mix)
```

¿Cómo se determina la longitud de una lista?
Para saber cuántos elementos hay en una lista, se usa la función `len`:

`print(len(mix))`

### ¿Cómo se accede a elementos específicos de una lista?

Se puede acceder a los elementos de una lista utilizando índices, donde el índice comienza en 0:

```python
print(mix[0])  # Primer elemento
print(mix[-1])  # Último elemento
```

### ¿Cómo se realizan operaciones de slicing en listas?

El slicing permite obtener sublistas a partir de una lista existente, especificando un rango de índices:

```python
print(mix[1:3])  # Desde el índice 1 hasta el 2 (el 3 no se incluye)
print(mix[:2])  # Desde el inicio hasta el índice 1
print(mix[2:])  # Desde el índice 2 hasta el final
```

### ¿Qué métodos de manipulación de listas existen?

- Añadir elementos al final: `append()`

```python
mix.append(False)
print(mix)
```

- Insertar elementos en una posición específica: `insert()`

```python
mix.insert(1, ["A", "B"])
print(mix)
```

- Encontrar la primera aparición de un elemento: `index()`

`print(mix.index(["A", "B"]))`

- Encontrar el mayor y menor elemento: `max()` y `min()`

```python
numbers = [1, 2, 3.5, 90, 100]
print(max(numbers))
print(min(numbers))
```

### ¿Cómo se eliminan elementos de una lista?

- Eliminar por índice: `del`

```python
del numbers[-1]
print(numbers)
```

- Eliminar una porción de la lista:

```python
del numbers[0:2]
print(numbers)
```

- Eliminar toda la lista:

`del numbers`

En Python, una **lista** es una colección ordenada y mutable (modificable) de elementos. Las listas son uno de los tipos de datos más utilizados y son extremadamente versátiles, permitiendo almacenar elementos de diferentes tipos, incluyendo otras listas.

### Creación de Listas

Puedes crear una lista usando corchetes `[]` y separando los elementos con comas.

**Ejemplo:**

```python
mi_lista = [1, 2, 3, 4, 5]
```

También es posible crear una lista vacía:

```python
mi_lista_vacia = []
```

### Acceso a Elementos

Puedes acceder a los elementos de una lista usando índices, que comienzan en `0` para el primer elemento.

**Ejemplo:**

```python
mi_lista = [10, 20, 30, 40, 50]
print(mi_lista[0])  # Salida: 10
print(mi_lista[2])  # Salida: 30
```

También puedes usar índices negativos para acceder a los elementos desde el final de la lista.

**Ejemplo:**

```python
print(mi_lista[-1])  # Salida: 50 (último elemento)
```

### Modificación de Elementos

Las listas en Python son mutables, lo que significa que puedes cambiar sus elementos.

**Ejemplo:**

```python
mi_lista[1] = 25
print(mi_lista)  # Salida: [10, 25, 30, 40, 50]
```

### Operaciones Comunes

1. **Agregar elementos:**
   - `append()`: Agrega un elemento al final de la lista.
   - `insert()`: Inserta un elemento en una posición específica.
   - `extend()`: Agrega múltiples elementos de otra lista o iterable.

   **Ejemplo:**

   ```python
   mi_lista.append(60)
   mi_lista.insert(2, 35)
   mi_lista.extend([70, 80])
   print(mi_lista)  # Salida: [10, 25, 35, 30, 40, 50, 60, 70, 80]
   ```

2. **Eliminar elementos:**
   - `remove()`: Elimina la primera aparición de un elemento específico.
   - `pop()`: Elimina y devuelve el elemento en la posición especificada (por defecto, el último elemento).
   - `clear()`: Elimina todos los elementos de la lista.

   **Ejemplo:**

   ```python
   mi_lista.remove(25)
   print(mi_lista)  # Salida: [10, 35, 30, 40, 50, 60, 70, 80]

   elemento = mi_lista.pop(3)
   print(elemento)  # Salida: 40
   print(mi_lista)  # Salida: [10, 35, 30, 50, 60, 70, 80]

   mi_lista.clear()
   print(mi_lista)  # Salida: []
   ```

3. **Rebanado (Slicing):**
   - Puedes obtener sublistas usando el operador de rebanado (`:`).

   **Ejemplo:**

   ```python
   mi_lista = [10, 20, 30, 40, 50]
   sublista = mi_lista[1:4]  # Sublista desde el índice 1 hasta el 3 (4 no incluido)
   print(sublista)  # Salida: [20, 30, 40]
   ```

4. **Longitud de una lista:**
   - Usa `len()` para obtener el número de elementos en la lista.

   **Ejemplo:**

   ```python
   print(len(mi_lista))  # Salida: 5
   ```

5. **Verificar existencia de un elemento:**
   - Usa el operador `in`.

   **Ejemplo:**

   ```python
   if 30 in mi_lista:
       print("El 30 está en la lista")
   ```

6. **Ordenar una lista:**
   - `sort()`: Ordena la lista en su lugar.
   - `sorted()`: Devuelve una nueva lista ordenada.

   **Ejemplo:**

   ```python
   mi_lista.sort()
   print(mi_lista)  # Salida: [10, 20, 30, 40, 50]

   nueva_lista = sorted([3, 1, 4, 2])
   print(nueva_lista)  # Salida: [1, 2, 3, 4]
   ```

### Iteración sobre una Lista

Puedes iterar sobre los elementos de una lista utilizando un bucle `for`.

**Ejemplo:**

```python
for elemento in mi_lista:
    print(elemento)
```

### Listas Anidadas

Las listas pueden contener otras listas, lo que permite la creación de estructuras de datos más complejas.

**Ejemplo:**

```python
lista_anidada = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(lista_anidada[1][2])  # Salida: 6 (elemento en la posición 2 de la sublista 1)
```

### Conclusión

Las listas son extremadamente útiles en Python debido a su flexibilidad y la gran cantidad de operaciones que puedes realizar con ellas. Son una herramienta esencial para manejar colecciones de datos en programas Python.