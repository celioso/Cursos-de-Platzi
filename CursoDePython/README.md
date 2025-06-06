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

## Método slice

Cuando asignamos una lista a una nueva variable, por ejemplo, `B = A`, no estamos creando una copia independiente. Ambas variables apuntan al mismo espacio de memoria. Así, cualquier cambio en `A` se reflejará en `B`.

### ¿Cómo evitar que dos listas apunten al mismo espacio de memoria?

Para evitar que dos variables apunten al mismo espacio de memoria, debemos crear una copia superficial de la lista original usando slicing. Por ejemplo:

- Crear una lista `A` con números del 1 al 5.
- Asignar `B = A` y luego imprimir ambas listas muestra que ambas son idénticas.
- Eliminar un elemento de `A` también lo elimina de `B`.

### ¿Cómo usar slicing para crear una copia de una lista?

Podemos utilizar slicing para copiar una lista sin que ambas variables apunten al mismo espacio de memoria. Por ejemplo:

```python
A = [1, 2, 3, 4, 5]
C = A[:]
```

Luego, verificamos los IDs de memoria:

```python
print(id(A))
print(id(C))
```

Ambos IDs serán diferentes, lo que indica que C es una copia independiente de A.

### ¿Por qué es importante entender la asignación de memoria en listas?

En Python, a diferencia de otros lenguajes, podemos almacenar diferentes tipos de datos en una colección. Entender cómo funciona la memoria es crucial para evitar errores en el código, especialmente en aplicaciones del mundo laboral.

## Listas de más dimensiones y Tuplas

Las matrices en Python son una herramienta poderosa que permite organizar datos en listas de listas, facilitando su manejo y manipulación.

### ¿Qué es una matriz en Python?

Una matriz es una colección ordenada de datos dispuestos en filas y columnas. Se representa como una lista de listas, donde cada sublista es una fila de la matriz.

### ¿Cómo iterar a través de una matriz?

Para iterar a través de una matriz en Python, se puede utilizar un bucle for anidado. Cada sublista (fila) se recorre individualmente:

- **Ejemplo de matriz:**
```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
```

- **Iterar e imprimir cada elemento:**

```python
for row in matrix:
    for element in row:
        print(element)
```

### ¿Cómo acceder a elementos específicos en una matriz?

Para acceder a un elemento específico en una matriz, se utilizan los índices de la fila y la columna. Por ejemplo, para acceder al número 9 en la matriz anterior, se usa `matrix[2][2]`.

- **Código:**

```python
print(matrix[2][2])  # Salida: 9
```

### ¿Qué significa que las matrices sean mutables?

Las matrices son mutables, lo que significa que se pueden modificar, añadir o eliminar elementos después de su creación. Este es un ejemplo básico:

- **Modificar un elemento:**

```python
matrix[0][0] = 10
print(matrix)  # Salida: [[10, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### ¿Cuál es la diferencia entre matrices y tuplas?

A diferencia de las matrices, las tuplas son inmutables, lo que significa que no se pueden modificar después de su creación. Las tuplas se utilizan para almacenar datos que no deben cambiar.

- **Ejemplo de tupla:**

`numbers = (1, 2, 3)`

Intentar modificar una tupla genera un error:

`numbers[0] = 10  # Genera TypeError: 'tuple' object does not support item assignment`

Parece que ya discutimos sobre las listas de más dimensiones y las tuplas, pero puedo expandir o aclarar cualquier parte si lo deseas. Aquí tienes un resumen y algunos detalles adicionales sobre cada concepto:

### Listas de Más Dimensiones

Las listas de más dimensiones en Python son simplemente listas que contienen otras listas como elementos. Esto te permite crear estructuras como matrices o tablas, donde los datos se organizan en filas y columnas.

#### Ejemplo de Lista Bidimensional:

```python
# Una matriz 3x3
matriz = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Acceder a un elemento: por ejemplo, el número 6
print(matriz[1][2])  # Salida: 6
```

En este ejemplo, `matriz` es una lista de listas, donde cada lista interna representa una fila de la matriz.

#### Ejemplo de Lista Tridimensional:

```python
# Un cubo 2x2x2
cubo = [
    [
        [1, 2], 
        [3, 4]
    ],
    [
        [5, 6], 
        [7, 8]
    ]
]

# Acceder a un elemento: por ejemplo, el número 7
print(cubo[1][1][0])  # Salida: 7
```

Aquí, `cubo` es una lista tridimensional, lo que significa que es una lista de listas de listas. Puedes pensar en esto como una estructura que contiene múltiples matrices.

### Tuplas

Las tuplas en Python son estructuras de datos muy similares a las listas, pero con dos diferencias clave:

1. **Inmutabilidad:** No se pueden modificar después de su creación. No puedes añadir, eliminar o cambiar elementos en una tupla.
2. **Sintaxis:** Se definen utilizando paréntesis `()`.

#### Ejemplo de Tupla:

```python
tupla = (1, 2, 3, 4, 5)

# Acceder a un elemento: por ejemplo, el número 3
print(tupla[2])  # Salida: 3

# Intentar modificar la tupla generará un error
# tupla[2] = 10  # Esto provocará un TypeError
```

Las tuplas son útiles cuando necesitas asegurarte de que los datos no cambien durante la ejecución de un programa.

#### Tuplas Anidadas (Más Dimensiones):

Al igual que las listas, las tuplas pueden anidarse para crear estructuras de datos más complejas:

```python
tupla_anidada = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9)
)

# Acceder a un elemento: por ejemplo, el número 8
print(tupla_anidada[2][1])  # Salida: 8
```

En este caso, `tupla_anidada` es una tupla de tuplas, lo que te permite trabajar con estructuras de datos que se asemejan a matrices.

### Comparación entre Listas y Tuplas

- **Mutabilidad:** Las listas son mutables (pueden cambiar), mientras que las tuplas son inmutables.
- **Velocidad:** Las tuplas son generalmente más rápidas que las listas, especialmente para operaciones de acceso.
- **Uso:** Las listas se utilizan cuando necesitas una estructura de datos que pueda cambiar. Las tuplas se utilizan cuando los datos deben permanecer constantes.

Si tienes alguna pregunta específica sobre estos temas o necesitas ejemplos adicionales, estaré encantado de ayudarte.

## Aplicación de Matrices

### Aplicación de Matrices

Las matrices son una herramienta fundamental en muchas áreas de la computación y las matemáticas. En Python, podemos usar listas dentro de listas para representar matrices bidimensionales (2D). Hoy, vamos a explorar varias aplicaciones prácticas de las matrices y cómo estas estructuras pueden ser usadas para representar tableros de juego.

### Representación de Tableros de Juego

Las matrices son ideales para representar tableros de juego en programación, como tableros de ajedrez, damas y otros juegos de mesa. Usar matrices para estos fines permite manejar fácilmente la disposición de las piezas y las reglas del juego.

### Ejemplo: Tablero de Ajedrez

Un tablero de ajedrez es una matriz de 8x8. En vez de representar solo las casillas blancas y negras, podemos representar las piezas de ajedrez. Usaremos letras para representar las piezas: `P` para peón, `R` para torre, `N` para caballo (knight), `B` para alfil, `Q` para reina y `K` para rey. Las piezas negras se representan con letras minúsculas y las blancas con letras mayúsculas.

```python
chess_board = [
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
]

print(chess_board
```

En este ejemplo, el `0` representa una casilla vacía.

### Movimiento de un Caballo

En ajedrez, los caballos (`N` para blanco y `n` para negro) se mueven en forma de “L”. Esto significa que pueden moverse dos casillas en una dirección y luego una casilla perpendicularmente, o una casilla en una dirección y luego dos casillas perpendicularmente.

Por ejemplo, si el caballo blanco está en la posición (7, 1) (segunda casilla de la última fila), las posiciones posibles a las que puede moverse son:

- (5, 0)
- (5, 2)
- (6, 3)

Es importante verificar que estas posiciones estén dentro de los límites del tablero y no contengan piezas blancas.

Si movemos el caballo de (7, 1) a (5, 2), el tablero se vería así:

```python
chess_board[7][1] = 0  # Casilla original del caballo ahora está vacía
chess_board[5][2] = 'N'  # Nueva posición del caballo

print(chess_board)
```

### Ejemplo: Tablero de Damas

Un tablero de damas también es una matriz de 8x8, pero además de las casillas alternas, debemos representar las piezas de los dos jugadores.

```python
checkers_board = [
    [0, 'b', 0, 'b', 0, 'b', 0, 'b'],
    ['b', 0, 'b', 0, 'b', 0, 'b', 0],
    [0, 'b', 0, 'b', 0, 'b', 0, 'b'],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    ['w', 0, 'w', 0, 'w', 0, 'w', 0],
    [0, 'w', 0, 'w', 0, 'w', 0, 'w'],
    ['w', 0, 'w', 0, 'w', 0, 'w', 0]
]

print(checkers_board)
```

En este ejemplo, el 0 representa una casilla vacía, 'w' representa una pieza blanca, y 'b' representa una pieza negra. Las filas superiores e inferiores están llenas de piezas en sus posiciones iniciales, mientras que las filas centrales están vacías.

### Aplicación de Matrices a Imágenes

Las matrices también son esenciales para la representación y manipulación de imágenes. Cada píxel de una imagen en escala de grises se puede representar como un valor en una matriz, donde cada valor varía del 0 (negro) al 255 (blanco).

### Ejemplo: Representación de una Imagen en Escala de Grises

Imaginemos una matriz de 5x5 que representa una imagen en escala de grises con un simple patrón.

```python
image = [
    [255, 0, 0, 0, 255],
    [0, 255, 0, 255, 0],
    [0, 0, 255, 0, 0],
    [0, 255, 0, 255, 0],
    [255, 0, 0, 0, 255]
]

print(image)
```

En esta matriz, el `255` representa píxeles blancos y el 0 representa píxeles negros. Este patrón podría visualizarse como una “X” blanca sobre un fondo negro si se dibujara.

### Aplicaciones en Otros Campos

Las matrices se utilizan en muchos otros campos además de los juegos y las imágenes. Aquí hay algunos ejemplos:

- **Análisis de Datos:** Las matrices se utilizan para almacenar y manipular grandes conjuntos de datos, realizar cálculos estadísticos y análisis de datos.
- **Inteligencia Artificial y Machine Learning:** Las matrices son esenciales para representar datos de entrada y salida, pesos de redes neuronales y otros parámetros en algoritmos de aprendizaje automático.
- **Computación Científica:** Las matrices se utilizan para resolver ecuaciones lineales, realizar simulaciones y modelar fenómenos científicos.
- **Gráficos por Computadora:** Las matrices se utilizan para representar y transformar objetos en gráficos 2D y 3D.

Las matrices son una herramienta poderosa que no solo facilita la representación de datos complejos, sino que también permite realizar operaciones y transformaciones avanzadas de manera eficiente. Al dominar el uso de matrices en Python, puedes abrir la puerta a un mundo de posibilidades en diversos campos de la ciencia, la ingeniería y la tecnología.

### Diccionarios

Los diccionarios en Python son una estructura que almacenan dos datos, la clave y el valor. Un ejemplo cotidiano es un diccionario físico donde buscamos el significado de una palabra y encontramos la palabra (clave) y su definición (valor). Veamos cómo se utilizan en código.

### ¿Cómo se crea un diccionario en Python?

Iniciamos creando una variable llamada numbers y especificamos el uso de diccionarios utilizando llaves. Asignamos valores a las claves:

```python
numbers = {1: "one", "2": "two", 3: "three"}
print(numbers)
```

### ¿Cómo se accede a los elementos de un diccionario?

Para consultar la información de una clave específica, utilizamos la indexación:

`print(numbers["2"])`

### ¿Cómo se eliminan elementos de un diccionario?

Para eliminar un elemento, utilizamos la clave del mismo:

```python
del information["edad"]
print(information)
```

### ¿Qué métodos existen para trabajar con diccionarios?

Podemos utilizar métodos propios de los diccionarios, como `keys()`, `values()`, e `items()`:

```python
# Obtener las claves
claves = information.keys()
print(claves)

# Obtener los valores
valores = information.values()
print(valores)

# Obtener los pares clave-valor
pares = information.items()
print(pares)
```

### ¿Cómo se crea un diccionario de diccionarios?

Podemos crear una agenda de contactos usando diccionarios de diccionarios:

```python
contactos = {
    "Carla": {"apellido": "Florida", "altura": 1.7, "edad": 30},
    "Diego": {"apellido": "Antesana", "altura": 1.75, "edad": 32}
}
print(contactos["Carla"])
```

Los **diccionarios** en Python son estructuras de datos que permiten almacenar pares de clave-valor. Cada clave en un diccionario es única, y se utiliza para acceder al valor asociado a esa clave. A diferencia de las listas y las tuplas, donde los elementos se acceden por su posición (índice), en un diccionario se accede a los valores a través de sus claves.

### Características de los Diccionarios:

1. **Clave-Valor:** Cada elemento en un diccionario tiene una clave y un valor asociado. Por ejemplo, en el par `'nombre': 'Juan'`, `'nombre'` es la clave y `'Juan'` es el valor.

2. **Inmutabilidad de Claves:** Las claves deben ser de un tipo de dato inmutable, como números, cadenas de texto (strings) o tuplas.

3. **Mutabilidad de Valores:** Los valores pueden ser de cualquier tipo de dato, incluidos otros diccionarios, listas, tuplas, etc.

4. **Acceso Rápido:** El acceso a los valores en un diccionario es muy eficiente gracias a su implementación basada en tablas hash.

### Creación de un Diccionario:

Puedes crear un diccionario usando llaves `{}` o la función `dict()`.

```python
# Usando llaves
diccionario = {
    'nombre': 'Juan',
    'edad': 30,
    'ciudad': 'Madrid'
}

# Usando dict()
diccionario2 = dict(nombre='Ana', edad=25, ciudad='Barcelona')
```

### Acceso a Valores:

Para acceder al valor asociado a una clave, se usa la sintaxis `diccionario[clave]`.

```python
print(diccionario['nombre'])  # Salida: Juan
print(diccionario2['edad'])  # Salida: 25
```

### Añadir o Modificar Elementos:

Para añadir un nuevo par clave-valor o modificar un valor existente, se utiliza la misma sintaxis de acceso.

```python
# Añadir un nuevo par clave-valor
diccionario['profesión'] = 'Ingeniero'

# Modificar un valor existente
diccionario['edad'] = 31

print(diccionario)
# Salida: {'nombre': 'Juan', 'edad': 31, 'ciudad': 'Madrid', 'profesión': 'Ingeniero'}
```

### Eliminar Elementos:

Puedes eliminar un par clave-valor usando `del` o el método `pop()`.

```python
# Usando del
del diccionario['ciudad']

# Usando pop (devuelve el valor eliminado)
profesion = diccionario.pop('profesión')

print(diccionario)
# Salida: {'nombre': 'Juan', 'edad': 31}
print(profesion)  # Salida: Ingeniero
```

### Métodos Útiles:

- **`keys()`**: Devuelve una vista con todas las claves del diccionario.
- **`values()`**: Devuelve una vista con todos los valores del diccionario.
- **`items()`**: Devuelve una vista con todos los pares clave-valor como tuplas.
- **`get(clave, valor_por_defecto)`**: Devuelve el valor asociado a la clave, o el valor por defecto si la clave no existe.

```python
# Ejemplos de métodos
claves = diccionario.keys()
valores = diccionario.values()
items = diccionario.items()

print(claves)   # Salida: dict_keys(['nombre', 'edad'])
print(valores)  # Salida: dict_values(['Juan', 31])
print(items)    # Salida: dict_items([('nombre', 'Juan'), ('edad', 31)])

# Usando get para evitar errores si la clave no existe
ciudad = diccionario.get('ciudad', 'No especificado')
print(ciudad)  # Salida: No especificado
```

### Diccionarios Anidados:

Los diccionarios pueden contener otros diccionarios como valores, permitiendo crear estructuras de datos complejas.

```python
estudiantes = {
    'Juan': {'edad': 20, 'ciudad': 'Madrid'},
    'Ana': {'edad': 22, 'ciudad': 'Barcelona'}
}

# Acceder a un valor en un diccionario anidado
print(estudiantes['Juan']['ciudad'])  # Salida: Madrid
```

### Iteración Sobre Diccionarios:

Puedes iterar sobre los diccionarios para trabajar con las claves, los valores o ambos.

```python
for clave, valor in diccionario.items():
    print(f"La clave es {clave} y el valor es {valor}")
# Salida:
# La clave es nombre y el valor es Juan
# La clave es edad y el valor es 31
```

Los diccionarios son muy útiles para almacenar y gestionar datos que tienen una relación directa de mapeo entre un identificador (clave) y sus características o atributos (valor).

## Comprehension Lists en Python

Una *Comprehension List* es una forma concisa de crear listas en Python, pues permite generar listas nuevas transformando cada elemento de una colección existente o creando elementos a partir de un rango. La sintaxis es compacta y directa, lo que facilita la comprensión del propósito de tu código de un vistazo.

La estructura básica de una Comprehension List es:

`[expresión for elemento in iterable if condición]`

Que se traduce a: “Crea una nueva lista evaluando `nueva_expresión` para cada elemento en el iterable.”

### Ejercicios:

1. **Doble de los Números**

Dada una lista de números [1, 2, 3, 4, 5], crea una nueva lista que contenga el doble de cada número usando una List Comprehension.

2. **Filtrar y Transformar en un Solo Paso**

Tienes una lista de palabras ["sol", "mar", "montaña", "rio", "estrella"] y quieres obtener una nueva lista con las palabras que tengan más de 3 letras y estén en mayúsculas.

3. **Crear un Diccionario con List Comprehension**

Tienes dos listas, una de claves ["nombre", "edad", "ocupación"] y otra de valores ["Juan", 30, "Ingeniero"]. Crea un diccionario combinando ambas listas usando una List Comprehension.

4. **Anidación de List Comprehensions**

Dada una lista de listas (una matriz):

```python
pythonCopiar código
matriz = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
```

Calcula la matriz traspuesta utilizando una List Comprehension anidada.

5. **Extraer Información de una Lista de Diccionarios**

Dada una lista de diccionarios que representan personas:

```python
pythonCopiar código
personas = [
    {"nombre": "Juan", "edad": 25, "ciudad": "Madrid"},
    {"nombre": "Ana", "edad": 32, "ciudad": "Madrid"},
    {"nombre": "Pedro", "edad": 35, "ciudad": "Barcelona"},
    {"nombre": "Laura", "edad": 40, "ciudad": "Madrid"}
]
```

Extrae una lista de nombres de personas que viven en “Madrid” y tienen más de 30 años.

6. **List Comprehension con un `else`**

Dada una lista de números [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], crea una nueva lista multiplicando por 2 los números pares y dejando los impares como están.

### Soluciones

1. **Doble de los Números**

```python
pythonCopiar código
numeros = [1, 2, 3, 4, 5]
dobles = [x * 2 for x in numeros]
print("Dobles:", dobles)
```

2. **Filtrar y Transformar en un Solo Paso**

```python
pythonCopiar código
palabras = ["sol", "mar", "montaña", "rio", "estrella"]
palabras_filtradas = [palabra.upper() for palabra in palabras if len(palabra) > 3]
print("Palabras filtradas y en mayúsculas:", palabras_filtradas)
```

3. **Crear un Diccionario con List Comprehension**

```python
pythonCopiar código
claves = ["nombre", "edad", "ocupación"]
valores = ["Juan", 30, "Ingeniero"]

diccionario = {claves[i]: valores[i] for i in range(len(claves))}
print("Diccionario creado:", diccionario)
```

4. **Anidación de List Comprehensions**

```python
pythonCopiar código
matriz = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
```

transpuesta_comprehension = [[fila[i] for fila in matriz] for i in range(len(matriz[0]))]
print("Transpuesta con List Comprehension:", transpuesta_comprehension)

5. **Extraer Información de una Lista de Diccionarios**

```python
pythonCopiar código
personas = [
    {"nombre": "Juan", "edad": 25, "ciudad": "Madrid"},
    {"nombre": "Ana", "edad": 32, "ciudad": "Madrid"},
    {"nombre": "Pedro", "edad": 35, "ciudad": "Barcelona"},
    {"nombre": "Laura", "edad": 40, "ciudad": "Madrid"}
]

nombres_madrid = [persona["nombre"] for persona in personas if persona["ciudad"] == "Madrid" and persona["edad"] > 30]
print("Nombres de personas en Madrid mayores de 30 años:", nombres_madrid
```

6. **List Comprehension con un `else`**

```python
pythonCopiar código
numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
transformados = [x * 2 if x % 2 == 0 else x for x in numeros]
print("Números transformados:", transformados)
```

Las *Comprehension Lists* en Python son una herramienta poderosa y versátil que permite escribir código más limpio y eficiente. Al dominar su uso, puedes realizar transformaciones y filtrados de datos de manera más concisa, lo que no solo mejora la legibilidad del código, sino que también puede optimizar su rendimiento.

Practicar con ejemplos como los presentados te ayudará a integrar esta técnica en tus proyectos de programación diaria, facilitando la manipulación de colecciones de datos de manera elegante y efectiva.

**Lecturas recomendadas**

[5. Data Structures — Python 3.12.5 documentation](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions "5. Data Structures — Python 3.12.5 documentation")


## Estructuras condicionales

En programación, las estructuras condicionales son esenciales para tomar decisiones basadas en ciertas condiciones. Por ejemplo, al usar la instrucción `IF` en Python, se puede verificar si una variable cumple con una condición específica y ejecutar el código correspondiente.

### ¿Cómo se usa la estructura IF en Python?

Para utilizar el `IF`, primero se define una variable, por ejemplo, `x = 10`. Luego, se escribe la estructura condicional usando la palabra reservada `IF` seguida de la condición, como `if x > 5:`. Si esta condición es verdadera, se ejecuta el código dentro del `IF`, que debe estar indentado.

```python
x = 10
if x > 5:
    print("x es mayor que 5")
```

### ¿Qué pasa si la condición del IF es falsa?

Si la condición del `IF` no se cumple, se puede utilizar la instrucción `else` para manejar el caso contrario. Por ejemplo, si `x` es menor o igual a 5, se ejecutará el bloque de código dentro del else.

```python
x = 3
if x > 5:
    print("x es mayor que 5")
else:
    print("x es menor o igual a 5")
```

### ¿Cómo se manejan múltiples condiciones?

Cuando hay múltiples condiciones, se puede usar `elif` (else if). Esto permite agregar condiciones adicionales entre `if` y `else`.

```python
x = 5
if x > 5:
    print("x es mayor que 5")
elif x == 5:
    print("x es igual a 5")
else:
    print("x es menor que 5")
```

### ¿Cómo se manejan múltiples condiciones en un solo IF?

Para evaluar múltiples condiciones en una sola sentencia `IF`, se pueden utilizar los operadores lógicos` and` y `or`. El operador `and` requiere que ambas condiciones sean verdaderas, mientras que el operador `or` requiere que al menos una condición sea verdadera.

```python
x = 15
y = 30
if x > 10 and y > 25:
    print("x es mayor que 10 y y es mayor que 25")
if x > 10 or y > 35:
    print("x es mayor que 10 o y es mayor que 35")
```

### ¿Qué es la negación en las condiciones?

La palabra reservada `not `se utiliza para negar una condición. Si una condición es verdadera, not la convierte en falsa, y viceversa.

```python
x = 15
if not x > 20:
    print("x no es mayor que 20")
```

### ¿Cómo se anidan las estructuras IF?

Los `IF` anidados permiten evaluar condiciones dentro de otras condiciones. Esto es útil para verificar múltiples niveles de requisitos.

```python
isMember = True
age = 15
if isMember:
    if age >= 15:
        print("Tienes acceso ya que eres miembro y mayor que 15")
    else:
        print("No tienes acceso ya que eres miembro, pero menor a 15 años")
else:
    print("No eres miembro y no tienes acceso")
```

Las **estructuras condicionales** en Python te permiten tomar decisiones en tu código basadas en condiciones específicas. Estas estructuras ejecutan bloques de código diferentes dependiendo de si una condición se evalúa como `True` o `False`.

### Estructuras Condicionales en Python

1. **`if`**: Evalúa una condición. Si la condición es `True`, ejecuta el bloque de código correspondiente.
2. **`elif`** (else if): Se utiliza para evaluar múltiples condiciones. Si la condición anterior es `False`, se evalúa la siguiente condición.
3. **`else`**: Se ejecuta si todas las condiciones anteriores son `False`.

### Sintaxis Básica

```python
if condición:
    # Bloque de código si la condición es True
elif otra_condición:
    # Bloque de código si la primera condición es False y esta es True
else:
    # Bloque de código si todas las condiciones anteriores son False
```

### Ejemplo Simple:

```python
edad = 18

if edad >= 18:
    print("Eres mayor de edad.")
else:
    print("Eres menor de edad.")
```

En este ejemplo, si la variable `edad` es mayor o igual a 18, se imprime "Eres mayor de edad". De lo contrario, se imprime "Eres menor de edad".

### Uso de `elif` para Múltiples Condiciones:

```python
nota = 85

if nota >= 90:
    print("Sobresaliente")
elif nota >= 80:
    print("Notable")
elif nota >= 70:
    print("Aprobado")
else:
    print("Reprobado")
```

En este caso, dependiendo del valor de `nota`, se imprime la calificación correspondiente. Si la `nota` es 85, se imprime "Notable".

### Condicionales Anidadas:

Puedes anidar estructuras `if`, `elif`, y `else` dentro de otras para crear condiciones más complejas.

```python
edad = 20
licencia = True

if edad >= 18:
    if licencia:
        print("Puedes conducir.")
    else:
        print("Necesitas una licencia para conducir.")
else:
    print("No tienes la edad suficiente para conducir.")
```

Aquí, primero se verifica si la persona es mayor de edad. Si lo es, se verifica si tiene licencia de conducir. Dependiendo de estas dos condiciones, se decide si puede conducir o no.

### Operadores Lógicos en Condicionales:

Puedes usar operadores lógicos como `and`, `or`, y `not` para combinar varias condiciones en una sola.

```python
edad = 25
tiene_permiso = True

if edad >= 18 and tiene_permiso:
    print("Puedes entrar al evento.")
else:
    print("No puedes entrar al evento.")
```

En este ejemplo, la persona solo puede entrar al evento si es mayor de edad y tiene permiso. Si cualquiera de estas condiciones es `False`, no se le permitirá la entrada.

### Expresiones Condicionales (Operador Ternario):

Python también permite escribir condicionales simples en una sola línea usando el operador ternario.

```python
mensaje = "Mayor de edad" if edad >= 18 else "Menor de edad"
print(mensaje)
```

Este código es equivalente al ejemplo anterior, pero en una sola línea. Si `edad >= 18`, `mensaje` será "Mayor de edad". De lo contrario, será "Menor de edad".

### Ejemplo Completo:

```python
temperatura = 30

if temperatura > 30:
    print("Hace mucho calor.")
elif 20 <= temperatura <= 30:
    print("El clima es agradable.")
elif 10 <= temperatura < 20:
    print("Hace un poco de frío.")
else:
    print("Hace mucho frío.")
```

En este ejemplo, dependiendo de la `temperatura`, se imprime un mensaje que describe cómo está el clima. La estructura `elif` permite evaluar varios rangos de temperatura.

Las estructuras condicionales son fundamentales para controlar el flujo de ejecución en los programas, permitiendo que el código tome decisiones dinámicamente según las condiciones dadas.

## Bucles y Control de Iteraciones

Aprender a automatizar el proceso de iteración en listas utilizando bucles y controles de iteración es fundamental para optimizar el manejo de datos en Python.

### ¿Cómo iterar una lista usando un bucle for?

Para iterar sobre una colección de datos, podemos usar un bucle for. Aquí se muestra cómo acceder a cada elemento de una lista de números del 1 al 6:

```python
numbers = [1, 2, 3, 4, 5, 6]
for i in numbers:
    print(f"i es igual a: {i}")
```

En este ejemplo, `i` representa cada elemento de la lista `numbers`.

### ¿Cómo iterar usando la función range?

La función `range` permite generar una secuencia de números. Se puede especificar el inicio, el fin y el paso:

```python
for i in range(10):
    print(i)  # Imprime del 0 al 9

for i in range(3, 10):
    print(i)  # Imprime del 3 al 9
```

### ¿Cómo usar condicionales dentro de un bucle for?

Se pueden incluir condicionales dentro del bucle for para realizar operaciones específicas:

```python
frutas = ["manzana", "pera", "uva", "naranja", "tomate"]
for fruta in frutas:
    if fruta == "naranja":
        print("naranja encontrada")
    print(fruta)
```

### ¿Cómo funciona el bucle while?

El bucle while ejecuta un bloque de código mientras se cumpla una condición:

```python
x = 0
while x < 5:
    print(x)
    x += 1
```

### ¿Qué hacer para evitar bucles infinitos?

Es importante modificar la condición dentro del bucle while para evitar que se ejecute indefinidamente:

```python
x = 0
while x < 5:
    print(x)
    x += 1
```

### ¿Cómo usar break y continue en bucles?

La palabra clave `break` se utiliza para salir del bucle prematuramente, mientras que continue omite la iteración actual y pasa a la siguiente:

```python
for i in numbers:
    if i == 3:
        break
    print(i)  # Termina al llegar a 3

for i in numbers:
    if i == 3:
        continue
    print(i)  # Omite el 3
```

Los **bucles** y el **control de iteraciones** son herramientas fundamentales en la programación que te permiten ejecutar un bloque de código repetidamente. Python ofrece varios tipos de bucles y mecanismos para controlar cómo se ejecutan estas iteraciones.

### Tipos de Bucles

1. **`for`**: Se utiliza para iterar sobre una secuencia (como una lista, tupla, diccionario, conjunto o cadena) o sobre un rango de números.
2. **`while`**: Repite un bloque de código mientras una condición dada sea `True`.

### Bucle `for`

El bucle `for` en Python es ideal para iterar sobre los elementos de una secuencia. La sintaxis básica es la siguiente:

```python
for elemento in secuencia:
    # Bloque de código a ejecutar
```

#### Ejemplo: Iterar sobre una lista

```python
numeros = [1, 2, 3, 4, 5]

for numero in numeros:
    print(numero)
```

En este ejemplo, el bucle `for` recorre la lista `numeros` y `numero` toma el valor de cada elemento en la lista durante cada iteración.

#### Ejemplo: Usando `range()`

La función `range()` se utiliza comúnmente con `for` para generar una secuencia de números.

```python
for i in range(5):  # Esto genera los números 0 a 4
    print(i)
```

El código anterior imprime los números del 0 al 4. También puedes especificar un inicio, fin y un paso:

```python
for i in range(1, 10, 2):
    print(i)
# Salida: 1, 3, 5, 7, 9
```

### Bucle `while`

El bucle `while` repite un bloque de código mientras una condición sea `True`. La sintaxis básica es:

```python
while condición:
    # Bloque de código a ejecutar
```

#### Ejemplo: Contador con `while`

```python
contador = 0

while contador < 5:
    print(contador)
    contador += 1
```

Este código comienza con `contador = 0` y sigue incrementándolo hasta que llega a 5. En cada iteración, se imprime el valor actual de `contador`.

### Control de Iteraciones

Python ofrece varias formas de controlar la ejecución de bucles:

1. **`break`**: Termina el bucle inmediatamente.
2. **`continue`**: Salta la iteración actual y pasa a la siguiente.
3. **`else`**: Un bloque que se ejecuta cuando el bucle termina de manera normal, es decir, sin que se haya usado `break`.

#### Ejemplo: Uso de `break`

```python
for i in range(10):
    if i == 5:
        break
    print(i)
# Salida: 0, 1, 2, 3, 4
```

El bucle se detiene cuando `i` llega a 5, debido a la instrucción `break`.

#### Ejemplo: Uso de `continue`

```python
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)
# Salida: 1, 3, 5, 7, 9
```

Aquí, `continue` hace que el bucle salte la iteración actual si `i` es par, por lo que solo se imprimen los números impares.

#### Ejemplo: Uso de `else` con Bucles

El bloque `else` en un bucle se ejecuta solo si el bucle no se interrumpe con `break`.

```python
for i in range(5):
    print(i)
else:
    print("Bucle completado.")
# Salida:
# 0
# 1
# 2
# 3
# 4
# Bucle completado.
```

Si el bucle se completa sin interrupciones, se ejecuta el bloque `else`. Sin embargo, si se usa `break`, el bloque `else` se omite.

#### Ejemplo: `else` con `break`

```python
for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("Bucle completado.")
# Salida:
# 0
# 1
# 2
```

En este caso, como el bucle se interrumpe con `break` cuando `i` es igual a 3, el bloque `else` no se ejecuta.

### Bucles Anidados

Puedes anidar bucles dentro de otros bucles para trabajar con estructuras de datos más complejas, como matrices.

```python
for i in range(3):
    for j in range(2):
        print(f"i = {i}, j = {j}")
```

Este código tiene un bucle `for` dentro de otro `for`. Para cada valor de `i`, el bucle interno recorre todos los valores de `j`.

### Bucles Infinito con `while`

Si la condición de un `while` siempre es verdadera, el bucle se ejecutará indefinidamente. Esto se conoce como un bucle infinito.

```python
while True:
    respuesta = input("¿Quieres salir? (s/n): ")
    if respuesta == 's':
        break
```

Este ejemplo sigue preguntando al usuario si quiere salir hasta que la respuesta sea `'s'`.

Estos son los fundamentos de los bucles y el control de iteraciones en Python. Son herramientas esenciales para ejecutar tareas repetitivas y para gestionar la lógica de flujo en tus programas.

## Generadores e Iteradores

Trabajar con iteradores y generadores en Python permite manejar grandes cantidades de datos de manera eficiente, sin necesidad de cargar todo en memoria.

### ¿Qué es un iterador y cómo se usa?

Un iterador en Python es un objeto que permite recorrer todos los elementos de una colección, uno a la vez, sin necesidad de usar índices. Para crear un iterador, se utiliza la función `iter()` y para obtener el siguiente elemento, se usa la función `next()`. Veamos un ejemplo:

```python
# Crear una lista
lista = [1, 2, 3, 4]

# Obtener el iterador de la lista
iterador = iter(lista)

# Usar el iterador para obtener elementos
print(next(iterador))  # Imprime: 1
print(next(iterador))  # Imprime: 2
print(next(iterador))  # Imprime: 3
print(next(iterador))  # Imprime: 4

# Intentar obtener otro elemento después de finalizar la iteración
print(next(iterador))  # Esto generará una excepción StopIteration
```

Los iteradores también pueden recorrer cadenas de texto:

```python
# Crear una cadena
texto = "hola mundo"

# Obtener el iterador de la cadena
iterador_texto = iter(texto)

# Iterar a través de la cadena
for caracter in iterador_texto:
    print(caracter)
```

### ¿Cómo crear un iterador con range para números impares?

La función `range` se puede usar para crear un iterador que recorra números impares:

```python
# Crear un iterador para números impares hasta 10
limite = 10
iterador_impares = iter(range(1, limite + 1, 2))

# Iterar a través de los números impares
for numero in iterador_impares:
    print(numero)
```

Para cambiar a números pares, solo se debe modificar el inicio del rango:

```python
# Crear un iterador para números pares hasta 10
iterador_pares = iter(range(0, limite + 1, 2))

# Iterar a través de los números pares
for numero in iterador_pares:
    print(numero)
```

### ¿Qué es un generador y cómo se utiliza?

Un generador es una función que produce una secuencia de valores sobre los cuales se puede iterar, usando la palabra clave `yield` en lugar de `return`. Aquí hay un ejemplo básico:

```python
def mi_generador():
    yield 1
    yield 2
    yield 3

# Usar el generador
for valor in mi_generador():
    print(valor)
```

### ¿Cómo crear un generador para la serie de Fibonacci?

La serie de Fibonacci es una secuencia donde cada número es la suma de los dos anteriores. Podemos crear un generador para producir esta serie:

```python
def fibonacci(limite):
    a, b = 0, 1
    while a < limite:
        yield a
        a, b = b, a + b

# Usar el generador para la serie de Fibonacci hasta 10
for numero in fibonacci(10):
    print(numero)
```

[9. Classes — Python 3.12.4 documentation](https://docs.python.org/3/tutorial/classes.html#iterators "9. Classes — Python 3.12.4 documentation")

[Functional Programming HOWTO — Python 3.12.4 documentation](https://docs.python.org/3/howto/functional.html#generators "Functional Programming HOWTO — Python 3.12.4 documentation")

## Uso de Funciones en Python

En Python, uno de los principios fundamentales es el de divide y vencerás. Esto se refiere a dividir el código en porciones más pequeñas para facilitar su legibilidad, mantenimiento y reutilización. Las funciones nos permiten encapsular lógica específica, evitando la duplicación de código.

### ¿Cómo se definen las funciones en Python?

Las funciones en Python se definen utilizando la palabra reservada `def,` seguida del nombre de la función y los parámetros que representan la información necesaria para su ejecución.

```python
def saludar():
    print("Hola, mundo")

saludar()
```

### ¿Cómo se utilizan los parámetros en una función?

Podemos agregar parámetros a una función para que reciba información dinámica. Por ejemplo, para saludar a una persona específica:

```python
def saludar(name):
    print(f"Hola, {name}")

saludar("Carla")
```

### ¿Cómo manejar múltiples parámetros en una función?

Las funciones pueden tener múltiples parámetros. Por ejemplo, para saludar con nombre y apellido:

```python
def saludar(name, last_name):
    print(f"Hola, {name} {last_name}")

saludar("Diego", "Antezano")
```

### ¿Qué ocurre si falta un argumento?

Si no se pasan todos los argumentos, Python generará un error indicando que falta un argumento posicional:

`saludar("Diego")`

### ¿Cómo definir valores predeterminados para parámetros?

Podemos asignar valores predeterminados a los parámetros, que se utilizarán si no se proporciona uno específico:

```python
def saludar(name, last_name="No tiene apellido"):
    print(f"Hola, {name} {last_name}")

saludar("Diego")
```

### ¿Cómo pasar parámetros por nombre?

Podemos pasar parámetros por nombre, lo que permite cambiar el orden de los argumentos:

`saludar(last_name="Florida", name="Carla")`

### ¿Cómo crear una calculadora con funciones en Python?

Podemos definir funciones para operaciones básicas y una función principal para manejarlas:

```python
def suma(a, b):
    return a + b

def resta(a, b):
    return a - b

def multiplicar(a, b):
    return a * b

def dividir(a, b):
    return a / b

def calculadora():
    while True:
        print("Seleccione una operación:")
        print("1. Suma")
        print("2. Resta")
        print("3. Multiplicación")
        print("4. División")
        print("5. Salir")
        
        opcion = int(input("Ingrese su opción: "))
        
        if opcion == 5:
            print("Saliendo de la calculadora.")
            break
        
        if opcion in [1, 2, 3, 4]:
            num1 = float(input("Ingrese el primer número: "))
            num2 = float(input("Ingrese el segundo número: "))
            
            if opcion == 1:
                print("La suma es:", suma(num1, num2))
            elif opcion == 2:
                print("La resta es:", resta(num1, num2))
            elif opcion == 3:
                print("La multiplicación es:", multiplicar(num1, num2))
            elif opcion == 4:
                print("La división es:", dividir(num1, num2))
        else:
            print("Opción no válida, por favor intente de nuevo.")

calculadora()
```

### ¿Qué se debe considerar al crear funciones en Python?

Es crucial tener en cuenta el tipo de datos que se manejan, validar entradas del usuario y asegurarse de que las funciones se llamen correctamente para evitar errores en la ejecución.

[PEP 8 – Style Guide for Python Code | peps.python.org](https://www.python.org/dev/peps/pep-0008/ "PEP 8 – Style Guide for Python Code | peps.python.org")

## Funciones Lambda y Programación Funcional en Python


### ¿Cómo utilizar lambda para operaciones básicas?

Para realizar operaciones sencillas con lambda, no necesitamos especificar el nombre de la función. Solo requerimos parámetros y la operación deseada. Por ejemplo, para sumar dos números, podemos definir una función lambda así:

```python
sumar = lambda a, b: a + b
print(sumar(10, 4))
```

### ¿Cómo utilizar lambda para multiplicaciones?

Podemos adaptar fácilmente lambda para realizar otras operaciones como la multiplicación:

```python
multiplicar = lambda a, b: a * b
print(multiplicar(80, 4))
```

### ¿Cómo aplicar lambda a elementos de una lista con map?

Cuando trabajamos con listas y queremos aplicar una función a cada elemento, map es útil junto con lambda. Por ejemplo, para obtener el cuadrado de los números del 0 al 10:

```python
numeros = list(range(11))
cuadrados = list(map(lambda x: x ** 2, numeros))
print("Cuadrados:", cuadrados)
```

### ¿Cómo filtrar elementos de una lista con lambda y filter?

Lambda también es útil para filtrar elementos que cumplen ciertas condiciones. Por ejemplo, para obtener los números pares de una lista:

```python
numeros_pares = list(filter(lambda x: x % 2 == 0, numeros))
print("Pares:", numeros_pares)
```

Como hemos visto, lambda ofrece una forma más sencilla de trabajar con funciones en Python sin comprometer su eficiencia. En la próxima clase, exploraremos temas más complejos donde las funciones serán el foco principal.

Las **funciones lambda** y la **programación funcional** en Python son conceptos poderosos que permiten escribir código más conciso, flexible y expresivo. Vamos a explorarlos en detalle.

### Funciones Lambda

Una **función lambda** en Python es una función anónima, es decir, una función que no tiene nombre y se define en una sola línea usando la palabra clave `lambda`. Estas funciones son útiles para operaciones simples y cortas que se pueden definir rápidamente sin la necesidad de una función formal.

#### Sintaxis de una Función Lambda

```python
lambda argumentos: expresión
```

- **`argumentos`**: Son los parámetros que la función tomará.
- **`expresión`**: Es una única expresión que se evalúa y devuelve como resultado de la función.

#### Ejemplo Simple

```python
# Función lambda que suma dos números
suma = lambda x, y: x + y

# Usar la función lambda
resultado = suma(3, 5)
print(resultado)  # Salida: 8
```

En este ejemplo, `suma` es una función lambda que toma dos argumentos, `x` y `y`, y devuelve su suma.

### Uso de Funciones Lambda con Funciones Integradas

Las funciones lambda son comúnmente usadas junto con funciones integradas como `map()`, `filter()`, y `sorted()`.

#### `map()`

`map()` aplica una función a todos los elementos de una secuencia.

```python
numeros = [1, 2, 3, 4]
cuadrados = list(map(lambda x: x ** 2, numeros))
print(cuadrados)  # Salida: [1, 4, 9, 16]
```

Aquí, `map()` aplica la función lambda a cada elemento de la lista `numeros`, devolviendo una nueva lista con los resultados.

#### `filter()`

`filter()` filtra los elementos de una secuencia según una función que devuelve `True` o `False`.

```python
numeros = [1, 2, 3, 4, 5, 6]
pares = list(filter(lambda x: x % 2 == 0, numeros))
print(pares)  # Salida: [2, 4, 6]
```

En este ejemplo, `filter()` utiliza la función lambda para seleccionar solo los números pares de la lista.

#### `sorted()`

`sorted()` ordena los elementos de una secuencia. Puedes usar una función lambda para definir la clave de ordenamiento.

```python
puntos = [(1, 2), (3, 1), (5, 4), (2, 0)]
ordenado_por_y = sorted(puntos, key=lambda punto: punto[1])
print(ordenado_por_y)  # Salida: [(2, 0), (3, 1), (1, 2), (5, 4)]
```

En este caso, `sorted()` ordena la lista de tuplas según el segundo elemento de cada tupla.

### Programación Funcional en Python

La **programación funcional** es un paradigma de programación que trata a las funciones como ciudadanos de primera clase, lo que significa que pueden ser pasadas como argumentos, retornadas desde otras funciones y asignadas a variables.

#### Principios Clave:

1. **Funciones como Primeras Clases**: Las funciones pueden ser asignadas a variables, almacenadas en estructuras de datos, y pasadas como argumentos.
2. **Inmutabilidad**: Prefiere el uso de datos inmutables, lo que significa que las estructuras de datos no se modifican después de su creación.
3. **Funciones Puras**: Una función pura es aquella que, dado el mismo conjunto de argumentos, siempre devuelve el mismo resultado y no tiene efectos secundarios.

#### Funciones de Orden Superior

Una función de orden superior es una función que toma una o más funciones como argumentos, o devuelve una función como resultado.

##### Ejemplo: Función de Orden Superior

```python
def aplicar_operacion(operacion, x, y):
    return operacion(x, y)

suma = lambda x, y: x + y
resultado = aplicar_operacion(suma, 5, 3)
print(resultado)  # Salida: 8
```

En este ejemplo, `aplicar_operacion` es una función de orden superior que recibe otra función `operacion` y dos números, aplicando `operacion` a estos números.

#### Composición de Funciones

La composición de funciones implica combinar funciones pequeñas para crear una función más compleja.

```python
def doble(x):
    return x * 2

def incrementar(x):
    return x + 1

def compuesto(f, g, x):
    return f(g(x))

resultado = compuesto(doble, incrementar, 3)
print(resultado)  # Salida: 8
```

Aquí, `compuesto` toma dos funciones (`f` y `g`) y un valor `x`, y aplica `g` a `x`, luego aplica `f` al resultado de `g(x)`.

### Funciones Integradas para Programación Funcional

- **`map(función, iterable)`**: Aplica `función` a cada elemento de `iterable`.
- **`filter(función, iterable)`**: Filtra `iterable` dejando solo los elementos donde `función` devuelva `True`.
- **`reduce(función, iterable)`**: Acumula los elementos de `iterable` aplicando `función` secuencialmente (requiere importar desde `functools`).

#### Ejemplo: `reduce()`

```python
from functools import reduce

numeros = [1, 2, 3, 4]
producto = reduce(lambda x, y: x * y, numeros)
print(producto)  # Salida: 24
```

Aquí, `reduce()` multiplica todos los números de la lista `numeros`.

### Ventajas y Consideraciones

- **Ventajas**:
  - Código más conciso y expresivo.
  - Facilita la creación de funciones reutilizables y composables.
  - Fomenta la inmutabilidad y la transparencia referencial.

- **Consideraciones**:
  - El abuso de funciones lambda puede hacer que el código sea difícil de leer.
  - La programación funcional puede ser menos intuitiva para principiantes en comparación con paradigmas más imperativos.

### Conclusión

Las funciones lambda y la programación funcional en Python ofrecen herramientas potentes para escribir código más modular, limpio y expresivo. Si bien no siempre es necesario adoptar la programación funcional en su totalidad, comprender estos conceptos y utilizarlos cuando sean apropiados puede mejorar la eficiencia y legibilidad de tu código.

## ¿Cómo realizar una función recursiva en Python?

La recursividad es una técnica fundamental en programación donde una función se llama a sí misma para resolver problemas complejos de manera más sencilla y estructurada.

### ¿Cómo se aplica la recursividad en el cálculo del factorial?

La recursividad se entiende mejor con ejemplos prácticos. El factorial de un número se define como el producto de todos los números desde ese número hasta 1. Por ejemplo, el factorial de 5 (5!) es 5 * 4 * 3 * 2 * 1.

En código Python, la función factorial se puede definir recursivamente de la siguiente manera:

```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
```

Este código sigue dos casos clave en la recursividad:

- **Caso base**: cuando `n` es 0, la función retorna 1.
- **Caso recursivo**: cuando n es mayor que 0, la función retorna n multiplicado por el factorial de n-1.

### ¿Cómo funciona la recursividad en la serie de Fibonacci?

La serie de Fibonacci es otra aplicación clásica de la recursividad. En esta serie, cada número es la suma de los dos anteriores, comenzando con 0 y 1. La fórmula es:

`[ F(n) = F(n-1) + F(n-2) ]`

El código Python para calcular el número n-ésimo en la serie de Fibonacci usando recursividad es el siguiente:

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

Aquí también se siguen dos casos:

- **Caso base**: cuando `n` es 0 o 1, la función retorna `n`.
- **Caso recursivo**: para otros valores de `n`, la función retorna la suma de `fibonacci(n-1)` y `fibonacci(n-2)`.

## Manejo de Excepciones en Python y uso de pass

Las excepciones en Python están organizadas en una jerarquía de clases, donde las excepciones más generales se encuentran en la parte superior y las más específicas en la parte inferior.

Esta organización jerárquica permite a los programadores manejar excepciones de manera más precisa y efectiva.

```python
numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
transformados = [x * 2 if x % 2 == 0 else x for x in numeros]
print("Números transformados:", transformados)
```

Por ejemplo, la excepción Exception es la clase base para la mayoría de las excepciones, y de ella derivan subclases como `ArithmeticError` y `ValueError`.

Comprender esta jerarquía es crucial para poder manejar las excepciones adecuadamente y elegir las excepciones específicas que se desean capturar.

A continuación se muestra un código que imprime la jerarquía de excepciones en Python:

```python
def print_exception_hierarchy(exception_class, indent=0):
    print(' ' * indent + exception_class.__name__)
    for subclass in exception_class.__subclasses__():
        print_exception_hierarchy(subclass, indent + 4)

# Imprimir la jerarquía comenzando desde la clase base Exception
print_exception_hierarchy(Exception)
```

Este código utiliza recursión para recorrer y mostrar las subclases de excepciones, permitiéndote visualizar cómo están organizadas y relacionadas entre sí.

Entender la jerarquía de excepciones en Python es fundamental para escribir código robusto y manejable. Al conocer las relaciones entre las diferentes excepciones, puedes capturar errores de manera más específica, lo que te permite implementar manejadores de excepciones más precisos y efectivos.

En Python, el manejo de excepciones se realiza utilizando bloques `try`, `except`, `else`, y `finally`. También, el uso de la palabra clave `pass` permite manejar situaciones excepcionales sin hacer nada, simplemente ignorando el error o condición. Aquí te explico ambos conceptos con ejemplos:

### 1. Manejo de Excepciones
El manejo de excepciones permite capturar errores durante la ejecución de un programa, evitando que este se detenga de forma abrupta.

La estructura básica es:

```python
try:
    # Código que puede lanzar una excepción
except TipoDeExcepción:
    # Código para manejar la excepción
else:
    # Código que se ejecuta si no ocurre ninguna excepción
finally:
    # Código que siempre se ejecuta (opcional)
```

### Ejemplo de Manejo de Excepciones

```python
try:
    numero = int(input("Introduce un número: "))
    print(f"El número ingresado es: {numero}")
except ValueError:
    print("Error: Debes introducir un número válido.")
else:
    print("No ocurrió ninguna excepción.")
finally:
    print("Finalizando la operación.")
```

- **try:** Intenta ejecutar el código que puede generar una excepción.
- **except:** Captura la excepción `ValueError` (que ocurre cuando intentas convertir un valor no numérico a entero).
- **else:** Se ejecuta si no hay ninguna excepción.
- **finally:** Este bloque siempre se ejecuta, ocurra o no una excepción.

### 2. Uso de `pass`
La palabra clave `pass` se utiliza para indicar que no se realizará ninguna acción en un bloque de código. Esto es útil cuando no deseas manejar la excepción de inmediato o cuando estás creando código de prueba.

### Ejemplo con `pass`

```python
try:
    numero = int(input("Introduce un número: "))
except ValueError:
    pass  # Ignora el error si ocurre un ValueError
else:
    print(f"El número ingresado es: {numero}")
```

En este caso, si el usuario introduce un valor que no es un número, el programa no hará nada con el error y continuará ejecutándose sin mostrar un mensaje de error.

### Resumen:
- **Manejo de Excepciones**: Se utiliza `try` y `except` para capturar y manejar errores.
- **`pass`**: Permite ignorar errores o implementar código sin realizar ninguna acción específica cuando ocurre una excepción.

**Lecturas recomendadas**

[4. Más herramientas para control de flujo — documentación de Python - 3.12.5](https://docs.python.org/es/3/tutorial/controlflow.html#pass "4. Más herramientas para control de flujo — documentación de Python - 3.12.5")

[8. Errors and Exceptions — Python 3.12.5 documentation](https://docs.python.org/es/3/tutorial/errors.html "8. Errors and Exceptions — Python 3.12.5 documentation")


¿Te estresas cuando te aparece un error en tu código? No te preocupes, todos los programadores nos enfrentamos a errores constantemente. De hecho, encontrar y solucionar errores es parte del trabajo diario de un programador. Sin embargo, lo que distingue a un buen programador de un excelente programador es la habilidad para manejar esos errores de manera efectiva. En este blog, exploraremos qué son las excepciones y los errores, por qué es importante manejarlos adecuadamente, y cómo hacerlo en Python.

### Errores y Excepciones

Los términos “errores” y “excepciones” en el código a menudo se utilizan indistintamente, pero tienen diferencias clave:

- **Errores:** Son problemas en el código que pueden ser sintácticos (como errores de escritura) o semánticos (como errores en la lógica del programa). Los errores detienen la ejecución del programa.
- **Excepciones:** Son eventos que ocurren durante la ejecución de un programa y que alteran el flujo normal del código. A diferencia de los errores, las excepciones pueden ser manejadas para evitar que el programa se detenga.

Comprender los errores y las excepciones es vital porque:

- **Mejora la calidad del código:** Permite escribir programas más robustos y menos propensos a fallos.
- **Facilita la depuración:** Ayuda a identificar y solucionar problemas de manera más eficiente.
- **Mejora la experiencia del usuario:** Evita que el programa se cierre abruptamente, ofreciendo mensajes de error claros y manejables.

### Errores Básicos en Python

Antes de profundizar en el manejo de excepciones, es importante familiarizarnos con algunos errores comunes en Python.

### SyntaxError

El `SyntaxError` ocurre cuando hay un error en la sintaxis del código. Por ejemplo:

```python
# Código con SyntaxError
print("Hola Mundo"
```

Resultado:

`SyntaxError: unexpected EOF while parsing`

### TypeError

El `TypeError` se produce cuando se realiza una operación en un tipo de dato inapropiado. Por ejemplo:

```python
# Código con TypeError
resultado = "10" + 5
```

Resultado:

`TypeError: can only concatenate str (not "int") to str`

Estos son solo algunos ejemplos de los errores más comunes que se pueden encontrar en Python. Ahora, veamos cómo manejar excepciones para evitar que estos errores detengan la ejecución de nuestro programa.

### La Estructura del try-except

En Python, el manejo de excepciones se realiza principalmente a través de la estructura `try-except`. Esta estructura permite intentar ejecutar un bloque de código y capturar las excepciones que puedan ocurrir, proporcionando una forma de manejar los errores de manera controlada. Esto no solo evita que el programa se detenga abruptamente, sino que también ofrece la oportunidad de informar al usuario sobre lo que salió mal y cómo puede solucionarlo.

### ¿Qué hace `try`?

La palabra clave `try` se utiliza para definir un bloque de código donde se anticipa que puede ocurrir un error. Python ejecuta este bloque y, si ocurre una excepción, transfiere el control al bloque `except`.

### ¿Qué hace `except`?

La palabra clave `except` define un bloque de código que se ejecuta si ocurre una excepción en el bloque `try`. Aquí es donde podemos manejar el error, limpiar el desorden, o proporcionar mensajes informativos al usuario.

### Estructura Básica

La estructura básica de `try-except` es la siguiente:

```python
try:
    # Código que puede generar una excepción
    pass
except NombreDeLaExcepcion:
    # Código que maneja la excepción
    pass
```
### ¿Por qué es importante manejar las excepciones?

Permitir que los errores sigan su curso sin control puede tener varias consecuencias negativas:

- **Interrupción del programa:** Un error no manejado puede hacer que tu programa se detenga abruptamente, causando frustración en los usuarios.
- **Pérdida de datos:** Si el programa se cierra inesperadamente, es posible que se pierdan datos importantes no guardados.
- **Mala experiencia del usuario:** Los usuarios prefieren programas que manejen errores de manera elegante y les proporcionen mensajes claros sobre lo que salió mal y cómo pueden solucionarlo.

Manejar las excepciones con `try-except` permite:

- **Continuidad del programa:** Permite que el programa continúe ejecutándose incluso cuando se encuentra un error.
- **Mensajes de error claros:** Proporciona mensajes específicos que pueden ayudar al usuario a corregir el problema.
- **Mejor depuración:** Facilita la identificación y corrección de errores, haciendo el proceso de depuración más eficiente.

### Ejemplo de try-except

```python
try:
    valor = int(input("Ingresa un número: "))
    resultado = 10 / valor
    print(f"El resultado es {resultado}")
except ValueError:
    print("Por favor, ingresa un número válido.")
except ZeroDivisionError:
    print("No se puede dividir por cero.")
```

Resultado:

```python
# Si el usuario ingresa "a":
Por favor, ingresa un número válido.

# Si el usuario ingresa "0":
No se puede dividir por cero.
```

### Jerarquía de Excepciones

En Python, las excepciones están organizadas en una jerarquía, donde las excepciones más generales se encuentran en la parte superior y las más específicas en la parte inferior. Por ejemplo:

- Exception
 - ArithmeticError
  	- ZeroDivisionError
 - ValueError
 
Conocer esta jerarquía es útil para manejar excepciones de manera más precisa y efectiva.

### Ejemplos Prácticos

#### Ejemplo 1: Manejo de ValueError

```python
try:
    edad = int(input("Introduce tu edad: "))
    print(f"Tu edad es {edad}")
except ValueError:
    print("Error: Debes introducir un número.")
```

Resultado:

```python
# Si el usuario ingresa "veinte":
Error: Debes introducir un número.
```

### Ejemplo 2: Manejo de múltiples excepciones

```python
try:
    divisor = int(input("Ingresa un número divisor: "))
    resultado = 100 / divisor
    print(f"El resultado es {resultado}")
except ValueError:
    print("Error: Debes introducir un número válido.")
except ZeroDivisionError:
    print("Error: No se puede dividir por cero.")
```

Resultado:

```python
# Si el usuario ingresa "cero":
Error: No se puede dividir por cero.
```

### Ejemplo 3: Manejo General de Excepciones

```python
try:
    nombre = input("Introduce tu nombre: ")
    print(f"Hola, {nombre}!")
except Exception as e:
    print(f"Ha ocurrido un error: {e}")
```

Resultado:

`Hola, Juan!`

### Explorando la Jerarquía de Excepciones en Python

En Python, las excepciones están organizadas en una jerarquía de clases, donde cada excepción específica es una subclase de la clase base `Exception`.

El código proporcionado utiliza recursión para imprimir esta jerarquía comenzando desde la clase base `Exception`. Cada clase de excepción se muestra indentada según su nivel en la jerarquía, lo que ayuda a visualizar cómo están relacionadas las excepciones entre sí.

```python
def print_exception_hierarchy(exception_class, indent=0):
    print(' ' * indent + exception_class.__name__)
    for subclass in exception_class.__subclasses__():
        print_exception_hierarchy(subclass, indent + 4)

# Imprimir la jerarquía comenzando desde la clase base Exception
print_exception_hierarchy(Exception)
```
Ahora te toca a ti! Ejecuta el código y observa cómo se imprime la jerarquía de excepciones, esto te permitirá comprender mejor cómo están estructuradas las excepciones y cómo puedes aprovechar esta estructura para manejar errores de manera más efectiva en tus programas.

### Uso de `pass`

Antes de concluir, hablemos del uso de `pass`. Imagina que estás creando una función que sabes que vas a necesitar, pero no quieres crear la lógica ahora mismo y sigues con el código. Si solo pones el `def` sin el cuerpo o código dentro del nivel de indentación de la función, te producirá un error `IndentationError`.

Puedes solucionar esto usando `pass`, que es una declaración nula; no hace nada cuando se ejecuta, pero es útil como un marcador de posición para que la estructura del código sea válida.

### Ejemplo de `pass`

```python
def mi_funcion():
    pass  # Marcador de posición para el cuerpo de la función

print("Continuando con el código...")
```

Resultado:

`Continuando con el código...`

En este ejemplo, `pass` permite definir la estructura de la función sin implementar la lógica de inmediato, evitando errores de indentación y permitiendo continuar con el desarrollo del código.

A medida que utilices nuevas herramientas en Python, como librerías y otros tipos de datos, te encontrarás con excepciones específicas de esas herramientas. Familiarizarte con las excepciones comunes de cada librería te permitirá manejarlas de manera más efectiva y escribir código más robusto. Recuerda, el manejo adecuado de excepciones no solo mejora tu código, sino que también te convierte en un programador más competente y confiable.

### ¿Cómo realizar una función recursiva en Python?

La recursividad es una técnica fundamental en programación donde una función se llama a sí misma para resolver problemas complejos de manera más sencilla y estructurada.

### ¿Cómo se aplica la recursividad en el cálculo del factorial?

La recursividad se entiende mejor con ejemplos prácticos. El factorial de un número se define como el producto de todos los números desde ese número hasta 1. Por ejemplo, el factorial de 5 (5!) es 5 * 4 * 3 * 2 * 1.

En código Python, la función factorial se puede definir recursivamente de la siguiente manera:

```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
```

Este código sigue dos casos clave en la recursividad:

- **Caso base:** cuando n es 0, la función retorna 1.
- **Caso recursivo:** cuando n es mayor que 0, la función retorna n multiplicado por el factorial de n-1.

### ¿Cómo funciona la recursividad en la serie de Fibonacci?

La serie de Fibonacci es otra aplicación clásica de la recursividad. En esta serie, cada número es la suma de los dos anteriores, comenzando con 0 y 1. La fórmula es:

`[ F(n) = F(n-1) + F(n-2) ]`

El código Python para calcular el número n-ésimo en la serie de Fibonacci usando recursividad es el siguiente:

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

Aquí también se siguen dos casos:

- **Caso base:** cuando n es 0 o 1, la función retorna n.
- **Caso recursivo:** para otros valores de n, la función retorna la suma de fibonacci(n-1) y fibonacci(n-2).

### Fundamentos de Programación Orientada a Objetos en Python

La programación orientada a objetos (POO) es un paradigma de la programación que se basa en organizar el software en objetos, los cuales son instancias de clases. Las clases actúan como plantillas genéricas que definen atributos y comportamientos. Por ejemplo, una clase “Persona” puede tener atributos como nombre, apellido y fecha de nacimiento.

### ¿Cómo se crean clases y objetos en Python?

Para crear una clase en Python, se utiliza la palabra reservada class seguida del nombre de la clase con la primera letra en mayúscula. Dentro de la clase, se define un constructor con la función `__init__`. Esta función inicializa los atributos del objeto.

#### Ejemplo de creación de una clase y objeto

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
    
    def saludar(self):
        print(f"Hola, mi nombre es {self.nombre} y tengo {self.edad} años")

# Crear objetos de la clase Persona
persona1 = Persona("Ana", 30)
persona2 = Persona("Luis", 25)

persona1.saludar()
persona2.saludar()
```

### ¿Qué son los métodos en una clase?

Los métodos son funciones definidas dentro de una clase que operan sobre los objetos de la misma. En el ejemplo anterior, saludar es un método de la clase Persona.

### ¿Cómo manejar una cuenta bancaria con POO?

Un ejemplo práctico de POO es la gestión de una cuenta bancaria. Creamos una clase BankAccount con métodos para depositar y retirar dinero, así como para activar y desactivar la cuenta.

### Ejemplo de clase BankAccount

```python
class BankAccount:
    def __init__(self, account_holder, balance):
        self.account_holder = account_holder
        self.balance = balance
        self.is_active = True
    
    def deposit(self, amount):
        if self.is_active:
            self.balance += amount
            print(f"Se ha depositado {amount}. Saldo actual: {self.balance}")
        else:
            print("No se puede depositar, cuenta inactiva")
    
    def withdraw(self, amount):
        if self.is_active:
            if amount <= self.balance:
                self.balance -= amount
                print(f"Se ha retirado {amount}. Saldo actual: {self.balance}")
            else:
                print("Fondos insuficientes")
        else:
            print("No se puede retirar, cuenta inactiva")
    
    def deactivate(self):
        self.is_active = False
        print("La cuenta ha sido desactivada")
    
    def activate(self):
        self.is_active = True
        print("La cuenta ha sido activada")

# Crear objetos de la clase BankAccount
cuenta1 = BankAccount("Ana", 500)
cuenta2 = BankAccount("Luis", 1000)

cuenta1.deposit(500)
cuenta2.withdraw(100)
cuenta1.deactivate()
cuenta1.deposit(200)
cuenta1.activate()
cuenta1.deposit(200)
```

### ¿Cómo se crean y manejan objetos en Python?

La creación de objetos sigue una sintaxis similar a la de la creación de variables, pero usando el nombre de la clase seguido de los parámetros necesarios para el constructor.

### Ejemplo de uso de la clase BankAccount

```python
# Creación de cuentas
cuenta1 = BankAccount("Ana", 500)
cuenta2 = BankAccount("Luis", 1000)

# Realización de operaciones
cuenta1.deposit(500)
cuenta2.withdraw(100)
cuenta1.deactivate()
cuenta1.deposit(200)  # No se puede depositar, cuenta inactiva
cuenta1.activate()
cuenta1.deposit(200)  # Depósito exitoso
```

## Fundamentos de la Programación Orientada a Objetos (POO)

La Programación Orientada a Objetos es un paradigma de programación que organiza el diseño del software en torno a objetos. Los objetos son instancias de clases, que pueden tener atributos (datos) y métodos (funciones).

**Conceptos Clave**

- **Clase:** Es un molde o plantilla que define los atributos y métodos que tendrán los objetos.
- **Objeto:** Es una instancia de una clase.
- **Atributo:** Es una variable que pertenece a una clase o a un objeto.
- **Método:** Es una función que pertenece a una clase o a un objeto.
- **Herencia:** Es un mecanismo por el cual una clase puede heredar atributos y métodos de otra clase.
- **Encapsulamiento:** Es el concepto de ocultar los detalles internos de un objeto y exponer sólo lo necesario.
- **Polimorfismo:** Es la capacidad de diferentes clases de ser tratadas como instancias de la misma clase a través de una interfaz común.

La Programación Orientada a Objetos (POO) es un paradigma de programación que se basa en el uso de **objetos** para diseñar y desarrollar aplicaciones. Este enfoque se centra en crear software que sea modular, reutilizable y más fácil de mantener. Aquí están los fundamentos clave de la POO:

### 1. **Clases y Objetos**
   - **Clase:** Una clase es un molde o plantilla que define las propiedades (atributos) y comportamientos (métodos) que tendrán los objetos creados a partir de ella. Por ejemplo, una clase `Coche` podría tener atributos como `color`, `marca`, y `modelo`, y métodos como `arrancar()` o `frenar()`.
   - **Objeto:** Un objeto es una instancia de una clase. Es un elemento concreto que se crea a partir de una clase, con valores específicos para sus atributos. Por ejemplo, un objeto de la clase `Coche` podría ser un coche rojo de la marca `Toyota` y modelo `Corolla`.

### 2. **Encapsulamiento**
   - El encapsulamiento consiste en ocultar los detalles internos de un objeto y exponer sólo lo necesario. Esto se logra mediante el uso de **modificadores de acceso** como `private`, `protected`, y `public`, que controlan qué partes de un objeto pueden ser accedidas o modificadas desde fuera de su clase. Esto protege los datos y asegura que solo se modifiquen de manera controlada.

### 3. **Herencia**
   - La herencia permite crear nuevas clases basadas en clases existentes. Una clase que hereda de otra (llamada **subclase** o **clase derivada**) toma los atributos y métodos de la clase base (llamada **superclase** o **clase padre**), y puede añadir o modificar funcionalidades. Esto fomenta la reutilización del código y la creación de jerarquías de clases. Por ejemplo, si tenemos una clase `Vehículo`, podemos crear una subclase `Coche` que herede de `Vehículo`.

### 4. **Polimorfismo**
   - El polimorfismo permite que un mismo método o función pueda tener diferentes comportamientos según el objeto que lo invoque. Existen dos tipos principales de polimorfismo:
     - **Polimorfismo en tiempo de compilación (sobrecarga):** Permite definir varios métodos con el mismo nombre pero diferentes parámetros.
     - **Polimorfismo en tiempo de ejecución (sobreescritura):** Permite que una subclase redefina un método de su superclase para modificar su comportamiento.

### 5. **Abstracción**
   - La abstracción consiste en representar conceptos esenciales sin incluir detalles de implementación específicos. Las clases abstractas y las interfaces son herramientas que permiten definir métodos sin implementarlos, dejando que las clases derivadas proporcionen la implementación. Esto facilita la creación de sistemas flexibles y extensibles.

### 6. **Modularidad**
   - La POO promueve la división del software en módulos o componentes independientes (objetos), que pueden ser desarrollados, testeados y mantenidos por separado, pero que funcionan juntos como un todo coherente.

### 7. **Relaciones entre Objetos**
   - Las clases y objetos pueden relacionarse de varias maneras, como:
     - **Asociación:** Una relación donde un objeto utiliza a otro.
     - **Agregación:** Una forma más débil de asociación, donde un objeto contiene referencias a otros objetos.
     - **Composición:** Una forma más fuerte de agregación, donde un objeto contiene y controla completamente a otros objetos.

### Ventajas de la POO:
- **Reutilización de código:** Las clases pueden reutilizarse en diferentes partes de un programa o en diferentes proyectos.
- **Facilidad de mantenimiento:** El encapsulamiento y la modularidad facilitan la localización y corrección de errores.
- **Facilidad de expansión:** La herencia y la abstracción permiten agregar nuevas funcionalidades sin alterar el código existente.
- **Flexibilidad:** El polimorfismo permite que el código sea más flexible y fácil de extender.

Estos fundamentos hacen de la POO un enfoque poderoso y ampliamente utilizado en el desarrollo de software moderno.

## Ejercicio Biblioteca con POO

### Reto

Desarrolla una concesionaria de vehículos en la cual se puedan gestionar las compras y ventas de vehículos. Un usuario podrá ver los vehículos disponibles, su precio y realizar la compra de uno. Aplica los conceptos de programación orientada a objetos vistos en este ejercicio.

## Herencia en POO con Python

El concepto de herencia en programación permite que una clase derive atributos y métodos de otra, facilitando la reutilización de código y la creación de estructuras jerárquicas lógicas. En este ejercicio, se aplica herencia para modelar una concesionaria que vende autos, bicicletas y camiones.

### ¿Cómo se crea la clase base para los vehículos?

Primero, se define una clase base llamada `Vehículo`, que contiene atributos comunes como marca, modelo, precio y disponibilidad. Los métodos básicos incluyen verificar disponibilidad, obtener el precio y vender el vehículo.

```python
class Vehículo:
    def __init__(self, marca, modelo, precio):
        self.marca = marca
        self.modelo = modelo
        self.precio = precio
        self.disponible = True

    def vender(self):
        if self.disponible:
            self.disponible = False
            print(f"El vehículo {self.marca} ha sido vendido.")
        else:
            print(f"El vehículo {self.marca} no está disponible.")

    def estado(self):
        return self.disponible

    def get_price(self):
        return self.precio
```

### ¿Cómo se implementa la herencia en las clases derivadas?

Las clases` Auto`, `Bicicleta` y `Camión` heredan de `Vehículo`. Cada una puede personalizar métodos específicos según sus necesidades.

```python
class Auto(Vehículo):
    def start(self):
        if self.disponible:
            return f"El motor del coche {self.marca} está en marcha."
        else:
            return f"El coche {self.marca} no está disponible."

    def stop(self):
        if self.disponible:
            return f"El motor del coche {self.marca} se ha detenido."
        else:
            return f"El coche {self.marca} no está disponible."
```

### ¿Cómo se manejan las instancias de las clases en la concesionaria?

Se crean instancias de `Auto`, `Cliente` y `Concesionaria` para manejar el inventario y las ventas.

```python
class Cliente:
    def __init__(self, nombre):
        self.nombre = nombre
        self.autos = []

    def comprar_auto(self, auto):
        if auto.estado():
            self.autos.append(auto)
            auto.vender()
        else:
            print(f"El auto {auto.marca} no está disponible.")

class Concesionaria:
    def __init__(self):
        self.inventario = []
        self.clientes = []

    def añadir_auto(self, auto):
        self.inventario.append(auto)

    def registrar_cliente(self, cliente):
        self.clientes.append(cliente)

    def mostrar_disponibles(self):
        for auto in self.inventario:
            if auto.estado():
                print(f"{auto.marca} {auto.modelo} está disponible por {auto.get_price()}.")
```

### ¿Cómo se aplican las operaciones en la concesionaria?

Finalmente, se crean instancias y se realizan operaciones para mostrar la funcionalidad del sistema.

```python
# Crear autos
auto1 = Auto("Toyota", "Corolla", 20000)
auto2 = Auto("Honda", "Civic", 22000)
auto3 = Auto("Ford", "Mustang", 30000)

# Crear cliente
cliente = Cliente("Carlos")

# Crear concesionaria
concesionaria = Concesionaria()
concesionaria.añadir_auto(auto1)
concesionaria.añadir_auto(auto2)
concesionaria.añadir_auto(auto3)
concesionaria.registrar_cliente(cliente)

# Mostrar autos disponibles
concesionaria.mostrar_disponibles()

# Comprar auto
cliente.comprar_auto(auto1)

# Mostrar autos disponibles después de la compra
concesionaria.mostrar_disponibles()
```

### ¿Qué beneficios trae la herencia en este contexto?

- **Reutilización de código:** Las clases derivadas heredan atributos y métodos comunes.
- **Mantenimiento:** Facilita el mantenimiento y la actualización del código.
- **Extensibilidad:** Permite agregar nuevas clases derivadas con facilidad.

**Lecturas recomendadas**

[9. Classes — Python 3.12.4 documentation](https://docs.python.org/3/tutorial/classes.html#inheritance "9. Classes — Python 3.12.4 documentation")

## Los 4 pilares de la programacion orientada a objetos

Programar con objetos puede parecer complejo al principio, pero entender sus pilares fundamentales te facilitará mucho la tarea. Vamos a ver cómo aplicar abstracción, encapsulamiento, herencia y polimorfismo en un código sencillo.

### ¿Qué es la abstracción en programación orientada a objetos?

La abstracción te permite definir estructuras básicas sin entrar en detalles específicos. En el código, hemos creado instancias de diferentes vehículos, como un auto, una bicicleta y un camión, asignándoles atributos como marca, modelo y precio. Este enfoque nos permite trabajar con conceptos generales antes de precisar características específicas.

### ¿Cómo se aplica el encapsulamiento?

El encapsulamiento se refiere a mantener los datos privados dentro de una clase y acceder a ellos solo mediante métodos públicos. En nuestro ejemplo, las variables de instancia de los vehículos son privadas. Solo podemos acceder a ellas a través de métodos específicos, como GetPrice o verificarDisponibilidad, asegurando así que los datos se manejen de manera controlada y segura.

### ¿Qué rol juega la herencia?

La herencia permite que una clase hija adopte atributos y métodos de una clase padre. Aquí, la clase auto hereda de la clase vehículo, lo que significa que todas las características y comportamientos definidos en vehículo están disponibles en auto sin necesidad de duplicar el código. Este principio facilita la reutilización y extensión del código.

### ¿Qué es el polimorfismo y cómo se usa?

El polimorfismo permite que diferentes clases respondan a los mismos métodos de maneras distintas. En nuestro caso, tanto el auto como la bicicleta heredan métodos de vehículo, pero cada uno los implementa de forma diferente. Por ejemplo, el método para indicar que el auto está en marcha difiere del método de la bicicleta, que no usa motor. Este comportamiento flexible es clave para escribir código más dinámico y reutilizable.

## Uso de super() en Python

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print("Hello! I am a person.")

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def greet(self):
        super().greet()
        print(f"Hi, I'm {self.name}, and I'm a student with ID: {self.student_id}")

# Crear instancia de Student y llamar a greet
student = Student("Ana", 20, "S12345")
student.greet()  # Output: Hello! I am a person.
                 #         Hi, I'm Ana, and I'm a student with ID: S12345
```

El uso de `super()` en Python está relacionado con la herencia de clases. Permite llamar métodos o acceder a atributos de la **clase padre** desde una **clase hija**, sin necesidad de referenciar explícitamente el nombre de la clase padre. Es especialmente útil cuando se trabaja con herencia múltiple o se desea extender la funcionalidad de la clase base sin sobrescribir completamente su comportamiento.

### Sintaxis básica de `super()`

```python
super().metodo_de_la_clase_padre()
```

Aquí, `super()` devuelve un objeto que representa a la clase base, permitiendo llamar a sus métodos o acceder a sus atributos.

### Ejemplo básico con herencia y `super()`

```python
class Animal:
    def __init__(self, nombre):
        self.nombre = nombre
    
    def hacer_sonido(self):
        print("El animal hace un sonido")

class Perro(Animal):
    def __init__(self, nombre, raza):
        # Llamamos al constructor de la clase padre con super()
        super().__init__(nombre)
        self.raza = raza
    
    def hacer_sonido(self):
        # Extendemos la funcionalidad del método de la clase padre
        super().hacer_sonido()
        print("El perro ladra")

# Uso de las clases
mi_perro = Perro("Firulais", "Golden Retriever")
print(mi_perro.nombre)  # Firulais
mi_perro.hacer_sonido() 
```

#### Salida:
```
El animal hace un sonido
El perro ladra
```

### Explicación:
1. **`super().__init__(nombre)`**: Llama al constructor de la clase base `Animal`, lo que permite que la clase hija `Perro` también inicialice la variable `nombre` que está definida en la clase `Animal`.
2. **`super().hacer_sonido()`**: Llama al método `hacer_sonido` de la clase base `Animal` antes de agregar el comportamiento específico de `Perro`.

### Beneficios de usar `super()`:
1. **Herencia Múltiple**: En casos donde una clase hereda de múltiples clases, `super()` sigue el **orden de resolución de métodos (MRO)**, lo que garantiza que se llame al método correcto en la cadena de herencia.
2. **Reutilización**: Permite reutilizar el código de la clase base, extendiendo su funcionalidad sin necesidad de duplicar código.
3. **Mantenibilidad**: Si el nombre de la clase base cambia, no es necesario modificar los métodos que usan `super()`, ya que no se hace referencia directa a la clase padre.

### Ejemplo con Herencia Múltiple

```python
class Mamifero:
    def __init__(self, nombre):
        self.nombre = nombre

    def hacer_sonido(self):
        print("Sonido de mamífero")

class Volador:
    def __init__(self, velocidad):
        self.velocidad = velocidad

    def volar(self):
        print(f"El animal vuela a {self.velocidad} km/h")

class Murcielago(Mamifero, Volador):
    def __init__(self, nombre, velocidad):
        super().__init__(nombre)  # Llama al constructor de Mamifero
        Volador.__init__(self, velocidad)  # Llama al constructor de Volador

# Uso de las clases
bat = Murcielago("Batty", 50)
bat.hacer_sonido()  # Sonido de mamífero
bat.volar()  # El animal vuela a 50 km/h
```

En este ejemplo, `Murcielago` hereda de ambas clases, `Mamifero` y `Volador`. `super()` es útil para llamar al constructor de `Mamifero`, pero también es necesario llamar explícitamente al constructor de `Volador` para inicializar `velocidad`.

### Resumen:
- `super()` es una herramienta poderosa para interactuar con clases base.
- Facilita la herencia en programas orientados a objetos, permitiendo la reutilización y extensión del comportamiento de las clases padres.
- Es crucial en escenarios de herencia múltiple para seguir el orden correcto de llamadas a métodos.

**Lecturas recomendadas**
[Funciones incorporadas — documentación de Python - 3.12.5](https://docs.python.org/es/3/library/functions.html#super "Funciones incorporadas — documentación de Python - 3.12.5")

## Superando los Fundamentos de Programación Orientada a Objetos en Python

Para entender mejor la Programación Orientada a Objetos (POO), es esencial recordar los conceptos básicos de atributos y métodos.

- **Atributos:** Son variables que pertenecen a una clase o a sus objetos. Definen las propiedades de un objeto. Por ejemplo, pensemos en una persona: ¿Qué caracteriza a una persona en general? Las personas tienen nombre, edad, dirección, etc. En términos de POO, estos serían los atributos de la clase Person.
- **Métodos:** Son funciones definidas dentro de una clase que operan sobre sus objetos. Definen los comportamientos de un objeto. Siguiendo con el ejemplo de una persona, ¿Qué acciones puede realizar una persona? Puede hablar, caminar, comer, etc. En POO, estas acciones serían métodos de la clase Person.

### Ejemplo Básico de una Clase

```python
class Person:
    def __init__(self, name, age):
        self.name = name  # Atributo
        self.age = age    # Atributo

    def greet(self):
        print(f"Hola, mi nombre es {self.name} y tengo {self.age} años.")  # Método

# Crear una instancia de la clase Person
persona1 = Person("Ana", 30)
persona1.greet()  # Output: Hola, mi nombre es Ana y tengo 30 años.
```

al usar herencia vimos el método init() que es le cosntructor, el mismo es llamado automáticamente cuando se crea una nueva instancia de una clase y se utiliza para inicializar los atributos del objeto.

### Ejemplo de Constructor

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Crear una instancia de Person
person1 = Person("Ana", 30)
print(person1.name)  # Output: Ana
print(person1.age)   # Output: 30
```

En este ejemplo, el constructor `__init__` inicializa los atributos `name` y `age` de la clase `Person`.

### Uso de `super()` en Python

La función `super()` en Python te permite acceder y llamar a métodos definidos en la superclase desde una subclase. Esto es útil cuando quieres extender o modificar la funcionalidad de los métodos de la superclase sin tener que repetir su implementación completa.

### Ejemplo de Uso de `super()`

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print("Hello! I am a person.")

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def greet(self):
        super().greet()
        print(f"Hi, I'm {self.name}, and I'm a student with ID: {self.student_id}")

# Crear instancia de Student y llamar a greet
student = Student("Ana", 20, "S12345")
student.greet()  # Output: Hello! I am a person.
                 #         Hi, I'm Ana, and I'm a student with ID: S12345
```

En este ejemplo:

- La clase Person define un método `greet()` que imprime un saludo genérico.
- La clase `Student`, que es una subclase de `Person`, utiliza `super().__init__(name, age)` para llamar al constructor de la superclase `Person` y luego sobrescribe el método `greet()` para agregar información específica del estudiante.

### Jerarquía de Clases y Constructores

¿Qué sucede si una clase tiene una clase padre y esa clase padre tiene otra clase padre? En este caso, usamos super() para asegurar que todas las clases padre sean inicializadas correctamente.

### Ejemplo de Jerarquía de Clases

```python
class LivingBeing:
    def __init__(self, name):
        self.name = name

class Person(LivingBeing):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def introduce(self):
        print(f"Hi, I'm {self.name}, {self.age} years old, and my student ID is {self.student_id}")

# Crear instancia de Student
student = Student("Carlos", 21, "S54321")
student.introduce()  # Output: Hi, I'm Carlos, 21 years old, and my student ID is S54321
```

En este ejemplo:

- LivingBeing es la clase base que inicializa el atributo `name`.
- Person es una subclase de LivingBeing que inicializa `name` a través de `super().__init__(name)` y luego inicializa age.
- `Student` es una subclase de `Person` que inicializa `name` y `age` a través de `super().__init__(name, age)` y luego inicializa `student_id`.

### Métodos que Vienen por Defecto en Python

En Python, todas las clases heredan de la clase base `object`. Esto significa que todas las clases tienen ciertos métodos por defecto, algunos de los cuales pueden ser útiles para personalizar el comportamiento de tus clases.

### Métodos por Defecto Más Comunes

- `__init__(self)`: Constructor de la clase. Es llamado cuando se crea una nueva instancia de la clase. Inicializa los atributos del objeto.
- `__str__(self)`: Devuelve una representación en cadena del objeto, utilizada por print() y str(). Este método es útil para proporcionar una representación legible del objeto.
- `__repr__(self)`: Devuelve una representación “oficial” del objeto, utilizada por repr(). Este método está diseñado para devolver una cadena que represente al objeto de manera que se pueda recrear.

### Ejemplo de Métodos `__str__` y `__repr__`

Vamos a crear una clase `Person `que sobrescriba los métodos `__str__` y `__repr__`.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name}, {self.age} años"

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age})"

# Crear instancias de Person
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

# Uso de __str__
print(person1)  # Output: Alice, 30 años

# Uso de __repr__
print(repr(person1))  # Output: Person(name=Alice, age=30)
```

### Explicación del Código

- El método `__str__` devuelve una representación en cadena del objeto, útil para mensajes de salida amigables.
- El método `__repr__` devuelve una representación más detallada del objeto, útil para la depuración.

Estos métodos proporcionan una manera conveniente de representar y comparar objetos, lo que facilita la depuración y el uso de los objetos en el código.

### Importancia de Aprender estos Conceptos

Comprender y utilizar super(), los métodos por defecto y los constructores es crucial para escribir código limpio, eficiente y reutilizable en Python. Estos conceptos permiten:

- **Extender Funcionalidades:** `super()` permite extender las funcionalidades de una superclase sin duplicar código.
- **Inicialización Correcta:** El uso adecuado de constructores asegura que todos los atributos sean inicializados correctamente.
- **Personalizar Representaciones:** Métodos como `__str__` y` __repr__` permiten personalizar cómo se representan los objetos, facilitando la depuración y el manejo de datos.
- **Comparar y Ordenar Objetos:** Métodos como `__eq__`, `__lt__`, etc., permiten definir cómo se comparan y ordenan los objetos, lo cual es esencial para muchas operaciones de datos.

## Manejo de Archivos .TXT

El manejo de archivos `.txt` en Python se realiza principalmente con las funciones integradas para abrir, leer, escribir y cerrar archivos. Python ofrece el método `open()` para trabajar con archivos, y soporta diversos modos de acceso según las necesidades (lectura, escritura, etc.).

### Operaciones básicas con archivos `.txt` en Python

1. **Abrir un archivo**: `open()` es la función utilizada para abrir archivos en diferentes modos.
   ```python
   archivo = open('archivo.txt', 'r')  # Abre el archivo en modo lectura
   ```
   Donde `'r'` indica que el archivo se abrirá en modo **lectura**.

2. **Leer el contenido de un archivo**: Existen varios métodos para leer archivos:
   - **Leer todo el archivo**: `read()`
   - **Leer una línea a la vez**: `readline()`
   - **Leer todas las líneas y almacenarlas en una lista**: `readlines()`

   Ejemplo de cómo leer un archivo:
   ```python
   with open('archivo.txt', 'r') as archivo:
       contenido = archivo.read()  # Lee todo el contenido
       print(contenido)
   ```
   Al usar `with`, el archivo se cierra automáticamente después de finalizar las operaciones.

3. **Escribir en un archivo**: Para escribir en un archivo, se usa el modo `'w'` (escritura), `'a'` (agregar al final) o `'x'` (crear un archivo nuevo).
   - **Sobrescribir** (modo `'w'`):
     ```python
     with open('archivo.txt', 'w') as archivo:
         archivo.write("Este es un nuevo contenido.\n")
     ```
   - **Agregar al final** (modo `'a'`):
     ```python
     with open('archivo.txt', 'a') as archivo:
         archivo.write("Este contenido se agrega al final.\n")
     ```

4. **Leer y escribir un archivo**: El modo `'r+'` permite leer y escribir en el mismo archivo.
   ```python
   with open('archivo.txt', 'r+') as archivo:
       contenido = archivo.read()
       archivo.write("Texto adicional.")
   ```

### Modos de apertura de archivos
- `'r'`: **Lectura**. Da error si el archivo no existe.
- `'w'`: **Escritura**. Sobrescribe el archivo si ya existe o lo crea si no.
- `'a'`: **Agregar**. Añade al final del archivo si existe o lo crea si no.
- `'x'`: **Crear**. Da error si el archivo ya existe.
- `'b'`: **Modo binario** (se combina con los otros modos, por ejemplo, `'rb'` para leer un archivo binario).

### Ejemplo completo

```python
# Escritura de un archivo
with open('archivo.txt', 'w') as archivo:
    archivo.write("Primera línea del archivo.\n")
    archivo.write("Segunda línea del archivo.\n")

# Lectura de un archivo
with open('archivo.txt', 'r') as archivo:
    for linea in archivo:
        print(linea.strip())

# Agregar más contenido al archivo
with open('archivo.txt', 'a') as archivo:
    archivo.write("Tercera línea del archivo.\n")
```

### Gestión de Excepciones al manejar archivos
Es buena práctica manejar posibles errores al trabajar con archivos, como archivos inexistentes o problemas de permisos.
```python
try:
    with open('archivo_inexistente.txt', 'r') as archivo:
        contenido = archivo.read()
except FileNotFoundError:
    print("El archivo no existe.")
except IOError:
    print("Error al leer o escribir en el archivo.")
```

### Cerrar archivos
Si no usas `with`, debes cerrar el archivo manualmente para liberar recursos:
```python
archivo = open('archivo.txt', 'r')
contenido = archivo.read()
archivo.close()  # Cierra el archivo
```

### Resumen:
- `open()` es la función principal para abrir archivos.
- Los archivos pueden leerse con `read()`, `readline()`, o `readlines()`.
- Se puede escribir o agregar al archivo con `write()`.
- Utiliza `with` para abrir archivos, ya que gestiona automáticamente el cierre del archivo.

**Lecturas recomendadas**

[7. Input and Output — Python 3.12.5 documentation](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files "7. Input and Output — Python 3.12.5 documentation")

## Manejo de Archivos CSV

### ¿Cuándo Usar CSV vs JSON?

En el mundo del manejo de datos, elegir el formato adecuado para almacenar y transferir información es crucial. Dos de los formatos más comunes son CSV (Comma-Separated Values) y JSON (JavaScript Object Notation). Cada uno tiene sus propias ventajas y desventajas, y es importante entender cuándo es más apropiado usar uno sobre el otro.

Estas son algunas consideraciones clave para decidir entre estos formatos, basadas en la estructura de los datos y los requisitos de la aplicación.

### Usamos CSV cuando:

- Los datos son tabulares y se ajustan bien a un formato de filas y columnas.
- Se requiere compatibilidad con hojas de cálculo y herramientas de análisis de datos que admiten CSV.
- La estructura de los datos es simple y plana.

### Usa JSON cuando:

- Necesitas representar datos jerárquicos o anidados.
- Los datos se transmitirán entre un cliente y un servidor en una aplicación web.
- Deseas almacenar configuraciones o datos de aplicación en un formato fácil de leer y modificar.

CSV es ideal para datos tabulares simples que necesitan ser fácilmente importados a hojas de cálculo o herramientas de análisis, mientras que JSON es más adecuado para datos complejos y jerárquicos, especialmente en aplicaciones web donde la legibilidad y la capacidad de anidar estructuras son importantes.

Al comprender las fortalezas de cada formato, puedes tomar decisiones informadas que optimicen el manejo y la transferencia de datos en tus proyectos.

El manejo de archivos CSV en Python se facilita mediante el módulo integrado `csv`, que proporciona herramientas para leer, escribir y procesar archivos CSV de manera sencilla.

### Operaciones básicas con archivos CSV en Python

#### 1. **Leer archivos CSV**
Puedes leer archivos CSV utilizando `csv.reader`, que devuelve cada fila del archivo como una lista de elementos.

**Ejemplo de lectura de un archivo CSV:**
```python
import csv

# Abrir y leer un archivo CSV
with open('archivo.csv', mode='r') as archivo:
    lector_csv = csv.reader(archivo)
    for fila in lector_csv:
        print(fila)
```

Cada fila del archivo se imprime como una lista, donde cada elemento corresponde a una celda de la fila CSV.

#### 2. **Leer archivos CSV con encabezados**
Si el archivo CSV tiene una fila de encabezados, puedes usar `csv.DictReader`, que devuelve cada fila como un diccionario, donde las claves son los nombres de las columnas.

**Ejemplo de lectura con encabezados:**
```python
import csv

# Leer un CSV con encabezado
with open('archivo_con_encabezado.csv', mode='r') as archivo:
    lector_csv = csv.DictReader(archivo)
    for fila in lector_csv:
        print(fila)  # Cada fila es un diccionario
```
En este caso, los nombres de las columnas del CSV son las claves de los diccionarios.

#### 3. **Escribir en archivos CSV**
Para escribir en un archivo CSV, utilizamos `csv.writer`, que permite agregar filas como listas.

**Ejemplo de escritura en un archivo CSV:**
```python
import csv

# Escribir datos en un archivo CSV
with open('archivo.csv', mode='w', newline='') as archivo:
    escritor_csv = csv.writer(archivo)
    escritor_csv.writerow(['Nombre', 'Edad', 'Ciudad'])  # Escribir encabezado
    escritor_csv.writerow(['Juan', 28, 'Madrid'])
    escritor_csv.writerow(['Ana', 22, 'Barcelona'])
```

El parámetro `newline=''` se usa para evitar líneas en blanco entre cada fila al escribir en el archivo.

#### 4. **Escribir en archivos CSV con encabezados**
Si deseas escribir datos a un archivo CSV con encabezados, puedes usar `csv.DictWriter`. Debes especificar los nombres de las columnas como una lista y pasar cada fila como un diccionario.

**Ejemplo de escritura con encabezados:**
```python
import csv

# Escribir un CSV con encabezado
with open('archivo_con_encabezado.csv', mode='w', newline='') as archivo:
    campos = ['Nombre', 'Edad', 'Ciudad']
    escritor_csv = csv.DictWriter(archivo, fieldnames=campos)
    
    escritor_csv.writeheader()  # Escribir encabezado
    escritor_csv.writerow({'Nombre': 'Juan', 'Edad': 28, 'Ciudad': 'Madrid'})
    escritor_csv.writerow({'Nombre': 'Ana', 'Edad': 22, 'Ciudad': 'Barcelona'})
```

#### 5. **Modificar archivos CSV**
Para modificar un archivo CSV, puedes leer su contenido, almacenarlo en una lista, realizar los cambios necesarios y luego escribir el contenido modificado en el archivo nuevamente.

**Ejemplo de modificación de un archivo CSV:**
```python
import csv

# Leer y modificar un archivo CSV
with open('archivo.csv', mode='r') as archivo:
    lector_csv = csv.reader(archivo)
    filas = list(lector_csv)

# Modificar una fila
filas[1][1] = '30'  # Cambiar la edad de Juan a 30

# Escribir los cambios de nuevo en el archivo
with open('archivo.csv', mode='w', newline='') as archivo:
    escritor_csv = csv.writer(archivo)
    escritor_csv.writerows(filas)
```

#### 6. **Manejo de Excepciones**
Es recomendable manejar posibles excepciones al trabajar con archivos, especialmente si el archivo CSV no existe o si hay problemas de permisos.

**Ejemplo de manejo de excepciones:**
```python
import csv

try:
    with open('archivo_inexistente.csv', mode='r') as archivo:
        lector_csv = csv.reader(archivo)
        for fila in lector_csv:
            print(fila)
except FileNotFoundError:
    print("El archivo no existe.")
except IOError:
    print("Error al leer el archivo.")
```

### Modos comunes de apertura de archivos CSV
- `'r'`: **Lectura**. Usado para leer el archivo.
- `'w'`: **Escritura**. Sobrescribe el archivo o lo crea si no existe.
- `'a'`: **Agregar**. Añade al final del archivo si existe.
- `'r+'`: **Lectura y escritura**. Lee y escribe en el archivo.

### Resumen:
- **Lectura**: Usa `csv.reader` para leer filas como listas y `csv.DictReader` para leer filas como diccionarios.
- **Escritura**: Usa `csv.writer` para escribir filas como listas y `csv.DictWriter` para escribir filas como diccionarios.
- **Excepciones**: Maneja posibles errores como archivos inexistentes o problemas de permisos.

Esta estructura es muy útil para manejar datos tabulares de forma eficiente en Python.

**Lecturas recomendadas**

[csv — CSV File Reading and Writing — documentación de Python - 3.12.5](https://docs.python.org/es/3/library/csv.html "csv — CSV File Reading and Writing — documentación de Python - 3.12.5")

## Manejo de Archivos JSON

En Python, el manejo de archivos JSON se realiza utilizando el módulo `json`, que proporciona funciones para leer y escribir archivos en formato JSON, una estructura ampliamente utilizada para intercambiar datos en aplicaciones web y APIs.

### 1. **Leer archivos JSON**
Para leer un archivo JSON en Python, se utiliza la función `json.load()`, que convierte el contenido JSON en una estructura de datos de Python, como diccionarios o listas.

**Ejemplo de lectura de un archivo JSON:**
```python
import json

# Leer un archivo JSON
with open('archivo.json', 'r') as archivo:
    datos = json.load(archivo)  # Convierte JSON en un diccionario o lista
    print(datos)
```

### 2. **Escribir en archivos JSON**
Para escribir datos en formato JSON, puedes usar `json.dump()`, que convierte los objetos de Python (diccionarios, listas, etc.) en una cadena JSON y los escribe en un archivo.

**Ejemplo de escritura en un archivo JSON:**
```python
import json

# Datos a escribir en formato JSON
datos = {
    'nombre': 'Juan',
    'edad': 30,
    'ciudad': 'Madrid'
}

# Escribir datos en un archivo JSON
with open('archivo.json', 'w') as archivo:
    json.dump(datos, archivo, indent=4)  # `indent=4` añade formato legible
```

### 3. **Leer y escribir cadenas JSON**
El módulo `json` también permite trabajar directamente con cadenas JSON, sin necesidad de archivos, utilizando las funciones `json.loads()` para leer y `json.dumps()` para escribir.

**Ejemplo de conversión de cadena JSON a Python:**
```python
import json

# Cadena en formato JSON
cadena_json = '{"nombre": "Ana", "edad": 25, "ciudad": "Barcelona"}'

# Convertir cadena JSON a un diccionario
datos = json.loads(cadena_json)
print(datos)
```

**Ejemplo de conversión de Python a cadena JSON:**
```python
import json

# Diccionario en Python
datos = {
    'nombre': 'Ana',
    'edad': 25,
    'ciudad': 'Barcelona'
}

# Convertir diccionario a cadena JSON
cadena_json = json.dumps(datos, indent=4)
print(cadena_json)
```

### 4. **Modificar archivos JSON**
Puedes modificar un archivo JSON leyendo su contenido, realizando cambios en los datos y luego escribiendo nuevamente el archivo.

**Ejemplo de modificación de un archivo JSON:**
```python
import json

# Leer archivo JSON
with open('archivo.json', 'r') as archivo:
    datos = json.load(archivo)

# Modificar los datos
datos['edad'] = 31

# Escribir los cambios de vuelta en el archivo
with open('archivo.json', 'w') as archivo:
    json.dump(datos, archivo, indent=4)
```

### 5. **Manejo de Excepciones**
Es importante manejar posibles excepciones, como errores de formato en el archivo JSON o la falta de permisos de lectura/escritura.

**Ejemplo de manejo de excepciones:**
```python
import json

try:
    with open('archivo.json', 'r') as archivo:
        datos = json.load(archivo)
except json.JSONDecodeError:
    print("Error: El archivo no tiene un formato JSON válido.")
except FileNotFoundError:
    print("Error: El archivo no fue encontrado.")
except IOError:
    print("Error: No se pudo acceder al archivo.")
```

### 6. **Formateo de JSON**
Puedes personalizar la forma en que se escribe el JSON en el archivo utilizando el parámetro `indent`, que agrega sangrías y hace el archivo más legible.

**Ejemplo con formato JSON legible:**
```python
import json

# Diccionario de Python
datos = {'nombre': 'Carlos', 'edad': 27, 'ciudad': 'Lima'}

# Convertir a cadena JSON con formato
json_formateado = json.dumps(datos, indent=4)
print(json_formateado)
```

### Resumen de funciones principales:
- **Lectura de archivos JSON**: `json.load()`
- **Escritura en archivos JSON**: `json.dump()`
- **Convertir cadena JSON a Python**: `json.loads()`
- **Convertir objeto de Python a cadena JSON**: `json.dumps()`

Estas herramientas son útiles cuando trabajas con datos en formato JSON, que es común en aplicaciones web, APIs y sistemas que requieren intercambio de información estructurada.

## Librería Statistics y Análisis Estadístico

En el análisis de datos, es fundamental comprender y utilizar diversas medidas estadísticas para interpretar correctamente la información. Estas medidas nos permiten resumir y describir las características principales de un conjunto de datos, facilitando la toma de decisiones informadas. Algunas de las medidas estadísticas más comunes y sus fórmulas asociadas son las siguientes:

### 1. Media (Promedio)

La **media aritmética** se calcula sumando todos los valores de un conjunto de datos y dividiendo entre la cantidad de valores.

![media aritmetica](images/media_aritmetica.png "media aritmetica")

Donde:

- nnn es el número total de valores.
- xix_ixi representa cada valor individual en el conjunto de datos.

### 2. Mediana

La **mediana** es el valor que separa la mitad superior de la mitad inferior de un conjunto de datos ordenado.

- Si n (el número de observaciones) es impar, la mediana es el valor en la posición 2n+1 después de ordenar los datos.

![Mediana](images/Mediana.png "Mediana")

Si n es par, la mediana es el promedio de los dos valores centrales, es decir:

![Mediana ecuación dos](images/images/Mediana2.png "Mediana ecuación dos")

### 3. Moda

La **moda** es el valor que aparece con mayor frecuencia en un conjunto de datos. No hay una fórmula específica para la moda; se determina contando la frecuencia de cada valor y eligiendo el que tiene la frecuencia más alta.

![Moda](images/Moda.png "Moda")

### 4. Desviación Estándar

La **desviación estándar** mide la dispersión de los datos en relación con la media. Se calcula como la raíz cuadrada de la varianza.

![Desviación Estándar](images/DesviacionEstandar.png "Desviación Estándar")

Donde:

- μ\muμ es la media de los datos.
- xix_ixi representa cada valor individual.
- nnn es el número total de valores.

5. Varianza

La **varianza** es el promedio de los cuadrados de las diferencias entre cada valor y la media del conjunto de datos.

![Varianza](images/Varianza.png "Varianza")

### 6. Máximo y Mínimo

- **Máximo** (max(x)): Es el valor más alto en un conjunto de datos.

max⁡(x)\max(x)

- **Mínimo **(min(x)): Es el valor más bajo en un conjunto de datos.

min⁡(x)\min(x)

![Máximo y mínimo](images/maxymin.png "Máximo y mínimo")

7. Rango

El **rango** es la diferencia entre el valor máximo y el valor mínimo en un conjunto de datos.

![Rango](images/rango.png "Rango")

Estas fórmulas matemáticas te permiten realizar un análisis estadístico básico de cualquier conjunto de datos, como las ventas mensuales en un negocio, y son fundamentales para extraer conclusiones y tomar decisiones basadas en datos.

Al aplicar estas medidas, podrás extraer conclusiones valiosas y tomar decisiones basadas en datos, lo que es crucial para el éxito en diversos campos, desde la investigación científica hasta la gestión empresarial.

**Lecturas recomendadas**

[statistics — Mathematical statistics functions — documentación de Python - 3.12.5](https://docs.python.org/es/3/library/statistics.html "statistics — Mathematical statistics functions — documentación de Python - 3.12.5")

## Biblioteca estándar en Python

Como programador, una de las mayores satisfacciones es encontrar la solución perfecta al problema que estás tratando de resolver, y hacerlo de manera eficiente.
¿Te imaginas tener a tu disposición un conjunto de herramientas que te permita escribir menos código pero lograr más? Eso es lo que ofrece la Biblioteca Estándar de Python.

Imagina poder saltar directamente a construir la lógica de tu aplicación sin preocuparte por las tareas rutinarias, porque ya tienes los módulos que necesitas listos para usar.
Aquí vamos a explorar cómo la Biblioteca Estándar de Python puede transformar tu manera de programar, dándote acceso inmediato a una vasta colección de herramientas que te permiten concentrarte en lo que realmente importa: resolver problemas de manera elegante y eficaz.

### ¿Qué es la Biblioteca Estándar de Python?

La Biblioteca Estándar de Python es como tener un conjunto de herramientas integradas directamente en el lenguaje que te ayudan a realizar una variedad de tareas sin tener que reinventar la rueda. Desde la manipulación de archivos, pasando por cálculos matemáticos complejos, hasta la creación de servidores web, la Biblioteca Estándar tiene módulos que simplifican casi cualquier tarea que te propongas.

### ¿Qué es una Librería y qué es un Módulo?

Antes de sumergirnos en cómo puedes aprovechar la Biblioteca Estándar, aclaremos dos conceptos clave:

- **Librería**: En Python, una librería es un conjunto organizado de módulos que puedes usar para añadir funcionalidades a tu código sin tener que escribirlas tú mismo. Piensa en ello como una colección de herramientas especializadas listas para usar.
- **Módulo**: Un módulo es un archivo de Python que contiene código que puedes reutilizar en tus proyectos. Un módulo puede incluir funciones, clases, y variables que te ayudan a resolver problemas específicos de manera eficiente.

Estos conceptos son fundamentales porque la Biblioteca Estándar está compuesta por una amplia variedad de módulos, cada uno diseñado para hacer tu vida como programador más fácil.

### La Conexión de la Biblioteca Estándar con tus Proyectos

La belleza de la Biblioteca Estándar radica en cómo cada módulo está diseñado para interactuar con otros, permitiéndote construir aplicaciones completas sin tener que buscar soluciones externas. Al trabajar en un proyecto, puedes confiar en que la Biblioteca Estándar tiene las herramientas necesarias para cubrir la mayoría de tus necesidades.

Por ejemplo, si estás trabajando en una aplicación que necesita interactuar con el sistema de archivos, el módulo os te permite manipular directorios y archivos de manera eficiente. Si tu aplicación necesita realizar operaciones matemáticas complejas, el módulo math ofrece un amplio rango de funciones listas para usar. Cada módulo tiene su propósito, pero todos están diseñados para trabajar juntos y hacer tu código más limpio y eficiente.

### Explorando Áreas Clave de la Biblioteca Estándar

Ahora, veamos algunas de las áreas más importantes que cubre la Biblioteca Estándar:

- **Manejo de Archivos y Sistema**: Módulos como os, shutil, y pathlib te permiten interactuar con el sistema de archivos, lo cual es esencial para casi cualquier proyecto.
- **Operaciones Matemáticas**: Módulos como math y random te proporcionan funciones matemáticas avanzadas y generación de números aleatorios.
- **Manejo de Fechas y Tiempos**: datetime y time te permiten trabajar con fechas y horas, lo cual es crucial para la programación de eventos o el registro de actividades.
- **Manipulación de Datos**: Módulos como json y csv son ideales para leer y escribir datos estructurados, algo común en el manejo de APIs y almacenamiento de información.
- **Redes y Comunicaciones**: Si estás construyendo aplicaciones que necesitan comunicarse a través de una red, socket y http.server te proporcionan las herramientas necesarias para gestionar conexiones y servidores web.

Estos módulos no solo te ahorran tiempo, sino que también te ayudan a escribir código más limpio y mantenible.

### ¿Qué es pip y Cuándo Deberíamos Considerar Instalar una Librería?

La Biblioteca Estándar es extremadamente poderosa, pero a veces necesitarás algo más específico o avanzado. Aquí es donde entra **pip**, una herramienta que te permite instalar librerías adicionales que no vienen incluidas en Python por defecto.

**¿Cuándo deberías considerar instalar una librería?**

- Cuando necesitas funcionalidades que no están cubiertas por la Biblioteca Estándar.
- Cuando quieres utilizar herramientas más especializadas para resolver problemas complejos.
- Cuando necesitas una versión más reciente o específica de un módulo.

Por ejemplo, si estás trabajando en análisis de datos, podrías necesitar `pandas`, una librería poderosa para la manipulación y análisis de datos que no está en la Biblioteca Estándar.

### ¿Cómo Instalar una Librería con pip?

Instalar una librería con `pip` es directo y simple. Abre tu terminal y ejecuta:

`pip install nombre-de-la-libreria`

Por ejemplo, para instalar pandas, simplemente escribirías:

`pip install pandas`

Esto descargará e instalará la librería desde [PyPI](https://pypi.org/ "PyPI"), un repositorio en línea donde se alojan miles de librerías para Python, y estará lista para ser utilizada en tu proyecto.

La Biblioteca Estándar de Python te ofrece un vasto conjunto de herramientas que puedes utilizar inmediatamente, permitiéndote escribir código eficiente y de alta calidad. Sin embargo, el mundo de Python no termina ahí. Te invito a explorar la [documentación oficial de la Biblioteca Estándar](https://docs.python.org/3/library/ "documentación oficial de la Biblioteca Estándar") para profundizar en los módulos disponibles, y no dudes en visitar [PyPI](https://pypi.org/ "PyPI") para descubrir librerías adicionales que pueden potenciar aún más tus proyectos. ¡El poder de Python está a tu disposición, y es hora de que lo aproveches al máximo!

La **biblioteca estándar de Python** es una colección de módulos y paquetes que están incluidos con la instalación de Python, lo que significa que no necesitas instalarlos por separado. Estos módulos proporcionan una amplia gama de funcionalidades que te permiten realizar diversas tareas comunes sin necesidad de depender de bibliotecas externas.

A continuación, te muestro algunas de las categorías más importantes de la biblioteca estándar de Python junto con ejemplos de uso:

### 1. **Manejo de Archivos y Directorios**
   - **`os`**: Proporciona funciones para interactuar con el sistema operativo, como la manipulación de archivos y directorios.
   - **`shutil`**: Ofrece operaciones de alto nivel en archivos y colecciones de archivos.
   - **`pathlib`**: Una forma moderna de manejar rutas de archivos y directorios.

   **Ejemplo:**
   ```python
   import os
   print(os.getcwd())  # Obtener el directorio de trabajo actual

   from pathlib import Path
   path = Path('mi_archivo.txt')
   print(path.exists())  # Verificar si el archivo existe
   ```

### 2. **Manejo de Fechas y Tiempos**
   - **`datetime`**: Ofrece funciones para manejar fechas, horas y períodos de tiempo.
   - **`time`**: Proporciona funciones relacionadas con el tiempo del sistema.

   **Ejemplo:**
   ```python
   from datetime import datetime
   ahora = datetime.now()
   print(ahora.strftime("%Y-%m-%d %H:%M:%S"))  # Formatear la fecha actual
   ```

### 3. **Manejo de Archivos JSON y CSV**
   - **`json`**: Manipula archivos JSON.
   - **`csv`**: Proporciona soporte para archivos CSV.

   **Ejemplo con JSON:**
   ```python
   import json
   datos = {"nombre": "Juan", "edad": 30}
   with open('datos.json', 'w') as archivo:
       json.dump(datos, archivo)
   ```

### 4. **Expresiones Regulares**
   - **`re`**: Módulo para trabajar con expresiones regulares.

   **Ejemplo:**
   ```python
   import re
   patron = r'\d+'
   texto = "El número es 12345"
   coincidencia = re.search(patron, texto)
   print(coincidencia.group())  # Devuelve "12345"
   ```

### 5. **Manejo de Excepciones y Errores**
   - **`warnings`**: Emite advertencias al usuario sin interrumpir el programa.
   - **`traceback`**: Proporciona una representación del seguimiento de excepciones.

   **Ejemplo:**
   ```python
   import warnings
   warnings.warn("Esto es solo una advertencia.")
   ```

### 6. **Módulos Matemáticos**
   - **`math`**: Proporciona funciones matemáticas básicas.
   - **`random`**: Permite la generación de números aleatorios.
   - **`statistics`**: Funciones estadísticas como la media, mediana y desviación estándar.

   **Ejemplo con `math`:**
   ```python
   import math
   print(math.sqrt(16))  # Devuelve 4.0
   ```

### 7. **Manejo de Compresión de Archivos**
   - **`zipfile`**: Manejo de archivos ZIP.
   - **`tarfile`**: Manejo de archivos TAR.

   **Ejemplo:**
   ```python
   import zipfile
   with zipfile.ZipFile('archivo.zip', 'r') as zip_ref:
       zip_ref.extractall('directorio_destino')
   ```

### 8. **Concurrencia y Paralelismo**
   - **`threading`**: Soporte para la creación de hilos.
   - **`multiprocessing`**: Permite la ejecución de múltiples procesos.

   **Ejemplo con `threading`:**
   ```python
   import threading
   def contar():
       for i in range(5):
           print(i)
   hilo = threading.Thread(target=contar)
   hilo.start()
   ```

### 9. **Redes y Protocolos de Internet**
   - **`socket`**: Soporte para la creación de aplicaciones de red.
   - **`http.server`**: Un servidor HTTP simple.

   **Ejemplo de servidor HTTP básico:**
   ```python
   from http.server import SimpleHTTPRequestHandler, HTTPServer

   puerto = 8080
   servidor = HTTPServer(('localhost', puerto), SimpleHTTPRequestHandler)
   print(f"Servidor HTTP corriendo en el puerto {puerto}")
   servidor.serve_forever()
   ```

### 10. **Manejo de Argumentos en la Línea de Comandos**
   - **`argparse`**: Módulo para analizar argumentos pasados por la línea de comandos.

   **Ejemplo:**
   ```python
   import argparse
   parser = argparse.ArgumentParser(description='Ejemplo de manejo de argumentos')
   parser.add_argument('nombre', help='El nombre del usuario')
   args = parser.parse_args()
   print(f"Hola, {args.nombre}")
   ```

### 11. **Serialización y Deserialización de Objetos**
   - **`pickle`**: Permite serializar objetos Python para almacenarlos y deserializarlos más tarde.

   **Ejemplo:**
   ```python
   import pickle
   datos = {'nombre': 'Ana', 'edad': 25}
   with open('datos.pickle', 'wb') as archivo:
       pickle.dump(datos, archivo)
   ```

### 12. **Soporte para Protocolos de Internet**
   - **`urllib`**: Para trabajar con URLs y realizar peticiones HTTP.
   - **`http.client`**: Módulo para gestionar conexiones HTTP.

   **Ejemplo:**
   ```python
   import urllib.request
   respuesta = urllib.request.urlopen('http://www.example.com')
   print(respuesta.read().decode('utf-8'))
   ```

### Resumen de la Biblioteca Estándar

La biblioteca estándar de Python es increíblemente versátil y abarca muchas áreas, desde la manipulación de archivos, operaciones matemáticas, hasta la gestión de redes y el uso de concurrencia. Al estar incluida con Python, es recomendable familiarizarse con los módulos más relevantes para tus proyectos.

## Librería Os, Math y Random

Las librerías **`os`**, **`math`** y **`random`** forman parte de la biblioteca estándar de Python y proporcionan funcionalidades esenciales para la interacción con el sistema operativo, operaciones matemáticas avanzadas y la generación de números aleatorios, respectivamente. A continuación te explico cada una con ejemplos.

### 1. **Librería `os`**
El módulo **`os`** te permite interactuar con el sistema operativo. Proporciona funciones para manejar archivos y directorios, manipular rutas, ejecutar comandos del sistema, entre otras.

#### Funciones comunes de `os`:

- **Obtener el directorio actual de trabajo**:
  ```python
  import os
  print(os.getcwd())  # Devuelve el directorio actual
  ```

- **Cambiar el directorio de trabajo**:
  ```python
  os.chdir('/ruta/nueva')
  print(os.getcwd())  # Comprueba el nuevo directorio
  ```

- **Listar archivos en un directorio**:
  ```python
  archivos = os.listdir('.')
  print(archivos)  # Lista todos los archivos en el directorio actual
  ```

- **Crear un nuevo directorio**:
  ```python
  os.mkdir('nuevo_directorio')  # Crea un directorio llamado 'nuevo_directorio'
  ```

- **Eliminar un archivo o directorio**:
  ```python
  os.remove('archivo.txt')  # Elimina un archivo
  os.rmdir('directorio_vacio')  # Elimina un directorio vacío
  ```

- **Obtener información del sistema**:
  ```python
  print(os.name)  # 'posix', 'nt', etc., dependiendo del sistema operativo
  ```

- **Ejecutar un comando del sistema**:
  ```python
  os.system('ls')  # Ejecuta el comando 'ls' en sistemas tipo UNIX
  ```

### 2. **Librería `math`**
El módulo **`math`** proporciona funciones matemáticas básicas y avanzadas que no están disponibles por defecto en Python, como funciones trigonométricas, logaritmos y constantes matemáticas.

#### Funciones comunes de `math`:

- **Raíz cuadrada**:
  ```python
  import math
  print(math.sqrt(16))  # Devuelve 4.0
  ```

- **Logaritmo natural y logaritmo base 10**:
  ```python
  print(math.log(10))       # Logaritmo natural (base e)
  print(math.log10(1000))   # Logaritmo base 10
  ```

- **Potencias**:
  ```python
  print(math.pow(2, 3))  # 2 elevado a la 3, devuelve 8.0
  ```

- **Constantes matemáticas**:
  ```python
  print(math.pi)    # Valor de pi (3.14159...)
  print(math.e)     # Valor de e (2.71828...)
  ```

- **Funciones trigonométricas**:
  ```python
  print(math.sin(math.pi / 2))  # Seno de 90 grados, devuelve 1.0
  print(math.cos(0))            # Coseno de 0, devuelve 1.0
  ```

- **Valor absoluto**:
  ```python
  print(math.fabs(-10))  # Devuelve 10.0
  ```

- **Redondeo hacia abajo (piso) y hacia arriba (techo)**:
  ```python
  print(math.floor(3.7))  # Devuelve 3
  print(math.ceil(3.3))   # Devuelve 4
  ```

### 3. **Librería `random`**
El módulo **`random`** proporciona herramientas para generar números aleatorios y realizar selecciones aleatorias, que son útiles en simulaciones, pruebas, y muchas otras aplicaciones.

#### Funciones comunes de `random`:

- **Generar un número aleatorio entre 0 y 1**:
  ```python
  import random
  print(random.random())  # Devuelve un número decimal aleatorio entre 0 y 1
  ```

- **Generar un número aleatorio entero dentro de un rango**:
  ```python
  print(random.randint(1, 10))  # Devuelve un entero entre 1 y 10 (incluidos)
  ```

- **Elegir un elemento aleatorio de una lista**:
  ```python
  opciones = ['rojo', 'verde', 'azul']
  print(random.choice(opciones))  # Devuelve un elemento aleatorio de la lista
  ```

- **Mezclar aleatoriamente una lista**:
  ```python
  lista = [1, 2, 3, 4, 5]
  random.shuffle(lista)  # Mezcla los elementos de la lista
  print(lista)
  ```

- **Generar un número flotante aleatorio en un rango específico**:
  ```python
  print(random.uniform(1.5, 10.5))  # Devuelve un número decimal aleatorio entre 1.5 y 10.5
  ```

- **Elegir varios elementos aleatorios de una lista (sin repetición)**:
  ```python
  lista = [1, 2, 3, 4, 5]
  print(random.sample(lista, 2))  # Devuelve 2 elementos aleatorios de la lista
  ```

### Ejemplos Combinando `os`, `math`, y `random`

Aquí te muestro un ejemplo que utiliza las tres bibliotecas juntas:

```python
import os
import math
import random

# Cambiar el directorio de trabajo
os.chdir('/tmp')

# Crear un archivo con números aleatorios y escribirlos en el archivo
with open('numeros_aleatorios.txt', 'w') as archivo:
    for _ in range(5):
        numero = random.uniform(0, 100)
        archivo.write(f"{numero:.2f}\n")  # Escribir un número aleatorio con 2 decimales

# Leer el archivo y calcular la raíz cuadrada de cada número
with open('numeros_aleatorios.txt', 'r') as archivo:
    for linea in archivo:
        numero = float(linea.strip())
        print(f"Raíz cuadrada de {numero}: {math.sqrt(numero):.2f}")
```

### Resumen:
- **`os`**: Proporciona herramientas para interactuar con el sistema operativo.
- **`math`**: Ofrece funciones matemáticas avanzadas como logaritmos, trigonometría y potencias.
- **`random`**: Genera números aleatorios y realiza operaciones aleatorias como elegir elementos de una lista.

Estas tres bibliotecas son esenciales para realizar tareas comunes como la manipulación de archivos, operaciones matemáticas y la generación de números aleatorios en Python.

**1. OS (Sistema Operativo):**

- `os.getcwd()` Retorna el directorio de trabajo actual.
- `os.chdir(path)`: Cambia el directorio de trabajo actual al especificado.
- `os.listdir(path)`: Lista los archivos y carpetas en el directorio especificado.
- `os.makedirs(path)`: Crea directorios de manera recursiva.
- `os.remove(path)`: Elimina el archivo especificado.
- `os.path.join(*paths)`: Une componentes de una ruta de manera segura según el sistema operativo.
- `os.path.exists(path)`: Verifica si una ruta existe.
- `os.rename(src, dst)`: Renombra un archivo o directorio.
- `os.environ`: Proporciona acceso a las variables de entorno del sistema.

**2. Módulo (Operaciones Matemáticas):**

- `math.sqrt(x)`: Retorna la raíz cuadrada de x.
- `math.pow(x, y)`: Eleva x a la potencia y (equivalente a `x ** y`).
- `math.ceil(x)`: Redondea un número hacia arriba (al entero más cercano).
- `math.floor(x)`: Redondea un número hacia abajo (al entero más cercano).
- `math.factorial(x)`: Retorna el factorial de x.
- `math.fabs(x)`: Retorna el valor absoluto de x (como número flotante).
- `math.log(x[, base])`: Retorna el logaritmo de x con base base (por defecto, base e).
- `math.sin(x), math.cos(x), math.tan(x):` Retorna el seno, coseno y tangente de x (en radianes).
- `math.pi`: Retorna el valor de π (pi).

**3. Módulo (Generación Aleatoria):**

- `random.random()`: Retorna un número flotante aleatorio entre 0.0 y 1.0.
- `random.randint(a, b)`: Retorna un entero aleatorio entre a y b (ambos inclusive).
- `random.choice(seq)`: Retorna un elemento aleatorio de una secuencia (como una lista).
- `random.shuffle(seq)`: Baraja una secuencia (lista) en su lugar.
- `random.sample(population, k)`: Retorna una lista de tamaño k con elementos aleatorios sin repetición de la population.
- `random.uniform(a, b)`: Retorna un número flotante aleatorio entre a y b.
- `random.gauss(mu, sigma)`: Retorna un número siguiendo una distribución normal (gaussiana) con media mu y desviación estándar `sigma`.

**Lecturas recomendadas**

[math — Funciones matemáticas — documentación de Python - 3.10.13](https://docs.python.org/es/3.10/library/math.html "math — Funciones matemáticas — documentación de Python - 3.10.13")

[os — Interfaces misceláneas del sistema operativo — documentación de Python - 3.10.13](https://docs.python.org/es/3.10/library/os.html "os — Interfaces misceláneas del sistema operativo — documentación de Python - 3.10.13")

[random — Generate pseudo-random numbers — documentación de Python - 3.12.5](https://docs.python.org/es/3/library/random.html "random — Generate pseudo-random numbers — documentación de Python - 3.12.5")

## Recapitulación de lo aprendido hasta ahora

### Recapitulación de lo aprendido hasta ahora

![Python Advanced Concepts Illustration](images/Python_Advanced_Concepts_Illustration.png)

Hasta este punto, has recorrido los fundamentos de Python: su sintaxis, las estructuras de control, las funciones y el manejo básico de la biblioteca estándar. Estos conceptos son esenciales para cualquier programador y te han preparado para el siguiente paso en tu camino.

### Motivación: La importancia de avanzar

Ahora, es momento de dar un paso más y enfrentar desafíos más complejos. Aprender a escribir código Pythonico y profesional no solo te permitirá resolver problemas de una manera más eficiente y limpia, sino que te abrirá la puerta a proyectos más ambiciosos y demandantes.

En esta nueva etapa, es crucial contar con conocimientos sólidos sobre:

- **Programación orientada a objetos (POO)**: cómo estructurar mejor tu código, creando clases y objetos que sean reutilizables y fáciles de entender.
- **Manejo avanzado de excepciones**: cómo gestionar errores de forma elegante y robusta.
- **Decoradores y generadores**: formas avanzadas de controlar el flujo de tu código y maximizar la eficiencia.
- **Módulos y paquetes**: cómo organizar y estructurar grandes proyectos de forma profesional.
- **Escritura de código eficiente y legible**: técnicas para escribir código que no solo funcione, sino que sea mantenible y optimizado.

### El Proyecto: El no negociable

La mejor manera de consolidar estos nuevos conocimientos es aplicarlos en un proyecto práctico. Esta sección final te guiará en el desarrollo de un proyecto completo, donde aplicarás lo aprendido en esta sección avanzada.

El proyecto será la herramienta clave que te permitirá interiorizar estos conceptos. A través de él, aprenderás no solo a resolver problemas complejos, sino a **pensar como un desarrollador profesional**. Recuerda: la teoría es importante, pero solo se convierte en aprendizaje real cuando la pones en práctica. Este proyecto es el “no negociable”: es lo que te hará aprender o aprender.

### ¿Qué podrás hacer con estos nuevos conocimientos?

Al completar este módulo y el proyecto, serás capaz de:

- Escribir código Python eficiente, limpio y estructurado.
- Resolver problemas complejos con un enfoque orientado a objetos.
- Utilizar módulos y paquetes para crear proyectos escalables.
- Manejar excepciones y errores de forma profesional, asegurando que tu código sea robusto.
- Trabajar en proyectos reales aplicando las mejores prácticas de la industria.

### Conclusión: El camino hacia la maestría

Este es un paso crucial en tu carrera como desarrollador. Con este nuevo nivel de comprensión, estarás preparado para abordar cualquier desafío que Python te presente y crear proyectos que impacten en el mundo real. ¡Es momento de avanzar, aprender y construir!

## Escribir código Pythonico y profesional

Escribir código Pythonico y profesional implica seguir ciertas convenciones y mejores prácticas que mejoran la legibilidad, mantenibilidad y eficiencia del código. Aquí hay algunas pautas y ejemplos:

### 1. **Nombres Significativos**
Usa nombres claros y descriptivos para variables, funciones y clases.

```python
def calculate_area(radius):
    """Calcula el área de un círculo dado su radio."""
    import math
    return math.pi * radius ** 2
```

### 2. **Funciones y Métodos Breves**
Las funciones deben realizar una única tarea y ser lo más cortas posible.

```python
def is_even(number):
    """Devuelve True si el número es par, de lo contrario False."""
    return number % 2 == 0
```

### 3. **Uso de Docstrings**
Documenta tus funciones y clases usando docstrings para describir su propósito y uso.

```python
class Circle:
    """Clase que representa un círculo."""

    def __init__(self, radius):
        """Inicializa el círculo con un radio."""
        self.radius = radius

    def area(self):
        """Calcula el área del círculo."""
        import math
        return math.pi * (self.radius ** 2)
```

### 4. **Listas por Comprensión**
Usa listas por comprensión para crear listas de manera más clara y eficiente.

```python
squares = [x**2 for x in range(10)]
```

### 5. **Manejo de Errores**
Usa excepciones para manejar errores de forma limpia.

```python
def divide(a, b):
    """Divide dos números, manejando división por cero."""
    try:
        return a / b
    except ZeroDivisionError:
        return "No se puede dividir entre cero."
```

### 6. **Uso de `with` para Recursos**
Utiliza el manejo de contexto `with` para asegurarte de que los recursos se liberan adecuadamente.

```python
with open('archivo.txt', 'r') as file:
    contenido = file.read()
```

### 7. **Organización del Código**
Organiza tu código en módulos y paquetes. Mantén una estructura lógica y usa un archivo `__init__.py` si es necesario.

### 8. **Formato de Código**
Usa herramientas como `black` o `flake8` para formatear tu código y mantener la consistencia.

### Ejemplo Completo

Aquí tienes un ejemplo completo que sigue las buenas prácticas:

```python
import math

class Circle:
    """Clase que representa un círculo."""

    def __init__(self, radius):
        """Inicializa el círculo con un radio."""
        self.radius = radius

    def area(self):
        """Calcula el área del círculo."""
        return math.pi * (self.radius ** 2)

def main():
    """Función principal del script."""
    radius = float(input("Introduce el radio del círculo: "))
    circle = Circle(radius)
    print(f"El área del círculo es: {circle.area():.2f}")

if __name__ == "__main__":
    main()
```

Este ejemplo muestra cómo estructurar el código de manera clara, usar docstrings, y manejar la entrada del usuario de forma segura y eficiente.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at pythonicCode](https://github.com/platzi/python-avanzado/tree/pythonicCode)

## Comentarios y Docstrings en Python

Los comentarios y docstrings son herramientas importantes en Python para mejorar la legibilidad y el mantenimiento del código. Aquí te explico cómo usarlos de manera efectiva.

### Comentarios

Los comentarios se utilizan para explicar el código, hacer aclaraciones sobre ciertas decisiones de diseño o proporcionar contexto sobre la lógica. Deben ser claros y concisos. En Python, los comentarios comienzan con el símbolo `#`.

#### Ejemplo de Comentarios

```python
# Este es un comentario de una sola línea

def calcular_area_circulo(radio):
    """Calcula el área de un círculo dado su radio."""
    # Usamos la constante pi de la librería math
    import math
    return math.pi * (radio ** 2)  # Devuelve el área
```

### Docstrings

Los docstrings son cadenas de documentación que se colocan al principio de funciones, métodos y clases para describir su propósito y cómo se deben usar. Se encierran entre triples comillas `"""` y pueden ocupar varias líneas. Esto permite que sean accesibles a través de la función `help()` en Python.

#### Ejemplo de Docstrings

```python
def calcular_area_circulo(radio):
    """
    Calcula el área de un círculo dado su radio.

    Parámetros:
    radio (float): El radio del círculo.

    Retorna:
    float: El área del círculo.
    """
    import math
    return math.pi * (radio ** 2)
```

### Mejores Prácticas para Comentarios y Docstrings

1. **Sé claro y conciso**: Los comentarios y docstrings deben ser fácilmente comprensibles y directos. Evita la jerga innecesaria.

2. **Usa el estilo de documentación estándar**: Para docstrings, se recomienda seguir convenciones como [PEP 257](https://www.python.org/dev/peps/pep-0257/) o [Google Style Guide](https://google.github.io/styleguide/pyguide.html#383-docstrings).

3. **Actualiza los comentarios y docstrings**: Si realizas cambios en el código, asegúrate de actualizar los comentarios y docstrings para reflejar esos cambios.

4. **Evita comentarios redundantes**: No escribas comentarios que repitan lo que el código ya dice. Por ejemplo, no es necesario comentar que una línea de código suma dos números si la línea ya es explícita.

### Ejemplo Completo con Comentarios y Docstrings

Aquí tienes un ejemplo más completo que incorpora tanto comentarios como docstrings:

```python
import math

class Circulo:
    """
    Clase que representa un círculo.

    Atributos:
    radio (float): El radio del círculo.
    """

    def __init__(self, radio):
        """
        Inicializa un círculo con el radio dado.

        Parámetros:
        radio (float): El radio del círculo.
        """
        self.radio = radio

    def area(self):
        """
        Calcula el área del círculo.

        Retorna:
        float: El área del círculo.
        """
        return math.pi * (self.radio ** 2)  # Área = π * r²

    def __str__(self):
        """Devuelve una representación en cadena del círculo."""
        return f"Círculo de radio {self.radio}"

# Ejemplo de uso
if __name__ == "__main__":
    circulo = Circulo(5)  # Crear un círculo con un radio de 5
    print(circulo)  # Imprime la representación del círculo
    print(f"Área: {circulo.area():.2f}")  # Imprime el área del círculo
```

En este ejemplo, se utilizan comentarios para aclarar partes del código y se proporcionan docstrings que describen la funcionalidad de la clase y sus métodos. Esto ayuda a otros desarrolladores (y a ti mismo en el futuro) a comprender rápidamente el propósito y el funcionamiento del código.

**Lecturas recomendadas**

[Google Colab](https://colab.research.google.com/drive/1CKFfLQPzvzFEYHS8NigzJm4vUHWOJgfu?usp=sharing)

[GitHub - platzi/python-avanzado at Docstrings](https://github.com/platzi/python-avanzado/tree/Docstrings)

## Scope y closures: variables locales y globales

En Python, **scope** (alcance) se refiere a la visibilidad y duración de una variable en diferentes partes del código, mientras que **closures** (cierres) son funciones que recuerdan el entorno en el que fueron creadas, incluso después de que el entorno haya terminado de ejecutarse. Vamos a desglosar ambos conceptos, comenzando por las variables locales y globales.

### 1. Variables Locales y Globales

#### Variables Locales
Las variables locales son aquellas definidas dentro de una función. Solo son accesibles dentro de esa función y se destruyen una vez que la función termina de ejecutarse.

```python
def funcion_local():
    variable_local = 10  # Variable local
    print("Dentro de la función:", variable_local)

funcion_local()
# print(variable_local)  # Esto generará un error porque variable_local no está definida aquí
```

#### Variables Globales
Las variables globales son aquellas definidas fuera de cualquier función. Son accesibles desde cualquier parte del código, incluyendo dentro de funciones. Sin embargo, para modificar una variable global dentro de una función, se debe usar la palabra clave `global`.

```python
variable_global = 20  # Variable global

def funcion_global():
    global variable_global  # Indica que se va a usar la variable global
    variable_global += 5  # Modifica la variable global
    print("Dentro de la función:", variable_global)

funcion_global()
print("Fuera de la función:", variable_global)
```

### 2. Closures

Los closures son funciones anidadas que recuerdan el entorno en el que fueron creadas, incluso si su función contenedora ha terminado de ejecutarse. Esto es útil para crear funciones que mantienen el estado entre llamadas.

#### Ejemplo de Closure

```python
def crear_multiplicador(factor):
    def multiplicar(numero):
        return numero * factor  # factor es una variable no local
    return multiplicar

multiplicador_por_2 = crear_multiplicador(2)
print(multiplicador_por_2(5))  # Salida: 10

multiplicador_por_3 = crear_multiplicador(3)
print(multiplicador_por_3(5))  # Salida: 15
```

En este ejemplo:
- `crear_multiplicador` es una función que devuelve otra función (`multiplicar`).
- `multiplicar` recuerda el valor de `factor` incluso después de que `crear_multiplicador` ha terminado de ejecutarse.
- Cada vez que se llama a `multiplicador_por_2` o `multiplicador_por_3`, se utiliza el valor de `factor` que se pasó cuando se creó el closure.

### Resumen de Alcance y Closures

- **Alcance**: Define donde una variable puede ser accedida (local, global, etc.).
  - Las **variables locales** solo se pueden usar dentro de la función donde fueron declaradas.
  - Las **variables globales** se pueden usar en cualquier parte del código.
  
- **Closures**: Permiten que una función "recuerde" el entorno en el que fue creada, manteniendo el acceso a variables no locales (que no están definidas dentro de ella) a través de su estado interno.

Estos conceptos son fundamentales para entender cómo funcionan las funciones en Python y cómo se puede gestionar el estado y la visibilidad de las variables en el código.

## Anotaciones de tipo

Las **anotaciones de tipo** en Python son una forma de especificar el tipo de variables, parámetros de funciones y valores de retorno. Aunque no afectan el funcionamiento del programa, facilitan la lectura del código y ayudan a herramientas de análisis estático a detectar errores. A continuación, te presento una explicación detallada y ejemplos sobre cómo usarlas.

### Sintaxis de Anotaciones de Tipo

1. **Parámetros de Funciones y Tipo de Retorno**

   Puedes usar las anotaciones de tipo en la definición de funciones para indicar qué tipo de datos se espera como argumento y qué tipo se devolverá:

   ```python
   def sumar(a: int, b: int) -> int:
       return a + b
   ```

   En este ejemplo, se espera que `a` y `b` sean enteros (`int`), y la función devolverá un entero.

2. **Variables**

   Aunque no es común, también puedes anotar el tipo de las variables:

   ```python
   nombre: str = "Juan"
   edad: int = 30
   ```

3. **Listas y Otras Colecciones**

   Puedes especificar el tipo de elementos en listas, tuplas y diccionarios utilizando el módulo `typing`:

   ```python
   from typing import List, Tuple, Dict

   def procesar_datos(nombres: List[str], edades: List[int]) -> Dict[str, int]:
       return dict(zip(nombres, edades))
   ```

   En este caso, `nombres` es una lista de cadenas (`List[str]`) y `edades` es una lista de enteros (`List[int]`). La función devuelve un diccionario que asocia cada nombre con una edad.

4. **Funciones Lambda**

   También puedes usar anotaciones de tipo con funciones lambda:

   ```python
   sumar: Callable[[int, int], int] = lambda x, y: x + y
   ```

   Aquí, `sumar` es una función lambda que toma dos enteros y devuelve un entero.

5. **Anotaciones en Clases**

   Las anotaciones de tipo también se pueden utilizar en las clases para definir atributos:

   ```python
   class Persona:
       def __init__(self, nombre: str, edad: int):
           self.nombre: str = nombre
           self.edad: int = edad
   ```

### Ejemplo Completo

Aquí tienes un ejemplo más completo que utiliza anotaciones de tipo en funciones y clases:

```python
from typing import List, Dict

class Estudiante:
    def __init__(self, nombre: str, calificaciones: List[float]):
        self.nombre: str = nombre
        self.calificaciones: List[float] = calificaciones

    def promedio(self) -> float:
        return sum(self.calificaciones) / len(self.calificaciones)

def registrar_estudiantes(estudiantes: List[Estudiante]) -> Dict[str, float]:
    return {estudiante.nombre: estudiante.promedio() for estudiante in estudiantes}

# Uso
est1 = Estudiante("Ana", [90, 80, 85])
est2 = Estudiante("Luis", [75, 80, 70])
resultado = registrar_estudiantes([est1, est2])
print(resultado)
```

### Ventajas de las Anotaciones de Tipo

- **Mejora la Legibilidad:** Facilita la comprensión del código, ya que queda claro qué tipos de datos se utilizan.
- **Detección Temprana de Errores:** Ayuda a herramientas de análisis estático (como `mypy`) a identificar posibles errores antes de la ejecución.
- **Mejor Integración en Editores:** Los editores pueden proporcionar autocompletado y sugerencias más precisas cuando tienen información sobre los tipos de datos.

### Conclusión

Las anotaciones de tipo en Python son una herramienta poderosa para mejorar la calidad del código y facilitar la colaboración en proyectos. Aunque no son obligatorias, su uso es altamente recomendable en proyectos grandes o en entornos donde se requiere mantener un código limpio y fácil de entender.

`mypy` es una herramienta de análisis estático para Python que se utiliza para verificar tipos de datos en tu código. Aquí tienes un desglose de sus principales funciones y beneficios:

### ¿Para qué sirve `mypy`?

1. **Verificación de Tipos Estáticos:**
   - `mypy` permite especificar anotaciones de tipo en tus funciones y variables, ayudando a identificar errores de tipo antes de que se ejecute el código. Esto es especialmente útil en un lenguaje como Python, que es dinámico y permite la escritura de código sin tipos explícitos.

2. **Detección de Errores:**
   - Ayuda a encontrar errores comunes relacionados con los tipos, como intentar sumar un número y una cadena, lo que puede causar errores en tiempo de ejecución.

3. **Mejoras en la Documentación:**
   - Las anotaciones de tipo actúan como documentación adicional. Facilitan la comprensión del código, ya que otros desarrolladores (o tú mismo en el futuro) pueden ver qué tipos de datos se esperan en cada función.

4. **Compatibilidad con Códigos Grandes:**
   - A medida que los proyectos crecen, puede volverse difícil rastrear los tipos de datos. `mypy` facilita el mantenimiento de códigos grandes al proporcionar un sistema de tipos que ayuda a garantizar la coherencia.

5. **Integración con Herramientas de Desarrollo:**
   - Puede integrarse con varios editores y entornos de desarrollo para proporcionar retroalimentación en tiempo real sobre los tipos mientras escribes código.

6. **Fomento de Buenas Prácticas:**
   - El uso de `mypy` fomenta la escritura de código más robusto y menos propenso a errores, promoviendo buenas prácticas de programación.

### Ejemplo de Uso

Aquí hay un ejemplo básico de cómo usar `mypy`:

1. **Escribir un archivo Python con anotaciones de tipo:**
   ```python
   def add(x: int, y: int) -> int:
       return x + y

   print(add(5, 3))
   ```

2. **Ejecutar `mypy` en el archivo:**
   ```bash
   mypy my_file.py
   ```

Si hay algún error de tipo, `mypy` lo señalará, ayudándote a corregirlo antes de que se ejecute el programa.

### Conclusión

`mypy` es una herramienta poderosa para mejorar la calidad del código en proyectos de Python, proporcionando verificación de tipos estáticos y ayudando a los desarrolladores a escribir un código más seguro y mantenible.

## Validación de tipos en métodos

La validación de tipos en métodos en Python es una práctica que se utiliza para asegurar que los argumentos de las funciones o métodos tienen los tipos esperados. Esto puede ayudar a prevenir errores en tiempo de ejecución y facilitar la comprensión del código. A continuación, se presentan varias maneras de implementar la validación de tipos en métodos:

### 1. Anotaciones de Tipo

Desde Python 3.5, se pueden utilizar anotaciones de tipo para indicar qué tipos de datos se esperan como argumentos y qué tipo se devolverá. Sin embargo, estas anotaciones son solo sugerencias y no imponen la validación de tipos en tiempo de ejecución. Aún así, se pueden usar herramientas como `mypy` para comprobar las anotaciones de tipo.

```python
def add(a: int, b: int) -> int:
    return a + b

print(add(5, 10))  # Salida: 15
```

### 2. Validación Manual de Tipos

Si deseas realizar la validación de tipos en tiempo de ejecución, puedes hacerlo utilizando `isinstance` dentro del método.

```python
def add(a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Los argumentos deben ser enteros.")
    return a + b

try:
    print(add(5, 10))  # Salida: 15
    print(add(5, '10'))  # Esto generará un TypeError
except TypeError as e:
    print(e)  # Salida: Los argumentos deben ser enteros.
```

### 3. Uso de Decoradores

También puedes crear un decorador para manejar la validación de tipos en múltiples funciones o métodos, lo que hace que tu código sea más limpio y reutilizable.

```python
def type_check(*arg_types):
    def decorator(func):
        def wrapper(*args):
            for a, t in zip(args, arg_types):
                if not isinstance(a, t):
                    raise TypeError(f"Argumento {a} no es de tipo {t.__name__}.")
            return func(*args)
        return wrapper
    return decorator

@type_check(int, int)
def add(a, b):
    return a + b

try:
    print(add(5, 10))  # Salida: 15
    print(add(5, '10'))  # Esto generará un TypeError
except TypeError as e:
    print(e)  # Salida: Argumento 10 no es de tipo int.
```

### 4. Librerías de Validación

Existen librerías externas como `pydantic` y `attrs` que pueden ayudarte a manejar la validación de tipos y la creación de clases de forma más robusta y concisa.

#### Ejemplo con Pydantic

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

item = Item(name="Libro", price=12.99)
print(item)
```

### Conclusión

La validación de tipos es una práctica importante en Python para asegurar la calidad y la robustez del código. Puedes optar por usar anotaciones de tipo junto con herramientas de análisis estático, o bien implementar validaciones manuales o usar decoradores para garantizar que los métodos reciban los tipos de datos esperados.

## Librería Collections y Enumeraciones

La librería `collections` de Python proporciona varios tipos de datos adicionales que pueden ser muy útiles para manipular estructuras de datos de manera más eficiente y con mayor flexibilidad. `Enum` es otra característica de Python, proporcionada por el módulo `enum`, que permite definir conjuntos de valores constantes, conocidos como enumeraciones, con nombres significativos.

### 1. `collections`: Tipos de Datos Especializados

Aquí tienes algunos de los tipos de datos más importantes en `collections`:

#### `Counter`
`Counter` es un diccionario especializado para contar elementos hashables, como caracteres en una cadena o elementos en una lista.

```python
from collections import Counter

texto = "banana"
contador = Counter(texto)
print(contador)  # Salida: Counter({'a': 3, 'n': 2, 'b': 1})
```

#### `deque`
`deque` (double-ended queue) es una estructura de datos que permite agregar y eliminar elementos desde ambos extremos de la cola de manera eficiente.

```python
from collections import deque

d = deque([1, 2, 3])
d.append(4)       # Agrega al final
d.appendleft(0)   # Agrega al principio
print(d)          # Salida: deque([0, 1, 2, 3, 4])
```

#### `defaultdict`
`defaultdict` es como un diccionario regular, pero permite especificar un valor por defecto para claves no existentes.

```python
from collections import defaultdict

def_dict = defaultdict(int)
def_dict['a'] += 1
print(def_dict)  # Salida: defaultdict(<class 'int'>, {'a': 1})
```

#### `namedtuple`
`namedtuple` permite definir tuplas con nombres para cada elemento, lo que las hace más legibles y permite acceder a los valores por nombre.

```python
from collections import namedtuple

Punto = namedtuple('Punto', ['x', 'y'])
p = Punto(3, 5)
print(p.x, p.y)  # Salida: 3 5
```

#### `OrderedDict`
`OrderedDict` es como un diccionario regular, pero mantiene el orden de los elementos según el orden de inserción.

```python
from collections import OrderedDict

orden_dict = OrderedDict()
orden_dict['a'] = 1
orden_dict['b'] = 2
orden_dict['c'] = 3
print(orden_dict)  # Salida: OrderedDict([('a', 1), ('b', 2), ('c', 3)])
```

### 2. `Enum`: Enumeraciones

Las enumeraciones son colecciones de constantes con nombres simbólicos que mejoran la legibilidad del código y evitan errores en valores repetidos o difíciles de entender.

Para crear una enumeración en Python, puedes usar la clase `Enum` del módulo `enum`:

```python
from enum import Enum

class DiaSemana(Enum):
    LUNES = 1
    MARTES = 2
    MIERCOLES = 3
    JUEVES = 4
    VIERNES = 5

print(DiaSemana.LUNES)         # Salida: DiaSemana.LUNES
print(DiaSemana.LUNES.name)    # Salida: LUNES
print(DiaSemana.LUNES.value)   # Salida: 1
```

#### Enumeraciones avanzadas: `IntEnum` y `auto()`
- **IntEnum**: Permite que las enumeraciones se comporten como enteros.
- **auto()**: Asigna automáticamente valores secuenciales.

```python
from enum import IntEnum, auto

class Nivel(IntEnum):
    BAJO = auto()
    MEDIO = auto()
    ALTO = auto()

print(Nivel.BAJO)      # Salida: Nivel.BAJO
print(Nivel.BAJO.value)  # Salida: 1
```

### Conclusión

- `collections` ofrece estructuras de datos eficientes para tareas comunes de manipulación de datos.
- `Enum` ayuda a definir constantes simbólicas, mejorando la legibilidad y evitando errores de valores ambiguos.

Ambas herramientas amplían las funcionalidades de Python de una manera eficaz y elegante.

**Archivos de la clase**

[1-fundamentos-avanzados-20241024t123600z-001.zip](https://static.platzi.com/media/public/uploads/1-fundamentos-avanzados-20241024t123600z-001_b1d58ae2-f393-443f-9aae-d4b533d48698.zip)

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at collections](https://github.com/platzi/python-avanzado/tree/collections)

## Decoradores en Python

Los decoradores en Python son funciones que modifican el comportamiento de otras funciones o métodos. Son una herramienta muy útil para añadir funcionalidades o preprocesamientos sin tener que cambiar el código original de la función decorada.

### ¿Qué es un Decorador?

Un decorador es una función que toma otra función como argumento y le añade funcionalidades adicionales. Devuelve una nueva función modificada o envuelta con el comportamiento adicional.

La sintaxis básica de un decorador utiliza el símbolo `@` seguido del nombre del decorador antes de la definición de la función que se quiere decorar:

```python
@mi_decorador
def funcion():
    pass
```

### Ejemplo Básico de Decorador

Aquí tienes un ejemplo de un decorador que muestra un mensaje antes y después de ejecutar la función:

```python
def mi_decorador(funcion):
    def wrapper():
        print("Antes de la función")
        funcion()
        print("Después de la función")
    return wrapper

@mi_decorador
def saludo():
    print("Hola")

saludo()
```

**Salida:**
```plaintext
Antes de la función
Hola
Después de la función
```

### Decoradores con Argumentos en las Funciones

Si la función original toma argumentos, el decorador debe adaptarse para recibirlos y pasarlos correctamente:

```python
def decorador_con_argumentos(funcion):
    def wrapper(*args, **kwargs):
        print("Llamando a la función con argumentos:", args, kwargs)
        resultado = funcion(*args, **kwargs)
        print("Resultado:", resultado)
        return resultado
    return wrapper

@decorador_con_argumentos
def suma(a, b):
    return a + b

suma(3, 5)
```

**Salida:**
```plaintext
Llamando a la función con argumentos: (3, 5) {}
Resultado: 8
```

### Decoradores Anidados

Se pueden aplicar varios decoradores a una misma función. En este caso, los decoradores se aplican en orden desde el más cercano a la función hacia el exterior:

```python
def decorador1(funcion):
    def wrapper():
        print("Decorador 1")
        funcion()
    return wrapper

def decorador2(funcion):
    def wrapper():
        print("Decorador 2")
        funcion()
    return wrapper

@decorador1
@decorador2
def mi_funcion():
    print("Función original")

mi_funcion()
```

**Salida:**
```plaintext
Decorador 1
Decorador 2
Función original
```

### Decoradores de Clase

Los decoradores no solo se limitan a funciones; también pueden aplicarse a clases para modificar su comportamiento. Aquí tienes un ejemplo de decorador que modifica el método `__init__` de una clase:

```python
def decorador_clase(cls):
    class NuevaClase(cls):
        def __init__(self, *args, **kwargs):
            print("Iniciando con decorador de clase")
            super().__init__(*args, **kwargs)
    return NuevaClase

@decorador_clase
class Persona:
    def __init__(self, nombre):
        self.nombre = nombre

p = Persona("Carlos")
```

**Salida:**
```plaintext
Iniciando con decorador de clase
```

### Decoradores Integrados en Python

Python ofrece algunos decoradores integrados, como:

- `@staticmethod`: Define métodos estáticos que no requieren una instancia de la clase.
- `@classmethod`: Define métodos de clase que reciben la clase como primer argumento en lugar de la instancia.
- `@property`: Define métodos como propiedades, permitiendo acceso a métodos como si fueran atributos.

### Decoradores con Argumentos Propios

A veces, es útil que un decorador acepte argumentos. En estos casos, el decorador se define dentro de otra función, lo cual permite que la función exterior reciba argumentos:

```python
def decorador_con_parametros(mensaje):
    def decorador(funcion):
        def wrapper(*args, **kwargs):
            print(mensaje)
            return funcion(*args, **kwargs)
        return wrapper
    return decorador

@decorador_con_parametros("Ejecutando función")
def resta(a, b):
    return a - b

resta(10, 3)
```

**Salida:**
```plaintext
Ejecutando función
```

### Conclusión

Los decoradores son una herramienta poderosa en Python que permite modificar funciones y métodos sin cambiar su implementación interna, siendo útiles para casos como:

- Validación de datos
- Manejo de excepciones
- Medición de tiempo de ejecución
- Creación de APIs

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at decoradores](https://github.com/platzi/python-avanzado/tree/decoradores)

## Decoradores anidados y con parámetros

Los decoradores anidados y con parámetros en Python permiten aplicar múltiples transformaciones a funciones o métodos de una forma flexible y reutilizable. Aquí te explico cómo funcionan ambos conceptos y te muestro ejemplos.

### 1. Decoradores Anidados

Los decoradores anidados son simplemente varios decoradores aplicados a una misma función, uno tras otro. Se aplican en el orden en que aparecen, de afuera hacia adentro. Esto significa que el decorador más cercano a la función será ejecutado primero.

**Ejemplo de decoradores anidados:**

```python
def decorador1(funcion):
    def wrapper(*args, **kwargs):
        print("Ejecutando decorador1")
        return funcion(*args, **kwargs)
    return wrapper

def decorador2(funcion):
    def wrapper(*args, **kwargs):
        print("Ejecutando decorador2")
        return funcion(*args, **kwargs)
    return wrapper

@decorador1
@decorador2
def saludo(nombre):
    print(f"Hola, {nombre}")

saludo("Carlos")
```

**Salida:**
```plaintext
Ejecutando decorador1
Ejecutando decorador2
Hola, Carlos
```

En este caso:
- `decorador2` se aplica primero, luego `decorador1`.
- `saludo` pasa por ambos decoradores antes de ejecutar su lógica.

### 2. Decoradores con Parámetros

Los decoradores con parámetros permiten personalizar el comportamiento del decorador. Para crear un decorador con parámetros, se define una función que recibe estos parámetros y que, a su vez, devuelve el decorador propiamente dicho.

**Ejemplo de decorador con parámetros:**

```python
def repetir(n):
    def decorador(funcion):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                funcion(*args, **kwargs)
        return wrapper
    return decorador

@repetir(3)
def saludo(nombre):
    print(f"Hola, {nombre}")

saludo("Carlos")
```

**Salida:**
```plaintext
Hola, Carlos
Hola, Carlos
Hola, Carlos
```

Aquí:
- `repetir(3)` llama al decorador `repetir` con `n=3`.
- El decorador envuelve a `saludo`, que se ejecutará 3 veces.

### 3. Decoradores Anidados con Parámetros

Podemos combinar ambas ideas aplicando varios decoradores con parámetros. Esto permite una gran flexibilidad y personalización en el comportamiento de las funciones decoradas.

**Ejemplo de decoradores anidados con parámetros:**

```python
def prefijo(texto):
    def decorador(funcion):
        def wrapper(*args, **kwargs):
            print(texto, end=" ")
            return funcion(*args, **kwargs)
        return wrapper
    return decorador

def repetir(n):
    def decorador(funcion):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                funcion(*args, **kwargs)
        return wrapper
    return decorador

@prefijo("Atención:")
@repetir(2)
def saludo(nombre):
    print(f"Hola, {nombre}")

saludo("Carlos")
```

**Salida:**
```plaintext
Atención: Hola, Carlos
Atención: Hola, Carlos
```

En este ejemplo:
- `@repetir(2)` envuelve la función `saludo` para ejecutarse 2 veces.
- `@prefijo("Atención:")` envuelve `saludo` añadiendo el prefijo "Atención:" antes de cada saludo.

### Detalles Técnicos

Cuando usas decoradores anidados con parámetros, el orden de ejecución sigue siendo de afuera hacia adentro. Esto permite aplicar las personalizaciones de los decoradores en el orden deseado, haciendo que cada paso dependa del resultado del anterior.

Esta técnica es especialmente útil cuando necesitas funcionalidades como:

- **Autorización y autenticación:** Un decorador puede verificar permisos antes de que otro ejecute la lógica principal.
- **Registro y monitoreo:** Puedes aplicar un decorador que registre la ejecución de una función antes de realizar otras acciones.
- **Manipulación de datos:** Un decorador puede transformar datos de entrada o salida antes de que otro decorador aplique sus cambios. 

## Uso de Decoradores en clases y métodos

En Python, los decoradores no solo se usan en funciones, también se pueden aplicar a clases y métodos para extender o modificar su comportamiento. Los decoradores en clases y métodos son especialmente útiles para aspectos transversales como la validación de permisos, el registro de accesos, el manejo de excepciones y la administración de cachés.

Aquí tienes una guía rápida sobre cómo funcionan y un ejemplo detallado:

### Decoradores en Métodos de Clase

Los decoradores pueden aplicarse a métodos específicos de una clase, y se comportan según el tipo de método:

- **Métodos de instancia** (los métodos normales que acceden a `self`).
- **Métodos de clase** (aquellos que usan `@classmethod` y acceden a `cls`).
- **Métodos estáticos** (usando `@staticmethod`, no acceden ni a `self` ni a `cls`).

Los decoradores en métodos de clase pueden hacer cosas como:

1. Controlar permisos antes de ejecutar el método.
2. Hacer un registro (logging) cada vez que se llama al método.
3. Manejar errores de forma consistente.

### Decoradores en Clases

Un decorador en una clase modifica el comportamiento de la clase en su conjunto, normalmente para extenderla o agregarle nuevas funcionalidades.

### Ejemplo de Uso

Imaginemos una clase `CuentaBancaria` donde queremos aplicar un decorador para registrar cada vez que se realiza una operación, y otro decorador para comprobar permisos de acceso a ciertas operaciones.

```python
from functools import wraps
import datetime

# Decorador para verificar permisos
def verificar_permisos(permiso_necesario):
    def decorador_metodo(func):
        @wraps(func)
        def envoltura(self, *args, **kwargs):
            if permiso_necesario not in self.permisos:
                raise PermissionError(f"No tienes permisos para realizar esta operación: {permiso_necesario}")
            return func(self, *args, **kwargs)
        return envoltura
    return decorador_metodo

# Decorador para registrar en log
def registrar_operacion(func):
    @wraps(func)
    def envoltura(self, *args, **kwargs):
        resultado = func(self, *args, **kwargs)
        operacion = f"{func.__name__} ejecutado el {datetime.datetime.now()}"
        self.historial.append(operacion)
        print(f"Registro: {operacion}")
        return resultado
    return envoltura

# Clase CuentaBancaria usando decoradores en sus métodos
class CuentaBancaria:
    def __init__(self, balance=0):
        self.balance = balance
        self.historial = []
        self.permisos = ['ver_balance', 'retirar']  # permisos actuales del usuario

    @registrar_operacion
    @verificar_permisos('ver_balance')
    def ver_balance(self):
        print(f"Balance actual: ${self.balance}")
        return self.balance

    @registrar_operacion
    @verificar_permisos('retirar')
    def retirar(self, cantidad):
        if cantidad > self.balance:
            raise ValueError("Fondos insuficientes")
        self.balance -= cantidad
        print(f"Retiro exitoso: ${cantidad}. Nuevo balance: ${self.balance}")
        return self.balance

# Uso de la clase con decoradores
cuenta = CuentaBancaria(1000)

# Operaciones
cuenta.ver_balance()
cuenta.retirar(200)

# Intento de retiro sin permisos
try:
    cuenta.permisos.remove('retirar')
    cuenta.retirar(100)
except PermissionError as e:
    print(e)
```

### Explicación

1. **`@verificar_permisos`**: Este decorador toma un parámetro, `permiso_necesario`, y se asegura de que el usuario tenga el permiso adecuado antes de ejecutar el método.
2. **`@registrar_operacion`**: Este decorador registra cada operación realizada, almacenándola en el atributo `historial`.

Este patrón es muy poderoso para clases que requieren varias verificaciones y registros en sus métodos, proporcionando una estructura de código limpia y modular.

## Métodos mágicos

Los **métodos mágicos** en Python (también llamados "métodos especiales" o "dunder methods" por la doble subrayado que los rodea) son funciones predefinidas en las clases que permiten personalizar el comportamiento de los objetos en varias situaciones, como en operaciones matemáticas, conversiones de tipos, comparaciones y manejo de contenedores. 

Estos métodos comienzan y terminan con doble subrayado (`__`), por ejemplo, `__init__` o `__str__`. Al definir estos métodos, se le puede indicar a Python cómo debe comportarse un objeto en distintos contextos.

### 1. Métodos de Inicialización y Representación

- **`__init__(self, ...)`**: Inicializador de una clase (similar a un constructor). Se ejecuta al crear una instancia.
- **`__str__(self)`**: Define el comportamiento del objeto cuando se convierte a una cadena con `str()`, como al usar `print()`.
- **`__repr__(self)`**: Representación oficial del objeto, útil para depuración. A menudo se usa en lugar de `__str__` para generar un mensaje detallado y debe proporcionar una salida que permita recrear el objeto cuando se usa `eval()`.

   ```python
   class Persona:
       def __init__(self, nombre, edad):
           self.nombre = nombre
           self.edad = edad

       def __str__(self):
           return f"Persona({self.nombre}, {self.edad})"
       
       def __repr__(self):
           return f"Persona(nombre={self.nombre!r}, edad={self.edad})"

   persona = Persona("Ana", 30)
   print(persona)         # Salida personalizada
   repr(persona)          # Representación oficial para depuración
   ```

### 2. Métodos de Operadores Aritméticos

Permiten personalizar el comportamiento de operadores (`+`, `-`, `*`, `/`, etc.).

- **`__add__(self, other)`**: Define el comportamiento de la adición (`self + other`).
- **`__sub__(self, other)`**: Define la resta (`self - other`).
- **`__mul__(self, other)`**: Define la multiplicación (`self * other`).
- **`__truediv__(self, other)`**: Define la división (`self / other`).

   ```python
   class Vector:
       def __init__(self, x, y):
           self.x = x
           self.y = y

       def __add__(self, other):
           return Vector(self.x + other.x, self.y + other.y)

       def __str__(self):
           return f"Vector({self.x}, {self.y})"

   v1 = Vector(2, 3)
   v2 = Vector(1, 5)
   print(v1 + v2)  # Salida: Vector(3, 8)
   ```

### 3. Métodos de Comparación

Permiten especificar el comportamiento para comparaciones (`==`, `<`, `>`, etc.).

- **`__eq__(self, other)`**: Define la comparación de igualdad (`self == other`).
- **`__lt__(self, other)`**: Define la comparación de "menor que" (`self < other`).
- **`__le__(self, other)`**: Define la comparación de "menor o igual" (`self <= other`).

   ```python
   class Persona:
       def __init__(self, nombre, edad):
           self.nombre = nombre
           self.edad = edad

       def __eq__(self, other):
           return self.edad == other.edad

       def __lt__(self, other):
           return self.edad < other.edad

   persona1 = Persona("Juan", 25)
   persona2 = Persona("Ana", 30)
   print(persona1 == persona2)  # Salida: False
   print(persona1 < persona2)   # Salida: True
   ```

### 4. Métodos de Acceso a Elementos

Permiten definir el comportamiento de acceso, modificación y eliminación de elementos en objetos como si fueran contenedores.

- **`__getitem__(self, key)`**: Define el comportamiento al acceder a un elemento (`self[key]`).
- **`__setitem__(self, key, value)`**: Define el comportamiento al asignar un valor (`self[key] = value`).
- **`__delitem__(self, key)`**: Define el comportamiento al eliminar un elemento (`del self[key]`).

   ```python
   class Contenedor:
       def __init__(self):
           self.datos = {}

       def __getitem__(self, key):
           return self.datos.get(key, None)

       def __setitem__(self, key, value):
           self.datos[key] = value

       def __delitem__(self, key):
           del self.datos[key]

   cont = Contenedor()
   cont["a"] = 10
   print(cont["a"])  # Salida: 10
   del cont["a"]
   ```

### 5. Métodos de Contexto (con `with`)

Permiten definir el comportamiento al usar un objeto con el bloque `with`, como al manejar archivos.

- **`__enter__(self)`**: Inicializa o prepara el objeto para el contexto.
- **`__exit__(self, exc_type, exc_val, exc_tb)`**: Define el comportamiento al salir del contexto, manejando cualquier excepción.

   ```python
   class Archivo:
       def __init__(self, nombre):
           self.nombre = nombre
           self.archivo = None

       def __enter__(self):
           self.archivo = open(self.nombre, "w")
           return self.archivo

       def __exit__(self, exc_type, exc_val, exc_tb):
           if self.archivo:
               self.archivo.close()

   with Archivo("ejemplo.txt") as f:
       f.write("Texto de ejemplo")
   ```

### Resumen

Los métodos mágicos permiten una gran personalización y facilitan la creación de clases que se comportan de forma coherente con los operadores y funciones de Python. Al definir estos métodos, puedes hacer que las clases respondan a operaciones y funciones nativas de Python, logrando una sintaxis más intuitiva y adaptada a tus necesidades.

**Archivos de la clase**

[metodos-y-estructuras.zip](https://static.platzi.com/media/public/uploads/metodos-y-estructuras_24a6f051-c67c-4120-80d2-4b58b1ab01d8.zip)

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at staticClassMethod](https://github.com/platzi/python-avanzado/tree/staticClassMethod)

## Sobrecarga de operadores

![A visual representation of Operator Overloading in Python.jpg](images/A_visual_representation_of_Operator_Overloading_in_Python.jpg)

Imagina que puedes hacer que tus clases personalizadas en Python se comporten como números, listas o cadenas de texto, permitiendo sumar objetos, compararlos y mucho más. ¿Qué pasaría si pudieras redefinir cómo tus clases responden a operaciones comunes como +, -, ==, o incluso <? Esa es la magia de la **sobrecarga de operadores**.

En esta clase, aprenderás a darle superpoderes a tus objetos para que puedan interactuar de manera intuitiva con los operadores estándar de Python. Ya no se trata solo de crear clases; ahora, tus clases podrán comportarse como cualquier otro tipo de dato nativo de Python, lo que hará tu código más limpio, legible y poderoso.

¿Quieres que tus objetos se sumen como fracciones o se comparen como personas? La sobrecarga de operadores te permitirá hacerlo. Al final de esta lección, estarás creando clases que pueden sumar, restar, comparar y mucho más, llevando tu programación en Python a otro nivel. ¡Vamos a descubrir cómo hacerlo.

### 1. ¿Qué es la Sobrecarga de Operadores?

Por defecto, los operadores en Python como + o == solo funcionan con tipos de datos predefinidos (números, cadenas, listas, etc.). Sin embargo, con la sobrecarga de operadores, podemos modificar cómo estos operadores funcionan con nuestras clases personalizadas.

**Ejemplo básico de sobrecarga de +:**

Imagina que tienes una clase Vector, y quieres sumar dos vectores usando el operador +. Para esto, usaremos el método mágico __add__.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(4, 1)

v3 = v1 + v2  # Sobrecarga de `+`
print(v3)  # Output: Vector(6, 4)
```

Aquí, __add__ define que la suma de dos objetos Vector es un nuevo Vector con la suma de sus componentes.

### 2. Sobrecarga de Comparación (==, <, >)

La sobrecarga no se limita a operadores aritméticos, también podemos redefinir operadores de comparación como ==, <, > para que comparen objetos en función de los atributos que queramos.

**Ejemplo de sobrecarga de == para comparar objetos:**

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    def __eq__(self, otra_persona):
        return self.nombre == otra_persona.nombre and self.edad == otra_persona.edad

p1 = Persona("Ana", 30)
p2 = Persona("Ana", 30)

print(p1 == p2)  # Output: True (Ambas personas tienen el mismo nombre y edad)
```

En este caso, __eq__ permite que el operador == compare dos personas por sus atributos nombre y edad.

### 3. Ejemplo de Sobrecarga de Otros Operadores

Aparte de + y ==, otros operadores pueden ser sobrecargados, como el operador de resta -, multiplicación *, y operadores de comparación como <, >. Veamos un ejemplo de sobrecarga del operador de comparación <.

**Ejemplo con el operador < (menor que):**

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    def __lt__(self, otra_persona):
        return self.edad < otra_persona.edad

p1 = Persona("Ana", 25)
p2 = Persona("Luis", 30)

print(p1 < p2)  # Output: True (Ana es menor que Luis)

```
Aquí, __lt__ permite comparar las edades de dos personas con el operador <.

### 4. Buenas Prácticas al Sobrecargar Operadores

1. **Usa la sobrecarga cuando tenga sentido**: No abuses de la sobrecarga de operadores. Solo la utilices cuando sea intuitivo y claro que un operador debe funcionar con tus clases.

2. **Mantén la consistencia**: Si sobrecargas un operador como +, asegúrate de que el comportamiento sea consistente con lo que los usuarios esperan (por ejemplo, que la suma de dos vectores realmente sume sus componentes).

3. **Documenta el comportamiento**: Aunque la sobrecarga de operadores puede hacer que tu código sea más limpio, es importante que documentes claramente cómo se comportan los operadores sobrecargados, especialmente si tienen un comportamiento no convencional.

### 5. Ejercicio Práctico: Sobrecargar el Operador + en una Clase de Fracciones

**Objetivo**: Implementa una clase Fraccion que permita sumar fracciones usando el operador +.

**Requerimientos:**

1. La clase debe tener numerador y denominador.
2. El operador + debe sumar dos fracciones y devolver el resultado simplificado.

```python
from math import gcd

class Fraccion:
    def __init__(self, numerador, denominador):
        self.numerador = numerador
        self.denominador = denominador

    def __add__(self, otra_fraccion):
        nuevo_num = self.numerador * otra_fraccion.denominador + otra_fraccion.numerador * self.denominador
        nuevo_den = self.denominador * otra_fraccion.denominador
        comun_divisor = gcd(nuevo_num, nuevo_den)
        return Fraccion(nuevo_num // comun_divisor, nuevo_den // comun_divisor)

    def __repr__(self):
        return f"{self.numerador}/{self.denominador}"

f1 = Fraccion(1, 4)
f2 = Fraccion(1, 2)

f3 = f1 + f2  # Suma de fracciones
print(f3)  # Output: 3/4
```

Este ejemplo muestra cómo redefinir el operador + para sumar fracciones de manera intuitiva.

¡Felicidades por completar esta clase sobre Sobrecarga de Operadores en Python! Ahora has aprendido cómo personalizar el comportamiento de los operadores en tus clases, lo que te permite crear código más intuitivo, limpio y poderoso.

Al comprender cómo los operadores pueden ser sobrecargados, has desbloqueado una nueva capa de flexibilidad en tus proyectos. Ya no tienes que conformarte con el comportamiento predeterminado de Python: ahora puedes hacer que tus clases se comporten como cualquier otro tipo de dato nativo.

Ahora es el momento de aplicar lo que has aprendido. ¡Ve y experimenta con tus propias clases y operadores!

La **sobrecarga de operadores** en Python permite redefinir el comportamiento de los operadores estándar (como `+`, `-`, `*`, `==`, etc.) para que puedan ser usados con objetos de una clase personalizada. A través de los **métodos especiales** (también conocidos como "métodos mágicos" o "dunder methods"), Python permite implementar esta funcionalidad en clases, facilitando la creación de clases con operaciones específicas.

### Ejemplo básico de sobrecarga de operadores

Supongamos que queremos una clase `Vector` que represente un vector en un plano 2D. Podríamos querer sumar dos instancias de `Vector` con el operador `+`, pero por defecto Python no sabe cómo hacer esta operación entre objetos de esta clase. Implementaríamos el método especial `__add__` para que la suma funcione:

```python
class Vector:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        raise TypeError("El operando debe ser de tipo Vector")

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(5, 7)
v3 = v1 + v2  # Esto llama a v1.__add__(v2)
print(v3)     # Salida: Vector(7, 10)
```

Aquí el método `__add__` permite usar `+` para sumar dos vectores, devolviendo un nuevo objeto `Vector`.

### Métodos especiales para sobrecargar operadores

Existen varios métodos especiales en Python para sobrecargar distintos operadores. Aquí tienes una lista de algunos operadores y los métodos correspondientes:

1. **Operadores Aritméticos**:
   - `+` : `__add__(self, other)`
   - `-` : `__sub__(self, other)`
   - `*` : `__mul__(self, other)`
   - `/` : `__truediv__(self, other)`
   - `//` : `__floordiv__(self, other)`
   - `%` : `__mod__(self, other)`
   - `**` : `__pow__(self, other)`

2. **Operadores de Comparación**:
   - `==` : `__eq__(self, other)`
   - `!=` : `__ne__(self, other)`
   - `<` : `__lt__(self, other)`
   - `<=` : `__le__(self, other)`
   - `>` : `__gt__(self, other)`
   - `>=` : `__ge__(self, other)`

3. **Operadores de Asignación Aritmética**:
   - `+=` : `__iadd__(self, other)`
   - `-=` : `__isub__(self, other)`
   - `*=` : `__imul__(self, other)`
   - `/=` : `__itruediv__(self, other)`

4. **Conversión de Tipos**:
   - `__int__(self)` para `int()`
   - `__float__(self)` para `float()`
   - `__str__(self)` para `str()`

### Ejemplo completo con varios operadores

Aquí tienes un ejemplo con una clase `Complejo` que representa números complejos, sobrecargando operadores aritméticos y de comparación:

```python
class Complejo:
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if isinstance(other, Complejo):
            return Complejo(self.real + other.real, self.imag + other.imag)
        raise TypeError("El operando debe ser de tipo Complejo")

    def __sub__(self, other):
        if isinstance(other, Complejo):
            return Complejo(self.real - other.real, self.imag - other.imag)
        raise TypeError("El operando debe ser de tipo Complejo")

    def __mul__(self, other):
        if isinstance(other, Complejo):
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return Complejo(real, imag)
        raise TypeError("El operando debe ser de tipo Complejo")

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __str__(self):
        return f"({self.real} + {self.imag}i)"

c1 = Complejo(2, 3)
c2 = Complejo(1, -1)

print(c1 + c2)  # Salida: (3 + 2i)
print(c1 - c2)  # Salida: (1 + 4i)
print(c1 * c2)  # Salida: (5 + 1i)
print(c1 == c2) # Salida: False
```

### Consideraciones al Sobrecargar Operadores

1. **Compatibilidad**: Al implementar métodos de sobrecarga, asegúrate de manejar correctamente errores de tipo, como en `isinstance()` para verificar tipos de entrada.
2. **Inmutabilidad vs. Mutabilidad**: Decide si los métodos devuelven nuevos objetos o modifican el objeto en sí, ya que puede afectar el uso de instancias en el programa.
3. **Legibilidad**: Usa sobrecarga solo cuando el significado del operador sea intuitivo para la clase; sobrecargar sin una lógica clara puede causar confusión en el código.

Sobrecargar operadores permite hacer que los objetos sean más expresivos y fáciles de manipular, lo cual resulta útil en programas complejos donde el uso de operadores personalizados aumenta la legibilidad y reduce errores.

## Implementación de `if __name__ == "__main__":`

El bloque `if __name__ == "__main__":` en Python se usa para que una sección específica de código solo se ejecute cuando el archivo es ejecutado directamente, y no cuando es importado como módulo en otro archivo. Este es un patrón común en Python, especialmente para escribir scripts y bibliotecas que puedan ser reutilizados y probados.

### ¿Cómo funciona `if __name__ == "__main__":`?

Cuando se ejecuta un archivo Python, el intérprete asigna el valor especial `"__main__"` a la variable `__name__` si el archivo se ejecuta directamente. Sin embargo, si el archivo es importado como un módulo en otro archivo, `__name__` toma el valor del nombre del archivo, sin ejecutar el bloque que depende de `"__main__"`.

### Estructura del Bloque `if __name__ == "__main__":`

```python
# archivo.py

def funcion_principal():
    print("Esta es la función principal del módulo.")

# Ejecutar solo si es el archivo principal
if __name__ == "__main__":
    print("El archivo se ejecuta directamente.")
    funcion_principal()
```

### Ejemplo de uso práctico

Supongamos que tienes un archivo, `calculadora.py`, que define algunas funciones matemáticas:

```python
# calculadora.py

def sumar(a, b):
    return a + b

def restar(a, b):
    return a - b

def multiplicar(a, b):
    return a * b

def dividir(a, b):
    return a / b

if __name__ == "__main__":
    print("Ejecutando pruebas...")
    print(sumar(5, 3))         # Salida: 8
    print(restar(5, 3))        # Salida: 2
    print(multiplicar(5, 3))   # Salida: 15
    print(dividir(5, 3))       # Salida: 1.666...
```

### Ejecución del Bloque

1. **Ejecución directa**: Si corres `calculadora.py` directamente (`python calculadora.py`), se imprimirá el mensaje `"Ejecutando pruebas..."` y el resultado de cada función.
2. **Importación como módulo**: Si importas `calculadora.py` en otro archivo, digamos `main.py`:

   ```python
   # main.py
   import calculadora

   resultado = calculadora.sumar(10, 5)
   print(resultado)  # Salida: 15
   ```

   En este caso, `calculadora.py` **no** ejecutará el bloque dentro de `if __name__ == "__main__":`, evitando la impresión del mensaje de prueba y otros cálculos no deseados.

### Ventajas de `if __name__ == "__main__":`

1. **Evita ejecución no deseada**: Permite ejecutar código solo cuando se ejecuta el archivo principal y evita la ejecución de pruebas o funciones adicionales cuando el archivo es importado.
2. **Modularidad**: Facilita el uso de módulos en diferentes proyectos sin cambios adicionales.
3. **Pruebas rápidas**: Permite agregar pruebas simples o mensajes para verificar el comportamiento del código en desarrollo, manteniéndolo separado de la funcionalidad principal.

En general, `if __name__ == "__main__":` ayuda a mantener el código organizado y reutilizable, al tiempo que permite pruebas internas sin interferir cuando el código se importa en otros archivos.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at ifNameMain](https://github.com/platzi/python-avanzado/tree/ifNameMain)

## Metaprogramación en Python

La metaprogramación en Python es una técnica que permite a los programas manipular o generar código durante su ejecución. Esto se puede lograr a través de varias funcionalidades del lenguaje, como las clases, los decoradores, las funciones de alto orden y los metaclases. Aquí tienes un resumen de las principales técnicas de metaprogramación en Python:

### 1. **Decoradores**

Los decoradores son una forma de modificar el comportamiento de una función o método. Puedes crear decoradores para agregar funcionalidades adicionales sin modificar el código original.

```python
def mi_decorador(func):
    def envoltura():
        print("Algo se está haciendo antes de llamar a la función.")
        func()
        print("Algo se está haciendo después de llamar a la función.")
    return envoltura

@mi_decorador
def saludar():
    print("¡Hola!")

saludar()
```

### 2. **Funciones de alto orden**

Las funciones de alto orden son aquellas que pueden recibir otras funciones como argumentos o devolverlas como resultados. Esto permite crear comportamientos dinámicos.

```python
def aplicar_funcion(func, valor):
    return func(valor)

def elevar_al_cuadrado(x):
    return x ** 2

resultado = aplicar_funcion(elevar_al_cuadrado, 5)
print(resultado)  # Salida: 25
```

### 3. **Clases y la creación dinámica de clases**

Puedes crear clases dinámicamente utilizando el método `type()`, lo que permite definir clases en tiempo de ejecución.

```python
def crear_clase(nombre):
    return type(nombre, (object,), {})

MiClase = crear_clase("ClaseDinamica")
objeto = MiClase()
print(type(objeto))  # Salida: <class '__main__.ClaseDinamica'>
```

### 4. **Metaclases**

Las metaclases son clases de clases, lo que significa que definen cómo se crean las clases. Puedes crear una metaclase para modificar la creación de clases.

```python
class MiMeta(type):
    def __new__(cls, nombre, bases, dct):
        dct['nuevo_atributo'] = 'Soy un nuevo atributo'
        return super().__new__(cls, nombre, bases, dct)

class MiClase(metaclass=MiMeta):
    pass

objeto = MiClase()
print(objeto.nuevo_atributo)  # Salida: Soy un nuevo atributo
```

### 5. **Reflexión y introspección**

Python permite inspeccionar objetos en tiempo de ejecución, lo que puede ser útil para metaprogramación. Puedes utilizar funciones como `getattr()`, `setattr()`, `hasattr()` y `dir()`.

```python
class Persona:
    def __init__(self, nombre):
        self.nombre = nombre

persona = Persona("Juan")
print(getattr(persona, 'nombre'))  # Salida: Juan
setattr(persona, 'edad', 30)
print(persona.edad)  # Salida: 30
```

### Ventajas y desventajas

**Ventajas:**
- Permite crear código más flexible y reutilizable.
- Facilita la creación de bibliotecas y frameworks.
- Puede simplificar el código al evitar duplicaciones.

**Desventajas:**
- Puede hacer que el código sea más difícil de entender.
- El uso excesivo puede llevar a un rendimiento reducido.
- Puede complicar la depuración debido a su naturaleza dinámica.

La metaprogramación es una poderosa herramienta en Python, pero debe usarse con cuidado para mantener la claridad y la mantenibilidad del código.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at metaprogramacion](https://github.com/platzi/python-avanzado/tree/metaprogramacion)

## Uso de *args y **kwargs

Los parámetros `*args` y `**kwargs` en Python son utilizados para pasar un número variable de argumentos a una función. Esto es útil cuando no se conoce de antemano cuántos argumentos se van a pasar a la función. Vamos a desglosar cada uno de ellos:

### `*args`

- **Significado**: `*args` permite pasar un número variable de argumentos posicionales a una función. Cuando se utiliza, Python agrupa todos los argumentos adicionales que se pasan a la función en una tupla.
- **Uso**: Se usa cuando no se sabe cuántos argumentos se van a pasar a la función.

#### Ejemplo de `*args`

```python
def sumar(*args):
    total = 0
    for numero in args:
        total += numero
    return total

resultado = sumar(1, 2, 3, 4, 5)
print(resultado)  # Salida: 15
```

En este ejemplo, la función `sumar` puede aceptar cualquier cantidad de argumentos, y todos ellos se suman.

### `**kwargs`

- **Significado**: `**kwargs` permite pasar un número variable de argumentos nombrados (keyword arguments) a una función. Python agrupa todos los argumentos adicionales en un diccionario.
- **Uso**: Se utiliza cuando se quieren pasar un número variable de argumentos con nombre.

#### Ejemplo de `**kwargs`

```python
def imprimir_info(**kwargs):
    for clave, valor in kwargs.items():
        print(f"{clave}: {valor}")

imprimir_info(nombre="Juan", edad=30, ciudad="Bogotá")
```

Salida:
```
nombre: Juan
edad: 30
ciudad: Bogotá
```

En este caso, la función `imprimir_info` recibe argumentos nombrados que se almacenan en un diccionario `kwargs`. Luego, se itera sobre ese diccionario para imprimir cada clave y su correspondiente valor.

### Uso combinado de `*args` y `**kwargs`

Puedes usar `*args` y `**kwargs` en la misma función. Sin embargo, siempre debes poner `*args` antes de `**kwargs`.

#### Ejemplo combinado

```python
def mostrar_datos(*args, **kwargs):
    print("Argumentos posicionales:", args)
    print("Argumentos con nombre:", kwargs)

mostrar_datos(1, 2, 3, nombre="Juan", edad=30)
```

Salida:
```
Argumentos posicionales: (1, 2, 3)
Argumentos con nombre: {'nombre': 'Juan', 'edad': 30}
```

### Resumen

- **`*args`**: Para argumentos posicionales variables, se agrupan en una tupla.
- **`**kwargs`**: Para argumentos nombrados variables, se agrupan en un diccionario.

Ambos son herramientas poderosas que permiten a las funciones ser más flexibles y adaptables a diferentes situaciones.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at Args](https://github.com/platzi/python-avanzado/tree/Args)

## Métodos privados y protegidos

En Python, los métodos privados y protegidos son convenciones utilizadas para controlar el acceso a los atributos y métodos de una clase. Aunque Python no implementa un verdadero encapsulamiento como otros lenguajes (por ejemplo, Java o C++), proporciona ciertas convenciones que ayudan a gestionar la accesibilidad. Aquí tienes un desglose de ambos conceptos:

### Métodos Privados

- **Definición**: Los métodos privados son aquellos que no deben ser accesibles desde fuera de la clase. Se utilizan para encapsular la lógica que no debería ser expuesta.
- **Convención**: Se definen precediendo el nombre del método con dos guiones bajos (`__`). Esto provoca que el método sea "name-mangled", es decir, el nombre del método se transforma internamente para incluir el nombre de la clase.

#### Ejemplo de Métodos Privados

```python
class MiClase:
    def __init__(self):
        self.__atributo_privado = "Soy privado"

    def __metodo_privado(self):
        return "Este es un método privado"

    def metodo_publico(self):
        return self.__metodo_privado()

objeto = MiClase()
print(objeto.metodo_publico())  # Salida: Este es un método privado

# Intentar acceder al método privado directamente resultará en un error
# print(objeto.__metodo_privado())  # Esto generará un AttributeError
```

### Métodos Protegidos

- **Definición**: Los métodos protegidos son aquellos que están destinados a ser utilizados solo dentro de la clase y sus subclases. Se pueden acceder desde fuera de la clase, pero se considera una mala práctica hacerlo.
- **Convención**: Se definen precediendo el nombre del método con un solo guion bajo (`_`).

#### Ejemplo de Métodos Protegidos

```python
class MiClase:
    def __init__(self):
        self._atributo_protegido = "Soy protegido"

    def _metodo_protegido(self):
        return "Este es un método protegido"

class SubClase(MiClase):
    def mostrar(self):
        return self._metodo_protegido()

objeto = SubClase()
print(objeto.mostrar())  # Salida: Este es un método protegido

# Acceder directamente al método protegido es posible, pero no recomendado
print(objeto._metodo_protegido())  # Salida: Este es un método protegido
```

### Resumen

- **Métodos Privados**:
  - Se definen con `__` (doble guion bajo).
  - No son accesibles desde fuera de la clase.
  - Se utilizan para ocultar la implementación.

- **Métodos Protegidos**:
  - Se definen con `_` (un solo guion bajo).
  - Se pueden acceder desde fuera de la clase, pero se desaconseja.
  - Se utilizan para indicar que un método no está destinado a ser usado públicamente.

Estas convenciones son importantes para mantener un diseño claro y seguro en tus clases, permitiendo que los desarrolladores comprendan mejor las intenciones detrás del acceso a ciertos métodos y atributos.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at privateProtectedMethods](https://github.com/platzi/python-avanzado/tree/privateProtectedMethods)

## Gestión avanzada de propiedades

La gestión avanzada de propiedades en Python se refiere a la manipulación de atributos de clase utilizando propiedades (`property`), así como técnicas más avanzadas como la creación de propiedades dinámicas, la validación de datos y la gestión de la encapsulación. Aquí te presento un resumen de cómo se puede hacer esto:

### 1. Uso de `property`

La función `property` permite definir métodos que se pueden utilizar como atributos. Esto es útil para encapsular el acceso a los atributos, permitiendo validaciones o transformaciones al obtener o establecer el valor.

#### Ejemplo básico de `property`

```python
class Persona:
    def __init__(self, nombre):
        self._nombre = nombre  # Atributo protegido

    @property
    def nombre(self):
        return self._nombre

    @nombre.setter
    def nombre(self, nuevo_nombre):
        if not isinstance(nuevo_nombre, str):
            raise ValueError("El nombre debe ser una cadena.")
        self._nombre = nuevo_nombre

persona = Persona("Juan")
print(persona.nombre)  # Salida: Juan

persona.nombre = "Carlos"  # Cambiar el nombre
print(persona.nombre)  # Salida: Carlos

# persona.nombre = 123  # Esto generaría un ValueError
```

### 2. Propiedades dinámicas

Puedes crear propiedades que calculen su valor en función de otros atributos de la clase.

#### Ejemplo de propiedades dinámicas

```python
class Rectangulo:
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto

    @property
    def area(self):
        return self.ancho * self.alto

rectangulo = Rectangulo(5, 10)
print(rectangulo.area)  # Salida: 50
```

### 3. Validación de propiedades

Las propiedades pueden ser utilizadas para validar datos antes de asignar un valor.

#### Ejemplo de validación

```python
class CuentaBancaria:
    def __init__(self, saldo_inicial):
        self._saldo = saldo_inicial

    @property
    def saldo(self):
        return self._saldo

    @saldo.setter
    def saldo(self, nuevo_saldo):
        if nuevo_saldo < 0:
            raise ValueError("El saldo no puede ser negativo.")
        self._saldo = nuevo_saldo

cuenta = CuentaBancaria(1000)
print(cuenta.saldo)  # Salida: 1000

cuenta.saldo = 500  # Cambiar el saldo
print(cuenta.saldo)  # Salida: 500

# cuenta.saldo = -100  # Esto generaría un ValueError
```

### 4. Propiedades computadas

Las propiedades también pueden ser utilizadas para computar valores basados en otros atributos. Esto es útil para mantener el código limpio y evitar duplicaciones.

#### Ejemplo de propiedades computadas

```python
class Circulo:
    def __init__(self, radio):
        self.radio = radio

    @property
    def area(self):
        import math
        return math.pi * (self.radio ** 2)

circulo = Circulo(5)
print(circulo.area)  # Salida: 78.53981633974483
```

### 5. Uso de `@classmethod` y `@staticmethod` con propiedades

Puedes utilizar `@classmethod` y `@staticmethod` para crear propiedades de clase y métodos estáticos, respectivamente.

#### Ejemplo de métodos de clase

```python
class Persona:
    cantidad_personas = 0

    def __init__(self, nombre):
        self.nombre = nombre
        Persona.cantidad_personas += 1

    @classmethod
    def total_personas(cls):
        return cls.cantidad_personas

persona1 = Persona("Juan")
persona2 = Persona("Maria")
print(Persona.total_personas())  # Salida: 2
```

### Resumen

- **Propiedades**: Permiten encapsular el acceso a los atributos de una clase, añadiendo validación y lógica adicional.
- **Propiedades dinámicas**: Se pueden crear basadas en otros atributos, lo que permite una computación perezosa.
- **Validación**: Permite validar datos antes de asignarlos a un atributo.
- **Propiedades computadas**: Facilitan el cálculo de valores basados en otros atributos.

La gestión avanzada de propiedades es esencial para diseñar clases robustas y mantener un buen encapsulamiento y validación.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at property](https://github.com/platzi/python-avanzado/tree/property)

## Métodos estáticos y de clase avanzados

En Python, los métodos estáticos y de clase (`@staticmethod` y `@classmethod`) son herramientas poderosas para gestionar el comportamiento de las clases sin tener que instanciar objetos. Aunque ambos tipos de métodos no actúan sobre una instancia, difieren en su uso y propósito:

### 1. Métodos Estáticos (`@staticmethod`)

Un método estático es una función que pertenece a una clase, pero no puede acceder ni modificar el estado de la clase o de sus instancias (no recibe el argumento `self` ni `cls`). Los métodos estáticos se utilizan cuando la funcionalidad de la función es relevante para la clase, pero no necesita interactuar con los atributos o métodos de clase o de instancia.

#### Uso de `@staticmethod` avanzado

En casos avanzados, los métodos estáticos pueden ser útiles para:
- Realizar cálculos complejos independientes de la clase.
- Encapsular funciones auxiliares que operan sobre datos externos.
- Realizar validaciones de datos que pueden usarse en múltiples métodos de la clase.

```python
class Calculadora:
    @staticmethod
    def sumar(a, b):
        return a + b

    @staticmethod
    def es_numero_par(x):
        return x % 2 == 0

# Uso sin instanciar la clase
print(Calculadora.sumar(10, 5))  # Salida: 15
print(Calculadora.es_numero_par(10))  # Salida: True
```

En este ejemplo, `sumar` y `es_numero_par` no dependen de ningún estado de clase, lo que las convierte en buenos candidatos para ser métodos estáticos.

### 2. Métodos de Clase (`@classmethod`)

Un método de clase recibe el primer argumento `cls`, que representa la propia clase y permite manipular los atributos de la clase o crear nuevas instancias. Los métodos de clase son útiles para trabajar con datos a nivel de clase y pueden ser usados para construir métodos de fábrica, modificar atributos de clase o realizar operaciones que afectan a todas las instancias de la clase.

#### Uso de `@classmethod` avanzado

Los métodos de clase son especialmente útiles en situaciones donde necesitas:
- Crear instancias de la clase desde datos o configuraciones específicas.
- Implementar métodos de fábrica que devuelvan instancias preconfiguradas.
- Acceder y modificar datos a nivel de clase, como contadores de instancias.

##### Ejemplo 1: Método de Fábrica

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    @classmethod
    def desde_cadena(cls, cadena):
        nombre, edad = cadena.split(',')
        return cls(nombre, int(edad))

# Crear una instancia utilizando un método de clase
persona = Persona.desde_cadena("Juan,30")
print(persona.nombre)  # Salida: Juan
print(persona.edad)    # Salida: 30
```

Aquí, el método `desde_cadena` actúa como un método de fábrica, creando instancias a partir de una cadena de texto específica.

##### Ejemplo 2: Contador de Instancias

```python
class Vehiculo:
    contador_instancias = 0

    def __init__(self, tipo):
        self.tipo = tipo
        Vehiculo.incrementar_contador()

    @classmethod
    def incrementar_contador(cls):
        cls.contador_instancias += 1

    @classmethod
    def obtener_total_vehiculos(cls):
        return cls.contador_instancias

# Crear varias instancias
coche = Vehiculo("Coche")
moto = Vehiculo("Moto")

# Ver el total de instancias
print(Vehiculo.obtener_total_vehiculos())  # Salida: 2
```

### Comparación y Cuándo Usar Cada Uno

| Método         | Usar cuando...                                                                                     |
|----------------|----------------------------------------------------------------------------------------------------|
| `@staticmethod`| No necesitas acceder ni modificar la clase ni sus instancias, pero la funcionalidad es relevante.  |
| `@classmethod` | Necesitas manipular el estado de la clase o crear instancias configuradas de forma específica.     |

### Resumen

- **Métodos Estáticos** (`@staticmethod`): Independientes del estado de la clase o instancias. Útiles para validaciones, cálculos y funciones auxiliares.
- **Métodos de Clase** (`@classmethod`): Tienen acceso a la clase y son útiles para modificar el estado de la clase o implementar métodos de fábrica.

Ambos métodos son útiles para escribir código reutilizable y pueden mejorar la organización y la legibilidad de las clases.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at staticClassMethodAvanzado](https://github.com/platzi/python-avanzado/tree/staticClassMethodAvanzado)

## Introducción a la concurrencia y paralelismo

La concurrencia y el paralelismo son conceptos clave en la programación de alto rendimiento y el diseño de aplicaciones modernas, especialmente en entornos donde se necesita manejar múltiples tareas al mismo tiempo para optimizar el uso de recursos. Aunque ambos términos están relacionados, representan enfoques diferentes para ejecutar varias tareas simultáneamente. Vamos a desglosar los conceptos básicos y las diferencias.

### Concurrencia

La **concurrencia** se refiere a la capacidad de un sistema para gestionar múltiples tareas al mismo tiempo, sin necesariamente ejecutarlas simultáneamente. En un entorno concurrente, las tareas pueden progresar en paralelo, aunque no estén ejecutándose al mismo tiempo, ya que el sistema cambia rápidamente entre tareas.

#### Ejemplo de concurrencia

Imagina que estás en una cocina preparando varias recetas: cortas los ingredientes de una receta, luego los de otra, y vas alternando. No completas una receta antes de comenzar con otra, sino que cambias entre ellas para hacer progresos en todas.

- **Multithreading**: Un enfoque común de concurrencia en el que múltiples "hilos" de ejecución se utilizan para dividir las tareas. En Python, se pueden utilizar con la biblioteca `threading` para tareas I/O intensivas (como la lectura y escritura en archivos o las solicitudes de red), debido a las limitaciones del Global Interpreter Lock (GIL).

```python
import threading

def tarea(nombre):
    print(f"Iniciando tarea: {nombre}")

hilo1 = threading.Thread(target=tarea, args=("Tarea 1",))
hilo2 = threading.Thread(target=tarea, args=("Tarea 2",))

hilo1.start()
hilo2.start()
hilo1.join()
hilo2.join()
```

### Paralelismo

El **paralelismo** es la capacidad de ejecutar múltiples tareas al mismo tiempo, aprovechando varios núcleos de CPU. En un entorno verdaderamente paralelo, varias tareas se ejecutan simultáneamente en diferentes núcleos, lo que es ideal para tareas que requieren un uso intensivo de CPU.

#### Ejemplo de paralelismo

Siguiendo el ejemplo de la cocina, en un entorno paralelo, tienes varios chefs en la cocina, y cada uno trabaja en una receta diferente simultáneamente.

- **Multiprocessing**: En Python, la biblioteca `multiprocessing` permite crear varios procesos independientes que pueden ejecutarse en paralelo, sin las limitaciones del GIL. Esto es útil para tareas que necesitan cálculos intensivos, como el procesamiento de grandes volúmenes de datos.

```python
from multiprocessing import Process

def tarea(nombre):
    print(f"Iniciando tarea: {nombre}")

proceso1 = Process(target=tarea, args=("Tarea 1",))
proceso2 = Process(target=tarea, args=("Tarea 2",))

proceso1.start()
proceso2.start()
proceso1.join()
proceso2.join()
```

### Diferencias clave entre concurrencia y paralelismo

| Característica   | Concurrencia                                                     | Paralelismo                                           |
|------------------|-------------------------------------------------------------------|-------------------------------------------------------|
| **Definición**   | Habilidad de manejar múltiples tareas al mismo tiempo.            | Ejecución simultánea de tareas en diferentes núcleos. |
| **Ejemplo**      | `threading` (conmutación rápida entre tareas).                    | `multiprocessing` (tareas ejecutadas en paralelo).    |
| **Aplicaciones** | Ideal para tareas I/O intensivas.                                 | Ideal para tareas CPU intensivas.                     |

### Concurrencia y paralelismo en Python: retos y limitaciones

Python maneja la concurrencia con hilos (`threading`) y el paralelismo con procesos (`multiprocessing`). Sin embargo, el **GIL** limita la ejecución de hilos en Python, lo que hace que el uso de `multiprocessing` sea preferible para tareas intensivas en CPU, mientras que `threading` es útil para tareas de I/O.

### Alternativas: `asyncio` para concurrencia asíncrona

Para manejar concurrencia de manera más eficiente, Python también ofrece `asyncio`, que permite trabajar con tareas asíncronas sin crear múltiples hilos o procesos. Es especialmente útil en aplicaciones de red y permite manejar tareas I/O en paralelo de manera eficiente.

```python
import asyncio

async def tarea(nombre):
    print(f"Iniciando tarea: {nombre}")
    await asyncio.sleep(1)
    print(f"Terminando tarea: {nombre}")

async def main():
    await asyncio.gather(tarea("Tarea 1"), tarea("Tarea 2"))

asyncio.run(main())
```

### Resumen

- **Concurrencia**: Capacidad de manejar múltiples tareas a la vez, alternando entre ellas. Ideal para tareas de I/O intensivas.
- **Paralelismo**: Ejecución simultánea de tareas en múltiples núcleos. Ideal para tareas CPU intensivas.
- **`asyncio`**: Permite manejar tareas asíncronas y concurrentes de manera eficiente sin usar múltiples hilos o procesos.

Cada enfoque tiene sus fortalezas y limitaciones, y la elección depende de la naturaleza de las tareas.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at concurrenciaParalelismo](https://github.com/platzi/python-avanzado/tree/concurrenciaParalelismo)

## Threading y multiprocessing en Python

## Uso de `threading` y `multiprocessing` en Python

![A_rectangular_image_representing_threads_and_multi-processing_in_Python.jpg](images/A_rectangular_image_representing_threads_and_multi-processing_in_Python.jpg)

Imagina que estás trabajando en una aplicación que necesita procesar múltiples tareas al mismo tiempo: desde manejar solicitudes web hasta realizar cálculos complejos de manera simultánea. A medida que las aplicaciones se vuelven más exigentes, las soluciones básicas de concurrencia ya no son suficientes. Aquí es donde entran las herramientas avanzadas de Python como **threading** y **multiprocessing**, que te permiten sacar el máximo provecho de tu CPU y gestionar tareas de manera eficiente y sin errores.

En esta clase, aprenderás a manejar escenarios más complicados, como evitar que los hilos interfieran entre sí, compartir datos de manera segura entre procesos y prevenir bloqueos que puedan detener tu aplicación. Prepárate para llevar la programación concurrente y paralela a un nivel más profesional y resolver problemas que los desarrolladores enfrentan en proyectos del mundo real.

### 1. Sincronización de Hilos en Python

Cuando varios hilos intentan acceder a un mismo recurso al mismo tiempo, pueden ocurrir problemas de coherencia. Para evitar esto, se utilizan mecanismos de sincronización, como Lock y RLock, que garantizan que solo un hilo acceda a un recurso crítico a la vez.

### Ejemplo: Uso de Lock para Evitar Condiciones de Carrera

```python
import threading

# Variable compartida
saldo = 0
lock = threading.Lock()  # Crear un Lock

def depositar(dinero):
    global saldo
    for _ in range(100000):
        with lock:  # Bloquear el acceso para evitar condiciones de carrera
            saldo += dinero

hilos = []
for _ in range(2):
    hilo = threading.Thread(target=depositar, args=(1,))
    hilos.append(hilo)
    hilo.start()

for hilo in hilos:
    hilo.join()

print(f"Saldo final: {saldo}")  # Esperamos ver 200000 como saldo
```

**Explicación:**

- El uso de Lock asegura que solo un hilo modifique la variable saldo en un momento dado, evitando que el resultado final sea incorrecto.

### 2. Compartir Datos entre Procesos con multiprocessing

A diferencia de los hilos, los procesos no comparten memoria de forma predeterminada. Para que dos procesos puedan compartir datos, Python proporciona herramientas como **multiprocessing.Queue** y **multiprocessing.Value**.

**Ejemplo: Compartir Datos con Queue en multiprocessing**

```python
import multiprocessing

def calcular_cuadrado(numeros, cola):
    for n in numeros:
        cola.put(n * n)

if __name__ == "__main__":
    numeros = [1, 2, 3, 4, 5]
    cola = multiprocessing.Queue()

    proceso = multiprocessing.Process(target=calcular_cuadrado, args=(numeros, cola))
    proceso.start()
    proceso.join()

    # Extraer resultados de la cola
    while not cola.empty():
        print(cola.get())
```

**Explicación:**

- Usamos Queue para que el proceso secundario pueda pasar datos de vuelta al proceso principal.

### 3. Problemas de Sincronización y Cómo Evitarlos

A medida que manejas tareas más complejas, es posible que te encuentres con problemas como deadlocks y race conditions. Entender estos problemas es crucial para escribir código concurrente robusto.

**Evitar Deadlocks con RLock**

Un deadlock ocurre cuando dos o más hilos se bloquean mutuamente al esperar por un recurso que está siendo utilizado por otro hilo. Para evitar esto, podemos usar RLock en lugar de Lock.

**Ejemplo: Uso de RLock para Evitar Deadlocks**

```python
import threading

class CuentaBancaria:
    def __init__(self, saldo):
        self.saldo = saldo
        self.lock = threading.RLock()

    def transferir(self, otra_cuenta, cantidad):
        with self.lock:
            self.saldo -= cantidad
            otra_cuenta.depositar(cantidad)

    def depositar(self, cantidad):
        with self.lock:
            self.saldo += cantidad

cuenta1 = CuentaBancaria(500)
cuenta2 = CuentaBancaria(300)

hilo1 = threading.Thread(target=cuenta1.transferir, args=(cuenta2, 200))
hilo2 = threading.Thread(target=cuenta2.transferir, args=(cuenta1, 100))

hilo1.start()
hilo2.start()

hilo1.join()
hilo2.join()

print(f"Saldo cuenta1: {cuenta1.saldo}")
print(f"Saldo cuenta2: {cuenta2.saldo}")
```

**Explicación:**

- Usamos RLock para evitar que múltiples operaciones simultáneas en una cuenta causen bloqueos.

### 4. Coordinación de Tareas con multiprocessing.Manager

Cuando los procesos deben compartir estructuras de datos complejas (como listas o diccionarios), podemos usar un Manager para crear un espacio de memoria compartido entre procesos.

**Ejemplo: Uso de Manager para Compartir Listas entre Procesos**

```python
import multiprocessing

def agregar_valores(lista_compartida):
    for i in range(5):
        lista_compartida.append(i)

if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        lista_compartida = manager.list()

        proceso1 = multiprocessing.Process(target=agregar_valores, args=(lista_compartida,))
        proceso2 = multiprocessing.Process(target=agregar_valores, args=(lista_compartida,))

        proceso1.start()
        proceso2.start()

        proceso1.join()
        proceso2.join()

        print(f"Lista compartida: {lista_compartida}")
```

**Explicación:**

- multiprocessing.Manager nos permite crear una lista compartida entre varios procesos, facilitando la comunicación entre ellos.

¡Lo lograste! Ahora tienes en tus manos poderosas técnicas para manejar múltiples tareas de forma eficiente. Aprendiste a sincronizar hilos para evitar errores, a compartir datos de manera segura entre procesos y a evitar bloqueos que podrían detener tus aplicaciones. Todo esto te prepara para enfrentar los desafíos del desarrollo de software moderno, donde la concurrencia y el paralelismo son esenciales para crear aplicaciones rápidas, eficientes y escalables.

Con estas herramientas avanzadas, tu código no solo será más rápido, sino también más robusto y confiable. Este es el tipo de conocimiento que te permite destacar en proyectos grandes y complejos. ¡Estás listo para aplicar todo lo que has aprendido y optimizar tus próximas creaciones en Python!

En Python, las bibliotecas `threading` y `multiprocessing` permiten ejecutar múltiples tareas en paralelo, pero tienen diferencias importantes. **`threading`** se usa principalmente para tareas de entrada/salida (I/O), como redes o acceso a archivos, debido a la limitación del **Global Interpreter Lock (GIL)**. **`multiprocessing`**, por otro lado, permite ejecutar código en múltiples procesos y es más adecuado para tareas que consumen mucha CPU, ya que evita el GIL.

A continuación, detallo cada biblioteca y su uso.

### 1. `threading`: Concurrencia a través de hilos

La biblioteca `threading` permite la creación y manejo de hilos en Python. Los hilos comparten el mismo espacio de memoria y recursos, lo que facilita la comunicación entre ellos, pero limita su uso en tareas intensivas en CPU debido al GIL.

#### Uso básico de `threading`

```python
import threading

def tarea(nombre):
    print(f"Iniciando {nombre}")
    # Simulación de trabajo
    for i in range(3):
        print(f"{nombre} ejecutando {i}")
    print(f"Terminando {nombre}")

# Crear hilos
hilo1 = threading.Thread(target=tarea, args=("Hilo 1",))
hilo2 = threading.Thread(target=tarea, args=("Hilo 2",))

# Iniciar hilos
hilo1.start()
hilo2.start()

# Esperar a que terminen
hilo1.join()
hilo2.join()
```

#### Ventajas y Limitaciones de `threading`

- **Ventajas**: Ideal para tareas de entrada/salida que pueden esperar (ej., operaciones de red, archivos).
- **Limitaciones**: Debido al GIL, solo un hilo puede ejecutar bytecode de Python a la vez, lo que limita la utilidad de `threading` para operaciones que consumen mucha CPU.

### 2. `multiprocessing`: Paralelismo con múltiples procesos

La biblioteca `multiprocessing` permite la ejecución de tareas en múltiples procesos, cada uno con su propio intérprete de Python, evitando el GIL. Esto es útil para tareas que requieren cálculos intensivos, como el procesamiento de datos.

#### Uso básico de `multiprocessing`

```python
from multiprocessing import Process

def tarea(nombre):
    print(f"Iniciando {nombre}")
    # Simulación de trabajo
    for i in range(3):
        print(f"{nombre} ejecutando {i}")
    print(f"Terminando {nombre}")

# Crear procesos
proceso1 = Process(target=tarea, args=("Proceso 1",))
proceso2 = Process(target=tarea, args=("Proceso 2",))

# Iniciar procesos
proceso1.start()
proceso2.start()

# Esperar a que terminen
proceso1.join()
proceso2.join()
```

#### Ventajas y Limitaciones de `multiprocessing`

- **Ventajas**: Permite un verdadero paralelismo, útil para tareas CPU intensivas. Cada proceso tiene su propio espacio de memoria y no está afectado por el GIL.
- **Limitaciones**: Cada proceso consume más memoria y tiene más sobrecarga en la comunicación entre procesos que los hilos.

### Comunicación entre Hilos y Procesos

#### `threading`: Comunicación a través de variables compartidas

En `threading`, los hilos pueden compartir variables y recursos de la misma clase, ya que todos operan en el mismo espacio de memoria. Es importante utilizar **bloqueos (locks)** para evitar problemas de sincronización.

```python
import threading

contador = 0
bloqueo = threading.Lock()

def incrementar():
    global contador
    for _ in range(1000):
        with bloqueo:  # Bloqueo para evitar condiciones de carrera
            contador += 1

# Crear hilos
hilos = [threading.Thread(target=incrementar) for _ in range(5)]

# Iniciar hilos
for hilo in hilos:
    hilo.start()

# Esperar a que terminen
for hilo in hilos:
    hilo.join()

print(f"Contador final: {contador}")
```

#### `multiprocessing`: Comunicación a través de colas y pipes

Dado que los procesos no comparten memoria, `multiprocessing` proporciona colas (`Queue`) y pipes (`Pipe`) para la comunicación.

```python
from multiprocessing import Process, Queue

def productor(q):
    for i in range(5):
        q.put(i)  # Añadir elementos a la cola
        print(f"Producto {i} añadido a la cola")

def consumidor(q):
    while not q.empty():
        item = q.get()  # Obtener elementos de la cola
        print(f"Producto {item} consumido")

cola = Queue()

# Crear procesos
p1 = Process(target=productor, args=(cola,))
p2 = Process(target=consumidor, args=(cola,))

# Iniciar procesos
p1.start()
p2.start()

# Esperar a que terminen
p1.join()
p2.join()
```

### Cuándo usar `threading` vs `multiprocessing`

| **Situación**                               | **Biblioteca recomendada** |
|---------------------------------------------|----------------------------|
| Tareas I/O intensivas (red, lectura/escritura)| `threading`                |
| Tareas CPU intensivas (procesamiento de datos)| `multiprocessing`          |
| Necesidad de comunicación sencilla entre tareas | `threading` con variables compartidas y locks |
| Necesidad de aislamiento de datos              | `multiprocessing` con `Queue` o `Pipe`         |

### Resumen

- **`threading`** es ideal para tareas concurrentes y basadas en I/O donde el GIL no es un problema.
- **`multiprocessing`** permite el verdadero paralelismo y es mejor para tareas intensivas en CPU.
- La **comunicación** entre hilos se hace mediante variables compartidas y bloqueos, mientras que en `multiprocessing` se utilizan `Queue` y `Pipe` para compartir datos entre procesos.

Ambas bibliotecas son útiles para optimizar el rendimiento en Python, pero elegir la correcta depende del tipo de tarea y del diseño de la aplicación.

## Asincronismo con asyncio

El asincronismo en Python permite ejecutar tareas concurrentes sin bloquear el flujo de ejecución del programa. La biblioteca **`asyncio`** de Python facilita la creación y gestión de tareas asíncronas, lo que es ideal para aplicaciones que necesitan manejar múltiples operaciones de entrada y salida (I/O) de manera eficiente, como las solicitudes de red, el acceso a bases de datos y la lectura de archivos.

### Fundamentos del Asincronismo con `asyncio`

- **Asincronismo**: Consiste en realizar tareas de manera concurrente, de modo que una tarea pueda comenzar mientras otra está en pausa, sin detener el flujo general del programa.
- **`async` y `await`**: Estas palabras clave son esenciales en `asyncio`. `async` se usa para definir una función asíncrona (`async def`), mientras que `await` se usa dentro de estas funciones para pausar su ejecución hasta que se complete una operación asíncrona.
- **Corutinas**: Las funciones asíncronas en Python son "corutinas", que pueden ser pausadas y reanudadas. Esto permite que otras corutinas se ejecuten mientras una está esperando, optimizando el rendimiento.

### Ejemplo Básico con `asyncio`

Supongamos que tenemos una tarea simple que se ejecuta durante algunos segundos, como simulación de una tarea de red o una solicitud a una API.

```python
import asyncio

async def tarea(nombre, duracion):
    print(f"Iniciando {nombre}")
    await asyncio.sleep(duracion)  # Simula una tarea que toma tiempo
    print(f"Terminando {nombre} después de {duracion} segundos")

# Ejecutar varias tareas
async def main():
    await asyncio.gather(
        tarea("Tarea 1", 2),
        tarea("Tarea 2", 3),
        tarea("Tarea 3", 1)
    )

# Ejecutar la función principal
asyncio.run(main())
```

### Explicación

- `async def tarea(nombre, duracion)`: Define una función asíncrona que simula el trabajo con `asyncio.sleep(duracion)`.
- `await asyncio.gather(...)`: Ejecuta varias tareas de forma concurrente. `gather` acepta múltiples corutinas y las ejecuta al mismo tiempo, permitiendo que las tareas esperen sin bloquear el flujo general del programa.

### Uso de `asyncio.gather` y `asyncio.sleep`

- **`asyncio.gather`**: Permite agrupar varias corutinas para ejecutarlas al mismo tiempo. Esto es útil cuando tienes varias tareas independientes que deseas que se completen en paralelo.
- **`asyncio.sleep`**: Simula una operación de espera asíncrona, como el tiempo que toma una solicitud a una API o la lectura de un archivo.

### Ejemplo Realista: Descarga Simultánea de Páginas Web

Imaginemos un programa que descarga contenido de varias URLs. Podemos simular el proceso de espera de las respuestas de una API con `asyncio.sleep`.

```python
import asyncio

async def descargar_pagina(url):
    print(f"Descargando {url}")
    await asyncio.sleep(2)  # Simula la espera de respuesta de la red
    print(f"Descarga completa: {url}")
    return f"Contenido de {url}"

async def main():
    urls = ["https://ejemplo.com/1", "https://ejemplo.com/2", "https://ejemplo.com/3"]
    
    # Ejecutar todas las descargas en paralelo
    resultados = await asyncio.gather(*(descargar_pagina(url) for url in urls))
    
    print("Resultados de las descargas:")
    for resultado in resultados:
        print(resultado)

asyncio.run(main())
```

### Explicación

- **Lista de URLs**: Simulamos una lista de URLs que queremos "descargar".
- **`await asyncio.gather(...)`**: Llama a `descargar_pagina` para cada URL en paralelo, esperando los resultados de todas las descargas antes de continuar.

### Ejecución Secuencial vs. Asíncrona

Si las descargas se ejecutaran de forma secuencial, cada una tendría que esperar a que la anterior termine. Con `asyncio.gather`, se ejecutan en paralelo, y el tiempo total es aproximadamente el de la tarea más larga.

### Buenas Prácticas en `asyncio`

1. **Usa `await` correctamente**: Solo se puede utilizar `await` dentro de funciones definidas con `async def`. Usar `await` permite pausar la corutina y ceder el control para que otras corutinas avancen.
2. **`asyncio.run(main())`**: Es la forma recomendada para ejecutar el bucle de eventos asíncrono en el programa principal.
3. **Evita el bloqueo**: Las funciones sin `await`, como operaciones de CPU o funciones de espera síncronas, bloquearán el flujo de ejecución asíncrono.

### Ventajas del Asincronismo con `asyncio`

- **Rendimiento**: Permite manejar miles de operaciones I/O sin necesidad de crear múltiples hilos o procesos.
- **Escalabilidad**: Ideal para aplicaciones que necesitan manejar muchas conexiones simultáneamente, como servidores web.
- **Eficiencia**: Reduce el consumo de memoria y CPU al evitar el uso de múltiples procesos o hilos.

### Resumen

- **Asincronismo** con `asyncio` permite que las tareas de entrada y salida se ejecuten sin bloquear el flujo.
- **Corutinas** (`async def`) y **await** son esenciales para ejecutar código de forma asíncrona.
- **`asyncio.gather`** ejecuta varias tareas en paralelo.
- `asyncio` es ideal para tareas de I/O intensivas y es la solución recomendada para aplicaciones que necesitan alto rendimiento en operaciones concurrentes de red o archivo. 

Este enfoque permite crear aplicaciones eficientes y escalables en entornos de alto rendimiento.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at asincronismo](https://github.com/platzi/python-avanzado/tree/asincronismo)

## Creación de módulos en Python

Crear módulos en Python es una manera de organizar y reutilizar el código, facilitando la mantenibilidad y claridad del proyecto. Un **módulo** es un archivo `.py` que contiene funciones, clases o variables que se pueden importar en otros archivos o módulos de un proyecto.

### 1. Creación de un Módulo Básico

Imaginemos que queremos crear un módulo llamado `matematica.py` que contenga funciones matemáticas personalizadas.

**Paso 1:** Crear el archivo del módulo.

En el mismo directorio del proyecto, crea un archivo llamado `matematica.py` y añade funciones en él. Por ejemplo:

```python
# matematica.py

def sumar(a, b):
    return a + b

def restar(a, b):
    return a - b

def multiplicar(a, b):
    return a * b

def dividir(a, b):
    if b == 0:
        raise ValueError("No se puede dividir por cero.")
    return a / b
```

**Paso 2:** Importar el módulo en otro archivo.

En un archivo diferente, como `main.py`, importa el módulo y utiliza sus funciones:

```python
# main.py

import matematica

resultado_suma = matematica.sumar(10, 5)
resultado_resta = matematica.restar(10, 5)

print("Suma:", resultado_suma)
print("Resta:", resultado_resta)
```

### 2. Importación Específica y Alias

Puedes importar funciones específicas o dar alias a las funciones o módulos para hacer el código más legible.

```python
# Importación específica
from matematica import sumar, dividir

print(sumar(3, 2))
print(dividir(10, 2))

# Alias para un módulo o función
import matematica as mat

print(mat.multiplicar(3, 2))
```

### 3. Uso de `__name__ == "__main__"`

El bloque `if __name__ == "__main__":` permite ejecutar pruebas o código dentro del propio módulo sin que se ejecute cuando el módulo es importado en otro archivo.

En `matematica.py`:

```python
# matematica.py

def sumar(a, b):
    return a + b

# Código de prueba
if __name__ == "__main__":
    print(sumar(5, 7))  # Esto solo se ejecutará si ejecutas matematica.py directamente
```

### 4. Creación de Paquetes

Un **paquete** es una colección de módulos organizados en una estructura de carpetas. Para crear un paquete, crea una carpeta con un archivo `__init__.py` y coloca módulos en ella. Esto permite importar submódulos.

Estructura de ejemplo de un paquete llamado `mi_paquete`:

```
mi_paquete/
    __init__.py
    matematica.py
    geometria.py
```

En `matematica.py` puedes colocar las funciones matemáticas, y en `geometria.py` funciones relacionadas con la geometría.

**Uso del paquete en un archivo principal:**

```python
# Importa el paquete y sus módulos
from mi_paquete import matematica, geometria

print(matematica.sumar(3, 2))
print(geometria.area_cuadrado(4))
```

### 5. Ejemplo Completo con el Paquete

#### Estructura de Archivos

```
mi_paquete/
    __init__.py
    matematica.py
    geometria.py
main.py
```

#### Contenido de `__init__.py`

```python
# mi_paquete/__init__.py

from .matematica import sumar, restar
from .geometria import area_cuadrado
```

#### Contenido de `geometria.py`

```python
# mi_paquete/geometria.py

def area_cuadrado(lado):
    return lado * lado
```

#### `main.py`

```python
# main.py

from mi_paquete import sumar, area_cuadrado

print("Suma:", sumar(3, 5))
print("Área del cuadrado:", area_cuadrado(4))
```

### Resumen

1. **Módulo**: Archivo `.py` con funciones, clases o variables reutilizables.
2. **Paquete**: Carpeta con un archivo `__init__.py` que organiza múltiples módulos.
3. **Importación específica y alias**: Personaliza la importación de módulos o funciones.
4. **Pruebas con `__name__ == "__main__"`**: Permite probar el código en el módulo mismo sin que se ejecute al importarse.

Estos conceptos son fundamentales para crear código bien organizado y fácil de mantener en Python.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at modulos](https://github.com/platzi/python-avanzado/tree/modulos)

## Gestión de paquetes

La **gestión de paquetes** en Python permite organizar, reutilizar y distribuir módulos o grupos de módulos de manera eficiente. Los **paquetes** son carpetas con módulos (archivos `.py`) y un archivo especial `__init__.py`, que convierte la carpeta en un paquete reconocible por Python. Esta organización es clave para proyectos de gran tamaño, donde diferentes funcionalidades se agrupan en paquetes separados, manteniendo el código limpio y modular.

### ¿Qué es un Paquete?

Un paquete en Python es una colección de módulos organizados en una estructura de carpetas, donde cada módulo puede contener funciones, clases y variables relacionadas. Esta estructura modular facilita:

1. **Organización del código** en componentes lógicos.
2. **Reutilización y distribución** del código en otros proyectos.
3. **Escalabilidad** en proyectos complejos, separando funcionalidades en varios paquetes.

### Estructura Básica de un Paquete

Un paquete debe incluir un archivo `__init__.py` en su carpeta principal para que Python reconozca la carpeta como un paquete importable. Aunque `__init__.py` puede estar vacío, generalmente se utiliza para importar submódulos o funciones comunes.

Ejemplo de una estructura de paquete:

```
mi_proyecto/
│
├── mi_paquete/
│   ├── __init__.py
│   ├── operaciones_matematicas.py
│   └── operaciones_texto.py
└── main.py
```

### Ejemplo de Paquete

#### Paso 1: Crear los Módulos en el Paquete

Dentro de `mi_paquete`, crea dos archivos:

- **`operaciones_matematicas.py`**: Contiene funciones matemáticas.
- **`operaciones_texto.py`**: Contiene funciones para manipular texto.

**Contenido de `operaciones_matematicas.py`:**

```python
# mi_paquete/operaciones_matematicas.py

def sumar(a, b):
    return a + b

def restar(a, b):
    return a - b
```

**Contenido de `operaciones_texto.py`:**

```python
# mi_paquete/operaciones_texto.py

def contar_palabras(texto):
    return len(texto.split())

def a_mayusculas(texto):
    return texto.upper()
```

#### Paso 2: Configurar `__init__.py`

El archivo `__init__.py` dentro de `mi_paquete` puede usarse para controlar qué funciones o módulos se exponen al importar el paquete. 

**Contenido de `__init__.py`:**

```python
# mi_paquete/__init__.py

from .operaciones_matematicas import sumar, restar
from .operaciones_texto import contar_palabras, a_mayusculas
```

### Paso 3: Usar el Paquete en `main.py`

Crea el archivo `main.py` fuera de la carpeta del paquete. Aquí puedes importar y usar las funciones definidas en `mi_paquete`.

**Contenido de `main.py`:**

```python
# main.py

from mi_paquete import sumar, restar, contar_palabras, a_mayusculas

print("Suma:", sumar(5, 3))
print("Resta:", restar(10, 4))
print("Número de palabras:", contar_palabras("Este es un ejemplo"))
print("Texto en mayúsculas:", a_mayusculas("texto en minúsculas"))
```

### Ejecución del Ejemplo

Para ejecutar el ejemplo:

1. Asegúrate de estar en el directorio donde se encuentra `main.py`.
2. Ejecuta `main.py` con:

   ```bash
   python main.py
   ```

La salida debería ser algo similar a:

```
Suma: 8
Resta: 6
Número de palabras: 4
Texto en mayúsculas: TEXTO EN MINÚSCULAS
```

### Gestión de Paquetes con `pip`

Si deseas compartir o distribuir el paquete, puedes crear un archivo `setup.py`, que facilita la instalación del paquete mediante `pip`. Esto es útil para distribuir el paquete a otros usuarios o para subirlo a PyPI (Python Package Index).

**Ejemplo de `setup.py`:**

```python
# setup.py

from setuptools import setup, find_packages

setup(
    name='mi_paquete',
    version='0.1',
    packages=find_packages(),
    description='Un paquete de ejemplo para operaciones matemáticas y de texto',
    author='Tu Nombre',
    author_email='tu_email@example.com',
)
```

Para instalar el paquete en tu entorno local:

```bash
pip install .
```

### Resumen

1. **Estructura de Paquete**: Organiza el proyecto en carpetas y módulos con un archivo `__init__.py`.
2. **Modularidad y Reutilización**: Puedes reutilizar módulos y funciones en otros proyectos.
3. **Distribución**: Usa `setup.py` para facilitar la instalación y compartir el paquete.

Crear paquetes en Python es fundamental para desarrollar aplicaciones escalables, modulares y reutilizables, permitiendo que el código sea mantenible y fácil de distribuir.

**Lecturas recomendadas**

[GitHub - platzi/python-avanzado at paquetes](https://github.com/platzi/python-avanzado/tree/paquetes)

## Publicación de paquetes en PyPI

Publicar un paquete en el **Python Package Index (PyPI)** permite que otros usuarios puedan instalarlo y utilizarlo fácilmente a través de `pip`. Aquí tienes una guía paso a paso para hacerlo.

### Paso 1: Preparar el Entorno y el Código

1. **Organiza tu proyecto**: Asegúrate de que tu paquete tiene la estructura correcta y un archivo `__init__.py` en cada carpeta de paquetes o subpaquetes.
   
2. **Crea `setup.py`**: Este archivo contiene la configuración para construir e instalar el paquete.

3. **Crea `README.md`** (opcional pero recomendado): Este archivo sirve como descripción detallada del paquete y se muestra en la página de PyPI.

### Paso 2: Crear el Archivo `setup.py`

En la raíz de tu proyecto, crea `setup.py` con la configuración básica del paquete.

Ejemplo de `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name='mi_paquete',  # Nombre único en PyPI
    version='0.1.0',    # Versión inicial
    packages=find_packages(),  # Encuentra automáticamente submódulos y subpaquetes
    description='Un paquete de ejemplo para operaciones matemáticas y de texto',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tu Nombre',
    author_email='tu_email@example.com',
    url='https://github.com/tu_usuario/mi_paquete',  # URL del repositorio
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
```

### Paso 3: Crear el Archivo `pyproject.toml` (opcional)

Para definir cómo construir el paquete, usa un archivo `pyproject.toml`, aunque no siempre es necesario. Este archivo define las dependencias de construcción, si tu proyecto tiene configuraciones específicas.

Ejemplo de `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
```

### Paso 4: Crear un Entorno Virtual y Empaquetar el Proyecto

1. **Instala `setuptools` y `wheel`** en tu entorno virtual:

   ```bash
   pip install setuptools wheel
   ```

2. **Empaqueta el proyecto**: Esto creará los archivos necesarios en la carpeta `dist/`.

   ```bash
   python setup.py sdist bdist_wheel
   ```

   Esto generará un archivo `.tar.gz` y un archivo `.whl` dentro de la carpeta `dist/`, que son los archivos de distribución del paquete.

### Paso 5: Crear una Cuenta en PyPI

1. **Regístrate en PyPI**: Si aún no tienes cuenta, ve a [https://pypi.org/account/register/](https://pypi.org/account/register/) y crea una.

2. **Configura la autenticación**: Una vez registrado, guarda tu nombre de usuario y contraseña, o crea un **token de API** para autenticación.

### Paso 6: Subir el Paquete a PyPI

1. **Instala `twine`**: Es la herramienta para subir paquetes a PyPI.

   ```bash
   pip install twine
   ```

2. **Sube el paquete a PyPI**:

   ```bash
   twine upload dist/*
   ```

3. **Iniciar sesión**: Twine te pedirá tus credenciales de PyPI, o puedes usar un token de API.

### Paso 7: Verificar la Publicación

1. Visita tu página de PyPI en [https://pypi.org/project/mi_paquete](https://pypi.org/project/mi_paquete) (reemplaza `mi_paquete` con el nombre de tu paquete).
2. Comprueba que la descripción, las dependencias y los archivos se hayan subido correctamente.

### Paso 8: Instalar el Paquete desde PyPI

Para probar el paquete recién publicado, instálalo con `pip`:

```bash
pip install mi_paquete
```

### Ejemplo Completo de la Estructura Final

Estructura típica de un proyecto que se subirá a PyPI:

```
mi_paquete/
│
├── mi_paquete/              # Paquete principal
│   ├── __init__.py
│   ├── operaciones_matematicas.py
│   └── operaciones_texto.py
│
├── README.md                # Descripción del proyecto (Markdown)
├── setup.py                 # Configuración de empaquetado
└── pyproject.toml           # (opcional) Configuración de compilación
```

### Consejos para Publicaciones en PyPI

1. **Incrementa la versión**: Cada vez que subas una actualización, incrementa la versión en `setup.py`.
2. **Buena documentación**: Un `README.md` claro y detallado ayudará a los usuarios a entender cómo usar el paquete.
3. **Pruebas**: Asegúrate de probar el paquete localmente antes de publicarlo.

Este flujo garantiza que el paquete esté listo y sea accesible desde PyPI, permitiendo que otros usuarios puedan instalarlo y utilizarlo de manera sencilla.

**Lecturas recomendadas**

[PyPI · The Python Package Index](https://pypi.org/)

## Implementación de un Sistema Completo

En esta última clase del curso, vamos a aplicar todos los conceptos aprendidos en las clases anteriores para desarrollar un sistema completo utilizando Python avanzado. El proyecto consistirá en la implementación de un **sistema de gestión de reservas para un hotel**. Este sistema gestionará:

1. **Reservas**: Creación y cancelación de reservas de habitaciones.
2. **Clientes**: Almacenamiento y gestión de la información de los clientes.
3. **Habitaciones**: Verificación de la disponibilidad de las habitaciones.
4. **Pagos**: Procesamiento de pagos de las reservas de forma asincrónica.

### Objetivos:

1. Integrar los módulos y paquetes del sistema para que cada funcionalidad esté organizada de manera eficiente.
2. Aplicar la programación asincrónica (asyncio) para manejar pagos concurrentes sin bloquear el sistema.
3. Utilizar las mejores prácticas Pythonicas, como las recomendaciones de PEP 8, manejo de tipos y validaciones.
4. Construir un sistema modular y reutilizable mediante la creación de un paquete Python.

### Requisitos:

1. Organizar el código en diferentes módulos y paquetes que gestionen las diferentes partes del sistema.
2. Aplicar programación concurrente y asincrónica para procesar múltiples reservas de manera eficiente.
3. Implementar validaciones básicas para asegurar que las reservas y los pagos sean gestionados correctamente.
4. El sistema debe ser capaz de agregar clientes, verificar la disponibilidad de habitaciones, gestionar reservas y procesar pagos de manera eficaz.

Este proyecto te ayudará a consolidar los conocimientos adquiridos durante el curso y será un ejemplo de cómo utilizar técnicas avanzadas en un entorno real para construir una aplicación completa y robusta en Python.

1. Estructura del Proyecto

Vamos a dividir nuestro sistema en diferentes módulos y paquetes para organizar el código de manera eficiente:

**Paquetes del proyecto:**

```python
hotel_management/
    __init__.py
    reservations.py
    customers.py
    rooms.py
    payments.py
```

1. `reservations.py`: Maneja la creación y cancelación de reservas.
2. `customers.py`: Gestiona la información de los clientes.
3. `rooms.py`: Gestiona la disponibilidad y características de las habitaciones.
4. `payments.py`: Procesa los pagos de las reservas.

### 2. Implementación de Módulos

**2.1. Módulo `reservations.py`**

Este módulo gestionará la lógica relacionada con la creación y cancelación de reservas.

```python
from collections import defaultdict
from datetime import datetime

class Reservation:
    def __init__(self, reservation_id, customer_name, room_number, check_in, check_out):
        self.reservation_id = reservation_id
        self.customer_name = customer_name
        self.room_number = room_number
        self.check_in = check_in
        self.check_out = check_out

class ReservationSystem:
    def __init__(self):
        # Utilizamos defaultdict para gestionar las reservas
        self.reservations = defaultdict(list)

    def add_reservation(self, reservation):
        """Agrega una nueva reserva al sistema."""
        self.reservations[reservation.room_number].append(reservation)
        print(f"Reserva creada para {reservation.customer_name} en la habitación {reservation.room_number}")

    def cancel_reservation(self, reservation_id):
        """Cancela una reserva existente por ID."""
        for room, reservations in self.reservations.items():
            for r in reservations:
                if r.reservation_id == reservation_id:
                    reservations.remove(r)
                    print(f"Reserva {reservation_id} cancelada")
                    return
        print("Reserva no encontrada")
```

**2.2. Módulo `customers.py`**

Este módulo gestionará la información de los clientes.

```python
class Customer:
    def __init__(self, customer_id, name, email):
        self.customer_id = customer_id
        self.name = name
        self.email = email

class CustomerManagement:
    def __init__(self):
        self.customers = {}

    def add_customer(self, customer):
        """Agrega un nuevo cliente al sistema."""
        self.customers[customer.customer_id] = customer
        print(f"Cliente {customer.name} agregado.")

    def get_customer(self, customer_id):
        """Obtiene la información de un cliente por ID."""
        return self.customers.get(customer_id, "Cliente no encontrado.")
```

**2.3. Módulo `rooms.py`**

Este módulo gestionará las habitaciones disponibles en el hotel.

```python
class Room:
    def __init__(self, room_number, room_type, price):
        self.room_number = room_number
        self.room_type = room_type
        self.price = price
        self.available = True

class RoomManagement:
    def __init__(self):
        self.rooms = {}

    def add_room(self, room):
        """Agrega una nueva habitación al sistema."""
        self.rooms[room.room_number] = room
        print(f"Habitación {room.room_number} agregada.")

    def check_availability(self, room_number):
        """Verifica si una habitación está disponible."""
        room = self.rooms.get(room_number)
        if room and room.available:
            print(f"Habitación {room_number} está disponible.")
            return True
        print(f"Habitación {room_number} no está disponible.")
        return False
```

2.4. Módulo payments.py
Este módulo procesará los pagos utilizando asincronismo con asyncio.

import asyncio
import random

async def process_payment(customer_name, amount):
    """Simula el procesamiento de un pago."""
    print(f"Procesando pago de {customer_name} por ${amount}...")
    await asyncio.sleep(random.randint(1, 3))  # Simula una operación de pago
    print(f"Pago de ${amount} completado para {customer_name}")
    return True

### 3. Implementación del Sistema Completo

En el archivo `main.py`, vamos a integrar los módulos y utilizar programación concurrente y asincrónica para procesar varias reservas y pagos al mismo tiempo.

```python
import asyncio
from hotel_management.reservations import Reservation, ReservationSystem
from hotel_management.customers import Customer, CustomerManagement
from hotel_management.rooms import Room, RoomManagement
from hotel_management.payments import process_payment
from hotel_management.datetime import datetime

async def main():
    # Inicializar sistemas
    reservation_system = ReservationSystem()
    customer_mgmt = CustomerManagement()
    room_mgmt = RoomManagement()

    # Crear habitaciones
    room_mgmt.add_room(Room(101, "Single", 100))
    room_mgmt.add_room(Room(102, "Double", 150))

    # Agregar clientes
    customer1 = Customer(1, "Alice", "alice@example.com")
    customer_mgmt.add_customer(customer1)

    customer2 = Customer(2, "Bob", "bob@example.com")
    customer_mgmt.add_customer(customer2)

    # Verificar disponibilidad de habitaciones
    if room_mgmt.check_availability(101):
        reservation = Reservation(1, "Alice", 101, datetime.now(), datetime.now())
        reservation_system.add_reservation(reservation)

        # Procesar pago asincrónicamente
        await process_payment("Alice", 100)

    if room_mgmt.check_availability(102):
        reservation = Reservation(2, "Bob", 102, datetime.now(), datetime.now())
        reservation_system.add_reservation(reservation)

        # Procesar pago asincrónicamente
        await process_payment("Bob", 150)

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. Explicación del Proyecto

En este proyecto, aplicamos los conceptos de las clases anteriores:

- **Módulos y paquetes**: Creamos un sistema modular donde cada parte (reservas, clientes, habitaciones, pagos) está bien organizada en su propio módulo.
- **Validaciones**: A través de la lógica implementada en la gestión de habitaciones y reservas.
- **Asincronismo (asyncio)**: Usamos asyncio para procesar pagos de manera concurrente sin bloquear el sistema.
- **Decoradores y métodos estáticos**: Aunque no se usan directamente aquí, podrían aplicarse para funcionalidades específicas como verificaciones adicionales o cálculo de descuentos.

### 5. Conclusión

Este proyecto final integra las diferentes áreas cubiertas en el curso, demostrando cómo las técnicas avanzadas de Python pueden combinarse para construir un sistema robusto y eficiente.