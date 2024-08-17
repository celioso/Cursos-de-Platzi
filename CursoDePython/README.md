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

