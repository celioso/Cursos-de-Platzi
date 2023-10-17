# Curso Práctico de Python: Creación de un CRUD

Python es un lenguaje de programación creado por Guido Van Rossum, con una sintaxis muy limpia, ideado para enseñar a la gente a programar bien. Se trata de un lenguaje interpretado o de script.

## Ventaja de python

- Legible: sintaxis intuitiva y estricta.
- Productivo: ahorra mucho código.
- Portable: para todo sistema operativo.
- Recargado: viene con muchas librerías por defecto.
Editor recomendado: Atom o Sublime Text.

Instalación de python
Existen dos versiones de Python que tienen gran uso actualmente, Python 2.x y Python 3.x, para este curso necesitas usar una versión 3.x

Para instalar Python solo debes seguir los pasos dependiendo del sistema operativo que tengas instalado.

### Windows
Para instalar Python en Windows ve al sitio https://www.python.org/downloads/ y presiona sobre el botón Download Python 3.7.3

Se descargará un archivo de instalación con el nombre python-3.7.3.exe , ejecútalo. Y sigue los pasos de instalación.

Al finalizar la instalación haz lo siguiente para corroborar una instalación correcta

1. Presiona las teclas Windows + R para abrir la ventana de Ejecutar.
2. Una vez abierta la ventana Ejecutar escribe el comando cmd y presiona ctrl+shift+enter para ejecutar una línea de comandos con permisos de administrador.
3. Windows te preguntará si quieres abrir el Procesador de comandos de Windows con permisos de administrador, presiona sí.
4. En la línea de comandos escribe python
Tu consola se mostrará así.
![consola de python](https://parzibyte.me/blog/wp-content/uploads/2017/11/Probando-python.png)

¡Ya estás listo para continuar con el curso!

### MacOS
La forma sencilla es tener instalado [homebrew](http://https://brew.sh/ "homebrew") y usar el comando:

** Para instalar la Versión 2.7**

`brew install python`
Para instalar la Versión 3.x

`brew install python3`
### Linux
Generalmente Linux ya lo trae instalado, para comprobarlo puedes ejecutar en la terminal el comando.
Versión 2.7
`python -v`
Versión 3.x
`python3 -v`
Si el comando arroja un error quiere decir que no lo tienes instalado, en ese caso los pasos para instalarlo cambian un poco de acuerdo con la distribución de linux que estés usando. Generalmente el gestor de paquetes de la distribución de Linux tiene el paquete de Python

**Si eres usuario de Ubuntu o Debian por ejemplo puedes usar este comando para instalar la versión 3.1:**

`$ sudo apt-get install python3.1`

**Si eres usuario de Red Hat o Centos por ejemplo puedes usar este comando para instalar python**

`$ sudo yum install python`

Si usas otra distribución o no has podido instalar python en tu sistema Linux dejame un comentario y vemos tu caso específico.

Si eres usuario habitual de linux también puedes [descargar los archivos](http://https://www.python.org/downloads/source/ "descargar los archivos") para instalarlo manualmente.

### Antes de empezar:
Para usar Python debemos tener un editor de texto abierto y una terminal o cmd (línea de comandos en Windows) como administrador.
No le tengas miedo a la consola, la consola es tu amiga.
Para ejecutar Python abre la terminal y escribe:
`python`
Te abrirá una consola de Python, lo notarás porque el prompt cambia y ahora te muestra tres simbolos de mayor que “ >>> “ y el puntero adelante indicando que puedes empezar a ingresar comandos de python.
` >>> `
En éste modo puedes usar todos los comandos de Python o escribir código directamente.

Nota: Si deseas ejecutar código de un archivo sólo debes guardarlo con [extension.py](http:/http://extension.py// "extension.py") y luego ejecutar en la terminal:
` $ python archivo.py`
Ten en cuenta que para ejecutar el archivo con extensión “.py” debes estar ubicado en el directorio donde tienes guardado el archivo.

**Para salir de Python** y regresar a la terminal debes usar el comando exit()

Cuando usamos Python debemos atender ciertas reglas de la comunidad para definir su estructura. Las encuentras en el libro [PEP8](http://https://www.python.org/dev/peps/pep-0008/ "PEP8").

**Tipos de datos en Python**

- **Enteros (int)**: en este grupo están todos los números, enteros y long:
*ejemplo: 1, 2.3, 2121, 2192, -123*

- **Booleanos (bool)**: Son los valores falso o verdadero, compatibles con todas las operaciones booleanas ( and, not, or ):
*ejemplo: True, False*

- **Cadenas (str)**: Son una cadena de texto:
*ejemplos: “Hola”, “¿Cómo estas?”*

- **Listas**: Son un grupo o array de datos, puede contener cualquiera de los datos anteriores:
*ejemplos: [1,2,3, ”hola” , [1,2,3] ], [1,“Hola”,True ]*

- **Diccionarios**: Son un grupo de datos que se acceden a partir de una clave:
*ejemplo: {“clave”:”valor”}, {“nombre”:”Fernando”}*

- **Tuplas:** también son un grupo de datos igual que una lista con la diferencia que una tupla después de creada no se puede modificar.
*ejemplos: (1,2,3, ”hola” , (1,2,3) ), (1,“Hola”,True ) (Pero jamás podremos cambiar los elementos dentro de esa Tupla)*

En Python trabajas con **módulo**s y **ficheros** que usas para importar las librerías.

**Funciones**
Las funciones las defines con def junto a un nombre y unos paréntesis que reciben los parámetros a usar. Terminas con dos puntos.

`def nombre_de_la_función(parametros):`

Después por indentación colocas los datos que se ejecutarán desde la función:
```python 
 >>> def my_first_function():
 ...	return “Hello World!” 
 ...    
 >>> my_first_function()
```
Hello World!

### Variables
Las variables, a diferencia de los demás lenguajes de programación, no debes definirlas, ni tampoco su tipo de dato, ya que al momento de iterarlas se identificará su tipo. Recuerda que en Python todo es un objeto.

```python
A = 3 
B = A
```
### Listas
Las listas las declaras con corchetes. Estas pueden tener una lista dentro o cualquier tipo de dato.
```python
 >>> L = [22, True, ”una lista”, [1, 2]] 
 >>> L[0] 
 22
```
**Tuplas**
Las tuplas se declaran con paréntesis, recuerda que no puedes editar los datos de una tupla después de que la has creado.
```python
 >>> T = (22, True, "una tupla", (1, 2)) 
 >>> T[0] 
 22
```
### Diccionarios
En los diccionarios tienes un grupo de datos con un formato: la primera cadena o número será la clave para acceder al segundo dato, el segundo dato será el dato al cual accederás con la llave. Recuerda que los diccionarios son listas de llave:valor.
```python
 >>> D = {"Kill Bill": "Tamarino", "Amelie": "Jean-Pierre Jeunet"} 
 >>> D["Kill Bill"]
 "Tamarino"
```
**Conversiones**
De flotante a entero:
```python
 >>> int(4.3)
 4
```
De entero a flotante:
```python
 >>> float(4) 
 4.0
```
De entero a string:
```python
>>> str(4.3) 
 "4.3"
```
De tupla a lista:
```python
>>> list((4, 5, 2)) 
 [4, 5, 2]
```
 
### Operadores Comunes
Longitud de una cadena, lista, tupla, etc.:
```python
>>> len("key") 
 3
```
Tipo de dato:
````python
>>> type(4) 
 < type int >
```
APLICAR una conversión a un conjunto como una lista:
```python
>>> map(str, [1, 2, 3, 4])
 ['1', '2', '3', '4']
```
Redondear un flotante con x número de decimales:
```python
>>> round(6.3243, 1)
 6.3
```
Generar un rango en una lista (esto es mágico):
```python
>>> range(5)
 [0, 1, 2, 3, 4]
 ```
Sumar un conjunto:

```python
 sum([1, 2, 4]) 
 7```
 
Organizar un conjunto:
```python
sorted([5, 2, 1]) 
 [1, 2, 5]
 ```
Conocer los comandos que le puedes aplicar a x tipo de datos:
````python
Li = [5, 2, 1]
dir(Li)
 ['append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']```
‘append’, ‘count’, ‘extend’, ‘index’, ‘insert’, ‘pop’, ‘remove’, ‘reverse’, ‘sort’ son posibles comandos que puedes aplicar a una lista.

Información sobre una función o librería:
```python
help(sorted) 
 (Aparecerá la documentación de la función sorted)
```
### Clases
Clases es uno de los conceptos con más definiciones en la programación, pero en resumen sólo son la representación de un objeto. Para definir la clase usas_ class_ y el nombre. En caso de tener parámetros los pones entre paréntesis.

Para crear un constructor haces una función dentro de la clase con el nombre init y de parámetros self (significa su clase misma), nombre_r y edad_r:
````python
 class Estudiante(object): 
 ... 	def __init__(self,nombre_r,edad_r): 
 ... 		self.nombre = nombre_r 
 ... 		self.edad = edad_r 
 ...
 ... 	def hola(self): 
 ... 		return "Mi nombre es %s y tengo %i" % (self.nombre, self.edad) 
 ... 
e = Estudiante(“Arturo”, 21) 
 print (e.hola())
 Mi nombre es Arturo y tengo 21
```
Lo que hicimos en las dos últimas líneas fue:

1. En la variable e llamamos la clase Estudiante y le pasamos la cadena Arturo” y el entero 21.

2. Imprimimos la función hola() dentro de la variable e (a la que anteriormente habíamos pasado la clase).

Y por eso se imprime la cadena “Mi nombre es Arturo y tengo 21”

### Métodos especiales

**cmp**(self,otro)
Método llamado cuando utilizas los operadores de comparación para comprobar si tu objeto es menor, mayor o igual al objeto pasado como parámetro.

**len**(self)
Método llamado para comprobar la longitud del objeto. Lo usas, por ejemplo, cuando llamas la función len(obj) sobre nuestro código. Como es de suponer el método te debe devolver la longitud del objeto.

**init**(self,otro)
Es un constructor de nuestra clase, es decir, es un “método especial” que se llama automáticamente cuando creas un objeto.

### Condicionales IF
Los condicionales tienen la siguiente estructura. Ten en cuenta que lo que contiene los paréntesis es la comparación que debe cumplir para que los elementos se cumplan.
```python
 if ( a > b ):
 	elementos 
 elif ( a == b ): 
 	elementos 
 else:
 	elementos```

### Bucle FOR
El bucle de for lo puedes usar de la siguiente forma: recorres una cadena o lista a la cual va a tomar el elemento en cuestión con la siguiente estructura:
```python
 for i in ____:
 	elementos```

Ejemplo:
```python
for i in range(10):
 	print i```

En este caso recorrerá una lista de diez elementos, es decir el _print i _de ejecutar diez veces. Ahora i va a tomar cada valor de la lista, entonces este for imprimirá los números del 0 al 9 (recordar que en un range vas hasta el número puesto -1).

**Bucle WHILE**
En este caso while tiene una condición que determina hasta cuándo se ejecutará. O sea que dejará de ejecutarse en el momento en que la condición deje de ser cierta. La estructura de un while es la siguiente:

```python
while (condición):
 	elementos```
Ejemplo:

```python
>>> x = 0 
 >>> while x < 10: 
 ... 	print x 
 ... 	x += 1```
 
En este ejemplo preguntará si es menor que diez. Dado que es menor imprimirá x y luego sumará una unidad a x. Luego x es 1 y como sigue siendo menor a diez se seguirá ejecutando, y así sucesivamente hasta que x llegue a ser mayor o igual a 10.

### Funciones principales en Turtle Graphics

Las funciones principales para animar nuestro objeto son las siguientes:

**forward(distance)**: Avanzar una determinada cantidad de píxeles.
**backward(distance)**: Retroceder una determinada cantidad de píxeles.
**left(angle):** Girar hacia la izquierda un determinado ángulo.
**right(angle)**: Girar hacia la derecha un determinado ángulo.

Por otro lado, puede que en ocasiones queramos desplazarnos de un punto a otro sin dejar rastro. Para ello utilizaremos las siguientes funciones:

**home(distance)**: Desplazarse al origen de coordenadas.
**goto((x, y))**: Desplazarse a una coordenada en concreto.
**pendown()**: Subir el lápiz para no mostrar el rastro.
**penup()**: Bajar el lápiz para mostrar el rastro.

Por último, puede que queramos cambiar el color o tamaño del lápiz. En ese caso utilizaremos las siguientes funciones gráficas:

**shape(‘turtle’):** Cambia al objeto tortuga.
**pencolor(color)**: Cambiar al color especificado.
**pensize(dimension)**: Tamaño de la punta del lápiz.

# Operadores aritmeticos
1. ( ) = parentesis
2. = multiplicacion
3. / = division
4. % = modulo
5. = suma
6. – = resta
7. **  = potencia
### Operadores racionales

1. = mayor que

3. < = menor que
4. = = mayor o igual que

6. = = menor o igual que

8. != = diferente
9. == = igualdad
10. = -> igual

### Operadores lógicos

1. Not
2. And
3. Or

### Prioridad de operadores en funcionalidad

1. ()
2. **
3. *,/, not
4. +, -, and
5. ,<,==,>=,<=, ! , or

### Operadores de asignación
a + = 5 Suma en Asignación
b - = 2 resta en Asignación
a * = 3 multiplicación en Asignación
a / = 3 división en Asignación
a ** = 2 potencia en Asignación
a % = 2 modulo en Asignación

### Variables y expresiones

Asignaciones = Crea una variable y asigna un valor
_age = 20 #Cuando una variable empieza por ’ _ ’ es una variable privada
PI = 3.14159 #Cuando una variable esta en mayúscula significa que es una constante
__do_not_touch = “something important” #Cuando esta variable esta en doble guion bajo = significa que el codigo se puede romper

Re asignaciones de variables
my_var = 2
my_var = my_var * 5
print(my_var)

Variables y Expresiones
Las variables pueden contener números y letras
No pueden comenzar con numero
Multiples palabras se unen con _
No se pueden utilizar palabras reservadas

### Operaciones con Strings en Python

Los strings tienen varios métodos que nosotros podemos utilizar.

- **upper**: convierte todo el string a mayúsculas
- **lower**: convierte todo el string a minúsculas
- **find**: encuentra el indice en donde existe un patrón que nosotros definimos
- **startswith**: significa que empieza con algún patrón.
- **endswith**: significa que termina con algún patrón.
- **capitalize**: coloca la primera letra en mayúscula y el resto en minúscula.

**in** y **not in** nos permite saber con cualquier secuencia sin una subsecuencia o substrings se encuentra adentro de la secuencia mayor.

**dir**: Nos dice todos los métodos que podemos utilizar dentro de un objeto.
**help**: nos imprime en pantalla el docstrings o comentario de ayuda o instrucciones que posee la función. Casi todas las funciones en Python las tienen.

### Operaciones con strings: Slices en python

Los* slices *en Python nos permiten manejar secuencias de una manera poderosa.

Slices en español significa ““rebanada””, si tenemos una secuencia de elementos y queremos una rebanada tenemos una sintaxis para definir qué pedazos queremos de esa secuencia.

`secuencia[comienzo: final: pasos]`

### Operaciones con strings: Slices en python


En realidad la notación slice es más sencilla de lo que parece. Este es el resumen de sus principales opciones:

a[inicio:final] = desde el elemento 'inicio' hasta 'final'-1
a[inicio:] = desde el elemento 'inicio' hasta el final del array
a[:final] = desde el primer elemento hasta elemento 'final'-1
a[:] = todos los elementos del array
a[::salto] = desde el elemento principio hasta el final del array de dendo saltos de en en x en x letras
Además de estos cuatro casos que son los más comunes, también puedes utilizar un tercer valor opcional llamado step o salto:

#### el número de elementos indicado por 'salto', es decir, de 2 en 2 por ejemplo

a[inicio: final:salto]
Otra de las opciones más interesantes del slice es que el principio y el final pueden ser números negativos. 
Esto indica que se empieza a contar desde el final del array:

a[-1] = selecciona el último elemento del array
a[-2:] = selecciona los dos últimos elementos del array
a[:-2] = selecciona todos los elementos excepto los dos últimos
Si existen menos elementos de los que quieres seleccionar, Python se porta bien y no muestra ningún error. 
Si por ejemplo el array sólo tiene un elemento y tratas de seleccionar `a[:-2]`, el resultado es una lista vacía en vez de un mensaje de error. Como es posible que a veces te interese mostrar un error en estos casos, es algo que deberías tener en cuenta.

`un_slice = 'Platano'`

#### MOSTRAMOS LA PABLABRA COMPLETA
print(un_slice[0:len(un_slice):1])
print(un_slice[0:len(un_slice)])
print(un_slice[:])

#### MOSTRAMOS UNA LETRA CONCRETA DE LA PABLABRA
print(un_slice[0])=  Nos muestra la primera letra de la palabra empezando por el principio
print(un_slice[-2]) = Nos muestra la segunda letra de la palabra empezando por el final

#### MOSTRAMOS LA PABLABRA DESDE UNA LETRA CONCRETA
print(un_slice[0:]) = Nos muestra la pablabra desde la posición cero hasta el final
print(un_slice[3:]) = Nos muestra la pablabra desde la posición tres hasta el final
print(un_slice[3::2]) = Nos muestra la pablabra desde la posición tres hasta el final dando saltos saltos de dos en dos letras

#### MOSTRAMOS LA PABLABRA HASTA UNA LETRA CONCRETA
print(un_slice[:len(un_slice)])  = Nos muestra la pablabra completa
print(un_slice[:len(un_slice)-2]) = Nos muestra la pablabra desde la posición cero hasta la posición 2 empezando por el final
print(un_slice[:len(un_slice):3]) = Nos muestra la pablabra desde la posición cero hasta el final dando saltos saltos de tres en tres letras

#### MOSTRAMOS LA PABLABRA DANDO SALTOS
print(un_slice[0:len(un_slice):2]) = Nos muestra la pablabra desde la posición cero hasta el final dando saltos saltos de dos en dos letras
print(un_slice[2:len(un_slice):4]) = Nos muestra la pablabra desde la posición dos hasta el final dando saltos saltos de cuatro en cuatro letras

#### For loops

Las iteraciones es uno de los conceptos más importantes en la programación. En Python existen muchas manera de iterar pero las dos principales son los **for loops** y **while loops**.

Los **for loops** nos permiten iterar a través de una secuencia y los while loops nos permiten iterara hasta cuando una condición se vuelva falsa.

**for loops:**

- Tienen dos keywords break y continue que nos permiten salir anticipadamente de la iteración
- Se usan cuando se quiere ejecutar varias veces una o varias instrucciones.
- for [variable] in [secuencia]:

Es una convención usar la letra `i` como variable en nuestro for, pero podemos colocar la que queramos.

`range`: Nos da un objeto rango, es un iterador sobre el cual podemos generar secuencias.

### While loops

Al igual que las for loops, las **while loops** nos sirve para iterar, pero las for loops nos sirve para iterar a lo largo de una secuencia mientras que las **while loops** nos sirve para iterar mientras una condición sea verdadera.

Si no tenemos un mecanismo para convertir el mecanismo en falsedad, entonces nuestro while loops se ira al infinito(infinite loop).

####Iterators and generators

Aunque no lo sepas, probablemente ya utilices iterators en tu vida diaria como programador de Python. Un iterator es simplemente un objeto que cumple con los requisitos del Iteration Protocol (protocolo de iteración) y por lo tanto puede ser utilizado en ciclos. Por ejemplo,
```python
for i in range(10):
    print(i)
```
En este caso, la función range es un iterable que regresa un nuevo valor en cada ciclo. Para crear un objeto que sea un iterable, y por lo tanto, implemente el protocolo de iteración, debemos hacer tres cosas:

- Crear una clase que implemente los métodos **iter** y **
**
- **ite**r debe regresar el objeto sobre el cual se iterará
- **next** debe regresar el siguiente valor y aventar la excepción StopIteration cuando ya no hayan elementos sobre los cual iterar.
Por su parte, los generators son simplemente una forma rápida de crear iterables sin la necesidad de declarar una clase que implemente el protocolo de iteración. Para crear un generator simplemente declaramos una función y utilizamos el keyword yield en vez de return para regresar el siguiente valor en una iteración. Por ejemplo,
```python
def fibonacci(max):
    a, b = 0, 1
    while a < max:
        yield a
        a, b = b, a+b
```
Es importante recalcar que una vez que se ha agotado un generator ya no podemos utlizarlo y debemos crear una nueva instancia. Por ejemplo,
```python
fib1 = fibonacci(20)
fib_nums = [num for num in fib1]
...
double_fib_nums = [num * 2 for num in fib1] # no va a funcionar
double_fib_nums = [num * 2 for num in fibonacci(30)] # sí funciona
```

### Uso de listas

Python y todos los lenguajes nos ofrecen *constructos* mucho más poderosos, haciendo que el desarrollo de nuestro software sea

- Más sofisticado
- Más legible
- Más fácil de implementar
Estos *constructos* se llaman **Estructuras de Datos** que nos permiten agrupar de distintas maneras varios valores y elementos para poderlos manipular con mayor facilidad.

Las **listas** las vas a utilizar durante toda tu carrera dentro de la programación e ingeniería de Software.

Las **listas** son una secuencia de valores. A diferencia de los strings, las listas pueden tener cualquier tipo de valor. También, a diferencia de los strings, son mutables, podemos agregar y eliminar elementos.

En Python, las listas son referenciales. Una lista no guarda en memoria los objetos, sólo guarda la referencia hacia donde viven los objetos en memoria

Se inician con `[]` o con la built-in function `list`.

### Operaciones con listas

Ahora que ya entiendes cómo funcionan las **listas**, podemos ver qué tipo de operaciones y métodos podemos utilizar para modificarlas, manipularlas y realizar diferentes tipos de cómputos con esta Estructura de Datos.

El operador **+(suma)** concatena dos o más listas.
El operador ***(multiplicación)** repite los elementos de la misma lista tantas veces los queramos multiplicar

Sólo podemos utilizar **+(suma) **y ***(multiplicación).**

Las listas tienen varios métodos que podemos utilizar.

- `append` nos permite añadir elementos a listas. Cambia el tamaño de la lista.
- `pop` nos permite sacar el último elemento de la lista. También recibe un índice y esto nos permite elegir qué elemento queremos eliminar.
- `sort` modifica la propia lista y ordenarla de mayor a menor. Existe otro método llamado `sorted`, que también ordena la lista, pero genera una nueva instancia de la lista
- `del` nos permite eliminar elementos vía indices, funciona con *slices*
- `remove` nos permite es pasarle un valor para que Python compare internamente los valores y determina cuál de ellos hace match o son iguales para eliminarlos.

### Diccionarios

Los diccionarios se conocen con diferentes nombres a lo largo de los lenguajes de programación como HashMaps, Mapas, Objetos, etc. En Python se conocen como **Diccionarios**.

Un diccionario es similar a una lista sabiendo que podemos acceder a través de un indice, pero en el caso de las listas este índice debe ser un número entero. Con los diccionarios puede ser cualquier objeto, normalmente los verán con **strings** para ser más explicitos, pero funcionan con muchos tipos de llaves…

Un diccionario es una asociación entre llaves(**keys**) y valores(**values**) y la referencia en Python es muy precisa. Si abres un diccionario verás muchas palabras y cada palabra tiene su definición.

Para iniciar un diccionario se usa `{}` o con la función `dict`

Estos también tienen varios métodos. Siempre puedes usar la función `dir` para saber todos los métodos que puedes usar con un objeto.

Si queremos ciclar a lo largo de un diccionario tenemos las opciones:

`keys`: nos imprime una lista de las llaves
`values`: nos imprime una lista de los valores
`items`: nos manda una lista de tuplas de los valores

### Diccionarios

Los diccionarios se conocen con diferentes nombres a lo largo de los lenguajes de programación como HashMaps, Mapas, Objetos, etc. En Python se conocen como **Diccionarios**.

Un diccionario es similar a una lista sabiendo que podemos acceder a través de un indice, pero en el caso de las listas este índice debe ser un número entero. Con los diccionarios puede ser cualquier objeto, normalmente los verán con **strings** para ser más explicitos, pero funcionan con muchos tipos de llaves…

Un diccionario es una asociación entre llaves(**keys**) y valores(**values**) y la referencia en Python es muy precisa. Si abres un diccionario verás muchas palabras y cada palabra tiene su definición.

Para iniciar un diccionario se usa `{}` o con la función `dict`

Estos también tienen varios métodos. Siempre puedes usar la función `dir` para saber todos los métodos que puedes usar con un objeto.

Si queremos ciclar a lo largo de un diccionario tenemos las opciones:

`keys`: nos imprime una lista de las llaves
`values`: nos imprime una lista de los valores
`items`: nos manda una lista de tuplas de los valores

ejemplos:

```python
>>> rea.items()
dict_items([('Pizza', 'la comida mas deliciosa del mundo'), ('Pasta', 'la Comida mas sabrosa de Italia')])
>>> for key in rea.keys():
...     print(key)
...
Pizza
Pasta
>>> for key in rea.values():
...     print(key)
...
la comida mas deliciosa del mundo
la Comida mas sabrosa de Italia
>>> for key, value in rea.items():
...     print(key, value)
...
Pizza la comida mas deliciosa del mundo
Pasta la Comida mas sabrosa de Italia
```

### Tuplas y conjuntos

Tuplas(tuples) son iguales a las listas, la única diferencia es que son** inmutables**, la diferencia con los strings es que pueden recibir muchos tipos valores. Son una serie de valores separados por comas, casi siempre se le agregan paréntesis para que sea mucho más legible.

Para poderla inicializar utilizamos la función `tuple`.

Uno de sus usos muy comunes es cuando queremos regresar más de un valor en nuestra función.

Una de las características de las Estructuras de Datos es que cada una de ellas nos sirve para algo especifico. No existe en programación una navaja suiza que nos sirva para todos. Los mejores programas son aquellos que utilizan la herramienta correcta para el trabajo correcto.

Conjutos(**sets**) nacen de la teoría de conjuntos. Son una de las Estructuras más importantes y se parecen a las listas, podemos añadir varios elementos al conjunto, pero **no pueden existir elementos duplicados**. A diferencia de los tuples podemos agregar y eliminar, son **mutables**.

Los sets se pueden inicializar con la función **set**. Una recomendación es inicializarlos con esta función para no causar confusión con los diccionarios.

- `add `nos sirve añadir elementos.
- `remove` nos permite eliminar elementos.


#### **Declaracion de tuplas**
a= 1,2,3
a=(1,2,3)
##### Operaciones con tuplas:
**Acceder a un valor mediante indice de tupla**
a[0]=1
**Conteo de cuantas veces está un valor en la tupla**
a.count(1)—>1

**Declaración conjutos o Sets**
a= set([1,2,3])
a={1,2,3}
##### Operaciones con conjuntos:

- NO se puede acceder a un valor mediante índice
- NO se puede agregar un valor ya existente, por ejemplo
**Agregar un valor a conjunto**
`a.add(1)`—> error!! (valor existente en set)
`a.add(20)`—> a={1,2,3,20}
Tenemos:
a={1,2,3,20}
b={3,4,5}
`a.intersection(b)`–> {3} **(elementos en común)**
`a.union(b)`—>{1,2,3,20,4,5} **(elementos de a + b pero sin repetir ninguno)**
`a.difference(b)`–>{1,2,20} **(elementos de a que no se encuentran en b)**
`b.difference(a)`–>{4,5} (elementos de b que no se encuentran en a)

## Python comprehensions

Las Comprehensions son constructos que nos permiten generar una secuencia a partir de otra secuencia.

Existen tres tipos de comprehensions:

- List comprehensions
`[element for element in element_list if element_meets_condition]`
- Dictionary comprehensions
`{key: element for element in element_list if element_meets_condition}`
- Sets comprehensions
`{element for element in element_list if elements_meets_condition}`

### Búsquedas binarias

Uno de los conceptos más importantes que debes entender en tu carrera dentro de la programación son los algoritmos. No son más que una secuencia de instrucciones para resolver un problema específico.

Búsqueda binaria lo único que hace es tratar de encontrar un resultado en una lista ordenada de tal manera que podamos razonar. Si tenemos un elemento mayor que otro, podemos simplemente usar la mitad de la lista cada vez.

### Decoradores

**Python** es un lenguaje que acepta **diversos** paradigmas como programación orientada a objetos y la programación funcional, siendo estos los temas de nuestro siguiente módulo.

Los **decoradores** son una función que envuelve a otra función para modificar o extender su comportamiento.

En Python las **funciones** son ciudadanos de primera clase, first class citizen, esto significan que las funciones pueden recibir funciones como **parámetros** y pueden **regresar** funciones. Los **decoradores** utilizan este concepto de manera fundamental.

### Decoradores en Python

En esta clase pondremos en práctica lo aprendido en la clase anterior sobre decoradores.

Por convención la función interna se llama` wrapper`,

Para usar los decoradores es con el símbolo de @(arroba) y lo colocamos por encima de la función. Es un *sugar syntax*

`*args **kwargs `son los argumentos que tienen keywords, es decir que tienen nombre y los argumentos posicionales, los **args**. Los asteriscos son simplemente una expansión.

### Decoradores en Python

En esta clase pondremos en práctica lo aprendido en la clase anterior sobre decoradores.

Por convención la función interna se llama` wrapper`,

Para usar los decoradores es con el símbolo de @(arroba) y lo colocamos por encima de la función. Es un *sugar syntax*

`*args **kwargs `son los argumentos que tienen keywords, es decir que tienen nombre y los argumentos posicionales, los **args**. Los asteriscos son simplemente una expansión.

#### DECORADORES EN PYTHON

Los decoradores sirven para ejecutar lógica del código antes y/o después de otra función, esto nos ayuda a generar funciones y código que pueda ser reutilizado fácilmente sin hacer más extenso nuestro código. Hay que recordar que si se genera una función dentro de otra solo existiera en ese scope(dentro de la función padre), si se quiere invocar una función varias veces dentro de otras se tiene que generar de manera global.

`**args y kwargs**`

Básicamente lo que hacen es pasar tal cual los valores de de los argumentos que se pasan a la función args hace referencias a listas y kwargs a elementos de un diccionario (llave: valor)

** args: **

```python
def test_valor_arg(n_arg, *args):
    print('primer valor normal: ', n_arg)

    For arg in args:
	print('este es un valor de *args: ',arg)

    print(type(args))

if__name__ == '__main__':

    test_valor_args('carlos','Karla','Paola','Elena')
```
- el tipo de valor y es una tupla
- solo poniendo argumentos divididos por comas los convierte
**kuargs: **

```python
def test_valor_kwargs(**kwargs):
    if kwargs is not None:
        for key,  value in kwargs.items():
            print('%s == %s' %(key,value))

    print(type(kwargs))

if __name__ == '__main__':

 test_valor_kwargs(caricatura='batman')
 ```
 
- el valor que te da es un diccionario
- toma los valores en los extremos de un signo igual
Este es un ejemplo usando los 2 en una función

```python
def test_valor_kwargs_args(*args, **kwargs):
    print(type(kwargs))
    print(kwargs)
    print('----------')
    print(type(args))
    print(args)

if __name__ == '__main__':
    test_valor_kwargs_args('flash', 'batman', caricatura='batman', empresa = 'dc')
```

### ¿Qué es la programación orientada a objetos?
La programación orientada a objetos es un paradigma de programación que otorga los medios para estructurar programas de tal manera que las propiedades y comportamientos estén envueltos en objetos individuales.

Para poder entender cómo modelar estos objetos tenemos que tener claros cuatro principios:

- Encapsulamiento.
- Abstracción
- Herencia
- Polimorfismo
Las clases simplemente nos sirven como un molde para poder generar diferentes instancias.

### Programación orientada a objetos en Python

Para declarar una clase en Python utilizamos la keyword `class`, después de eso le damos el nombre. Una convención en Python es que todas las clases **empiecen con mayúscul**a y se continua con **CamelCase**.

Un método fundamental es dunder init(**__init__**). Lo único que hace es inicializar la clase basado en los parámetros que le damos al momento de construir la clase.

`self` es una referencia a la clase. Es una forma internamente para que podamos acceder a las propiedades y métodos.

#### Scopes and namespaces

En Python, un name, también conocido como identifier, es simplemente una forma de otorgarle un nombre a un objeto. Mediante el nombre, podemos acceder al objeto. Vamos a ver un ejemplo:
```python
my_var = 5

id(my_var) # 4561204416
id(5) # 4561204416
```
En este caso, el identifier **my_var** es simplemente una forma de acceder a un objeto en memoria (en este caso el espacio identificado por el número 4561204416 ). Es importante recordar que un name puede referirse a cualquier tipo de objeto (aún las funciones).

```python
def echo(value):
    return value

a = echo

a(‘Billy’) # 3
```
Ahora que ya entendimos qué es un name podemos avanzar a los namespaces (espacios de nombres). Para ponerlo en palabras llanas, un namespace es simplemente un conjunto de names.

En Python, te puedes imaginar que existe una relación que liga a los nombres definidos con sus respectivos objetos (como un diccionario). Pueden coexistir varios namespaces en un momento dado, pero se encuentran completamente aislados. Por ejemplo, existe un namespace específico que agrupa todas las variables globales (por eso puedes utilizar varias funciones sin tener que importar los módulos correspondientes) y cada vez que declaramos una módulo o una función, dicho módulo o función tiene asignado otro namespace.

A pesar de existir una multiplicidad de namespaces, no siempre tenemos acceso a todos ellos desde un punto específico en nuestro programa. Es aquí donde el concepto de scope (campo de aplicación) entra en juego.

Scope es la parte del programa en el que podemos tener acceso a un namespace sin necesidad de prefijos.

En cualquier momento determinado, el programa tiene acceso a tres scopes:

- El scope dentro de una función (que tiene nombres locales)
- El scope del módulo (que tiene nombres globales)
- El scope raíz (que tiene los built-in names)

Cuando se solicita un objeto, Python busca primero el nombre en el scope local, luego en el global, y por último, en el raíz. Cuando anidamos una función dentro de otra función, su scope también queda anidado dentro del scope de la función padre.

```python
def outer_function(some_local_name):
    def inner_function(other_local_name):
         # Tiene acceso a la built-in function print y al nombre local some_local_name
         print(some_local_name) 
        
         # También tiene acceso a su scope local
         print(other_local_name)
```
Para poder manipular una variable que se encuentra fuera del scope local podemos utilizar los keywords global y nonlocal.
```python
some_var_in_other_scope = 10

def some_function():
     global some_var_in_other_scope
     
     Some_var_in_other_scope += 1
```

### Introducción a Click

**Click** es un pequeño framework que nos permite crear aplicaciones de Línea de comandos. Tiene cuatro decoradores básicos:

- **@click_group**: Agrupa una serie de comandos
- **@click_command**: Aca definiremos todos los comandos de nuestra apliacion
- **@click_argument**: Son parámetros necesarios
- @**click_option**: Son parámetros opcionales
Click también realiza las conversiones de tipo por nosotros. Esta basado muy fuerte en decoradores.

### Crear ambiente virtual

1. `pip install virtualenv`
2. `python -m venv venv` o `virtualenv --python=python3 venv`

## o 

1. `virtualenv venv`
2. `python -m venv venv`
3. `venv\scripts\activate`

### ENTORNO VIRTUAL EN WINDOWS
luego seguí estos pasos.
1. pip Install virtualenv
2. Pip Install virtualenvwrapper-win:
3.  una vez echo esto anote el comando que dice el profe con una ligera modificación
estando en la carpeta “platzi-ventas” ejecuto el comando:
• `virtualenv --python=python venv`

si ya ha respondido la consola con done, ingreso a la carpeta con el nombre que se creo** “venv”**
• **cd ** venv/Scripts

ejecuto el archivo activate
activate, inmediatamente inicializa el ambiente virtual tal como al profe!! EJ:
… \CURSO PLATZI\CURSO_PYTHON_3\PLATZI-VENTAS\venv\Script> activate
Y por fin ingresas al entorno virtual
(venv) C:\Users…\CURSO PLATZI\CURSO_PYTHON_3\PLATZI-VENTAS\venv
luego regresa a la carpeta donde esta es setup.py en la Carpeta PLATZI-VENTAS y ejecuta :
`pip install --editable .`

### Clients
Modelaremos a nuestros clientes y servicios usando lo aprendido en clases anteriores sobre programación orientada a objetos y clases.

`@staticmethod` nos permite declarar métodos estáticos en nuestra clase. Es un método que se puede ejecutar sin necesidad de una instancia de una clase. No hace falta que reciba `self` como parámetro.

### Tablas con mejor formatoTablas con mejor formato
¡Hola a todos! Les comparto una mejora que le hice al código para que la tabla tenga un mejor formato.
Utilicé un módulo llamado Tabulate, que pueden encontrar en la página oficial de PyPI o solo instalarlo con $ `pip install tabulate`.

El código del método list queda así:

```python
from tabulate import tabulate

@clients.command()
@click.pass_context
def list(ctx):
    """List all clients"""
    client_service = ClientService(ctx.obj['clients_table'])
    clients_list = client_service.list_clients()

    headers = [field.capitalize() for field in Client.schema()]
    table = []

    for client in clients_list:
        table.append(
            [client['name'],
             client['company'],
             client['email'],
             client['position'],
             client['uid']])

    print(tabulate(table, headers))
```
### para ejecutar el archivo

se crea él ambienté virtual con `pip install --editable .` esto se ejecuta en donde se encuentra el archivo en mi caso (venv) `C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoPracticodePythonCreaciondeunCRUD\proyectos\proyectoCRUD>`

ya se puede ejecutar el archivo pv y los comandos son los siguientes `pv clients --help`  para ver `pv clients --help` ayuda, `pv clients list` para ver la lista de clientes  y para crear un cliente nuevo se usa `pv clients create`

### par aejecutar el archivo

se crea el anviente virtual con `pip install --editable .` esto se ejecuta en donde se encuentra el archivo en mi caso (venv) `C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoPracticodePythonCreaciondeunCRUD\proyectos\proyectoCRUD>`

ya se puwde ejecutar el archivo pv y los comandos son los siguientees `pv clients --help`  para ver `pv clients --help` ayuda, `pv clients list` para ver la lista de de clientes  y para crear un cliente nuevo se usa `pv clients create`, para actualizar un usuario `pv clients update <id_cliente>`

### Manejo de errores y jerarquía de errores en Python

**Python** tiene una amplia jerarquía de errores que nos da posibilidades para definir errores en casos como donde no se pueda leer un archivo, dividir entre cero o si existen problemas en general en nuestro código Python. El problema con esto es que nuestro programa termina, es diferente a los errores de sintaxis donde nuestro programa nunca inicia.

Para **aventar** un error en Python utilizamos la palabra `raise`. Aunque Python nos ofrece muchos errores es **buena práctica** definir errores específicos de nuestra aplicación y usar los de Python para extenderlos.

Podemos generar nuestros propios errores creando una clase que extienda de `BaseException`.

Si queremos **evitar** que termine nuestro programa cuando ocurra un error, debemos tener una estrategia. Debemos utilizar **try / except** cuando tenemos la posibilidad de que un pedazo de nuestro código falle

- `try`: significa que se ejecuta este código. Si es posible, solo ponemos una sola línea de código ahí como buena práctica
- `except`: es nuestro manejo del error, es lo que haremos si ocurre el error. Debemos ser específicos con el tipo de error que vamos a atrapar.
- `else`: Es código que se ejecuta cuando no ocurre ningún error.
- `finally`: Nos permite obtener un bloque de código que se va a ejecutar sin importar lo que pase.


### Aplicaciones de Python en el mundo realAplicaciones de Python en el mundo real

Python tiene muchas aplicaciones:

En las ciencias tiene muchas librerías que puedes utilizar como analisis de las estrellas y astrofisica; si te interesa la medicina puedes utilizar **Tomopy** para analizar tomografías. También están las librerías más fuertes para la ciencia de datos **numpy**, **Pandas** y **Matplotlib**

En CLI por si te gusta trabajar en la nube y con datacenters, para sincronizar miles de computadoras:

- aws
- gcloud
- rebound
- geeknote
Aplicaciones Web:

- Django
- Flask
- Bottle
- Chalice
- Webapp2
- Gunicorn
- Tornado

### Python 2 vs 3 (Conclusiones)

No es recomendable empezar con Python 2 porque tiene fecha de vencimiento para el próximo año.

**PEP** = Python Enhancement Proposals

Los **PEP** son la forma en la que se define como avanza el lenguaje. Existen tres PEPs que debes saber.

- **PEP8** es la guía de estilo de cómo escribir programas de Python. Es importante escribir de manera similiar para que nuestro software sea legible para el resto de la comunidad
- **PEP25**7 nos explica cómo generar buena documentación en nuestro código
- **PEP20**

```python
import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```