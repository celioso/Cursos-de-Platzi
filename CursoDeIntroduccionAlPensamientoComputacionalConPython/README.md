# Curso de Introducción al Pensamiento Computacional con Python

### Introducción al Cómputo++

1. **"Primera computadora"** creada por los griegos, calcula la posición del sol, luna y algunas constelaciones.

2. En 1801, Telar de Jacquar, separa el resultado de la informacion que contiene las instrucciones.

3. Siglo XIX, **Motor analítico de Babbage**, haciendo uso del avances en mecánica (engranajes) de su época logra separar instrucciones de calculo y realizar varios cálculos a la vez.

4. Finales siglo XIX, **ENIAC** (Eectronic Numerical Integrator and Computer) **usaba sistema decimal**. creada por Alan Turing y Alonso. Esta época exigía exactitud en los cálculos que hacían que compañías dependieran de esta(Compañías Ferroviarias por ejemplo). Turing y Alonso, Descubrieron que matemáticamente todos los algoritmos podíamos reducirlos a una secuencia de soluciones matemáticas.

5. En 1945, Arquitectura de Von Neumann, **EDVAC** (Electronic Discrete Variable Automatic Computer) usaba sistema binario. Su aporte fue el descubrimiento que dentro de los componentes electrónicos se puede usar una serie de hadward para realizar el computo y almacenar datos dentro de memoria.

6. En 1950, Microchip, ejemplo Apple 1

7. En siglo XX, **Arquitectura de Feymann**, aporta las bases matemáticas de computo cuántico.

### Introducción a los lenguajes de programación

**¿Qué es un lenguaje de programación?** Es un lenguaje formal que, mediante una serie de instrucciones, le permite a un programador escribir un conjunto de órdenes, acciones consecutivas, datos y algoritmos para, de esa forma, crear programas que controlen el comportamiento físico y lógico de una máquina.

**¿Qué tipos de lenguaje de programación existen?** El lenguaje de programación es la base para construir todas las aplicaciones digitales que se utilizan en el día a día y se clasifican en dos tipos principales: lenguaje de bajo nivel y de alto nivel.

*Lenguaje de programación de bajo nivel** Son lenguajes totalmente orientados a la máquina.

Este lenguaje sirve de interfaz y crea un vínculo inseparable entre el hardware y el software.

Además, ejerce un control directo sobre el equipo y su estructura física. Para aplicarlo adecuadamente es necesario que el programador conozca sólidamente el hardware. Éste se subdivide en dos tipos:

**Lenguaje máquina** Es el más primitivo de los lenguajes y es una colección de dígitos binarios o bits (0 y 1) que la computadora lee e interpreta y son los únicos idiomas que las computadoras entienden.

Ejemplo: 10110000 01100001

No entendemos muy bien lo que dice ¿verdad? Por eso, el lenguaje ensamblador nos permite entender mejor a qué se refiere éste código.

**Lenguaje ensamblador** El lenguaje ensamblador es el primer intento de sustitución del lenguaje de máquina por uno más cercano al utilizado por los humanos.

Un programa escrito en éste lenguaje es almacenado como texto (tal como programas de alto nivel) y consiste en una serie de instrucciones que corresponden al flujo de órdenes ejecutables por un microprocesador.

Sin embargo, dichas máquinas no comprenden el lenguaje emsamblador, por lo que se debe convertir a lenguaje máquina mediante un programa llamado Ensamblador.

Este genera códigos compactos, rápidos y eficientes creados por el programador que tiene el control total de la máquina.

Ejemplo: MOV AL, 61h (asigna el valor hexadecimal 61 al registro “AL”)

**Lenguaje de programación de alto nivel** Tienen como objetivo facilitar el trabajo del programador, ya que utilizan unas instrucciones más fáciles de entender.

Además, el lenguaje de alto nivel permite escribir códigos mediante idiomas que conocemos (español, inglés, etc.) y luego, para ser ejecutados, se traduce al lenguaje de máquina mediante traductores o compiladores.

**Traductor** Traducen programas escritos en un lenguaje de programación al lenguaje máquina de la computadora y a medida que va siendo traducida, se ejecuta.

**Compilador** Permite traducir todo un programa de una sola vez, haciendo una ejecución más rápida y puede almacenarse para usarse luego sin volver a hacer la traducción.

### Elementos básicos de Python

- ¿Qué es un lenguaje de programación? Es un lenguaje formal que, mediante una serie de instrucciones, le permite a un programador escribir un conjunto de órdenes, acciones consecutivas, datos y algoritmos para, de esa forma, crear programas que controlen el comportamiento físico y lógico de una máquina.

- ¿Qué tipos de lenguaje de programación existen? El lenguaje de programación es la base para construir todas las aplicaciones digitales que se utilizan en el día a día y se clasifican en dos tipos principales: lenguaje de bajo nivel y de alto nivel.

1. **Lenguaje de programación** de bajo nivel Son lenguajes totalmente orientados a la máquina. Este lenguaje sirve de interfaz y crea un vínculo inseparable entre el hardware y el software.

2. **Lenguaje de programación de alto nivel** Tienen como objetivo facilitar el trabajo del programador, ya que utilizan unas instrucciones más fáciles de entender.

3. L**enguaje específico de dominio** En desarrollo de software e ingeniería de dominio, un lenguaje específico de dominio, o "lenguaje específico del dominio", (en inglés domain-specific language, DSL) es un lenguaje de programación o especificación dedicado a resolver un problema en particular, representar un problema específico y proveer una técnica para solucionar una situación particular. El concepto no es nuevo pero se ha vuelto más popular debido al aumento del uso de modelaje específico del dominio.1​

4. **Lenguaje de propósito general** Se llama lenguaje de propósito general al lenguaje de programación cuyos procedimientos, instrucciones y estructuras de datos están diseñados para resolver todo tipo de problemas.

5. **Lenguaje de interpretados y compilados** La principal diferencia entre un lenguaje compilado y uno interpretado es que el lenguaje compilado requiere un paso adicional antes de ser ejecutado, la compilación, que convierte el código que escribes a lenguaje de máquina. Un lenguaje interpretado, por otro lado, es convertido a lenguaje de máquina a medida que es ejecutado.

Ejemplos de lenguajes compilados incluyen C, C++, Java, Go y Rust, entre muchos otros. Ejemplos de lenguajes interpretados incluyen Ruby, Python y JavaScript, entre muchos otros. A todos estos lenguajes se les conoce como lenguajes de alto nivel.

```python
1 + 1 # 1
'Platzi' / 5 # Error
'Platzi' * 3 #PlatziPlatziPlatzi
#PrintStatement
print('Hola mundo')
'Mr' + '. Internatuta'
2 + 2
```

**¿Que es un objeto?** Concepto, abstracción o cosa con límites bien definidos y con significado para el problema que se está manejando

- Escalares vs No escalares -Tipos int. float. bool. str.

```python
#Definiendo variables
my_int  = 1
my_float = 1.0
my_bool = True
my_none = None
my_str = 'Hola'

#Imprimiendo el tipo
type(my_int)
type(my_float)
type(my_bool)
type(my_none)
type(my_str)
```

**¿Que pasa si ejecutas esto?**

```python
1 + 1 
2 - 5
2.0 * 3
6 // 2
6 // 4
6 / 4
7 % 2
2 ** 2
```

Resultado

```python
>>> 1 + 1
2
>>> 2 - 5
-3
>>> 2.0 * 3
6.0
>>> 6 // 2
3
>>> 6 // 4
1
>>> 6 / 4
1.5
>>> 7 % 2
1
>>> 2 ** 2
```
Expresion:
![expresion](images/expresion.png "hola")