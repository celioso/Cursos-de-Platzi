# Curso de Estructuras de Datos Lineales con Python

**Resumen:**

En esta clase se hizo una puntualización de lo que se debe saber en sí mismo para trabajar en este curso:

**Requisitos Mínimos:**

- Elementos léxicos de python o Keywords
- Convenciones de estilo PEP8
- Operadores, Manejo de Strings y Literals.
- Entender sobre Listas, Tuplas, Conjuntos, Diccionarios.

**Tener claro:**

- Declaración de funciones
- Recursividad
- Funciones anidades
- High Order Functions
- Funciones lambda.
- Programación Orientada a Objetos

**Nice to Have:**

Manejo de excepciones
Manipulación de archivos.

### Python tiene sus propias estructuras:

- Listas = `[]`
- Tuplas = `()`
- Conjuntos (sets) = `{}`
- Diccionarios = `{key: value}`

**Tipos de colecciones**
Nos referimos a las estructuras de datos. Una colección es un grupo de cero o más elementos que pueden tratarse como una unidad conceptual.

**Tipos de datos.**

- Non-zeo Value
- Cero
- null
- undefined
Estos tipos de dato también pueden formar parte de una colección. Existen colecciones de tipo Dinámicas que son aquelas que pueden variar su tamaño y las **Inmutables** que no cambian su tamaño.

**Estructuras Lineales**

De forma general encontramos estructuras de datos lineales que están definidas por sus índices. Es decir puedo encotnrarme estrcuturas de datos lineales que sean dinámicas o inmutables, de ello variarán sus propiedades, como poner un elemento al final, (sucesor) o no.

Te encontrarás con Listas, Listas Ordenadas, Cola, Cola de prioridad y más.

- Es decir está ordenadas por posición.
- Solo el primer elemento `no` tiene predecesor

Ej:

- Una fila para un concierto
- Una pila de platos por lavar, o una pila de libros por leer.
- Checklist, una lista de mercado, la lista de Schindler

**Estructuras Jerárquicas**

Estructuras basadas en una jerarquia definida.
Los árboles pueden tener n números de nieleves hacia abajo o adyacentes. Te encotnrarás con árboles Binarios, Montículos.

- Ordendas como árbol invertido (raices)
- Solo el primer nodo `no` tiene predecesores, pero si sucesores.
- Es un sistema de padres e Hijos.

Ej:

- Libros, Capítulos, Temas.
- Abuelos, Madres, Hijos.

**Estructuras Grafos:**

- Cada dato puede tener varios predecesores y sucesores, se les llama vecinos
- Los elementos se relecionan entre si con n relaciones.

Ej:

- Vuelos aéreos, sistemas de recomendación
- La mismísima interntet es un grafo

**Estructuras Desordenadas:**

Estructuras como Bolsa, Bolsa ordenada, Conjuntos, DIccionarios, Diccionario ordenados.
- No tienen orden en particular
- No hay predecesores o sucesores.

Ej:

- Una bolsa de gomitas, no sabe de qué color te va a tocar.

**Estructuras Ordenadas:**

Son estructuras que imponen un orden con una regla. Generalmente una regla de orden.
`item <= item(i+1)` Es decir que el tiem que sigue es el primer elemento +1.

Ej:

- Los directorios telefónicos, los catálogos.

**Conclusión:**

Suponga que tiene un dataset con muchos datos, una colección de libros, música, fotos, y desea ordenar esta colección, ante esta situación siempre existe el Trade Off entre `rapidez/costo/memoria` El conocimeinto de las propiedades de las colecciones te facilitará la selección de estructura de datos según sea el caso y velar por un software eficiente.

![](https://static.platzi.com/media/user_upload/ed1-ca54b039-9574-4ac2-9075-79de80998d37.jpg)

### Tipos de colecciones

![Tipos de colecciones](https://static.platzi.com/media/user_upload/2022-08-15%20%282%29-6a25edb2-5501-4852-a9e6-53e3287f944d.jpg "Tipos de colecciones")

### Operaciones Esenciales en Colecciones:
Algunas operaciones básicas responden necesidades puntuales como saber:

- Tamaño: Las dimensiones
- Pertenencia: Si un elemento pertenece o no.
- Recorrido: Pasar por los elementos
- String: Converir la colección a un string.
- Igualdad: Comparar colecciones
- Concatenación: Unir o sumar listas
- Conversión de tipo: Convertir tipos de datos entre si
- Insertar, Remover, Reemplazar, Acceder a n elementos en n posición.

### Colecciones Incorporadas en Python

**Listas**: Propósito general, de - índices con tamaños dinámicos. Ordenables `lista =[]`.
Usaria las listas para almacenar una serie de números, una lista de palabras,y básicamente cualquier cosa.

**Tuplas**: Inmutables, no se pueden añadir más elementos. Utiles para constantes por ejemplo coordenadas, direcciones. Es de tipo secuencial. `tupla =()`
Las usuaría cuando sé exactamente el tamaño que tendrán mis datos.

**Conjuntos**: Almacenan objetos no duplicados.(Teoría de conjuntos), son de acceso rápido, aceptan operaciones lógicas, son desordenados. `set()` `conjunto={1,2,3,4}`.
Usaría un casteo entre conjuntos y listas cuando quiero eliminar duplicados de una lista.

**Diccionarios**: Pares de llaver valor, arrays asociativos (hash maps), son desordenados, y muy rápidos para hacer consultas. `diccionario ={'Llave':"Valor"}`
Los usaría para almacenar datos, listas, objetos que perfectamente pueden volverse un `dataframe`, o un `defaultdict`.

### Arrays:
Es una estructura de datos lineal, las estructuras de datos son representaciones internas de una colección de información, por lo que un array puede representarme de una forma particular y con unas características puntuales.

- Elemento: Valor almacenado en las posiciones del array
- Indice: Referencia a la posición del elemento.

En a memoria los arrays se almacenan de manera consecutiva, los bits se guardan casilla por casilla consecutivamente.

El array tiene una capacidad de almacenamiento. Puedo tener arrays en 1,2 y/o 3 dimensiones. A mayor complejidad dimensional, es decir, si aumenta la dimensión se hace más complicado acceder a estos datos, se recomienda en python trabajar con `dimensiones <2`

**NOTA**: Los arrays son un tipo de listas, pero las listas no son arrays. Los arrays son diferentes y poseen las siguientes restricciones:

**No pueden:**

- Agregar posiciones
- Remover Posiciones
- Modificar su tamaño
- Su capacidad define al crearse

Los arrays se usan en los sprites de los videojuegos, o en un menú de opciones. Son opciones definidas.

El módulo array de python solo almacena números y caracteres, está basado en listas. Sin embargo tiene funciones reducidas, pero podemos crear nuestros propios arrays.

### Creando un array

Al crear el array por nosotros mismos podremos entender como es que funciona un arreglo y le podremos dar vida según nuestras necesidades. Los métodos que se crearán para el arreglo de ejemplo e incluyendo los métodos adicionales en el reto son:

**Clase padre**

- Crearse
- Longitud
- Representación string
- Pertenencia
- índice
- Reemplazo

**Instancia de la clase Array**

Estas funciones no sobre escriben métodos, heredan todas las funcionalidades de la clase padre, es decir, todos los métodos.
Los aprovecharemos para crear lo métodos del challenge.

- Randomizar
- Sumar

**Nota**: Recuerda que en este script estamos hablando e módulos de python por lo que es buena idea tener tu archivo `__init__.py` para python trate los directorios que contienen archivos como paquetes. Este archivo puede estar vacío o inicializar código para todo el paquete. ¿Quieres saber más? consulta en: [Modules](https://docs.python.org/3/tutorial/modules.html "Modules")

### Creando un array

Al crear el array por nosotros mismos podremos entender como es que funciona un arreglo y le podremos dar vida según nuestras necesidades. Los métodos que se crearán para el arreglo de ejemplo e incluyendo los métodos adicionales en el reto son:

**Clase padre**

- Crearse
- Longitud
- Representación string
- Pertenencia
- índice
- Reemplazo

**Instancia de la clase Array**

Estas funciones no sobre escriben métodos, heredan todas las funcionalidades de la clase padre, es decir, todos los métodos.
Los aprovecharemos para crear lo métodos del challenge.

- Randomizar
- Sumar

**Nota**: Recuerda que en este script estamos hablando e módulos de python por lo que es buena idea tener tu archivo `__init__.py` para python trate los directorios que contienen archivos como paquetes. Este archivo puede estar vacío o inicializar código para todo el paquete. ¿Quieres saber más? consulta en: [Modules](https://docs.python.org/3/tutorial/modules.html "Modules")

```python
class Array:
    def __init__(self, capacity, fill_value=None):
        """
        
        capacity = Será la capacidad que deseamos almacenar en el array.
        fill_value =None Existe para que por defecto nuestro array no tenga nada.


        """

        # CREACIÓN:
        # 1. Estos items se guardaran en una lista vacía, pero usaremos métodos propios.
        self.items = list()

        # 2. Llenaremos nuestra lista vacía con los valores:
 
	# generados según la capacidad deseada para el arreglo.
        # Se añade fill_value para darle esos espacios donde 
	# se almacenaran nuestros datos.
        # Es como hacer hoyos en la tierra donde plantar, 
	# aún no existe la planta, pero si el espacio que ocupará.

        for i in range(capacity):
            self.items.append(fill_value)

        
        # LONGITUD:
        # 1. Definimos método privado usando __len__,
	# usando dundders para que nadie acceda a este.
        # Me define la longitud del arreglo.

        def __len__(self):
            return len(self.items)


        # REPRESENTACIÓN STRING
        # 1. Representación en cadena de caracteres.
        # Parseo el items  a un str.

        def __str__(self):
            return str(self.items)

        # ITERADOR
        # Nos servirá para recorrer la estructura.
        # El método  iter me permitirá recorrer la estructura con su método interno next()

        def __iter__(self):
            return iter(self.items)

        # OBTENER ELEMENTO
        # Para obtener elemento necesito el elemento y el índice 
	# al cual llamo con la notación de []
        # Con el fin de saber su ubicación
        def __getitem__(self, index):
            return self.items[index]

        # REEMPLAZAR ELEMENTOS
        # Suscribimos elementos en el indice con el nuevo elemento.
        def __setitem__(self, index, new_item):
            self.items[index] = new_item




if __name__ == '__main__':
    arreglo = Array(3)
    # Ubicación en memoria
    print(arreglo) #<__main__.Array object at 0x0000020954591FA0>
    
    # Me retorna los espacios vacíos del array, los hoyos de para las plantas.
    print(arreglo.items) #[None, None, None]
    

    # Para llenar los datos debo usar .items o lista vacía,
    # para poder acceder a los elementos del arreglo.
    # Aquí evidencio como se llenan los datos.
    # [1, None, None]
    # [1, 2, None]
    # [1, 2, 3]
    for i in range(0, len(arreglo.items)):
        arreglo.items[i] = i + 1
        print(arreglo.items)
       


    # Usando los métodos que creamos para el arreglo.
    length = arreglo.items.__len__()
    print("El arreglo tiene como largo : "+ str(length))

    # Retorno un str
    strings = arreglo.items.__str__()
    print(type(strings))

    # Creo un Objeto lista iterador y lo recorro con next
    iterador = arreglo.items.__iter__()
    print(iterador)
    print(next(iterador))
   
    # Consigo el elemnto en la posición 1
    consigo_elemento = arreglo.items.__getitem__(1)
    print(consigo_elemento)

    # Ingreso un elemento específicado.
    arreglo.items.__setitem__(1,"Arreglo terminado!")
    print(arreglo.items) 
```

#### Instancia del padre

```python
'''
1. Crear un array 
2. Incorporar un método para poblar sus slots con números aleatorios o secuenciales
3. Incluye un método que sume todos lo valores del array.

'''
from arrays_custom import Array
import random as r



class Array_challenge(Array):

    def __random_sequential__(self,randomize=False, sequential=False):
        self.randomize = randomize
        self.sequential = sequential
        

        # Busco el largo para tener un límite claro.
        # Si el parámetro randomize es True entonces
        # Creo números aleatorios usando l como número base
        # Uso setitems para insertar elementos por cada elemento
        l = self.items.__len__()
        if randomize:
            for index in range(0,l):
                number = r.randrange(l**l)
                self.items.__setitem__(index, number)

        if sequential:
            # Aquí se puede optimizar para un algoritmo de 
	    # ordenamiento más efectivo, en función del tamaño del arreglo
	    # Ver por ejemplo MergeSort.
            self.items = sorted(self.items)

        return self.items

    def __sum_array__(self):
        return sum(self.items)

if __name__ == '__main__':
    solution = Array_challenge(3)
    print("""Solución:

    1. Creo números aleatorios y/u ordenados dependiendo de los dos parámetros (True)
    Así obtengo números aleatorios, o números aleatorios y ordenados secuencialmente.

    2. Retorno la suma de os valores aleatorios.
    """)

    print("1: ", solution.__random_sequential__(True,True))
    print("2: ", solution.__sum_array__())
```


### Nodes y singly linked list

Las estructuras linked consisten en nodos conectados a otros, los más comunes son sencillos o dobles. No se accede por índice sino por recorrido. Es decir se busca en la lista de nodos hasta encontrar un valor.

- **Data**: Será el valor albergado en un nodo.
- **Next**: Es la referencia al siguiente nodo en la lista
- **Previous**: Será el nodo anterior.
- **Head**: Hace referencia al primer nodo en la lista
- **Tail**: Hace referencia al último nodo.
**¿Cómo funciona en memoria los Linked Estructures?**

Estas estructuras de datos hablan de `nodos/datos` repartidos en memoria, diferentes a los arrays que son contiguos. Los nodos se conectan a diferentes espacios en memoria, podemos acceder a los datos saltando en memoria, siendo mucho más ágil. Los nodos nos sirven para crear otras estructuras más complejas, como **Stacks**, **Queues**, las llamadas pilas y colas. Es posible optimizar partes del código usando nodos.

**Double Linked Structure.**
Estos hacen que el nodo haga referencia al siguiente nodo y al anterior, es decir nos va a permitir ir en ambas direcciones. También nos permitirá realizar “formas” y contextos circulares.

- El ejemplo clave aquí será función de `ctrl+z` y `ctrl+y` Estas opciones nos permiten hacer y deshacer un proceso en Windows.

- El historial del navegador también es un buen ejemplo al permitirnos navegar entre el pasado y el presente.

**Nota**: Los **linked list** tienen una desventaja importante, si la lista crece mucho será más costoso computacionalmente recorrer toda la lista.
Es importante saber cuando usarlas y cuando no.

### Crear Nodos.
Cada nodo almacenará un valor y cada nodo tiene un puntero que llevará a otro nodo con otro valor y así obtener los datos allí almacenados.
Es muy útil al tener infromación dispersa en memoria y cuando queremos que sean consultas ágiles, es importante entender que los nodos son la base para implementaciones más elaboradas de estructuras de datos, **Stacks**, **Qeues**, **Deque**, **Doubly**, **Singly List**, **Circular list**, **Graphs**.

Cada estructura de datos servirá para un propósito dentro de un contexto, por ejemplo los grafos acíclicos, donde se usan para sistemas de recomendaciones al mostrar las relaciones entre objetos o representar los tipos de redes que se forman entre nodos. Para crear un nodo:

- Creamos una clase `Node`
- Referimos valores mediante argumentos de instancias.
- Unimos los nodos iterando entre referencias.
Este script tiene como propósito crear nodos.

**Constructor:**

data= El dato del nodo.
next= está por defecto en None, porque en una serie de nodos el +ultimo te lleva a ninguna parte.

```python
class Node():
    def __init__(self, data, _next=None):
        # Atributos
        self.data = data
        self.next = _next
```

A continuación la instancia de la clase Node

```python
from node import Node

node1 = None
node2 = Node("A", None)
# Los nodos al ser secuanciales permiten refrencias a cualquier lugar.
node3 = Node("B", node2)

def show_relations():
    '''
    Este script perfectamente puede ser una función que recibe al nodo como parámetro pythony llamo para mostrar las
    relaciones.

    '''
    print("Esto es la ubicación en memoria de los nodos")
    print(node2) #<node.Node object at 0x0000022017FEAD30>
    print(node3) #<node.Node object at 0x0000022017FEAD90>
    print("Esto es el dato y muestra la relación entre nodos")
    print(node2.data,"-->", node2.next) #A --> None
    print(node3.data,"-->", node3.next) #B --> <node.Node object at 0x0000015356D6AD30>
    print("El siguiente dato del nodo es:")
    # Se refiere al nodo que está conectado y luego al dato que este contiene.
    print(node3.next.data) #'A'
    print("Creando el nodo1 y mostrando datos y relacion con nodo3 obtenemos: ")
    # Asignar una propiedad a un elemento para volverlo nodo.
    # Al intanciar la clase con una relación estamos ligando los nodos.
    node1= Node("C", node3)
    print(node1.data,"-->", node1.next)#C --> <node.Node object at 0x0000022017FEAD90>

def create_nodes():
    print("Creo nodos que se asignan a un solo valor en memoria, en este caso a node2: ")
    for node in range(5): #n --> <node.Node object at 0x000001B2AD991FD0>
        head = Node(node, node2)
        print(head.data,"-->", head.next)


def run():
    show_relations()
    create_nodes()

if __name__ == '__main__':
    run()
```

### Lista enlazada lineal simple (singly linked list) – Implementación en Python
![Node](https://programacionycacharreo.files.wordpress.com/2018/10/singly_linked_list.jpeg "Node")

Una lista enlazada (linked list en inglés), es un tipo de estructura de datos compuesta de nodos. Cada nodo contiene los datos de ese nodo y enlaces a otros nodos.

Se pueden implementar distintos tipos de listas enlazadas. En este post vamos a implementar una lista enlazada lineal simple (singly linked list). En este tipo de listas, cada nodo contiene sus datos y un enlace al siguiente nodo. Además la lista tendrá un método para contar el número de elementos de la lista, un método para insertar un elemento en la lista y un método para eliminar un elemento de la lista.

En primer lugar, definimos una clase que va a ser la clase Node. Los objetos de esta contendrán sus propios datos y un enlace al siguiente elemento de la lista:

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

A continuación definimos la clase de la lista SinglyLinkedList, que contiene el primer elemento de la lista:

```python
class SinglyLinkedList:
    def __init__(self, head):
        self.head = head
```

El método que cuenta los elementos de la lista, length(), primero comprueba que la lista no esté vacía, y luego recorre todos los elementos de la lista incrementando un contador por cada elemento. Al final devuelve el contador:

```python
 def length(self) -> int: 
        current = self.head
        if current is not None:
            count = 1

            while current.next is not None:
                count += 1
                current = current.next
            return count
        else:
            return 0
```

El siguiente método, insert(datos, posición), inserta un elemento tras la posición indicada. Si se indica la posición 0, el nuevo elemento pasa a ser la cabecera de la lista. En esta implementación, si la posición que se pasa como argumento excede el tamaño de la lista,el elemento se inserta al final:

```python
def insert(self, data, position):
       new_node = Node(data)

       if position == 0:
           new_node.next = linked_list.head
           linked_list.head = new_node
       else:
           current = linked_list.head
           k = 1
           while current.next is not None and k < position:
               current = current.next
               k += 1
           new_node.next = current.next
           current.next = new_node
```
El método delete(posición) borra el elemento en la posición pasada como parámetro. Si es el primer elemento la lista de la cabeza pasa a ser el segundo elemento. Si se encuentra el elemento en la lista y se borra devolvemos True, en caso contrario devolvemos False:

```python
def delete(self, position):
       if position != 1:
           current = self.head
           k = 1
           while current.next is not None and k < position - 1:
               current = current.next
               k += 1
           if current.next is not None:
               current.next = current.next.next
               return True
           else:
               return False
       else:
           self.head = self.head.next
           return True
```

Creamos la lista

```python
linked_list = SinglyLinkedList(Node(1))
```

Rellenamos la lista

```python
 for i in range(2,10):
        linked_list.insert(i, i-1)
```

Insertamos un elemento

```python
   linked_list.insert(999,3)
```

Eliminamos un elemento

```python
  linked_list.delete(6)
```

Mostramos la lista

```python
    current = linked_list.head
    while current is not None:
		print(current.data)
        current = current.next
```
 # **Nota**: Crear un archivo en visual se usa el codigo `code <nombre_del_archivo>` ejemplo `code prueba.py`

 Prara ingresar en la teerminal se usa `py`

```python
from node import Node

# * Creación de los nodos enlazados (linked list)
head = None
for count inrange(1,6):
    head = Node(count, head)

# * Recorrer e imprimir valores de la lista
probe = head
print("Recorrido de la lista:")
while probe != None:
    print(probe.data)
    probe = probe.next
Recorrido dela lista:
5
4
3
2
1
# * Busqueda de un elemento
probe = head
target_item = 2
while probe != Noneand target_item != probe.data:
    probe = probe.next

if probe != None:
    print(f'Target item {target_item} has been found')
else:
    print(f'Target it {target_item} is not in the lisked list')

Target item 2 has been found
# * Remplazo de un elemento
probe = head
target_item = 3
new_item = "Z"

while probe != None and target_item != probe.data:
    probe = probe.next

if probe != None:
    probe.data = new_item
    print(f"{new_item} replace the old value in the node number {target_item}")
else:
    print(f"The target item {target_item} is not in the linked list")

Z replace the old value in the node number 3
The target item 3 is not in the linked list

Recorrido de la lista:
5
4
Z
2
1
# * Insertar un nuevo elemento/nodo al inicio(head)
head = Node("F", head)
Recorrido dela lista:
F
5
4
Z
2
1
# * Insertar un nuevo elemento/nodo al final(tail)
new_node = Node("K")
if head is None:
    head = new_node
else:
    probe = head
    while probe.next != None:
        probe = probe.next
    probe.next = new_node
Recorrido dela lista:
F
5
4
Z
2
1
K
# * Eliminar un elmento/nodo al inicio(head)
removed_item = head.data
head = head.next
print("Removed_item: ",end="")
print(removed_item)

Removed_item: F

Recorrido dela lista:
5
4
Z
2
1
K
# * Eliminar un elmento/nodo al final(tail)
removed_item = head.data
if head.nextisNone:
    head = None
else:
    probe = head
    while probe.next.next != None:
        probe = probe.next
    removed_item = probe.next.data
    probe.next = None

print("Removed_item: ",end="")
print(removed_item)
Removed_item: K

Recorrido dela lista:
5
4
Z
2
1
# * Agregar un nuevo elemento/nodo por "indice" inverso(Cuenta de Head - Tail)
# new_item = input("Enter new item: ")
# index = int(input("Enter the position to insert the new item: "))
new_item = "10"
index = 3

if head is None orindex <= 0:
    head = Node(new_item, head)
else:
    probe = head
    whileindex > 1 and probe.next != None:
        probe = probe.next
        index -= 1
    probe.next = Node(new_item, probe.next)

# * Agregar un nuevo elemento/nodo por "indice" inverso(Cuenta de Head - Tail)

Recorrido dela lista:
5
4
Z
10
2
1
# * Eliminar un nuevo elemento/nodo por "indice" inverso(Cuenta de Head - Tail)
index = 3

if head is None or index <= 0:
    removed_item = head.data
    head = head.next
    print(removed_item)
else:
    probe = head
    whileindex > 1 and probe.next.next != None:
        probe = probe.next
        index -= 1
    removed_item = probe.next.data
    probe.next = probe.next.next

    print("Removed_item: ",end="")
    print(removed_item)

Removed_item: 10
Recorrido dela lista:
5
4
Z
2
1
```

### listas enlazadas circulares
Las listas enlazadas circulares son un tipo de lista enlazada en la que el último nodo apunta hacia el headde la lista en lugar de apuntar None. Esto es lo que los hace circulares.

**Las listas enlazadas circulares tienen bastantes casos de uso interesantes:**
Dar la vuelta al turno de cada jugador en un juego multijugador
Gestionar el ciclo de vida de la aplicación de un sistema operativo determinado
Implementando un montón de Fibonacci
Así es como se ve una lista enlazada circular:

![](https://files.realpython.com/media/Group_22.cee69a15dbe3.png)

Una de las ventajas de las listas enlazadas circulares es que puede recorrer toda la lista comenzando en cualquier nodo. Dado que el último nodo apunta al headde la lista, debe asegurarse de dejar de atravesar cuando llegue al punto de partida. De lo contrario, terminarás en un bucle infinito.

En términos de implementación, las listas enlazadas circulares son muy similares a la lista enlazada individualmente. La única diferencia es que puede definir el punto de partida cuando recorre la lista:

```python
class CircularLinkedList:
    def __init__(self):
        self.head = None

    def traverse(self, starting_point=None):
        if starting_point is None:
            starting_point = self.head
        node = starting_point
        while node is not None and (node.next != starting_point):
            yield node
            node = node.next
        yield node

    def print_list(self, starting_point=None):
        nodes = []
        for node in self.traverse(starting_point):
            nodes.append(str(node))
        print(" -> ".join(nodes))
```

Atravesar la lista ahora recibe un argumento adicional starting_point, que se usa para definir el inicio y (debido a que la lista es circular) el final del proceso de iteración. Aparte de eso, gran parte del código es el mismo que teníamos en nuestra LinkedListclase.

```python 
>>> from node import Node
>>> index = 1
>>> new_item = "ham"
>>> head = Node(None, None)
>>> head.next = head
>>> probe = head
>>> while index > 0 and probe.next != head:
...     probe = probe.next
...     index -= 1
...
>>> probe.next = Node(new_item, probe.next)
>>> print(probe.next.data)
ham
```
### Cómo utilizar listas doblemente enlazadas
Las listas doblemente enlazadas se diferencian de las listas enlazadas individualmente en que tienen dos referencias:

- El previous campo hace referencia al nodo anterior.
- El next campo hace referencia al siguiente nodo.
El resultado final se ve así:
![](https://files.realpython.com/media/Group_23.a9df781f6087.png)

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.previous = None
```
Este tipo de implementación le permitiría atravesar una lista en ambas direcciones en lugar de solo atravesar usando next. Puede utilizar next para avanzar y previous retroceder.

En términos de estructura, así es como se vería una lista doblemente enlazada:

![](https://files.realpython.com/media/Group_21.7139fd0c8abb.png)

```python
>>> from double_linked_list import Node, TwoWayNode
>>> head = TwoWayNode(1)
>>> tail = head
>>> for data in range(2, 6):
...     tail.next = TwoWayNode(data, tail)
...     tail = tail.next
...
>>> probe = tail
>>> while probe != None:
...     print(probe.data)
...     probe = probe.previous
5
4
3
2
1
```
### Pilas (Stacks)

[Metodos de Stacks](https://drive.google.com/file/d/1kvRoA5qXGoUY8i1Mnx6T-7BphTTn109F/view?usp=drive_link "Metodos de Stacks")

Las pilas (stacks) son una estructura de datos donde tenemos una colección de elementos, y sólo podemos hacer dos cosas:

- añadir un elemento al final de la pila
- sacar el último elemento de la pila

Una manera común de visualizar una pila es imaginando una torre de panqueques, donde una vez que ponemos un panqueque encima de otro, no podemos sacar el anterior hasta que se hayan sacado todos los que están encima.

A pesar de su simplicidad, las pilas son estructuras relativamente comunes en ciertas áreas de la computación, en especial para implementar o simular evaluación de expresiones, recursión, scope, …

### LIFO (Last In First Out)
Las pilas son estructuras de tipo LIFO, lo cual quiere decir que el último elemento añadido es siempre el primero en salir.

De alguna forma, podemos decir que una pila es como si fuera una lista o array, en el sentido que es una colección, pero a diferencia de los arrays y otras colecciones, en las pilas solo accedemos al elemento que esté “encima de la pila”, el último elemento. Nunca manipulamos ni accedemos a valores por debajo del último elemento.

### crear u stack

```python
>>> from stack import Stack
>>> food = Stack()
>>> food.push("egg")
>>> food.push("ham")
>>> food.push("spam")
>>> food.pop()
'spam'
>>> food.peek()
'ham'
>>> food.clear()
>>> food.peek()
'the stack is empty'
```
[Metodo stack](https://geekflare.com/es/python-stack-implementation/ "Metodo stack")
### Queues

[metodo de queue](https://drive.google.com/file/d/1EHv6Y4hjbMK7gkEf4hzvvKddpkGKm8UD/view?usp=drive_link "metodo de queue")

[Queue](https://geekflare.com/es/python-queue-implementation/ "Queue")

En una cola, los elementos se agregan al final y se eliminan desde el frente. Esto se asemeja a una fila en la vida real, donde la primera persona en llegar es la primera en ser atendida.

**Las operaciones principales en una cola son**:
**Enqueue (encolar)**:
- Agrega un elemento al final de la cola.
**Dequeue (desencolar)**:
- Elimina y devuelve el elemento que está en el frente de la cola.
**Front (frente)**:
- Devuelve el elemento que está en el frente de la cola sin eliminarlo.
**Rear (trasero)**:
- Devuelve el elemento que está al final de la cola sin modificar la cola.
**Is Empty (está vacía)**:
- Verifica si la cola está vacía o no.

### Ejercicio de clase

```python
>>> from list_based_queue import ListQueue
>>> food = ListQueue()
>>> food.enqueue("egg")
>>> food.enqueue("ham")
>>> food.enqueue("spam")
>>> food.traverse()
spam
ham
egg
>>>
```

### Queue basada en dos stacks

```python
>>> from stack_based_queue import Queue
>>> numbers = Queue()
>>> numbers.enqueue(5)
>>> numbers.enqueue(6)
>>> numbers.enqueue(7)
>>> print(numbers.inboud_stack)
[5, 6, 7]
>>> print(numbers.outbound_stack)
[]
>>> numbers.dequeue()
5
>>> print(numbers.inboud_stack)
[]
>>> print(numbers.outbound_stack)
[7, 6]
>>> numbers.dequeue()
6
>>> print(numbers.outbound_stack)
[7]
>>> numbers.dequeue()
7
>>> print(numbers.outbound_stack)
[]
>>>
```

### Queue basada en dos stacks

```python
>>> from stack_based_queue import Queue
>>> numbers = Queue()
>>> numbers.enqueue(5)
>>> numbers.enqueue(6)
>>> numbers.enqueue(7)
>>> print(numbers.inboud_stack)
[5, 6, 7]
>>> print(numbers.outbound_stack)
[]
>>> numbers.dequeue()
5
>>> print(numbers.inboud_stack)
[]
>>> print(numbers.outbound_stack)
[7, 6]
>>> numbers.dequeue()
6
>>> print(numbers.outbound_stack)
[7]
>>> numbers.dequeue()
7
>>> print(numbers.outbound_stack)
[]
>>>
```

### Queue Basada en nodos

No pensé que fuéramos a ver aquí Big O, pero a la par del curso voy leyendo un libro llamado **Data Structure and Algorithmic Thinking with Python** y al principio viene esta bella tabla.

![Data Structure](https://static.platzi.com/media/user_upload/Captura%20desde%202022-10-11%2001-23-47-d0da3528-d28a-4562-a9b5-8fcdde707fb8.jpg "Data Structure")

```python
>>> from node_based_queue import Queue
>>> food = Queue()
>>> food.enqueue("eggs")
>>> food.enqueue("ham")
>>> food.enqueue("spam")
>>> food.head
<node_based_queue.TwoWayNode object at 0x00000266A1A2CFD0>
>>> food.head.data
'eggs'
>>> food.head.next.data
'ham'
>>> food.tail.data
'spam'
>>> food.tail.previous.data
'ham'
>>> food.count
3
>>> food.dequeue()
'eggs'
>>> food.head.data
'ham'
>>>
```

### Reto: simulador de playlist musical

```python
 from music_player import Track, MediaPlayerQueue
>>> track1 = Track("Highway to hell")
>>> track2 = Track("Go!")
>>> track3 = Track("Light years")
>>> track4 = Track("Heartbreaker")
>>> track5 = Track("Breath me")
>>> track6 = Track("How to dissappear completery")
>>> media_player = MediaPlayerQueue()
>>> media_player.add_track(track1)
>>> media_player.add_track(track2)
>>> media_player.add_track(track3)
>>> media_player.add_track(track4)
>>> media_player.add_track(track5)
>>> media_player.add_track(track6)
>>> media_player.play()
Count: 6
Now playing Highway to hell
Now playing Go!
Now playing Light years
Now playing Heartbreaker
Now playing Breath me
Now playing How to dissappear completery
```