# Curso de ECMAScript: Historia y Versiones de JavaScript

Se inicia git con `git init`

luego par ainiciar el proyecto se utiliza `npm init` y seguimos los siguientes pasos como version, autor, etc.

```linux
This utility will walk you through creating a package.json file.
It only covers the most common items, and tries to guess sensible defaults.

See `npm help init` for definitive documentation on these fields
and exactly what they do.

Use `npm install <pkg>` afterwards to install a package and
save it as a dependency in the package.json file.

Press ^C at any time to quit.
package name: (curso-ecmascript)
version: (1.0.0)
description: (index.js)
entry point: (index.js)
test command:
git repository:
keywords: javascript, ecmascript, node.js
author: Mario Celis <celioso1@hotmail.com>
license: (ISC) MIT
About to write to C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoDeECMAScriptHistoriaYVersionesDeJavaScript\js\curso-ecmascript\package.json:

{
  "name": "curso-ecmascript",
  "version": "1.0.0",
  "description": "(index.js)",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [
    "javascript",
    "ecmascript",
    "node.js"
  ],
  "author": "Mario Celis <celioso1@hotmail.com>",
  "license": "MIT"
}


Is this OK? (yes)
```

### ES6: let y const, y arrow functions

En **ECMAScript 6** (ES6 o ES2015) fueron publicadas varias características nuevas que dotaron de gran poder al lenguaje, dos de estas son una nueva forma de declaración de variables con `let` y `const`, y funciones flechas.

**La nueva forma para declarar variables con let y const**

Hasta ahora aprendiste a declarar variables con `var`, sin embargo, a partir de la especificación de ES6 se agregaron nuevas formas para la declaración de variables.

Las nuevas palabras reservadas `let` y const resuelven varios problemas con `var` como el scope, hoisting, variables globales, re-declaración y re-asignación de variables.

**Variables re-declaradas y re-asignadas**

*La re-declaración es volver a declarar una variable, y la re-asignación es volver a asignar un valor*. Entonces cada palabra reservada tiene una forma diferente de manejar variables:

- Una variable declarada con `var` puede ser re-declarada y re-asignada.
- Una variable declarada con `let` puede ser re-asignada, pero no re-declarada.
- Una variable declarada con `const` no puede ser re-declarada, ni re-asignada. Su declaración y asignación debe ser en una línea, caso contrario habrá un error.
En conclusión, si intentas re-declarar una variable declarada con let y const habrá un error de “variable ya declarada”; por otro lado, si intentas re-asignar una variable declarada con const existirá un “error de tipo”.

En los demás casos, JavaScript lo aceptará como válidos, algo problemático con `var`, por eso deja de utilizarlo.

**Ejemplo de declaración y asignación en diferentes líneas**

```javaScript
// Declaración de variables
var nameVar 
let nameLet

// Asignación de variables
nameVar= "soy var"
nameLet = "soy let"
```
Aunque realmente lo que pasa si no asignas un valor en la declaración, JavaScript le asigna un valor `undefined`.

**Ejemplo de declarar y asignar con const en diferentes líneas de código**
```javaScript
const pi  // SyntaxError: Missing initializer in const declaration.
pi = 3.14
```
**Ejemplo de re-declaración de variables**

```javaScript
var nameVar = "soy var"
let nameLet = "soy let"
const nameConst = "soy const"

// Re-declaración de variables
var nameVar = "var soy" 
console.log(nameVar) // 'var soy'

let nameLet = "let soy" // SyntaxError: Identifier 'nameLet' has already been declared.

const nameConst = "const soy" //SyntaxError: Identifier 'nameConst' has already been declared.
```
Ejemplo de re-asignación de variables
```javaScript
var nameVar = "soy var"
let nameLet = "soy let"
const nameConst = "soy const"

// Re-asignación de variables
nameVar = "otro var"
console.log(nameVar) // 'otro var'

nameLet = "otro let"
console.log(nameVar) // otro let'

nameConst = "otro const" //TypeError: Assignment to constant variable.
```
Ten en cuenta que los errores pararán la ejecución de tu programa.

**Scope**
En el tema del scope, `let` y `const` tienen un scope de bloque y `var` no.
```javaScript
{
var nameVar = "soy var"
let nameLet = "soy let"
}

console.log(nameVar) // 'soy var'
console.log(nameLet) // ReferenceError: nameLet is not defined
```
Todo el tema de Scope tiene su propio curso que deberías haber tomado: Curso de Closures y Scope en JavaScript

**Objeto global**

En variables globales, let y constno guardan sus variables en el objeto global (window, global o globalThis), mientras que var sí los guarda.
```javaScript
var nameVar = "soy var"
let nameLet = "soy let"
const nameConst = "soy const"

globalThis.nameVar   // 'soy var'
globalThis.nameLet   // undefined
globalThis.nameConst  // undefined
```
Esto es importante para que no exista re-declaración de variables.

**Funciones flecha**
Las funciones flecha (*arrow functions*) consiste en una **función anónima** con la siguiente estructura:
```javaScript
//Función tradicional
function nombre (parámetros) {
    return valorRetornado
}

//Función flecha
const nombre = (parámetros) => {
    return valorRetornado
}
```
Se denominan función flecha por el elemento `=>` en su sintaxis.

**Omitir paréntesis en las funciones flecha**
Si existe un solo parámetro, puedes omitir los paréntesis.
```javaScript
const porDos = num => {
    return num * 2
}
```
**Retorno implícito**
Las funciones flecha tienen un retorno implícito, es decir, se puede omitir la palabra reservada `return`, para que el **código sea escrito en una sola línea**.
```javaScript
//Función tradicional
function suma (num1, num2) {
    return num1 + num2
}

//Función flecha
const suma = (num1, num2) => num1 + num2
```
Si el retorno requiere de más líneas y aún deseas utilizarlo de manera implícita, deberás envolver el cuerpo de la función entre paréntesis.

```javaScript
const suma = (num1, num2) => (
    num1 + num
)
```
[Funciones Flecha - javaScript | MDN](https://developer.mozilla.org/es/docs/Web/JavaScript/Reference/Functions/Arrow_functions "Funciones Flecha - javaScript | MDN")

[Curso de Closures y Scope en JavaScript](https://platzi.com/cursos/javascript-closures-scope/ "Curso de Closures y Scope en JavaScript")

### ES6: strings

Las **plantillas literales** (*template literals*) consisten en crear cadenas de caracteres que puedan contener variables sin utilizar la concatenación. Esto mejora la legibilidad y la mantenibilidad del código.

**Concatenación de caracteres**

Antes de ES6, si querías crear una cadena larga o un mensaje elaborado, debías utilizar la concatenación. La concatenación de caracteres consiste en unir uno o varios caracteres, como si fuera una suma.
```JavaScript
var nombre = "Andres"
var edad = 23
var mensaje = "Mi nombre es " + nombre + " y tengo " + edad + " años."

console.log(mensaje)
// 'Mi nombre es Andres y tengo 23 años.'
```

Esto trae varios **problemas en la legibilidad y mantenibilidad del código**. Se convierte cada vez más complejo en mensajes más extensos o el estar pendiente de agregar espacios antes o después de cada variable concatenada.

**Cómo utilizar las plantillas literales**
Las plantillas literales añadidas en ES6, se emplea el caracter [acento grave](https://elcodigoascii.com.ar/codigos-ascii/acento-grave-codigo-ascii-96.html "acento grave") **( ` )**, que no es una comilla simple **( ’)**, para envolver el mensaje.Para incluir las variables se utiliza la sintaxis `${variable}`.
```JavaScript
var nombre = "Andres"
var edad = 23

var mensaje = `Mi nombre es ${nombre} y tengo ${edad} años.`

console.log(mensaje)
// 'Mi nombre es Andres y tengo 23 años.'
```
De esta manera el código es más legible y que pueda mantenerse.

**Plantilla multilínea**
La plantilla multilínea consiste en crear mensajes que contengan varias líneas separadas entre sí, utilizando las [plantillas literales](https://platzi.com/clases/1815-ecmascript-6/26121-default-params-y-concatenacion/ "plantillas literales"). Antes de ES6, la forma de crear una plantilla multilínea era agregar `\n` al `string`.
```JavaScript
var mensaje = "Línea 1 \n" + "línea 2"

console.log(mensaje)
// 'Línea 1
// línea 2'
```

Con ES6 solamente necesitas utilizar las plantillas literales.
```JavaScript
const mensaje = `Línea 1
línea 2`

console.log(mensaje)
// 'Línea 1
// línea 2'
```

### ES6: parámetros por defecto

Los **parámetros por defecto** (*default params*) **consisten en establecer un valor por defecto a los parámetros de una función**, para asegurar que el código se ejecute correctamente en el caso de que no se establezcan los argumentos correspondientes en la invocación de la función.

**Cómo era utilizar valores por defecto antes de ES6**

Tal como puedes ver en el siguiente código, la función sumar recibe dos parámetros y retorna el valor total. Sin embargo, si alguien no decide poner alguno o todos los parámetros necesarios, pues que el programa no funcionará correctamente.

```JavaScript
function sumar(number1, number2){
  return number1 + number2
}

sumar(3,4) // 7
sumar(3)   // NaN  
sumar()    // NaN
```
Antes de ES6, se debía establecer una variable y utilizar el operador OR `( ||)` con el valor por defecto necesario. El caracter guion bajo `( _)` lo utilizo para diferenciar el parámetro de la función de la variable declarada.
```JavaScript
function sumar(number1, number2){
  var _number1 = number1 || 0
  var _number2 = number2 || 0
  
  return _number1 + _number2
}

sumar(3,4) // 7
sumar(3)   // 3
sumar()    // 0
```
**Cómo utilizar los parámetros por defecto**
Con los parámetros por defectos añadidos en ES6, eliminamos las declaraciones para mejorar la legibilidad y el mantenimiento del código de la siguiente manera:

```JavaScript
function sumar(number1 = 0, number2 = 0){
  return number1 + number2
}

sumar(3,4) // 7
sumar(3)   // 3
sumar()    // 0
```
Puedes utilizar cualquier valor, siempre que sea necesario.

**Posición de los parámetros por defecto**

Si obligatoriamente necesitas el valor como argumento, ten presente que los parámetros por defecto siempre deben estar en las posiciones finales.

```JavaScript
// ❌ Mal
function sumar(number1 = 0, number2) { ... }
sumar(3)   // number1 = 3 y number2 = undefined 

// ✅ Bien
function sumar(number1, number2 = 0) { ... }
sumar(3)   // number1 = 3 y number2 = 0
```

### ES6: asignación de desestructuración

La desestructuración (destructuring) consiste en extraer los valores de arrays o propiedades de objetos en distintas variables.

Desestructuración de objetos
La desestructuración de objetos implica extraer las propiedades de un objeto en variables. Mediante el mismo nombre de la propiedad del objeto con la siguiente sintaxis:

```JavaScript
const objeto = { 
    prop1: "valor1",
    prop2: "valor2",
    ... 
} 

// Desestructuración
const { prop1, prop2 } = objeto
```
Antes de ES6, necesitabas acceder al objeto con la notación punto o corchetes por cada propiedad que se necesita y asignar ese valor a una variable diferente.
```JavaScript
var usuario = { nombre: "Andres", edad: 23, plataforma: "Platzi" }

var nombre = usuario.nombre
var edad = usuario.edad
var plataforma = usuario["plataforma"]

console.log(nombre)  // 'Andres' 
console.log(edad)  // 23
console.log(plataforma)  // 'Platzi'
```
Con la desestructuración puedes realizar lo mismo, pero en una sola línea, provocando que el código seas más legible y mantenible.

```JavaScript
const usuario = { nombre: "Andres", edad: 23, plataforma: "Platzi" }

const { nombre, edad, plataforma } = usuario

console.log(nombre)  // 'Andres' 
console.log(edad)  // 23
console.log(plataforma)  // 'Platzi'
```
**Cambiar el nombre de las variables con desestructuración**
Si no te agrada el nombre de la propiedad del objeto, puedes cambiarlo utilizando la siguiente sintaxis:
```JavaScript
const objeto = { prop1: "valor1", prop2: "valor2", ... } 

// Desestructuración
const { prop1: newProp1, prop2: newProp2 } = objeto
```
Por ejemplo:
```JavaScript
const usuario = { nombre: "Andres", edad: 23, plataforma: "Platzi" }

const { nombre: name, edad: age, plataforma: platform } = usuario

console.log(name)  // 'Andres' 
console.log(age)  // 23
console.log(platform)  // 'Platzi'

console.log(nombre)   // Uncaught ReferenceError: nombre is not defined
```
**Desestructuración en parámetros de una función**
Si envías un objeto como argumento en la invocación a la declaración de una función, puedes utilizar la desestructuración en los parámetros para obtener los valores directamente. Ten en cuenta que el nombre debe ser igual a la propiedad del objeto.
```JavaScript
const usuario = { nombre: "Andres", edad: 23, plataforma: "Platzi" }

function mostrarDatos( { nombre, edad, plataforma } ){
    console.log(nombre, edad, plataforma) 
}

mostrarDatos(usuario) // 'Andres', 23, 'Platzi'
```
**Desestructuración de arrays**
La desestructuración de arrays consiste en extraer los valores de un array en variables, utilizando la **misma posición del array** con una sintaxis similar a la desestructuración de objetos.
```JavaScript
const array = [ 1, 2, 3, 4, 5 ]

// Desestructuración
const [uno, dos, tres ] = array

console.log(uno) // 1
console.log(dos) // 2
console.log(tres) // 3
```
Desestructuración para valores retornados de una función
Cuando una función retorna un array, puedes guardarlo en una variable. Por ende, puedes utilizar la desestructuración para utilizar esos valores por separado de manera legible.

En el siguiente ejemplo, la función useState retorna un array con dos elementos: un valor y otra función actualizadora.

```JavaScript
function useState(value){
    return [value, updateValue()]
}

//Sin desestructuración 
const estado = useState(3)
const valor = estado[0]
const actualizador = estado[1]

//Con desestructuración 
const [valor, actualizador] = useState(3)
```
**Lo que puedes hacer con desestructuración, pero no es recomendable**
Si necesitas un elemento en cierta posición, puedes utilizar la separación por comas para identificar la variable que necesitas.

```JavaScript
const array = [ 1, 2, 3, 4, 5 ]

const [ ,,,,  cinco ] = array

console.log(cinco) // 5
```
Como los arrays son un tipo de objeto, puedes utilizar la desestructuración de objetos mediante el **índice y utilizando un nombre para la variable**.
```JavaScript
const array = [ 1, 2, 3, 4, 5 ]

const {4: cinco} = array

console.log(cinco) // 5
```

### ES6: spread operator

El **operador de propagación** (spread operator), como su nombre lo dice, consiste en **propagar los elementos de un iterable**, ya sea un array o string utilizando tres puntos (...) dentro de un array.

```JavaScript
// Para strings
const array = [ ..."Hola"]    // [ 'H', 'o', 'l', 'a' ]

// En arrays
const otherArray = [ ...array]   //[ 'H', 'o', 'l', 'a' ]
```
También se utiliza para **objetos**, pero esta característica fue añadida en versiones posteriores de ECMAScript y es denominada [propiedades de propagación](https://platzi.com/clases/3504-ecmascript-nuevo/51771-expresiones-regulares/ "propiedades de propagación").

**Cómo copiar arrays utilizando el operador de propagación**

Para realizar una copia de un array, deberás tener cuidado de la **referencia en memoria**. Los arrays se guardan en una referencia en la memoria del computador, al crear una copia, este tendrá la misma referencia que el original. Debido a esto, **si cambias algo en la copia, también lo harás en el original**.

```JavaScript
const originalArray = [1,2,3,4,5]
const copyArray = originalArray
copyArray[0] = 0

originalArray // [0,2,3,4,5]
originalArray === copyArray  // true
```
Para evitar esto, utiliza el operador de propagación para crear una copia del *array* que utilice una **referencia en memoria diferente al original**.

```JavaScript
const originalArray = [1,2,3,4,5]
const copyArray = [...originalArray]
copyArray[0] = 0

originalArray // [1,2,3,4,5]
copyArray // [0,2,3,4,5]
originalArray === copyArray  // false
```
**Cómo unir arrays y añadir elementos con el operador de propagación**

Para unir dos arrays con el operador de propagación, simplemente debes separarlos por comas en un *array*.

```JavaScript
const array1 = [1,2,3]
const number = 4
const array2 = [5,6,7]

const otherArray = [ ...array1, number, ...array2 ]

otherArray // [1,2,3,4,5,6,7]
```
**Cuidado con la copia en diferentes niveles de profundidad**
El operador de propagación sirve para producir una copia en **un solo nivel de profundidad**, esto quiere decir que si existen objetos o *arrays* dentro del *array* a copiar. Entonces los sub-elementos en cada nivel, tendrán la **misma referencia de memoria en la copia y en el original**.

```JavaScript
const originalArray = [1, [2,3] ,4,5]
const copyArray = [...originalArray]

originalArray[1] === copyArray[1] // true
```
La manera de solucionar es más compleja, tendrías que emplear el operador de propagación para cada elemento en cada nivel de profundidad.

Sin embargo, recientemente salió una forma de producir una copia profunda con [StructuredClone](https://developer.mozilla.org/en-US/docs/Web/API/structuredClone "StructuredClone"), aunque es una característica muy reciente, así que revisa que navegadores tienen soporte.

```JavaScript
const originalArray = [1, [2,3] ,4,5]
const copyArray = structuredClone(originalArray)

originalArray === copyArray  // false
originalArray[1] === copyArray[1] // false
```
Este comportamiento también sucede para objetos dentro de otros objetos, u objetos dentro de arrays.

**Parámetro rest**

El parámetro rest consiste en **agrupar el residuo de elementos** mediante la sintaxis de tres puntos (...) seguido de una variable que contendrá los elementos en un *array*.

Esta característica sirve para crear funciones que acepten cualquier número de argumentos para agruparlos en un *array*.

```JavaScript
function hola (primero, segundo, ...resto) {
  console.log(primero, segundo)  // 1 2
  console.log(resto) // [3,4,5,6]
}

hola(1,2,3,4,5)
```
También sirve para obtener los elementos restantes de un *array* u objeto usando [desestructuración](https://platzi.com/clases/3504-ecmascript-nuevo/51756-asignacion-de-destructuracion/ "desestructuración").

```JavaScript
const objeto = {
  nombre: "Andres",
  age: 23,
  plataforma: "Platzi"
}
const array = [0,1,2,3,4,5]

const {plataforma, ...usuario} = objeto
cons [cero, ...positivos] = array

usuario // { nombre: 'Andres', age: 23 }
positivos // [ 1, 2, 3, 4, 5 ]
```
**Posición del parámetro rest**

El parámetro *rest* **siempre deberá estar en la última posición** de los parámetros de la función, caso contrario existirá un error de sintaxis.
```JavaScript
// ❌ Mal
function hola (primero, ...rest, ultimo) { ... }
// SyntaxError: Rest element must be last element. 
```

**Diferencias entre el parámetro rest y el operador de propagación**

Aunque el parámetro *rest* y el operador de propagación utilicen la misma sintaxis, son diferentes.

El parámetro *rest* agrupa el **residuo de elementos** y siempre debe estar en la última posición, mientras que el operador de propagación **expande los elementos de un iterable en un array** y no importa en que lugar esté situado.

```JavaScript
const array = [1,2,3,4,5]

function hola (primero, segundo, ...resto) { // <- Parámetro Rest
  console.log(primero, segundo)  // 1 2
  console.log(resto) // [3,4,5, "final"]
}

hola(...array, "final") //<- Operador de propagación
//Lo mismo que hacer -> hola(1,2,3,4,5, "final")
```

### ES6: object literals

Los **objetos literales** consiste en crear objetos a partir de variables **sin repetir el nombre**. Antes de ES6, para crear un objeto a partir de variables consistía en la siguiente manera:

```JavaScript
const nombre = "Andres"
const edad = 23

const objeto = {
    nombre: nombre, 
    edad: edad
}

objeto // { nombre: 'Andres', edad: 23 }
```
**Cómo utilizar objetos literales**

Con los parámetros de objeto puedes **obviar la repetición de nombres**, JavaScript creará la propiedad a partir del nombre de la variable con su respectivo valor.

```JavaScript
const nombre = "Andres"
const edad = 23

const objeto = {nombre, edad}

objeto // { nombre: 'Andres', edad: 23 }
```

El resultado es el mismo, pero sin la necesidad de repetir palabras. Puedes combinarlo con variables que su propiedad tiene un nombre diferente.

```JavaScript
const nombre = "Andres"
const edad = 23
const esteEsUnID = 1

const objeto = {
    nombre, 
    edad,
    id: esteEsUnID
}

objeto // { nombre: 'Andres', edad: 23, id: 1 }
```

### ES6: promesas

Una **promesa** es una forma de manejar el asincronismo en JavaScript y se representa como un objeto que puede generar un valor único a futuro, que tiene dos estados, o está resuelta o incluye una razón por la cual no ha sido resuelta la solicitud.

**Cómo utilizar las promesas**
Solamente ten presente que la clase Promise y sus métodos then y catch fueron añadidos en ES6. Esto resuelve un problema del manejo del asincronismo con *callbacks*, llamado [Callback Hell](https://miro.medium.com/max/721/0*iiecmuTLPBqbxd5V.jpeg "Callback Hell").

El argumento de la clase `Promise` es una función que recibe dos parámetros:

- resolve: cuando la promesa es **resuelta**.
- reject: cuando la promesa es **rechazada**.
Puedes utilizar cualquier nombre, siempre y cuando sepas su funcionamiento.
```JavaScript
const promesa = () => {
  return new Promise((resolve, reject) => {
    if (something) {
      //true o false
      resolve("Se ha resuelto la promesa")
    } else {
      reject("Se ha rechazado la promesa")
    }
  })
}

promesa()
  .then(respuesta => console.log(respuesta)) //En caso que se ejecute resolve
  .catch(error => console.log(error)) //En caso que se ejecute reject
```
**Cursos para entender el asincronismo en JavaScript**
Si aún no sabes en qué consiste el asincronismo, no te preocupes, existen cursos completos de este tema.

- [Curso de JavaScript Engine (V8) y el Navegador](https://platzi.com/cursos/javascript-navegador/ "Curso de JavaScript Engine (V8) y el Navegador")
- [Curso de Asincronismo con JavaScript](https://platzi.com/cursos/asincronismo-js-2019/ "Curso de Asincronismo con JavaScript")

### Clases

Esta clase es difícil de entender si no se tienen unas bases teóricas sobre la Programación Orientada a Objetos y sobre aspectos de JavaScript como el this 😰 Intentaré definir estos elementos según lo que conozco para crear el concepto de lo que son las clases para quienes no lo pudieron entender bien: ㅤ Comencemos por el aspecto **teórico**: ㅤ

- **Clases**: Es una plantilla. Una definición genérica de algo que tiene atributos (datos/variables) y métodos (acciones/funciones) y desde la cual se pueden crear objetos.
- **Objetos**: Un elemento real que fue creada con base en una clase (plantilla) y que hereda (contiene) sus atributos y métodos. ㅤ
¿Lo vemos con un ejemplo?: Tenemos una clase Animal que tiene como atributos: especie, edad, patas y tiene como métodos: dormir, comer, caminar. A partir de esa clase genérica podemos instanciar objetos de ese tipo, como los siguientes: ㅤ

- **Objeto perro**: especie: canino, edad: 3, patas: 4. Puede dormir, comer y caminar.
- **Objeto paloma**: especie: ave, edad: 1, patas: 2. Puede dormir, comer y caminar.
- **Objeto gato**: especie: felino, edad: 2, patas: 4. Puede dormir, comer y caminar. ㅤ *Estos tres objetos fueron creados con base en la clase Animal (a esto se le llama instanciar un objeto a partir de una clase), y por ende, cada uno es un objeto de tipo Animal y cada uno tiene los atributos y métodos definidos en la clase*.

ㅤ Ahora, a nivel más **técnico**, utilizamos los siguientes conceptos: ㅤ

- **Constructor**: Es un método que contiene una serie de instrucciones que se encargan de inicializar un objeto cuando es instanciado a partir de esa clase. Básicamente, asigna los valores de los atributos que le enviemos a ese objeto nuevo. Es una función que se ejecuta automáticamente.
- **Getter y Setter**: Son funciones sencillas de entender: obtener el valor de un atributo o establecerlo. Se crean de esta manera por un concepto de la POO denominado encapsulamiento, que consiste, entre otras cosas, en limitar el acceso a las clases para tener mayor control sobre ellas.
- **This**: Con este objeto de contexto hacemos referencia al propio objeto que se está instanciando y no a la clase.

```JavaScript
// Declaración de la clase Animal
class Animal {

	// Constructor: le enviamos a la clase los valores para los atributos del nuevo objeto (como argumentos) y el constructor se encarga de asignarlos:
	// (Recordar: this hace referencia al objeto).
	constructor(especie, edad, patas) {
		this.especie = especie; // Asignar atributo especie al objeto
		this.edad = edad; // Asignar atributo edad al objeto
		this.patas = patas; // Asignar atributo patas al objeto
	}

	// Métodos de la clase: pueden contener cualquier lógica.
	dormir() {
		return 'Zzzz';
	}

	comer() {
		return 'Yummy!';
	}

	caminar() {
		return '¡Caminando!, la la la';
	}

	// Getter y Setter (solo para edad para no alargar)
	// (Recordar: this hace referencia al objeto)
	get getEdad() {
		return this.edad;
	}

	set setEdad(newEdad) {
		this.edad= newEdad;
	}
}

// Ahora instanciemos los objetos: tendremos perro, paloma y gato como objetos de tipo Animal. Al enviar el valor de los atributos como argumentos, el constructor automáticamente los asigna al nuevo objeto.
const perro = new Animal('canino', 3, 4);
const paloma = new Animal('ave', 1, 2);
const gato = new Animal('felino', 2, 4);

// Podemos acceder a los métodos desde cada objeto:
perro.dormir();	// Retorna 'Zzzz'
paloma.comer(); // Retorna 'Yummy!'
gato.caminar(); // Retorna '¡Caminando!, la la la'

// Usamos los getter para obtener los valores de los atributos y los setters para reasignarlos.
perro.getEdad; // Retorna 3
gato.setEdad = 3; // Cambia su atributo edad a 3
```

Sé que es largo y tedioso por tanto aspecto teórico, pero cuando lo entiendes se abre todo un mundo de posibilidades al momento de programar (no solo con JavaScript). Espero que sea de ayuda 😉

### ES6: module

Para que el código de JavaScript sea más ordenado, legible y mantenible; ES6 introduce una forma de manejar código en **archivos de manera modular**. Esto involucra exportar funciones o variables de un archivo, e **importarlas** en otros archivos donde se necesite.

#### Cómo utilizar los módulos de ECMAScript

Para explicar cómo funciona las exportaciones e importaciones de código, debes tener mínimo dos archivos, uno para **exportar** las funcionalidades y otro que las **importe** para ejecutarlas.

Además, si iniciaste un proyecto con NPM (*Node Package Manager*) con Node.js, necesitas especificar que el código es modular en el archivo package.json de la siguiente manera:

```JavaScript
// package.json
{   ...
    "type": "module"
}
```

#### Qué son las exportaciones de código

Las exportaciones de código consisten en **crear funciones o variables para utilizarlas en otros archivos** mediante la palabra reservada `export`.

Existen dos formas de exportar, antes de declarar la funcionalidad, o entre llaves `{}`.

Por ejemplo, en el archivo `math_function.js` declaramos una función para sumar dos valores, el cual lo exportaremos.

```JavaScript
//math_function.js
export const add = (x,y) => {
    return x + y
}
```

```JavaScript
//math_function.js
const add = (x,y) => {
    return x + y
}

export { add, otherFunction, ... }
```
#### Qué son las importaciones de código
Las importaciones de código consiste en **usar funciones o variables de otros archivos** mediante la palabra reservada import, que deberán estar siempre lo más arriba del archivo y utilizando el **mismo nombre que el archivo original**.

Existen dos formas de exportar, antes de declarar la funcionalidad, o entre llaves `{}`.

Por ejemplo, importamos la función `add` del archivo `math_function.js` para utilizarla en un archivo `main.js`.

```JavaScript
// main.js
import { add, otherFunction } from './math_functions.js'

add(2,2) //4
```
Si importamos el módulo con un nombre diferente, existirá un error de sintaxis.
```JavaScript
// Erróneo
import { suma } from './math_functions.js'

suma(2,2) //SyntaxError: The requested module '/src/archivo1.js' does not provide an export named 'suma'
```
**Para importar todas las funcionalidades de un archivo se utiliza un asterisco** `(*)` y se puede cambiar el nombre para evitar la repetición de variables o funciones a través de la palabra reservada `as`.
```JavaScript
// main.js
import * as myMathModule from './math_functions.js';

myMathModule.add(2,2) //4
myMathModule.otherFunction()
...
```
#### Exportaciones por defecto
**Si solo UN valor será exportado**, entonces se puede utilizar `export default`. De esta manera no es necesario las llaves `{}` al exportar e importar.
```JavaScript
//math_function.js
export default function add (x,y){
    return x + y;
}
```
Adicionalmente, no se puede usar `export default` antes de declaraciones `const`, `let` o `var`, pero puedes exportarlas al final.

```JavaScript
// ❌ Erróneo
export default const add  = (x,y) => {
    return x + y;
}

// ✅ Correcto
const add  = (x,y) => {
    return x + y;
}

export default add
```
#### Importaciones por defecto
Si únicamente un valor será importado, entonces se puede utilizar **cualquier nombre en la importación**. De esta manera no es necesario las llaves `{}`.

```JavaScript
//Las siguientes importaciones son válidas
import  add  from './math_functions.js'
import  suma  from './math_functions.js'
import  cualquierNombre  from './math_functions.js'
```
Sin embargo, es recomendable utilizar siempre el nombre de la función, para evitar confusiones.

#### Combinar ambos tipos de exportaciones e importaciones
Teniendo las consideraciones de importaciones y exportaciones, nombradas y por defecto, entonces podemos combinarlas en un mismo archivo.

```JavaScript
// module.js
export const myExport = "hola"
function myFunction() { ... }

export default myFunction

// main.js
import myFunction, { myExport } from "/module.js"
```

### ES6: generator

Los **generadores** son funciones especiales que pueden pausar su ejecución, luego volver al punto donde se quedaron, recordando su scope y seguir retornando valores.

Estos se utilizan para guardar la **totalidad de datos infinitos**, a través de una función matemática a valores futuros. De esta manera ocupan poca memoria, con respecto a si creamos un *array* u objeto.

Cómo utilizar generadores
La sintaxis de los generadores comprende lo siguiente:

- La palabra reservada `function* `(con el asterisco al final).
- La palabra reservada `yield` que hace referencia al valor retornado cada vez que se invoque, recordando el valor anterior.
- Crear una variable a partir de la función generadora.
- El método next devuelve un objeto que contiene una propiedad `value` con cada valor de `yield`; y otra propiedad done con el valor `true` o `false` si el generador ha terminado.
Si el generador se lo invoca y ha retornado todos sus valores de yield, entonces devolverá el objeto con las propiedades `value` con `undefined` y un `done` con `true`.

```JavaScript
// Declaración
function* nombre(parámetros){
    yield (primer valor retornado)
    yield (segundo valor retornado)
    ...
    yield (último valor retornado)

}

//Crear el generador
const generador = nombre(argumentos)

// Invocacioens
generador.next().value //primer valor retornado
generador.next().value //segundo valor retornado
...
generador.next().value //último valor retornado
```
#### Ejemplo de un generador
Por ejemplo, creemos un generador para retornar tres valores.


```JavaScript
function* generator(){
    yield 1
    yield 2
    yield 3
}

const generador = generator()

generador.next().value //1
generador.next().value //2
generador.next().value //3
generador.next() // {value: undefined, done: true}
```

- [Documentación de generadores](https://developer.mozilla.org/es/docs/Web/JavaScript/Reference/Global_Objects/Generator "Documentación de generadores")
#### Cómo utilizar for of y for in
Existen dos nuevas formas de utilizar ciclos repetitivos. El bucle for valor of iterable recorre iterables, como arrays, `Map`, `Set` e incluso un generador.

El valor es cada elemento del iterable puede tener cualquier nombre, por eso se inicia con `let nombre`.

```JavaScript
const array = [5, 4, 3, 2, 1]

for (let numero of array) {
  console.log(numero) // 5 4 3 2 1
}
```
Sin embargo, debes tener en cuenta que solo podrás acceder a sus valores, y no a sus referencias, por lo que si quieres cambiar los elementos del array, necesitarás un índice `array[indice]`.

```JavaScript
for (let numero of array) {
  valor *= 2 
  console.log(numero) // 10 8 6 4 2
}
 
console.log(array) // [ 5, 4, 3, 2, 1 ]
```
Si intentas recorrer un objeto de esta forma `for elemento of objeto`, te ocurrirá un error, porque un **objeto no es un iterabl**e. En su lugar puedes utilizar `for elemento in objeto`, que recorrerá las propiedades del objeto.
```JavaScript
const objeto = { a: 1, b: 2, c: 3 }

for (let elemento in objeto) {
  console.log(elemento) // 'a' 'b' 'c'
}
```
Sin embargo, si utilizas `for elemento in array`, no dará un error, pero el resultado no será el esperado, ya que los arrays son un tipo de objeto donde cada propiedad es el índice del valor del array o del iterable. Por lo que debes tener cuidado.
```JavaScript
const array = [5, 4, 3, 2, 1]

for (let elemento in array) {
  console.log(elemento) // '0' '1' '2' '3' '4'
}

/* const array = {
	'0': 5,
  '1': 4,
  '2': 3,
  '3': 2,
  '4': 1
}*/
```
##### Lecturas recomendadas
[Iteradores y generadores - JavaScript | MDN](https://developer.mozilla.org/es/docs/Web/JavaScript/Guide/Iterators_and_Generators "Iteradores y generadores - JavaScript | MDN")

[Generador - JavaScript | MDN](https://developer.mozilla.org/es/docs/Web/JavaScript/Reference/Global_Objects/Generator "Generador - JavaScript | MDN")

### ES6: set-add

`Set` es una nueva estructura de datos para almacenar **elementos únicos**, es decir, sin elementos repetidos.

#### Cómo utilizar los Sets

Para iniciar un `Set`, se debe crear una instancia de su clase a partir de un iterable. Generalmente, un iterable es un *array*.
`const set = new Set(iterable)`

#### Cómo manipular los Sets
Para manipular estas estructuras de datos, existen los siguientes métodos:

- `add(value)`: añade un nuevo valor.
- `delete(value)`: elimina un elemento que contiene el `Set`, retorna un booleano si existía o no el valor.
- `has(value)`: retorna un booleano si existe o no el valor dentro del `Set`.
- `clear(value)`: elimina todos los valores del `Set`.
- `size`: retorna la cantidad de elementos del `Set`.

### ES7: exponentiation operator y array includes

La siguiente versión de **ECMAScript** fue publicada en 2016. Las siguientes características de ES7 o ES2016 que aprenderás son: el método `includes` de *arrays* y el operador de potenciación.

#### Operador de potenciación
El operador de potenciación (exponential operator) consiste en elevar una base a un exponente utilizando el doble asterisco `(**)`.

`base ** exponente`

Por ejemplo, el cubo de 2 es igual a 8, matemáticamente expresado sería: $2^3=8$.

```JavaScript
const potencia = 2**3

console.log(potencia) // 8
```
#### Método includes
El método includes determina si un *array* o *string* incluye un determinado elemento. Devuelve `true` o `false`, si existe o no respectivamente.

Este método recibe dos argumentos:

- El **elemento** a comparar.
- El **índice inicial** desde donde comparar hasta el último elemento.
#### Índices positivos y negativos

Los índices positivos comienzan desde 0 hasta la longitud total menos uno, de **izquierda** a **derecha** del *array*.

`[0,1,2,3, ...., lenght-1]`

Los índices negativos comienzan desde -1 hasta el negativo de la longitud total del *array*, de **derecha a izquierda**.

`[-lenght, ...,  -3, -2, -1]`

**Ejemplos utilizando el método includes**

El método includes se utiliza para *arrays* y *strings*. El método es sensible a mayúsculas, minúsculas y espacios.

```JavaScript
//Utilizando strings
const saludo = "Hola mundo"

saludo.includes("Hola") // true
saludo.includes("Mundo") // false
saludo.includes(" ") // true
saludo.includes("Hola", 1) // false
saludo.includes("mundo", -5) // true
```

```JavaScript
// Utilizando arrays
const frutas = ["manzana", "pera", "piña", "uva"]

frutas.includes("manzana") // true
frutas.includes("Pera") // false
frutas.includes("sandía") // false
frutas.includes("manzana", 1) // false
frutas.includes("piña", -1) // false
frutas[0].includes("man") // true
```
#### Lecturas recomendadas
[Curso de Manipulación de Arrays en JavaScript - Platzi](https://platzi.com/cursos/arrays "Curso de Manipulación de Arrays en JavaScript - Platzi")

### ES8: object entries y object values

Los métodos de **transformación de objetos a arrays** sirven para obtener la información de las propiedades, sus valores o ambas.

#### Obtener los pares de valor de un objeto en un array
`Object.entries()` devuelve un array con las entries en forma `[propiedad, valor]` del objeto enviado como argumento.

```JavaScript
const usuario = {
    name: "Andres",
    email: "andres@correo.com",
    age: 23
}

Object.entries(usuario) 
/* 
[
  [ 'name', 'Andres' ],
  [ 'email', 'andres@correo.com' ],
  [ 'age', 23 ]
]  
*/
```
#### Obtener las propiedades de un objeto en un array
`Object.keys()` devuelve un array con las propiedades (keys) del objeto enviado como argumento.

```JavaScript
const usuario = {
    name: "Andres",
    email: "andres@correo.com",
    age: 23
}

Object.keys(usuario) 
// [ 'name', 'email', 'age' ]
```
#### Obtener los valores de un objeto en un array
`Object.values()` devuelve un *array* con los valores de cada propiedad del objeto enviado como argumento.

```JavaScript
const usuario = {
    name: 'Andres',
    email: "andres@correo.com",
    age: 23
}

Object.values(usuario) 
// [ 'Andres', 'andres@correo.com', 23 ]
```

#### Lecturas recomendadas
[Object.entries() - JavaScript | MDN](https://developer.mozilla.org/es/docs/Web/JavaScript/Reference/Global_Objects/Object/entries "Object.entries() - JavaScript | MDN")

[Object.values() - JavaScript | MDN](https://developer.mozilla.org/es/docs/Web/JavaScript/Reference/Global_Objects/Object/values "Object.values() - JavaScript | MDN")

[Curso de Manipulación de Arrays en JavaScript - Platzi](https://platzi.com/cursos/arrays/ "Curso de Manipulación de Arrays en JavaScript - Platzi")

### ES8: string padding y trailing commas

Las siguientes características de ES8 o ES2017 que aprenderás son: rellenar un string y trailing commas.

#### Rellenar un string o padding
El padding consiste en rellenar un string por el principio o por el final, con el carácter especificado, repetido hasta que complete la longitud máxima.

Este método recibe dos argumentos:

- La longitud máxima a rellenar, incluyendo el `string` inicial.
- El `string` para rellenar, por defecto, es un espacio.
Si la longitud a rellenar es menor que la longitud del string actual, entonces no agregará nada.

#### Método padStart
El método `padStart` completa un `string` con otro string en el inicio hasta tener un total de caracteres especificado.

```JavaScript
'abc'.padStart(10) // "       abc"
'abc'.padStart(10, "foo") // "foofoofabc"
'abc'.padStart(6,"123465") // "123abc"
'abc'.padStart(8, "0") // "00000abc"
'abc'.padStart(1) // "abc"
```

#### Método padEnd
El método `padEnd` completa un string con otro string en el final hasta tener un total de caracteres especificado.
```JavaScript
'abc'.padEnd(10) // "abc       "
'abc'.padEnd(10, "foo") // "abcfoofoof"
'abc'.padEnd(6, "123456") // "abc123"
'abc'.padEnd(1) // "abc"
```
#### Trailing commas
Las *trailing commas* consisten en comas al final de objetos o *arrays* que faciliten añadir nuevos elementos y evitar errores de sintaxis.
```JavaScript
const usuario = {
    name: 'Andres',
    email: "andres@correo.com",
    age: 23, //<-- Trailing comma
}

const nombres = [
    "Andres",
    "Valeria",
    "Jhesly", //<-- Trailing comma
 ]
```
##### Lecturas recomendadas
[String.prototype.padStart() - JavaScript | MDN](https://developer.mozilla.org/es/docs/Web/JavaScript/Reference/Global_Objects/String/padStart "String.prototype.padStart() - JavaScript | MDN")
