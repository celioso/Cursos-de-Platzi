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