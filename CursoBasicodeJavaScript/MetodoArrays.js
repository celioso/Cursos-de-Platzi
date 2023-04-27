var articulos = [
    {nombre: "bici", costo: 3000},
    {nombre: "Tv", costo: 2500},
    {nombre: "Libro", costo: 320},
    {nombre: "Celular", costo: 10000},
    {nombre: "Laptop", costo: 20000},
    {nombre: "Teclado", costo: 500},
    {nombre: "Audifonos", costo: 1700},
];

// filtrar los datos

var articulosFiltrados = articulos.filter(function(articulo){
    return articulo.costo <= 500;
});

console.log(articulosFiltrados);

//metodo map

var nombreArticulos = articulos.map(function(articulo){
    return articulo.nombre
});

//  .find busca el articulo exacto

var encuentraArticulo = articulo.find(function(articulo){
    return articulo.nombre === "Laptop";
});

//.forEach

articulos.forEach(function(articulo){
    console.log(articulo.nombre);
});

//.some

var articulosBaratos = articulos.some(function(articulo){
        return articulo.costo <= 700;
});

//Eliminando elementos de un Array
/*El método .push() nos permite agregar uno o más elementos al final de un array. A continuación veremos un ejemplo aplicado con un array que contiene números:*/

let numArray = [1, 2, 3, 4, 5];
function newNum(){
    numArray.push(6,7);
    console.log(numArray);
}
newNum();

//Como podemos ver, al momento de ejecutar la función se agregan los números 6 y 7 al array. Ahora revisemos un ejemplo con strings:

let txtArray = ["Ana", "Juan", "Diego", "Lautaro"];
function addCharacters(){
    txtArray.push("Chris", "Maria");
    console.log(txtArray);
}

addCharacters();

//Como podemos ver, agregamos dos cadenas de strings al ejecutar la función donde tenemos txtArray.push(). Es decir, indico el array al que voy agregar elementos, uso el método .push(), y dentro de .push() indico los elementos que quiero agregar al string. Finalmente, el console.log() lo uso para revisar en la consola si esto sucedió o no.

/*.shift()
Ahora pasemos a la otra cara de la moneda donde necesitamos eliminar un elemento del array. .shift() eliminar el primer elemento de un array, es decir, elimina el elemento que esté en el índice 0.*/

let array = [1, 2, 3, 4, 5]
console.log(array);

let shiftArray = array.shift();
console.log(array);

//Como vemos, luego de aplicar .shift() se eliminó exitosamente el primer elemento del array. ¿Y si quisiéramos eliminar el último elemento? Pasemos al bonus track de esta clase 🙌🏼.

//Bonus Track
//Si ya entendiste cómo funciona .shift() aplicar .pop() te será pan comido 🍞. El método .pop() eliminará el último elemento de un array. En este sentido, si tenemos un array de 5 elementos, pop() eliminará el elemento en el índice 4. Utilicemos el mismo ejemplo pero usando este método.

let array = [1, 2, 3, 4, 5];
console.log(array);
let shiftArray = array.pop();
console.log(array);

// arregla el Bug

export function solution(cars) {
    // 👇 Este es el código que no funciona
    return cars.filter(function (car) {
      if (car.licensePlate) {
        return true;
      } else {
        return false;
      }
    });
  }