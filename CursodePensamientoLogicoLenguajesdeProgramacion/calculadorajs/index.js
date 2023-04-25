//Encuentra el numero mayor de una lista de numeros en un Array//

let numeros = [5, 10, 15, 70, 8]
let numeroMaximo = 0
let tamaño = numeros.length

for(let i = 0; i < tamaño;i++)
{
  if(numeroMaximo < numeros[i])
  {
    numeroMaximo = numeros[i]
  }
}

console.log("El número mayor es: " + numeroMaximo)

let maximo = Math.max(5, 10, 15, 70, 8)
console.log("El número mayor es: " + maximo)

let minimo = Math.min(5, 10, 15, 70, 8)
console.log("El número menor es: " + minimo)