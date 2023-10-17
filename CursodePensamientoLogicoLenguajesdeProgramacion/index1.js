var numeros = [3, 4, 8, 7, 21, 45, 24, 87, 142, 24, 14, 38, 43, 13, 180];
var max = 0

for (let i = 0; i < numeros.length; i++) {
  if (max < numeros[i]) {
    max = numeros[i]
  }
}
console.log("El número mayor es: " + max)

