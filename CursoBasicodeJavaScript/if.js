var edad = 18;

if (edad === 18) {
    console.log("Puedes votar, seá tu 1ra votación");
} else if (edad > 18){
    console.log("Puedes votar de nuevo");
}   else if (edad > 18){
    console.log("Puedes votar de nuevo");
} else if (edad > 18){
    console.log("Puedes votar de nuevo");
} 
else {
    console.log("Aun no puedes votar")
}


condition ? true: false;

var numero = 1;

var resultado = numero === 1 ? "Si soy un uno" : "No, soy un uno";

//Cómo anidar condicionales al programar

function calcularDescuento(articulos, precio) {
    var precioFinal
  
    if (articulos <= 5) {
      //Hasta 5 artículos
      precioFinal = precio * (1 - 0.1)
    } else if (articulos > 5 && articulos <= 10) {
      //De 6 a 10 artículos
      precioFinal = precio * (1 - 0.15)
    } else {
      //De 10 artículos en adelante
      precioFinal = precio * (1 - 0.2)
    }
  
    return precioFinal
  }
  
  //Operador ternario
  calcularDescuento(4, 10) // 9
  calcularDescuento(8, 20) // 17
  calcularDescuento(15, 50) // 40

  function esPar(numero){
    return numero % 2 === 0 ? "Es par" : "No es par"
}

esPar(2) // "Es par"
esPar(3) // "No es par"