var numero = 1;

switch (numero){
    case 1:
        console.log("soy uno!");
        break;
    case 10:
        console.log("Soy un 10!");
        break;
    case 100:
        console.log("soy un  100!");
        break;
    default:
        console.log("No soy nada!");
        break;
}


//solucion del reto
export function solution(article) {
    // Tu código aquí 👈
    switch (article) {
      case "computadora":
        return "Con mi computadora puedo programar usando JavaScript";
        break;
  
      case "celular":
        return "En mi celular puedo aprender usando la app de Platzi";
        break;
  
      case "cable":
        return "¡Hay un cable en mi bota!";
        break;
  
      default:
        return "Artículo no encontrado";
        break;
    }
  }