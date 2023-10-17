var miAuto = {
    marca: "Toyota",
    modelo: "Corolla",
    year: 2020,
    detalleDelAuto: function(){
        console.log(`Mi Auto ${this.modelo} es del años ${this.year}, marca ${this.marca}`)
    }
};

//para buscar el Objeto

miAuto.marca;
miAuto.year;

//llamar la función

miAuto.detalleDelAuto();

//Objects: Función constructora

function car(brand, model, year){
    this.brand = brand;
    this.model = model;
    this.year = year;
}

//agregar un objeto

var autoNuevo = new car("Tesla", "Model 3", 2020);
var autoNuevo2 = new car("Tesla","Model X", 2018);
var autoNuevo3 = new car("Toyota", "corolla", 2020);


//reto

function car(brand, model, year){
    this.brand = brand;
    this.model = model;
    this.year = year;
}

var cars = [];

// Se agrega la cantidad de carros que tiene la condición
for (i = 0; i < 30; i++){
    var brand = prompt("Ingresa la marca del auto");
    var model = prompt("Ingresa el modelo del auto");
    var year = prompt("Ingresa el año del auto");
    cars.push(new auto (brand, model, year));
}

//muestra todos los datos agregados al terminar 

for(let i = 0 ; i < car.length ; i++){
    console.log(car[i]);
  }
  

//Permiso para conducir

 
export function solution(car) {
  // Tu código aquí 👈
  !car["licensePlate"] ? car.drivingLicense = false : car.drivingLicense = true
  return car
}

function solution(car) {
    if (car.licensePlate === null || car.licensePlate === undefined) {
      car.drivingLicense = false;
    }
    else {
      car.drivingLicense = true;
    }
    return car;
  }