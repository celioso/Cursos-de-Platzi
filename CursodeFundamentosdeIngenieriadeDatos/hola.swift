import Foundation

// Definir una estructura Persona
struct Persona {
    var nombre: String
    var edad: Int
    
    // Método para saludar
    func saludar() {
        print("¡Hola! Mi nombre es \(nombre) y tengo \(edad) años.")
    }
}

// Crear una instancia de la estructura Persona
let persona1 = Persona(nombre: "Mario", edad: 30)

// Llamar al método saludar
persona1.saludar()
