class Vehicle:
    def __init__(self, brand, model, price):
        self.brand = brand
        self.model = model
        self.price = price
        self.is_available = True

    def sell(self):
        if self.is_available:
            self.is_available = False
            print(f"El vehiculo {self.brand}. Ha sido vendido")
        else:
            print(f"El vehiculo {self.brand}. No está disponible")

    def check_availeble(self):
        return self.is_available
    
    def get_price(self):
        return self.price
    
    def start_engine(self):
        raise NotImplementedError("Este metodo debe ser implementado por la subclase")
    
    def stop_engine(self):
        raise NotImplementedError("Este metodo debe ser implementado por la subclase")
    
class Car(Vehicle):
    def start_engine(self):
        if not self.is_available:
            return f"El motor del coche {self.brand} está en marcha"
        else:
            return f"El coches {self.brand} no está disponible"
            
    def stop_engine(self):
        if self.is_available:
            return f"El motor del coche {self.brand} se ha detenido"
        else:
            return f"El coche {self.brand} no está disponible"
        
class Bike (Vehicle):
    def start_engine(self):
        if not self.is_available:
            return f"La bicicleta {self.brand} está en marcha"
        else:
            return f"La bicicleta{self.brand} no está disponible"
            
    def stop_engine(self):
        if self.is_available:
            return f"La bicicleta {self.brand} se ha detenido"
        else:
            return f"La bicicleta {self.brand} no está disponible"
        
class Truck(Vehicle):
    def start_engine(self):
        if not self.is_available:
            return f"El motor del camión {self.brand} está en marcha"
        else:
            return f"El camión {self.brand} no está disponible"
            
    def stop_engine(self):
        if self.is_available:
            return f"El motor del camión {self.brand} se ha detenido"
        else:
            return f"El camión {self.brand} no está disponible"
        
class Customer:
    def __init__(self, name):
        self.name = name 
        self.purchased_vehicles = []

    def buy_vehicle(self, vehicle: Vehicle):
        if vehicle.check_availeble():
            vehicle.sell()
            self.purchased_vehicles.append()
        else:
            print(f"Lo siento, {vehicle.brand} no está disponible")

    def inquire_vehicle(self, vehicle: Vehicle):
        if vehicle.check_availeble():
            availablily = "Diponible"
        else:
            availablily = "No disponible"
        print(f"El {vehicle.brand} esta {availablily} y cuesta {vehicle.get_price()}")

class Dealership:
    def __init__(self):
        self.inventary = []
        self.custumers = []

    def add_vehicles(self, vehicle: Vehicle):
        self.inventary.append(vehicle)
        print(f"El {vehicle.brand} ha sido añadido al inventario")

    def register_customers(self, customer: Customer):
        self.custumers.append(customer)
        print(f"El cliente {customer.name} ha sido añadido")

    def show_availeble_vehicle(self):
        print("Vehiculos disponibles en la tienda")
        for vehicle in self.inventary:
            if vehicle.check_available():
                print(f"- {vehicle.brand} por {vehicle.get_price()}")