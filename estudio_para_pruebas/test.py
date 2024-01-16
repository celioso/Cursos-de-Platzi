def datos (self,name, years):
    self.name=name
    self.years = years
    name = input("cual es su nombre ")
    years = input("ingrese su edad ")


print("Hola, bienvenido a nuestra compañia")
print(f"Su nombre es {name} y tiene {years}")

if __name__=="__main__":
    print("Hola, bienvenido a nuestra compañia")
    datos()