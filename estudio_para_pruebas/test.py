def datos ():
    name = input("cual es su nombre ")
    years = input("ingrese su edad ")
    print(f"Su nombre es {name} y tiene {years}")


def hola():
    print("Hola, bienvenido a nuestra compañia")

def despedida():
    print("Eso es todo")


if __name__=="__main__":
    hola()
    datos()
    despedida()