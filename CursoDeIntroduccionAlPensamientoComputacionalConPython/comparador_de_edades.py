print("compararor de edades")
name_1 = input("Engrese el primer nombre: ")
edad_1 = int(input("Engrese la edad: "))
name_2 =input("Engrese el segundo nombre: ")
edad_2 = int(input("Engrese la edad: "))

if edad_1 > edad_2:
    print(f"{name_1} tiene {edad_1} años y {name_2} tiene {edad_2}, quiere decir que {name_1} es mayor que {name_2}")

elif edad_1 < edad_2:
    print(f"{name_1} tiene {edad_1} años y {name_2} tiene {edad_2}, quiere decir que {name_2} es mayor que {name_1}")
else:
    print("Ambos tiene la misma edad")