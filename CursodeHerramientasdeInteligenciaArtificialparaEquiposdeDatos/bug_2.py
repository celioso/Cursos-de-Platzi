def calcular_promedio(notas):
    suma = 0
    contador = 0
    
    for nota in notas:  # Corregido 'on' por 'in'
        suma += nota
        contador += 1
    
    promedio = suma / contador
    
    if promedio >= 60:  # Añadidos dos puntos
        mensaje = "Aprobado"
    else:
        mensaje = "Reprobado"  # Añadidos dos puntos
    
    return promedio, mensaje

notas = [80, 75, 90, 65, 50]
promedio, resultado = calcular_promedio(notas)

print("El promedio es: " + str(promedio))  # Convertir promedio a cadena
print("El resultado es: " + resultado)
