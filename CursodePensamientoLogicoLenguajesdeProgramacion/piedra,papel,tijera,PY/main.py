import random
#1 es piedra, 2 es papel, 3 es tijera
comienza = random.randint(1, 3)
jugador = 0
pc = comienza
jugador = int(input("Elige: 1 para piedra , 2 para papel , 3 para tijera: "))           
#alert("Elegiste "+ jugador)
if jugador == 1:
  print("Elegiste piedra")
elif jugador == 2:
  print("Elegiste papel")
elif jugador == 3:
  print("Elegiste tijera")
else:
  print("Perdiste")  
  
if pc == 1:
  print("Pc eligió piedra")
elif pc == 2:
  print("Pc eligió papel")
elif pc == 3:
  print("Pc eligió tijera")

#combate
if pc == jugador:
  print("EMPATE")
elif ((jugador == 1 and pc == 3) or (jugador == 2 and pc == 1)or (jugador == 3 and pc == 1)):
  print("GANASTE")
else:
  print("PERDISTE")
    
  