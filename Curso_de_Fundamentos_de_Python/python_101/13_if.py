if True:
  print('debería ejecutarse')

if False:
  print('nunca se ejecuta ')


pet = input('¿Cual es tu mascota favorita ')

if pet == 'perro':
  print('genial tienes buen gusto ')

elif pet == 'gato':
  print('espero tengas suerte ')

elif pet == 'pez':
  print('Eres lo maximo')

else:
  print('No tienes ninguna mascota interesante')


'''
stock = int(input('Digito el stock => '))

if stock >= 100 and stock <= 1000:
  print('El stock es correcto')

else:
  print('El stock es incorrecto')
  '''

num = int(input('coloca un número: ')) 

if num % 2 == 0: 
    print(f'El {num} es par')
else:
    print(f'El {num} es impar')