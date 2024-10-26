x = 100

def local_function():
    x = 10 # Variable local
    print(f'El valor de la variable es {x}')

def show_global():
    print(f'El valor d ela variable gloval es {x}')
#local_function()
print(x) # Genera error