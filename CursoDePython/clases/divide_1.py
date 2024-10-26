def divide(a: int, b:int) -> float:
    # validar que ambos parametros sean enteros
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError('Ambos parametros deben ser enteros.')
    # Verificamos que el divisor no sea cero
    if b == 0:
        raise ValueError('El divisor no puede ser cero.')
    
    return a/b

# Ejemplo de uso

try:
    res= divide(10,2) # Error de tdivisión por cero
    print(res)
except (ValueError, TypeError) as e:
    print(f'Error: {e}')