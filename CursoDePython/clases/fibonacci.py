def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    
number = 3
print(fibonacci(number))

def number_natural(n):
    if n == 0:
        return 0
    else:
            return n + number_natural(n-1)
    
number = 8
print("Suma:",number_natural(number))