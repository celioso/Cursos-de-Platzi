def sum(a, b):
    """
    >>> sum(5, 7)
    11
    >>> sum(4,-4)
    0
    """
    return a + b


def subtract(a, b):
    return a - b

def multiplication(a, b):
    return a * b

def division(a, b):
    """
    >>> division(10, 0)
    traceback (most recent call last):
    ValueError: La división por cero no está permitida
    """
    if b == 0:
        raise ZeroDivisionError("No se permite la división por cero")
    return a / b