def outer_function():
    x = 'enclosing'
    def inner_finction():
        nonlocal x
        x = 'modified'
        print(f'El valor en inner es: {x}')
    inner_finction()
    print(f'El valor outer: {x}')
outer_function()