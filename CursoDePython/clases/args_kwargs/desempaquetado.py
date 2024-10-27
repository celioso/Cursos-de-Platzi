# Desempaquetado args
def add(a,b,c):
    return a+b+c

value = (1,2,3)
print(add(*value))

def show_info(name, age):
    print(f'Name: {name}, Age: {age}')

data = {'name':'Ana', 'age': 28}
show_info(**data)