import mymodule

mymodule.greeting("Jonathan")

# Importe el módulo llamado mymodule y acceda al diccionario person1:

import mymodule

a = mymodule.person1["age"]
print(a)

import mymodule as mx

a = mx.person1["age"]
print(a)

from mymodule import person1

print (person1["age"])
