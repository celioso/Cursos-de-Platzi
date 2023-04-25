name = 'Mario'
middle_name = 'Alexander'
last_name = "Vargas Celis"
age = 30
print(name)
print(last_name)

full_name = name+" " + last_name
print(full_name)

quote = "I'm Nicolas"
print(quote)

quote2 = 'she said "hello"'
print(quote2)

# format
template = "Hola, mi nombre es "+name+" y mi apelledo es "+ last_name
print("v1 ",template)

template = "Hola, mi nombre es {} y mi apellido {}".format(name, last_name)
print("v2", template)

template =f"Hola, mi nombre es {name} y mi apellido es {last_name}"
print("v3 ", template)

template =f"hola, mi nombre es {name} {middle_name} y mi apellido es {last_name}, y mi edad es {age}"
print("v4 ", template)