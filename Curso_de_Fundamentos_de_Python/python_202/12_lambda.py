def increment(x):
  return x + 1
  
increment_v2 = lambda x : x +1 

result = increment(10)
print(11)

result = increment_v2(20)
print(result)

full_name = lambda name,middle_name, last_name: f'full name is {name.title()} {middle_name.title()} {last_name.title()}'

text = full_name('Mario','Alexander','Perez casa')
print(text)
