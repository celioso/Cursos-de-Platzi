price = 100 # global
result =  200


def increment():
  price = 200
  price = price + 10
  print(price)
  return price

print(price)
price_2 = increment()
print(price_2)
print(result)