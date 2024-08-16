to_do = ["Dirigirnos al hotel",
        "Ir a almorzar",
        "Visitar un museo",
        "volver al hotel"]

print(to_do)
numbers = [1, 2, 3, 4, "Cinco"]
print(numbers)
print(type(numbers))
mix =["uno", 2, 3.14, True, [1, 2, 3]]
print(mix)
print(len(mix))
print("Elemento", mix)
print("Primer elemento", mix[0])
print("segundo elemento", mix[1])
print("Ultimo elemento", mix[-1])
string = "Hola mundo"
print("Primer elemento", string[0])
print("segundo elemento", string[1])
print("Ultimo elemento", string[-1])
print(mix[2:-2])
print(mix)
mix.append(False)
print(mix)
mix.append(["a", "b"])
print(mix)
mix.insert(1,["a", "b"])
print(mix)
print(mix.index(["a", "b"]))
numbers = [1, 2, 100.01, 90.45,  3, 4,5]
print(numbers)
print("Mayor:", max(numbers))
print("Manor:", min(numbers))

del numbers[-1]
print(numbers)
del numbers[:2]
print(numbers)
del numbers
print(numbers)
