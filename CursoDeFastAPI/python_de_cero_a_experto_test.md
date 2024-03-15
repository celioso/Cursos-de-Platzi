###  Python: De Cero a Experto

1. **Tienes un set o conjunto de Python almacenado en la variable trips. ¿Cómo obtienes su cantidad de elementos?**

**R/:** len(trips)

2. **¿Cuál será el resultado del siguiente bloque de código?**

```python
a = {1,2}
b = {2,3}
print(a & b)
```
**R/:** {2}

3. **Dado el siguiente bloque de código:**

![](https://static.platzi.com/media/user_upload/list_comprehension-07114201-7d69-4066-af58-6787f4bd7c86.jpg)

**¿Cuál de las siguientes respuestas con List Comprehension nos peermite obtener el mismo resultad con una sintaxis más corta?**

**R/:** `n = [i - 1 for i in range(1,6) if i <= 2]`

4. **¿Cuál de las siguientes estructuras de datos NO nos permite duplicar elementos?**
**R/:** Set

5. ¿Cuál de las siguientes funciones SIEMPRE devuelve la misma cantidad de elementos de la lista original?
**R/:** map

6. **Dado el siguiente bloque de código:**
```python
original = [1, 2, 3, 4, 5]
new = []

for x in original:
    new.append(x * 2)
```

**¿Cuál de las siguientes respuestas es la sintaxis más corta para obtener el mismo resultado?**

**R/:** `new = list(map(lambda x: x * 2, original))`

7. **¿Qué uso se le da al método map() de Python?**

**R/:** Aplica una función sobre todos los elementos de un iterable y devuelve otro iterable tipo map.

8. **¿Qué es una variable en programación?**
**R/:** Un espacio en memoria al que se le da un nombre para guardar algún dato.

9. **¿Para qué sirve type() en Python?**

**R/:** Indica el tipo de dato de una variable u objeto.

10. **¿Qué obtenemos al utilizar el operador + en strings? Por ejemplo:**

`'Hola,' + " " + 'Platzinauta'`
**R/:** Se concatena las cadenas de texto.

11. **¿Qué se obtiene al ejecutar la siguiente conversión en Python?**
![](https://static.platzi.com/media/user_upload/carbon-e06f99b3-047f-4513-b58a-8416d00404ce.jpg)

**R/:** 4

12. ¿Qué se obtiene al ejecutar la siguiente línea en Python?

`print((8 / 2) + 4 * 8)`

Considera el orden en que se ejecutan los operadores aritméticos.

**R/:** 36

13. **¿Qué es una lista en Python?**

**R/:** Son un tipo de datos donde se pueden almacenar colecciones de datos de cualquier tipo.

14. **¿Cuál es la diferencia entre un ciclo for y un while?**

**R/:** Con el ciclo for tenemos definido cuántas veces iteramos, mientras que en el while es indefinido hasta que se cumpla la condición que indica.

15. **¿Qué es un entorno virtual?**

**R/:** La herramienta de Python para aislar o encapsular proyectos con sus propios paquetes y versiones sin afectar a otros proyectos y entornos virtuales.

16. **¿Qué herramienta nos permite trabajar con entornos virtuales en Python 3?**

**R/:** venv

17. **¿Qué herramienta nos permite instalar paquetes de Python como dependencias en nuestros proyectos?**

**R/:** pip

18. **¿Con qué comando creamos entornos virtuales en Python 3?**

**R/:** python3 -m venv [ruta del entorno virtual]

19. **¿Qué herramienta nos permite aislar y encapsular nuestros proyectos y los paquetes de terceros que este utilice, aunque la versión de Python y el sistema operativo sigan siendo los mismos para todos los proyectos?**

**R/:** virtualenv 

20. **¿Con qué comando instalamos el paquete requests en su versión 2.27.1?**

**R/:** pip install requests==2.27.1

21. **En tu proyecto A necesitas matplotlib en su versión 3.5, pero tu proyecto B necesita el mismo paquete en su versión 3.6. ¿Cuál es la mejor forma de trabajar para no generar conflictos entre ambos paquetes?**

**R/:** Aislando cada proyecto en su propio ambiente virtual para instalar la versión correcta del paquete en cada uno sin afectar al otro.

22. **¿Cómo puedes agregar SQLAlchemy a tu aplicación?**

**R/:** Ejecutando el comando "pip install sqlalchemy"

23. **¿Cuál es la sintaxis correcta para obtener todos los datos de una tabla con SQLALchemy?**

**R/:** db.query(Model).all()

24. **¿Cuál es la sintaxis correcta para eliminar un registro de una tabla con SQLALchemy?**

**R/:** db.query(Model).filter(condition).delete()

25. **¿Cuál es la clase utilizada para crear routers en FastAPI?**

**R/:** APIRouter


26. **¿En qué framework está basado FastAPI?**

**R/:** Starlette

27. **¿Cuál es la ruta para acceder a la documentación autogenerada?**

**R/:** /docs

28. **¿De qué clase debe heredar un esquema de datos de Pydantic?**

**R/:** BaseModel

29. **¿Cómo puedes validar que un número sea mayor o igual a otro?**

**R/:** Usando el parámetro ge

30. **Si tuvieras un esquema llamado Book, ¿cómo le indicarías a una ruta que debe devolver datos de este tipo?**

**R/:** jsonable_encoder