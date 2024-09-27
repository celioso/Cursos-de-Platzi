# Curso de Expresiones Regulares - Examen

1. **El símbolo que denota cualquier caracter es:**

**R/:** .

2. **Para usar un rango de caracteres (por ejemplo 0-9 o a-g) lo tenemos que delimitar en la expresión con:**

**R/:** []

3. **Si quisiera hacer que una expresión no distinga entre mayúsculas y minúsculas, debo agregarle la bandera:**

**R/:** /i

4. **El patrón `a.` significa exactamente:**

**R/:** una a y cualquier otro caracter

5. **El patrón a* significa exactamente:**

**R/:** cero o más a

6. **El patrón `a?` significa exactamente:**

**R/:** cero o sólo una a

7. **El patrón `a+` significa exactamente:**

**R/:** una o más a

8. **La clase `\w` es equivalente a:**

**R/:** `[a-zA-Z0-9_]`

9. **Para delimitar el número de ocurrencias de alguna expresión (una, dos o tres, en este caso), el número de repeticiones buscadas se denota por:**

**R/:** `{1,3}`

10. **Si quiero encontrar grupos de sólo 3 caracteres numéricos, cuál de las siguientes expresiones NO funciona:**

**R/:** `\d+…`

11. **¿Cuál de las siguientes expresiones sería útil para encontrar palabras que empiecen con letra mayúscula?**

**R/:** `/[A-Z][a-z]+/`

12. **El patrón `/\[\d+?\],.*/` ¿con cuál opción hará match?:**

**R/:** `[12],34,56,78`

13. **Una expresión greedy significa que se intentará encontrar el patrón:**

**R/:** tantas veces como sea posible

14. **La expresión `^1.*` encuentra:**

**R/:** una línea que empiece con 1

15. **El caracter que denota cualquier fin de línea es:?**

**R/:** `$`

16. **La expresión `[^rm][a-z]+` encontrará:**

**R/:** Todas las palabras que NO empiecen con “r” o “m”

17. **Cuál de las siguientes líneas NO hará match con la expresión `/^[\w\s]+$/`:**

**R/:** `12,34,56,78`

18. **Cuál de las siguientes líneas SÍ hará match con la expresión**
`/[a-z]{6,}@?gmail\.com`

**R/:** prueba@gmail.com

19. **Supongamos que tenemos un archivo csv con códigos de producto, todos los códigos son dos caracteres seguidos por 8 a 10 dígitos, pero sólo queremos los productos que empiecen con AB o CD, ¿qué expresión usarías?**

**R/:** `/^(AB|CD)\d{8,10}$/`

20. **Cuando tenemos un problema y lo planeamos solucionar con expresiones regulares, entonces:**

**R/:** tardaremos un poco más en solucionarlo, pero tendremos una solución robusta y duradera
