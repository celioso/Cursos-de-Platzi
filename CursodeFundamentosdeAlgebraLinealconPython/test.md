# Curso de Fundamentos de Álgebra Lineal con Python

1. **¿Cómo debo representar el escalar 5 según la definición matemática con Python?**
   
**R//=** escalar = 5

2. **¿Qué código debo usar para crear un vector con los elementos 2 y 3 ([2,3])?**
 
**R//=** vector = np.array([2,3])

3. **Cuando calculo la dimensión del escalar 3.1 con el comando escalar.shape, ¿qué respuesta obtengo?**
   
**R//=** AttributeError: 'float' object has no attribute 'shape'

4. **Para calcular la dimensión de un tensor debo usar la instrucción/atributo:**
   
**R//=** tensor.shape

5. **Transponer la matriz [[1, 2, 3, 4] [5, 6, 7, 8] [9,10, 11,12]] da como resultado:**
    
**R//=** [[ 1 5 9] [ 2 6 10] [ 3 7 11] [ 4 8 12]]

6. **Sumar la matriz [[1 3] [5 6]] con el escalar 3 (matriz+escalar) da lo mismo que hacer:**
    
**R//=** [[ 4 6] [ 8 9]]

7. **El producto interno de una matriz de dimensiones (3,2) y un vector de dimensión (2, ) nos devuelve:**
    
**R//=** Un vector de dimension (3,)

8. **Cuando tengo dos matrices A y B, con A de dimensiones (3,2) y B (2,3), entonces con respecto al producto interno:**
    
**R//=** En este caso siempre puedo calcular el producto interno, aunque da distinto resultado hacer A.dot(B) y B.dot(A)

9. **¿El producto interno es conmutativo? O sea si tengo A y B matrices, tal que es posible calcular el producto interno, hacer A.dot(B) siempre es lo mismo que hacer B.dot(A)?**
    
**R//=** No, solamente en casos particulares podría ser lo mismo.

10. **Dadas A y B, tal que A.dot(B) está definido, entonces al transponer(A.dot(B)) siempre es igual a:**
    
**R//=** B.T.dot(A.T)

11. **Dado el sistema de ecuaciones lineales `y = 7x+2 y = 3x+5` ¿Cuál es la solución del sistema?**
    
**R//=** x = 3/4 y = 7+1/4**

12. **¿Las matrices singulares tienen inversa?**
    
**R//=** No, una matriz cuadrada se dice singular cuando no es invertible.

13. **Un sistema de ecuaciones lineales con 4 ecuaciones y 4 incógnitas, que tiene solución, se puede representar en su forma matricial por:**
    
**R//=** 1 matriz A de 4x4, 1 vector x de dimensión 4, 1 vector b de dimensión 4

14. **Un sistema de ecuaciones lineales en R2 tiene infinitas soluciones cuando:**
    
**R//=** Todas las ecuaciones lineales que forman parte del sistema están sobre la misma recta y por lo tanto todos los puntos de la recta son solución.

15. **¿Cuánto da la norma del vector [1/2,1/2,1/2,1/2]? Observación: la norma que vimos durante la clase es la norma 2**
    
**R//=** 1

16. **¿Cuánto da la norma 0 (cero) del vector [-50,-25,0,25,100,-300]?**
    
**R//=** 5

17. **¿Qué ángulo forman los vectores v1 = [0,1] y v2 = [1,0]?**
    
**R//=** 90 grados, porque norma_v1 * norma_v2 * np.cos(np.deg2rad(90)) da 0

18. **¿Cómo puedo comprobar si una matriz A es simétrica?**
    
**R//=** A == A.T

19. **¿Son ortogonales los vectores v = [7,-7,3], u=[1,1,0] y w=[0,0,1]?**
    
**R//=** No, porque no todas las combinaciones entre los vectores dan 0 al calcular el producto interno

20. **Si estoy en un espacio de dimensión 3 y tengo una matriz, ¿cuántas filas / columnas ortogonales puedo tener como máximo?**
    
**R//=** 3 vectores ortonormales (filas o columnas), porque cuando quiera tener un vector más ortonormal a todos los que ya tengo no será posible lograrlo con esa cantidad de dimensiones.

21. **Cuál de estas afirmaciones es VERDADERA respecto al determinante de la Matriz**
[[1 0 0 0 0]
[0 2 0 0 0]
[0 0 3 0 0]
[0 0 0 4 0]
[0 0 0 0 5]]
    
**R//=** El determinante de la matriz da 120, ya que al ser diagonal la matriz, el determinante es la multiplicación de todos los elementos de la diagonal.

22. **¿Es importante definir funciones auxiliares que estén fuera de nuestro código principal?**
    
**R//=** Es importante porque simplificamos el mantenimiento de nuestro código y ahorramos tiempo. Además cualquier mejora a una función externa automáticamente estará disponible para todos los notebooks en los que estemos trabajando y la misma función sea utilizada.