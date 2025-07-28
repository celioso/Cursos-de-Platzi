# Curso de Álgebra Lineal Aplicada para Machine Learning

1. **Al aplicar una matriz a un vector lo que obtenemos es:**
   
**R//=** Un vector transformado linealmente.

2. **Un autovector de una matriz de 2x2 es aquel que cuando le aplico la matriz al autovector:**
 
**R//=** Su dirección no cambia.

3. **Dada la matriz [[3 4] [3 2]], ¿Cuáles son sus autovalores y autovectores asociados?**
   
**R//=** autovector_1 = [0.8, 0.6] autovalor_1 = 6 autovector_2 = [-raiz(2)/2, raiz(2)/2] autovalor_2 = -1

4. **Una matriz A no cuadrada, ¿Cuándo se puede descomponer?**
   
**R//=** Siempre, podemos usar la descomposición en valores singulares (SVD).

5. **Una forma simple de visualizar el efecto que la aplicación de una matriz A de 2x2 tiene es:**
    
**R//=** Graficar el círculo unitario y el circulo unitario transformado.

6. **Usar np.linalg.svd para descomponer una matriz por el método SVD nos devuelve 3 objetos U, D, V ¿Qué es D?**
    
**R//=** Un vector que contiene los valores singulares en orden descendente.

7. **Cuando importamos una imagen a una matriz usando `np.array(list(imagen.getdata(band=0)), float) obtenemos:`**
    
**R//=** Un vector con el valor de cada pixel de la imagen.

8. **Al descomponer por SVD a una matriz que contiene los pixeles de una imagen podemos reducir su tamaño y consecuentemente la definición al:**
    
**R//=** Elegir la cantidad de valores singulares que conservaremos.

9. **¿Qué es PCA?**
    
**R//=** Un método para reducir dimensiones que rotan los ejes.

10. **Cuando preparamos nuestros datos para aplicar PCA es importante que estén entre [0,1] o [-1,1] y estandarizarlos (por ejemplo dividir todos los elementos por el máximo valor que pueden tomar nuestros datos) porque:**
    
**R//=** Proyecta los datos originales en las direcciones que maximizan la varianza.

11. **Usando el algoritmo PCA de la librería sklearn.decomposition, ¿cómo especifico que quiero conservar el 80% de la varianza contenida en los datos?**
    
**R//=** n_components = 0.80