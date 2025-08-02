# Curso de Regresión Logística con Python y scikit-learn

1. **¿Para qué tipo de tareas funciona el algoritmo de regresión logística?**
   
**R//=** Clasificación

2. **¿Cuál es la función base en la regresión logística?**
 
**R//=** Sigmoid

3. **NO es una ventaja de la regresión logística:**
   
**R//=** Inferencia en la importancia de los datos.

4. **Una desventaja de la regresión logística es que se degrada su performance ante datasets linealmente no separables. ¿Esto es verdadero o falso?**
   
**R//=** Verdadero

5. **¿Con qué tipo de datasets es preferible usar regresión logística?**
    
**R//=** Binomial

6. **El rate de probabilidad de ocurrencia sobre probabilidad de no ocurrencia tipo (0.80 / 0.20 = 4) se denomina:**
    
**R//=** Odds

7. **¿Cuál es la fórmula de la regresión logística?**
    
**R//=** 
   $$
   \hat{y} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
   $$


8. **¿Cómo se denomina al proceso para llevar todos los datos de un dataset a una misma escala numérica?**
    
**R//=** Normalización

9. **Eliminar valores nulos hace parte del proceso de preparación de datos en regresión logística:**
    
**R//=** Verdadero

10. **¿Por qué es necesario realizar un análisis exploratorio de datos antes de entrenar un modelo de machine learning?**
    
**R//=** Ayuda a entender el comportamiento de los datos y cómo trabajarlos.

11. **Un dataset en cuya variable objetivo es true o false en cuanto a detección de spam en un mail es considerado:**
    
**R//=** Binomial

12. **¿Evaluando el MLE es preferible un resultado menor o mayor?**
    
**R//=** Es indiferente

13. **La función de coste evalúa las diferencias en la predicción del modelo y el valor real a predecir. ¿Esto es verdadero o falso?**
    
**R//=** Verdadero

14. **Para importar la librería necesaria para la regresión logística desde sklearn se debe usar la siguiente línea de código:**
    
**R//=** from sklearn.linear_model import LogisticRegression

15. **¿Qué se busca evitar al aplicar regularizers en regresión logística?**
    
**R//=** Overfitting

16. **¿Cuál es la técnica de regularización basada en el valor absoluto de los pesos?**
    
**R//=** L1

17. **¿Cómo se usa en scikit-learn la técnica de regresión logística multiclase que separa una clase y la evalúa contra el resto para clasificar?**
    
**R//=** ovr (One vs Rest)

18. **Todos los solvers en logistic regression funcionan con todas las combinaciones de regulatizers y tipo de regresion multiple:**
    
**R//=** Falso

19. **Para realizar la predicción de la clase puedo utilizar el método:**
    
**R//=** predict

20. **¿Qué retorna el método `predict_proba`?**
    
**R//=** La probabilidad para cada clase.