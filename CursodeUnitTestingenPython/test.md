# Curso de Unit Testing en Python

1. **¿Qué tipo de prueba se debe realizar para validar el correcto funcionamiento de componentes individuales del código?**
   
**R//=** Pruebas unitarias

2. **Si un sistema comienza a fallar cuando diferentes componentes interactúan, ¿qué tipo de prueba no fue efectiva?**
 
**R//=** ATesting de Integración

3. **Para verificar si el software cumple las expectativas del usuario final, ¿qué prueba debe realizarse?**
   
**R//=** Pruebas de aceptación

4. **¿Cuál es la ventaja principal de automatizar las pruebas con Python?**
   
**R//=** Detectar errores más rápido y en mayor cantidad

5. **En una prueba unitaria con Python, si una función devuelve 5 pero se esperaba 7, ¿cómo se detectaría el error?**
    
**R//=** Usando assert para comparar el resultado esperado con el real

6. **¿Qué método en TestCase se utiliza para preparar los recursos antes de cada prueba?**
    
**R//=** setUp()

7. **¿Qué comando se usa para obtener una salida más detallada al ejecutar las pruebas unitarias?**
    
**R//=** pytest -v

8. **¿Cuál es el propósito principal del método assertEqual en UnitTest?**
    
**R//=** Verificar que dos valores son iguales.

9. **¿Qué método sería adecuado para verificar si un valor está dentro de una lista en UnitTest?**
    
**R//=** assertIn

10. **¿Cuál es el propósito principal del decorador @skip en una prueba unitaria?**
    
**R//=** Omitir temporalmente una prueba que aún no debe ejecutarse

11. **¿Cuál es la diferencia clave entre @skipIf y @skipUnless?**
    
**R//=** @skipIf salta una prueba si una condición es verdadera, mientras que @skipUnless salta una prueba si la condición es falsa.

12. **¿Cuál sería un buen nombre para una prueba de un método withdraw que reduce el saldo con un valor positivo?**
    
**R//=** test_withdraw_positive_amount_reduce_balance

13. **¿Cómo puedes ejecutar las pruebas de Doctest en un archivo Python?**
    
**R//=** Usando el comando python -m doctest nombre_archivo.py

14. **¿Qué comando en Coverage te permite generar un reporte visual de la cobertura de código en HTML?**
    
**R//=** coverage html

15. **Si quieres analizar qué partes de tu código no han sido probadas, ¿qué deberías hacer?**
    
**R//=** Revisar el reporte HTML de Coverage