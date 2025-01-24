# Curso Práctico de SQL

1. **¿Por qué cambió de nombre de SEQUEL a SQL?**
   
**R//=** Derechos de autor.

2. **¿Cuál de estos es un operador unario?**
 
**R//=** Proyección

3. **Las funciones de agregación se usan en conjunto con la sentencia:**
   
**R//=** GROUP BY

4. **¿Qué función te permite usar una tabla remota como origen de datos?**
   
**R//=** dblink

5. **¿Cuál de los tipos de JOIN produce un producto cartesiano completo?**
    
**R//=** CROSS JOIN

6. **¿Cuál es el "comodín" para sustituir un carácter?**
    
**R//=** _

7. **¿Cuál es el ordenamiento por defecto al usar ORDER BY?**
    
**R//=** Ascendente

8. **¿Qué número de tupla regresa la consulta LIMIT 1 OFFSET 105?**
    
**R//=** 106

9. **FETCH FIRST 15 ROWS ONLY es igual a LIMIT 15**
    
**R//=** Verdadero

10. **Se pueden usar subqueries solo en la cláusula FROM**
    
**R//=** Falso

11. **¿Cómo se genera un rango dinámico cuando no se conocen los parámetros del set a buscar?**
    
**R//=** Usando un Subquery

12. **DATE_PART se usa para fechas y EXTRACT no.**
    
**R//=** Falso

13. **¿En qué sentencias se usa DATE_PART?**
    
**R//=** SELECT y WHERE

14. **¿A qué es equivalente el operador :: de postgreSQL?**
    
**R//=** CAST

15. **¿Qué resultado arroja el operador * entre dos rangos?**

**R//=** Los elementos en común entre los dos.
    
16. **¿Cuál es la principal diferencia entre usar MAX y LIMIT para obtener máximos?**
    
**R//=** MAX sirve para subsets.

17. **La única limitante al hacer un self join es que no se puede usar GROUP BY**
    
**R//=** Falso

18. **¿En qué se convierte un LEFT JOIN al añadir la cláusula: WHERE tabla_2.id IS NULL?**
    
**R//=** En un exclusive LEFT JOIN

19. **La diferencia simétrica contiene...**
    
**R//=** Los elementos que no pertenecen a ambas tablas.

20. **¿Por qué es deseable usar ROW_NUMBER para generar el triángulo?**
    
**R//=** Porque es independiente del orden de los datos.

21. **Para generar un rango se usan el valor inicial, el valor final y...**
    
**R//=** El delta

22. **Se pueden utilizar expresiones regulares para filtrar resultados.**
    
**R//=** Verdadero

23. **¿Cuál de las siguientes es una ventaja de las BD distribuidas?**
    
**R//=** Desarrollo modular.

24. **En un query distribuido la latencia de red se considera despreciable.**
    
**R//=** Falso

25. **Desventaja de aplicar sharding.**
    
**R//=** Baja elasticidad

26. **¿Qué se enfocan en reducir las window functions?**
    
**R//=** Self JOIN

27. **DENSE_RANK elimina los espacios en un RANK regular.**
    
**R//=** Verdadero