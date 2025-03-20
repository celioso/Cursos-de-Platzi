# Curso Práctico de Bases de Datos en AWS

1. **¿De cuánto es el periodo de retención de una base de datos en RDS?**
   
**R//=** 1 a 35 días.

2. **¿Cuál base de datos AWS es recomendable utilizar para manejar cargas productivas?**
 
**R//=** Aurora.

3. **¿Qué se debe activar en la RDS para conectarnos a la BD externamente?**
   
**R//=** Habilitar la opción de public access.

4. **¿Cuál base de datos solo permite 1 BD por instancia creada?**
   
**R//=** Oracle.

5. **Al borrar nuestra RDS ¿qué sucede con los backups manuales?**
    
**R//=** Quedan intactos.

6. **¿Con qué granularidad puedo reestablecer un backup automático?**
    
**R//=** Segundos.

7. **Actualmente eres dueño de un blog que publica noticias de actualidad. Recientemente has notado un incremento en su base de datos principal a nivel de lectura y quieres minimizar esta carga para que las solicitudes de escritura no se vean afectadas ¿qué deberías hacer?**
    
**R//=** Implementar Replicas de Lectura.

8. **¿Qué funcionalidad debería usar de RDS para mejorar la disponibilidad de su base de datos?**
    
**R//=** Multi AZ.

9. **Necesitas migrar tu base de datos posgreSQL de on-premise a la nube a RDS. ¿Qué tipo de migración deberías realizar si quieres mantener el mismo motor?**
    
**R//=** Homogénea.

10. **¿Cuál de los siguientes casos sería una migración Heterogénea en AWS?**
    
**R//=** Oracle a Aurora.

11. **¿Cuál base de datos tiene mejor rendimiento en RDS?**
    
**R//=** Aurora.

12. **¿Cuál es el límite de tamaño para una base de datos en Aurora?**
    
**R//=** 64TB.

13. **El CIO de la empresa te ha pedido que migres la base de datos actual PosgreSQL a la nube de AWS, sin embargo, te ha hecho énfasis en que quiere una base de datos de alto desempeño, altamente disponible y de ser posible serverless ¿cuál base de datos recomendarías utilizar?**
    
**R//=** Aurora.

14. **DynamoDB es una base de datos relacional.**
    
**R//=** Falso.

15. **Al momento de crear la tabla DynamoDB ¿puedes habilitar la opción de encriptación?**

**R//=** Verdadero.

16. **¿En cuál de los siguientes escenarios utilizarías DynamoDB?**

**R//=** Almacenamiento de sesiones de una página web.

17. **En DynamoDB ¿en dónde se almacenan los datos?**

**R//=** Particiones.

18. **¿Es recomendable hacer operaciones Scan en una tabla?**

**R//=** No es recomendado hacerlo.

19. **¿Pueden las operaciones Scan consumirse de las lecturas aprovisionadas de una tabla?**

**R//=** Sí - Depende del tamaño de la tabla y del valor aprovisionado.

20. **¿Cuál es la principal funcionalidad de los streams en DynamoDB?**

**R//=** Proporciona una secuencia ordenada por tiempo de cambios del nivel del elemento en cualquier tabla.

21. **¿Por cuánto tiempo se guarda información de los streams?**

**R//=** 24 Horas.

22. **¿Qué podrías hacer para mejorar el rendimiento de una DynamoDB?**

**R//=** Utilizar DAX.

23. **¿Es recomendable aprovisionar una tabla de DynamoDB con capacidad de lectura y escritura altas para evitar problemas de rendimiento?**

**R//=** Falso.

24. **¿Qué uso tiene la sort key en DynamoDB?**

**R//=** Permite realizar ordenamiento de la tabla por el campo de que tenga la sort key.

25. **¿Con qué motor de base de datos es compatible Aurora serverless?**

**R//=** MySQL