# Curso de PostgreSQL

1. **¿Qué es Postgresql?**
   
**R//=** Un motor de base de datos.

2. **¿Cuál versión de PostgreSQL es recomentable instalar?**
 
**R//=** La penúltima o antepenúltima versión

3. **¿Cuál es el comando que nos muestra la lista de todos los comandos tipo backslash disponibles en la consola?**
   
**R//=** \?

4. **Con PgAdmin sólo puedes modificar la estructura de base de datos de manera visual, no hay editor de código.**
   
**R//=** Falso

5. **¿Cuáles son los 3 archivos principales de configuración de PostgreSQL?**
    
**R//=** postgresql.conf, pg_hba.conf y pg_ident.conf.

6. **¿Cuál comando nos sirve para activar la medición de tiempo de las consultas en la consola?**
    
**R//=** \timing

7. **La relación entre objetos tangibles normalmente se traduce en:**
    
**R//=** Una tabla relacional.

8. **¿Cuál de los siguientes tipos de datos permite texto?**
    
**R//=** Character Varying

9. **Un forma de representar relaciones entre tablas es por medio de:**
    
**R//=** Llaves foráneas.

10. **La creación de llaves primarias y llaves foráneas en una tabla sólo se puede hacer con el comando CREATE TABLE.**
    
**R//=** Falso

11. **¿En qué consiste la creación de particiones de una tabla en Postgresql?**
    
**R//=** Dividir lo que es lógicamente una tabla grande en piezas físicas más pequeñas.

12. **Es importante crear ROLES y dejar de usar el predeterminado de Postgres porque:**
    
**R//=** Permite crear una estructura de permisos a la medida.

13. **¿Cuáles dos acciones podemos capturar de una tabla maestra usando llaves foráneas?**
    
**R//=** Al borrar y actualizar.

14. **El tipo de dato SERIAL sirve para:**
    
**R//=** Crear valores consecutivos para una columna.

15. **La única forma de insertar datos en una tabla usando un generador de datos aleatoreos es:**

**R//=** No existe una única forma.
    
16. **En la teoría de conjuntos, el INNER JOIN corresponde a:**
    
**R//=** Intersección

17. **ON CONFLICT DO nos permite:**
    
**R//=** Decidir qué hacer en caso de conflicto al insertar valores en una tabla.

18. **¿Es posible usar bloques condicionales de tipo IF en una consulta de PostgreSQL?**
    
**R//=** Sí. Usando CASE WHEN.

19. **La principal diferencia entre Vistas y Vistas Materializadas es:**
    
**R//=** Las Vistas no almacenan los datos en disco. Las Vistas Materializadas sí.

20. **¿En una PL/PgSQL se puede ejecutar código tanto SQL cómo no SQL?**
    
**R//=** Sí. Siempre y cuando el lenguaje indicado sea plpgsql.

21. **¿Los TRIGGERS pueden ser usados para ignorar inserts?**
    
**R//=** Sí. Se usa con BEFORE y RETURN NULL.

22. **Podemos usar DBLINK en medio de una consulta**
    
**R//=** Verdadero

23. **Dos comandos de cierre para una transacción son:**
    
**R//=** COMMIT y ROLLBACK

24. **Las extensiones nos permiten:**
    
**R//=** Extender la funcionalidad de PostgreSQL.

25. **¿Qué formato debes usar si queremos ver la consulta equivalente a nuestra base de datos al momento de hacer Backup?**
    
**R//=** Plain

26. **Vacuum Full es peligroso porque:**
    
**R//=** Bloquea las tablas durante el proceso.

27. **Las replicas funcionan usando:**
    
**R//=** Archivos WAL.

28. **¿Es posible actualizar la información en una réplica?**
    
**R//=** Falso

29. **Para alivianar los IOPS de la base de datos podemos:**
    
**R//=** Todas son correctas

30. **La principal limitación de rendimiento que tenemos en bases de datos es:**
    
**R//=** Los IOPS