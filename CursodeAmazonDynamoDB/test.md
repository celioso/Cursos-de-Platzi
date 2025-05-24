# Curso de Amazon DynamoDB

1. **En una base de datos NoSQL debo definir previamente todo el esquema.**
   
**R//=** Falso

2. **DynamoDB requiere de instalación y mantenimiento de infrastructura.**
 
**R//=** Falso. Es un servicio serverless.

3. **¿Cuál es la manera más óptima de crear una llave primaria?**
   
**R//=** Todas las opciones son correctas.

4. **Un usuario con programmatic access puede acceder a AWS usando**
   
**R//=** Solo el SDK o la línea de comando.

5. **¿Cuál es el comando usado para cargar data a dynamoDB usando el AWS CLI?**
    
**R//=** aws dynamodb put-item ....

6. **Una característica de las consultas tipo Scan es**
    
**R//=** Consultan todos los items de la tabla uno a uno hasta encontrar al que cumpla con los filtros.

7. **Los índices globales**
    
**R//=** Crean una tabla réplica basada en el nuevo atributo para poder ejecutar queries sobre él mismo sin depender de la llave primaria.

8. **Al realizar un query sobre un item con un peso de 16kb se usan**
    
**R//=** 4 Unidades de lectura.

9. **Los streams de dynamodb pueden ser usados para**
    
**R//=** Crear un filtro de registros y replicar los objetos a una nueva tabla.

10. **Los exports al S3 son usados mayormente para**
    
**R//=** Realizar queries a la data desde otro servicio AWS como Athena o Glue.

11. **Cloudwatch Insight con dynamodb nos permite**
    
**R//=** Obtener una visión clara sobre el comportamiento de una tabla en DynamoDB, incluyendo la identificación de los ítems más requeridos.

12. **La función principal del TTL (Time To Live) es**
    
**R//=** Eliminar registros expirados basados en una ventana de tiempo determinada por el administrador de la tabla.

13. **DynamoDB escala automáticamente**
    
**R//=** Agregando más unidades de lectura/escritura basado en las métricas de cloudwatch.