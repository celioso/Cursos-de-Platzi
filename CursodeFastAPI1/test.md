# Curso de Databricks: Arquitectura Delta Lake

1. **Para modelar datos en una API creada con FastAPI, ¿qué herramienta se recomienda utilizar?**
   
**R//=** Pydantic

2. **Si necesitas documentar automáticamente los endpoints de tu API, ¿qué característica de FastAPI te sería útil?**
 
**R//=** Integración con OpenAPI

3. **Si necesitas validar que un campo de email en un modelo de datos sea un correo electrónico válido, ¿qué deberías hacer?**
   
**R//=** Utilizar un validador de email específico en Pydantic.

4. **¿Cómo podrías evitar que el ID de un modelo 'Customer' sea enviado por el cliente al crear un nuevo registro en FastAPI?**
   
**R//=** Usando herencia para crear un modelo 'CustomerCreate' sin el ID.

5. **¿Cómo podrías implementar un sistema de IDs incrementales para los registros de 'Customer' en una base de datos en memoria en FastAPI?**
    
**R//=** Contando los elementos en la lista de 'Customers' y usando ese número como el nuevo ID.

6. **¿Cuál es la ventaja principal de usar SQLModel en FastAPI para manejar bases de datos?**
    
**R//=** Permite integrar modelos de Pydantic con SQLAlchemy para manejar bases de datos sin escribir SQL.

7. **¿Cómo se asegura que un campo en un modelo SQLModel se guarde en la base de datos?**
    
**R//=** Definiendo el campo con la clase Field y estableciendo un valor por defecto.

8. **¿Qué método HTTP deberías usar para eliminar un recurso en FastAPI?**
    
**R//=** DELETE

9. **Si necesitas obtener un recurso específico por su ID en FastAPI, ¿qué método HTTP es más apropiado?**
    
**R//=** GET

10. **¿Cuál es el método HTTP más adecuado para actualizar parcialmente un recurso en FastAPI?**
    
**R//=** PATCH

11. **¿Qué código de estado HTTP es más apropiado para indicar que un recurso ha sido actualizado exitosamente en FastAPI?**
    
**R//=** HTTP 201 Created

12. **¿Cuál es el propósito de utilizar entornos virtuales al trabajar con FastAPI?**
    
**R//=** Permiten mantener las dependencias del proyecto separadas de otros proyectos.

13. **¿Qué método se debe utilizar para crear automáticamente todas las tablas en la base de datos al iniciar una aplicación FastAPI?**
    
**R//=** Usar el método create_all_tables en el evento de inicio de la aplicación.

14. **Para implementar una relación muchos-a-muchos entre clientes y planes en FastAPI, ¿qué elemento adicional es necesario crear?**
    
**R//=** Una tabla intermedia

15. **Para validar que un 'customer' y un 'plan' existen antes de crear una relación en FastAPI, ¿qué deberías hacer?**

**R//=** Usar una excepción HTTP 404 si no existen
    
16. **Si quieres validar un campo adicional en tu modelo, ¿qué deberías hacer?**
    
**R//=** Agregar un nuevo Field Validator para el campo

17. **¿Cuál es la ventaja principal de usar un StaticPool en la configuración de pruebas de una base de datos?**
    
**R//=** Permite crear una base de datos temporal en memoria, evitando múltiples archivos.

18. **¿Qué función cumple el método 'drop_all' en el contexto de pruebas de bases de datos?**
    
**R//=** Borra todas las tablas para limpiar la memoria antes de nuevas pruebas.

19. **Al modificar un endpoint que no está relacionado directamente con 'customer', ¿cómo podrías asegurarte de que no afectaste su funcionalidad?**
    
**R//=** Ejecutando pruebas automatizadas

20. **¿Cuál es la ventaja principal de utilizar relaciones muchos a muchos en bases de datos relacionales?**
    
**R//=** Permite evitar la duplicación de datos y organizar la información de manera más eficiente.

21. **Si necesitas actualizar un campo específico de un 'customer' sin afectar otros campos, ¿qué técnica deberías usar?**
    
**R//=** Enviar solo los campos a actualizar en el body del request

22. **¿Cuál es la ventaja principal de utilizar un Api router en una aplicación FastAPI?**
    
**R//=**  Permite agrupar endpoints para facilitar el soporte y cambios sin modificar todos los endpoints.

23. **Al modificar el modelo de datos en FastAPI y agregar un nuevo campo, ¿cuál es la mejor práctica para reflejar estos cambios en la base de datos?**
    
**R//=** Utilizar migraciones

24. **Si quieres registrar un middleware en tu aplicación FastAPI, ¿qué método deberías utilizar?**
    
**R//=** app.middleware

25. **¿Por qué es importante sobrescribir las dependencias en una aplicación FastAPI durante las pruebas?**
    
**R//=** Para que las pruebas usen una base de datos de testing en lugar de la de producción.

26. **Al implementar autenticación en FastAPI, ¿qué código de estado HTTP deberías retornar si las credenciales proporcionadas son incorrectas?**
    
**R//=** 401 Unauthorized