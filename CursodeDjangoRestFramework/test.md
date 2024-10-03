# Curso de Django Rest Framework

1. **¿Por qué se utiliza JSON en las interacciones entre sistemas a través de APIs?**
   
**R//=** Porque permite estructurar datos de forma legible y eficiente.

2. **¿Qué debes agregar en settings.py para integrar Django REST Framework al proyecto?**
 
**R//=** Añadir 'rest_framework' en INSTALLED_APPS

3. **¿Qué herramienta automatiza el formato de código en Django conforme al estándar PEP8?to para hacer un proyecto de ciencia de datos?**
   
**R//=** Black

4. **Si tienes un modelo en Django, ¿cómo puedes convertir los datos de una instancia de ese modelo en formato JSON usando DRF?**
   
**R//=** Usando un Serializador

5. **¿Cuál es una de las principales funciones de los Serializadores en DRF?**
    
**R//=** Transformar objetos Python a JSON

6. **¿Qué método HTTP usamos para modificar un recurso en una vista de detalle en Django?**
    
**R//=** PUT.

7. **Al realizar una eliminación de un recurso con DELETE en Django, ¿qué código de estado HTTP se debe devolver?**
    
**R//=** 204 No Content

8. **Si un recurso es modificado con éxito utilizando PUT, ¿qué sucede después en la base de datos?**
    
**R//=** Se actualiza el recurso con los nuevos datos usando el método save() del serializador

9. **¿Cómo aseguramos que los datos de una solicitud PUT sean válidos antes de guardar un recurso modificado en Django?**
    
**R//=** Usamos el método is_valid() del serializador

10. **¿Qué clase de permiso se debe usar en Django REST Framework para asegurar que un usuario esté autenticado antes de acceder a un endpoint?**
    
**R//=** IsAuthenticated

11. **Si se deben validar múltiples campos en conjunto, como el estado de vacaciones y un número de contacto, ¿qué método se debe usar?**
    
**R//=** Usar el método validate para acceder a todos los campos y aplicar la lógica.

12. **¿Qué método usarías para retornar un error de validación en formato JSON cuando se envían datos incorrectos en el API?**
    
**R//=** raise serializers.ValidationError

13. **¿Qué método de Django se usa para construir URLs dinámicas en pruebas unitarias?**
    
**R//=** reverse()

14. **¿Qué sucede cuando un usuario anónimo alcanza el límite de solicitudes definido en Django REST?**
    
**R//=** Recibe un error “Too Many Requests”

15. **¿Qué técnica puedes implementar en Django REST para limitar la cantidad de solicitudes que un usuario puede hacer en un período de tiempo?**
    
**R//=** Throttling