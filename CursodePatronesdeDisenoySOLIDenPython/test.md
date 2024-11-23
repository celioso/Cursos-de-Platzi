# Curso de Patrones de Diseño y SOLID en Python

1. **¿Cómo podrías aplicar el principio abierto/cerrado para añadir un nuevo método de notificación sin modificar el código existente?**
   
**R//=** Crear una nueva clase que implemente la interfaz de notificación y añadirla como opción en el servicio.

2. **Si tienes una clase que implementa múltiples comportamientos, como imprimir y escanear, pero solo necesitas uno de ellos, ¿qué principio de diseño deberías aplicar para evitar que la clase esté obligada a implementar métodos innecesarios?**
 
**R//=** Principio de Segregación de Interfaces (ISP)

3. **Cuando una interfaz tiene demasiados métodos y no todas las clases que la implementan utilizan todos esos métodos, ¿qué acción deberías considerar para mejorar la cohesión y reducir el acoplamiento?**
   
**R//=** Segregar la interfaz en interfaces más pequeñas

4. **Si decides modificar una interfaz que afecta a muchas clases, ¿qué principio de diseño te ayudaría a minimizar el impacto de esos cambios en el resto del sistema?**
   
**R//=** El principio de diseño que te ayudaría a minimizar el impacto de los cambios en el sistema es el Principio de Segregación de Interfaces (o ISP, por sus siglas en inglés: Interface Segregation Principle).

5. **Si deseas implementar el principio de inversión de dependencias en un servicio de pagos, ¿qué patrón de diseño sería más adecuado para evitar que la clase de alto nivel dependa de la creación de clases de bajo nivel?**
    
**R//=** Patrón Factory

6. **Al aplicar el principio de inversión de dependencias, ¿cuál es la principal ventaja de hacer que la clase de alto nivel dependa de interfaces en lugar de implementaciones concretas?**
    
**R//=** YFacilita la modificación y prueba del código

7. **¿Cómo podrías aplicar el principio de responsabilidad única para reestructurar un archivo de código que contiene múltiples clases relacionadas con el procesamiento de pagos?**
    
**R//=** Identificando y separando las clases en módulos específicos según su contexto y responsabilidad.

8. **Al reestructurar un código que implementa el principio de inversión de dependencias, ¿qué deberías considerar al definir las clases de alto y bajo nivel?**
    
**R//=** Asegurarte de que las clases de alto nivel dependan de interfaces y no de implementaciones concretas.

9. **Si quisieras mejorar la comunicación entre diferentes clases en un sistema de reservas, ¿qué tipo de patrón de diseño deberías considerar?**
    
**R//=** Patrón de diseño de comportamiento

10. **Si quisieras implementar un sistema de pagos que permita cambiar el método de procesamiento en tiempo de ejecución, ¿qué patrón de diseño sería el más adecuado?**
    
**R//=** El patrón de diseño estrategia, ya que permite intercambiar algoritmos de manera flexible.

11. **Si estás diseñando un sistema de notificación que debe cambiar su comportamiento en función de la disponibilidad de información de contacto, ¿cómo implementarías el patrón de estrategia para seleccionar el notificador adecuado?**
    
**R//=** Implementando un método que permita cambiar el notificador en tiempo de ejecución según la información disponible.

12. **Si deseas agregar funcionalidades a un objeto en tiempo de ejecución sin modificar su estructura original, ¿cuál sería la mejor estrategia a seguir?**
    
**R//=** Utilizar el patrón decorador para añadir responsabilidades dinámicamente.

13. **Al aplicar el patrón decorador, ¿qué importancia tiene definir un protocolo para el servicio original y los decoradores?**
    
**R//=** Es crucial para asegurar que tanto el servicio original como los decoradores implementen la misma interfaz, permitiendo así una integración fluida.

14. **Si quisieras construir un objeto complejo en Python y deseas que su creación sea flexible y escalable, ¿qué patrón de diseño deberías aplicar?**
    
**R//=** Patrón Builder

15. **Al implementar el patrón Builder, ¿cuál es la principal ventaja que obtienes en comparación con un constructor tradicional?**

**R//=** Mayor flexibilidad en la creación de objetos
    
16. **Al implementar el método build en un Builder, ¿qué validaciones son necesarias para asegurar que se puede crear la instancia del servicio de pagos?**
    
**R//=** Es necesario validar que todos los atributos requeridos estén presentes y no sean None antes de crear la instancia del servicio de pagos.

17. **Si tuvieras que diseñar un sistema de notificación para un servicio de pagos, ¿qué componente del patrón Observer sería esencial para gestionar las suscripciones?**
    
**R//=** Una clase manager que maneje los listeners y sus notificaciones.

18. **Al implementar el patrón Observer, ¿qué método es crucial en la interfaz listener para recibir actualizaciones de eventos?**
    
**R//=** Un método update que reciba un evento como parámetro.

19. **¿Cómo implementarías un sistema que notifique a diferentes componentes sobre el estado de un pago (exitoso o fallido) utilizando el patrón Observer en Python?**
    
**R//=** Definiendo una interfaz de listener con un método notify y gestionando los listeners en un manager.

20. **En el contexto del patrón Observer, si un listener no está suscrito al manager, ¿qué sucederá cuando se notifique un evento de pago?**
    
**R//=** El listener no recibirá la notificación del evento.

21. **Si estás diseñando un sistema de validación de pagos y necesitas que cada validación se realice de manera secuencial, ¿cuál patrón de diseño sería más adecuado para implementar?**
    
**R//=** Chain of Responsibility

22. **¿Qué patrón de diseño sería más adecuado implementar para crear un sistema de procesamiento de pagos que permita elegir entre diferentes pasarelas de pago según la situación?**
    
**R//=** El patrón de diseño más adecuado para crear un sistema de procesamiento de pagos que permita elegir entre diferentes pasarelas de pago según la situación es el Patrón Estrategia (Strategy Pattern).

23. **Al diseñar un sistema que valide información antes de procesar un pago, ¿qué patrón de diseño sería más efectivo para manejar múltiples validaciones de manera ordenada?**
    
**R//=** Cadena de Responsabilidades

24. **Al diseñar un sistema de procesamiento de pagos, ¿qué patrón de diseño sería más adecuado para permitir la adición de nuevos métodos de pago sin modificar el código existente?**
    
**R//=** Patrón de Estrategia

25. **Si necesitas que tu código sea flexible y fácil de extender, ¿cuál de los principios SOLID te ayudaría a evitar dependencias rígidas entre clases?**
    
**R//=** Principio de Inversión de Dependencias (DIP)