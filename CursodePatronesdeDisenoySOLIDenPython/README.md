# Curso de Patrones de Diseño y SOLID en Python

## Patrones de Diseño y Principios SOLID en Python

El mundo del desarrollo de software ha evolucionado considerablemente en las últimas décadas gracias a dos pilares fundamentales: los patrones de diseño, que llevan 30 años organizando el caos del código, y los principios SOLID, que llevan 20 años haciendo la vida de los programadores más sencilla. Juntos, permiten construir sistemas más mantenibles, escalables y flexibles. En este curso, aprenderemos a aplicar ambos conceptos mientras desarrollamos un procesador de pagos en Python, mejorando el código a lo largo del proceso para crear una solución robusta y eficiente.

## ¿Por qué es importante aprender patrones de diseño y principios SOLID?

Los patrones de diseño y los principios SOLID ofrecen soluciones probadas a problemas comunes en el desarrollo de software. Aunque llevan décadas en uso, siguen siendo esenciales porque:

- Mejoran la mantenibilidad del código.
- Aumentan la flexibilidad y escalabilidad.
- Facilitan la implementación de pruebas, tanto unitarias como de integración.
- Optimizan el rendimiento del código.
- Mejoran la experiencia de desarrollo, haciéndolo más claro y ordenado.

## ¿Cómo aplicaremos los patrones de diseño y los principios SOLID?

A lo largo del curso, trabajaremos sobre una base de código inicial que no cumple con ninguno de los principios SOLID ni sigue patrones de diseño. Esto nos permitirá ver, de manera práctica, cómo transformar el código paso a paso. Usaremos Python para construir un sistema de procesamiento de pagos, aplicando mejoras incrementales entre cada lección, las cuales servirán como retos para que los estudiantes reflexionen y discutan en la sección de comentarios.

## ¿Qué conocimientos previos necesitas?

- **Curso de Python básico**: Fundamental para entender los conceptos de lenguaje aplicados durante el curso.
- **Curso de pruebas unitarias en Python**: Opcional, pero útil para facilitar las pruebas del código que desarrollaremos.

Además, es imprescindible contar con un entorno de desarrollo adecuado que incluya:

- Un editor de código.
- Un manejador de paquetes como PIP, UV o Poetry.
- Un entorno virtual para manejar dependencias.
- GIT y GitHub para el manejo del repositorio del proyecto.

Los principios SOLID son un conjunto de cinco directrices que ayudan a los desarrolladores a crear software más mantenible y escalable. Cada letra representa un principio:

1. **S**: Single Responsibility Principle (SRP) - Cada clase debe tener una  nica responsabilidad.
2. **O**: Open/Closed Principle (OCP) - Las entidades deben estar abiertas a la extensión, pero cerradas a la modificación.
3. **L**: Liskov Substitution Principle (LSP) - Los objetos de una clase derivada deben poder reemplazar objetos de la clase base sin afectar el comportamiento.
4. **I**: Interface Segregation Principle (ISP) - Es mejor tener varias interfaces específicas que una interfaz general.
5. **D**: Dependency Inversion Principle (DIP) - Las dependencias deben ser abstraídas; las clases de alto nivel no deben depender de las de bajo nivel.
Estos principios aumentan la calidad del código y facilitan su mantenimiento.

## Principio de Responsabilidad Única (SRP) en Python

El principio de responsabilidad única es uno de los pilares de la construcción de software, establecido por Robert C. Martin. Este principio indica que una clase o función debe tener solo una razón para cambiar, lo que mejora la mantenibilidad y reduce la complejidad. Implementarlo no solo facilita las pruebas unitarias, sino que también incrementa la reutilización del código y disminuye el acoplamiento, promoviendo un sistema más cohesivo y fácil de escalar.

### ¿Qué implica el principio de responsabilidad única?

- **Responsabilidad única**: Una clase o función debe encargarse de una única tarea, lo que evita que se ocupe de múltiples aspectos.
- **Razón para cambiar**: El código debe tener una única causa para ser modificado, asegurando que los cambios sean claros y controlados.

### ¿Qué problemas soluciona este principio?

- **Mantener el código**: Al dividir responsabilidades, el código se vuelve más fácil de mantener y más económico de modificar.
- **Reutilización del código**: Las funciones con una única responsabilidad pueden ser utilizadas en diferentes contextos sin necesidad de duplicar código.
- **Pruebas unitarias más simples**: Las funciones con responsabilidades bien definidas son más fáciles de probar, reduciendo la carga de trabajo en el desarrollo.

### ¿Cómo saber cuándo aplicar el principio de responsabilidad única?

- **Múltiples razones para cambiar**: Si una clase o función tiene varias razones para ser modificada, es un buen indicio de que tiene más responsabilidades de las que debería.
- **Alta complejidad y difícil mantenimiento**: Si se encuentra complicado añadir nuevas funcionalidades o corregir errores, es probable que la falta de una clara definición de responsabilidades esté afectando el código.
- **Dificultad para realizar pruebas unitarias**: Si preparar una prueba implica mucho trabajo o configurar demasiados elementos, es señal de que el principio no se está siguiendo correctamente.
- **Duplicación de código**: Si una funcionalidad, como una validación, está replicada en varios lugares, se debería extraer a una única función y reutilizarla donde sea necesario.

### ¿Qué hacer cuando encuentras duplicación de responsabilidades?

Cuando se identifica la duplicación de código o responsabilidades mal distribuidas, el enfoque ideal es reorganizar ese código en una función específica que pueda ser reutilizada en todos los puntos necesarios. Esto no solo reduce el trabajo redundante, sino que también asegura que los cambios futuros se realicen en un solo lugar.

## Procesador de Pagos con Stripe en Python

Construir un procesador de pagos es un desafío común en la industria del desarrollo de software, y en este curso lo haremos aplicando los principios SOLID y patrones de diseño. El código inicial será básico y lo iremos refactorizando para ajustarlo a buenas prácticas a medida que avanzamos.

### ¿Cómo funciona el código inicial?

El código comienza con una clase simple que tiene un único método llamado processTransaction. Este método recibe dos parámetros: customerData y paymentData. Dentro de este método, se llevan a cabo tres validaciones básicas:

- Verificación de que el cliente tiene nombre.
- Verificación de que tiene información de contacto.
- Comprobación de que tiene una fuente de pago válida.

### ¿Cómo se realiza el procesamiento del pago?

El procesamiento del pago se gestiona a través de la integración con Stripe. Utilizamos la clave API de Stripe, que se almacena en las variables de entorno, y creamos un cargo basado en la cantidad y los datos proporcionados. El método maneja cualquier error con un bloque `try-except`, enviando una notificación si el pago falla.

### ¿Cómo se notifican los resultados?

El código incluye mecanismos para notificar al cliente por correo electrónico o mensaje de texto. Sin embargo, dado que no hay un servidor SMTP configurado, se usan ejemplos comentados para ilustrar cómo sería el envío de correos. En el caso de mensajes de texto, se utiliza un mock que simula una pasarela de envío de SMS.

### ¿Cómo se registra la información de la transacción?

Finalmente, el método guarda los detalles de la transacción en un archivo `transactions.log`. Este archivo contiene información como el nombre del cliente, el monto cobrado y el estado final del pago. Estos registros son útiles para futuras consultas y auditorías.

### ¿Qué modificaciones se pueden hacer al procesador?

El código es flexible, y podemos modificar la información del cliente y del pago para probar diferentes escenarios. Por ejemplo, podemos cambiar el nombre del cliente, la cantidad a cobrar y el método de pago, incluyendo el uso de tokens de tarjetas en lugar de números de tarjeta. Stripe proporciona varios tokens y números de prueba que pueden ser utilizados para simular transacciones en diferentes condiciones.

### ¿Qué papel juegan las variables de entorno?

Una parte esencial del código es la configuración de las variables de entorno, que se manejan mediante el módulo `.env` de Python. Este módulo carga la clave API de Stripe desde el archivo `.env`, lo que asegura que la clave esté protegida y accesible solo durante la ejecución del programa.

**Lecturas recomendadas**

[Stripe API Reference](https://stripe.com/docs/api)

## Aplicar el Principio de Responsabilidad Única (SRP)

Para aplicar el principio de responsabilidad única (SRP) sobre un procesador de pagos en Python, se debe organizar el código para que cada clase o método tenga una única responsabilidad. A continuación, te explicaré cómo refactorizar el código original, identificando responsabilidades y dividiéndolo en componentes más manejables.

### ¿Cómo estaba estructurado el código original?

El código inicial contenía varias responsabilidades en una sola clase llamada `PaymentProcessor`. Estas responsabilidades incluían:

- Validación de datos del cliente
- Validación de los datos de pago
- Procesamiento del pago con Stripe
- Envío de notificaciones (email o SMS)
- Registro de la transacción en logs

Este enfoque infringe el principio SRP, ya que una sola clase está encargada de múltiples tareas, lo que hace difícil mantener y escalar el código.

### ¿Cómo refactorizar el código aplicando SRP?

El primer paso fue identificar las distintas responsabilidades dentro de la clase. Se encontraron cuatro bloques importantes de responsabilidades:

1. Validación de datos del cliente.
2. Validación de datos del pago.
3. Procesamiento del pago.
4. Notificación y logging de transacciones.

### ¿Cómo organizar las nuevas clases?

**1. ¿Cómo separar la validación de datos del cliente?**
Se creó una clase `CustomerValidation` con el método `validate`, encargado exclusivamente de validar los datos del cliente, como nombre y contacto. El código de validación fue extraído de la clase `PaymentProcessor` y movido aquí.

**2. ¿Cómo manejar la validación del pago?**

Al igual que en la validación del cliente, se creó una clase `PaymentDataValidator`, encargada de validar los datos de pago, como la fuente de pago (por ejemplo, si se provee una tarjeta o token válido).

**3. ¿Cómo procesar el pago sin romper SRP?**

El procesamiento de pago es una responsabilidad que corresponde a la interacción con la API de Stripe. En este caso, se mantuvo esta lógica en una clase llamada **StripePaymentProcessor**, asegurando que solo procese pagos.

**4. ¿Cómo gestionar las notificaciones?**

Se creó una clase `Notifier`, que maneja el envío de notificaciones, ya sea por email o SMS. Esto ayuda a aislar la lógica de notificaciones del resto del código, permitiendo cambios futuros sin afectar otras partes.

**5. ¿Cómo registrar logs de transacciones?**

Se añadió una clase `TransactionLogger` dedicada al registro de las transacciones. Esta clase se encarga de capturar información de la transacción y guardarla en los logs del sistema.

**¿Cómo coordinar todas estas clases?**

Se unieron todas estas clases bajo una nueva entidad `PaymentService`, que orquesta la interacción entre ellas. Esta clase permite coordinar la validación, procesamiento, notificaciones y registro de transacciones de manera eficiente y con SRP.

**¿Cómo se manejan los errores y excepciones?**

Cada clase maneja sus propias excepciones, asegurando que las validaciones levanten errores específicos cuando los datos son incorrectos. Además, se agregó manejo de excepciones con `try-except` para capturar fallas en el procesamiento de pagos, lo que permite gestionar errores de manera controlada.

## Principio Abierto/Cerrado (OCP) en Python

El principio abierto-cerrado (Open-Closed Principle) es clave para mantener la flexibilidad y estabilidad en el desarrollo de software, permitiendo que el código sea ampliado sin ser modificado. Este principio garantiza que el software pueda evolucionar de manera eficiente sin afectar las funcionalidades ya probadas, lo que es fundamental en un entorno de cambio constante como el de las empresas tecnológicas.

### ¿Qué es el principio abierto-cerrado?
El principio abierto-cerrado establece que el software debe estar abierto para su extensión pero cerrado para su modificación. Esto significa que es posible añadir nuevas funcionalidades sin alterar el código existente, lo que ayuda a evitar errores y mantiene la estabilidad del sistema.

### ¿Cómo se aplica en el desarrollo de software?

- **Extensión sin modificación**: se agregan nuevas características utilizando mecanismos como interfaces, abstracciones o polimorfismos. En lenguajes como Python, estas herramientas son parte del propio lenguaje y permiten ampliar comportamientos sin alterar la base de código original.
- **Cerrado para modificación**: protege el código validado, encapsulando las funcionalidades y asegurando que las nuevas extensiones no rompan lo que ya está en uso.

### ¿Cuáles son los beneficios de aplicarlo?

- **Menos errores**: al no tocar el código existente, se minimizan los errores derivados de cambios imprevistos.
- **Actualizaciones más rápidas**: la extensión del software se vuelve más ágil, lo que es crucial cuando hay cambios constantes de requisitos en las empresas.
- **Estabilidad del sistema**: el código probado y validado permanece inalterado, lo que facilita el desarrollo de nuevas funcionalidades sin riesgos.

### ¿Cuándo deberías aplicar el principio abierto-cerrado?

Este principio es útil cuando necesitas añadir nuevas funcionalidades sin afectar el código existente. Un ejemplo común es la integración de una nueva pasarela de pagos en un sistema ya funcional o agregar un método de notificación sin cambiar las implementaciones actuales. También es recomendable en contextos donde los requisitos del sistema cambian frecuentemente, como en la construcción de plataformas de pago o servicios con muchas interacciones externas.

### ¿Cómo puedes aplicarlo en tu día a día?

Reflexiona sobre cómo usarías este principio al agregar nuevas pasarelas de pago al sistema que estás desarrollando. Piensa también en qué momentos, durante tu experiencia, has extendido funcionalidades sin modificar la base de código. Este enfoque te permitirá adaptarte rápidamente a las necesidades cambiantes de la empresa mientras mantienes la estabilidad.

El **Principio Abierto/Cerrado** (Open/Closed Principle, OCP) establece que **una clase debe estar abierta para su extensión, pero cerrada para su modificación**. Esto significa que deberíamos poder agregar nuevas funcionalidades a una clase sin cambiar su código existente, lo cual es esencial para preservar la estabilidad del software y minimizar los errores.

Una forma común de aplicar el OCP en Python es utilizando la **herencia** o la **composición** para extender el comportamiento de las clases sin modificar su código base.

### Ejemplo de una Clase que Viola el OCP

Imaginemos una clase que calcula el descuento para una tienda en línea. La clase `DiscountCalculator` contiene lógica para aplicar distintos tipos de descuentos. Sin embargo, si necesitamos agregar un nuevo tipo de descuento, tendríamos que modificar la clase, violando así el principio OCP.

```python
class DiscountCalculator:
    def calculate(self, price, discount_type):
        if discount_type == "percentage":
            return price * 0.90  # 10% de descuento
        elif discount_type == "fixed":
            return price - 20  # Descuento fijo de $20
        else:
            return price
```

Si se agrega un nuevo tipo de descuento, como un descuento para clientes frecuentes, sería necesario modificar el método `calculate`, lo que va en contra del principio OCP.

### Solución: Aplicando OCP

Para aplicar el OCP, separamos cada tipo de descuento en clases independientes, que pueden extenderse sin necesidad de modificar la lógica existente. Utilizaremos **polimorfismo** mediante una clase base abstracta `DiscountStrategy`.

```python
from abc import ABC, abstractmethod

class DiscountStrategy(ABC):
    @abstractmethod
    def apply_discount(self, price):
        pass

class PercentageDiscount(DiscountStrategy):
    def apply_discount(self, price):
        return price * 0.90  # 10% de descuento

class FixedDiscount(DiscountStrategy):
    def apply_discount(self, price):
        return price - 20  # Descuento fijo de $20

class NoDiscount(DiscountStrategy):
    def apply_discount(self, price):
        return price  # Sin descuento

class DiscountCalculator:
    def __init__(self, strategy: DiscountStrategy):
        self.strategy = strategy

    def calculate(self, price):
        return self.strategy.apply_discount(price)
```

### Explicación

- **Clase `DiscountStrategy`**: Es una clase abstracta que define el método `apply_discount`, el cual debe ser implementado por cada tipo de descuento.
- **Clases `PercentageDiscount`, `FixedDiscount` y `NoDiscount`**: Cada una implementa una estrategia de descuento específica. Puedes agregar nuevas estrategias sin modificar las clases existentes.
- **Clase `DiscountCalculator`**: Su método `calculate` utiliza una instancia de `DiscountStrategy` para aplicar el descuento. Esta clase no necesita ser modificada al agregar un nuevo tipo de descuento.

### Uso del Código

Podemos usar `DiscountCalculator` con distintas estrategias de descuento de forma flexible:

```python
# Crear instancias de estrategias de descuento
percentage_discount = PercentageDiscount()
fixed_discount = FixedDiscount()
no_discount = NoDiscount()

# Aplicar descuentos
calculator = DiscountCalculator(percentage_discount)
print(f"Precio con descuento del 10%: ${calculator.calculate(100)}")

calculator.strategy = fixed_discount
print(f"Precio con descuento fijo de $20: ${calculator.calculate(100)}")

calculator.strategy = no_discount
print(f"Precio sin descuento: ${calculator.calculate(100)}")
```

### Ventajas de Aplicar el Principio OCP

1. **Extensibilidad**: Puedes agregar nuevos tipos de descuento sin modificar las clases existentes.
2. **Mantenibilidad**: Al no modificar el código base, reduces el riesgo de introducir errores en el código existente.
3. **Flexibilidad**: Puedes intercambiar estrategias de descuento dinámicamente, haciendo el sistema más adaptable.

### Conclusión

Aplicar el Principio Abierto/Cerrado permite que el sistema sea extensible y flexible. Este enfoque de diseño reduce la posibilidad de errores al minimizar los cambios en el código existente y fomenta la reutilización de componentes a través del polimorfismo.