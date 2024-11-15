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

## Aplicar el Principio Abierto/Cerrado (OCP)

El principio de abierto-cerrado promueve la creación de código que permita extender funcionalidades sin modificar el comportamiento original. A continuación, veremos cómo se puede aplicar este principio en el contexto de un procesador de pagos que debe añadir nuevas pasarelas sin cambiar el código existente.

### ¿Cómo se aplica el principio abierto-cerrado en un procesador de pagos?

La clave para implementar el principio abierto-cerrado en un sistema de pagos es diseñar el código de manera que pueda admitir nuevas pasarelas sin modificar la estructura existente. Esto se logra utilizando clases abstractas o interfaces que actúan como intermediarios. Así, los procesadores de pagos específicos, como Stripe, pueden heredar de estas clases abstractas y añadir su propia lógica sin afectar el código original.

### ¿Qué cambios se hicieron para implementar una nueva pasarela?
Para incorporar una nueva pasarela de pagos en este caso:

- Se creó una nueva carpeta con ejemplos de antes y después de aplicar el principio abierto-cerrado.
- Los datos del cliente y el pago se refactorizaron utilizando **Pydantic**, que es la librería más popular en Python para validaciones de datos.
- Se introdujeron modelos de datos tipados para **CustomerData** y **PaymentData**, con campos claros como montos (enteros) y fuentes de pago (cadenas de texto).

### ¿Cómo se manejan los datos con Pydantic?

Pydantic permite definir modelos de datos que facilitan la validación y tipado. Por ejemplo, el modelo `CustomerData` contiene atributos como `name` (nombre del cliente) y `contact_info` (información de contacto), que incluye campos opcionales como teléfono o correo electrónico. Esto hace que la manipulación de datos sea más clara y segura.

### ¿Cómo se crean nuevas pasarelas de pagos usando clases abstractas?

Primero, se define una **clase abstracta** que representará un procesador de pagos. Esta clase no contiene lógica interna, sino que se utiliza para definir la firma del método principal, `processTransaction`. Los procesadores de pagos como Stripe implementan esta clase abstracta, heredando su forma y añadiendo la lógica específica para procesar las transacciones.

- Se define una clase abstracta `PaymentProcessor` que incluye el método `processTransaction`.
- Stripe, por ejemplo, hereda de esta clase abstracta para implementar su propia lógica de procesamiento.
- El servicio de pagos ya no depende de la implementación concreta de Stripe, sino que interactúa con la clase abstracta, lo que facilita añadir nuevas pasarelas sin tocar el código base.

### ¿Cómo se manejan las notificaciones de confirmación?

Se siguió una estrategia similar para las notificaciones. Al igual que con los procesadores de pagos, se creó una clase abstracta `Notifier` que define la firma del método `SendConfirmation`. Esto permite crear diferentes implementaciones, como un **notificador por correo electrónico** o un **notificador por SMS**, sin afectar la estructura del código original.

- Se introdujo un `EmailNotifier` y un `SMSNotifier`, ambos heredando de la clase abstracta `Notifier`.
- El código decide dinámicamente qué tipo de notificación enviar según los datos del cliente, ya sea por correo o por SMS.

### ¿Cómo se extendió el código sin modificar su estructura original?

Al final, el servicio de pagos se extendió para que permitiera enviar notificaciones vía SMS sin cambiar su base de código. Esto se logró creando una nueva clase `SMSNotifier` que también hereda de `Notifier`. El código puede cambiar entre el envío de correos o mensajes de texto con solo ajustar la implementación del servicio, cumpliendo así con el principio abierto-cerrado.

Para aplicar el **Principio Abierto/Cerrado** (Open/Closed Principle, OCP), necesitamos estructurar el código de manera que se pueda **extender sin modificar** el código existente. Esto se puede hacer mediante la creación de clases abstractas o interfaces que se implementan o heredan en clases específicas, permitiendo la adición de nuevas funcionalidades sin cambiar la clase base.

A continuación, te muestro un ejemplo y cómo aplicarle el OCP.

### Ejemplo de Código que Viola el OCP

Supongamos que estamos desarrollando un sistema para calcular el costo de envío de un pedido. Inicialmente, solo tenemos el cálculo de envío estándar, pero más adelante necesitamos agregar otros tipos de envío, como envío exprés y envío internacional. Una implementación que viola el OCP podría verse así:

```python
class ShippingCalculator:
    def calculate_shipping(self, order, shipping_type):
        if shipping_type == "standard":
            return order.weight * 5  # Envío estándar: $5 por kg
        elif shipping_type == "express":
            return order.weight * 10  # Envío exprés: $10 por kg
        elif shipping_type == "international":
            return order.weight * 20  # Envío internacional: $20 por kg
        else:
            raise ValueError("Invalid shipping type")
```

Cada vez que se agrega un nuevo tipo de envío, hay que modificar la clase `ShippingCalculator`, lo cual viola el principio OCP, ya que no está **cerrada a modificaciones**.

### Solución: Aplicando el OCP

Para cumplir con el principio OCP, podemos crear una clase base o interfaz llamada `ShippingStrategy` y luego extenderla para cada tipo de envío, manteniendo el código de `ShippingCalculator` **cerrado a modificaciones** pero **abierto a la extensión**.

```python
from abc import ABC, abstractmethod

# Interfaz o clase abstracta para la estrategia de envío
class ShippingStrategy(ABC):
    @abstractmethod
    def calculate(self, order):
        pass

# Estrategia de envío estándar
class StandardShipping(ShippingStrategy):
    def calculate(self, order):
        return order.weight * 5

# Estrategia de envío exprés
class ExpressShipping(ShippingStrategy):
    def calculate(self, order):
        return order.weight * 10

# Estrategia de envío internacional
class InternationalShipping(ShippingStrategy):
    def calculate(self, order):
        return order.weight * 20

# Clase principal que utiliza la estrategia de envío
class ShippingCalculator:
    def __init__(self, strategy: ShippingStrategy):
        self.strategy = strategy

    def calculate_shipping(self, order):
        return self.strategy.calculate(order)
```

### Explicación

- **Clase `ShippingStrategy`**: Es una clase abstracta que define el método `calculate`. Esta clase sirve como interfaz para las estrategias de envío.
- **Clases `StandardShipping`, `ExpressShipping`, `InternationalShipping`**: Implementan `ShippingStrategy` y proporcionan el cálculo específico de cada tipo de envío.
- **Clase `ShippingCalculator`**: Utiliza una instancia de `ShippingStrategy` para calcular el costo de envío sin importar el tipo. De esta manera, se puede extender `ShippingCalculator` con nuevos tipos de envío sin modificar su código.

### Uso del Código

Supongamos que tenemos un objeto `order` con un atributo `weight`. Así es como podríamos calcular el costo de envío usando diferentes estrategias:

```python
class Order:
    def __init__(self, weight):
        self.weight = weight

# Crear un pedido
order = Order(weight=10)  # Ejemplo de pedido de 10 kg

# Calcular el costo de envío estándar
calculator = ShippingCalculator(StandardShipping())
print(f"Envío estándar: ${calculator.calculate_shipping(order)}")

# Calcular el costo de envío exprés
calculator.strategy = ExpressShipping()
print(f"Envío exprés: ${calculator.calculate_shipping(order)}")

# Calcular el costo de envío internacional
calculator.strategy = InternationalShipping()
print(f"Envío internacional: ${calculator.calculate_shipping(order)}")
```

### Ventajas de Aplicar el OCP

1. **Extensibilidad**: Podemos añadir nuevos tipos de envío (por ejemplo, "Envío nocturno") simplemente creando una nueva clase que herede de `ShippingStrategy`, sin necesidad de modificar `ShippingCalculator`.
2. **Mantenibilidad**: Al no tener que modificar `ShippingCalculator`, se reduce el riesgo de introducir errores en la lógica existente.
3. **Reusabilidad**: Cada estrategia de envío es una clase independiente, por lo que puede reutilizarse y probarse de forma aislada.

### Conclusión

Al aplicar el **Principio Abierto/Cerrado** hemos hecho que el código sea más **modular** y fácil de **extender**. Esto nos permite agregar funcionalidades nuevas sin necesidad de modificar el código existente, lo cual es esencial para mantener un sistema robusto y escalable.

## Principio de Sustitución de Liskov (LSP) en Python

El principio de sustitución de Liskov (LSP) es clave para garantizar la coherencia y la interoperabilidad en sistemas orientados a objetos. Propone que las subclases deben ser intercambiables con sus clases base sin alterar el comportamiento esperado. Esto evita problemas inesperados y asegura que las clases que implementen una interfaz o hereden de otra puedan utilizarse de manera consistente, facilitando la reutilización del código y reduciendo errores en tiempo de ejecución.

### ¿Qué establece el principio de sustitución de Liskov?

Este principio, propuesto por Barbara Liskov, establece que las subclases deben ser sustituibles por sus clases base sin afectar el comportamiento del programa. Es esencial para asegurar que el sistema se mantenga coherente y funcione correctamente cuando se emplean clases derivadas.

### ¿Cómo se aplican las subclases en LSP?

Las subclases deben respetar el contrato de la clase base. Esto significa que:

- No se puede cambiar la firma de los métodos.
- No se deben agregar nuevos atributos que afecten la funcionalidad de los métodos existentes.
- La interfaz y los tipos deben mantenerse compatibles.

### ¿Qué errores evita el principio de sustitución?

El LSP ayuda a evitar errores como:

- Excepciones inesperadas cuando se requieren parámetros adicionales no previstos en la clase base.
- Cambios en el tipo de retorno de los métodos que interrumpen la compatibilidad entre clases.

### ¿Cuáles son los beneficios del principio de sustitución?

- **Reutilización del código**: Las clases que cumplen con LSP pueden ser utilizadas en diferentes contextos sin modificaciones.
- C**ompatibilidad de interfaces**: Facilita que las clases puedan interactuar de forma coherente sin errores inesperados.
- **Reducción de errores en tiempo de ejecución**: El código se mantiene predecible y coherente, disminuyendo la posibilidad de fallos imprevistos.

### ¿Cuándo aplicar el principio de sustitución de Liskov?

Es necesario aplicarlo cuando:

- Hay violación de precondiciones o poscondiciones, es decir, cuando los parámetros o el tipo de retorno de los métodos cambian.
- Se presentan excepciones inesperadas al usar subclases, lo que indica que no se puede hacer una sustitución sencilla entre ellas.

## Aplicar el Principio de Sustitución de Liskov (LSP)

El principio de sustitución de Liskov nos permite escribir código flexible y reutilizable, pero también requiere que todas las clases o protocolos cumplan con una firma coherente. En esta clase, hemos aplicado este principio en Python reemplazando las clases abstractas con protocolos y detectando un bug deliberado que rompe este principio. Vamos a analizar cómo lo resolvimos y cómo aseguramos que las clases de notificación sean intercambiables sin modificar el código base.

### ¿Cómo reemplazamos las clases abstractas por protocolos?

- Se sustituyeron las clases abstractas por protocolos en Python.
- Los protocolos actúan de manera similar a las interfaces en otros lenguajes de programación.
- En este caso, el `Notifier` y el `PaymentProcessor` fueron convertidos en protocolos.
- Los métodos dentro de los protocolos fueron documentados usando docstrings en formato NumPy para mejorar la claridad.

### ¿Cómo se introdujo y detectó el bug?

- Se introdujo un bug a propósito al cambiar la clase `SMSNotifier`.
- El bug hizo que el método `SendConfirmation` no cumpliera con la firma requerida, ya que estaba aceptando un parámetro adicional: `SMSGateway`.
- Esto provocaba que no fuera intercambiable con la clase `EmailNotifier`, lo que viola el principio de sustitución de Liskov.
- Para detectarlo, se utilizó el debugger y un análisis de la firma del método.

### ¿Qué desafíos presenta el principio de sustitución de Liskov?

- Mantener la consistencia en las firmas de los métodos entre clases hijas y protocolos es crucial.
- Es fundamental evitar introducir parámetros adicionales o cambiar las firmas de los métodos, ya que esto rompe la intercambiabilidad de las clases.

Para aplicar el **Principio de Sustitución de Liskov** (LSP), las subclases deben comportarse de manera que puedan sustituir a su clase base sin alterar el funcionamiento del sistema. Esto implica que la subclase debe cumplir con el contrato de la clase base en cuanto a comportamiento, sin añadir restricciones o modificar expectativas fundamentales.

### Ejemplo: Aplicación Correcta del LSP en Python

Imaginemos que estamos desarrollando un sistema para un zoológico que maneja diferentes tipos de aves. Tenemos una clase `Bird` que representa un ave general con un método `fly`. Queremos agregar subclases `Eagle` (que puede volar) y `Penguin` (que no vuela). 

#### Ejemplo Incorrecto: Violación del LSP

En este caso, al intentar hacer que `Penguin` herede de `Bird` pero no pueda volar, estamos violando el LSP, ya que `Penguin` no cumple con el comportamiento esperado (poder volar) de `Bird`.

```python
class Bird:
    def fly(self):
        print("I can fly!")

class Eagle(Bird):
    pass

class Penguin(Bird):
    def fly(self):
        raise Exception("I can't fly!")
```

Con este diseño, `Penguin` no se comporta como un `Bird` común, ya que lanza una excepción en lugar de volar, lo cual rompe el LSP. Si un programa intenta llamar al método `fly` en una instancia de `Penguin` y espera que funcione igual que en `Bird`, se generará un error inesperado.

### Solución: Aplicando el LSP

Para cumplir con el LSP, podemos rediseñar la estructura usando una clase base `Bird` con dos subclases específicas: `FlyingBird` y `NonFlyingBird`. Esto nos permite representar correctamente tanto a las aves que vuelan como a las que no, sin que una interfiera en la funcionalidad de la otra.

```python
from abc import ABC, abstractmethod

# Clase base abstracta
class Bird(ABC):
    @abstractmethod
    def make_sound(self):
        pass

# Subclase para aves que pueden volar
class FlyingBird(Bird):
    def fly(self):
        print("I can fly!")

# Subclase para aves que no vuelan
class NonFlyingBird(Bird):
    def fly(self):
        raise NotImplementedError("This bird can't fly.")

# Aves específicas
class Eagle(FlyingBird):
    def make_sound(self):
        print("Screech!")

class Penguin(NonFlyingBird):
    def make_sound(self):
        print("Honk!")
```

### Explicación

- **Clase `Bird`**: Es una clase base abstracta que define el método `make_sound` que todas las aves deben implementar.
- **Clases `FlyingBird` y `NonFlyingBird`**: Son clases intermedias que especifican si un ave puede volar o no. `FlyingBird` implementa el método `fly`, mientras que `NonFlyingBird` levanta una excepción controlada con `NotImplementedError` para indicar que estas aves no vuelan.
- **Clases `Eagle` y `Penguin`**: Estas representan tipos específicos de aves que implementan el método `make_sound` de la clase base `Bird`. `Eagle` puede volar gracias a que hereda de `FlyingBird`, y `Penguin` no vuela porque hereda de `NonFlyingBird`.

### Uso del Código

Podemos crear instancias de `Eagle` y `Penguin` y usarlas de manera segura sin que una viole el comportamiento esperado de la otra.

```python
# Crear instancias
eagle = Eagle()
penguin = Penguin()

# Ejecutar métodos
eagle.make_sound()    # Screech!
eagle.fly()           # I can fly!

penguin.make_sound()  # Honk!
# penguin.fly()        # Esto lanza NotImplementedError, lo cual es el comportamiento esperado
```

### Ventajas de Aplicar el LSP

1. **Consistencia en el Comportamiento**: Las subclases cumplen con el comportamiento esperado sin generar resultados inesperados o excepciones imprevistas.
2. **Reusabilidad**: Las clases `FlyingBird` y `NonFlyingBird` actúan como plantillas reutilizables para cualquier tipo de ave que vuele o no vuele, respectivamente.
3. **Mantenibilidad**: Se puede extender la jerarquía con nuevas clases de aves sin tener que modificar la lógica existente.

### Conclusión

El Principio de Sustitución de Liskov nos ayuda a crear una jerarquía de clases que se comporta de manera predecible, cumpliendo con las expectativas del usuario de la clase base. En este ejemplo, hemos creado un diseño que permite agregar nuevos tipos de aves sin romper la funcionalidad de las existentes, cumpliendo así con el LSP y mejorando la estabilidad y extensibilidad del sistema.

## Principio de Segregación de Interfaces (ISP) en Python

El principio de segregación de interfaces (ISP) es clave en la construcción de software flexible y modular. Su enfoque evita que las clases dependan de interfaces que no necesitan, promoviendo la cohesión y disminuyendo el acoplamiento. Este principio es esencial cuando trabajamos con dispositivos multifuncionales, como impresoras y escáneres, ya que cada dispositivo solo debe implementar lo que realmente usa.

### ¿Qué establece el principio de segregación de interfaces?

Este principio dice que los clientes no deben depender de interfaces que no utilizan. En el caso de una impresora multifuncional, por ejemplo, esta debería implementar interfaces para imprimir y escanear por separado. Si solo imprime, no necesita la capacidad de escaneo, y viceversa.

### ¿Cuáles son las ventajas de aplicar este principio?

- **Mejora la cohesión y reduce el acoplamiento**: Al separar los comportamientos, las clases son más especializadas y enfocadas en una tarea concreta.
- **Reutilización de componentes**: Las interfaces segregadas permiten reutilizar partes del código sin tener que implementar todos los comportamientos en una misma clase.
- **Aislamiento de cambios**: Si una interfaz, como la de impresión, cambia, las demás (como la de escaneo) no se ven afectadas.
- **Facilidad para realizar pruebas unitarias**: Al tener interfaces pequeñas y específicas, es más sencillo probar cada comportamiento de manera aislada.

### ¿Cuándo debemos aplicar el principio de segregación de interfaces?

- **Interfaces con demasiados métodos irrelevantes**: Si una interfaz contiene muchos métodos que no son necesarios para todas las clases, es el momento de dividirla.
- **Clases que no usan todos los métodos**: Cuando una clase no necesita todos los métodos de una interfaz, esto indica que es necesario implementar el ISP.
- **Cambios que afectan a muchas clases**: Si al modificar una interfaz varias clases se ven afectadas, es un claro signo de que el principio de segregación es necesario.

### ¿Cómo podrías aplicar el principio de segregación de interfaces a tu código?

Este principio es útil en sistemas donde ciertos módulos o clases tienen funcionalidades diversas. Para el procesador de pagos, ¿cómo segmentarías las interfaces para que cada clase implemente solo lo que necesita? Por ejemplo, podrías separar el procesamiento de tarjetas y la gestión de transacciones en interfaces diferentes, garantizando que los cambios en una parte del sistema no afecten otras funcionalidades.

El **Principio de Segregación de Interfaces** (Interface Segregation Principle, ISP) establece que los clientes no deberían estar obligados a depender de interfaces que no utilizan. En otras palabras, es preferible dividir interfaces grandes en interfaces más pequeñas y específicas para que las clases solo tengan que implementar los métodos que realmente necesitan. Este principio ayuda a reducir la complejidad, mejorar la cohesión y hacer el código más mantenible.

### Ejemplo de Código que Viola el ISP

Imaginemos que estamos desarrollando una aplicación para gestionar distintos tipos de trabajadores. Tenemos una interfaz `Worker` con métodos que representan diferentes tipos de tareas que los trabajadores pueden realizar: programar, diseñar y testear. Sin embargo, no todos los trabajadores realizan todas las tareas; por ejemplo, un programador no necesariamente diseña.

```python
class Worker:
    def code(self):
        raise NotImplementedError

    def design(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

class Programmer(Worker):
    def code(self):
        print("Coding...")

    def design(self):
        raise NotImplementedError("Programmers don't design.")

    def test(self):
        raise NotImplementedError("Programmers don't test.")

class Designer(Worker):
    def design(self):
        print("Designing...")

    def code(self):
        raise NotImplementedError("Designers don't code.")

    def test(self):
        raise NotImplementedError("Designers don't test.")
```

### Problema

Este diseño viola el ISP porque obliga a `Programmer` y `Designer` a implementar métodos que no utilizan (`design` y `test` para el programador, `code` y `test` para el diseñador). Esto hace que el código sea confuso y propenso a errores, y obliga a que cada clase tenga métodos que no son relevantes para su rol.

### Solución: Aplicando el ISP

Para cumplir con el ISP, podemos dividir la interfaz `Worker` en interfaces más pequeñas y específicas para cada tipo de tarea, de manera que cada clase solo implemente los métodos que realmente necesita.

```python
from abc import ABC, abstractmethod

# Interfaces separadas para cada tarea específica
class Coder(ABC):
    @abstractmethod
    def code(self):
        pass

class Designer(ABC):
    @abstractmethod
    def design(self):
        pass

class Tester(ABC):
    @abstractmethod
    def test(self):
        pass

# Implementaciones específicas
class Programmer(Coder):
    def code(self):
        print("Coding...")

class GraphicDesigner(Designer):
    def design(self):
        print("Designing...")

class QualityAnalyst(Tester):
    def test(self):
        print("Testing...")
```

### Explicación

- **Interfaces `Coder`, `Designer`, y `Tester`**: Hemos dividido la interfaz `Worker` en tres interfaces más pequeñas que representan tareas específicas. Cada interfaz tiene un solo método abstracto que debe ser implementado por cualquier clase que la herede.
- **Clases `Programmer`, `GraphicDesigner`, y `QualityAnalyst`**: Ahora cada clase implementa solo la interfaz relevante a sus responsabilidades, sin tener que preocuparse por métodos innecesarios.

### Uso del Código

Podemos ahora crear instancias de `Programmer`, `GraphicDesigner`, y `QualityAnalyst` sin preocuparnos por métodos que no tengan sentido en el contexto de cada rol.

```python
# Crear instancias
programmer = Programmer()
designer = GraphicDesigner()
tester = QualityAnalyst()

# Ejecutar métodos específicos
programmer.code()    # Coding...
designer.design()    # Designing...
tester.test()        # Testing...
```

### Ventajas de Aplicar el ISP

1. **Reducción de la Complejidad**: Cada clase implementa solo los métodos que necesita, lo cual reduce la complejidad y hace que el código sea más fácil de entender.
2. **Mejor Mantenibilidad**: El código es más mantenible porque no hay métodos irrelevantes en las clases.
3. **Mayor Cohesión**: Cada interfaz está enfocada en una responsabilidad específica, haciendo que el código sea más cohesivo y fácil de extender.

### Conclusión

El **Principio de Segregación de Interfaces** promueve la creación de interfaces más pequeñas y específicas, lo que permite que las clases solo implementen métodos que son relevantes para sus roles específicos. En este ejemplo, el rediseño asegura que cada tipo de trabajador solo tenga que implementar los métodos que realmente necesita, haciendo el código más limpio y modular.

## Aplicar el Principio de Segregación de Interfaces (ISP)

Implementar el principio de segregación de interfaces es clave para mantener el código limpio y flexible en sistemas complejos como un procesador de pagos. En este caso, se abordaron varias mejoras dentro del procesador de pagos que incluyen la creación de métodos específicos para reembolsos y recurrencias, junto con la correcta segregación de las interfaces según las capacidades de cada procesador de pago.

### ¿Qué cambios se realizaron en el procesador de pagos?

- Se agregaron dos nuevos métodos: reembolso y creación de recurrencias.
- Se implementó un segundo procesador de pagos, el procesador offline, que simula pagos en efectivo. Sin embargo, este procesador no puede realizar reembolsos ni crear recurrencias.
- El método `processTransaction` fue modificado para no depender del atributo de Stripe, ya que ahora hay múltiples procesadores.

### ¿Por qué falló la implementación del principio de segregación de interfaces?

El procesador de pagos offline implementaba métodos que no podía usar, como los reembolsos y las recurrencias, lo que violaba el principio de segregación de interfaces. Este principio establece que una clase no debe depender de métodos que no puede implementar o usar.

### ¿Cómo se corrigió el problema?

- Se crearon dos nuevos protocolos: uno para reembolsos (`RefundPaymentProtocol`) y otro para recurrencias (`RecurringPaymentProtocol`).
- Estos protocolos definen exclusivamente los métodos para esas acciones, eliminando la necesidad de que procesadores como el offline implementen métodos que no necesitan.
- Los procesadores que pueden realizar todas las acciones, como Stripe, ahora implementan los tres protocolos: uno para cobros, otro para reembolsos y otro para recurrencias.

### ¿Qué otros ajustes se hicieron en el servicio?

- Se agregaron atributos opcionales para los procesadores de reembolsos y de recurrencias (`RefundProcessor` y `RecurringProcessor`), permitiendo que cada tipo de procesador maneje solamente las acciones que le corresponden.
- Se implementaron validaciones para evitar excepciones en caso de que un procesador no soporte ciertas acciones. Si el procesador no soporta reembolsos o recurrencias, se lanza una excepción con un mensaje claro.

### ¿Cómo afecta este cambio al procesador de Stripe y al procesador offline?

- En el caso de Stripe, ahora maneja los tres protocolos, permitiendo cobrar, hacer reembolsos y gestionar pagos recurrentes.
- El procesador offline, al no manejar reembolsos ni recurrencias, ya no tiene que implementar esos métodos, cumpliendo con el principio de segregación de interfaces.

Para aplicar el **Principio de Segregación de Interfaces** (ISP) en el diseño de código, es clave asegurarse de que las clases no estén obligadas a depender de métodos que no usan. Esto se logra dividiendo interfaces grandes en varias interfaces más pequeñas y específicas, de manera que cada clase solo implemente los métodos que necesita.

### Ejemplo Práctico: Aplicación del ISP en Python

Imaginemos un sistema para manejar dispositivos electrónicos. Inicialmente, podríamos diseñar una interfaz llamada `Device` con métodos que representen varias funciones que un dispositivo podría tener, como encender, apagar y reproducir música. Sin embargo, no todos los dispositivos tienen las mismas capacidades. Un parlante puede reproducir música, pero una lámpara no.

#### Ejemplo Incorrecto: Violación del ISP

A continuación, un diseño que viola el ISP, en el que todos los dispositivos deben implementar todos los métodos, incluso si algunos no son relevantes.

```python
class Device:
    def turn_on(self):
        raise NotImplementedError

    def turn_off(self):
        raise NotImplementedError

    def play_music(self):
        raise NotImplementedError

class Speaker(Device):
    def turn_on(self):
        print("Speaker is now on.")

    def turn_off(self):
        print("Speaker is now off.")

    def play_music(self):
        print("Playing music...")

class Lamp(Device):
    def turn_on(self):
        print("Lamp is now on.")

    def turn_off(self):
        print("Lamp is now off.")

    def play_music(self):
        raise NotImplementedError("Lamps cannot play music.")
```

En este diseño, la clase `Lamp` debe implementar el método `play_music`, a pesar de que una lámpara no reproduce música. Esto viola el ISP, ya que `Lamp` depende de un método que no le corresponde.

### Solución Correcta: Aplicación del ISP

Para aplicar el ISP, dividimos la interfaz `Device` en interfaces más pequeñas y específicas. Cada dispositivo solo implementará los métodos que realmente necesita.

```python
from abc import ABC, abstractmethod

# Interfaces específicas
class PowerDevice(ABC):
    @abstractmethod
    def turn_on(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass

class MusicPlayer(ABC):
    @abstractmethod
    def play_music(self):
        pass

# Clases que implementan las interfaces relevantes
class Speaker(PowerDevice, MusicPlayer):
    def turn_on(self):
        print("Speaker is now on.")

    def turn_off(self):
        print("Speaker is now off.")

    def play_music(self):
        print("Playing music...")

class Lamp(PowerDevice):
    def turn_on(self):
        print("Lamp is now on.")

    def turn_off(self):
        print("Lamp is now off.")
```

### Explicación

- **Interfaz `PowerDevice`**: Esta interfaz define los métodos `turn_on` y `turn_off`, comunes a dispositivos que se pueden encender y apagar.
- **Interfaz `MusicPlayer`**: Define solo el método `play_music`, exclusivo para dispositivos que pueden reproducir música.
- **Clase `Speaker`**: Implementa tanto `PowerDevice` como `MusicPlayer`, ya que puede encenderse y reproducir música.
- **Clase `Lamp`**: Implementa solo `PowerDevice`, ya que solo necesita los métodos de encendido y apagado.

### Uso del Código

Al aplicar el ISP, podemos usar `Speaker` y `Lamp` sin preocuparnos por métodos innecesarios o errores derivados de métodos irrelevantes.

```python
# Crear instancias de dispositivos
speaker = Speaker()
lamp = Lamp()

# Usar métodos relevantes
speaker.turn_on()     # Speaker is now on.
speaker.play_music()  # Playing music...
speaker.turn_off()    # Speaker is now off.

lamp.turn_on()        # Lamp is now on.
lamp.turn_off()       # Lamp is now off.
```

### Ventajas de Aplicar el ISP

1. **Código Más Limpio y Específico**: Cada clase implementa solo los métodos que necesita, eliminando métodos innecesarios.
2. **Mayor Cohesión y Mantenibilidad**: Las interfaces pequeñas y específicas facilitan el mantenimiento y la extensión del código.
3. **Mejora en la Legibilidad**: El código es más fácil de entender, ya que cada clase y cada interfaz tienen una responsabilidad clara.

### Conclusión

Aplicar el **Principio de Segregación de Interfaces** hace que el código sea más modular, fácil de mantener y específico. En este ejemplo, hemos logrado un diseño en el que cada dispositivo electrónico implementa solo los métodos que realmente utiliza, cumpliendo con el ISP y optimizando el sistema para agregar nuevos dispositivos sin generar dependencias innecesarias.

## Principio de Inversión de Dependencias (DIP) en Python

El principio de inversión de dependencias (Dependency Inversion Principle) es uno de los pilares en la construcción de software robusto, ya que busca disminuir la dependencia entre módulos de alto y bajo nivel, mejorando la flexibilidad y testabilidad del código. Este principio establece que tanto los módulos de alto como de bajo nivel deben depender de abstracciones, no de implementaciones concretas.

### ¿En qué consiste el principio de inversión de dependencias?

La definición formal indica que los módulos de alto nivel, que contienen la lógica de negocio, no deben depender de los módulos de bajo nivel, que gestionan los detalles de implementación. Ambos deben depender de abstracciones. Esto garantiza que los detalles de implementación dependan de las abstracciones y no al revés. Así, se facilita el cambio de implementaciones sin afectar al sistema principal.

### ¿Cómo se aplica este principio en un sistema real?

Un ejemplo claro es un gestor de notificaciones con una interfaz que define el método `enviar mensaje`. Este gestor es un módulo de alto nivel que no depende de cómo se implementa el envío de mensajes. Las clases de bajo nivel, como el notificador por email o por SMS, implementan esa interfaz, y el gestor puede cambiar de una a otra sin modificar su código. Esto muestra cómo el principio reduce el acoplamiento y facilita el mantenimiento.

### ¿Qué beneficios trae el principio de inversión de dependencias?

- **Modularidad:** Al abstraer las implementaciones, las clases de alto nivel se mantienen independientes de los detalles.
- **Flexibilidad**: Cambiar algoritmos o implementaciones es sencillo, ya que el sistema depende de interfaces y no de clases específicas.
- **Testabilidad**: Facilita el uso de mocks en pruebas unitarias, simulando comportamientos sin depender de entornos complejos, como bases de datos.

### ¿Cuándo aplicar el principio de inversión de dependencias?

- Cuando hay alto acoplamiento entre módulos de alto y bajo nivel, dificultando el mantenimiento.
- Si se presentan problemas para cambiar implementaciones sin afectar el resto del sistema.
- Al realizar pruebas unitarias complicadas por la dependencia directa de implementaciones concretas.
Cuando se dificulta la reutilización de componentes o el escalado del sistema.

El **Principio de Inversión de Dependencias** (Dependency Inversion Principle, DIP) establece que:
1. Los módulos de alto nivel no deberían depender de módulos de bajo nivel. Ambos deberían depender de abstracciones (interfaces).
2. Las abstracciones no deberían depender de los detalles. Los detalles deberían depender de abstracciones.

En esencia, el DIP sugiere que debemos evitar dependencias directas entre clases de alto y bajo nivel, favoreciendo la introducción de interfaces o abstracciones. Esto mejora la modularidad, la mantenibilidad y facilita el cambio de componentes del sistema.

### Ejemplo: Aplicación del DIP en Python

Supongamos que tenemos un sistema en el que una clase `OrderProcessor` gestiona pedidos y necesita enviar notificaciones cuando un pedido se procesa. Inicialmente, `OrderProcessor` depende de una implementación específica de notificación (`EmailNotifier`). Esto hace que `OrderProcessor` dependa de un detalle concreto, violando el DIP.

#### Ejemplo Incorrecto: Violación del DIP

```python
class EmailNotifier:
    def send_email(self, message):
        print(f"Sending email with message: {message}")

class OrderProcessor:
    def __init__(self):
        self.notifier = EmailNotifier()

    def process_order(self, order):
        # Procesar el pedido (lógica ficticia)
        print(f"Processing order: {order}")
        self.notifier.send_email("Order processed successfully.")
```

Aquí, `OrderProcessor` depende directamente de `EmailNotifier`. Si quisiéramos cambiar el tipo de notificación (por ejemplo, usar notificaciones por SMS), tendríamos que modificar `OrderProcessor`, lo cual es una violación del DIP.

### Solución Correcta: Aplicando el DIP

Para aplicar el DIP, introducimos una abstracción (interfaz) `Notifier`, de la cual `OrderProcessor` depende, en lugar de depender de `EmailNotifier` directamente. Esto permite usar cualquier clase que implemente la interfaz `Notifier`, mejorando la flexibilidad y el mantenimiento.

```python
from abc import ABC, abstractmethod

# Interfaz Notifier
class Notifier(ABC):
    @abstractmethod
    def send(self, message):
        pass

# Implementación concreta para enviar notificaciones por email
class EmailNotifier(Notifier):
    def send(self, message):
        print(f"Sending email with message: {message}")

# Implementación concreta para enviar notificaciones por SMS
class SMSNotifier(Notifier):
    def send(self, message):
        print(f"Sending SMS with message: {message}")

# OrderProcessor depende de la abstracción Notifier
class OrderProcessor:
    def __init__(self, notifier: Notifier):
        self.notifier = notifier

    def process_order(self, order):
        # Procesar el pedido (lógica ficticia)
        print(f"Processing order: {order}")
        self.notifier.send("Order processed successfully.")
```

### Explicación

- **Interfaz `Notifier`**: Es una abstracción que define el método `send`. `OrderProcessor` depende de esta interfaz, en lugar de una implementación concreta.
- **Clases `EmailNotifier` y `SMSNotifier`**: Ambas implementan la interfaz `Notifier` y proporcionan métodos específicos para enviar notificaciones.
- **Clase `OrderProcessor`**: Ahora depende de `Notifier`, no de `EmailNotifier`. Al pasar una instancia de `Notifier` al constructor de `OrderProcessor`, podemos cambiar el tipo de notificación sin modificar la clase `OrderProcessor`.

### Uso del Código

Podemos crear una instancia de `OrderProcessor` con cualquier implementación de `Notifier`, cumpliendo con el DIP.

```python
# Usando EmailNotifier
email_notifier = EmailNotifier()
processor_with_email = OrderProcessor(email_notifier)
processor_with_email.process_order("Order #1")
# Output:
# Processing order: Order #1
# Sending email with message: Order processed successfully.

# Usando SMSNotifier
sms_notifier = SMSNotifier()
processor_with_sms = OrderProcessor(sms_notifier)
processor_with_sms.process_order("Order #2")
# Output:
# Processing order: Order #2
# Sending SMS with message: Order processed successfully.
```

### Ventajas de Aplicar el DIP

1. **Modularidad**: Podemos cambiar la implementación del servicio de notificación sin modificar la clase `OrderProcessor`.
2. **Facilidad de Testeo**: Podemos pasar una implementación de `Notifier` simulada (mock) a `OrderProcessor` para pruebas unitarias.
3. **Mantenibilidad y Extensibilidad**: Podemos añadir nuevas formas de notificación (por ejemplo, por WhatsApp) sin cambiar el código de `OrderProcessor`.

### Conclusión

El **Principio de Inversión de Dependencias** permite diseñar sistemas más flexibles y modulares al reducir las dependencias directas entre módulos de alto y bajo nivel. En este ejemplo, `OrderProcessor` depende de una abstracción (`Notifier`) en lugar de una implementación concreta, haciendo que sea fácil modificar o extender el comportamiento del sistema sin modificar el código central.

## Aplicar el Principio de Inversión de Dependencias (DIP)

El Principio de Inversión de Dependencias (DIP) establece que las clases de alto nivel no deben depender de clases de bajo nivel, sino de abstractions, como interfaces o protocolos. Los puntos básicos son:

1. **Dependencias hacia abstracciones**: Un módulo de alto nivel no debe depender de un módulo de bajo nivel. Ambos deberían depender de abstracciones.
2. **Inversión de dependencias**: Las dependencias deben ser inyectadas en lugar de ser creadas dentro de las clases, promoviendo un código más flexible y fácil de modificar.
3. **Facilitación de pruebas**: Permite que las pruebas unitarias sean más simples, ya que se pueden inyectar dependencias simuladas.
Este principio mejora la mantenibilidad y escalabilidad del código.

El principio de inversión de dependencias es clave para mejorar la arquitectura del código, y este ejemplo sobre el servicio de pagos lo ilustra perfectamente. En lugar de que la clase de alto nivel dependa de detalles de las clases de bajo nivel, se trabaja con abstracciones, lo que permite mayor flexibilidad al añadir nuevas funcionalidades.

### ¿Cumple el código con el principio de inversión de dependencias?

Sí, el código cumple con el principio desde el momento en que el servicio de pagos (clase de alto nivel) no depende directamente de los detalles de los procesadores de pagos, validadores o notificadores, sino de interfaces o protocolos. Esto significa que, si alguna de las implementaciones cambia, el servicio de alto nivel no requiere modificaciones, ya que interactúa únicamente con las abstracciones.

### ¿Qué detalles pueden mejorar para aplicar mejor el principio?

Aunque la implementación está alineada con el principio de inversión de dependencias, hay un detalle por mejorar: las clases de alto nivel aún instancian clases de bajo nivel dentro de sus atributos, como los validadores o el logger. Para eliminar esta dependencia, se recomienda remover esas “default factories” y en su lugar, inyectar todas las dependencias desde fuera del servicio. Esto garantizaría una mayor adherencia al principio.

### ¿Cómo se instancia el servicio de pagos aplicando el principio?

El código muestra claramente cómo se instancian las dependencias antes de crear el servicio de pagos. Se crea el procesador de pagos de Stripe, un notificador de email, los validadores de clientes y pagos, así como el logger. Luego, se pasa cada dependencia al PaymentService, asegurando que el servicio de alto nivel no conozca los detalles internos de las clases de bajo nivel.

- Se instancia el procesador de Stripe y se reutiliza para recurrencias y reembolsos.
- Si se requiere un servicio con otro procesador, como uno offline, el cambio es sencillo porque las clases están desacopladas.

### ¿Cómo permite la inyección de dependencias mejorar la flexibilidad del código?

La inyección de dependencias facilita la flexibilidad, ya que permite cambiar las implementaciones sin modificar el servicio principal. Si en lugar de Stripe se requiere otro procesador o un notificador distinto como SMS, el código solo necesita ajustar las instancias que se pasan al servicio, sin afectar el código principal.

El **Principio de Inversión de Dependencias** (Dependency Inversion Principle, DIP) establece que:
1. Los módulos de alto nivel no deberían depender de módulos de bajo nivel. Ambos deberían depender de abstracciones (interfaces).
2. Las abstracciones no deberían depender de los detalles. Los detalles deberían depender de abstracciones.

En esencia, el DIP sugiere que debemos evitar dependencias directas entre clases de alto y bajo nivel, favoreciendo la introducción de interfaces o abstracciones. Esto mejora la modularidad, la mantenibilidad y facilita el cambio de componentes del sistema.

### Ejemplo: Aplicación del DIP en Python

Supongamos que tenemos un sistema en el que una clase `OrderProcessor` gestiona pedidos y necesita enviar notificaciones cuando un pedido se procesa. Inicialmente, `OrderProcessor` depende de una implementación específica de notificación (`EmailNotifier`). Esto hace que `OrderProcessor` dependa de un detalle concreto, violando el DIP.

#### Ejemplo Incorrecto: Violación del DIP

```python
class EmailNotifier:
    def send_email(self, message):
        print(f"Sending email with message: {message}")

class OrderProcessor:
    def __init__(self):
        self.notifier = EmailNotifier()

    def process_order(self, order):
        # Procesar el pedido (lógica ficticia)
        print(f"Processing order: {order}")
        self.notifier.send_email("Order processed successfully.")
```

Aquí, `OrderProcessor` depende directamente de `EmailNotifier`. Si quisiéramos cambiar el tipo de notificación (por ejemplo, usar notificaciones por SMS), tendríamos que modificar `OrderProcessor`, lo cual es una violación del DIP.

### Solución Correcta: Aplicando el DIP

Para aplicar el DIP, introducimos una abstracción (interfaz) `Notifier`, de la cual `OrderProcessor` depende, en lugar de depender de `EmailNotifier` directamente. Esto permite usar cualquier clase que implemente la interfaz `Notifier`, mejorando la flexibilidad y el mantenimiento.

```python
from abc import ABC, abstractmethod

# Interfaz Notifier
class Notifier(ABC):
    @abstractmethod
    def send(self, message):
        pass

# Implementación concreta para enviar notificaciones por email
class EmailNotifier(Notifier):
    def send(self, message):
        print(f"Sending email with message: {message}")

# Implementación concreta para enviar notificaciones por SMS
class SMSNotifier(Notifier):
    def send(self, message):
        print(f"Sending SMS with message: {message}")

# OrderProcessor depende de la abstracción Notifier
class OrderProcessor:
    def __init__(self, notifier: Notifier):
        self.notifier = notifier

    def process_order(self, order):
        # Procesar el pedido (lógica ficticia)
        print(f"Processing order: {order}")
        self.notifier.send("Order processed successfully.")
```

### Explicación

- **Interfaz `Notifier`**: Es una abstracción que define el método `send`. `OrderProcessor` depende de esta interfaz, en lugar de una implementación concreta.
- **Clases `EmailNotifier` y `SMSNotifier`**: Ambas implementan la interfaz `Notifier` y proporcionan métodos específicos para enviar notificaciones.
- **Clase `OrderProcessor`**: Ahora depende de `Notifier`, no de `EmailNotifier`. Al pasar una instancia de `Notifier` al constructor de `OrderProcessor`, podemos cambiar el tipo de notificación sin modificar la clase `OrderProcessor`.

### Uso del Código

Podemos crear una instancia de `OrderProcessor` con cualquier implementación de `Notifier`, cumpliendo con el DIP.

```python
# Usando EmailNotifier
email_notifier = EmailNotifier()
processor_with_email = OrderProcessor(email_notifier)
processor_with_email.process_order("Order #1")
# Output:
# Processing order: Order #1
# Sending email with message: Order processed successfully.

# Usando SMSNotifier
sms_notifier = SMSNotifier()
processor_with_sms = OrderProcessor(sms_notifier)
processor_with_sms.process_order("Order #2")
# Output:
# Processing order: Order #2
# Sending SMS with message: Order processed successfully.
```

### Ventajas de Aplicar el DIP

1. **Modularidad**: Podemos cambiar la implementación del servicio de notificación sin modificar la clase `OrderProcessor`.
2. **Facilidad de Testeo**: Podemos pasar una implementación de `Notifier` simulada (mock) a `OrderProcessor` para pruebas unitarias.
3. **Mantenibilidad y Extensibilidad**: Podemos añadir nuevas formas de notificación (por ejemplo, por WhatsApp) sin cambiar el código de `OrderProcessor`.

### Conclusión

El **Principio de Inversión de Dependencias** permite diseñar sistemas más flexibles y modulares al reducir las dependencias directas entre módulos de alto y bajo nivel. En este ejemplo, `OrderProcessor` depende de una abstracción (`Notifier`) en lugar de una implementación concreta, haciendo que sea fácil modificar o extender el comportamiento del sistema sin modificar el código central.

## Reestructuración de un proyecto en Python

Los patrones de diseño son una herramienta poderosa en el desarrollo de software que permite resolver problemas comunes de una manera eficiente y reutilizable. En esta introducción, exploraremos qué son, por qué existen y cuáles son los principales tipos de patrones, como los creacionales, estructurales y de comportamiento, junto con ejemplos prácticos que te permitirán comprender su utilidad en la creación de software de calidad.

### ¿Qué son los patrones de diseño?

Los patrones de diseño son soluciones reutilizables para problemas recurrentes en el desarrollo de software. Se pueden comparar con recetas en el mundo culinario: cada patrón es una guía aplicable a ciertos escenarios o problemas. Al igual que las recetas se ajustan a ciertos ingredientes, los patrones de diseño se aplican en situaciones específicas dentro de la programación.

### ¿Por qué existen los patrones de diseño?

Estos patrones surgen de la experiencia acumulada de los autores del libro *Design Patterns*, conocido como “The Gang of Four”. A través de su trabajo, recopilaron soluciones que ayudan a crear código mantenible y reutilizable, lo que es crucial en el desarrollo colaborativo. Al adoptar estos patrones, los desarrolladores pueden comunicarse de manera más efectiva, utilizando un lenguaje común que simplifica el intercambio de ideas y la colaboración en revisiones de código.

### ¿Cuáles son las categorías de los patrones de diseño?

Existen tres categorías principales:

1. **Patrones creacionales**: Se centran en la creación de instancias de objetos. Son útiles cuando la creación de un objeto es compleja o tiene muchas dependencias. Ejemplos incluyen:

 - Singleton
 - Factory Method
 - Abstract Factory
 - Builder
 - Prototype

2. **Patrones estructurales**: Se enfocan en la composición de clases y objetos para crear estructuras eficientes. Permiten organizar clases de manera óptima. Algunos ejemplos son:

 - Adapter
 - Bridge
 - Composite
 - Decorator
 - Facade
 - Flyweight
 - Proxy

3. **Patrones de comportamiento**: Ayudan a mejorar la comunicación y asignación de responsabilidades entre objetos, resolviendo interacciones complejas entre clases. Ejemplos incluyen:

 - Observer
 - Strategy
 - Command
 - Iterator
 - Mediator
 - State
 - Visitor

### ¿Cómo se aplican los patrones de diseño en la industria?

En proyectos reales, no es necesario usar todos los patrones de diseño. La clave está en seleccionar aquellos que mejor se adapten al contexto específico del proyecto. En este curso, se verán algunos de los patrones más utilizados en la industria, aplicables al tipo de software que desarrollaremos, y serán clave para resolver problemas comunes de manera eficiente.

Reestructurar un proyecto en Python implica organizar su estructura de archivos y carpetas para que sea modular, mantenible y escalable, especialmente en proyectos complejos. Una estructura bien organizada facilita el desarrollo, las pruebas, la colaboración en equipo y la integración de herramientas de CI/CD.

Aquí tienes una guía para estructurar un proyecto en Python, especialmente útil para aplicaciones de tamaño medio a grande.

---

### Estructura Común para un Proyecto en Python

Imaginemos que tienes un proyecto llamado `my_project`. La estructura típica podría verse así:

```
my_project/
│
├── my_project/              # Paquete principal de la aplicación
│   ├── __init__.py          # Marca este directorio como un paquete Python
│   ├── main.py              # Archivo principal para ejecutar la aplicación
│   ├── config.py            # Configuración de la aplicación
│   ├── controllers/         # Lógica de controladores o lógica de negocio
│   │   ├── __init__.py
│   │   └── user_controller.py
│   ├── models/              # Modelos y clases de datos
│   │   ├── __init__.py
│   │   └── user.py
│   ├── services/            # Servicios o lógica de negocio
│   │   ├── __init__.py
│   │   └── user_service.py
│   ├── repositories/        # Capa de acceso a datos
│   │   ├── __init__.py
│   │   └── user_repository.py
│   ├── utils/               # Utilidades y funciones auxiliares
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── schemas/             # Definición de esquemas o validaciones
│       ├── __init__.py
│       └── user_schema.py
│
├── tests/                   # Pruebas unitarias y funcionales
│   ├── __init__.py
│   ├── test_main.py         # Pruebas de integración para main.py
│   ├── controllers/         # Pruebas de controladores
│   ├── models/              # Pruebas de modelos
│   ├── services/            # Pruebas de servicios
│   └── repositories/        # Pruebas de acceso a datos
│
├── scripts/                 # Scripts de gestión (deploy, migraciones)
│   └── run_migration.py
│
├── .env                     # Variables de entorno (NO subir a control de versiones)
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Documentación general del proyecto
├── .gitignore               # Archivos y carpetas ignoradas por git
└── setup.py                 # Script de instalación para pip
```

### Explicación de Cada Directorio y Archivo

- **my_project/**: Este es el paquete principal de tu aplicación, que incluye la mayor parte de la lógica y componentes del proyecto.

  - **`__init__.py`**: Marca el directorio como un paquete. Si usas namespaces, puedes añadir configuraciones básicas.
  
  - **`main.py`**: Punto de entrada de la aplicación. En una aplicación web, aquí podría residir el servidor principal, mientras que en una CLI se definiría el comando principal.
  
  - **`config.py`**: Define configuraciones globales (puedes cargar variables de `.env` o definir configuraciones específicas para cada entorno).
  
  - **controladores (controllers/)**: Aquí reside la lógica de los controladores, que gestionan la lógica de flujo de la aplicación. Para una API, por ejemplo, este directorio puede incluir controladores para diferentes endpoints.

  - **modelos (models/)**: Define la estructura de datos (o modelos ORM, si usas una base de datos). Cada modelo representa una entidad o clase de datos.

  - **servicios (services/)**: Contiene la lógica de negocio. Aquí colocas las operaciones específicas y servicios que actúan sobre los datos y no necesariamente dependen del flujo del usuario.

  - **repositorios (repositories/)**: Capa de acceso a datos que interactúa con la base de datos o con APIs externas. Permite la separación del almacenamiento de datos y la lógica de negocio.

  - **utilidades (utils/)**: Funciones auxiliares que pueden ser usadas en distintas partes del proyecto (validaciones, formateo de datos, etc.).

  - **esquemas (schemas/)**: Definiciones de esquemas o validaciones de datos (por ejemplo, usando librerías como `pydantic` o `marshmallow`). Sirve para definir y validar los datos que entran o salen del sistema.

- **tests/**: Directorio para pruebas unitarias, funcionales e integradas. Organizado de forma similar al proyecto principal para encontrar y estructurar mejor las pruebas de cada módulo.

- **scripts/**: Contiene scripts auxiliares, como scripts de migraciones, herramientas de despliegue, o utilidades de mantenimiento del sistema.

- **`.env`**: Archivo para variables de entorno, como claves API o configuraciones específicas del entorno, que no deben subirse al repositorio.

- **`requirements.txt`**: Lista de dependencias del proyecto, necesaria para instalar las librerías con `pip`.

- **`README.md`**: Documentación general del proyecto. Aquí puedes describir el propósito del proyecto, instrucciones de instalación y guías de uso.

- **`.gitignore`**: Archivo de configuración para ignorar archivos específicos en el control de versiones (por ejemplo, `.env`, archivos de caché, etc.).

- **`setup.py`**: Script de configuración para la instalación del paquete si estás planeando distribuir el proyecto como una biblioteca o una herramienta instalable.


### Consejos Adicionales

1. **Usa herramientas de linting y formateo**: Utiliza `flake8` o `pylint` para asegurar la calidad del código, y `black` o `autopep8` para mantener el formato del código.

2. **Pruebas automatizadas**: Configura tus pruebas con `pytest` o `unittest` para que corran automáticamente en un servidor de CI/CD (por ejemplo, usando GitHub Actions o GitLab CI).

3. **Documentación**: Es recomendable documentar tanto el código como el README.md, e incluso usar docstrings y herramientas como Sphinx si el proyecto es grande.

4. **Configuración flexible**: En `config.py`, carga variables de entorno usando una librería como `dotenv` para facilitar el cambio de entornos (producción, desarrollo, etc.).

### Ejemplo de Código para `config.py`

```python
import os
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

class Config:
    DEBUG = os.getenv("DEBUG", "False") == "True"
    DATABASE_URL = os.getenv("DATABASE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
```

### Ejemplo de `main.py` con Configuración de Entrada

```python
from my_project.config import Config
from my_project.controllers.user_controller import UserController

def main():
    print("Starting application...")
    user_controller = UserController()
    user_controller.run()

if __name__ == "__main__":
    main()
```

### Beneficios de una Buena Estructura de Proyecto

- **Mantenibilidad**: La separación de responsabilidades hace que el código sea más fácil de modificar y mantener.
- **Escalabilidad**: Una estructura bien organizada permite agregar nuevas funcionalidades de manera ordenada.
- **Colaboración**: Facilita el trabajo en equipo, ya que cada miembro puede trabajar en un módulo sin interferir en otros.
- **Pruebas**: Las pruebas se integran de forma natural en la estructura, permitiendo que cada módulo tenga sus pruebas específicas.

Organizar y estructurar bien un proyecto en Python es esencial para el éxito a largo plazo. Mantener estas buenas prácticas hará que el desarrollo sea más eficiente y sostenible.