# Curso Básico de Cálculo Diferencial para Data Science e Inteligencia Artificial

## ¿Qué es el cálculo diferencial?

El **cálculo diferencial** es una rama de las matemáticas que se enfoca en el **estudio de cómo cambian las funciones**. Su objetivo principal es **entender y calcular la tasa de cambio** de una cantidad con respecto a otra.

### 🔍 ¿Qué estudia el cálculo diferencial?

1. **Derivadas**:
   La derivada de una función en un punto mide **la pendiente** o la **tasa de cambio instantánea** de la función en ese punto.
   Por ejemplo, si una función representa la posición de un objeto, su derivada representa la **velocidad**.

2. **Límites**:
   El concepto de **límite** es fundamental para definir formalmente qué es una derivada.

### 📐 Aplicaciones del cálculo diferencial:

* En física: para analizar **movimiento, velocidad y aceleración**.
* En economía: para estudiar **costos marginales y tasas de crecimiento**.
* En ingeniería: para **optimizar diseños** o analizar sistemas dinámicos.
* En machine learning: para **ajustar modelos** mediante el cálculo del gradiente.

### 🧠 Ejemplo básico:

Si tienes una función:

$$
f(x) = x^2
$$

Su derivada es:

$$
f'(x) = 2x
$$

Esto significa que en $x = 3$, la tasa de cambio de la función es $2 \times 3 = 6$.

### Resumen

Antes de estudiar el cálculo diferencial es necesario comprender: ¿Qué es el cálculo?, ¿Para qué nos sirve?, ¿Cuál es el propósito? Empecemos por definiciones genéricas.

**Conceptos generales de Cálculo diferencial e integral**

#### Cálculo:

Es realizar operaciones de manera dada para llegar a un resultado.

#### Cálculo diferencial:

Parte del cálculo infinitesimal (que estudia las funciones cuando tienen cambios muy pequeños, cercanos a cero) y del análisis matemático que estudia cómo cambian las funciones continuas cuando sus variables sufren cambios infinitesimales. El principal objeto de estudio en el cálculo diferencial es la **derivada** (o razón de cambio infinitesimal). Un ejemplo de esto es el cálculo de la velocidad instantánea de un objeto en movimiento.

#### Cálculo integral:

Estudio de la anti derivación, es decir, la operación inversa a la derivada. Busca reconstruir funciones a partir de su razón de cambio. El cálculo integral está fuera del alcance de este curso.

## ¿Qué es un límite?

Un **límite** en matemáticas describe el **comportamiento de una función o secuencia** a medida que sus entradas se **acercan a un valor específico**.

### 📌 Definición básica:

El **límite de una función** $f(x)$ cuando $x$ se acerca a un valor $a$, se escribe como:

$$
\lim_{x \to a} f(x)
$$

Esto significa:

> "¿Qué valor se **acerca** la función $f(x)$ cuando $x$ se **aproxima a $a$**?"

### 🧠 Ejemplo sencillo:

Supón que tienes:

$$
f(x) = 2x
$$

Entonces:

$$
\lim_{x \to 3} 2x = 6
$$

Porque cuando te acercas a $x = 3$, $f(x) = 2x$ se acerca a 6.

### ⚠️ Importancia de los límites:

* Permiten **definir la derivada**, que es el núcleo del cálculo diferencial.
* Ayudan a **analizar funciones en puntos problemáticos**, como donde no están definidas exactamente.
* Son fundamentales en el estudio de **continuidad** y **análisis matemático**.

### 🧮 ¿Y si el valor no existe?

Por ejemplo:

$$
\lim_{x \to 0} \frac{1}{x}
$$

Este **no tiene límite finito**, porque a medida que $x$ se acerca a 0, el valor **crece infinitamente** (positivo o negativo, dependiendo del lado).

### Resumen

Un límite es un análisis de tendencia respecto a un punto. Es decir, hacia donde va el valor de nuestra función a medida que “**x**” se acerca al valor “**a**”, pero sin tocar nunca ese valor “**a**”. Por eso se llama límite, porque “tenemos prohibido tocar” ese valor “**a**”.

![Fórmula de un límite](images/formulalimita.jpg)

Una analogía para entender el límite es la siguiente: supón que tienes una fogata encendida frente a tí, y quieres comprobar que el fuego que estás viendo existe, que tus ojos no te engañan. Para ello, acercas las manos al fuego, cada vez más despacio, a ver si las sientes más calientes, pero nunca llegas a estar directamente en el fuego porque te puedes quemar.

## Resolución de límites e interpretación

¡Perfecto! Vamos a ver cómo **resolver límites paso a paso** y cómo **interpretarlos** gráficamente y conceptualmente.

### 🧮 **Ejemplo 1: Límite directo**

### Problema:

$$
\lim_{x \to 2} (3x + 1)
$$

### ✏️ Resolución:

Este es un límite simple. Sustituimos directamente el valor de $x$:

$$
\lim_{x \to 2} (3x + 1) = 3(2) + 1 = 7
$$

### ✅ Interpretación:

A medida que $x$ se **acerca a 2**, el valor de $3x + 1$ se **acerca a 7**. No hay discontinuidades ni problemas, porque la función es lineal y continua.

### 🧮 **Ejemplo 2: Límite con indeterminación**

### Problema:

$$
\lim_{x \to 1} \frac{x^2 - 1}{x - 1}
$$

### 🔍 Paso 1: Sustituimos directamente

$$
\frac{1^2 - 1}{1 - 1} = \frac{0}{0}
$$

Esto es una **indeterminación**, así que tenemos que **simplificar**.

### 🔧 Paso 2: Factorizamos el numerador

$$
\frac{x^2 - 1}{x - 1} = \frac{(x - 1)(x + 1)}{x - 1}
$$

Cancelamos el término común:

$$
= x + 1 \quad \text{(cuando } x \ne 1\text{)}
$$

### ✅ Resultado del límite:

$$
\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = \lim_{x \to 1} (x + 1) = 2
$$

### 📊 Interpretación gráfica

En el ejemplo anterior, la función original tiene una **discontinuidad evitable** en $x = 1$, pero **el límite existe** porque los valores de la función se acercan a 2 desde ambos lados.

### 📘 Ejemplo 3: Límite infinito

$$
\lim_{x \to 0^+} \frac{1}{x}
$$

Cuando $x \to 0$ **desde la derecha** (valores positivos muy pequeños), la función:

$$
\frac{1}{x} \to +\infty
$$

### ✅ Interpretación:

El límite **no existe como número finito**, pero decimos que **tiende a infinito**.

### 🔚 Resumen

| Tipo de límite         | Qué hacer                                    |
| ---------------------- | -------------------------------------------- |
| Sustitución directa    | Solo evalúa normalmente                      |
| Indeterminación 0/0    | Factoriza, racionaliza o usa L'Hôpital       |
| Infinito o no definido | Analiza comportamiento a izquierda y derecha |
| Límite lateral         | Usa $x \to a^-$ o $x \to a^+$                |

### Resumen

Para resolver un [límite](https://platzi.com/clases/2726-calculo-diferencial-ds/46053-que-es-un-limite/ "límite") veamos el concepto de límites laterales. Un límite lateral es cuando nos acercamos a un valor por un lado, ya sea por la izquierda o la derecha. Los límites laterales se denotan de la siguiente manera.

![limite lateral izquierda.PNG](images/limitelateralizquierda.jpg)

Esto es un límite lateral a la izquierda. Un límite lateral a la derecha se denota del mismo modo, excepto que elevamos la constante “a” al signo positivo.

![limite lateral derecha](images/limitelateralderecha.jpg)

Los límites laterales son importantes, ya que para que el [límite](https://platzi.com/clases/2726-calculo-diferencial-ds/46053-que-es-un-limite/ "límite") en general exista, los dos límites laterales deben ser iguales.

![limite general](images/limitegeneral.jpg)

#### Ejercicio de límites e interpretación

Resolvamos el siguiente límite

![ejercicio de limite](images/ejerciciodelimite.jpg)

Si nos damos cuenta, la función no está definida en dos, porque al hacer `x=2` nos queda una división entre cero. **Para resolver límites, debemos recurrir a trucos** como la factorización en este caso. Fíjate que el numerador es una diferencia de cuadrados, por lo que podemos reescribir el límite como.

![factorizacion de limite](images/factorizaciondelimite.jpg)

En este caso, podemos simplemente evaluar el límite y nos queda que es igual a cuatro. Esto quiere decir que a medida que nos vamos acercando a dos, la función se aproxima a cuatro. Pero recuerda, **la función no está definida en dos**.

![Vista gráfica del límite](images/Vistagraficadellimite.jpg)

## Definición de la derivada

La **derivada** de una función mide cómo cambia su valor en respuesta a pequeños cambios en su variable independiente. Esencialmente, responde a la pregunta:

> **¿Qué tan rápido cambia una función en un punto?**

### 📌 **Definición formal (con límites)**

La derivada de una función $f(x)$ en un punto $x = a$ se define como:

$$
f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}
$$

🔍 Esto representa la **pendiente de la recta tangente** a la curva de $f(x)$ en el punto $x = a$.

### 📉 Interpretación gráfica

* Si la derivada es **positiva**, la función **sube** en ese punto.
* Si la derivada es **negativa**, la función **baja** en ese punto.
* Si la derivada es **cero**, la función tiene un **punto crítico** (puede ser un máximo, mínimo o un punto de inflexión).

### 📘 Ejemplo

Sea $f(x) = x^2$. Entonces:

$$
f'(x) = \lim_{h \to 0} \frac{(x + h)^2 - x^2}{h}
= \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h}
= \lim_{h \to 0} \frac{2xh + h^2}{h}
= \lim_{h \to 0} (2x + h) = 2x
$$

✅ Por lo tanto, la derivada de $f(x) = x^2$ es $f'(x) = 2x$.

### Resumen

El problema de la derivada nace de tratar de **encontrar la recta tangente a un punto en una curva**. La primera solución (por parte de Isaac Newton) fué tomar una recta secante, es decir una recta que corta a la curva en dos puntos.

**Encontrando la pendiende de la recta tangente**

![Recta secante a una curva y su pendiente](images/Rectasecanteaunacurvaysupendiente.jpg)

De la imagen anterior nos damos cuenta que estos dos puntos están dados por ![1.PNG](images/1.jpg) y ![2.PNG](images/2.jpg) donde “h” es la distancia horizontal entre dichos puntos. Mediante estos dos puntos, podemos calcular la pendiente de la recta secante con la fórmula ![3.PNG](images/3.jpg), donde m es la pendiente, ![4.PNG](images/4.jpg) y ![5.PNG](images/5.jpg) corresponden a las coordenadas de ![p1.PNG](images/p1.jpg), y ![x1.PNG](images/x1.jpg) y ![y1.PNG](images/y1.jpg) a las coordenadas de p1.PNG![p1.PNG](images/p1.jpg).

Sin embargo, esto nos da la pendiente de la recta secante. Queremos encontrar la de la recta tangente. Para ello, debemos recortar la distancia “**h**”, hasta que sea muy cercana a cero. Entonces tomamos el límite [9.PNG](images/9.jpg), y esto nos da la pendiente de la recta tangente en un punto x de la curva. A este límite es lo que llamamos **derivada**.

La **derivada** también se puede ver en términos de incrementos. El numerador sería el incremento ![7y.PNG](images/7y.jpg) entre las funciones o también el valor de y, mientras que el denominador sería el incremento [8.PNG](images/8x.jpg) entre los valores de x.

Recuerda que el objetivo del curso no es hacer cálculo de la manera tradicional (a lápiz y papel), si no **entender los fundamentos matemáticos que se aplican en distintos algoritmos de inteligencia artificial**.