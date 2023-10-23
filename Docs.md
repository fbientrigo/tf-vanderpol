# Conceptos:

## GradientTape
La GradientTape es una parte fundamental de la diferenciación automática en TensorFlow. Aquí tienes cómo funciona:

Cuando abres una GradientTape, TensorFlow comienza a "grabar" todas las operaciones que ocurren dentro de ese bloque.

Puedes usar el método .watch(variable) para registrar una variable específica si deseas calcular gradientes con respecto a ella.

Luego, puedes realizar operaciones matemáticas con esas variables registradas.

Cuando cierras la GradientTape, TensorFlow puede calcular automáticamente los gradientes de las operaciones registradas con respecto a las variables registradas. Estos gradientes son útiles para optimizar funciones y ajustar modelos de aprendizaje automático mediante algoritmos de optimización como el descenso de gradiente.

ejemplo corto que ilustra cómo usar `tf.GradientTape` para calcular gradientes en TensorFlow:

```python
import tensorflow as tf

# Definimos una función simple, f(x) = x^2
def f(x):
    return x**2

# Punto en el que queremos calcular el gradiente
x = tf.constant(2.0)

# Abre una GradientTape para rastrear las operaciones y calcular gradientes
with tf.GradientTape() as tape:
    # Indica que deseamos rastrear la variable 'x'
    tape.watch(x)
    
    # Calcula la función f(x)
    y = f(x)

# Calcula el gradiente de y con respecto a x
gradiente = tape.gradient(y, x)

# Imprime el resultado
print("f(x) =", y.numpy())  # f(x) = 4.0
print("Gradiente de f(x) con respecto a x =", gradiente.numpy())  # Gradiente de f(x) con respecto a x = 4.0
```

En este ejemplo, primero definimos una función simple `f(x) = x^2`. Luego, utilizamos `tf.constant(2.0)` para crear un tensor `x` con el valor 2.0.

Dentro del bloque `with tf.GradientTape() as tape:`, indicamos que deseamos rastrear la variable `x` utilizando `tape.watch(x)`. Luego, calculamos la función `f(x)` y almacenamos el resultado en `y`.

Finalmente, utilizamos `tape.gradient(y, x)` para calcular el gradiente de `y` con respecto a `x`. El resultado se almacena en la variable `gradiente` y lo imprimimos. En este caso, el gradiente es 4.0, ya que la derivada de `x^2` con respecto a `x` es 2x, y cuando `x` es 2.0, el gradiente es igual a 4.0.


___

# Documentación de TensorFlow: tf.dynamic_stitch y tf.dynamic_partition


## `tf.dynamic_stitch`

### Descripción

`tf.dynamic_stitch` es una función en TensorFlow que permite combinar valores de tensores dispersos en un nuevo tensor resultante. Esta función es útil cuando se necesita combinar valores de diferentes tensores en una sola estructura de datos.

### Sintaxis

```python
tf.dynamic_stitch(indices, data, name=None)
```

- `indices`: Una lista de índices que especifica cómo combinar los datos de entrada.
- `data`: Una lista de tensores que contienen los datos a combinar.

### Ejemplo

```python
import tensorflow as tf

# Definir índices y datos
indices = [tf.constant([0, 2]), tf.constant([1, 3])]
data = [tf.constant([10, 20]), tf.constant([30, 40])]

# Usar tf.dynamic_stitch para combinar los datos
result = tf.dynamic_stitch(indices, data)

print(result)
```

## `tf.dynamic_partition`

### Descripción

`tf.dynamic_partition` es una función en TensorFlow que permite dividir un tensor en particiones basadas en una condición específica. Cada partición resultante contiene elementos que cumplen con la condición dada.

### Sintaxis

```python
tf.dynamic_partition(data, partitions, num_partitions, name=None)
```

- `data`: El tensor que se va a dividir en particiones.
- `partitions`: Un tensor que especifica la partición a la que se asigna cada elemento en `data`.
- `num_partitions`: El número total de particiones.

### Ejemplo

```python
import tensorflow as tf

# Definir un tensor de datos
data = tf.constant([10, 20, 30, 40, 50])

# Definir las particiones
partitions = tf.constant([0, 1, 1, 0, 2])

# Usar tf.dynamic_partition para dividir el tensor en particiones
result = tf.dynamic_partition(data, partitions, 3)

print(result)
```

Estos son ejemplos básicos de cómo usar `tf.dynamic_stitch` y `tf.dynamic_partition` en TensorFlow. Estas funciones son útiles en una variedad de situaciones donde es necesario combinar o dividir datos de manera dinámica, esto es aplicado en la función: ``optimizer_function_factory``
