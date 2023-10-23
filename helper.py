import numpy as np
import tensorflow as tf

def differential_equation_loss(model_u, x, t):
    """
    Calcula la pérdida de una ecuación diferencial parcial (PDE).

    Args:
        model_u (tf.keras.Model): El modelo que predice la solución de la PDE.
        x (tf.Tensor): Coordenadas espaciales.
        t (tf.Tensor): Coordenadas temporales.

    Returns:
        tf.Tensor: La pérdida asociada a la PDE (Ecuación de Burgers).
    """
    with tf.GradientTape() as tape1:
        tape1.watch(x) # <--- seguimiento de gradiente
        with tf.GradientTape(persistent=True) as tape2: #graba las gradientes
            tape2.watch(x)
            tape2.watch(t)
            u = model_u(x, t) # el modelo es una funcion u(x,t)
        # luego calculamos las gradientes
        du_dt = tape2.gradient(u, t)
        du_dx = tape2.gradient(u, x)
    du2_dx2 = tape1.gradient(du_dx, x) # 2 threads de tape permiten 2da derivada
    
    # Aqui va la ecuación que se trabaja
    coefficient = 0.01 / np.pi
    f = du_dt + u * du_dx - coefficient * du2_dx2


    del tape2 # se quita de la memoria
    return f

def optimizer_function_factory(model, loss, input_u, output_u, input_f):
    """
    Fábrica para crear una función requerida por tfp.optimizer.lbfgs_minimize.

    Args:
        model (tf.keras.Model): Una instancia de tf.keras.Model o sus subclases.
        loss (función): Una función con la firma loss_value = loss(pred_y, true_y).
        input_u (tf.Tensor): Los datos de entrada para U (predicción).
        output_u (tf.Tensor): Los datos de salida para U (predicción).
        input_f (tf.Tensor): Datos de entrada para la ecuación diferencial parcial.

    Returns:
        función: Una función con la firma loss_value, gradients = f(model_parameters).
    """
    x_u = input_u[:, 0:1]
    t_u = input_u[:, 1:2]
    x_f = input_f[:, 0:1]
    t_f = input_f[:, 1:2]
    model(x_u, t_u)

    # Obtener las formas de todos los parámetros entrenables en el modelo
    shapes = tf.shape_n(model.trainable_weights)
    n_tensors = len(shapes)

    # Usaremos tf.dynamic_stitch y tf.dynamic_partition más adelante, por lo que necesitamos
    # preparar la información requerida primero
    count = 0
    idx = []  # índices de unión (stitch)
    part = []  # índices de partición (partition)

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """Actualiza los parámetros del modelo con un tensor 1D de TensorFlow.

        Args:
            params_1d (tf.Tensor): Un tensor 1D que representa los parámetros entrenables del modelo.
        """
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_weights[i].assign(tf.reshape(param, shape))

    # Ahora creamos una función que será devuelta por esta fábrica
    @tf.function
    def f(params_1d):
        """Una función que puede ser utilizada por tfp.optimizer.lbfgs_minimize.

        Esta función es creada por la fábrica.

        Args:
           params_1d (tf.Tensor): Un tensor 1D.

        Returns:
            float: Una pérdida escalar y los gradientes con respecto a los parámetros (params_1d).
        """
        # Usamos GradientTape para calcular el gradiente de la pérdida con respecto a los parámetros
        with tf.GradientTape() as tape:
            # Actualizamos los parámetros en el modelo
            assign_new_model_parameters(params_1d)
            # Calculamos la pérdida
            loss_value = loss(output_u, model(x_u, t_u), differential_equation_loss(model, x_f, t_f))

        # Calculamos los gradientes y los convertimos en un tensor 1D de TensorFlow
        grads = tape.gradient(loss_value, model.trainable_weights)
        grads = tf.dynamic_stitch(idx, grads)

        # Imprimimos la iteración y la pérdida
        f.iter.assign_add(1)
        tf.print("Iteración:", f.iter, "pérdida:", loss_value)

        return loss_value, grads

    # Almacenamos esta información como atributos para que podamos usarlos fuera del ámbito
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    return f

