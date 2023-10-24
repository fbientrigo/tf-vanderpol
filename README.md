# tf-vanderpol
Non linear differential equation modeled via a neural network for educative ends;

## TensorFlow

[TensorFlow](https://www.tensorflow.org/) es una plataforma de código abierto para el aprendizaje automático y la inteligencia artificial. Ofrece una amplia gama de herramientas, bibliotecas y recursos para desarrollar modelos de aprendizaje automático y realizar cálculos numéricos eficientes en tensores.

# Instalar Environment
Basta con correr el environment que se encuentra, posee tensorflow2 y tensorflow-probabilities junto a las librerias basicas de progamación cientifica

```
conda env create -f environment.yml
```

## Instalar usando pip
```
pip install virtualenv
virtualenv tf23
./tf23/Scripts/activate
pip install -r requirements.txt
```

# Sistema de datos
## h5py
It lets you store huge amounts of numerical data, and easily manipulate that data from NumPy. For example, you can slice into multi-terabyte datasets stored on disk, as if they were real NumPy arrays. Thousands of datasets can be stored in a single file, categorized and tagged however you want.