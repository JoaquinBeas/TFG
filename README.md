# Copia de Modelos de Difusión en MNIST

**Autor:** Joaquin Beas

## Descripción

Este proyecto tiene como objetivo replicar el comportamiento de un modelo de difusión entrenado en el dataset MNIST, y posteriormente comparar las salidas del modelo maestro y del modelo copia habiendo estado este entrenado con el output del modelo maestro. Despues comparar el comportamiento de estos modelos entre sí.

El esquema basico de funcionamiento podemos encontrarlo en docs\proyect_schema.png, el cual explica que partiendo de un dataset mnist dividimos en trainset y testset, con el trainset entrenamos un modelo de difusion, con este modelo predecimos el ruido sobre una entrada y luego se lo restamos a esta misma entrada para crear un dataset Mnist sin labels. 
A este nuevo dataset le asignamos labels segun prediga el modelo para formar un nuevo dataset creado por el modelo de difusion teacher con labels, este dataset lo volveremos a dividir en trainset y testset para entrenar un modelo student.