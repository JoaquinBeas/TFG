# Copia de Modelos de Difusión en MNIST

**Autor:** Joaquin Beas

## Descripción

Este proyecto tiene como objetivo replicar el comportamiento de un modelo de difusión entrenado en el dataset MNIST, y posteriormente comparar las salidas del modelo maestro y del modelo copia, tanto entre sí como con los datos reales (ground truth).

El proceso se divide en tres etapas principales:

1. **Entrenamiento del Modelo Maestro:**  
   Se entrena un modelo de difusión sobre MNIST para generar imágenes a partir de una entrada.

2. **Entrenamiento del Modelo Copia:**  
   Utilizando las salidas generadas por el modelo maestro, se crea un dataset que se emplea para entrenar un segundo modelo (modelo copia o estudiante), con el objetivo de replicar el comportamiento del maestro.

3. **Evaluación y Comparación:**  
   Se comparan las salidas de ambos modelos entre sí y contra los datos reales conocidos (ground truth). Para ello se utilizan métricas cuantitativas (por ejemplo, MSE, FID, SSIM) y análisis visual, permitiendo identificar desviaciones y evaluar la fidelidad de la copia.

## Estructura de src

fuente principal del proyecto.
  - **models/:**
    Implementaciones de los modelos de difusión.
    - `diffusion_mnist.py`: Define la arquitectura y comportamiento del modelo maestro.
    - `diffusion_copy.py`: Define la arquitectura y comportamiento del modelo copia.
  - **data/:**
    Funciones para el manejo y preprocesamiento de datos.  
    - `dataset.py`: Define el dataset, incluyendo la gestión de datos reales.
    - `data_loader.py`: Contiene los cargadores de datos.
  - **utils/:**
    Funciones auxiliares para el proyecto (cálculo de métricas, helpers, etc.).
    - `metrics.py`: Funciones para el cálculo de métricas de evaluación (ej. MSE, FID, SSIM).
    - `helpers.py`: Funciones utilitarias diversas.
  - `config.py`: Funciones para cargar y gestionar configuraciones del proyecto.
