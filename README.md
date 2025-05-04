# Còpia de models mitjançant models de difusió en MNIST

**Autor:** Joaquin Beas

## Descripció

Aquest projecte implementa un pipeline de *distil·lació de coneixement* basat en dades sintètiques generades per un model de difusió entrenat sobre el conjunt de dades MNIST. El flux de treball és el següent:

1. Entrenar un **model teacher** (classificador MNIST) utilitzant el conjunt de dades MNIST original.
2. Entrenar un **model de difusió** per aprendre a eliminar soroll de les imatges de MNIST.
3. Generar un **dataset sintètic amb etiquetes**:

   * Mostrejar imatges amb soroll utilitzant el model de difusió.
   * Predir les seves etiquetes mitjançant el model teacher.
4. Entrenar un **model student** (classificador MNIST) utilitzant el dataset sintètic generat.
5. Avaluar i comparar el rendiment dels models teacher i student.

L'esquema d'alt nivell es troba a `docs/proyect_schema.pdf`.

## Requisits

* Dependències (instal·lar amb `pip install -r requirements.txt`):

  * matplotlib==3.10.1
  * numpy==2.2.5
  * Pillow==11.2.1
  * scikit\_learn==1.6.1
  * torch==2.6.0+cu126
  * torchsummary==1.5.1
  * torchvision==0.21.0+cu126
  * tqdm==4.67.1

## Ús

### Opcions disponibles

| Opció               | Tipus / valors                                                                              | Per defecte                | Descripció                                          |
| ------------------- | ------------------------------------------------------------------------------------------- | -------------------------- | --------------------------------------------------- |
| `--mnist`           | {`mnist_cnn`, `mnist_complex_cnn`, `decision_tree`, `resnet_preact`}                        | `resnet_preact`            | Model teacher MNIST per entrenar o carregar.        |
| `--diffusion`       | {`diffusion_guided_unet`, `diffusion_resnet`, `diffusion_unet`, `unet`, `conditional_unet`} | `conditional_unet`         | Model difusió per entrenar o carregar.              |
| `--student`         | Mateixos valors que `--mnist`                                                               | Segueix valor de `--mnist` | Model student MNIST (utilitza teacher per defecte). |
| `--train-mnist`     | `flag`                                                                                      | desactivat                 | Entrenar el model teacher MNIST (si no, carrega).   |
| `--train-diffusion` | `flag`                                                                                      | desactivat                 | Entrenar el model de difusió (si no, carrega).      |
| `--train-student`   | `flag`                                                                                      | desactivat                 | Entrenar el model student MNIST (si no, carrega).   |
| `--gen-synth`       | `flag`                                                                                      | desactivat                 | Generar dataset sintètic amb difusió + teacher.     |

## Estructura del projecte

```plain
├── docs/                          # Documentació i diagrames
│   └── proyect_schema.pdf         # Diagrama del flux de treball
├── src/                           # Codi font
│   ├── data/                      # Checkpoints, samples i datasets MNIST
│   ├── diffusion_models/          # Definicions dels models de difusió
│   ├── mnist_models/              # Definicions dels models MNIST
│   └── utils/                     # Utilitats: configs, dataloaders, enums
├── evaluate_mnist_models.py       # Script per comparar models teacher vs student
├── main.py                        # Orquestra pipeline (teacher, difusió, syntètic, student)
├── requirements.txt               # Llista de dependències Python
└── README.md                      # Aquest document
```

## Descripció dels scripts principals

### `synthetic_dataset.py`

* `SyntheticDataset.generate_dataset(...)`: Muestra n imatges amb soroll, etiqueta-les i desa aquelles amb confiança ≥ llindar.
* `SyntheticDataset.generate_balanced_dataset(...)`: Genera un dataset balancejat en format IDX gzip, ús de sampling condicional.

### `train_diffussion_model.py`

* `DiffussionTrainer`: Entrena el model de difusió amb MSE, early stopping, checkpoints i genera exemples per època.

### `train_mnist_model.py`

* `MnistTrainer`: Entrena el classificador MNIST (real o sintètic) amb cross-entropy, early stopping i checkpoints.

### `evaluate_mnist_models.py`

* Avalua i compara l'estadística (precisió, pèrdua) dels models teacher i student sobre testset MNIST.

---

> Projecte desenvolupat com a part del TFG en Aprenentatge Profund i Models de Difusió.
