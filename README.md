# Còpia de models mitjançant models de difusió en MNIST

**Autor:** Joaquín Beas

## Descripció

Aquest projecte implementa un pipeline complet de replicació de coneixement de classificadors MNIST emprant models de difusió per generar un dataset sintètic.

El flux de treball principal:

1. Entrenar o carregar un **model teacher** (classificador MNIST) amb dades reals de MNIST.
2. Entrenar o carregar un **model de difusió** per generar imatges MNIST.
3. Generar i filtrar un **dataset sintètic** etiquetat segons la confiança del model teacher.
4. Entrenar o carregar un **model student** amb el dataset sintètic.
5. Avaluar i comparar el rendiment dels models teacher vs student en el conjunt de test real de MNIST.

## Requisits

Instal·la les dependències amb:

```bash
pip install -r requirements.txt
```

El fitxer `requirements.txt` hauria d'incloure com a mínim:

* torch
* torchvision
* numpy
* pillow
* scikit-learn
* tqdm

## Estructura del projecte

```plain
.
├── docs/                         # Documentació i memòria
├── src/                          # Codi fuente del projecte
│   ├── diffusion_models/         # Arquitectures U-Net per DDPM
│   │   ├── diffusion_unet.py
│   │   └── diffusion_unet_conditional.py
│   ├── mnist_models/             # Models per classificació MNIST
│   │   ├── mnist_simple_cnn.py
│   │   ├── mnist_compressed_cnn.py
│   │   ├── mnist_decision_tree.py
│   │   └── resnet_preact.py
│   ├── utils/                    # Funcions auxiliars i configuració
│   │   ├── backbone_utils.py
│   │   ├── blocks.py
│   │   ├── config.py
│   │   ├── cosine_variance_schedule.py
│   │   ├── data_loader.py
│   │   ├── diffusion_models_enum.py
│   │   ├── fp16_util.py
│   │   ├── gaussian_diffusion.py
│   │   ├── initialization.py
│   │   ├── mnist_models_enum.py
│   │   ├── nn.py
│   │   ├── resnet.py
│   │   ├── training_plot.py
│   │   ├── unet_conditional.py
│   │   ├── unet_guided.py
│   │   └── unet_teacher.py
│   ├── synthetic_dataset.py     # Generació i filtrat de dataset sintètic
│   ├── train_diffussion_model.py# Entrenament de models de difusió
│   ├── train_mnist_model.py     # Entrenament de classificadors MNIST
│   └── main.py                  # Script principal de pipeline
├── README.md                     # Aquest fitxer
├── requirements.txt              # Dependències Python
├── .gitignore                    # Fitxers i carpetes ignorades per Git
└── docs/memoria.pdf              # Memòria del TFG
```

## Ús

El script `main.py` orquestra tot el pipeline. Exemples d'ús:

```bash
# 1. Entrenar o carregar el model teacher amb MNIST real
python main.py --train-mnist

# 2. Entrenar o carregar el model de difusió
python main.py --train-diffusion

# 3. Generar un dataset sintètic de 60000 imatges amb confiança ≥ 0.9
python main.py --gen-synth --num-samples 60000 --confidence-threshold 0.9

# 4. Entrenar o carregar el model student amb el dataset sintètic
python main.py --train-student

# 5. Pipeline complet en un sol pas
python main.py --train-mnist --train-diffusion --gen-synth --train-student
```

## Flags principals

* `--mnist-model {mnist_simple_cnn, mnist_complex_cnn, resnet_preact, decision_tree}`
  Model MNIST que s'utilitza tant per teacher com per student. **Default:** `resnet_preact`.

* `--diffusion-model {diffusion_unet, conditional_unet}`
  Model de difusió per generar imatges. **Default:** `conditional_unet`.

* `--num-samples INT`
  Nombre d'imatges sintètiques a generar quan s'utilitza `--gen-synth`. **Default:** `60000`.

* `--confidence-threshold FLOAT`
  Llindar de confiança per filtrar les imatges sintètiques. **Default:** `0.8`.

* Flags de pipeline:

  * `--train-mnist`     : Entrena o carrega el model teacher MNIST.
  * `--train-diffusion` : Entrena o carrega el model de difusió.
  * `--gen-synth`       : Genera el dataset sintètic utilitzant difusió + teacher.
  * `--train-student`   : Entrena o carrega el model student amb dades sintètiques.

## Descripció dels scripts

### `synthetic_dataset.py`

* Conté la classe `SyntheticDataset` que permet:

  1. **Muestreig** de dades sintètiques mitjançant dos models de difusió: `diffusion_unet` i `conditional_unet`, seleccionables via flag.
  2. **Filtrat** de mostres segons la confiança del teacher (probabilitat màxima) amb llindar ajustable.
  3. **Generació de datasets balancejats**, especificant el nombre de mostres per classe (opció `--balanced --samples-per-class X`).
  4. **Exportació** dels conjunts en format PNG + JSON, o com fitxers IDX gzip compatibles amb MNIST.

### `train_diffussion_model.py`

* Defineix la classe `DiffussionTrainer` amb suport per:

  * Entrenar un **U-Net bàsic** (`diffusion_unet.py`) o un **U-Net condicional** (`diffusion_unet_conditional.py`).
  * Configuració de l’schedule de betas (linear o cosine) i nombre de passes `T`.
  * Checkpoints periòdics i recuperació automàtica (resume) de l’entrenament.
  * Ajust de l’scheduler d’aprenentatge i early stopping per evitar overfitting.

### `train_mnist_model.py`

* Encapsula la classe `MnistTrainer` per entrenar classificadors sobre:

  * **Dades reals** de MNIST o **datasets sintètics** prèviament generats.
  * Models configurables: `mnist_simple_cnn`, `mnist_complex_cnn`, `resnet_preact` o `DecisionTreeClassifier` de scikit-learn.
  * Entrenament amb `CrossEntropyLoss`, optimitzador configurable (Adam o SGD), early stopping i lògica de logger.

### `mnist_simple_cnn.py` i `mnist_complex_cnn.py`

* Definició de dues arquitectures de CNN:

  * **Simple CNN** (\~422 k paràmetres), adequat per un baseline ràpid.
  * **Complex CNN** (\~5 k paràmetres amb GlobalAveragePooling), optimitzat per baix cost computacional.

### `resnet_preact.py`

* Implementació de **ResNet Pre-activation** (\~272 k paràmetres), model teacher estàndard amb alta precisió.

### `diffusion_unet.py` i `diffusion_unet_conditional.py`

* Arquitectures de **DDPM**:

  * U-Net genèric per a difusió.
  * U-Net condicional que incorpora embeddings de classe per generar imatges dirigides.

---
>Projecte desenvolupat com a part del TFG en Aprenentatge Profund i Models de Difusió.
