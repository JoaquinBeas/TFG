# run_diffusion_training.py
# -----------------------------------------------------------
# Lanza el entrenamiento completo del modelo de difusión
# condicional usando los parámetros por defecto del proyecto.
# -----------------------------------------------------------
from src.train_diffussion_model import DiffussionTrainer
from src.utils.diffusion_models_enum import DiffusionModelType


def main() -> None:
    trainer = DiffussionTrainer(
        model_type=DiffusionModelType.CONDITIONAL_UNET,  # modelo por defecto
        num_epochs=100, 
        batch_size=64,# 100 épocas fijas
        early_stopping_patience=100                      # no se detiene antes
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
