import os
import torch

from src.synthetic_dataset import SyntheticDataset
from src.train_diffussion_model import DiffussionTrainer, DiffusionModelType
from src.train_mnist_model import MnistTrainer, MNISTModelType
from src.utils.config import (
    DEVICE,
    TRAIN_DIFFUSION_MODEL_DIR,
    TRAIN_MNIST_MODEL_COPY_DIR,
    TRAIN_MNIST_MODEL_DIR,
    MODEL_IMAGE_SIZE,
    MODEL_IN_CHANNELS,
    TIMESTEPS,
    MNIST_N_CLASSES
)


def main():
    # Flags de ejecución
    train_mnist = False
    train_diffusion = False
    train_diffusion_copy = True
    gemerate_synthetic_dataset = False

    # Nombres de modelo (strings)
    mnist_model_name = "resnet_preact"       # mnist_cnn, mnist_complex_cnn, decision_tree, resnet_preact
    diffusion_model_name = "conditional_unet"  # diffusion_guided_unet, diffusion_resnet, diffusion_unet, conditional_unet
    mnist_model_name_copy = "resnet_preact"

    # ----- MNIST TEACHER -----
    if mnist_model_name.lower() == "mnist_cnn":
        selected_mnist_model = MNISTModelType.SIMPLE_CNN
    elif mnist_model_name.lower() == "mnist_complex_cnn":
        selected_mnist_model = MNISTModelType.COMPLEX_CNN
    elif mnist_model_name.lower() == "decision_tree":
        selected_mnist_model = MNISTModelType.DECISION_TREE
    elif mnist_model_name.lower() == "resnet_preact":
        selected_mnist_model = MNISTModelType.RESNET_PREACT
    else:
        raise ValueError(f"Modelo MNIST desconocido: {mnist_model_name}")

    # ----- DIFUSIÓN -----
    name = diffusion_model_name.lower()
    if name == "diffusion_guided_unet":
        selected_diffusion_model = DiffusionModelType.GUIDED_UNET
    elif name == "diffusion_resnet":
        selected_diffusion_model = DiffusionModelType.RESNET
    elif name in ("diffusion_unet", "unet"): 
        selected_diffusion_model = DiffusionModelType.UNET
    elif name == "conditional_unet":
        selected_diffusion_model = DiffusionModelType.CONDITIONAL_UNET
    else:
        raise ValueError(f"Modelo de difusión desconocido: {diffusion_model_name}")

    # ----- MNIST STUDENT CONFIG -----
    if mnist_model_name_copy.lower() == "mnist_cnn":
        selected_mnist_model_copy = MNISTModelType.SIMPLE_CNN
    elif mnist_model_name_copy.lower() == "mnist_complex_cnn":
        selected_mnist_model_copy = MNISTModelType.COMPLEX_CNN
    elif mnist_model_name_copy.lower() == "decision_tree":
        selected_mnist_model_copy = MNISTModelType.DECISION_TREE
    elif mnist_model_name_copy.lower() == "resnet_preact":
        selected_mnist_model_copy = MNISTModelType.RESNET_PREACT
    else:
        raise ValueError(f"Modelo MNIST desconocido para copia: {mnist_model_name_copy}")

    # ----- MODELO MNIST TEACHER -----
    if train_mnist:
        print("Entrenando modelo MNIST teacher...")
        trainer_mnist = MnistTrainer(
            model_type=selected_mnist_model,
            num_epochs=20,
            learning_rate=0.002,
            batch_size=64
        )
        avg_loss, accuracy = trainer_mnist.train_model()
        print(f"MNIST Teacher: Pérdida={avg_loss:.4f}, Precisión={accuracy:.2f}%")
        model_mnist = trainer_mnist.get_model()
    else:
        print("Cargando MNIST teacher desde checkpoint...")
        if selected_mnist_model == MNISTModelType.SIMPLE_CNN:
            from src.mnist_models.mnist_simple_cnn import MNISTCNN
            model_mnist = MNISTCNN()
        elif selected_mnist_model == MNISTModelType.COMPLEX_CNN:
            from src.mnist_models.mnist_complex_cnn import MNISTNet1
            model_mnist = MNISTNet1()
        elif selected_mnist_model == MNISTModelType.RESNET_PREACT:
            from src.mnist_models.resnet_preact import ResNetPreAct
            # Inline config mínimo
            class Cfg: pass
            cfg = Cfg(); cfg.model = Cfg()
            cfg.model.in_channels = 1
            cfg.model.n_classes = 10
            cfg.model.base_channels = 16
            cfg.model.block_type = 'basic'
            cfg.model.depth = 20
            cfg.model.remove_first_relu = False
            cfg.model.add_last_bn = False
            cfg.model.preact_stage = [True, True, True]
            model_mnist = ResNetPreAct(cfg)
        else:
            from src.mnist_models.mnist_decision_tree import MNISTDecisionTree
            model_mnist = MNISTDecisionTree(max_depth=40)
        checkpoint = os.path.join(TRAIN_MNIST_MODEL_DIR, selected_mnist_model.value, "last_model.pt")
        model_mnist.load_state_dict(torch.load(checkpoint, map_location=torch.device(DEVICE)))
        model_mnist.eval()
        print(f"MNIST Teacher cargado desde: {checkpoint}")
    # ----- MODELO DIFUSIÓN -----
    if train_diffusion:
        print("Entrenando modelo de difusión...")
        trainer_diffusion = DiffussionTrainer(
            model_type=selected_diffusion_model,
            num_epochs=120,
            learning_rate=0.001,
            batch_size=64,
            early_stopping_patience=100
        )
        avg_loss_test = trainer_diffusion.train_model()
        print(f"Difusión Test Loss: {avg_loss_test:.6f}")
        model_diffusion = trainer_diffusion.get_model()
    else:
        print("Cargando difusión desde checkpoint...")
        if selected_diffusion_model == DiffusionModelType.GUIDED_UNET:
            from src.diffusion_models.diffusion_guided_unet import DiffusionGuidedUnet
            model_diffusion = DiffusionGuidedUnet()
        elif selected_diffusion_model == DiffusionModelType.RESNET:
            from src.diffusion_models.diffusion_resnet import DiffusionResnet
            model_diffusion = DiffusionResnet()
        elif selected_diffusion_model == DiffusionModelType.UNET:
            from src.diffusion_models.diffusion_unet import DiffusionUnet
            model_diffusion = DiffusionUnet(
                MODEL_IMAGE_SIZE,
                MODEL_IN_CHANNELS,
                TIMESTEPS
            )
        elif selected_diffusion_model == DiffusionModelType.CONDITIONAL_UNET:
            from src.diffusion_models.diffusion_unet_conditional import ConditionalDiffusionModel
            model_diffusion = ConditionalDiffusionModel()
        else:
            raise ValueError(f"Modelo de difusión desconocido: {diffusion_model_name}")
        ckpt = os.path.join(TRAIN_DIFFUSION_MODEL_DIR, selected_diffusion_model.value, "last_model.pt")
        model_diffusion.load_state_dict(torch.load(ckpt, map_location=torch.device(DEVICE)))
        model_diffusion.eval()
        print(f"Difusión cargada desde: {ckpt}")
    if gemerate_synthetic_dataset:
        # ----- GENERAR DATASET SINTÉTICO -----
        print("Generando dataset sintético...")
        synthetic_dataset = SyntheticDataset(model_diffusion, model_mnist)
        synthetic_dataset.generate_balanced_dataset(max_per_class=100)

    # ----- MODELO MNIST STUDENT -----
    if train_diffusion_copy:
        print("Entrenando modelo MNIST student...")
        trainer_student = MnistTrainer(
            model_type=selected_mnist_model_copy,
            num_epochs=20,
            learning_rate=0.002,
            batch_size=64,
            model_path=TRAIN_MNIST_MODEL_COPY_DIR,
            use_synthetic_dataset=True
        )
        avg_loss_s, acc_s = trainer_student.train_model()
        print(f"MNIST Student: Pérdida={avg_loss_s:.4f}, Precisión={acc_s:.2f}%")
    else:
        print("Cargando MNIST student desde checkpoint...")
        if selected_mnist_model_copy == MNISTModelType.SIMPLE_CNN:
            from src.mnist_models.mnist_simple_cnn import MNISTCNN
            model_student = MNISTCNN()
        elif selected_mnist_model_copy == MNISTModelType.COMPLEX_CNN:
            from src.mnist_models.mnist_complex_cnn import MNISTNet1
            model_student = MNISTNet1()
        elif selected_mnist_model_copy == MNISTModelType.RESNET_PREACT:
            from src.mnist_models.resnet_preact import ResNetPreAct
            cfg = Cfg(); cfg.model = Cfg()
            cfg.model.in_channels = 1; cfg.model.n_classes = 10
            cfg.model.base_channels = 16; cfg.model.block_type = 'basic'
            cfg.model.depth = 20; cfg.model.remove_first_relu = False
            cfg.model.add_last_bn = False; cfg.model.preact_stage = [True,True,True]
            model_student = ResNetPreAct(cfg)
        else:
            from src.mnist_models.mnist_decision_tree import MNISTDecisionTree
            model_student = MNISTDecisionTree(max_depth=40)
        ckpt_s = os.path.join(TRAIN_MNIST_MODEL_COPY_DIR, selected_mnist_model_copy.value, "last_model.pt")
        model_student.load_state_dict(torch.load(ckpt_s, map_location=torch.device(DEVICE)))
        model_student.eval()
        print(f"MNIST Student cargado desde: {ckpt_s}")
if __name__ == "__main__":
    main()
