import os
import argparse
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Mapping strings to enum types
MNIST_MAP = {
    "mnist_cnn": MNISTModelType.SIMPLE_CNN,
    "mnist_compressed_cnn": MNISTModelType.COMPRESSED_CNN,
    "decision_tree": MNISTModelType.DECISION_TREE,
    "resnet_preact": MNISTModelType.RESNET_PREACT,
}

DIFF_MAP = {
    # "diffusion_guided_unet": DiffusionModelType.GUIDED_UNET,
    # "diffusion_resnet": DiffusionModelType.RESNET,
    "diffusion_unet": DiffusionModelType.UNET,
    "unet": DiffusionModelType.UNET,
    "conditional_unet": DiffusionModelType.CONDITIONAL_UNET,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or load MNIST teacher, diffusion model, and MNIST student."
    )
    parser.add_argument("--mnist", choices=MNIST_MAP.keys(), default="resnet_preact",
                        help="Type of MNIST teacher model.")
    parser.add_argument("--diffusion", choices=DIFF_MAP.keys(), default="conditional_unet",
                        help="Type of diffusion model.")
    parser.add_argument("--student", choices=MNIST_MAP.keys(),
                        help="Type of MNIST student model (defaults to teacher type).")
    parser.add_argument("--train-mnist", action="store_true",
                        help="Train MNIST teacher (otherwise load checkpoint).")
    parser.add_argument("--train-diffusion", action="store_true",
                        help="Train diffusion model (otherwise load checkpoint).")
    parser.add_argument("--train-student", action="store_true",
                        help="Train MNIST student (otherwise load checkpoint).")
    parser.add_argument("--gen-synth", action="store_true",
                        help="Generate synthetic dataset using diffusion & teacher.")
    return parser.parse_args()


def select(enum_map, key, desc):
    try:
        return enum_map[key.lower()]
    except KeyError:
        raise ValueError(f"Unknown {desc}: {key}")


def load_mnist_model(model_type: MNISTModelType) -> torch.nn.Module:
    # Instantiate architecture
    if model_type == MNISTModelType.SIMPLE_CNN:
        from src.mnist_models.mnist_simple_cnn import MNISTCNN as M
        model = M()
    elif model_type == MNISTModelType.COMPRESSED_CNN:
        from src.mnist_models.mnist_compressed_cnn import MNISTNet1 as M
        model = M()
    elif model_type == MNISTModelType.RESNET_PREACT:
        from src.mnist_models.resnet_preact import ResNetPreAct as R
        # Minimal inline config
        class C: pass
        cfg = C(); cfg.model = C()
        cfg.model.in_channels = 1
        cfg.model.n_classes = MNIST_N_CLASSES
        cfg.model.base_channels = 16
        cfg.model.block_type = 'basic'
        cfg.model.depth = 20
        cfg.model.remove_first_relu = False
        cfg.model.add_last_bn = False
        cfg.model.preact_stage = [True, True, True]
        model = R(cfg)
    else:
        from src.mnist_models.mnist_decision_tree import MNISTDecisionTree as D
        model = D(max_depth=40)
    # Load checkpoint
    ckpt = os.path.join(TRAIN_MNIST_MODEL_DIR, model_type.value, "last_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    logger.info(f"MNIST model loaded from {ckpt}")
    return model


def load_diffusion_model(model_type: DiffusionModelType) -> torch.nn.Module:
    if model_type == DiffusionModelType.UNET:
        from src.diffusion_models.diffusion_unet import DiffusionUnet as U
        model = U(MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS)
    else:
        from src.diffusion_models.diffusion_unet_conditional import ConditionalDiffusionModel as C
        model = C()
    ckpt = os.path.join(TRAIN_DIFFUSION_MODEL_DIR, model_type.value, "last_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    logger.info(f"Diffusion model loaded from {ckpt}")
    return model


def main():
    args = parse_args()

    # Determine types
    mnist_type = select(MNIST_MAP, args.mnist, "MNIST teacher model")
    diff_type = select(DIFF_MAP, args.diffusion, "diffusion model")
    student_type = select(MNIST_MAP, args.student or args.mnist, "MNIST student model")

    # MNIST teacher
    if args.train_mnist:
        logger.info("Training MNIST teacher...")
        trainer = MnistTrainer(
            model_type=mnist_type,  # Por defecto: ResNetPreAct
            num_epochs=30, 
            learning_rate=0.002, 
            batch_size=64
        )
        loss, acc = trainer.train_model()
        logger.info(f"Teacher loss={loss:.4f}, acc={acc:.2f}%")
        mnist_model = trainer.get_model()
    else:
        logger.info("Loading MNIST teacher...")
        mnist_model = load_mnist_model(mnist_type)

    # Diffusion
    if args.train_diffusion:
        logger.info("Training diffusion model...")
        trainer = DiffussionTrainer(model_type=diff_type, num_epochs=120,
                                    learning_rate=0.001, batch_size=64,
                                    early_stopping_patience=100)
        loss = trainer.train_model()
        logger.info(f"Diffusion loss={loss:.6f}")
        diffusion_model = trainer.get_model()
    else:
        logger.info("Loading diffusion model...")
        diffusion_model = load_diffusion_model(diff_type)

    # Synthetic dataset
    if args.gen_synth:
        logger.info("Generating synthetic dataset...")
        SyntheticDataset(diffusion_model, mnist_model).generate_balanced_dataset(max_per_class=100)

    # MNIST student
    if args.train_student:
        logger.info("Training MNIST student...")
        trainer = MnistTrainer(model_type=student_type, num_epochs=20,
                                learning_rate=0.002, batch_size=64,
                                model_path=TRAIN_MNIST_MODEL_COPY_DIR,
                                use_synthetic_dataset=True)
        loss, acc = trainer.train_model()
        logger.info(f"Student loss={loss:.4f}, acc={acc:.2f}%")
        student_model = trainer.get_model()
    else:
        logger.info("Loading MNIST student...")
        student_model = load_mnist_model(student_type)

    logger.info("Evaluando student model en test-MNIST original")
    eval_trainer = MnistTrainer(
        model_type=student_type,
        num_epochs=1,
        learning_rate=0.0,
        batch_size=64
    )
    eval_trainer.model = student_model
    eval_trainer.model.to(DEVICE)
    _, acc_student_mnist = eval_trainer.evaluate_on_loader(
        eval_trainer.test_loader,
        loader_name="MNIST-Test"
    )
    logger.info(f"Student model accuracy on MNIST test: {acc_student_mnist:.2f}%")


def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    main()
