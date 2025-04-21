import os
import torch
from torchvision.utils import save_image
from src.utils.config import DEVICE, SAVE_SYNTHETIC_DATASET_DIR

class SyntheticDataset:
    def __init__(self, diffusion_model, mnist_model):
        """
        Initialize the dataset generator with already loaded models.
        
        Args:
            diffusion_model (nn.Module): An already instantiated diffusion model.
            mnist_model (nn.Module): An already instantiated MNIST classifier.
        """
        self.diffusion_model = diffusion_model.to(DEVICE)
        self.diffusion_model.eval()
        self.mnist_model = mnist_model.to(DEVICE)
        self.mnist_model.eval()

    def generate_dataset(self, n_samples, output_path=SAVE_SYNTHETIC_DATASET_DIR):
        """
        Genera n_samples imágenes, las etiqueta con el teacher MNIST y las guarda
        en subcarpetas 0–9 para poder cargarlas con ImageFolder.
        """
        with torch.no_grad():
            # 1) Muestreo
            samples = self.diffusion_model.sampling(n_samples, device=DEVICE)
            # 2) Etiquetado
            outputs = self.mnist_model(samples)
            labels = outputs.argmax(dim=1)

        # 3) Prepara carpetas
        if os.path.exists(output_path):
            # opcional: borrar contenido previo
            # shutil.rmtree(output_path)
            pass
        os.makedirs(output_path, exist_ok=True)

        # 4) Guarda cada imagen en su carpeta de clase
        for idx, (img, lbl) in enumerate(zip(samples.cpu(), labels.cpu())):
            class_dir = os.path.join(output_path, str(int(lbl.item())))
            os.makedirs(class_dir, exist_ok=True)
            # save_image ya normaliza de [0,1] a [0,255] y crea PNG por defecto
            save_image(img, os.path.join(class_dir, f"{idx:05d}.png"))

        print(f"Dataset sintético guardado en estructura ImageFolder en: {output_path}")
        return output_path
