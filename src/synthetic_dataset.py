from concurrent.futures import ThreadPoolExecutor
import json
import os
from typing import Optional
import torch
from torchvision.utils import save_image
from src.utils.config import DEVICE, SAVE_SYNTHETIC_DATASET_DIR, MNIST_DATA_LOADERS_DIR, MNIST_N_CLASSES
from torchvision.datasets import MNIST
from tqdm import tqdm
import torch.nn.functional as F

class SyntheticDataset:
    def __init__(self, diffusion_model, mnist_model):
        """
        Initialize the dataset generator with already loaded models.
        
        Args:
            diffusion_model (nn.Module): An instantiated diffusion model.
            mnist_model (nn.Module): An instantiated MNIST classifier.
        """
        self.diffusion_model = diffusion_model.to(DEVICE)
        self.diffusion_model.eval()
        self.mnist_model = mnist_model.to(DEVICE)
        self.mnist_model.eval()
        self.device = DEVICE

    def generate_dataset(
        self,
        n_samples,
        output_path: str = SAVE_SYNTHETIC_DATASET_DIR,
        confidence_threshold: float = 0.75
    ):
        """
        Genera n_samples imágenes, las etiqueta con el teacher MNIST y las guarda
        en subcarpetas 0–9 para poder cargarlas con ImageFolder.
        Sólo guarda las muestras cuya confianza (max softmax) ≥ confidence_threshold.
        """
        with torch.no_grad():
            # 1) Muestreo
            samples = self.diffusion_model.sampling(n_samples, device=self.device)
            # 2) Normalización ([-1,1]), igual que en tus dataloaders
            mean = torch.tensor([0.5], device=self.device).view(1,1,1,1)
            std  = torch.tensor([0.5], device=self.device).view(1,1,1,1)
            samples = (samples - mean) / std
            # 3) Etiquetado
            logits      = self.mnist_model(samples)
            probs       = F.softmax(logits, dim=1)
            labels      = probs.argmax(dim=1).cpu()
            confidences = probs.max(dim=1).values.cpu()

        # 4) Filtrado por confianza
        keep_mask = confidences >= confidence_threshold
        kept = keep_mask.sum().item()
        print(f"Filtradas {kept}/{n_samples} muestras con confianza ≥ {confidence_threshold}")

        # Preparo tensores CPU ya filtrados
        samples_cpu     = samples.cpu()[keep_mask]
        probs_cpu       = probs.cpu()[keep_mask]
        labels_cpu      = labels[keep_mask]
        confidences_cpu = confidences[keep_mask]

        # Preparo mean/std en CPU para des-normalizar al guardar
        mean_cpu = mean.cpu().view(1,1,1,1)
        std_cpu  = std.cpu().view(1,1,1,1)

        # 5) Prepara carpetas
        if os.path.exists(output_path):
            # opcional: limpia carpeta previa
            # shutil.rmtree(output_path)
            pass
        os.makedirs(output_path, exist_ok=True)

        # 6) Guarda cada imagen en su carpeta de clase
        for idx, (img, prob, conf, lbl) in enumerate(
            zip(samples_cpu, probs_cpu, confidences_cpu, labels_cpu)
        ):
            # Reconstruyo rango [0,1] para save_image
            img_vis = img * std_cpu + mean_cpu

            png_path  = os.path.join(output_path, f"{idx:05d}.png")
            json_path = os.path.join(output_path, f"{idx:05d}.json")

            save_image(img_vis, png_path)
            data = {
                'label':      int(lbl),
                'probs':      prob.tolist(),
                'confidence': float(conf)
            }
            with open(json_path, 'w') as f:
                json.dump(data, f)

        print(f"Dataset sintético guardado en: {output_path}")
        return output_path

    def generate_balanced_dataset(
        self,
        output_path: str = SAVE_SYNTHETIC_DATASET_DIR,
        batch_size: int = 200,
        io_workers: int = 4,
        max_per_class: Optional[int] = None,
        confidence_threshold: float = 0.85
    ) -> str:
        """
        Genera un dataset balanceado usando sampling condicional de tu modelo de difusión.
        - Hasta `min(original_count, max_per_class)` muestras por clase.
        - Sólo guarda imágenes con confidence ≥ `confidence_threshold`.
        Muestra progreso e información de confianza media por lote.
        """
        mnist_train = MNIST(root=MNIST_DATA_LOADERS_DIR, train=True, download=False)
        counts = torch.bincount(mnist_train.targets, minlength=MNIST_N_CLASSES).tolist()
        if isinstance(max_per_class, int) and max_per_class > 0:
            counts = [min(c, max_per_class) for c in counts]

        os.makedirs(output_path, exist_ok=True)
        for cls in range(MNIST_N_CLASSES):
            os.makedirs(os.path.join(output_path, str(cls)), exist_ok=True)

        executor = ThreadPoolExecutor(max_workers=io_workers)
        for cls, total_needed in enumerate(counts):
            generated = 0
            pbar = tqdm(total=total_needed, desc=f"Clase {cls}", unit="img")

            while generated < total_needed:
                batch = min(batch_size, total_needed - generated)
                labels = torch.full((batch,), cls, dtype=torch.long, device=self.device)
                samples = self.diffusion_model.sampling(
                    n_samples=batch,
                    labels=labels,
                    device=self.device,
                    guide_w=0.5
                )

                with torch.no_grad():
                    logits      = self.mnist_model(samples)
                    confidences = F.softmax(logits, dim=1).max(dim=1).values.cpu().tolist()
                avg_conf = sum(confidences) / len(confidences)
                pbar.write(f"Batch de {batch}: avg_conf={avg_conf:.2f}")

                valid_count = 0
                for img, conf in zip(samples.cpu(), confidences):
                    if conf < confidence_threshold:
                        continue

                    idx = generated
                    img_path = os.path.join(output_path, str(cls), f"{idx:05d}.png")
                    executor.submit(save_image, img, img_path)
                    generated += 1
                    valid_count += 1
                    if generated >= total_needed:
                        break
                pbar.update(valid_count)
                pbar.set_postfix({'needed': total_needed - generated})
            pbar.close()
            if generated < total_needed:
                print(f"Warning: clase {cls} generadas {generated}/{total_needed}")
        executor.shutdown(wait=True)
        print(f"Dataset balanceado generado en {output_path}")
        return output_path