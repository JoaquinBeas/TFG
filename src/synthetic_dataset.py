from concurrent.futures import ThreadPoolExecutor
import json
import os
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
        batch_size: int = 100,
        io_workers: int = 4,
        max_per_class=None,
        confidence_threshold: float = 0.0
    ):
        """
        Genera un dataset balanceado con la misma distribución de clases que MNIST original.
        Solo cuenta las muestras cuya confianza (max softmax) ≥ confidence_threshold.
        """
        # 1) Cálculo de cuántas muestras por clase
        mnist_train    = MNIST(root=MNIST_DATA_LOADERS_DIR, train=True, download=False)
        target_counts  = torch.bincount(mnist_train.targets, minlength=MNIST_N_CLASSES).tolist()
        if isinstance(max_per_class, int) and max_per_class > 0:
            target_counts = [min(cnt, max_per_class) for cnt in target_counts]
        total_needed = sum(target_counts)

        # 2) Limpia / crea carpetas
        if os.path.exists(output_path):
            for entry in os.listdir(output_path):
                path = os.path.join(output_path, entry)
                if os.path.isdir(path):
                    for f in os.listdir(path):
                        os.remove(os.path.join(path, f))
                else:
                    os.remove(path)
        os.makedirs(output_path, exist_ok=True)
        for c in range(MNIST_N_CLASSES):
            os.makedirs(os.path.join(output_path, str(c)), exist_ok=True)

        executor       = ThreadPoolExecutor(max_workers=io_workers)
        counts         = [0] * MNIST_N_CLASSES
        total_generated = 0

        # Mean/std en GPU para normalizar cada batch
        mean = torch.tensor([0.5], device=self.device).view(1,1,1,1)
        std  = torch.tensor([0.5], device=self.device).view(1,1,1,1)
        # Los llevamos también a CPU para des-normalizar
        mean_cpu = mean.cpu().view(1,1,1,1)
        std_cpu  = std.cpu().view(1,1,1,1)

        while total_generated < total_needed:
            with torch.no_grad():
                # 3) Muestreo + normalización
                samples = self.diffusion_model.sampling(batch_size, device=self.device)
                samples = (samples - mean) / std

                # 4) Etiquetado
                logits      = self.mnist_model(samples)
                probs       = F.softmax(logits, dim=1)
                preds       = probs.argmax(dim=1).cpu()
                confidences = probs.max(dim=1).values.cpu()

            # 5) Filtrar, contar y guardar
            samples_cpu = samples.cpu()
            for i, (img, p, conf) in enumerate(zip(samples_cpu, preds, confidences)):
                print("confianza",conf)
                if conf < confidence_threshold:
                    continue
                cls = int(p)
                if counts[cls] < target_counts[cls]:
                    idx = counts[cls]
                    counts[cls] += 1
                    total_generated += 1

                    # Des-normalizo para guardar
                    img_vis = img * std_cpu + mean_cpu

                    class_dir    = os.path.join(output_path, str(cls))
                    img_path     = os.path.join(class_dir, f"{idx:05d}.png")
                    prob_path    = os.path.join(class_dir, f"{idx:05d}.json")

                    executor.submit(save_image, img_vis, img_path)
                    data = {
                        'label':      cls,
                        'probs':      probs[i].cpu().tolist(),
                        'confidence': float(conf)
                    }
                    executor.submit(lambda v, pth: json.dump(v, open(pth, 'w')), data, prob_path)

                    if total_generated >= total_needed:
                        break

            tqdm.write(f"Progreso: {total_generated}/{total_needed} muestras. Conteo: {counts}")

        executor.shutdown()
        print(f"Dataset balanceado generado en {output_path}")
        return output_path
