from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
import struct
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
        max_per_class: Optional[int] = None,
        confidence_threshold: float = 0.85
    ) -> str:
        """
        Genera un dataset balanceado en formato IDX gzip usando sampling condicional.
        - Hasta `min(original_count, max_per_class)` muestras por clase.
        - Sólo incluye imágenes con confidence ≥ `confidence_threshold`.
        Mantiene prints de debug y barra de progreso.
        """
        os.makedirs(output_path, exist_ok=True)

        mnist_train = MNIST(root=MNIST_DATA_LOADERS_DIR, train=True, download=False)
        target_counts = torch.bincount(mnist_train.targets, minlength=MNIST_N_CLASSES).tolist()
        if isinstance(max_per_class, int) and max_per_class > 0:
            target_counts = [min(c, max_per_class) for c in target_counts]
        total_needed = sum(target_counts)

        images_list = []  # numpy uint8 arrays 28x28
        labels_list = []  # int labels

        mean     = torch.tensor([0.5], device=self.device).view(1,1,1,1)
        std      = torch.tensor([0.5], device=self.device).view(1,1,1,1)
        mean_cpu = mean.cpu().view(1,1,1,1)
        std_cpu  = std.cpu().view(1,1,1,1)

        # 4) Estado de generación
        counts = [0]*MNIST_N_CLASSES
        total_generated = 0
        thr_frac = confidence_threshold

        # 5) Barra de progreso global
        pbar = tqdm(total=total_needed, desc="Total samples", unit="img")

        # 6) Bucle de muestreo con debug
        while total_generated < total_needed:
            batch = min(batch_size, total_needed - total_generated)
            labels = torch.full((batch,), -1, dtype=torch.long, device=self.device)
            # preparar labels por clase
            idx_offset = 0
            for cls, cnt in enumerate(counts):
                need = target_counts[cls] - cnt
                if need <= 0:
                    continue
                take = min(need, batch - idx_offset)
                labels[idx_offset:idx_offset+take] = cls
                idx_offset += take
                if idx_offset >= batch:
                    break

            # sampling condicional
            samples = self.diffusion_model.sampling(
                n_samples=batch,
                labels=labels,
                device=self.device,
                guide_w=0.5
            )  # Tensor [batch,1,28,28] en [0,1]

            # clasificación y cálculo de confianza
            with torch.no_grad():
                samples_norm = (samples - mean) / std
                logits      = self.mnist_model(samples_norm)
                confidences = F.softmax(logits, dim=1).max(dim=1).values.cpu().tolist()
            avg_conf = sum(confidences) / len(confidences)
            print(f"Batch de {batch}: avg_conf={avg_conf:.2f}")

            # filtrar y acumular
            valid_count = 0
            for img, label, conf in zip(samples.cpu(), labels.cpu().tolist(), confidences):
                if conf < thr_frac:
                    continue
                if counts[label] >= target_counts[label]:
                    continue
                # desnormalize y a uint8
                arr = ((img * std_cpu + mean_cpu).squeeze(0)
                       .mul(255).clamp(0,255).byte().numpy())
                images_list.append(arr)
                labels_list.append(label)
                counts[label] += 1
                total_generated += 1
                valid_count += 1
                pbar.update(1)
                if total_generated >= total_needed:
                    break

            if valid_count == 0:
                thr_frac = max(thr_frac - 0.01, 0.5)
        pbar.close()

        magic_img = 0x00000803
        magic_lbl = 0x00000801
        num = len(images_list)
        H, W = (1,28,28)[-2:]

        img_path = os.path.join(output_path, "train-images-idx3-ubyte.gz")
        lbl_path = os.path.join(output_path, "train-labels-idx1-ubyte.gz")
        with gzip.open(img_path, "wb") as f:
            f.write(struct.pack(">IIII", magic_img, num, H, W))
            for arr in images_list:
                f.write(arr.tobytes())
        with gzip.open(lbl_path, "wb") as f:
            f.write(struct.pack(">II", magic_lbl, num))
            f.write(bytes(labels_list))

        print(f"Dataset IDX generado: {num} muestras → {output_path}")
        return output_path