# src/mnist_models/mnist_decision_tree.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from src.utils.data_loader import get_mnist_dataloaders

class MNISTDecisionTree(nn.Module):
    """
    Árbol de decisión adaptado a la interfaz nn.Module para encajar en MnistTrainer:
    - Entrena el DecisionTreeClassifier con MNIST real en el constructor.
    - Tiene un parámetro dummy en el grafo para que el optimizador no falle.
    - forward(x) devuelve un one‑hot [N,10] con la clase predicha al 100% de confianza.
    """

    def __init__(self, max_depth: int = None, **dt_kwargs):
        super().__init__()
        # Parámetro “dummy” que formará parte del grafo y evitará lista de params vacía
        self.dummy_param = nn.Parameter(torch.zeros(1))

        # Entrenamos el árbol con MNIST real
        self.clf = DecisionTreeClassifier(max_depth=max_depth, **dt_kwargs)
        train_loader, _ = get_mnist_dataloaders()
        X_parts, y_parts = [], []
        for imgs, labels in train_loader:
            arr = imgs.view(imgs.size(0), -1).cpu().numpy()
            X_parts.append(arr)
            y_parts.append(labels.cpu().numpy())
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        self.clf.fit(X, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor [N,1,28,28], normalizado [-1,1].
        Devuelve un tensor [N,10] one‑hot con un 1.0 en la clase predicha,
        más dummy_param para mantener el grad_fn.
        """
        # 1) Extraer numpy y predecir clases
        x_np = x.detach().cpu().view(x.size(0), -1).numpy()
        preds = self.clf.predict(x_np)  # array shape (N,)

        # 2) Construir one‑hot en el device de x
        N = x.size(0)
        device = x.device
        proba_tensor = torch.zeros(N, 10, device=device, dtype=torch.float32)
        idxs = torch.arange(N, device=device)
        proba_tensor[idxs, torch.from_numpy(preds).to(device)] = 1.0

        # 3) Asegurar que dummy_param está en el mismo dispositivo
        dummy = self.dummy_param
        if dummy.device != device:
            dummy = dummy.to(device)

        # 4) Sumar dummy_param (broadcast a [N,10]) para enganchar el grad_fn
        return proba_tensor + dummy
