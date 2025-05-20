"""
Re-trains a fresh MNIST **teacher** from scratch every time this script runs,
plots the learning curves and **removes the model checkpoints folder at the end**
so no artefacts remain on disk.

Outputs
-------
* `learning_accuracy_<timestamp>.png` – loss & accuracy por epoch.
* `batch_metrics_<timestamp>.png`     – loss & accuracy por batch del test.

The images are saved in `src/data/plots/` (created if absent). After plotting we
delete `src/data/models_teacher/` to keep the workspace clean.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from src.mnist_models.resnet_preact import ResNetPreAct  # noqa: E402

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# Defaults (no argparse) ------------------------------------------------------
# ----------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOTS_DIR = Path("src/data/plots")
MODEL_DIR = Path("src/data/models_teacher")  # created, used, then deleted
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Model definition ------------------------------------------------------------
# ----------------------------------------------------------------------------
class Cfg: pass
cfg = Cfg(); cfg.model = Cfg()
cfg.model.in_channels    = 1
cfg.model.n_classes      = 10
cfg.model.base_channels  = 16
cfg.model.block_type     = 'basic'
cfg.model.depth          = 20
cfg.model.remove_first_relu = False
cfg.model.add_last_bn    = False
cfg.model.preact_stage   = [True, True, True]
model = ResNetPreAct(cfg).to(DEVICE)

# ----------------------------------------------------------------------------
# Data loaders ----------------------------------------------------------------
# ----------------------------------------------------------------------------

def get_loaders(batch_size: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("src/data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("src/data", train=False, download=True, transform=tfm)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_ld, test_ld

# ----------------------------------------------------------------------------
# Training & evaluation loops -------------------------------------------------
# ----------------------------------------------------------------------------

def train_one_epoch(model, loader, opt):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        running_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    loss_hist, acc_hist = [], []
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            preds = logits.argmax(1)
            batch_acc = (preds == y).sum().item() / y.size(0)

            loss_hist.append(loss.item())
            acc_hist.append(batch_acc)

            running_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (
        running_loss / total,
        correct / total,
        loss_hist,
        acc_hist,
    )

# ----------------------------------------------------------------------------
# Plotting helpers ------------------------------------------------------------
# ----------------------------------------------------------------------------

def save_learning_curves(train_losses, test_losses, train_accs, test_accs, ts):
    plt.figure(figsize=(10, 4))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve")
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train acc")
    plt.plot(test_accs, label="Test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy evolution")
    plt.legend()

    out = PLOTS_DIR / f"learning_accuracy_{ts}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_batch_metrics(loss_hist, acc_hist, ts):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist)
    plt.title("Batch loss (test)")
    plt.subplot(1, 2, 2)
    plt.plot(acc_hist)
    plt.title("Batch accuracy (test)")
    out = PLOTS_DIR / f"batch_metrics_{ts}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out

# ----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_loaders(BATCH_SIZE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, opt)
        vl, va, _, _ = evaluate(model, test_loader)

        train_losses.append(tl)
        train_accs.append(ta)
        test_losses.append(vl)
        test_accs.append(va)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"train loss {tl:.4f} acc {ta*100:.2f}% | "
            f"test loss {vl:.4f} acc {va*100:.2f}%"
        )

    # Final evaluation batch-wise
    _, _, loss_hist, acc_hist = evaluate(model, test_loader)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot1 = save_learning_curves(train_losses, test_losses, train_accs, test_accs, ts)
    plot2 = save_batch_metrics(loss_hist, acc_hist, ts)

    # Save model checkpoint before deleting folder (optional reference)
    chk_path = MODEL_DIR / f"model_teacher_{ts}.pt"
    torch.save(model, chk_path)
    print("Checkpoint saved at", chk_path)

    # Remove the whole models folder to leave workspace clean
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    print("Removed folder", MODEL_DIR)

    print("Plots saved:")
    print("  •", plot1)
    print("  •", plot2)
    print(f"Final test accuracy: {test_accs[-1]*100:.2f}%")


if __name__ == "__main__":
    main()
