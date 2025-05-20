import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from src.diffusion_models.diffusion_guided_unet import DiffusionGuidedUnet
from src.diffusion_models.diffusion_resnet import DiffusionResnet
from src.diffusion_models.diffusion_unet import DiffusionUnet
from src.diffusion_models.diffusion_unet_conditional import ConditionalDiffusionModel
from src.utils.config import (
    DEVICE,
    MNIST_BATCH_SIZE,
    MNIST_EPOCHS,
    MNIST_LEARNING_RATE,
    TRAIN_DIFFUSION_MODEL_DIR,
    TRAIN_DIFFUSION_SAMPLES_DIR,
    MODEL_IMAGE_SIZE,
    MODEL_IN_CHANNELS,
    TIMESTEPS,
    MNIST_N_CLASSES
)
from src.utils.data_loader import get_mnist_dataloaders
from src.utils.diffusion_models_enum import DiffusionModelType

class DiffussionTrainer:
    def __init__(
        self,
        model_type: DiffusionModelType = DiffusionModelType.CONDITIONAL_UNET,
        num_epochs=100,
        learning_rate=MNIST_LEARNING_RATE,
        batch_size=MNIST_BATCH_SIZE,
        early_stopping_patience=100,
        model_path=TRAIN_DIFFUSION_MODEL_DIR,
        image_path=TRAIN_DIFFUSION_SAMPLES_DIR
    ):
        self.device = torch.device(DEVICE)
        full_train_loader, self.test_loader = get_mnist_dataloaders(batch_size=batch_size)
        train_dataset = full_train_loader.dataset
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Instanciar modelo según enum
        # if model_type == DiffusionModelType.GUIDED_UNET:
        #     self.model = DiffusionGuidedUnet().to(self.device)
        # elif model_type == DiffusionModelType.RESNET:
        #     self.model = DiffusionResnet().to(self.device)
        if model_type == DiffusionModelType.UNET:
            self.model = DiffusionUnet(
                image_size=MODEL_IMAGE_SIZE,
                in_channels=MODEL_IN_CHANNELS,
                timesteps=TIMESTEPS
            ).to(self.device)
        elif model_type == DiffusionModelType.CONDITIONAL_UNET:
            self.model = ConditionalDiffusionModel().to(self.device)
        else:
            raise ValueError(f"Unknown diffusion model: {model_type}")

        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.checkpoint_dir = os.path.join(model_path, model_type.value)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        self.image_path = os.path.join(image_path, model_type.value)
        os.makedirs(self.image_path, exist_ok=True)
        print(f"Image dir: {self.image_path}")
        self.early_stopping_patience = early_stopping_patience

    def train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            data = data * 2.0 - 1.0
            noise = torch.randn_like(data)
            self.optimizer.zero_grad()
            if isinstance(self.model, ConditionalDiffusionModel):
                loss = self.model.p_losses(data, labels)
            else:
                pred = self.model(data, noise)
                loss = self.criterion(pred, noise)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} - Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.6f}")

    def evaluate_on_loader(self, loader, loader_name=""):
        self.model.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                data = data * 2.0 - 1.0
                noise = torch.randn_like(data)
                if isinstance(self.model, ConditionalDiffusionModel):
                    loss = self.model.p_losses(data, labels)
                else:
                    pred = self.model(data, noise)
                    loss = self.criterion(pred, noise)
                total_loss += loss.item() * data.size(0)
                total += data.size(0)
        avg_loss = total_loss / total
        if loader_name:
            print(f"{loader_name} avg loss: {avg_loss:.6f}")
        else:
            print(f"Avg loss: {avg_loss:.6f}")
        return avg_loss

    def train_model(self):
        best_val, patience = float('inf'), 0
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n=== Epoch {epoch} ===")
            self.train_epoch(epoch)
            val_loss = self.evaluate_on_loader(self.val_loader, loader_name="Validation")

            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{epoch}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Generate samples
            if isinstance(self.model, ConditionalDiffusionModel):
                # Sample every 10 epochs
                if epoch % 10 == 0:
                    for cls in range(self.model.num_classes):
                        labels = torch.full((1,), cls, device=self.device, dtype=torch.long)
                        sample_imgs, _ = self.model.sample(labels, n_sample=1, guide_w=2)
                        save_image(sample_imgs, os.path.join(self.image_path, f"epoch{epoch}_class{cls}.png"), normalize=True)
                else:
                    print(f"Skipping conditional sampling at epoch {epoch}")
            else:
                sample = self.model.sampling(9)
                save_image(sample, os.path.join(self.image_path, f"epoch{epoch}.png"), nrow=3, normalize=True)
                print(f"Saved sample at epoch {epoch}")

            # Early stopping
            if val_loss < best_val:
                best_val, patience = val_loss, 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    print("Early stopping.")
                    break

        test_loss = self.evaluate_on_loader(self.test_loader, loader_name="Test")
        final_ckpt = os.path.join(self.checkpoint_dir, "last_model.pt")
        torch.save(self.model.state_dict(), final_ckpt)
        print(f"Final model saved: {final_ckpt}")
        return test_loss

    def get_model(self):
        return self.model
