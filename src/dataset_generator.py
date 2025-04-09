import torch
from src.utils.config import DEVICE

class DataSetGenerator:
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

    def generate_dataset(self, n_samples, output_path="generated_dataset.pt"):
        """
        Generates a dataset using the diffusion model's sampling and MNIST model's predictions.
        
        Args:
            n_samples (int): Number of images to generate.
            output_path (str): Path to save the generated dataset.
        
        Returns:
            dict: Contains 'samples' (images) and 'labels' (predicted).
        """
        with torch.no_grad():
            # Generate samples using the diffusion model.
            samples = self.diffusion_model.sampling(n_samples, device=DEVICE)
            # Predict labels for the generated samples using the MNIST model.
            outputs = self.mnist_model(samples)
            predicted_labels = outputs.argmax(dim=1)
        
        dataset = {"samples": samples.cpu(), "labels": predicted_labels.cpu()}
        torch.save(dataset, output_path)
        print(f"Dataset generated and saved to: {output_path}")
        return dataset
