import torch
import torchvision
from torch.utils.data import DataLoader
from math import log2
from typing import Dict

from src.progan.config import config
from src.data_factory.percolation import generate_percolation_data

class CNNLoss(torch.nn.Module):
    
    def __init__(
        self,
        loss_function: torch.nn.modules.loss._Loss,
        cnn: torch.nn.Module,
        wanted_output: float = 0.5928
    ) -> None:
        
        super(CNNLoss, self).__init__()
        
        self.wanted_output = wanted_output
        self.loss_function = loss_function
        self.cnn = cnn
        self.cnn.eval()

    def forward(
        self,
        generated_images: torch.tensor,
    ) -> torch.tensor:
        
        return self._cnn_loss(generated_images)
        
    def _cnn_loss(
        self, 
        generated_images: torch.tensor,
    ) -> torch.tensor:
    
        predicted_output = self.cnn(generated_images)
        wanted_output_ = torch.full_like(predicted_output, self.wanted_output, dtype=torch.float32)

        return self.loss_function(wanted_output_, predicted_output)


def get_loader(
    image_size: int, 
    dataset_size: int = 5000,
    statistical_control_parameter: float = 0.5928
) -> torch.utils.data.dataloader.DataLoader:
    
    batch_size = config['BATCH_SIZES'][int(log2(image_size / 4))]

    dataset, _ = generate_percolation_data(
        dataset_size=dataset_size, 
        lattice_size=image_size, 
        p_list=[statistical_control_parameter], 
        split=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True,
    )

    return loader

def plot_to_tensorboard(
    writer: torch.utils.tensorboard.writer.SummaryWriter, 
    losses: Dict[str, float],  
    real: torch.tensor, 
    fake: torch.tensor, 
    global_step: int
) -> None:

    for key, value in losses.items():
        writer.add_scalar(key, value, global_step=global_step)

    with torch.no_grad():
       
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=global_step)
        writer.add_image("Fake", img_grid_fake, global_step=global_step)

def gradient_penalty(
    critic: torch.nn.Module, 
    real: torch.tensor, 
    fake: torch.tensor, 
    alpha: float, 
    train_step: int, 
    device: str = "cpu"
) -> torch.tensor:

    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

