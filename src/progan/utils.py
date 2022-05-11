import torch
import torchvision
from torch.utils.data import DataLoader
from math import log2
import src.progan.config as config
from src.data_factory.percolation import generate_percolation_data
from src.progan.config import PERCOLATION_CRITICAL_CONTROL_PARAMETER

def get_loader(image_size, dataset_size=5000):
    
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset, _ = generate_percolation_data(dataset_size=dataset_size, lattice_size=image_size, p_list=[PERCOLATION_CRITICAL_CONTROL_PARAMETER], split=False)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader, dataset

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(
    writer, 
    loss_critic,  
    real, 
    fake, 
    tensorboard_step
):

    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)

def gradient_penalty(
    critic, 
    real, 
    fake, 
    alpha, 
    train_step, 
    device="cpu"
):

    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
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

