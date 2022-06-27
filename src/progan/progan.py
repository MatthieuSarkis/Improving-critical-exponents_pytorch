from math import log2
import numpy as np
import os
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Optional

from src.progan.generator import Generator
from src.progan.discriminator import Discriminator
from src.progan.utils import gradient_penalty, plot_to_tensorboard, get_loader, CNNLoss
from src.cnn.cnn import CNN
from src.data_factory.percolation import generate_percolation_data

class ProGan():

    def __init__(
        self,
        noise_dim: int,
        in_channels: int,
        channels_img: int,
        learning_rate: float,
        factors: List[int],
        device: str,
        logs_path: str,
        path_to_trained_model: Optional[str] = None,
        cnn_path: Optional[str] = None,
        stat_phys_model: str = 'percolation',
        statistical_control_parameter: float = 0.5928,
        use_tensorboard: bool = False,
        cnn_model_path: Optional[str] = None
    ) -> None:

        self.learning_rate = learning_rate
        self.device = device
        self.generated_images_path = logs_path
        if path_to_trained_model is None:
            self.logs_dir_checkpoints, self.logs_dir_images, self.save_dir_losses, self.logs_dir_tensorboard = self.build_logs_directories(logs_path=logs_path, use_tensorboard=use_tensorboard)
        self.noise_dim = noise_dim
        self.stat_phys_model = stat_phys_model
        self.statistical_control_parameter = statistical_control_parameter
        self.use_tensorboard = use_tensorboard

        self.generator = Generator(
            noise_dim=noise_dim,
            in_channels=in_channels,
            factors=factors,
            img_channels=channels_img,
        ).to(device)

        self.critic = Discriminator(
            in_channels=in_channels,
            img_channels=channels_img,
            factors=factors
        ).to(device)

        self.opt_generator = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(0.0, 0.99)
        )

        self.opt_critic = optim.Adam(
            self.critic.parameters(),
            lr=learning_rate,
            betas=(0.0, 0.99)
        )

        if 'cuda' in device:
            self.scaler_critic = torch.cuda.amp.GradScaler()
            self.scaler_gen = torch.cuda.amp.GradScaler()

        if use_tensorboard:
            self.writer = SummaryWriter(self.logs_dir_tensorboard)

        if cnn_path is not None:

            cnn_checkpoint = torch.load(cnn_model_path, map_location=torch.device(self.device))
            cnn_checkpoint['constructor_args']['device'] = self.device
            self.cnn = CNN(**cnn_checkpoint['constructor_args'])
            self.cnn.load_state_dict(cnn_checkpoint['model_state_dict'])

            self.cnn_criterion = CNNLoss(
                loss_function=nn.L1Loss(reduction='sum'),
                cnn=self.cnn,
                wanted_output=statistical_control_parameter
            )

        else:
            self.cnn = None

        if path_to_trained_model is not None:
            self.load_checkpoint(path_to_trained_model=path_to_trained_model)

        self.losses = {
            "Wasserstein Distance": {"Global Step": [], "Loss": []},
            "CNN Loss": {"Global Step": [], "Loss": []}
        }

    def _train(
        self,
        start_train_at_img_size: int,
        progressive_epochs: List[int],
        lambda_gp: float,
        fixed_noise: torch.tensor,
        batch_sizes: List[int],
        dataset_size: int,
        save_model: bool = True,
        cnn_loss_ratio: Optional[float] = None
    ) -> None:

        self.losses = {
            "Wasserstein Distance": {"Global Step": [], "Loss": []},
            "CNN Loss": {"Global Step": [], "Loss": []}
        }

        self.generator.train()
        self.critic.train()

        global_step = 0
        step = int(log2(start_train_at_img_size / 4))

        for num_epochs in progressive_epochs[step:]:

            alpha = 1e-5

            loader = get_loader(
                image_size=4 * 2**step,
                batch_sizes=batch_sizes,
                dataset_size=dataset_size,
                stat_phys_model=self.stat_phys_model,
                statistical_control_parameter=self.statistical_control_parameter
            )

            for epoch in range(num_epochs):

                print(
                    "\n Global Step [{}/{}], Epoch [{}/{}], Current image size: {}, Final image size: {}, stat_phys_model: {}, percolation parameter: {:.4f}".format(
                        global_step+1,
                        sum(progressive_epochs),
                        epoch+1, num_epochs,
                        4 * 2**step,
                        4 * 2**(len(progressive_epochs) - 1),
                        self.stat_phys_model,
                        self.statistical_control_parameter)
                )

                global_step, alpha = self._train_one_step(
                    loader=loader,
                    dataset_size=dataset_size,
                    step=step,
                    epoch=epoch,
                    lambda_gp=lambda_gp,
                    progressive_epochs=progressive_epochs,
                    fixed_noise=fixed_noise,
                    global_step=global_step,
                    alpha=alpha,
                    cnn_loss_ratio=cnn_loss_ratio
                )

                if save_model:
                    self.save_checkpoint()
                    self.dump_losses()

            step += 1

    def _train_one_step(
        self,
        loader,
        dataset_size: int,
        step: int,
        epoch: int,
        lambda_gp: float,
        progressive_epochs: List[int],
        fixed_noise: torch.tensor,
        global_step: int,
        alpha: float,
        cnn_loss_ratio: Optional[float] = None
    ) -> Tuple[int, float]:

        loop = tqdm(loader, leave=True)
        image_size = 4 * 2**step
        use_cnn = self.cnn is not None and image_size == 128

        # for logs
        cumulative_wasserstein_distance = 0
        cumulative_cnn_loss = 0

        for batch_idx, real in enumerate(loop):

            real = real.to(self.device)
            cur_batch_size = real.shape[0]

            noise = torch.randn(cur_batch_size, self.noise_dim, 1, 1).to(self.device)

            # Training on GPU
            if 'cuda' in self.device:

                # Training the critic
                with torch.cuda.amp.autocast():

                    fake = self.generator(noise, alpha, step)
                    critic_real = self.critic(real, alpha, step)
                    critic_fake = self.critic(fake.detach(), alpha, step)
                    gp = gradient_penalty(self.critic, real, fake, alpha, step, device=self.device)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp + 0.001 * torch.mean(critic_real ** 2)

                self.opt_critic.zero_grad()
                self.scaler_critic.scale(loss_critic).backward()
                self.scaler_critic.step(self.opt_critic)
                self.scaler_critic.update()

                # Training the generator
                with torch.cuda.amp.autocast():

                    gen_fake = self.critic(fake, alpha, step)
                    loss_gen = -torch.mean(gen_fake)

                    if use_cnn:
                        cnn_loss = self.cnn_criterion(fake)
                        loss_gen += cnn_loss_ratio * cnn_loss / fake.shape[0]   # because we used reduction='sum'

                self.opt_generator.zero_grad()
                self.scaler_gen.scale(loss_gen).backward()
                self.scaler_gen.step(self.opt_generator)
                self.scaler_gen.update()

            # Training on CPU
            else:

                # Training the critic
                fake = self.generator(noise, alpha, step)
                critic_real = self.critic(real, alpha, step)
                critic_fake = self.critic(fake.detach(), alpha, step)
                gp = gradient_penalty(self.critic, real, fake, alpha, step, device=self.device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp + 0.001 * torch.mean(critic_real ** 2)

                self.opt_critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()

                # Training the generator
                gen_fake = self.critic(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)

                if use_cnn:
                    cnn_loss = self.cnn_criterion(fake)
                    loss_gen += cnn_loss_ratio * cnn_loss / fake.shape[0]   # because we used reduction='sum'

                self.opt_generator.zero_grad()
                loss_gen.backward()
                self.opt_generator.step()

            alpha += cur_batch_size / (0.5 * progressive_epochs[step] * dataset_size) # Update alpha and ensure less than 1
            alpha = min(alpha, 1)

            # for logs
            cumulative_wasserstein_distance += (torch.sum(critic_real) - torch.sum(critic_fake)).item()
            if use_cnn:
                cumulative_cnn_loss += cnn_loss.item()

            if batch_idx  == len(loop) - 1:

                with torch.no_grad():
                    fixed_fakes = 0.5 * (torch.sign(self.generator(fixed_noise, alpha, step)) + 1) # casting the spins to +1 and -1 and shifting to 0, 1
                    img_grid_real = torchvision.utils.make_grid(real[:8])
                    img_grid_fake = torchvision.utils.make_grid(fake[:8])
                    save_image(img_grid_real, os.path.join(self.logs_dir_images, 'size={}_epoch={}_real.png'.format(image_size, epoch)))
                    save_image(img_grid_fake, os.path.join(self.logs_dir_images, 'size={}_epoch={}_fake.png'.format(image_size, epoch)))

                if self.use_tensorboard:

                    plot_to_tensorboard(
                        writer=self.writer,
                        losses={"Loss Critic": loss_critic.item(), "Loss CNN": cnn_loss.item()} if use_cnn else {"Loss Critic": loss_critic.item()},
                        real=real.detach(),
                        fake=fixed_fakes.detach(),
                        global_step=global_step,
                    )

                global_step += 1

            loop.set_postfix(
                gp=gp.item(),
                critic_loss=loss_critic.item(),
            )

        # for logs
        cumulative_wasserstein_distance /= dataset_size
        cumulative_cnn_loss /= dataset_size
        self.losses['Wasserstein Distance']['Global Step'].append(global_step)
        self.losses['Wasserstein Distance']['Loss'].append(cumulative_wasserstein_distance)
        if use_cnn:
            self.losses['CNN Loss']['Global Step'].append(global_step)
            self.losses['CNN Loss']['Loss'].append(cumulative_cnn_loss)

        return global_step, alpha

    def generate_images(
        self,
        steps: int,
        n_images: int = 100
    ) -> None:

        image_size = 2**(steps + 2)

        os.makedirs(os.path.join(self.generated_images_path, 'real'), exist_ok=True)
        os.makedirs(os.path.join(self.generated_images_path, 'fake'), exist_ok=True)

        self.generator.eval()

        for i in tqdm(range(n_images)):

            if self.stat_phys_model == 'percolation':

                real_image, _ = generate_percolation_data(
                    dataset_size=1,
                    lattice_size=image_size,
                    p_list=[self.statistical_control_parameter],
                    split=False
                )

            elif self.stat_phys_model == "ising":

                with open('./data/ising/L={}/T={:.4f}.bin'.format(image_size, self.statistical_control_parameter), 'rb') as f:
                   real_image = torch.frombuffer(buffer=f.read(), dtype=torch.int8, offset=0).reshape(-1, 1, image_size, image_size)
                   idx = np.random.randint(real_image.shape[0])
                   real_image = real_image[idx: idx+1].type(torch.float32)

            real_image = ((real_image + 1) / 2).type(torch.int8)

            with torch.no_grad():

                noise = torch.randn(size=(1, self.noise_dim, 1, 1), device=self.device, dtype=torch.float32)
                fake_image = 0.5 * (torch.sign(self.generator(noise, alpha=1.0, steps=steps)) + 1) # casting the spins to +1 and -1 and shifting to 0, 1.
                fake_image = fake_image.cpu().type(torch.int8).view(image_size, image_size).numpy()
                np.save(os.path.join(self.generated_images_path, 'real', 'real_L={}_p={:.4f}_#{}.npy'.format(image_size, self.statistical_control_parameter, i+1)), real_image)
                np.save(os.path.join(self.generated_images_path, 'fake', 'fake_L={}_p={:.4f}_#{}.npy'.format(image_size, self.statistical_control_parameter, i+1)), fake_image)

        print("*** Images Generated ***")

    def save_checkpoint(self) -> None:

        gen_checkpoint_path = os.path.join(self.logs_dir_checkpoints, 'generator.pt')
        critic_checkpoint_path = os.path.join(self.logs_dir_checkpoints, 'critic.pt')

        gen_checkpoint = {
            "state_dict": self.generator.state_dict(),
            "optimizer": self.opt_generator.state_dict(),
        }

        critic_checkpoint = {
            "state_dict": self.critic.state_dict(),
            "optimizer": self.opt_critic.state_dict(),
        }

        torch.save(gen_checkpoint, gen_checkpoint_path)
        torch.save(critic_checkpoint, critic_checkpoint_path)

    def load_checkpoint(
        self,
        path_to_trained_model: str
    ) -> None:

        gen_checkpoint_file = os.path.join(path_to_trained_model, 'checkpoints', 'generator.pt')
        critic_checkpoint_file = os.path.join(path_to_trained_model, 'checkpoints', 'critic.pt')

        gen_checkpoint = torch.load(gen_checkpoint_file, map_location=self.device)
        critic_checkpoint = torch.load(critic_checkpoint_file, map_location=self.device)
        self.generator.load_state_dict(gen_checkpoint["state_dict"])
        self.critic.load_state_dict(critic_checkpoint["state_dict"])
        self.opt_generator.load_state_dict(gen_checkpoint["optimizer"])
        self.opt_critic.load_state_dict(critic_checkpoint["optimizer"])

        for param_group in self.opt_generator.param_groups:
            param_group["lr"] = self.learning_rate
        for param_group in self.opt_critic.param_groups:
            param_group["lr"] = self.learning_rate

    @staticmethod
    def build_logs_directories(
        logs_path: str,
        use_tensorboard: bool
    ) -> Tuple[str, str, str, Optional[str]]:

        logs_dir_checkpoints = os.path.join(logs_path, 'checkpoints')
        os.makedirs(logs_dir_checkpoints, exist_ok=True)

        logs_dir_images = os.path.join(logs_path, 'images')
        os.makedirs(logs_dir_images, exist_ok=True)

        logs_dir_losses = os.path.join(logs_path, 'losses')
        os.makedirs(logs_dir_losses, exist_ok=True)

        if use_tensorboard:
            logs_dir_tensorboard = os.path.join(logs_path, 'tensorboard')
            os.makedirs(logs_dir_tensorboard, exist_ok=True)

        else:
            logs_dir_tensorboard = None

        return logs_dir_checkpoints, logs_dir_images, logs_dir_losses, logs_dir_tensorboard

    def dump_losses(self) -> None:

        with open(os.path.join(self.save_dir_losses, 'losses.json'), 'w') as f:
            json.dump(self.losses, f,  indent=4, separators=(',', ': '))
