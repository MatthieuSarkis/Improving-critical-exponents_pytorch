from datetime import datetime
from math import log2
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from typing import List, Tuple

from src.progan.generator import Generator
from src.progan.discriminator import Discriminator
from src.progan.utils import gradient_penalty, plot_to_tensorboard, get_loader
import src.progan.config as config

class ProGan():

    def __init__(
        self,
        z_dim: int,
        in_channels: int,
        channels_img: int,
        learning_rate: float,
        device: str,
        logs_path: str,
        load_model: bool = False,
    ) -> None:

        self.learning_rate = learning_rate
        self.device = device
        self.logs_dir_checkpoints, self.logs_dir_images, self.logs_dir_tensorboard = self.build_logs_directories(logs_path=logs_path)
        self.z_dim = z_dim

        self.gen = Generator(
            z_dim, 
            in_channels, 
            img_channels=channels_img,
        ).to(device)

        self.critic = Discriminator(
            in_channels, 
            img_channels=channels_img,
        ).to(device)

        self.opt_gen = optim.Adam(
            self.gen.parameters(), 
            lr=learning_rate, 
            betas=(0.0, 0.99)
        )

        self.opt_critic = optim.Adam(
            self.critic.parameters(), 
            lr=learning_rate, 
            betas=(0.0, 0.99)
        )

        self.scaler_critic = torch.cuda.amp.GradScaler()
        self.scaler_gen = torch.cuda.amp.GradScaler()

        self.writer = SummaryWriter(self.logs_dir_tensorboard)

        if load_model:
            self.load_checkpoint(
                gen_checkpoint_file=config.LOAD_CHECKPOINT_GEN_PATH,
                critic_checkpoint_file=config.LOAD_CHECKPOINT_CRITIC_PATH
            )

    def _train(
        self,
        start_train_at_img_size: int,
        progressive_epochs: List[int],
        lambda_gp: float,
        fixed_noise: torch.tensor,
        dataset_size: int,
        save_model: bool = True,
    ) -> None:

        self.gen.train()
        self.critic.train()

        tensorboard_step = 0
        step = int(log2(start_train_at_img_size / 4))

        for num_epochs in progressive_epochs[step:]:

            alpha = 1e-5
            loader, dataset = get_loader(image_size=4 * 2**step, dataset_size=dataset_size)  # 4->0, 8->1, 16->2, 32->3, 64->4, 128->5

            for epoch in range(num_epochs):

                print("(Current image size: {}) Epoch [{}/{}]".format(4 * 2**step, epoch+1, num_epochs))

                tensorboard_step, alpha = self._train_one_step(
                    loader=loader,
                    dataset=dataset,
                    step=step,
                    lambda_gp=lambda_gp,
                    progressive_epochs=progressive_epochs,
                    fixed_noise=fixed_noise,
                    tensorboard_step=tensorboard_step,
                    alpha=alpha
                )

                if save_model:
                    self.save_checkpoint()

            step += 1

    def _train_one_step(
        self,
        loader,
        dataset,
        step: int,
        lambda_gp: float,
        progressive_epochs: List[int],
        fixed_noise: torch.tensor,
        tensorboard_step: int,
        alpha: float,
    ) -> Tuple[int, float]:

        loop = tqdm(loader, leave=True)

        for batch_idx, real in enumerate(loop):

            real = real.to(self.device)
            cur_batch_size = real.shape[0]

            noise = torch.randn(cur_batch_size, self.z_dim, 1, 1).to(self.device)

            with torch.cuda.amp.autocast():

                fake = self.gen(noise, alpha, step)
                critic_real = self.critic(real, alpha, step)
                critic_fake = self.critic(fake.detach(), alpha, step)
                gp = gradient_penalty(self.critic, real, fake, alpha, step, device=self.device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp + (0.001 * torch.mean(critic_real ** 2))

            self.opt_critic.zero_grad()
            self.scaler_critic.scale(loss_critic).backward()
            self.scaler_critic.step(self.opt_critic)
            self.scaler_critic.update()

            with torch.cuda.amp.autocast():

                gen_fake = self.critic(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)

            self.opt_gen.zero_grad()
            self.scaler_gen.scale(loss_gen).backward()
            self.scaler_gen.step(self.opt_gen)
            self.scaler_gen.update()

            # Update alpha and ensure less than 1
            alpha += cur_batch_size / ((progressive_epochs[step] * 0.5) * len(dataset))
            alpha = min(alpha, 1)

            if batch_idx % 500 == 0:

                with torch.no_grad():
                    fixed_fakes = torch.sign(self.gen(fixed_noise, alpha, step)) # casting the spins to +1 and -1

                plot_to_tensorboard(
                    self.writer,
                    loss_critic.item(),
                    real.detach(),
                    fixed_fakes.detach(),
                    tensorboard_step,
                )

                tensorboard_step += 1

            loop.set_postfix(
                gp=gp.item(),
                loss_critic=loss_critic.item(),
            )

        return tensorboard_step, alpha

    def generate_images(
        self,
        steps: int, 
        n_images: int = 100
    ) -> None:

        self.gen.eval()

        for i in range(n_images):

            with torch.no_grad():
                noise = torch.randn(size=(1, self.z_dim, 1, 1), device=self.device, dtype=torch.float32)
                img = torch.sign(self.gen(noise, alpha=1.0, steps=steps)) # casting the spins to +1 and -1
                save_image(img, os.path.join(self.logs_dir_images, 'img_{}.png'.format(i)))

        self.gen.train()


    def save_checkpoint(self) -> None:

        print("=> Saving checkpoint")

        gen_checkpoint_path = os.path.join(self.logs_dir_checkpoints, 'generator.pt')
        critic_checkpoint_path = os.path.join(self.logs_dir_checkpoints, 'critic.pt')
    
        gen_checkpoint = {
            "state_dict": self.gen.state_dict(),
            "optimizer": self.opt_gen.state_dict(),
        }

        critic_checkpoint = {
            "state_dict": self.critic.state_dict(),
            "optimizer": self.opt_critic.state_dict(),
        }

        torch.save(gen_checkpoint, gen_checkpoint_path)
        torch.save(critic_checkpoint, critic_checkpoint_path)

    def load_checkpoint(
        self,
        gen_checkpoint_file: str,
        critic_checkpoint_file: str, 
    ) -> None:

        print("=> Loading checkpoint")
        gen_checkpoint = torch.load(gen_checkpoint_file, map_location="cuda")
        critic_checkpoint = torch.load(critic_checkpoint_file, map_location="cuda")
        self.gen.load_state_dict(gen_checkpoint["state_dict"])
        self.critic.load_state_dict(critic_checkpoint["state_dict"])
        self.opt_gen.load_state_dict(gen_checkpoint["optimizer"])
        self.opt_critic.load_state_dict(critic_checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in self.opt_gen.param_groups:
            param_group["lr"] = self.learning_rate
        for param_group in self.opt_critic.param_groups:
            param_group["lr"] = self.learning_rate

    @staticmethod
    def build_logs_directories(logs_path: str) -> Tuple[str, str, str]:
    
        logs_dir = os.path.join(logs_path, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
        logs_dir_checkpoints = os.path.join(logs_dir, 'checkpoints')
        logs_dir_images = os.path.join(logs_dir, 'images')
        logs_dir_tensorboard = os.path.join(logs_dir, 'tensorboard')
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(logs_dir_checkpoints, exist_ok=True)
        os.makedirs(logs_dir_images, exist_ok=True)
        os.makedirs(logs_dir_tensorboard, exist_ok=True)

        return logs_dir_checkpoints, logs_dir_images, logs_dir_tensorboard