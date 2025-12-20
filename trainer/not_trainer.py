from trainer.base_trainer import BaseTrainer
from hydra.utils import instantiate, get_class

from metrics import L2
from models import Discriminator, UNet

import torch
import numpy as np


class NOTrainer(BaseTrainer):
    """
    Class for training Neural Optimal Transport model
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize NOT trainer
        """

        super().__init__(*args, **kwargs)

    def setup_models(self):
        """
        Setup models required for NOT training: UNet, ResNet
        """

        ### Добавить instatinate
        self.generator = UNet(in_channels=4, 
                              out_channels=3, 
                              base_factor=32, 
                              bilinear=True).to(self.device)
        
        self.critic = Discriminator(in_channels=3).to(self.device)


        ### чекать конфиг
        if 1 != 2:
            pass
            # self.generator.load_state_dict(...)
            # self.logger.log_custom_message('UNet weights were loaded')

            # self.critic.load_state_dict(...)
            # self.logger.log_custom_message('ResNet weights were loaded')

        
    def setup_optimizers(self):
        """
        Setup optimizers for generator and critic
        """

        ### Добавить instatinate
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=1e-4, weight_decay=1e-10)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-10)


    def setup_losses(self):
        """
        Setup losses for each task
        """

        self.ot_loss = L2()
        self.gan_loss = lambda G_x, x1: self.critic(G_x) - self.critic(x1)
    
    def to_train(self):
        """
        Set generator and critic parameters to train mode
        """
        
        self.generator.train()
        self.critic.train()

    def to_eval(self):
        """
        Set generator and critic parameters to eval mode
        """

        self.generator.eval()
        self.critic.eval()

    def _sample_noise(self, params: tuple):
        """
        Sample standart normal variable Z


        Args:
        -----
        params : tuple
            Params for standart normal noise
        """

        B, C, H, W = params

        return torch.randn(B, 1, H, W).to(self.device) * np.sqrt(self.config.model_params.noise_variance)

    def train_critic(self, x0: torch.Tensor, 
                           x1: torch.Tensor):
        """
        Function to fit only critic with fixed generator
        """

        self.optimizer_critic.zero_grad()

        z = self._sample_noise(x0.shape)

        ### Freeze generator parameters
        with torch.no_grad():
            tilde_x = self.generator(x0, z)
        
        loss_critic = -(self.gan_loss(tilde_x, x1)).mean()
        loss_critic.backward()

        self.optimizer_critic.step()

        return

    def train_generator(self, x0: torch.Tensor):
        """
        Unfreeze generator and fit it after several critic training steps

        Args:
        -----
        x0 : torch.Tensor
            Input data from p^s
        """

        self.optimizer_generator.zero_grad()

        z = self._sample_noise(x0.shape)

        tilde_x = self.generator(x0, z)

        ot_loss = self.ot_loss(x0, tilde_x).mean()
        loss_gen = ot_loss + self.critic(tilde_x).mean()

        loss_gen.backward()

        self.optimizer_generator.step()

        return
    

    def _next_batch(self):
        """
        Safe batch iteration

        Returns:
        --------
        Dict
            Dict with p^t and p^s batches of images
        """
        try:
            batch = next(self.dataloader_iter)
        except:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)

        return batch
    

    def _norm_batch(self, batch: dict):
        """
        Extract images from batches and norm to [-1, 1]

        Args:
        -----
        batch : Dict
            Dict with p^t and p^s batches of images

        Returns:
        --------
        Tuple(torch.Tensor, torch.Tensor)
            Tuple with normed batches of images
        """
        x0, x1 = (batch['images_X'].to(self.device) * 2 - 1,
                    batch['images_Y'].to(self.device) * 2 - 1)
        
        return x0, x1
    
    def _get_objects(self):
        """
        Safe way to get objects: get batch -> norm images

        Returns:
        --------
        Tuple(torch.Tensor, torch.Tensor)
            Tuple with normed batches of images
        """
        batch = self._next_batch()
        x0, x1 = self._norm_batch(batch)

        return x0, x1
    

    def train_iter(self):
        """
        Make one training iteration over all batches for NOT model:

        ...

        Returns:
        --------
        dict
            Dictionary containing the loss value for the training step
        """

        self.optimizer_generator.zero_grad()
        self.optimizer_critic.zero_grad()


        ### Fit only critic
        for _ in range(self.config.model_params.critic_fit_iters):
            x0, x1 = self._get_objects()
            self.train_critic(x0, x1)

        ### One step fit for generator
        x0, x1 = self._get_objects()
        self.train_generator(x0)

        ### Forward pass
        with torch.no_grad():
            z = self._sample_noise(x0.shape)
            tilde_x = self.generator(x0, z)

            ot_part = self.ot_loss(x0, tilde_x)
            gan_part = (self.critic(tilde_x) - self.critic(x1))

            lagrangian_loss = ot_part.mean() - gan_part.mean()

        return {'Lagrangian': lagrangian_loss.item(),
                'OT Loss': ot_part.mean().item(),
                'GAN Loss': gan_part.mean().item()
                }


    def inference(self, batch: torch.Tensor):
        """
        Generate batch of images from input batch

        Args:
        -----
        batch : torch.Tensor
            Samples from p^t
        """

        batch = batch.to(self.device) * 2 - 1
        z = self._sample_noise(batch.shape).to(self.device)

        with torch.no_grad():
            out = self.generator(batch, z)

        return (out.cpu().clamp(-1, 1) + 1) / 2