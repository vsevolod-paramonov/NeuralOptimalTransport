from trainer.base_trainer import BaseTrainer
from hydra.utils import instantiate, get_class

from metrics import L2

from models import Discriminator, UNet

import torch


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
                              bilinear=True)
        
        self.critic = Discriminator(in_channels=3)


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

    def train_epoch(self):
        """
        Make one training iteration over all batches for NOT model:

        ...

        Returns:
        --------
        dict
            Dictionary containing the loss value for the training step
        """

        cumm_loss = 0.0

        self.optimizer_generator.zero_grad()
        self.optimizer_critic.zero_grad()

        for batch in self.dataloader:

            ### Sample from p^s and p^t => [0, 1] -> [-1, 1]
            x_0, x_1 = (batch['images_X'].to(self.device) * 2 - 1,
                        batch['images_Y'].to(self.device) * 2 - 1)

            B, C, H, W = x_0.shape

            ### Sample noise 
            z = torch.randn((B, 1, H, W)).to(self.device)

            ### Generator output
            with torch.no_grad():
                G_x = self.generator(x_0, z)

            ### OT loss + GAN-Loss
            loss = self.ot_loss(x_0, G_x).mean() + self.gan_loss(x_1, G_x).mean()

            loss.backward()

            ### Update generator and critic parameters
            self.optimizer_generator.step()
            self.optimizer_critic.step()

            cumm_loss += loss.item() * B

            break

        cumm_loss /= len(self.dataloader)

        return {'Loss': cumm_loss}
