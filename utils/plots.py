import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import torchvision
from omegaconf import DictConfig


def before_after_OT(path: str,
                    input_images: torch.Tensor, 
                    output_images: torch.Tensor):
    """
    Plot source images and generation
    """
    
    fig, ax = plt.subplots(2, input_images.shape[0], figsize=(20, 10))

    for i in range(input_images.shape[0]):

        img1 = input_images[i, :, :, :]
        img2 = output_images[i, :, :, :]

        ax[0][i].imshow(img1.permute(1,2,0).detach().numpy())
        ax[1][i].imshow(img2.permute(1,2,0).detach().numpy())

        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])

        ax[0][0].set_ylabel(r'$X_0 \sim p^S$', fontsize=15)
        ax[1][0].set_ylabel(r'$G_{\theta}(X_0, Z)$', fontsize=15)

    plt.savefig(os.path.join(path, 'output.pdf'), 
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)
    

def save_images(path: str,
                output_images: torch.Tensor):
    """
    Save each image from batch separately
    """

    for i in range(output_images.shape[0]):
        torchvision.utils.save_image(output_images[i], f"{path}/sample{i+1}.png")

