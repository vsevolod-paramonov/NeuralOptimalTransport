import matplotlib.pyplot as plt
import torch
import numpy as np

path = '/Users/vsevolodparamonov/NeuralOptimalTransport/inference/target/output.pdf'

def before_after_OT(input_images: torch.Tensor, 
                    output_images: torch.Tensor):
    """
    Plot source images and generation
    """
    
    fig, ax = plt.subplots(2, input_images.shape[0], figsize=(20, 10))

    for i in range(input_images.shape[0]):

        img1 = input_images[i, :, :, :]
        img2 = output_images[i, :, :, :]

        img1 = (img1.clamp(-1, 1) + 1) / 2
        img2 = (img1.clamp(-1, 1) + 1) / 2

        ax[0][i].imshow(img1.permute(1,2,0).detach().numpy())
        ax[1][i].imshow(img2.permute(1,2,0).detach().numpy())

        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])

        ax[0][0].set_ylabel(r'$X_0 \sim p^S$', fontsize=15)
        ax[1][0].set_ylabel(r'$G_{\theta}(X_0, Z)$', fontsize=15)

    plt.savefig(path, 
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)