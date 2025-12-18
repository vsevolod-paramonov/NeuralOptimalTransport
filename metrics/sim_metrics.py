import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import numpy as np


class L2:
    """
    L2 distance between img1 and img2
    L2 = 1/N ||img1 - img2||^2
    """

    def __call__(self, img1: torch.Tensor, 
                       img2: torch.Tensor):

        return F.mse_loss(img1, img2).item()


class PSNRMetric:
    """
    Peak Signal-to-Noise Ratio
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """

    def __call__(self, img1: torch.Tensor, 
                       img2: torch.Tensor):
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        B = img1_np.shape[0]
        
        psnr_list = [
            compare_psnr(
                img1_np[i].transpose(1, 2, 0),
                img2_np[i].transpose(1, 2, 0),
                data_range=1.0
            )
            for i in range(B)
        ]
        return np.mean(psnr_list)

class SSIMMetric:
    """
    Structural similarity index measure
    https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    """
    def __call__(self, img1: torch.Tensor, 
                       img2: torch.Tensor):
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        
        B = img1_np.shape[0]
        
        ssim_list = [
            compare_ssim(
                img1_np[i].transpose(1, 2, 0),
                img2_np[i].transpose(1, 2, 0),
                data_range=1.0, channel_axis=-1
            )
            for i in range(B)
        ]
        
        return np.mean(ssim_list)

class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity
    https://distancia.readthedocs.io/en/latest/LPIPS.html
    """
    def __init__(self, net='alex', device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def __call__(self, img1: torch.Tensor, 
                       img2: torch.Tensor):
        img1 = img1.to(self.device) * 2 - 1
        img2 = img2.to(self.device) * 2 - 1
        
        with torch.no_grad():
            dists = self.model(img1, img2)
            
        return dists.flatten().cpu().numpy()
