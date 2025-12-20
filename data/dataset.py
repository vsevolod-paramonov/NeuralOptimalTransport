import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DomenDataset(Dataset):
    """
    Dataset class for loading domen images
    """

    def __init__(self, root_dir: str, 
                       img_size: int,
                       transform: v2.Compose = None
                       ):
        """
        Initialize the dataset

        Args:
        -----
        root_dir : str
            Path to the directory containing the dataset
        transform : v2.Compose, optional
            Transform to be applied on a sample (default: None)
        """
        
        self.root_dir = root_dir
        self.transform = transform if transform is not None else v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.CenterCrop(img_size),  
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.samples = [
            os.path.join(self.root_dir, file_name)
            for file_name in os.listdir(self.root_dir)
        ]


    def __getitem__(self, index: int):
        """
        Get image by index


        Args:
        -----
        index : int
            Index of the sample to retrieve


        Returns:
        -------
        tuple
            Image from certain domain
        """

        img_path = self.samples[index]
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        return image


    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns:
        --------
        int
            Number of samples in the dataset
        """

        return len(self.samples)