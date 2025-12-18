import torch
import random
from omegaconf import DictConfig
from data.dataset import DomenDataset
from torch.utils.data import DataLoader


class DomenLoader:
    """
    DataLoader class for loading domen images
    """

    def __init__(self, config: DictConfig, 
                       domen_X: DomenDataset,
                       domen_Y: DomenDataset,
                       ):
        """
        Initialize the DataLoader

        Args:
        -----
        config : DictConfig
            Configuration dictionary containing data parameters
        domen_X : DomenDataset
            Dataset with images from X domain
        domen_Y : DomenDataset
            Dataset with images from Y domain
        """

        self.config = config
  
        self.batch_size = 3
        self.shuffle = True

        self.domen_X = domen_X
        self.domen_Y = domen_Y

        self.len_X = len(self.domen_X)
        self.len_Y = len(self.domen_Y)
   
        self.max_len = max(self.len_X, self.len_Y)


    def __iter__(self):
        """
        Create an iterator for the DataLoader
        """

        self.indices_X = list(range(self.len_X))
        self.indices_Y = list(range(self.len_Y))

        if self.shuffle:
            random.shuffle(self.indices_X)
            random.shuffle(self.indices_Y)

        self.current = 0
        return self
    
    def __next__(self):
        """
        Get the next batch of data


        Returns:
        --------
        Dict
            Dictionary with batches of images from both domains
        """

        if self.current >= self.max_len:
            raise StopIteration

        batch_indices_X = self.indices_X[self.current:self.current + self.batch_size]
        batch_indices_Y = self.indices_Y[self.current:self.current + self.batch_size]

        images_X = [self.domen_X[idx] for idx in batch_indices_X]
        images_Y = [self.domen_Y[idx] for idx in batch_indices_Y]

        batch_X = torch.stack(images_X)
        batch_Y = torch.stack(images_Y)

        self.current += self.batch_size

        return {'images_X': batch_X,
                'images_Y': batch_Y}
    
    def __len__(self):
        """
        Return num of pairs
        """

        return self.max_len