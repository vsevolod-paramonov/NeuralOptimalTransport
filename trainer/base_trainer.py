import os
import shutil
import torch
from torchvision.transforms import v2
from PIL import Image

from data import DomenDataset, DomenLoader
from utils import before_after_OT

from abc import abstractmethod
from logger import logwriter


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self.config.training.device

    
    def setup(self):
        self.setup_experiment_dir()

        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()

        self.logwriter._log_custom_message('Datasets and DataLoaders are setted up!')

        self.setup_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_losses()

        self.logwriter._log_custom_message('Models are setted up!')

    def setup_logger(self):

        exp_logs_dir = os.path.join(self.experiment_dir, 'logs')
        os.makedirs(exp_logs_dir, exist_ok=True)

        self.logwriter = logwriter.Logger(exp_logs_dir)

    def setup_experiment_dir(self):
        self.experiment_dir = os.path.join(os.getcwd(), self.config.experiments.experiment_dir, self.config.experiments.exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def setup_datasets(self):
        self.source_dataset = DomenDataset(self.config.data.source_images)
        self.target_dataset = DomenDataset(self.config.data.target_images)
        
    def setup_dataloaders(self):
        self.dataloader = DomenLoader(self.config, self.source_dataset, self.target_dataset)

    def training_loop(self):
        self.to_train()

        self.logwriter._log_custom_message('Started fitting')
        self.iter = 0

        self.dataloader_iter = iter(self.dataloader)

        for i in range(1, self.config.training.num_iters + 1):

            train_loss = self.train_iter()

            # if i % self.config.train.checkpoint_step == 0 and i > 0:
            #     self.save_checkpoint()

            if i % self.config.training.sampling_iters == 0 and i > 0:
                self.logwriter._log_custom_message('Sampling images')
                self.generate_images()

            self.logwriter._log_metrics(train_loss, i)

            self.iter += 1

        self.logwriter._log_custom_message('Fitting ended')

    
    def generate_images(self):

        self.to_eval()

        assert len(os.listdir(self.config.sampling.source_path)) > 0, 'Pass images to sample!'

        if os.path.exists(self.config.sampling.target_path):
            shutil.rmtree(self.config.sampling.target_path)

        ### Clear images from previous sampling
        os.makedirs(self.config.sampling.target_path)

        to_tensor = v2.Compose([
            v2.Resize((178, 178)),
            v2.CenterCrop(178),  
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        samples = torch.stack([
            to_tensor(
            Image.open(os.path.join(self.config.sampling.source_path, file_name)).convert('RGB')
            )
            for file_name in os.listdir(self.config.sampling.source_path)
            ])
        
        output = self.inference(samples)

        before_after_OT(samples, output)

        return 


    @abstractmethod
    def inference(self, batch):
        pass

    @abstractmethod
    def setup_models(self):
        pass

    @abstractmethod
    def setup_optimizers(self):
        pass

    @abstractmethod
    def setup_schedulers(self):
        pass

    @abstractmethod
    def setup_losses(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def make_example(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def to_train(self):
        pass

    @abstractmethod
    def to_eval(self):
        pass