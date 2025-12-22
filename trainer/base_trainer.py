import os
import shutil
import torch
import numpy as np
from torchvision.transforms import v2
from PIL import Image

from data import DomenDataset, DomenLoader
from utils import before_after_OT, save_images
from metrics import (L2,
                     SSIMMetric,
                     PSNRMetric,
                     LPIPSMetric,
                     FID)


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

        if self.config.validation.metrics is not None:
            self.generate_validation()
            self.setup_metrics()

        self.logwriter._log_custom_message('Models are setted up!')

    def setup_logger(self):

        exp_logs_dir = os.path.join(self.experiment_dir, 'logs')
        os.makedirs(exp_logs_dir, exist_ok=True)

        self.logwriter = logwriter.Logger(exp_logs_dir)

    def setup_experiment_dir(self):
        self.experiment_dir = os.path.join(os.getcwd(), self.config.experiments.experiment_dir, self.config.experiments.exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def setup_datasets(self):
        self.source_dataset = DomenDataset(self.config.data.source_images, self.config.data.image_size)
        self.target_dataset = DomenDataset(self.config.data.target_images, self.config.data.image_size)
        
    def setup_dataloaders(self):
        self.dataloader = DomenLoader(self.config, self.source_dataset, self.target_dataset)

    def setup_metrics(self):
        self.metrics = dict()
        
        for metric_name in self.config.validation.metrics:
            metric_class = globals().get(metric_name)

            if metric_name == 'LPIPSMetric':
                metric_class = metric_class(net='alex', device=self.device)
            if metric_name == 'FID':
                metric_class = metric_class(device=self.device)

            self.metrics[metric_name] = metric_class

    def training_loop(self):
        self.to_train()

        self.logwriter._log_custom_message('Started fitting')
        self.iter = 0

        self.dataloader_iter = iter(self.dataloader)

        for i in range(1, self.config.training.num_iters + 1):

            train_loss = self.train_iter()

            ### Infer images from 'inference' folder
            if i % self.config.training.sampling_iters == 0 and i > 0:
                self.logwriter._log_custom_message('Sampling images')
                self.generate_images()

            ### Save models checkpoints
            if i % self.config.training.checkpoint_step == 0 and i > 0:
                self.save_checkpoint()
                self.logwriter._log_custom_message('Checkpoint was saved!')

            ### Calculate metrics
            if self.config.validation.metrics is not None and i % self.config.training.metrics_step == 0 and i > 0:
                self.logwriter._log_custom_message('Calculate metrics...')
                metrics = self.run_validation()
                train_loss.update(metrics)

            self.logwriter._log_metrics(train_loss, i)

            self.iter += 1

        self.logwriter._log_custom_message('Fitting ended')

        
    def generate_validation(self):
        """
        Prepare samples for validation
        """

        val_path = self.config.validation.val_path
        source_path = os.path.join(val_path, 'source')
        target_path = os.path.join(val_path, 'target')

        os.makedirs(val_path, exist_ok=True)

        if os.path.exists(source_path) == True:
            shutil.rmtree(source_path)
        os.makedirs(source_path, exist_ok=True)

        if os.path.exists(target_path) == True:
            shutil.rmtree(target_path)
        os.makedirs(target_path, exist_ok=True)
        
        np.random.seed(self.config.experiments.seed)
        random_images = np.random.choice(range(len(self.source_dataset)), 
                                         size=self.config.validation.val_size,
                                         replace=False)

        batch_val_images = torch.stack([
            self.source_dataset[i] for i in random_images
        ])

        save_images(source_path, batch_val_images)
       

    def run_validation(self):
        """
        Run validation and calculate metrics
        """
        self.to_eval()

        ### Setup paths
        val_path = self.config.validation.val_path
        source_path = os.path.join(val_path, 'source')
        target_path = os.path.join(val_path, 'target')

        ### Get source data
        to_tensor = v2.Compose([
            v2.Resize((self.config.data.image_size, self.config.data.image_size)),
            v2.CenterCrop(self.config.data.image_size),  
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        source_data = torch.stack([
            to_tensor(
            Image.open(os.path.join(source_path, file_name)).convert('RGB')
            )
            for file_name in sorted(os.listdir(source_path))
            ])
        
        ### Run sampling
        target_data = self.inference(source_data)

        ### Save images for FID calculation
        save_images(target_path, target_data)

        ### Run metrics calculation
        metric_results = dict()

        for metric_name, metric_class in self.metrics.items():
            if metric_name == 'FID':
                metric_value = metric_class(source_path, target_path)
            elif metric_name == 'LPIPSMetric':
                metric_value = metric_class(source_data, target_data)
            else:
                metric_value = metric_class()(source_data, target_data)

            metric_results[metric_name] = metric_value
            self.logwriter._log_custom_message(f'{metric_name} was calculated')

        return metric_results

    
    def generate_images(self):
        """
        Infer images from /inference/source folder
        """
        self.to_eval()

        assert len(os.listdir(self.config.sampling.source_path)) > 0, 'Pass images to sample!'

        if os.path.exists(self.config.sampling.target_path):
            shutil.rmtree(self.config.sampling.target_path)

        ### Clear images from previous sampling
        os.makedirs(self.config.sampling.target_path)

        to_tensor = v2.Compose([
            v2.Resize((self.config.data.image_size, self.config.data.image_size)),
            v2.CenterCrop(self.config.data.image_size),  
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

        before_after_OT(self.config.sampling.target_path, samples, output)

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