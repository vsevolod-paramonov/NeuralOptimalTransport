import os
import tqdm
import torch
# from mtdatasets.dataset import LanguageDataset, TranslatorDataset
# from mtdatasets.dataloader import TranslatorDataLoader
from abc import abstractmethod
from logger import logwriter

import sys

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.exp.device 

    
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
        self.experiment_dir = os.path.join(os.getcwd(), self.config.exp.exp_dir, self.config.exp.exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def setup_datasets(self):
        pass
        
    def setup_dataloaders(self):
        self.train_loader = TranslatorDataLoader(self.train_dataset, batch_size=self.config.train.batch_size, shuffle=True)
        self.val_loader = TranslatorDataLoader(self.val_dataset, batch_size=self.config.train.batch_size, shuffle=False)

    def training_loop(self):
        self.to_train()

        self.logwriter._log_custom_message('Started fitting')
        self.iter = 0
        self.cur_epoch = 0

        for i in range(self.config.train.epoch_num):

            self.cur_epoch = i
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()

            train_loss.update(val_loss)

            if i % self.config.train.checkpoint_step == 0 and i > 0:
                self.save_checkpoint()


            self.logwriter._log_metrics(train_loss, i)

        self.logwriter._log_custom_message('Fitting ended')

    @torch.inference_mode()
    def inf(...):
        pass


            
    @abstractmethod
    def inference(self, seq):
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