import os
import torch

from data import DomenDataset, DomenLoader

from abc import abstractmethod
from logger import logwriter

domen_X = '/Users/vsevolodparamonov/Downloads/img_align_celeba'
domen_Y = '/Users/vsevolodparamonov/Downloads/cropped'
experiment_dir = '/Users/vsevolodparamonov/NeuralOptimalTransport/experiments'
exp_name = 'default'
epoch_num = 10


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cpu'

    
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
        self.experiment_dir = os.path.join(os.getcwd(), experiment_dir, exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def setup_datasets(self):
        self.source_dataset = DomenDataset(domen_X)
        self.target_dataset = DomenDataset(domen_Y)
        
    def setup_dataloaders(self):
        self.dataloader = DomenLoader(..., self.source_dataset, self.target_dataset)

    def training_loop(self):
        # self.to_train()

        self.logwriter._log_custom_message('Started fitting')
        self.iter = 0
        self.cur_epoch = 0

        for i in range(epoch_num):

            self.cur_epoch = i
            train_loss = self.train_epoch()

            # if i % self.config.train.checkpoint_step == 0 and i > 0:
            #     self.save_checkpoint()


            self.logwriter._log_metrics(train_loss, i)

        self.logwriter._log_custom_message('Fitting ended')

            
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