import os
import shutil
import torch
from torchvision.transforms import v2
from PIL import Image

from data import DomenDataset, DomenLoader
from utils import before_after_OT

from abc import abstractmethod
from logger import logwriter

domen_X = '/Users/vsevolodparamonov/Downloads/img_align_celeba'
domen_Y = '/Users/vsevolodparamonov/Downloads/cropped'
experiment_dir = '/Users/vsevolodparamonov/NeuralOptimalTransport/experiments'
exp_name = 'default'
source_path = '/Users/vsevolodparamonov/NeuralOptimalTransport/inference/source'
target_path = '/Users/vsevolodparamonov/NeuralOptimalTransport/inference/target'
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

            self.generate_images()

            self.logwriter._log_metrics(train_loss, i)

            break

        self.logwriter._log_custom_message('Fitting ended')

    
    def generate_images(self):

        assert len(os.listdir(source_path)) > 0, 'Pass images to sample!'

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        ### Clear images from previous sampling
        os.makedirs(target_path)

        to_tensor = v2.Compose([
                                v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True)
                                ])
        
        samples = torch.stack([
            to_tensor(
            Image.open(os.path.join(source_path, file_name)).convert('RGB')
            )
            for file_name in os.listdir(source_path)
            ])
        
        print(samples.shape)
        
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