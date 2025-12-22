import hydra
from omegaconf import DictConfig
from trainer.not_trainer import NOTrainer
import torch
import numpy as np


@hydra.main(config_path="/Users/vsevolodparamonov/NeuralOptimalTransport/configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig):

    ### Fix seed
    np.random.seed(cfg.experiments.seed)
    torch.random.manual_seed(cfg.experiments.seed)
    
    ### Setup trainer
    trainer = NOTrainer(cfg)
    trainer.setup()

    ### Fit if required
    if cfg.training.fit == True:
        trainer.training_loop()

    ### Otherwise inference
    else:
        assert cfg.training.fit != True and (cfg.checkpoint.generator_checkpoint is not None 
                and cfg.checkpoint.critic_checkpoint is not None), 'No checkpoint was found'

    ### Run inference
    trainer.generate_images()

if __name__ == "__main__":
    main()