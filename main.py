import hydra
from omegaconf import DictConfig
from trainer.not_trainer import NOTrainer
import torch
import numpy as np

seed = 52

@hydra.main(config_path="/Users/vsevolodparamonov/NeuralOptimalTransport/configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig):

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    trainer = NOTrainer(cfg)

    trainer.setup()

    trainer.training_loop()
    
    # if not cfg.inference.inference_mode:
    #     trainer.training_loop()

    # trainer.translate_text(cfg.inference.test_path, 'test_pred')

if __name__ == "__main__":
    main()