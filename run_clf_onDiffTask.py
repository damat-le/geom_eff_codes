from __future__ import annotations
import pandas as pd
import numpy as np
import torch

from src.datasets import MazeDataset, MazeDataModule
from src.experiments import MazeExperiment
from src.models import BetaVAE_CLF


class MazeDatasetEncoded(MazeDataset):

    def get_size(self, batch):
        if len(batch.shape) == 1:
            return (self.img_size,)
        else:
            return (batch.shape[0], self.img_size)

class VaeLatentDataset(MazeDataModule):
    def __init__(
            self, 
            data_path: str, 
            num_labels: int,
            model4latentrepr: str,
            train_val_test_split: list[int], 
            train_batch_size: int = 8, 
            val_batch_size: int = 8, 
            test_batch_size: int = 8, 
            patch_size: int | tuple = (256, 256),
            num_workers: int = 0, 
            pin_memory: bool = False, 
            **kwargs
        ):
        super().__init__(data_path, num_labels, train_val_test_split, train_batch_size, val_batch_size, test_batch_size, patch_size, num_workers, pin_memory, **kwargs)
        self.model4latentrepr = model4latentrepr
    
    def setup(self, stage: str | None = None) -> None:
        imgs = pd.read_csv(self.data_dir, header=None).values
        dataset = MazeDataset(imgs, num_labels=self.num_labels)

        # load model for latent repr from checkpoint
        model_config = os.path.join(self.model4latentrepr, 'hparams.yaml')
        with open(model_config, 'r') as file:
            try:
                model_config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        model_ckpt = os.path.join(self.model4latentrepr, 'checkpoints/last.ckpt')
        state = torch.load(model_ckpt)['state_dict']
        state = {k[6:]: v for k, v in state.items()} 
        model4latentrepr = BetaVAE_CLF(
            **model_config['model_params']
        )
        model4latentrepr.load_state_dict(state_dict=state, strict=0)
        model4latentrepr.eval()

        # get latent repr
        X, y = dataset[:]
        mu, logvar = model4latentrepr.encode(X)
        Z = model4latentrepr.reparameterize(mu, logvar)
        
        Z = Z.detach().numpy()
        y = y.detach().numpy()

        latent_data = np.concatenate((Z, y), axis=1)

        idx1 = self.train_val_test_split[0]
        idx2 = self.train_val_test_split[1]
        # mazes_cont50k_noExit_biasedL90R10_1
        self.train_dataset = MazeDatasetEncoded(latent_data[:idx1], num_labels=self.num_labels, img_size=Z.shape[1])
        self.val_dataset = MazeDatasetEncoded(latent_data[idx1:idx2], num_labels=self.num_labels, img_size=Z.shape[1])
        self.test_dataset = MazeDatasetEncoded(latent_data[idx2:], num_labels=self.num_labels, img_size=Z.shape[1])

class MazeExperiment(MazeExperiment):
    def on_validation_end(self) -> None:
        pass

if __name__ == '__main__':
    import argparse
    import os
    import yaml
    from torch import set_float32_matmul_precision
    from lightning_fabric.utilities.seed import seed_everything
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.strategies import DDPStrategy
    
    from src.models import models

    set_float32_matmul_precision('medium')


    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')
    parser.add_argument('--task_to_train',
                        dest="task_to_train",
                        help =  'task to train',
                        type=int,
                        default=0)
    parser.add_argument('--task_to_skip',
                        dest="task_to_skip",
                        help =  'task to skip',
                        type=int,
                        default=0)
    
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    task_to_train = args.task_to_train 
    task_to_skip = args.task_to_skip
    config['model_params']['clf_task_num'] = task_to_train


    tb_logger =  TensorBoardLogger(
        save_dir= os.path.join(
            config['logging_params']['save_dir'],
            f"ReprTask{task_to_skip}",
            f'clf_task{task_to_train}'
            ),
        name=config['logging_params']['name'],
    )
    tb_logger.log_hyperparams(config)

    # For reproducibility
    seed_everything(
        config['exp_params']['manual_seed'], 
        True
        )

    model = models[config['model_params']['name']](**config['model_params'])

    experiment = MazeExperiment(
        model,
        config['exp_params']
        )

    data = VaeLatentDataset(
        **config["data_params"], 
        pin_memory=len(config['trainer_params']['devices']) != 0
        )

    data.setup()
    runner = Trainer(
        logger=tb_logger,
        #log_every_n_steps=25,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2, 
                dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                monitor= "val_loss",
                save_last= True),
            ],
        strategy=DDPStrategy(find_unused_parameters=False),
        **config['trainer_params']
        )

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)
    print("======= Done =======")
