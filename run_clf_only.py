from typing import List, Union, Sequence, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
from dataset import MazeDataset
import pandas as pd
import numpy as np

class MazeDatasetEncoded(MazeDataset):

    def get_size(self, batch):
        if len(batch.shape) == 1:
            return (self.img_size,)
        else:
            return (batch.shape[0], self.img_size)

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        num_labels: int, # l'ho inserito io
        img_size: int,
        train_val_test_split: List[int], #l'ho inserito io
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.num_labels = num_labels
        self.img_size = img_size
        self.train_val_test_split = train_val_test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        imgs = pd.read_csv(self.data_dir, header=None).values
        
        idx1 = self.train_val_test_split[0]
        idx2 = self.train_val_test_split[1]
        # mazes_cont50k_noExit_biasedL90R10_1
        self.train_dataset = MazeDatasetEncoded(imgs[:idx1], num_labels=self.num_labels, img_size=self.img_size)
        self.val_dataset = MazeDatasetEncoded(imgs[idx1:idx2], num_labels=self.num_labels, img_size=self.img_size)
        self.test_dataset = MazeDatasetEncoded(imgs[idx2:], num_labels=self.num_labels, img_size=self.img_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            #collate_fn=lambda data: tuple(data)
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            #collate_fn=lambda data: tuple(data)
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            #collate_fn=lambda data: tuple(data)
        )
     
class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: Module, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(
            *results,
            optimizer_idx=optimizer_idx,
            batch_idx = batch_idx
            )
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(
            *results,
            optimizer_idx = optimizer_idx,
            batch_idx = batch_idx
            )
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
            )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model,self.params['submodel']).parameters(),
                    lr=self.params['LR_2']
                    )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0],
                    gamma = self.params['scheduler_gamma']
                    )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1],
                            gamma = self.params['scheduler_gamma_2']
                            )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

if __name__ == '__main__':
    import argparse
    import os
    import yaml
    from pathlib import Path
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.plugins import DDPPlugin
    from models import *

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    tb_logger =  TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
    )
    tb_logger.log_hyperparams(config)

    # For reproducibility
    seed_everything(
        config['exp_params']['manual_seed'], 
        True
        )

    model = vae_models[config['model_params']['name']](**config['model_params'])

    experiment = VAEXperiment(
        model,
        config['exp_params']
        )

    data = VAEDataset(
        **config["data_params"], 
        pin_memory=len(config['trainer_params']['gpus']) != 0
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
        strategy=DDPPlugin(find_unused_parameters=False),
        **config['trainer_params']
        )

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)