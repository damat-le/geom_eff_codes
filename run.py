from __future__ import annotations
import os
import yaml
import argparse
from pathlib import Path
from torch import set_float32_matmul_precision
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.models import models
from src.datasets import MazeDataModule
from src.experiments import MazeExperiment

set_float32_matmul_precision('medium')

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
    name=config['model_params']['name'],
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

data = MazeDataModule(
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


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/LatentSpace").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)