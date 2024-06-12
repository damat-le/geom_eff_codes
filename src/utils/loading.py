import yaml
import torch
import pandas as pd 

from src.datasets import MazeDataset
from src.models import models

def load_model_from_checkpoint(ckp_path, config):
    # load pytorch model checkpoint from file
    model_type = config['model_params']['name']
    model = models[model_type](**config['model_params'])
    if not torch.cuda.is_available():
        map_location=torch.device('cpu')
    else:
        map_location=torch.device('cuda')
    state = torch.load(ckp_path, map_location=map_location)['state_dict']
    # remove 'model.' prefix from state dict keys
    state = {k[6:]: v for k, v in state.items()} 
    model.load_state_dict(state, strict=0) 
    model = model.eval()
    return model

def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def load_dataset(config):
    path2imgs = config['data_params']['data_path']
    #path2imgs = '../' + path2imgs
    imgs = pd.read_csv(path2imgs, header=None).values
    dataset = MazeDataset(imgs, config["data_params"]["num_labels"])
    return dataset

def load_experiment(config_path, ckp_path, data_path=None):
    config = load_config(config_path)
    model = load_model_from_checkpoint(ckp_path, config)
    if data_path is not None:
        config['data_params']['data_path'] = data_path
    dataset = load_dataset(config)
    return dataset, model, config
