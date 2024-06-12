import torch 
import numpy as np
import pandas as pd
import src.utils.loading as lu
from src.datasets import MazeDataset
from typing import Iterable
from sklearn.manifold import MDS, Isomap

import random 
RANDOM_SEED = 29485
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class BaseContainer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Results:
    
    tasks = ['base', 'task0', 'task1', 'task2', 'task3', 'taskAll', 'alignBias', 'bottomBias']
    capacities = ['high', 'low']

    def __init__(
            self,
            path2data: str,
            models: dict,
            idxs: Iterable[int],
            ) -> None:

        self.models = models
        self.idxs = idxs

        self.X, self.y = self._get_Xy(path2data)

    def get_sample(self, idx: int|None =None):
        if idx is None:
            idx = np.random.choice(self.idxs)

        encodings = {}
        for task in self.tasks:
            encodings[task] = {}
            for c in self.capacities:
                model_container = self.__getattribute__(task).__getattribute__(c)
                encodings[task][c] = {
                    'Z': model_container.Z[idx],
                    'y': model_container.y[idx],
                    'emb': model_container.emb[idx]
                }
                
        return self.X[idx], self.y[idx], encodings
        
    def reindex(self, idxs):
        self.X = self.X[idxs]
        self.y = self.y[idxs]
        for task in self.tasks:
            for c in self.capacities:
                model_container = self.__getattribute__(task).__getattribute__(c)
                model_container.Z = model_container.Z[idxs]
                model_container.y = model_container.y[idxs]
                model_container.emb = model_container.emb[idxs]
        return None

    def load(self, **MDS_kwargs):
        for task in self.tasks:
            task_container = BaseContainer()
            for c in self.capacities:
                model_container = BaseContainer()
                Z, y = self._get_encodings(task, c)
                emb = self._get_embeddings(Z, **MDS_kwargs)
                
                model_container.__setattr__('Z', Z)
                model_container.__setattr__('y', y)
                model_container.__setattr__('emb', emb)

                task_container.__setattr__(c, model_container)
            self.__setattr__(task, task_container)
        return None

    def _get_Xy(self, path2data):
        df = pd.read_csv(path2data, header=None).values
        dataset = MazeDataset(df)
        X, y = dataset[self.idxs]
        X = X.detach().numpy()
        y = y.detach().numpy()
        return X, y
    
    def _get_labels(self, task):

        def get_new_labels_task0(label):
            if label in [0,1]:
                return 0
            elif label in [2,3,4]:
                return 1
            elif label in [5,6,7]:
                return 2
            elif label in [8,9,10]:
                return 3
            elif label in [11,12]:
                return 4

        y = self.y.copy()

        if task=='bottomBias':
            condition = y[:,1] <= 6
        elif task=='alignBias':
            condition = y[:,0] == y[:,1]
        if (task=='task0') or (task=='taskAll'):
            labels = np.vectorize(get_new_labels_task0)(y)
            y = labels[:,0]*5 + labels[:,1]
        elif task=='task1':
            condition = y[:,0] == y[:,1]         
        elif task=='task2':
            condition = y[:,0] <= y[:,1]
        elif task=='task3':
            condition = (y[:,0] < 6) | ((y[:,0] >= 6) & (y[:,1] >= 6))
        elif task=='base':
            pass

        try:
            y = np.where(condition, 1, 0)
        except:
            pass

        return y
    
    def _get_encodings(self, task, capacity):
        model_dict = self.models[task][capacity]
        config  = lu.load_config(model_dict['config'])
        model = lu.load_model_from_checkpoint(model_dict['ckpt'], config)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        model.to(device)
        model.eval()

        X = self.X.copy().reshape(-1, 1, 13, 13)
        X = torch.tensor(X, dtype=torch.float32, device=device)

        mu, logvar = model.encode(X)
        Z = mu.cpu().detach().numpy()

        return Z, self._get_labels(task)

    def _get_embeddings(self, Z, **kwargs):
        #mds = Isomap(**kwargs)
        mds = MDS(**kwargs)
        emb = mds.fit_transform(Z)
        return emb
