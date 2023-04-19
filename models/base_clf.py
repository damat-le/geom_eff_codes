import torch
import numpy as np
from torch import nn
from torch import Tensor
from sklearn.metrics import f1_score

class LatentClassifier(nn.Module):

    def __init__(
        self, 
        latent_dim: int = 5, 
        clf_task_num: int = 0,
        **kwargs):

        super().__init__()
        
        clf_tasks_dict = {
            0 : self.map_label2idx_task0,
            1 : self.map_label2idx_task1,
            2 : self.map_label2idx_task2,
            3 : self.map_label2idx_task3,
        }

        self.clf_task_num = clf_task_num
        self.get_task_labels = clf_tasks_dict[clf_task_num]

        # Build Classifier
        if self.clf_task_num == 0:
            self.clf = nn.Sequential(
                nn.Linear(latent_dim, 1500),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.25),
                nn.Linear(1500, 25),
                nn.Softmax(dim=1)
            )
        elif self.clf_task_num in [1, 2, 3]:
            self.clf = nn.Sequential(
                nn.Linear(latent_dim, 1024),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )

    @staticmethod
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
    
    def map_label2idx_task0(self, labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        labels = np.vectorize(self.get_new_labels_task0)(labels)

        if len(labels.shape) == 1:
            res = labels[0]*5 + labels[1]
            return torch.tensor(res, device=device, dtype=torch.long)
        else:
            res = labels[:,0]*5 + labels[:,1]
            return torch.tensor(res, device=device, dtype=torch.long)
        
    @staticmethod
    def map_label2idx_task1(labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        if len(labels.shape) == 1:
            res = np.where(labels[0]==labels[1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)
        else:
            res = np.where(labels[:,0]==labels[:,1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)

    @staticmethod
    def map_label2idx_task2(labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        if len(labels.shape) == 1:
            res = np.where(labels[0]<=labels[1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)
        else:
            res = np.where(labels[:,0]<=labels[:,1],1,0).reshape(-1,1)
            return torch.tensor(res, device=device, dtype=torch.float)

    @staticmethod
    def map_label2idx_task3(labels: Tensor) -> Tensor:
        device = labels.device
        labels = labels.cpu()
        if len(labels.shape) == 1:
            condition = (labels[0] < 6) | ((labels[0] >= 6) & (labels[1] >= 6))
        else:
            condition = (labels[:,0] < 6) | ((labels[:,0] >= 6) & (labels[:,1] >= 6)) 
        res = np.where(condition,1,0).reshape(-1,1)
        return torch.tensor(res, device=device, dtype=torch.float)
    
    def forward(self, z: Tensor, **kwargs) -> Tensor:
        labels = kwargs['labels']
        task_labels = self.get_task_labels(labels)
        preds = self.clf(z)
        return preds, task_labels
    
    def loss_function(self, *args, **kwargs):
        preds, task_labels = args
        if self.clf_task_num == 0:
            loss = nn.functional.cross_entropy(preds, task_labels)
            pred_labels = torch.argmax(preds, dim=1)
            f1 = f1_score(task_labels.cpu(), pred_labels.cpu(), average='macro')
        elif self.clf_task_num in [1, 2, 3]:
            loss = nn.functional.binary_cross_entropy(preds, task_labels)
            pred_labels = np.where(preds.cpu()>=.5, 1, 0)
            f1 = f1_score(task_labels.cpu(), pred_labels, zero_division=0)

        return {'loss': loss, 'f1_score': f1}
