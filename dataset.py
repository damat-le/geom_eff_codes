import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST
import zipfile
import pandas as pd
from torchvision.transforms import Lambda
import numpy as np


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

class MazeDataset(Dataset):

    def __init__(self, data, num_labels=2, img_size=13):
        self.img_size = img_size
        self.num_labels = num_labels
        self.imgs = data
        self.transform = Lambda(lambda x: torch.Tensor(x.reshape(*self.get_size(x))))

    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        img = self.imgs[idx, :-self.num_labels]
        label = self.imgs[idx, -self.num_labels:]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

    def get_size(self, batch):
        if len(batch.shape) == 1:
            return (1, self.img_size, self.img_size)
        else:
            return (batch.shape[0], 1, self.img_size, self.img_size)
    
    def sample(self, label_up, label_lc, n=1, idx=None):
        if idx is not None:
            img = self.imgs[idx:idx+1, :-self.num_labels]
            label = self.imgs[idx:idx+1, -self.num_labels:]
            return [idx], img, label
        if label_up is None:
            label_up = np.random.randint(0, 13)
        if label_lc is None:
            label_lc = np.random.randint(0, 13)
        condition = (self.imgs[:, -2] == label_up) & (self.imgs[:, -1] == label_lc)
        idxs = np.where(condition)
        row_idxs = np.unique(idxs[0])
        selected_row_idxs = np.random.choice(row_idxs, n)
        imgs = self.imgs[selected_row_idxs, :-self.num_labels]
        labels = self.imgs[selected_row_idxs, -self.num_labels:]
        return selected_row_idxs, imgs, labels
    
    def _check_integrity(self) -> bool:
        return True


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

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
        self.train_val_test_split = train_val_test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        # train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.CenterCrop(148),
        #                                       transforms.Resize(self.patch_size),
        #                                       transforms.ToTensor(),])
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size),
        #                                     transforms.ToTensor(),])
        
        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )
        
        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )

#       ========================= MNIST ======================================
        # data = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
        # m = len(data)
        # self.train_dataset, self.val_dataset = random_split(data, [int(m*.8), int(m*.2)])
        # self.test_dataset = MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

#       ========================= MazeDataset ======================================
        imgs = pd.read_csv(self.data_dir, header=None).values
        
        idx1 = self.train_val_test_split[0]
        idx2 = self.train_val_test_split[1]
        # mazes_cont50k_noExit_biasedL90R10_1
        self.train_dataset = MazeDataset(imgs[:idx1], num_labels=self.num_labels)
        self.val_dataset = MazeDataset(imgs[idx1:idx2], num_labels=self.num_labels)
        self.test_dataset = MazeDataset(imgs[idx2:], num_labels=self.num_labels)

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
     