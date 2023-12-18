from __future__ import annotations

from torch import TensorType
from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: TensorType) -> list[TensorType]:
        raise NotImplementedError

    def decode(self, input: TensorType) -> any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> TensorType:
        raise NotImplementedError

    def generate(self, x: TensorType, **kwargs) -> TensorType:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: TensorType) -> TensorType:
        pass

    @abstractmethod
    def loss_function(self, *inputs: any, **kwargs) -> TensorType:
        pass
