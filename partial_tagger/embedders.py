from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import torch
from torch.nn import Module

EmbedderInputs = TypeVar("EmbedderInputs")


class BaseEmbedder(ABC, Generic[EmbedderInputs], Module):
    @abstractmethod
    def forward(
        self, inputs: EmbedderInputs, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass
