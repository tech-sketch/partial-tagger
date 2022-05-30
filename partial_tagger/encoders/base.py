from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.nn import Module

from ..crf.nn import CRF


class BaseCRFEncoder(ABC, Module):
    def __init__(self, num_tags: int) -> None:
        super(BaseCRFEncoder, self).__init__()

        self.num_tags = num_tags
        self.crf = CRF(num_tags)

    @abstractmethod
    def forward(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass
