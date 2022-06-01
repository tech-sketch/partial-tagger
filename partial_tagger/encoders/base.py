from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.nn import Module

from ..crf.nn import CRF


class BaseCRFEncoder(ABC, Module):
    """Base class of all encoders with CRF.

    Args:
        num_tags: An integer representing the number of tags.
    """

    def __init__(self, num_tags: int) -> None:
        super(BaseCRFEncoder, self).__init__()

        self.num_tags = num_tags
        self.crf = CRF(num_tags)

    @abstractmethod
    def forward(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes log potentials from the given embeddings.

        Args:
            embeddings: A [batch_size, sequence_length, embedding_size] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, num_tag, num_tags] float tensor
            representing log potentials.
        """
        pass
