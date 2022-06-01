from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import torch
from torch.nn import Module

EmbedderInputs = TypeVar("EmbedderInputs")


class BaseEmbedder(ABC, Generic[EmbedderInputs], Module):
    """Base class of all embedders."""

    @abstractmethod
    def forward(
        self, inputs: EmbedderInputs, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes embeddings from the given inputs.

        Args:
            inputs: Any inputs feeding into an embedder.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, embedding_size] float tensor
            representing embeddings.
        """
        pass
