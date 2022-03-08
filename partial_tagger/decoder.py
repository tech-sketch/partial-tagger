from abc import abstractmethod, ABC
from typing import Optional, Tuple

import torch
from torch.nn import Module


class Decoder(ABC, Module):
    """Base class of all decoders."""

    @abstractmethod
    def forward(
        self, text_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the best tag sequence from the given text feature and its confidence score.

        Args:
            text_features: A [batch_size, sequence_length, feature_size] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A tuple of tensors.
            A [batch_size] float tensor representing the confidence score
            and A [batch_size, sequence_length] integer tensor representing
            the best tag sequence.
        """
        pass
