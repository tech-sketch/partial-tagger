from abc import abstractmethod, ABC
from typing import Optional

import torch
from torch.nn import Module


class LossFunction(ABC, Module):
    """Base class of all loss functions."""

    @abstractmethod
    def forward(
        self,
        text_features: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss from the given text feature.

        Args:
            text_features: A [batch_size, sequence_length, feature_size] float tensor.
            y: A [batch_size, sequence, num_tags] boolean tensor
            indicating the target tag sequence.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A scalar float tensor representing loss.
        """
        pass
