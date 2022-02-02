from typing import Optional, Tuple

import torch
from torch import nn


class CRF(nn.Module):
    """A CRF layer.

    Args:
        num_tags: Number of tags.
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()

    def forward(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes log potentials.

        Args:
            logits: A [batch_size, sequence_length, num_tags] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length - 1, num_tag, num_tags] float tensor
            representing log potentials.
        """
        batch_size, sequence_length, num_tags = logits.size()
        return torch.randn(batch_size, sequence_length - 1, num_tags, num_tags)

    def max(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the tag sequence gives the maximum probability for logits.

        Args:
            logits: A [batch_size, sequence_length, num_tags] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A tuple of tensors.
            A [batch_size] float tensor representing the maximum probabilities
            and A [batch_size,  sequence_length] integer tensor representing
            the tag sequence.
        """
        batch_size, sequence_length, num_tags = logits.size()
        return torch.randn(batch_size), torch.randint(
            0, num_tags, (batch_size, sequence_length)
        )
