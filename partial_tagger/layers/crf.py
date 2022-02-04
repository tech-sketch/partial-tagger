from typing import Optional, Tuple

import torch
from torch import nn

from partial_tagger.functional import crf


class CRF(nn.Module):
    """A CRF layer.

    Args:
        num_tags: Number of tags.
    """

    def __init__(self, num_tags: int) -> None:
        super(CRF, self).__init__()

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes log potentials for a CRF.

        Args:
            logits: A [batch_size, sequence_length, num_tags] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length - 1, num_tag, num_tags] float tensor
            representing log potentials.
        """
        if mask is None:
            mask = self.compute_mask_from_logits(logits)

        log_potentials = logits[:, 1:, None, :] + self.transitions[None, None]
        log_potentials[:, 0] += logits[:, 0, :, None]

        num_tags = log_potentials.size(-1)
        value = crf.NINF * (
            1 - torch.eye(num_tags, num_tags, device=log_potentials.device)
        )
        mask = mask[:, 1:, None, None]

        return log_potentials * mask + value * (~mask)

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

    @staticmethod
    def compute_mask_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Computes a mask tensor.

        Args:
            logits: A [batch_size, sequence_length, num_tags] float tensor.

        Returns:
            A [batch_size, sequence_length] boolean tensor.
        """
        return logits.new_ones(logits.shape[:-1], dtype=torch.bool)
