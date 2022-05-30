from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.nn import Module


class BaseCRFDecoder(ABC, Module):
    """Base class of all decoders for CRF."""

    @abstractmethod
    def forward(
        self, log_potentials: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the best tag sequence from the given log potentials.

        Args:
            log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length] integer tensor representing
            the best tag sequence.

        """
        pass
