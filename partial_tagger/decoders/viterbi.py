from typing import List, Optional

import torch
from torch import nn

from ..crf import functional as F
from .base import BaseCRFDecoder


class ViterbiDecoder(BaseCRFDecoder):
    """A Viterbi decoder for CRF.

    Args:
        padding_index: An integer for padded elements.
    """

    def __init__(self, padding_index: Optional[int] = -1) -> None:
        super(ViterbiDecoder, self).__init__()

        self.padding_index = padding_index

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
        if mask is None:
            mask = log_potentials.new_ones(log_potentials.shape[:-2], dtype=torch.bool)

        log_potentials.requires_grad_()

        with torch.enable_grad():
            _, tag_indices = F.decode(log_potentials)

        return tag_indices * mask + self.padding_index * (~mask)


class ConstrainedViterbiDecoder(ViterbiDecoder):
    """A constrained Viterbi decoder.

    Args:
        padding_index: An integer for padded elements.
        start_constraints: A list of boolean indicating allowed start tags.
        end_constraints: A list of boolean indicating allowed end tags. .
        transition_constraints: A nested list of boolean indicating allowed transition.
    """

    def __init__(
        self,
        start_constraints: List[bool],
        end_constraints: List[bool],
        transition_constraints: List[List[bool]],
        padding_index: Optional[int] = -1,
    ) -> None:
        super(ConstrainedViterbiDecoder, self).__init__(padding_index)

        self.start_constraints = nn.Parameter(
            torch.tensor(start_constraints), requires_grad=False
        )
        self.end_constraints = nn.Parameter(
            torch.tensor(end_constraints), requires_grad=False
        )
        self.transition_constraints = nn.Parameter(
            torch.tensor(transition_constraints), requires_grad=False
        )

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
        if mask is None:
            mask = log_potentials.new_ones(log_potentials.shape[:-2], dtype=torch.bool)

        constrained_log_potentials = F.constrain_log_potentials(
            log_potentials,
            mask,
            self.start_constraints,
            self.end_constraints,
            self.transition_constraints,
        )

        return super(ConstrainedViterbiDecoder, self).forward(
            constrained_log_potentials, mask
        )
