from typing import List, Optional, Tuple

import torch
from torch import nn

from partial_tagger.crf.core import CRF
from partial_tagger.decoder import Decoder
from partial_tagger.functional import crf


class ViterbiDecoder(Decoder):
    """A Viterbi decoder for a CRF layer.

    Args:
        crf: A CRF layer.
        padding_index: An integer for padded elements.
    """

    def __init__(self, crf: CRF, padding_index: Optional[int] = -1) -> None:
        super(ViterbiDecoder, self).__init__()

        self.crf = crf
        self.padding_index = padding_index

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
        if mask is None:
            mask = text_features.new_ones(text_features.shape[:-1], dtype=torch.bool)

        log_potentials = self.crf(text_features, mask)

        max_log_probability, tag_indices = crf.decode(log_potentials)

        tag_indices = tag_indices * mask + self.padding_index * (~mask)

        return max_log_probability, tag_indices


class ConstrainedViterbiDecoder(Decoder):
    """A constrained Viterbi decoder for a CRF layer.

    Args:
        crf: A CRF layer.
        padding_index: An integer for padded elements.
        start_constraints: A list of boolean indicating allowed start tags.
        end_constraints: A list of boolean indicating allowed end tags. .
        transition_constraints: A nested list of boolean indicating allowed transition.
    """

    def __init__(
        self,
        crf: CRF,
        start_constraints: List[bool],
        end_constraints: List[bool],
        transition_constraints: List[List[bool]],
        padding_index: Optional[int] = -1,
    ) -> None:
        super(ConstrainedViterbiDecoder, self).__init__()

        self.crf = crf

        self.start_constraints = nn.Parameter(
            torch.tensor(start_constraints), requires_grad=False
        )
        self.end_constraints = nn.Parameter(
            torch.tensor(end_constraints), requires_grad=False
        )
        self.transition_constraints = nn.Parameter(
            torch.tensor(transition_constraints), requires_grad=False
        )

        self.padding_index = padding_index

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
        if mask is None:
            mask = text_features.new_ones(text_features.shape[:-1], dtype=torch.bool)

        log_potentials = self.crf(text_features, mask)

        max_log_probability, tag_indices = crf.constrained_decode(
            log_potentials,
            mask=mask,
            start_constraints=self.start_constraints,
            end_constraints=self.end_constraints,
            transition_constraints=self.transition_constraints,
            padding_index=self.padding_index,
        )

        return max_log_probability, tag_indices
