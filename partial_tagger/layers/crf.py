from abc import abstractmethod
from typing import List, Optional, Tuple

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
            A [batch_size, sequence_length, num_tag, num_tags] float tensor
            representing log potentials.
        """
        if mask is None:
            mask = self.compute_mask_from_logits(logits)

        num_tags = logits.size(-1)
        initial_mask = torch.eye(num_tags, num_tags, device=logits.device).bool()

        # log potential from the dummy initial token to the first token
        initial_log_potentials = logits[:, [0], :, None] * initial_mask + crf.NINF * (
            ~initial_mask
        )
        log_potentials = torch.cat(
            (
                initial_log_potentials,
                logits[:, 1:, None, :] + self.transitions[None, None],
            ),
            dim=1,
        )

        mask_value = crf.NINF * (~initial_mask)
        mask = mask[..., None, None]

        return log_potentials * mask + mask_value * (~mask)

    def max(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_index: Optional[int] = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the tag sequence gives the maximum probability for logits.

        Args:
            logits: A [batch_size, sequence_length, num_tags] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.
            padding_index: An integer for padded elements.

        Returns:
            A tuple of tensors.
            A [batch_size] float tensor representing the maximum log probabilities
            and A [batch_size,  sequence_length] integer tensor representing
            the tag sequence.
        """
        if mask is None:
            mask = self.compute_mask_from_logits(logits)

        with torch.enable_grad():
            log_potentials = self(logits, mask)
            max_log_probability, tag_indices = crf.decode(log_potentials)

        tag_indices = tag_indices * mask + padding_index * (~mask)

        return max_log_probability, tag_indices

    @staticmethod
    def compute_mask_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Computes a mask tensor.

        Args:
            logits: A [batch_size, sequence_length, num_tags] float tensor.

        Returns:
            A [batch_size, sequence_length] boolean tensor.
        """
        return logits.new_ones(logits.shape[:-1], dtype=torch.bool)


class BaseDecoder(nn.Module):
    """Base class of all decoder."""

    @abstractmethod
    def forward(
        self, log_potentials: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class Decoder(BaseDecoder):
    """A vanilla decoder for a CRF layer.

    Args:
        padding_index: An integer for padded elements.
    """

    def __init__(self, padding_index: Optional[int] = -1) -> None:
        super(Decoder, self).__init__()

        self.padding_index = padding_index

    def forward(
        self, log_potentials: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the tag sequence gives the maximum probability for log potentials.

        Args:
            log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A tuple of tensors.
            A [batch_size] float tensor representing the maximum log probabilities
            and A [batch_size, sequence_length] integer tensor representing
            the tag sequence.
        """
        if mask is None:
            mask = log_potentials.new_ones(log_potentials.shape[:-2], dtype=torch.bool)

        max_log_probability, tag_indices = crf.decode(log_potentials)

        tag_indices = tag_indices * mask + self.padding_index * (~mask)

        return max_log_probability, tag_indices


class ConstrainedDecoder(BaseDecoder):
    """A decoder for a CRF layer with constraints.

    Args:
        padding_index: An integer for padded elements.
        start_constraints: A list of boolean.
        end_constraints: A list of boolean.
        transition_constraints: A list of boolean.
    """

    def __init__(
        self,
        start_constraints: List[bool],
        end_constraints: List[bool],
        transition_constraints: List[List[bool]],
        padding_index: Optional[int] = -1,
    ) -> None:
        super(BaseDecoder, self).__init__()

        self.padding_index = padding_index

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the tag sequence gives the maximum probability for log potentials.

        Args:
            log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A tuple of tensors.
            A [batch_size] float tensor representing the maximum log probabilities
            and A [batch_size, sequence_length] integer tensor representing
            the tag sequence.
        """
        if mask is None:
            mask = log_potentials.new_ones(log_potentials.shape[:-2], dtype=torch.bool)

        max_log_probability, tag_indices = crf.constrained_decode(
            log_potentials,
            mask=mask,
            start_constraints=self.start_constraints,
            end_constraints=self.end_constraints,
            transition_constraints=self.transition_constraints,
            padding_index=self.padding_index,
        )

        return max_log_probability, tag_indices
