from typing import Optional

import torch
from torch import nn

from .base import BaseCRFEncoder


class LinearCRFEncoder(BaseCRFEncoder):
    """A linear CRF encoder.

    Args:
        embedding_size: An integer representing the embedding size.
        num_tags:  An integer representing the number of tags.
    """

    def __init__(self, embedding_size: int, num_tags: int) -> None:
        super(LinearCRFEncoder, self).__init__(num_tags)

        self.linear = nn.Linear(embedding_size, num_tags)

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
        return self.crf(self.linear(embeddings), mask)
