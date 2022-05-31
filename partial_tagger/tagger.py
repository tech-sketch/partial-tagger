from typing import Generic, Optional, Tuple

import torch
from torch.nn import Module

from .decoders import BaseCRFDecoder
from .embedders import BaseEmbedder, EmbedderInputs
from .encoders import BaseCRFEncoder


class Tagger(Generic[EmbedderInputs], Module):
    """Sequence tagger.

    Args:
        embedder: An embedder.
        encoder: An encoder.
        decoder: A decoder.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        encoder: BaseCRFEncoder,
        decoder: BaseCRFDecoder,
    ) -> None:
        super(Tagger, self).__init__()

        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, inputs: EmbedderInputs, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes log potentials and tag sequence.

        Args:
            inputs: An inputs representing input data feeding into embedder.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A scalar float tensor representing loss.
        """
        embeddings = self.embedder(inputs, mask)
        log_potentials = self.encoder(embeddings, mask)
        tag_indices = self.decoder(log_potentials, mask)
        return log_potentials, tag_indices
