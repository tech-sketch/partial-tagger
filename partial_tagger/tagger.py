from typing import Generic, Optional, Tuple

import torch
from torch.nn import Module

from .decoder import Decoder
from .feature_extractor import FeatureExtractor, TaggerInputs
from .loss_function import LossFunction


class Tagger(Module, Generic[TaggerInputs]):
    """Sequence tagger.

    Args:
        feature_extractor: A feature extractor.
        loss_function: A loss function.
        decoder: A decoder.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        loss_function: LossFunction,
        decoder: Decoder,
    ) -> None:
        super(Tagger, self).__init__()

        self.feature_extractor = feature_extractor
        self.loss_function = loss_function
        self.decoder = decoder

    def compute_loss(
        self, inputs: TaggerInputs, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes loss of a tagger.

        Args:
            inputs: A TaggerInputs representing input data.
            y: A [batch_size, sequence, num_tags] boolean tensor
            indicating the target tag sequence.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A scalar float tensor representing loss.
        """
        text_features = self.feature_extractor(inputs, mask)
        return self.loss_function(text_features, y, mask)

    def predict(
        self, inputs: TaggerInputs, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the best tag sequence from inputs and its confidence score.

        Args:
            inputs: A TaggerInputs representing input data.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A tuple of tensors.
            A [batch_size] float tensor representing the confidence score
            and A [batch_size, sequence_length] integer tensor representing
            the best tag sequence.
        """
        text_features = self.feature_extractor(inputs, mask)
        return self.decoder(text_features, mask)
