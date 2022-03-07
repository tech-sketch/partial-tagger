from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import torch
from torch.nn import Module

TaggerInputs = TypeVar("TaggerInputs")


class FeatureExtractor(Module, Generic[TaggerInputs]):
    """Base class of all feature extractors."""

    @abstractmethod
    def forward(
        self, inputs: TaggerInputs, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes a text feature from the given input.

        Args:
            inputs: A TaggerInputs representing input data.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, feature_size] float tensor.
        """
        pass
