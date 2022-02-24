from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from partial_tagger.functional import crf
from partial_tagger.layers.crf import CRF, BaseDecoder, Decoder

TaggerInputs = Union[
    torch.Tensor, Dict[Any, torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor]
]


class CRFTagger(nn.Module):
    """A sequence tagger with CRF.

    Args:
        feature_size: Dimension of output vectors of feature_extractor.
        feature_extractor: Feature extraction network.
        num_tags: Number of tags.
        use_kernel: Boolean indicating if a kernel is used.
        decoder: Decoder to computes the most likely tag sequence.
    """

    def __init__(
        self,
        feature_size: int,
        feature_extractor: nn.Module,
        num_tags: int,
        use_kernel: Optional[bool] = True,
        decoder: Optional[BaseDecoder] = None,
    ) -> None:
        if feature_size != num_tags and not use_kernel:
            raise ValueError(
                "Feature size doesn't match with num_tags."
                "Please fix feature_size or set use_kernel True."
            )

        super(CRFTagger, self).__init__()

        self.feature_extractor = feature_extractor
        self.crf_layer = CRF(num_tags)

        self.kernel: nn.Module
        if use_kernel:
            self.kernel = nn.Linear(feature_size, num_tags)
        else:
            self.kernel = nn.Identity()

        self.decoder: BaseDecoder
        if decoder is None:
            self.decoder = Decoder()
        else:
            self.decoder = decoder

    def forward(
        self,
        inputs: TaggerInputs,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the tag sequence gives the maximum probability for inputs.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A tuple of tensors.
            A [batch_size] float tensor representing the maximum probabilities
            and A [batch_size,  sequence_length] integer tensor representing
            the tag sequence.
        """
        with torch.no_grad():
            features = self.feature_extractor(inputs, mask)
            logits = self.kernel(features)

        log_potentials = self.crf_layer(logits, mask)

        return self.decoder(log_potentials, mask)

    def compute_loss(
        self,
        inputs: TaggerInputs,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss of a tagger.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            y: A [batch_size, sequence_length, num_tags] boolean tensor
            indicating an active tag at each index.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A float tensor representing loss.
        """
        features = self.feature_extractor(inputs, mask)

        logits = self.kernel(features)

        log_potentials = self.crf_layer(logits, mask)

        return crf.log_likelihood(log_potentials, y, mask).sum().neg()


class PartialCRFTagger(CRFTagger):
    """A sequence tagger with CRF for partially annotated data.

    Args:
        feature_size: Dimension of output vectors of feature_extractor.
        feature_extractor: Feature extraction network.
        num_tags: Number of tags.
        use_kernel: Boolean indicating if a kernel is used.
    """

    def compute_loss(
        self,
        inputs: TaggerInputs,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss of a tagger.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            y: A [batch_size, sequence_length, num_tags] boolean tensor
            indicating all active tags at each index.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A float tensor representing loss.
        """
        features = self.feature_extractor(inputs, mask)

        logits = self.kernel(features)

        log_potentials = self.crf_layer(logits, mask)

        return crf.marginal_log_likelihood(log_potentials, y, mask).sum().neg()


class ExpectedEntityRatioPartialCRFTagger(CRFTagger):
    """A sequence tagger for partially annotated data with expected entity ratio loss.

    Args:
        feature_size: Dimension of output vectors of feature_extractor.
        feature_extractor: Feature extraction network.
        num_tags: Number of tags.
        use_kernel: Boolean indicating if a kernel is used.
        decoder: Decoder to computes the most likely tag sequence.
        outside_index: An integer value representing the O tag.
        eer_loss_weight: A float value representing EER loss coefficient.
        entity_ratio: A float value representing entity ratio.
        entity_ratio_margin: A float value representing EER loss margin.
    """

    def __init__(
        self,
        feature_size: int,
        feature_extractor: nn.Module,
        num_tags: int,
        use_kernel: bool = True,
        decoder: Optional[BaseDecoder] = None,
        outside_index: int = 0,
        eer_loss_weight: float = 10.0,
        entity_ratio: float = 0.15,
        entity_ratio_margin: float = 0.05,
    ) -> None:
        super(ExpectedEntityRatioPartialCRFTagger, self).__init__(
            feature_size,
            feature_extractor,
            num_tags,
            use_kernel,
            decoder,
        )

        self.outside_index = outside_index
        self.eer_loss_weight = eer_loss_weight
        self.entity_ratio = entity_ratio
        self.entity_ratio_margin = entity_ratio_margin

    def compute_loss(
        self,
        inputs: TaggerInputs,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss of a tagger.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            y: A [batch_size, sequence_length, num_tags] boolean tensor
            indicating all active tags at each index.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A float tensor representing loss.
        """
        features = self.feature_extractor(inputs, mask)

        logits = self.kernel(features)

        log_potentials = self.crf_layer(logits, mask)

        # marginal likelihood
        score = crf.multitag_sequence_score(log_potentials, y, mask)
        log_Z = crf.forward_algorithm(log_potentials)

        # expected entity ratio
        p = torch.autograd.grad(log_Z.sum(), log_potentials, create_graph=True)[0].sum(
            dim=-1
        )
        if mask is not None:
            p *= mask[..., None]
        expected_entity_count = (
            p[:, :, : self.outside_index].sum()
            + p[:, :, self.outside_index + 1 :].sum()
        )
        expected_entity_ratio = expected_entity_count / p.sum()
        eer_loss = torch.clamp(
            (expected_entity_ratio - self.entity_ratio).abs()
            - self.entity_ratio_margin,
            min=0,
        )

        return (log_Z - score).mean() + self.eer_loss_weight * eer_loss
