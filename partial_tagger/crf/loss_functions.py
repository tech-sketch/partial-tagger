from typing import Optional

import torch

from partial_tagger.crf.core import CRF
from partial_tagger.functional import crf
from partial_tagger.loss_function import LossFunction


class NegativeLogLikelihood(LossFunction):
    """Negative log likelihood loss for CRF.

    Args:
        crf: A CRF layer.
    """

    def __init__(self, crf: CRF) -> None:
        super(NegativeLogLikelihood, self).__init__()

        self.crf = crf

    def forward(
        self,
        text_features: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes negative log likelihood of CRF from the given text feature.

        Args:
            text_features: A [batch_size, sequence_length, feature_size] float tensor.
            y: A [batch_size, sequence, num_tags] boolean tensor
            indicating the target tag sequence.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A scalar float tensor representing loss.
        """
        log_potentials = self.crf(text_features, mask)
        return crf.log_likelihood(log_potentials, y, mask).sum().neg()


class NegativeMarginalLogLikelihood(LossFunction):
    """Negative marginal log likelihood loss for CRF.

    Args:
        crf: A CRF layer.
    """

    def __init__(self, crf: CRF) -> None:
        super(NegativeMarginalLogLikelihood, self).__init__()

        self.crf = crf

    def forward(
        self,
        text_features: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes negative log likelihood of CRF from the given text feature.

        Args:
            text_features: A [batch_size, sequence_length, feature_size] float tensor.
            y: A [batch_size, sequence, num_tags] boolean tensor
            indicating the target tag sequence.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A scalar float tensor representing loss.
        """
        log_potentials = self.crf(text_features, mask)
        return crf.marginal_log_likelihood(log_potentials, y, mask).sum().neg()


class ExpectedEntityRatioLoss(LossFunction):
    """Expected entity ratio loss for CRF.

    Args:
        crf: A CRF layer.
        outside_index: An integer value representing the O tag.
        eer_loss_weight: A float value representing EER loss coefficient.
        entity_ratio: A float value representing entity ratio.
        entity_ratio_margin: A float value representing EER loss margin.
    """

    def __init__(
        self,
        crf: CRF,
        outside_index: int,
        eer_loss_weight: float = 10.0,
        entity_ratio: float = 0.15,
        entity_ratio_margin: float = 0.05,
    ) -> None:
        super(ExpectedEntityRatioLoss, self).__init__()

        self.crf = crf
        self.outside_index = outside_index
        self.eer_loss_weight = eer_loss_weight
        self.entity_ratio = entity_ratio
        self.entity_ratio_margin = entity_ratio_margin

    def forward(
        self,
        text_features: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes negative log likelihood of CRF from the given text feature.

        Args:
            text_features: A [batch_size, sequence_length, feature_size] float tensor.
            y: A [batch_size, sequence, num_tags] boolean tensor
            indicating the target tag sequence.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A scalar float tensor representing loss.
        """
        with torch.enable_grad():
            log_potentials = self.crf(text_features, mask)

            # log partition
            log_Z = crf.forward_algorithm(log_potentials)

            # marginal probabilities
            p = torch.autograd.grad(log_Z.sum(), log_potentials, create_graph=True)[
                0
            ].sum(dim=-1)

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

        # marginal likelihood
        score = crf.multitag_sequence_score(log_potentials, y, mask)

        return (log_Z - score).mean() + self.eer_loss_weight * eer_loss
