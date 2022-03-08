import pytest
import torch

from partial_tagger.crf import CRF
from partial_tagger.crf.loss_functions import (
    ExpectedEntityRatioLoss,
    NegativeLogLikelihood,
    NegativeMarginalLogLikelihood,
)


@pytest.fixture
def crf(test_data_for_shape_check2: tuple) -> CRF:
    (
        (
            batch_size,
            sequence_length,
            feature_size,
            num_tags,
        ),
        _,
        _,
    ) = test_data_for_shape_check2
    return CRF(feature_size, num_tags)


@torch.no_grad()
def test_negative_log_likelihood_returns_correct_shape(
    test_data_for_shape_check2: tuple, crf: CRF
) -> None:
    (
        (batch_size, sequence_length, _, _),
        text_features,
        tag_bitmap,
    ) = test_data_for_shape_check2
    loss_function = NegativeLogLikelihood(crf)
    loss = loss_function(text_features, tag_bitmap)

    assert loss.size() == torch.Size()


@torch.no_grad()
def test_negative_marginal_log_likelihood_returns_correct_shape(
    test_data_for_shape_check2: tuple, crf: CRF
) -> None:
    (
        (batch_size, sequence_length, _, _),
        text_features,
        tag_bitmap,
    ) = test_data_for_shape_check2
    loss_function = NegativeMarginalLogLikelihood(crf)
    loss = loss_function(text_features, tag_bitmap)

    assert loss.size() == torch.Size()


@torch.no_grad()
def test_expected_entity_ratio_loss_returns_correct_shape(
    test_data_for_shape_check2: tuple, crf: CRF
) -> None:
    (
        (batch_size, sequence_length, _, _),
        text_features,
        tag_bitmap,
    ) = test_data_for_shape_check2
    loss_function = ExpectedEntityRatioLoss(crf, outside_index=0)
    loss = loss_function(text_features, tag_bitmap)

    assert loss.size() == torch.Size()
