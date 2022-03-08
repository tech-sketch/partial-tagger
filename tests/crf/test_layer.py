import pytest
import torch

from partial_tagger.crf import CRF
from partial_tagger.functional import crf
from tests import helpers


@pytest.fixture
def model(feature_size: int, num_tags: int) -> CRF:
    return CRF(feature_size, num_tags)


def test_crf_forward_returns_correct_shape(
    model: CRF, test_data_for_shape_check2: tuple
) -> None:
    (
        (
            batch_size,
            sequence_length,
            _,
            num_tags,
        ),
        text_features,
        _,
    ) = test_data_for_shape_check2

    with torch.no_grad():
        log_potentials = model(text_features)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )


def test_crf_forward_returns_correct_shape_if_sequence_length_is_one(
    model: CRF, feature_size: int, num_tags: int
) -> None:
    batch_size = 3
    sequence_length = 1
    text_features = torch.randn(batch_size, sequence_length, feature_size)

    log_potentials = model(text_features)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )


def test_crf_forward_returns_log_potentials_yield_likelihood_valid_as_probability(
    model: CRF, num_tags: int, test_data_small2: tuple
) -> None:
    (batch_size, sequence_length), text_features, _ = test_data_small2

    log_potentials = model(text_features)

    total_log_p = torch.tensor([crf.NINF] * batch_size)
    for tag_bitmap in helpers.iterate_possible_one_hot_tag_bitmap(
        batch_size, sequence_length, num_tags
    ):
        total_log_p = torch.logaddexp(
            total_log_p, crf.log_likelihood(log_potentials, tag_bitmap)
        )

    assert torch.allclose(total_log_p.exp(), torch.ones_like(total_log_p))


def test_crf_forward_returns_correctly_masked_log_potentials(
    model: CRF, feature_size: int
) -> None:
    batch_size = 3
    sequence_length = 20
    text_features = torch.randn(batch_size, sequence_length, feature_size)
    mask = torch.tensor(
        [[True] * (sequence_length - 2 * i) + [False] * 2 * i for i in range(3)]
    )

    log_potentials = model(text_features, mask)

    assert helpers.check_log_potentials_mask(log_potentials, mask)
