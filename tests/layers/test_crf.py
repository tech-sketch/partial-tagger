import pytest
import torch

from partial_tagger.functional import crf
from partial_tagger.layers import CRF
from tests import helpers


@pytest.fixture
def model(transitions: torch.Tensor) -> CRF:
    model = CRF(transitions.size(0))
    model.transitions.data = transitions
    return model


def test_crf_forward_returns_correct_shape(test_data_for_shape_check: tuple) -> None:
    (batch_size, sequence_length, num_tags), logits, _, _ = test_data_for_shape_check

    model = CRF(num_tags)

    assert model(logits).size() == torch.Size(
        [batch_size, sequence_length - 1, num_tags, num_tags]
    )


def test_crf_max_returns_correct_shape(test_data_for_shape_check: tuple) -> None:
    (batch_size, sequence_length, num_tags), logits, _, _ = test_data_for_shape_check

    model = CRF(num_tags)

    max_probabilities, tag_indices = model.max(logits)

    assert max_probabilities.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])


def test_crf_forward_returns_log_potentials_yield_likelihood_valid_as_probability(
    model: CRF, test_data_by_hand: tuple
) -> None:
    (batch_size, sequence_length, num_tags), logits, _ = test_data_by_hand

    log_potentials = model(logits)

    total_log_p = torch.tensor([crf.NINF] * batch_size)
    for tag_bitmap in helpers.iterate_possible_one_hot_tag_bitmap(
        batch_size, sequence_length, num_tags
    ):
        total_log_p = torch.logaddexp(
            total_log_p, crf.log_likelihood(log_potentials, tag_bitmap)
        )

    assert torch.allclose(total_log_p.exp(), torch.ones_like(total_log_p))
