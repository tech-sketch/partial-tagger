import pytest
import torch

from partial_tagger.functional import crf
from partial_tagger.layers import CRF
from partial_tagger.layers.crf import ConstrainedDecoder, Decoder
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
        [batch_size, sequence_length, num_tags, num_tags]
    )


def test_decoder_returns_correct_shape(test_data_for_shape_check: tuple) -> None:
    (batch_size, sequence_length, num_tags), logits, _, _ = test_data_for_shape_check

    model = CRF(num_tags)
    decoder = Decoder()

    max_probabilities, tag_indices = decoder(model(logits))

    assert max_probabilities.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])


def test_constrained_decoder_returns_correct_shape(
    test_data_for_shape_check: tuple,
) -> None:
    (batch_size, sequence_length, num_tags), logits, _, _ = test_data_for_shape_check

    model = CRF(num_tags)
    decoder = ConstrainedDecoder(
        [True] * num_tags, [True] * num_tags, [[True] * num_tags] * num_tags
    )

    max_probabilities, tag_indices = decoder(model(logits))

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


def test_crf_forward_returns_correctly_masked_log_potentials(model: CRF) -> None:
    batch_size = 3
    sequence_length = 20
    num_tags = 5
    logits = torch.randn(batch_size, sequence_length, num_tags)
    mask = torch.tensor(
        [[True] * (sequence_length - 2 * i) + [False] * 2 * i for i in range(3)]
    )

    log_potentials = model(logits, mask)

    assert helpers.check_log_potentials_mask(log_potentials, mask)


def test_crf_forward_returns_tensor_if_sequence_length_equals_to_one(
    model: CRF,
) -> None:
    batch_size = 3
    sequence_length = 1
    num_tags = 5
    logits = torch.randn(batch_size, sequence_length, num_tags)

    log_potentials = model(logits)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )


def test_decoder_returns_correct_tag_indices(
    model: CRF, test_data_by_hand: tuple
) -> None:
    _, logits, expected_tag_indices = test_data_by_hand
    decoder = Decoder()

    _, tag_indices = decoder(model(logits))

    assert torch.equal(tag_indices, expected_tag_indices)


def test_decoder_returns_score_valid_as_probability(
    model: CRF, test_data_by_hand: tuple
) -> None:
    _, logits, tag_indices = test_data_by_hand
    decoder = Decoder()
    log_potentials = model(logits)
    expected_log_p = crf.log_likelihood(
        log_potentials, crf.to_tag_bitmap(tag_indices, log_potentials.size(-1))
    )

    max_log_p, _ = decoder(log_potentials)

    assert torch.allclose(max_log_p, expected_log_p)
