import pytest
import torch

from partial_tagger.crf import functional as F
from partial_tagger.crf.nn import CRF

from .. import helpers


@pytest.fixture
def model(num_tags: int) -> CRF:
    return CRF(num_tags)


@torch.no_grad()
def test_crf_forward_returns_expected_shape_tensor(
    model: CRF, test_data_for_shape_check: tuple
) -> None:
    (
        (batch_size, sequence_length, _, num_tags),
        _,
        logits,
        *_,
    ) = test_data_for_shape_check

    log_potentials = model(logits)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )


@torch.no_grad()
def test_crf_forward_handles_if_sequence_length_is_one(
    model: CRF, num_tags: int
) -> None:
    batch_size = 3
    sequence_length = 1
    logits = torch.randn(batch_size, sequence_length, num_tags)

    log_potentials = model(logits)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )


@torch.no_grad()
def test_crf_forward_returns_log_potentials_yield_likelihood_valid_as_probability(
    model: CRF, test_data_small: tuple
) -> None:
    (batch_size, sequence_length, num_tags), *_ = test_data_small

    logits = torch.randn(batch_size, sequence_length, num_tags)

    log_potentials = model(logits)

    total_log_p = torch.tensor([F.NINF] * batch_size)
    for tag_indices in helpers.iterate_possible_tag_indices(sequence_length, num_tags):
        total_log_p = torch.logaddexp(
            total_log_p, F.log_likelihood(log_potentials, torch.tensor(tag_indices))
        )

    assert torch.allclose(total_log_p.exp(), torch.ones_like(total_log_p))


@torch.no_grad()
def test_crf_forward_returns_expected_masked_log_potentials(
    model: CRF, num_tags: int
) -> None:
    batch_size = 3
    sequence_length = 20
    logits = torch.randn(batch_size, sequence_length, num_tags)
    mask = torch.tensor(
        [[True] * (sequence_length - 2 * i) + [False] * 2 * i for i in range(3)]
    )

    log_potentials = model(logits, mask)

    assert helpers.check_log_potentials_mask(log_potentials, mask)
