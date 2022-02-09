import pytest
import torch

from partial_tagger.functional import crf
from tests import helpers


def test_log_likelihood_returns_correct_shape(test_data_for_shape_check: tuple) -> None:
    (batch_size, _, _), _, log_potentials, tag_bitmap = test_data_for_shape_check
    expected_size = torch.Size([batch_size])

    log_p = crf.log_likelihood(log_potentials, tag_bitmap)

    assert log_p.size() == expected_size


def test_marginal_log_likelihood_returns_correct_shape(
    test_data_for_shape_check: tuple,
) -> None:
    (batch_size, _, _), _, log_potentials, tag_bitmap = test_data_for_shape_check
    expected_size = torch.Size([batch_size])

    log_p = crf.marginal_log_likelihood(log_potentials, tag_bitmap)

    assert log_p.size() == expected_size


def test_log_likelihood_valid_as_probability(test_data_small: tuple) -> None:
    (batch_size, sequence_length, num_tags), log_potentials = test_data_small

    total_log_p = torch.tensor([crf.NINF] * batch_size)
    for tag_bitmap in helpers.iterate_possible_one_hot_tag_bitmap(
        batch_size, sequence_length, num_tags
    ):
        total_log_p = torch.logaddexp(
            total_log_p, crf.log_likelihood(log_potentials, tag_bitmap)
        )

    assert torch.allclose(total_log_p.exp(), torch.ones_like(total_log_p))


def test_marginal_log_likelihood_valid_as_probability(test_data_small: tuple) -> None:

    shape, log_potentials = test_data_small

    tag_bitmap = torch.ones(shape, dtype=torch.bool)
    log_p = crf.marginal_log_likelihood(log_potentials, tag_bitmap)

    assert torch.allclose(log_p.exp(), torch.ones_like(log_p))


def test_marginal_log_likelihood_matches_log_likelihood_if_one_hot_tag_bitmap_is_given(
    test_data_small: tuple,
) -> None:
    shape, log_potentials = test_data_small

    for tag_bitmap in helpers.iterate_possible_one_hot_tag_bitmap(*shape):
        a = crf.log_likelihood(log_potentials, tag_bitmap)
        b = crf.marginal_log_likelihood(log_potentials, tag_bitmap)

        assert torch.allclose(a, b)


def test_forward_algorithm_returns_value_same_as_brute_force(
    test_data_small: tuple,
) -> None:
    _, log_potentials = test_data_small

    log_Z = crf.forward_algorithm(log_potentials)
    expected_log_Z = helpers.compute_log_normalizer_by_brute_force(log_potentials)

    assert torch.allclose(log_Z, expected_log_Z)


def test_amax_returns_value_same_as_brute_force(test_data_small: tuple) -> None:
    _, log_potentials = test_data_small

    max_score = crf.amax(log_potentials)
    expected_max_score, _ = helpers.compute_best_tag_indices_by_brute_force(
        log_potentials
    )

    assert torch.allclose(max_score, expected_max_score)


def test_decode_returns_value_same_as_brute_force(test_data_small: tuple) -> None:
    _, log_potentials = test_data_small

    max_log_probability, tag_indices = crf.decode(log_potentials)

    max_score, expected_tag_indices = helpers.compute_best_tag_indices_by_brute_force(
        log_potentials
    )
    log_Z = helpers.compute_log_normalizer_by_brute_force(log_potentials)
    expected_max_log_probability = max_score - log_Z

    assert torch.allclose(max_log_probability, expected_max_log_probability)
    assert torch.allclose(tag_indices, expected_tag_indices)


@pytest.mark.parametrize(
    "tag_indices, num_tags, expected, partial_index",
    [
        (
            torch.tensor([[0, 1, 2, 3, 4]]),
            5,
            torch.tensor(
                [
                    [
                        [True, False, False, False, False],
                        [False, True, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, True, False],
                        [False, False, False, False, True],
                    ]
                ]
            ),
            None,
        ),
        (torch.tensor([-100, -1, 5, 100]), 5, torch.tensor([[[False] * 5] * 4]), None),
        (
            torch.tensor([[0, 1, 2, 3, 4, -1, -1]]),
            5,
            torch.tensor(
                [
                    [
                        [True, False, False, False, False],
                        [False, True, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, True, False],
                        [False, False, False, False, True],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ]
            ),
            -1,
        ),
    ],
)
def test_tag_bitmap_returns_expected_value(
    tag_indices: torch.Tensor, num_tags: int, expected: torch.Tensor, partial_index: int
) -> None:
    assert torch.equal(
        crf.to_tag_bitmap(tag_indices, num_tags, partial_index), expected
    )
