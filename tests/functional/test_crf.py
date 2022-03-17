from unittest.mock import patch

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


def test_sequence_score_computes_mask_correctly(
    test_data_with_mask: tuple,
) -> None:
    (_, _, num_tags), log_potentials, tag_indices, mask = test_data_with_mask
    tag_bitmap = crf.to_tag_bitmap(tag_indices, num_tags)

    with patch("torch.Tensor.mul") as m:
        crf.sequence_score(log_potentials, tag_bitmap, mask)
        used_mask = m.call_args[0][0]

        assert helpers.check_sequence_score_mask(used_mask, tag_indices, mask)


@pytest.mark.parametrize(
    "partial_index",
    [i for i in range(5)],
)
def test_multitag_sequence_score_correctly_masks_log_potentials(
    test_data_with_mask: tuple, partial_index: int
) -> None:
    (_, _, num_tags), log_potentials, tag_indices, mask = test_data_with_mask
    tag_bitmap = crf.to_tag_bitmap(tag_indices, num_tags, partial_index=partial_index)

    with patch("partial_tagger.functional.crf.forward_algorithm") as m:
        crf.multitag_sequence_score(log_potentials, tag_bitmap, mask)
        constrained_log_potentials = m.call_args[0][0]

        assert helpers.check_constrained_log_potentials(
            log_potentials, constrained_log_potentials, tag_indices, mask, partial_index
        )


@pytest.mark.parametrize(
    "log_potentials, mask, start_constraints, end_constraints, transition_constraints",
    [
        (
            torch.randn(3, 20, 5, 5, requires_grad=True),
            torch.ones((3, 20), dtype=torch.bool),
            torch.tensor([True, False, False, True, True]),  # 0, 3, 4 are allowed
            torch.tensor([False, True, True, False, False]),  # 2, 3 are allowed
            torch.tensor(
                [
                    [True, False, True, True, True],  # 0->1 is not allowed
                    [True, True, True, True, True],  # no constraints
                    [True, True, False, True, True],  # 2->2 is not allowed
                    [True, False, True, True, True],  # 3->1 is not allowed
                    [True, False, False, False, False],  # only 4->0 is allowed
                ]
            ),
        ),
        (
            torch.zeros(3, 20, 5, 5, requires_grad=True),
            torch.ones((3, 20), dtype=torch.bool),
            torch.tensor([False, False, True, True, True]),
            torch.tensor([False, False, True, True, True]),
            torch.tensor([[True] * 5] * 5),
        ),
    ],
)
def test_constrained_decode_returns_tag_indices_under_constraints(
    log_potentials: torch.Tensor,
    mask: torch.Tensor,
    start_constraints: torch.Tensor,
    end_constraints: torch.Tensor,
    transition_constraints: torch.Tensor,
) -> None:
    constrained_log_potentials = crf.constrain_log_potentials(
        log_potentials, mask, start_constraints, end_constraints, transition_constraints
    )
    _, tag_indices = crf.decode(constrained_log_potentials)

    assert helpers.check_tag_indices_satisfies_constraints(
        tag_indices, start_constraints, end_constraints, transition_constraints
    )


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
        (
            torch.tensor([[4, 1, 2, 3, 4, 0, 0]]),
            5,
            torch.tensor(
                [
                    [
                        [False, False, False, False, True],
                        [False, True, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, True, False],
                        [False, False, False, False, True],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ]
            ),
            0,
        ),
    ],
)
def test_tag_bitmap_returns_expected_value(
    tag_indices: torch.Tensor, num_tags: int, expected: torch.Tensor, partial_index: int
) -> None:
    assert torch.equal(
        crf.to_tag_bitmap(tag_indices, num_tags, partial_index), expected
    )
