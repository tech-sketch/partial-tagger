import torch

from partial_tagger import crf
from tests.helpers import compute_log_normalizer_by_brute_force


def test_log_likelihood_returns_correct_shape(test_data_for_shape_check: tuple) -> None:
    (batch_size, _, _), _, log_potentials, tag_bitmap = test_data_for_shape_check

    assert crf.log_likelihood(log_potentials, tag_bitmap).size() == torch.Size(
        [batch_size]
    )


def test_marginal_log_likelihood_returns_correct_shape(
    test_data_for_shape_check: tuple,
) -> None:
    (batch_size, _, _), _, log_potentials, tag_bitmap = test_data_for_shape_check

    assert crf.marginal_log_likelihood(log_potentials, tag_bitmap).size() == torch.Size(
        [batch_size]
    )


def test_forward_algorithm_returns_value_same_as_brute_force(
    test_data_by_hand_for_crf_functions: tuple,
) -> None:
    _, log_potentials, _ = test_data_by_hand_for_crf_functions

    log_Z = crf.forward_algorithm(log_potentials)
    expected_log_Z = compute_log_normalizer_by_brute_force(log_potentials)

    assert torch.allclose(log_Z, expected_log_Z)
