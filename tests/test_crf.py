import torch

from partial_tagger import crf


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
