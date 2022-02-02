import torch

from partial_tagger.layers import CRF


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
