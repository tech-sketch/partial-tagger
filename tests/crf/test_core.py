import torch

from partial_tagger.crf.core import CRF


def test_crf_forward_returns_correct_shape(test_data_for_shape_check2: tuple) -> None:
    (
        batch_size,
        sequence_length,
        feature_size,
        num_tags,
    ), text_features = test_data_for_shape_check2
    crf = CRF(feature_size, num_tags)

    with torch.no_grad():
        log_potentials = crf(text_features)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )
