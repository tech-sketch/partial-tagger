import pytest
import torch

from partial_tagger.crf.core import CRF
from partial_tagger.crf.decoders import ConstrainedViterbiDecoder, ViterbiDecoder


@pytest.fixture
def crf(test_data_for_shape_check2: tuple) -> CRF:
    (
        batch_size,
        sequence_length,
        feature_size,
        num_tags,
    ), _ = test_data_for_shape_check2
    return CRF(feature_size, num_tags)


def test_viterbi_decoder_returns_correct_shape(
    test_data_for_shape_check2: tuple, crf: CRF
) -> None:
    (batch_size, sequence_length, _, _), text_features = test_data_for_shape_check2
    decoder = ViterbiDecoder(crf)
    confidence, tag_indices = decoder(text_features)

    assert confidence.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])


def test_constrained_viterbi_decoder_returns_correct_shape(
    test_data_for_shape_check2: tuple, crf: CRF
) -> None:
    (
        batch_size,
        sequence_length,
        _,
        num_tags,
    ), text_features = test_data_for_shape_check2
    decoder = ConstrainedViterbiDecoder(
        crf, [True] * num_tags, [True] * num_tags, [[True] * num_tags] * num_tags
    )

    confidence, tag_indices = decoder(text_features)

    assert confidence.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])
