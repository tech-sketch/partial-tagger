import pytest
import torch

from partial_tagger.crf import CRF
from partial_tagger.crf.decoders import ConstrainedViterbiDecoder, ViterbiDecoder
from partial_tagger.functional import crf


@pytest.fixture
def model_for_shape_check(feature_size: int, num_tags: int) -> CRF:
    return CRF(feature_size, num_tags)


@pytest.fixture
def model(num_tags: int, transitions: torch.Tensor) -> CRF:
    model = CRF(num_tags, num_tags)
    model.transitions.data = transitions
    model.kernel.weight.data = torch.eye(num_tags, num_tags)
    model.kernel.bias.data = torch.zeros(num_tags)
    return model


@torch.no_grad()
def test_viterbi_decoder_returns_correct_shape(
    model_for_shape_check: CRF, test_data_for_shape_check2: tuple
) -> None:
    (batch_size, sequence_length, _, _), text_features, _ = test_data_for_shape_check2
    decoder = ViterbiDecoder(model_for_shape_check)
    confidence, tag_indices = decoder(text_features)

    assert confidence.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])


@torch.no_grad()
def test_constrained_viterbi_decoder_returns_correct_shape(
    model_for_shape_check: CRF, test_data_for_shape_check2: tuple
) -> None:
    (
        (
            batch_size,
            sequence_length,
            _,
            num_tags,
        ),
        text_features,
        _,
    ) = test_data_for_shape_check2
    decoder = ConstrainedViterbiDecoder(
        model_for_shape_check,
        [True] * num_tags,
        [True] * num_tags,
        [[True] * num_tags] * num_tags,
    )

    confidence, tag_indices = decoder(text_features)

    assert confidence.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])


@torch.no_grad()
def test_decoder_returns_correct_tag_indices(
    model: CRF, test_data_by_hand: tuple
) -> None:
    _, text_features, expected_tag_indices = test_data_by_hand
    decoder = ViterbiDecoder(model)

    _, tag_indices = decoder(text_features)

    assert torch.equal(tag_indices, expected_tag_indices)


@torch.no_grad()
def test_decoder_returns_score_valid_as_probability(
    model: CRF, test_data_by_hand: tuple
) -> None:
    _, text_features, tag_indices = test_data_by_hand
    decoder = ViterbiDecoder(model)
    log_potentials = model(text_features)
    expected_log_p = crf.log_likelihood(
        log_potentials, crf.to_tag_bitmap(tag_indices, log_potentials.size(-1))
    )

    max_log_p, _ = decoder(text_features)

    assert torch.allclose(max_log_p, expected_log_p)
