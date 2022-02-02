import pytest
import torch
from torch import nn

from partial_tagger.taggers import CRFTagger, PartialCRFTagger, TaggerInputs


class DummyFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: TaggerInputs, mask: torch.Tensor) -> torch.Tensor:
        if isinstance(inputs, dict):
            for key in inputs.keys():
                break
            return inputs[key]
        elif isinstance(inputs, (tuple, list)):
            return inputs[0]
        else:
            return inputs


def test_crf_tagger_raises_value_error() -> None:
    feature_size = 256
    num_tags = 5

    with pytest.raises(ValueError):
        CRFTagger(feature_size, DummyFeatureExtractor(), num_tags, use_kernel=False)


def test_crf_tagger_forward_returns_correct_shape(
    test_data_for_shape_check: tuple,
) -> None:
    (batch_size, sequence_length, num_tags), logits, _, _ = test_data_for_shape_check

    tagger = CRFTagger(num_tags, DummyFeatureExtractor(), num_tags, False)

    max_probabilities, tag_indices = tagger(logits)

    assert max_probabilities.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])


def test_crf_tagger_compute_loss_returns_correct_shape(
    test_data_for_shape_check: tuple,
) -> None:
    (batch_size, _, num_tags), logits, _, tag_bitmap = test_data_for_shape_check

    tagger = CRFTagger(num_tags, DummyFeatureExtractor(), num_tags)

    loss = tagger.compute_loss(logits, tag_bitmap)

    assert loss.size() == torch.Size([batch_size])


def test_partial_crf_tagger_compute_loss_returns_correct_shape(
    test_data_for_shape_check: tuple,
) -> None:
    (batch_size, _, num_tags), logits, _, tag_bitmap = test_data_for_shape_check

    tagger = PartialCRFTagger(num_tags, DummyFeatureExtractor(), num_tags, False)

    loss = tagger.compute_loss(logits, tag_bitmap)

    assert loss.size() == torch.Size([batch_size])
