from typing import Optional, Tuple

import torch

from partial_tagger.decoder import Decoder
from partial_tagger.feature_extractor import FeatureExtractor
from partial_tagger.loss_function import LossFunction
from partial_tagger.tagger import Tagger


class DummyFeatureExtractor(FeatureExtractor[torch.Tensor]):
    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return inputs


class DummyLossFunction(LossFunction):
    def forward(
        self,
        text_features: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return text_features.sum()


class DummyDecoder(Decoder):
    def forward(
        self, text_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return text_features.sum(dim=(-1, -2)), text_features.sum(dim=-1).long()


def test_tagger_compute_loss_returns_correct_shape(
    test_data_for_shape_check2: tuple,
) -> None:
    _, text_features, tag_bitmap = test_data_for_shape_check2
    feature_extractor = DummyFeatureExtractor()
    loss_function = DummyLossFunction()
    decoder = DummyDecoder()

    tagger = Tagger[torch.Tensor](feature_extractor, loss_function, decoder)

    assert tagger.compute_loss(text_features, tag_bitmap).size() == torch.Size()


def test_tagger_predict_returns_correct_shape(
    test_data_for_shape_check2: tuple,
) -> None:
    (
        (batch_size, sequence_length, _, _),
        text_features,
        tag_bitmap,
    ) = test_data_for_shape_check2
    feature_extractor = DummyFeatureExtractor()
    loss_function = DummyLossFunction()
    decoder = DummyDecoder()

    tagger = Tagger[torch.Tensor](feature_extractor, loss_function, decoder)

    confidence, tag_indices = tagger.predict(text_features, tag_bitmap)

    assert confidence.size() == torch.Size([batch_size])
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])
