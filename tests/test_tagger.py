from typing import Optional

import torch

from partial_tagger.decoders.viterbi import ViterbiDecoder
from partial_tagger.embedders import BaseEmbedder
from partial_tagger.encoders.linear import LinearCRFEncoder
from partial_tagger.tagger import Tagger


class PassThroughEmbedder(BaseEmbedder[torch.Tensor]):
    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return inputs


def test_tagger_forward_returns_expected_shape_tensor(
    test_data_for_shape_check: tuple,
) -> None:
    embedding_size = 256
    (batch_size, sequence_length, num_tags), *_ = test_data_for_shape_check

    tagger = Tagger[torch.Tensor](
        PassThroughEmbedder(),
        LinearCRFEncoder(embedding_size, num_tags),
        ViterbiDecoder(),
    )
    inputs = torch.randn((batch_size, sequence_length, embedding_size))

    log_potentials, tag_indices = tagger(inputs)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )
    assert tag_indices.size() == torch.Size([batch_size, sequence_length])
