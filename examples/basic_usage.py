import torch
from torch import optim

from partial_tagger.crf import CRF
from partial_tagger.crf.decoders import ViterbiDecoder
from partial_tagger.crf.loss_functions import NegativeMarginalLogLikelihood
from partial_tagger.functional.crf import to_tag_bitmap


def main() -> None:
    feature_size = 768
    num_tags = 5
    batch_size = 3
    sequence_length = 10

    text_features = torch.randn(batch_size, sequence_length, feature_size)
    tags = torch.randint(0, num_tags, (batch_size, sequence_length))
    tag_bitmap = to_tag_bitmap(tags, num_tags, 0)

    crf = CRF(feature_size, num_tags)
    loss_function = NegativeMarginalLogLikelihood(crf)
    decoder = ViterbiDecoder(crf)

    optimizer = optim.Adam(crf.parameters())

    num_epochs = 100
    for _ in range(num_epochs):
        loss = loss_function(text_features, tag_bitmap)
        loss.backward()
        optimizer.step()

    decoder(text_features)
