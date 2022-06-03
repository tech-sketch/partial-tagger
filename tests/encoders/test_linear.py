import torch

from partial_tagger.encoders.linear import LinearCRFEncoder


def test_linear_crf_encoder_returns_expected_shape_tensor(
    test_data_for_shape_check: tuple,
) -> None:
    (
        batch_size,
        sequence_length,
        embedding_size,
        num_tags,
    ), *_ = test_data_for_shape_check

    encoder = LinearCRFEncoder(embedding_size, num_tags)
    embeddings = torch.randn((batch_size, sequence_length, embedding_size))

    log_potentials = encoder(embeddings)

    assert log_potentials.size() == torch.Size(
        [batch_size, sequence_length, num_tags, num_tags]
    )
