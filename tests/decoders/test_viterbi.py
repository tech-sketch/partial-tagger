import torch

from partial_tagger.decoders.viterbi import ConstrainedViterbiDecoder, ViterbiDecoder


@torch.no_grad()
def test_viterbi_decoder_returns_expected_shape_tensor(
    test_data_for_shape_check: tuple,
) -> None:
    (
        (batch_size, sequence_length, _),
        _,
        log_potentials,
        _,
    ) = test_data_for_shape_check

    decoder = ViterbiDecoder()

    assert decoder(log_potentials).size() == torch.Size([batch_size, sequence_length])


@torch.no_grad()
def test_constrained_viterbi_decoder_returns_expected_shape_tensor(
    test_data_for_shape_check: tuple,
) -> None:
    (
        (batch_size, sequence_length, num_tags),
        _,
        log_potentials,
        _,
    ) = test_data_for_shape_check

    decoder = ConstrainedViterbiDecoder(
        [True] * num_tags, [True] * num_tags, [[True] * num_tags] * num_tags
    )

    assert decoder(log_potentials).size() == torch.Size([batch_size, sequence_length])
