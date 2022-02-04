import pytest
import torch


@pytest.fixture
def test_data_for_shape_check() -> tuple:
    batch_size = 3
    sequence_length = 20
    num_tags = 5
    logits = torch.randn(batch_size, sequence_length, num_tags)
    log_potentials = torch.randn(batch_size, sequence_length - 1, num_tags, num_tags)
    y = torch.randint(0, num_tags, (batch_size, sequence_length))
    tag_bitmap = torch.nn.functional.one_hot(y, num_tags).bool()
    return (batch_size, sequence_length, num_tags), logits, log_potentials, tag_bitmap


@pytest.fixture
def test_data_small() -> tuple:
    batch_size = 2
    sequence_length = 3
    num_tags = 5
    log_potentials = torch.randn(batch_size, sequence_length - 1, num_tags, num_tags)
    return (batch_size, sequence_length, num_tags), log_potentials
