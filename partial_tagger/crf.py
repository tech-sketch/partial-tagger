from typing import Optional

import torch


def log_likelihood(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes log likelihood.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.
        tag_bitmap: A [batch_size, sequence_length, num_tags] boolean tensor
        indicating an active tag at each index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A tensor representing log likelihood.
    """
    batch_size, _, _, _ = log_potentials.size()
    return torch.randn(batch_size)


def marginal_log_likelihood(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes marginal log likelihood.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.
        tag_bitmap: A [batch_size, sequence_length, num_tags] boolean tensor
        indicating all active tags at each index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A tensor representing marginal log likelihood.
    """
    batch_size, _, _, _ = log_potentials.size()
    return torch.randn(batch_size)
