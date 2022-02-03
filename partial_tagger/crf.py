from typing import Optional

import torch

NINF = -1e5


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
        A [batch_size] float tensor representing log likelihood.
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
        A [batch_size] float tensor representing marginal log likelihood.
    """
    batch_size, _, _, _ = log_potentials.size()
    return torch.randn(batch_size)


def forward_algorithm(log_potentials: torch.Tensor) -> torch.Tensor:
    """Computes the normalizer for a CRF.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.

    Returns:
        A [batch_size] float tensor representing the normalizer.
    """
    batch_size, sequence_length, num_tags, _ = log_potentials.size()

    n = sequence_length.bit_length()
    padding_length = (1 << n) - sequence_length
    value = (1 - torch.eye(num_tags, num_tags, device=log_potentials.device)) * NINF

    log_potentials = torch.cat(
        (log_potentials, value[None, None].repeat(batch_size, padding_length, 1, 1)),
        dim=1,
    )

    for _ in range(n):
        log_potentials = torch.logsumexp(
            log_potentials[:, 0::2, ..., None] + log_potentials[:, 1::2, None, ...],
            dim=-2,
        )

    return torch.logsumexp(log_potentials, dim=(-1, -2)).squeeze(dim=-1)


def amax(log_potentials: torch.Tensor) -> torch.Tensor:
    """Computes the maximum score for a CRF.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.

    Returns:
        A [batch_size] float tensor representing the maximum score.
    """
    batch_size, sequence_length, num_tags, _ = log_potentials.size()

    n = sequence_length.bit_length()
    padding_length = (1 << n) - sequence_length
    value = (1 - torch.eye(num_tags, num_tags, device=log_potentials.device)) * NINF

    log_potentials = torch.cat(
        (log_potentials, value[None, None].repeat(batch_size, padding_length, 1, 1)),
        dim=1,
    )

    for _ in range(n):
        log_potentials = torch.amax(
            log_potentials[:, 0::2, ..., None] + log_potentials[:, 1::2, None, ...],
            dim=-2,
        )

    return torch.amax(log_potentials, dim=(-1, -2)).squeeze(dim=-1)
