from typing import Callable, Optional, Tuple

import torch

NINF = torch.finfo(torch.float16).min


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

    score = sequence_score(log_potentials, tag_bitmap, mask)
    log_Z = forward_algorithm(log_potentials)

    return score - log_Z


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

    score = multitag_sequence_score(log_potentials, tag_bitmap, mask)
    log_Z = forward_algorithm(log_potentials)

    return score - log_Z


def normalize(log_potentials: torch.Tensor, normalizer: Callable) -> torch.Tensor:
    """Normalizes log potentials based on normalizer.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.

    Returns:
        A [batch_size] float tensor representing the normalized value.
    """
    batch_size, sequence_length, num_tags, _ = log_potentials.size()

    n = sequence_length.bit_length()
    padding_length = (1 << n) - sequence_length
    padding_value = (
        1 - torch.eye(num_tags, num_tags, device=log_potentials.device)
    ).mul(NINF)[None, None]

    log_potentials = torch.cat(
        (
            log_potentials,
            padding_value.repeat(batch_size, padding_length, 1, 1),
        ),
        dim=1,
    )

    for _ in range(n):
        log_potentials = normalizer(
            log_potentials[:, 0::2, ..., None] + log_potentials[:, 1::2, None, ...],
            dim=-2,
        )

    return normalizer(normalizer(log_potentials, dim=-2), dim=-1).squeeze(dim=-1)


def forward_algorithm(log_potentials: torch.Tensor) -> torch.Tensor:
    """Computes the normalizer for a CRF.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.

    Returns:
        A [batch_size] float tensor representing the normalizer.
    """
    return normalize(log_potentials, torch.logsumexp)


def amax(log_potentials: torch.Tensor) -> torch.Tensor:
    """Computes the maximum score for a CRF.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.

    Returns:
        A [batch_size] float tensor representing the maximum score.
    """

    def _amax(inputs: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.max(inputs, dim=dim).values

    return normalize(log_potentials, _amax)


def decode(log_potentials: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the tag sequence gives the maximum probability for log potentials.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.

    Returns:
        A tuple of tensors. The first tensor is a [batch_size] float tensor
        representing the maximum log probability. The second tensor is
        a [batch_size, sequence_length] integer tensor representing the tag sequence.
    """
    max_score = amax(log_potentials)
    log_Z = forward_algorithm(log_potentials)

    (tag_matrix,) = torch.autograd.grad(max_score.sum(), log_potentials)
    tag_matrix = tag_matrix.long()

    tag_bitmap = tag_matrix.sum(dim=-2)

    tag_indices = tag_bitmap.argmax(dim=-1)

    return max_score - log_Z, tag_indices


def constrained_decode(
    log_potentials: torch.Tensor,
    mask: torch.Tensor,
    start_constraints: torch.Tensor,
    end_constraints: torch.Tensor,
    transition_constraints: torch.Tensor,
    padding_index: Optional[int] = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the tag sequence gives the maximum probability for log potentials
    with start/end/transition constraints.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.
        mask: A [batch_size, sequence_length] boolean tensor.
        start_constraints: A [num_tags] boolean tensor.
        end_constraints: A [num_tags] boolean tensor.
        transition_constraints: A [num_tags, num_tags] boolean tensor.
        padding_index: An integer for padded elements.

    Returns:
        A tuple of tensors. The first tensor is a [batch_size] float tensor
        representing the maximum log probability. The second tensor is
        a [batch_size, sequence_length] integer tensor representing the tag sequence.
    """
    # Apply end constraints
    end_constraints = (
        torch.arange(log_potentials.size(1), device=log_potentials.device)[None].lt(
            mask.sum(dim=-1).sub(2)[..., None]
        )
        ^ (~mask[:, 1:])
    )[..., None, None] | end_constraints[None, None, None]
    constrained_log_potentials = log_potentials * end_constraints + NINF * (
        ~end_constraints
    )

    # Apply start constraints
    start_constraints = start_constraints[None, :, None]
    constrained_log_potentials[:, 0] = constrained_log_potentials[
        :, 0
    ] * start_constraints + NINF * (~start_constraints)

    # Apply transition constraints
    transition_constraints = transition_constraints[None, None] | (
        ~mask[:, 1:, None, None]
    )
    constrained_log_potentials = (
        constrained_log_potentials * transition_constraints
        + NINF * (~transition_constraints)
    )

    max_log_probability, tag_indices = decode(constrained_log_potentials)

    tag_indices = tag_indices * mask + padding_index * (~mask)
    return max_log_probability, tag_indices


def sequence_score(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the sequence score based on the given tag_bitmap.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.
        tag_bitmap: A [batch_size, sequence_length, num_tags] boolean tensor
        indicating an active tag at each index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A [batch_size] float tensor representing the sequence score.
    """
    if mask is None:
        mask = tag_bitmap.new_ones(tag_bitmap.shape[:-1], dtype=torch.bool)

    tag_bitmap = tag_bitmap & mask[..., None]
    tag_matrix = tag_bitmap[:, :-1, :, None] & tag_bitmap[:, 1:, None, :]

    return log_potentials.mul(tag_matrix).sum(dim=(1, 2, 3))


def multitag_sequence_score(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the sequence score of all tag sequences matching.

    Args:
        log_potentials: A [batch_size, sequence_length - 1, num_tags, num_tags]
        float tensor.
        tag_bitmap: A [batch_size, sequence_length, num_tags] boolean tensor
        indicating all active tags at each index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A [batch_size] float tensor representing the sequence score.
    """
    if mask is None:
        mask = tag_bitmap.new_ones(tag_bitmap.shape[:-1], dtype=torch.bool)

    tag_bitmap = tag_bitmap | (~mask)[..., None]
    tag_matrix = tag_bitmap[:, :-1, :, None] & tag_bitmap[:, 1:, None, :]

    constrained_log_potentials = log_potentials * tag_matrix + NINF * (~tag_matrix)
    return forward_algorithm(constrained_log_potentials)


def to_tag_bitmap(
    tag_indices: torch.Tensor, num_tags: int, partial_index: Optional[int] = None
) -> torch.Tensor:
    """Computes tag_bitmap from the given tag_indices.

    Args:
        tag_indices: A [batch_size, sequence_length] integer tensor.
        num_tags: An integer value representing the number of tags.
        partial_index: An integer value representing the index for partial label.

    Returns:
        A [batch_size, sequence_length, num_tags] boolean tensor.
        indicating an active tag at each index.
    """
    tag_bitmap = torch.arange(num_tags, device=tag_indices.device)[None, None].eq(
        tag_indices[..., None]
    )

    if partial_index is None:
        return tag_bitmap

    partial_mask = tag_indices.eq(partial_index)
    return tag_bitmap | partial_mask[..., None]
