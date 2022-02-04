from itertools import product
from typing import Generator, Tuple

import torch

from partial_tagger.functional.crf import NINF


def iterate_possible_tag_indices(
    sequence_length: int, num_tags: int
) -> Generator[tuple, None, None]:
    yield from product(range(num_tags), repeat=sequence_length)


def iterate_possible_one_hot_tag_bitmap(
    batch_size: int, sequence_length: int, num_tags: int
) -> Generator[torch.Tensor, None, None]:
    for tag_indices in iterate_possible_tag_indices(sequence_length, num_tags):
        tag_bitmap = []
        for active in tag_indices:
            bitmap = [False] * num_tags
            bitmap[active] = True
            tag_bitmap.append(bitmap)
        yield torch.tensor([tag_bitmap] * batch_size)


def compute_log_normalizer_by_brute_force(log_potentials: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, num_tags, _ = log_potentials.size()
    log_Z = torch.tensor([NINF] * batch_size)
    for b in range(batch_size):
        for tag_indices in iterate_possible_tag_indices(sequence_length + 1, num_tags):
            tag_indices_score = torch.tensor(0.0)
            for i, (j, k) in enumerate(zip(tag_indices[:-1], tag_indices[1:])):
                tag_indices_score += log_potentials[b, i, j, k]
            log_Z[b] = torch.logaddexp(log_Z[b], tag_indices_score)
    return log_Z


def compute_best_tag_indices_by_brute_force(
    log_potentials: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, num_tags, _ = log_potentials.size()
    best_tag_indices = torch.tensor(
        [[-1] * (sequence_length + 1) for _ in range(batch_size)]
    )
    max_scores = torch.tensor([NINF] * batch_size)
    for b in range(batch_size):
        max_score = torch.tensor(NINF)
        for tag_indices in iterate_possible_tag_indices(sequence_length + 1, num_tags):
            tag_indices_score = torch.tensor(0.0)
            for i, (j, k) in enumerate(zip(tag_indices[:-1], tag_indices[1:])):
                tag_indices_score += log_potentials[b, i, j, k]
            if tag_indices_score.gt(max_score):
                best_tag_indices[b] = torch.tensor(tag_indices)
                max_score = tag_indices_score
        max_scores[b] = max_score
    return max_scores, best_tag_indices
