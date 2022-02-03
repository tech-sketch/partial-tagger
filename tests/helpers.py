from itertools import product

import torch

from partial_tagger.crf import NINF


def compute_log_normalizer_by_brute_force(log_potentials: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, num_tags, _ = log_potentials.size()
    log_Z = torch.tensor([NINF] * batch_size)
    for b in range(batch_size):
        for tag_indices in product(range(num_tags), repeat=sequence_length + 1):
            tag_indices_score = torch.tensor(0.0)
            for i, (j, k) in enumerate(zip(tag_indices[:-1], tag_indices[1:])):
                tag_indices_score += log_potentials[b, i, j, k]
            log_Z[b] = torch.logaddexp(log_Z[b], tag_indices_score)
    return log_Z
