from typing import Optional

import torch
from torch import nn

from partial_tagger.functional import crf


class CRF(nn.Module):
    """A CRF layer.

    Args:
        feature_size: An integer representing a feature size:
        num_tags: An integer representing the number of tags.
    """

    def __init__(self, feature_size: int, num_tags: int) -> None:
        super(CRF, self).__init__()

        self.kernel = nn.Linear(feature_size, num_tags)
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(
        self, text_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes log potentials for a CRF.

        Args:
            text_features: A [batch_size, sequence_length, feature_size] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, num_tag, num_tags] float tensor
            representing log potentials.
        """
        if mask is None:
            mask = text_features.new_ones(text_features.shape[:-1], dtype=torch.bool)

        logits = self.kernel(text_features)
        num_tags = logits.size(-1)
        initial_mask = torch.eye(num_tags, num_tags, device=logits.device).bool()

        # log potential from the dummy initial token to the real initial token
        initial_log_potentials = logits[:, [0], :, None] * initial_mask + crf.NINF * (
            ~initial_mask
        )
        log_potentials = torch.cat(
            (
                initial_log_potentials,
                logits[:, 1:, None, :] + self.transitions[None, None],
            ),
            dim=1,
        )

        mask_value = crf.NINF * (~initial_mask)
        mask = mask[..., None, None]

        return log_potentials * mask + mask_value * (~mask)
