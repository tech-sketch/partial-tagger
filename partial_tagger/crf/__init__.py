import torch

from .layer import CRF  # NOQA

# Negative infinity
NINF = torch.finfo(torch.float16).min
