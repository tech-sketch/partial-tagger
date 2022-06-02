# Sequence Tagger for Partially Annotated Dataset in PyTorch

This is a CRF tagger for partially annotated dataset in PyTorch. You can easily utilize
marginal log likelihood for CRF (Tsuboi, et al., 2008). The implementation of this library is based on Rush, 2020.


## Usage

First, import some modules as follows.

```py
from partial_tagger.crf.nn import CRF
from partial_tagger.crf import functional as F
```

Initialize `CRF` by giving it the number of tags.

```py
num_tags = 2
crf = CRF(num_tags)
```

Prepare incomplete tag sequence (partial annotation) and convert it to a tag bitmap.  
This tag bitmap represents the target value for CRF.

```py
# 0-1 indicates a true tag
# -1 indicates that a tag is unknown
incomplete_tags = torch.tensor([[0, 1, 0, 1, -1, -1, -1, 1, 0, 1]])

tag_bitmap = F.to_tag_bitmap(incomplete_tags, num_tags=num_tags, partial_index=-1)

```

Compute marginal log likelihood from logits.

```py
batch_size = 1
sequence_length = 10
# Dummy logits
logits = torch.randn(batch_size, sequence_length, num_tags)

log_potentials = crf(logits)

loss = F.marginal_log_likelihood(log_potentials, tag_bitmap).sum().neg()
```

## Installation

To install this package:

```bash
pip install partial-tagger
```

## References

- Yuta Tsuboi, Hisashi Kashima, Shinsuke Mori, Hiroki Oda, and Yuji Matsumoto. 2008. [Training Conditional Random Fields Using Incomplete Annotations](https://aclanthology.org/C08-1113/). In _Proceedings of the 22nd International Conference on Computational Linguistics (Coling 2008)_, pages 897–904, Manchester, UK. Coling 2008 Organizing Committee.
- Alexander Rush. 2020. [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/). In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations_, pages 335–342, Online. Association for Computational Linguistics.
