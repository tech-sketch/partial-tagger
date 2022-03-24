# Sequence Tagger for Partially Annotated Dataset in PyTorch

This is a CRF tagger for partially annotated dataset in PyTorch.


## Usage

First, import some classes.

```py
from partial_tagger.crf import CRF
from partial_tagger.crf.loss_functions import (
    ExpectedEntityRatioLoss,
    NegativeMarginalLogLikelihood,
)
from partial_tagger.crf.decoders import ViterbiDecoder
from partial_tagger.functional.crf import to_tag_bitmap
```

Initialize `CRF` by giving it the dimension of feature vector and the number of tags.

```py
feature_size = 768
num_tags = 5

crf = CRF(feature_size, num_tags)
```

Prepare incomplete tag sequence (partial annotation) and convert it to a tag bitmap.  
This tag bitmap represents the target value for CRF and is used for loss function. 

```py
# 0-4 indicates a true tag
# -1 indicates that a tag is unknown 
incomplete_tags = torch.tensor([[0, -1, 2, -1, 4, 0, 1, -1, 3, 4]])

tag_bitmap = to_tag_bitmap(incomplete_tags, num_tags=num_tags, partial_index=-1)

```

Compute loss from a feature vector and a tag bitmap.

```py

batch_size = 3
sequence_length = 10
# Dummy feature vector. You can use your own.
text_features = torch.randn(batch_size, sequence_length, feature_size)

loss_function = NegativeMarginalLogLikelihood(crf)
# You can use ExpectedEntityRatioLoss.
# loss_function = ExpectedEntityRatioLoss(crf, outside_index=0)

loss = loss_function(text_features, tag_bitmap)
```

Decode the best tag sequence from the given feature vector.

```py

decoder = ViterbiDecoder(crf)

_, tag_sequence = decoder(text_features)
```


## Installation

To install this package:

```bash
pip install partial-tagger
```

## References

- Alexander Rush. 2020. [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/). In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations_, pages 335–342, Online. Association for Computational Linguistics.
- Thomas Effland and Michael Collins. 2021. [Partially Supervised Named Entity Recognition via the Expected Entity Ratio Loss](https://aclanthology.org/2021.tacl-1.78/). _Transactions of the Association for Computational Linguistics_, 9:1320–1335.
- Yuta Tsuboi, Hisashi Kashima, Shinsuke Mori, Hiroki Oda, and Yuji Matsumoto. 2008. [Training Conditional Random Fields Using Incomplete Annotations](https://aclanthology.org/C08-1113/). In _Proceedings of the 22nd International Conference on Computational Linguistics (Coling 2008)_, pages 897–904, Manchester, UK. Coling 2008 Organizing Committee.
