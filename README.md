# Sequence Tagger for Partially Annotated Dataset in PyTorch

This is a CRF tagger for partially annotated dataset in PyTorch.
This adopts the same algorithm of `torch-struct`. So it runs faster than other implementation.


## Installation

To install this package:

```bash
pip install partial-tagger
```

## References

- Alexander Rush. 2020. [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/). In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 335–342, Online. Association for Computational Linguistics.
- Thomas Effland and Michael Collins. 2021. [Partially Supervised Named Entity Recognition via the Expected Entity Ratio Loss](https://aclanthology.org/2021.tacl-1.78/). Transactions of the Association for Computational Linguistics, 9:1320–1335.
- Yuta Tsuboi, Hisashi Kashima, Shinsuke Mori, Hiroki Oda, and Yuji Matsumoto. 2008. [Training Conditional Random Fields Using Incomplete Annotations](https://aclanthology.org/C08-1113/). In Proceedings of the 22nd International Conference on Computational Linguistics (Coling 2008), pages 897–904, Manchester, UK. Coling 2008 Organizing Committee.
