# Attention-Is-All-You-Need

PyTorch Implementation of: Vaswani, Ashish, et al. "Attention is all you need." _Advances in neural information processing systems_ 30 (2017).

https://arxiv.org/abs/1706.03762

<br/>

<br/>

## How to use

#### 1. select Vocab model

- you can choose `VocabBasic` or `VocabSPM`
- `VocabBasic`: character-level vocab model without using any sentence processor
- `VocabSPM`: token-level vocab model using [sentencepiece](https://github.com/google/sentencepiece) (trained with [kowiki](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C))

<br/>

#### 2. set input lines

- set input lines in `lines` from `sample.py`

<br/>

#### 3. run `sample.py`

- you will get returns from transformer

<br/>

<br/>

## References

- https://paul-hyun.github.io/transformer-01/
- https://wikidocs.net/31379
