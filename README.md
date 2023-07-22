# Attention-Is-All-You-Need

PyTorch Implementation of: Vaswani, Ashish, et al. "Attention is all you need." _Advances in neural information processing systems_ 30 (2017).

https://arxiv.org/abs/1706.03762

<br/>

<br/>

## A minimal usage of transformer

- you can customize the transformer's configuration in `config.py`

  ```python
  # model
  n_vocab = 8000+7 # only for sentencepiece
  n_seq = 1000
  n_layer = 6
  n_head = 8
  d_emb = 512
  d_hidden = 2048
  dropout = 0.1
  scale = (512//8)**(1/2)
  ```

- then you can use the transformer as follows:

  ```python
  import config
  from transformer import Transformer
  
  model = Transformer(config)
  ```

- the required input & ouput format is as follows:

  ```python
  import torch
  
  # encoder input: should be a list of list (of encoded ids of encoder input)
  x_enc = torch.tensor([[2, 10, 6714, 3], [2, 7, 3, 0]])		# torch.Size([2, 4])
  
  # decoder input: should be a list of list (of encoded ids of decoder input)
  x_dec = torch.tensor([[2, 68, 3, 0, 0, 0], [2, 182, 4922, 1032, 4, 3]])		# torch.Size([2, 6])
  
  # decoder output (result)
  y_dec = model(x_enc, x_dec)		# torch.Size([2, 6, 512])
  ```

<br/>

<br/>

## How to use sample.py

| flag             | description              | example                                             | default                           |
| ---------------- | ------------------------ | --------------------------------------------------- | --------------------------------- |
| -v, --vocab      | vocab model name         | `sentencepiece`                                     | `basic`                           |
| -p, --vocab_path | vocab model path         | `src/vocab/spm/kowiki_8000.model`                   | `src/vocab/spm/kowiki_8000.model` |
| -e, --enc        | encoder inputs (as list) | `"the truth is" "hello from the other side"`        | required                          |
| -d, --dec        | decoder inputs (as list) | `"I am Iron man" "at least i can say that i tried"` | required                          |

<br/>

<br/>

## References

- https://paul-hyun.github.io/transformer-01/
- https://wikidocs.net/31379
