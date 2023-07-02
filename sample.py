import torch
import torch.nn as nn
from vocab import VocabSPM
from model.transformer import Transformer


if __name__ == '__main__':
    # vocab
    vocab = VocabSPM("src/vocab/kowiki_8000.model")

    # config
    config = dict(
        n_vocab=len(vocab),
        n_seq=200,
        n_layer=6,
        n_head=8,
        d_emb=128,
        d_hidden=100,
        scale=100**(1/2),
    )

    # test inputs
    lines = [
        "겨울은 추워요.",
        "아 빨리 끝내고 트위치 방송 보면서 잠이나 자고싶다."
    ]

    x_enc = []
    x_dec = []

    for line in lines:
        x_enc.append(torch.tensor(vocab.EncodeAsIds(line)))
        x_dec.append(torch.tensor([vocab.PieceToId('[BOS]')]))

    x_enc = nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=0)
    x_dec = nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=0)

    # transformer
    transformer = Transformer(config)

    # test outputs
    y_dec = transformer(x_enc, x_dec)

    print(y_dec.shape)