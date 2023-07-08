import argparse
import torch
import torch.nn as nn
from vocab import VocabSPM
from config import Config
from model.transformer import Transformer


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', required=True)
    parser.add_argument('-i'', --inputs', dest='inputs', nargs='+', required=True)
    args = parser.parse_args()

    # vocab
    vocab = VocabSPM(args.model)

    # config
    config = Config(
        n_vocab=len(vocab), 
        n_seq=200,
        n_layer=6,
        n_head=8,
        d_emb=128,
        d_hidden=100,
        scale=100**(1/2),
    )

    # test inputs
    lines = args.inputs

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