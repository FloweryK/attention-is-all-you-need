import argparse
import torch
import torch.nn as nn
import config
from vocab import VocabSPM, BOS, PAD
from transformer import Transformer


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', dest='vocab', required=True)
    parser.add_argument('-e',  '--enc', dest='x_encs', nargs='+', required=True)
    parser.add_argument('-d',  '--dec', dest='x_decs', nargs='+', required=True)
    args = parser.parse_args()

    # vocab
    vocab = VocabSPM(args.vocab)

    # test inputs
    x_enc = []
    x_dec = []

    for enc, dec in zip(args.x_encs, args.x_decs):
        x_enc.append(torch.tensor(vocab.EncodeAsIds(enc)))
        x_dec.append(torch.tensor(vocab.EncodeAsIds(dec)))

    x_enc = nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=PAD)
    x_dec = nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=PAD)

    # transformer
    transformer = Transformer(config)

    # test outputs
    y_dec = transformer(x_enc, x_dec)

    assert len(args.x_encs) == len(args.x_decs)
    
    for i in range(len(args.x_encs)):
        print(f"\n#{i}")
        print('\t x_enc:', x_enc[i])
        print('\t x_dec:', x_dec[i])
        print('\t y_dec:', y_dec[i])
