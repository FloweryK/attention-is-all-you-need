import argparse
import torch
import torch.nn as nn
from vocab import VocabBasic, VocabSPM, PAD, BOS, EOS
import config
from transformer import Transformer


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', dest='vocab', default='basic')
    parser.add_argument('-p', '--vocab_path', dest='vocab_path', default='src/vocab/spm/kowiki_8000.model')
    parser.add_argument('-e',  '--enc', dest='x_encs', nargs='+', required=True)
    parser.add_argument('-d',  '--dec', dest='x_decs', nargs='+', required=True)
    args = parser.parse_args()

    # vocab
    if args.vocab == 'basic':
        vocab = VocabBasic(lines=args.x_encs + args.x_decs)
    elif args.vocab == 'sentencepiece':
        vocab = VocabSPM(path=args.vocab_path)

    # test inputs
    x_enc = []
    x_dec = []

    for enc, dec in zip(args.x_encs, args.x_decs):
        x_enc.append(torch.tensor([BOS] + vocab.encode_as_ids(enc) + [EOS]))
        x_dec.append(torch.tensor([BOS] + vocab.encode_as_ids(dec) + [EOS]))

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
