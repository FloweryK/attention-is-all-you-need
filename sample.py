import torch
from vocab import VocabBasic, VocabSPM
from transformer.transformer import Transformer


if __name__ == '__main__':
    # test input
    lines = [
        "겨울은 추워요.",
        "아 빨리 끝내고 트위치 방송 보면서 잠이나 자고싶다."
    ]

    # load vocab
    # vocab = VocabSPM("src/vocab/kowiki_8000.model")
    vocab = VocabBasic(lines)

    # encode & decode
    x_enc = []
    x_dec = []

    for line in lines:
        print(vocab.encode_as_pieces(line))
        print(vocab.encode_as_ids(line))

        x_enc.append(torch.tensor(vocab.encode_as_ids(line)))
        x_dec.append(torch.tensor([vocab.piece_to_id("[BOS]")]))
    
    x_enc = torch.nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=0)
    x_dec = torch.nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=0)

    # transformer
    transformer = Transformer(n_vocab=len(vocab), n_seq=256, n_emb=200, n_hidden=100, n_head=2, n_layer=3)

    # test output
    y_dec, probs_enc, probs1_dec, probs2_dec = transformer(x_enc, x_dec)

    print('x_enc:\n', x_enc)
    print('x_dec:\n', x_dec)
    print('y_dec:\n', y_dec)
    print('probs_enc:\n', probs_enc)
    print('probs1_dec:\n', probs1_dec)
    print('probs2_dec:\n', probs2_dec)