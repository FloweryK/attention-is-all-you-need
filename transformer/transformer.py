import torch.nn as nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, n_vocab, n_seq, n_emb, n_hidden, n_head, n_layer):
        super().__init__()

        self.encoder = Encoder(n_vocab, n_seq, n_emb, n_hidden, n_head, n_layer)
        self.decoder = Decoder(n_vocab, n_seq, n_emb, n_hidden, n_head, n_layer)

    def forward(self, x_inputs, x_outputs):
        y_enc, probs_enc = self.encoder(x_inputs)
        y_dec, probs1_dec, probs2_dec = self.decoder(x_outputs, x_inputs, y_enc)

        return y_dec, probs_enc, probs1_dec, probs2_dec