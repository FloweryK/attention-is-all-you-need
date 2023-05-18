import torch
import torch.nn as nn
from transformer.attention import MultiHeadAttention
from transformer.embedding import InputEmbedding, PositionalEmbedding
from transformer.feedforward import PoswiseFeedForwardNet


class DecoderLayer(nn.Module):
    def __init__(self, n_emb, n_hidden, n_head, eps=1e-12):
        super().__init__()

        self.attention1 = MultiHeadAttention(n_emb, n_hidden, n_head)
        self.norm1 = nn.LayerNorm(n_emb, eps)
        self.attention2 = MultiHeadAttention(n_emb, n_hidden, n_head)
        self.norm2 = nn.LayerNorm(n_emb, eps)
        self.feedforward = PoswiseFeedForwardNet(n_emb)
        self.norm3 = nn.LayerNorm(n_emb, eps)

    
    def forward(self, x_dec, mask_dec, y_enc, mask_enc):
        y1_dec, prob1_dec = self.attention1(x_dec, x_dec, x_dec, mask_dec)
        y1_dec = self.norm1(x_dec + y1_dec)

        y2_dec, prob2_dec = self.attention2(y1_dec, y_enc, y_enc, mask_enc)
        y2_dec = self.norm2(y1_dec + y2_dec)

        y = self.norm3(y2_dec + self.feedforward(y2_dec))

        return y, prob1_dec, prob2_dec
        


class Decoder(nn.Module):
    def __init__(self, n_vocab, n_seq, n_emb, n_hidden, n_head, n_layer):
        super().__init__()

        self.embedding_input = InputEmbedding(n_vocab, n_emb)
        self.embedding_positional = PositionalEmbedding(n_seq, n_emb)
        self.layers = nn.ModuleList([DecoderLayer(n_emb, n_hidden, n_head) for _ in range(n_layer)])

    def forward(self, x_dec, x_enc, y_enc):
        # x_dec: (n_batch, n_seq_dec)
        # x_enc: (n_batch, n_seq_enc, n_emb)
        # y_enc: (n_batch, n_seq_enc, n_emb)

        # mask_dec: (n_batch, n_seq_dec, n_seq_dec)
        mask_dec_pad = x_dec.eq(0).unsqueeze(1).expand(x_dec.size(0), x_dec.size(1), x_dec.size(1))
        mask_dec_tri = torch.ones_like(mask_dec_pad).triu(diagonal=1)
        mask_dec = torch.gt(mask_dec_pad + mask_dec_tri, 0)
        
        # mask_enc: (n_batch, n_seq_enc, n_seq_dec)
        mask_enc = x_enc.eq(0).unsqueeze(1).expand(x_enc.size(0), x_dec.size(1), x_enc.size(1))

        # embedding: (n_batch, n_seq_dec, n_emb)
        y_dec = self.embedding_input(x_dec) + self.embedding_positional(x_dec)

        # layers
        probs1_dec, probs2_dec = [], []
        for layer in self.layers:
            y_dec, prob1_dec, prob2_dec = layer(y_dec, mask_dec, y_enc, mask_enc)
            probs1_dec.append(prob1_dec)
            probs2_dec.append(prob2_dec)

        return y_dec, probs1_dec, probs2_dec