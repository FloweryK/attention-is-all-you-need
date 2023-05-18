import torch.nn as nn
from transformer.attention import MultiHeadAttention
from transformer.embedding import InputEmbedding, PositionalEmbedding
from transformer.feedforward import PoswiseFeedForwardNet


class EncoderLayer(nn.Module):
    def __init__(self, n_emb, n_hidden, n_head, eps=1e-12):
        super().__init__()

        self.attention = MultiHeadAttention(n_emb, n_hidden, n_head)
        self.norm1 = nn.LayerNorm(n_emb, eps)
        self.feedforward = PoswiseFeedForwardNet(n_emb)
        self.norm2 = nn.LayerNorm(n_emb, eps)
    
    def forward(self, x, mask):
        # x:    (n_batch, n_seq, n_emb)
        # mask: (n_batch, n_seq, n_seq)
        # prob: (n_batch, n_seq, n_seq)
        # y:    (n_batch, n_seq, n_emb)
        y, prob = self.attention(x, x, x, mask)
        y = self.norm1(x + y)
        y = self.norm2(y + self.feedforward(y))

        return y, prob
        


class Encoder(nn.Module):
    def __init__(self, n_vocab, n_seq, n_emb, n_hidden, n_head, n_layer):
        super().__init__()

        self.embedding_input = InputEmbedding(n_vocab, n_emb)
        self.embedding_positional = PositionalEmbedding(n_seq, n_emb)
        self.layers = nn.ModuleList([EncoderLayer(n_emb, n_hidden, n_head) for _ in range(n_layer)])

    def forward(self, x):
        # x: (n_batch, n_seq)

        # mask: (n_batch, n_seq, n_seq)
        mask = x.eq(0).unsqueeze(1).expand(x.size(0), x.size(1), x.size(1))

        # embedding: (n_batch, n_seq, n_emb)
        y = self.embedding_input(x) + self.embedding_positional(x)

        # layers
        probs = []
        for layer in self.layers:
            y, prob = layer(y, mask)
            probs.append(prob)

        return y, probs