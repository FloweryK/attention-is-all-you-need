import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, n_vocab, n_emb):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_emb)

    def forward(self, x):
        # x: (n_batch, n_seq) -> (n_batch, n_seq, n_emb)
        x = self.embedding(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, n_seq, n_emb):
        super().__init__()
        
        # create positional embeddings
        i_seq = torch.arange(n_seq, dtype=torch.float).unsqueeze(1)
        i_emb = torch.pow(10000, -torch.arange(0, n_emb, dtype=torch.float) / n_emb)
        embeddings = torch.zeros(n_seq, n_emb)
        embeddings[:, 0::2] = torch.sin(i_seq * i_emb[0::2])
        embeddings[:, 1::2] = torch.sin(i_seq * i_emb[1::2])

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
    
    def forward(self, x):
        n_batch = x.size(0)
        n_seq = x.size(1)

        # i_seq: (n_batch, n_seq)
        i_seq = torch.arange(n_seq, device=x.device, dtype=x.dtype).expand(n_batch, n_seq).contiguous() + 1
        i_seq = i_seq.masked_fill(x.eq(0), 0)

        # x: (n_batch, n_seq) -> (n_batch, n_seq, n_emb)
        x = self.embedding(i_seq)
        return x

