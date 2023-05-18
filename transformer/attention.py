import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, n_hidden, dropout=0.1):
        super().__init__()
        self.scale = 1 / (n_hidden ** 0.5)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask):
        # Q, K, V: (n_batch, n_seq, n_emb)
        # scores: (n_batch, n_seq, n_seq)
        score = torch.matmul(Q, K.transpose(-1, -2))
        score = torch.mul(score, self.scale)
        score = score.masked_fill(mask, -1e9)

        # probs: (n_batch, n_seq, n_seq)
        prob = F.softmax(score, dim=-1)
        prob = self.dropout(prob)

        # context: (n_batch, n_seq, n_emb)
        context = torch.matmul(prob, V)

        return context, prob


class MultiHeadAttention(nn.Module):
    def __init__(self, n_emb, n_hidden, n_head, dropout=0.1):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_head = n_head

        self.linear_Q = nn.Linear(n_emb, n_head * n_hidden)
        self.linear_K = nn.Linear(n_emb, n_head * n_hidden)
        self.linear_V = nn.Linear(n_emb, n_head * n_hidden)
        self.attention = ScaledDotProductAttention(n_hidden)
        self.linear = nn.Linear(n_head * n_hidden, n_emb)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, Q, K, V, mask):
        n_batch = Q.size(0)

        # Q, K, V: (n_batch, n_seq, n_emb)
        # q, k, v: (n_batch, n_head, n_seq, n_hidden)
        q = self.linear_Q(Q).view(n_batch, -1, self.n_head, self.n_hidden).transpose(1, 2)
        k = self.linear_K(K).view(n_batch, -1, self.n_head, self.n_hidden).transpose(1, 2)
        v = self.linear_V(V).view(n_batch, -1, self.n_head, self.n_hidden).transpose(1, 2)

        # mask: (n_batch, n_seq, n_seq) -> (n_batch, n_head, n_seq, n_seq)
        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # prob: (n_batch, n_head, n_seq, n_seq)
        # context: (n_batch, n_head, n_seq, n_hidden)
        context, prob = self.attention(q, k, v, mask)

        # context: (n_batch, n_head, n_seq, n_hidden) -> (n_batch, n_seq, n_head * n_hidden)
        context = context.transpose(1, 2).contiguous().view(n_batch, -1, self.n_head * self.n_hidden)

        # 굳이 multihead를 쓸 필요가 있었나?

        # y: (n_batch, n_seq, n_emb)
        y = self.dropout(self.linear(context))

        return y, prob



        
