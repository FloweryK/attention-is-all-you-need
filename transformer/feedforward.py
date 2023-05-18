import torch.nn as nn
import torch.nn.functional as F


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, n_emb, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_emb, out_channels=n_emb*4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=n_emb*4, out_channels=n_emb, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (n_batch, n_seq, n_emb)
        # y: (n_batch, n_seq, n_emb)
        x = self.conv1(x.transpose(1, 2))
        x = F.gelu(x)
        x = self.conv2(x).transpose(1, 2)
        x = self.dropout(x)
        return x

