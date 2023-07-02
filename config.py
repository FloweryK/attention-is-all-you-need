class Config:
    n_layer = 6
    n_head = 8
    d_emb = 128
    d_hidden = 100
    scale = 100**(1/2)

    def __init__(self, n_vocab, n_seq):
        self.n_vocab = n_vocab
        self.n_seq = n_seq