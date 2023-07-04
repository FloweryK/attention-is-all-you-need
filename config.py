class Config:
    def __init__(self, n_vocab, n_seq, n_layer, n_head, d_emb, d_hidden, scale, **kwargs):
        self.n_vocab = n_vocab
        self.n_seq = n_seq
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_emb = d_emb
        self.d_hidden = d_hidden
        self.scale = scale

        for key, value in kwargs.items():
            setattr(self, key, value)