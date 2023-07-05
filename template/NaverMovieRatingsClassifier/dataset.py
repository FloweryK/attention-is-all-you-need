import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RatingsDataset(Dataset):
    def __init__(self, vocab, path):
        super().__init__()

        self.vocab = vocab
        self.path = path
        self.labels = []
        self.sentences = []

        self.load()
        self.len = len(self.labels)
    
    def load(self):        
        df = pd.read_csv(self.path, sep='\t', engine='python')

        # pre-encode dataset
        for _, row in tqdm(df.iterrows()):
            if type(row['document']) != str:
                continue

            label = row['label']
            pieces = [self.vocab.piece_to_id(piece) for piece in row['document']]

            self.labels.append(label)
            self.sentences.append(pieces)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        label = self.labels[index]
        x_enc = self.sentences[index]
        x_dec = [self.vocab.piece_to_id("[BOS]")]
        return (
            torch.tensor(label),
            torch.tensor(x_enc),
            torch.tensor(x_dec)
        )


def collate_fn(inputs):
    labels, enc_inputs, dec_inputs = list(zip(*inputs))

    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [torch.stack(labels, dim=0), enc_inputs, dec_inputs]
    return batch