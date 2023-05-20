import sentencepiece as spm


class VocabBasic:
    def __init__(self, lines):
        self.char2id = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[SEP]": 4,
            "[CLS]": 5,
            "[MASK]": 6,
        }
        self.id2char = {}

        self.make_ids(lines)
    
    def __len__(self):
        return len(self.char2id)
    
    def make_ids(self, lines):
        id_cur = 0

        # make char2id
        for line in lines:
            for char in line:
                if char not in self.char2id:
                    self.char2id[char] = id_cur
                    id_cur += 1
        
        # make id2char
        for id, char in self.char2id.items():
            self.id2char[id] = char
    
    def encode_as_pieces(self, line):
        return [char for char in line]

    def encode_as_ids(self, line):
        return [self.char2id[char] for char in line]

    def piece_to_id(self, piece):
        return self.char2id[piece]


class VocabSPM(spm.SentencePieceProcessor):
    def __init__(self, path):
        super().__init__()
        self.load(path)
