import pickle

class Vocabulary(object):
    def __init__(self, vocab, specials=['<pad>', '<sos>', '<eos>']):
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2
        self.eos_index = 3
        
        self.itos = list(specials)
        for word in vocab:
            self.itos.append(word)
        
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)
    
    def save_vocab(self, vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(self, f)