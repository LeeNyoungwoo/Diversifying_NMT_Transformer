from torch.utils.data import Dataset
import torch
import pickle as pc

class Our_Handler(Dataset):
    def __init__(self, src_path, tgt_path, vocab, tokenizer, max_len=100, is_test=False):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if not is_test:
            with open(src_path, encoding='utf-8') as f:
                self.src_corpus = [line.strip().split('\n') for line in f.readlines()]

            with open(tgt_path, encoding='utf-8') as f:
                self.tgt_corpus = [line.strip().split('\n') for line in f.readlines()]
        else:
            with open(src_path, 'rb') as f:
                self.src_corpus = pc.load(f)
            
            with open(tgt_path, 'rb') as f:
                self.tgt_corpus = pc.load(f)
    
    def __len__(self):
        return len(self.src_corpus)
    
    def __getitem__(self, item):
        enc_inputs = []
        dec_inputs = []
        dec_outputs = []
        
        for idx, (src_sent, tgt_sent) in enumerate(zip(self.src_corpus[item], self.tgt_corpus[item])):
            if len(src_sent) > self.max_len:
                src_sent = self.truncating(src_sent)
            if len(tgt_sent) > self.max_len:
                tgt_sent = self.truncating(tgt_sent)
            src_ids = self.convert_tokens_to_ids(src_sent)
            tgt_ids = self.convert_tokens_to_ids(tgt_sent)
            
            # make encoder input, in this case, we add just padded tokens
            # we can add <sos> token and <eos> token
            padded_enc_input = self.padding(src_ids)
            
            # make decoder input and output
            # for decoder input, we add <sos> token
            # for decoder output, we add <eos> token
            dec_input = [self.vocab.sos_index] + tgt_ids
            dec_output = tgt_ids + [self.vocab.eos_index]
            
            padded_dec_input = self.padding(dec_input)
            padded_dec_output = self.padding(dec_output)
            
            enc_inputs.append(padded_enc_input)
            dec_inputs.append(padded_dec_input)
            dec_outputs.append(padded_dec_output)
        
        enc_inputs = self.convert_data_type(enc_inputs)
        dec_inputs = self.convert_data_type(dec_inputs)
        dec_outputs = self.convert_data_type(dec_outputs)
        
        return enc_inputs[0], dec_inputs[0], dec_outputs[0]
    
    def convert_data_type(self, inputs):
        return torch.tensor(inputs)
    
    def padding(self, sent):
        return sent + [self.vocab.pad_index for _ in range(self.max_len - len(sent))]
    
    def truncating(self, sent):
        return sent[:self.max_len]
    
    def convert_tokens_to_ids(self, sent):
        tokens = self.tokenizer.tokenizer(sent)
        token2idx = []
        
        for idx, token in enumerate(tokens):
            token2idx.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        
        return token2idx