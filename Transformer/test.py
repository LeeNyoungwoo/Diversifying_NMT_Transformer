import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
from torch.utils.data import DataLoader

from spm_tokenize import *
from spm_vocab import *
from spm_handler import *
import os
import numpy as np

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-max_len', type=int, default=32)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    
    opt = parser.parse_args()

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Load Tokenizer and Vocab...')
    sp_tokenizer = Tokenizer(is_train=False, model_prefix='spm')
    sp_vocab = sp_tokenizer.vocab
    
    print(f'Load the extended vocab...')
    vocab = Vocabulary.load_vocab('./data/vocab')
    
    ######################TEST DATA######################
    # fitting the test dataset dir
    test_data_dir = ['./data/test/newstest2014_en', './data/test/newstest2014_de']
    test_dataset = Our_Handler(src_path=test_data_dir[0], 
                            tgt_path=test_data_dir[1],
                            vocab=vocab, 
                            tokenizer=sp_tokenizer,
                            max_len=32,
                            is_test=True)

    test_dataloader = DataLoader(test_dataset,
                            batch_size=64,
                            shuffle=False,
                            drop_last=True)
    opt.test = test_dataloader
    opt.test_len = len(test_dataloader)
    ####################################################
    model = get_model(opt, len(vocab), len(vocab))
    
    model.eval()
    opt.src_pad = opt.trg_pad = vocab.pad_index

    test_loss = 0.
    test_ppl = 0.
    for batch_idx, (enc_input, dec_input, dec_output) in enumerate(opt.test):

        enc_input = enc_input.to(opt.device)
        dec_input = dec_input.to(opt.device)
        dec_output = dec_output.to(opt.device)

        src_mask, trg_mask = create_masks(enc_input, dec_input, opt)

        with torch.no_grad():
            preds = model(enc_input, dec_input, src_mask, trg_mask)

        ys = dec_output.contiguous().view(-1)

        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)

        test_loss += loss.item()
        test_ppl += np.exp(loss.item())
        
    avg_test_loss = test_loss / len(opt.test)
    avg_ppl = test_ppl / len(opt.test)
    print(f'Test loss: {avg_test_loss:.3f}, Test perpelxity: {avg_ppl:.3f}')
    
if __name__ == '__main__':
    main()
