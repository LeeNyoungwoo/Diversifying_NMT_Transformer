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

def convert_tokens_to_ids(vocab, tokenizer, sent):
    tokens = tokenizer.tokenizer(sent)
    token2idx = []

    for idx, token in enumerate(tokens):
        token2idx.append(vocab.stoi.get(token, vocab.unk_index))

    return token2idx

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, vocab, tokenizer):
    
    model.eval()
    indexed = []
    #sentence = SRC.preprocess(sentence)
    print(sentence)
    sentence = convert_tokens_to_ids(vocab, tokenizer, sentence)

    sentence = Variable(torch.LongTensor([sentence]))
    if opt.device == 'cuda':
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, vocab, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
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
                            max_len=256,
                            is_test=True)

    test_dataloader = DataLoader(test_dataset,
                            batch_size=8,
                            shuffle=False,
                            drop_last=True)
    opt.test = test_dataloader
    opt.test_len = len(test_dataloader)
    ####################################################
    model = get_model(opt, len(vocab), len(vocab))
    
    while True:
        opt.text =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text=="q":
            break
        if opt.text=='f':
            fpath =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opt.text = ' '.join(open(opt.text, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase = translate(opt, model, SRC, TRG)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
