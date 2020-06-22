import argparse
import time
import torch
from Models import get_model
from Process import *
from spm_tokenize import *
from spm_vocab import *
from spm_handler import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from torch.utils.data import DataLoader
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
import pickle as pc
from metrics import *

def convert_tokens_to_ids(vocab, tokenizer, sent):
    tokens = tokenizer.tokenizer(sent)
    token2idx = []

    for idx, token in enumerate(tokens):
        token2idx.append(vocab.stoi.get(token, vocab.unk_index))

    return token2idx

def decode_tokens(tokenizer, sentences):
    result = []
    for sent in sentences:
        sent = sent.split(' ')
        
        result.append(tokenizer.tokenize.DecodePieces(sent))
    return result

def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    
    print(regex)
    print(text)
    for mo in text:
        print(mo)
        print(dict[mo[mo.startswith():mo.endswith()]])
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, vocab, tokenizer):
    
    model.eval()
    indexed = []
    #sentence = SRC.preprocess(sentence)

    sentence = convert_tokens_to_ids(vocab, tokenizer, sentence)

    sentence = Variable(torch.LongTensor([sentence]))
    sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, vocab, opt)

    #return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)
    
    return sentence

def translate(opt, model, vocab, tokenizer):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        result = translate_sentence(sentence + '.', model, opt, vocab, tokenizer)
        

        result = decode_tokens(tokenizer, result)
        for each in result[:opt.k]:
            translated.append(each.capitalize())

    #return (' '.join(translated))
    return translated

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
    
    with open(test_data_dir[0], 'rb') as f:
        src_corpus = pc.load(f)
    with open(test_data_dir[1], 'rb') as f:
        tgt_corpus = pc.load(f)
    ####################################################
    model = get_model(opt, len(vocab), len(vocab))
    
    #opt.text = 'How are you?'
    
    pred = []
    refer = []
    for batch_idx, (text, tgt) in enumerate(zip(src_corpus, tgt_corpus)):

        opt.text = text[0]
        
        phrase = translate(opt, model, vocab, sp_tokenizer)
        
        pred.append(phrase)
        refer.append(tgt)
        
    #phrase = translate(opt, model, vocab, sp_tokenizer)

    with open('translation_result', 'wb') as f:
        pc.dump(pred, f)
    with open('reference_result', 'wb') as f:
        pc.dump(refer, f)
        
    #compute_metrics(refer, pred)
    
def compute_metrics(reference, translation):
    bleu_result = []
    for idx, (refer, trans) in enumerate(zip(reference, translation)):
        for each in trans:
            bleu_result.append(bleu_compute(refer[0], each))
    
#    print(f'RFB result: {rfb_compute(bleu_result):.3f}')
    pwb_result = []
    
    print(len(translation))
    for idx, each in enumerate(translation):
        pwb_result.append(pwb_compute(each))
    print(f'PWB result: {sum(pwb_result):.3f}')
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
