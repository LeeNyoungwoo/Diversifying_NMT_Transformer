import argparse
import time
import torch
from Models import get_model
from Process import *
from spm_enko_tokenize import *
from spm_vocab import *
from spm_enko_handler import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from torch.utils.data import DataLoader
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam_ko import beam_search
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
    
    for mo in text:
        print(mo)
        print(dict[mo[mo.startswith():mo.endswith()]])
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
    
    model.eval()
    indexed = []
    #sentence = SRC.preprocess(sentence)

    sentence = convert_tokens_to_ids(src_vocab, src_tokenizer, sentence)

    sentence = Variable(torch.LongTensor([sentence]))
    sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, src_vocab, trg_vocab, opt)

    #return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)
    
    return sentence

def translate(opt, model, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences[:1]:
        print('sentence: ', sentence)
        result = translate_sentence(sentence + '.', model, opt, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer)
        result = decode_tokens(trg_tokenizer, result)
        for each in result[:opt.k]:
            print('each: ', each.capitalize())
            translated.append(each.capitalize())
    
    print('translated: ', translated)

    #return (' '.join(translated))
    return translated

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-bleu', type=bool, default=False)
    
    opt = parser.parse_args()

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Load Tokenizer and Vocab...')
    en_sp_tokenizer = Tokenizer(is_train=False, model_prefix='spm_en')
    ko_sp_tokenizer = Tokenizer(is_train=False, model_prefix='spm_ko')

    print(f'Load the extended vocab...')
    en_vocab = Vocabulary.load_vocab('./ko_data/en_vocab')
    ko_vocab = Vocabulary.load_vocab('./ko_data/ko_vocab')
    
    ######################TEST DATA######################
    # fitting the test dataset dir
    test_data_dir = ['./ko_data/test/test_ko_en.en', './ko_data/test/test_ko_en.ko']
    
    with open(test_data_dir[1], encoding='utf-8') as f:
        src_corpus = [line.strip().split('\n') for line in f.readlines()]
    with open(test_data_dir[0], encoding='utf-8') as f:
        tgt_corpus = [line.strip().split('\n') for line in f.readlines()]
    ####################################################
    model = get_model(opt, len(ko_vocab), len(en_vocab))
    
    #opt.text = 'How are you?'
    if opt.bleu == False:
        pred = []
        refer = []
        for batch_idx, (text, tgt) in enumerate(zip(src_corpus, tgt_corpus)):

            opt.text = text[0]
            print('text: ', text)
            print('tgt: ', tgt)

            phrase = translate(opt, model, ko_vocab, en_vocab, ko_sp_tokenizer, en_sp_tokenizer)

            pred.append(phrase)
            refer.append(tgt)

        #phrase = translate(opt, model, vocab, sp_tokenizer)
        with open('translation_result', 'w') as file:
            for data in pred:
                file.write(data[0]+'\n')
        with open('reference_result', 'w') as file:
            for data in refer:
                file.write(data[0]+'\n')
    else:
        with open('translation_result', 'rb') as f:
            translation = pc.load(f)
        with open('reference_result', 'rb') as f:
            reference = pc.load(f)
            
        compute_metrics(reference, translation)
    
def filtering(sentence):
    filter_sent = []
    for sent in sentence:
        if sent == '':
            continue
        filter_sent.append(sent)
    
    return filter_sent[:3]

def compute_metrics(reference, translation):
    bleu_result = []
    for idx, (refer, trans) in enumerate(zip(reference, translation)):
        bleu_result.append(rfb_bleu_compute(refer[0], trans[0]))
    
    print(sum(bleu_result))
    print(f'RFB result: {rfb_compute(bleu_result):.3f}')
    pwb_result = []
    
    print(len(translation))
    for idx, each in enumerate(translation):

        each = filtering(each)

        pwb_result.append(pwb_compute(each))
    print(f'PWB result: {sum(pwb_result):.3f}')
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
