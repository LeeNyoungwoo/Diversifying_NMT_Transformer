import spacy
import re
import sentencepiece as spm

class tokenize(object):
    def __init__(self, lang=None, tokenize_type='spm', input_file=None, model_prefix=None, vocab_size=None, model_type=None):
        
        self.tokenizer_type = tokenizer_type
        
        # For processing SentencePiece
        self.input_file = input_file
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        if tokenizer_type == 'spacy':
            self.tokenize = spacy.load(lang)
        elif tokenizer_type == 'spm':
            self.tokenize = build_sp()
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        
        if self.tokenizer_type == 'spacy':
            return [tok.text for tok in self.tokenize.tokenizer(sentence) if tok.text != " "]
        elif self.tokenizer_type == 'spm':
            return [tok for tok in self.tokenize.encode(sentence)]
            
    def build_sp(self):
        templates = '--input={} --model_prefix={} --vocab_size={} --model_type={}'
        cmd = templates.format(self.input_file, self.model_prefix, self.vocab_size, self.model_type)
        spm.SentencePieceTrainer.Train(cmd)
        
        sp = spm.SentencePieceProcessor()
        sp.Load('{}.model'.format(self.model_prefix))
        with open('{}.vocab'.format(self.model_prefix), encoding='utf-8') as f:
            vocabs = [doc.strip().split('\t')[0] for doc in f]
        
        return sp
        
        
