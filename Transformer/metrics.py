import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def bleu_compute(reference, translation):
    reference = reference.split()
    translation = translation.split()
    
    return sentence_bleu([reference], translation, smoothing_function=SmoothingFunction().method7, weights=[1./3, 1./3, 1./3])

def rfb_compute(bleu_score_list):
    return sum(bleu_score_list) / len(bleu_score_list)

def pwb_compute(translation_list):
    pwb_score = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            
            pwb_score.append(bleu_compute(translation_list[i], translation_list[j]))
    
    assert len(pwb_score) == 20
    
    return sum(pwb_score) / len(pwb_score)

def deq_compute(rfb, rfb_base, pwb, pwb_base):
    return (pwb_base - pwb) / (rfb_base - rfb)