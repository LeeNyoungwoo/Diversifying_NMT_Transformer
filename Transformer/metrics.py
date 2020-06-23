import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

def rouge_compute(reference, translation):
    rouge = Rouge()
    scores = rouge.get_scores(reference, translation)
    return np.array([scores[0]["rouge-l"]["p"], scores[0]["rouge-l"]["r"], scores[0]["rouge-l"]["f"]])

def rfb_bleu_compute(reference, translation):
    reference = reference.split()
    translation = translation.split()

    return corpus_bleu([reference], [translation], smoothing_function=SmoothingFunction().method3)

def pwb_bleu_compute(reference, translation):
    reference = reference.split()
    translation = translation.split()

    return corpus_bleu([reference], [translation], smoothing_function=SmoothingFunction().method7)


def rfb_compute(bleu_score_list):
    return sum(bleu_score_list) / len(bleu_score_list)

def pwb_compute(translation_list):
    pwb_score = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            
            pwb_score.append(pwb_bleu_compute(translation_list[i], translation_list[j]))
    
    assert len(pwb_score) == 6
    
    return sum(pwb_score) / len(pwb_score)

def deq_compute(rfb, rfb_base, pwb, pwb_base):
    return (pwb_base - pwb) / (rfb_base - rfb)