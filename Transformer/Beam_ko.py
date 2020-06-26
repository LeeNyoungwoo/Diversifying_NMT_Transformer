import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(src, model, vocab, opt):
    
    init_tok = vocab.stoi['<sos>']
    src_mask = (src != vocab.stoi['<pad>']).unsqueeze(-2)
    #print(src)
    #print(f'src: {src.size()}, src_mask: {src_mask.size()}')
    e_output = model.encoder(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]])
    outputs = outputs.cuda()
    
    trg_mask = nopeak_mask(1, opt)
    
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask, policy=True))
    out = F.softmax(out, dim=-1)
    #print(f'out : {out.size()}')
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long()
    outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, s_vocab, t_vocab, opt):
    
    outputs, e_outputs, log_scores = init_vars(src, model, s_vocab, opt)

    #print('\nbeam_search start\n')
    #print(f'outputs: {outputs.size()}')
    #print(f'e_outputs: {e_outputs.size()}')
    #print(f'log_scores: {log_scores.size()}')
    eos_tok = s_vocab.stoi['<eos>']
    src_mask = (src != s_vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
    
        trg_mask = nopeak_mask(i, opt)

        out = model.out(model.decoder(outputs[:,:i], e_outputs, src_mask, trg_mask, policy=True))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    length = []

    if ind is None:
        for output in outputs:
#             out_list = output.tolist()
#             out_list.append(3)
#             output = torch.Tensor(out_list)
            length.append((output==eos_tok).nonzero()[0])
        
        result = []
        for idx in range(opt.k):
            result.append(' '.join([t_vocab.itos[tok] for tok in outputs[idx][1:length[idx]]]))
        #return ' '.join([vocab.itos[tok] for tok in outputs[ind][1:length]])

        return result

    else:
        for output in outputs:
            length.append((output==eos_tok).nonzero()[0])
        
        result = []
        for idx in range(opt.k):
            result.append(' '.join([t_vocab.itos[tok] for tok in outputs[idx][1:length[idx]]]))
        #return ' '.join([vocab.itos[tok] for tok in outputs[ind][1:length]])

        return result
