import argparse
import time, os
import torch
from torch import nn

from Models import get_model
#from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle
from torch.utils.data import DataLoader
from spm_tokenize import *
from spm_vocab import *
from spm_handler import *
from tqdm import tqdm, trange

def train_model(model, opt):
    
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    loss_log = tqdm(total=0, bar_format='{desc}', position=2)   
    # for epoch in range(opt.epochs):
    for epoch in trange(opt.epochs, desc="Epoch", position=0):
        if epoch == 1:
            break
        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
                    
        for batch_idx, (enc_input, dec_input, dec_output) in enumerate(opt.train): 
            if batch_idx == 1:
                break
                
            enc_input = enc_input.transpose(0,1).to(opt.device)
            dec_input = dec_input.transpose(0,1).to(opt.device)
            dec_output = dec_output.to(opt.device)
            
            #trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(enc_input, dec_input, opt)
            
            preds = model(enc_input, dec_input, src_mask, trg_mask)
            
            #ys = trg[:, 1:].contiguous().view(-1)
            ys = dec_output.contiguous().view(-1)

            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()
            
            total_loss += loss.item()
            avg_loss = total_loss/opt.printevery
            
            if (batch_idx + 1) % opt.printevery == 0:
                p = int(100 * (batch_idx + 1) / opt.train_len)
                if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
                else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                total_loss = 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('--is_train', type=bool, default=False)
    
    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # baseline code
    #read_data(opt)
    #SRC, TRG = create_fields(opt)
    #opt.train = create_dataset(opt, SRC, TRG)
    
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # modified version for tokenize, vocab, dataset
    if opt.is_train:
        filenames = ['data/europarl-v7.de-en.en', 'data/europarl-v7.de-en.de']
        
        print(f'Build Tokenizer and Vocab...')
        sp_tokenizer = Tokenizer(is_train=True, filenames=filenames, tokenizer_type='spm', input_file='./data/concat_data.txt', model_prefix='spm', vocab_size=32000, model_type='bpe')
        sp_vocab = sp_tokenizer.vocab
    else:
        print(f'Load Tokenizer and Vocab...')
        sp_tokenizer = Tokenizer(is_train=False, model_prefix='spm')
        sp_vocab = sp_tokenizer.vocab
    
    if opt.is_train:
        print(f'Extend Vocab...')
        vocab = Vocabulary(sp_vocab)
        vocab.save_vocab('./data/vocab')
    else:
        print(f'Load the extended vocab...')
        vocab = Vocabulary.load_vocab('./data/vocab')
    
    train_dataset = Our_Handler(src_path='./data/europarl-v7.de-en.en', tgt_path='./data/europarl-v7.de-en.de', vocab=vocab, tokenizer=sp_tokenizer, max_len=256)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    opt.train = train_dataloader
    opt.src_pad = opt.trg_pad = vocab.pad_index
    opt.train_len = len(train_dataloader)
    
    ######################DEV DATA######################
    # fitting the dev dataset dir
    dev_data_dir = ['data/dev/newstest2013.en', 'data/dev/newstest2013.de']
    dev_dataset = Our_Handler(src_path=dev_data_dir[0], 
                            tgt_path=dev_data_dir[1],
                            vocab=vocab, 
                            tokenizer=sp_tokenizer,
                            max_len=256)
    
    dev_dataloader = DataLoader(dev_dataset,
                            batch_size=32,
                            shuffle=False,
                            drop_last=True)
    
    ####################################################
    
    ######################TEST DATA######################
    # fitting the test dataset dir
    test_data_dir = ['data/test/newstest2014.en', 'data/test/newstest2014.de']
    test_dataset = Our_Handler(src_path=test_data_dir[0], 
                            tgt_path=test_data_dir[1],
                            vocab=vocab, 
                            tokenizer=sp_tokenizer,
                            max_len=256,
                            is_test=True)
    
    test_dataloader = DataLoader(test_dataset,
                            batch_size=32,
                            shuffle=False,
                            drop_last=True)
    
    ####################################################
    #model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    model = get_model(opt, len(sp_vocab), len(sp_vocab))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Use {torch.cuda.device_count()} GPUs')
        
    model = model.to(opt.device)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt)

    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
