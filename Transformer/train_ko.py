import argparse
import time, os
import torch
from torch import nn

from Models import get_model
#from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts,Triangular
from Batch import create_masks
import dill as pickle
import pandas as pd
from torch.utils.data import DataLoader
from spm_enko_tokenize import *
from spm_vocab import *
from spm_enko_handler import *
from tqdm import tqdm, trange
from nltk.translate.bleu_score import corpus_bleu

# Early Stopping
class EarlyStopping():
    def __init__(self, patience=30, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience  = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
    
# Save csv file
def write_csv_file(loss, loss_name='loss'):
    df = pd.DataFrame({loss_name: loss})
    df.to_csv(loss_name+'_result.csv', index=False, encoding='UTF8')

def train_model(model, opt):
    
    print("training model...")
    # model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    early_stopping = EarlyStopping(patience=3, verbose=1)

    loss_log = tqdm(total=0, bar_format='{desc}', position=2)
    
    training_loss_list = []
    val_loss_list = []
    
    for epoch in range(opt.epochs):
    # for epoch in trange(opt.epochs, desc="Epoch", position=0):
        
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        # Training the model
        model.train()
        total_loss = 0
                    
        for batch_idx, (enc_input, dec_input, dec_output) in enumerate(opt.train): 
        # for batch_idx, (enc_input, dec_input, dec_output) in enumerate(tqdm(opt.train, desc="Iteration", ncols=100, position=1)):
            
            enc_input = enc_input.to(opt.device)
            dec_input = dec_input.to(opt.device)
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
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                training_loss_list.append(avg_loss)
                total_loss = 0
    
        ## Validating the model
        model.eval()
        val_loss = 0
        val_step = 0
        early_stopped = False
        
        with torch.no_grad():
            for batch_idx, (enc_input, dec_input, dec_output) in enumerate(opt.validation): 
            # for batch_idx, (enc_input, dec_input, dec_output) in enumerate(tqdm(opt.validation, desc="Iteration", ncols=100, position=1)):
                enc_input = enc_input.to(opt.device)
                dec_input = dec_input.to(opt.device)
                dec_output = dec_output.to(opt.device)

                src_mask, trg_mask = create_masks(enc_input, dec_input, opt)

                preds = model(enc_input, dec_input, src_mask, trg_mask)

                ys = dec_output.contiguous().view(-1)

                loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
                val_loss += loss.item()
                val_step += 1

        val_loss = val_loss/val_step
        val_loss_list.append(val_loss)
        print("epoch %d, loss = %.3f" %(epoch+1, val_loss))
        
        dst = 'koen_100len_32batch'
        print("saving weights to " + dst + "/...")
        torch.save(model.state_dict(), f'{dst}/model_weights')
        print("weights and field pickles saved to " + dst)

        if early_stopping.validate(val_loss):
            break

    write_csv_file(training_loss_list, 'train_loss')
    write_csv_file(val_loss_list, 'val_loss')

def test_model(model, opt):
    print("Testing model...")
    model.eval()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    loss_log = tqdm(total=0, bar_format='{desc}', position=2)   
    # for epoch in range(opt.epochs):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (enc_input, dec_input, dec_output) in enumerate(tqdm(opt.test, desc="Iteration", position=0)):
            enc_input = enc_input.to(opt.device)
            dec_input = dec_input.to(opt.device)
            dec_output = dec_output.to(opt.device)

            src_mask, trg_mask = create_masks(enc_input, dec_input, opt)

            preds = model(enc_input, dec_input, src_mask, trg_mask)

            ys = dec_output.contiguous().view(-1)

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            test_loss += loss.item()
            test_avg_loss = test_loss/opt.printevery

            if (batch_idx + 1) % opt.printevery == 0:
                p = int(100 * (batch_idx + 1) / opt.test_len)
                test_loss = 0
        
        print("%dm: Evaluated loss = %.3f\n" %\
        ((time.time() - start)//60, test_avg_loss))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=500)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--is_test', type=bool, default=False)
    
    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # baseline code
    #read_data(opt)
    #SRC, TRG = create_fields(opt)
    #opt.train = create_dataset(opt, SRC, TRG)
    
    if not os.path.exists('./ko_data'):
        os.makedirs('./ko_data')

    # modified version for tokenize, vocab, dataset
    if opt.is_train:
        source_file_name = 'ko_data/train_ko_en.en'
        target_file_name = 'ko_data/train_ko_en.ko'
        
        print(f'Build Tokenizer and Vocab...')
        en_sp_tokenizer = Tokenizer(is_train=True, filename=source_file_name, tokenizer_type='spm', model_prefix='spm_en', vocab_size=32000, model_type='bpe')
        ko_sp_tokenizer = Tokenizer(is_train=True, filename=target_file_name, tokenizer_type='spm', model_prefix='spm_ko', vocab_size=32000, model_type='bpe')
        en_sp_vocab = en_sp_tokenizer.vocab
        ko_sp_vocab = ko_sp_tokenizer.vocab
    else:
        print(f'Load Tokenizer and Vocab...')
        en_sp_tokenizer = Tokenizer(is_train=False, model_prefix='spm_en')
        ko_sp_tokenizer = Tokenizer(is_train=False, model_prefix='spm_ko')

        en_sp_vocab = en_sp_tokenizer.vocab
        ko_sp_vocab = ko_sp_tokenizer.vocab
    
    if opt.is_train:
        print(f'Extend Vocab...')
        en_vocab = Vocabulary(en_sp_vocab)
        ko_vocab = Vocabulary(ko_sp_vocab)
        en_vocab.save_vocab('./ko_data/en_vocab')
        ko_vocab.save_vocab('./ko_data/ko_vocab')
    else:
        print(f'Load the extended vocab...')
        en_vocab = Vocabulary.load_vocab('./ko_data/en_vocab')
        ko_vocab = Vocabulary.load_vocab('./ko_data/ko_vocab')

    #En-Ko
#     train_dataset = Our_Handler(src_path='./ko_data/train_ko_en.en', tgt_path='./ko_data/train_ko_en.ko', 
#                                 en_vocab=en_vocab, ko_vocab=ko_vocab, en_tokenizer=en_sp_tokenizer, ko_tokenizer=ko_sp_tokenizer, max_len=100)
    #Ko-En
    train_dataset = Our_Handler(src_path='./ko_data/train_ko_en.ko', tgt_path='./ko_data/train_ko_en.en', 
                                en_vocab=en_vocab, ko_vocab=ko_vocab, en_tokenizer=en_sp_tokenizer, ko_tokenizer=ko_sp_tokenizer, max_len=90)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    opt.train = train_dataloader
    #En-Ko
    opt.src_pad = en_vocab.pad_index
    opt.trg_pad = ko_vocab.pad_index
    
    #Ko-En
    opt.src_pad = en_vocab.pad_index
    opt.trg_pad = ko_vocab.pad_index
    
    opt.train_len = len(train_dataloader)
    
    ######################DEV DATA######################
    # fitting the dev dataset dir
    dev_data_dir = ['./ko_data/dev/dev_ko_en.en', './ko_data/dev/dev_ko_en.ko']
    
    #En-Ko
#     dev_dataset = Our_Handler(src_path=dev_data_dir[0], 
#                             tgt_path=dev_data_dir[1],
#                             en_vocab=en_vocab,ko_vocab=ko_vocab, 
#                             en_tokenizer=en_sp_tokenizer, ko_tokenizer=ko_sp_tokenizer,
#                             max_len=100)
    #Ko-En
    dev_dataset = Our_Handler(src_path=dev_data_dir[1], 
                            tgt_path=dev_data_dir[0],
                            en_vocab=en_vocab,ko_vocab=ko_vocab, 
                            en_tokenizer=en_sp_tokenizer, ko_tokenizer=ko_sp_tokenizer,
                            max_len=90)
    
    dev_dataloader = DataLoader(dev_dataset,
                            batch_size=32,
                            shuffle=False,
                            drop_last=True)
    opt.validation = dev_dataloader
    opt.val_len = len(dev_dataloader)
    ####################################################

    ######################TEST DATA######################
    # fitting the test dataset dir
    test_data_dir = ['./ko_data/test/test_ko_en.en', './ko_data/test/test_ko_en.ko']
    
    #En-Ko
#     test_dataset = Our_Handler(src_path=test_data_dir[0], 
#                             tgt_path=test_data_dir[1],
#                             en_vocab=en_vocab,ko_vocab=ko_vocab, 
#                             en_tokenizer=en_sp_tokenizer, ko_tokenizer=ko_sp_tokenizer,
#                             max_len=100,
#                             # is_test=True
#                             is_test=False)
    #Ko-En
    test_dataset = Our_Handler(src_path=test_data_dir[1], 
                            tgt_path=test_data_dir[0],
                            en_vocab=en_vocab,ko_vocab=ko_vocab, 
                            en_tokenizer=en_sp_tokenizer, ko_tokenizer=ko_sp_tokenizer,
                            max_len=90,
                            # is_test=True
                            is_test=False)
    
    test_dataloader = DataLoader(test_dataset,
                            batch_size=32,
                            shuffle=False,
                            drop_last=True)
    opt.test = test_dataloader
    opt.test_len = len(test_dataloader)
    ####################################################
    
    #En-Ko
#     model = get_model(opt, len(en_vocab), len(ko_vocab))
    
    #Ko-En
    model = get_model(opt, len(ko_vocab), len(en_vocab))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Use {torch.cuda.device_count()} GPUs')
        
    model = model.to(opt.device)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        # opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
        opt.sched = Triangular(opt.optimizer, num_epochs=opt.epochs, warm_up=opt.epochs//4, cool_down=opt.epochs//4)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    if opt.is_test:
        test_model(model, opt)
    else:
        train_model(model, opt)
#         if opt.floyd is False:
#             promptNextAction(model, opt)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt):

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
            # if saved_once == 0:
            #     pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
            #     pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
            #     saved_once = 1
            
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
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()
