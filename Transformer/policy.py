import torch
from torch import nn
import numpy as np
    
class SamplePolicy(nn.Module):
    def __init__(self, K, head_num):
        super(SamplePolicy, self).__init__()
        
        #self.src_len = src_len
        self.K = K
        self.head_num = head_num
        
        #self.counting = np.zeros((batch_size, max_len))
        
    def forward(self, attention_weight):
        # attention_weight shape == (beam_size, head_num, decoding_step, src_len)
        attention_weight = attention_weight.transpose(1, 2)
        # attention_weight shape == (beam_size, decoding_step, head_num, src_len)
        time_step = attention_weight.size(1)
        src_len = attention_weight.size(3)

        counting = torch.zeros((src_len))
            
        # calculate attention weight
        for h in range(self.head_num):
            candidate = torch.argmax(attention_weight[:, time_step-1, h], dim=-1)
            counting[candidate] += 1
                
        # confusing condition
        if max(counting) <= self.K:
            sampled_head = np.random.randint(low=0, high=self.head_num, size=1)
            attention_weight[:, time_step-1] = attention_weight[:, time_step-1, sampled_head]
            
        attention_weight = attention_weight.transpose(1, 2)
        return attention_weight
        
class OurPolicy(nn.Module):
    def __init__(self, K, head_num):
        super(OurPolicy, self).__init__()
        
        self.K = K
        self.head_num = head_num
        
    def forward(self, attention_weight):
        # attention_weight shape == (beam_size, head_num, decoding_step, src_len)
        attention_weight = attention_weight.transpose(1, 2)
        # attention_weight shape == (beam_size, decoding_step, head_num, src_len)
        #print(f'attn: {attention_weight.size()}')
        time_step = attention_weight.size(1)
        src_len = attention_weight.size(3)
        
        counting = torch.zeros((src_len))
        checking_table = torch.zeros((self.head_num, src_len))
        
        # calculate attention weight
        for h in range(self.head_num):
            candidate = torch.argmax(attention_weight[:, time_step-1, h], dim=-1)
            counting[candidate] += 1
            checking_table[h][candidate] = 1
                
        # confusing condition
        if max(counting) <= self.K:
            max_val = max(counting)
            max_indices = (counting == max_val).nonzero()
            
            max_heads = []
            for idx in max_indices:
                tb = checking_table[:, idx.item()].nonzero().view(-1)
                tmp = [each.item() for each in tb]
                max_heads.extend(tmp)

            attention_weight = self.merging_attn(attention_weight, max_heads, time_step)

        attention_weight = attention_weight.transpose(1, 2)
        return attention_weight
    
    def merging_attn(self, attention_weight, max_heads, time_step):
        sampled_head = np.random.randint(low=0, high=len(max_heads), size=1)
        max_heads = np.array(max_heads)
        sampled_head = max_heads[sampled_head]
        
        # forcing most likely heads to sampled head
        for head in max_heads:
            attention_weight[:, time_step-1, head] = attention_weight[:, time_step-1, sampled_head]
        
        return attention_weight
            
  