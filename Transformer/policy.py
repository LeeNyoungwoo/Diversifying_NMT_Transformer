import torch
import numpy as np

class SamplePolicy(nn.Module):
    def __init__(self, batch_size, src_len, T, K, head_num):
        super(SamplePolicy, self).__init__()
        
        self.batch_size = batch_size
        self.src_len = src_len
        self.K = K
        self.T = T
        self.head_num = head_num
        
        #self.counting = np.zeros((batch_size, max_len))
        
    def forward(self, attention_weight):
    
        for t in range(self.T):
            counting = torch.zeros((self.batch_size, self.src_len))
            
        # calculate attention weight
        for h in range(self.head_num):
            candidate = torch.argmax(attention_weight[h], dims=-1)
            
            for idx in range(self.batch_size):
                counting[idx, candidate[idx]] += 1
        
        # confusing condition
        if max(counting) <= self.K:
            sampled_head = np.random.uniform(low=0, high=self.head_num-1, size=self.head_num)
            
            attention_weight[:] = attention_weight[sampled_head]
            
        
        return attention_weight
        