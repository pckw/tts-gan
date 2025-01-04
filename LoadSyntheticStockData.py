# Generator synthetic Running and Jumping data 
# Made them to a Pytorch Dataset 

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
import os

class SyntheticStockDataset(Dataset):
    def __init__(self, 
                 stock_model_path,
                 channels = 6,
                 latent_dim = 100,
                 seq_len = 150,
                 sample_size = 1000
                 ):
        
        self.sample_size = sample_size
        
        #Generate Stock Data
        gen_net = Generator(seq_len=seq_len, channels=channels, latent_dim=latent_dim)
        ckp = torch.load(stock_model_path)
        gen_net.load_state_dict(ckp['gen_state_dict'])
        
        
        #generate syntheticstock data label is 0
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_stock = gen_net(z)
        self.syn_stock = self.syn_stock.detach().numpy()
        self.stock_label = np.zeros(len(self.syn_stock))
    
        
        self.combined_train_data = self.syn_stock
        self.combined_train_label = self.stock_label
        
        print(self.combined_train_data.shape)
        print(self.combined_train_label.shape)
        
        
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        return self.combined_train_data[idx], self.combined_train_label[idx]
    
    
