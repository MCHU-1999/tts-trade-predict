import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool = nn.AdaptiveMaxPool2d((1, None))
           
        self.fc = nn.Sequential(nn.Conv1d(in_channels, 2, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv1d(2, in_channels, 1, bias=False))
        # self.fc = nn.Sequential(nn.Conv1d(in_channels, 1, 1, bias=False),
        #                        nn.ReLU())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_x = self.avg_pool(x)
        # max_x = self.max_pool(x)
        # avg_out = self.fc(avg_x)
        # max_out = self.fc(max_x)
        x = self.fc(x)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = avg_out + max_out
        return self.sigmoid(out)

class Transformer1d(nn.Module):
    def __init__(self, out_feature=6, n_length=256, d_model=256, nhead=16, dim_feedforward=2048, dropout=0.1):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.out_feature = out_feature
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,
            batch_first=True,
        )
        self.ca = ChannelAttention(5)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.Linear(self.d_model, self.out_feature)
        # self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, src):
        # out = out.permute(2, 0, 1)
        src = self.ca(src)
        out = self.encoder(src)
        # out = out.mean(0)
        out = self.decoder(out)
        out = self.relu(out)
        out = out.view(-1, 6)
        
        return out    
    