import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from train import DEVICE
import torch
import torch.nn as nn
device = torch.device(DEVICE)


# 這塊是別人寫的，現在還沒修到我可以用的程度。
class Time2Vec(nn.Module):
    def __init__(self, activation='sin', hidden_dim=512):
        super(Time2Vec, self).__init__()
        if activation == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos
        self.out_features = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, 2)

    def forward (self, x):
        # 荻取x的尺寸信息
        batch_size = x.shape[0]
        sentence_len = x.shape[1]
        in_features = x.shape[2]

        # 初始化权重和偏置
        self.w0 = nn.parameter.Parameter(torch.randn(batch_size, in_features, 1)).to(device)
        self.b0 = nn.parameter.Parameter(torch.randn(batch_size, sentence_len, 1)).to(device)
        self.w = nn.parameter.Parameter(torch.randn(batch_size, in_features, self.out_features - 1)).to(device)
        self.b = nn.parameter.Parameter(torch.randn(batch_size, sentence_len, in_features - 1)).to(device)

        # 运算
        print(self.b.shape)
        v1 = self.activation(torch.matmul(x, self.w) + self.b)
        v2 = torch.matmul(x, self.w0) + self.b0
        v3 = torch.cat([v1, v2], -1)
        x = self.fc1(v3)
        return x

# 這塊是從另一個模型上面抄襲來降低維度，現在沒有在用。
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=4, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
           
        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes//ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv1d(in_planes//ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_out = self.fc(avg_x)
        max_out = self.fc(max_x)
        out = avg_out + max_out
        out = out.permute(0, 2, 1)
        return self.sigmoid(out)
    
class Transformer1d(nn.Module):
    def __init__(self, out_feature=6, n_length=256, d_model=1, nhead=1, dim_feedforward=2048, dropout=0.1):
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
            norm_first=True,
        )
        self.ttv = Time2Vec()
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.Linear(self.d_model*self.n_length, self.out_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, src):
        src = src.reshape(-1, self.n_length, 1)
        src = self.ttv(src)
        print(src.shape)

        src = src.permute(0, 2, 1)
        out = self.encoder(src)
        # print(out.shape)
        out = out.reshape(out.shape[0], -1)
        out = self.decoder(out)
        out = self.relu(out)
        out = out.view(-1, 6)
        return out    
    