import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

FILES = os.listdir(f'./data/dataset_x/')

# example of dataset create
class KlineDataset(Dataset):
    # data loading
    def __init__(self, _x:list, _y:list):
        _x = np.array(_x)
        _y = np.array(_y)

        self.x = torch.from_numpy(_x)
        self.y = torch.from_numpy(_y)
        self.n_samples = _y.shape[0]

    # indexing
    def __getitem__(self, index):
        each_x = np.loadtxt(f'./data/dataset_x/BTCUSDT_{index}.txt', delimiter=',', dtype=np.float32, skiprows=0)
        x_data = np.reshape(each_x, (5, 5, 90))
        x_data = np.transpose(x_data, (1, 0, 2))
        # print(x_data.shape)
        x_data = torch.from_numpy(x_data)
        return x_data, self.y[index]

    # return the length of our dataset
    def __len__(self):

        return self.n_samples
    
def split_and_load(test_size=0.2, random_state=101):
    print('loading file into dataset')

    y = np.loadtxt('./data/BTCUSDT_dataset_y.csv', delimiter=',', dtype=np.float32, skiprows=0)
    x = np.arange(0, len(y))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test
    

# ========================================================================================================================
# Test Function
# ========================================================================================================================
if __name__ == "__main__":
    print(KlineDataset().x.shape)