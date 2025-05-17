import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import cv2


class MySSMDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data_len = len(self.data)

        max_space_len = 0
        for i in range(0, self.data_len):
            item_space_len = len(self.data[i]['short_space'])
            if item_space_len > max_space_len:
                max_space_len = item_space_len

        for j in range(0, self.data_len):
            item_space = self.data[j]['short_space']
            if item_space.shape[0] < max_space_len:
                new_item_space = np.zeros((max_space_len, 8))
                # new_item_space = np.zeros((max_space_len, 10))
                new_item_space[:item_space.shape[0], :] = item_space
                self.data[j]['short_space'] = new_item_space
        print(f'Total number of samples = {self.data_len} ; max_space_len = {max_space_len}')

        print(f'The dataset was successfully loaded from: {data_path}')



    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data_len
    


