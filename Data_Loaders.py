import math

import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split



class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data_medium.csv', delimiter=',')
        collided_set = np.where(self.data[:, -1] == 1)
        not_collided_set = np.where(self.data[:, -1] == 0)
        collisions = self.data[collided_set]
        non_collisions = self.data[not_collided_set]
        choices = np.random.choice(len(non_collisions), np.math.ceil(len(collisions)*0.6), replace=False)
        non_collisions_subset = non_collisions[choices]
        self.collision_count = len(collisions)
        self.non_collision_count = len(non_collisions_subset)
        self.data = np.concatenate((collisions, non_collisions_subset), axis=0)
        np.random.shuffle(self.data)


        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data)  # fits and transforms

        pickle.dump(self.scaler, open("submission/saved/scaler.pkl", "wb"), protocol=1)

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        input = self.normalized_data[idx, :-1]
        label = self.normalized_data[idx, -1]
        return {'input': input, 'label': label}

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        set_count = len(self.nav_dataset)
        train_count = math.ceil(set_count * 0.8)
        test_count = set_count - train_count
        self.train_loader, self.test_loader \
            = torch.utils.data.random_split(self.nav_dataset, [train_count, test_count])

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']


if __name__ == '__main__':
    main()
