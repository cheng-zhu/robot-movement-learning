import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        ones = 0
        real_keep = []
        for i in range(int(data.size/7)):
            if data[i][6] == 1:
                real_keep.append(i)
                ones += 1
        indices = list(range(int(data.size/7)))
        np.random.seed(31)
        np.random.shuffle(indices)

        ratio = 2.8
        zeros = 0
        for i in indices:
            if zeros == int(ratio * ones):
                break
            if data[i][6] == 0:
                real_keep.append(i)
                zeros += 1

        self.data = [data[i] for i in real_keep]
        self.data = np.array(self.data)
        np.random.shuffle(self.data)

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# __getitem__() must return a dict with entries {'input': x, 'label': y}
        x = self.normalized_data[idx, :-1].astype('float32')
        y = self.normalized_data[idx, -1].astype('float32')
        dic = {'input': x, 'label': y}
        return dic


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
        test_split = 0.25
        shuffle_dataset = True
        random_seed = 31

        dataset_size = len(self.nav_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        
        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)
        
        self.train_loader = data.DataLoader(self.nav_dataset, batch_size=batch_size, sampler=train_sampler)
        self.test_loader = data.DataLoader(self.nav_dataset, batch_size=batch_size, sampler=test_sampler)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
