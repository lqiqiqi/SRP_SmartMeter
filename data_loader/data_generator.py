import os
import numpy as np
import math
import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def shuffle():
    validation_split = 0.1
    random_seed = 42
    dataset_size = 2000

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices, val_indices


class TxtDataset(Dataset):  # 这是一个Dataset子类
    def __init__(self, y, config):
        self.config = config
        highdata = torch.from_numpy(y).type('torch.FloatTensor')
        highdata_log = torch.div(torch.log(torch.mul(highdata, 1000) + 1), math.log(100))
        self.Label = highdata_log
        self.GroundTruth = highdata
        lowdata = torch.nn.functional.interpolate(highdata, scale_factor=1/self.config.scale_factor)
        lowdata_log = torch.div(torch.log(torch.mul(lowdata, 1000) + 1), math.log(100))
        self.Data = lowdata_log

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        groundtruth = self.GroundTruth[index]
        return txt, label, groundtruth  # 返回标签

    def __len__(self):
        return len(self.Data)


class DataGenerator:

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.filedir = config.data_dir
        #获取文件夹中的文件名称列表
        self.filenames = os.listdir(self.filedir)

        #TODO 数据要用log处理

    def read_combine_data(self, indices):
        for i in indices:
            filepath = self.filedir + '/' + self.filenames[i]
            # 遍历单个文件，读取行数
            tempData = np.expand_dims(np.genfromtxt(filepath, delimiter=','), 0)
            if i % 20 == 0:
                print('Loading No.{} file'.format(i))
            if i == indices[0]:
                data = tempData
            else:
                data = np.concatenate((data, tempData))
        return np.expand_dims(data, 1)


    def load_dataset(self, train_indices, test_indices):

        if self.dataset == 'train':
            print('Loading train datasets...')

            data = self.read_combine_data(train_indices)
            txt = TxtDataset(data, self.config)

            return DataLoader(dataset=txt, num_workers=self.config.num_threads, batch_size=self.config.batch_size,
                              shuffle=False)

        elif self.dataset == 'test':
            print('Loading test datasets...')

            data = self.read_combine_data(test_indices) # 2000 out of index
            txt = TxtDataset(data, self.config)

            return DataLoader(dataset=txt, num_workers=self.config.num_threads,
                              batch_size=self.config.test_batch_size,
                              shuffle=False)

        elif self.dataset == 'debug':
            print('Loading debug datasets...')
            # test_set = get_test_set(self.data_dir, self.test_dataset, self.scale_factor, is_gray=is_gray,
            #                         normalize=False)

            data = self.read_combine_data(test_indices[:20])
            txt = TxtDataset(data, self.config)

            return DataLoader(dataset=txt, num_workers=self.config.num_threads,
                              batch_size=self.config.batch_size,
                              shuffle=False)

