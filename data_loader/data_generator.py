import os
import numpy as np
import math
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class TxtDataset(Dataset):  # 这是一个Dataset子类
    def __init__(self, y):
        highdata = torch.from_numpy(y).type('torch.FloatTensor')
        highdata_log = torch.div(torch.log(torch.mul(highdata, 1000) + 1), math.log(100))
        self.Label = highdata_log
        lowdata = torch.nn.functional.interpolate(highdata, scale_factor=0.2)
        lowdata_log = torch.div(torch.log(torch.mul(lowdata, 1000) + 1), math.log(100))
        self.Data = lowdata_log

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label  # 返回标签

    def __len__(self):
        return len(self.Data)


class DataGenerator:

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.filedir = config.data_dir
        #获取文件夹中的文件名称列表
        self.filenames=os.listdir(self.filedir)
        #TODO 数据要用log处理

    def read_combine_data(self, i1, i2):
        for i in range(i1, i2+1):
            filepath = self.filedir + '/' + self.filenames[i]
            # 遍历单个文件，读取行数
            tempData = np.expand_dims(np.genfromtxt(filepath, delimiter=','), 0)
            if i % 20 == 0:
                print('Loading No.{} file'.format(i))
            if i == i1:
                data = tempData
            else:
                data = np.concatenate((data, tempData))
        return np.expand_dims(data, 1)


    def load_dataset(self, ):
        if self.config.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if self.dataset == 'train':
            print('Loading train datasets...')

            data = self.read_combine_data(0, 1800)
            txt = TxtDataset(data)

            return DataLoader(dataset=txt, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)

        elif self.dataset == 'test':
            print('Loading test datasets...')

            data = self.read_combine_data(1801, 2000)
            txt = TxtDataset(data)

            return DataLoader(dataset=txt, num_workers=self.config.num_threads,
                              batch_size=self.config.test_batch_size,
                              shuffle=False)

        elif self.dataset == 'debug':
            print('Loading test datasets...')
            # test_set = get_test_set(self.data_dir, self.test_dataset, self.scale_factor, is_gray=is_gray,
            #                         normalize=False)

            data = self.read_combine_data(1, 20)
            txt = TxtDataset(data)

            return DataLoader(dataset=txt, num_workers=self.config.num_threads,
                              batch_size=self.config.batch_size,
                              shuffle=False)

