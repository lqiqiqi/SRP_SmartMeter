import os
import numpy as np
import math
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def load_data(filepath):
    # data = np.genfromtxt(filepath, delimiter=',')
    data = []
    with open(filepath) as FileObj:
        for lines in FileObj:
            data.append(float(lines))
        data = np.array(data)
    return data

def shuffle():
    validation_split = 0.125
    random_seed = 40 # 本来是42
    dataset_size = 16000

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices, val_indices

def sequence_random_crop(sequence: object, sequence_len: object) -> object:
    # print(len(sequence), sequence_len)
    start = np.random.randint(0, len(sequence) - sequence_len)
    return sequence_crop(sequence, sequence_len, start)

def sequence_crop(sequence, sequence_len, start):
    if len(sequence) >= start + sequence_len:
        cropped = sequence[start: start + sequence_len]
    else:
        raise Exception('Wrong Size.')
    return cropped

class TxtDataset(Dataset):  # 这是一个Dataset子类
    def __init__(self, config, dataset):
        self.config = config
        self.filenames_ = []
        for file in os.listdir(self.config.data_dir):
            self.filenames_.append(os.path.join(self.config.data_dir, file))
        if dataset == 'train':
            indices, _ = shuffle()
        elif dataset == 'test':
            _, indices = shuffle()
        elif dataset == 'debug':
            _, indices = shuffle()
            indices = indices[:200]
        self.filenames = [self.filenames_[i] for i in indices]

    def __getitem__(self, index):
        # highdata_raw二维，经过dataloader输出为三维，与conv层match
        raw0 = load_data(self.filenames[index])
        raw = sequence_random_crop(raw0, 10000)

        highdata_raw = np.expand_dims(raw, 0)
        highdata = torch.from_numpy(highdata_raw).type('torch.FloatTensor')
        highdata_log = torch.div(torch.log(torch.mul(highdata, 1000) + 1), math.log(100))

        # interpolate需要三维输入，先升为三维后，再squeeze降维
        # lowdata = torch.nn.functional.interpolate(highdata.unsqueeze(0), scale_factor=1 / self.config.scale_factor).squeeze(0)

        # take 函数结果变为一维
        lowdata = torch.take(highdata, torch.arange(0, highdata.size()[1], self.config.scale_factor).long()).unsqueeze(0)
        lowdata_log = torch.div(torch.log(torch.mul(lowdata, 1000) + 1), math.log(100))

        return lowdata_log, highdata_log, highdata  # 返回txt, label, groundtruth

    def __len__(self):
        return len(self.filenames)


class DataGenerator:

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.filedir = config.data_dir
        #获取文件夹中的文件名称列表
        self.filenames = os.listdir(self.filedir)

    def load_dataset(self):

        if self.dataset == 'train':
            print('Loading train datasets...')

            txt = TxtDataset(self.config, 'train')

            return DataLoader(dataset=txt, num_workers=self.config.num_threads, batch_size=self.config.batch_size,
                              shuffle=False)

        elif self.dataset == 'test':
            print('Loading test datasets...')

            txt = TxtDataset(self.config, 'test')

            return DataLoader(dataset=txt, num_workers=self.config.num_threads,
                              batch_size=self.config.test_batch_size,
                              shuffle=True)

        elif self.dataset == 'debug':
            print('Loading debug datasets...')

            txt = TxtDataset(self.config, 'debug')

            return DataLoader(dataset=txt, num_workers=self.config.num_threads,
                              batch_size=self.config.batch_size,
                              shuffle=True)

