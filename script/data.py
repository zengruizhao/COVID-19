# coding=utf-8
"""
@File   : data.py
@Time   : 2020/04/13
@Author : Zengrui Zhao
"""
from torch.utils.data import Dataset
from pathlib2 import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pickle

class Data(Dataset):
    def __init__(self, rootDir=Path('/home/zzr/Data/XinGuan/original'),
                 mode='train',
                 size=(224, 224)):
        self.rootDir = rootDir
        self.size = size
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.getList()
        self.mean, self.std = 0.5968, 0.2877
        if mode == 'train':
            self.to_tensorImg = \
                transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(degrees=90, resample=Image.BILINEAR),
                                    transforms.RandomApply([transforms.RandomRotation(90)], p=.5),
                                    transforms.RandomApply([transforms.ColorJitter(.1, .1, .1, .1)], p=.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((self.mean, ), (self.std, ))])
        else:
            self.to_tensorImg = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((self.mean, ), (self.std, ))])

    def __getitem__(self, item):
        label = 0 if 'Non' in str(self.list[item]) else 1
        img = Image.open(str(self.list[item])).convert('L').resize(self.size)

        return self.to_tensorImg(img), label

    def __len__(self):
        return len(self.list)

    def getList(self):
        txtPath = Path(self.rootDir) / 'split'
        neg, pos = '', ''
        for i in os.listdir(txtPath):
            if self.mode in i:
                if 'Non' in i:
                    neg = i
                else:
                    pos = i

        with open(Path(txtPath / neg), 'r') as f:
            neg = f.readlines()

        with open(Path(txtPath / pos), 'r') as f:
            pos = f.readlines()

        neg = list(map(lambda x: Path(self.rootDir) / 'NonCOVID' / x.strip(), neg))
        pos = list(map(lambda x: Path(self.rootDir) / 'COVID' / x.strip(), pos))
        self.list = (neg + pos)
        random.shuffle(self.list)

    def get_mean_std(self, type='train', mean_std_path='./mean.pkl'):
        """
        计算数据集的均值和标准差
        :param type: 使用的是那个数据集的数据，有'train', 'test', 'testing'
        :param mean_std_path: 计算出来的均值和标准差存储的文件
        :return:
        """
        means, stdevs = 0, 0
        num_imgs = len(self.list)
        for data in tqdm(self.list):
            img = np.array(Image.open(str(data)).convert('L').resize(self.size)) / 255.
            means += img.mean()
            stdevs += img.std()

        means = np.asarray(means) / num_imgs
        stdevs = np.asarray(stdevs) / num_imgs

        print("{} : normMean = {}".format(type, means))
        print("{} : normstdevs = {}".format(type, stdevs))

        # 将得到的均值和标准差写到文件中，之后就能够从中读取
        # with open(mean_std_path, 'wb') as f:
        #     pickle.dump(means, f)
        #     pickle.dump(stdevs, f)
        #     print('pickle done')

class Lung(Data):
    def __init__(self, rootDir=Path('/home/zzr/Data/XinGuan/lung'),
                 mode='train',
                 size=(160, 240)):
        super().__init__(rootDir, mode, size)
        self.mean, self.std = .5536, .2696
        if mode == 'train':
            self.to_tensorImg = \
                transforms.Compose([
                    transforms.RandomOrder([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply([
                            transforms.RandomRotation(45,
                                                      resample=Image.BILINEAR)], p=.5),
                        transforms.RandomApply([
                            transforms.ColorJitter(.1, .1, .1, .1)], p=.5)]),
                        transforms.ToTensor(),
                        transforms.Normalize((self.mean, ), (self.std, ))])
        else:
            self.to_tensorImg = \
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((self.mean, ),(self.std, ))])

    def getList(self):
        path = os.path.join(self.rootDir, self.mode)
        self.list = []
        for i in os.listdir(path):
            for j in os.listdir(Path(path) / i):
                self.list.append(Path(path) / i / j)
        # self.list.sort()
        # random.shuffle(self.list)

    def __getitem__(self, item):
        label = 0 if 'NonCOVID' in str(self.list[item]).split('/') else 1
        img = Image.open(str(self.list[item])).convert('L').resize(self.size)

        return self.to_tensorImg(img), label, str(self.list[item]).split('/')[-1]

if __name__ == '__main__':
    datas = Lung(mode='test', size=(320, 480))
    # datas.get_mean_std()
    for data in datas:
        print(data[-1])
