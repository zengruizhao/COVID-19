# coding=utf-8
"""
@File   : train_testSplit.py
@Time   : 2020/04/15
@Author : Zengrui Zhao
"""
import os
import shutil
import random

path = '/home/zzr/Data/XinGuan/picked'
outPath = '/home/zzr/Data/XinGuan/data'

for i in os.listdir(path):
    lists = os.listdir(os.path.join(path, i))
    random.shuffle(lists)
    train = lists[:int(len(lists) * .7)]
    test = lists[int(len(lists) * .7):]
    for j in train:
        savePath = os.path.join(outPath, 'train', i)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        shutil.copy(os.path.join(path, i, j), os.path.join(savePath, j))

    for j in test:
        savePath = os.path.join(outPath, 'test', i)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        shutil.copy(os.path.join(path, i, j), os.path.join(savePath, j))