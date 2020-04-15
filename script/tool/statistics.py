# coding=utf-8
"""
@File   : statistics.py
@Time   : 2020/04/15
@Author : Zengrui Zhao
"""
import os
from pathlib2 import Path
from PIL import Image
import numpy as np

path = '/home/zzr/Data/XinGuan/lung/train'
height, width = [], []
for i in os.listdir(path):
    for j in os.listdir(Path(path) / i):
        img = Image.open(str(Path(path) / i / j))
        height.append(img.size[0])
        width.append(img.size[1])

print(np.mean(height), np.mean(width))