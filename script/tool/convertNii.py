# coding=utf-8
"""
@File   : convertNii.py
@Time   : 2020/04/15
@Author : Zengrui Zhao
"""
import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

def main():
    path = '/home/zzr/Data/XinGuan/seg'
    outPath = '/home/zzr/Data/XinGuan/picked/seg'
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for file in os.listdir(path):
        image = nib.load(os.path.join(path, file))
        data = image.get_fdata()
        for i in range(data.shape[-1]):
            img = data[..., i]
            img = np.flipud(np.rot90(img))
            # print(np.unique(img))
            # plt.imshow(img, cmap='gray')
            # plt.show()
            mpimg.imsave(os.path.join(outPath, f'{file}_{str(i)}.png'), img, cmap='gray')

def read():
    path = '/home/zzr/Data/XinGuan/picked/seg'
    for i in os.listdir(path):
        img = np.array(Image.open(os.path.join(path, i)).convert('L'))
        plt.imshow(img, cmap='gray')
        plt.show()

if __name__ == '__main__':
    # main()
    read()
