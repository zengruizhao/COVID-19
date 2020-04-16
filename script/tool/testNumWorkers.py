# coding=utf-8
"""
@File   : testNumWorkers.py
@Time   : 2020/04/16
@Author : Zengrui Zhao
"""
import torch
import sys
import time
from torch.utils.data import DataLoader
sys.path.append('..')
from data import Lung

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    for num_workers in range(6, 10):  # 遍历worker数
        kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
        train_set = Lung(size=(320, 480))
        train_loader = DataLoader(train_set,
                                  batch_size=32,
                                  drop_last=True,
                                  shuffle=True,
                                  **kwargs)
        start = time.time()
        for epoch in range(10):
            for batch_idx, (data, target) in enumerate(train_loader):  # 不断load
                pass

        end = time.time()
        print(f"Finish with:{(end - start):.5f} second, num_workers={num_workers}")