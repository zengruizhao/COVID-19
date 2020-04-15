# coding=utf-8
"""
@File   : evaluate.py
@Time   : 2020/04/14
@Author : Zengrui Zhao
"""
import numpy as np
import torch
from model import Vgg
from data import Data
import argparse
from torch.utils.data import DataLoader

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batchSize', type=int, default=128)
    parse.add_argument('--dataDir', type=str, default='/home/zzr/Data/XinGuan')
    parse.add_argument('--model', type=str,
                       default='/home/zzr/Project/COVID-19/model/200414-131422_Vgg/out_20.pth')
    return parse.parse_args()

def eval(model, dataloader):
    model.eval()
    tp, tn, total = 0, 0, 0
    with torch.no_grad():
        for img, lb in dataloader:
            img, lb = img.to(device), lb.to(device)
            outputs = model(img).view(img.shape[0], -1)
            predicted = np.array(torch.max(outputs, dim=1)[1].cpu())
            predictedP = np.where(predicted==1)
            predictedN = np.where(predicted==0)
            tp += (lb[predictedP] == 1).sum()
            tn += (lb[predictedN] == 0).sum()
            total += lb.size()[0]

    f1 = (2 * tp + .01) / (total + tp - tn + .001)
    acc = (tp + tn + .01) / (total + .01)
    print(f'f1: {f1:.4f}, '
          f'acc: {acc:.4f}')

    return f1, acc

def main(args):
    model = Vgg().to(device)
    model.load_state_dict(torch.load(args.model))
    testSet = Data(rootDir=args.dataDir, mode='test')
    valDataloader = DataLoader(testSet,
                               batch_size=args.batchSize,
                               drop_last=False,
                               shuffle=False,
                               pin_memory=False)
    f1, acc = eval(model, valDataloader)

if __name__ == '__main__':
    args = parseArgs()
    main(args)
