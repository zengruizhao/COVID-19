# coding=utf-8
"""
@File   : evaluate.py
@Time   : 2020/04/14
@Author : Zengrui Zhao
"""
import numpy as np
import torch
from model import Vgg
from data import Lung
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batchSize', type=int, default=128)
    parse.add_argument('--dataDir', type=str, default='/home/zzr/Data/XinGuan/lung')
    parse.add_argument('--model', type=str,
                       default='/home/zzr/Project/COVID-19/model/200417-083930_Vgg/out_450.pth')
    return parse.parse_args()

def evalSingLung(model, dataloader):
    model.eval()
    tp, tn, total = 0, 0, 0
    with torch.no_grad():
        for img, lb, _ in dataloader:
            img, lb = img.to(device), lb.to(device)
            outputs = model(img).view(img.shape[0], -1)
            predicted = np.array(torch.max(outputs, dim=1)[1].cpu())
            predictedP = np.where(predicted==1)
            predictedN = np.where(predicted==0)
            tp += (lb[predictedP] == 1).sum()
            tn += (lb[predictedN] == 0).sum()
            total += lb.size()[0]

    f1 = (2 * tp + .01) / (total + tp - tn + .01)
    acc = (tp + tn + .01) / (total + .01)
    print(f'f1: {f1:.4f}, '
          f'acc: {acc:.4f}')

    return f1, acc

def evalPatient(model, dataloader):
    model.eval()
    table = {}
    with torch.no_grad():
        for img, lb, names in dataloader:
            img, lb = img.to(device), np.array(lb)
            outputs = F.softmax(model(img), dim=1).view(img.shape[0], -1)
            prob = np.array(outputs.cpu()[:, 1])
            for idx, name in enumerate(names):
                table[name] = [lb[idx], prob[idx]]
    
    return table

def decision(table):
    """
    :param table:
    :return:
    """
    names = sorted([key for key in table.keys()])
    result = {}
    publicName = names[0][:-5]
    labelpre = table[names[0]][0]
    probpre = table[names[0]][1]
    result[publicName] = (labelpre, probpre)
    for name in names[1:]:
        label = table[name][0]
        prob = table[name][1]
        if publicName == name[:-5]:
            result[publicName] = (label, probpre, prob)
        else:
            publicName = name[:-5]
            labelpre, probpre = label, prob
            result[publicName] = (labelpre, probpre)

    return result

def metrics(result):
    total = len(result)
    tp, tn = 0, 0
    for key, val in result.items():
        label = val[0]
        prob = val[1] if len(val) == 2 else (val[1] + val[2]) / 2
        # prob = val[1] if len(val) == 2 else np.max(val[1:])
        if prob > .5 and label == 1:
            tp += 1
        if prob < .5 and label == 0:
            tn += 1

    f1 = (2 * tp + .01) / (total + tp - tn + .01)
    acc = (tp + tn + .01) / (total + .01)
    print(f'f1: {f1:.4f}, '
          f'acc: {acc:.4f}')

def main(args):
    model = Vgg().to(device)
    model.load_state_dict(torch.load(args.model))
    testSet = Lung(rootDir=args.dataDir, mode='test', size=(320, 480))
    valDataloader = DataLoader(testSet,
                               batch_size=args.batchSize,
                               drop_last=False,
                               shuffle=False,
                               pin_memory=False)
    # evalSingLung(model, valDataloader)
    table = evalPatient(model, valDataloader)
    result = decision(copy.deepcopy(table))
    metrics(copy.deepcopy(result))

if __name__ == '__main__':
    args = parseArgs()
    main(args)
