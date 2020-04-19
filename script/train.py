# coding=utf-8
"""
@File   : train.py
@Time   : 2020/04/13
@Author : Zengrui Zhao
"""
import numpy as np
import torch
from model import Vgg
from data import Lung
import argparse
import time
import os.path as osp
import os
from logger import get_logger
from torch.utils.data import DataLoader
from torch import nn
from ranger import Ranger
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=500)
    parse.add_argument('--batchSize', type=int, default=32)
    parse.add_argument('--lr', type=int, default=1e-4)
    parse.add_argument('--inputSize', default=(320, 480))
    parse.add_argument('--valBatchSize', type=int, default=64)
    parse.add_argument('--name', type=str, default='Vgg')
    parse.add_argument('--numWorkers', type=int, default=7)
    parse.add_argument('--dataDir', type=str, default='/home/zzr/Data/XinGuan/lung')
    parse.add_argument('--logDir', type=str, default='../log')
    parse.add_argument('--tensorboardDir', type=str, default='../tensorboard')
    parse.add_argument('--modelDir', type=str, default='../model')
    parse.add_argument('--evalFrequency', type=int, default=2)
    parse.add_argument('--msgFrequency', type=int, default=10)
    parse.add_argument('--saveFrequency', type=int, default=50)
    return parse.parse_args()

def eval(model, dataloader, logger):
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
    logger.info(f'f1: {f1:.4f}, '
                f'acc: {acc:.4f}')

    return f1, acc

def main(args, logger):
    writer = SummaryWriter(args.tensorboardDir)
    model = Vgg().to(device)
    trainSet = Lung(rootDir=args.dataDir, mode='train', size=args.inputSize)
    valSet = Lung(rootDir=args.dataDir, mode='test', size=args.inputSize)
    trainDataloader = DataLoader(trainSet,
                                 batch_size=args.batchSize,
                                 drop_last=True,
                                 shuffle=True,
                                 pin_memory=False,
                                 num_workers=args.numWorkers)
    valDataloader = DataLoader(valSet,
                               batch_size=args.valBatchSize,
                               drop_last=False,
                               shuffle=False,
                               pin_memory=False,
                               num_workers=args.numWorkers)
    criterion = nn.CrossEntropyLoss()
    optimizer = Ranger(model.parameters(), lr=args.lr)
    iter = 0
    runningLoss = []
    for epoch in range(args.epoch):
        if epoch != 0 and epoch % args.evalFrequency == 0:
            f1, acc = eval(model, valDataloader, logger)
            writer.add_scalars('f1_acc', {'f1': f1,
                                          'acc': acc}, iter)

        if epoch != 0 and epoch % args.saveFrequency == 0:
            modelName = osp.join(args.subModelDir, 'out_{}.pth'.format(epoch))
            # 防止分布式训练保存失败
            stateDict = model.modules.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(stateDict, modelName)

        for img, lb in trainDataloader:
            # array = np.array(img)
            # for i in range(array.shape[0]):
            #     plt.imshow(array[i, 0, ...], cmap='gray')
            #     plt.show()
            iter += 1
            img, lb = img.to(device), lb.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs.squeeze(), lb.long())
            loss.backward()
            optimizer.step()
            runningLoss.append(loss.item())

            if iter % args.msgFrequency == 0:
                avgLoss = np.mean(runningLoss)
                runningLoss = []
                lr = optimizer.param_groups[0]['lr']
                logger.info(f'epoch: {epoch} / {args.epoch}, '
                            f'iter: {iter} / {len(trainDataloader) * args.epoch}, '
                            f'lr: {lr}, '
                            f'loss: {avgLoss:.4f}')
                writer.add_scalar('loss', avgLoss, iter)

    eval(model, valDataloader, logger)
    modelName = osp.join(args.subModelDir, 'final.pth')
    stateDict = model.modules.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(stateDict, modelName)

if __name__ == '__main__':
    args = parseArgs()
    uniqueName = time.strftime('%y%m%d-%H%M%S_') + args.name
    args.uniqueName = uniqueName
    # 每次创建作业使用不同的tensorboard目录
    args.subTensorboardDir = osp.join(args.tensorboardDir, args.uniqueName)
    # 保存模型的目录
    args.subModelDir = osp.join(args.modelDir, args.uniqueName)

    # 创建所有用到的目录
    for subDir in [args.subTensorboardDir,
                    args.subModelDir,
                    args.logDir]:
        if not osp.exists(subDir):
            os.makedirs(subDir)

    logFileName = osp.join(args.logDir, args.uniqueName + '.log')
    logger = get_logger(logFileName)

    for k, v in args.__dict__.items():
        logger.info(f'{k}: {v}')

    main(args, logger=logger)
