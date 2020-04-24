# coding=utf-8
"""
@File   : model.py
@Time   : 2020/04/13
@Author : Zengrui Zhao
"""
from pretrainedmodels import se_resnext50_32x4d, alexnet, resnet34, vgg13_bn, vgg11_bn
from torch import nn
from torchsummary import summary
import torch

class Vgg(nn.Module):
    def __init__(self, classes=2):
        super(Vgg, self).__init__()
        model = vgg11_bn()
        firstLayer = list(model.children())[1]
        firstLayer[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        firstLayer[-1] = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(*firstLayer)
        self.classifier = nn.Sequential(nn.Conv2d(512, 1024, 1),
                                        nn.BatchNorm2d(1024),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(1024, classes, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Seresnext(nn.Module):
    def __init__(self, classes=2):
        super(Seresnext, self).__init__()
        model = se_resnext50_32x4d()
        firstLayer = list(model.children())[0]
        firstLayer[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(firstLayer) + list(model.children())[1:-1]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Conv2d(2048, classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Resnet(nn.Module):
    def __init__(self, classes=2):
        super(Resnet, self).__init__()
        model = resnet34()
        features = list(model.children())[:-1]
        features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(512, classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Alexnet(nn.Module):
    def __init__(self, classes=2):
        super(Alexnet, self).__init__()
        model = alexnet()
        firstLayer = list(model.children())[1]
        firstLayer[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.features = nn.Sequential(*firstLayer)
        firstLayer[-1] = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(*firstLayer)
        self.classifier = nn.Sequential(nn.Conv2d(256, 512, 1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 1024, 1),
                                        nn.BatchNorm2d(1024),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(1024, classes, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Vgg().to(device)
    print(model)
    summary(model, (1, 320, 480))
