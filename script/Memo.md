## 记录

- 目前对比几个模型, VGG相对较好, 猜想原因为其他模型的第一层为尺寸较大的卷积, VGG使用3x3卷积
- 网络模型使用vgg11_bn
- 输入图像尺寸越大，效果越好，目前输入尺寸为320X480
- 学习率为1e-4
- 单肺图像的平均宽高为167x234
- 网络在最后一个池化层之后只使用一个卷积层
## 尝试
- 因为单个肺的形状为长方形，因此输入图像尺寸不要设置为正方形
- 每次训练结果都不一样,
