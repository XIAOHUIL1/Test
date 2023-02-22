import torch
from torch import nn


class VGG(nn.Module):
    # 初始化函数中传入features也就是make_features生成的网络结构，classnum代表分类的数量，init_weight代表是否需要将网络初始化
    def __init__(self, features, class_num=1000, init_weight=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # Dropout 随机失活神经元 理解为灭霸打响指
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            # ReLU 提高模型准确率和神经网络的非线性能力
            nn.ReLU(True),
            nn.Linear(2048, class_num)
        )

        if init_weight:
            self.__initialize__weight()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)


# VGG的conv stride为1，padding为2 maxpool的size为2，stride为2
# 提取特征网络结构，传入一个配置变量cfg list类型，到时候根据需要传入不同配置的列表就可以了
def make_features(cfg: list):
    layers = []
    # 输入RGB彩色图片，三通道
    in_channels = 3
    # for循环来遍历配置列表
    for v in cfg:
        # 如果遍历到pool元素，代表是一个最大池化层，所以创建一个最大池化下采样层
        if v == "pool":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 起始通道数为3，后续根据字典更新
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            # 每个卷积层都采用ReLU激活函数，所以将刚刚定义好的卷积层和ReLu接起来，添加进layers列表当中
            layers += [conv2d, nn.ReLU(True)]
            # 每当通过一个卷积层时，输出深度out_channels发生了变化，变成了v，将in_channels改成v
            in_channels = v
            # 用nn.Sequantial函数将列表通过非关键字参数的形式传入进去，layers之前的*号代表通过非关键字参数传入进去的
            return nn.Sequential(*layers)


# 创建字典，每一个key对应一个vgg模型，参数全部输入进去
cfgs = {
    'vgg11': [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'vgg13': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'vgg16': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
}


def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning:model number {} not in cfgs dict!".format(model_name))
        model = vgg(make_features(cfg), **kwargs)
        return model


vgg_model = vgg(model_name='vgg13')
