from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),  # 要采样的话在这里改变stride
            #nn.BatchNorm2d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),  # 采样之后注意保持feature map的大小不变
            #nn.BatchNorm2d(outchannel),
            nn.Dropout(0.1)
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)  # 注意激活

class MySalEncoder_ResNet34(nn.Module):
    def __init__(self):
        super(MySalEncoder_ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )  # 开始的部分
        self.body = self.makelayers([3, 4, 6, 3])  # 具有重复模块的部分
        self.classifier = nn.Linear(512, 128)  # 末尾的部分

    def makelayers(self, blocklist):  # 注意传入列表而不是解列表
        self.layers = []
        for index, blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64 * 2 ** (index - 1), 64 * 2 ** index, 1, 2, bias=False),
                    #nn.BatchNorm2d(64 * 2 ** index)
                )
                self.layers.append(ResidualBlock(64 * 2 ** (index - 1), 64 * 2 ** index, 2, shortcut))  # 每次变化通道数时进行下采样
            for i in range(0 if index == 0 else 1, blocknum):
                self.layers.append(ResidualBlock(64 * 2 ** index, 64 * 2 ** index, 1))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

