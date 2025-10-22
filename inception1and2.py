
import torch
from torch import nn
from torch.nn import functional as F
from argparse import ArgumentParser
# from torchsummary import summary
# import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--class_num", type=int, default=5)
    parser.add_argument('--drop_out_prob', type=int, default=.2)

    args = parser.parse_args()
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        'kernel size must be 3 or 7'
        padding = 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 压缩通道
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # 压缩通道
        x = torch.cat([avg_out, max_out], dim=1)  # [b, 1, h, w]
        x = self.conv1(x)
        return self.sigmoid(x)



class Inception1(nn.Module): #class A(B)是类A继承了类B的方法。这里就是inception集成了nn.module的方法
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs): #**kwage是将除了前面显示列出的参数外的其他参数, 以dict结构进行接收.
        super(Inception1, self).__init__(**kwargs) #这里的super语法是按照python2.x语法写的，在python3.x里面可以直接写成 super().__init__(**kwargs)
        #这个从父类nn.Module继承实例初始化方法干的是一个将新来的其他参数按照nn.Module的初始化方法添加到子类Inception所创建的实例中。
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1)
        self.p3_3 = nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x)) #这个F是前面import的functional，这里就是吧第一层输出放到relu里面，输出p1作为path1的输出
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_3(F.relu(self.p3_2(F.relu(self.p3_1(x))))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1) #批量/样本输入都是dim=0也就是行，输出/特征/通道数是dim=1给concat起来


class Inception2(nn.Module): #class A(B)是类A继承了类B的方法。这里就是inception集成了nn.module的方法
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs): #**kwage是将除了前面显示列出的参数外的其他参数, 以dict结构进行接收.
        super(Inception2, self).__init__(**kwargs) #这里的super语法是按照python2.x语法写的，在python3.x里面可以直接写成 super().__init__(**kwargs)
        #这个从父类nn.Module继承实例初始化方法干的是一个将新来的其他参数按照nn.Module的初始化方法添加到子类Inception所创建的实例中。
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=(3, 1), padding=(1, 0))
        self.p2_3 = nn.Conv2d(c2[1], c2[2], kernel_size=(1, 3), padding=(0, 1))
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(3, 1), padding=(1, 0))
        self.p3_3 = nn.Conv2d(c3[1], c3[2], kernel_size=(1, 7), padding=(0, 3))
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x)) #这个F是前面import的functional，这里就是吧第一层输出放到relu里面，输出p1作为path1的输出
        p2 = F.relu(self.p2_3(F.relu(self.p2_2(F.relu(self.p2_1(x))))))
        p3 = F.relu(self.p3_3(F.relu(self.p3_2(F.relu(self.p3_1(x))))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))


b2 = nn.Sequential(Inception1(64, 32, (32, 64), (32, 64, 96), 16),
                   nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))


b3 = nn.Sequential(Inception2(208, 64, (96, 128, 160), (128, 160, 192), 32),
                   # nn.MaxPool2d(kernel_size=2, padding=1, stride=2),
                   nn.AdaptiveMaxPool2d((1, 1)),
                   nn.Flatten())
# b3 = nn.Sequential(Inception2(208, 64, (96, 128, 160), (128, 160, 192), 32))
                   # nn.MaxPool2d(kernel_size=2, padding=1, stride=2),)

attention = nn.Sequential(SpatialAttention(3))
# def googlenet_bk1(args):
#     return nn.Sequential(b1, b2, b3, nn.Linear(448, args.class_num))  # 构造器里面是可以套构造器的。
# def googlenet_bk2(args):
#     return nn.Sequential(b1, b2, b3, nn.Linear(448, args.class_num))  # 构造器里面是可以套构造器的
#

class googlenet_1and2(nn.Module): 
    def __init__(self, args): #**kwage是将除了前面显示列出的参数外的其他参数, 以dict结构进行接收.
        super(googlenet_1and2, self).__init__()
        self.model = nn.Sequential(b1, b2, b3)
        self.spatial_attention = nn.Sequential(attention)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, args.class_num)
        self.lstm = nn.LSTM(input_size=448, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True,
                            dropout=0.5)
    def forward(self, x):
        temp = self.model(x)
        temp = temp.reshape(-1, 1, 448)
        temp,_ = self.lstm(temp)
        temp = self.flatten(temp)
        temp = self.linear(temp)
        return temp


# net = googlenet_1and2(args)
# # #
# # #
# # #
# # #
# X = torch.rand(size=(1, 1, 2, 100))
# # # for layer in net:
# # #     X = layer(X)
# # #     print(layer.__class__.__name__,'output shape:\t', X.shape)
# summary(net, (1, 2, 100), batch_size=1, device="cpu")
