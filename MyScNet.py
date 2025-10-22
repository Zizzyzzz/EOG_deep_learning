import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import MS_CAM
import torch.utils.model_zoo as model_zoo

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 2 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)
        return out

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
        return torch.cat((p1, p2, p3, p4), dim=1)

class residual_SCNet_block(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(residual_SCNet_block, self).__init__()

        self.in_channel = int(input_channel/4)
        self.branch1 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
        self.branch2 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
        self.branch3 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
        # self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.aft_extract = nn.Sequential(
            nn.Conv2d(self.in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2)
        )

    def forward(self, x):
        split_x = torch.split(x, self.in_channel, dim=1)
        residual = split_x[0]
        y1 = self.branch1(split_x[1])
        x2 = torch.cat([split_x[2], y1], dim=3)
        y2 = self.branch2(x2)
        x3 = torch.cat([split_x[3], y2], dim=3)
        y3 = self.branch3(x3)
        output = torch.cat([residual, y1, y2, y3], dim=3)
        output = self.aft_extract(output)
        return output

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

b2 = nn.Sequential(residual_SCNet_block(64, 64))
b3 = nn.Sequential(residual_SCNet_block(64, 32))
b4 = nn.Sequential(residual_SCNet_block(32, 32))
b5 = nn.Sequential(residual_SCNet_block(32, 32))
channel_atten = MS_CAM(channels=32, r=4)

class residual_SCNet_block_two_layer(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(residual_SCNet_block_two_layer, self).__init__()

        self.in_channel = int(input_channel/2)
        self.branch1 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

        self.aft_extract = nn.Sequential(
            nn.Conv2d(self.in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2)
        )

    def forward(self, x):
        split_x = torch.split(x, self.in_channel, dim=1)
        residual = split_x[0]
        y1 = self.branch1(split_x[1])
        # x2 = torch.cat([split_x[2], y1], dim=3)
        # y2 = self.branch2(x2)
        # x3 = torch.cat([split_x[3], y2], dim=3)
        # y3 = self.branch2(x3)
        output = torch.cat([residual, y1], dim=3)
        output = self.aft_extract(output)
        return output

class residual_SCNet_block_eight_layer(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(residual_SCNet_block_eight_layer, self).__init__()

        self.in_channel = int(input_channel/8)
        self.branch1 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

        self.branch2 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

        self.branch3 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

        self.branch4 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

        self.branch5 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

        self.branch6 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
        self.branch7 = nn.Sequential(SCBottleneck(self.in_channel, self.in_channel, norm_layer=nn.BatchNorm2d),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
        self.aft_extract = nn.Sequential(
            nn.Conv2d(self.in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2)
        )

    def forward(self, x):
        split_x = torch.split(x, self.in_channel, dim=1)
        residual = split_x[0]
        y1 = self.branch1(split_x[1])

        x2 = torch.cat([split_x[2], y1], dim=3)
        y2 = self.branch2(x2)

        x3 = torch.cat([split_x[3], y2], dim=3)
        y3 = self.branch3(x3)

        x4 = torch.cat([split_x[4], y3], dim=3)
        y4 = self.branch4(x4)

        x5 = torch.cat([split_x[5], y4], dim=3)
        y5 = self.branch4(x5)

        x6 = torch.cat([split_x[6], y5], dim=3)
        y6 = self.branch4(x6)

        x7 = torch.cat([split_x[7], y6], dim=3)
        y7 = self.branch4(x7)
        output = torch.cat([residual, y1, y2, y3, y4, y5, y6, y7], dim=3)
        output = self.aft_extract(output)
        return output

b2_twolayer = nn.Sequential(residual_SCNet_block_two_layer(64, 64))

b2_eightlayer = nn.Sequential(residual_SCNet_block_eight_layer(64, 64))
class My_SCNet(nn.Module):
    def __init__(self, args):
        super(My_SCNet, self).__init__()
        self.pre_part = nn.Sequential(b1)
        self.resi_SC_block1 = nn.Sequential(b2)
        self.resi_SC_block2 = nn.Sequential(b3)
        self.resi_SC_block3 = nn.Sequential(b4)
        self.resi_SC_block4 = nn.Sequential(b5)
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.linear = nn.Linear(18432, args.class_num)
    def forward(self, x):
        output = self.pre_part(x)
        output = self.resi_SC_block1(output)
        output = self.resi_SC_block2(output)
        output = self.resi_SC_block3(output)
        output = self.resi_SC_block4(output)
        output = self.flatten(output)
        output = self.linear(output)
        return output

class My_SCNet_attention(nn.Module):
    def __init__(self, args):
        super(My_SCNet_attention, self).__init__()
        self.pre_part = nn.Sequential(b1)
        self.resi_SC_block1 = nn.Sequential(b2)
        self.resi_SC_block2 = nn.Sequential(b3)
        self.resi_SC_block3 = nn.Sequential(b4)
        # self.resi_SC_block4 = nn.Sequential(b5)
        # self.Conv1 =nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=(1, 19), stride=1, padding=0),
        #
        # )
        # self.Conv2 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), padding=0)
        self.MS_CAM = nn.Sequential(channel_atten)
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.linear = nn.Linear(11840, args.class_num)
    def forward(self, x):
        output = self.pre_part(x)
        output = self.resi_SC_block1(output)
        # former_output = self.Conv1(output)
        output = self.resi_SC_block2(output)
        output = self.resi_SC_block3(output)
        # self.featuremap = output.detach()
        # output = self.resi_SC_block4(output)
        # latter_output = self.Conv2(output)
        output_atten = self.MS_CAM(output)
        output_atten = self.flatten(output_atten)
        self.featuremap = output_atten.detach() #
        output_atten = self.linear(output_atten)
        return output_atten

class My_SCNet_oneSC(nn.Module):
    def __init__(self, args):
        super(My_SCNet_oneSC, self).__init__()
        self.pre_part = nn.Sequential(b1)
        self.resi_SC_block1 = nn.Sequential(b2)
        # self.resi_SC_block2 = nn.Sequential(b3)
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.linear = nn.Linear(9856, args.class_num)
    def forward(self, x):
        output = self.pre_part(x)
        output = self.resi_SC_block1(output)
        # output = self.resi_SC_block2(output)
        output = self.flatten(output)
        output = self.linear(output)
        return output

class Only_MS_CAM(nn.Module):
    def __init__(self, args):
        super(Only_MS_CAM, self).__init__()
        self.pre_part = nn.Sequential(b1)
        # self.resi_SC_block1 = nn.Sequential(b2)
        # self.resi_SC_block2 = nn.Sequential(b3)
        self.MS_CAM = nn.Sequential(MS_CAM(channels=64, r=4))
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.linear = nn.Linear(6400, args.class_num)
    def forward(self, x):
        output = self.pre_part(x)
        # output = self.resi_SC_block1(output)
        # output = self.resi_SC_block2(output)
        output_atten = self.MS_CAM(output)
        output = self.flatten(output)
        output = self.linear(output)
        return output