import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import MS_CAM
#这里实现论文中提及的消融实验模型也就是将SC CNN 替换成CNN

channel_atten = MS_CAM(channels=32, r=4)
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

class residual_CNN_block(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(residual_CNN_block, self).__init__()

        self.in_channel = int(input_channel/4)

        self.branch1 = nn.Sequential(nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding = 1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
        self.branch2 = nn.Sequential(nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding = 1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
        self.branch3 = nn.Sequential(nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding = 1),
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

channel_atten = MS_CAM(channels=4, r=4)
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))
b2 = nn.Sequential(residual_CNN_block(64, 64))
b3 = nn.Sequential(residual_CNN_block(64, 32))
b4 = nn.Sequential(residual_CNN_block(32, 4))

class My_CNN_attention(nn.Module):
    def __init__(self, args):
        super(My_CNN_attention, self).__init__()
        self.pre_part = nn.Sequential(b1)
        self.resi_CNN_block1 = nn.Sequential(b2)
        self.resi_CNN_block2 = nn.Sequential(b3)
        self.resi_CNN_block3 = nn.Sequential(b4)

        self.MS_CAM = nn.Sequential(channel_atten)
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.linear = nn.Linear(1480, args.class_num)
    def forward(self, x):
        output = self.pre_part(x)
        output = self.resi_CNN_block1(output)
        # former_output = self.Conv1(output)
        output = self.resi_CNN_block2(output)
        output = self.resi_CNN_block3(output)
        self.featuremap = output.detach()

        output_atten = self.MS_CAM(output)
        output_atten = self.flatten(output_atten)
        # self.featuremap = output_atten.detach() #
        output_atten = self.linear(output_atten)
        return output_atten