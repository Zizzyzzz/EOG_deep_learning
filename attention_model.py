
import torch
from torch import nn
from torch.nn import functional as F
import copy
import math
from copy import deepcopy
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3):
#         super(SpatialAttention, self).__init__()
#
#         'kernel size must be 3 or 7'
#         padding = 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # 压缩通道
#         max_out, _ = torch.max(x, dim=1, keepdim=True)   # 压缩通道
#         x = torch.cat([avg_out, max_out], dim=1)  # [b, 1, h, w]
#         x = self.conv1(x)
#         return self.sigmoid(x)
# class EncoderLayer(nn.Module):
#     '''
#     An encoder layer
#
#     Made up of self-attention and a feed forward layer.
#     Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
#     '''
#     def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
#         self.size = size
#
#
#
#     def forward(self, x_in):
#         "Transformer Encoder"
#         x = self.sublayer_output[0](x_in, lambda x: self.self_attn(x_in))  # Encoder self-attention
#         return self.sublayer_output[1](x, self.feed_forward)
#
class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model=50, d_ff=100, dropout=0.2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
#
class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
#
class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
#
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
#
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.2):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(nn.Conv1d(448, 448, kernel_size=1, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query):
        "Implements Multi-head attention"
        nbatches = query.size(0)
        key = query
        value = query
        query = self.convs[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        torch.cuda.empty_cache()
        return self.linear(x)

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x, x, x)
        # Apply normalization and dropout
        x = self.norm1(x + self.dropout1(attn_output))

        # Apply feed-forward network
        ff_output = self.ff(x)
        # Apply normalization and dropout
        x = self.norm2(x + self.dropout2(ff_output))

        return x

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
                   nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2),
                   )

b3 = nn.Sequential(Inception2(208, 64, (96, 128, 160), (128, 160, 192), 32),
                   nn.Flatten(start_dim=-2, end_dim=-1))
                   # nn.MaxPool2d(kernel_size=2, padding=1, stride=2),)
# attn = TransformerBlock(input_dim=50, num_heads=1, ff_dim=128, dropout=0.2)
# ff = PositionwiseFeedForward(50, 100, 0.2)
# attention = nn.Sequential(SpatialAttention(3))
# def googlenet_bk1(args):
#     return nn.Sequential(b1, b2, b3, nn.Linear(448, args.class_num))  # 构造器里面是可以套构造器的。
# def googlenet_bk2(args):
#     return nn.Sequential(b1, b2, b3, nn.Linear(448, args.class_num))  # 构造器里面是可以套构造器的
#
class One_inception(nn.Module):  #w8 2 run
    def __init__(self, args): #**kwage是将除了前面显示列出的参数外的其他参数, 以dict结构进行接收.
        super(One_inception, self).__init__()
        self.model = nn.Sequential(b1, b2, nn.Flatten(), nn.Linear(10400, args.class_num))
    def forward(self, x):
        temp = self.model(x)
        return temp

class googlenet_1and2(nn.Module):
    def __init__(self, args): #**kwage是将除了前面显示列出的参数外的其他参数, 以dict结构进行接收.
        super(googlenet_1and2, self).__init__()
        self.inception_1 = nn.Sequential(b1, b2)
        self.inception_2 = nn.Sequential(b3)
        self.pool = nn.MaxPool2d(kernel_size=7, stride=3)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.early_fusion = nn.Sequential(
            nn.Conv1d(208, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
        )
        self.later_fusion = nn.Sequential(
            nn.Conv1d(448, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
        )
        self.linear = nn.Sequential(
         nn.Linear(1536, args.class_num)
        )
        self.linear2 =nn.Sequential(nn.Linear(1536, args.class_num))

    def forward(self, x):
        # # early_and_late_fusion
        early_temp = self.inception_1(x)
        late_temp = self.inception_2(early_temp)
        early_temp = self.flatten(early_temp)
        early_temp = self.early_fuiision(early_temp)
        late_temp = self.later_fusion(late_temp)
        early_temp = torch.flatten(early_temp, start_dim=-2, end_dim=-1)
        late_temp = torch.flatten(late_temp, start_dim=-2, end_dim=-1)
        early_temp = self.linear(early_temp)
        late_temp = self.linear2(late_temp)
        fusion_feature = early_temp+late_temp
        # # only early_fusion

        return fusion_feature


class googlenet_1and2_early(nn.Module):
    def __init__(self, args):  # **kwage是将除了前面显示列出的参数外的其他参数, 以dict结构进行接收.
        super(googlenet_1and2_early, self).__init__()
        self.inception_1 = nn.Sequential(b1, b2)
        self.inception_2 = nn.Sequential(b3)
        self.pool = nn.MaxPool2d(kernel_size=7, stride=3)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.early_fusion = nn.Sequential(
            nn.Conv1d(208, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
        )
        # self.later_fusion = nn.Sequential(
        #     nn.Conv1d(448, 32, kernel_size=5, stride=1, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(.2),
        #     nn.Dropout(args.drop_out_prob),
        # )
        self.linear = nn.Sequential(
            nn.Linear(23936, args.class_num)
        )
        # self.linear2 = nn.Sequential(nn.Linear(1536, args.class_num))

    def forward(self, x):
        # # early_and_late_fusion
        early_temp = self.inception_1(x)
        late_temp = self.inception_2(early_temp)
        early_temp = self.flatten(early_temp)
        early_temp = self.early_fusion(early_temp)
        # late_temp = self.later_fusion(late_temp)
        early_temp = torch.flatten(early_temp, start_dim=-2, end_dim=-1)
        late_temp = torch.flatten(late_temp, start_dim=-2, end_dim=-1)
        # early_temp = self.linear(early_temp)
        # late_temp = self.linear2(late_temp)
        fusion_feature = torch.cat((early_temp, late_temp), dim=1)
        fusion_feature = self.linear(fusion_feature)
        # # only early_fusion

        return fusion_feature

class googlenet_1and2_late(nn.Module):
    def __init__(self, args):  # **kwage是将除了前面显示列出的参数外的其他参数, 以dict结构进行接收.
        super(googlenet_1and2_late, self).__init__()
        self.inception_1 = nn.Sequential(b1, b2)
        self.inception_2 = nn.Sequential(b3)
        self.pool = nn.MaxPool2d(kernel_size=7, stride=3)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        # self.early_fusion = nn.Sequential(
        #     nn.Conv1d(208, 32, kernel_size=5, stride=1, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(.2),
        #     nn.Dropout(args.drop_out_prob),
        # )
        self.later_fusion = nn.Sequential(
            nn.Conv1d(448, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
        )
        self.linear = nn.Sequential(
            nn.Linear(11936, args.class_num)
        )
        # self.linear2 = nn.Sequential(nn.Linear(1536, args.class_num))

    def forward(self, x):
        # # early_and_late_fusion
        early_temp = self.inception_1(x)
        late_temp = self.inception_2(early_temp)
        early_temp = self.flatten(early_temp)
        # early_temp = self.early_fusion(early_temp)
        late_temp = self.later_fusion(late_temp)
        early_temp = torch.flatten(early_temp, start_dim=-2, end_dim=-1)
        late_temp = torch.flatten(late_temp, start_dim=-2, end_dim=-1)
        # early_temp = self.linear(early_temp)
        # late_temp = self.linear2(late_temp)
        fusion_feature = torch.cat((early_temp, late_temp), dim=1)
        fusion_feature = self.linear(fusion_feature)
        # # only early_fusion

        return fusion_feature
# if __name__ == '__main__':
#     a = torch.rand(size=(64, 448, 50))
#     b=1
#     h= 5
#     d_model = 50
#     attn = TransformerBlock(input_dim=50, num_heads=5, ff_dim=128, dropout=0.1)
#     k = attn(a)
#     b=1