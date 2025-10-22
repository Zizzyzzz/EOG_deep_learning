import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
import os
class Multi_scale_CNN(nn.Module):
    def __init__(self, args):
        super(Multi_scale_CNN, self).__init__()
        self.CNN_branch_1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2)
            # nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=2),
            # nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.CNN_branch_2 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2)
            # nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1),
            # nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # self.featrue_dropout = nn.Dropout(0.5)

        self.CNN_branch_3 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=1)
            # nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1),
            # nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(4480, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, args.class_num)
        )
    def forward(self, x):
        branch_1 = self.CNN_branch_1(x)
        branch_2 = self.CNN_branch_2(x)
        branch_3 = self.CNN_branch_3(x)
        branch_1 = torch.flatten(branch_1, start_dim=-2, end_dim=-1)
        branch_2 = torch.flatten(branch_2, start_dim=-2, end_dim=-1)
        branch_3 = torch.flatten(branch_3, start_dim=-2, end_dim=-1)
        fusion_featrue = torch.cat((branch_1, branch_2,branch_3), dim=1)
        fusion_featrue = self.fc(fusion_featrue)
        # fusion_featrue = self.featrue_dropout(fusion_featrue)
            # if(self.opts["featrue_bottleneck"]):
            #     return fusion_featrue
            # else:
            #     return self.bottle_1(fusion_featrue)
        del branch_1, branch_2
        torch.cuda.empty_cache()
        return fusion_featrue

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(64 * 24, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, args.class_num)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(torch.flatten(x, start_dim=1, end_dim=-1))
        return x

class CNN_net3work(nn.Module):
    def __init__(self, args):
        super(CNN_net3work, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2)
            # nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=2)
            # nn.Conv1d(64, 128,kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=2)
            # nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=2)

        )
        self.fc=nn.Sequential(
            nn.Linear(32 * 49, 256),
            # nn.Linear(256*5, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, args.class_num)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(torch.flatten(x, start_dim=1, end_dim=-1))
        return x


class EOGdataset(Dataset):
    def __init__(self, sample):
        self.raw_data = sample

    def __getitem__(self, idx):
        # data = self.raw_data[0, idx*2:idx*2+2, 0:100] #########conv2d用
        # label = self.raw_data[0, idx*2, -1] #########conv2d用
        data = self.raw_data[idx*2:idx*2+2, 0:100]
        label = self.raw_data[idx*2, -1]
        data = data.reshape(1, 2, 100) #########conv2d用

        return data, label

    def __len__(self):
        # return int(self.raw_data.shape[1] / 2)#########conv2d用
        return int(self.raw_data.shape[0] / 2)


def extract_data_set(i, i_dataset):
    dir = "F:\ZengZhengPhd\zengz\phd1\分数据集跑的EOGdeep\EOG_deep_learning_tiaocan\preprocessing"
    if i_dataset == 0:
        data_dir = os.path.join(dir, "subject_1_npy")
    elif i_dataset == 1:
        data_dir = os.path.join(dir, "subject_2_npy")
    elif i_dataset == 2:
        data_dir = os.path.join(dir, "subject_3_npy")
    elif i_dataset == 3:
        data_dir = os.path.join(dir, "subject_4_npy")
    elif i_dataset == 4:
        data_dir = os.path.join(dir, "subject_5_npy")
    else:
        data_dir = dir

    Blink = np.load(os.path.join(data_dir, "blink_subject_sample.npy"), allow_pickle=True)
    Down = np.load(os.path.join(data_dir, "down_subject_sample.npy"), allow_pickle=True)
    Left = np.load(os.path.join(data_dir, "left_subject_sample.npy"), allow_pickle=True)
    Right = np.load(os.path.join(data_dir, "right_subject_sample.npy"), allow_pickle=True)
    Up = np.load(os.path.join(data_dir, "up_subject_sample.npy"), allow_pickle=True)

    Blink_set = torch.tensor([])
    Down_set = torch.tensor([])
    Left_set = torch.tensor([])
    Right_set = torch.tensor([])
    Up_set = torch.tensor([])
    data_set = torch.tensor([])

    temp_blink = Blink[0][i]
    temp_blink[:, 100] = 0
    temp_blink = torch.tensor(temp_blink)
    Blink_set = torch.cat([Blink_set, temp_blink], 0)

    temp_down = Down[0][i]
    temp_down[:, 100] = 1
    temp_down = torch.tensor(temp_down)
    Down_set = torch.cat([Down_set, temp_down], 0)

    temp_left = Left[0][i]
    temp_left[:, 100] = 2
    temp_left = torch.tensor(temp_left)
    Left_set = torch.cat([Left_set, temp_left], 0)

    temp_right = Right[0][i]
    temp_right[:, 100] = 3
    temp_right = torch.tensor(temp_right)
    Right_set = torch.cat([Right_set, temp_right], 0)

    temp_up = Up[0][i]
    temp_up[:, 100] = 4
    temp_up = torch.tensor(temp_up)
    Up_set = torch.cat([Up_set, temp_up], 0)
    data_set = torch.cat([Blink_set, Down_set, Left_set, Right_set, Up_set], 0)

    return data_set