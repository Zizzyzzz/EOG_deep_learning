import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
import os



def config_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_train_validation_data_set(i, i_dataset, percent):
    config_random_seed(0)
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

    Blink_val_set = torch.tensor([])
    Down_val_set = torch.tensor([])
    Left_val_set = torch.tensor([])
    Right_val_set = torch.tensor([])
    Up_val_set = torch.tensor([])
    val_data_set = torch.tensor([])

    temp_blink = Blink[0][i]
    temp_blink[:, 100] = 0
    num = int(np.size(temp_blink, axis=0) * percent)
    if (num % 2) != 0:
        num = num + 1
    temp_val_blink = temp_blink[0:num, :]
    temp_blink = temp_blink[num:, :]
    temp_blink = torch.tensor(temp_blink)
    temp_val_blink = torch.tensor(temp_val_blink)
    Blink_set = torch.cat([Blink_set, temp_blink], 0)
    Blink_val_set = torch.cat([Blink_val_set, temp_val_blink], 0)

    temp_down = Down[0][i]
    temp_down[:, 100] = 1
    num = int(np.size(temp_down, axis=0) * percent)
    if (num % 2) != 0:
        num = num + 1
    temp_val_down = temp_down[0:num, :]
    temp_down = temp_down[num:, :]
    temp_down = torch.tensor(temp_down)
    temp_val_down = torch.tensor(temp_val_down)
    Down_set = torch.cat([Down_set, temp_down], 0)
    Down_val_set = torch.cat([Down_val_set, temp_val_down], 0)

    temp_left = Left[0][i]
    temp_left[:, 100] = 2
    num = int(np.size(temp_left, axis=0) * percent)
    if (num % 2) != 0:
        num = num + 1
    temp_val_left = temp_left[0:num, :]
    temp_left = temp_left[num:, :]
    temp_left = torch.tensor(temp_left)
    temp_val_left = torch.tensor(temp_val_left)
    Left_set = torch.cat([Left_set, temp_left], 0)
    Left_val_set = torch.cat([Left_val_set, temp_val_left], 0)

    temp_right = Right[0][i]
    temp_right[:, 100] = 3
    num = int(np.size(temp_right, axis=0) * percent)
    if (num % 2) != 0:
        num = num + 1
    temp_val_right = temp_right[0:num, :]
    temp_right = temp_right[num:, :]
    temp_right = torch.tensor(temp_right)
    temp_val_right = torch.tensor(temp_val_right)
    Right_set = torch.cat([Right_set, temp_right], 0)
    Right_val_set = torch.cat([Right_val_set, temp_val_right], 0)

    temp_up = Up[0][i]
    temp_up[:, 100] = 4
    num = int(np.size(temp_up, axis=0) * percent)
    if (num % 2) != 0:
        num = num + 1
    temp_val_up = temp_up[0:num, :]
    temp_up = temp_up[num:, :]
    temp_up = torch.tensor(temp_up)
    temp_val_up = torch.tensor(temp_val_up)
    Up_set = torch.cat([Up_set, temp_up], 0)
    Up_val_set = torch.cat([Up_val_set, temp_val_up], 0)

    data_set = torch.cat([Blink_set, Down_set, Left_set, Right_set, Up_set], 0)
    val_data_set = torch.cat([Blink_val_set, Down_val_set, Left_val_set, Right_val_set, Up_val_set], 0)
    return data_set, val_data_set