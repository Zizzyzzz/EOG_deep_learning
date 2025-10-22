import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
import os


def Metric_count(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    Blink_pred = np.where(y_pred == 0)[0]
    Blink_true = np.where(y_true == 0)[0]
    Blink_not_Pred = np.where(y_pred != 0)[0]
    Blink_not_true = np.where(y_true != 0)[0]
    TP_Blink = len(set(Blink_pred) & set(Blink_true))
    FN_Blink = len(Blink_true) - TP_Blink
    TN_Blink =len(set(Blink_not_Pred) & set(Blink_not_true))
    FP_Blink = len(Blink_not_true) - TN_Blink
    Blink_sen = TP_Blink / (TP_Blink+FN_Blink)
    Blink_spe = TN_Blink / (TN_Blink+FP_Blink)
    Blink_acc = TP_Blink / len(Blink_true)

    Down_pred = np.where(y_pred == 1)[0]
    Down_true = np.where(y_true == 1)[0]
    Down_not_Pred = np.where(y_pred != 1)[0]
    Down_not_true = np.where(y_true != 1)[0]
    TP_Down = len(set(Down_pred) & set(Down_true))
    FN_Down= len(Down_true) - TP_Down
    TN_Down = len(set(Down_not_Pred) & set(Down_not_true))
    FP_Down = len(Down_not_true) - TN_Down
    Down_sen = TP_Down / (TP_Down + FN_Down)
    Down_spe = TN_Down / (TN_Down + FP_Down)
    Down_acc = TP_Down / len(Down_true)

    Left_pred = np.where(y_pred == 2)[0]
    Left_true = np.where(y_true == 2)[0]
    Left_not_Pred = np.where(y_pred != 2)[0]
    Left_not_true = np.where(y_true != 2)[0]
    TP_Left = len(set(Left_pred) & set(Left_true))
    FN_Left = len(Left_true) - TP_Left
    TN_Left = len(set(Left_not_Pred) & set(Left_not_true))
    FP_Left = len(Left_not_true) - TN_Left
    Left_sen = TP_Left / (TP_Left + FN_Left)
    Left_spe = TN_Left / (TN_Left + FP_Left)
    Left_acc = TP_Left / len(Left_true)

    Right_pred = np.where(y_pred == 3)[0]
    Right_true = np.where(y_true == 3)[0]
    Right_not_Pred = np.where(y_pred != 3)[0]
    Right_not_true = np.where(y_true != 3)[0]
    TP_Right = len(set(Right_pred) & set(Right_true))
    FN_Right = len(Right_true) - TP_Right
    TN_Right = len(set(Right_not_Pred) & set(Right_not_true))
    FP_Right = len(Right_not_true) - TN_Right
    Right_sen = TP_Right / (TP_Right + FN_Right)
    Right_spe = TN_Right / (TN_Right + FP_Right)
    Right_acc = TP_Right / len(Right_true)

    Up_pred = np.where(y_pred == 4)[0]
    Up_true = np.where(y_true == 4)[0]
    Up_not_Pred = np.where(y_pred != 4)[0]
    Up_not_true = np.where(y_true != 4)[0]
    TP_Up = len(set(Up_pred) & set(Up_true))
    FN_Up = len(Up_true) - TP_Up
    TN_Up = len(set(Up_not_Pred) & set(Up_not_true))
    FP_Up = len(Up_not_true) - TN_Up
    Up_sen = TP_Up / (TP_Up + FN_Up)
    Up_spe = TN_Up / (TN_Up + FP_Up)
    Up_acc = TP_Up / len(Up_true)

    Sen = (Blink_sen + Down_sen + Left_sen+Right_sen+Up_sen)/5
    Spe = (Blink_spe+Down_spe+Left_spe+Right_spe+Up_spe)/5

    return Sen,Spe, Blink_acc, Down_acc, Left_acc, Right_acc, Up_acc