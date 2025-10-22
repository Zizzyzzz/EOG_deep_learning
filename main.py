# This is a sample Python script.
import numpy as np
# from torchstat import stat
import torch
from thop import profile
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Transformert_experiment import My_Transfromer
import torch.nn.functional as F
from argparse import ArgumentParser
import pathlib
import os
import argparse
import glob
from tqdm import tqdm
from CNN_model2 import CNN_eyesay, VGG_NC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from setting import Config
from CNN_model import Classifier
from CNN_model import CNN_net3work
from CNN_model import Multi_scale_CNN
from CNN_model import extract_data_set
from CNN_model import EOGdataset
from Count_metric import Metric_count
from Extract_train_validation import extract_train_validation_data_set
import random
# import VggTEST
# from attention_model import googlenet_1and2
# from attention_model import One_inception
# from MyScNet import My_SCNet
from MyScNet import My_SCNet_attention
# from ablation_CNN import My_CNN_attention
# from inception1and2 import googlenet_1and2
# from inception121 import googlenet_121
def config_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def construct_LOSO_set(opts, test_sub_id, i_dataset):
     # bilnk-0 Down-1 Left-2 Right-3 Up-4
    # random_number = random.sample(range(1, 50), 1)
    # while random_number == test_sub_id:
    #     random_number = random.sample(range(1, 50), 1)
    dataset_num = np.array([5, 15, 10, 15, 14])
    train_set = torch.tensor([])
    val_set = torch.tensor([])
    test_set = torch.tensor([])
    for i in range(dataset_num[i_dataset]):
        if i == test_sub_id:
            # temp = extract_data_set(i)
            # val_set = temp
            test_set = extract_data_set(i, i_dataset)

        # elif i == test_sub_id:
        #     test_set = extract_data_set(i)
        else:
            temp_train, temp_val = extract_train_validation_data_set(i, i_dataset, percent=0.2)
            # temp = temp.reshape(-1, 1, 101) ################conv2d用
            train_set = torch.cat([train_set, temp_train], 0)
            val_set = torch.cat([val_set, temp_val], 0)

    print('The train/validatiion/test set have splited over,test_subject is ', test_sub_id+1)
    train_dataset = EOGdataset(sample=train_set)
    val_dataset = EOGdataset(sample=val_set)
    test_dataset = EOGdataset(sample=test_set)


    train_data_loader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                   pin_memory=True, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=True,
                                 shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                  pin_memory=True, shuffle=False)
    return train_data_loader, val_data_loader, test_data_loader

def train_c (iterator, model, optimizer, epoch):
    print(f"being to train {epoch}")
    model.train()
    loop = tqdm(enumerate(iterator), total=len(iterator), leave=False)
    for idx, batch in loop:
        optimizer.zero_grad()
        y_true = batch[1].to(device)
        y_pred = model(batch[0].type(torch.float32).to(device))
        loss = loss_criterion(y_pred, y_true.long())
        # wirte scalars to monitor
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(loss=loss.item())
        writer.add_scalar('Loss/train', loss.item(), idx + epoch * len(loop))
    print("Train loss: {:.6f}  ".format(loss))

def val_c(iterator, model, epoch):
    model.eval()
    print("begin to validataion")
    val_loss = []
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            y_true = (batch[1].to(device))
            y_pred = model(batch[0].type(torch.float32).to(device))
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            loss = loss_criterion(y_pred, y_true.long())
            val_loss.append(loss)
    val_loss = torch.stack(val_loss).mean()
    val_loss = torch.stack([val_loss], dim=0).mean()
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    y_pred = F.softmax(y_pred, dim=-1)
    y_pred = torch.argmax(y_pred, dim=-1)
    c = (y_true == y_pred).squeeze()
    acc = c.sum().type(torch.float32) / len(y_true)
    print("Epoch: {:.0f} Val Acc: {:.6f} Val_loss: {:.6f} ".format(epoch, acc, val_loss))
    # writer.add_scalar("Loss/val", val_loss.item(), epoch)
    # writer.add_scalar("Acc/val", acc.item(), epoch)
    print("validation end")
    del y_true
    del y_pred
    return val_loss, acc



def test_c(iterator, model,save_path):
    print("begin to test")
    model.eval()
    test_loss = []
    y_true_list = []
    y_prob_list = []
    for idx, batch in enumerate(iterator):
        y_true = (batch[1].to(device))
        y_out = model(batch[0].type(torch.float32).to(device))
        y_prob = F.softmax(y_out, dim=-1)
        y_true_list.append(y_true.detach().cpu().numpy())
        y_prob_list.append(y_prob.detach().cpu().numpy())
        loss = loss_criterion(y_out, y_true.long())
        test_loss.append(loss.item())
    test_loss = np.hstack(test_loss).mean()
    y_true = np.hstack(y_true_list)
    y_prob = np.vstack(y_prob_list)

    y_pred = np.argmax(y_prob, axis=-1)
    Metric_count(y_true, y_pred)
    c = np.where(y_true == y_pred)[0]
    acc = len(c) / len(y_true)
    Sen, Spe, Blink_acc, Down_acc, Left_acc, Right_acc, Up_acc = Metric_count(y_true, y_pred)
    print("Epoch: {:.0f} Test Acc: {:.6f}  ".format(epoch, acc))
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Acc/test", acc, epoch)
    # np.savez(os.path.join(save_path, "resut.npz"), y_pred=y_pred, y_true=y_true, y_prob=y_prob)
    print("test end")
    return acc, Sen, Spe, Blink_acc, Down_acc, Left_acc, Right_acc, Up_acc


def test_c_final(iterator, model,save_path):
    print("begin to test")
    model.eval()
    test_loss = []
    y_true_list = []
    y_prob_list = []
    for idx, batch in enumerate(iterator):
        y_true = (batch[1].to(device))
        y_out = model(batch[0].type(torch.float32).to(device))
        y_prob = F.softmax(y_out, dim=-1)
        y_true_list.append(y_true.detach().cpu().numpy())
        y_prob_list.append(y_prob.detach().cpu().numpy())
        loss = loss_criterion(y_out, y_true.long())
        test_loss.append(loss.item())
    test_loss = np.hstack(test_loss).mean()
    y_true = np.hstack(y_true_list)
    y_prob = np.vstack(y_prob_list)

    y_pred = np.argmax(y_prob, axis=-1)
    Metric_count(y_true, y_pred)
    c = np.where(y_true == y_pred)[0]
    acc = len(c) / len(y_true)
    Sen, Spe, Blink_acc, Down_acc, Left_acc, Right_acc, Up_acc = Metric_count(y_true, y_pred)
    print("Epoch: {:.0f} Test Acc: {:.6f}  ".format(epoch, acc))
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Acc/test", acc, epoch)
    # np.savez(os.path.join(save_path, "resut.npz"), y_pred=y_pred, y_true=y_true, y_prob=y_prob)
    print("test end")
    return acc, Sen, Spe, Blink_acc, Down_acc, Left_acc, Right_acc, Up_acc, y_pred, y_true


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, default=r"./result")
    parser.add_argument("--tb_log_dir", type=str, default="./tb_logs")
    # experiments settings
    parser.add_argument("--mode", type=str, default="maintance")
    parser.add_argument("--class_num", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--lr', type=float, default=5*1e-3) #8*1e-3
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=int, default=0.4*1e-3)
    parser.add_argument('--drop_out_prob', type=int, default=.2)


    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)  # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_random_seed(0)
    dataset_num = np.array([5, 15, 10, 15, 14])

    test_acc = []
    for i_dataset in range(5):
        best_epoch = np.zeros((dataset_num[i_dataset], 1))
        Spe_epoch = np.zeros((dataset_num[i_dataset], 1))
        Sen_epoch = np.zeros((dataset_num[i_dataset], 1))
        Blink_epoch = np.zeros((dataset_num[i_dataset], 1))
        Down_epoch = np.zeros((dataset_num[i_dataset], 1))
        Left_epoch = np.zeros((dataset_num[i_dataset], 1))
        Right_epoch = np.zeros((dataset_num[i_dataset], 1))
        Up_epoch = np.zeros((dataset_num[i_dataset], 1))
        print('正在分析的数据集是', i_dataset)
        confuse_matrix = np.zeros([5, 5])
        for test_sub_id in range(dataset_num[i_dataset]):
            # test_sub_id = 30
            # model = Classifier(opts).to(device)
            # model = Multi_scale_CNN(opts).to(device)
            # model = VggTEST.net3(opts).to(device)
            # model = VggTEST.vgg_sequen_al(opts).to(device)
            # model = inception1.googlenet_bk2(opts).to(device)
            # model = inception1and3.googlenet_1and3(opts).to(device)
            model = My_SCNet_attention(opts).to(device)

            #用于计算模型参数
            # model = My_SCNet_attention(opts)
            input1 = torch.randn(256, 1, 2, 100).to(device)
            flops, params = profile(model, inputs=(input1,))
            print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            print('Params = ' + str(params / 1000 ** 2) + 'M')

            # model_best = deepcopy(model)
            train_data_loader, val_data_loader, test_data_loader = construct_LOSO_set(opts, test_sub_id, i_dataset)

            optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
            loss_criterion = nn.CrossEntropyLoss()
            writer = SummaryWriter(log_dir= os.path.join(opts.tb_log_dir, str(test_sub_id)))
            # scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.4, last_epoch=-1)
            best_acc = 0
            for epoch in range(opts.max_epochs):
                train_c(train_data_loader, model, optimizer, epoch)  # train
                val_loss, val_acc = val_c(val_data_loader, model, epoch)  # validataion
                scheduler.step()
                if val_acc > best_acc:
                    best_acc = val_acc
                    model_best = deepcopy(model)
                    print('更新了新的模型')
                    nothing = test_c(test_data_loader, model_best, opts.output_path)
            acc, Sen, Spe, Blink_acc, Down_acc, Left_acc, Right_acc, Up_acc, y_pred, y_true = test_c_final(test_data_loader, model_best, opts.output_path)
            best_epoch[test_sub_id][0] = acc
            Sen_epoch[test_sub_id][0] = Sen
            Spe_epoch[test_sub_id][0] = Spe
            Blink_epoch[test_sub_id][0] = Blink_acc
            Down_epoch[test_sub_id][0] = Down_acc
            Right_epoch[test_sub_id][0] = Right_acc
            Left_epoch[test_sub_id][0] = Left_acc
            Up_epoch[test_sub_id][0] = Up_acc
            for i in range(len(y_pred)):
                a = int(y_pred[i])
                b = int(y_true[i])
                confuse_matrix[a][b] = confuse_matrix[a][b]+1
            a=1

                # if acc > best_acc:
                #     best_acc = acc
                #     model_best = deepcopy(model)
                # scheduler.step(val_loss)  # update learning
            # test_c_acc = test_c(test_data_loader, model, opts.output_path)
            # test_best_acc = test_c(test_data_loader, model_best, opts.output_path)
            # print('第x个人的最高准确率是x', test_sub_id, test_best_acc)
            # test_acc.append(test_c_acc)
        # test_acc = np.array(test_acc)
        # np.save("multi3_lr8_gama0.4_step50_epoch200_acc.npy", test_acc)
        np.save(str(i_dataset) + "branch_best_epoch200", best_epoch)
        np.save(str(i_dataset) + "branch_Sen_epoch200", Sen_epoch)
        np.save(str(i_dataset) + "branch_Spe_epoch200", Spe_epoch)
        np.save(str(i_dataset) + "branch_Blink_epoch200", Blink_epoch)
        np.save(str(i_dataset) + "branch_Down_epoch200", Down_epoch)
        np.save(str(i_dataset) + "branch_Right_epoch200", Right_epoch)
        np.save(str(i_dataset) + "branch_Left_epoch200", Left_epoch)
        np.save(str(i_dataset) + "branch_Up_epoch200", Up_epoch)
        np.save(str(i_dataset) + "Confuse_matrix", confuse_matrix)
        a=1

