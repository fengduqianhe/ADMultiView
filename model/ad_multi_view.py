import torch
from tqdm import tqdm
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from wrap.ADNI import AdniDataSet
from setting import parse_opts
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import roc_curve, auc

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [5]))
start = datetime.now()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class single_net(nn.Module):
    def __init__(self, f=4):
        super(single_net, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1',
                               nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2,
                                         padding=2,
                                         dilation=2))
        self.layer1.add_module('bn1', nn.BatchNorm2d(num_features=8))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool2d(kernel_size=2, stride=1))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=2,
                                         padding=2,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm2d(num_features=32))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool2d(kernel_size=2, stride=1))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2,
                                         padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm2d(num_features=64))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool2d(kernel_size=2, stride=1))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv3',
                               nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2,
                                         padding=2,
                                         dilation=2))
        self.layer4.add_module('bn3', nn.BatchNorm2d(num_features=128))
        self.layer4.add_module('relu3', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling3', nn.MaxPool2d(kernel_size=2, stride=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class FirstNet(nn.Module):

    def __init__(self, f=8):
        super(FirstNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                  dilation=1))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=0,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm3d(num_features=16 * f))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer4.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer4.add_module('relu4', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))

        # self.fc = nn.Sequential()
        self.fc1 = nn.Linear(512+128*3, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        self.fc_att1 = nn.Linear(90 * 8, 128)
        self.fc_att2 = nn.Linear(90 * 8, 128)
        self.fc_att3 = nn.Linear(90 * 8, 128)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.single_list1 = nn.ModuleList([single_net(f=4) for i in range(90)])
        self.single_list2 = nn.ModuleList([single_net(f=4) for i in range(90)])
        self.single_list3 = nn.ModuleList([single_net(f=4) for i in range(90)])
        # self.single = single_net(f=4)


    def forward(self, x1, x2, x3, x4):
        branch1 = []
        branch2 = []
        branch3 = []

        for i_num in range(0, 90):
            branch1.append(self.single_list1[i_num].forward(x2[:, i_num]).unsqueeze(1))
            branch2.append(self.single_list2[i_num].forward(x3[:, i_num]).unsqueeze(1))
            branch3.append(self.single_list3[i_num].forward(x4[:, i_num]).unsqueeze(1))
        attention_x1 = torch.cat(branch1, dim=1).squeeze()
        attention_x2 = torch.cat(branch2, dim=1).squeeze()
        attention_x3 = torch.cat(branch3, dim=1).squeeze()

        attention_x1 = attention_x1.reshape((attention_x1.shape[0], -1))
        attention_x2 = attention_x2.reshape((attention_x2.shape[0], -1))
        attention_x3 = attention_x3.reshape((attention_x3.shape[0], -1))

        attention_x1 = self.fc_att1(attention_x1)
        attention_x2 = self.fc_att2(attention_x2)
        attention_x3 = self.fc_att3(attention_x3)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.shape[0], -1)
        # x1 = torch.cat((attention_x1, attention_x2, attention_x3), 1)
        x1 = torch.cat((x1, attention_x1, attention_x2, attention_x3), 1)

        # x1 = torch.cat((x1, attention_x3), 1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        return x1, attention_x1, attention_x2, attention_x3


if __name__ == "__main__":
    sets = parse_opts()
    sets.gpu_id = [0]

    '''AD VS NC'''

    train_data_path = '/ADNIMERGE_ADNI1_BL_PROCESSED_AD_NC.csv'
    val_data_path = '/ADNIMERGE_ADNI2_BL_PROCESSED_AD_NC.csv'
    
    train_img_path = '/BL818_processed/'
    val_img_path = '/BL776_processed/'

    train_data_path = '/ADNIMERGE_ADNI2_BL_PROCESSED_AD_NC.csv'
    val_data_path = '/ADNIMERGE_ADNI1_BL_PROCESSED_AD_NC.csv'
    
    train_img_path = '/data1/qiaohezhe/MRI_MMSE/BL776_processed/'
    val_img_path =  '/data1/qiaohezhe/MRI_MMSE/BL818_processed/'


    train_data = AdniDataSet(train_data_path, train_img_path, sets)
    val_data = AdniDataSet(val_data_path, val_img_path, sets)
    """将训练集划分为训练集和验证集"""
    # train_db, val_db = torch.utils.data.random_split(train_data, [700, 180])
    print('train:', len(train_data), 'validation:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=12, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=12, shuffle=True)
    print("Train data load success")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    early_stopping = EarlyStopping(patience=50, verbose=True)
    model.to(device)
    result_list = []
    epochs = 100

    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct = 0
        total = 0
        model.train()
        running_loss = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, patch1, patch2, patch3, labels = data
            inputs, patch1, patch2, patch3, labels = inputs.to(device), patch1.to(device), patch2.to(device), \
                                                     patch3.to(device), labels.to(device)

            optimizer.zero_grad()
            logps, branch_logps1, branch_logps2, branch_logps3 = model.forward(inputs, patch1.float(), patch2.float(),
                                                                               patch3.float())
            loss = criterion(logps, labels)
            # _, predict = torch.max(logps, 1)
            print(torch.softmax(logps, dim=1)[0, :])
            predict = torch.softmax(logps, dim=1)[:, 1]
            print("predict", predict)

            predict[predict >= 0.6] = 1
            predict[predict < 0.6] = 0
            predict = predict.long()
            labels = labels.long()

            print("train ground truth", labels)
            print("train predict", predict)

            correct += (predict == labels).sum().item()
          
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, running_loss / len(train_loader)))
        print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
        running_loss = 0
        correct = 0
        total = 0
        classnum = 2
        target_num = torch.zeros((1, classnum))
        predict_num = torch.zeros((1, classnum))
        acc_num = torch.zeros((1, classnum))
        model.eval()
        roc_label = []
        roc_predict = []
        with torch.no_grad():
            print("validation...")
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                inputs, patch1, patch2, patch3, labels = data
                inputs, patch1, patch2, patch3, labels = inputs.to(device), patch1.to(device), patch2.to(device), \
                                                         patch3.to(device), labels.to(device)

                output, branch1_logps, branch2_logps, branch3_logps = model(inputs, patch1.float(), patch2.float(),
                                                                            patch3.float())
                # _, predict = torch.max(output, 1)
                predict = torch.softmax(output, dim=1)[:, 1]
                print("predict", predict)
                predict[predict >= 0.6] = 1
                predict[predict < 0.6] = 0
                loss = criterion(output, labels)
                running_loss += loss.item()
                total += labels.size(0)
                print("valid ground truth", labels)
                print("valid predict", predict)
                predict = predict.long()
                labels = labels.long()
                correct += (predict == labels).sum().item()
                '''calculate  Recall Precision F1'''
                pre_mask = torch.zeros(output.size()).scatter_(1, predict.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                tar_mask = torch.zeros(output.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)

                roc_label += labels.tolist()
                roc_output = torch.softmax(torch.transpose(output, 0, 1), dim=0)
                roc_output = roc_output.tolist()
                roc_predict += roc_output[1]

            val_loss = running_loss / len(val_loader)
            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            # 精度调整
            recall = (recall.numpy()[0] * 100).round(3)
            precision = (precision.numpy()[0] * 100).round(3)
            F1 = (F1.numpy()[0] * 100).round(3)

            print(roc_label)
            print(roc_predict)
            fpr, tpr, threshold = roc_curve(roc_label, roc_predict, pos_label=1)  ###计算真正率和假正率
            roc_auc = auc(fpr, tpr)  ###计算auc的值

            print("The accuracy of valid {} images: {}%".format(total, 100 * correct / total))
            print(
                "The accuracy of valid {} images: recall {}, precision {}, F1 {}".format(total, recall, precision, F1))
            result_list.append(
                [epoch, train_loss, val_loss, correct / total, recall[1], precision[1], F1[1], roc_auc])

            # 输入日志
            name = ['epoch', 'train_loss', 'val_loss', 'val_acc', 'recall', 'precision', 'F1', 'AUC']
            result = pd.DataFrame(columns=name, data=result_list)

            result.to_csv("ad_multi_view.csv", mode='w', index=False,
                          header=True)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
             
                print("Early stopping")
                break

    stop = datetime.now()
    print("Running time: ", stop - start)
