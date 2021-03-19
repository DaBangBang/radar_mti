import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn, optim, cuda, cos, sin
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import natsort
import wandb
import os
import glob
import re
import warnings
import argparse

predict_dir = 'D:/data_signal_MTI/project_util_3/result_for_paper/'
save_dir =  'D:/data_signal_MTI/project_util_3/result_for_paper/'
epochs = 100001
learning_rate = 0.001

device = 'cuda:0' if cuda.is_available() else 'cpu'
# wandb.init(project="optimize_affine")

def l2_loss(r_out, train_label):
    # w = 10
    loss = torch.mean(torch.abs(train_label[:,0] - r_out))
    # z_loss = torch.mean(torch.abs(train_label[:,1] - z_out))
    # print(r_loss, z_loss)
    # loss = r_loss + w*z_loss
    return loss

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()
        # self.w1 = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.w2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.t1 = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.t2 = nn.Parameter(torch.randn(1), requires_grad=True)
        
    def forward(self, t_r):
        r_out = t_r + self.t1 
        # z_out = t_z + self.t2
        return r_out, self.t1

class Op_Data(Dataset):
    def __init__(self, train_range, label_set):
                
        self.train_range = torch.tensor(np.array(train_range))
        # self.train_zeta = torch.tensor(np.array(train_zeta))
        self.train_label = torch.tensor(np.array(label_set))
        self.len = np.array(train_range).shape[0]

        print("train_function", self.train_range.size(), self.train_label.size())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.train_range[idx], self.train_label[idx]

model = Affine()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

def train_function(train_loader):
    model.train()
    loss_all = []
    for i, (train_range, train_label) in enumerate(train_loader, 0):
        train_range = train_range.float().to(device)
        # train_zeta = train_zeta.float().to(device)
        train_label = train_label.float().to(device)
        optimizer.zero_grad()
        r_out, t1 = model(train_range)
        loss = l2_loss(r_out, train_label)
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        r_out = r_out.cpu().detach().numpy()
        # z_out = z_out.cpu().detach().numpy()
        # w1 = w1.cpu().detach().numpy()
        # w2 = w2.cpu().detach().numpy()
        t1 = t1.cpu().detach().numpy()
        # t2 = t2.cpu().detach().numpy()
        # var = np.array([w1, w2])
        var = np.array([t1])

    return np.mean(np.array(loss_all)), r_out, var

if __name__ == '__main__':
    
    predict = np.load(predict_dir + 'expect_r_music_pad_100%_fold.npy')
    predict_zeta = np.load(predict_dir + 'expect_z_music_pad_100%_fold.npy')
    g_t = np.load(predict_dir + 'label_z_2dfft_fold.npy')
    print(g_t.shape)
    for i in range(predict.shape[0]):
        actual_range = predict[i]
        if actual_range > 2380 or actual_range <= 0:
            predict[i] = np.nan
            predict_zeta[i] = np.nan
           
    k = ~np.isnan(predict)
    predict = predict[k]
    predict_zeta = predict_zeta[k]

    g_t = g_t[k]

    print(predict.shape, g_t.shape)
    train_data = Op_Data(train_range= predict, label_set= g_t)
    train_loader = DataLoader(dataset=train_data, batch_size=10000, shuffle=False)

    for epoch in range(epochs):
        training_loss, r_out, var = train_function(train_loader)
        if epoch%100 == 0:
            print('loss', training_loss)
            # wandb.log({'Loss': training_loss}, step=epoch)
        if epoch%1000 == 0:
            print(np.array(r_out).shape, np.array(var).shape)
            # torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'scaling.pt'))
            np.save(save_dir + 'expect_r_music_pad_100%_fold_optimize', np.array(r_out))
            # np.save(save_dir + 'expect_z_%4_esprit_pad_op_trans', np.array(z_out))
            np.save(save_dir + 'optimize_r_music', np.array(var))