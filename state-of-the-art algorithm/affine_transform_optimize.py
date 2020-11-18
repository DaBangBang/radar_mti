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

predict_dir = 'D:/data_signal_MTI/project_util_3/prediction_result/'
save_dir =  'D:/data_signal_MTI/project_util_3/prediction_result/RT_2dfft/'
epochs = 50001
learning_rate = 0.001

def l2_loss(output, train_label):
    loss = torch.mean((train_label - output)**2)
    return loss

def spherical_to_cartesian(expect_r, expect_z, label_z):
    xp = expect_r * np.cos(label_z[:,2]) * np.sin(expect_z)
    yp = expect_r * np.sin(label_z[:,2]) + 100
    zp = expect_r * np.cos(label_z[:,2]) * np.cos(expect_z)
    expect_p = np.array([[xp, yp, zp]])
    expect_p = np.swapaxes(np.swapaxes(expect_p,0,2),1,2).reshape((-1,3))
    
    x = label_z[:,0] * np.cos(label_z[:,2]) * np.sin(label_z[:,1])
    y = label_z[:,0] * np.sin(label_z[:,2]) + 100
    z = label_z[:,0] * np.cos(label_z[:,2]) * np.cos(label_z[:,1])

    expect_gt = np.array([[x, y, z]])
    expect_gt = np.swapaxes(np.swapaxes(expect_gt,0,2),1,2).reshape((-1,3))
    
    return expect_p, expect_gt


class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()
        self.roll = nn.Parameter(torch.randn(1), requires_grad=True)
        self.pitch = nn.Parameter(torch.randn(1), requires_grad=True)
        self.yaw = nn.Parameter(torch.randn(1), requires_grad=True)
        self.tx = nn.Parameter(torch.randn(1), requires_grad=True)
        self.ty = nn.Parameter(torch.randn(1), requires_grad=True)
        self.tz = nn.Parameter(torch.randn(1), requires_grad=True)


    def construct_transform(self):

        tensor_0 = torch.zeros(1)
        tensor_1 = torch.ones(1)
        
        RX = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, cos(self.roll), -sin(self.roll)]),
                torch.stack([tensor_0, sin(self.roll), cos(self.roll)])]).reshape(3,3)

        RY = torch.stack([
                torch.stack([cos(self.pitch), tensor_0, sin(self.pitch)]),
                torch.stack([tensor_0, tensor_1, tensor_0]),
                torch.stack([-sin(self.pitch), tensor_0, cos(self.pitch)])]).reshape(3,3)

        RZ = torch.stack([
                torch.stack([cos(self.yaw), -sin(self.yaw), tensor_0]),
                torch.stack([sin(self.yaw), cos(self.yaw), tensor_0]),
                torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)
        
        f_r = RX@RY@RZ
        f_t = torch.stack([self.tx, self.ty, self.tz]).reshape(-1)
        return f_r, f_t

    def forward(self, x):
        f_r, f_t = self.construct_transform()
        output = (x@f_r.T) + f_t
        return output, f_r, f_t

class Op_Data(Dataset):
    def __init__(self, train_set, label_set):
                
        self.train_data = torch.tensor(train_set)
        self.train_label = torch.tensor(np.array(label_set))
        self.len = np.array(train_set).shape[0]

        print("train_function", self.train_data.size(), self.train_label.size())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.train_data[idx], self.train_label[idx]

model = Affine()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_function(train_loader):
    model.train()
    loss_all = []
    for i, (train_data, train_label) in enumerate(train_loader, 0):
        train_data = train_data.float()
        train_label = train_label.float()
        optimizer.zero_grad()
        output, f_r, f_t = model(train_data)
        loss = l2_loss(output, train_label)
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        rotation = f_r.cpu().detach().numpy()
        translation = f_t.cpu().detach().numpy()

    return np.mean(np.array(loss_all)), rotation, translation

if __name__ == '__main__':
    
    predict_r = np.load(predict_dir + 'expect_r_%4_2dfft_pad.npy')
    predict_zeta = np.load(predict_dir + 'expect_z_%4_2dfft_pad.npy')
    g_t = np.load(predict_dir + 'label_z_%4.npy')
    expect_p, expect_gt = spherical_to_cartesian(predict_r, predict_zeta, g_t)
    train_data = Op_Data(train_set= expect_p, label_set= expect_gt)
    train_loader = DataLoader(dataset=train_data, batch_size=10200, shuffle=False)

    for epoch in range(epochs):
        training_loss, f_r, f_t = train_function(train_loader)
        if epoch%100 == 0:
            print('loss', training_loss)
        if epoch%1000 == 0:
            print(np.array(f_r).shape, np.array(f_t).shape)
            np.save(save_dir + 'rotation_2dfft', np.array(f_r))
            np.save(save_dir + 'translation_2dfft', np.array(f_t))