import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn, optim, cuda
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import natsort
import wandb
import os
import glob
import re
import warnings
import argparse

warnings.filterwarnings("ignore")

signal_dir = '/data/data_signal_MTI/project_util/signal_all_w_mti_cutoff_12/'
label_dir = '/data/data_signal_MTI/project_util/label_all/'

model_path = '/home/nakorn/weight_bias/wandb/run-20200818_162113-2k83zzsu/aoa_fir_6cov_1.pt'
save_predict_path = '/home/nakorn/weight_bias/test_data/'
all_trajectory = 117

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=500)
parser.add_argument('-batch_size', type=int, default=4000)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-zero_padding', type=int, default=0)
parser.add_argument('-test_batch_size', type=int, default= 9860)
parser.add_argument('-loss_weight', type=int, default=1)
parser.add_argument('-save_to_wandb', type=bool, default=False)
parser.add_argument('-test_only', type=bool, default=False)
parser.add_argument('-wmodel', default='cnn+fc+reg')
args = parser.parse_args()

# epochs = 3000
# batch_size = 2000
# learning_rate = 0.001
bin_resolution = 0.17 ## millimeter = 4.6 cm
# padding = 0

train_all = []
test_all = []
train_label_all = []
test_label_all = []
device = 'cuda:1' if cuda.is_available() else 'cpu'

if args.save_to_wandb:
    wandb.init(project="cnn-fc-117", dir='/home/nakorn/weight_bias')

def RMSE_loss(out_z, label, wmodel):

    if 'cnn+fc' == wmodel:
        m_r = meshgrid()
        expect_z = torch.matmul(out_z, m_r)
        mse_z = mse_loss(expect_z, label[:,1])
        rmse_z = torch.sqrt(mse_z)
        loss = rmse_z
    elif 'cnn+fc+reg' == args.wmodel:
        expect_z = out_z.view(-1)
        mse_z = mse_loss(expect_z, label[:,1])
        rmse_z = torch.sqrt(mse_z)
        loss = rmse_z

    return loss, expect_z

def cartesian_to_spherical(label):
    y_offset = 105
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,2])
    phi = np.arctan2(label[:,0,1] - y_offset , np.sqrt(label[:,0,0]**2 + label[:,0,2]**2))

    # print(r)
    return r, zeta, phi

def meshgrid():
    m_r = torch.arange(-4.1667, 4.1667, bin_resolution).to(device)
    # print(m_r, m_r.shape)
    return m_r

def data_preparation(data_iq, label):
    
    # n_pad = ((0,0),(0,0),(0,0),(0,args.zero_padding))
    # data_real = np.pad(data_real, pad_width=n_pad, mode='constant', constant_values=0)

    data_fft_range = np.fft.fft(data_iq, axis=2) / data_iq.shape[2]
    data_fft_velocity = np.fft.fftshift(np.fft.fft(data_fft_range, axis=1) / data_iq.shape[1], axes=1)
    half_velocity_bin = data_iq.shape[1] / 2
    data_fft_angle = np.fft.fftshift(np.fft.fft(data_fft_velocity, axis=3) / data_iq.shape[3], axes=3) # angle fft
    data_fft_angle = abs(data_fft_angle)
    data_fft_angle = data_fft_angle[:,int(half_velocity_bin-10):int(half_velocity_bin+10),:int(data_iq.shape[2]/4),:]

    data_fft_angle = np.swapaxes(data_fft_angle, 1,2)
    data_fft_angle = np.float64(data_fft_angle)

    print(data_fft_angle.shape)

    r, zeta, phi = cartesian_to_spherical(label)
    # print(r[0], zeta[0], phi[0])
    label = np.array([r, zeta, phi])
    label = np.float64(label.T)
    # print(label[0, :], label.shape)
    return data_fft_angle, label

# def plot_function():



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 2D-CNN Layer
        self.encode_conv1 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=(3,3), stride = 1, padding=(1,1))
        self.encode_conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3,3), stride = 2, padding=(1,1))
        self.encode_conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3), stride = 1, padding=(1,1))
        self.encode_conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride = 2, padding=(1,1))
        # self.encode_conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride = 1, padding=(1,1))
        # self.encode_conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride = 2, padding=(1,1))

        self.fc1 = nn.Linear(in_features=5*8*1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)        


    def forward(self, x):

        x = F.relu(self.encode_conv1(x))
        x = F.relu(self.encode_conv2(x))
        x = F.relu(self.encode_conv3(x))
        x = F.relu(self.encode_conv4(x))
        # x = F.leaky_relu(self.encode_conv5(x))
        # x = F.leaky_relu(self.encode_conv6(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Radar_train_Dataset(Dataset):
    def __init__(self, real_part, label_file):
 
        data_real = np.load(real_part)
        label = np.load(label_file)
        
        data_fft_modulus, label = data_preparation(data_real, label)
        
        train_all.extend(data_fft_modulus)
        train_label_all.extend(label)
                
        self.train_data = torch.tensor(np.array(train_all))
        self.train_label = torch.tensor(np.array(train_label_all))
        self.len = np.array(train_all).shape[0]

        print("train_function", self.train_data.size(), self.train_label.size())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.train_data[idx], self.train_label[idx]

class Radar_test_Dataset(Dataset):

    def __init__(self, real_part, label_file):
        
        data_real = np.load(real_part)
        label = np.load(label_file)

        data_fft_modulus, label = data_preparation(data_real, label)
        
        test_all.extend(data_fft_modulus)
        test_label_all.extend(label)
                
        self.test_data = torch.tensor(np.array(test_all))
        self.test_label = torch.tensor(np.array(test_label_all))
        self.len = np.array(test_all).shape[0]

        print("test_function", self.test_data.size(), self.test_label.size())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.test_data[idx], self.test_label[idx]


model = Model()
# wandb.watch(model)
if args.test_only:
    model.load_state_dict(torch.load(model_path))
model.to(device)

mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train_function(train_loader):
    model.train()
    avg_mini_train_loss = []
    for i, (train_data, train_labels) in enumerate(train_loader, 0):
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        train_data = train_data.float()
        train_labels = train_labels.float()

        optimizer.zero_grad()
        out_z = model(train_data)
        loss, expect_z = RMSE_loss(out_z, train_labels, args.wmodel)
        loss.backward()
        optimizer.step()
        avg_mini_train_loss.append(loss.item())

    return np.mean(np.array(avg_mini_train_loss)) 
        
def test_function(test_loader):
    model.eval()
    avg_mini_test_loss = []
    for i, (test_data, test_labels) in enumerate(test_loader,0):
        test_data, test_labels = test_data.to(device), test_labels.to(device)
        test_data = test_data.float()
        test_labels = test_labels.float()
        out_z = model(test_data)
        loss, expect_z= RMSE_loss(out_z, test_labels, args.wmodel)
        # loss = loss*(1e-2)
        avg_mini_test_loss.append(loss.item())
        test_labels = test_labels.cpu().detach().numpy()
        expect_z = expect_z.cpu().detach().numpy()

    return np.mean(np.array(avg_mini_test_loss)), test_labels, expect_z 
    
if __name__ == '__main__':
    
    
    count = 0
    for f_name in range(all_trajectory):
        count += 1
        real_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy'
        label_name = label_dir + 'label_' + str(count) + '.npy'
      
        if count%4 == 0:
            test_data = Radar_test_Dataset(real_part= real_name,  label_file=label_name)
        else:
            train_data = Radar_train_Dataset(real_part= real_name, label_file=label_name)
            
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size)


    if args.test_only:
        test_loss, label, expect_z = test_function(test_loader)
        np.save(save_predict_path + 'label_z_%4', label)
        np.save(save_predict_path + 'expect_z_%4', expect_z)
        print(test_loss, expect_z.shape)

    else :
        for epoch in range(args.epochs):
            # print("======> epoch =", epoch)
            train_loss = train_function(train_loader)
            
            if args.save_to_wandb:
                wandb.log({'Train_loss': train_loss}, step=epoch)
            
            # print(">>>>>> train_loss <<<<<<", train_loss)
            if epoch%10 == 0:
                test_loss, label, expect_z = test_function(test_loader)
                
                print(">>> test_loss, epoch   <<<<<", epoch , test_loss)
                
                if args.save_to_wandb:
                    plt.figure(1)
                    plt.plot(label[:, 1])
                    plt.plot(expect_z[:])
                    plt.ylabel('rmse zeta')
                    plt.xlabel('number of test point')
                    wandb.log({'distance_z': plt}, step=epoch)
                    wandb.log({'Test_loss': test_loss}, step=epoch)
    
    if args.save_to_wandb:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'aoa_fir_6cov_1.pt'))


    
