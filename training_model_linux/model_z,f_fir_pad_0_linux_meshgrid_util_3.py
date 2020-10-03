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

signal_dir = '/data/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
label_dir = '/data/data_signal_MTI/project_util_3/label_all/'

model_path = '/home/nakorn/weight_bias/wandb/run-20200930_200650-c0cxja7k/files/aoa_fir_6cov_1.pt'
save_predict_path = '/data/data_signal_MTI/project_util_3/prediction_result/'
all_trajectory = 120

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=1001)
parser.add_argument('-batch_size', type=int, default=2500)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-zero_padding', type=int, default=0)
parser.add_argument('-test_batch_size', type=int, default= 10200)
parser.add_argument('-loss_weight', type=int, default=100)
parser.add_argument('-save_to_wandb', type=bool, default=False)
parser.add_argument('-test_only', type=bool, default=False)
parser.add_argument('-wmodel', default='cnn+fc+reg')
parser.add_argument('-cuda', type=int, default=0)
args = parser.parse_args()

# epochs = 3000
# batch_size = 2000
# learning_rate = 0.001
bin_resolution = 0.44879 ## millimeter = 4.6 cm
 # test_range_fft to predict zeta
# bin_resolution = 0.049866
# padding = 0

train_all = []
test_all = []
train_label_all = []
test_label_all = []
device = 'cuda:'+ str(args.cuda) if cuda.is_available() else 'cpu'

if args.save_to_wandb:
    wandb.init(project="training-120-trajactory-range", dir='/home/nakorn/weight_bias')

def L2_loss(out_z, label):

    m_r = meshgrid()
    expect_z = torch.matmul(out_z, m_r)
    mse = mse_loss(expect_z, label[:,1])
    mse = mse*args.loss_weight
    
    return mse, expect_z

def cartesian_to_spherical(label):
    
    y_offset = 100
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,2])
    phi = np.arctan2(label[:,0,1] - y_offset , np.sqrt(label[:,0,0]**2 + label[:,0,2]**2))
    # print(zeta)
    # print(r)
    return r, zeta, phi

def meshgrid():
    m_r = torch.arange(-1.5708, 1.5708, bin_resolution).to(device)
    # print(m_r, m_r.shape)
    return m_r

def data_preparation(data_iq, label):

    data_fft = np.fft.fft(data_iq, axis=2)
    data_fft_v = np.fft.fftshift(np.fft.fft(data_fft, axis=1), axes=1)
    data_fft_a = np.fft.fftshift(np.fft.fft(data_fft_v, axis=3), axes=3)
    data_fft_modulus = abs(data_fft_a)
    data_fft_modulus = np.mean(data_fft_modulus, axis=1)

    # test_range_fft to predict zeta
    # data_fft_modulus = np.swapaxes(data_fft_modulus, 1,2)
    # data_fft_modulus = data_fft_modulus[:,:,:int(data_fft_modulus.shape[2]/8)]
    
    data_fft_modulus = data_fft_modulus[:,:int(data_fft_modulus.shape[1]/8),:]
    data_fft_modulus = np.float64(data_fft_modulus)

    r, zeta, phi = cartesian_to_spherical(label)
    # print(r[0], zeta[0], phi[0])
    label = np.array([r, zeta, phi])
    label = np.float64(label.T)
    # print(label[0, :], label.shape)
    return data_fft_modulus, label

# def plot_function():



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 2D-CNN Layer
        self.encode_conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride = 1, padding=1)
        self.encode_conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride = 1 , padding=1)
        self.encode_conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride = 1, padding=1)
        self.encode_conv4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride = 1, padding=1)
        self.encode_conv5 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride = 1, padding=1)


    def forward(self, x):

        x = F.relu(self.encode_conv1(x))
        x = F.relu(self.encode_conv2(x))
        x = F.relu(self.encode_conv3(x))
        x = F.relu(self.encode_conv4(x))
        # x = F.leaky_relu(self.encode_conv5(x))
        # x = F.leaky_relu(self.encode_conv6(x))

        x_1 = self.encode_conv5(x)
        x_1 = x_1.view(x_1.size(0), -1)
        x_1 = F.softmax(x_1, dim=1)
        
        return x_1

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
        loss, expect_z = L2_loss(out_z, train_labels)
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
        loss, expect_z = L2_loss(out_z, test_labels)
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


    
