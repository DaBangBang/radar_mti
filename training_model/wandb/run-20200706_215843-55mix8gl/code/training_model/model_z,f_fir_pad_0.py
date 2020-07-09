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
folder_name = 'D:/data_signal_MTI/data_ball_move_39_real_imag_clean/p*'

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=400)
parser.add_argument('-batch_size', type=int, default=2000)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-zero_padding', type=int, default=0)
parser.add_argument('-test_batch_size', type=int, default= 2032)
args = parser.parse_args()

# epochs = 3000
# batch_size = 2000
# learning_rate = 0.001
mm = 1e-3
bin_resolution = 0.17 ## millimeter = 4.6 cm
# padding = 0

train_all = []
test_all = []
train_label_all = []
test_label_all = []
device = 'cuda' if cuda.is_available() else 'cpu'
wandb.init(project="model_zeta-phi_fir_angle")

def RMSE_loss(out_z, out_phi, label):
    loss_weight = 4
    m_r = meshgrid()
    expect_z = torch.matmul(out_z, m_r)
    expect_phi = torch.matmul(out_phi, m_r)
    mse_z = mse_loss(expect_z, label[:,1])
    mse_phi = mse_loss(expect_phi, label[:,2])
    rmse_z = torch.sqrt(mse_z)
    rmse_phi = torch.sqrt(mse_phi)
    loss = rmse_z + loss_weight*rmse_phi
    return loss, expect_z, expect_phi, rmse_z, rmse_phi

def cartesian_to_spherical(label):
    y_offset = 110
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,1])
    phi = np.arctan2((np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2)) , label[:,0,2])
    # print(r)
    return r, zeta, phi

def meshgrid():
    m_r = torch.arange(-4.1667, 4.1667, bin_resolution).to(device)
    # print(m_r, m_r.shape)
    return m_r

def data_preparation(data_real, label):
    
    # data_fft_modulus = abs(data_real)
    # data_fft_modulus = data_fft_modulus[:,:,:50,:]

    n = data_real.shape[3]
    data_fft_modulus = np.fft.fftshift(np.fft.fft(data_real, axis=3) / n, axes=3) # angle fft
    data_fft_modulus = abs(data_fft_modulus)
    data_fft_modulus = data_fft_modulus[:,:,:50,:]

    # plt.imshow(data_fft_modulus[100,:,10,:])
    # plt.show() 

    data_fft_modulus = np.swapaxes(data_fft_modulus, 1,2)
    # print(data_fft_modulus.shape)
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
        self.encode_conv1 = nn.Conv2d(in_channels=50, out_channels=4, kernel_size=(3,3), stride = 1, padding=(1,1))
        self.encode_conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3,3), stride = 2, padding=(1,1))
        self.encode_conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3), stride = 1, padding=(1,1))
        self.encode_conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride = 2, padding=(1,1))
        # self.encode_conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride = 1, padding=(1,1))
        # self.encode_conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride = 2, padding=(1,1))

        self.fc1_1 = nn.Linear(in_features=8*3*8, out_features=200)
        self.fc1_2 = nn.Linear(in_features=8*3*8, out_features=200)
        # self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.fc2_1 = nn.Linear(in_features=200, out_features=50)
        self.fc2_2 = nn.Linear(in_features=200, out_features=50)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.leaky_relu(self.encode_conv1(x))
        x = F.leaky_relu(self.encode_conv2(x))
        x = F.leaky_relu(self.encode_conv3(x))
        x = F.leaky_relu(self.encode_conv4(x))
        # x = F.leaky_relu(self.encode_conv5(x))
        # x = F.leaky_relu(self.encode_conv6(x))

        x = x.view(x.size(0), -1)
        x_1 = F.leaky_relu(self.fc1_1(x))
        x_2 = F.leaky_relu(self.fc1_2(x))
        
        x_1 = F.softmax(self.fc2_1(x_1), dim=1)
        x_2 = F.softmax(self.fc2_2(x_2), dim=1)

        return x_1, x_2

class Radar_train_Dataset(Dataset):
    def __init__(self, real_part, label_file):
 
        data_real = np.load(real_part[0])
        label = np.load(label_file[0])
        
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
        
        data_real = np.load(real_part[0])
        label = np.load(label_file[0])
        
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
        out_z, out_phi = model(train_data)
        loss, expect_z, expect_phi, rmse_z, rmse_phi = RMSE_loss(out_z, out_phi, train_labels)
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
        out_z, out_phi = model(test_data)
        loss, expect_z, expect_phi, rmse_z, rmse_phi = RMSE_loss(out_z, out_phi, test_labels)
        # loss = loss*(1e-2)
        avg_mini_test_loss.append(loss.item())
        test_labels = test_labels.cpu().detach().numpy()
        expect_z = expect_z.cpu().detach().numpy()
        expect_phi = expect_phi.cpu().detach().numpy()
        rmse_z = rmse_z.cpu().detach().numpy()
        rmse_phi = rmse_phi.cpu().detach().numpy()

    return np.mean(np.array(avg_mini_test_loss)), test_labels, expect_z, expect_phi, rmse_z, rmse_phi 
    
if __name__ == '__main__':
    
    
    folder_name = glob.glob(folder_name)
    folder_name = natsort.natsorted(folder_name)
    count = 0
    for f_name in folder_name:
        count += 1
        real_name = f_name + '/doppler_fft_zero_pad_0_fir*'
        real_name = glob.glob(real_name)
  
        label_name = f_name +'/radar_pos_label_*'
        label_name = glob.glob(label_name)
      
        if count%4 == 0:
            test_data = Radar_test_Dataset(real_part= real_name,  label_file=label_name)
        else:
            train_data = Radar_train_Dataset(real_part= real_name, label_file=label_name)
            
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size)

    for epoch in range(args.epochs):
        # print("======> epoch =", epoch)
        train_loss = train_function(train_loader)
        wandb.log({'Train_loss': train_loss}, step=epoch)
        
        # print(">>>>>> train_loss <<<<<<", train_loss)
        if epoch%10 == 0:
            test_loss, expect, expect_z, expect_phi, rmse_z, rmse_phi = test_function(test_loader)
            plt.figure(1)
            plt.plot(expect[:500, 1])
            plt.plot(expect_z[:500])
            plt.ylabel('rmse zeta')
            plt.xlabel('number of test point')
            plt.figure(2)
            plt.plot(expect[:500, 2])
            plt.plot(expect_phi[:500])
            plt.ylabel('rmse phi')
            plt.xlabel('number of test point')
            print(">>> test_loss, epoch   <<<<<", epoch , test_loss)
            print(">>> label_z, label_phi <<<<<", rmse_z, rmse_phi)
            print(">>> [label_z, test_z]  <<<<<", expect[0,1], expect_z[0])
            print(">>> [label_phi, test_phi] <<", expect[0,2], expect_phi[0])
            wandb.log({'distance': plt}, step=epoch)
            wandb.log({'Test_loss': test_loss}, step=epoch)
    

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'aoa_fir_6cov_1.pt'))


    
