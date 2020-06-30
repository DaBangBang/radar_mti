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
warnings.filterwarnings("ignore")

folder_name = 'D:/data_signal_MTI/data_ball_move_39_real_imag_clean/p*'

epochs = 1500
batch_size = 2000
learning_rate = 0.001
mm = 1e-3
bin_resolution = 46.8410 ## millimeter = 4.6 cm
padding = 0
train_all = []
test_all = []
train_label_all = []
test_label_all = []
device = 'cuda' if cuda.is_available() else 'cpu'
wandb.init(project="model_r_abs_pad_")

def L2_loss(output, label):
    m_r = meshgrid()
    expect = torch.matmul(output, m_r)
    mse = mse_loss(expect, label)
    return mse, expect

def cartesian_to_spherical(label):
    y_offset = 110
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    # print(r)
    return r

def meshgrid():
    m_r = torch.arange(0, bin_resolution*25, bin_resolution).to(device)
    return m_r

def data_preparation(data_real, data_imag, label):
    
    data_complex = data_real + 1j*data_imag
    # print(data_complex.shape)
    n_pad = ((0,0),(0,0),(0,padding),(0,0))
    raw_iq = np.pad(data_complex, pad_width=n_pad, mode='constant', constant_values=0)
    raw_iq = np.mean(raw_iq, axis=1)
    data_fft = np.fft.fft(raw_iq, axis=1)
    data_fft_modulus = abs(data_fft / data_fft.shape[1])
    
    # plt.plot(data_fft_modulus[0,:,0])
    # plt.plot(raw_iq[0,:,0].imag)
    # plt.show()
    
    data_fft_modulus = np.swapaxes(data_fft_modulus, 1,2)
    data_fft_modulus = data_fft_modulus[:,:,:int(data_fft_modulus.shape[2]/8)]
    data_fft_modulus = np.float64(data_fft_modulus)

    label = cartesian_to_spherical(label)
    label = np.float64(label)

    return data_fft_modulus, label

# def plot_function():



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 2D-CNN Layer
        self.encode_conv1 = nn.Conv1d(in_channels=12, out_channels=4, kernel_size=3, stride = 1, padding=1)
        self.encode_conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride = 2, padding=1)
        self.encode_conv3 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride = 1, padding=1)
        self.encode_conv4 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride = 2, padding=1)
        self.encode_conv5 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride = 1, padding=1)
        self.encode_conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride = 2, padding=1)

        self.fc1 = nn.Linear(in_features=7*8, out_features=200)
        # self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.fc3 = nn.Linear(in_features=200, out_features=25)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.leaky_relu(self.encode_conv1(x))
        x = F.leaky_relu(self.encode_conv2(x))
        x = F.leaky_relu(self.encode_conv3(x))
        x = F.leaky_relu(self.encode_conv4(x))
        # x = F.leaky_relu(self.encode_conv5(x))
        # x = F.leaky_relu(self.encode_conv6(x))

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

class Radar_train_Dataset(Dataset):
    def __init__(self, real_part, imag_part, label_file):
 
        data_real = np.load(real_part[0])
        data_imag = np.load(imag_part[0])

        label = np.load(label_file[0])
        
        data_fft_modulus, label = data_preparation(data_real, data_imag, label)
        
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

    def __init__(self, real_part, imag_part, label_file):
        
        data_real = np.load(real_part[0])
        data_imag = np.load(imag_part[0])

        label = np.load(label_file[0])
        
        data_fft_modulus, label = data_preparation(data_real, data_imag, label)
        
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
wandb.watch(model)
model.to(device)

mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_function(train_loader):
    model.train()
    avg_mini_train_loss = []
    for i, (train_data, train_labels) in enumerate(train_loader, 0):
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        train_data = train_data.float()
        train_labels = train_labels.float()

        optimizer.zero_grad()
        output = model(train_data)
        loss, expect = L2_loss(output, train_labels)
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
        output = model(test_data)
        loss, expect = L2_loss(output, test_labels)
        # loss = loss*(1e-2)
        avg_mini_test_loss.append(loss.item())

    return np.mean(np.array(avg_mini_test_loss)), expect.cpu().detach().numpy(), test_labels.cpu().detach().numpy()
    
if __name__ == '__main__':
    
    
    folder_name = glob.glob(folder_name)
    folder_name = natsort.natsorted(folder_name)
    count = 0
    for f_name in folder_name:
        count += 1
        real_name = f_name + '/raw_signal_real_*'
        real_name = glob.glob(real_name)
        imag_name = f_name + '/raw_signal_imag_*'
        imag_name = glob.glob(imag_name)
        label_name = f_name +'/radar_pos_label_*'
        label_name = glob.glob(label_name)
        # print(real_name, imag_name)
        if count%4 == 0:
            test_data = Radar_test_Dataset(real_part= real_name, imag_part= imag_name, label_file=label_name)
        else:
            train_data = Radar_train_Dataset(real_part= real_name, imag_part= imag_name, label_file=label_name)
            
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=2032)

    for epoch in range(epochs):
        # print("======> epoch =", epoch)
        train_loss = train_function(train_loader)
        wandb.log({'Train_loss': train_loss})
        # print(">>>>>> train_loss <<<<<<", train_loss)
        if epoch%10 == 0:
            test_loss, expect, expect_label = test_function(test_loader)
            plt.plot(expect)
            plt.plot(expect_label)
            plt.ylabel('number of test point')
            print(">>>>>> test_loss <<<<<< epoch", epoch , test_loss)
            wandb.log({"distance": plt})
            wandb.log({'Test_loss': test_loss})
    

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'AE_1.pt'))


    
