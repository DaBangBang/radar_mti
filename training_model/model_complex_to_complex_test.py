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

folder_name = 'D:/data_signal_MTI/data_ball_move_39_real_imag_clean/p*'

epochs = 500
batch_size = 500
learning_rate = 0.001
train_all = []
test_all = []
train_label_all = []
test_label_all = []
device = 'cuda' if cuda.is_available() else 'cpu'

def L2_fft_loss(output, label):
        # print(x.size())
        L2_all_label = []
        output = output.permute(0,2,1)
        fft_out = torch.fft(output, 2)
        modulus_fft = torch.sqrt(fft_out[:,:,0]**2 + fft_out[:,:,1]**2)
        for i in range(label.size(1)):
            
            l2_loss = torch.sqrt(label[:,i,:]**2 + modulus_fft**2)
            # print(l2_loss)
            l2_loss = torch.mean(l2_loss)
            
            L2_all_label.append(l2_loss)
            
        L2_all_label = torch.tensor(L2_all_label, requires_grad=True)
        L2_all_label = torch.mean(l2_loss)
        # print(L2_all_label)
        # print('l2all',  L2_all_label.size())
        # print('modulus', modulus_fft.size())
        
        return L2_all_label

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 2D-CNN Layer
        self.encode_conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride = 1, padding=1)
        self.encode_conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride = 2, padding=1)
        self.encode_conv3 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride = 1, padding=1)
        self.encode_conv4 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride = 2, padding=1)
        # self.encode_conv5 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride = 1, padding=1)
        # self.encode_conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride = 2, padding=1)

        self.fc1 = nn.Linear(in_features=50*8, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=400)

        self.decode_conv1 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.decode_conv2 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.decode_conv3 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.decode_conv4 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.upsampling = nn.Upsample(scale_factor=2)

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
        x = F.leaky_relu(self.fc2(x))
        x = x.view(x.size(0), 16, 25) # 25*16 = 400

        x = F.leaky_relu(self.upsampling(self.decode_conv1(x)))
  
        x = F.leaky_relu(self.upsampling(self.decode_conv2(x)))
        x = F.leaky_relu(self.upsampling(self.decode_conv3(x)))
        x = self.decode_conv4(x)

        return x

class Radar_train_Dataset(Dataset):
    def __init__(self, real_part, imag_part, label_file):
 
        data_real = np.load(real_part[0])
        data_imag = np.load(imag_part[0])
        data_real = data_real[:,0,:,0]
        data_imag = data_imag[:,0,:,0]
        data_real_image = np.swapaxes(np.array([data_imag, data_real]), 0,1)
        data_real_image = np.float64(data_real_image)
        train_all.extend(data_real_image)
        # plt.plot(data_real_image[0,0,:])
        # plt.plot(data_real_image[0,1,:])
        # plt.show()
        label = np.load(label_file)
        label = np.float64(label)
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
        data_real = data_real[:,0,:,0]
        data_imag = data_imag[:,0,:,0]
        data_real_image = np.swapaxes(np.array([data_imag, data_real]), 0,1)
        data_real_image = np.float64(data_real_image)
        test_all.extend(data_real_image)

        label = np.load(label_file)
        label = np.float64(label)
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
model.load_state_dict(torch.load('D:/signal_MTI/training_model/wandb/run-20200619_005640-242r74ue/AE_1.pt'))
model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_function(train_loader):
    model.train()
    avg_mini_train_loss = []
    for i, (train_data, train_labels) in enumerate(train_loader, 0):
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        train_data = train_data.float()
        train_labels = train_labels.float()

        optimizer.zero_grad()
        output = model(train_data)
        loss = L2_fft_loss(output, train_labels)
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
        loss = L2_fft_loss(output, test_labels)
        # loss = loss*(1e-2)
        avg_mini_test_loss.append(loss.item())

    return np.mean(np.array(avg_mini_test_loss)), output.detach().cpu().numpy()
    
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
        label_name = f_name +'/simulate_radar_1chirp_0pad.npy'
        
        # print(real_name, imag_name)

        if count%4 == 0:
            test_data = Radar_test_Dataset(real_part= real_name, imag_part= imag_name, label_file=label_name)
        else:
            train_data = Radar_train_Dataset(real_part= real_name, imag_part= imag_name, label_file=label_name)
            
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=2032)
    test_loss, output = test_function(test_loader)
    np.save('D:/data_signal_MTI/data_ball_move_39_graph/result_pad_0', output)
    print(">>>>>> test_loss <<<<<<", test_loss)
    print("output_shape", output.shape)

    
