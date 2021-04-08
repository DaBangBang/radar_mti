import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn, optim, cuda
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import natsort
import wandb
import os
import glob
import re
import warnings
import argparse
from sklearn.model_selection import KFold
from torch.autograd import Variable

warnings.filterwarnings("ignore")

signal_dir = '/data/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
label_dir = '/data/data_signal_MTI/project_util_3/label_all/'
test_dir = '/data/data_signal_MTI/project_util_3/result_paper/'
expect_r_remove = '/data/data_signal_MTI/project_util_3/expect_r_remove_outlier.npy'

model_path = '/home/nakorn/weight_bias/wandb/run-20200930_201418-1dzier7q/files/fir_6cov_1.pt'
save_predict_path = '/data/data_signal_MTI/project_util_3/prediction_result/'
data_save_path = '/data/data_signal_MTI/project_util_3'
all_trajectory = 120

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=1501)
parser.add_argument('-batch_size', type=int, default=1000)
parser.add_argument('-learning_rate', type=float, default= 0.001)
parser.add_argument('-zero_padding', type=int, default=0)
parser.add_argument('-test_batch_size', type=int, default = 15000)
parser.add_argument('-loss_weight', type=int, default=3)
parser.add_argument('-save_to_wandb', type=bool, default=False)
parser.add_argument('-test_only', type=bool, default=False)
parser.add_argument('-mesh_w', type=float, default=0.1)
parser.add_argument('-use_mesh', type=bool, default=False)
parser.add_argument('-cuda', type=int, default=0)
args = parser.parse_args()
adc_range = 512
r_pad = ((0,0),(0,0),(0, args.zero_padding*adc_range),(0,0))
range_res = 45.74 / (1+args.zero_padding)

train_all = []
test_all = []
train_label_all = []
test_label_all = []
mae_each_fold = []
sd_each_fold = []
max_num = 8
min_num = 1

device = 'cuda:'+ str(args.cuda) if cuda.is_available() else 'cpu'

def L2_loss(output, label, op_w):
    
    m_r = meshgrid()
    
    if args.use_mesh :
        adj = 1 + args.mesh_w*torch.tanh(op_w)
        m_r = adj*m_r
    expect = torch.matmul(output, m_r)
    
    # mse = mse_loss(expect, label)
    mae = mae_loss(expect, label)
    return mae, expect

def cartesian_to_spherical(label):
    y_offset = 100
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    return r

def meshgrid():
    m_r = torch.arange(0, range_res*40, range_res).to(device)
    # print("m_r", m_r.shape)
    # print(m_r)
    return m_r

def augmented(data_fft_modulus, label):
    power_spectral = data_fft_modulus**2
    power_spectral_all = np.concatenate((power_spectral/10, power_spectral/100), axis=0)

    label_all = np.concatenate((label, label), axis=0)

    return power_spectral_all, label_all

def data_preparation(data_iq, label):
    print(data_iq.shape)
    data_iq_pad = np.pad(data_iq, pad_width= r_pad, mode='constant', constant_values=0)
    print(data_iq_pad.shape)
    data_fft = np.fft.fft(data_iq_pad, axis=2)

    #test_angle_fft to predict zeta
    # data_fft_v = np.fft.fftshift(np.fft.fft(data_fft, axis=1), axes=1)
    # data_fft_a = np.fft.fftshift(np.fft.fft(data_fft_v, axis=3), axes=3)

    data_fft_modulus = abs(data_fft)
    data_fft_modulus = np.mean(data_fft_modulus, axis=1)

    # data_fft_modulus = data_fft_modulus[:,:int(data_fft_modulus.shape[1]/8),:]    

    data_fft_modulus = np.swapaxes(data_fft_modulus, 1,2)
    data_fft_modulus = data_fft_modulus[:,:,:40]
    # data_fft_modulus = np.expand_dims(data_fft_modulus, axis=1)
    
    data_fft_modulus = np.float64(data_fft_modulus)
    
    # plt.plot(data_fft_modulus[0,0,:])
    # plt.show()
    
    label = cartesian_to_spherical(label)
    label = np.float64(label)

    # data_fft_modulus_aug, label_aug = augmented(data_fft_modulus, label)

    return data_fft_modulus, label
    # return 10*np.log10(data_fft_modulus), label

# def plot_function():



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 2D-CNN Layer
        self.encode_conv1 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride = 1, padding=1)
        self.encode_conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride = 1 , padding=1)
        self.encode_conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride = 1, padding=1)
        self.encode_conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride = 1, padding=1)
        self.encode_conv5 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride = 1, padding=1)

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.op_w = nn.Parameter(torch.randn(1), requires_grad=args.use_mesh)
     

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.leaky_relu(self.encode_conv1(x))
        x = F.leaky_relu(self.encode_conv2(x))
        x = F.leaky_relu(self.encode_conv3(x))
        x = F.leaky_relu(self.encode_conv4(x))


        x_1 = self.encode_conv5(x)
        x_1 = x_1.view(x_1.size(0), -1)
        x_1 = F.softmax(x_1, dim=1)

        
        return x_1, self.op_w

class Radar_train_Dataset(Dataset):
    def __init__(self, train_data, train_label):
               
        self.sig_train = torch.tensor(np.array(train_data))
        self.sig_label = torch.tensor(np.array(train_label))
        self.len = np.array(train_data).shape[0]

        print("train_function", self.sig_train.size(), self.sig_label.size())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.sig_train[idx], self.sig_label[idx]

class Radar_test_Dataset(Dataset):

    def __init__(self, test_data, test_label):
       
        self.sig_test = torch.tensor(np.array(test_data))
        self.sig_label = torch.tensor(np.array(test_label))
        self.len = np.array(test_data).shape[0]

        print("test_function", self.sig_test.size(), self.sig_label.size())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # print(idx)
        return self.sig_test[idx], self.sig_label[idx]


model = Model()

# wandb.watch(model)
if args.test_only:
    model.load_state_dict(torch.load(model_path))
model.to(device)
# mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train_function(train_loader):
    global model, optimizer
    model.train()
    avg_mini_train_loss = []

    for i, (train_data, train_labels) in enumerate(train_loader, 0):
        # print(i)
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        train_data = train_data.float()
    
        rand_num = ((max_num - min_num)*torch.rand(1) + min_num).to(device)
        
        if random.getrandbits(1):
            train_data = train_data * rand_num
        else:
            train_data = train_data / rand_num

        train_labels = train_labels.float()
        optimizer.zero_grad()
        output, op_w = model(train_data)
        loss, expect_r = L2_loss(output, train_labels, op_w)
        loss.backward()
        optimizer.step()
        avg_mini_train_loss.append(loss.item())

    return np.mean(np.array(avg_mini_train_loss)) 
        
def test_function(test_loader):
    global model, optimizer
    model.eval()
    avg_mini_test_loss = []
    for i, (test_data, test_labels) in enumerate(test_loader,0):
        test_data, test_labels = test_data.to(device), test_labels.to(device)
        test_data = test_data.float()
        test_labels = test_labels.float()
        output, op_w = model(test_data)
        loss, expect_r = L2_loss(output, test_labels, op_w)
        
        test_labels = test_labels.cpu().detach().numpy()
        op_w = op_w.cpu().detach().numpy()
        expect_r = expect_r.cpu().detach().numpy()
        avg_mini_test_loss.append(loss.item())

    return np.mean(np.array(avg_mini_test_loss)), test_labels, expect_r, op_w 

def evaluation(label, expect_r):
    mat_mae = np.mean(np.abs(label-expect_r))
    mat_rmse = np.sqrt(np.mean((label-expect_r)**2))
    mae_each_fold.append(mat_mae)
    sd_each_fold.append(mat_rmse)
    np.save(test_dir + 'mae_each_fold_zone_8', np.array(mae_each_fold))
    np.save(test_dir + 'rmse_each_fold_zone_8', np.array(sd_each_fold))
    print("all_mae = ", np.mean(mae_each_fold))
    print("all_rmse = ", np.mean(sd_each_fold))

if __name__ == '__main__':
    
    fold = 0
    count = 0
    data_iq = []
    label_all = []
    reject_list = []
    # reject_list = [np.arange(81,121)]
    # np.save(test_dir + 'reject_list', reject_list)

    # for f_name in range(all_trajectory):
    #     count += 1
    #     iq_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy' 
    #     label_name = label_dir + 'label_' + str(count) + '.npy'

    #     data_post, label_post = data_preparation(np.load(iq_name), np.load(label_name))
        
    #     # if count not in reject_list: 
    #     data_iq.append(data_post)
    #     label_all.append(label_post)

    #     print(np.array(data_iq).shape, np.array(label_all).shape)

    data_iq = np.load(data_save_path + '/signal_range_8c_3p.npy')
    label_all = np.load(data_save_path + '/label_range_8c_3p.npy')
    label_zone = np.load(data_save_path + '/label_aoa_16c_10p.npy')
    label_azimuth = label_zone[:,1]
    
    data_iq = np.array(data_iq)
    label_all = np.array(label_all)
    # print(data_iq[0,10,0,:], label_all[0,10])
    # print(np.min(label_all), np.max(label_all))
    # print(label_all.shape[0], label_all.shape)
    data_iq = np.reshape(data_iq, (-1, *data_iq.shape[-2:]))
    label_all = np.reshape(label_all, (-1))
    predict_r_remove = np.load(expect_r_remove)
    print(predict_r_remove.shape, predict_r_remove[10])
    print(max(label_azimuth), min(label_azimuth))
    # outlier_ = np.max(label_all)

    for i in range(predict_r_remove.shape[0]):
        actual_range = predict_r_remove[i]
        # print(actual_range)
        if actual_range < 92 or actual_range > 350:
            predict_r_remove[i] = np.nan
            
            
    k = ~np.isnan(predict_r_remove)
    data_iq = data_iq[k]
    label_all = label_all[k]
    label_azimuth = label_azimuth[k]

    print(data_iq.shape, label_all.shape)
    
    # print(data_iq[10,0,:], label_all[10])

    train_index = []
    test_index = []

    for ii in range(label_azimuth.shape[0]):
        if  -0.27706 <= label_azimuth[ii] < -0.07034:
            test_index.append(ii)
        else:
            train_index.append(ii)
    test_index = np.array(test_index)
    train_index = np.array(train_index)
    print(np.array(test_index).shape, np.array(train_index).shape)

    fold += 1
    model = Model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.save_to_wandb:
        run = wandb.init(project="model_zone_generalization", name="test_range_zone_8_"+str(fold), dir='/home/nakorn/weight_bias', reinit=True)
    
    train_data, train_label, test_data, test_label = data_iq[train_index], label_all[train_index] \
                                                        ,data_iq[test_index], label_all[test_index]
    
    # Select with norm dis
    select_train = np.random.choice(train_data.shape[0], 8500, replace=False)
    select_test = np.random.choice(test_data.shape[0], 1700, replace=False)
    
    train_data = train_data[select_train]
    train_label = train_label[select_train]
    test_data = test_data[select_test]
    test_label = test_label[select_test]
    np.save(data_save_path+ '/select_train_zone_8', select_train)
    np.save(data_save_path+ '/select_test_zone_8', select_test)
    # ============================================ 

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    np.save(test_dir + 'train_index_fold_zone_8' + str(fold), np.array(train_index))
    np.save(test_dir + 'test_index_fold_zone_8' + str(fold), np.array(test_index))

    train_set = Radar_train_Dataset(train_data=train_data, train_label=train_label)
    test_set = Radar_test_Dataset(test_data=test_data, test_label=test_label)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size)

    print(train_loader)
    if args.test_only:
        test_loss, label, expect_r, op_w = test_function(test_loader)
        print(expect_r.shape)
        np.save(save_predict_path + 'expect_r_zone_8', expect_r)

    else:
        for epoch in range(args.epochs):
            # print("======> epoch =", epoch)
            train_loss = train_function(train_loader)
            
            if args.save_to_wandb:
                wandb.log({'Train_loss': train_loss}, step=epoch)
            
            # print(">>>>>> train_loss <<<<<<", train_loss)
            if epoch%10 == 0:
                test_loss, label, expect_r, op_w = test_function(test_loader)
                rmse = np.sqrt(np.mean((label-expect_r)**2))
                print(">>>>>> test_loss <<<<<< epoch", epoch , test_loss, rmse)
                np.save(test_dir + 'expec_r_fold_zone_8' + str(fold), np.array(expect_r))
                
                if args.save_to_wandb:
                    plt.plot(label[:])
                    plt.plot(expect_r[:])
                    plt.ylabel('r distance')
                    plt.xlabel('number of test point')
                    wandb.log({'distance': plt}, step=epoch)
                    wandb.log({'Test_loss': test_loss}, step=epoch)
                    wandb.log({'Meshgrid_weight' : op_w}, step=epoch)
    
            if args.save_to_wandb and (epoch%500 == 0):
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_test_zone_8'+str(fold)+'_ep_'+str(epoch)+'.pt'))
        run.finish()
        evaluation(label, expect_r)

    
