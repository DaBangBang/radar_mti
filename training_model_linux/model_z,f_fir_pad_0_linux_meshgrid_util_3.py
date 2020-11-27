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
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

signal_dir = '/data/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
label_dir = '/data/data_signal_MTI/project_util_3/label_all/'
test_dir = '/data/data_signal_MTI/project_util_3/10_Fold/30%_data/test_data_2/'

model_path = '/home/nakorn/weight_bias/wandb/run-20200930_200650-c0cxja7k/files/aoa_fir_6cov_1.pt'
save_predict_path = '/data/data_signal_MTI/project_util_3/prediction_result/'
all_trajectory = 120

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=1001)
parser.add_argument('-batch_size', type=int, default=1000)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-zero_padding', type=int, default=0)
parser.add_argument('-test_batch_size', type=int, default= 4080)
parser.add_argument('-loss_weight', type=int, default=10)
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
mae_each_fold = []
sd_each_fold = []

device = 'cuda:'+ str(args.cuda) if cuda.is_available() else 'cpu'


def L2_loss(out_z, label):

    m_r = meshgrid()
    m_r = -1*m_r
    expect_z = torch.matmul(out_z, m_r)
    # mse = mse_loss(expect_z, label[:,1])
    # print(expect_z.size(), label.size())
    mae = mae_loss(expect_z, label[:,1])
    # mse = mse*args.loss_weight
    mae = mae*args.loss_weight

    return mae, expect_z
     
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
        self.encode_conv1 = nn.Conv1d(in_channels=64, out_channels=8, kernel_size=3, stride = 1, padding=1)
        self.encode_conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride = 1 , padding=1)
        self.encode_conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride = 1, padding=1)
        self.encode_conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride = 1, padding=1)
        self.encode_conv5 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride = 1, padding=1)


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
    model.train()
    avg_mini_train_loss = []
    rand_i = torch.tensor([1, 4, 8, 16, 32, 64]).to(device)
    for i, (train_data, train_labels) in enumerate(train_loader, 0):
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        train_data = train_data.float()

        rand_div = torch.randint(low=0, high=5, size=(1,)).to(device)
        train_data = train_data / rand_i[rand_div[0]]

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

def evaluation(label, expect_r):
    mat_mae = np.mean(np.abs(label[:,1]-expect_r))
    mat_sd = np.sqrt(np.mean((label[:,1]-expect_r)**2))
    mae_each_fold.append(mat_mae)
    sd_each_fold.append(mat_sd)
    np.save(test_dir + 'mae_aoa_each_fold', np.array(mae_each_fold))
    np.save(test_dir + 'sd_aoa_each_fold', np.array(sd_each_fold))
    print("all_mae = ", np.mean(mae_each_fold))
    print("all_sd = ", np.mean(sd_each_fold))

if __name__ == '__main__':
    
    fold = 0
    count = 0
    data_iq = []
    label_all = []
    reject_list = []
    # reject_list = np.load(test_dir + 'reject_list.npy')
    
    # print(ff, yy)
    for f_name in range(all_trajectory):
        count += 1
        iq_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy'
        label_name = label_dir + 'label_' + str(count) + '.npy'
      
        data_post, label_post = data_preparation(np.load(iq_name), np.load(label_name))
        
        # if count not in reject_list: 
        data_iq.append(data_post)
        label_all.append(label_post)
        print(np.array(data_iq).shape, np.array(label_all).shape)



    data_iq = np.array(data_iq)
    label_all = np.array(label_all)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(data_iq):
    # train_index1 = np.arange(0,30)
    # # train_index11 = np.arange(30,40)
    # train_index2 = np.arange(40,70)
    # # train_index22 = np.arange(70,80)
    # train_index3 = np.arange(80,110)
    # # train_index33 = np.arange(110,120)
    # # train_index = np.concatenate([train_index1, train_index11, train_index2, train_index22, train_index3, train_index33])
    # train_index = np.concatenate([train_index1, train_index2, train_index3])

    # test_index1 = np.arange(30,40)
    # test_index2 = np.arange(70,80)
    # test_index3 = np.arange(110,120)
    # test_index = np.concatenate([test_index1, test_index2, test_index3])
        fold += 1
        model = Model()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.save_to_wandb:
            run = wandb.init(project="training-120-trajactory-range", name="test_aoa_10_fold_weight1-64,30%_peak_mae_3n_"+str(fold), dir='/home/nakorn/weight_bias', reinit=True)
        
        reject_list = np.random.choice(np.arange(1,108), size=76 , replace= False)
        train_index = np.delete(train_index, reject_list)
        print(train_index)
        train_data, train_label, test_data, test_label = data_iq[train_index], label_all[train_index] \
                                                            ,data_iq[test_index], label_all[test_index]
        train_data = np.reshape(train_data, (-1, *train_data.shape[-2:]))
        train_label = np.reshape(train_label, (-1, *train_label.shape[-1:]))
        test_data = np.reshape(test_data, (-1, *test_data.shape[-2:]))
        test_label = np.reshape(test_label, (-1, *test_label.shape[-1:]))
        print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
        np.save(test_dir + 'train_aoa_index_fold_' + str(fold), np.array(train_index))
        np.save(test_dir + 'test_aoa_index_fold_' + str(fold), np.array(test_index))

        train_set = Radar_train_Dataset(train_data=train_data, train_label=train_label)
        test_set = Radar_test_Dataset(test_data=test_data, test_label=test_label)

        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size)

        if args.test_only:
            test_loss, label, expect_z = test_function(test_loader)
            np.save(save_predict_path + 'label_z_%4_robot_3', label)
            np.save(save_predict_path + 'expect_z_%4_robot_3', expect_z)
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
                
                
                if args.save_to_wandb and (epoch%500 == 0):  
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'aoa_10_fold_rd_'+str(fold)+'_ep_'+str(epoch)+'.pt'))
            run.finish()
            evaluation(label, expect_z)
        


    
