import numpy as np
import glob
import natsort
import matplotlib.pyplot as plt
import argparse
from scipy import stats

signal_dir = 'D:/data_signal_MTI/project_util/signal_all_w_mti_cutoff_12/'
label_dir = 'D:/data_signal_MTI/project_util/label_all/'
all_trajectory = 117

range_rmse_all = []
azimuth_rmse_all = []

k = 1000
m = 1
mm = 1e-3
cm = 1e-2
us = 1e-6
MHz = 1e6
c = 299792458 * m


parser = argparse.ArgumentParser()
parser.add_argument('-kernel_size', type=int, default=5)
parser.add_argument('-guard', type=int, default=1)
parser.add_argument('-weight_kernel', type=int, default=3 )
parser.add_argument('-adc_samples', type=int, default=256)
parser.add_argument('-angle_samples', type=int, default=4)
parser.add_argument('-sampling_rate', type=float, default= 18750*k)
parser.add_argument('-freqz_slope', type=float, default = 199.975 * MHz / us)
parser.add_argument('-range_pad', type=int, default=0)
parser.add_argument('-angle_pad', type=int, default=0)

args = parser.parse_args()

kernel_c_i, kernel_c_j = int(args.kernel_size / 2), int(args.kernel_size / 2)
n_pad = ((0,0),(kernel_c_i,kernel_c_i),(kernel_c_j,kernel_c_j),(0,0))

a_pad = ((0,0),(0,0),(0,0),(0,args.angle_pad))
r_pad = ((0,0),(0,0),(0,args.range_pad),(0,0))

adc_sampling_time = args.adc_samples / args.sampling_rate
bandwith = adc_sampling_time * args.freqz_slope
range_resolution = c / (2 * bandwith)

range_resolution = range_resolution / ((args.range_pad + args.adc_samples) / args.adc_samples) 
angle_resolution = 2 / (args.angle_samples + args.angle_pad)

print("range_res: ", range_resolution / mm, "angle_res:", angle_resolution)

def F_cfar(test_data_all):
   
    data_kernel = test_data_all.copy()
    data_kernel = np.pad(data_kernel, pad_width=n_pad, mode='constant', constant_values = 0)
    tresh_plane = np.zeros((test_data_all.shape[0], test_data_all.shape[1],
                                test_data_all.shape[2], test_data_all.shape[3] ))
    # print(data_kernel.shape)
    for i in range(test_data_all.shape[1])[:]:
        for j in range(test_data_all.shape[2])[:]:
            kernel = data_kernel[:,i:i+args.kernel_size,j:j+args.kernel_size,:].copy()
            kernel[:, (kernel_c_i-args.guard):(kernel_c_i+args.guard)+1, 
                    (kernel_c_j-args.guard):(kernel_c_j+args.guard)+1,:] = 0
            ca_kernel = args.weight_kernel*np.mean(kernel, axis=(1,2))
            tresh_plane[:,i,j,:] = ca_kernel
    
    tresh_doppler = (test_data_all > tresh_plane)

    return tresh_doppler
    

def F_2DFFT(test_data_all, aoa_data):
   
    n = test_data_all.shape[1] / 2
    print(n)
    tresh_doppler = F_cfar(test_data_all)
    tresh_doppler = tresh_doppler[:, int(n-10):int(n+10), :int(test_data_all.shape[2]/4), :]

    print(tresh_doppler.shape)
    print(label_all.shape)
    which_frame, velocity, range_r, antenna = np.where(tresh_doppler == True)

    num_vr = np.array([which_frame, velocity, range_r])
    max_num_vr = np.max(num_vr, axis=1)
   
    plt.imshow(tresh_doppler[0,:,:,0])
    plt.show()

    for i in range(max_num_vr[0]+1):
        k = np.where(num_vr[0,:] == i)
        
        r_pos = stats.mode(num_vr[2,k[0]])
        v_pos = stats.mode(num_vr[1,k[0]])
        
        # print(i , r_pos[0], v_pos[0])
        if r_pos[0]:

            r_pos = np.array(r_pos[0] + 12)
            v_pos = np.array(v_pos[0] + 8)        

            
            r_pos_mm = (r_pos[0]*(range_resolution / mm))
            
            a_pos = np.argmax(aoa_data[i, v_pos[0], r_pos[0], :])
            center_a = int(aoa_data.shape[3] / 2)
            angle_res = np.pi / args.angle_samples + args.angle_pad
            a_pos_rad = (a_pos - center_a)*angle_res 

            # print(r_pos_mm, a_pos_rad)
            range_rmse_all, azimuth_rmse_all = calcurate_rmse(r_pos_mm, a_pos_rad, i)

        # plt.plot(aoa_data[i, v_pos[0], r_pos[0], :])
        # plt.show()

    range_rmse_all = np.sqrt(np.mean(range_rmse_all))
    azimuth_rmse_all = np.sqrt(np.mean(azimuth_rmse_all))
    print("range_rmse: ",range_rmse_all, "azimuth_rmse: ", azimuth_rmse_all)
    # plt.imshow(tresh_doppler[0,:,:,0])
    # plt.show()


def calcurate_rmse(r_pos_mm, a_pos_rad, i):
    # print(label_all[i,0], r_pos_mm)
    if r_pos_mm != 0:
        range_rmse = (label_all[i,0] - r_pos_mm)**2
        range_rmse_all.append(range_rmse)
        
        azimuth_rmse = (label_all[i,1] - a_pos_rad)**2
        azimuth_rmse_all.append(azimuth_rmse)
        
    return np.array(range_rmse_all), np.array(azimuth_rmse_all)

def cartesian_to_spherical(label):
    
    y_offset = 110
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,2])
    phi = np.arctan2(label[:,0,1] - y_offset , np.sqrt(label[:,0,0]**2 + label[:,0,2]**2))
    
    return r, zeta, phi


if __name__ == '__main__':
    
    test_data_all = []
    label_all = []
    count = 0

    for f_name in range(all_trajectory):
        count += 1
        real_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy' 
        label_name = label_dir + 'label_' + str(count) + '.npy'
        
        if count%4 == 0:
            test_data = np.load(real_name)
            print(test_data.shape)
            label = np.load(label_name)
            
            r, zeta, phi = cartesian_to_spherical(label)
            label = np.array([r, zeta, phi])
            label = label.T

            test_data_all.extend(test_data)
            label_all.extend(label)    
        else:
            print('...')

    test_data_all = np.array(test_data_all)
    # test_data_all = abs(test_data_all)
    label_all = np.array(label_all)
    test_data_all = test_data_all[::10]
    label_all = label_all[::10]
    print(test_data_all.shape, label_all.shape)

    test_data_all = np.complex64(test_data_all)
    
    test_data_all = np.pad(test_data_all, pad_width=r_pad, mode='constant', constant_values=0)
    data_fft_range = np.fft.fft(test_data_all, axis=2) / test_data_all.shape[2]
    data_fft_velocity = np.fft.fftshift(np.fft.fft(data_fft_range, axis=1) / test_data_all.shape[1], axes=1)
    half_velocity_bin = test_data_all.shape[1] / 2
    data_fft_angle = np.fft.fftshift(np.fft.fft(data_fft_velocity, axis=3) / test_data_all.shape[3], axes=3) # angle fft
    data_fft_angle = abs(data_fft_angle)


    plt.plot(abs(data_fft_range[17,0,:,0]))
    plt.show()

    F_2DFFT(abs(data_fft_velocity), data_fft_angle)
    
    # print(test_data_all.shape) 
     