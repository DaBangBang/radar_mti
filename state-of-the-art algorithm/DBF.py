import numpy as np
import glob
import natsort
import matplotlib.pyplot as plt
import argparse
from scipy import stats
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from matplotlib.patches import Circle

signal_dir = 'D:/data_signal_MTI/project_util/signal_all_w_mti_cutoff_12/'
label_dir = 'D:/data_signal_MTI/project_util/label_all/'
all_trajectory = 117

range_square_all = []
aoa_square_all = []
range_estimate_all =[]
aoa_estimate_all = []

angle_pad = 256
range_pad = 1024

r_pad = ((0,0),(0,0),(0,range_pad),(0,0))
a_pad = ((0,angle_pad))

k = 1000
m = 1
mm = 1e-3
cm = 1e-2
us = 1e-6
MHz = 1e6
c = 299792458 * m

adc_samples = 256
sampling_rate = 18750*k
freqz_slope = 199.975 * MHz / us
angle_samples = 4

adc_sampling_time = adc_samples / sampling_rate
bandwith = adc_sampling_time * freqz_slope
range_resolution = c / (2 * bandwith)
range_resolution = range_resolution / ((range_pad + adc_samples) / adc_samples) 
angle_resolution = 2 / (angle_samples + angle_pad)

print("range_res: ", range_resolution / mm, "angle_res:", angle_resolution)

def cartesian_to_spherical(label):
    
    y_offset = 105
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,2])
    phi = np.arctan2(label[:,0,1] - y_offset , np.sqrt(label[:,0,0]**2 + label[:,0,2]**2))
    
    return r, zeta, phi

def detect_peaks(data_fft_v, data_fft, label):
    
    for i in range(data_fft_v.shape[0]):
        coor_xy = []
        for j in range(data_fft_v.shape[3]):
            coor_x, coor_y = np.where(data_fft_v[i,:,:,j] == np.amax(data_fft_v[i,:,:,j]))
            coor_xy.append([coor_x, coor_y])
            
        coor_xy = np.array(coor_xy)
        coor_x_mode,_ = stats.mode(coor_xy[:,0,:])
        coor_y_mode,_ = stats.mode(coor_xy[:,1,:])
        # print(coor_x_mode[0][0], coor_y_mode[0][0])
        
        range_estimation = coor_y_mode[0][0]*(range_resolution / mm)
        range_estimate_all.append(range_estimation)
        range_square = (label[i,0] - range_estimation)**2
        range_square_all.append(range_square)
        
        '''
        '''
        

        # print(coor_x_mode[0][0], coor_y_mode[0][0])
        # fig,ax = plt.subplots(1)
        # ax.set_aspect('equal')
        # ax.imshow(abs(data_fft_v[i,:,:,j]))
        # circ = Circle((coor_y,coor_x), 1, color='red')
        # ax.add_patch(circ)
        # plt.show()
             
        data_fft_a = data_fft[i,coor_x_mode[0][0],coor_y_mode[0][0],:]
        data_fft_a = np.pad(data_fft_a, pad_width=a_pad, mode='constant', constant_values=0)
        # print(data_fft_a.shape)
        data_fft_a = np.fft.fftshift(np.fft.fft(data_fft_a, axis=0) / data_fft_a.shape[0], axes=0) # fft_angle
        data_fft_a = abs(data_fft_a)
        
        detect_angle_peak(data_fft_a, label, i)
    

def detect_angle_peak(data_fft_a, label, i):

    peak= np.where(data_fft_a == np.amax(data_fft_a))
    aoa_estimate = (- (np.pi / data_fft_a.shape[0]) * peak[0]) + (np.pi/2)
    # print(aoa_estimate.shape)
    aoa_estimate_all.append(aoa_estimate[0])
    aoa_square = (label[i,1] - aoa_estimate[0])**2
    aoa_square_all.append(aoa_square)
    # print(aoa_estimate) 
    # mark = [peak[0]]
    # plt.plot(data_fft_a, '-gD', markevery=mark)
    # plt.show()
        

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
            test_data = np.complex64(test_data)
            label = np.load(label_name)
            
            r, zeta, phi = cartesian_to_spherical(label)
            label = np.array([r, zeta, phi])
            label = label.T
            
            test_data = np.pad(test_data, pad_width=r_pad, mode='constant', constant_values=0)
            print(test_data.shape)
            data_fft = np.fft.fft(test_data, axis=2) / test_data.shape[2] # fft_range
            data_fft = np.fft.fftshift(np.fft.fft(data_fft, axis=1) / test_data.shape[1], axes=1) # fft_velocity
            half_velocity_bin = test_data.shape[1] / 2
            detect_peaks(abs(data_fft[:,:,:int(test_data.shape[2]/4),:]), data_fft, label)
            print(np.array(range_square_all).shape, np.array(aoa_square_all).shape)

        else:
            print('...')
    np.save('D:/data_signal_MTI/project_util/test_data/dbf_r%4', np.array(range_estimate_all))
    np.save('D:/data_signal_MTI/project_util/test_data/dbf_z%4', np.array(aoa_estimate_all))

    
    print("range_rmse", "aoa_rmse")
    print(np.sqrt(np.mean(np.array(range_square_all))), np.sqrt(np.mean(np.array(aoa_square_all))) )
    
