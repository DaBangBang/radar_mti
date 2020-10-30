import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
from scipy import stats
import operator

signal_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
pad_multi_range = 4
pad_multi_angle = 20


adc_range = 512
Rx = 8
r_pad = ((0,0),(0,0),(0, pad_multi_range*adc_range),(0,0))
a_pad = ((0,0),(0,0),(0,0),(0, pad_multi_angle*Rx))
all_trajectory = 120
range_res = 45.74 / (1+pad_multi_range)
angle_res = 3.1415 / (Rx*(pad_multi_angle+1)-1)
# angle_res = 0.448

m_r = np.arange(-1.5708, 1.5708, angle_res)
print(m_r)

def range_estimate(range_fft):
    possible_range = range_fft[:,:,:50+(pad_multi_range*50),:]
    range_max = np.argmax(possible_range, axis=2)
    range_max = stats.mode(range_max, axis=2)
    range_max = stats.mode(range_max[0][:], axis=1)
    plt.plot(range_fft[30,0,:50+(pad_multi_range*50),0])
    plt.show()
    return range_max[0].reshape(-1)

def angle_estimation(doppler_fft, doa_fft):
    angle_max_all = []
    for i in range(doppler_fft.shape[0]):
        dop_idx = []
        range_idx = []
        for j in range(doppler_fft.shape[3]):
            xy = np.where(doppler_fft[i,:,:,j] == np.amax(doppler_fft[i,:,:,j]))
            dop_idx.append(xy[0][0])
            range_idx.append(xy[1][0])
        dop_max = stats.mode(dop_idx)
        range_max = stats.mode(range_idx)
        angle_max = np.argmax(doa_fft[i, dop_max[0][0], range_max[0][0], :])
        angle_max_all.append(angle_max)
        # plt.plot(doa_fft[i, dop_max[0][0], range_max[0][0], :])
        # plt.show()

    return angle_max_all

if __name__ == '__main__':
    count = 0
    for f_name in range(all_trajectory)[:5]:
        count += 1
        real_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy'
        if count%4 == 0:
            test_data = np.load(real_name)
            test_data = test_data[:50]
            test_data = np.pad(test_data, pad_width= r_pad, mode='constant', constant_values=0)
            test_data = np.fft.fft(test_data, axis=2)
            range_max = range_estimate(abs(test_data))
            
            doppler_fft = np.fft.fftshift(np.fft.fft(test_data, axis=1), axes=1)
            doa_fft = np.pad(test_data, pad_width= a_pad, mode='constant', constant_values=0)
            doa_fft = np.fft.fftshift(np.fft.fft(doa_fft, axis=3), axes=3)
            angle_max = angle_estimation(abs(doppler_fft),abs(doa_fft))
         
            # print(range_max)
            actual_range = range_res*range_max
            actual_doa = m_r[angle_max]
            print(actual_range, actual_doa)
            