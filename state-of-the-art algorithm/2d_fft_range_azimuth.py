import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
from scipy import stats
import operator

signal_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
pad_multi_range = 3
pad_multi_angle = 20


adc_range = 512
Rx = 8
r_pad = ((0,0),(0, pad_multi_range*adc_range),(0,0))
a_pad = ((0,0),(0,0),(0, pad_multi_angle*Rx))
all_trajectory = 120
range_res = 45.74 / (1+pad_multi_range)
angle_res = 3.1415 / (Rx*(pad_multi_angle+1)-1)
# angle_res = 0.448

m_r = np.arange(1.5708, -1.5708, -angle_res)
print(m_r)

def range_estimate(range_fft):
    possible_range = range_fft[:,:50+(pad_multi_range*50),:]
    range_max = np.argmax(possible_range, axis=1)
    range_max = stats.mode(range_max, axis=1)
    range_max = stats.mode(range_max[0][:], axis=0)
    return range_max[0].reshape(-1)[0]

def angle_estimation(doppler_fft, doa_fft):
    angle_max_all = []
    dop_idx = []
    range_idx = []

    doppler_fft = doppler_fft[:,:50+(pad_multi_range*50),:]
    for i in range(doppler_fft.shape[2]):
        xy = np.where(doppler_fft[:,:,i] == np.amax(doppler_fft[:,:,i]))
        dop_idx.append(xy[0][0])
        range_idx.append(xy[1][0])
    dop_max = stats.mode(dop_idx)
    range_max = stats.mode(range_idx)
    angle_max = np.argmax(doa_fft[dop_max[0][0], range_max[0][0], :])

    return angle_max

if __name__ == '__main__':
    count = 0
    expect_r = []
    expect_z = []
    for f_name in range(all_trajectory):
        count += 1
        real_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy'
        if count%4 == 0:
            test_data = np.load(real_name)
            for i in range(test_data.shape[0]):
                data_frame = test_data[i]
                # print(data_frame.shape)
                data_frame_pad = np.pad(data_frame, pad_width= r_pad, mode='constant', constant_values=0)
                range_fft = np.fft.fft(data_frame_pad, axis=1)
                doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
                doppler_fft_pad = np.pad(doppler_fft, pad_width= a_pad, mode='constant', constant_values=0)
                doa_fft = np.fft.fftshift(np.fft.fft(doppler_fft_pad, axis=2), axes=2)
                
                range_max = range_estimate(abs(range_fft))
                angle_max = angle_estimation(abs(doppler_fft),abs(doa_fft))
         
                actual_range = range_res*range_max
                actual_doa = m_r[angle_max]
                expect_r.append(actual_range)
                expect_z.append(actual_doa)
            print("finish")
    expect_r = np.array(expect_r)
    expect_z = np.array(expect_z)
    np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_2dfft_pad', np.array(expect_r))
    np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_2dfft_pad', np.array(expect_z))
                
            