import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
from scipy import stats
import operator

signal_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
fold_dir = 'D:/data_signal_MTI/project_util_3/10_fold_validation/100%_data/test_data/test_index_fold_'
pad_multi_range = 3
pad_multi_angle = 20


adc_range = 512
Rx = 8
r_pad = ((0,0),(0, pad_multi_range*adc_range),(0,0))
a_pad = ((0,0),(0,0),(0, pad_multi_angle*Rx))
all_trajectory = 120
range_res = 47.6 / (1+pad_multi_range)
angle_res = 3.1415 / (Rx*(pad_multi_angle+1)-1)
# angle_res = 0.448

m_r = np.arange(1.5708, -1.5708, -angle_res)
# print(m_r)

def range_estimate(range_fft):
    possible_range = range_fft[:,:10+(pad_multi_range*10),:]
    range_max = np.argmax(possible_range, axis=1)
    range_max = stats.mode(range_max, axis=1)
    range_max = stats.mode(range_max[0][:], axis=0)
    return range_max[0].reshape(-1)[0]

def angle_estimation(doppler_fft, doa_fft):
    angle_max_all = []
    dop_idx = []
    range_idx = []

    doppler_fft = doppler_fft[:,:10+(pad_multi_range*10),:]
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
    c_outlier = 0
    for j in range(all_trajectory):
        real_name = signal_dir + 'raw_iq_w_mti_' + str(j+1) + '.npy'
        print(j+1, real_name)
        test_data = np.load(real_name)
        for k in range(test_data.shape[0]):
            data_frame = test_data[k]
            data_frame_pad = np.pad(data_frame, pad_width= r_pad, mode='constant', constant_values=0)
            range_fft = np.fft.fft(data_frame_pad, axis=1)
            doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
            doppler_fft_pad = np.pad(doppler_fft, pad_width= a_pad, mode='constant', constant_values=0)
            doa_fft = np.fft.fftshift(np.fft.fft(doppler_fft_pad, axis=2), axes=2)
            

            range_max = range_estimate(abs(range_fft))
            angle_max = angle_estimation(abs(doppler_fft),abs(doa_fft))
            
            
            actual_range = range_res*range_max
            actual_doa = m_r[angle_max]
            
            
            kk = abs(range_fft[0,:40,0])
            kk_i =np.argmax(kk)
            kk_x =np.mean(abs(doa_fft), axis=0)
            kk_x =np.max(kk_x, axis=0)
            print(kk_i)
            plt.figure(1)
            plt.plot(kk)
            plt.figure(2)
            plt.plot(kk_x)
            plt.show()
            
            if actual_range < 94 or actual_range > 0 :
                print(actual_range)
                # plt.plot(abs(range_fft[0,:,0]))
                # plt.plot(abs(range_fft[0,:200,1]))
                # plt.plot(abs(range_fft[0,:200,2]))
                # plt.plot(abs(range_fft[0,:200,3]))
                # plt.plot(abs(range_fft[0,:200,4]))
                # plt.plot(abs(range_fft[0,:200,5]))
                # plt.plot(abs(range_fft[0,:200,6]))
                # plt.plot(abs(range_fft[0,:200,7]))
                # plt.show()
                c_outlier += 1

            

            expect_r.append(actual_range)
            expect_z.append(actual_doa)

        
        
        print("finish")
        print("outlier_each_filer", c_outlier)
    expect_r = np.array(expect_r)
    expect_z = np.array(expect_z)
    np.save('D:/data_signal_MTI/project_util_3/result_for_paper/expect_r_remove_outlier', np.array(expect_r))
    np.save('D:/data_signal_MTI/project_util_3/result_for_paper/expect_z_remove_outlier', np.array(expect_z))
                
            