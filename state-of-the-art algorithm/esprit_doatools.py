import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator
import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.plotting as doaplot
import doatools


MHz = 1e6
GHz = 1e9
us = 1e-6
m = 1
k = 1000
c = 299792458 * m
fs = 13*MHz
Bw = 3.1508 * GHz

pad_multi_range = 3
pad_multi_angle = 20

all_trajectory = 120

adc_range = 512
Rx = 8

M = 100
p = 1

Ts = adc_range/fs
r_pad = (0, pad_multi_range*adc_range)
a_pad = ((0,pad_multi_range*adc_range),(0, pad_multi_angle*Rx))

signal_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
fold_dir = 'D:/data_signal_MTI/project_util_3/10_fold_validation/100%_data/test_data/test_index_fold_'

def range_estimation(data_frame):
    data_frame = np.pad(data_frame, pad_width=r_pad, mode='constant', constant_values=0)
    data_frame = np.reshape(data_frame, (-1,1))
    cov_matrix = data_frame@data_frame.conj().transpose()
    estimator = estimation.Esprit1D(wavelength)
    resolved, estimates = estimator.estimate(cov_matrix, 1, unit='deg')
    print("rocation", estimates.locations)
    # f = np.arange(0, fs, (fs/(data_frame.shape[0])))
    
    # cov_matrix = np.zeros((M,M))
    # for i in range((data_frame.shape[0])-M):
    #     cov_matrix = cov_matrix + data_frame[i:i+M]@data_frame[i:i+M].conj().transpose()
    # U, E, V = np.linalg.svd(cov_matrix)
    # S = U[:, 0:p]
    # S1 = S[0:M-1, :]
    # S2 = S[1:M, :]
    # P = np.linalg.pinv(S1)@S2
    # eig_p, v_p = np.linalg.eig(P)
    # omega = np.angle(eig_p)
    # f_est = fs*omega/(2*np.pi)
    # actual_r = f_est[0] * c * Ts / Bw / 2

    # return actual_r*1e3

wavelength = 1.0 # Normalized wavelength. Recall that d0 = wavelength / 2.
d0 = wavelength / 2
power_source = 1.0
power_noise = 1.0 # SNR = 0 dB
n_snapshots = 100
ula = model.UniformLinearArray(8, d0)
rula = model.UniformLinearArray(512, 6.27)

def angle_estimation(data_frame):
    data_frame = np.pad(data_frame, pad_width=a_pad, mode='constant', constant_values=0)
    data_frame = np.swapaxes(data_frame,0,1)
    cov_matrix = data_frame@data_frame.conj().transpose()
    # grid = estimation.FarField1DSearchGrid()
    estimator = estimation.Esprit1D(wavelength)
    resolved, estimates = estimator.estimate(cov_matrix, 1)
    print(estimates.locations)
    # U, E, V = np.linalg.svd(cov_matrix)
    # S = U[:, 0:p]
    # S1 = S[0:Rx-1, :]
    # S2 = S[1:Rx, :]
    # P = np.linalg.pinv(S1)@S2
    # eig_p, v_p = np.linalg.eig(P)
    # actual_doa = np.arcsin(np.angle(eig_p)/np.pi)
    
    # return actual_doa[0]

if __name__ == '__main__':
    count = 0
    expect_r = []
    expect_z = []
    for i in range(10):
        test_dir = fold_dir + str(i+1) +'.npy'
        print(test_dir)
        fold_num = np.load(test_dir)
        print(fold_num)
        for j in fold_num[:1]:
            print(j+1)
            real_name = signal_dir + 'raw_iq_w_mti_' + str(j+1) + '.npy'
            test_data = np.load(real_name)
            for k in range(test_data.shape[0])[:1]:
                actual_range = range_estimation(test_data[k,0,:,0])
                actual_doa = angle_estimation(test_data[k,0,:,:])
                
                # if  actual_range > 2380 or actual_range <= 0:
                #     expect_r.append(np.nan)
                #     expect_z.append(np.nan)
                #     print(actual_range, "outlier")
                # else:
                #     expect_r.append(actual_range)
                #     expect_z.append(-1*actual_doa)
            print("finish")
    expect_r = np.array(expect_r)
    expect_z = np.array(expect_z)
    # np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_r_esprit_pad_100%_fold_cut', np.array(expect_r))
    # np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_z_esprit_pad_100%_fold_cut', np.array(expect_z))