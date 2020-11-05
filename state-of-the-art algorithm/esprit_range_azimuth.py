import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator

MHz = 1e6
GHz = 1e9
us = 1e-6
m = 1
k = 1000
c = 299792458 * m
fs = 13*MHz
Bw = 3.1508 * GHz
pad_multi_range = 0
pad_multi_angle = 0

all_trajectory = 120

adc_range = 512
Rx = 8

M = 100
p = 1

Ts = adc_range/fs
r_pad = (0, pad_multi_range*adc_range)
a_pad = ((0,pad_multi_range*adc_range),(0, pad_multi_angle*Rx))

signal_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'

def range_estimation(data_frame):
    data_frame = np.pad(data_frame, pad_width=r_pad, mode='constant', constant_values=0)
    data_frame = np.reshape(data_frame, (-1,1))
    f = np.arange(0, fs, (fs/(data_frame.shape[0])))
    cov_matrix = np.zeros((M,M))
    for i in range((data_frame.shape[0])-M):
        cov_matrix = cov_matrix + data_frame[i:i+M]@data_frame[i:i+M].conj().transpose()
    U, E, V = np.linalg.svd(cov_matrix)
    S = U[:, 0:p]
    S1 = S[0:M-1, :]
    S2 = S[1:M, :]
    P = np.linalg.pinv(S1)@S2
    eig_p, v_p = np.linalg.eig(P)
    omega = np.angle(eig_p)
    f_est = fs*omega/(2*np.pi)
    actual_r = f_est[0] * c * Ts / Bw / 2

    return actual_r*1e3

def angle_estimation(data_frame):
    data_frame = np.pad(data_frame, pad_width=a_pad, mode='constant', constant_values=0)
    data_frame = np.swapaxes(data_frame,0,1)
    cov_matrix = data_frame@data_frame.conj().transpose()
    U, E, V = np.linalg.svd(cov_matrix)
    S = U[:, 0:p]
    S1 = S[0:Rx-1, :]
    S2 = S[1:Rx, :]
    P = np.linalg.pinv(S1)@S2
    eig_p, v_p = np.linalg.eig(P)
    actual_doa = np.arcsin(np.angle(eig_p)/np.pi)
    
    return actual_doa[0]

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
                actual_range = range_estimation(test_data[i,0,:,0])
                actual_doa = angle_estimation(test_data[i,0,:,:])
                expect_r.append(actual_range)
                expect_z.append(actual_doa)
            print("finish")
    expect_r = np.array(expect_r)
    expect_z = np.array(expect_z)
    np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_esprit_nopad', np.array(expect_r))
    np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_esprit_nopad', np.array(expect_z))