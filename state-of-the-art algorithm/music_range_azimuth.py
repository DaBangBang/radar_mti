import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator

MHz = 1e6
us = 1e-6
m = 1
k = 1000
c = 299792458 * m
num_source = 1

SamplingRate = 12500 * k
FrequencySlope = 80.00 * MHz / us
ADCSamples = 512
ADCSamplingTime = ADCSamples / SamplingRate
RealBandwidth = FrequencySlope * ADCSamplingTime

search_distance = np.linspace(0, ADCSamples*0.0457, ADCSamples)
range_search = search_distance.size
print(search_distance.shape)

signal_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
all_trajectory = 120

# test_data = []

def range_steering_vector(range_search):
    n_vector = np.arange(0, ADCSamples)
    delta_t = n_vector / ADCSamples * ADCSamplingTime
    r_v = np.exp(1j * 2 * np.pi * 2 * RealBandwidth / c / ADCSamplingTime * delta_t * range_search)
    # print(r_v/np.sqrt(n_vector.size))
    return r_v/np.sqrt(n_vector.size)

if __name__ == '__main__':
    count = 0
    for f_name in range(all_trajectory)[:5]:
        count += 1
        real_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy'
        if count%4 == 0:
            test_data = np.load(real_name)
            test_range = test_data[:,0,:,0]
            test_azimuth = test_data[:,0,:,:]
            for i in range(test_range.shape[0]):
                signal_r = test_range[i,:]
                signal_r = signal_r.reshape((-1,1))
                # n_pad = ((0,512))
                # signal_r = np.pad(signal_r, n_pad, mode='constant', constant_values=0)
                cov_r = signal_r@signal_r.conj().transpose()
                u, V = LA.eig(cov_r)
                idx = u.argsort()[::-1]
                u = u[idx]
                V = V[:, idx]
                Qn = V[:, num_source:range_search]
                pspectrum = np.zeros(range_search)
                for j in range(range_search):
                    av = range_steering_vector(search_distance[j])
                    av = av.reshape((-1,1))
                    # print(Qn.shape, av.shape)
                    pspectrum[i] = 1 / LA.norm((Qn.conj().transpose()@av))
                plt.plot(search_distance, abs(np.fft.fft(signal_r, axis=0)/ signal_r.shape[0]))
                plt.plot(search_distance, pspectrum)
                plt.show()        
            print("eeee")            

