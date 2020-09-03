import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator

signal_dir = 'D:/data_signal_MTI/project_util/signal_all_w_mti_cutoff_12/'
label_dir = 'D:/data_signal_MTI/project_util/label_all/'
all_trajectory = 117
range_estimate_music_all = []
aoa_estimate_esprit_all = []
L = 1  # number of sources
# N = 4  # number of ULA elements 
snr = 10 # signal to noise ratio
count = 0

us = 1e-6
GHz = 1e9
MHz = 1e6
m = 1
c = 299792458 * m
k = 1000
mm = 1e-3
cm = 1e-2
RampEndTime = 20 * us
IdleTime = 9 * us
Bandwidth = 3.9995 * GHz
Frequency = 77 * GHz
FrequencySlope = 199.975 * MHz / us
ADCSamples = 256
SamplingRate = 18750 * k
ADCStartTime = 3 * us
ADCSamplingTime = ADCSamples / SamplingRate
RealBandwidth = ADCSamplingTime * FrequencySlope
N = ADCSamples

array = np.linspace(0, (N-1) / N * ADCSamplingTime, N)
Ranges = np.linspace(100, 500 ,400) #mm search
print(Ranges.shape)
numRanges = Ranges.size


def cartesian_to_spherical(label):
    
    y_offset = 105
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,2])
    phi = np.arctan2(label[:,0,1] - y_offset , np.sqrt(label[:,0,0]**2 + label[:,0,2]**2))
    
    return r, zeta, phi

def array_response_vector(array,obj_distance):
    N = array.shape
    v = np.exp(-1j * 2 * np.pi * array * (RealBandwidth / ADCSamplingTime * 2 * obj_distance / c))
    return v/np.sqrt(N)

def music(CovMat,L,N,array,Ranges):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Ranges are the grid of directions in the azimuth angular domain
    _,V = LA.eig(CovMat)
    Qn  = V[:,L:N]
    numRanges = Ranges.size
    pspectrum = np.zeros(numRanges)
    for i in range(numRanges):
        av = array_response_vector(array,Ranges[i])
        pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
    psindB    = np.log10(10*pspectrum/pspectrum.min())
    # DoAsMUSIC,_= ss.find_peaks(psindB,height=1, distance=1)
    DoAsMUSIC = np.where(psindB == np.amax(psindB))
    return DoAsMUSIC[0],pspectrum

def esprit(CovMat,L,N):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = LA.eig(CovMat)
    S = U[:,0:L]
    Phi = LA.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs,_ = LA.eig(Phi)
    DoAsESPRIT = np.angle(eigs)
    # DoAsESPRIT = 49000*DoAsESPRIT/(2*np.pi)
    return DoAsESPRIT

#=============================================================



for f_name in range(all_trajectory):
    count += 1
    real_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy' 
    label_name = label_dir + 'label_' + str(count) + '.npy'
    
    if count%4 == 0:

        raw_iq = np.load(real_name)
        raw_iq = np.complex64(raw_iq)
        label = np.load(label_name)
        print(label.shape)
        
        r, zeta, phi = cartesian_to_spherical(label)
        label = np.array([r, zeta, phi])
        label = label.T
        
        for i in range(raw_iq.shape[0]):
            single_frame = raw_iq[i,0,:,:]
            # single_frame = np.reshape(single_frame, (256,1))
            # print(single_frame.shape)
            # single_frame = np.swapaxes(single_frame, 0,1)

            CovMat = single_frame@single_frame.conj().transpose()
            DoAsMUSIC, psindB = music(CovMat,L,N,array,Ranges)            
            # DoAsESPRIT = esprit(CovMat,L,N)
            
            
            range_estimate_music = DoAsMUSIC
            # print(DoAsESPRIT, range_estimate_music)
            range_estimate_music_all.append(range_estimate_music)
            # aoa_estimate_esprit_all.append(aoa_estimate_esprit)
            # print(aoa_estimate_music, aoa_estimate_esprit) 
            
            plt.subplot(223)
            plt.plot(Ranges,psindB)
            plt.plot(Ranges[DoAsMUSIC],psindB[DoAsMUSIC],'x')
            plt.title('MUSIC')
            plt.legend(['pseudo spectrum','Estimated DoAs'])
            
            plt.subplot(224)
            plt.plot(DoAsESPRIT,np.zeros(L),'x')
            plt.title('ESPRIT')
            plt.legend(['Actual DoAs','Estimated DoAs'])

            plt.show()

    else:
        print('...')

# aoa_estimate_music_all = np.array(aoa_estimate_music_all)
# aoa_estimate_esprit_all = np.array(aoa_estimate_esprit_all)

np.save('D:/data_signal_MTI/project_util/test_data/music_r%4', np.array(range_estimate_music_all))
# np.save('D:/data_signal_MTI/project_util/test_data/esprit_z%4', np.array(aoa_estimate_esprit_all))











