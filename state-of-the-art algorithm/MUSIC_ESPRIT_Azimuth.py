import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator

signal_dir = 'D:/data_signal_MTI/project_util/signal_all_w_mti_cutoff_12/'
label_dir = 'D:/data_signal_MTI/project_util/label_all/'
all_trajectory = 117
aoa_estimate_music_all = []
aoa_estimate_esprit_all = []
L = 1  # number of sources
N = 4  # number of ULA elements 
snr = 10 # signal to noise ratio
count = 0

array = np.linspace(0,(N-1)/2,N)
Angles = np.linspace(-np.pi/2,np.pi/2,360)
numAngles = Angles.size

def cartesian_to_spherical(label):
    
    y_offset = 105
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,2])
    phi = np.arctan2(label[:,0,1] - y_offset , np.sqrt(label[:,0,0]**2 + label[:,0,2]**2))
    
    return r, zeta, phi

def array_response_vector(array,theta):
    N = array.shape
    v = np.exp(1j*2*np.pi*array*np.sin(theta))
    return v/np.sqrt(N)

def music(CovMat,L,N,array,Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _,V = LA.eig(CovMat)
    Qn  = V[:,L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(array,Angles[i])
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
    DoAsESPRIT = np.arcsin(np.angle(eigs)/np.pi)
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
            single_frame = np.swapaxes(single_frame, 0,1)

            CovMat = single_frame@single_frame.conj().transpose()
            DoAsMUSIC, psindB = music(CovMat,L,N,array,Angles)            
            DoAsESPRIT = esprit(CovMat,L,N)

            aoa_estimate_music = (((1/2)*DoAsMUSIC) - 90)*(np.pi / 180)
            aoa_estimate_esprit = DoAsESPRIT

            aoa_estimate_music_all.append(aoa_estimate_music)
            aoa_estimate_esprit_all.append(aoa_estimate_esprit)
            # print(aoa_estimate_music, aoa_estimate_esprit) 
            
            # plt.subplot(223)
            # plt.plot(Angles,psindB)
            # plt.plot(Angles[DoAsMUSIC],psindB[DoAsMUSIC],'x')
            # plt.title('MUSIC')
            # plt.legend(['pseudo spectrum','Estimated DoAs'])
            
            # plt.subplot(224)
            # plt.plot(DoAsESPRIT,np.zeros(L),'x')
            # plt.title('ESPRIT')
            # plt.legend(['Actual DoAs','Estimated DoAs'])

            # plt.show()

    else:
        print('...')

# aoa_estimate_music_all = np.array(aoa_estimate_music_all)
# aoa_estimate_esprit_all = np.array(aoa_estimate_esprit_all)

np.save('D:/data_signal_MTI/project_util/test_data/music_z%4', np.array(aoa_estimate_music_all))
np.save('D:/data_signal_MTI/project_util/test_data/esprit_z%4', np.array(aoa_estimate_esprit_all))











