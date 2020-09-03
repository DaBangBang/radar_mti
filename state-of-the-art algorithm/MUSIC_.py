import numpy as np
import glob
import natsort
import matplotlib.pyplot as plt
import argparse
from scipy import stats
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from matplotlib.patches import Circle
import scipy.linalg as la

signal_dir = 'D:/data_signal_MTI/project_util/signal_all_w_mti_cutoff_12/'
label_dir = 'D:/data_signal_MTI/project_util/label_all/'
all_trajectory = 117
aoa_estimate_all = []

M = 1

def detect_peaks(music_spec):
    peak = np.where(music_spec == np.amax(music_spec))
    mark = [peak[0]]
    aoa_estimate = peak[0]*0.00174533 - 1.5708
    aoa_estimate_all.append(aoa_estimate) 
    # plt.plot(music_spec, '-gD', markevery=mark)
    # plt.show()


def MUSIC(raw_iq):
    
    for i in range(raw_iq.shape[0]):
        
        single_frame = raw_iq[i,0,:,:]
        single_frame = np.swapaxes(single_frame, 0,1)
        # print(single_frame.shape)
        Rxx = single_frame * np.matrix.getH(np.asmatrix(single_frame)) / single_frame.shape[1]
        D,E = la.eig(Rxx)
        idx = D.argsort()[::-1]
        lmbd = D[idx]
        E = E[:, idx]
        En = E[:, M:len(E)]

        music_spec = []

        for zeta in np.arange(-90,90+1,0.1):
        
            a = np.array([[1, np.exp(1j*np.pi*np.sin(zeta*np.pi/180)), 
                np.exp(2*1j*np.pi*np.sin(zeta*np.pi/180)), np.exp(3*1j*np.pi*np.sin(zeta*np.pi/180))]])
            a = np.swapaxes(a, 0,1)
            aH = np.matrix.getH(np.asmatrix(a))
            EnH = np.matrix.getH(np.asmatrix(En))
            aHEn = np.matmul(aH, En)
            aHEnEnH = np.matmul(aHEn, EnH)
            aHEnEnHa = np.matmul(aHEnEnH, a)
            Pspec = 1 / aHEnEnHa
            music_spec.append(abs(Pspec))
        
        music_spec = np.array(music_spec)
        detect_peaks(music_spec[:,0,0])
        # print(music_spec.shape)
        # plt.figure(1)
        # plt.plot(music_spec[:,0,0])
        # plt.show()    

def cartesian_to_spherical(label):
    
    y_offset = 105
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
            print(np.linspace(0,(N-1)/2,N))
            test_data = np.load(real_name)
            test_data = np.complex64(test_data)
            label = np.load(label_name)
            print(label.shape)
            
            r, zeta, phi = cartesian_to_spherical(label)
            label = np.array([r, zeta, phi])
            label = label.T
            MUSIC(test_data)
            
         

        else:
            print('...')

    np.save('D:/data_signal_MTI/project_util/test_data/music_z%4', np.array(aoa_estimate_all))
