import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

N = 1000
raw_iq = np.load('D:/data_signal_MTI/raw_iq.npy')
single_frame = raw_iq[97,0,:,:]
single_frame = np.swapaxes(single_frame, 0,1)
print(raw_iq.shape)
a_pad = ((0,0),(0,0),(0,0),(0,177))

#fft method
# range_fft = np.fft.fft(raw_iq, axis=2) / raw_iq.shape[2]
# velocity_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1) / raw_iq.shape[1], axes=1)
# velocity_fft_pad = np.pad(velocity_fft, pad_width= a_pad, mode='constant', constant_values =0)
# angle_fft = np.fft.fftshift(np.fft.fft(velocity_fft_pad, axis=3) / velocity_fft_pad.shape[3], axes=3)

# plt.figure(1)
# plt.imshow(abs(velocity_fft[97,:,:,0]))

# plt.figure(2)
# plt.plot(abs(angle_fft[97,32,5,:]))
# plt.plot(abs(angle_fft[97,31,5,:]))
# plt.plot(abs(angle_fft[97,33,5,:]))


#music method
M = 2
Rxx = single_frame * np.matrix.getH(np.asmatrix(single_frame)) / N
D,E = la.eig(Rxx)
idx = D.argsort()[::-1]
lmbd = D[idx]
E = E[:, idx]
En = E[:, M:len(E)]

# print(En.shape)



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
print(music_spec.shape)
plt.figure(3)
plt.plot(music_spec[:,0,0])
plt.show()