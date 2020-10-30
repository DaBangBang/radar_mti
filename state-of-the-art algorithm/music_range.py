import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator

N = 512
s_N = 512
L = 2

f1 = 1.000e9
f2 = 1.020e9
fs = 4*f1

signal = []
f=[]
f_inc=fs/(N)
f = np.linspace(0, fs, s_N)
f = f/fs
num_f_search = f.size
print(num_f_search)
array = np.arange(0,N)


def steering_vector(array, theta):
    v = np.exp(1j*2*np.pi*array*theta)
    # print(v.shape)
    return v/np.sqrt(N)

for i in range(N):
    mix = np.sqrt(1.000)*np.exp(1j*2*np.pi*(f1/fs)*i) + np.sqrt(0.500)*np.exp(1j*2*np.pi*(f2/fs)*i)
    signal.append(mix)

signal = np.array(signal)
signal = signal.reshape((-1, 1))
signal_fft = np.fft.fft(signal, axis=0)
signal_fft = abs(signal_fft)

# signal_e = np.zeros((s_N, s_N))
# for n in range(N-s_N):
#     signal_e = signal_e + signal[n:n+s_N]@signal[n:n+s_N].conj().transpose()

signal_e = signal@signal.conj().transpose()
print(signal_e.shape)
u, V = LA.eig(signal_e)
idx = u.argsort()[::-1]
u = u[idx]
V = V[:, idx]
Qn = V[:, L:N]
pspectrum = np.zeros(num_f_search)
for i in range(num_f_search):
    av = steering_vector(array, f[i])
    # qn = Qn[:, i]
    # qn = qn.reshape((-1,1))
    av = av.reshape((-1,1))
    print(Qn.shape, av.shape)
    pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
print(pspectrum.shape)
psindb = np.log10(10*pspectrum/pspectrum.min())

plt.plot(f, np.log10(10*signal_fft/signal_fft.min()))
plt.plot(f, psindb)
plt.show()
print(pspectrum.shape)