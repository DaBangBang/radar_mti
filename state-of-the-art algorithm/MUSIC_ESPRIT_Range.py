import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
N = 1000
M = 100
s = 2

f1 = 1.000e9
f2 = 1.020e9
fs = 4*f1

signal = []
f=[]
f_inc=fs/(N)

def steering_vector(f_inc, N, M, fs):
    steering = []
    for k in range(M):
        s_vector = []
        for i in range(N):
            w = 2*np.pi*i*(f_inc/fs)
            s_vector.append(np.exp(1j*w*k))
        steering.append(s_vector)
    steering = np.array(steering)
    return steering

steering = steering_vector(f_inc, N, M, fs)

for i in range(N):
    f.append(i*f_inc)
    mix = np.sqrt(1.000)*np.exp(1j*2*np.pi*(f1/fs)*i) + np.sqrt(0.10)*np.exp(1j*2*np.pi*(f2/fs)*i)
    signal.append(mix)

signal = np.array(signal)
signal = signal + np.sqrt(0.10) * (np.random.randn(N) + 1j*np.random.randn(N))
f = np.array(f)

fft_s = np.fft.fft(signal)
fft_a = abs(fft_s)

# print(steering)
signal_a = np.reshape(signal, (-1, 1))
cov_s = np.zeros((M,M))

for ii in range(N-M+1):
    cov_s = cov_s + (signal_a[ii:ii+M]@signal_a[ii:ii+M].conj().transpose())
    

u,V = LA.eig(cov_s)
idx = u.argsort()[::-1]
u = u[idx]
V = V[:,idx]
V = V[:, s:M]
den = np.zeros((N,1))
print(v)
for n in range(M-s):
    v = V[:,n]
    v = np.reshape(v, (-1,1))
    xn = steering.conj().transpose()
    den = den+(abs(xn@v))**2
    
psd_music = 1/den 
# print(psd_music)

U, E, D = LA.svd(cov_s)
S = U[:,0:s]
S1 = S[0:M-1,:]
S2 = S[1:M,:]
P = LA.pinv(S1)@S2
ev,_ = LA.eig(P)
idx = ev.argsort()[::-1]
ev = ev[idx]
print(ev)
wn = np.angle(ev)
f_est = fs*wn/(2*np.pi)

print(f_est)



plt.plot(f,20*np.log(fft_a/max(fft_a)))
plt.plot(f,10*np.log(psd_music/max(psd_music)))
plt.axvline(x=f_est[0])
plt.axvline(x=f_est[1])
plt.xlim((0.98e9, 1.04e9))
plt.ylim((-150, 10))
plt.show()

# print(cov_s.shape, signal_a.shape)

