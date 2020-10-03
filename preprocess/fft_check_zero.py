import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

signal_dir = 'D:/data_signal_MTI/project_util_2/signal_all_w_mti_cutoff_12/'
label_dir = 'D:/data_signal_MTI/project_util_2/label_all/'

all_trajectory = 117
count = 0

test_signal_all = []
test_label_all = []

def animate(i):
    line1.set_ydata(test_signal_all[i,0,:,0])
    print(i)
    return line1,

def cartesian_to_spherical(label):
    y_offset = 105
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    return r

def data_preparation(data_iq, label):

    
    label = cartesian_to_spherical(label)
    label = np.float64(label)

    return data_fft_modulus, label

# def check_zero_function():

for f_name in range(all_trajectory)[:80]:
    count += 1
    real_name = signal_dir + 'raw_iq_w_mti_' + str(count) + '.npy' 
    label_name = label_dir + 'label_' + str(count) + '.npy'

    if count%4 == 0:
        data_iq = np.load(real_name)
        label = np.load(label_name)
        # data_fft_modulus, label = data_preparation(data_iq, label)

        test_signal_all.extend(data_iq)
        test_label_all.extend(label)
        print(np.array(test_signal_all).shape) 
    
    else:
        print('pass')

test_signal_all = np.array(test_signal_all)
test_signal_all = test_signal_all[4400:4700]
test_signal_all = abs(np.fft.fft(test_signal_all, axis=2))**2
count = 0
for i in range(test_signal_all.shape[0]):
    if np.max(test_signal_all[i,0,:,0]) > 10*np.mean(test_signal_all):
        count += 1
    else:
        count += 0
print(count)


print(np.mean(test_signal_all), np.max(test_signal_all), np.min(test_signal_all))
test_signal_all = test_signal_all / 1000
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
line1, = ax1.plot(test_signal_all[0,0,:,0])
ax1.set_ylim([-5, 100000])

ani = FuncAnimation(fig, animate, frames=300, interval=20)
plt.show()
