import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

angle_pad = 128
a_pad = ((0,0),(0,0),(0,0),(0,angle_pad))
angle_fft_all = []
x = np.linspace(-90, 90, 140)
y = np.linspace(-90, 90, 140)
xx, yy = np.meshgrid(x, y)
print(yy.shape)

def animate(i):
    axDop.set_array(angle_fft[i,:150,:])
    # circ = Circle((10,10), 1 , color='red')
    # axDop.add_patch(circ)
    print(i)
    return axDop

signal = np.load('D:/data_signal_MTI/project_util/test_data/raw_iq_w_mti_1.npy')
signal = signal[0:600,:,:,:]
range_fft = np.fft.fft(signal, axis = 2) / signal.shape[2]
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis = 1) / signal.shape[1], axes= 1 )
doppler_fft = np.pad(doppler_fft, pad_width=a_pad, mode='constant', constant_values=0)
angle_fft = np.fft.fftshift(np.fft.fft(doppler_fft, axis=3)/doppler_fft.shape[3], axes=3)
# angle_fft = np.amax(abs(angle_fft), axis=1)-+
angle_fft = abs(angle_fft[:,32,:,:])
# for i in range(doppler_fft.shape[0]):
#     coor_x, coor_y = np.where(doppler_fft[i,:,:,0] == np.amax(doppler_fft[i,:,:,0]))
#     selected_doppler = doppler_fft[i, coor_x[0], coor_y[0], :]
#     data_fft_a = np.pad(selected_doppler, pad_width=a_pad, mode='constant', constant_values=0)
#     data_fft_a = np.fft.fftshift(np.fft.fft(data_fft_a, axis=0) / data_fft_a.shape[0], axes=0)
#     angle_fft_all.append(abs(data_fft_a))


# angle_fft_all = np.array(angle_fft_all)
# mesh_angle = np.matmul(angle_fft_all, yy)
# print(mesh_angle.shape)

fig = plt.figure(1)
axDop = plt.imshow(angle_fft[0,:150,:], cmap='jet', aspect= 'equal')


ani = FuncAnimation(fig, animate, frames=450, interval = 25)
plt.show()