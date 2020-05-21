import numpy as np
import glob
import natsort
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

adcSamples = 1000
chirp = 32
frame_number = 500

def animate(i):
    line1.set_ydata(abs(range_fft[i,0,:,0]))
    ax2.set_array(velocity_fft[i,:,:200,0])
    print(i)
    return line1, ax2

def runGraphInitial():

    global line1, ax2

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.set_ylim([-1,100])
    line1, = ax1.plot(abs(range_fft[0,0,:,0]))
    
    ax2 = fig.add_subplot(122)
    ax2 = plt.imshow(velocity_fft[0,:,:200,0], aspect= 'auto', origin='lower' , cmap='jet', 
         extent=[0, velocity_fft.shape[2]-1, velocity_fft.shape[1]/2., -velocity_fft.shape[1]/2.], interpolation = 'catrom')
    
    ani = FuncAnimation(fig, animate, frames= frame_number  ,interval = 25)
    plt.show()

def dopplerFFT():
    n = chirp
    dop_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1) / n, axes=1)
    dop_fft = abs(dop_fft)
    return dop_fft

def rangeFFT():
    n = adcSamples
    range_fft = np.fft.fft(raw_signal, axis=2) / n
    return range_fft

def subtraction(raw_iq, b_raw_iq):

    b_raw_iq = np.mean(b_raw_iq, axis=(0,1))
    print(raw_iq.shape, b_raw_iq.shape)
    signal_sub = raw_iq - b_raw_iq

    return signal_sub

def main():

    global raw_signal, range_fft, velocity_fft

    folder_name = glob.glob('D:/SmallTrack/data/data_stage_acrylic_mov/p*')
    folder_name = natsort.natsorted(folder_name)
    
    for f_name in folder_name:
        print(f_name)
        real_name = f_name + '/raw_signal_real.npy'
        imag_name = f_name + '/raw_signal_imag.npy'
        real_part = np.load(real_name)
        imag_part = np.load(imag_name)
        print(real_part.shape)
        raw_iq = real_part + 1j*imag_part
        raw_iq = np.complex64(raw_iq)
        n_pad = ((0,0),(0,0),(0,5000),(0,0))
        raw_iq = np.pad(raw_iq, pad_width=n_pad, mode='constant', constant_values=0)
        raw_iq = raw_iq[:,:,:,:2]

    b_folder_name = glob.glob('D:/SmallTrack/data/data_stage_acrylic_mov/b*')
    b_folder_name = natsort.natsorted(b_folder_name)

    for b_f_name in b_folder_name:
        print(b_f_name)
        real_name = b_f_name + '/raw_bg_real.npy'
        imag_name = b_f_name + '/raw_bg_imag.npy'
        real_part = np.load(real_name)
        imag_part = np.load(imag_name)
        print(real_part.shape)
        b_raw_iq = real_part + 1j*imag_part 
        b_raw_iq = np.complex64(b_raw_iq)
        n_pad = ((0,0),(0,0),(0,5000),(0,0))
        b_raw_iq = np.pad(b_raw_iq, pad_width=n_pad, mode='constant', constant_values=0)
        b_raw_iq = b_raw_iq[:,:,:,:2]

    print(raw_iq.shape, b_raw_iq.shape)
    raw_signal = subtraction(raw_iq, b_raw_iq)
    range_fft = rangeFFT()
    velocity_fft = dopplerFFT()
    
    runGraphInitial()
    # plt.plot(abs(np.array(range_fft[0])[0,0,:,0]))
    # plt.imshow(abs(np.array(range_fft[0])[0,:,:,0]))

    # plt.show()

if __name__ == "__main__":
    main() 