import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import natsort
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
import itertools
from pylab import *
import scipy.signal as signal

frame_number = 600
chirp = 32
adcSamples = 200
TxRx = 12

def animate(i):
    line1.set_ydata(abs(range_fft[i,0,:,0]))
    ax2.set_array(velocity_fft[i,:,:,0])
    print(i)
    return line1, ax2

def runGraphInitial():
    global line1, ax2

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.set_ylim([-1,100])
    line1, = ax1.plot(abs(range_fft[0,0,:,0]))
    
    ax2 = fig.add_subplot(122)
    ax2 = plt.imshow(velocity_fft[0,:,:,0], aspect= 'auto', origin='lower' , cmap='jet', 
         extent=[0, velocity_fft.shape[2]-1, velocity_fft.shape[1]/2., -velocity_fft.shape[1]/2.], interpolation = 'catrom')
    
    ani = FuncAnimation(fig, animate, frames= frame_number  ,interval = 25)
    plt.show()

def dopplerFFT():
    n = range_fft.shape[1]
    dop_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1) / n, axes=1)
    dop_fft = abs(dop_fft)
    return dop_fft

def rangeFFT():
    n = raw_iq.shape[2]
    range_fft = np.fft.fft(raw_iq, axis=2) / n
    return range_fft

def stoveMTI():
    stove_sub = []  
    
    for i in range(raw_iq.shape[0]):
        
        if i == raw_iq.shape[0]-1:
            stove = raw_iq[i]
        else:
            stove = raw_iq[i+1] - raw_iq[i]

        stove_sub.append(stove)

    return np.reshape(np.array(stove_sub), (frame_number, chirp, adcSamples, TxRx))

def firMTI():
    M_order = 97
    slowtime_sampling = 25*32
    nyq =  slowtime_sampling / 2
    Fc = 20
    hp_filter = signal.firwin(M_order, cutoff = Fc / nyq , window="hamming", pass_zero=False)
    signal_out = signal.filtfilt(hp_filter, 1, raw_iq, axis=0)
    # signal_out = signal.filtfilt(lf_filter, 1, range_fft, axis=0)

    plt.figure(1)
    plt.plot(raw_iq[:,0,0])
    plt.plot(signal_out[:,0,0])

    plt.show()

    return np.reshape(signal_out, (600,32,200,12))

def iirMTI():
    M_order = 12
    slowtime_sampling = 25*32
    nyq = slowtime_sampling / 2
    Fc = 20
    hp_filter, a = signal.butter(M_order, Fc/nyq, btype='high', analog=False)
    signal_out = signal.filtfilt(hp_filter, a, raw_iq, axis=0)
    
    plt.figure(1)
    plt.plot(raw_iq[:,0,0])
    plt.plot(signal_out[:,0,0])
    print('iir--runing')
    plt.show()

    return np.reshape(signal_out, (frame_number, chirp, adcSamples, TxRx))

def bgSubtraction():
   
    window = 25
    bg_win = raw_iq[:window]
    signal_sub = raw_iq - np.mean(bg_win, axis=(0,1))
    
    return signal_sub
    

def bgSubtraction_update():

    signal_sub = []
    window = 25
    for i in range(raw_iq.shape[0]):
        
        if  i >= window:

            if i%5 == 0:
                sub_mean = np.mean(raw_iq[i-window:i], axis=(0,1))
                print('update', i-window, i)

            signal = raw_iq[i] - sub_mean
            signal_sub.append(signal)
            

        else:
            signal = raw_iq[i] - np.mean(raw_iq[0:window], axis=(0,1))
            signal_sub.append(signal)

        # print(signal_sub.shape, sub_mean.shape)
    signal_sub = np.array(signal_sub)
    
    return signal_sub

def plot_range_fft():
    fft_plot = np.reshape(range_fft, (frame_number*chirp, adcSamples, TxRx))
    fft_plot = 20*np.log10(fft_plot/32767)
    ax = plt.subplot(111)
    im = ax.imshow(abs(fft_plot[:,:,0]).T, aspect='auto', cmap='jet')
    plt.colorbar(im)
    plt.xlabel('frame (time)')
    plt.ylabel('range (z-axis)')
    plt.show()


def main():

    global range_fft, velocity_fft, raw_iq, SamplingRate

    k = 1000
    SamplingRate = 18750 * k
    dt = 1. / SamplingRate

    folder_name = glob.glob('D:/SmallTrack/data/data_ppl_walk_slow/pos_config_3*')
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
    
    time_scale = np.arange(0, raw_iq.shape[2], 1)
    time_scale = time_scale / SamplingRate

    # print(raw_iq.shape)
    # print(1/dt)

    #### ---------------- Matthew ash paper - IEEE sensor -------------------
    # static bg subtraction
    # raw_iq = bgSubtraction()

    # bg subtraction MTI with updating
    # raw_iq = bgSubtraction_update() # pre-processing using bg subtraction technique
    # print(raw_iq)

    # stove MTI technique
    # raw_iq = np.reshape(raw_iq,(frame_number*chirp, adcSamples, TxRx))
    # raw_iq = stoveMTI() # pre-processing using fir stove technique
    
    # FIR M=99 cut-off 20 hz
    # raw_iq = np.reshape(raw_iq,(frame_number*chirp, adcSamples, TxRx))
    # raw_iq = firMTI()

    # IIR M=12 cut-off 20 hz
    raw_iq = np.reshape(raw_iq,(frame_number*chirp, adcSamples, TxRx))
    raw_iq = iirMTI()

    #### --------------------------------------------------------------------

    # print(raw_iq.shape)
    range_fft = rangeFFT()
    plot_range_fft()
    # N = range_fft.shape[2]
    # F_step = SamplingRate / N
    # freq = np.arange(0, N*F_step, F_step)
    # plt.plot(freq, abs(range_fft[0,0,:,0])) 
    # plt.show()

    # velocity_fft = dopplerFFT()
    # print(velocity_fft.shape)
    # print(range_fft.shape, velocity_fft.shape)

    # runGraphInitial()


    

if __name__ == '__main__':
    main()