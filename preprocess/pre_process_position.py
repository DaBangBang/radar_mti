#%%
import numpy as np
import cmath as cm
import math as mp
# import scipy.fftpack
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import re
import natsort

def callBinfile(fname):
    
    # print(fname)    
    fid = open(fname,'r')
    fid = np.fromfile(fid, np.int16)
    print(fid.shape)
    fid = fid[:49283072] # for 376 frame 15 sec 25 fps

    '''
        read data from bin file. Pls see data structure in MMwave radar device
    '''
    frameNumber = 376
    numADCSamples = 512
    numADCbits = 16
    numTx = 2
    numChirps = 16
    numLanes = 4
    '''
    --------------------------------------------------------------------------------
    '''
    # print(fid.shape)
    Tx1, Tx2, Tx3 = list(), list(), list()
    adcData = np.reshape(fid, (-1,numLanes*2))
    adcData_complex = adcData[:,[0,1,2,3]] + cm.sqrt(-1)*adcData[:,[4,5,6,7]]
    adcData_complex = np.reshape(adcData_complex,(frameNumber,numADCSamples*numTx*numChirps,-1))
    for i in range(0, numTx*numChirps*numADCSamples, numTx*numADCSamples):
        Tx1.append(adcData_complex[:,i:i+numADCSamples,:])
        Tx2.append(adcData_complex[:,i+numADCSamples:i+(numADCSamples*2),:])
        # Tx3.append(adcData_complex[:,i+(numADCSamples*2):i+(numADCSamples*3),:])
    
    adcData_Tx1 = np.array(Tx1)
    adcData_Tx2 = np.array(Tx2)
    # adcData_Tx3 = np.array(Tx3)

    # list_adc = [adcData_Tx1, adcData_Tx2, adcData_Tx3]
    list_adc = [adcData_Tx1, adcData_Tx2]
    adcData_Tx_all = np.concatenate(list_adc, axis = 3)

    # plt.show()
    
    '''
        return Tx1, Tx2, Tx3 - TDM - MIMO
        with shape (chirp, frame, adc_sample, Rx)
    '''
    return np.swapaxes(adcData_Tx_all, 0,1)  

def animate(i):
    
    line1.set_ydata(abs(range_fft[i,0,:,0]))
    # line1.set_ydata(range_fft[i,:])
    
    # Re = IQall[i,0,:,0].real
    # Im = IQall[i,0,:,0].imag
    # line2.set_ydata(Re)
    # line3.set_ydata(Im)

    # frame = range_fft[i,:,:,3]
    # axIm.set_array(abs(frame))

    # check_hp = np.argmax(range_fft[i,0,:,0])
    # print(check_hp)
    # check_hp = 2
    # line4.set_ydata(phase[i,:,check_hp,0])
    # line5.set_ydata(phase[i,:,check_hp,1])
    

    # frame2 = range_doppler[i,:,:,0]
    # frame2 = Vefft[0,i,:20,22:42]
    # axDop.set_array(frame2)


    
    # return line1, line2, line3, line4, line5, axDop 
    # return line1, line2, line3, axDop, axIm
    return line1,

def runGraphInitial():
    
    global line1, line2, line3, axIm, axIm, axDop, line4, line5

    fig = plt.figure(1)
    ''' 
        Initial condition fft 
        plot 1d fft
    '''
    ax1 = fig.add_subplot(221)
    line1, = ax1.plot(abs(range_fft[0,0,:,0]))
    # line1, = ax1.plot(range_fft[0,:])
    ax1.set_ylim([-5,200])

    '''
        plot IQ data (raw data)
    '''
    # ax2 = fig.add_subplot(222)
    # color_red = 'tab:red'
    # color_blue = 'tab:blue'
    # Re = IQall[0,0,:,0].real
    # Im = IQall[0,0,:,0].imag
    # line2, = ax2.plot(Re, color_blue)
    # ax3 = ax2.twinx()
    # line3, = ax3.plot(Im, color_red)

    '''
        imshow range-chirp , 
        imshow range-doppler 
    # '''
    # axIm = fig.add_subplot(223)
    # frame = range_fft[0,:,:,3]
    # axIm = plt.imshow(abs(frame), aspect = 'auto', interpolation = 'catrom')

    '''
        phase stability
    '''
    # phIm = fig.add_subplot(223)
    # line4, = phIm.plot(phase[0,:,20,0], color_blue)
    # phIm2 = phIm.twinx()
    # line5, = phIm2.plot(phase[0,:,20,1], color_red)
    # phIm.set_ylim([-200,200])
    # phIm2.set_ylim([-200,200])

    '''
        imshow range-doppler(velocity)
    # '''
    # axDop = fig.add_subplot(224)
    # veFrame = range_doppler[0,:,:,0]
    # veFrame = Vefft[0,0,:20,22:42]
    
    # axDop = plt.imshow(veFrame, aspect= 'auto', origin='lower' , cmap='jet'
    # , interpolation = 'catrom')
    # axDop = plt.imshow(veFrame, aspect= 'auto', origin='lower' , cmap='jet')
    # plt.colorbar(axDop)

    ani = FuncAnimation(fig, animate, frames=250  ,interval = 20)
    plt.show()

def fft_range_function():
    n  = IQall.shape[2]
    range_fft = np.fft.fft(IQall, axis=2) / n
    # range_fft = abs(range_fft)
    # range_fft = 20*np.log10(range_fft/32767)
    
    return range_fft

def fftVelocity():
    
    n = range_fft.shape[1]
    veFrame = np.fft.fftshift(np.fft.fft(range_fft, axis=1) / n, axes=1)
    veFrame = abs(veFrame)
    veFrame = 20*np.log10(veFrame/32767)
    # return abs(veFrame) ## for plot
    return veFrame

def movingAvg_OneD():
    
    '''
    moving averange background subtraction on 1-d fft 
    '''

    fftFrame = range_fft
    print("swap first ", fftFrame.shape)
    fftFrameSum = []
    for i in range(fftFrame.shape[0]-14):

        fftMeanFront = np.mean(fftFrame[i:i+14,:,:,:], axis=0)
        # print(fftMeanFront)
        # fftMeanBack = np.mean(fftFrame[i+16:i+21,:,:,:], axis=0)
        # fftMeanAll = [fftMeanFront, fftMeanBack] 
        # fftMeanAll = np.array(fftMeanAll)
        # fftMeanAll = np.mean(fftMeanAll, axis=0)
        fftMeanSub = fftFrame[i+14,:,:,:] - fftMeanFront
        fftFrameSum.append(fftMeanSub)

    fftFrameSum = np.array(fftFrameSum)
    '''
    '''

    # # fftFrameSum = abs(fftFrameSum)
    # fftFrameSum = np.swapaxes(fftFrameSum,0,1)
    # print("swap second ", fftFrameSum.shape)

    return fftFrameSum


def main():

    global IQall, range_fft, range_doppler, phase

    
    folder_name = glob.glob('D:/data_signal_MTI/project_util_3/signal_triangle_raw')
    # folder_name.sort(key=lambda f: int(re.sub('\D','',f)))
    # folder_name = natsort.natsorted(folder_name)
    # print(folder_name)
    # data_bg = np.load('D:/SmallTrack/data_config_2_moving/data_bg_mov_4/data_bg.npy')
    # data_bg = abs(data_bg)
    # print(data_bg.shape)
    name_count = 0

    for sub_f in folder_name:
    
        print(sub_f)
        bin_insub = sub_f + '/*.bin'
        sub_bin = glob.glob(bin_insub)
        sub_bin = natsort.natsorted(sub_bin)
        # sub_bin.sort(key=lambda f: int(re.sub('\D','',f)))
        range_fft_mean_real = []
        range_fft_mean_imag = []

        for ss in sub_bin:
            name_count += 1
            print('sub_file', ss)
            IQall = callBinfile(ss)
            IQreal = np.float16(IQall.real)
            IQimag = np.float16(IQall.imag)
    
            # IQall = np.float32(IQall)
            print(IQall.shape, IQall[0,0,0,0], IQreal[0,0,0,0], IQimag[0,0,0,0])
        
            # range_fft = fft_range_function()
            # range_fft_sub = abs(range_fft) - abs(data_bg)
            # print("range_after_sub", range_fft_sub.shape)
            # range_fft_sub = range_fft_sub[:,:,1:33,:,]
            # print("range_after_reduce",range_fft_sub.shape)
            # range_fft_sub = np.float32(range_fft_sub)
            # range_fft_mean_all.extend(range_fft_sub)
            # IQall = IQall[:,0,:,0]
            # range_fft_mean_real.extend(IQreal)
            # range_fft_mean_imag.extend(IQimag)
            # runGraphInitial()
            save_real = 'D:/data_signal_MTI/project_util_3/signal_triangle_preprocess/raw_signal_real_' + str(name_count)
            save_imag = 'D:/data_signal_MTI/project_util_3/signal_triangle_preprocess/raw_signal_imag_' + str(name_count)
            np.save(save_real, IQreal)
            np.save(save_imag, IQimag)
        # range_fft_mean_real = np.array(range_fft_mean_real)
        # range_fft_mean_imag = np.array(range_fft_mean_imag)
        # save_name_real = sub_f + '/raw_signal_real_'
        # save_name_imag = sub_f + '/raw_signal_imag_'

        # print("path save =", save_name_real, "shape", range_fft_mean_real.shape, "check", range_fft_mean_real[0,0,0,0])
        # print("path save =", save_name_imag, "shape", range_fft_mean_imag.shape, "check", range_fft_mean_imag[0,0,0,0])
        # np.save(save_name_real + str(name_count), range_fft_mean_real)
        # np.save(save_name_imag + str(name_count), range_fft_mean_imag)


    # range_fft_mean_all = np.array(range_fft_mean_all)
    # print(range_fft_mean_all.shape)
    # np.save('D:/SmallTrack/data_pos_test_2/signal_all', range_fft_mean_all)
  
    
if __name__ == '__main__':
    main()
    
