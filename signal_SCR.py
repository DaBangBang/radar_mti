import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import itertools
from scipy import stats

training_cell_along_range = 21
training_cell_along_range = int(training_cell_along_range/2)
guarding_cell_along_range = 15
guarding_cell_along_range = int(guarding_cell_along_range/2)

def kernel(range_fft_no_sub, range_fft_bg_sub, range_fft_mov, range_fft_stove, range_fft_fir, range_fft_iir ):
    print(range_fft_bg_sub.shape)
    peak_value = range_fft_bg_sub
    weight = 1.5
    tresh_plane = np.zeros((range_fft_bg_sub.shape[0], range_fft_bg_sub.shape[1],
                            range_fft_bg_sub.shape[2], range_fft_bg_sub.shape[3]))
    
    for j in range(range_fft_bg_sub.shape[2]):
        kernel_ = peak_value[:, :, max(0,j-training_cell_along_range):j+training_cell_along_range+1, :].copy()
        kernel_[:, :, max(0,j-guarding_cell_along_range):j+guarding_cell_along_range+1, :] = np.nan
        kernel_ = weight*np.nanmean(kernel_, axis= 2)            
        tresh_plane[:,:,j,:] = kernel_
    
    # range_tresh ====> avg_clutter after MTI
    range_tresh = range_fft_bg_sub.copy()
    range_tresh_mov = range_fft_mov.copy()
    range_tresh_stove = range_fft_stove.copy()
    range_tresh_fir = range_fft_fir.copy()
    range_tresh_iir = range_fft_iir.copy()

    for m, n, o, p in itertools.product(range(range_tresh.shape[0]), range(range_tresh.shape[1]), 
                                    range(range_tresh.shape[2]), range(range_tresh.shape[3])):        

        if tresh_plane[m,n,o,p] < peak_value[m,n,o,p]:
            range_tresh[m,n,o,p] = 0
            range_tresh_mov[m,n,o,p] = 0
            range_tresh_stove[m,n,o,p] = 0
            range_tresh_fir[m,n,o,p] = 0
            range_tresh_iir[m,n,o,p] = 0

    return range_tresh, range_tresh_mov, range_tresh_stove, range_tresh_fir, range_tresh_iir

def cal_scr(range_tresh, range_tresh_mov, range_tresh_stove
            ,range_tresh_fir, range_tresh_iir, range_fft_no_sub, range_fft_bg_sub, range_fft_mov, 
            range_fft_stove, range_fft_fir, range_fft_iir):

    
    Cout = np.mean(range_tresh, axis=2)
    Cout_c = np.mean(range_tresh_mov, axis=2)
    Cin = np.mean(range_fft_no_sub, axis=2)

    
    Scr_ = 10*np.log10(Cin/Cout)
    Scr_c = 10*np.log10(Cin/Cout_c)
    print(Scr_.shape)

    # ------------- cal signal to noise ratio --------------------
    peak_bg_sub = np.max(range_fft_bg_sub, axis =2)
    peak_mov = np.max(range_fft_mov, axis =2)
    peak_stove = np.max(range_fft_stove, axis =2)
    peak_fir = np.max(range_fft_fir, axis =2)
    peak_iir = np.max(range_fft_iir, axis =2)
    
    std_bg_sub = np.std(range_tresh, axis= 2)
    std_mov = np.std(range_tresh_mov, axis= 2)
    std_stove = np.std(range_tresh_stove, axis= 2)
    std_fir = np.std(range_tresh_fir, axis= 2)
    std_iir = np.std(range_tresh_iir, axis= 2)

    snr_bg_sub = 10*np.log10(peak_bg_sub/std_bg_sub)
    snr_mov = 10*np.log10(peak_mov/std_mov)
    snr_stove = 10*np.log10(peak_stove/std_stove)
    snr_fir = 10*np.log10(peak_fir/std_fir)
    snr_iir = 10*np.log10(peak_iir/std_iir)

    # ------------- save scr_ ---------------------------
    # np.save('D:/data_signal_MTI/data_ball_circle/Scr_iir', Scr_)
    # ------------- plot sample clutter -----------------
    plt.figure(1)
    # plt.plot(range_tresh[5,0,:,0])
    plt.plot(range_fft_no_sub[35,0,:,0])
    # plt.plot(range_fft_bg_sub[35,0,:,0], color='black')
    plt.plot(range_fft_mov[35,0,:,0], color='orange')
    plt.plot(np.full(100, Cin[35,0,0]), color='black')
    plt.plot(np.full(100, Cout[35,0,0]), color='red')
    # plt.plot(np.full(100, Cout_c[5,0,0]), color='blue')

    plt.figure(2)
    plt.plot(Scr_[:,0,0])
    plt.plot(Scr_c[:,0,0])
    
    plt.figure(3)
    plt.plot(snr_bg_sub[:,0,0])
    plt.plot(snr_mov[:,0,0])
    plt.plot(snr_stove[:,0,0])
    plt.plot(snr_fir[:,0,0])
    plt.plot(snr_iir[:,0,0])
    # print(peak_bg_sub[30,0,0], peak_compare[30,0,0])
    # print(std_bg_sub[30,0,0], std_compare[30,0,0])
    # print(snr_bg_sub[30,0,0], snr_compare[30,0,0])
    print(np.mean(Scr_[5:]), np.mean(Scr_c[5:]))
    print(np.mean(snr_bg_sub), np.mean(snr_mov))
    plt.show()
    

def main():
    range_fft_no_sub = np.load('D:/data_signal_MTI/data_ball_circle/range_66_no_sub.npy')
    range_fft_bg_sub = np.load('D:/data_signal_MTI/data_ball_circle/range_66_bg_sub.npy') # depend on MTI
    
    range_fft_mov = np.load('D:/data_signal_MTI/data_ball_circle/range_66_mov_avg.npy')
    range_fft_stove = np.load('D:/data_signal_MTI/data_ball_circle/range_66_stove.npy')
    range_fft_iir = np.load('D:/data_signal_MTI/data_ball_circle/range_66_fir.npy')
    range_fft_fir = np.load('D:/data_signal_MTI/data_ball_circle/range_66_iir.npy')
    
    range_fft_no_sub = abs(range_fft_no_sub)
    range_fft_bg_sub = abs(range_fft_bg_sub)
    range_fft_mov = abs(range_fft_mov)
    range_fft_stove = abs(range_fft_stove)
    range_fft_fir = abs(range_fft_fir)
    range_fft_iir = abs(range_fft_iir)
  
    range_tresh, range_tresh_mov, range_tresh_stove, range_tresh_fir, range_tresh_iir = kernel(range_fft_no_sub, range_fft_bg_sub, range_fft_mov,
                                                                                                range_fft_stove, range_fft_fir, range_fft_iir)
    cal_scr(range_tresh, range_tresh_mov, range_tresh_stove
            ,range_tresh_fir, range_tresh_iir, range_fft_no_sub, range_fft_bg_sub, range_fft_mov, 
            range_fft_stove, range_fft_fir, range_fft_iir)
    
    # ====================== scr ====================================
    # s1 = np.load('D:/data_signal_MTI/data_ball_circle/Scr_bg_sub.npy')
    # s2 = np.load('D:/data_signal_MTI/data_ball_circle/Scr_mov_avg.npy')
    # s3 = np.load('D:/data_signal_MTI/data_ball_circle/Scr_stove.npy')
    # s4 = np.load('D:/data_signal_MTI/data_ball_circle/Scr_fir.npy')
    # s5 = np.load('D:/data_signal_MTI/data_ball_circle/Scr_iir.npy')
    # plt.plot(s1[1:,0,0])
    # plt.plot(s2[1:,0,0])
    # plt.plot(s3[1:,0,0])
    # plt.plot(s4[1:,0,0])
    # plt.plot(s5[1:,0,0])
    # plt.show()


if __name__ == '__main__':
    main()