import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import itertools

training_cell_along_range = 21
training_cell_along_range = int(training_cell_along_range/2)
guarding_cell_along_range = 15
guarding_cell_along_range = int(guarding_cell_along_range/2)

def kernel(range_fft_no_sub, range_fft_bg_sub):
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
    
    for m, n, o, p in itertools.product(range(range_tresh.shape[0]), range(range_tresh.shape[1]), 
                                    range(range_tresh.shape[2]), range(range_tresh.shape[3])):        

        if tresh_plane[m,n,o,p] < peak_value[m,n,o,p]:
            range_tresh[m,n,o,p] = 0


    return range_tresh

def cal_scr(range_tresh, range_fft_no_sub, range_fft_bg_sub):

    Cout = np.mean(range_tresh, axis=2)
    Cin = np.mean(range_fft_no_sub, axis=2)

    Scr_ = 10*np.log10(Cin/Cout)
    print(Scr_.shape)

    # ------------- save scr_ ---------------------------
    # np.save('D:/data_signal_MTI/data_ball_circle/Scr_iir', Scr_)
    # ------------- plot sample clutter -----------------
    plt.figure(1)
    plt.plot(range_tresh[0,0,:,0])
    # plt.plot(range_fft_no_sub[0,0,:,0])
    plt.plot(range_fft_bg_sub[0,0,:,0])
    # plt.plot(np.full(100, Cin[0,0,0]))
    plt.plot(np.full(100, Cout[0,0,0]))

    plt.figure(2)
    plt.plot(Scr_[:,0,0])
    plt.show()

def main():
    range_fft_no_sub = np.load('D:/data_signal_MTI/data_ball_circle/range_66_no_sub.npy')
    range_fft_bg_sub = np.load('D:/data_signal_MTI/data_ball_circle/range_66_stove.npy') # depend on MTI
    range_fft_no_sub = abs(range_fft_no_sub)
    range_fft_bg_sub = abs(range_fft_bg_sub)
    range_trash = kernel(range_fft_no_sub, range_fft_bg_sub)
    cal_scr(range_trash, range_fft_no_sub, range_fft_bg_sub)
    
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