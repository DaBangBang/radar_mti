import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plotFuction(velocity_fft, doppler_tresh, tresh_plane):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221, projection='3d')
    X, Y = np.mgrid[:velocity_fft.shape[1], :velocity_fft.shape[2]]
    ax1.plot_surface(X, Y, abs(velocity_fft[0,:,:,0]), cmap='jet', alpha=0.7)
    ax1.plot_surface(X, Y, tresh_plane, alpha=1)

    ax2 = fig.add_subplot(222)
    ax2 = plt.imshow(doppler_tresh, aspect='auto')
    

    ax3 = fig.add_subplot(223)
    ax3 = plt.imshow(abs(velocity_fft[0,:,:,0]), aspect= 'auto', origin='lower' , cmap='jet', 
         extent=[0, velocity_fft.shape[2]-1, velocity_fft.shape[1]/2., -velocity_fft.shape[1]/2.], interpolation = 'catrom')
    
    plt.show()

    

def kernal(velocity_fft):
    weight = 8
    peak_value = abs(velocity_fft)
    doppler_tresh = np.zeros((velocity_fft.shape[1], velocity_fft.shape[2]))
    tresh_plane = np.zeros((velocity_fft.shape[1], velocity_fft.shape[2]))
    
            # training cell [i-3, j-3:j+4]
            #               [i-2, j-3:j+4]
            #               [i-1, j-3:j-1], [i-1, j+2:j+4]
            #               [i-0, j-3:j-1], [i-0, j+2:j+4]
            #               [i+1, j-3:j-1], [i+1, j+2:j+4]
            #               [i+2, j-3:j+4]
            #               [i+3, j-3:j+4] {7x7}

            # guarding cell [i-1, j-1:j+2]
            #               [i-0, j-2][i-0, j+2]
            #               [i+1, j-1:j+2] {3x3}

            # cell          [i,j]
   
    for i in range(velocity_fft.shape[1]):
        for j in range(velocity_fft.shape[2]):

            kernal_ = np.concatenate((peak_value[0, i-3:i-2, j-3:j+4, 0].reshape(-1),
                                      peak_value[0, i-2:i-1, j-3:j+4, 0].reshape(-1),  
                                      peak_value[0, i-1:i-0, j-3:j-1, 0].reshape(-1), peak_value[0, i-1:i+0, j+2:j+4, 0].reshape(-1),
                                      peak_value[0, i-0:i+1, j-3:j-1, 0].reshape(-1), peak_value[0, i-0:i+1, j+2:j+4, 0].reshape(-1),
                                      peak_value[0, i+1:i+2, j-3:j-1, 0].reshape(-1), peak_value[0, i+1:i+2, j+2:j+4, 0].reshape(-1),
                                      peak_value[0, i+2:i+3, j-3:j+4, 0].reshape(-1),
                                      peak_value[0, i+3:i+4, j-3:j+4, 0].reshape(-1)))            
            kernal_avg = weight*np.mean(kernal_)
            tresh_plane[i,j] = kernal_avg

            cell = peak_value[0,i,j,0]
            
            if cell > kernal_avg :
                doppler_tresh[i,j] = 1
                print('ij',i,j)

    return doppler_tresh, tresh_plane                          

def main():
    velocity_fft = np.load('D:/data_signal_MTI/data_ball_circle/dop_66.npy')
    doppler_tresh, tresh_plane = kernal(velocity_fft)
    plotFuction(velocity_fft, doppler_tresh, tresh_plane)
    # print(velocity_fft[0,0, 100:101,0].shape)

if __name__ == "__main__":
    main()