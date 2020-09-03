import numpy as np
import glob
import natsort
import matplotlib.pyplot as plt

f_name = 'D:/data_signal_MTI/data_ppl_walk_slow/pos_config_66.626'
test_data_all = []

def complex_combine(data_real, data_imag):
    data_real = np.load(data_real)
    data_imag = np.load(data_imag)
    # print(data_real.shape)
    data_real = data_real[:,0,:,0]
    data_imag = data_imag[:,0,:,0]
    # data_complex = data_real + 1j*data_imag
    # data_complex = data_complex[:,0,:,0]
    data_complex = np.array([data_real, data_imag])
    data_complex = np.swapaxes(data_complex, 0, 2)
    # print(data_complex.shape)
    # data_fft = np.fft.fft(data_complex, axis=1) 
    # data_fft = abs(data_fft)
    # # data_fft = np.mean(data_fft, axis=1)
    
    # # select axis = 1
    # # data_fft = data_fft[:,:,0]
    print(data_complex.shape)
    # plt.plot(data_complex[:,100])
    # plt.plot(data_fft[100,:])
    # plt.show()
    np.save('D:/data_signal_MTI/data_ball_move_39_graph/test_siren', data_complex)


if __name__ == '__main__':
    
    # folder_name = glob.glob(folder_name)
    # folder_name = natsort.natsorted(folder_name)
    count = 0

    # for f_name in folder_name[:1]:
    #     count += 1
    real_name = f_name + '/raw_signal_real.npy'
    # real_name = glob.glob(real_name)
    imag_name = f_name + '/raw_signal_imag.npy'
    # imag_name = glob.glob(imag_name)
    print(real_name)
    complex_combine(real_name, imag_name)

        
     