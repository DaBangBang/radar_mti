import numpy as np
import glob
import natsort

dir_project = 'D:/data_signal_MTI/project_util_2/'
sig_dir = 'D:/data_signal_MTI/project_util_2/signal_all_wo_mti'
label_dir = 'D:/data_signal_MTI/project_util_2/label_all'

signal_folder = [dir_project + 'signal_circle_preprocess/', dir_project + 'signal_square_preprocess/', 
                dir_project + 'signal_triangle_preprocess/' ]

label_folder = [dir_project + 'label_circle/', dir_project + 'label_square/', dir_project + 'label_triangle/']

# print(signal_folder, label_folder)
count = 1

for i in range(len(signal_folder)):
    
    signal_real = signal_folder[i] + 'raw_signal_real_*'
    signal_imag = signal_folder[i] + 'raw_signal_imag_*'
    label = label_folder[i] + 'radar_pos_label_*'
    signal_real = natsort.natsorted(glob.glob(signal_real))
    signal_imag = natsort.natsorted(glob.glob(signal_imag))
    label = natsort.natsorted(glob.glob(label))

    # signal_real = np.load(signal_real)
    # signal_imag = np.load(signal_imag)

    print(len(signal_real), len(signal_imag), len(label))

    for i in range(len(label)):
        real = np.load(signal_real[i])
        imag = np.load(signal_imag[i])
        label_ = np.load(label[i])
        real = real[25:370]
        imag = imag[25:370]

        k = label_.shape[0]
        m = k - real.shape[0]
        label_ = label_[:(k-m)]

        print(label_[5:].shape)
        
        np.save(sig_dir + '/raw_signal_real_' + str(count), real[5:])
        np.save(sig_dir + '/raw_signal_imag_' + str(count), imag[5:])
        np.save(label_dir + '/label_' + str(count), label_[5:])
        count += 1

