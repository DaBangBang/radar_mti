import numpy as np
import glob
import natsort
import plotly.graph_objects as go
import re   

label_dir = 'D:/data_signal_MTI/data_ball_move_39_label/radar*'
r_dir = 'D:/data_signal_MTI/data_ball_move_39_real_imag/raw_signal_real_*'
i_dir = 'D:/data_signal_MTI/data_ball_move_39_real_imag/raw_signal_imag_*'

save_radar_clean_dir = 'D:/data_signal_MTI/data_ball_move_39_real_imag_clean/'

folder_name = glob.glob(label_dir)
folder_name = natsort.natsorted(folder_name)
r_dir = glob.glob(r_dir)
r_dir = natsort.natsorted(r_dir)
i_dir = glob.glob(i_dir)
i_dir = natsort.natsorted(i_dir)
all_label = []

for f_name in folder_name[:]:
    for r_name in r_dir[:]:
        for i_name in i_dir[:]:
            n_label = np.array(list(map(int, re.findall("\d+", f_name))))
            r_label = np.array(list(map(int, re.findall("\d+", r_name))))
            i_label = np.array(list(map(int, re.findall("\d+", i_name))))    
            if n_label[1] == r_label[1] == i_label[1]:
                print(f_name)
                print(r_name)
                print(i_name)
                print("=================")
                label = np.load(f_name)
                real = np.load(r_name)
                imag = np.load(i_name)
                sub_sample = real.shape[0] - label.shape[0]
                real = real[sub_sample:]
                imag = imag[sub_sample:]
                print(label.shape, real.shape, imag.shape)
                print("================")
                np.save(save_radar_clean_dir + 'raw_signal_real_' + str(r_label[1]), real)
                np.save(save_radar_clean_dir + 'raw_signal_imag_' + str(i_label[1]), imag)
                
                

