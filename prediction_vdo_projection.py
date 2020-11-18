import cv2
import numpy as np
import glob
import pyqtgraph as pg
from numpy.linalg import inv, norm
import sys
from scipy import ndimage, misc, stats
import collections
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import re
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

expect_r_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_esprit_pad.npy'
expect_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_esprit_pad.npy'
label_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/label_z_%4.npy'
trans_file = 'D:/data_signal_MTI/project_util_3/prediction_result/2dfft_op_param_trans.npy'

label_dir = 'D:/data_signal_MTI/project_util_3/label_all/'
rt_circle_dir = 'D:/data_signal_MTI/project_util_3/label_circle/rt_matrix.txt'
rt_square_dir = 'D:/data_signal_MTI/project_util_3/label_square/rt_matrix.txt'
rt_triangle_dir = 'D:/data_signal_MTI/project_util_3/label_triangle/rt_matrix.txt'
rt_dir = [rt_circle_dir, rt_square_dir, rt_triangle_dir]
#for robot
# label_dir = 'D:/data_signal_MTI/project_util_3/label_all_robot_3/'
# rt_robot_dir = 'D:/data_signal_MTI/project_util_3/label_robot_3/rt_matrix.txt'
# rt_dir = [rt_robot_dir]
trajectories = 120

def cam_config(file_vdo):
    global cap, fps, out
    # cap = cv2.VideoCapture(0+ cv2.CAP_DSHOW)
    cap = cv2.VideoCapture('D:/data_signal_MTI/project_util_3/' + file_vdo)
    # cap = cv2.VideoCapture('C:/Users/nakorn-vision/Videos/Logitech/LogiCapture/2020-06-09_21-25-24.mp4')
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FPS, )
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

def frameTrigger():

    frame_green = frame[710, 260, 1]
    frame_green = frame_green > 128
    cv2.imshow('green', frame[710:800, 250:280, :])
    return frame_green

def label_all():
    label_a = []
    rt_all = []
    tr_all = []

    for i in range(trajectories):
        label = np.load(label_dir + 'label_' + str(i+1) + '.npy')
        label_a.append(label)
    
    for f_file in rt_dir:
        f = open(f_file, "r")
        f_rt = f.read()
        string_split = np.float32(re.findall(r"[-+]?\d*\.\d+|\d+", f_rt))
        for s in range(0, len(string_split), 12):
            rotation = [[string_split[s], string_split[s+1], string_split[s+2]], 
                [string_split[s+3], string_split[s+4], string_split[s+5]],
                [string_split[s+6], string_split[s+7], string_split[s+8]]]           
            translation = [[string_split[s+9], string_split[s+10], string_split[s+11]]]
            rt_all.append(rotation)
            tr_all.append(translation)
        f.close()

    return np.array(label_a), np.array(rt_all), np.array(tr_all)

def prediction_all(rt_all, tr_all):
    test_idx = []
    rt_cut = []
    tr_cut = []
    count = 0
    for i in range(trajectories):
        count += 1
        if count%4 == 0:
        # if True:
            rt_cut.append(rt_all[i])
            tr_cut.append(tr_all[i])
            test_idx.append(i)

    expect_r = np.load(expect_r_file)
    expect_z = np.load(expect_z_file)
    # print(expect_r.shape, expect_z.shape)
    label_z = np.load(label_z_file)
    trans = np.load(trans_file)

    x = label_z[:,0] * np.cos(label_z[:,2]) * np.sin(label_z[:,1])
    y = label_z[:,0] * np.sin(label_z[:,2]) + 100
    z = label_z[:,0] * np.cos(label_z[:,2]) * np.cos(label_z[:,1])

    expect_xyz = np.array([[x, y, z]])
    expect_xyz = np.swapaxes(expect_xyz,0,2)
    expect_xyz = expect_xyz.reshape((len(test_idx), -1, 3, 1))

    xp = (expect_r + trans[0]) * np.cos(label_z[:,2]) * np.sin(expect_z + trans[1])
    yp = (expect_r + trans[0]) * np.sin(label_z[:,2]) + 100
    zp = (expect_r + trans[0]) * np.cos(label_z[:,2]) * np.cos(expect_z + trans[1]) 

    expect_p = np.array([[xp,yp,zp]])
    expect_p = np.swapaxes(expect_p,0,2)
    # f_r = np.load('D:/data_signal_MTI/project_util_3/prediction_result/RT_2dfft/rotation_2dfft.npy')
    # f_t = np.load('D:/data_signal_MTI/project_util_3/prediction_result/RT_2dfft/translation_2dfft.npy')
    # expect_p = expect_p.reshape(-1,3)@f_r.T + f_t
    
    expect_p = expect_p.reshape((len(test_idx), -1, 3, 1))
    print(expect_p.shape)
    
  
    
    return expect_xyz, expect_p, np.array(rt_cut), np.array(tr_cut), test_idx

def cal_back_to_uv(label_a, predict, rotation, translation, camera_matrix, predict_flag):
    
    if predict_flag:
        translation = np.swapaxes(translation, 1, 2)
        w_c = np.matmul(rotation, label_a) + translation
        uv_c = np.matmul(camera_matrix, w_c)

        w_p = np.matmul(rotation, predict) + translation
        uv_p = np.matmul(camera_matrix, w_p)
        return uv_c, uv_p

    else:
        label_a = np.swapaxes(label_a, 2, 3)
        translation = np.swapaxes(translation, 1, 2)
        w_c = np.matmul(rotation, label_a) + translation
        uv_c = np.matmul(camera_matrix, w_c)
        return uv_c, 0

def cam_run(camera_matrix):
    
    global frame
    out = cv2.VideoWriter('D:/data_signal_MTI/project_util_3/prediction_result/prediction_esprit_trans.mp4', 
            cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1280,720))
    location = collections.deque(maxlen=5)
    location_c = collections.deque(maxlen=5)
    location_p = collections.deque(maxlen=5)
    # location = []
    # location_c = []
    # location_p = []

    flag_next_frame = False
    frame_count = 0
    real_frame_count = 0
    trigger_frame_label = 0
    s_f_com = 0
    s_f_com_flag = False

    ## if use prediction turn on all (for label and prediction)
    ## for all label 
    label_a, rt_all, tr_all = label_all()
    print(label_a.shape)
    label_a = np.swapaxes(label_a, 0,1)
    predict = 0
    predict_flag = False

    ## for prediction
    label_a, predict, rt_all, tr_all, test_idx = prediction_all(rt_all, tr_all)
    label_a = np.swapaxes(label_a, 0,1)
    predict = np.swapaxes(predict, 0,1)
    predict_flag = True
   
    uv_c, uv_p = cal_back_to_uv(label_a, predict, rt_all, tr_all, camera_matrix, predict_flag)
    
    file_vdo = ['circle_counter_clockwise.mp4', 'square_counter_clockwise.mp4', 'triangle_counter_clockwise.mp4']
    # file_vdo = ['robot_3.mp4']

    for n_vdo in file_vdo:
        
        cam_config(n_vdo)
        ret, frame = cap.read()
        
        while ret == True:
            
            frame_trigger = frameTrigger()
            if frame_trigger:
                
                if 340 > frame_count > 24+5 and (predict_flag == False):
                    point = uv_c[real_frame_count, trigger_frame_label, :, 0] // uv_c[real_frame_count, trigger_frame_label, 2, 0]
                    location.append((int(point[0]),int(point[1])))
                    for j in range(len(location)):
                        cv2.circle(frame, location[j], 4, (255,0,255), -1)
                    cv2.imshow('org', frame)
                    cv2.waitKey(10)
                    real_frame_count += 1
                
                elif 340 > frame_count > 24+5 and (predict_flag == True):
                    if test_idx[s_f_com] == trigger_frame_label:
                        point_c = uv_c[real_frame_count, s_f_com, :, 0] // uv_c[real_frame_count, s_f_com, 2, 0]
                        point_p = uv_p[real_frame_count, s_f_com, :, 0] // uv_p[real_frame_count, s_f_com, 2, 0]
                        location_c.append((int(point_c[0]),int(point_c[1])))
                        location_p.append((int(point_p[0]),int(point_p[1])))
                        for j in range(len(location_c)):
                            cv2.circle(frame, location_c[j], 4, (255,0,255), -1)
                            cv2.circle(frame, location_p[j], 4, (255,0,0), -1)
                        out.write(frame)
                        cv2.imshow('org', frame)
                        # cv2.waitKey()
                        real_frame_count += 1
                        s_f_com_flag = True
                    else:
                        cv2.imshow('org', frame)
                
                elif frame_count == 345 and s_f_com_flag:
                    trigger_frame_label += 1
                    s_f_com += 1
                    s_f_com_flag = False
                
                elif frame_count == 345 :
                    trigger_frame_label += 1
                

                frame_count += 1
            
            else:
                frame_count = 0
                real_frame_count = 0
                location = collections.deque(maxlen=5)
                location_c = collections.deque(maxlen=5)
                location_p = collections.deque(maxlen=5)
            
            
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break

def main():
    camera_matrix = np.load("./preprocess/camera_matrix.npy")
    '''
        fine tune focal length
    '''
    camera_matrix[0,0] -= 200
    camera_matrix[1,1] -= 200
    
    print(camera_matrix)
    cam_run(camera_matrix)

if __name__ == '__main__':
    main()
    